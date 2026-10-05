"""
scripts/migrate_to_graph.py — 存量数据迁进知识结构图谱 (10-05)

把 58 个 workspace 的历史产物搬进 `facts.db` 的主题层:

    1. 补写 fact 库  —— 34 个有 fact 的 workspace 里只有 3 个的 fact 真的进过
       facts.db(实测 2505 条里 714 条能对上, 1791 条从没入库)。因为 fact 库
       写入是 9-29 才加的, 而那 31 个 workspace 都是 9-29 之前跑的。
    2. 剥 prompt 占位符 —— 624 条(24%)的 claim 末尾带着字面量 `[paper_id]`
       (模型把 prompt 里的示例标记漏进输出了)。**必须在算 id 之前剥**: 留着
       的话这条 fact 的内容哈希就永久带上了这个噪声, 以后同一件事被干净地
       重抽会得到不同的 id, 制造一对永远对不上的孪生 fact。
    3. 导入已有主题地图 —— 3 个 workspace 有 _thematic_map.json, 把主题/关系
       upsert 进表, 并把它们的 fact_ids 映射到 facts.db 的全局 id。

**幂等**: 全部走内容哈希 + INSERT OR IGNORE, 跑两遍结果一致。主题层是累积
生长的, 从不删除 —— 见 agent/infra/theme_graph.py 模块头的模型说明。

用法:
    venv\\Scripts\\python.exe scripts\\migrate_to_graph.py            # 只做数据迁移
    venv\\Scripts\\python.exe scripts\\migrate_to_graph.py --dry      # 只报表不写库
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

os.chdir(ROOT)

from agent.infra import theme_graph as tg  # noqa: E402
from agent.infra.db import connect  # noqa: E402
from agent.prompts_survey import (  # noqa: E402
    clean_fact,
    strip_paper_id_from_claim,
)
from agent.prompts_survey import (  # noqa: E402
    count_placeholder_forms as _count_in_text,
)
from agent.tools.facts import add_facts_from_extraction, fact_id_for  # noqa: E402

WS_ROOT = ROOT / "data" / "surveys" / "working"

# prompt 里给模型的占位符被漏进输出的形态。10-05 起**不再在这里自带正则** ——
# 统一调 prompts_survey.strip_paper_id_from_claim, 它是唯一真相源。
# 之前这里另写一套 4 种形态, 漏了 "[2209.02093]" 裸 id, 于是库里同时存在
# "剥过的" 和 "没剥过的" 两种 claim(767 条), 而 fact.id 是内容哈希 ——
# 标注只差一个标点就算成两条不同的 fact, 去重机制被自己的脏数据废掉。
# 形态清单和新增形态的回归测试见 prompts_survey._PLACEHOLDER_RES。
def strip_placeholders(text: str) -> tuple[str, int]:
    """剥掉 prompt 占位符, 返回 (干净文本, 剥掉几个)。"""
    if not text:
        return "", 0
    n = sum(count_placeholder_forms(text).values())
    return strip_paper_id_from_claim(text), n


def load_workspace(ws: Path) -> list[dict]:
    fp = ws / "_facts.jsonl"
    if not fp.exists():
        return []
    out = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def cleaned_facts(ws: Path) -> tuple[list[dict], int, int]:
    """读一个 workspace 的 fact 并做清洗。返回 (清洗后, 空 claim 数, 剥掉的占位符数)。"""
    out, n_empty, n_ph = [], 0, 0
    for f in load_workspace(ws):
        try:
            cf = clean_fact(dict(f))
        except Exception:
            cf = dict(f)
        claim, k = strip_placeholders(cf.get("claim") or "")
        n_ph += k
        if not claim:
            n_empty += 1
            continue
        cf["claim"] = claim
        out.append(cf)
    return out, n_empty, n_ph


def backfill_fact_library(dirs: list[Path], stats: Counter) -> None:
    """把清洗后的 fact 补写进 facts.db(幂等, 按内容哈希收敛)。"""
    for ws in dirs:
        batch, _, _ = cleaned_facts(ws)
        if not batch:
            continue
        r = add_facts_from_extraction(batch, source_type="paper_abstract")
        stats["入库-新建 fact"] += r["new_facts"]
        stats["入库-命中已有"] += r["inserted"] - r["new_facts"]
        stats["入库-多出处"] += r["multi_source"]
        stats["入库-跳过"] += r["skipped"]


def import_thematic_maps(dirs: list[Path], ws_fid_to_global: dict[str, str],
                         dry: bool) -> Counter:
    """把已有 _thematic_map.json 的主题/关系 upsert 进表。

    宏观主题**不在这里造** —— 10-05 才加的 macros, 那时这些文件都没有这一层,
    编一个占位的宏观主题等于凭空发明上位结构。宏观层由 backfill_graph 事后补。
    """
    imported: Counter = Counter()
    for ws in dirs:
        tm_path = ws / "_thematic_map.json"
        if not tm_path.exists():
            continue
        try:
            tmap = json.loads(tm_path.read_text(encoding="utf-8"))
        except Exception:
            imported["地图解析失败"] += 1
            continue
        themes = tmap.get("themes") or []
        if not themes:
            continue
        imported["有地图的 workspace"] += 1
        if dry:
            continue
        with connect("facts") as conn:
            id_by_name: dict[str, int] = {}
            for t in themes:
                name = (t.get("name") or "").strip()
                if not name:
                    continue
                tid = tg.upsert_theme(
                    conn, name, t.get("statement", ""), t.get("notes", ""))
                id_by_name[name] = tid
                gids = []
                for fid in t.get("fact_ids") or []:
                    g = ws_fid_to_global.get(f"{ws.name}\x00{fid}")
                    if g:
                        gids.append(g)
                    else:
                        imported["fact_id 反查失败"] += 1
                n = tg.link_facts(conn, tid, gids, asserted_by=ws.name)
                imported["主题"] += 1
                imported["归属边"] += n
                imported["主题无 fact"] += 1 if n == 0 else 0
            for t in themes:
                src = id_by_name.get((t.get("name") or "").strip())
                if not src:
                    continue
                for rel in t.get("relations") or []:
                    dst = id_by_name.get((rel.get("to") or "").strip())
                    if not dst:
                        imported["关系指向不存在的主题"] += 1
                        continue
                    tg.add_relation(conn, src, dst, rel.get("note", "") or "")
                    imported["关系边"] += 1
    return imported


def count_placeholder_forms(dirs: list[Path]) -> Counter:
    """占位符按形态分开计数。

    为什么要分开: 总数对不上就说明正则又漏了新形态, 而漏掉的那一种恰恰可能是
    最毒的。10-05 就是先只匹配 `[paper_id]`(436 处), 补上 `[paper_id: 真实id]`
    之后才发现另外 187 处 —— 后者会把真实 paper_id 写进内容哈希, 直接废掉
    fact.id 的去重机制。总数吻合不等于没漏, **分形态**才看得出来。

    10-05 补: 又漏了 `[2209.02093]` 裸 id(767 条) —— 这次不是"漏写正则",
    是这里和 prompts_survey 各有一套、覆盖范围不同。形态清单现已收敛到
    prompts_survey._PLACEHOLDER_RES, 这里只负责按形态报数。
    """
    forms: Counter = Counter()
    names = ("[paper_id: 真实id] 污染内容哈希", "[[...]] 双括号", "[paper_id] 纯占位",
             "<paper_id> 尖括号", "裸 [arxiv_id] 光秃秃",
             "[期刊 id]", "裸 paper_id: id")
    for ws in dirs:
        for f in load_workspace(ws):
            counts = _count_in_text(f.get("claim") or "")
            for i, k in counts.items():
                if k:
                    forms[names[i] if i < len(names) else f"形态{i}"] += k
    return forms


def report(dirs: list[Path], stats: Counter, imported: Counter,
           n_placeholder: int, forms: Counter, dry: bool) -> None:
    print("=" * 68)
    print("存量迁移报表" + ("(dry-run, 未写库)" if dry else ""))
    print("=" * 68)
    print(f"workspace 扫描 {len(dirs)} 个, 有 fact 的 "
          f"{sum(1 for d in dirs if load_workspace(d))} 个")
    print(f"fact 读入 {stats['fact 读入']} / 可用 {stats['fact 可用']} / "
          f"丢弃 {stats['空 claim 丢弃']}")
    print(f"剥掉的 prompt 占位符 {n_placeholder} 处:")
    for k, v in forms.most_common():
        print(f"    {v:>5}  {k}")
    if not dry:
        print(f"\n入库: 新建 fact {stats['入库-新建 fact']} | 命中已有 "
              f"{stats['入库-命中已有']} | 合并出多出处 {stats['入库-多出处']} | "
              f"跳过 {stats['入库-跳过']}")
        print(f"\n主题地图导入: workspace {imported['有地图的 workspace']} | "
              f"主题 {imported['主题']} | 归属边 {imported['归属边']} | "
              f"关系边 {imported['关系边']}")
        for k in ("fact_id 反查失败", "关系指向不存在的主题", "主题无 fact"):
            if imported[k]:
                print(f"  ⚠ {k}: {imported[k]}")
    with connect("facts") as conn:
        s = tg.stats(conn)
    print("\n--- facts.db 现状 ---")
    for k, v in s.items():
        print(f"  {k:<20} {v}")
    cov = s["facts_linked"] * 100 // max(1, s["facts_total"])
    print(f"\nfact 主题覆盖率: {s['facts_linked']}/{s['facts_total']} ({cov}%)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只报表不写库")
    args = ap.parse_args()
    dry = args.dry
    if not dry:
        tg.init_schema()

    stats: Counter = Counter()
    # workspace 原始 fact_id -> facts.db 全局 id。旧 fact_id 体系污染严重
    # (668 个 id 对应 2505 条 fact), 所以必须**按内容**反查, 不能直接当主键用。
    ws_fid_to_global: dict[str, str] = {}
    n_placeholder = 0
    dirs = sorted(d for d in WS_ROOT.iterdir() if d.is_dir()) if WS_ROOT.exists() else []

    for ws in dirs:
        raw = load_workspace(ws)
        if not raw:
            continue
        cleaned, n_empty, n_ph = cleaned_facts(ws)
        n_placeholder += n_ph
        stats["fact 读入"] += len(raw)
        stats["fact 可用"] += len(cleaned)
        stats["空 claim 丢弃"] += n_empty
        stats["剥掉占位符"] += n_ph
        # 逐条重算而不是 zip(raw, cleaned): 空 claim 会被丢弃, 按位置 zip 会错位,
        # 把 A 的旧 fact_id 映到 B 的内容上 —— 而这个映射直接决定主题地图导入时
        # 哪条 fact 挂到哪个主题, 错位了不会报错, 只会让结果悄悄错
        for f in raw:
            old = f.get("fact_id")
            if not old:
                continue
            try:
                cf = clean_fact(dict(f))
            except Exception:
                cf = dict(f)
            claim, _ = strip_placeholders(cf.get("claim") or "")
            if not claim:
                continue
            g = fact_id_for(claim, cf.get("fact_type") or "")
            # 同一个旧 id 可能对应多条(旧 id 污染), 留第一条不覆盖
            ws_fid_to_global.setdefault(f"{ws.name}\x00{old}", g)

    if not dry:
        backfill_fact_library(dirs, stats)
    imported = import_thematic_maps(dirs, ws_fid_to_global, dry)
    report(dirs, stats, imported, n_placeholder, count_placeholder_forms(dirs), dry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
