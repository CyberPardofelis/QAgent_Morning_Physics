"""
scripts/backfill_graph.py — 给没有主题地图的 workspace 回填主题层 (10-05)

**为什么分块而不是抽样**

`_organize.organize_facts_into_themes` 里有个 THEME_MAP_MAX_FACTS=200 的硬上限:
超过就按 subject 均匀抽样, 被抽掉的 fact **永远不会被任何主题覆盖**。实测
`669cfc28a53e` 有 309 条 fact, 硬上限会静默丢掉 35% —— 而那正是要总结的目标数据。

抽样的前提是"主题地图一次调用吃下全部 fact", 那是旧设计(每次跑画一张独立的图)。
现在主题是**全局累积**的, 于是可以:
    1. 把 fact 分块(每块 <= CHUNK), 每块独立跑一次主题归纳
    2. 所有块产出的主题**按名 upsert 进同一张表**(同名即同一主题)
    3. 块与块之间的主题自然合并/共存

这样一条 fact 都不会丢, 而且块越多主题越细 —— 正好是"核心对象"层想要的粒度。
代价: 一次 workspace 要多次调用, 但用 m3.1 flash 单块 25 秒左右。

**幂等**: 主题按名 upsert, 归属边 INSERT OR IGNORE。重跑不会产生重复。

用法:
    venv\\Scripts\\python.exe scripts\\backfill_graph.py --dry          # 只报表
    venv\\Scripts\\python.exe scripts\\backfill_graph.py --limit 3     # 先跑 3 个看效果
    venv\\Scripts\\python.exe scripts\\backfill_graph.py                # 全量
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

os.chdir(ROOT)

from agent.infra import theme_graph as tg  # noqa: E402
from agent.infra.db import connect  # noqa: E402
from agent.prompts_survey import clean_fact  # noqa: E402
from agent.tools.facts import fact_id_for  # noqa: E402
from agent.tools.surveys import _organize as O  # noqa: E402

WS_ROOT = ROOT / "data" / "surveys" / "working"

# 2026-10-06 修: 原来硬编码 "MiniMax-M3.1-Flash-Preview", **不在白名单里**
# (白名单只有 M2 / M2.1 / M2.5 / M2.7 / M3)。写错不会立刻暴露 —— 每次调用都
# 400, 而 cost_guard 是"先计费再发请求", 于是整轮跑完额度就没了, 一条主题都
# 没落库。这跟 10-05 去重脚本踩的是同一个坑。
MODEL = "MiniMax-M3"


def preflight(model: str, n_calls: int = 1) -> None:
    """发第一条之前先验模型名和额度, 否则跑砸掉的是一整天的预算。"""
    from agent.llm_clients import MINIMAX_VALID_MODELS
    if model not in MINIMAX_VALID_MODELS:
        raise SystemExit(
            f"[中止] 模型名 {model!r} 不在 MiniMax 白名单里。\n"
            f"  有效: {', '.join(MINIMAX_VALID_MODELS)}\n"
            f"  写错不会立刻暴露 —— 每次调用都 400, 而 cost_guard 先计费,\n"
            f"  整轮跑完额度就没了还一条主题都没落库。")
    from agent.cost_guard import get_cost_guard
    from config import Config
    used, ceil = get_cost_guard().current_count(), Config.get_cost_ceiling_daily()
    left = max(0, ceil - used)
    print(f"[preflight] 模型 {model} ✓ · 今日额度 {used}/{ceil}, 剩余 {left}")
    if n_calls and left < n_calls:
        raise SystemExit(
            f"[中止] 剩余额度 {left} < 预计至少要发的 {n_calls} 次。\n"
            f"  先调高 cost_ceiling_daily 再跑, 或者用 --limit 少跑几个。")

# **按字符分块, 不按条数。** 实测(m3.1 flash):
#     3360 字符 (23 条)  ->  82s  成功
#     9458 字符 (60 条)  -> 154s  成功
#    13194 字符 (140 条) ->  卡死 43 分钟, timeout=600 没被真正执行
# 所以 140 条/块是**按条数封顶**时踩出来的坑: 同一块数下 prompt 大小能差 4 倍
# (fact 的 claim 长度差异很大), 条数根本不是那个决定输出的量。
# 8000 字符是个实测安全线 —— 超过它的两次都出了事, 没超过的三次都正常。
CHUNK_CHARS = 8000
CHUNK_FACTS = 60          # 双重保险: 极长的 claim 也不会把块撑爆

# 10-05: 测试/占位 workspace, 必须硬跳过。三个:
#   640ab2bae07b  topic="Test", 96% 是 "论文核心方法为 method 4" 这种把字段名
#                当正文的垃圾, 聚出来全是"技术路线识别/方法贡献骨架"这类**元主题**;
#                而且每条 fact 都进了那几个主题, 任意两个之间都"共享 5 篇论文",
#                把跨域边榜首占满 —— 真边被挤到看不见。
#   1614c621f2dc  topic="测试 P6.3 持续脏", 2 条 fact 全是字面量 "test [2401.99999]"
#   ad47e85173c1  topic="测试 P6.3 自动重写", 同上。2401.99999 不是真实 arXiv id。
#
# **必须硬跳过**: 续跑判据是"asserted_by 有没有出现在 theme_fact 里",
# 清理之后 theme_fact 全没了 -> 判定为"没跑过" -> 又给它跑一遍 -> 垃圾复活。
# 这个坑真踩过两次。
SKIP_WORKSPACES = {"640ab2bae07b", "1614c621f2dc", "ad47e85173c1"}

_PLACEHOLDER = re.compile(
    r"\[\s*paper[_\s-]?id\s*(?::\s*[^\]\[]{0,40})?\s*\]"
    r"|\[\[\s*[^\]\[]{1,40}\s*\]\]"
    r"|<\s*paper[_\s-]?id\s*>"
    r"|(?<![\w.])paper[_\s-]?id\s*:\s*\d{4}\.\d{4,5}(?![\w])",
    re.I,
)


def strip_ph(text: str) -> str:
    out = _PLACEHOLDER.sub("", text or "")
    return re.sub(r"[\s，,。;；:：]{2,}", lambda m: m.group(0)[0], out).strip()


def load_facts(ws: Path) -> list[dict]:
    fp = ws / "_facts.jsonl"
    if not fp.exists():
        return []
    out = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            f = json.loads(line)
        except Exception:
            continue
        try:
            cf = clean_fact(dict(f))
        except Exception:
            cf = dict(f)
        c = strip_ph(cf.get("claim") or "")
        if not c:
            continue
        cf["claim"] = c
        gid = fact_id_for(c, cf.get("fact_type") or "")
        cf["_global_id"] = gid
        # **必须把 fact_id 换成全局 id 再喂给主题地图**。
        # 旧 fact_id 污染极严重(实测 668 个 id 对应 2505 条 fact, 71% 是裸
        # `fN`, `f1` 一个人就对应 407 种不同文本) —— 拿它当标识, 主题地图
        # 根本无法区分两条不同的 fact, 校验时也会把它们当同一条丢掉。
        # 换成内容哈希后: 唯一、稳定, 而且主题返回的 id 可以直接写进
        # theme_fact(有真外键), 不需要再反查一遍。
        cf["fact_id"] = gid
        out.append(cf)
    return out


def split_chunks(facts: list[dict]) -> list[list[dict]]:
    """按 **claim 总字符数**切块, 同时受条数上限约束。

    为什么不用"固定 N 条": 实测同是 140 条的块, prompt 能从 3360 到 13194 字符
    (claim 长度差异极大), 而 13194 那次**卡死 43 分钟**、timeout 形同虚设。
    决定输出和稳定性的是 prompt 大小, 不是条数。
    """
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    n = 0
    for f in facts:
        L = len(f.get("claim") or "") + 40          # 40 是 [fact_id] (type) 的开销
        if cur and (n + L > CHUNK_CHARS or len(cur) >= CHUNK_FACTS):
            chunks.append(cur)
            cur, n = [], 0
        cur.append(f)
        n += L
    if cur:
        chunks.append(cur)
    return chunks


def run_one(ws: Path, facts: list[dict], dry: bool) -> dict:
    """一个 workspace: 分块 -> 主题归纳 -> upsert -> 宏观层。"""
    rep = {"ws": ws.name, "facts": len(facts), "chunks": 0, "themes": 0,
           "links": 0, "relations": 0, "macros": 0, "warn": Counter(), "err": ""}
    chunks = split_chunks(facts) or [facts]
    for ch in chunks:
        rep["chunks"] += 1
        if dry:
            continue
        try:
            tmap = O.organize_facts_into_themes(
                ch, paper_titles={}, provider="minimax", model=MODEL)
        except Exception as e:                      # noqa: BLE001
            rep["err"] = f"{type(e).__name__}: {e}"[:200]
            continue
        themes = tmap.get("themes") or []
        if not themes:
            rep["warn"]["块无主题"] += 1
            for w in tmap.get("warnings") or []:
                if "未被任何主题覆盖" in w:
                    rep["warn"]["抽样丢弃"] += 1
            continue
        with connect("facts") as conn:
            id_by_name = {}
            for t in themes:
                name = (t.get("name") or "").strip()
                if not name:
                    continue
                tid = tg.upsert_theme(conn, name, t.get("statement", ""),
                                      t.get("notes", ""))
                id_by_name[name] = tid
                rep["themes"] += 1
                # fact_id 已经是全局 id(见 load_facts), 直接就是 theme_fact.fact_id
                gids = [x for x in (t.get("fact_ids") or []) if x.startswith("fact_")]
                rep["links"] += tg.link_facts(conn, tid, gids, asserted_by=ws.name)
                if not gids:
                    rep["warn"]["主题零归属"] += 1
            for t in themes:
                src = id_by_name.get((t.get("name") or "").strip())
                if not src:
                    continue
                for rel in t.get("relations") or []:
                    dst = id_by_name.get((rel.get("to") or "").strip())
                    if not dst:
                        continue                       # 块内看不到块外主题 -> 跳过
                    tg.add_relation(conn, src, dst, rel.get("note", "") or "")
                    rep["relations"] += 1
    return rep


def macro_layer(dry: bool) -> int:
    """宏观层: **只对未归类的主题**跑一次, 并优先并入已有板块(10-06 改)。

    这是"主题和 workspace 无关"的直接体现 —— 宏观主题是整个语料的上位结构,
    逐 workspace 各聚一次会得到一堆互不相干的半截图。

    **为什么从"全量重聚"改成"只聚未归类的"(10-06 实测踩的坑)**:
    原来是 `WHERE EXISTS(...)` 全量取, 主题数只增不减 —— 209 个主题一次塞进
    prompt = 19351 字符, 而 timeout=300 秒。实测 6575 字符就要 160 秒, 所以
    **必然超时**, 连撞两次, 宏观板块一个没落。更糟的是这不会随时间自愈:
    下次回填只会让 prompt 更长。

    改成只取 `macro_id IS NULL` 之后, 这次的输入就是实际待归类的 28 个,
    prompt 约 2500 字符, 一次过。附带两个好处:
      - **历史结构只增不改**: 上一轮定好的板块边界不会被这轮换个措辞推翻。
      - **天然幂等**: 全都归完类之后, 待处理数是 0, 直接跳过。

    **statement 要截断**: 这一层要判断的是"哪几个主题属于同一块", 靠主题名
    + 领域判断的前 80 字就够, 全文只会撑爆预算。
    """
    with connect("facts") as conn:
        rows = conn.execute("""
            SELECT t.name, t.statement,
                   (SELECT COUNT(*) FROM theme_fact tf WHERE tf.theme_id=t.id) AS n
              FROM theme t
             WHERE t.macro_id IS NULL
               AND EXISTS (SELECT 1 FROM theme_fact tf WHERE tf.theme_id=t.id)
             ORDER BY n DESC
        """).fetchall()
        existing = conn.execute("""
            SELECT m.name, m.statement,
                   (SELECT COUNT(*) FROM theme t WHERE t.macro_id=m.id) AS n
              FROM theme_macro m
             ORDER BY n DESC
        """).fetchall()
    themes = [{"name": r["name"],
               "statement": (r["statement"] or "")[:80],
               "n": r["n"],
               "fact_ids": [str(i)] * r["n"]} for i, r in enumerate(rows)]
    ex = [{"name": r["name"], "statement": (r["statement"] or "")[:80],
           "n_themes": r["n"]} for r in existing]
    if len(themes) < 2:
        print(f"未归类主题只有 {len(themes)} 个, 不做宏观聚类"
              f" (已有板块 {len(ex)} 个)")
        return 0
    est = (sum(len(t["name"]) + len(t["statement"]) + 40 for t in themes)
           + sum(len(e["name"]) + len(e["statement"]) for e in ex) + 1200)
    print(f"[macro] 未归类 {len(themes)} 个主题 + 已有 {len(ex)} 个板块, "
          f"prompt 约 {est} 字符 (statement 已截到 80 字)")
    if est > 12000:
        print("[macro] ⚠ 仍偏大(安全线 8000), 这次可能因超时失败; "
              "失败会如实报告而不是挂起")
    if dry:
        return 0
    macros, warns = O.cluster_themes_into_macros(
        themes, model=MODEL, existing_macros=ex)
    for w in warns:
        print(f"  ⚠ {w}")
    if not macros:
        return 0
    n_new = n_reuse = 0
    with connect("facts") as conn:
        for m in macros:
            reused = bool(m.get("reused"))
            # 复用已有板块时**传空 statement/notes = 保留原文**。upsert_macro
            # 用 COALESCE(NULLIF(?,''), statement), 空串就等于"不改"。
            # 这次模型只看到这批未归类主题, 看不到板块里原有的几十个主题,
            # 它重写的 statement 必然是残缺的(只描述新增的那几个)。
            mid = tg.upsert_macro(
                conn, m["name"],
                "" if reused else m.get("statement", ""),
                "" if reused else m.get("notes", ""))
            n_reuse += reused
            n_new += not reused
            for tn in m["theme_names"]:
                conn.execute(
                    "UPDATE theme SET macro_id = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE name = ?", (mid, (tn or "").strip()))
    print(f"[macro] 落库 板块 {len(macros)} 个 (新建 {n_new} / 复用 {n_reuse}), "
          f"覆盖 {sum(len(m['theme_names']) for m in macros)} 个主题")
    return len(macros)


def main() -> int:
    # 必须先声明再读: 下面 add_argument(default=MODEL) 已经用到它了
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个")
    ap.add_argument("--skip-macro", action="store_true")
    ap.add_argument("--macro-only", action="store_true",
                    help="只跑宏观层(回填已完成的场合用)")
    ap.add_argument("--model", default=MODEL, help=f"默认 {MODEL}")
    args = ap.parse_args()

    MODEL = args.model

    tg.init_schema()
    if args.macro_only:
        # 10-06 修: 原来写死 `macro_layer(False)`, **把 --dry 丢了** ——
        # `--macro-only --dry` 会真的发请求。实测撞上过: 想先看 prompt 估算
        # 有多大, 结果一发就是 300 秒起步的真实调用, 而"0 个板块"这个输出
        # 看起来跟真跑过一模一样, 事后根本分不清跑没跑。
        if not args.dry:
            preflight(MODEL, n_calls=1)
        n = macro_layer(args.dry)
        print(f"宏观板块 {n} 个" + ("  (--dry: 只报估算, 没发请求)" if args.dry else ""))
        with connect("facts") as conn:
            s = tg.stats(conn)
        for k, v in s.items():
            print(f"  {k:<20} {v}")
        return 0

    with connect("facts") as conn:
        done = {r[0] for r in conn.execute(
            "SELECT DISTINCT asserted_by FROM theme_fact")}
    dirs = sorted(d for d in WS_ROOT.iterdir() if d.is_dir())
    todo = []
    for d in dirs:
        if (d / "_facts.jsonl").exists() and d.name not in done \
                and d.name not in SKIP_WORKSPACES:
            todo.append(d)
    if args.limit:
        todo = todo[:args.limit]
    print(f"待回填 workspace {len(todo)} 个 (已入库 {len(done)} 个)\n")

    if not args.dry and not args.skip_macro:
        # 粗估: 每个 workspace 至少 1 次主题归纳, 再加 1 次宏观聚类
        preflight(MODEL, n_calls=len(todo) + 1)

    t0 = time.time()
    for i, d in enumerate(todo, 1):
        facts = load_facts(d)
        if not facts:
            continue
        r = run_one(d, facts, args.dry)
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {r['ws']} facts={r['facts']:<4} 块={r['chunks']} "
              f"主题+={r['themes']:<3} 边+={r['links']:<4} 关系+={r['relations']:<3} "
              f"累计 {el/60:.1f}min"
              + (f"  ❌ {r['err']}" if r["err"] else "")
              + (f"  ⚠ {dict(r['warn'])}" if r['warn'] else ""))
        sys.stdout.flush()

    if not args.dry and not args.skip_macro:
        print("\n--- 宏观层(全局) ---")
        n = macro_layer(False)
        print(f"宏观板块 {n} 个")

    with connect("facts") as conn:
        s = tg.stats(conn)
    print("\n--- facts.db 现状 ---")
    for k, v in s.items():
        print(f"  {k:<20} {v}")
    print(f"\n覆盖率 {s['facts_linked']}/{s['facts_total']} "
          f"({s['facts_linked']*100//max(1,s['facts_total'])}%) | "
          f"一条 fact 落多个主题的: {s['multi_theme_facts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
