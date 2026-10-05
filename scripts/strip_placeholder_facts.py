"""
scripts/strip_placeholder_facts.py — 剥掉存量 fact 正文里的 paper_id 标注 (10-05)

背景: `prompts_survey.strip_paper_id_from_claim` 从 9-29 起就在抽取路径上剥标注,
但 `migrate_to_graph.py` 当时另写了一套 4 种形态的正则, **漏了 `[2209.02093]`
裸 id**。存量数据走迁移脚本进来, 绕过了生产路径那套, 于是库里 767 条(17.7%)
fact 正文还带着标注。

为什么这不是洁癖:
    `fact.id` 是内容哈希。claim 末尾多一个 `,[2501.10017]` 和一个 `。` 之外的
    标点差异, 就算成**两条不同的 fact**。实测剥完立刻能精确合并 57 条 ——
    其中 11 条是同一句 `该论文采用硬件高效 ansatz 作为核心方法` 被 11 个
    workspace 各抽一遍, 只因末尾 id 写法不同而裂成 11 行。
    标注不清掉, 后面任何语义去重都在跟自己的噪声打架。

安全前提(已实测, `_tmp_cite_probe.py`):
    767 个方括号 id **全部是本文自己的** arxiv id, 0 个是引用别篇论文,
    0 条 fact 里含多个 id, 767 处全在句尾。所以一刀切剥不会丢真引用。

**先备份再跑。** 合并不可逆, 因此每次合并都写一条 `fact_merge` 审计行,
保留被合并行的原始 content 与 id, 事后可逐条还原。

用法:
    venv\\Scripts\\python.exe scripts\\strip_placeholder_facts.py --dry
    venv\\Scripts\\python.exe scripts\\strip_placeholder_facts.py
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Windows 控制台默认 GBK, 打印 ✅/❌ 会 UnicodeEncodeError —— 而这正好发生在
# **报错分支**上, 等于把错误信息本身给吞了。10-05 实测。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.infra.db import connect  # noqa: E402
from agent.prompts_survey import (  # noqa: E402
    count_placeholder_forms,
    strip_paper_id_from_claim,
)
from agent.tools.facts import fact_id_for  # noqa: E402

MERGE_AUDIT_SQL = """
CREATE TABLE IF NOT EXISTS fact_merge (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kept_id      TEXT NOT NULL,      -- 合并后保留的 fact.id
    dropped_id   TEXT NOT NULL,      -- 被合并掉的 fact.id
    reason       TEXT NOT NULL,      -- placeholder_strip / semantic_merge
    dropped_content TEXT NOT NULL,   -- 被合并行的原始正文(可还原)
    kept_content    TEXT NOT NULL,
    detail       TEXT DEFAULT '',    -- JSON: 判同依据等
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_fact_merge_kept ON fact_merge(kept_id);
CREATE INDEX IF NOT EXISTS idx_fact_merge_dropped ON fact_merge(dropped_id);
"""


def backup(db: Path, tag: str) -> Path:
    dst = db.with_suffix(db.suffix + f".bak-{tag}")
    shutil.copy2(str(db), str(dst))
    return dst


def collect_fixes(conn: sqlite3.Connection) -> list[dict]:
    """找出需要剥标注的 fact, 并算出剥完之后各自的新 id。"""
    rows = conn.execute(
        "SELECT id, content, fact_type, subject, confidence FROM fact"
    ).fetchall()
    fixes = []
    for r in rows:
        content = r["content"] or ""
        if not count_placeholder_forms(content).get(4) and not any(
            count_placeholder_forms(content).values()
        ):
            continue
        clean = strip_paper_id_from_claim(content)
        if clean == content:
            continue
        if not clean:
            # 剥完什么都不剩 = 整条 fact 就是个占位符, 不能留空 content
            continue
        new_id = fact_id_for(clean, r["fact_type"] or "")
        if new_id == r["id"]:
            continue
        fixes.append({
            "old_id": r["id"], "new_id": new_id, "content": content,
            "clean": clean, "fact_type": r["fact_type"] or "", "subject": r["subject"] or "",
            "confidence": r["confidence"],
        })
    return fixes


def plan(conn: sqlite3.Connection, fixes: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    """把 fix 分成两类: 单纯改主键, 以及剥完与既有 fact 撞车(可精确合并)。

    必须**按执行顺序模拟**出最终 id 集合, 不能只跟"当前已存在的 id"比。
    10-05 实测: 只比对现有 id 时, 一条 fact 剥完的新 id 恰好等于**另一条待改
    fact 剥完后的新 id**, 于是它在 plan 阶段看着"不撞车"(走 rename),
    等前面那条先改完占了这个 id, 它自己再改就 UNIQUE 失败。
    两个 id 相同 = 两条 fact 剥完正文一样, 本质是同一个合并, 只是分两次撞上。
    """
    existing = {r[0] for r in conn.execute("SELECT id FROM fact")}
    will_exist = set(existing)
    renames, merges = [], []
    for f in fixes:
        if f["new_id"] in will_exist:
            merges.append((f["new_id"], f["old_id"]))
        else:
            renames.append(f)
            will_exist.add(f["new_id"])   # 后面那条会撞上这一条
    return renames, merges


def apply_rename(conn: sqlite3.Connection, f: dict) -> None:
    """改主键, 并把 theme_fact / fact_source 的引用一起搬过去。

    顺序与外键: 整个过程必须开 `defer_foreign_keys`。
    `connect()` 开了 PRAGMA foreign_keys=ON 且**立即**检查, 于是两种直觉写法
    都会炸(10-05 实测):
        先改子表引用 -> new_id 此刻还不在 fact 表里 -> FOREIGN KEY failed;
        先改 fact.id -> theme_fact / fact_source 立刻悬空, 而 fact_source 的
                        ON DELETE CASCADE 会把子表数据**级联删掉**, 且不可逆。
    正确做法: 推迟到 COMMIT 时统一校验, 中间态允许悬空。
        1. 子表引用改指 new_id
        2. fact.id 改成 new_id
        3. 提交时校验
    提交后再跑一次 PRAGMA foreign_key_check 兜底 —— 不干净就报错, 不静默留下。

    fact_source 有 UNIQUE(fact_id, source_type, source_id), theme_fact 主键是
    (theme_id, fact_id): 搬引用时都可能撞上已存在的同一行。撞了就只留一条 ——
    出处和主题归属都是"证据", 重复没有增益。注意不能用 UPDATE OR IGNORE:
    它对 UNIQUE 冲突是整行**跳过**, 会把该搬的行和无关行混为一谈(实测会漏搬)。
    """
    old, new = f["old_id"], f["new_id"]
    conn.execute("PRAGMA defer_foreign_keys = ON")
    conn.execute("UPDATE fact SET content=? WHERE id=?", (f["clean"], old))

    # ---- fact_source ----
    # UNIQUE(fact_id, source_type, source_id)。逐行搬, 撞了就跳过(同 fact 的
    # 同一出处已有行 —— 出处是"证据", 重复没有增益)。
    for sid, stype, s_id in conn.execute(
            "SELECT id, source_type, source_id FROM fact_source WHERE fact_id=?",
            (old,)).fetchall():
        dup = conn.execute(
            "SELECT id FROM fact_source WHERE fact_id=? AND source_type=? AND source_id=?",
            (new, stype, s_id)).fetchone()
        if dup:
            conn.execute("DELETE FROM fact_source WHERE id=?", (sid,))
        else:
            conn.execute("UPDATE fact_source SET fact_id=? WHERE id=?", (new, sid))

    # ---- theme_fact ----
    # PK(theme_id, fact_id)。**UNIQUE 是立即检查的, defer_foreign_keys 管不着**
    # (它只推迟外键), 所以"先 UPDATE 再去重"必然中途炸 UNIQUE —— 10-05 在副本上
    # 实测两次都撞这个。只能**先解决冲突再改引用**:
    #   目标行不存在 -> 直接改指; 已存在 -> 把 asserted_by 并进去, 删掉旧行。
    # asserted_by 是"这条归属是哪次抽取断言的", 并进去而不是丢一个。
    for rid, theme_id, asserted_by in conn.execute(
            "SELECT rowid, theme_id, asserted_by FROM theme_fact WHERE fact_id=?",
            (old,)).fetchall():
        exist = conn.execute(
            "SELECT rowid, asserted_by FROM theme_fact WHERE theme_id=? AND fact_id=?",
            (theme_id, new)).fetchone()
        if exist is None:
            conn.execute("UPDATE theme_fact SET fact_id=? WHERE rowid=?", (new, rid))
            continue
        merged = _merge_asserted_by(exist[1], asserted_by)
        conn.execute("UPDATE theme_fact SET asserted_by=? WHERE rowid=?", (merged, exist[0]))
        conn.execute("DELETE FROM theme_fact WHERE rowid=?", (rid,))

    conn.execute("UPDATE fact SET id=? WHERE id=?", (new, old))


def _merge_asserted_by(a: str, b: str) -> str:
    """两个溯源值合并去重, 空串当没有。"""
    seen, out = set(), []
    for v in (a or "", b or ""):
        for part in v.split(","):
            part = part.strip()
            if part and part not in seen:
                seen.add(part)
                out.append(part)
    return ",".join(out)


def apply_merge(conn: sqlite3.Connection, kept_id: str, dropped_id: str,
                reason: str = "placeholder_strip") -> None:
    """把 dropped 的出处与主题归属并到 kept, 记审计, 删 dropped。

    `theme_fact` 不用 INSERT OR IGNORE 搬: 撞上 (同一 theme 下已有这条 fact) 时
    OR IGNORE 会**整行跳过**, 于是 dropped 带来的 asserted_by 就这么丢了 ——
    而 asserted_by 是"这条归属是哪次抽取断言的", 丢了补不回来。
    做法: 先把 dropped 的归属抓进临时表, 插不进去的(撞车)再显式合并
    asserted_by(去重后拼接), 空串当没有。
    """
    drop_row = conn.execute(
        "SELECT content FROM fact WHERE id=?", (dropped_id,)).fetchone()
    keep_row = conn.execute(
        "SELECT content FROM fact WHERE id=?", (kept_id,)).fetchone()
    conn.execute(
        "INSERT OR IGNORE INTO fact_source (fact_id, source_type, source_id, locator,"
        " quote, grounded, self_referential) SELECT ?, source_type, source_id, locator,"
        " quote, grounded, self_referential FROM fact_source WHERE fact_id=?",
        (kept_id, dropped_id))
    # theme_fact 逐行搬: 撞车(同 theme 下 kept 已有这条 fact)时合并 asserted_by。
    # INSERT OR IGNORE 在这里会**整行跳过**, 把 dropped 带来的溯源悄悄丢掉。
    for theme_id, asserted_by in conn.execute(
            "SELECT theme_id, asserted_by FROM theme_fact WHERE fact_id=?",
            (dropped_id,)).fetchall():
        exist = conn.execute(
            "SELECT rowid, asserted_by FROM theme_fact WHERE theme_id=? AND fact_id=?",
            (theme_id, kept_id)).fetchone()
        if exist is None:
            conn.execute(
                "INSERT INTO theme_fact (theme_id, fact_id, asserted_by) VALUES (?,?,?)",
                (theme_id, kept_id, asserted_by))
        else:
            conn.execute(
                "UPDATE theme_fact SET asserted_by=? WHERE rowid=?",
                (_merge_asserted_by(exist[1], asserted_by), exist[0]))
    conn.execute(
        "INSERT INTO fact_merge (kept_id, dropped_id, reason, dropped_content,"
        " kept_content) VALUES (?,?,?,?,?)",
        (kept_id, dropped_id, reason,
         (drop_row[0] if drop_row else ""), (keep_row[0] if keep_row else "")))
    conn.execute("DELETE FROM fact WHERE id=?", (dropped_id,))


def fk_snapshot(conn: sqlite3.Connection) -> set[tuple]:
    """当前外键违规的集合(表, rowid, 父表, fkid), 用来做前后对比。"""
    return {tuple(r) for r in conn.execute("PRAGMA foreign_key_check").fetchall()}


def check_integrity(conn: sqlite3.Connection,
                    fk_before: set[tuple] | None = None
                    ) -> tuple[list[str], list[str]]:
    """提交前的自检。改主键是高危操作, 悬空引用/孤儿行必须是**报错**而不是
    安静留下 —— 前几轮 run 就是在中途 IntegrityError / 自检拦下才发现顺序错。

    返回 (阻断项, 提示项)。**分开是关键**: 库里本来就有 8 处 theme_relation
    悬空(早先隔离测试 workspace 时把 theme id 0/1 删了, 边没跟着删)。混进
    阻断项, 这个脚本就永远跑不动; 但直接忽略又等于放过新问题 ——
    所以按集合差集判"是不是本次弄坏的", 老账照样报出来但不拦。
    实测教训: 第一版把提示写进同一个 list, 结果脚本对着"仅提示"也回滚。
    """
    errors, notes = [], []
    bad = fk_snapshot(conn)
    new_fk = bad - (fk_before or set())
    if new_fk:
        errors.append(f"本次新增外键悬空 {len(new_fk)} 处: {sorted(new_fk)[:3]}")
    elif bad:
        notes.append(f"库中本就有 {len(bad)} 处外键悬空(与本次无关): {sorted(bad)[:2]}")
    orphan_fs = conn.execute(
        "SELECT COUNT(*) FROM fact_source WHERE fact_id NOT IN (SELECT id FROM fact)"
    ).fetchone()[0]
    if orphan_fs:
        errors.append(f"fact_source 孤儿 {orphan_fs} 条")
    orphan_tf = conn.execute(
        "SELECT COUNT(*) FROM theme_fact WHERE fact_id NOT IN (SELECT id FROM fact)"
    ).fetchone()[0]
    if orphan_tf:
        errors.append(f"theme_fact 孤儿 {orphan_tf} 条")
    orphan_t = conn.execute(
        "SELECT COUNT(*) FROM theme_fact WHERE theme_id NOT IN (SELECT id FROM theme)"
    ).fetchone()[0]
    if orphan_t:
        errors.append(f"theme_fact 主题悬空 {orphan_t} 条")
    dup = conn.execute(
        "SELECT COUNT(*) FROM (SELECT content FROM fact GROUP BY content"
        " HAVING COUNT(*)>1)").fetchone()[0]
    if dup:
        errors.append(f"仍有 {dup} 组同 content 的 fact(应当已合并)")
    return errors, notes


def run_all(conn: sqlite3.Connection, renames: list[dict],
            merges: list[tuple[str, str]]) -> None:
    """执行全部改主键与合并, 包含"剥完才撞车"的收尾合并。

    单独抽出来是为了**可测**: 这些逻辑原先直接写在 main() 里, 于是
    10-05 的反向验证里"关掉收尾合并"和"把阻断项降级成提示"两条都杀不掉任何
    测试 —— 逻辑根本没被单测覆盖过。抽成函数后才谈得上护栏。
    """
    for f in renames:
        apply_rename(conn, f)
    for kept, dropped in merges:
        apply_merge(conn, kept, dropped)
    collapse_late_duplicates(conn)


def collapse_late_duplicates(conn: sqlite3.Connection) -> int:
    """剥完之后才撞成同 content 的组合, 收尾合并。

    两条 fact 原本各自带**不同**的标注(所以 id 不同、内容也不同), 剥完变成
    同一句。它们既不在 plan 的 existing 里、也不在 plan 的模拟集合里 ——
    因为 plan 比的是 new_id 而不是 new content。10-05 实测: 真库 15 组。
    """
    late = conn.execute(
        "SELECT content FROM fact GROUP BY content HAVING COUNT(*)>1").fetchall()
    n = 0
    for (content,) in late:
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM fact WHERE content=? ORDER BY id", (content,))]
        keeper = ids[0]
        for drop in ids[1:]:
            apply_merge(conn, keeper, drop, reason="placeholder_strip_late")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只报表不写库")
    ap.add_argument("--no-backup", action="store_true", help="跳过备份(不建议)")
    args = ap.parse_args()

    from agent.infra.db import DB_FILES
    db = ROOT / "data" / DB_FILES["facts"]

    with connect("facts") as conn:
        fixes = collect_fixes(conn)
        renames, merges = plan(conn, fixes)
        forms: Counter = Counter()
        for f in fixes:
            for k, v in count_placeholder_forms(f["content"]).items():
                if v:
                    forms[k] += v

        print("=" * 68)
        print("存量 fact 标注剥离" + ("(dry-run, 未写库)" if args.dry else ""))
        print("=" * 68)
        print(f"含占位符的 fact: {len(fixes)} 条")
        print(f"剥后主键不变(纯噪声, 跳过): {len(fixes) - len(renames) - len(merges)}")
        print(f"需要改主键: {len(renames)}    剥完与既有 fact 撞车(可精确合并): {len(merges)}")
        if args.dry:
            print("\ndry-run 结束, 未写库。")
            return 0

        if not args.no_backup:
            dst = backup(db, datetime.now().strftime("%m%d-%H%M"))
            print(f"\n已备份 -> {dst.name}")
        # DDL 单独一次 executescript: 它会隐式提交, 混在数据操作里会让
        # "回滚了数据但表建出来了" 这种半截状态出现(上一轮 run 就这样)。
        conn.executescript(MERGE_AUDIT_SQL)
        # defer_foreign_keys 是**每个事务**重置的, 且必须在 BEGIN **之后**设置,
        # 否则事务一开始它就失效了。10-05 实测: 顺序写反 -> 8 处外键悬空,
        # 是自检拦下来的(见 check_integrity), 没有污染数据。
        conn.execute("BEGIN")
        conn.execute("PRAGMA defer_foreign_keys = ON")
        fk_before = fk_snapshot(conn)
        run_all(conn, renames, merges)

        problems, notes = check_integrity(conn, fk_before)
        for n in notes:
            print(f"[提示] {n}")
        if problems:
            conn.rollback()
            print("\n[自检不通过, 已回滚]")
            for p in problems:
                print(f"   - {p}")
            return 1
        conn.commit()
        post, post_notes = check_integrity(conn, fk_before)
        if post:
            print("\n[提交后自检不通过(数据已写, 请用备份还原)]:")
            for p in post:
                print(f"   - {p}")
            return 1

        left = sum(
            sum(count_placeholder_forms(r[0]).values())
            for r in conn.execute("SELECT content FROM fact"))
        total = conn.execute("SELECT COUNT(*) FROM fact").fetchone()[0]
        merged = conn.execute("SELECT COUNT(*) FROM fact_merge").fetchone()[0]
        print(f"完成: fact {total} 条, 残留占位符 {left} 处")
        print(f"合并审计写入 fact_merge {merged} 条")
        print("自检通过: 无悬空外键 / 无孤儿行 / 无同 content 重复")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
