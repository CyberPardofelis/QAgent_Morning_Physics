"""
scripts/quarantine_pseudo_themes.py — 隔离占位符 workspace 造出的伪主题 (10-05)

背景: workspace `640ab2bae07b` 的 topic 就是 "Test", 87 条 fact 里 79 条(96%)
是占位符内容 —— "论文核心方法为 method 4" / "该论文摘要概述为 Abstract 4" /
"论文作者信息未知,署名情况缺失"。它聚类产出的主题全是**元主题**(方法贡献骨架 /
技术路线识别 / 贡献增量判定 / 未解问题界定 / 来源可信审计 / 实证有效性检验),
讲的是"怎么读一篇论文"而不是物理。

危害不在于它们假, 而在于它们**恰好是图谱里最强的跨域边**: 那批论文的每条 fact
都被塞进这几个主题, 于是任意两个伪主题之间都"共享 5 篇论文", 直接把跨域边
榜首占满。真边(贫瘠高原那几条 4-5 篇)被挤到看不见。

**按 workspace 隔离, 不按主题名**
    硬编码主题名治标 —— 换个模型跑一遍名字就变了(实测: 第一次是"技术路线识别/
    贡献增量判定/未解问题界定/...", 第二次是"方法贡献骨架/未来问题缺口/..."),
    清理脚本第二遍一个都删不到。这里改成: 删掉该 workspace 断言的**全部**归属边,
    再删因此变空的主题。与名字无关, 幂等。

**只删主题和归属边, 不删 fact 本身** —— fact 留在库里, 将来真需要还能重新归类。
删 theme 时 theme_relation 靠外键 CASCADE 自动跟着走。

配套: scripts/backfill_graph.py 的 SKIP_WORKSPACES 必须含这个 workspace。
    否则续跑判据(theme_fact 里有没有 asserted_by)在清理后会判定它"没跑过",
    又给它跑一遍, 垃圾主题复活 —— 这个坑真踩过一次。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

os.chdir(ROOT)
import sqlite3  # noqa: E402

# 占位符特征: 把字段名+序号当成了正文
PLACEHOLDER = re.compile(
    r"(method|contrib|outlook|abstract|title|author)\s*\d+"
    r"|占位|metadata|Abstract\s*\d|Method\s*\d", re.I)

# 整个隔离的 workspace。三个都是**测试/占位**数据, 不是真语料:
#   640ab2bae07b  topic="Test", 87 条 fact 里 79 条(96%)是 "论文核心方法为 method 4"
#                这种把字段名当正文的垃圾, 聚出来的全是"技术路线识别/方法贡献骨架"
#                这类**元主题**; 更糟的是每条 fact 都进了那几个主题, 任意两个之间
#                都"共享 5 篇论文", 把跨域边榜首占满。
#   1614c621f2dc  topic="测试 P6.3 持续脏", 2 条 fact 全是字面量 "test [2401.99999]"
#   ad47e85173c1  topic="测试 P6.3 自动重写", 同上
# 2401.99999 根本不是真实 arXiv id, 是测试时编的。
QUARANTINE_WS = ("640ab2bae07b", "1614c621f2dc", "ad47e85173c1")

# 真实主题, 只删它里面沾的脏边(这些 fact 来自别的 workspace, 主题要留)
PRUNE_ONLY = ["量子态验证"]


def main() -> int:
    db = ROOT / "data" / "knowledge" / "facts.db"
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")

    print("=" * 70)
    print(f"隔离 {len(QUARANTINE_WS)} 个测试/占位 workspace 断言的归属边")
    print("=" * 70)
    ph = ",".join("?" * len(QUARANTINE_WS))
    affected = c.execute(f"""
        SELECT tf.asserted_by AS ws, t.name AS theme, COUNT(*) AS n
          FROM theme_fact tf JOIN theme t ON t.id = tf.theme_id
         WHERE tf.asserted_by IN ({ph})
         GROUP BY tf.asserted_by, t.name ORDER BY n DESC""",
                         QUARANTINE_WS).fetchall()
    for r in affected:
        print(f"  {r['n']:>4} 条  [{r['ws']}] {r['theme']}")

    # 1) 删这些 workspace 断言的全部归属边
    n_links = c.execute(
        f"SELECT COUNT(*) FROM theme_fact WHERE asserted_by IN ({ph})",
        QUARANTINE_WS).fetchone()[0]
    c.execute(f"DELETE FROM theme_fact WHERE asserted_by IN ({ph})", QUARANTINE_WS)
    print(f"\n删掉归属边 {n_links} 条")

    # 2) 删因此变空的主题。判据是"删完之后一条 theme_fact 都没有"——
    #    混合主题(还挂着别处的 fact)会自动留下, 不会误删。
    gone = [r[0] for r in c.execute("""
        SELECT t.name FROM theme t
         WHERE NOT EXISTS (SELECT 1 FROM theme_fact tf WHERE tf.theme_id = t.id)""")]
    if gone:
        c.executemany("DELETE FROM theme WHERE name = ?", [(g,) for g in gone])
    print(f"随之消失的空主题 {len(gone)} 个: {'、'.join(gone)}")

    # 3) 别的 workspace 里的脏边, 只删边留主题
    for name in PRUNE_ONLY:
        row = c.execute("SELECT id FROM theme WHERE name = ?", (name,)).fetchone()
        if not row:
            continue
        tid = row["id"]
        bad = [r["fact_id"] for r in c.execute("""
            SELECT tf.fact_id AS fact_id, f.content AS txt
              FROM theme_fact tf JOIN fact f ON f.id = tf.fact_id
             WHERE tf.theme_id = ?""", (tid,)).fetchall()
            if PLACEHOLDER.search(r["txt"] or "")]
        if bad:
            c.executemany("DELETE FROM theme_fact WHERE theme_id=? AND fact_id=?",
                          [(tid, b) for b in bad])
        print(f"  {name}: 删掉 {len(bad)} 条脏边, 主题保留")
    c.commit()

    print("\n--- 清理后 ---")
    for k, q in [("主题", "SELECT COUNT(*) FROM theme"),
                 ("归属边", "SELECT COUNT(*) FROM theme_fact"),
                 ("关系边", "SELECT COUNT(*) FROM theme_relation"),
                 ("fact(应不变)", "SELECT COUNT(*) FROM fact"),
                 ("已挂主题的 fact", "SELECT COUNT(DISTINCT fact_id) FROM theme_fact")]:
        print(f"  {k:<18} {c.execute(q).fetchone()[0]}")
    print("\n被隔离的 fact 本身仍在 fact 库里(没有主题归属), "
          "可用 graph_query.orphan_facts() 查到并重新归类。")
    print("⚠ backfill_graph.py 的 SKIP_WORKSPACES 必须含这个 workspace, "
          "否则续跑会再跑一遍, 垃圾主题复活。")
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
