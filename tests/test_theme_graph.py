"""
tests/test_theme_graph.py — 知识结构图谱的表与查询 (10-05)

锁的是三件最容易悄悄坏掉的事:

1. **主题不属于任何 run**。`theme` / `theme_macro` 上没有 run_id, 唯一带 run
   语义的是 `theme_fact.asserted_by` 且它是**溯源**不是归属。改表时最容易
   的错误就是"顺手加个 run_id 做隔离"—— 那会把累积生长的结构退化成"每次跑
   画一张独立的图", 也就是这个模块头写死要防的那个错误建模。
2. **外键真的连着 fact 层**。`theme_fact.fact_id → fact(id)`, 所以删 fact 会
   自动清掉它的所有主题归属; 编不存在的 fact_id 必须被拒。
3. **跨域查询的 join 键是 source_id(论文), 不是 fact_id**。这两个只差一个键,
   写错时不报错、结果少两个数量级(实测写错时 63 篇横跨论文只产出 1 条边)。

测试跑在**临时库**上(不碰 data/knowledge/facts.db), 但表定义直接 import
生产代码 —— 这样改 schema 会被这些测试挡住。
"""
from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

tg = importlib.import_module("agent.infra.theme_graph")
gq = importlib.import_module("agent.infra.graph_query")

# 生产 schema 原文: fact / fact_source 由 facts 模块自己建, 这里只建最小可用的
FACTS_SCHEMA = """
CREATE TABLE fact (
    id TEXT PRIMARY KEY, content TEXT NOT NULL, fact_type TEXT DEFAULT 'fact',
    subject TEXT DEFAULT '', confidence REAL DEFAULT 1.0);
CREATE TABLE fact_source (
    id INTEGER PRIMARY KEY AUTOINCREMENT, fact_id TEXT NOT NULL,
    source_type TEXT NOT NULL, source_id TEXT NOT NULL, locator TEXT DEFAULT '',
    quote TEXT DEFAULT '', grounded INTEGER DEFAULT 1,
    self_referential INTEGER DEFAULT 0,
    FOREIGN KEY (fact_id) REFERENCES fact(id) ON DELETE CASCADE,
    UNIQUE (fact_id, source_type, source_id));
"""


@pytest.fixture
def conn():
    """真 schema 的内存库。**不 mock schema** —— 改生产 SCHEMA_SQL 会被这里挡住。"""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON")
    c.executescript(FACTS_SCHEMA)
    c.executescript(tg.SCHEMA_SQL)
    yield c
    c.close()


def add(c, fid, content="内容", paper="P1", stype="paper_abstract"):
    c.execute("INSERT OR IGNORE INTO fact (id, content) VALUES (?,?)", (fid, content))
    c.execute(
        "INSERT OR IGNORE INTO fact_source (fact_id, source_type, source_id) "
        "VALUES (?,?,?)", (fid, stype, paper))


# ============================================================
# 模型: 主题不属于 run
# ============================================================

def test_structural_tables_have_no_run_id(conn):
    """硬性护栏: 主题层**没有 run 归属**。

    反向验证: 给 theme 表加一列 run_id, 这条立刻红。
    这是 10-05 踩过的建模错误 —— "一个 workspace 一张主题图" 这个边界是
    我自己造的, 用户从没这么说过。给主题加 run_id 会让结构退化成
    "每轮重画", 而不是累积生长。
    """
    for t in ("theme", "theme_macro", "theme_paradigm", "theme_relation"):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info([{t}])")}
        assert "run_id" not in cols, f"{t} 上不该有 run_id(当前 {cols})"
        assert "workspace" not in cols, f"{t} 上不该有 workspace(当前 {cols})"
    # 唯一允许带 run 语义的是 theme_fact.asserted_by, 且它叫 asserted(断言)不是 owned
    tf = {r[1] for r in conn.execute("PRAGMA table_info([theme_fact])")}
    assert "asserted_by" in tf, tf
    assert "run_id" not in tf and "owner" not in tf, tf


def test_themes_merge_by_name_across_runs(conn):
    """同名主题在不同来源里是**同一个** —— 累积生长的核心语义。"""
    add(conn, "f1", "c1", "P1")
    a = tg.upsert_theme(conn, "贫瘠高原", "第一版判断")
    b = tg.upsert_theme(conn, "贫瘠高原", "第二版判断")
    assert a == b, "同名必须落到同一个 id"
    assert conn.execute("SELECT COUNT(*) FROM theme").fetchone()[0] == 1
    # 后来的 statement 会更新(累积, 不丢信息)
    assert conn.execute(
        "SELECT statement FROM theme WHERE id=?", (a,)).fetchone()[0] == "第二版判断"


def test_norm_theme_name_only_strips_punctuation():
    """归一只动空白和标点, 不动字词 —— 动字词的"智能归一"会把不同主题糊一起。"""
    assert tg.norm_theme_name(" 张量网络、表示法。 ") == "张量网络表示法"
    assert tg.norm_theme_name("张量网络表示法") == "张量网络表示法"
    # 不同的字必须保持不同
    assert tg.norm_theme_name("张量网络表示") != tg.norm_theme_name("张量网络优化")


# ============================================================
# 外键与幂等
# ============================================================

def test_theme_fact_rejects_unknown_fact_id(conn):
    """不存在的 fact_id 必须被拒 —— 外键真的连着 fact 层。"""
    tid = tg.upsert_theme(conn, "T")
    with pytest.raises(sqlite3.IntegrityError):
        tg.link_facts(conn, tid, ["不存在的fact"], "ws")
    conn.rollback()


def test_link_facts_is_idempotent_and_keeps_first_assertor(conn):
    """重复写入不产生重复行, 且 asserted_by 保留**首次**写入的值。"""
    add(conn, "f1")
    tid = tg.upsert_theme(conn, "T")
    tg.link_facts(conn, tid, ["f1"], "ws-A")
    tg.link_facts(conn, tid, ["f1"], "ws-B")
    rows = conn.execute(
        "SELECT fact_id, asserted_by FROM theme_fact").fetchall()
    assert rows == [("f1", "ws-A")], f"实得 {rows}"


def test_deleting_fact_cascades_to_theme_links(conn):
    """删 fact -> 它的所有主题归属自动消失。这条靠外键保证, 不靠应用层自觉。"""
    add(conn, "f1")
    tid = tg.upsert_theme(conn, "T")
    tg.link_facts(conn, tid, ["f1"], "ws")
    conn.execute("DELETE FROM fact WHERE id='f1'")
    assert conn.execute("SELECT COUNT(*) FROM theme_fact").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM theme").fetchone()[0] == 1, "主题本身不该被删"


def test_relation_has_no_type_column(conn):
    """关系边没有 type —— 10-05 删掉的那个噪声标签不许被加回来。"""
    cols = {r[1] for r in conn.execute("PRAGMA table_info([theme_relation])")}
    assert "type" not in cols, cols
    assert {"src_theme", "dst_theme", "note"} <= cols, cols


def test_theme_fact_is_many_to_many(conn):
    """主题↔fact 必须多对多: 一条 fact 落多个主题, 一个主题含多个 fact。"""
    for i in range(3):
        add(conn, f"f{i}")
    ta, tb = tg.upsert_theme(conn, "A"), tg.upsert_theme(conn, "B")
    tg.link_facts(conn, ta, ["f0", "f1"], "ws")
    tg.link_facts(conn, tb, ["f1", "f2"], "ws")
    r = conn.execute("""
        SELECT COUNT(*) FROM (SELECT fact_id FROM theme_fact
                               GROUP BY fact_id HAVING COUNT(DISTINCT theme_id)>1)
    """).fetchone()[0]
    assert r == 1, "f1 应同时落 A 和 B"


# ============================================================
# 查询层
# ============================================================

@pytest.fixture
def wired(monkeypatch, conn):
    """把查询层接到内存库上。"""
    monkeypatch.setattr(gq, "_conn", lambda: _Ctx(conn), raising=False)
    return conn


class _Ctx:
    """让 with _conn() as c 能用, 且 close 时不关掉外部 conn。"""

    def __init__(self, c):
        self.c = c

    def __enter__(self):
        return self.c

    def __exit__(self, *a):
        return False


def _build(wired):
    """一个能区分"共享论文"和"共享 fact"的小图。"""
    # P1 的两条 fact 分别落 A 和 B  -> 跨域关联(共享论文)
    add(wired, "f1", "论文一的方法", "P1")
    add(wired, "f2", "论文一的结果", "P1")
    a = tg.upsert_theme(wired, "方法论")
    b = tg.upsert_theme(wired, "结果")
    tg.link_facts(wired, a, ["f1"], "ws")
    tg.link_facts(wired, b, ["f2"], "ws")
    # 无关的 C
    add(wired, "f3", "另一篇", "P2")
    c = tg.upsert_theme(wired, "别处")
    tg.link_facts(wired, c, ["f3"], "ws")
    wired.commit()


def test_themes_sharing_papers_joins_on_source_not_fact(wired):
    """**核心防回归**: 跨域边 join 的是论文, 不是 fact。

    反向验证: 把 graph_query 里的 `s2.source_id = s1.source_id` 改回
    `tfb.fact_id = tfa.fact_id`, 这条会红(结果变成 0 条)。
    """
    _build(wired)
    pairs = gq.themes_sharing_papers(min_papers=1)
    assert len(pairs) == 1, f"应产出 1 条跨域边, 实得 {pairs}"
    p = pairs[0]
    assert {p["theme_a"], p["theme_b"]} == {"方法论", "结果"}
    assert p["n_papers"] == 1 and p["papers"] == ["P1"]


def test_themes_sharing_facts_is_a_different_question(wired):
    """共享同一条 fact 是"边界划错"的信号, 和共享论文是两回事。"""
    _build(wired)
    assert gq.themes_sharing_facts() == [], "本图里没有共享 fact 的主题对"
    add(wired, "f4", "同一件事", "P3")
    ta = tg.get_theme_id(wired, "方法论")
    tb = tg.get_theme_id(wired, "别处")
    tg.link_facts(wired, ta, ["f4"], "ws")
    tg.link_facts(wired, tb, ["f4"], "ws")
    wired.commit()
    got = gq.themes_sharing_facts()
    assert len(got) == 1 and got[0]["n_facts"] == 1, got


def test_bridges_finds_hub_themes(wired):
    """桥接主题 = 同时和多个主题共享论文。"""
    _build(wired)
    # 让第三面和方法论共享论文 P1(它俩共用同一条 fact)
    add(wired, "f5", "论文一的第三面", "P1")
    c2 = tg.upsert_theme(wired, "第三面")
    tg.link_facts(wired, tg.get_theme_id(wired, "方法论"), ["f5"], "ws")
    tg.link_facts(wired, c2, ["f5"], "ws")
    wired.commit()
    b = gq.bridges(min_papers=1)
    hubs = {x["theme"]: x for x in b}
    assert "方法论" in hubs, f"方法论应是桥接主题, 实得 {b}"
    assert {"结果", "第三面"} <= set(hubs["方法论"]["connects"]), hubs["方法论"]
    assert "别处" not in hubs["方法论"]["connects"], "别处只跟第三面有关, 不该出现在方法论上"


def test_themes_of_paper_shows_how_a_paper_is_split(wired):
    """论文不是节点, 但要能问出"这篇被怎么切开了"。"""
    _build(wired)
    got = {g["theme"]: g["n_facts"] for g in gq.themes_of_paper("P1")}
    assert got == {"方法论": 1, "结果": 1}, got
    grp = next(g for g in gq.themes_of_paper("P1") if g["theme"] == "方法论")
    assert grp["facts"][0]["content"] == "论文一的方法"


def test_tree_reports_unassigned_themes(wired):
    """还没跑宏观层时, 主题挂在 macro=None 根下 —— 不能凭空消失。"""
    _build(wired)
    t = gq.tree()
    assert len(t) == 1 and t[0]["macro"] is None
    assert {x["name"] for x in t[0]["themes"]} == {"方法论", "结果", "别处"}


def test_tree_nests_under_macro_after_macro_assigned(wired):
    mid = tg.upsert_macro(wired, "上层板块", "为什么这几个该放一起")
    tid = tg.upsert_theme(wired, "方法论", macro_id=mid)
    _build(wired)
    wired.execute("UPDATE theme SET macro_id=? WHERE id=?", (mid, tid))
    wired.commit()
    t = gq.tree()
    assert t[0]["macro"] == "上层板块"
    assert {x["name"] for x in t[0]["themes"]} == {"方法论"}
    loose = t[1]
    assert loose["macro"] is None
    assert {x["name"] for x in loose["themes"]} == {"结果", "别处"}


def test_paper_theme_span_counts_papers_bridging_themes(wired):
    _build(wired)
    sp = gq.paper_theme_span()
    assert sp["total_papers"] == 2
    assert sp["by_n_themes"] == {1: 1, 2: 1}
    assert sp["multi_theme_papers"] == 1


def test_orphan_facts_are_reported_not_deleted(wired):
    """没被认领的 fact 仍在库里 —— 里面有定义, 用户说过"定义也是有主题的"。"""
    _build(wired)                      # f1..f3 都有主题
    add(wired, "f9", "某个还没归类的定义")
    wired.commit()
    assert gq.orphan_facts() == ["某个还没归类的定义"]
    assert conn_count(wired, "fact") == 4, "孤儿 fact 不该被删"
    assert conn_count(wired, "theme_fact") == 3


def conn_count(c, table):
    return c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_stats_shape(wired):
    _build(wired)
    s = tg.stats(wired)
    for k in ("macros", "themes", "links", "relations", "facts_total",
              "facts_linked", "facts_unlinked", "themes_without_fact"):
        assert k in s, k
    assert s["themes"] == 3 and s["facts_linked"] == 3
    assert s["themes_without_fact"] == 0
