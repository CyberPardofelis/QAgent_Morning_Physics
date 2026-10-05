"""
agent.infra.graph_query — 结构图谱的查询层 (10-05)

建了图还不够, 得能问出跨域的东西。这一层提供三类问题:

  A. 主题 ↔ 主题 有没有关系?      `themes_sharing_papers` / `themes_sharing_facts`
  B. 某篇论文横跨哪些主题?        `papers_of_theme` / `themes_of_paper`
  C. 全局结构长什么样?            `tree` / `bridges` / `orphan_facts`

**为什么"共享论文"是跨域关联的主力**: 实测 233 篇论文里 78 篇(33%)被 2+ 个
workspace 反复抽取, 而跨 workspace 的**相同 fact 文本只有 3 条**。也就是说
语料里天然重叠的是**论文**不是 fact —— 同一篇论文在一次次抽取里被切出不同侧面。
所以跨域的桥是: 主题A 和主题B 各自吃下了同一篇论文的不同 fact。
这也正是用户建模时说的"一篇论文可以放在多个主题之内"。

三层模型(见 theme_graph):
    宏观主题 → 主题 → 核心对象(fact), 论文是 fact 的出处, **不是节点**
"""
from __future__ import annotations

import sqlite3
from typing import Any

from . import theme_graph as tg


def _conn():
    from agent.infra.db import connect
    return connect("facts")


# ============================================================
# A. 主题之间
# ============================================================

def themes_sharing_papers(min_papers: int = 1, limit: int = 200) -> list[dict[str, Any]]:
    """两个主题之间**共享论文** -> 这就是跨域关联的边。

    注意这不是"两个主题相似", 而是"同一篇论文的不同侧面分别落进了这两个主题"。
    比词向量相似度可靠得多: 跨领域的主题分成两簇正是因为它们不共享词汇,
    而相似度恰恰奖励词汇重合(10-05 实测前 26 对跨领域 0 对)。
    """
    with _conn() as c:
        # 关键: join 的是 **source_id(论文)**, 不是 fact_id。
        # 写错成 fact_id 的话算出来是"两个主题共享同一条 fact"(边界划错的信号),
        # 而不是"两个主题共享同一篇论文"(跨域关联)。两者用的是同一对表, 只差
        # 一个 join 键, 跑在真数据上才看得出来: 写错时 63 篇横跨论文只产出 1 条边。
        rows = c.execute("""
            SELECT ta.name AS a, tb.name AS b,
                   COUNT(DISTINCT s1.source_id) AS n_papers,
                   GROUP_CONCAT(DISTINCT s1.source_id) AS papers
              FROM theme_fact tfa
              JOIN fact_source s1  ON s1.fact_id  = tfa.fact_id
              JOIN fact_source s2  ON s2.source_id = s1.source_id
              JOIN theme_fact tfb  ON tfb.fact_id = s2.fact_id
              JOIN theme ta ON ta.id = tfa.theme_id
              JOIN theme tb ON tb.id = tfb.theme_id
             WHERE tfa.theme_id < tfb.theme_id
             GROUP BY tfa.theme_id, tfb.theme_id
            HAVING COUNT(DISTINCT s1.source_id) >= ?
             ORDER BY n_papers DESC, a, b
             LIMIT ?
        """, (min_papers, limit)).fetchall()
    return [{"theme_a": r[0], "theme_b": r[1], "n_papers": r[2],
             "papers": (r[3] or "").split(",")} for r in rows]


def themes_sharing_facts(limit: int = 200) -> list[dict[str, Any]]:
    """两个主题共享**同一条 fact** -> 真正的语义重叠(边已经画出来了的那种)。

    和 `themes_sharing_papers` 互补: 共享论文是"同一工作的不同侧面",
    共享 fact 是"同一件事被归到了两块"(说明边界划错了, 该考虑合并主题)。
    """
    with _conn() as c:
        rows = c.execute("""
            SELECT ta.name, tb.name, COUNT(*) AS n
              FROM theme_fact tfa
              JOIN theme_fact tfb ON tfb.fact_id = tfa.fact_id
              JOIN theme ta ON ta.id = tfa.theme_id
              JOIN theme tb ON tb.id = tfb.theme_id
             WHERE tfa.theme_id < tfb.theme_id
             GROUP BY tfa.theme_id, tfb.theme_id
             ORDER BY n DESC LIMIT ?
        """, (limit,)).fetchall()
    return [{"theme_a": r[0], "theme_b": r[1], "n_facts": r[2]} for r in rows]


def bridges(min_papers: int = 2, limit: int = 50) -> list[dict[str, Any]]:
    """桥接主题: 同时和**多个**其他主题共享论文的主题。

    这就是"这个领域里哪几块在互相借方法"的答案 —— 也是跨域关联最值得看的地方。
    """
    with _conn() as c:
        rows = c.execute("""
            WITH pairs AS (
                SELECT tfa.theme_id AS a, tfb.theme_id AS b,
                       COUNT(DISTINCT s1.source_id) AS n
                  FROM theme_fact tfa
                  JOIN fact_source s1  ON s1.fact_id  = tfa.fact_id
                  JOIN fact_source s2  ON s2.source_id = s1.source_id
                  JOIN theme_fact tfb  ON tfb.fact_id = s2.fact_id
                 WHERE tfa.theme_id < tfb.theme_id
                 GROUP BY tfa.theme_id, tfb.theme_id
                HAVING COUNT(DISTINCT s1.source_id) >= ?
            )
            SELECT t.name, COUNT(DISTINCT p.b) AS n_branches,
                   GROUP_CONCAT(DISTINCT tn.name) AS connects
              FROM pairs p
              JOIN theme t  ON t.id  = p.a
              JOIN theme tn ON tn.id = p.b
             GROUP BY p.a
             ORDER BY n_branches DESC LIMIT ?
        """, (min_papers, limit)).fetchall()
    return [{"theme": r[0], "n_branches": r[1],
             "connects": (r[2] or "").split(",")} for r in rows]


# ============================================================
# B. 论文(不是节点, 是 fact 的出处)
# ============================================================

def themes_of_paper(paper_id: str) -> list[dict[str, Any]]:
    """一篇论文落在哪些主题里 —— 以及**每个主题吃下了它的哪几条 fact**。

    后面这半句是关键: 只知道"这篇论文在 3 个主题里"没什么用, 知道
    "纠缠浓缩协议 12 条在主题A / 分层隐形传态 8 条在主题B / 可行性 5 条在主题C"
    才知道这个研究对象是怎么被切开的。
    """
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("""
            SELECT t.name AS theme, f.content, f.fact_type
              FROM fact_source s
              JOIN fact f        ON f.id = s.fact_id
              JOIN theme_fact tf ON tf.fact_id = f.id
              JOIN theme t       ON t.id = tf.theme_id
             WHERE s.source_id = ?
        """, (paper_id,)).fetchall()
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["theme"], []).append(
            {"fact_type": r["fact_type"], "content": r["content"]})
    return [{"theme": k, "n_facts": len(v), "facts": v} for k, v in out.items()]


def papers_of_theme(theme_name: str, limit: int = 200) -> list[str]:
    with _conn() as c:
        rows = c.execute("""
            SELECT DISTINCT s.source_id
              FROM theme_fact tf
              JOIN fact_source s ON s.fact_id = tf.fact_id
              JOIN theme t ON t.id = tf.theme_id
             WHERE t.name = ? LIMIT ?
        """, (theme_name, limit)).fetchall()
    return [r[0] for r in rows]


# ============================================================
# B2. 主题的核心对象(L3) —— 三层树的叶子
# ============================================================

def facts_by_theme(theme_name: str | None = None,
                   limit: int = 20000) -> dict[str, list[dict[str, Any]]]:
    """主题名 -> 该主题下的核心对象(fact)。

    主题地图 UI 要**一次**把三层树建起来, 所以 theme_name=None 时返回全部主题的
    映射(一条 SQL, 不是每个主题查一次 —— 182 个主题逐个查就是 182 次往返)。
    详情面板只看单个主题时传 theme_name, 走**同一个函数、同一段解析**: 不另开
    一个 facts_of_theme, 免得两处实现各自漂移(10-05 剥占位符就是生产路径认 5 种
    形态、迁移脚本只认 4 种, 结果库里留下 17.7% 没剥干净的 fact)。

    limit 是**全局**上限而非每主题上限: 主题数随语料增长, "每主题 200"在 182 个
    主题上就是 36000 行, 那不是保护是雪崩。
    """
    sql = """
        SELECT t.name AS theme, f.id AS fact_id, f.content, f.fact_type,
               (SELECT COUNT(*) FROM fact_source s WHERE s.fact_id = f.id) AS n_sources
          FROM theme_fact tf
          JOIN theme t ON t.id = tf.theme_id
          JOIN fact  f ON f.id = tf.fact_id
    """
    params: list[Any] = []
    if theme_name:
        sql += " WHERE t.name = ?"
        params.append(theme_name)
    sql += " ORDER BY t.name, f.id LIMIT ?"
    params.append(limit)

    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(sql, params).fetchall()
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["theme"], []).append(
            {"fact_id": r["fact_id"], "content": r["content"],
             "fact_type": r["fact_type"], "n_sources": r["n_sources"]})
    return out


# ============================================================
# C. 全局结构
# ============================================================

def tree() -> list[dict[str, Any]]:
    """宏观主题 → 主题 的树。没建宏观层时 macro 为 None, 主题直接挂根。"""
    with _conn() as c:
        c.row_factory = sqlite3.Row
        macros = c.execute(
            "SELECT id, name, statement FROM theme_macro ORDER BY name").fetchall()
        out = []
        for m in macros:
            th = c.execute("""
                SELECT t.name, t.statement,
                       (SELECT COUNT(*) FROM theme_fact tf WHERE tf.theme_id=t.id) AS n_facts,
                       (SELECT COUNT(DISTINCT s.source_id) FROM theme_fact tf
                          JOIN fact_source s ON s.fact_id=tf.fact_id
                         WHERE tf.theme_id=t.id) AS n_papers
                  FROM theme t WHERE t.macro_id=? ORDER BY n_facts DESC
            """, (m["id"],)).fetchall()
            out.append({"macro": m["name"], "statement": m["statement"],
                        "themes": [dict(r) for r in th]})
        loose = c.execute("""
            SELECT t.name, t.statement,
                   (SELECT COUNT(*) FROM theme_fact tf WHERE tf.theme_id=t.id) AS n_facts
              FROM theme t WHERE t.macro_id IS NULL ORDER BY n_facts DESC
        """).fetchall()
        if loose:
            out.append({"macro": None, "statement": "(还没跑宏观聚类)",
                        "themes": [dict(r) for r in loose]})
    return out


def orphan_facts(limit: int = 50) -> list[str]:
    """还没被任何主题认领的 fact。

    这些不是垃圾: 里面有定义、有元描述、也有真正的新方向。用户 10-05 明确说过
    "定义也可以放进主题里面, 因为定义也是有主题的"。
    """
    with _conn() as c:
        rows = c.execute("""
            SELECT f.content FROM fact f
             WHERE NOT EXISTS (SELECT 1 FROM theme_fact tf WHERE tf.fact_id=f.id)
             LIMIT ?
        """, (limit,)).fetchall()
    return [r[0] for r in rows]


def paper_theme_span() -> dict[str, Any]:
    """论文横跨多少主题的分布 —— 图谱「连通度」的体检指标。"""
    with _conn() as c:
        rows = c.execute("""
            SELECT s.source_id, COUNT(DISTINCT tf.theme_id) AS n
              FROM fact_source s
              JOIN theme_fact tf ON tf.fact_id = s.fact_id
             GROUP BY s.source_id
        """).fetchall()
    d: dict[int, int] = {}
    for _pid, n in rows:
        d[n] = d.get(n, 0) + 1
    return {"total_papers": len(rows), "by_n_themes": dict(sorted(d.items())),
            "multi_theme_papers": sum(v for k, v in d.items() if k >= 2)}


def overview() -> dict[str, Any]:
    with _conn() as c:
        s = tg.stats(c)
    s["paper_span"] = paper_theme_span()
    return s
