"""
agent.tools.facts — 跨来源的原子事实库 (9-29)

背景 (来自一次讨论 + 随后的实测):
    fact 是所有下游任务的通用中间层 —— 综述章节、论文对比、问答都从 fact 起。
    但现状是每次现抽: `analyze_paper` 抽一遍, `survey_generate` 再抽一遍,
    prompt 还不一样, 抽出来的东西不一致。

    更要紧的是: 实测发现**编造出在抽取阶段**(旧 prompt 会顺着
    "ion trap + nonorthogonal" 自行补出"离子附加能级"这种原文没有的机制)。
    而一条 fact 一旦有了**多个独立出处**, 这种凭空引申就会暴露成孤例 ——
    只有一个可疑来源 vs 三个独立来源都说有, 可信度完全不同。
    所以 [fact, paper] 这种多出处结构不只是"更全", 它是**可审计**的前提。

设计要点(都来自讨论中定的取舍):
    1. fact 与 source 分表。一条 fact 可有多个出处, 单表做不到。
    2. `fact.id` 用**内容哈希**, 绝不用 `{paper_id}::{序号}`。
       9-29 修过一个 P0: 那个前缀方案在单篇论文内就有 97% 映射错误;
       跨来源后一条 fact 属于多个 paper, 前缀根本不成立。
    3. **只追加, 不做自动合并**。自动去重的价值很大(合并证据), 但阈值错了
       合并错**不可逆**(合错了无法知道原来三条分别是什么)。先积累看真实
       重复率, 再定策略。
    4. 保留 grounded 标记(落地审查结果)。fact 库大了以后要能筛掉不可信的。
    5. 与 memories.db 分开: 记忆是"用户想记住的主观偏好", fact 是"从来源
       抽出的客观命题"。检索层以后可以共享, 但存储和生命周期必须分开 ——
       混在一起会出现"用户说'我喜欢块排版'被当成文献事实引用"。

当前接入点(第一版只做一条路径, 验证假设后再扩):
    survey 的 fact 抽取 -> 写库
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# Schema
# ============================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fact (
    id          TEXT PRIMARY KEY,          -- 内容哈希(不含 paper_id 语义)
    content     TEXT NOT NULL,             -- 去语境化的自足陈述
    fact_type   TEXT DEFAULT 'fact',       -- method/result/conclusion/limitation/hypothesis/claim
    subject     TEXT DEFAULT '',           -- 讨论的对象/概念, 供按概念聚合检索
    confidence  REAL DEFAULT 1.0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_source (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id      TEXT NOT NULL,
    source_type  TEXT NOT NULL,            -- paper_abstract/paper_fulltext/conversation/manual/derived
    source_id    TEXT NOT NULL,            -- arxiv id / conversation id / paper_id
    locator      TEXT DEFAULT '',          -- 章节/页码/消息序号
    quote        TEXT DEFAULT '',          -- 原文摘录(grounding 依据)
    grounded     INTEGER DEFAULT 1,        -- 落地审查是否通过
    self_referential INTEGER DEFAULT 0,    -- claim 是否带"本文/作者"等自指
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fact_id) REFERENCES fact(id) ON DELETE CASCADE,
    UNIQUE (fact_id, source_type, source_id)      -- 同一来源重复抽取不重复入库
);

CREATE INDEX IF NOT EXISTS idx_fact_subject     ON fact(subject);
CREATE INDEX IF NOT EXISTS idx_fact_type        ON fact(fact_type);
CREATE INDEX IF NOT EXISTS idx_fact_source_sid  ON fact_source(source_id);
CREATE INDEX IF NOT EXISTS idx_fact_source_fid  ON fact_source(fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_source_type ON fact_source(source_type);
"""

# 同一 fact 在不同来源下的重复计数(判断"几个独立工作都这么说")
# 注意顺序: WHERE -> GROUP BY -> HAVING -> ORDER BY。
# n_sources 是聚合别名, 所以 min_sources 过滤必须走 HAVING, 且要排在 ORDER BY 之前。
_SOURCE_COUNT_SQL = """
SELECT f.id, f.content, f.fact_type, f.subject, f.confidence,
       COUNT(DISTINCT s.source_id) AS n_sources,
       GROUP_CONCAT(DISTINCT s.source_type) AS source_types
FROM fact f LEFT JOIN fact_source s ON s.fact_id = f.id
{where}
GROUP BY f.id
{having}{order}
"""


def init_schema() -> None:
    """建表(幂等)。"""
    from agent.infra.db import connect
    with connect("facts") as conn:
        conn.executescript(SCHEMA_SQL)
    logger.info("[facts] facts.db schema 初始化完成")


# ============================================================
# id 生成
# ============================================================

_PUNCT_RE = re.compile(r"[\s　，。、；：！？,.;:!?\"'“”‘’()（）\[\]【】]+")


def fact_id_for(content: str, fact_type: str = "") -> str:
    """
    由内容算出稳定 id(内容哈希)。

    设计理由:
        - 绝不能含 paper_id —— 跨来源后一条 fact 属于多个 paper
        - 内容相同 → id 相同 → 天然把"同一命题的多处抽取"收敛成一条 fact,
          多个出处挂在 fact_source 上。这正是 [fact, paper] 结构的核心。
        - 稳定: 同样的内容再抽一次不会产生新行。

    归一化: 去标点空白 + 小写, 避免"同一句话不同标点"算成两条。
    """
    norm = _PUNCT_RE.sub("", (content or "").lower())
    if not norm:
        # 空内容也要有个稳定 id, 否则 insert 会失败
        norm = hashlib.sha1((content or "").encode("utf-8")).hexdigest()
    h = hashlib.sha1(f"{fact_type}|{norm}".encode()).hexdigest()
    return f"fact_{h[:16]}"


# ============================================================
# 写入
# ============================================================

def add_fact(
    content: str,
    fact_type: str = "fact",
    subject: str = "",
    source_type: str = "paper_abstract",
    source_id: str = "",
    locator: str = "",
    quote: str = "",
    grounded: bool = True,
    self_referential: bool = False,
    confidence: float = 1.0,
) -> dict[str, Any]:
    """
    追加一条 fact 及其一个出处。**只追加, 不做自动合并。**

    同内容 + 同来源的重复抽取会被 UNIQUE 约束挡掉(幂等);
    同内容 + 不同来源 → 复用同一条 fact, 追加第二个出处(合并证据)。

    Returns: {"fact_id": str, "created": bool, "source_added": bool}
    """
    if not content or not content.strip():
        raise ValueError("content 不能为空")
    if not source_id:
        raise ValueError("source_id 不能为空(fact 必须可溯源)")

    init_schema()
    fid = fact_id_for(content, fact_type)
    now_iso = None

    from agent.infra.db import connect
    with connect("facts") as conn:
        conn.row_factory = __import__("sqlite3").Row
        row = conn.execute("SELECT id FROM fact WHERE id = ?", (fid,)).fetchone()
        created = row is None
        if created:
            conn.execute(
                "INSERT INTO fact (id, content, fact_type, subject, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (fid, content.strip(), fact_type, subject or "", confidence),
            )
        else:
            conn.execute(
                "UPDATE fact SET updated_at = CURRENT_TIMESTAMP, "
                "subject = CASE WHEN subject = '' THEN ? ELSE subject END WHERE id = ?",
                (subject or "", fid),
            )

        src = conn.execute(
            "SELECT id FROM fact_source WHERE fact_id = ? AND source_type = ? AND source_id = ?",
            (fid, source_type, source_id),
        ).fetchone()
        source_added = src is None
        if source_added:
            conn.execute(
                "INSERT INTO fact_source "
                "(fact_id, source_type, source_id, locator, quote, grounded, self_referential) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (fid, source_type, source_id, locator or "", quote or "",
                 1 if grounded else 0, 1 if self_referential else 0),
            )
        _ = now_iso

    return {"fact_id": fid, "created": created, "source_added": source_added}


def add_facts_from_extraction(
    facts: list[dict[str, Any]],
    source_type: str = "paper_abstract",
    source_id_key: str = "paper_id",
    only_grounded: bool = True,
) -> dict[str, Any]:
    """
    把一次 fact 抽取的结果整批入库。

    Args:
        facts: action_extract_facts 产出的 fact 列表(含 paper_id/claim/
              support_quote/_grounded/_self_referential)
        source_type: 来源类型
        only_grounded: True = 只收通过落地审查的(未通过的记录进库但标 grounded=0,
                       方便事后复盘; False = 一律收)

    Returns: {"total": n, "inserted": n, "skipped": n, "ungrounded": n, "new_facts": n}
    """
    init_schema()
    stats = {"total": len(facts), "inserted": 0, "skipped": 0,
             "ungrounded": 0, "new_facts": 0, "multi_source": 0}
    for f in facts:
        content = (f.get("claim") or "").strip()
        sid = f.get(source_id_key) or ""
        if not content or not sid:
            stats["skipped"] += 1
            continue
        grounded = f.get("_grounded", True) is not False
        if not grounded:
            stats["ungrounded"] += 1
        if only_grounded and not grounded:
            stats["skipped"] += 1
            continue
        try:
            r = add_fact(
                content=content,
                fact_type=f.get("fact_type", "fact"),
                subject=(f.get("subject") or ""),
                source_type=source_type,
                source_id=sid,
                locator=f.get("locator", ""),
                quote=(f.get("support_quote") or ""),
                grounded=grounded,
                self_referential=bool(f.get("_self_referential", False)),
            )
            stats["inserted"] += 1
            if r["created"]:
                stats["new_facts"] += 1
            if r["source_added"] and not r["created"]:
                stats["multi_source"] += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("[facts] 入库失败: %s", e)
            stats["skipped"] += 1
    return stats


# ============================================================
# 读取
# ============================================================

def get_fact(fact_id: str) -> dict[str, Any] | None:
    """取一条 fact(含它的全部出处)。"""
    import sqlite3
    init_schema()
    from agent.infra.db import connect
    with connect("facts") as conn:
        conn.row_factory = sqlite3.Row
        f = conn.execute("SELECT * FROM fact WHERE id = ?", (fact_id,)).fetchone()
        if not f:
            return None
        out = dict(f)
        out["sources"] = [dict(r) for r in conn.execute(
            "SELECT * FROM fact_source WHERE fact_id = ? ORDER BY id", (fact_id,)
        ).fetchall()]
    return out


def list_facts(
    subject: str = "",
    fact_type: str = "",
    source_id: str = "",
    source_type: str = "",
    min_sources: int = 0,
    only_grounded: bool = False,
    limit: int = 100,
    order_by: str = "n_sources",
) -> list[dict[str, Any]]:
    """
    列 fact, 带出处计数。

    Args:
        min_sources: 只看有 >= N 个独立出处的 —— 即"几个来源都这么说"。
                     这是 fact 库相对"每篇论文各存一份"的核心增量。
        only_grounded: 过滤掉未通过落地审查的
        order_by: "n_sources" | "recent" | "subject"
    """
    import sqlite3
    init_schema()
    conds, params = [], []
    if subject:
        conds.append("f.subject = ?")
        params.append(subject)
    if fact_type:
        conds.append("f.fact_type = ?")
        params.append(fact_type)
    if source_id:
        conds.append("EXISTS (SELECT 1 FROM fact_source x WHERE x.fact_id = f.id AND x.source_id = ?)")
        params.append(source_id)
    if source_type:
        conds.append("EXISTS (SELECT 1 FROM fact_source x WHERE x.fact_id = f.id AND x.source_type = ?)")
        params.append(source_type)
    if only_grounded:
        conds.append("NOT EXISTS (SELECT 1 FROM fact_source x WHERE x.fact_id = f.id AND x.grounded = 0)")

    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    order = {
        "n_sources": "ORDER BY n_sources DESC, f.created_at DESC",
        "recent": "ORDER BY f.created_at DESC",
        "subject": "ORDER BY f.subject, n_sources DESC",
    }.get(order_by, "ORDER BY n_sources DESC, f.created_at DESC")
    # having 的 ? 只在 min_sources 给了值时才出现在 SQL 里, 参数要同步
    having = "HAVING COUNT(DISTINCT s.source_id) >= ? " if min_sources else ""
    sql = _SOURCE_COUNT_SQL.format(where=where, having=having, order=order) + "LIMIT ?"
    params_all = list(params) + ([min_sources] if min_sources else []) + [limit]

    from agent.infra.db import connect
    with connect("facts") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params_all).fetchall()
    return [dict(r) for r in rows]


def get_sources(fact_id: str) -> list[dict[str, Any]]:
    """取某条 fact 的全部出处。"""
    import sqlite3
    init_schema()
    from agent.infra.db import connect
    with connect("facts") as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM fact_source WHERE fact_id = ? ORDER BY id", (fact_id,)
        ).fetchall()]


def stats() -> dict[str, Any]:
    """fact 库概览(给 UI / 诊断用)。"""
    import sqlite3
    init_schema()
    from agent.infra.db import connect, path_of
    p = path_of("facts")
    out: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return out
    with connect("facts") as conn:
        conn.row_factory = sqlite3.Row
        out["n_facts"] = conn.execute("SELECT COUNT(*) FROM fact").fetchone()[0]
        out["n_sources"] = conn.execute("SELECT COUNT(*) FROM fact_source").fetchone()[0]
        out["multi_source"] = conn.execute(
            "SELECT COUNT(*) FROM (SELECT fact_id FROM fact_source "
            "GROUP BY fact_id HAVING COUNT(DISTINCT source_id) > 1)"
        ).fetchone()[0]
        out["ungrounded"] = conn.execute(
            "SELECT COUNT(DISTINCT fact_id) FROM fact_source WHERE grounded = 0"
        ).fetchone()[0]
        out["by_source_type"] = {
            r["source_type"]: r["n"] for r in conn.execute(
                "SELECT source_type, COUNT(*) AS n FROM fact_source GROUP BY source_type"
            ).fetchall()
        }
        out["by_type"] = {
            r["fact_type"]: r["n"] for r in conn.execute(
                "SELECT fact_type, COUNT(*) AS n FROM fact GROUP BY fact_type"
            ).fetchall()
        }
    out["size_bytes"] = p.stat().st_size
    return out


def dedup_candidates(similarity_note: str = "") -> list[dict[str, Any]]:
    """
    找出**可能**重复但 id 不同的 fact 候选, 供人工/后续策略判断。

    第一版刻意**不做自动合并**(不可逆), 只提供候选让上层决定。
    当前实现: 同一 subject 下、content 去重后前 8 字相同的不同 fact。
    这是保守的候选生成, 不做语义相似度判断。
    """
    facts = list_facts(limit=2000, order_by="recent")
    by_key: dict[tuple[str, str], list[dict]] = {}
    for f in facts:
        key = (f.get("subject") or "", (f.get("content") or "")[:8])
        by_key.setdefault(key, []).append(f)
    out = []
    for (subj, prefix), group in by_key.items():
        if len(group) > 1:
            out.append({
                "subject": subj, "prefix": prefix, "n": len(group),
                "fact_ids": [g["id"] for g in group],
                "note": similarity_note,
            })
    return out
