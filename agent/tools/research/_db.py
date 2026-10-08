"""
agent.tools.research._db — research.db 长期库 (P7 阶段 2)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §7

================================ 为什么要有这个库, workspace 不够吗 ============================
workspace(单次任务的 JSONL)已经能记全一次验证。但它记不住**跨任务**的东西:

  同一个命题, 10 月 8 日验出 refuted(反例 n=40), 11 月换台机器又验一次 —— 这两条证据
  在两个 workspace 里彼此看不见。命题一旦可判定且值得验, 通常会被反复碰到
  (换范围、换环境、论文里被引用时重新检查)。这个库就是让它们相遇的地方。

所以 claim 表用**内容哈希做主键**(对标 agent/tools/facts.py:105 `fact_id_for`):
同命题不同任务自动合并成一行, evidence 多次追加不覆盖。于是下面三个问题能直接查:
  - 这条命题验过几次?
  - 每次结论一致吗?(不一致 = 翻车, audit 会报 verdict_conflict)
  - 有没有命题的 verdict 前后矛盾?

★ 有一条纪律写死在结构里: **不覆盖**。同 claim 的 evidence 只 append,
因为"证据被覆盖"和"证据被追加"在科研上的区别是, 前者让你永远查不清出问题时到底跑过什么。

历史:
  - 2026-10-08 P7.2 首版
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/research/research.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS task (
  task_hash   TEXT PRIMARY KEY,
  question    TEXT,
  created_at  TEXT,
  status      TEXT,          -- running / done / failed
  rounds      INTEGER DEFAULT 0,
  tokens      INTEGER DEFAULT 0
);

-- id 是 claim_id_for() 的内容哈希 (见 _provenance.claim_id_for)
CREATE TABLE IF NOT EXISTS claim (
  id           TEXT PRIMARY KEY,
  task_hash    TEXT,
  claim_text   TEXT,
  scope        TEXT,         -- 变量域与范围; 空 = 不可判定, 不许宣称 proved
  decidable    INTEGER DEFAULT 1,
  verdict      TEXT,         -- proved/refuted/partial/failed/inconclusive/untested/undecidable
  n_evidence   INTEGER DEFAULT 0,
  first_seen   TEXT,
  updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  claim_id    TEXT,
  task_hash   TEXT,
  kind        TEXT,          -- code / solver / lean / citation
  verdict     TEXT,
  reason      TEXT,
  code        TEXT,
  stdout      TEXT,
  stderr      TEXT,
  elapsed_ms  INTEGER,
  env         TEXT,          -- JSON 字符串
  created_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_claim_task   ON claim(task_hash);
CREATE INDEX IF NOT EXISTS idx_ev_claim     ON evidence(claim_id);
CREATE INDEX IF NOT EXISTS idx_ev_task      ON evidence(task_hash);
"""


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema() -> None:
    """建表(幂等)。import 本模块时自动跑, 跟 facts/surveys 一致。"""
    with _connect() as conn:
        conn.executescript(SCHEMA_SQL)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ============================================================
# task
# ============================================================
def upsert_task(task_hash: str, question: str, status: str = "running") -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO task (task_hash, question, created_at, status) VALUES (?,?,?,?) "
            "ON CONFLICT(task_hash) DO UPDATE SET status=excluded.status",
            (task_hash, question, _now(), status),
        )


def finish_task(task_hash: str, status: str = "done", rounds: int = 0, tokens: int = 0) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE task SET status=?, rounds=?, tokens=? WHERE task_hash=?",
            (status, rounds, tokens, task_hash),
        )


def get_task(task_hash: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM task WHERE task_hash=?", (task_hash,)).fetchone()
    return dict(row) if row else None


def list_tasks(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM task ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ============================================================
# claim (内容哈希主键 —— 同命题自动合并)
# ============================================================
def upsert_claim(
    claim_id: str,
    claim_text: str,
    task_hash: str = "",
    scope: str = "",
    decidable: bool = True,
    verdict: str = "pending",
) -> None:
    """
    写入/更新一条命题。同 claim_id **合并**, 不新建行。

    注意 verdict 用 `excluded.verdict` 直接覆盖 —— 这是**唯一**允许覆盖的字段,
    因为它是"当前最新判定"而非历史。历史判定在 evidence 表里, 永远保留。
    """
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO claim (id, task_hash, claim_text, scope, decidable, verdict,
                               first_seen, updated_at)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                verdict = excluded.verdict,
                scope   = CASE WHEN excluded.scope != '' THEN excluded.scope ELSE claim.scope END,
                updated_at = excluded.updated_at
            """,
            (claim_id, task_hash, claim_text, scope, 1 if decidable else 0,
             verdict, _now(), _now()),
        )


def get_claim(claim_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM claim WHERE id=?", (claim_id,)).fetchone()
    return dict(row) if row else None


def list_claims(task_hash: str = "", limit: int = 200) -> list[dict[str, Any]]:
    if task_hash:
        sql = "SELECT * FROM claim WHERE task_hash=? ORDER BY updated_at DESC LIMIT ?"
        args = (task_hash, limit)
    else:
        sql = "SELECT * FROM claim ORDER BY updated_at DESC LIMIT ?"
        args = (limit,)
    with _connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def claims_by_verdict(verdict: str) -> list[dict[str, Any]]:
    """按判定查命题 —— 比如"把历史 refuted 的都翻出来" """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM claim WHERE verdict=? ORDER BY updated_at DESC", (verdict,)
        ).fetchall()
    return [dict(r) for r in rows]


# ============================================================
# evidence (只 append, 永不覆盖)
# ============================================================
def add_evidence(
    ev: dict[str, Any],
    task_hash: str = "",
    env: dict[str, Any] | None = None,
) -> int:
    """
    追加一条证据, 返回自增 id。

    调用前先 upsert_claim —— 这里顺手把 claim.n_evidence 维护起来,
    免得再写一遍 COUNT 查询。
    """
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO evidence (claim_id, task_hash, kind, verdict, reason, code,
                                  stdout, stderr, elapsed_ms, env, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ev.get("claim_id") or "",
                task_hash,
                ev.get("kind", ""),
                ev.get("verdict", ""),
                ev.get("reason", ""),
                (ev.get("code") or "")[:20000],
                (ev.get("stdout") or "")[:20000],
                (ev.get("stderr") or "")[:8000],
                int(ev.get("elapsed_ms") or 0),
                json.dumps(env or ev.get("env") or {}, ensure_ascii=False),
                _now(),
            ),
        )
        cid = ev.get("claim_id") or ""
        if cid:
            conn.execute(
                "UPDATE claim SET n_evidence = "
                "(SELECT COUNT(*) FROM evidence WHERE claim_id = ?), updated_at = ? "
                "WHERE id = ?",
                (cid, _now(), cid),
            )
        return int(cur.lastrowid or 0)


def list_evidence(claim_id: str = "", task_hash: str = "", limit: int = 200) -> list[dict[str, Any]]:
    if claim_id:
        sql, args = "SELECT * FROM evidence WHERE claim_id=? ORDER BY id", (claim_id,)
    elif task_hash:
        sql, args = "SELECT * FROM evidence WHERE task_hash=? ORDER BY id", (task_hash,)
    else:
        sql, args = "SELECT * FROM evidence ORDER BY id DESC LIMIT ?", (limit,)
    with _connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def claim_verdict_history(claim_id: str) -> list[str]:
    """
    某命题的判定历史(按时间)。**翻车检测的原始数据** ——
    出现互斥的 proved/refuted 就是结论打架, audit 会报 verdict_conflict。
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT verdict FROM evidence WHERE claim_id=? ORDER BY id", (claim_id,)
        ).fetchall()
    return [r["verdict"] for r in rows if r["verdict"]]


# ============================================================
# 统计
# ============================================================
def stats() -> dict[str, Any]:
    """库概览(诊断/UI 用)"""
    with _connect() as conn:
        n_task = conn.execute("SELECT COUNT(*) FROM task").fetchone()[0]
        n_claim = conn.execute("SELECT COUNT(*) FROM claim").fetchone()[0]
        n_ev = conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
        rows = conn.execute(
            "SELECT verdict, COUNT(*) c FROM claim GROUP BY verdict"
        ).fetchall()
    return {
        "db_path": str(_DB_PATH),
        "tasks": n_task,
        "claims": n_claim,
        "evidence": n_ev,
        "claims_by_verdict": {r["verdict"]: r["c"] for r in rows},
    }


# import 时建表(与 facts/surveys 行为一致)
try:
    init_schema()
except Exception as e:      # pragma: no cover
    logger.warning("[research._db] init_schema 失败(非致命): %s", e)
