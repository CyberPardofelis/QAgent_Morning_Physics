"""
agent.tools.research.research_oeis — OEIS 整数序列检索工具 (P7 阶段 1, L0)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §1.5.2

⚠️ 唯一需要联网的工具(oeis.org)。网络失败时**优雅降级**, 不抛异常打断对话。

⚠️ exposed 默认 False。
"""
from __future__ import annotations

import logging
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace, hash_question
from agent.tools.registry import register_tool

from ._flags import RESEARCH_EXPOSED
from ._provenance import record
from ._retrieve import RetrieveError, lookup

logger = logging.getLogger(__name__)


@register_tool(name="research_oeis", cost_tier="L0", exposed=RESEARCH_EXPOSED)
def research_oeis(
    sequence: str,
    max_terms: int = 10,
    keep: int = 5,
    claim: str = "",
) -> dict[str, Any]:
    """
    查 OEIS 并逐项对账, 返回最强匹配。

    Args:
        sequence: "1,2,4,8,16" 这样的整数序列, ≥2 项
        max_terms: 最多送多少项
        keep: 返回前几条候选
        claim: 这在验什么(写进证据)
    """
    th = hash_question(f"oeis:{sequence}")
    ws = ResearchWorkspace(th, question=claim or f"oeis {sequence}")
    ws.set_status(phase="oeis", step=1, total=1, message=sequence)

    try:
        out = lookup(sequence, max_terms=max_terms, keep=keep)
    except (RetrieveError, ValueError) as e:
        ev = record(
            ws, kind="citation", verdict="failed",
            reason=f"OEIS 查询失败: {e}",
            code=str(sequence), claim_id="",
        )
        return {
            "ok": False, "verdict": "failed", "evidence": [ev],
            "summary": str(e),
            "workspace": str(ws.path),
            "cost": {"tokens": 0, "elapsed_ms": 0},
        }

    best = out.get("best")
    ev = record(
        ws,
        kind="citation",
        verdict=out["verdict"],
        reason=out["reason"],
        code=f"OEIS query: {out['query']}",
        stdout=(f"{best['number']} | {best['name']} | {best['formula']}" if best else "(无匹配)"),
        claim_id="",
        extra={"oeis_number": best["number"] if best else None,
               "n_exact_matches": out["n_exact_matches"],
               "candidates": out["candidates"]},
    )

    logger.info("[research_oeis] %s -> %s", out["query"], best["number"] if best else "no match")
    return {
        "ok": True,
        "verdict": out["verdict"],
        "evidence": [ev],
        "summary": out["reason"],
        "best_match": best,
        "candidates": out["candidates"],
        "n_terms": out["n_terms"],
        "n_exact_matches": out["n_exact_matches"],
        "workspace": str(ws.path),
        "cost": {"tokens": 0, "elapsed_ms": 0},
    }
