"""
agent.tools.research.research_z3 — SMT 求解器工具 (P7 阶段 1, L0)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §6.1

为什么是 L0: 它不调 LLM。约束是 SMT-LIB2 文本(不是可执行代码), z3 自带求解超时,
所以直接调比绕沙箱更快也更干净 —— 沙箱那套 AST 门对 SMT-LIB2 不适用。

⚠️ exposed 默认 False (QAGENT_RESEARCH_EXPOSE=0)。
"""
from __future__ import annotations

import logging
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace, hash_question
from agent.tools.registry import register_tool

from ._flags import RESEARCH_EXPOSED
from ._provenance import claim_id_for, record
from ._z3_tools import DEFAULT_TIMEOUT_MS, Z3Error, solve

logger = logging.getLogger(__name__)


@register_tool(name="research_z3", cost_tier="L0", exposed=RESEARCH_EXPOSED)
def research_z3(
    constraints: str,
    goal: str = "sat",
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    claim: str = "",
) -> dict[str, Any]:
    """
    求解 SMT-LIB2 约束并落证据。

    goal="prove" 时按证明语义映射: unsat→proved, sat→refuted(附反例赋值)。
    """
    th = hash_question(f"z3:{goal}:{constraints}")
    ws = ResearchWorkspace(th, question=claim or "z3 solve")
    ws.set_status(phase="z3", step=1, total=1, message=goal)

    try:
        out = solve(constraints, goal=goal, timeout_ms=timeout_ms)
    except Z3Error as e:
        ev = record(
            ws, kind="solver", verdict="failed",
            reason=f"Z3 求解失败: {e}",
            code=constraints, claim_id=claim_id_for(claim, constraints[:200]),
            extra={"goal": goal},
        )
        return {
            "ok": False, "verdict": "failed", "evidence": [ev],
            "summary": ev["reason"], "result": "error",
            "workspace": str(ws.path),
            "cost": {"tokens": 0, "elapsed_ms": 0},
        }

    ev = record(
        ws,
        kind="solver",
        verdict=out["verdict"],
        reason=out["reason"],
        code=constraints,
        stdout=out["model"],
        elapsed_ms=out["elapsed_ms"],
        claim_id=claim_id_for(claim, constraints[:200]),
        extra={"goal": goal, "result": out["result"],
               "n_assertions": out["n_assertions"], "timeout_ms": out["timeout_ms"]},
    )

    logger.info("[research_z3] goal=%s -> %s (%sms)", goal, out["result"], out["elapsed_ms"])
    return {
        "ok": out["result"] != "unknown",
        "verdict": out["verdict"],
        "evidence": [ev],
        "summary": out["reason"],
        "result": out["result"],
        "model": out["model"],
        "n_assertions": out["n_assertions"],
        "workspace": str(ws.path),
        "cost": {"tokens": 0, "elapsed_ms": out["elapsed_ms"]},
    }
