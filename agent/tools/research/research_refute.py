"""
agent.tools.research.research_refute — 专攻找反例 (P7 阶段 1, L0)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §1.5.1 判据 2

============================ 为什么要有这个编排工具 =================================
"找反例"有三条强度递增的路, 但 LLM 每次都要自己在脑子里权衡, 容易挑弱的走:

    图枚举 (n≤7 穷尽)   <   Z3 (全域级反例)   <   都不是 → 如实说不知道

本工具把这个策略**固化成代码**: 先穷举(便宜、确定性、范围全覆盖),
再 Z3(贵但能出全域反例), 都没有就如实报 partial/inconclusive。

⚠️ **本工具永远不会返回 proved** —— 这是刻意的。
   "找不到反例" ≠ "命题成立"。证实类命题走 research_sympy 或 research_z3(goal="prove")。
   把"找反例失败"说成"证明了"是科研里最常见的误判, 这里从代码层堵死。

⚠️ exposed 默认 False。
"""
from __future__ import annotations

import logging
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace, hash_question
from agent.tools.registry import register_tool

from ._enumerate import GRAPH_PROPERTIES, dispatch
from ._flags import RESEARCH_EXPOSED
from ._provenance import claim_id_for, record
from ._z3_tools import DEFAULT_TIMEOUT_MS, Z3Error, solve

logger = logging.getLogger(__name__)


@register_tool(name="research_refute", cost_tier="L0", exposed=RESEARCH_EXPOSED)
def research_refute(
    claim: str,
    graphs: str = "",
    constraints: str = "",
    n_max: int = 7,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> dict[str, Any]:
    """
    依次尝试图枚举 → Z3, 找到反例立刻返回。

    Args:
        claim: 要反驳的命题
        graphs: 图性质名(空则跳过图枚举这一路)
        constraints: SMT-LIB2 约束(空则跳过 Z3 这一路)
        n_max: 图枚举上限
        timeout_ms: Z3 超时
    """
    th = hash_question(f"refute:{claim}")
    ws = ResearchWorkspace(th, question=claim)
    cid = claim_id_for(claim, graphs or constraints[:200])
    ws.set_status(phase="refute", step=1, total=1, message=claim[:60])

    attempts: list[str] = []
    evidence: list[dict[str, Any]] = []

    # ---------- 路线 1: 图枚举(穷尽优先, 便宜且确定) ----------
    if graphs:
        try:
            out = dispatch(task="graph", prop=graphs, n_max=n_max)
            ev = record(
                ws, kind="code", verdict=out["verdict"], reason=out["reason"],
                code=f"# research_refute 图枚举 prop={graphs} n_max={n_max}",
                stdout=str(out.get("counterexample") or ""),
                elapsed_ms=out["elapsed_ms"], claim_id=cid,
                extra={"route": "enumerate", "counterexample": out.get("counterexample"),
                       "checked": out.get("checked")},
            )
            evidence.append(ev)
            attempts.append(f"图枚举({graphs}, n≤{n_max}): {out['verdict']}, "
                            f"检查了 {out.get('checked')} 个图")

            if out["verdict"] == "refuted":
                return _finish(ws, "refuted", out["reason"], evidence, attempts,
                               counterexample=out.get("counterexample"), route="enumerate")
        except ValueError as e:
            attempts.append(f"图枚举参数错: {e}")

    # ---------- 路线 2: Z3(全域级反例) ----------
    if constraints:
        try:
            out = solve(constraints, goal="prove", timeout_ms=timeout_ms)
            ev = record(
                ws, kind="solver", verdict=out["verdict"], reason=out["reason"],
                code=constraints, stdout=out["model"], elapsed_ms=out["elapsed_ms"],
                claim_id=cid,
                extra={"route": "z3", "result": out["result"], "goal": "prove"},
            )
            evidence.append(ev)
            attempts.append(f"Z3(goal=prove): {out['result']} -> {out['verdict']}")

            if out["verdict"] == "refuted":
                return _finish(ws, "refuted", out["reason"], evidence, attempts,
                               counterexample=out["model"], route="z3")
            if out["verdict"] == "proved":
                # 罕见但合法: 约束组 unsat 说明"假设成立 ∧ 结论失败"不可满足。
                # 这时"反驳失败且命题被证明成立" —— 如实报, 但要说清是谁证实的。
                return _finish(ws, "proved",
                               "Z3 证明该命题在给定前提下成立(unsat), 反驳不成立。",
                               evidence, attempts, counterexample=None, route="z3")
        except Z3Error as e:
            attempts.append(f"Z3 失败: {e}")

    # ---------- 都没结论 ----------
    if not evidence:
        reason = (
            "没有提供任何可执行的反驳路线。research_refute 需要至少给一个: "
            f"graphs={list(GRAPH_PROPERTIES)[:4]}... 之一, 或 constraints=SMT-LIB2 文本。"
        )
        return {
            "ok": False, "verdict": "inconclusive", "evidence": [],
            "summary": reason, "attempts": attempts,
            "workspace": str(ws.path),
            "cost": {"tokens": 0, "elapsed_ms": 0},
        }

    return _finish(ws, "partial",
                   "两条路线都没找到反例。⚠️ **找不到反例不等于命题成立** —— "
                   "这只是这些范围内的有限检查。证实类命题请用 research_sympy(化简到 0) "
                   "或 research_z3(goal=\"prove\")。",
                   evidence, attempts, counterexample=None, route="all")


def _finish(ws, verdict: str, reason: str, evidence: list[dict], attempts: list[str],
            *, counterexample=None, route: str = "") -> dict[str, Any]:
    """统一出口"""
    elapsed = sum(e.get("elapsed_ms", 0) for e in evidence)
    return {
        "ok": verdict == "refuted",
        "verdict": verdict,
        "evidence": evidence,
        "summary": reason,
        "counterexample": counterexample,
        "attempts": attempts,
        "route_used": route,
        "workspace": str(ws.path),
        "cost": {"tokens": 0, "elapsed_ms": elapsed},
    }
