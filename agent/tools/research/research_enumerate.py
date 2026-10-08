"""
agent.tools.research.research_enumerate — 有限枚举反例搜索工具 (P7 阶段 1, L0)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §1.5.1 判据 2 / §6.1

⚠️ 本工具**永远不返回 proved** —— 穷尽到 n≤7 仍不等于全域成立。
   它只能给 refuted(找到反例)或 partial(有限范围内无反例)。
   证实类命题请用 research_sympy(化简到 0)或 research_z3(goal="prove" 拿到 unsat)。

⚠️ exposed 默认 False。
"""
from __future__ import annotations

import logging
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace, hash_question
from agent.tools.registry import register_tool

from ._enumerate import GRAPH_PROPERTIES, SEQUENCE_PROPERTIES, dispatch
from ._flags import RESEARCH_EXPOSED
from ._provenance import record

logger = logging.getLogger(__name__)


@register_tool(name="research_enumerate", cost_tier="L0", exposed=RESEARCH_EXPOSED)
def research_enumerate(
    prop: str,
    task: str = "graph",
    n_max: int = 7,
    kind: str = "primes",
    n_terms: int = 20,
    claim: str = "",
) -> dict[str, Any]:
    """
    在可穷举对象上检查性质, 找反例。

    Args:
        prop: 性质名(graph/sequence 两套, 见 schema)
        task: graph / sequence
        n_max: graph 时最大顶点数(≤7, graph_atlas 上限)
        kind: sequence 时用哪个内置序列
        n_terms: sequence 时项数
        claim: 这在验什么(写进证据)
    """
    label = f"enumerate:{task}:{prop}"
    th = hash_question(f"{label}:{n_max}:{kind}:{n_terms}")
    ws = ResearchWorkspace(th, question=claim or label)
    ws.set_status(phase="enumerate", step=1, total=1, message=label)

    try:
        out = dispatch(task=task, prop=prop, n_max=n_max, kind=kind, n_terms=n_terms)
    except ValueError as e:
        ev = record(
            ws, kind="code", verdict="failed", reason=f"参数错误: {e}", code="",
            claim_id="",
            extra={"task": task, "prop": prop},
        )
        # 参数错就把可用清单一并给出去, 省得 LLM 猜一轮
        avail = (GRAPH_PROPERTIES if task == "graph" else SEQUENCE_PROPERTIES)
        return {
            "ok": False, "verdict": "failed", "evidence": [ev],
            "summary": f"{e}。可用性质: {', '.join(sorted(avail))}",
            "workspace": str(ws.path),
            "cost": {"tokens": 0, "elapsed_ms": 0},
        }

    ev = record(
        ws,
        kind="code",
        verdict=out["verdict"],
        reason=out["reason"],
        code=f"# research_enumerate task={task} prop={prop} n_max={n_max} kind={kind}",
        stdout=str(out.get("counterexample") or ""),
        elapsed_ms=out["elapsed_ms"],
        claim_id="",
        extra={"task": task, "prop": prop, "checked": out.get("checked"),
               "counterexample": out.get("counterexample"),
               "atlas_exhaustive": out.get("atlas_exhaustive")},
    )

    logger.info("[research_enumerate] %s/%s -> %s (%s checked)",
                task, prop, out["verdict"], out.get("checked", "?"))
    return {
        "ok": True,
        "verdict": out["verdict"],
        "evidence": [ev],
        "summary": out["reason"],
        "counterexample": out.get("counterexample"),
        "checked": out.get("checked"),
        "atlas_exhaustive": out.get("atlas_exhaustive"),
        "workspace": str(ws.path),
        "cost": {"tokens": 0, "elapsed_ms": out["elapsed_ms"]},
    }
