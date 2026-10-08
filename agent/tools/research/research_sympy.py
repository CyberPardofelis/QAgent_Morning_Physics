"""
agent.tools.research.research_sympy — SymPy 符号计算工具 (P7 阶段 0, L0)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §6.1

存在的意义 (§1.5.1 判据 1): 要下"proved"这种全域结论, **只能用符号计算**。
浮点数恰好为 0 / 小 n 枚举没失败, 都只支持 partial —— 这是数值与符号的硬边界。

安全: expr 是 LLM 生成的字符串, 绝不直接喂 sympify。
     `_sympy_tools.build_code` 用 repr() 把 expr/params 嵌成字面量, 再走沙箱 AST 门。

⚠️ exposed 默认 False (QAGENT_RESEARCH_EXPOSE=0)。
"""
from __future__ import annotations

import logging
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace, hash_question
from agent.tools.registry import register_tool

from ._flags import RESEARCH_EXPOSED, RESEARCH_SANDBOX_TIMEOUT
from ._sympy_tools import sympy_op

logger = logging.getLogger(__name__)


@register_tool(name="research_sympy", cost_tier="L0", exposed=RESEARCH_EXPOSED)
def research_sympy(
    expr: str,
    op: str = "simplify",
    params: str = "",
    timeout: int = RESEARCH_SANDBOX_TIMEOUT,
) -> dict[str, Any]:
    """
    做一次符号计算并落一条证据。

    Args:
        expr: SymPy 表达式字符串, 自由变量直接写符号名 (如 'x**2 + 2*x + 1')
        op: simplify/expand/factor/cancel/apart/trigsimp/integrate/diff/
            series/limit/sum/solve/eqsolve/isprime
        params: 额外参数, 空格分隔。integrate/diff 需变量名("x");
            series/limit/sum 需 "变量 起点 终点/阶数" (如 "x 0 10")
        timeout: 沙箱硬超时秒数

    Returns:
        §6.4 的 5 字段形态
    """
    task_hash = hash_question(f"sympy:{op}:{expr}")
    ws = ResearchWorkspace(task_hash, question=f"{op}({expr})")

    out = sympy_op(expr, op, params, scratch_dir=str(ws.scratch_dir), timeout=timeout)

    # 符号计算的结果本身就是证据: 若化简到 0, 就是"等于 0"的精确证据
    verdict = "proved" if (out.get("ok") and out.get("result_str") == "0") else (
        "partial" if out.get("ok") else "failed"
    )
    reason = (
        "符号化简结果为 0 —— 精确判定成立"
        if verdict == "proved"
        else ("符号计算完成, 结果非零(不构成证明)" if out.get("ok")
              else f"符号计算失败: {out.get('error_type', '?')}")
    )

    from ._provenance import build_evidence
    ev = build_evidence(
        ws.evidence_count() + 1,      # 编号从 ev_0001 起(与 _execute 一致)
        kind="code",
        verdict=verdict,
        reason=reason,
        code=out.get("code", ""),
        stdout=out.get("stdout", ""),
        stderr=out.get("stderr", ""),
        elapsed_ms=out.get("elapsed_ms", 0),
        extra={"op": op, "expr": expr, "params": params,
               "result_str": out.get("result_str", ""), "latex": out.get("latex", "")},
    )
    ws.append_evidence(ev)

    logger.info("[research_sympy] %s(%s) -> %s", op, expr[:40], out.get("result_str", ""))
    return {
        "ok": bool(out.get("ok")),
        "verdict": verdict,
        "evidence": [ev],
        "summary": reason,
        "result_str": out.get("result_str", ""),
        "latex": out.get("latex", ""),
        "stdout": out.get("stdout", ""),
        "stderr": out.get("stderr", ""),
        "workspace": str(ws.path),
        "cost": {"tokens": 0, "elapsed_ms": out.get("elapsed_ms", 0)},
    }
