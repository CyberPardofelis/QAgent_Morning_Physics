"""
agent.tools.research.research_lean — Lean 4 形式化验证工具 (P7.3, L0)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §23

为什么需要它
------------
用户的要求: 「结论对不对」和「结论有没有证据」最后都必须问机器, LLM 的
判断也得基于**符号或数值分析、Lean 之类的结果**。

Lean kernel 是这几种后端里**最强的一档**: 它验证的是证明项本身(proof
term), 不是代码逻辑。kernel 本身只有一万多行、信任假设极小, 这是整个
形式化验证的根基。sympy/numpy 验的是"我对这个式子的化简"; Lean 验的是
"这个推理链每一步都合法"。

为什么是 L0: 它不调 LLM。Lean 源码是文本, kernel 自己判真假,
和 research_z3 同理直接调比绕沙箱干净。

★ 判定判据(最容易踩的坑) ★
Lean 遇到 `sorry` 会报 warning 但**编译通过, 退出码仍是 0**。实测:
    theorem t (n : Nat) : n*2+1 ≥ 1 := by sorry
    → exit=0, 输出 "warning: declaration uses 'sorry'"

所以「退出码 0」**不等于**证明了。本工具的判定是三条一起看:
    exit_code == 0  且  没有 `: error:`  且  没有 `declaration uses 'sorry'`

⚠️ Lean 没装时**如实说没装**, 返回 failed 而不是假装能证。
   V2 里需要 Mathlib 的定理(费马小定理/sqrt2无理)暂时跑不了 ——
   Mathlib 缓存约 3GB 且走 GitHub release, 国内网络下不了。
   但不依赖 Mathlib 的核心定理(归纳、代数基础)现在就能验。

⚠️ exposed 默认 False (QAGENT_RESEARCH_EXPOSE=0)。
"""
from __future__ import annotations

import logging
import time
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace, hash_question
from agent.tools.registry import register_tool

from ._flags import RESEARCH_EXPOSED
from ._lean import check_lean, lean_available, lean_version
from ._provenance import claim_id_for, record

logger = logging.getLogger(__name__)


@register_tool(name="research_lean", cost_tier="L0", exposed=RESEARCH_EXPOSED)
def research_lean(
    code: str,
    claim: str = "",
    timeout: int = 300,
) -> dict[str, Any]:
    """
    用 Lean 4 kernel 检查一段形式化证明, 并落证据。

    ★ 什么时候该用它 ★
    - 需要**全域、无穷**层面的结论时(有限枚举证明不了"对所有 n")
    - 命题本身能写成形式化陈述时(定理、性质、算法正确性)
    - 想要比"符号化简"更强的保证时

    ★ 什么时候不该用 ★
    - 数值计算、具体反例搜索 → 用 research_run_code / research_enumerate
    - 开放式问题(哥德巴赫猜想本身) → Lean 也证不了
    - 需要 Mathlib 的大型定理 → 当前环境 Mathlib 没装好

    Args:
        code: Lean 4 源码。应为若干 theorem 声明。
              不依赖 Mathlib 的核心定理可以直接用(如 Nat.add_zero)。
        claim: 这段代码想证明的命题(写进证据, 供报告引用)
        timeout: 编译硬超时秒数

    Returns:
        ok / verdict / evidence[] / summary / proved / sorry_used /
        lean_version / cost
    """
    if not lean_available():
        return {
            "ok": False,
            "verdict": "failed",
            "evidence": [],
            "summary": ("Lean 未安装 —— 本工具不能用来做机器判定。"
                        "需要 Mathlib 的定理当前也跑不了。"),
            "proved": False,
            "sorry_used": False,
            "lean_version": "absent",
            "cost": {"tokens": 0, "elapsed_ms": 0},
        }

    t0 = time.time()
    res = check_lean(code, timeout=timeout)
    elapsed_ms = int((time.time() - t0) * 1000)
    ver = lean_version()

    th = hash_question(f"lean:{claim or code[:200]}")
    ws = ResearchWorkspace(th, question=claim or "lean check")
    ws.set_status(phase="lean", step=1, total=1,
                  message="formalize" if res.proved else "check")

    if res.proved:
        verdict = "proved"
        reason = (f"Lean kernel 验证通过(无 sorry、无 error) —— {ver}\n"
                  f"★ 这是证明项级别的验证: kernel 检查的是每个推理步骤的合法性, "
                  f"不是代码逻辑。")
    elif res.get("sorry_used"):
        verdict = "inconclusive"
        reason = ("源码里有 `sorry` —— 证明**没有写完**。\n"
                  "★ 注意: Lean 遇到 sorry 仍会编译通过(退出码 0), "
                  "所以不能只看编译成功与否。\n"
                  f"位置: {'; '.join(res.get('warnings', [])[:3])}")
    else:
        verdict = "failed"
        reason = ("Lean 编译失败 —— 证明项不合法或写错了。\n"
                  + "\n".join(res.get("errors", [])[:5]))

    ev = record(
        ws, kind="solver", verdict=verdict, reason=reason, code=code,
        stdout=res.get("output", ""), elapsed_ms=elapsed_ms,
        claim_id=claim_id_for(claim, code[:200]),
        extra={
            "lean_version": ver,
            "sorry_used": res.get("sorry_used", False),
            "exit_code": res.get("exit_code"),
            "n_errors": len(res.get("errors", [])),
            "backend": "lean4-kernel",
        },
    )

    logger.info("[research_lean] proved=%s sorry=%s (%dms)",
                res.proved, res.get("sorry_used"), elapsed_ms)

    return {
        "ok": res.proved,
        "verdict": verdict,
        "evidence": [ev],
        "summary": reason,
        "proved": res.proved,
        "sorry_used": res.get("sorry_used", False),
        "errors": res.get("errors", [])[:10],
        "lean_version": ver,
        "workspace": str(ws.path),
        "cost": {"tokens": 0, "llm_calls": 0, "elapsed_ms": elapsed_ms},
    }