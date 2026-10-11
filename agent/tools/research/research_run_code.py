"""
agent.tools.research.research_run_code — 受限沙箱执行工具 (P7 阶段 0, L0)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §6.1

⚠️ exposed 默认 False (QAGENT_RESEARCH_EXPOSE=0): 开发调通用, LLM 暂不见。
   接入时把该 env 设 1, 不必改代码。

这个文件同时提供内部函数 `_execute()`, 供 research_verify 复用 ——
"跑一次 + 判一次 + 记一条证据" 这套动作只该有一份实现, 否则两处会慢慢漂移。
"""
from __future__ import annotations

import logging
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace, hash_question
from agent.tools.registry import register_tool

from ._flags import RESEARCH_EXPOSED, RESEARCH_SANDBOX_TIMEOUT
from ._provenance import build_evidence, judge_verdict
from ._sandbox import SandboxRejected, run_code

logger = logging.getLogger(__name__)


# ============================================================
# 内部: 执行 + 判定 + 落证据 (research_verify 复用)
# ============================================================
def _execute(
    code: str,
    ws: ResearchWorkspace,
    *,
    purpose: str = "",
    timeout: int = RESEARCH_SANDBOX_TIMEOUT,
    claim_id: str = "",
    claim: str = "",
    kind: str = "code",
    extra: dict[str, Any] | None = None,
):
    """
    在 ws 的 scratch 里跑 code, 判定 verdict, 追加一条证据到 _evidence.jsonl。

    Returns:
        (SandboxResult | None, evidence_dict)

    SandboxResult 为 None 表示代码没过静态门(此时 evidence 照样落盘,
    verdict=failed —— **被拒也是证据**, 不该悄悄消失)。
    """
    ws.clear_scratch()
    # 编号从 ev_0001 起(不是 ev_0000): 报告里人读的锚点, 0 号看着像"没有证据"
    index = ws.evidence_count() + 1

    try:
        result = run_code(code, ws.scratch_dir, timeout=timeout)
    except SandboxRejected as e:
        # 静态门拒绝: 没执行, 但必须记录, 否则调 LLM 时看不到"为什么没跑"
        verdict, reason = "failed", f"代码被沙箱静态门拒绝: {e.reason}"
        ev = build_evidence(
            index, kind=kind, verdict=verdict, reason=reason,
            code=code, elapsed_ms=0, claim_id=claim_id, extra=extra,
        )
        ws.append_evidence(ev)
        return None, ev

    # code 必须显式传: judge_verdict 的第二层门(复核 REFUTED 的反例是否
    # 真成立)要看代码形态才能发现遍历顺序类 bug。
    # claim 同理: PROVED 侧的机器复核要认出「代码把待证命题当前提用」,
    # 光看代码和 stdout 认不出哪个词是结论 —— 那是命题自带的。
    verdict, reason = judge_verdict(result, code=code, claim=claim)

    # ★ stdout 抽取与 exit code 解耦 ★
    # judge_verdict 的第一优先级是「没跑成 -> failed」, 于是**先看退出码、
    # 后看 stdout**。但 stdout 里可能已经有一个自证有效的制品(见文件头事故)。
    # 这里: 崩了但 stdout 非空时, 用 ok=True 再判一次 —— 让标记解析与制品门
    # 照常跑。若它给出确定结论(而且带上了可复核的制品), 就采信, 并标注崩溃。
    # 若仍然判 failed, 说明 stdout 里确实没有可用证据, 维持原判。
    crashed_with_output = False
    if not getattr(result, "ok", True) and (result.stdout or "").strip():
        import copy as _copy
        lenient = _copy.copy(result)
        lenient.ok = True
        lenient.timed_out = False
        lenient.rejected = False
        v2, r2 = judge_verdict(lenient, code=code, claim=claim)
        if v2 in ("proved", "refuted", "partial"):
            verdict = v2
            reason = ("⚠ 程序在打印上述结果后崩溃(exit!=0), "
                      "但 stdout 里留下了可复核的制品, 据此采信; "
                      "崩前未能完成的部分不作数。\n  " + r2)
            crashed_with_output = True

    ev = build_evidence(
        index,
        kind=kind,
        verdict=verdict,
        reason=reason,
        code=code,
        stdout=result.stdout,
        stderr=result.stderr or result.reject_reason,
        elapsed_ms=result.elapsed_ms,
        claim_id=claim_id,
        extra={**(extra or {}),
               "crashed_with_output": crashed_with_output},
    )
    ws.append_evidence(ev)
    return result, ev


# ============================================================
# LLM 工具入口
# ============================================================
@register_tool(name="research_run_code", cost_tier="L0", exposed=RESEARCH_EXPOSED)
def research_run_code(
    code: str,
    purpose: str = "",
    timeout: int = RESEARCH_SANDBOX_TIMEOUT,
) -> dict[str, Any]:
    """
    在受限沙箱里执行 Python, 返回输出与证据编号。

    Args:
        code: 要执行的 Python 源码(不要 markdown 包裹)
        purpose: 这段代码要验什么(一句话, 进证据记录)
        timeout: 硬超时秒数

    Returns:
        §6.4 的 5 字段形态 + code/stdout 明细
    """
    task_hash = hash_question(purpose or "adhoc:research_run_code")
    ws = ResearchWorkspace(task_hash, question=purpose or "ad-hoc run")
    ws.set_status(phase="run_code", step=1, total=1, message=purpose)

    result, ev = _execute(code, ws, purpose=purpose, timeout=timeout)

    payload = {
        "ok": ev["verdict"] not in ("failed",),
        "verdict": ev["verdict"],
        "evidence": [ev],
        "summary": ev["reason"],
        "workspace": str(ws.path),
        "cost": {"tokens": 0, "elapsed_ms": ev["elapsed_ms"]},
    }
    # 执行细节直接摊开给 LLM 看 —— 它要靠 stdout 里的 VERDICT 行自己理解发生了什么
    payload["stdout"] = ev.get("stdout", "")
    payload["stderr"] = ev.get("stderr", "")
    if result is not None:
        payload["timed_out"] = result.timed_out
        payload["rejected"] = result.rejected
        payload["error_type"] = result.error_type
    logger.info("[research_run_code] verdict=%s %sms", ev["verdict"], ev["elapsed_ms"])
    return payload
