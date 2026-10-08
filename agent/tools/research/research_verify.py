"""
agent.tools.research.research_verify — ★ 科研验证主入口 (P7 阶段 0, L1)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §6.2 / §8

闭环 (每次调用一轮):
    命题 → [1 次 LLM 生成验证代码] → 沙箱执行 → 机器判定 verdict
         → 落证据 _evidence.jsonl → 失败则把错误原文喂回 LLM 修正 (最多 N 轮)
         → 返回结论 + 证据链

为什么是这个形状 (与综述的区别):
- 综述的"对错"是覆盖度审计(引用统计), 本功能的"对错"来自**执行结果**;
- LLM 只负责**生成代码**, 不负责下结论 —— verdict 由 `_provenance.judge_verdict`
  从代码里的 VERDICT 标记 + 执行成功与否算出。这条分工是整个功能立得住的原因。

成本: 1 次验证通常 1 次 LLM (重试时 +1/轮)。cost_tier="L1"。

⚠️ exposed 默认 False (QAGENT_RESEARCH_EXPOSE=0)。调通后再接入。
"""
from __future__ import annotations

import logging
import re
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace, hash_question
from agent.prompts_research import build_code_prompt, format_attempts, template_for
from agent.tools.llm import _call_llm
from agent.tools.registry import register_tool

from ._flags import RESEARCH_EXPOSED, RESEARCH_MAX_RETRY, RESEARCH_SANDBOX_TIMEOUT
from ._provenance import VERDICTS, claim_id_for
from .research_run_code import _execute

logger = logging.getLogger(__name__)

#: 围栏代码块提取。LLM 有时加解释文字, 只取代码部分。
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """
    从 LLM 输出里抽出代码。

    优先取 ```python 围栏; 没有围栏时, 如果整段看起来就是代码(没有中文解说),
    就整段当代码用 —— 有些模型会直接吐裸代码。
    """
    if not text:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    stripped = text.strip()
    # 裸代码的启发式: 以 import / from / 符号 / def 开头, 且不含解释性中文
    if re.match(r"^\s*(import |from |#|def |class )", stripped):
        return stripped
    return ""


# ============================================================
# 内部: 生成验证代码
# ============================================================
def _generate_code(
    claim: str,
    scope: str,
    how: str,
    *,
    retry_context: str = "",
    timeout: int = 180,
    quantifier: str = "",
) -> tuple[str, str]:
    """
    调 LLM 生成验证代码。返回 (code, raw_llm_output)。

    code 为空串说明 LLM 没吐出可执行代码 —— 调用方要把它当一次失败尝试处理。

    Args:
        quantifier: forall/exists/none (P4 加)。写进 prompt 告诉模型 verdict 该往哪个方向判。
    """
    prompt = build_code_prompt(claim, scope, how, retry_context, quantifier=quantifier)
    tmpl = template_for(how)
    if tmpl:
        # 给写法示范比讲规则管用(LLM 漏 VERDICT 那行的高发点)
        prompt += f"\n【同方法的写法示范 —— 照这个骨架改, 别自己发挥】\n```python\n{tmpl}```\n"

    raw = _call_llm(
        prompt,
        temperature=0.2,      # 生成代码要低温, 别让它发挥
        top_p=0.8,
        max_tokens=4000,
        timeout=timeout,
    )
    return extract_code(raw), raw


# ============================================================
# 工具入口
# ============================================================
@register_tool(name="research_verify", cost_tier="L1", exposed=RESEARCH_EXPOSED)
def research_verify(
    claim: str,
    scope: str = "",
    how: str = "symbolic",
    timeout: int = RESEARCH_SANDBOX_TIMEOUT,
    task_hash: str = "",
    code: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    验证一条命题, 返回带证据的结论。

    Args:
        claim: 待验命题(可判定形式)
        scope: 变量域与范围
        how: symbolic / enumerate / numeric / graph / solver
        timeout: 沙箱硬超时秒数
        task_hash: 复用已有工作区; 不传按 claim 自动生成
        code: 直接给验证代码, **跳过 LLM**(0 token), 调试/复现用
        dry_run: 只生成代码不执行

    Returns:
        §6.4 的 5 字段形态: ok / verdict / evidence[] / summary / cost
    """
    th = task_hash or hash_question(claim)
    ws = ResearchWorkspace(th, question=claim)
    cid = claim_id_for(claim, scope)
    ws.append_spec({"claim_text": claim, "scope": scope, "how": how, "claim_id": cid})

    attempts = ws.load_attempts()
    last_ev: dict[str, Any] | None = None
    used_llm = 0

    # 用户显式传了 code → 不该让 LLM 插手重写他写的代码(那是他自己的东西, 他自己改)。
    # 2026-10-08 实测踩坑: 无条件 `code = ""` 导致手写代码失败后偷偷去调 LLM 重新生成。
    code_provided = bool(code.strip())
    max_try = 1 if code_provided else RESEARCH_MAX_RETRY + 1

    for attempt in range(max_try):
        ws.set_status(phase="verify", step=attempt + 1,
                      total=max_try, message=f"验证 {claim[:60]}")

        # ---- 1. 拿代码 ----
        if code:
            cur_code = code
            retry_ctx = ""
        else:
            retry_ctx = format_attempts(attempts)
            cur_code, raw = _generate_code(claim, scope, how, retry_context=retry_ctx)
            used_llm += 1
            if not cur_code:
                attempts.append({"attempt": attempt, "error_type": "NoCode",
                                 "stderr": f"LLM 没输出可执行代码:\n{raw[:500]}"})
                ws.append_attempt(attempts[-1])
                continue

        if dry_run:
            return {
                "ok": True, "verdict": "untested",
                "evidence": [{"id": "ev_0001", "kind": "code", "code": cur_code,
                              "verdict": "untested", "reason": "dry_run: 未执行"}],
                "summary": f"dry_run —— 生成代码如下, 未执行 (attempt {attempt+1})",
                "generated_code": cur_code,
                "workspace": str(ws.path),
                "cost": {"tokens": 0, "elapsed_ms": 0},
            }

        # ---- 2. 跑 + 判 + 记 ----
        _, ev = _execute(
            cur_code, ws,
            purpose=f"verify: {claim[:80]}",
            timeout=timeout,
            claim_id=cid,
            extra={"attempt": attempt + 1, "how": how, "scope": scope},
        )
        last_ev = ev

        verdict = ev["verdict"]
        # 只有 failed 才值得让 LLM 改 —— untested 是"代码没写标记",
        # 重试同样代码没用(那是 prompt 问题, P2 的 research_audit 再查)
        if verdict != "failed":
            break

        attempts.append({
            "attempt": attempt,
            "error_type": ev.get("reason", "")[:60],
            "stdout": ev.get("stdout", "")[-800:],
            "stderr": ev.get("stderr", "")[-1500:],
            "claim_id": cid,
        })
        ws.append_attempt(attempts[-1])
        if not code_provided:
            code = ""      # 重试时重新让 LLM 生成(带修正上下文)

    # ---- 3. 汇总 ----
    all_ev = ws.load_evidence()
    verdict = last_ev["verdict"] if last_ev else "inconclusive"
    if verdict not in VERDICTS:      # 防御: judge_verdict 已保证合法, 这里兜底
        verdict = "untested"

    return {
        "ok": verdict in ("proved", "refuted"),
        "verdict": verdict,
        "evidence": [last_ev] if last_ev else [],
        "all_evidence_ids": [e.get("id") for e in all_ev],
        "summary": last_ev["reason"] if last_ev else "没有产出任何证据",
        "attempt_count": len(attempts),
        "workspace": str(ws.path),
        "env": last_ev.get("env", {}) if last_ev else {},
        "cost": {
            "tokens": 0,
            "llm_calls": used_llm,
            "elapsed_ms": last_ev.get("elapsed_ms", 0) if last_ev else 0,
        },
    }
