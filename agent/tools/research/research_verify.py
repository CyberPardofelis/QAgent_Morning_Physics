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
from agent.prompts_research import (
    build_code_prompt, build_lean_prompt, format_attempts, needs_lean,
    plan_method, template_for,
)
from agent.tools.llm import _call_llm, usage_snapshot
from agent.tools.registry import register_tool

from ._flags import RESEARCH_EXPOSED, RESEARCH_MAX_RETRY, RESEARCH_SANDBOX_TIMEOUT
from ._lean import check_lean, mathlib_available
from ._provenance import VERDICTS, build_evidence, claim_id_for, extract_verdict_marker
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


#: LLM 调用被上游挡住时返回的**错误字符串**特征。这些不是代码,
#: extract_code 抽出来必然是空串 —— 关键是要区分"模型没写代码"和
#: "请求压根没发出去", 两者的处置完全不同(后者重试毫无意义)。
_BLOCKED_MARKERS = (
    "[CostGuard]",
    "cost_guard 阻断",
    "未发出 LLM 请求",
    "未知 LLM provider",
    "429",
    "rate limit",
    "insufficient",
)


def _is_blocked(raw: str) -> bool:
    """判断 LLM 输出是不是「上游把请求挡住了」而不是「模型没给代码」。"""
    if not raw:
        return False
    low = raw.lower()
    return any(m.lower() in low for m in _BLOCKED_MARKERS)


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
    # ★ 机器做的方法选型预判 ★
    # 这套 V2 题是按「裸枚举必死」设计的(1e6/1e7/K9/100皇后),
    # 而实测里 LLM 栽的跟头几乎全是选型。所以先把命题里的规模标记抽出来、
    # 算出「裸枚举来不来得及」, 注入 prompt 让它自己判断 ——
    # 而不是等它写完一坨三重循环再被超时打死。
    plan = plan_method(claim, how)
    if plan["note"]:
        prompt = f"{plan['note']}\n\n{prompt}"
        how = plan["how"]          # 选型预判可以改写建议方法
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


#: 提取 Lean 围栏代码块。Lean prompt 要求 ```lean 包起来。
_LEAN_FENCE_RE = re.compile(r"```(?:lean4?|Lean4?)?\s*\n(.*?)```",
                            re.DOTALL)


def extract_lean_code(text: str) -> str:
    """从 LLM 输出里抽 Lean 代码。没围栏时, 只要看着像 Lean 就整段当代码。"""
    if not text:
        return ""
    m = _LEAN_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    s = text.strip()
    if re.search(r"^\s*(?:import|theorem|example|lemma|axiom|#print)",
                 s, re.MULTILINE):
        return s
    return ""


def _generate_lean(
    claim: str,
    scope: str,
    *,
    retry_context: str = "",
    timeout: int = 180,
) -> tuple[str, str]:
    """调 LLM 生成 Lean 证明。返回 (lean_code, raw)。"""
    raw = _call_llm(
        build_lean_prompt(claim, scope, retry_context),
        temperature=0.1,       # 形式化代码更不能发挥
        top_p=0.8,
        max_tokens=4000,
        timeout=timeout,
    )
    return extract_lean_code(raw), raw


#: 证据强度排序 —— 「更好」= 结论更确定。
#: refuted 排在 proved 后面而不是前面: 两者都是**定论**, 但一道题"被证伪"往往
#: 意味着工具没能理解命题, 而"被证明"更可能是真的完成了任务。
#: 真正的反例会带着具体见证值, 那是另一种信号, 不靠这个排序表达。
_VERDICT_RANK = {
    "proved": 5,
    "refuted": 4,
    "partial": 3,
    "inconclusive": 2,
    "untested": 1,
    "failed": 0,
}


def _better_evidence(a: dict | None, b: dict | None) -> dict | None:
    """两条证据里挑更好的那条(强度相同时取后一条, 保留最新的修正)。"""
    if a is None:
        return b
    if b is None:
        return a
    ra = _VERDICT_RANK.get(a.get("verdict", ""), -1)
    rb = _VERDICT_RANK.get(b.get("verdict", ""), -1)
    return b if rb >= ra else a


# ============================================================
# Lean 硬路由 —— 形式化命题根本不该走 Python
# ============================================================
def _verify_via_lean(
    claim: str,
    ws: ResearchWorkspace,
    *,
    scope: str = "",
    claim_id: str = "",
    max_try: int | None = None,
    llm_timeout: int = 300,
    lean_timeout: int = 900,
) -> dict[str, Any]:
    """
    形式化命题走 Lean kernel, 不生成 Python。

    ★ 为什么是硬路由而不是「LLM 自己选」★
    2026-10-08 V2 实测: hard-015/016 明确要求 Lean, 但链路只会生成 Python,
    LLM 于是用 sympy 硬凑 —— hard-016 最后输出的是一个「有界枚举 + 循环论证」
    的**假 PROVED**。它不是不会 Lean, 是**根本没有 Lean 这条路可选**。
    跟 research_z3/research_enumerate 从不被调用是同一个病: 能力缺路由。

    ★ 判定 = Lean 的四件套, 缺一不可 ★
        exit_code == 0  且  无 ": error:"  且  无 sorry  且  无 sorryAx/自带 axiom
    只看退出码会把 `by sorry` 判成 proved —— 编译器对 sorry 只报 warning。

    重试策略: Lean 的编译器报错是**极其精确**的(unknown constant 会告诉你在哪),
    所以每次失败都把报错原文喂回去, 基本两三轮就能收敛。
    """
    attempts = ws.load_attempts()
    _u_before = usage_snapshot()
    used_llm = 0
    last: dict[str, Any] | None = None
    limit = max_try if max_try is not None else RESEARCH_MAX_RETRY + 1
    blocked_reason = ""

    for attempt in range(limit):
        ws.set_status(phase="verify_lean", step=attempt + 1, total=limit,
                      message=f"Lean 形式化 {claim[:50]}")
        retry_ctx = format_attempts(attempts)
        code, raw = _generate_lean(claim, scope, retry_context=retry_ctx,
                                   timeout=llm_timeout)
        used_llm += 1
        if not code:
            if _is_blocked(raw):
                blocked_reason = raw[:300]
                break
            attempts.append({"attempt": attempt, "error_type": "NoLeanCode",
                             "stderr": f"LLM 没输出 Lean 代码:\n{raw[:500]}"})
            ws.append_attempt(attempts[-1])
            continue

        res = check_lean(code, timeout=lean_timeout)
        proved = bool(res.get("proved"))
        # 形式化后端不产出 VERDICT 标记 —— Lean kernel 本身就是判定者。
        verdict = "proved" if proved else "inconclusive"
        why = _lean_reason(res)

        last = build_evidence(
            ws.evidence_count() + 1,
            kind="lean",
            verdict=verdict,
            reason=why,
            code=code,
            stdout=res.get("output", "")[-4000:],
            stderr="\n".join(res.get("errors", [])[:10]),
            elapsed_ms=0,
            claim_id=claim_id,
            extra={"attempt": attempt + 1, "how": "lean",
                   "scope": scope, "mathlib": res.get("mathlib")},
        )
        ws.append_evidence(last)
        if proved:
            break
        # 把编译器原文喂回去重试 —— Lean 的报错精确, 这是最高效的反馈回路
        attempts.append({
            "attempt": attempt,
            "error_type": why[:120],
            "stdout": res.get("output", "")[-1200:],
            "stderr": "\n".join(res.get("errors", [])[:10])[:1500],
            "claim_id": claim_id,
        })
        ws.append_attempt(attempts[-1])

    _u_after = usage_snapshot()
    if blocked_reason:
        return {"ok": False, "verdict": "blocked", "blocked": True,
                "blocked_reason": blocked_reason, "evidence": [],
                "all_evidence_ids": [], "backend": "lean",
                "summary": f"请求被上游阻断, 没有真的跑过: {blocked_reason}",
                "attempt_count": len(attempts), "workspace": str(ws.path),
                "env": {},
                "cost": {"tokens": _u_after["total_tokens"] - _u_before["total_tokens"],
                         "llm_calls": used_llm, "elapsed_ms": 0}}
    if last is None:
        return {"ok": False, "verdict": "untested", "evidence": [],
                "all_evidence_ids": [], "backend": "lean",
                "summary": "Lean 后端没有产出任何证据",
                "attempt_count": len(attempts), "workspace": str(ws.path),
                "env": {}, "cost": {"tokens": 0, "llm_calls": used_llm,
                                    "elapsed_ms": 0}}
    return {
        "ok": last["verdict"] == "proved",
        "verdict": last["verdict"],
        "evidence": [last],
        "all_evidence_ids": [last.get("id")],
        "backend": "lean",
        "summary": last["reason"],
        "attempt_count": len(attempts),
        "workspace": str(ws.path),
        "env": last.get("env", {}),
        "cost": {"tokens": _u_after["total_tokens"] - _u_before["total_tokens"],
                 "llm_calls": used_llm, "elapsed_ms": 0},
    }


def _lean_reason(res: dict[str, Any]) -> str:
    """把 Lean 检查结果翻成一句给 LLM 看的话。"""
    if not res.get("available"):
        return "Lean 不可用 —— 不能用它做机器判定"
    if res.get("proved"):
        return ("Lean kernel 编译通过: 退出码 0, 无 error, 无 sorry, "
                "公理审计无 sorryAx —— 证明成立")
    bits = []
    if res.get("timed_out") or res.get("exit_code") == -2:
        bits.append("编译超时")
    if res.get("errors"):
        bits.append(f"{len(res['errors'])} 处编译错误")
    if res.get("sorry_used"):
        bits.append("用了 sorry 或 axiom(等于没证)")
    return "Lean 未证成: " + "、".join(bits or ["未知原因"])


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

    # ★ Lean 硬路由 ★
    # 命题点名 Lean/Mathlib 时, 走形式化后端, **不生成 Python**。
    # 没装 Mathlib 时如实说明并回落 Python —— 但要在返回里说清楚走的是哪条路,
    # 不能让调用方以为"已经形式化验证过了"。
    if needs_lean(claim):
        if mathlib_available():
            return _verify_via_lean(claim, ws, scope=scope, claim_id=cid)
        logger.warning(
            "[research_verify] 命题要求形式化, 但 Mathlib 不可用, 回落 Python。"
            "注意: 这**不构成形式化证明**。")

    attempts = ws.load_attempts()
    last_ev: dict[str, Any] | None = None
    best_ev: dict[str, Any] | None = None
    used_llm = 0
    blocked_reason = ""     # 上游阻断(cost_guard/限流)时的原因; 非空 = 压根没跑
    # 成本要真数字。之前 cost.tokens 硬编码 0 —— 成本统计骗人会让人以为
    # 跑 20 道题不花钱, 而「先量再优化」要求用量可见。取本次调用前后的
    # 差值, 这样单题成本独立于别的工具调用。
    _u_before = usage_snapshot()

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
                # ★ 阻断要当场停, 不能当"代码没生成出来"空转 ★
                # 真实事故(2026-10-08): cost_guard 配额耗尽时, _call_llm
                # 返回的是一段**错误字符串**(不是代码)。extract_code 对它
                # 返回 "", 于是走 NoCode 分支 —— 但那不是"模型没写出来",
                # 重试也没用(配额不会因为重试而恢复)。于是白白空转 3 轮,
                # 最后报 "没有产出任何证据"。
                #
                # 后果是评估数据被污染: 12 道题记成判错, 实际是没跑。
                # **"没跑"和"判错"必须分开统计**, 否则准确率虚高或虚低。
                if _is_blocked(raw):
                    attempts.append({"attempt": attempt, "error_type": "Blocked",
                                     "stderr": raw[:500]})
                    ws.append_attempt(attempts[-1])
                    last_ev = None
                    blocked_reason = raw[:300]
                    break
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
            claim=claim,
            extra={"attempt": attempt + 1, "how": how, "scope": scope},
        )
        last_ev = ev

        # ★ 保留**历史最好**的一条, 不是最后一条 ★
        #
        # 为什么: 2026-10-08 V2 实跑发现 BBP 与 100 皇后是**时好时坏**的 ——
        # 同一道题, 有时第 3 轮就 PROVED, 有时烧完 6 轮还是 failed。
        # 只看最后一次, 意味着「第 3 轮明明做对了, 却因为第 6 轮崩了而被记成失败」。
        # 这不是保守, 是**丢掉已经拿到手的证据**。
        #
        # 取"最好"是安全的, 因为每一条都过了机器门 —— 门不认的结论根本没进来。
        # 排序即证据强度: proved > refuted > partial > inconclusive > failed/untested。
        best_ev = _better_evidence(best_ev, ev)

        verdict = ev["verdict"]
        # ★ 只有 failed 才值得让 LLM 改 —— untested 是"代码没写标记",
        # 重试同样代码没用(那是 prompt 问题, P2 的 research_audit 再查)
        #
        # ★★ 机器门推翻了 LLM 的声明时, 必须重试 ★★
        # 2026-10-08 加。机器门(R1 有限枚举冒充全域 / R2 循环论证 /
        # REFUTED 三层见证值检查)会给出一段**很具体的**理由("你只查了
        # |p|,|q|<=200"、"你把 sqrt(2) 是无理数当前提用了"、"那个参考值
        # 是源码里的字面量")。这段理由是机器给的, 比"再试一次"有用得多。
        #
        # ⚠️ 三种触发条件, 区别很关键, 不能一刀切 ★
        #
        #  (a) failed            —— 代码没跑成, 重试无争议
        #  (b) machine_overrode  —— 机器推翻了 LLM 的声明, 有客观缺陷
        #  (c) inconclusive      —— LLM 自己说"定不了"
        #
        # **(partial 不重试)**。sqrt2 那道事故就是这么来的:
        # attempt 1 老老实实判 PARTIAL, 重试时被推着升级成假的 PROVED。
        # partial 是一个**诚实的、有边界的**结论, 重试它等于逼模型多要一点,
        # 而多要的那一点往往不是多算了, 是多说了一句。禁掉。
        #
        # ★ 但「自己判的 partial」和「机器降下来的 partial」必须分开 ★
        # 条件写的是 `declared != verdict`, 所以:
        #   自己判 partial  -> declared == verdict     -> machine_overrode=False -> break
        #   机器把 REFUTED 降成 partial(R3 靶子错了)
        #                   -> declared != verdict     -> machine_overrode=True  -> **重试**
        # 后者恰恰**需要**重试, 因为它身上带着一个客观缺陷:
        # 实测(2026-10-08 V2 hard-011 Petersen 色数) z3 已经证成了「不能 2 着色」,
        # 整体却判了 REFUTED, R3 降成 partial —— 此时缺的是把 3 着色构造修好,
        # 而机器给的判词("输出自相矛盾")会原样喂回去。停在 partial 就再也修不了。
        #
        # 但 (c) 必须重试。V2 hard-001 Pell: 它的代码算出「所有正整数解: []」
        # 然后判 INCONCLUSIVE —— 这是**实现错了却没能自我怀疑**(真解是
        # x=1766319049, y=226153980)。inconclusive 没有任何虚假声明需要保护,
        # 而且它自己的 stdout 就在上下文里, 重试时 LLM 能看到那个空结果并改对。
        # 唯一的风险是烧 token, 而每次尝试都过机器门, 门不认的结论流不下去。
        declared, _raw_marker = extract_verdict_marker(ev.get("stdout", ""))
        machine_overrode = declared is not None and declared != verdict
        gave_up = verdict == "inconclusive"

        # 重试 = (跑挂了) 或 (机器推翻了它) 或 (它自己说定不了)。
        # 除此之外一律 break —— 尤其是它自己给了 partial。
        if not (verdict == "failed" or machine_overrode or gave_up):
            break

        attempts.append({
            "attempt": attempt,
            # machine_overrode 时, reason 就是**机器的判决理由**, 直接当反馈喂回去
            "error_type": ("MachineGate: " + ev.get("reason", ""))[:120]
            if machine_overrode else ev.get("reason", "")[:60],
            "declared_by_llm": declared,
            "judged_by_machine": verdict,
            "gave_up_by_llm": gave_up,
            "stdout": ev.get("stdout", "")[-800:],
            "stderr": ev.get("stderr", "")[-1500:],
            "claim_id": cid,
        })
        ws.append_attempt(attempts[-1])
        if not code_provided:
            code = ""      # 重试时重新让 LLM 生成(带修正上下文)

    # ---- 3. 汇总 ----
    all_ev = ws.load_evidence()
    _u_after = usage_snapshot()
    # ★ 汇总用**最好的一条**, 不是最后一条 ★
    # 见 _better_evidence: BBP / 100 皇后实测时好时坏, 只看最后一次会
    # 把「第 3 轮已经 PROVED」的结果记成「第 6 轮崩了」。证据都在 _evidence.jsonl 里,
    # 一条都没丢, 只是报告该引用哪一条的问题。
    final_ev = best_ev or last_ev
    verdict = final_ev["verdict"] if final_ev else "inconclusive"
    if verdict not in VERDICTS:      # 防御: judge_verdict 已保证合法, 这里兜底
        verdict = "untested"

    # ★ blocked: 上游把请求挡住了, 压根没跑 ★
    # 单独标出来而不是混进 inconclusive —— 统计准确率时必须能把
    # 「没跑」和「判错」分开, 否则数字没意义。
    if blocked_reason:
        return {
            "ok": False,
            "verdict": "blocked",
            "blocked": True,
            "blocked_reason": blocked_reason,
            "evidence": [],
            "all_evidence_ids": [],
            "summary": f"请求被上游阻断, 没有真的跑过: {blocked_reason}",
            "attempt_count": len(attempts),
            "workspace": str(ws.path),
            "env": {},
            "cost": {
                "tokens": _u_after["total_tokens"] - _u_before["total_tokens"],
                "prompt_tokens": _u_after["prompt_tokens"] - _u_before["prompt_tokens"],
                "completion_tokens": _u_after["completion_tokens"] - _u_before["completion_tokens"],
                "llm_calls": used_llm,
                "elapsed_ms": 0,
            },
        }

    return {
        "ok": verdict in ("proved", "refuted"),
        "verdict": verdict,
        "evidence": [final_ev] if final_ev else [],
        "all_evidence_ids": [e.get("id") for e in all_ev],
        "summary": final_ev["reason"] if final_ev else "没有产出任何证据",
        "attempt_count": len(attempts),
        "workspace": str(ws.path),
        "env": final_ev.get("env", {}) if final_ev else {},
        "cost": {
            "tokens": _u_after["total_tokens"] - _u_before["total_tokens"],
            "prompt_tokens": _u_after["prompt_tokens"] - _u_before["prompt_tokens"],
            "completion_tokens": _u_after["completion_tokens"] - _u_before["completion_tokens"],
            "llm_calls": used_llm,
            "elapsed_ms": final_ev.get("elapsed_ms", 0) if final_ev else 0,
        },
    }
