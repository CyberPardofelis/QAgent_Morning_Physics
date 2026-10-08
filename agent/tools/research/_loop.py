"""
agent.tools.research._loop — 生成→执行→验证→反馈→修正 的迭代循环 (P7 阶段 2)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §10 P2

================================ 形态对标 SURVEY_P6.5_REACT.md ============================
沿用综述 ReAct 的那套骨架, 因为它已经被跑通:
    state dict → LLM 决策 → 执行 action → 把错误/观测喂回 → 继续

但科研这边有一处**刻意的不同**: 综述的 action 由 LLM 自由选(搜/抽/写/收尾),
科研的 action 是**受约束的** —— 能把命题变成 proved/refuted 的动作是有限几个
(符号 / 枚举 / Z3 / 反驳), 所以决策只是"下一条验哪条 + 用哪种方法",
且**每个候选动作都有规则兜底**: LLM 决策失败时按 spec 里的 how 顺序走,
不会出现"LLM 一抽风就死循环"。

============================ 为什么循环里必须有"兜底" ==============================
LLM 决策偶尔会给出非法 claim_id 或不存在的方法。综述那边同类问题导致
"续跑不但没省调用, 还把范式层换成了劣质旧路线, 而 UI 里只看得到一行 warning"。
所以这里: 非法决策 → log + 走规则兜底 + 计入 stats.decide_fallback,
**并且在最终报告里如实写出**。

历史:
  - 2026-10-08 P7.2 首版
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace
from agent.prompts_research import format_attempts

from ._claim import normalize_claim
from ._flags import RESEARCH_MAX_RETRY
from ._provenance import record
from .research_run_code import _execute

logger = logging.getLogger(__name__)

#: 验证方法 → 可用性检查。枚举方法必须带 prop, Z3 必须带 constraints。
_METHOD_ORDER = ("solver", "enumerate", "refute", "symbolic", "numeric", "graph")

#: 一次 investigate 默认最多几轮
DEFAULT_MAX_ROUNDS = 3


# ============================================================
# state
# ============================================================
def new_state(question: str, task_hash: str, max_rounds: int) -> dict[str, Any]:
    return {
        "question": question,
        "task_hash": task_hash,
        "phase": "plan",
        "specs": [],
        "by_id": {},
        "pending": [],
        "round": 0,
        "max_rounds": max_rounds,
        "errors": [],
        "finish_requested": False,
        # 本次跑**新增**的证据。收尾时只把这部分落库 —— 见文件末尾 finalize 调用处的注释:
        # 用 ws.load_evidence() 会把历史证据重复 INSERT, 并刷新旧 claim 的 updated_at
        # 让历史 verdict 在"本轮"复活(P4.5 实测踩到)。
        "new_evidence": [],
        "stats": {
            "planned": 0, "dropped_dupes": 0, "verified": 0, "refuted": 0,
            "proved": 0, "partial": 0, "failed": 0, "inconclusive": 0,
            "undecidable": 0, "decide_fallback": 0, "llm_calls": 0,
            "elapsed_ms": 0,
        },
    }


# ============================================================
# 决策
# ============================================================
def decide_next(state: dict[str, Any]) -> dict[str, Any] | None:
    """
    选下一条要验的命题。**纯规则, 不调 LLM**。

    为什么这层不用 LLM:
      - 能产生结论的动作就那几个, 规则决策的可预测性 > LLM 的"聪明"
      - 省 token(P2 每次 investigate 的 LLM 开销应该全花在建模和写代码上)
      - LLM 决策错了会死循环, 规则不会

    选谁: 优先级 = (能否定 > 能证实 > 其它), 同级按 spec 里的 priority。
    理由: 先找反例 —— 找到一条 refuted 就说明用户想证的东西是错的, 越早知道越好。
    """
    pending = [state["by_id"][cid] for cid in state["pending"] if cid in state["by_id"]]
    if not pending:
        return None

    def score(spec: dict[str, Any]):
        how = spec.get("how", "symbolic")
        tier = 0 if how in ("enumerate", "refute", "solver") else 1
        return (tier, spec.get("priority", 99), spec.get("claim_id", ""))

    chosen = sorted(pending, key=score)[0]
    return {"claim_id": chosen["claim_id"], "how": chosen.get("how", "symbolic")}


# ============================================================
# 单条验证
# ============================================================
def verify_one(
    spec: dict[str, Any],
    ws: ResearchWorkspace,
    state: dict[str, Any],
    *,
    decide: dict[str, Any],
    code: str = "",
) -> dict[str, Any]:
    """
    验一条命题, 落证据, 返回该命题当前的判定。

    这里**不调 LLM 生成代码**(调用方负责, 因为 generate 与 verify 分开更好测)——
    code 为空时直接判 failed, 把"没给代码"如实报出来。
    """
    how = decide.get("how") or spec.get("how") or "symbolic"
    scope = spec.get("scope", "")
    cid = spec["claim_id"]

    if not code:
        ev = record(
            ws, kind="code", verdict="failed",
            reason="没有可执行的验证代码 —— 跳过 (调用方未提供)",
            claim_id=cid, extra={"how": how, "scope": scope, "attempt": 0},
        )
        spec["state"] = "failed"
        state.setdefault("new_evidence", []).append(ev)
        return ev

    result, ev = _execute(
        code, ws,
        purpose=f"verify: {spec['claim_text'][:80]}",
        claim_id=cid,
        extra={"how": how, "scope": scope,
               "round": state["round"], "priority": spec.get("priority", 99)},
    )
    spec["state"] = ev["verdict"]
    # 记进本次清单, 收尾时只落这部分(见 new_state 里 new_evidence 的注释)
    state.setdefault("new_evidence", []).append(ev)
    return ev


def retry_with_feedback(
    spec: dict[str, Any],
    ws: ResearchWorkspace,
    attempts: list[dict[str, Any]],
) -> str:
    """
    把失败原因喂回 LLM, 拿修正后的代码。

    这是闭环里"反馈 → 修正"那一环 —— 没有它, LLM 只会重复犯同一个错
    (同样的 prompt 再问一次, 很可能得到同样的 bug)。

    Returns:
        修正后的代码; 拿不到就返回空串
    """
    from .research_verify import _generate_code

    ctx = format_attempts(attempts)
    if not ctx:
        return ""
    code, _raw = _generate_code(
        spec["claim_text"], spec.get("scope", ""),
        spec.get("how", "symbolic"), retry_context=ctx,
    )
    return code


# ============================================================
# 报告 + 落库
# ============================================================
def finalize(
    state: dict[str, Any],
    ws: ResearchWorkspace,
    evidence: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """
    收尾: 渲染报告 + 审计 + 落 research.db。

    Args:
        evidence: **本次新增**的证据(不是 ws.load_evidence() 的全量历史)。
            理由见 new_state 里 new_evidence 的注释 —— 传全量会把历史重复写进
            evidence 表, 并刷新旧 claim 的 updated_at 让历史 verdict 复活。

    Returns:
        (report_md, audit_result)

    ★ 审计结果直接写进报告尾部 —— 让"结论可不可信"和"结论是什么"出现在同一张纸上。

    ★ 审计只看**本次**证据是刻意的: 审计要回答的是"这份报告里的结论站不站得住",
      而报告就是用本次证据渲染的。历史证据里翻的车由 `research_audit` 单独查
      (它读 _evidence.jsonl 全量), 不该让一次新跑的报告永远挂着陈年告警。
    """
    from ._claim import audit_claims, render_report
    from ._db import add_evidence, finish_task, upsert_claim, upsert_task
    from ._provenance import env_fingerprint

    env = env_fingerprint()
    specs = state["specs"]

    # ---- 落库: claim 内容哈希合并, evidence 只 append ----
    try:
        upsert_task(state["task_hash"], state["question"], status="running")
        for spec in specs:
            upsert_claim(
                spec["claim_id"], spec["claim_text"], task_hash=state["task_hash"],
                scope=spec.get("scope", ""), decidable=spec.get("decidable", True),
                verdict=spec.get("state", "pending"),
            )
        for ev in evidence:
            add_evidence(ev, task_hash=state["task_hash"], env=env)
        finish_task(
            state["task_hash"], status="done",
            rounds=state["round"], tokens=state["stats"].get("llm_calls", 0),
        )
    except Exception as e:
        logger.warning("[research.loop] 落库失败(非致命, 报告仍出): %s", e)

    # ---- 渲染 + 审计 ----
    md = render_report(specs, evidence, question=state["question"], env=env)
    aud = audit_claims(specs, evidence, report_md=md)

    if aud["n_errors"]:
        md += (
            f"\n## ⚠ 审计告警({aud['n_errors']} 处硬伤 / {aud['n_warnings']} 处警告)\n\n"
            "> 下列问题由 `research_audit` 自动检出(0 LLM), **报告里的结论在修复前不可信**:\n\n"
        )
        for it in aud["issues"]:
            mark = "🔴" if it["level"] == "error" else "🟡"
            cid = f" [{it['claim_id']}]" if it.get("claim_id") else ""
            md += f"- {mark} `{it['code']}`{cid} — {it['detail']}\n"
    else:
        md += f"\n## 审计\n\n- ✅ {aud['summary']}\n"

    ws.save_report(md)
    return md, aud


# ============================================================
# 主循环 (生成器, 对 UI/工具流式吐进度)
# ============================================================
def run_loop(
    question: str,
    ws: ResearchWorkspace,
    state: dict[str, Any],
    code_provider,
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> Iterator[str]:
    """
    跑完整个循环, 逐步 yield 进度。

    Args:
        code_provider: 可调用对象 (spec, decide, attempt_index) -> code 字符串。
            把它做成注入的, 是为了让测试能不调 LLM 直接喂代码 ——
            P0 的教训是"生成"和"执行"耦合后就没法脱离 LLM 单测了。
    """
    started = time.time()

    for rnd in range(1, max_rounds + 1):
        state["round"] = rnd
        decide = decide_next(state)
        if decide is None:
            yield "\n✅ 所有命题都已得到判定, 提前收尾\n"
            break

        spec = state["by_id"][decide["claim_id"]]
        state["pending"].remove(decide["claim_id"])
        yield (f"\n🔬 第 {rnd}/{max_rounds} 轮 — [{decide['how']}] {spec['claim_text'][:60]}\n")

        # ---- 主尝试 ----
        code = code_provider(spec, decide, 0)
        state["stats"]["llm_calls"] += (1 if code else 0)
        ev = verify_one(spec, ws, state, decide=decide, code=code)
        yield f"   → {ev['verdict']}: {ev['reason'][:90]}\n"

        # ---- 失败才重试: untested/partial 重试同题无意义 ----
        attempts: list[dict[str, Any]] = []
        if ev["verdict"] == "failed" and RESEARCH_MAX_RETRY > 0:
            attempts.append({
                "attempt": 0, "error_type": ev.get("reason", "")[:60],
                "stdout": ev.get("stdout", "")[-800:],
                "stderr": ev.get("stderr", "")[-1500:],
            })
            ws.append_attempt({**attempts[-1], "claim_id": spec["claim_id"]})
            for k in range(1, RESEARCH_MAX_RETRY + 1):
                yield f"   ↻ 带着错误重试 ({k}/{RESEARCH_MAX_RETRY})\n"
                code2 = code_provider(spec, decide, k, feedback=attempts)
                state["stats"]["llm_calls"] += (1 if code2 else 0)
                ev2 = verify_one(spec, ws, state, decide=decide, code=code2)
                yield f"     → {ev2['verdict']}: {ev2['reason'][:80]}\n"
                if ev2["verdict"] != "failed":
                    ev = ev2
                    break
                attempts.append({
                    "attempt": k, "error_type": ev2.get("reason", "")[:60],
                    "stdout": ev2.get("stdout", "")[-800:],
                    "stderr": ev2.get("stderr", "")[-1500:],
                })
                ws.append_attempt({**attempts[-1], "claim_id": spec["claim_id"]})

        # ---- 统计 ----
        v = ev.get("verdict", "failed")
        state["stats"]["verified"] += 1
        if v in state["stats"]:
            state["stats"][v] += 1

        ws.set_status(
            phase="verify", step=rnd, total=max_rounds,
            message=f"{v}: {spec['claim_text'][:40]}",
            stats=state["stats"],
        )

    state["stats"]["elapsed_ms"] = int((time.time() - started) * 1000)

    # ---- 收尾 ----
    # 只把**本次新增**的证据落库。不能用 ws.load_evidence() —— 那是全量历史,
    # 会把上几轮跑的结果又 INSERT 一遍(同一次验证记录 3 次), 并刷新旧 claim 的
    # updated_at 让历史 verdict 在"本轮"复活。详见 research_investigate
    # ._finish_without_verify 的注释。
    md, aud = finalize(state, ws, list(state.get("new_evidence") or []))
    yield (f"\n{'✅' if aud['ok'] else '⚠️'} 审计: {aud['summary']}\n")
    yield f"📄 报告: {ws.path / '_report.md'}\n"


# ============================================================
# 工具层用的小工具
# ============================================================
def specs_from_planning(
    planning: dict[str, Any],
    question: str = "",
) -> tuple[list[dict[str, Any]], int]:
    """
    把 research_plan 的 LLM 输出转成规范化的 specs 列表。

    Args:
        planning: LLM 的 JSON 输出
        question: **原始研究问题**。P4 加 —— 少了它就查不出量词被偷换
            (见 _claim.quantifier_conflict)。P4 实跑里「对所有 n, n²<2n」
            被拆成「存在 n 使 n²<2n」并判 proved, 就是漏了这一步。

    Returns:
        (specs, 重复条数)
    """
    from ._claim import dedupe_claims

    raw_specs = planning.get("specs") or []
    specs = []
    for i, s in enumerate(raw_specs, 1):
        if not isinstance(s, dict):
            continue
        specs.append(normalize_claim(
            s.get("claim_text", ""), s.get("scope", ""), s.get("how", ""),
            quantifier=s.get("quantifier", ""),
            question=question,
            priority=s.get("priority", i),
        ))
    # 建模阶段自己说不可判定的, 也收进来(它们要进"需人判断"区)
    for u in (planning.get("undecidable") or []):
        if isinstance(u, str) and u.strip():
            specs.append(normalize_claim(u.strip(), how="undecidable", question=question))

    # ★ 这里原本还有一道"原问题含 UNDECIDABLE_MARKERS 就整条拦下"的防线, 已删。
    #   它靠关键词黑名单判断问题是否不可判定("优美/有趣/最优算法/本质上") ——
    #   而黑名单只会越长, 且漏得比不拦还危险: 「最小的不存在的素数」「n>100000
    #   的阶乘」这类根本不含任何标记词, 照样放行。
    #   现在统一由 _precheck 闸门在建模后逐条判断(问 LLM "它和原问题是同一件事吗"
    #   + "它本身可判定吗"), 那是语义判断, 不是关键词匹配。
    specs, dropped = dedupe_claims(specs)
    return specs, dropped
