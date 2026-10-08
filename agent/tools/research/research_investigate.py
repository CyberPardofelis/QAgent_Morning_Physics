"""
agent.tools.research.research_investigate — ★ 编排主工具 (P7 阶段 2, L2)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §10 P2

================================ 它是什么 ==================================
一个完整研究任务的入口: 问题 → 建模 → 逐条验证(带修正重试) → 报告 → 审计。
流式产出(和 survey_generate 一样是生成器), UI 能实时看到每轮在干什么。

============================ 为什么所有验证都长成"生成代码" ============================
这个模块看起来绕了一条路: 明明已经有 research_sympy / research_z3 / research_enumerate
这些专用工具了, 为什么还要让 LLM 生成 Python 代码?

因为**统一**。如果 investigate 有三条并行的验证路径(symbolic 走工具、enumerate 走工具、
solver 走工具), 那么:
  - 证据格式会分叉 → audit_claims 得写三套解析
  - 超时/沙箱/重试逻辑得写三遍
  - 测一个 case 要准备三种输入形态

统一成"LLM 写代码 → 沙箱跑 → 机器判定"之后:
  - 证据格式只有一种(code + stdout + verdict)
  - 工具层只管一件事: 生成什么代码
  - LLM 要调 Z3 就写 z4py 代码, 要枚举图就写 networkx 代码 —— **它在沙箱里跑**,
    于是超时保护、证据落盘、审计全都自动生效

代价是 LLM 生成的代码质量不如专用工具(专用工具是手写且确定性的)。
所以两种都在: investigate 适合"开放式探索"(不知道该用什么方法),
专用工具适合"我已经知道要干什么"(快、稳、0 token)。

⚠️ exposed 默认 False (QAGENT_RESEARCH_EXPOSE=0)。
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace, hash_question
from agent.prompts_research import build_modeling_prompt
from agent.tools.registry import register_tool

from ._flags import RESEARCH_EXPOSED
from ._loop import finalize, new_state, run_loop, specs_from_planning
from .research_plan import _extract_json
from .research_verify import _generate_code

logger = logging.getLogger(__name__)


def _make_code_provider(state: dict[str, Any]):
    """
    造一个 code_provider 闭包给 run_loop 用。

    注入而不是硬编码的理由(P0 的教训): 生成"代码"和"执行代码"耦合后,
    测试就没法不烧 token 地跑闭环。做成注入的, 测试直接喂固定代码字符串。
    """

    def provider(spec: dict, decide: dict, attempt: int, feedback: list | None = None):
        from agent.prompts_research import format_attempts

        # 量词必须一路带到代码生成 —— 它决定 verdict 往哪个方向判(P4 加)
        quant = spec.get("quantifier", "")
        if attempt == 0:
            code, _raw = _generate_code(
                spec["claim_text"], spec.get("scope", ""), decide.get("how", "symbolic"),
                quantifier=quant)
        else:
            ctx = format_attempts(feedback or [])
            code, _raw = _generate_code(
                spec["claim_text"], spec.get("scope", ""),
                decide.get("how", "symbolic"), retry_context=ctx,
                quantifier=quant,
            )
        state["stats"]["llm_calls"] += 1
        return code

    return provider


@register_tool(name="research_investigate", cost_tier="L2", is_streaming=True,
               exposed=RESEARCH_EXPOSED)
def research_investigate(
    question: str,
    max_rounds: int = 3,
    task_hash: str = "",
    plan_only: bool = False,
) -> Iterator[str]:
    """
    跑一次完整研究任务(流式)。

    Args:
        question: 研究问题
        max_rounds: 最多验证几轮(每轮一条命题)
        task_hash: 复用工作区; 不传按 question 生成
        plan_only: True = 只建模不验证(等价于 research_plan, 但走同一个流水线)
    """
    from agent.tools.llm import _call_llm

    th = task_hash or hash_question(question)
    ws = ResearchWorkspace(th, question=question)
    state = new_state(question, th, max_rounds)

    # ---------- Phase 1: 建模 ----------
    yield "🧠 建模: 把问题拆成可判定命题...\n"
    raw = _call_llm(build_modeling_prompt(question), temperature=0.3,
                    max_tokens=3000, timeout=180)
    data = _extract_json(raw)
    state["stats"]["llm_calls"] += 1

    if not data:
        yield ("❌ 建模失败: LLM 没吐出可解析的 JSON。\n"
               "   建议: 把问题写具体些(含变量与范围), 或直接用 research_run_code 手写代码。\n")
        ws.set_status(phase="plan", step=1, total=1, message="建模失败")
        return

    specs, dropped = specs_from_planning(data, question=question)
    state["specs"] = specs
    state["by_id"] = {s["claim_id"]: s for s in specs}
    state["stats"]["planned"] = len(specs)
    state["stats"]["dropped_dupes"] = dropped
    state["pending"] = [s["claim_id"] for s in specs if s.get("decidable", True)]

    ws.save_plan({"specs": data.get("specs", []), "normalized": specs})
    yield f"   → {len(specs)} 条命题({len(state['pending'])} 条可自动验证"
    if dropped:
        yield f", 去重 {dropped} 条"
    yield ")\n"

    # ★ 命题回显(P4 加): 把量词和"不可判定"的原因摆到台面上。
    #   P4 实跑发现, 失败模式是"偷换问题后自信作答" —— 问哥德巴赫, 答的是弱哥德巴赫;
    #   问"这段证明是优美的", 判成 proved。报告通篇讲被偷换后的命题, 不回显根本看不出来。
    #   显式打印量词, 是让误读至少**可见**的成本最低的办法。
    _Q_MARK = {"forall": "∀ 所有", "exists": "∃ 存在", "none": "∄ 不存在", "": "?"}
    for s in specs[:6]:
        mark = "✓" if s.get("decidable") else "⨯"
        q = _Q_MARK.get(s.get("quantifier", ""), "?")
        yield f"     {mark} [{q}|{s.get('how')}] {s['claim_text'][:56]}\n"
    if len(specs) > 6:
        yield f"     … 另有 {len(specs) - 6} 条\n"

    blocked = [s for s in specs if not s.get("decidable", True) and s.get("warn")]
    for s in blocked:
        for w in s["warn"]:
            if "量词冲突" in w or "不可判定" in w:
                yield f"     ⚠️ {w}\n"
                yield "        （这条不会被自动验证 —— 验的也不是你问的那个）\n"

    if plan_only:
        ws.set_status(phase="plan", step=1, total=1, message="plan_only")
        yield f"\n📄 建模结果: {ws.path / '_plan.json'}\n"
        return

    if not state["pending"]:
        yield ("⚠️ 没有一条命题被判为可自动验证 —— 全进'需人判断'区了。\n"
               "   这通常说明问题太开放; 试试缩小范围或改成可枚举的形式。\n")
        _finish_without_verify(state, ws)
        return

    # ---------- Phase 2: 验证循环 ----------
    yield from run_loop(
        question, ws, state,
        code_provider=_make_code_provider(state),
        max_rounds=max_rounds,
    )


def _finish_without_verify(state: dict, ws: ResearchWorkspace) -> None:
    """无可验命题时也要出报告 + 审计(哪怕是空的), 别什么都不留"""
    # ★ 传**空列表**, 不是 ws.load_evidence()。P4.5 实测踩到:
    #   load_evidence() 会把**历史**证据一起捞回来, 而 finalize 干两件事:
    #     1. add_evidence() 把它们又 INSERT 一遍 → 同一次验证结果被记录 3 遍
    #        (实测审美题 13:49/14:14/14:22 各一份, evidence 表 21/22/23 与 56/57/58 与 105/106/107)
    #     2. add_evidence() 会顺带刷新 claim.updated_at → 13:49 的旧 `proved`
    #        在"本轮"被复活, 混进基线统计, 让已经修好的防线看起来毫无效果
    #   本次一条都没验, 就该只写本次的状态; 历史证据本来就躺在 evidence 表里,
    #   它是"只 append 永不覆盖"的设计, 不需要靠重复写来保留。
    finalize(state, ws, [])
    ws.set_status(phase="done", step=0, total=0, message="无可验命题")
