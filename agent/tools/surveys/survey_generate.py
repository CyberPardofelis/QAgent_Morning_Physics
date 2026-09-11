"""
agent.tools.surveys.survey_generate — 综述生成主入口 (L2 工具)

从原 surveys.py 抽出 (P1.7 重构):
- @register_tool survey_generate (顶层入口)
- ReAct 主循环 (Phase A 候选池 → step 1+ LLM 决策 → 5 action → 收尾)
- stream yield 进度 + 章节内容给 UI chat stream_chunk signal

其它全部:
- CRUD/health_check        → _crud.py
- 候选池/引用/relevance  → _pool.py
- 5 action + decision     → _actions.py
- 独立 audit 工具         → survey_audit.py
"""
import logging
from typing import Iterator, Dict, Any

from agent.tools.registry import register_tool
from agent.infra.survey_workspace import SurveyWorkspace, hash_topic

from ._pool import _build_candidate_pool
from ._actions import (
    _decide_action,
    _dispatch_action,
    _append_react_trace,
    _finalize_survey,
)

logger = logging.getLogger(__name__)


@register_tool(name="survey_generate", is_streaming=True, cost_tier="L2")
def survey_generate(
    topic: str,
    max_steps: int = 20,
    max_pool: int = 50,                # 9-08 17:00 加: 候选池上限, 默认 50 (LLM 可显式传更大值)
    dry_run: bool = False,
) -> Iterator[str]:
    """
    生成综述 (P6.5 ReAct 改写) — L2 档, 流式 yield 进度 + 章节内容.

    ReAct 主循环:
      Step 0: 初始候选池 (Phase A, 0 LLM)
      Step 1..max_steps: LLM 观察 state → 决策 action → 派发 → yield obs
        action: search_more / extract_facts / write_section / finish
      收尾: theme + sections 落 surveys.db, _audit.md, 最终 markdown

    Args:
        topic: 综述主题, 例 "量子纠缠度量计算"
        max_steps: ReAct 循环上限 (默认 20, spec §7.1)
        dry_run: True = 不调 LLM, 只走 Phase A 候选池构建

    Yields:
        流式进度 + 章节内容, chat stream_chunk signal 消费
        (is_streaming=True 标记, agent.py 派发层识别为 generator)
    """
    if not topic or not topic.strip():
        yield "❌ topic 不能为空\n"
        return

    topic = topic.strip()

    # 0. workspace 准备
    topic_hash = hash_topic(topic)
    ws = SurveyWorkspace(topic_hash, topic=topic)
    ws.set_status("init", 0, 0, f"topic={topic}")

    yield f"🔍 开始综述: {topic}\n"

    # 1. Step 0: Phase A 初始候选池 (0 LLM, summary_items.db + chroma + arxiv fallback, 9-08 多源召回)
    # 9-09 改: 9-09 19:30 user 拍板 返 (pool, build_status) tuple, 让候选池为空时能区分 arxiv 失败原因
    pool, build_status = _build_candidate_pool(topic, max_pool=max_pool)
    if not pool:
        ws.set_status("failed", 0, 0, "pool empty")
        arxiv_status = build_status.get("arxiv_status", "unknown")
        # 9-09 user 拍板: 检测中文 topic, 给英文关键词建议 (arxiv 是英文 API)
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in topic)
        zh_hint = (
            "\n- 💡 检测到中文 topic, arxiv 是英文 API, 建议用英文关键词重试 "
            "(例: graph state hypergraph state entanglement)"
            if has_chinese else ""
        )
        yield (
            f"❌ 候选池为空\n"
            f"- topic: {topic!r}\n"
            f"- summary_items.db 没匹配到 paper\n"
            f"- arxiv fallback: {arxiv_status}\n"
            f"- 建议: 换个主题词 (优先英文), 或先跑 `抓文 → analyze_paper` 准备数据"
            f"{zh_hint}\n"
        )
        return
    ws.save_pool(pool)
    yield f"📚 初始候选池: {len(pool)} 篇\n\n"

    # state 初始化
    # 9-08 user 拍板: 加 outline 字段, 把"写章节"拆成 write_outline + write_section 两个 action
    # outline_sections: list of {title, section_type, sub_points, is_transition}
    # 9-08 user 拍板: 加 section_gaps 字段, LLM 写章节时如果觉得还缺文章, 会在 markdown 末尾
    #   标注 `<!-- QAgent: missing_queries=[...] -->`  →  system 解析后存到 section_gaps
    #   → action_finish 看到 section_gaps 不空就拒绝结束, ReAct 回到 search_more 补
    state: Dict[str, Any] = {
        "topic": topic,
        "topic_hash": topic_hash,
        "pool": pool,
        "facts": [],
        "sections": [],
        "outline": None,                # 综述大纲原始 LLM 输出 (str)
        "outline_sections": [],         # 大纲结构化: [{title, section_type, sub_points, is_transition}, ...]
        "section_gaps": [],             # 章节缺漏: [{section_idx, title, queries: [...]}]
        "unused_pool_papers": [],       # 9-08 user 拍板: 候选池中未被任何章节引用的 paper_ids
                                        # build_react_prompt 提示 LLM, 让写章节时能补充引用
        "errors": [],
        "step": 0,
        "finish_requested": False,
        "max_steps": max_steps,
    }

    if dry_run:
        ws.save_audit(
            f"# dry_run audit\n- topic: {topic}\n- pool_size: {len(pool)}\n- workspace: {ws.path}\n"
        )
        yield (
            f"## survey_generate dry_run\n\n"
            f"- topic: {topic}\n"
            f"- 候选池: {len(pool)}\n"
            f"- workspace: `{ws.path}`\n"
            f"- 跳过 ReAct 循环 (dry_run=True)\n"
        )
        return

    # 2. ReAct 主循环
    while state["step"] < state["max_steps"] and not state["finish_requested"]:
        state["step"] += 1
        yield f"\n--- Step {state['step']} ---\n"

        # 2.1 LLM 决策 (step 1 用 mock, 后续 step 接真实 _call_llm + parse_decision)
        decision = _decide_action(state)

        # 9-08 17:25 user 拍板: 主循环只显示"调用了 X 工具", 不显示 thought (LLM 内部想法) 和 args (query 等详情)
        yield f"➡️ 调用工具: {decision['action']}\n"

        # 2.2 派发 action (try/except 包裹, 失败塞 state["errors"], 不中断)
        try:
            for chunk in _dispatch_action(decision, state, ws):
                if chunk:
                    yield chunk
        except Exception as e:
            from ._actions import _friendly_error
            friendly = _friendly_error(e, decision["action"])
            state["errors"].append({
                "step": state["step"],
                "action": decision["action"],
                "msg": friendly,
            })
            yield f"👁️ 观察: ❌ {friendly}\n"

        # 2.3 trace 落盘 (spec §6 _react_trace.jsonl)
        _append_react_trace(ws, {
            "step": state["step"],
            "thought": decision["thought"],
            "action": decision["action"],
            "args": decision.get("args", {}),
            "n_errors": len(state["errors"]),
        })

    # 3. 收尾 (step 1 stub — 后续 step 5 落 surveys.db + audit + 最终 markdown)
    final = _finalize_survey(state, ws)
    yield final
