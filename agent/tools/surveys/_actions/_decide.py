"""
surveys._actions._decide — 拆出自原 _actions.py (9-20 体检重构)

包含函数: _decide_action, _state_driven_decide
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)
def _decide_action(state: dict[str, Any]) -> dict[str, Any]:
    """
    ReAct LLM 真决策 (step 6 真实实现).

    1. 调 minimax LLM 拿 decision (build_react_prompt)
    2. parse JSON {thought, action, args}
    3. 失败 fallback 到 _state_driven_decide (state 规则保底)

    9-08 19:30 加: finish_rejected 时直接走 state 规则 (不调 LLM, 不浪费 LLM 调用,
    因为 finish 刚被拒, 已经知道下一步该调什么, 没必要再问 LLM)
    """
    from agent.prompts_survey import build_react_prompt
    from agent.tools.llm import _call_llm

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    fallback = _state_driven_decide(state)

    # 9-08 19:30: finish 刚被拒 → 直接用 state 规则 + forced action hint
    if state.get("finish_rejected"):
        forced = state.pop("_forced_action_hint", None)
        if forced:
            action, args = forced
            return {
                "thought": f"[finish 被拒自动执行] {action}({args})",
                "action": action,
                "args": args,
                "_source": "auto_from_finish_reject",
            }
        # 没 hint 就走 state 规则
        return fallback

    try:
        prompt = build_react_prompt(state)
        out = _call_llm(
            prompt,
            provider=_PROVIDER,
            model=_MODEL,
            enable_thinking=False,
            temperature=0.3,
            max_tokens=1000,
            timeout=60,
        )
        if out and "<think>" in out:
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
        m = re.search(r"\{[\s\S]*\}", out or "")
        if not m:
            raise ValueError("LLM 输出无 JSON")
        decision = json.loads(m.group(0))
        action = decision.get("action", "")
        if action not in ("search_more", "extract_facts", "write_outline", "write_section", "finish"):
            raise ValueError(f"未知 action: {action!r}")
        if not isinstance(decision.get("args"), dict):
            decision["args"] = {}
        decision.setdefault("thought", "")
        decision["_source"] = "llm"  # 标记是 LLM 决策, 不是 fallback
        return decision
    except Exception as e:
        # LLM 决策失败 -> fallback state 规则 (spec §1.3 容错)
        return {
            "thought": f"[fallback] LLM 决策失败 ({type(e).__name__}: {str(e)[:80]}), 走 state 规则",
            "action": fallback["action"],
            "args": fallback["args"],
            "_source": "fallback",
        }


def _state_driven_decide(state: dict[str, Any]) -> dict[str, Any]:
    """
    state-driven fallback 规则 (LLM 决策失败时用, 也是测试兜底).

    9-08 user 拍板的固定候选池规则:
      - 候选池 < 20  → search_more (硬规则, 优先)
      - 候选池 ≥ 40 且 fact < 5  → extract_facts (硬规则, 优先)
      - 20 ≤ pool < 40  → LLM 自由(中间区不强制, 让 LLM 按主题判断)
    9-08 user 拍板的拆分:
      - 没 outline → write_outline (硬规则, 写章节前必须先生成大纲)
      - 有 outline 且未写完所有章节 → write_section(section_idx)
      - 写完所有 outline 章节 → finish
    9-08 user 拍板的缺漏反馈:
      - section_gaps 不空 → search_more(用 gap 的 query 扩候选池, 优先)
      - search_more 完成后 → write_section 重写该 gap 章节 (section_idx=gap.section_idx)
    """
    from agent.tools.config import get_min_pool_papers

    pool_size = len(state["pool"])
    fact_count = len(state["facts"])
    section_count = len(state["sections"])
    outline_sections = state.get("outline_sections", [])
    section_gaps = state.get("section_gaps", []) or []

    # ========== 候选池硬规则 (9-08 抽 env QAGENT_SURVEY_MIN_POOL, 默认 30; 充裕 = min+10) ==========
    _MIN_POOL = get_min_pool_papers()
    if pool_size < _MIN_POOL:
        return {"action": "search_more", "args": {"query": state["topic"], "k": 5}}
    if pool_size >= _MIN_POOL + 10 and fact_count < 5:
        return {
            "action": "extract_facts",
            "args": {"paper_ids": [p["paper_id"] for p in state["pool"][:5]]},
        }

    # ========== 缺漏优先: section_gaps 不空 → 补文章 ==========
    # 用最新 gap 的第一个 query 调 search_more 扩池
    # (如果有多个 gap, 一次只补一个, 避免 step 浪费)
    if section_gaps:
        first_gap = section_gaps[0]
        gap_queries = first_gap.get("queries", [])
        if gap_queries:
            return {
                "action": "search_more",
                "args": {"query": gap_queries[0], "k": 5},
            }
        # gap 没 query, 退化: 重写该 section_idx 章节 (让 LLM 重新评估)
        return {
            "action": "write_section",
            "args": {"section_idx": first_gap.get("section_idx", 0)},
        }

    # ========== 大纲规则: 写章节前必须先 write_outline ==========
    if not outline_sections:
        return {"action": "write_outline", "args": {}}

    # ========== 写章节: 按 outline_sections 顺序, 用 section_idx 标识 ==========
    if section_count < len(outline_sections):
        return {
            "action": "write_section",
            "args": {"section_idx": section_count},  # 0-based, 下一章
        }

    # ========== 收尾 ==========
    return {"action": "finish", "args": {}}
