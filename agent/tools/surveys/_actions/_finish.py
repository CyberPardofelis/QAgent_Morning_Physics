"""
surveys._actions._finish — 拆出自原 _actions.py (9-20 体检重构)

包含函数: action_finish
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)
def action_finish(args: dict[str, Any], state: dict[str, Any], ws) -> Iterator[str]:
    """
    finish action (9-08 19:30 三重校验 + 自动打回执行):
      1. 章节数校验: section_count < len(outline_sections) → 拒绝, 自动触发 write_section
      2. 缺漏校验: state["section_gaps"] 不空 → 拒绝, 自动触发对应工具
      3. 全部通过 → 真正 finish (state["finish_requested"] = True)

    9-08 19:30 user 拍板:
      - 0 引用候选 不算"打回重写" 条件 (删 9-08 18:13 的 unused 校验)
      - finish 被拒要 **自动** 调用相关工具去解决, 不等用户说
        (实现: 标记 state["finish_rejected"]=True, 主循环下一轮 _decide_action 不调 LLM, 走 state 规则)
    """
    outline_sections = state.get("outline_sections", []) or []
    section_count = len(state.get("sections", []))
    gaps = state.get("section_gaps", []) or []

    # ========== 1. 章节数校验 ==========
    if outline_sections and section_count < len(outline_sections):
        missing_idx = list(range(section_count, len(outline_sections)))
        missing_titles = ", ".join(
            f"'{outline_sections[i].get('title', '?')}'" for i in missing_idx[:5]
        )
        more = f" 等 {len(missing_idx)} 章" if len(missing_idx) > 5 else ""
        # 9-08 19:30 拍板: finish 被拒要自动执行相关工具 (write_section)
        state["finish_rejected"] = True
        state["_forced_action_hint"] = ("write_section", {"section_idx": section_count})
        yield (
            f"❌ finish 被拒: 章节未写完! "
            f"已写 {section_count}/{len(outline_sections)} 章, "
            f"还差 {len(missing_idx)} 章 ({missing_titles}{more})\n"
            f"👁️ 自动触发: write_section(section_idx={section_count})\n"
        )
        return

    # ========== 2. 缺漏校验 ==========
    if gaps:
        gap_titles = ", ".join(f"'{g.get('title', '?')}'" for g in gaps[:5])
        # 9-08 19:30 拍板: 自动执行 — 优先 search_more (用 gap query), 没 query 就 write_section
        first_gap = gaps[0]
        gap_queries = first_gap.get("queries", [])
        if gap_queries and "内容过短" not in gap_queries[0]:
            # 缺漏是 search_more 类的, 自动调 search_more
            state["finish_rejected"] = True
            state["_forced_action_hint"] = ("search_more", {"query": gap_queries[0], "k": 5})
            yield (
                f"❌ finish 被拒: 还有 {len(gaps)} 个章节缺漏/空章节未补 ({gap_titles} 等)\n"
                f"👁️ 自动触发: search_more(query='{gap_queries[0]}')\n"
            )
        else:
            # 缺漏是"内容过短" 或没 query, 自动调 write_section 重写
            target_idx = first_gap.get("section_idx", section_count)
            state["finish_rejected"] = True
            state["_forced_action_hint"] = ("write_section", {"section_idx": target_idx})
            yield (
                f"❌ finish 被拒: 还有 {len(gaps)} 个章节缺漏/空章节未补 ({gap_titles} 等)\n"
                f"👁️ 自动触发: write_section(section_idx={target_idx}) 重写缺漏章节\n"
            )
        return

    # ========== 3. 全部通过 → 真正 finish ==========
    state["finish_rejected"] = False
    state["finish_requested"] = True
    yield (
        f"👁️ 观察: ✅ 收尾确认 (章节 {section_count}/{len(outline_sections)} 已写完, 无缺漏) → 🏁\n"
    )
