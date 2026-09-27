"""
surveys._actions._write_outline — 拆出自原 _actions.py (9-20 体检重构)

包含函数: action_write_outline
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

from .._outline import normalize_outline_sections
from ._helpers import _friendly_error

logger = logging.getLogger(__name__)
def action_write_outline(args: dict[str, Any], state: dict[str, Any], ws) -> Iterator[str]:
    """
    write_outline action (9-08 新增, user 拍板拆分 write_section):
      - 调 minimax LLM 走 build_outline_prompt 生成综述大纲(5 段固定: intro/progress/method_compare/gap/outlook)
      - 解析 JSON: {sections: [{title, section_type, sub_points, is_transition}, ...]}
      - 存 state["outline"] (raw) + state["outline_sections"] (结构化 list)
      - 落 workspace/_outline.json (SurveyWorkspace.save_outline 已有)
      - outline_sections 锁住后, write_section 按 section_idx 顺序写

    大纲生成失败不重试(LLM 调用本身就贵, 失败就让 LLM 下一轮重选 action)
    """
    from agent.prompts_survey import build_outline_prompt
    from agent.tools.config import get_max_section_tokens
    from agent.tools.llm import _call_llm

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    # 已存在大纲 → 跳过(避免 LLM 重复调)
    if state.get("outline_sections"):
        # 9-08 17:25 改: silent 跳过, 不 yield 啰嗦
        return

    yield "📋 生成综述大纲\n"

    try:
        prompt = build_outline_prompt(topic=state["topic"], papers=state["pool"])
        out = _call_llm(
            prompt,
            provider=_PROVIDER,
            model=_MODEL,
            enable_thinking=False,
            temperature=0.3,
            max_tokens=get_max_section_tokens(),   # 9-08 23:50 user 拍板: 10000 → 20000
            timeout=180,
        )
    except Exception as e:
        friendly = _friendly_error(e, "write_outline")
        state["errors"].append({
            "step": state["step"], "action": "write_outline",
            "msg": friendly,
        })
        yield f"  ❌ LLM 失败: {friendly}\n"
        return

    # 剥 <think>
    if out and "<think>" in out:
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()

    # parse JSON {sections: [...]}
    parsed: dict[str, Any] | None = None
    if out:
        # 1) 直接
        try:
            v = json.loads(out.strip())
            if isinstance(v, dict) and "sections" in v:
                parsed = v
        except (json.JSONDecodeError, SyntaxError):
            pass
        # 2) regex 抓最大 {...}
        if parsed is None:
            m = re.search(r"\{[\s\S]*\}", out)
            if m:
                try:
                    v = json.loads(m.group(0))
                    if isinstance(v, dict) and "sections" in v:
                        parsed = v
                except (json.JSONDecodeError, SyntaxError):
                    pass
        # 3) 修常见尾巴
        if parsed is None:
            for tail in ['"}]', '"}', '}', ']']:
                try:
                    s = out.strip().rstrip(",\n ") + tail
                    v = json.loads(s)
                    if isinstance(v, dict) and "sections" in v:
                        parsed = v
                        break
                except (json.JSONDecodeError, SyntaxError):
                    pass

    if not parsed or not isinstance(parsed.get("sections"), list) or not parsed["sections"]:
        state["errors"].append({
            "step": state["step"], "action": "write_outline",
            "msg": f"JSON parse 失败: {(out or '')[:200]}",
        })
        yield "  ❌ 大纲 JSON parse 失败, 下一轮重试\n"
        return

    # 9-27: 规范化逻辑抽到 _outline.normalize_outline_sections, 与"用户手改大纲"
    # 共用同一套校验 —— 否则会出现"LLM 拼写错误被容错、用户拼写错误不被容错"。
    normalized, warnings = normalize_outline_sections(parsed["sections"], fill_to_five=True)
    for w in warnings:
        logger.info("[write_outline] %s", w)

    if not normalized:
        state["errors"].append({
            "step": state["step"], "action": "write_outline",
            "msg": "规范化后 sections 为空 (全部非法, section_type 不在 5 段列表)",
        })
        # 9-08 17:25 删: 错误时只 yield 一次, 简短
        yield "❌ 大纲生成失败\n"
        return

    # 存 state + 落盘 (silent, 9-08 17:25 删了 💾 落盘 yield)
    state["outline"] = out or ""
    state["outline_sections"] = normalized
    try:
        ws.save_outline({"sections": normalized, "raw": out or ""})
    except Exception as e:
        logger.warning("[write_outline] save_outline 失败: %s", e)

    # 9-08 17:25 改: 大纲 yield 简洁版, 只显示章节数 + 标题列表
    yield f"✅ 大纲 {len(normalized)} 章\n"
    for i, sec in enumerate(normalized):
        yield f"  {i+1}. [{sec['section_type']}] {sec['title']}\n"
