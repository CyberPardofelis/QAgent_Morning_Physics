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

    # 规范化 sections
    # 9-08 17:00 user 拍板: section_type 容错 + 至少 5 段
    #   - LLM 偶尔拼写错: "Method-Compare" / "GAP" / "out look" → 强制 normalize 到 5 段合法值
    #   - LLM 偷懒只返 2 段 → 补到 5 段 (intro/progress/method_compare/gap/outlook)
    VALID_TYPES = {"intro", "progress", "method_compare", "gap", "outlook"}

    def _normalize_sec_type(raw: str) -> str | None:
        """拼写/分隔符容错, 返回合法 section_type 或 None"""
        s = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not s:                          # 9-08 修: 空字符串不参与匹配 (避免 "" in "outlook" 误命中)
            return None
        # 直接命中
        if s in VALID_TYPES:
            return s
        # 模糊匹配
        for v in VALID_TYPES:
            if v in s or s in v:
                return v
        # 关键词 fallback
        if "intro" in s or "background" in s or "introduction" in s:
            return "intro"
        if "progress" in s or "timeline" in s or "milestone" in s:
            return "progress"
        if "method" in s or "compare" in s or "comparison" in s:
            return "method_compare"
        if "gap" in s or "limit" in s or "challenge" in s:
            return "gap"
        if "outlook" in s or "future" in s or "direction" in s:
            return "outlook"
        return None

    normalized = []
    seen_types: set = set()
    for sec in parsed["sections"]:
        if not isinstance(sec, dict):
            continue
        title = (sec.get("title") or "").strip()
        sec_type = _normalize_sec_type(sec.get("section_type", ""))
        if not title or not sec_type:
            continue  # 跳过非法 section (避免污染 outline)
        # 9-08 防重复: 同一 section_type 出现多次, 只保留第一个, 后续的去重
        if sec_type in seen_types:
            continue
        seen_types.add(sec_type)
        sub_points = sec.get("sub_points") or []
        if not isinstance(sub_points, list):
            sub_points = [str(sub_points)]
        sub_points = [str(sp).strip() for sp in sub_points if str(sp).strip()]
        if not sub_points:
            sub_points = ["(待定)"]
        is_transition = str(sec.get("is_transition", "no")).strip().lower()
        normalized.append({
            "title": title,
            "section_type": sec_type,
            "sub_points": sub_points,
            "is_transition": "yes" if is_transition in ("yes", "true", "1") else "no",
        })

    # 9-08 17:00 user 拍板: LLM 偷懒只返 < 5 段时, 用 default 补足到 5 段
    # 9-08 23:50 user 拍板: 升级到 claim-centric 视角, sub_points 从 3 个涨到 5-8 个,
    # 每个 sub_point 用完整判断式陈述句 (跟 SURVEY_OUTLINE_PROMPT 措辞对齐)
    DEFAULT_OUTLINE = [
        {"title": "引言", "sec_type": "intro", "sub_points": [
            "本领域的研究背景和核心科学/工程问题",
            "已有综述覆盖的范围和本综述的差异化定位",
            "本综述采用的分类法和分析框架",
            "本综述的覆盖范围和时间窗口",
            "全文组织结构",
        ], "is_transition": "yes"},
        {"title": "关键进展时间线", "sec_type": "progress", "sub_points": [
            "领域的早期奠基工作和初始共识",
            "第一次方法论突破及其驱动力",
            "中期关键优化和性能提升",
            "近期(2015-2020)的方法论转变",
            "当前主流共识(2020-2025)",
            "演化的内在驱动力: 为什么会有这些转折",
            "未解决的科学/工程问题",
        ], "is_transition": "no"},
        {"title": "方法分类与对比", "sec_type": "method_compare", "sub_points": [
            "按机制维度的分类法和命名规范",
            "按性能维度的对比和基准测试结果",
            "按适用场景的对比和工程取舍",
            "各类方法的核心 trade-off",
            "混合方法/组合方法的最新进展",
            "不同分类法之间的等价性讨论",
        ], "is_transition": "no"},
        {"title": "研究空白与不足", "sec_type": "gap", "sub_points": [
            "技术瓶颈: 当前方法在 X 场景无法扩展",
            "实验限制: 已有 benchmark 的覆盖盲区",
            "理论未解: 基础理论未能解释的现象",
            "跨领域迁移失败的反例",
            "为什么这些问题长期未解决(方法论或生态原因)",
        ], "is_transition": "no"},
        {"title": "未来方向与展望", "sec_type": "outlook", "sub_points": [
            "短期可做方向(1-2 年内可见成果)",
            "长期研究方向(5 年+的开放问题)",
            "跨领域迁移机会(A 领域方法到 B 领域)",
            "标准化展望(基准/评测/复现规范)",
        ], "is_transition": "no"},
    ]
    if len(normalized) < 5:
        existing = {s["section_type"] for s in normalized}
        for default in DEFAULT_OUTLINE:
            if default["sec_type"] not in existing:
                normalized.append({
                    "title": default["title"],
                    "section_type": default["sec_type"],
                    "sub_points": default["sub_points"],
                    "is_transition": default["is_transition"],
                })
                existing.add(default["sec_type"])
                if len(normalized) >= 5:
                    break
        # 9-08 17:25 删: 补足信息是 internal, 不 yield

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
