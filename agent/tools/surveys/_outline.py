"""
agent.tools.surveys._outline — 大纲结构规范化 / 校验 / 渲染 (9-27)

为什么单独抽:
    大纲原本只由 LLM 生成(_actions/_write_outline), 规范化逻辑内嵌在那个
    generator 里, 无法复用。用户可编辑大纲后, LLM 生成的 和 用户手改的
    必须走**同一套**校验 —— 否则会出现"LLM 的拼写错误被容错、用户的
    拼写错误不被容错"这种不一致。

    5 段固定模板 (intro/progress/method_compare/gap/outlook) 是**保底结构**,
    不是限制: 校验允许 5 段之外的 custom 类型, 只是在段数不足时才用默认模板补齐。

数据形状 (与 _outline.json / surveys.db 一致):
    {
      "title": str,
      "section_type": "intro"|"progress"|"method_compare"|"gap"|"outlook"|"custom",
      "sub_points": [str, ...],
      "is_transition": "yes"|"no",
    }
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 5 段保底结构
VALID_TYPES = ("intro", "progress", "method_compare", "gap", "outlook")

# 5 段保底模板 —— 仅在 LLM/用户给的段数不足时补齐, 不覆盖已有章节
DEFAULT_OUTLINE: list[dict[str, Any]] = [
    {
        "title": "引言", "sec_type": "intro",
        "sub_points": [
            "本领域的研究背景和核心科学/工程问题",
            "已有综述覆盖的范围和本综述的差异化定位",
            "本综述采用的分类法和分析框架",
            "本综述的覆盖范围和时间窗口",
            "全文组织结构",
        ],
        "is_transition": "yes",
    },
    {
        "title": "关键进展时间线", "sec_type": "progress",
        "sub_points": [
            "领域的早期奠基工作和初始共识",
            "第一次方法论突破及其驱动力",
            "中期关键优化和性能提升",
            "近期的方法论转变",
            "当前主流共识",
            "演化的内在驱动力: 为什么会有这些转折",
            "未解决的科学/工程问题",
        ],
        "is_transition": "no",
    },
    {
        "title": "方法分类与对比", "sec_type": "method_compare",
        "sub_points": [
            "按机制维度的分类法和命名规范",
            "按性能维度的对比和基准测试结果",
            "按适用场景的对比和工程取舍",
            "各类方法的核心 trade-off",
            "混合方法/组合方法的最新进展",
            "不同分类法之间的等价性讨论",
        ],
        "is_transition": "no",
    },
    {
        "title": "研究空白与不足", "sec_type": "gap",
        "sub_points": [
            "技术瓶颈: 当前方法在部分场景无法扩展",
            "实验限制: 已有 benchmark 的覆盖盲区",
            "理论未解: 基础理论未能解释的现象",
            "跨领域迁移失败的反例",
            "为什么这些问题长期未解决(方法论或生态原因)",
        ],
        "is_transition": "no",
    },
    {
        "title": "未来方向与展望", "sec_type": "outlook",
        "sub_points": [
            "短期可做方向(1-2 年内可见成果)",
            "长期研究方向(5 年+的开放问题)",
            "跨领域迁移机会",
            "标准化展望(基准/评测/复现规范)",
        ],
        "is_transition": "no",
    },
]


def normalize_sec_type(raw: str) -> str | None:
    """拼写/分隔符容错 → 合法 section_type; 无法判定返回 custom。"""
    s = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not s:
        return None
    if s in VALID_TYPES:
        return s
    # 匹配时再去掉分隔符, 否则 "out look"→"out_look" 匹配不上 "outlook"
    flat = s.replace("_", "")
    for v in VALID_TYPES:
        if v in flat or flat in v:
            return v
    if any(k in flat for k in ("intro", "background", "introduction")):
        return "intro"
    if any(k in flat for k in ("progress", "timeline", "milestone")):
        return "progress"
    if any(k in flat for k in ("method", "compare", "comparison")):
        return "method_compare"
    if any(k in flat for k in ("gap", "limit", "challenge")):
        return "gap"
    if any(k in flat for k in ("outlook", "future", "direction")):
        return "outlook"
    # 未知类型 → 保留为 custom(而不是丢弃)。用户自定义章节是有价值的,
    # 5 段模板只是保底, 不是白名单。
    return "custom"


def normalize_outline_sections(
    raw_sections: list[Any],
    fill_to_five: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    把任意来源的 sections 列表规范化成内部结构。

    Args:
        raw_sections: 原始列表, 元素可以是 dict 或 str(JSON 行)
        fill_to_five: 段数不足 5 时是否用 DEFAULT_OUTLINE 补齐(保底结构)

    Returns:
        (normalized, warnings) —— warnings 是被丢弃/降级的说明, 供 UI 展示
    """
    warnings: list[str] = []
    normalized: list[dict[str, Any]] = []
    seen_types: set[str] = set()

    for idx, sec in enumerate(raw_sections or []):
        if isinstance(sec, str):
            import json
            try:
                sec = json.loads(sec)
            except Exception:
                warnings.append(f"第 {idx + 1} 段不是合法 JSON, 已跳过")
                continue
        if not isinstance(sec, dict):
            warnings.append(f"第 {idx + 1} 段不是对象, 已跳过")
            continue

        title = (sec.get("title") or "").strip()
        sec_type = normalize_sec_type(sec.get("section_type", ""))
        if not title:
            warnings.append(f"第 {idx + 1} 段缺 title, 已跳过")
            continue
        if not sec_type:
            warnings.append(f"第 {idx + 1} 段 section_type 无法识别, 归为 custom")
            sec_type = "custom"
        # 同一 section_type 只保留第一个, 避免 "progress" 出现两遍导致重复
        if sec_type in seen_types:
            warnings.append(f"第 {idx + 1} 段 section_type={sec_type} 重复, 已跳过")
            continue
        seen_types.add(sec_type)

        sp = sec.get("sub_points") or []
        if isinstance(sp, str):
            sp = [sp]
        if not isinstance(sp, list):
            sp = [str(sp)]
        sub_points = [str(x).strip() for x in sp if str(x).strip()]
        if not sub_points:
            sub_points = ["(待定)"]
            warnings.append(f"第 {idx + 1} 段缺 sub_points, 已填占位")

        is_transition = str(sec.get("is_transition", "no")).strip().lower()
        normalized.append({
            "title": title,
            "section_type": sec_type,
            "sub_points": sub_points,
            "is_transition": "yes" if is_transition in ("yes", "true", "1") else "no",
        })

    if fill_to_five and len(normalized) < 5:
        existing = {s["section_type"] for s in normalized}
        for d in DEFAULT_OUTLINE:
            if d["sec_type"] in existing:
                continue
            normalized.append({
                "title": d["title"],
                "section_type": d["sec_type"],
                "sub_points": list(d["sub_points"]),
                "is_transition": d["is_transition"],
            })
            existing.add(d["sec_type"])
            if len(normalized) >= 5:
                break
        if len(normalized) >= 5:
            warnings.append(f"原大纲只有 {len(normalized) - (5 - len(existing) or 0)} 段, 已用保底模板补齐")

    return normalized, warnings


def render_outline_md(sections: list[dict[str, Any]]) -> str:
    """
    把大纲渲染成给用户/LLM 看的 markdown。

    用 [i] 标号 —— 用户要改某一章时可以直接说"改第 3 章"或"第 3 章标题改成 X"。
    """
    if not sections:
        return "(大纲为空)"
    lines: list[str] = []
    for i, s in enumerate(sections):
        marker = "  ⟵ 过渡章(不强求引用)" if s.get("is_transition") == "yes" else ""
        lines.append(
            f"**[{i}] `{s.get('section_type', '?')}` {s.get('title', '?')}**{marker}"
        )
        for sp in s.get("sub_points") or []:
            lines.append(f"    - {sp}")
        lines.append("")
    return "\n".join(lines).rstrip()


def outline_to_json(sections: list[dict[str, Any]]) -> str:
    """导出成紧凑 JSON, 供 survey_outline(action='update') 回填。"""
    import json
    return json.dumps({"sections": sections}, ensure_ascii=False, indent=2)
