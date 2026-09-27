"""
agent.tools.surveys.survey_outline — 大纲查看 / 编辑工具 (L0, 0 LLM)

9-27 新增。

为什么需要:
    之前大纲由 survey_generate 的 ReAct 循环内部生成, 生成后**立刻锁死**
    (_actions/_write_outline: `if state.get("outline_sections"): return`),
    然后同一次运行里把 5 章全写完。用户看到不对只能整体重跑 ——
    这正是"综述工具很别扭"的来源之一。

    工作区其实早就把大纲持久化到 _outline.json 且可回读, 只是没有编辑入口。
    本工具提供那个入口: 先跑 survey_generate(phase="plan") 生成大纲,
    用本工具查看/修改, 再跑 survey_generate(phase="write") 按改后的大纲写正文。

零 LLM 调用(L0 档): 纯本地 JSON 读写 + 校验。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from agent.infra.survey_workspace import SurveyWorkspace, hash_topic
from agent.tools.registry import register_tool

from ._outline import normalize_outline_sections, outline_to_json, render_outline_md

logger = logging.getLogger(__name__)


@register_tool(name="survey_outline", is_streaming=False, cost_tier="L0")
def survey_outline(
    topic: str,
    action: str = "view",
    sections: str = "",
) -> str:
    """
    查看 / 修改某主题的综述大纲 (0 LLM 调用, 纯本地操作)。

    何时用:
    - 用户说"看看大纲"/"改一下大纲"/"第 3 章标题改成 X"/"加一章 XX" → 先 view 再 update
    - 写完正文后想调整章节结构 → update 之后可再跑 survey_generate(phase="write")

    典型流程:
      1. survey_generate(topic="X", phase="plan")        # 生成大纲并停住
      2. survey_outline(topic="X", action="view")        # 展示给用户看
      3. survey_outline(topic="X", action="update", sections='{"sections":[...]}')
      4. survey_generate(topic="X", phase="write")       # 按改后的大纲写正文

    参数:
        topic:   综述主题(必须与生成时完全一致, 大纲按 topic hash 定位)
        action:  "view" 查看 / "update" 覆盖更新
        sections: action="update" 时必填。JSON 字符串, 形如
                  {"sections": [{"title": "...", "section_type": "intro",
                                "sub_points": ["...", "..."], "is_transition": "no"}]}
                  也接受顶层直接是数组。section_type 可用 5 段保底类型
                  (intro/progress/method_compare/gap/outlook) 或 "custom"。

    返回:
        markdown 格式的大纲; update 时附带规范化警告
    """
    if not topic or not topic.strip():
        return "❌ topic 不能为空"

    topic = topic.strip()
    action = (action or "view").strip().lower()
    if action not in ("view", "update"):
        return f"❌ action 只能是 'view' 或 'update', 收到 {action!r}"

    ws = SurveyWorkspace(hash_topic(topic), topic=topic)

    # ---------------- view ----------------
    if action == "view":
        raw = ws.load_outline()
        if not raw:
            return (
                f"❌ 主题「{topic}」还没有大纲。\n"
                f"先跑 survey_generate(topic=\"{topic}\", phase=\"plan\") 生成大纲。"
            )
        sections = raw.get("sections") or []
        if not sections:
            return f"❌ 主题「{topic}」的大纲是空的, 建议重跑 phase=\"plan\"。"
        body = render_outline_md(sections)
        return (
            f"## 综述大纲 — {topic}\n"
            f"共 {len(sections)} 章 (workspace: `{ws.path}`)\n\n"
            f"{body}\n\n"
            f"---\n"
            f"要改就用 `survey_outline(topic=\"{topic}\", action=\"update\", sections=...)`,\n"
            f"改完再跑 `survey_generate(topic=\"{topic}\", phase=\"write\")`。"
        )

    # ---------------- update ----------------
    if not sections or not sections.strip():
        return "❌ action=\"update\" 时必须传 sections (JSON 字符串)"

    try:
        parsed = json.loads(sections)
    except json.JSONDecodeError as e:
        return f"❌ sections 不是合法 JSON: {e}"

    if isinstance(parsed, dict):
        raw_sections = parsed.get("sections")
    elif isinstance(parsed, list):
        raw_sections = parsed
    else:
        return f"❌ sections 应是 {{\"sections\": [...]}} 或直接的数组, 收到 {type(parsed).__name__}"

    if not isinstance(raw_sections, list):
        return "❌ sections 字段必须是数组"

    # 用户编辑时不做 5 段补齐 —— 用户删到 4 章就是想要 4 章。
    # 只有 LLM 生成路径才需要保底模板填充。
    normalized, warnings = normalize_outline_sections(raw_sections, fill_to_five=False)
    if not normalized:
        msgs = "\n".join(f"  - {w}" for w in warnings) or "  (无详细信息)"
        return f"❌ 规范化后没有有效章节:\n{msgs}"

    try:
        existing = ws.load_outline() or {}
        ws.save_outline({
            "sections": normalized,
            "raw": existing.get("raw", ""),
            "edited_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "edited": True,
        })
    except Exception as e:
        logger.exception("[survey_outline] save failed")
        return f"❌ 保存大纲失败: {e}"

    warn_md = ""
    if warnings:
        warn_md = "\n\n**规范化提示**:\n" + "\n".join(f"- {w}" for w in warnings)

    return (
        f"✅ 大纲已更新 — {topic}\n"
        f"共 {len(normalized)} 章 (workspace: `{ws.path}`){warn_md}\n\n"
        f"{render_outline_md(normalized)}\n\n"
        f"---\n"
        f"下一步: `survey_generate(topic=\"{topic}\", phase=\"write\")` 按这份大纲写正文。"
    )


@register_tool(name="survey_outline_export", is_streaming=False, cost_tier="L0")
def survey_outline_export(topic: str) -> str:
    """
    导出某主题大纲的可编辑 JSON 模板 (0 LLM)。

    用途: 用户想把大纲拿到外部编辑器改, 改完再粘回来 update。
    """
    if not topic or not topic.strip():
        return "❌ topic 不能为空"
    topic = topic.strip()
    ws = SurveyWorkspace(hash_topic(topic), topic=topic)
    raw = ws.load_outline()
    sections: list[dict[str, Any]] = (raw or {}).get("sections") or []
    if not sections:
        return f"❌ 主题「{topic}」还没有大纲, 先跑 survey_generate(phase=\"plan\")。"
    return (
        f"## 可编辑大纲 JSON — {topic}\n\n"
        f"改完贴回来: `survey_outline(topic=\"{topic}\", action=\"update\", sections=<改后的 JSON>)`\n\n"
        f"```json\n{outline_to_json(sections)}\n```"
    )
