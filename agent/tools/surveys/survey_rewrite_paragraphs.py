"""
agent.tools.surveys.survey_rewrite_paragraphs — 单段重写工具 (L2)

9-27 新增。

为什么需要:
    现在的重写路径只有两种粒度, 都很粗:
      - 整章重写: 20000 max_tokens 全部重新生成, 改一段的代价是重写整章
      - 整体重跑: 从候选池重来, 几十次 LLM 调用
    用户说"第 3 段太空了, 帮我补一下" / "gap 那章最后一段没给反例" 时, 理想是
    **只重写那一段**, 其余原封不动, 上下文还能衔接。

设计:
    - 把已落库的章节 markdown 切成段(split_paragraphs)
    - 用"当前全文(标注哪些要重写) + 前后段上下文 + 候选池 fact"构造 prompt
    - LLM 只返回替换段落, 拼回原段落列表
    - 引用审计: 重写后重新扫 [paper_id], 候选池外的会报出来(和 write_section 同口径)
    - 段落数不变时, 索引稳定; 若 LLM 返回的段数与目标段数不一致, 以 LLM 为准并警告

不依赖 LLM 的部分(split/join/上下文抽取)都在 _paragraphs.py, 有独立单测。
"""
from __future__ import annotations

import logging
import re
from typing import Any

from agent.infra.survey_workspace import SurveyWorkspace, hash_topic
from agent.tools.registry import register_tool

from ._paragraphs import (
    extract_context,
    join_paragraphs,
    render_outline_for_rewrite,
    split_paragraphs,
)

logger = logging.getLogger(__name__)

_PROVIDER = "minimax"
_MODEL = "MiniMax-M3"

# LLM 偶尔会带上 [第 N 段] 标记 / 标题 / 代码块围栏, 清洗掉
_MARK_RE = re.compile(r"^\s*\[第\s*\d+\s*段[^\]]*\]\s*", re.MULTILINE)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^\s*```.*$", re.MULTILINE)


def _clean_rewritten(text: str) -> str:
    """清掉 LLM 输出里不该有的结构标记。"""
    if not text:
        return ""
    t = _MARK_RE.sub("", text)
    t = _HEADING_RE.sub("", t)
    t = _FENCE_RE.sub("", t)
    return t.strip()


def _split_output_blocks(text: str, expected: int) -> list[str]:
    """
    把 LLM 返回的替换段落切成 expected 块。

    优先按空行切; 若块数不符, 尝试按"第 N 段"标记切; 仍不符就按空行平均分。
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text or "") if b.strip()]
    if not blocks:
        return []
    if len(blocks) == expected:
        return blocks
    # 带标记时按标记切
    if re.search(r"\[第\s*\d+\s*段", text or ""):
        parts = re.split(r"\[第\s*\d+\s*段[^\]]*\]", text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) == expected:
            return parts
    return blocks


@register_tool(name="survey_rewrite_paragraphs", is_streaming=False, cost_tier="L2")
def survey_rewrite_paragraphs(
    topic: str,
    section_idx: int,
    para_indices: str = "",
    instruction: str = "",
    dry_run: bool = False,
) -> str:
    """
    重写已生成综述里某一章的指定段落, 其余段落保持不变 (L2 档)。

    何时用:
    - 用户说"第 3 章第 2 段太空了, 补一下" / "gap 那章最后一段没给反例"
    - 对某段不满意, 但不想整章重写(整章重写要 20000 tokens 全重来)

    典型流程:
      1. survey_outline(topic, action="view")          看有哪些章
      2. survey_show_section(topic, section_idx)         看章节有哪几段(可先用, 不调 LLM)
      3. survey_rewrite_paragraphs(topic, section_idx=2, para_indices="1,2",
                                   instruction="补一个 2023 年后的反例")

    参数:
        topic:        综述主题(必须与生成时一致)
        section_idx:  章节序号(0-based, 对应大纲里的 [i])
        para_indices: 要重写的段号, 逗号分隔, 如 "1" / "0,2,3"。
                      不传则重写**整章的所有段落**。
        instruction:  额外要求, 如"更具体一些"/"补一个反例"/"语气收紧"
        dry_run:      True = 只切分并展示结构, 不调 LLM 不落库(0 token, 用来试参数)

    返回:
        markdown 报告: 重写前后的段落对照 + 引用审计
    """
    if not topic or not topic.strip():
        return "❌ topic 不能为空"
    topic = topic.strip()

    try:
        section_idx = int(section_idx)
    except (TypeError, ValueError):
        return f"❌ section_idx 必须是整数, 收到 {section_idx!r}"

    ws = SurveyWorkspace(hash_topic(topic), topic=topic)
    outline = ws.load_outline() or {}
    outline_sections = outline.get("sections") or []
    if not outline_sections:
        return f"❌ 主题「{topic}」还没有大纲, 先跑 survey_generate(phase=\"plan\")。"
    if not 0 <= section_idx < len(outline_sections):
        return (
            f"❌ section_idx={section_idx} 越界 (共 {len(outline_sections)} 章, 0-based)。\n"
            f"用 survey_outline(topic=\"{topic}\", action=\"view\") 看章节列表。"
        )

    sec_meta = outline_sections[section_idx]
    sec_type = sec_meta.get("section_type", "intro")
    sec_title = sec_meta.get("title", "")
    outline_md = "\n".join(f"- {sp}" for sp in (sec_meta.get("sub_points") or []))

    # ---- 找这一章的现有正文(优先 workspace 回退, 再查 DB) ----
    content = _find_section_content(topic, section_idx, sec_title, ws)
    if not content:
        return (
            f"❌ 第 {section_idx} 章「{sec_title}」还没有正文, 无法局部重写。\n"
            f"先跑 survey_generate(topic=\"{topic}\", phase=\"write\") 生成正文。"
        )

    paras = split_paragraphs(content)
    if not paras:
        return f"❌ 第 {section_idx} 章正文为空或无法切分, 无法局部重写。"

    # ---- 解析目标段号 ----
    if para_indices and para_indices.strip():
        try:
            targets = sorted({int(x) for x in para_indices.replace("，", ",").split(",") if x.strip()})
        except ValueError:
            return f"❌ para_indices 必须是逗号分隔的整数, 收到 {para_indices!r}"
    else:
        targets = [p["idx"] for p in paras if p["kind"] != "heading"]

    bad = [t for t in targets if not 0 <= t < len(paras)]
    if bad:
        return (
            f"❌ 段号越界: {bad}。本章共 {len(paras)} 段 (0-{len(paras) - 1})。\n"
            f"用 dry_run=True 看结构。"
        )
    if not targets:
        return f"❌ 没有可重写的段 (本章共 {len(paras)} 段, 全部是标题?)"

    if dry_run:
        return (
            f"## dry_run — 第 {section_idx} 章「{sec_title}」\n"
            f"共 {len(paras)} 段, 将重写第 {targets} 段\n\n"
            f"{render_outline_for_rewrite(paras, targets)}\n"
            f"---\n"
            f"(dry_run: 未调 LLM, 未落库)"
        )

    # ---- 组 prompt ----
    from agent.prompts_survey import build_paragraph_rewrite_prompt
    from agent.tools.llm import _call_llm

    pool = ws.load_pool()
    facts = ws.load_facts()
    prompt = build_paragraph_rewrite_prompt(
        topic=topic,
        section_type=sec_type,
        section_title=sec_title,
        outline_section=outline_md,
        current_structure=render_outline_for_rewrite(paras, targets),
        context=extract_context(paras, targets),
        target_idxs=targets,
        papers=pool,
        facts=facts,
        user_instruction=instruction,
    )

    try:
        out = _call_llm(
            prompt,
            provider=_PROVIDER, model=_MODEL,
            enable_thinking=False, temperature=0.4,
            max_tokens=8000, timeout=240,
        )
    except Exception as e:
        from ._actions import _friendly_error
        return f"❌ LLM 调用失败: {_friendly_error(e, 'rewrite_paragraphs')}"

    if not out or not out.strip():
        return "❌ LLM 返回空内容, 未做任何修改。"
    if "<think>" in out:
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()

    blocks = _split_output_blocks(out, len(targets))
    if not blocks:
        return "❌ 未能从 LLM 输出中解析出替换段落, 未做任何修改。"
    blocks = [_clean_rewritten(b) for b in blocks]
    blocks = [b for b in blocks if b]
    if not blocks:
        return "❌ 清洗后没有有效段落, 未做任何修改。"

    warn = ""
    if len(blocks) != len(targets):
        warn = (
            f"\n⚠️ 目标 {len(targets)} 段, LLM 实际返回 {len(blocks)} 段 —— "
            f"已按返回顺序替换前 {len(blocks)} 段。"
        )

    # ---- 拼回 ----
    by_idx = {p["idx"]: p for p in paras}
    for i, t in enumerate(targets):
        if i < len(blocks):
            by_idx[t]["text"] = blocks[i]
    new_paras = [by_idx[p["idx"]] for p in paras]
    new_content = join_paragraphs(new_paras)

    # ---- 引用审计(和 write_section 同口径) ----
    from ._pool import _audit_section_paper_ids
    pool_ids = {p["paper_id"] for p in pool}
    before_bucket = _audit_section_paper_ids(content, pool_ids)
    after_bucket = _audit_section_paper_ids(new_content, pool_ids)

    # ---- 落库 ----
    sec_id = _save_section_content(topic, section_idx, sec_title, new_content)
    if sec_id is None:
        return (
            f"⚠️ 段落已重写, 但**没能落库**(找不到对应的 section 记录)。\n\n"
            f"{new_content}"
        )

    return (
        f"✅ 已重写第 {section_idx} 章「{sec_title}」的第 {targets} 段\n"
        f"- section_id: {sec_id}\n"
        f"- 段落数: {len(paras)} (不变)\n"
        f"- 字符数: {len(content)} → {len(new_content)}\n"
        f"- 引用(候选池内): {len(before_bucket['in_pool'])} → {len(after_bucket['in_pool'])}\n"
        f"- 引用(候选池外/幻觉): {len(before_bucket['out_of_pool'])} → {len(after_bucket['out_of_pool'])}\n"
        f"{warn}\n\n"
        f"### 重写后的第 {section_idx} 章\n\n{new_content}\n"
        f"---\n"
        f"### 重写段落对照\n\n{_diff_blocks(paras, new_paras, targets)}\n"
        f"\n⚠️ 若引用数骤降, 说明新段落没挂够 [paper_id] —— 审计会标出来。"
    )


def _diff_blocks(before: list[dict[str, Any]], after: list[dict[str, Any]], targets: list[int]) -> str:
    b = {p["idx"]: p["text"] for p in before}
    a = {p["idx"]: p["text"] for p in after}
    out = []
    for i in sorted(targets):
        if i in a:
            out.append(f"**[第 {i} 段]**\n\n- 旧: {b.get(i, '')[:200]}\n- 新: {a[i][:200]}")
    return "\n\n".join(out) or "(无变化)"


def _find_section_content(topic: str, section_idx: int, sec_title: str, ws) -> str:
    """按 (topic, section_idx) 找已写好的章节正文。"""
    from ._crud import list_sections, list_themes
    for t in list_themes():
        if (t.get("name") or "").strip() != topic:
            continue
        secs = list_sections(t["id"])
        if 0 <= section_idx < len(secs):
            return secs[section_idx].get("content_md") or ""
        # 章节数对不上时退回按标题找
        for s in secs:
            if (s.get("title") or "").strip() == (sec_title or "").strip():
                return s.get("content_md") or ""
    return ""


def _save_section_content(topic: str, section_idx: int, sec_title: str, content: str):
    """把新正文写回 survey_section。返回 section_id, 找不到返回 None。"""
    from ._crud import list_sections, list_themes, update_section
    for t in list_themes():
        if (t.get("name") or "").strip() != topic:
            continue
        secs = list_sections(t["id"])
        target = None
        if 0 <= section_idx < len(secs):
            target = secs[section_idx]
        else:
            target = next(
                (s for s in secs if (s.get("title") or "").strip() == (sec_title or "").strip()),
                None,
            )
        if target:
            update_section(target["id"], content_md=content)
            return target["id"]
    return None
