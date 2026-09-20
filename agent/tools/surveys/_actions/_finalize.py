"""
surveys._actions._finalize — 拆出自原 _actions.py (9-20 体检重构)

包含函数: _finalize_survey
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)
def _finalize_survey(state: Dict[str, Any], ws) -> str:
    """
    收尾 (step 5 真实实现):
      - 算 audit: 候选池 0 引用论文
      - 落 workspace/_audit.md
      - 落 surveys.db theme (已经在 write_section 第一次创建)
      - yield 最终 markdown 报告
    """
    topic = state["topic"]
    pool = state["pool"]
    sections = state["sections"]
    pool_ids = {p["paper_id"] for p in pool}

    # 覆盖度 audit
    cited_ids: set = set()
    for r in sections:
        cited_ids.update(r.get("in_pool", []))
    uncited = pool_ids - cited_ids

    # ========== 9-09 user 拍板: 相关性分布统计 ==========
    # 9-09 加了 per-paper LLM relevance 评分 (_relevance_score 0-10)
    # 0-3 = 误召, 4-6 = 弱相关, 7-8 = 强相关, 9-10 = 核心
    rel_high = []   # 9-10 核心
    rel_mid = []    # 7-8 强相关
    rel_low = []    # 4-6 弱相关
    rel_noise = []  # 0-3 误召 (理论上已被 relevance filter 踢光, 这里兜底)
    rel_no_score = []  # 没有 _relevance_score 字段的 (旧调用 / filter 跳过)
    for p in pool:
        s = p.get("_relevance_score")
        pid = p.get("paper_id", "?")
        title = (p.get("title") or "?")[:80]
        is_adj = p.get("_is_adjacent_field", False)
        item = (pid, title, is_adj, p.get("_relevance_reason", ""))
        if s is None:
            rel_no_score.append(item)
        elif s >= 9:
            rel_high.append(item)
        elif s >= 7:
            rel_mid.append(item)
        elif s >= 4:
            rel_low.append(item)
        else:
            rel_noise.append(item)

    audit_lines = [
        f"# 覆盖度审计 — {topic}",
        "",
        f"- workspace: `{ws.path}`",
        f"- 候选池: {len(pool)}",
        f"- 抽 fact: {len(state['facts'])}",
        f"- 章节: {len(sections)} (theme_id={state.get('theme_id', '?')})",
        f"- ReAct steps: {state['step']}",
        f"- 错误数: {len(state['errors'])}",
        "",
        "## 每章节引用统计",
        "",
        "| 章节 | 类型 | 候选池内 | 候选池外(幻觉) |",
        "|---|---|---|---|",
    ]
    for r in sections:
        audit_lines.append(
            f"| {r['title']} | {r['type']} | "
            f"{len(r['in_pool'])} | {len(r['out_of_pool'])} |"
        )

    # 9-09 加: 相关性分布段 (跟覆盖度并列, 取代单看"0 引用")
    total = len(pool)
    audit_lines.extend([
        "",
        f"## 🎯 候选池相关性分布 (LLM 0-10 评分, 9-09 user 拍板加)",
        "",
        f"- 核心 (9-10): **{len(rel_high)}** 篇 ({len(rel_high)*100//max(1,total)}%)",
        f"- 强相关 (7-8): **{len(rel_mid)}** 篇 ({len(rel_mid)*100//max(1,total)}%)",
        f"- 弱相关 (4-6): {len(rel_low)} 篇 ({len(rel_low)*100//max(1,total)}%)",
        f"- 误召/邻近 (0-3): {len(rel_noise)} 篇 ({len(rel_noise)*100//max(1,total)}%)",
    ])
    if rel_no_score:
        audit_lines.append(
            f"- 未评分 (filter 跳过/旧调用): {len(rel_no_score)} 篇"
        )
    # 列邻近子领域 paper (is_adjacent_field=True), 标 ❌
    adjacent_papers = [item for item in (rel_mid + rel_low + rel_noise) if item[2]]
    if adjacent_papers:
        audit_lines.extend([
            "",
            f"## ⚠️ 邻近子领域 paper (is_adjacent_field=True, 实际不应被综述引用)",
            "",
        ])
        for pid, title, _, reason in adjacent_papers[:20]:
            audit_lines.append(f"- `{pid}`: {title}  —  *{reason[:100]}*")
        if len(adjacent_papers) > 20:
            audit_lines.append(f"- ... 还有 {len(adjacent_papers) - 20} 篇")

    audit_lines.extend([
        "",
        f"## ⚠️ 候选池 0 引用论文 ({len(uncited)} / {len(pool_ids)} = "
        f"{len(uncited)*100//max(1,len(pool_ids))}%)",
        "",
    ])
    for pid in sorted(uncited):
        p = next((x for x in pool if x["paper_id"] == pid), None)
        title = (p or {}).get("title", "?")[:80]
        rel_info = ""
        if p and p.get("_relevance_score") is not None:
            rel_info = f"  (相关性 {p['_relevance_score']}/10)"
        audit_lines.append(f"- `{pid}`: {title}{rel_info}")
    audit_md = "\n".join(audit_lines)
    ws.save_audit(audit_md)
    ws.set_status(
        "done", 1, 1,
        f"theme_id={state.get('theme_id', '?')}, sections={len(sections)}, audit=ok",
    )

    # 最终 markdown 报告
    report = [
        f"## survey_generate 完成 — {topic}",
        "",
        f"- workspace: `{ws.path}`",
        f"- theme_id: `{state.get('theme_id', '?')}` (surveys.db survey_theme.id)",
        f"- 候选池: {len(pool)} 篇",
        f"- 抽 fact: {len(state['facts'])} 条",
        f"- 章节: {len(sections)} 段",
        f"- ReAct steps: {state['step']}",
        f"- 错误数: {len(state['errors'])}",
        f"- 覆盖度审计: `{ws.path}/_audit.md`",
        "",
    ]

    # ========== 9-08 user 拍板: 综述结构先展示, 再展开 ==========
    # 用 outline_sections 列出"完整大纲", 让 user 在收尾报告里先看结构, 再看章节引用
    outline_sections = state.get("outline_sections", [])
    if outline_sections:
        report.extend([
            "## 综述结构 (大纲)",
            "",
        ])
        for i, sec in enumerate(outline_sections):
            sps = " / ".join(sec.get("sub_points", []) or ["(无)"])
            report.append(f"**{i+1}. [{sec.get('section_type', '?')}] {sec.get('title', '?')}**  ({'过渡' if sec.get('is_transition') == 'yes' else '实质'})")
            report.append(f"   - 子点: {sps}")
        report.append("")

    report.extend([
        "### 章节 (section_id, 落库 surveys.db)",
        "",
    ])
    for r in sections:
        in_pids_str = ", ".join(f"[[{pid}]]" for pid in r['in_pool']) or "(无)"
        out_pids_str = ", ".join(f"[[{pid}]]" for pid in r['out_of_pool']) or "(无)"
        report.append(
            f"- `{r['section_id']}` [{r['type']}] {r['title']}\n"
            f"  - 候选池内引用 ({len(r['in_pool'])}): {in_pids_str}\n"
            f"  - 候选池外引用/幻觉 ({len(r['out_of_pool'])}): {out_pids_str}"
        )
    report.append("")
    report.append(f"### 候选池 0 引用 ({len(uncited)} / {len(pool_ids)}):")
    if uncited:
        for pid in sorted(uncited):
            p = next((x for x in pool if x["paper_id"] == pid), None)
            title = (p or {}).get("title", "?")[:80]
            rel_info = ""
            if p and p.get("_relevance_score") is not None:
                rel_info = f"  (相关性 {p['_relevance_score']}/10)"
            report.append(f"- `[[{pid}]]`: {title}{rel_info}")
    else:
        report.append("  (无, 所有候选池 paper 都被引用过)")

    # ========== 9-09 user 拍板: 相关性分布快速摘要 ==========
    # 跟上面的 0 引用列表并列, 一眼能看出 pool 健康度
    adjacent_papers = [
        (pid, title, is_adj, reason)
        for pid, title, is_adj, reason in (rel_mid + rel_low + rel_noise)
        if is_adj
    ]
    report.append("")
    report.append("### 🎯 候选池相关性分布 (LLM 0-10 评分)")
    report.append("")
    report.append(f"- 核心 (9-10): **{len(rel_high)}** 篇")
    report.append(f"- 强相关 (7-8): **{len(rel_mid)}** 篇")
    report.append(f"- 弱相关 (4-6): {len(rel_low)} 篇")
    report.append(f"- 误召/邻近 (0-3): {len(rel_noise)} 篇")
    if adjacent_papers:
        report.append("")
        report.append(f"⚠️ **邻近子领域 paper: {len(adjacent_papers)} 篇**, 建议人工 review 是否真有引用必要")
        for pid, title, _, reason in adjacent_papers[:5]:
            report.append(f"  - `[[{pid}]]`: {title[:60]}  —  *{reason[:80]}*")
        if len(adjacent_papers) > 5:
            report.append(f"  - ... 还有 {len(adjacent_papers) - 5} 篇 (见 `_audit.md`)")

    # ========== 9-09 user 拍板: pre-outline fact 主题对齐审查摘要 (mechanism B) ==========
    fact_stats = state.get("_fact_review_stats")
    if fact_stats and fact_stats.get("total", 0) > 0:
        report.append("")
        report.append("### 📋 pre-outline fact 主题对齐审查 (mechanism B, 9-09 加)")
        report.append("")
        report.append(f"- 主题对齐: **{fact_stats['aligned']}** 条 ({fact_stats['aligned']*100//fact_stats['total']}%)")
        report.append(f"- 邻近子领域 (标 ❌): {fact_stats['misaligned']} 条")
        if fact_stats.get("fallback", 0) > 0:
            report.append(f"- LLM fallback (默认对齐): {fact_stats['fallback']} 条")
        report.append(f"- 总计: {fact_stats['total']} 条 fact")
        if fact_stats["misaligned"] > 0:
            report.append("")
            # 列前 5 条 misaligned fact (只显示 paper_id + claim 短, 不暴露具体 fact_id/全文)
            report.append("**邻近子领域 fact 前 5 条** (不作为综述核心引用):")
            for f in state.get("facts", []):
                if f.get("_is_topic_aligned") is False and len([1 for x in report if "**邻近子领域 fact 前 5 条**" in x]) > 0:
                    pid = f.get("paper_id", "?")
                    claim = (f.get("claim") or "")[:80]
                    reason = (f.get("_alignment_reason") or "")[:80]
                    report.append(f"  - paper `[[{pid}]]`: {claim}...  —  *{reason}*")
                    if len([1 for line in report if line.startswith("  - paper")]) >= 5:
                        break
    if state["errors"]:
        report.append("")
        report.append(f"### ⚠️ 错误 ({len(state['errors'])} 条)")
        for e in state["errors"][:10]:  # 最多列 10 条
            report.append(f"- step {e.get('step', '?')} [{e.get('action', '?')}] {e.get('msg', '?')}")
        if len(state["errors"]) > 10:
            report.append(f"- ... 还有 {len(state['errors']) - 10} 条")
    report.append("")
    report.append(f"### 审计详情")
    report.append(f"见 `{ws.path}/_audit.md`")

    return "\n".join(report)
