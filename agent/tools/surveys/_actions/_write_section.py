"""
surveys._actions._write_section — 拆出自原 _actions.py (9-20 体检重构)

包含函数: action_write_section
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)
def action_write_section(args: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """
    write_section action (step 4 真实实现):
      - 调 minimax LLM 写单个章节 markdown
      - 复用 P6 build_section_prompt + _audit_section_paper_ids
      - 自动重写 (P6.3): out_of_pool > 0 时加约束重试, 最多 2 次
      - 章节落 surveys.db (create_section, paper_ids 逗号分隔)
      - step 4 用非流式 LLM 拿完整 markdown 再 yield (真流式要 _call_llm_stream, 留 P+)
    """
    from agent.tools.llm import _call_llm
    from agent.tools.config import get_max_section_tokens
    from agent.prompts_survey import build_section_prompt

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    # 9-08 user 拍板: 章节内容必须基于 outline (先 write_outline 再 write_section)
    # title / section_type / sub_points 全部从 outline_sections[section_idx] 拿
    # 旧的 args.title / args.sub_points 字段废弃, 但允许 LLM 老调用通过 (会忽略)
    outline_sections = state.get("outline_sections", [])
    if not outline_sections:
        yield (
            f"❌ write_section: 大纲未生成, 必须先 write_outline (state['outline_sections'] 空)\n"
            f"👁️ 观察: 跳到下一轮, LLM 应选 write_outline\n"
        )
        state["errors"].append({
            "step": state["step"], "action": "write_section",
            "msg": "大纲未生成, write_section 被拒",
        })
        return

    # section_idx 默认 = 已写章节数 (下一章)
    section_idx = args.get("section_idx")
    if section_idx is None:
        section_idx = len(state["sections"])
    if not isinstance(section_idx, int) or section_idx < 0 or section_idx >= len(outline_sections):
        yield (
            f"❌ write_section: section_idx={section_idx} 越界 "
            f"(outline 共 {len(outline_sections)} 章, 已写 {len(state['sections'])} 章)\n"
        )
        state["errors"].append({
            "step": state["step"], "action": "write_section",
            "msg": f"section_idx={section_idx} 越界",
        })
        return

    outline_sec = outline_sections[section_idx]
    title = outline_sec["title"]
    sec_type = outline_sec["section_type"]
    sub_points = outline_sec["sub_points"]
    outline_section_md = "\n".join(f"- {sp}" for sp in sub_points)

    yield f"✍️ 写章节 [{sec_type}] {title}\n"

    # theme: 第一次创建, 后续复用 (silent, 9-08 17:25 删了 📌 新建 theme yield)
    if "theme_id" not in state:
        state["theme_id"] = create_theme(
            name=state["topic"],
            description=f"P6.5 survey_generate 自动创建 at {datetime.now().isoformat(timespec='seconds')}",
        )
    theme_id = state["theme_id"]

    pool_ids = {p["paper_id"] for p in state["pool"]}

    # 自动重写 loop (P6.3: 最多 2 次, spec §3.3)
    MAX_REWRITE = 2
    sec_md = ""
    bucket = {"in_pool": [], "out_of_pool": [], "total_cites": 0}
    for rewrite_i in range(MAX_REWRITE + 1):
        try:
            section_prompt = build_section_prompt(
                topic=state["topic"],
                section_type=sec_type,
                section_title=title,
                outline_section=outline_section_md,
                papers=state["pool"],
            )
            if rewrite_i > 0:
                # 重写时加约束: 只能用候选池内 paper_id (silent, 9-08 17:25 删了 🔁 重写 yield)
                extra_constraint = (
                    f"\n\n【P6.3 自动重写 {rewrite_i}/{MAX_REWRITE}】"
                    f"\n⚠️ 你上一次的输出引用了候选池外的 paper_id ({bucket['out_of_pool']}),"
                    f"\n这会被自动审计标 ❌ (幻觉风险). 现在重写, **只能用候选池内**的 paper_id:"
                    f"\n候选池 paper_id 列表: {sorted(pool_ids)}"
                    f"\n请删除/替换所有候选池外引用, 然后再 output 完整章节 markdown。"
                )
                section_prompt = section_prompt + extra_constraint

            sec_md = _call_llm(
                section_prompt,
                provider=_PROVIDER,
                model=_MODEL,
                enable_thinking=False,
                temperature=0.4,
                max_tokens=get_max_section_tokens(),  # 9-08 23:50 user 拍板
                timeout=240,
            )
        except Exception as e:
            # 异常不重试 (silent, 9-08 17:25 删了 ❌ LLM 失败 yield)
            sec_md = f"⚠️ 章节生成失败: {_friendly_error(e, 'write_section')}\n\n(占位章节: {title})"
            state["errors"].append({
                "step": state["step"], "action": "write_section",
                "msg": _friendly_error(e, "write_section"),
            })
            break

        # 剥 <think>
        if sec_md and "<think>" in sec_md:
            sec_md = re.sub(r"<think>.*?</think>", "", sec_md, flags=re.DOTALL).strip()

        # 审计引用
        bucket = _audit_section_paper_ids(sec_md, pool_ids)
        if not bucket["out_of_pool"] or rewrite_i >= MAX_REWRITE:
            break
        # out_of_pool > 0 且还能重写

    # ========== 9-08 17:45 user 反馈: 章节内容为空 / 过短检测 ==========
    # user 实测 "第 2 章 progress 实际是空的——LLM 写了大标题但没正文" (#48)
    # 兜底: 如果 LLM 写的章节内容 < 200 字符 (排除纯标题/纯# 后), 标记空章节 + 阻断
    # 9-08 18:15 升级: 不再 silent 标 gaps, 而是完整阻断 — 不 yield 章节内容 + 不落 db +
    #   不 append sections (章节数自然少 1) + 标 section_gaps + 立即 return
    # 这样 action_finish 看到: ① 章节数少 1 (拒) + ② section_gaps 不空 (拒) → 双保险
    # user 也看不到"空章节被 yield 出去" 的污染
    sec_md_clean = (sec_md or "").strip()
    effective_lines = [
        ln for ln in sec_md_clean.split("\n")
        if ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith("---")
    ]
    effective_content = "\n".join(effective_lines).strip()
    if len(effective_content) < 200:
        # 9-08 18:15 升级: 阻断 (return), 防止空章节内容被 yield / 落 db / 算章节数
        state["section_gaps"].append({
            "section_idx": section_idx,
            "title": title,
            "queries": [f"章节 '{title}' 内容过短 ({len(effective_content)} 字符), 需要重写并补充 800-1500 字内容"],
        })
        state["errors"].append({
            "step": state["step"], "action": "write_section",
            "msg": f"章节 '{title}' 内容过短: {len(effective_content)} 字符 (要求 ≥ 200)",
            "effective_content_preview": effective_content[:100],
        })
        yield f"⚠️ 章节 '{title}' 内容过短 ({len(effective_content)} 字符), 触发重写\n"
        return  # 阻断: 不 yield 章节内容 / 不落 db / 不 append sections / 不算章节数

    # ========== 9-08 user 拍板: 解析 missing_queries 标注 ==========
    # LLM 写章节时如果觉得还缺某些方向的文献, 会在 markdown 末尾加:
    #   `<!-- QAgent: missing_queries=["BB84 原始协议", "MDI-QKD"] -->`
    # system 抓出来后存 state["section_gaps"], 并从 sec_md 删掉标注 (避免污染正文)
    # action_finish 看到 section_gaps 不空 → 拒绝 finish → 回到 search_more 循环
    _MISSING_Q_RE = re.compile(
        r'<!--\s*QAgent:\s*missing_queries\s*=\s*\[(.*?)\]\s*-->',
        re.DOTALL,
    )
    m_missing = _MISSING_Q_RE.search(sec_md or "")
    if m_missing:
        # 解析 query 列表 (简单 split, 容忍单引号/双引号)
        raw_queries = m_missing.group(1)
        queries: List[str] = []
        try:
            import ast as _ast
            parsed = _ast.literal_eval(f"[{raw_queries}]")
            if isinstance(parsed, list):
                queries = [str(q).strip() for q in parsed if str(q).strip()]
        except Exception:
            # 退化: 手动 split
            queries = [
                q.strip().strip('"').strip("'").strip()
                for q in raw_queries.split(",")
                if q.strip()
            ]
        # 删掉标注 (避免污染正文)
        sec_md = _MISSING_Q_RE.sub("", sec_md).rstrip() + "\n"
        if queries:
            gap_entry = {
                "section_idx": section_idx,
                "title": title,
                "queries": queries,
            }
            state["section_gaps"].append(gap_entry)
            # 9-08 17:25 删: 不在主循环 yield 缺漏详情 (queries 是具体内容, 污染显示)
            # state['section_gaps'] 仍存, action_finish 会检查

    # 章节内容整体 yield (review 用, 9-08 17:25 保留)
    yield f"\n--- [{sec_type}] {title} ---\n"
    for line in sec_md.split("\n"):
        yield line + "\n"
    yield "--- 章节结束 ---\n"

    # 落 surveys.db (silent, 9-08 17:25 删了 💾 落盘 yield)
    try:
        sec_id = create_section(
            theme_id=theme_id,
            section_type=sec_type,
            title=title,
            content_md=sec_md or "",
            paper_ids=",".join(bucket["in_pool"]),
        )
        state["sections"].append({
            "section_id": sec_id,
            "section_idx": section_idx,
            "title": title,
            "type": sec_type,
            "in_pool": bucket["in_pool"],
            "out_of_pool": bucket["out_of_pool"],
        })
    except Exception as e:
        friendly = _friendly_error(e, "create_section")
        state["errors"].append({
            "step": state["step"], "action": "write_section.create_section",
            "msg": friendly,
        })

    # 写完章节后清掉对应 idx 的 gap (silent, 9-08 17:25 删了 ✅ 清理 gap yield)
    state["section_gaps"] = [
        g for g in state.get("section_gaps", [])
        if g.get("section_idx") != section_idx
    ]

    # 覆盖度 audit → 传递给写作阶段 (silent, 9-08 17:25 删了 📊 覆盖度 audit yield 列 paper_ids)
    # 但 state['unused_pool_papers'] 仍更新, build_react_prompt 给 LLM 看
    all_cited: set = set()
    for r in state.get("sections", []):
        all_cited.update(r.get("in_pool", []))
    all_cited.update(bucket.get("in_pool", []))
    unused = sorted(pool_ids - all_cited)
    state["unused_pool_papers"] = unused
    # 不 yield 详情, 避免 paper_ids 列表污染显示

    # 9-08 17:25 改: 高层 yield 只显示"章节 X/Y 完成"
    yield f"👁️ 章节 {len(state['sections'])}/{len(outline_sections)} 完成\n"
