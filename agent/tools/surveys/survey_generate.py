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
from collections.abc import Iterator
from typing import Any

from agent.infra.survey_workspace import SurveyWorkspace, hash_topic
from agent.tools.registry import register_tool

from ._actions import (
    _append_react_trace,
    _decide_action,
    _dispatch_action,
    _finalize_survey,
)
from ._pool import _build_candidate_pool

logger = logging.getLogger(__name__)


@register_tool(name="survey_generate", is_streaming=True, cost_tier="L2")
def survey_generate(
    topic: str,
    max_steps: int = 20,
    max_pool: int = 50,                # 9-08 17:00 加: 候选池上限, 默认 50 (LLM 可显式传更大值)
    dry_run: bool = False,
    phase: str = "all",
    section_idx: int = -1,
    organize: bool = False,
) -> Iterator[str]:
    """
    生成综述 (P6.5 ReAct 改写) — L2 档, 流式 yield 进度 + 章节内容.

    Args:
        topic: 综述主题, 例 "量子纠缠度量计算"
        max_steps: ReAct 循环上限 (默认 20)
        dry_run: True = 不调 LLM, 只走 Phase A 候选池构建
        phase: "all" (默认, 一次跑完) / "plan" / "write" / "organize" — 见下
        section_idx: >=0 时只写第 section_idx 章 (单章重写), 配合 phase="write"
        organize: True = 走**主题驱动**路线 (Phase B/C), 章节结构由 fact 内容归纳
                  而非预设模板。默认 False = 旧的 5 段模板路线(保留, 可对照回退)。

    phase 拆分 (9-27, 为"大纲可编辑"而加):
        "plan"  — 只跑到大纲: 候选池 → fact → 大纲, 然后**停住**。
                   产物落 workspace/_pool.json, _facts.jsonl, _outline.json。
                   之后可以用 survey_outline(action="view"/"update") 改大纲。
        "write" — 从 workspace 读回 pool/facts/大纲(可以是用户改过的),
                   只写章节正文 + 收尾审计。不重复跑 Phase A / fact 抽取。
        "all"   — plan + write 一次跑完(向后兼容的默认行为)。

    单章重写:
        phase="write", section_idx=k → 只重写第 k 章, 其余章节不动。
        已有正文会被新正文替换(同一 section 记录被覆盖, 不追加)。

    Yields:
        流式进度 + 章节内容, chat stream_chunk signal 消费
        (is_streaming=True 标记, agent.py 派发层识别为 generator)
    """
    if not topic or not topic.strip():
        yield "❌ topic 不能为空\n"
        return

    topic = topic.strip()
    phase = (phase or "all").strip().lower()
    if phase not in ("all", "plan", "write"):
        yield f"❌ phase 只能是 'all' / 'plan' / 'write', 收到 {phase!r}\n"
        return

    # 0. workspace 准备
    topic_hash = hash_topic(topic)
    ws = SurveyWorkspace(topic_hash, topic=topic)

    if phase == "write":
        yield from _run_write_phase(topic, ws, section_idx=section_idx, max_steps=max_steps)
        return

    if phase == "organize" or organize:
        yield from _run_organize_phase(topic, ws, topic_hash, max_pool=max_pool)
        return

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
    # 9-18 加: pool_audit 字段, 把 _build_candidate_pool 的 audit 报告塞进 state,
    #           让 ReAct 主循环 LLM 决策时能看到 (候选池是否够格 / 是否需要 search_more / 是否要换方向)
    state: dict[str, Any] = {
        "topic": topic,
        "topic_hash": topic_hash,
        "pool": pool,
        "pool_audit": build_status.get("audit_state", {}),   # 9-18 加
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

    # 9-18 加: yield audit 报告给用户看 (高层总结, 不污染 UI)
    audit_summary = state.get("pool_audit", {})
    if audit_summary:
        meets = audit_summary.get("pool_meets_target", False)
        rounds = audit_summary.get("rounds", [])
        final = audit_summary.get("final_pool_size", 0)
        target = audit_summary.get("target_min_pool", 0)
        if meets:
            yield (
                f"✅ 候选池审核通过: {final}/{target} 篇, "
                f"经过 {len(rounds)} 轮 audit\n"
            )
        else:
            yield (
                f"⚠️ 候选池审核后仍不足: {final}/{target} 篇 "
                f"(已重试 {len(rounds)-1} 次, 走完所有 audit 轮)\n"
                f"💡 ReAct 后续 search_more 时建议调整方向\n"
            )

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
    if phase == "plan":
        yield from _yield_plan_summary(state, ws, topic)
        return
    final = _finalize_survey(state, ws)
    yield final


# ============================================================
# phase="plan": 跑到大纲就停 (9-27)
# ============================================================

def _yield_plan_summary(state: dict[str, Any], ws, topic: str):
    """plan 阶段收尾: 汇报大纲, 告诉用户下一步怎么走。"""
    from ._outline import render_outline_md

    outline = state.get("outline_sections") or []
    ws.set_status("planned", 0, len(outline), "大纲已生成, 等待用户确认/编辑")

    yield (
        f"\n## ✋ 大纲已生成, 已停在这里等你过目\n\n"
        f"- 候选池: {len(state.get('pool', []))} 篇\n"
        f"- 原子 fact: {len(state.get('facts', []))} 条\n"
        f"- 大纲: {len(outline)} 章\n"
        f"- workspace: `{ws.path}`\n"
    )
    if outline:
        yield render_outline_md(outline) + "\n"
    yield (
        f"---\n"
        f"接下来你可以:\n"
        f"  1. **直接开写**: `survey_generate(topic=\"{topic}\", phase=\"write\")`\n"
        f"  2. **先改大纲**: `survey_outline(topic=\"{topic}\", action=\"view\")` 看详情,\n"
        f"     改完 `survey_outline(..., action=\"update\", sections=...)`,\n"
        f"     再 `survey_generate(..., phase=\"write\")`\n"
        f"  3. **只重写某一章**: `survey_generate(topic=\"{topic}\", phase=\"write\", section_idx=2)`\n"
    )


# ============================================================
# phase="organize": 主题驱动路线 (Phase B + C)
# ============================================================

def _run_organize_phase(topic: str, ws, topic_hash: str, max_pool: int = 50):
    """
    主题驱动综述 (9-29 接入, 用户洞察):
        综述写不长、不像综述, 根因是以**文章**为锚点。真实的人写综述是以**信息**
        为锚点: 收集 fact → 把 fact 归入主题、看它们之间什么关系 → 围绕主题写叙述。
        "把 fact 归类"才是智识劳动本身, 所以它必须是一个独立且**有产物**的阶段。

    流程:
        1. Phase A  候选池 (0 LLM, 复用 _build_candidate_pool)
        2. fact 抽取 (每篇 1 次 LLM) + 落地审查(每 8 条 1 次, 见 _fact_audit)
        3. Phase B  主题地图 → workspace/_thematic_map.json (可编辑复用)
        4. Phase C  逐主题写叙述 (引文由 fact_id 机械映射, 模型碰不到 paper_id)
        5. 章节落 surveys.db + 覆盖度审计

    与旧路线的区别只在 3/4 —— 1/2/5 完全共用, 所以两条路线可以互相对照。
    """
    from ._organize import (
        organize_facts_into_themes,
        render_survey,
        save_thematic_map,
        write_theme_narrative,
    )

    # ---- Phase A: 候选池 ----
    yield f"🔍 主题驱动模式: {topic}\n"
    pool, build_status = _build_candidate_pool(topic, max_pool=max_pool)
    if not pool:
        ws.set_status("failed", 0, 0, "pool empty")
        yield "❌ 候选池为空, 无法继续。\n- 建议: 换个主题词(优先英文)或先抓文\n"
        return
    ws.save_pool(pool)
    ws.set_status("pool", len(pool), 0, "候选池已建")
    yield f"📚 候选池: {len(pool)} 篇\n"

    # ---- Phase 2: fact 抽取 ----
    yield "\n📝 抽取原子 fact...\n"
    from ._actions import action_extract_facts
    # 9-27 修: state 必须带 topic —— action_extract_facts 末尾的**主题对齐审查**
    # 要用它。漏了会 KeyError('topic'), 那套审查静默降级(实测踩到, 日志里
    # 表现为 "fact 审查失败 (KeyError: 'topic'), 跳过")。
    extract_state = {
        "topic": topic,          # 主题对齐审查要用; 缺了就静默失败
        "pool": pool,
        "facts": [],
        "errors": [],
        "step": 0,
        "finish_requested": False,
        "max_steps": 1,
    }
    for chunk in action_extract_facts({"paper_ids": [p["paper_id"] for p in pool]},
                                      extract_state, ws):
        if chunk:
            yield chunk
    facts = extract_state["facts"] or ws.load_facts()
    usable = [f for f in facts if _fact_is_usable(f)]
    ws.set_status("facts", len(usable), len(facts), "fact 抽取完成")
    if not usable:
        yield "❌ 没有可用 fact (全部未通过落地审查或抽取失败), 停止。\n"
        return
    bad = [f for f in facts if not _fact_is_usable(f)]
    if bad:
        yield f"   (另有 {len(bad)} 条 fact 未通过落地审查, 已剔除)\n"
    yield f"✅ 可用 fact: {len(usable)} 条 / 共 {len(facts)} 条\n"

    # ---- Phase B: 主题地图 ----
    yield "\n🧭 归纳主题结构 (从 fact 内容, 而非预设模板)...\n"
    titles = {p["paper_id"]: (p.get("title") or "") for p in pool}
    theme_map = organize_facts_into_themes(usable, paper_titles=titles)
    if not theme_map["themes"]:
        for w in theme_map.get("warnings", []):
            yield f"   ⚠ {w}\n"
        yield "❌ 主题地图生成失败。\n"
        return
    save_thematic_map(ws, theme_map)
    yield f"✅ 归纳出 {len(theme_map['themes'])} 个主题:\n"
    for i, t in enumerate(theme_map["themes"], 1):
        yield f"   [{i}] {t['name']} ({len(t['fact_ids'])} 条 fact)\n"
        if t.get("statement"):
            yield f"       {t['statement']}\n"
    for w in theme_map.get("warnings", []):
        yield f"   ⚠ {w}\n"

    # ---- Phase C: 逐主题写叙述 ----
    yield "\n✍️ 按主题写叙述...\n"
    arc = theme_map.get("narrative_arc") or []
    order = [t["name"] for t in theme_map["themes"]]
    if arc:
        rank = {n: i for i, n in enumerate(arc)}
        order = sorted(order, key=lambda n: (rank.get(n, len(rank)), order.index(n)))

    narratives: dict[str, dict[str, Any]] = {}
    for name in order:
        theme = next((t for t in theme_map["themes"] if t["name"] == name), None)
        if not theme:
            continue
        r = write_theme_narrative(theme, facts)
        narratives[name] = r
        if r["text"]:
            yield f"   ✅ {name}: {len(r['text'])} 字符 "
            yield f"(用 {len(r['used_fact_ids'])} 条 fact)\n"
        for w in r.get("warnings", []):
            yield f"      ⚠ {w}\n"

    # ---- 组装 + 落库 ----
    doc = render_survey(theme_map, narratives, title=topic,
                        subtitle="本综述章节结构由文献内容归纳得出(Phase B 主题地图), 非预设模板。")
    from ._crud import create_section
    theme_id = _find_or_create_theme(topic, edited=True)
    created = 0
    for t in theme_map["themes"]:
        text = (narratives.get(t["name"], {}) or {}).get("text", "").strip()
        if not text:
            continue
        used = (narratives.get(t["name"], {}) or {}).get("used_fact_ids", [])
        pids = sorted({f["paper_id"] for f in usable if f.get("fact_id") in set(used)})
        create_section(theme_id=theme_id, title=t["name"],
                       section_type=t.get("section_type", "custom"),
                       content_md=text, paper_ids=",".join(pids))
        created += 1
    ws.save_audit(_organize_audit(theme_map, narratives, usable))
    ws.set_status("done", created, len(theme_map["themes"]),
                  f"主题驱动完成, theme_id={theme_id}")
    yield (f"\n## 主题驱动综述完成 — {topic}\n"
           f"- theme_id: `{theme_id}`\n"
           f"- 主题: {len(theme_map['themes'])} 个, 写入 {created} 章\n"
           f"- fact: {len(usable)} 条可用 / {len(facts)} 条共抽\n"
           f"- 篇幅: {len(doc)} 字符\n"
           f"- 审计: `{ws.path}/_audit.md`\n\n")
    yield doc


def _fact_is_usable(f: dict[str, Any]) -> bool:
    from agent.prompts_survey import _is_usable_fact
    return _is_usable_fact(f)


def _organize_audit(theme_map: dict, narratives: dict, facts: list) -> str:
    """主题驱动路线的覆盖度审计(按主题而非按章节)。"""
    lines = ["# 覆盖度审计 (主题驱动路线)", ""]
    total_used = set()
    for t in theme_map.get("themes", []):
        nar = narratives.get(t["name"], {}) or {}
        used = nar.get("used_fact_ids", [])
        total_used.update(used)
        lines.append(
            f"## {t['name']}  ({len(t.get('fact_ids', []))} 条 fact 可用, "
            f"{len(used)} 条被引用, {nar.get('dropped_sentences', 0)} 句因无出处剔除)"
        )
        if t.get("statement"):
            lines.append(f"- 领域判断: {t['statement']}")
        for rel in t.get("relations", []):
            lines.append(f"- 关系 {rel.get('type')} → {rel.get('to')}: {rel.get('note')}")
        for w in nar.get("warnings", []):
            lines.append(f"- ⚠ {w}")
        lines.append("")
    all_ids = {f.get("fact_id") for f in facts}
    unused = sorted(all_ids - total_used)
    lines.append("## fact 使用率")
    lines.append(f"- 总计: {len(all_ids)} 条可用 fact")
    lines.append(f"- 被引用: {len(total_used)} 条 ({len(total_used) * 100 // max(1, len(all_ids))}%)")
    lines.append(f"- 未被引用: {len(unused)} 条")
    if theme_map.get("gaps"):
        lines.append("")
        lines.append("## 文献覆盖空白")
        lines += [f"- {g}" for g in theme_map["gaps"]]
    return "\n".join(lines)


# ============================================================
# phase="write": 从 workspace 读回产物, 只写章节 (9-27)
# ============================================================

def _run_write_phase(topic: str, ws, section_idx: int = -1, max_steps: int = 20):
    """
    写作阶段: pool / facts / 大纲 全部从 workspace 读回(可能是用户改过的)。

    好处:
      - 不重复跑 Phase A / fact 抽取(省掉几十次 LLM 调用)
      - 用户改完大纲后, 直接重写正文, 不用把整个 pipeline 重来一遍
      - section_idx >= 0 时只写那一章(单章重写), 其余章节不动
    """
    pool = ws.load_pool()
    outline_raw = ws.load_outline() or {}
    outline_sections = outline_raw.get("sections") or []
    facts = ws.load_facts()

    if not pool:
        yield (
            f"❌ 主题「{topic}」的 workspace 里没有候选池。\n"
            f"先跑 `survey_generate(topic=\"{topic}\", phase=\"plan\")` 生成大纲。\n"
            f"(workspace: `{ws.path}`)\n"
        )
        return
    if not outline_sections:
        yield (
            f"❌ 主题「{topic}」还没有大纲。\n"
            f"先跑 `survey_generate(topic=\"{topic}\", phase=\"plan\")`。\n"
            f"(workspace: `{ws.path}`)\n"
        )
        return

    # fact 里的对齐标记在 _facts.jsonl 里(旧 run 可能没有), 过滤交给渲染层兜底
    if isinstance(section_idx, int) and 0 <= section_idx < len(outline_sections):
        target_desc = f"第 {section_idx} 章 [{outline_sections[section_idx].get('section_type')}]"
        outline_sections = [outline_sections[section_idx]]
        yield (
            f"✍️ 单章模式: 只写 {target_desc} "
            f"{outline_sections[0].get('title', '')}\n"
            f"📚 从 workspace 读回: pool {len(pool)} 篇 / fact {len(facts)} 条 / 大纲原 {len(outline_raw.get('sections') or [])} 章\n\n"
        )
    else:
        yield (
            f"✍️ 写作阶段\n"
            f"📚 从 workspace 读回: pool {len(pool)} 篇 / fact {len(facts)} 条 / 大纲 {len(outline_sections)} 章"
            f"{'(用户已编辑)' if outline_raw.get('edited') else ''}\n\n"
        )

    edited = outline_raw.get("edited", False)
    state: dict[str, Any] = {
        "topic": topic,
        "topic_hash": ws.topic_hash,
        "pool": pool,
        "pool_audit": {},          # plan 阶段的 audit 不再重跑
        "facts": facts,
        "sections": [],
        "outline": outline_raw.get("raw", ""),
        "outline_sections": outline_sections,
        "outline_edited": edited,
        "section_gaps": [],
        "unused_pool_papers": [],
        "errors": [],
        "step": 0,
        "finish_requested": False,
        "max_steps": max_steps if section_idx < 0 else 1,
    }

    # 找到该主题已有的 theme_id, 避免每章都新建一个 theme
    state["theme_id"] = _find_or_create_theme(topic, edited)

    if section_idx >= 0:
        # 单章重写: 先删掉旧章节, 让 write_section 重新落一条(否则会重复追加)
        _drop_existing_section(state["theme_id"], outline_sections[0])

    while state["step"] < state["max_steps"] and not state["finish_requested"]:
        state["step"] += 1
        decision = _decide_action(state)
        yield f"➡️ 调用工具: {decision['action']}\n"
        try:
            for chunk in _dispatch_action(decision, state, ws):
                if chunk:
                    yield chunk
        except Exception as e:
            from ._actions import _friendly_error
            friendly = _friendly_error(e, decision["action"])
            state["errors"].append({
                "step": state["step"], "action": decision["action"], "msg": friendly,
            })
            yield f"👁️ 观察: ❌ {friendly}\n"
        _append_react_trace(ws, {
            "step": state["step"],
            "thought": decision["thought"],
            "action": decision["action"],
            "args": decision.get("args", {}),
            "n_errors": len(state["errors"]),
        })

    # 单章重写不做覆盖度审计(只写了一章, 审计统计会误导)
    if section_idx >= 0:
        for r in state.get("sections", []):
            yield (
                f"\n✅ 第 {section_idx} 章重写完成 "
                f"(section_id={r.get('section_id')}, 引用 {len(r.get('in_pool', []))} 篇)\n"
            )
        ws.set_status("done", 1, 1, f"单章重写 section_idx={section_idx}")
        return

    yield _finalize_survey(state, ws)


def _find_or_create_theme(topic: str, edited: bool) -> int:
    """找该 topic 已有的 theme, 找不到才新建 (phase 续跑时避免重复 theme)。"""
    from ._crud import create_theme, list_themes
    for t in list_themes():
        if (t.get("name") or "").strip() == topic:
            return t["id"]
    return create_theme(
        name=topic,
        description=f"survey_generate 自动创建 at {__import__('datetime').datetime.now().isoformat(timespec='seconds')}"
        + (" (用户编辑过大纲)" if edited else ""),
    )


def _drop_existing_section(theme_id: int, outline_sec: dict[str, Any]) -> int:
    """单章重写前删掉同标题的旧章节, 避免重写后出现两份。"""
    from ._crud import delete_section, list_sections
    n = 0
    for sec in list_sections(theme_id):
        if (sec.get("title") or "").strip() == (outline_sec.get("title") or "").strip():
            delete_section(sec["id"])
            n += 1
    return n
