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
import hashlib
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
    target_chars: int = 0,             # 9-30 加: 综述目标篇幅, 0=按 fact 数自动反推
    dry_run: bool = False,
    phase: str = "all",
    section_idx: int = -1,
    organize: bool = False,
    resume: bool = True,             # 10-02 加: 崩溃续跑, 见下方说明
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
        resume: True(默认) = 崩溃续跑。上一轮如果死在流水线中途(实测卡死在主题归纳
                  那一步, 已经跑掉 76 分钟 / 151 次调用), 且参数没变, 就从最后一个
                  完成的阶段接着走, 不重跑前面的阶段。resume=False = 强制从头跑
                  (改了 prompt 想重测时用)。续跑时一定会 yield 一行「♻️ 续跑」,
                  绝不静默接上旧数据。

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
        yield from _run_organize_phase(topic, ws, topic_hash, max_pool=max_pool,
                                  target_chars=target_chars, resume=resume)
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

def _phase_write(ws, theme_map, order, by_theme, facts, usable,
                 target_chars: int, from_stage: str, sig: str):
    """Phase C: 逐主题写叙述。产出 narratives dict。

    成文是最慢的一段(v60d 实测 26 次调用 / 1720 秒), 所以按**主题**逐个存 ——
    挂在第 5 个主题时崩了, 前 4 个主题的正文不用重写。
    """
    from ._organize import write_theme_narrative, write_theme_narrative_from_claims

    n_themes = max(1, len(theme_map["themes"]))
    if target_chars <= 0:
        target_chars = max(1200, int(len(usable) * 0.32))
    per_theme_chars = max(300, target_chars // n_themes)
    yield f"🎯 目标篇幅 {target_chars} 字符, 分 {n_themes} 个主题, 每主题约 {per_theme_chars} 字符\n"

    narratives: dict[str, dict[str, Any]] = {}
    if from_stage == "done":
        narratives = ws.load_narratives() or {}
        if narratives:
            yield f"\n♻️ 成文: 沿用 checkpoint 的 {len(narratives)} 个主题正文\n"
    for name in order:
        theme = next((t for t in theme_map["themes"] if t["name"] == name), None)
        if not theme:
            continue
        if name in narratives:
            r = narratives[name]
            yield f"   ♻️ {name}: {len(r.get('text') or '')} 字符 (沿用)\n"
            for w in r.get("warnings", []):
                yield f"      ⚠ {w}\n"
            continue
        tclaims = by_theme.get(name) or []
        r = (write_theme_narrative_from_claims(theme, tclaims, target_chars=per_theme_chars)
             if tclaims else write_theme_narrative(theme, facts))
        narratives[name] = r
        ws.save_narratives(narratives)      # 每写完一个主题就落盘
        ws.save_checkpoint("narratives", signature=sig,
                           done_themes=list(narratives.keys()))
        if r["text"]:
            yield f"   ✅ {name}: {len(r['text'])} 字符 "
            yield f"(用 {len(r.get('used_claim_ids') or r.get('used_fact_ids') or [])} 条)\n"
        for w in r.get("warnings", []):
            yield f"      ⚠ {w}\n"
    return narratives


def _resolve_resume(ws, topic_hash: str, max_pool: int, target_chars: int,
                    resume: bool, stages: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
    """决定从哪个阶段接着跑, 并返回要打印的提示行。

    抽出来是因为续跑判定有 4 个分支, 塞在 _run_organize_phase 里会把那个函数
    的复杂度顶爆(C901)。逻辑本身很直白, 见下面三条规则。
    """
    sig = hashlib.sha1(f"{topic_hash}|{max_pool}|{target_chars}".encode()).hexdigest()[:12]
    lines: list[str] = []
    ck = ws.load_checkpoint() if resume else None

    if ck and ck.get("signature") != sig:
        lines.append("♻️ 上轮 checkpoint 参数对不上(改了 max_pool/target_chars), 从头重跑\n")
        ws.clear_checkpoint()
        return "pool", {"signature": sig, "lines": lines}

    if ck:
        st = ck.get("stage")
        if st in stages:
            nxt = stages[min(stages.index(st) + 1, len(stages) - 1)]
            lines.append(
                f"♻️ 续跑: 上轮停在「{st}」阶段之后 ({ck.get('updated_at', '?')}), "
                f"从「{nxt}」接着走\n")
            return st, {"signature": sig, "lines": lines}
        ws.clear_checkpoint()

    return "pool", {"signature": sig, "lines": lines}


def _phase_facts(topic: str, ws, pool: list[dict[str, Any]], from_stage: str,
                 sig: str):
    """Phase 2: fact 抽取 (含续跑)。产出 (facts, from_stage)。

    续跑回来的 from_stage 可能被**回退** —— checkpoint 说 fact 已经抽好了,
    但 _facts.jsonl 是空的(被手动删了/上轮抽了 0 条), 这时必须老实重跑,
    不能拿着空 fact 往下走。
    """
    facts: list[dict[str, Any]] = []
    if from_stage in ("facts", "themes", "claims", "narratives", "done"):
        facts = ws.load_facts()
        yield f"\n♻️ fact 抽取: 沿用 checkpoint 的 {len(facts)} 条\n"
    if facts:
        yield from _yield_usable_summary(facts)
        return facts, from_stage

    from_stage = "pool"              # checkpoint 说有 fact 但文件空了 -> 老实重跑
    yield "\n📝 抽取原子 fact...\n"
    # 9-30: 抽取前先清掉上一轮遗留的 fact。workspace 按主题 hash 复用, 之前
    # 只 append 不清, 同一主题重跑会一路累积(实测堆到 1089 行, 同一篇论文
    # 攒了 42 条, 而配额本来是 4-12)。更糟的是抽取一旦失败,
    # `or ws.load_facts()` 会静默端上这些陈旧数据 —— 产出看着正常, 实际是旧数据。
    _stale = ws.clear_facts()
    if _stale:
        yield f"   (清掉上一轮遗留 fact {_stale} 条)\n"
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
    # 9-30 修: 原来写的是 `extract_state["facts"] or ws.load_facts()` ——
    # 抽取**全军覆没**时会静默退回 workspace 里的旧 fact。而 workspace 按主题
    # hash 复用, 同主题重跑就是上一轮的数据。实测: 51 篇抽了 0 条, 端上来 685 条
    # 陈旧的, 综述照常产出, 看上去完全正常 —— 实际建在旧数据上。
    facts = extract_state["facts"]
    stale = ws.load_facts()
    if not facts and stale:
        # 有意保留兜底(比如续跑 phase=write), 但**必须喊出来**, 不能静默
        print(f"[organize] ⚠️ 本轮抽取 0 条 fact, 沿用 workspace 里的 {len(stale)} 条旧数据", flush=True)
        facts = stale
    elif not facts:
        print("[organize] ❌ 本轮抽取 0 条 fact 且 workspace 无历史数据", flush=True)
    ws.save_checkpoint("facts", signature=sig, n_facts=len(facts))
    yield from _yield_usable_summary(facts)
    return facts, from_stage


def _yield_usable_summary(facts: list[dict[str, Any]]):
    """打印 fact 可用情况。抽出来是因为"续跑/新抽"两条路都要说同一段话。"""
    usable = [f for f in facts if _fact_is_usable(f)]
    ws_bad = [f for f in facts if not _fact_is_usable(f)]
    if ws_bad:
        yield f"   (另有 {len(ws_bad)} 条 fact 未通过落地审查, 已剔除)\n"
    yield f"✅ 可用 fact: {len(usable)} 条 / 共 {len(facts)} 条\n"


def _run_organize_phase(topic: str, ws, topic_hash: str, max_pool: int = 50,
                        target_chars: int = 0, resume: bool = True):
    """
    主题驱动综述 (9-29 接入, 用户洞察):
        综述写不长、不像综述, 根因是以**文章**为锚点。真实的人写综述是以**信息**
        为锚点: 收集 fact → 把 fact 归入主题、看它们之间什么关系 → 围绕主题写叙述。
        "把 fact 归类"才是智识劳动本身, 所以它必须是一个独立且**有产物**的阶段。

    流程:
        1. Phase A  候选池 (0 LLM, 复用 _build_candidate_pool)
        2. fact 抽取 (每篇 1 次 LLM) + 落地审查(见 _fact_audit)
        3. Phase B  主题地图 → workspace/_thematic_map.json (可编辑复用)
        4. Phase C  逐主题写叙述 (引文由 fact_id 机械映射, 模型碰不到 paper_id)
        5. 章节落 surveys.db + 覆盖度审计

    与旧路线的区别只在 3/4 —— 1/2/5 完全共用, 所以两条路线可以互相对照。

    崩溃续跑 (10-02):
        这条流水线是严格串行的, 中途挂掉就整轮作废 —— v60e 实测跑了 76 分钟 /
        151 次调用死在主题归纳那一步, 而 fact 早就落盘在 _facts.jsonl 里了。
        每个阶段完成时打一个 checkpoint, 下次参数没变就从最后一个完成的阶段接着走。

        两个防错设计 (都是"静默接上旧数据"这类最难查的 bug):
        1. **signature 必须逐位相同**: max_pool / target_chars 改了就不续 ——
           否则会把旧参数下的中间产物接到新参数上, 输出看着完全正常但结论是错的。
        2. **续跑时一定 yield 一行「♻️ 续跑」**。宁可吵, 不可静默。
        候选池也直接从 checkpoint 读, 不重跑召回+审查(顺带省 18 次调用)。
    """
    from ._organize import (
        load_thematic_map,
        organize_facts_into_themes,
        render_survey,
        save_thematic_map,
    )

    # ---------- 崩溃续跑: 读 checkpoint ----------
    # 阶段顺序 = 流水线顺序, "已完成的最后一个阶段"决定从哪接着跑
    _STAGES = ("pool", "facts", "themes", "claims", "narratives", "done")
    from_stage, ck_msgs = _resolve_resume(
        ws, topic_hash, max_pool, target_chars, resume, _STAGES)
    _sig = ck_msgs["signature"]
    yield from ck_msgs["lines"]

    # ---- Phase A: 候选池 ----
    yield f"🔍 主题驱动模式: {topic}\n"
    pool: list[dict[str, Any]] = []
    if from_stage != "pool":
        pool = ws.load_pool()
    if not pool:
        from_stage = "pool"          # checkpoint 说有池子但文件没了 -> 老实重跑
        pool, build_status = _build_candidate_pool(topic, max_pool=max_pool)
        if not pool:
            ws.set_status("failed", 0, 0, "pool empty")
            yield "❌ 候选池为空, 无法继续。\n- 建议: 换个主题词(优先英文)或先抓文\n"
            return
        ws.save_pool(pool)
        ws.save_checkpoint("pool", signature=_sig,
                           pool_ids=[p["paper_id"] for p in pool])
        yield f"📚 候选池: {len(pool)} 篇\n"
    else:
        yield f"📚 候选池(沿用 checkpoint): {len(pool)} 篇\n"
    ws.set_status("pool", len(pool), 0, "候选池已建")

    # ---- Phase 2: fact 抽取 ----
    # 抽成独立生成器: 这段本身分支就多(续跑判定 + 遗留清理 + 兜底告警),
    # 留在主函数里会把 C901 顶爆。
    facts, from_stage = yield from _phase_facts(topic, ws, pool, from_stage, _sig)
    usable = [f for f in facts if _fact_is_usable(f)]
    ws.set_status("facts", len(usable), len(facts), "fact 抽取完成")
    if not usable:
        yield "❌ 没有可用 fact (全部未通过落地审查或抽取失败), 停止。\n"
        return

    # ---- Phase B: 主题地图 ----
    theme_map: dict[str, Any] | None = None
    if from_stage in ("themes", "claims", "done"):
        theme_map = load_thematic_map(ws)
    if theme_map and theme_map.get("themes"):
        yield f"\n♻️ 主题地图: 沿用 checkpoint 的 {len(theme_map['themes'])} 个主题\n"
    else:
        from_stage = "facts"        # checkpoint 说有主题地图但文件没了/空了 -> 重跑
        theme_map = None
    if theme_map is None:
        yield "\n🧭 归纳主题结构 (从 fact 内容, 而非预设模板)...\n"
        titles = {p["paper_id"]: (p.get("title") or "") for p in pool}
        theme_map = organize_facts_into_themes(usable, paper_titles=titles)
        if not theme_map["themes"]:
            for w in theme_map.get("warnings", []):
                yield f"   ⚠ {w}\n"
            yield "❌ 主题地图生成失败。\n"
            return
        save_thematic_map(ws, theme_map)
        ws.save_checkpoint("themes", signature=_sig,
                           n_themes=len(theme_map["themes"]))
    yield f"✅ 归纳出 {len(theme_map['themes'])} 个主题:\n"
    for i, t in enumerate(theme_map["themes"], 1):
        yield f"   [{i}] {t['name']} ({len(t['fact_ids'])} 条 fact)\n"
        if t.get("statement"):
            yield f"       {t['statement']}\n"
    for w in theme_map.get("warnings", []):
        yield f"   ⚠ {w}\n"

    # ---- Phase B2: 命题合并 (9-30 加, 10-03 改成范式层) ----
    # 缺这一层, 综述就没有"综": fact 逐条成句、逐条挂引用, 同一篇论文在一段话里
    # 被重复引 5 次(40 篇那次实测)。合并成命题后, 一句话可以挂多篇。
    #
    # 10-03: 改走**范式层**(主题 -> 技术路线 -> 命题)。原路线按 subject 分簇
    # 再让 LLM 合并, 但 subject 是"按论文取的标签"(实测 93% 的 fact 其 subject
    # 只对应一篇论文), LLM 正确地不合并, 命题 ≈1:1 对应论文 -> 均引只有 0.94/句。
    # 范式层实测均引 12.0/句。任何一步失败都退回原路线。
    claims: list[dict[str, Any]] = []
    if from_stage in ("claims", "done"):
        claims = ws.load_claims() or []
    if claims:
        yield f"\n♻️ 命题合并: 沿用 checkpoint 的 {len(claims)} 条命题\n"
    else:
        from_stage = "themes"
        yield "\n🔗 归纳技术路线并合并成命题...\n"
        from ._organize import condense_facts_by_paradigm, condense_facts_by_theme
    try:
        claims = condense_facts_by_paradigm(usable, theme_map)
    except Exception as e:
        logger.warning("[organize] 范式层异常, 退回主题合并: %s", e)
        claims = []
    if not claims:
        try:
            claims = condense_facts_by_theme(usable, theme_map)
        except Exception as e:
            logger.warning("[organize] 命题合并失败, 退回 fact 直写: %s", e)
            claims = []
        if claims:
            ws.save_claims(claims)
            ws.save_checkpoint("claims", signature=_sig, n_claims=len(claims))
    if claims:
        n_cited = len({p for c in claims for p in c.get("paper_ids", []) if p})
        # 10-02: 原来打的是 `len(usable) -> len(claims)`, 但 usable 里有一大截
        # **主题地图根本没覆盖**的 fact(实测 v60g: 241 条里 49 条)。把它们算进
        # 分子, 于是"丢得越多, 压缩比越好看" —— 方向完全反了。
        # 真正该看的是"进了合并器的 fact 被压成了几条命题"。
        n_covered = len({fid for c in claims for fid in c.get("fact_ids", [])})
        yield (f"✅ {len(usable)} 条可用 fact -> {len(claims)} 条命题 "
               f"(覆盖 {n_covered} 条, 未覆盖 {len(usable) - n_covered} 条; "
               f"压缩比 {n_covered / max(len(claims), 1):.2f}:1; "
               f"涉及 {n_cited} 篇论文)\n")
    else:
        yield "⚠ 命题合并不可用, 退回逐 fact 直写(引文会偏稀疏)\n"

    # ---- Phase C: 逐主题写叙述 ----
    yield "\n✍️ 按主题写叙述...\n"
    arc = theme_map.get("narrative_arc") or []
    order = [t["name"] for t in theme_map["themes"]]
    if arc:
        rank = {n: i for i, n in enumerate(arc)}
        order = sorted(order, key=lambda n: (rank.get(n, len(rank)), order.index(n)))

    by_theme: dict[str, list[dict[str, Any]]] = {}
    for c in claims:
        by_theme.setdefault(c.get("theme_name", ""), []).append(c)

    # 9-30: 篇幅目标。综述该"用较少的语言描述宏观情况", 不是越长越好。
    # 40 篇那次写出了 27000 字(675 字/篇), 用户要的是 60 篇配 6000 字(100 字/篇)。
    # 按主题数摊开, 没给就按"每篇约 110 字符"的密度反推一个合理值。
    # (具体计算搬到 _phase_write 里 —— 它要用 target_chars 算每主题配额。)
    narratives = yield from _phase_write(
        ws, theme_map, order, by_theme, facts, usable, target_chars, from_stage, _sig)

    # ---- 组装 + 落库 ----
    doc = render_survey(theme_map, narratives, title=topic, pool=pool,
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
    # 跑完了才清 checkpoint —— 清掉之后下次同参数重跑就是干净的一轮,
    # 而不是接着上一轮的 narratives 走(resume=False 同样能强制重来)。
    ws.clear_checkpoint()
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
