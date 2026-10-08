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
from agent.tools.config import (
    get_gap_fill_max_papers,
    get_gap_fill_max_queries,
    get_gap_fill_rounds,
)
from agent.tools.registry import register_tool

from ._actions import (
    _append_react_trace,
    _decide_action,
    _dispatch_action,
    _finalize_survey,
)
from ._pool import _build_candidate_pool

logger = logging.getLogger(__name__)


# ============================================================
# 续跑签名用的 prompt/逻辑版本号
# ============================================================
# 为什么必须有: 续跑签名原本只覆盖 (topic_hash | max_pool | target_chars)。
# 10-03 把命题层从 subject 合并换成范式层, 签名一模一样 -> 上一轮按旧逻辑
# 产出的 themes/claims 会被当成"同参数的合法产物"接上, 输出看着完全正常,
# 结论却是旧逻辑的。这就是"静默接上旧数据"最难查的那一类。
#
# **改了任何 prompt 常量 / 阶段顺序 / 合并逻辑, 就把这个日期往后挪一天。**
# 代价是所有现存 checkpoint 作废(从头重跑一次); 不挪的代价是静默产出
# 上一个版本的结论。后者贵得多。
# 10-08: fact 抽取 prompt 加了"本篇独有性"铁律(见 prompts_survey
# SURVEY_FACT_PROMPT 第 6 条)—— 抽 fact 时避开转述他人和纯定义型内容。
# 不动 JSON schema, 所以老 fact 仍可读; 但**抽 fact 这步**的产物语义变了,
# 版本号必须跟着走, 否则复跑/续跑会拿同一个版本号跑出不同的 fact。
PROMPT_VERSION = "2026-10-08"


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
    from ._organize import (derive_target_chars, per_theme_target_chars,
                            write_theme_narrative,
                            write_theme_narrative_from_claims)

    themes = theme_map["themes"]
    n_themes = max(1, len(themes))
    if target_chars <= 0:
        # 10-08: 原来写死 `max(1200, len(usable) * 0.32)`, 实测**永远撞下限**
        # (263 条 fact * 0.32 = 84 -> target 恒等于 1200), 于是 fact 再多配额
        # 也不涨, 写作层在配额上写不下 -> 模型自造「(后续略)」占位符糊过去。
        target_chars = derive_target_chars(len(usable))
    # 配额**逐主题**算: 各主题命题数不同, 用同一个值会让命题多的章再次写不下
    # (实测「解码算法设计」7 条命题被按 1 条的配额写)。
    quota = {
        (t.get("name") or "").strip(): per_theme_target_chars(
            target_chars, n_themes, len(by_theme.get(t.get("name") or "") or []))
        for t in themes
    }
    yield (f"🎯 目标篇幅 {target_chars} 字符, 分 {n_themes} 个主题 "
           f"(配额按各主题命题数在 {min(quota.values()) if quota else 300}"
           f"~{max(quota.values()) if quota else 300} 字符之间)\n")

    narratives: dict[str, dict[str, Any]] = {}
    # 10-04 修: 原来只判 `== "done"`, 而 save_checkpoint("done") 全仓库从来没
    # 调过, 所以这行是死代码, _narratives.json 逐主题落盘的收益完全没兑现。
    # 真正会出现的续跑点是 "narratives" —— 写到第 5 个主题崩了, 前 4 个得留下。
    if from_stage in ("narratives", "done"):
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
        # 配额逐主题取, 别用全局单值
        r = (write_theme_narrative_from_claims(
                theme, tclaims, target_chars=quota.get(name, 300))
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
    # signature 必须逐位相同才续跑 —— PROMPT_VERSION 在最前面, 所以改 prompt /
    # 改阶段逻辑会自动作废旧 checkpoint(见 PROMPT_VERSION 处的说明)。
    sig = hashlib.sha1(
        f"{PROMPT_VERSION}|{topic_hash}|{max_pool}|{target_chars}".encode()
    ).hexdigest()[:12]
    lines: list[str] = []
    ck = ws.load_checkpoint() if resume else None

    if ck and ck.get("signature") != sig:
        lines.append(
            "♻️ 上轮 checkpoint 参数对不上(改了 max_pool/target_chars 或 prompt 版本), "
            "从头重跑\n"
        )
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


# ============================================================
# Phase B1.5: 宏观主题
# ============================================================

def _phase_macros(ws, theme_map: dict[str, Any]):
    """在主题之上再聚一层宏观板块, 原地写进 theme_map 并补写盘。

    10-05 加。抽成独立生成器(而不是内联在 _run_organize_phase 里)的理由跟
    _phase_claims 一样: 这块有"复用/不用"的分支 + 失败降级, 内联就没法测。

    **为什么这一层要放在 themes 判定之后、而不是塞进 organize_facts_into_themes**:
    磁盘上已有的 _thematic_map.json 全部没有 macros 字段(实测 56 个 workspace
    里只有 3 个有主题地图, 都是这次改动之前写的)。只改产出侧, 那 3 个永远
    补不上这一层; 而它们恰恰是最该补的(有真实 fact 覆盖)。所以这里对
    "复用了 checkpoint 的主题地图" 也照样跑一次。

    代价要说清楚: 复用路径会多出 1 次 LLM 调用。这是**有意的**取舍 ——
    补齐缺失的层级比"续跑绝不多调一次"更要紧, 而且它只发生一次(写进文件
    之后 macros 就非空了, 之后每次续跑都直接跳过)。这一行提示是刻意加的,
    免得用户以为续跑没省下调用。
    """
    from ._organize import cluster_themes_into_macros, save_thematic_map

    if theme_map.get("macros"):
        return                      # 已有(新跑的或上次补过的) -> 一次调用都不花
    n_themes = len(theme_map.get("themes") or [])
    if n_themes < 2:
        theme_map["macros"] = []
        theme_map["macro_warnings"] = [f"只有 {n_themes} 个主题, 不做宏观聚类"]
        return

    yield f"\n🏛️ 归纳宏观板块(把 {n_themes} 个主题再聚一层, 1 次调用)...\n"
    try:
        macros, warns = cluster_themes_into_macros(theme_map["themes"])
    except Exception as e:
        # 宏观层是**加法** —— 缺了它单层主题结构照样能用。所以这里降级但必须
        # 吵出来(这个模块开头就写着"宁可吵, 不可静默")。
        logger.warning("[organize] 宏观主题层异常: %s", e)
        theme_map["macros"] = []
        theme_map["macro_warnings"] = [f"宏观主题层异常: {e}"]
        yield "   ⚠ 宏观主题层异常, 保留单层主题结构(不影响综述正文)\n"
        return

    theme_map["macros"] = macros
    theme_map["macro_warnings"] = warns
    if not macros:
        yield "   ⚠ 宏观主题层没产出, 保留单层主题结构(不影响综述正文)\n"
        return
    # 补写盘: 新跑的 workspace 上面刚 save 过一次, 这里再写一次把 macros 带上;
    # 复用 checkpoint 的旧 workspace 全靠这一次才拿到这一层。
    save_thematic_map(ws, theme_map)
    yield f"   🏛️ {len(macros)} 个宏观板块已写入 {ws.path.name}/_thematic_map.json\n"


# ============================================================
# Phase B2: 命题合并
# ============================================================

def _phase_claims(ws, usable: list[dict[str, Any]], theme_map: dict[str, Any],
                  from_stage: str, sig: str,
                  paper_titles: dict[str, str] | None = None):
    """主题 -> (技术路线) -> 命题。产出 (claims, from_stage)。

    缺这一层, 综述就没有"综": fact 逐条成句、逐条挂引用, 同一篇论文在一段话里
    被重复引 5 次(40 篇那次实测)。合并成命题后, 一句话可以挂多篇。

    10-03 改走**范式层**(主题 -> 技术路线 -> 命题)。原路线按 subject 分簇再让
    LLM 合并, 但 subject 是"按论文取的标签"(实测 93% 的 fact 其 subject 只对应
    一篇论文), LLM 正确地不合并, 命题 ≈1:1 对应论文 -> 均引只有 0.94/句。
    范式层实测均引 12.0/句。任何一步失败都退回原路线。

    10-04 抽成独立生成器, 与 _phase_facts / _phase_write 同一形状。原因不是
    C901, 是**这个块以前根本没法测**: 它内联在 _run_organize_phase 里, 而
    现有的续跑测试是 `assert "load_claims" in inspect.getsource(...)` —— 只
    验证那行字符串在源码里, 不验证它返回值。于是下面两个 bug 一起从测试缝里
    漏了过去(10-03 上线, 11 天):
      (a) 原来的 `try: claims = condense_facts_by_paradigm(...)` 缩进在
          if/else **之外** -> 续跑时也会无条件重跑范式层, 刚读出来的 checkpoint
          被直接覆盖, "沿用 checkpoint" 那行提示是假的;
      (b) import 写在 else 里 -> 走 if 分支时那个名字是未绑定的局部变量, 必然
          UnboundLocalError, 被 except 吞掉 -> claims=[] -> 退主题合并 -> 那个
          名字同样没绑定, 再抛一次 -> 最后落到"退回 fact 直写"。
      合起来的效果: 续跑不但没省调用, 还把范式层(均引 12.0/句)换成了代码自己
      标注为劣质的旧路线(0.94/句), 而 UI 里只看得到一行 logger.warning。
    另: 原来的 save_claims / save_checkpoint 嵌在 `if not claims:` 里面, 等于
    **只有范式层失败走了兜底时才落盘** —— 主路线从不写 _claims.json。实测佐证:
    10-03 15:09 走范式路线跑完的 workspace 没有 _claims.json, 13:18 范式失败
    走兜底的反而有。
    """
    claims: list[dict[str, Any]] = []
    paradigms: dict[str, list[dict[str, Any]]] = {}
    if from_stage in ("claims", "narratives", "done"):
        claims = ws.load_claims() or []
    if claims:
        # 复用路径: 一定不碰 LLM。这条 if/else 边界是本函数存在的理由。
        # 范式层不重跑也不重落盘: _paradigms.json 是当初生成这些 claims 时写的,
        # 已经在磁盘上; 再写一遍只会用一份新的 created_at 覆盖掉旧的溯源信息。
        yield f"\n♻️ 命题合并: 沿用 checkpoint 的 {len(claims)} 条命题\n"
    else:
        from_stage = "themes"
        yield "\n🔗 归纳技术路线并合并成命题...\n"
        from ._organize import condense_facts_by_paradigm, condense_facts_by_theme
        try:
            claims, paradigms = condense_facts_by_paradigm(usable, theme_map)
        except Exception as e:
            logger.warning("[organize] 范式层异常, 退回主题合并: %s", e)
            claims, paradigms = [], {}
        if paradigms:
            # 10-05: 范式层以前只把名字拍扁进 claim 的 statement 字符串, 用完
            # 就丢 —— 于是磁盘上从来没有"技术路线"这一层。而 fact -> 范式 ->
            # 主题 的多归属结构正是"一篇论文同时落在多个主题里"的来源。
            ws.save_paradigms(paradigms)
            n_par = sum(len(v) for v in paradigms.values())
            yield f"   🧭 归纳出 {n_par} 条技术路线, 覆盖 {len(paradigms)} 个主题\n"
        if not claims:
            # 10-04 加: 范式层挂掉原先只在 logger.warning 里, 用户的流式输出看不到。
            # 而这一层直接决定引文密度(实测 12.0/句 vs 旧路线 0.94/句), 静默退回
            # 等于"看着正常但质量掉了一档" —— 正是这个模块开头写的"宁可吵, 不可
            # 静默"要防的那一类。所以这里明确告诉用户走了兜底。
            yield ("   ⚠ 范式层没产出, 已退回主题合并(引文密度会明显变稀)\n")
            try:
                claims = condense_facts_by_theme(usable, theme_map)
            except Exception as e:
                logger.warning("[organize] 命题合并失败, 退回 fact 直写: %s", e)
                claims = []
        if claims:
            ws.save_claims(claims)
            ws.save_checkpoint("claims", signature=sig, n_claims=len(claims))

    if not claims:
        yield "⚠ 命题合并不可用, 退回逐 fact 直写(引文会偏稀疏)\n"
        return claims, from_stage

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
    return claims, from_stage


def _phase_gap_fill(topic: str, ws, theme_map: dict[str, Any],
                    pool: list[dict[str, Any]],
                    facts: list[dict[str, Any]],
                    usable: list[dict[str, Any]], sig: str):
    """Phase B1: 拿 Phase B 的覆盖空白再检索一轮, 重跑主题地图(10-06 加)。

    **这是"评阅结果回不到主循环"的修复**
    organize 路线没有 ReAct 循环(见 `survey_generate` 开头 `phase == "organize"`
    那个 early return), 而老路线的 `section_gaps -> action_search_more` 只认
    `write_section` 里的 `<!-- QAgent: missing_queries -->` 标记。于是 Phase B
    吐出来的空白只能进渲染文件, 永远变不成"再检索一轮"。

    实测(10-06, neutral atom quantum computing): Phase B 给出 31 条空白, 里面
    有一条 "Atom transport / rearrangement protocols are absent from this corpus"
    —— 这正是"再多拉一批原子输运论文就能补上"的那种, 但工具读到它的时候综述
    已经写完了, 那条空白就只是一行字。

    **一轮就够, 而且默认只跑 1 轮**: 每轮的成本是
    1 次判定 + N 次检索 HTTP + M 篇 summarize + M 次 fact 抽取 + 一次主题重跑。
    M 取 20 的话一轮约 40 次调用, 已经跟整个 Phase 2 (39 篇 × 2 轮审查) 一个量级。
    回补两轮以上时**新增的空白往往还是补不上**(那说明缺的是别的领域), 多花一轮
    换不来东西, 所以 `QAGENT_SURVEY_GAP_FILL_ROUNDS` 默认 1。

    返回 (theme_map, pool, usable, fill_log)
    """
    rounds_max = get_gap_fill_rounds()
    fill_log: list[dict[str, Any]] = []
    theme_map = dict(theme_map)

    for rnd in range(1, rounds_max + 1):
        gaps = [g for g in (theme_map.get("gaps") or []) if (g or "").strip()]
        if not gaps:
            yield f"\n🩹 覆盖空白回补: 第 {rnd} 轮, 但没有空白, 跳过\n"
            break

        yield (f"\n🩹 覆盖空白回补 第 {rnd}/{rounds_max} 轮 "
               f"({len(gaps)} 条空白, 看哪些补得上)...\n")
        from ._pool._query_expand import gaps_to_queries
        verdict = gaps_to_queries(topic, gaps, max_queries=get_gap_fill_max_queries())
        if not verdict.get("ok"):
            yield f"   ⚠ 判定失败 ({verdict.get('error')}), 本轮不回补\n"
            break
        fillable = verdict.get("fillable") or []
        unfillable = verdict.get("unfillable") or []
        if unfillable:
            yield (f"   打分后取前 {len(fillable)} 条回补, "
                   f"其余 {len(unfillable)} 条判定补不上(如实留在清单里)\n")
        if not fillable:
            yield "   没有可回补的空白, 结束回补\n"
            break

        # ---- 检索 + 逐篇 summarize + 进池 (复用老路线那个 action) ----
        from ._actions import action_extract_facts, action_search_more
        before_ids = {p.get("paper_id") for p in pool}
        gstate = {
            "topic": topic, "pool": list(pool), "facts": [],
            "errors": [], "step": 0, "finish_requested": False, "max_steps": 1,
        }
        cap = get_gap_fill_max_papers()
        for item in fillable:
            q = item["query"]
            yield f"   → 检索「{q[:64]}」\n"
            try:
                for line in action_search_more(
                        {"query": q, "k": max(2, cap // len(fillable)),
                         "source": "all"}, gstate, ws):
                    if line and line.startswith(("⚠", "👁", "🔎")):
                        yield f"   {line}"
            except Exception as e:                      # noqa: BLE001
                yield f"   ⚠ 检索「{q[:48]}」失败: {type(e).__name__}: {str(e)[:80]}\n"

        new_papers = [p for p in gstate["pool"]
                      if p.get("paper_id") not in before_ids]
        # 超上限就砍: 后面每篇都要 summarize + 抽 fact, 这里不设闸会失控
        if len(new_papers) > cap:
            yield f"   (候选 {len(new_papers)} 篇 > 上限 {cap}, 只取前 {cap} 篇)\n"
            new_papers = new_papers[:cap]
        if not new_papers:
            yield "   没召回到新论文, 结束回补\n"
            break

        pool = list(gstate["pool"])
        ws.save_pool(pool)
        yield f"   新论文 {len(new_papers)} 篇已入池\n"

        # ---- 对新论文抽 fact ----
        before_facts = len(facts)
        estate = {
            "topic": topic, "pool": pool, "facts": [],
            "errors": [], "step": 0, "finish_requested": False, "max_steps": 1,
        }
        try:
            for line in action_extract_facts(
                    {"paper_ids": [p["paper_id"] for p in new_papers]}, estate, ws):
                if line and line.startswith(("👁", "✅", "📚")):
                    yield f"   {line}"
        except Exception as e:                          # noqa: BLE001
            yield f"   ⚠ 新论文 fact 抽取失败: {type(e).__name__}: {str(e)[:80]}\n"

        new_facts = estate.get("facts") or []
        if not new_facts:
            yield "   新论文一条 fact 都没抽出来, 结束回补\n"
            break
        facts = facts + new_facts
        usable = [f for f in facts if _fact_is_usable(f)]
        yield (f"   fact {before_facts} → {len(facts)} 条 "
               f"(可用 {len(usable)})\n")
        # **不要再调 ws.save_facts(facts)** —— 10-06 实测这行必崩:
        # SurveyWorkspace 根本没有 save_facts 方法(全仓库只有这一处提到它),
        # AttributeError 正好砸在"回补真的抽到了 fact"这一刻 —— 也就是回补
        # 唯一有价值的时刻。而且它是多余的: action_extract_facts 自己在
        # _extract_facts.py:158 就 `ws.append_fact(f)` 逐条落盘了,
        # 主线 `_stage_facts`(415-425 行)同样从不单独存 fact。

        # ---- 重跑主题地图(10-07 改: 增量并入, 不再从零重跑) ----
        #
        # 原来这里是 `organize_facts_into_themes(usable, ...)` —— 拿**全部** fact
        # 从零归纳一遍。实测(10-07 经典阴影第二轮)这是「83 篇池最后只引用 19 篇」
        # 的直接原因:
        #   353 条 fact 里 LLM 归纳出 **34 个主题**, 撞 max_themes=8 上限 ->
        #   26 个主题连带 **223 条 fact(63%)被整块截掉**, 这些 fact 再也没机会
        #   进命题层/参考文献。而且主题名每轮重新生成, 按名合并命中经常是 0。
        # 现在只归纳**这一轮新增的** fact, 骨架沿用上一轮的, 结果按名并进旧主题,
        # 旧主题一条都不砍。见 organize_facts_into_themes 的 prev_map/induce_facts。
        titles = {p["paper_id"]: (p.get("title") or "") for p in pool}
        from ._organize import organize_facts_into_themes, save_thematic_map
        new_usable = [f for f in new_facts if _fact_is_usable(f)]
        tmap = organize_facts_into_themes(
            usable, paper_titles=titles,
            prev_map=theme_map, induce_facts=new_usable)
        if not tmap.get("themes"):
            yield "   ⚠ 重跑主题地图失败, 沿用上一轮结果\n"
            break
        old_names = {t["name"] for t in theme_map.get("themes") or []}
        new_names = {t["name"] for t in tmap["themes"]}
        kept = len(new_names & old_names)
        theme_map = tmap
        save_thematic_map(ws, theme_map)
        yield (f"   主题地图增量并入: {len(old_names)} → {len(new_names)} 个主题"
               f"(新增 {len(new_names - old_names)}, 按名合并命中 {kept})\n")
        if not kept:
            # 「宁可吵, 不可静默」。增量模式下这条基本不该触发(旧主题原样保留),
            # 触发就说明旧主题真的一条都没保住, 下游产物会接不上。
            yield ("   ⚠ 按名合并命中 0: 旧主题名一个都没保住。依赖主题名的下游\n"
                   "     产物(_claims / _narratives / survey_section)会接不上。\n")
        fill_log.append({"round": rnd, "n_new_papers": len(new_papers),
                         "n_new_facts": len(new_facts),
                         "n_themes": len(new_names),
                         "n_kept_by_name": kept,
                         "acted": [{"gap": f["gap"], "query": f["query"],
                                    "score": f["score"]} for f in fillable],
                         "judged_not_acted": list(verdict.get("unfillable") or []),
                         "n_judged": len(gaps)})
        # 把**这一轮真的判过、真的搜过的那些空白原文**挂到主题地图上。
        # 注意挂的是**第一轮的原文**, 不是 tmap["gaps"] —— 后者是重跑后模型
        # **重新生成**的一批, 和第一轮逐字不同的概率极高。
        # 10-06 实测踩过: 拿第一轮的 filled 去标第二轮的列表, 命中 0,
        # 32 条**全部**被打上"判定补不上", 包括刚搜成功的 5 条。那比静默更糟 ——
        # 它看着权威、很确定、而且是错的。
        prev = theme_map.get("gap_fill_rounds") or []
        theme_map["gap_fill_rounds"] = prev + fill_log[-1:]

    return theme_map, pool, usable, fill_log


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

    崩溃续跑 (10-02 起, 10-04 修):
        这条流水线是严格串行的, 中途挂掉就整轮作废 —— v60e 实测跑了 76 分钟 /
        151 次调用死在主题归纳那一步, 而 fact 早就落盘在 _facts.jsonl 里了。
        每个阶段完成时打一个 checkpoint, 下次参数没变就从最后一个完成的阶段接着走。

        三个防错设计 (都是"静默接上旧数据"这类最难查的 bug):
        1. **signature 必须逐位相同**: signature = sha1(PROMPT_VERSION | topic_hash |
           max_pool | target_chars)。参数或 prompt 版本任一变了就不续 —— 否则会把
           旧条件下的中间产物接到新条件上, 输出看着完全正常但结论是错的。
           (PROMPT_VERSION 是 10-04 补的: 之前只覆盖三个参数, 于是 10-03 把命题层
            从 subject 换成范式层时签名没变, 旧产物会被当成同参数合法产物接上。)
        2. **续跑时一定 yield 一行「♻️ 续跑」**。宁可吵, 不可静默。
        3. **阶段梯 _STAGES 的每一级都真的会被写、也真的会被读**。10-04 之前
           "done" 从没被写过(→ 读端是死代码), "narratives" 只写不读(→ 续跑会重跑
           最贵的主题地图那一步), 命题层的 save 还嵌在 fallback 分支里(→ 主路线
           从不落盘)。改这套的时候, 写端和读端的阶段集合必须一起核对。
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
    if from_stage in ("themes", "claims", "narratives", "done"):
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
        # 10-05: 关系不再有 type 标签, 只显示那句具体的话(见 _organize 里
        # "不再存 type" 的注释)。空 note 的边也要显示 —— 少了 type 它就
        # 什么都不剩, 静默丢掉等于假装没有这条关系。
        for rel in t.get("relations") or []:
            note = (rel.get("note") or "").strip()
            yield (f"       ↔ 「{rel.get('to')}」: {note}\n" if note
                    else f"       ↔ 「{rel.get('to')}」(无说明)\n")
    for w in theme_map.get("warnings", []):
        yield f"   ⚠ {w}\n"

    # ---- Phase B1: 覆盖空白回补 (10-06 加) ----
    # 这是"评阅完不回传给主循环"的修复。organize 路线是**直线管线**, 没有 ReAct
    # 循环可回(见本文件 survey_generate 开头 phase=="organize" 那个 early return),
    # 而老路线那套 section_gaps -> search_more 机制只认 write_section 里的
    # `<!-- QAgent: missing_queries -->` 标记。于是 Phase B 归纳出的 31 条空白
    # 全部只能进渲染, 一条都没转化成"再检索一轮" —— 实测跑完还是那 39 篇。
    #
    # 这里补上那一段: 空白 -> 判定可补不可补 -> 可补的配英文检索式 -> 检索 +
    # 逐篇 summarize -> 对新论文抽 fact -> **重跑主题地图** -> 再算宏观层。
    # 主题地图本身是分块 + 按名合并的, 重跑会自动把新 fact 并进已有主题。
    if get_gap_fill_rounds() > 0:
        theme_map, pool, usable, fill_log = yield from _phase_gap_fill(
            topic, ws, theme_map, pool, facts, usable, _sig)
    else:
        fill_log = []
    if fill_log:
        yield (f"\n🩹 覆盖空白回补: {fill_log[0]['n_new_papers']} 篇新论文, "
               f"{fill_log[0]['n_new_facts']} 条新 fact\n")

    # ---- Phase B1.5: 宏观主题层 (10-05 加) ----
    yield from _phase_macros(ws, theme_map)
    if theme_map.get("macros"):
        by_name = {t["name"]: len(t.get("fact_ids") or []) for t in theme_map["themes"]}
        yield f"\n🏛️ 归纳出 {len(theme_map['macros'])} 个宏观板块:\n"
        for i, m in enumerate(theme_map["macros"], 1):
            n_facts = sum(by_name.get(n, 0) for n in m["theme_names"])
            yield (f"   ({i}) {m['name']} — 含 {len(m['theme_names'])} 个主题 / "
                    f"{n_facts} 条 fact\n")
            if m.get("statement"):
                yield f"       {m['statement']}\n"
            for cn in m.get("cross_notes") or []:
                yield f"       ↔ {cn}\n"
        for w in theme_map.get("macro_warnings") or []:
            yield f"   ⚠ {w}\n"

    # ---- Phase B2: 命题合并 (9-30 加, 10-03 改成范式层) ----
    # 10-08: 把论文标题传下去 —— 支撑判定("这篇是在提出还是在使用那个技术")
    # 靠标题判断最准, "A Bayesian analysis of classical shadows" 本身就说清了。
    claims, from_stage = yield from _phase_claims(
        ws, usable, theme_map, from_stage, _sig)

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
    # 10-04 加: 阶段梯 _STAGES 里有 "done", 但从来没人写过它。补上之后, 万一在
    # 最后一步(清 checkpoint 之前)崩掉, 下次能识别"已经跑完了"而不是当成
    # narratives 阶段重跑。正常路径下一行就被下面的 clear_checkpoint 清掉。
    ws.save_checkpoint("done", signature=_sig, n_sections=created)
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
            # 10-05: 不再有 type —— 只记那句具体的话
            lines.append(f"- 关联 → {rel.get('to')}: {rel.get('note') or '(无说明)'}")
        for w in nar.get("warnings", []):
            lines.append(f"- ⚠ {w}")
        lines.append("")
    all_ids = {f.get("fact_id") for f in facts}
    unused = sorted(all_ids - total_used)
    lines.append("## fact 使用率")
    lines.append(f"- 总计: {len(all_ids)} 条可用 fact")
    lines.append(f"- 被引用: {len(total_used)} 条 ({len(total_used) * 100 // max(1, len(all_ids))}%)")
    lines.append(f"- 未被引用: {len(unused)} 条")
    if theme_map.get("macros"):
        lines.append("")
        lines.append("## 宏观板块 (10-05)")
        n2t = {t["name"]: len(t.get("fact_ids", [])) for t in theme_map["themes"]}
        for m in theme_map["macros"]:
            lines.append(
                f"### {m['name']}  ({len(m['theme_names'])} 个主题 / "
                f"{sum(n2t.get(n, 0) for n in m['theme_names'])} 条 fact)")
            if m.get("statement"):
                lines.append(f"- 板块判断: {m['statement']}")
            for cn in m.get("cross_notes") or []:
                lines.append(f"- 跨板块关联: {cn}")
            lines.append(f"- 含主题: {'、'.join(m['theme_names'])}")
    if theme_map.get("macro_warnings"):
        lines.append("")
        lines += [f"- ⚠ {w}" for w in theme_map["macro_warnings"]]
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

    # 10-07 修: 这句原来写的是「fact 里的对齐标记在 _facts.jsonl 里(旧 run 可能
    # 没有)」—— **10-06 之前它一直是假的**。审查结论只改内存对象, 而 append_fact
    # 早在抽取循环里就把 fact 写死进文件了; 实测 2786 行里 `_is_topic_aligned`
    # 出现 0 次。现在 _extract_facts 末尾会把带标记的 fact 重写回盘, 但**旧
    # workspace 仍然是"没有标记"**, 所以这里仍然要防一手 ——
    # 读出来的 fact 若没有标记, 渲染层不能当成"全都通过审查"。
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
