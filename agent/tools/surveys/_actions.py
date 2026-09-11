"""
agent.tools.surveys._actions — ReAct 5 个 action + decision + finalize (LLM-heavy 部分)

从原 surveys.py 抽出 (P1.7 重构):
- _friendly_error / _decide_action / _state_driven_decide / _dispatch_action
- _append_react_trace (写 _react_trace.jsonl)
- 5 个 action handler: search_more / extract_facts / write_outline / write_section / finish
- _finalize_survey (收尾报告)

L2 工具档, 每个 action 走完整 LLM 链 (provider=minimax, model=MiniMax-M3)
"""
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, Iterator, List, Optional

from ._pool import (
    _safe_json_loads_array,
    _audit_section_paper_ids,
    _arxiv_search,
)
from ._crud import create_section, create_theme

logger = logging.getLogger(__name__)


# ============================================================
# ReAct helpers
# ============================================================
def _friendly_error(e: Exception, action: str) -> str:
    """把异常翻译成友好提示 (spec §4). step 1 简单实现, 后续 step 扩展查表."""
    err_type = type(e).__name__
    msg = str(e)[:200]
    return f"{action} 失败 ({err_type}): {msg}"


def _decide_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ReAct LLM 真决策 (step 6 真实实现).

    1. 调 minimax LLM 拿 decision (build_react_prompt)
    2. parse JSON {thought, action, args}
    3. 失败 fallback 到 _state_driven_decide (state 规则保底)

    9-08 19:30 加: finish_rejected 时直接走 state 规则 (不调 LLM, 不浪费 LLM 调用,
    因为 finish 刚被拒, 已经知道下一步该调什么, 没必要再问 LLM)
    """
    from agent.tools.llm import _call_llm
    from agent.prompts_survey import build_react_prompt

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    fallback = _state_driven_decide(state)

    # 9-08 19:30: finish 刚被拒 → 直接用 state 规则 + forced action hint
    if state.get("finish_rejected"):
        forced = state.pop("_forced_action_hint", None)
        if forced:
            action, args = forced
            return {
                "thought": f"[finish 被拒自动执行] {action}({args})",
                "action": action,
                "args": args,
                "_source": "auto_from_finish_reject",
            }
        # 没 hint 就走 state 规则
        return fallback

    try:
        prompt = build_react_prompt(state)
        out = _call_llm(
            prompt,
            provider=_PROVIDER,
            model=_MODEL,
            enable_thinking=False,
            temperature=0.3,
            max_tokens=1000,
            timeout=60,
        )
        if out and "<think>" in out:
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
        m = re.search(r"\{[\s\S]*\}", out or "")
        if not m:
            raise ValueError("LLM 输出无 JSON")
        decision = json.loads(m.group(0))
        action = decision.get("action", "")
        if action not in ("search_more", "extract_facts", "write_outline", "write_section", "finish"):
            raise ValueError(f"未知 action: {action!r}")
        if not isinstance(decision.get("args"), dict):
            decision["args"] = {}
        decision.setdefault("thought", "")
        decision["_source"] = "llm"  # 标记是 LLM 决策, 不是 fallback
        return decision
    except Exception as e:
        # LLM 决策失败 -> fallback state 规则 (spec §1.3 容错)
        return {
            "thought": f"[fallback] LLM 决策失败 ({type(e).__name__}: {str(e)[:80]}), 走 state 规则",
            "action": fallback["action"],
            "args": fallback["args"],
            "_source": "fallback",
        }


def _state_driven_decide(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    state-driven fallback 规则 (LLM 决策失败时用, 也是测试兜底).

    9-08 user 拍板的固定候选池规则:
      - 候选池 < 20  → search_more (硬规则, 优先)
      - 候选池 ≥ 40 且 fact < 5  → extract_facts (硬规则, 优先)
      - 20 ≤ pool < 40  → LLM 自由(中间区不强制, 让 LLM 按主题判断)
    9-08 user 拍板的拆分:
      - 没 outline → write_outline (硬规则, 写章节前必须先生成大纲)
      - 有 outline 且未写完所有章节 → write_section(section_idx)
      - 写完所有 outline 章节 → finish
    9-08 user 拍板的缺漏反馈:
      - section_gaps 不空 → search_more(用 gap 的 query 扩候选池, 优先)
      - search_more 完成后 → write_section 重写该 gap 章节 (section_idx=gap.section_idx)
    """
    from agent.tools.config import get_min_pool_papers

    pool_size = len(state["pool"])
    fact_count = len(state["facts"])
    section_count = len(state["sections"])
    outline_sections = state.get("outline_sections", [])
    section_gaps = state.get("section_gaps", []) or []

    # ========== 候选池硬规则 (9-08 抽 env QAGENT_SURVEY_MIN_POOL, 默认 30; 充裕 = min+10) ==========
    _MIN_POOL = get_min_pool_papers()
    if pool_size < _MIN_POOL:
        return {"action": "search_more", "args": {"query": state["topic"], "k": 5}}
    if pool_size >= _MIN_POOL + 10 and fact_count < 5:
        return {
            "action": "extract_facts",
            "args": {"paper_ids": [p["paper_id"] for p in state["pool"][:5]]},
        }

    # ========== 缺漏优先: section_gaps 不空 → 补文章 ==========
    # 用最新 gap 的第一个 query 调 search_more 扩池
    # (如果有多个 gap, 一次只补一个, 避免 step 浪费)
    if section_gaps:
        first_gap = section_gaps[0]
        gap_queries = first_gap.get("queries", [])
        if gap_queries:
            return {
                "action": "search_more",
                "args": {"query": gap_queries[0], "k": 5},
            }
        # gap 没 query, 退化: 重写该 section_idx 章节 (让 LLM 重新评估)
        return {
            "action": "write_section",
            "args": {"section_idx": first_gap.get("section_idx", 0)},
        }

    # ========== 大纲规则: 写章节前必须先 write_outline ==========
    if not outline_sections:
        return {"action": "write_outline", "args": {}}

    # ========== 写章节: 按 outline_sections 顺序, 用 section_idx 标识 ==========
    if section_count < len(outline_sections):
        return {
            "action": "write_section",
            "args": {"section_idx": section_count},  # 0-based, 下一章
        }

    # ========== 收尾 ==========
    return {"action": "finish", "args": {}}


def _dispatch_action(decision: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """派发 action 到 5 个 handler (9-08 加 write_outline, 拆 write_section 前的结构生成)"""
    action = decision["action"]
    args = decision.get("args", {})
    if action == "search_more":
        yield from action_search_more(args, state, ws)
    elif action == "extract_facts":
        yield from action_extract_facts(args, state, ws)
    elif action == "write_outline":
        yield from action_write_outline(args, state, ws)
    elif action == "write_section":
        yield from action_write_section(args, state, ws)
    elif action == "finish":
        yield from action_finish(args, state, ws)
    else:
        yield f"⚠️ 未知 action: {action!r}\n"


def _append_react_trace(ws, trace_entry: Dict[str, Any]) -> None:
    """写 _react_trace.jsonl (spec §6). 失败不阻塞主流程."""
    try:
        trace_path = ws.path / "_react_trace.jsonl"
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("[survey_generate] _react_trace 写盘失败: %s", e)


# ============================================================
# 5 个 action handler
# ============================================================
def action_search_more(args: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """
    search_more action (step 2 真实实现):
      1. arxiv API 抓 k 篇
      2. 每篇调 minimax LLM 跑单篇 summarize (4 字段)
      3. 写 summary_items.db (source='survey_search')
      4. 加进 state['pool']

    单篇失败不阻塞 batch (spec §7.2 D).
    """
    from agent.tools.summary import save_summary_item
    from agent.tools.llm import _call_llm
    from agent.prompts_survey import build_summarize_prompt

    query = args.get("query") or state["topic"]
    k = args.get("k", 5)
    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    # 9-08 17:20 user 拍板: 不显示具体论文内容 (paper_id/标题/row_id) , 只显示高层动作
    # 之前 yield 太多细节: "📥 arxiv 返回 N 篇, 开始单篇 summarize"、"  [i/N] summarize xxx: title..."、
    # "    ✅ 入库 (row_id=123)" 等 — 全是污染, 去掉
    yield "🔎 去找论文了\n"

    # 1. arxiv API 抓 (silent)
    try:
        raw_papers = _arxiv_search(query, k)
    except Exception as e:
        friendly = _friendly_error(e, "search_more.arxiv")
        state["errors"].append({
            "step": state["step"], "action": "search_more.arxiv",
            "msg": friendly,
        })
        yield f"❌ 找论文失败: {friendly}\n"
        return

    if not raw_papers:
        yield "⚠️ 没找到新论文\n"
        yield f"👁️ 候选池 {len(state['pool'])} 篇 (无新增)\n"
        return

    # 2. 单篇 summarize + 写库 (silent, 不 yield 单篇进度)
    new_pool = []
    for i, p in enumerate(raw_papers, 1):
        paper_id = p["paper_id"]
        try:
            # 调 LLM 抽 4 字段
            sum_prompt = build_summarize_prompt(
                paper_id=paper_id,
                title=p.get("title", ""),
                authors=p.get("authors", ""),
                abstract=p.get("abstract", ""),
            )
            out = _call_llm(
                sum_prompt,
                provider=_PROVIDER,
                model=_MODEL,
                enable_thinking=False,
                temperature=0.2,
                max_tokens=10000,   # 9-08 user 拍板: 上限 10000
                timeout=180,
            )
            # 剥 <think>
            if out and "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
            # parse JSON (容错)
            m = re.search(r"\{[\s\S]*\}", out or "")
            if not m:
                raise ValueError(f"summarize LLM 输出无 JSON: {(out or '')[:200]}")
            try:
                summary_dict = json.loads(m.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"summarize JSON parse 失败: {e}")

            # 写 summary_items.db (source='survey_search', spec §7.2 C)
            save_summary_item(
                paper_id=paper_id,
                source="survey_search",
                title=p.get("title", ""),
                authors=p.get("authors", ""),
                abstract=p.get("abstract", ""),
                core_method=summary_dict.get("core_method", ""),
                core_contrib=summary_dict.get("core_contrib", ""),
                keywords=summary_dict.get("keywords", ""),
                outlook=summary_dict.get("outlook", ""),
            )
            # 加进 state['pool'], 沿用 Phase A 候选池 dict 格式
            new_pool.append({
                "paper_id": paper_id,
                "title": p.get("title", ""),
                "year": str(p.get("year", "")),
                "journal": "arxiv",
                "core_method": summary_dict.get("core_method", ""),
                "core_contrib": summary_dict.get("core_contrib", ""),
                "abstract": p.get("abstract", ""),
                "outlook": summary_dict.get("outlook", ""),
                "keywords": summary_dict.get("keywords", ""),
                "_source": "arxiv_api_via_survey_search",
            })
        except Exception as e:
            # 单篇失败不阻塞 batch (spec §7.2 D), 但不 yield 错误详情
            state["errors"].append({
                "step": state["step"], "action": "search_more.summarize",
                "paper": paper_id, "msg": _friendly_error(e, "search_more.summarize"),
            })
            # 静默继续

    state["pool"].extend(new_pool)
    # 9-08 17:20 user 拍板: 只显示"找到 N 篇" 这种高层总结
    yield f"👁️ 找到 {len(new_pool)} 篇, 累计候选池 {len(state['pool'])} 篇\n"


def action_extract_facts(args: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """
    extract_facts action (step 3 真实实现):
      - 对每篇 paper 调 minimax LLM 抽 3-5 条 fact (JSON 数组)
      - 复用 P6 build_fact_prompt + _safe_json_loads_array 容错
      - fact 写 workspace/_facts.jsonl, 加 state['facts']
      - 单篇失败不阻塞 batch
    """
    from agent.tools.llm import _call_llm
    from agent.prompts_survey import build_fact_prompt

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    # 默认抽 state['pool'] 全部; 也可指定 paper_ids 子集
    target_ids = args.get("paper_ids")
    if target_ids:
        targets = [p for p in state["pool"] if p["paper_id"] in target_ids]
    else:
        targets = list(state["pool"])

    if not targets:
        yield "⚠️ extract_facts: 0 篇目标 (paper_ids 不在 pool 中)\n"
        yield f"👁️ 观察: extract_facts 0 篇, 累计 facts={len(state['facts'])} 条\n"
        return

    yield f"📝 extract_facts: 抽 {len(targets)} 篇, 调 {_PROVIDER}/{_MODEL}\n"

    for i, paper in enumerate(targets, 1):
        paper_id = paper["paper_id"]
        # 9-08 17:20 user 拍板: 不显示具体 fact_id / 单篇进度
        try:
            fact_prompt = build_fact_prompt(
                paper_id=paper_id,
                title=paper.get("title", ""),
                year=str(paper.get("year", "")),
                journal=paper.get("journal", ""),
                core_method=paper.get("core_method", ""),
                core_contrib=paper.get("core_contrib", ""),
                abstract=paper.get("abstract", ""),
                outlook=paper.get("outlook", ""),
            )
            out = _call_llm(
                fact_prompt,
                provider=_PROVIDER,
                model=_MODEL,
                enable_thinking=False,
                temperature=0.2,
                max_tokens=10000,   # 9-08 user 拍板: 上限 10000
                timeout=180,
            )
            # 剥 <think>
            if out and "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
            # 容错 parse JSON 数组
            m = re.search(r"\[[\s\S]*", out or "")
            if not m:
                raise ValueError(f"LLM 输出无 JSON 数组: {(out or '')[:200]}")
            facts = _safe_json_loads_array(m.group(0))
            if facts is None:
                raise ValueError(f"JSON parse 失败 (含容错): {(out or '')[:200]}")
            # 落 _facts.jsonl
            for f in facts:
                if not isinstance(f, dict):
                    continue
                f.setdefault("paper_id", paper_id)
                ws.append_fact(f)
                state["facts"].append(f)
        except Exception as e:
            # 单篇失败不阻塞 batch, 不 yield 详情
            state["errors"].append({
                "step": state["step"], "action": "extract_facts",
                "paper": paper_id, "msg": _friendly_error(e, "extract_facts"),
            })

    # 9-08 17:20 user 拍板: 只显示"抽了 N 条 fact" 高层
    yield f"👁️ 抽完 {len(targets)} 篇, 累计 {len(state['facts'])} 条 fact\n"

    # ========== 9-09 user 拍板: pre-outline fact 主题对齐审查 (mechanism B) ==========
    # 跟 _relevance_filter_pool (mechanism A) 配套 — paper 严判只过滤 pool,
    # 但 pool 内的 fact 仍可能是从邻近子领域 paper 抽出来的"挂羊头卖狗肉" 内容
    # 这里对每条 fact 做主题对齐审查, 不删非对齐 fact, 只打标 (audit 用)
    from agent.tools.config import get_fact_review_threshold
    fact_threshold = get_fact_review_threshold()
    if fact_threshold > 0 and state["facts"]:
        from agent.tools.surveys._pool import _review_facts_topic_alignment
        yield "🔎 fact 主题对齐审查中...\n"
        try:
            scored_facts, fact_stats = _review_facts_topic_alignment(
                topic=state["topic"],
                facts=state["facts"],
                threshold=fact_threshold,
            )
            # 把审查结果写回每条 fact
            for i, f in enumerate(state["facts"]):
                fid = f.get("fact_id")
                if not fid:
                    # 用同样的回退 key 逻辑
                    pid = f.get("paper_id", "?")
                    claim_hash = hash(str(f.get("claim", ""))[:50]) & 0xFFFF
                    fid = f"{pid}:{i}:{claim_hash}"
                s = scored_facts.get(fid, {})
                f["_topic_alignment_score"] = s.get("score", 5)
                f["_is_topic_aligned"] = s.get("is_topic_aligned", True)
                f["_alignment_reason"] = s.get("reason", "")
            # 存 stats 给 finalize 用
            state["_fact_review_stats"] = fact_stats
            yield (
                f"👁️ fact 审查: 对齐 {fact_stats['aligned']} / 邻近 {fact_stats['misaligned']}"
                f" / fallback {fact_stats['fallback']} (共 {fact_stats['total']} 条)\n"
            )
        except Exception as e:
            friendly = _friendly_error(e, "fact_review")
            state["errors"].append({
                "step": state["step"], "action": "fact_review",
                "msg": friendly,
            })
            yield f"👁️ fact 审查失败 ({friendly}), 跳过 — 用原始 fact 列表\n"
    else:
        # fact 审查关闭 或 无 fact, 不 yield 任何东西
        pass


def action_write_outline(args: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """
    write_outline action (9-08 新增, user 拍板拆分 write_section):
      - 调 minimax LLM 走 build_outline_prompt 生成综述大纲(5 段固定: intro/progress/method_compare/gap/outlook)
      - 解析 JSON: {sections: [{title, section_type, sub_points, is_transition}, ...]}
      - 存 state["outline"] (raw) + state["outline_sections"] (结构化 list)
      - 落 workspace/_outline.json (SurveyWorkspace.save_outline 已有)
      - outline_sections 锁住后, write_section 按 section_idx 顺序写

    大纲生成失败不重试(LLM 调用本身就贵, 失败就让 LLM 下一轮重选 action)
    """
    from agent.tools.llm import _call_llm
    from agent.tools.config import get_max_section_tokens
    from agent.prompts_survey import build_outline_prompt, SECTION_REF_QUOTAS

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    # 已存在大纲 → 跳过(避免 LLM 重复调)
    if state.get("outline_sections"):
        # 9-08 17:25 改: silent 跳过, 不 yield 啰嗦
        return

    yield f"📋 生成综述大纲\n"

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
    parsed: Optional[Dict[str, Any]] = None
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
        yield f"  ❌ 大纲 JSON parse 失败, 下一轮重试\n"
        return

    # 规范化 sections
    # 9-08 17:00 user 拍板: section_type 容错 + 至少 5 段
    #   - LLM 偶尔拼写错: "Method-Compare" / "GAP" / "out look" → 强制 normalize 到 5 段合法值
    #   - LLM 偷懒只返 2 段 → 补到 5 段 (intro/progress/method_compare/gap/outlook)
    VALID_TYPES = {"intro", "progress", "method_compare", "gap", "outlook"}

    def _normalize_sec_type(raw: str) -> Optional[str]:
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
    # 注: 完整大纲(含子点)落 workspace/_outline.json + _finalize_survey 报告里展示, 不在主循环啰嗦 yield
    # (之前加了一堆 📑 === 综述结构 === 大段 markdown 把主循环进度淹没了, 已回退)


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


def action_finish(args: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """
    finish action (9-08 19:30 三重校验 + 自动打回执行):
      1. 章节数校验: section_count < len(outline_sections) → 拒绝, 自动触发 write_section
      2. 缺漏校验: state["section_gaps"] 不空 → 拒绝, 自动触发对应工具
      3. 全部通过 → 真正 finish (state["finish_requested"] = True)

    9-08 19:30 user 拍板:
      - 0 引用候选 不算"打回重写" 条件 (删 9-08 18:13 的 unused 校验)
      - finish 被拒要 **自动** 调用相关工具去解决, 不等用户说
        (实现: 标记 state["finish_rejected"]=True, 主循环下一轮 _decide_action 不调 LLM, 走 state 规则)
    """
    outline_sections = state.get("outline_sections", []) or []
    section_count = len(state.get("sections", []))
    gaps = state.get("section_gaps", []) or []

    # ========== 1. 章节数校验 ==========
    if outline_sections and section_count < len(outline_sections):
        missing_idx = list(range(section_count, len(outline_sections)))
        missing_titles = ", ".join(
            f"'{outline_sections[i].get('title', '?')}'" for i in missing_idx[:5]
        )
        more = f" 等 {len(missing_idx)} 章" if len(missing_idx) > 5 else ""
        # 9-08 19:30 拍板: finish 被拒要自动执行相关工具 (write_section)
        state["finish_rejected"] = True
        state["_forced_action_hint"] = ("write_section", {"section_idx": section_count})
        yield (
            f"❌ finish 被拒: 章节未写完! "
            f"已写 {section_count}/{len(outline_sections)} 章, "
            f"还差 {len(missing_idx)} 章 ({missing_titles}{more})\n"
            f"👁️ 自动触发: write_section(section_idx={section_count})\n"
        )
        return

    # ========== 2. 缺漏校验 ==========
    if gaps:
        gap_titles = ", ".join(f"'{g.get('title', '?')}'" for g in gaps[:5])
        # 9-08 19:30 拍板: 自动执行 — 优先 search_more (用 gap query), 没 query 就 write_section
        first_gap = gaps[0]
        gap_queries = first_gap.get("queries", [])
        if gap_queries and "内容过短" not in gap_queries[0]:
            # 缺漏是 search_more 类的, 自动调 search_more
            state["finish_rejected"] = True
            state["_forced_action_hint"] = ("search_more", {"query": gap_queries[0], "k": 5})
            yield (
                f"❌ finish 被拒: 还有 {len(gaps)} 个章节缺漏/空章节未补 ({gap_titles} 等)\n"
                f"👁️ 自动触发: search_more(query='{gap_queries[0]}')\n"
            )
        else:
            # 缺漏是"内容过短" 或没 query, 自动调 write_section 重写
            target_idx = first_gap.get("section_idx", section_count)
            state["finish_rejected"] = True
            state["_forced_action_hint"] = ("write_section", {"section_idx": target_idx})
            yield (
                f"❌ finish 被拒: 还有 {len(gaps)} 个章节缺漏/空章节未补 ({gap_titles} 等)\n"
                f"👁️ 自动触发: write_section(section_idx={target_idx}) 重写缺漏章节\n"
            )
        return

    # ========== 3. 全部通过 → 真正 finish ==========
    state["finish_rejected"] = False
    state["finish_requested"] = True
    yield (
        f"👁️ 观察: ✅ 收尾确认 (章节 {section_count}/{len(outline_sections)} 已写完, 无缺漏) → 🏁\n"
    )


# ============================================================
# 收尾 (step 5)
# ============================================================
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
