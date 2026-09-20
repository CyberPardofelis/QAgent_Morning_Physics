"""
surveys._actions._search_more — 拆出自原 _actions.py (9-20 体检重构)

包含函数: action_search_more
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)
def action_search_more(args: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """
    search_more action (step 2 真实实现):
      1. arxiv API 抓 k 篇
      2. 每篇调 minimax LLM 跑单篇 summarize (4 字段)
      3. 写 summary_items.db (source='survey_search')
      4. 加进 state['pool']

    单篇失败不阻塞 batch (spec §7.2 D).

    9-18 加: 多源召回 (9-18 user 拍板, "survey 工具应该用搜索工具搜索, 不要总用 arxiv").
      - args["sources"]: list[str], 候选 ["arxiv", "aps", "iop"], 默认 ["arxiv"] 向后兼容
      - args["source"]: 字符串简写 "arxiv"/"aps"/"iop"/"all", 跟 sources 二选一
      - 多源时, 各路并行抓, 按 paper_id dedup, 然后跑 summarize
      - 控制台打印每路召回数
    """
    from agent.tools.summary import save_summary_item
    from agent.tools.llm import _call_llm
    from agent.prompts_survey import build_summarize_prompt

    query = args.get("query") or state["topic"]
    k = args.get("k", 5)

    # 9-18 加: 解析多源参数 (向后兼容, 不传 = arxiv-only)
    if "sources" in args and args["sources"]:
        sources = [str(s).lower().strip() for s in args["sources"] if s]
    elif args.get("source"):
        # 单字符串简写 "all" → ["arxiv","aps","iop"]
        s = str(args["source"]).lower().strip()
        sources = ["arxiv", "aps", "iop"] if s == "all" else [s]
    else:
        sources = ["arxiv"]  # 向后兼容

    # 过滤无效值
    valid_sources = {"arxiv", "aps", "iop"}
    sources = [s for s in sources if s in valid_sources]
    if not sources:
        sources = ["arxiv"]

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    # 9-08 17:20 user 拍板: 不显示具体论文内容 (paper_id/标题/row_id) , 只显示高层动作
    yield "🔎 去找论文了\n"

    # 9-18 加: 多源并行召回. 单源失败不阻塞 batch.
    raw_papers: List[Dict[str, Any]] = []
    per_source_count: Dict[str, int] = {}
    per_source_journal: Dict[str, str] = {}
    seen_in_search: set = set()
    # 9-18 修: 给每个 paper 标 _src 标签 (arxiv / aps / iop), 后面 journal 标签检测用这个,
    # 不要用 per_source_journal.get("aps") 这种全局判断 (会误把 IOP paper 标成 APS)
    for src in sources:
        try:
            if src == "arxiv":
                papers = _arxiv_search(query, k)
                journal_tag = "arxiv"
            elif src == "aps":
                papers = _openalex_aps_search(query, k)
                journal_tag = "APS"
            elif src == "iop":
                papers = _openalex_iop_search(query, k)
                journal_tag = "IOP"
            else:
                continue
            added = 0
            for p in papers:
                pid = p.get("paper_id", "")
                if not pid or pid in seen_in_search:
                    continue
                seen_in_search.add(pid)
                # 9-18 修: 在 paper dict 上贴 _src 标签, 后面检测用这个
                p["_src"] = src
                raw_papers.append(p)
                added += 1
            per_source_count[src] = added
            per_source_journal[src] = journal_tag
            print(
                f"[action_search_more] {src} 召回  +{added} 篇  query={query[:60]!r}",
                flush=True,
            )
        except Exception as e:
            friendly = _friendly_error(e, f"search_more.{src}")
            state["errors"].append({
                "step": state["step"], "action": f"search_more.{src}",
                "msg": friendly,
            })
            print(
                f"[action_search_more] {src} ❌ {friendly}",
                flush=True,
            )

    if not raw_papers:
        sources_label = "+".join(sources)
        yield f"⚠️ 没找到新论文 ({sources_label})\n"
        yield f"👁️ 候选池 {len(state['pool'])} 篇 (无新增)\n"
        return

    # 2. 单篇 summarize + 写库 (silent, 不 yield 单篇进度)
    # 9-18 加: 按 paper 来源标 _source (multi-source 时审计用)
    # 9-18 修: 用 p["_src"] (召回阶段贴的标签) 准确分类, 不再用 journal 字符串猜
    #   (原来用 "APS" in per_source_journal.get("aps") 全局判断, 永远 True, 导致 IOP paper 被错标成 APS)
    new_pool = []
    for i, p in enumerate(raw_papers, 1):
        paper_id = p["paper_id"]
        # 9-18 修: 用 _src 标签准确判定来源 (召回阶段已贴)
        src_tag = p.get("_src", "")
        journal = p.get("journal") or ""
        if src_tag == "arxiv":
            source_tag = "arxiv_api_via_survey_search"
            journal_label = "arxiv"
        elif src_tag == "aps":
            source_tag = "openalex_aps_api_via_survey_search"
            journal_label = journal or "APS"
        elif src_tag == "iop":
            source_tag = "openalex_iop_api_via_survey_search"
            journal_label = journal or "IOP"
        else:
            source_tag = "external_api_via_survey_search"
            journal_label = journal or "external"

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
                "journal": journal_label,
                "core_method": summary_dict.get("core_method", ""),
                "core_contrib": summary_dict.get("core_contrib", ""),
                "abstract": p.get("abstract", ""),
                "outlook": summary_dict.get("outlook", ""),
                "keywords": summary_dict.get("keywords", ""),
                "_source": source_tag,
            })
        except Exception as e:
            # 单篇失败不阻塞 batch (spec §7.2 D), 但不 yield 错误详情
            state["errors"].append({
                "step": state["step"], "action": "search_more.summarize",
                "paper": paper_id, "msg": _friendly_error(e, "search_more.summarize"),
            })
            # 静默继续

    state["pool"].extend(new_pool)
    # 9-18 加: yield 多源召回汇总 (silent 高层, 不算"具体论文内容")
    src_breakdown = ", ".join(f"{s}+{per_source_count.get(s, 0)}" for s in sources)
    # 9-08 17:20 user 拍板: 只显示"找到 N 篇" 这种高层总结
    yield f"👁️ 找到 {len(new_pool)} 篇 ({src_breakdown}), 累计候选池 {len(state['pool'])} 篇\n"
