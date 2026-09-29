"""
surveys._actions._extract_facts — 拆出自原 _actions.py (9-20 体检重构)

包含函数: action_extract_facts
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any

from .._pool import _safe_json_loads_array
from ._helpers import _friendly_error

logger = logging.getLogger(__name__)
def action_extract_facts(args: dict[str, Any], state: dict[str, Any], ws) -> Iterator[str]:
    """
    extract_facts action (step 3 真实实现):
      - 对每篇 paper 调 minimax LLM 抽 3-5 条 fact (JSON 数组)
      - 复用 P6 build_fact_prompt + _safe_json_loads_array 容错
      - fact 写 workspace/_facts.jsonl, 加 state['facts']
      - 单篇失败不阻塞 batch
    """
    from agent.prompts_survey import build_fact_prompt
    from agent.tools.llm import _call_llm

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    # 默认抽 state['pool'] 全部; 也可指定 paper_ids 子集
    target_ids = args.get("paper_ids")
    targets = [p for p in state["pool"] if p["paper_id"] in target_ids] if target_ids else list(state["pool"])

    if not targets:
        yield "⚠️ extract_facts: 0 篇目标 (paper_ids 不在 pool 中)\n"
        yield f"👁️ 观察: extract_facts 0 篇, 累计 facts={len(state['facts'])} 条\n"
        return

    yield f"📝 extract_facts: 抽 {len(targets)} 篇, 调 {_PROVIDER}/{_MODEL}\n"

    for _i, paper in enumerate(targets, 1):
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
            from agent.prompts_survey import clean_fact
            for _i, raw in enumerate(facts):
                if not isinstance(raw, dict):
                    continue
                # 9-29: 规范化 —— 去模型自写的 [paper_id] 标注、重编全局唯一 id、
                # 加确定性质检标记(术语是否超出 support_quote / 是否带自指措辞)。
                # 实测: 上一版"离子阱/附加能级"那组编造**出在抽取阶段**, 写作阶段
                # 只是照抄, 所以 prompt 纪律 + 确定性检查要同时上。
                f = clean_fact(raw, paper_id, _i)
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

    # ========== 9-29: fact 落地审查 ==========
    # 编造的源头在抽取阶段(实测"离子内附加能级"原文没有, 抽取器自行补全),
    # 写作阶段只是照抄。这里对抽出的 fact 做一次语义审查, 超纲的打 _grounded=False,
    # 写作阶段会据此剔除。
    if state["facts"]:
        from agent.tools.surveys._fact_audit import audit_fact_grounding
        pool = state.get("pool") or []
        src = {
            p.get("paper_id", ""): " ".join([
                p.get("title", "") or "", p.get("abstract", "") or "",
                p.get("core_method", "") or "", p.get("core_contrib", "") or "",
            ])
            for p in pool
        }
        yield "🔎 fact 落地审查中(核对 claim 有没有超出原文)...\n"
        try:
            st = audit_fact_grounding(state["facts"], src)
            failed_note = f" / 未能判定 {st['failed']}" if st["failed"] else ""
            yield (
                f"👁️ 审查: 通过 {st['grounded']} / 超纲 {st['ungrounded']}{failed_note}"
                f" (共 {len(state['facts'])} 条)\n"
            )
            if st["ungrounded"]:
                state["errors"].append({
                    "step": state["step"], "action": "fact_grounding_audit",
                    "msg": f"{st['ungrounded']} 条 fact 的 claim 超出原文, 写作时会剔除",
                })
        except Exception as e:
            friendly = _friendly_error(e, "fact_audit")
            state["errors"].append({
                "step": state["step"], "action": "fact_grounding_audit",
                "msg": friendly,
            })
            yield f"👁️ 审查失败 ({friendly}), 沿用原 fact\n"

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
            for _i, f in enumerate(state["facts"]):
                fid = f.get("fact_id")
                if not fid:
                    # 用同样的回退 key 逻辑
                    pid = f.get("paper_id", "?")
                    claim_hash = hash(str(f.get("claim", ""))[:50]) & 0xFFFF
                    fid = f"{pid}:{_i}:{claim_hash}"
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
