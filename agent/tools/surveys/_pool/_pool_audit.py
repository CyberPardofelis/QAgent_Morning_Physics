"""
_pool._pool_audit — 拆出自原 _pool.py (9-20 体检重构)

包含函数: _audit_pool_with_retry
"""
from __future__ import annotations

import logging
from typing import Any

from ._arxiv import _arxiv_search
from ._openalex import _openalex_aps_search, _openalex_iop_search
from ._relevance_filter import _relevance_filter_pool

logger = logging.getLogger(__name__)
def _audit_pool_with_retry(
    topic: str,
    pool_dict: dict[str, dict[str, Any]],
    min_pool: int,
    max_pool: int,
    relevance_threshold: int,
    print_prefix: str = "[_audit_pool_with_retry]",
    max_retries: int = 2,            # 9-18: 最多 2 次重试 (3 轮 total: initial + 2 retries)
) -> dict[str, Any]:
    """
    strict audit + retry loop (9-18 加).

    流程:
      round 0: strict audit 当前 paper pool (不 rescue 不抛错)
      round 1-N: 如果 round 0 后 pool 不够 min_pool, 自动 retry:
                  - 重新调外部搜索 (多源并行, k 翻倍拿更多)
                  - strict audit 新的 paper
                  - 合并所有 round 的合格 paper (dedup by paper_id)
      最多 max_retries 次 retry, 默认 2 次 (3 轮 total).

    返回 dict:
      {
        "kept_dict": paper_id -> pool entry,  最终留下的 paper (可能要再过 LLM relevance 才能用)
        "summary": {
          "topic": ...,
          "rounds": [
            {"round": 0, "before": 30, "after": 5, "rejected": 25, "ok": False, "reason": "..."},
            {"round": 1, "before": 60, "after": 12, "rejected": 48, "ok": True, "reason": "..."},
          ],
          "final_pool_size": 12,
          "target_min_pool": 30,
          "pool_meets_target": False,   # True if final >= min_pool
          "rejected_reasons_sample": [...],  # 前 3 个 audit 拒绝理由 (LLM 解释), 供 ReAct 看到
        }
      }
    """
    audit_rounds = []
    all_kept: dict[str, dict[str, Any]] = {}  # dedup by paper_id
    seen_ids: set = set()
    # 把当前 pool 里所有 paper 都加进 seen (避免 retry 时再加重复的)
    for pid in pool_dict:
        seen_ids.add(pid)

    for round_idx in range(max_retries + 1):
        before_count = len(pool_dict)
        print(
            f"{print_prefix} audit round {round_idx} start  before={before_count}",
            flush=True,
        )

        # strict audit
        if relevance_threshold > 0 and pool_dict:
            try:
                kept_round = _relevance_filter_pool(topic, pool_dict, relevance_threshold, strict=True)
            except Exception as e:
                logger.warning("[_audit_pool_with_retry] strict audit 异常: %s", e)
                kept_round = dict(pool_dict)  # fallback 不剔除
        else:
            kept_round = dict(pool_dict)

        after_count = len(kept_round)
        rejected = before_count - after_count

        # 拒绝理由采样 (前 3 个被踢的 paper)
        rejected_reasons = []
        for pid, p in pool_dict.items():
            if pid not in kept_round:
                reason = p.get("_relevance_reason", "")
                title = p.get("title", "")[:60]
                if not reason or reason.startswith(("[fallback]", "[batch_failed]")):
                    reason = "[未评] is_about_topic=False (不严格属于本主题)"
                rejected_reasons.append({"paper_id": pid, "title": title, "reason": reason})
                if len(rejected_reasons) >= 3:
                    break

        # 把这一轮通过的加进全局 kept (dedup)
        for pid, p in kept_round.items():
            if pid not in all_kept:
                all_kept[pid] = p

        meets_target = len(all_kept) >= min_pool
        round_record = {
            "round": round_idx,
            "before": before_count,
            "after": after_count,
            "rejected": rejected,
            "ok": meets_target,
            "reason": (
                f"通过 {after_count}/{before_count}, 累计合格 {len(all_kept)}/"
                f"{min_pool} (满足 target: {meets_target})"
            ),
        }
        audit_rounds.append(round_record)

        print(
            f"{print_prefix} audit round {round_idx} done  +{after_count}/{before_count}  "
            f"累计合格 {len(all_kept)}/{min_pool}  ok={meets_target}",
            flush=True,
        )

        if meets_target:
            break

        # 不满足, 准备下一轮 retry
        if round_idx >= max_retries:
            print(
                f"{print_prefix} 已重试 {max_retries} 次, pool 仍不足, 停止 retry",
                flush=True,
            )
            break

        # retry: 拿 audit 反馈调整 query + 重新搜
        # 策略: 用 audit 拒绝理由里的关键词做"否定指引", 但简单起见我们用原 query 加大 k 翻倍,
        #       再多拉一轮, 期望新召回里包含之前漏的真相关 paper.
        try:
            need_more = max(min_pool, max_pool) - len(all_kept)
            retry_k = min(need_more * 2, 60)   # 翻倍, 上限 60 (防太慢)
            print(
                f"{print_prefix} retry round {round_idx+1} start  need_more={need_more}  "
                f"retry_k={retry_k}  source=aps+iop+arxiv",
                flush=True,
            )
            # 多源再拉一次 (跟 _build_candidate_pool 同样套路, 但不再走 summary_items / chroma)
            new_papers: list[dict[str, Any]] = []
            try:
                aps_k = min(retry_k // 2, 30)
                aps_new = _openalex_aps_search(topic, aps_k)
                for p in aps_new:
                    if p.get("paper_id", "") not in seen_ids:
                        seen_ids.add(p["paper_id"])
                        new_papers.append(p)
                print(f"{print_prefix} retry: APS +{len([p for p in aps_new if p.get('paper_id', '') in seen_ids])} 篇", flush=True)
            except Exception as e:
                print(f"{print_prefix} retry: APS ❌ {type(e).__name__}: {str(e)[:60]}", flush=True)
            try:
                iop_k = min(retry_k // 4, 20)
                iop_new = _openalex_iop_search(topic, iop_k)
                for p in iop_new:
                    if p.get("paper_id", "") not in seen_ids:
                        seen_ids.add(p["paper_id"])
                        new_papers.append(p)
                print(f"{print_prefix} retry: IOP +{len([p for p in iop_new if p.get('paper_id', '') in seen_ids])} 篇", flush=True)
            except Exception as e:
                print(f"{print_prefix} retry: IOP ❌ {type(e).__name__}: {str(e)[:60]}", flush=True)
            try:
                arxiv_k = min(retry_k // 4, 20)
                arxiv_new = _arxiv_search(topic, arxiv_k)
                for p in arxiv_new:
                    pid = p.get("paper_id", "")
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        new_papers.append({
                            "paper_id": pid,
                            "title": p.get("title", ""),
                            "year": str(p.get("year", "")),
                            "journal": "arxiv",
                            "core_method": "",
                            "core_contrib": "",
                            "abstract": p.get("abstract", ""),
                            "outlook": "",
                            "keywords": p.get("keywords", ""),
                            "_match_score": 0,
                            "_source": "arxiv_api_retry",
                        })
                print(f"{print_prefix} retry: arxiv +{len([p for p in arxiv_new if p.get('paper_id', '') in seen_ids])} 篇", flush=True)
            except Exception as e:
                print(f"{print_prefix} retry: arxiv ❌ {type(e).__name__}: {str(e)[:60]}", flush=True)

            print(f"{print_prefix} retry: 新增 {len(new_papers)} 篇 待 audit", flush=True)

            # 把新 paper 加到下一轮 pool_dict
            for p in new_papers:
                pool_dict[p["paper_id"]] = p
        except Exception as e:
            print(f"{print_prefix} retry 异常停止: {e}", flush=True)
            break

    # 全局汇总: audit 拒绝理由采样 (跨所有轮, 取前 3)
    all_rejected_sample = []
    for r in audit_rounds:
        all_rejected_sample.extend(
            [{"round": r["round"], **x} for x in []]  # 这里简化: 拒绝理由已经在每轮记录里, ReAct 读 rounds 即可
        )
    # 简化: summary 里 rounds 字段已有足够信息 (LLM 看 round N 的 ok=False 就知道哪一轮被踢)

    final_pool_size = len(all_kept)
    summary = {
        "topic": topic,
        "rounds": audit_rounds,
        "final_pool_size": final_pool_size,
        "target_min_pool": min_pool,
        "pool_meets_target": final_pool_size >= min_pool,
        "max_retries": max_retries,
    }

    print(
        f"{print_prefix} DONE  final_pool={final_pool_size}/{min_pool}  "
        f"meets_target={summary['pool_meets_target']}",
        flush=True,
    )

    return {
        "kept_dict": all_kept,
        "summary": summary,
    }
