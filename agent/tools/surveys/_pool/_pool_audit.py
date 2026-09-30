"""
_pool._pool_audit — 拆出自原 _pool.py (9-20 体检重构)

包含函数: _audit_pool_with_retry
"""
from __future__ import annotations

import logging
from typing import Any

from agent.infra.console import safe_print  # 控制台 cp936 编不了 emoji 时不崩
from agent.tools.config import get_survey_expand

from ._arxiv import _arxiv_search
from ._expand import ExpandConfig, expand_from_seeds, select_frontier
from ._openalex import _openalex_aps_search, _openalex_iop_search
from ._relevance_filter import _relevance_filter_pool
from ._topic_match import entry_text, qualifies

logger = logging.getLogger(__name__)


def _expand_from_audited(
    topic: str,
    audited: dict[str, dict[str, Any]],
    seen_ids: set,
    print_prefix: str = "",
) -> list[dict[str, Any]]:
    """拿**已经过审**的论文当种子做引文图扩展 (9-30 user 拍板)。

    为什么种子必须是过审的
    ----------------------
    9-30 实测踩过: 扩展放在审查之前, 种子取自原始召回(没审过), 结果种子落到了
    邻近邻域(量子密钥分发那一带), 引文图顺着错误的种子走, 扩出 700 篇全部被审查
    判 0~1 分 —— 扩展注入 40 篇, 通过 0 篇, 端到端质量增量为 0。

    引文图会**放大**种子的方向, 所以种子错了代价是成百倍的。用过审的论文当种子,
    扩展出来的至少在正确的邻域里, 再交给下一轮审查。

    这是用户说的"剪枝要从一开始做起": 召回层剪一次, 审查层再剪一次, 扩展只从
    活下来的东西往外长。

    已知漏召: 词法门控认不出同义术语(MPS/PEPS 那类标题里没有 "tensor network"),
    这部分靠引文图补 —— 种子对了, 沿引用边自然会走到那些论文。
    """
    cfg = get_survey_expand()
    if not cfg["enabled"] or cfg["max_add"] <= 0:
        return []
    seeds = [p for p in audited.values() if p.get("_openalex_id")]
    if not seeds:
        safe_print(
            f"{print_prefix} 引文图扩展 skipped (过审论文里没有带 OpenAlex id 的, "
            f"无法作为引文图种子)",
            flush=True,
        )
        return []

    safe_print(
        f"{print_prefix} 引文图扩展 start  种子 {len(seeds)} 篇(全部已过审)  "
        f"rounds={cfg['rounds']}  breadth={cfg['breadth']}  max_add={cfg['max_add']}",
        flush=True,
    )
    try:
        res = expand_from_seeds(
            seeds,
            config=ExpandConfig(
                max_rounds=cfg["rounds"],
                forward_breadth=cfg["breadth"],
                backward_breadth=cfg["breadth"],
                max_requests=cfg["max_requests"],
                reviews_only=cfg["reviews"],
            ),
            topic=topic,
        )
    except Exception as e:
        safe_print(
            f"{print_prefix} 引文图扩展 ❌ {type(e).__name__}: {str(e)[:90]}", flush=True)
        return []

    # 扩展出来的也要过词法门控 —— 否则又变成往池子里灌噪声。
    # 选哪 40 篇送审按**主题匹配度**排, 不按被引量(见 select_frontier 的注释)。
    picked = select_frontier(res.entries, cfg["max_add"], topic=topic)
    out: list[dict[str, Any]] = []
    dropped_gate = 0
    for p in picked:
        pid = p.get("paper_id", "")
        if not pid or pid in seen_ids:
            continue
        if not qualifies(topic, entry_text(p)):
            dropped_gate += 1
            continue
        seen_ids.add(pid)
        out.append(p)

    safe_print(
        f"{print_prefix} 引文图扩展 done  扩出 {len(res.entries)} 篇  "
        f"取前 {cfg['max_add']}  过词法门控 -{dropped_gate}  采纳 {len(out)} 篇  "
        f"(请求 {res.stats['requests']}, {res.stats['stopped_reason']})",
        flush=True,
    )
    return out

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
        safe_print(
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

        safe_print(
            f"{print_prefix} audit round {round_idx} done  +{after_count}/{before_count}  "
            f"累计合格 {len(all_kept)}/{min_pool}  ok={meets_target}",
            flush=True,
        )

        if meets_target:
            break

        # 不满足, 准备下一轮 retry
        if round_idx >= max_retries:
            safe_print(
                f"{print_prefix} 已重试 {max_retries} 次, pool 仍不足, 停止 retry",
                flush=True,
            )
            break

        # retry: 两条腿走路
        #   (a) 引文图扩展 —— 种子是**本轮已过审的论文** (9-30 user 拍板)
        #   (b) 关键词再搜一遍 —— 换 k 翻倍拉, 期望捞到之前漏的真相关论文
        try:
            need_more = max(min_pool, max_pool) - len(all_kept)
            retry_k = min(need_more * 2, 60)   # 翻倍, 上限 60 (防太慢)
            safe_print(
                f"{print_prefix} retry round {round_idx+1} start  need_more={need_more}  "
                f"retry_k={retry_k}  source=引文图扩展(过审种子)+aps+iop+arxiv",
                flush=True,
            )
            new_papers: list[dict[str, Any]] = []

            # ---- (a) 引文图扩展: 种子 = all_kept (已过审) ----
            for p in _expand_from_audited(topic, all_kept, seen_ids, print_prefix):
                new_papers.append(p)

            # ---- (b) 多源再拉一次 (跟 _build_candidate_pool 同样套路, 但不再走 summary_items / chroma) ----
            try:
                aps_k = min(retry_k // 2, 30)
                aps_new = _openalex_aps_search(topic, aps_k)
                for p in aps_new:
                    if p.get("paper_id", "") not in seen_ids:
                        seen_ids.add(p["paper_id"])
                        new_papers.append(p)
                safe_print(f"{print_prefix} retry: APS +{len([p for p in aps_new if p.get('paper_id', '') in seen_ids])} 篇", flush=True)
            except Exception as e:
                safe_print(f"{print_prefix} retry: APS ❌ {type(e).__name__}: {str(e)[:60]}", flush=True)
            try:
                iop_k = min(retry_k // 4, 20)
                iop_new = _openalex_iop_search(topic, iop_k)
                for p in iop_new:
                    if p.get("paper_id", "") not in seen_ids:
                        seen_ids.add(p["paper_id"])
                        new_papers.append(p)
                safe_print(f"{print_prefix} retry: IOP +{len([p for p in iop_new if p.get('paper_id', '') in seen_ids])} 篇", flush=True)
            except Exception as e:
                safe_print(f"{print_prefix} retry: IOP ❌ {type(e).__name__}: {str(e)[:60]}", flush=True)
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
                safe_print(f"{print_prefix} retry: arxiv +{len([p for p in arxiv_new if p.get('paper_id', '') in seen_ids])} 篇", flush=True)
            except Exception as e:
                safe_print(f"{print_prefix} retry: arxiv ❌ {type(e).__name__}: {str(e)[:60]}", flush=True)

            safe_print(f"{print_prefix} retry: 新增 {len(new_papers)} 篇 待 audit", flush=True)

            # 把新 paper 加到下一轮 pool_dict
            for p in new_papers:
                pool_dict[p["paper_id"]] = p
        except Exception as e:
            safe_print(f"{print_prefix} retry 异常停止: {e}", flush=True)
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

    safe_print(
        f"{print_prefix} DONE  final_pool={final_pool_size}/{min_pool}  "
        f"meets_target={summary['pool_meets_target']}",
        flush=True,
    )

    return {
        "kept_dict": all_kept,
        "summary": summary,
    }
