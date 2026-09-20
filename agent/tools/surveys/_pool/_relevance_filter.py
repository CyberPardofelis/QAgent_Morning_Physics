"""
_pool._relevance_filter — 拆出自原 _pool.py (9-20 体检重构)

包含函数: _relevance_filter_pool
"""
from __future__ import annotations

import logging
import re
from typing import Any

from agent.tools.config import get_relevance_batch_size

from ._utils import _safe_json_loads_array

logger = logging.getLogger(__name__)
def _relevance_filter_pool(
    topic: str,
    pool_dict: dict[str, dict[str, Any]],
    threshold: int,
    strict: bool = False,    # 9-18 加: strict=True 时不 rescue 不抛错, 全 0 返空 dict
) -> dict[str, dict[str, Any]]:
    """
    per-paper LLM 相关性评分精筛 (9-09 user 拍板加).

    9-18 加 strict 参数:
      - strict=False (默认, 向后兼容): 保留 rescue + all_fallback 兜底 (9-09 行为)
      - strict=True: 严格审核, 不 rescue 不抛错, 全 0 返空 dict (给 retry loop 用)
                      9-18 user 拍板: rescue 兜底导致"主题不符也硬撑", 不严格.

    输入: pool_dict (paper_id -> pool entry)
    处理: 5 篇/批 (env 可调) 调 LLM, 每篇 0-10 打分
    输出: pool_dict, score < threshold 的 paper 剔除, 其余加 _relevance_score 字段

    容错 (strict=False):
      - 整批 LLM 失败 → 该批 5 篇保留 (降级, 不剔除)
      - 单篇 JSON parse 失败 → 该篇用 score=5 (中性) 保留
      - 全部过滤后剩 0 篇 → 抛 ValueError, 调用方 fallback 用原 pool

    容错 (strict=True):
      - 跟 strict=False 一致, 但不 rescue 不抛错
      - 全 0 返空 dict, 让调用方 retry 或 bailout
    """
    from agent.prompts_survey import build_relevance_score_prompt
    from agent.tools.llm import _call_llm

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"
    batch_size = get_relevance_batch_size()

    items = list(pool_dict.values())
    if not items:
        return pool_dict

    logger.info(
        "[_relevance_filter_pool] topic=%r, 候选池 %d 篇, batch_size=%d, threshold=%d, strict=%s",
        topic[:50], len(items), batch_size, threshold, strict,
    )

    scored: dict[str, dict[str, Any]] = {}  # paper_id -> {score, reason, is_about_topic, is_adjacent_field}

    # 批量化
    for i in range(0, len(items), batch_size):
        batch = items[i: i + batch_size]
        try:
            prompt = build_relevance_score_prompt(topic, batch)
            out = _call_llm(
                prompt,
                provider=_PROVIDER,
                model=_MODEL,
                enable_thinking=False,
                temperature=0.2,
                max_tokens=2000,    # 5 篇 × ~100 token/篇 + JSON overhead
                timeout=120,
            )
            # 剥 <think>
            if out and "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
            # 容错 parse
            results = _safe_json_loads_array(out or "")
            if results is None:
                raise ValueError(f"relevance batch JSON parse 失败: {(out or '')[:200]}")
            # 写入
            for r in results:
                if not isinstance(r, dict):
                    continue
                pid = r.get("paper_id", "")
                if not pid:
                    continue
                score = r.get("score")
                if not isinstance(score, (int, float)):
                    score = 5  # 中性 fallback
                # 9-09 严判: 提取 is_about_topic (跟 score 冗余但更清晰, 缺则从 score 回填)
                about_raw = r.get("is_about_topic")
                if about_raw is None:
                    is_about = int(score) >= 8
                else:
                    is_about = bool(about_raw)
                scored[pid] = {
                    "score": int(score),
                    "reason": str(r.get("reason", ""))[:200],
                    "is_about_topic": is_about,
                    "is_adjacent_field": bool(r.get("is_adjacent_field", False)),
                }
            # 漏评的 (LLM 没返) 用中性分保留
            for p in batch:
                pid = p.get("paper_id", "")
                if pid and pid not in scored:
                    scored[pid] = {
                        "score": 5, "reason": "[fallback] LLM 未评, 用中性分",
                        "is_about_topic": False, "is_adjacent_field": False,
                    }
        except Exception as e:
            # 整批失败: 整批 5 篇用中性分保留 (降级, 不剔除)
            logger.warning("[_relevance_filter_pool] batch %d 失败 (整批保留): %s", i // batch_size, e)
            for p in batch:
                pid = p.get("paper_id", "")
                if pid and pid not in scored:
                    scored[pid] = {
                        "score": 5, "reason": f"[batch_failed] {type(e).__name__}",
                        "is_about_topic": False, "is_adjacent_field": False,
                    }

    # 把评分写回 pool_dict
    for pid, p in pool_dict.items():
        s = scored.get(pid, {})
        p["_relevance_score"] = s.get("score", 5)
        p["_relevance_reason"] = s.get("reason", "")
        p["_is_about_topic"] = s.get("is_about_topic", False)
        p["_is_adjacent_field"] = s.get("is_adjacent_field", False)

    # 9-09 严判: 双重过滤 — score >= threshold **AND** is_about_topic = True
    # is_about_topic 是 LLM 显式给的二元判断, 比单纯 score 阈值更可靠
    # 邻近子领域 paper (is_about_topic=False) 一律踢掉, 即便 score=7
    kept = {
        pid: p for pid, p in pool_dict.items()
        if p["_relevance_score"] >= threshold and p["_is_about_topic"]
    }

    # 9-18: strict 模式 — 不 rescue 不抛错, 全 0 返空
    if strict:
        if not kept:
            print(
                f"[_relevance_filter_pool] strict 模式: 全部 {len(pool_dict)} 篇被踢光, 返空 dict "
                f"(topic={topic[:50]!r})",
                flush=True,
            )
            return {}
        n_filtered = len(pool_dict) - len(kept)
        print(
            f"[_relevance_filter_pool] strict 模式: 过滤掉 {n_filtered} 篇, 保留 {len(kept)} 篇 "
            f"(topic={topic[:50]!r})",
            flush=True,
        )
        return kept

    # 9-09 优化: 全部 paper 都标 [fallback] / [batch_failed] → LLM 整体失败, 不应用任何过滤
    all_fallback = all(
        p.get("_relevance_reason", "").startswith(("[fallback]", "[batch_failed]"))
        for p in pool_dict.values()
    )
    if all_fallback:
        logger.warning(
            "[_relevance_filter_pool] 所有 paper 都是 LLM fallback/batch_failed 标记, "
            "跳过 relevance 过滤, 原样返回 (LLM 整体不可用, 让 chroma/summary_items/arxiv 兜底)"
        )
        return dict(pool_dict)

    if len(kept) < 5 and len(kept) > 0:
        # 兜底: 取剩余的按 score 倒序, 补足到 5 篇 (避免下游 "候选池为空" 假阳)
        before_rescue = len(kept)
        remaining = sorted(
            (p for pid, p in pool_dict.items() if pid not in kept),
            key=lambda x: (-x["_relevance_score"], not x["_is_about_topic"]),  # 优先 is_about_topic=True
        )
        for p in remaining[:5 - len(kept)]:
            kept[p["paper_id"]] = p
        logger.warning(
            "[_relevance_filter_pool] threshold=%d 严判后剩 %d 篇 < 5, 兜底补足到 %d 篇 (含邻近子领域, 审计时会标 ❌)",
            threshold, before_rescue, len(kept),
        )

    if not kept:
        # 全部 0 分 → 抛错, 调用方 fallback 用原 pool
        raise ValueError("relevance filter 把所有 paper 踢光 (可能主题太偏或 LLM 全失败)")

    n_filtered = len(pool_dict) - len(kept)
    if n_filtered > 0:
        logger.info(
            "[_relevance_filter_pool] 过滤掉 %d 篇 (score<%d 或 is_about_topic=False), 保留 %d 篇",
            n_filtered, threshold, len(kept),
        )

    return kept
