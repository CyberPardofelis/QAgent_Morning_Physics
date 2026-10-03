"""
_pool._facts_review — 拆出自原 _pool.py (9-20 体检重构)

包含函数: _review_facts_topic_alignment
"""
from __future__ import annotations

import logging
import re
from typing import Any

from .._parallel import run_ordered
from ._utils import _safe_json_loads_array

logger = logging.getLogger(__name__)
def _review_facts_topic_alignment(
    topic: str,
    facts: list[dict[str, Any]],
    threshold: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """
    per-fact 主题对齐审查 (9-09 user 拍板加).

    输入:
      - topic: 综述主题
      - facts: list of fact dict (来自 state["facts"], 每条至少有 fact_id/paper_id/claim)
      - threshold: >= threshold 算主题对齐, < threshold 算邻近子领域

    输出:
      - scored_facts: dict[fact_id] -> {score, is_topic_aligned, reason}
      - stats: {aligned: N, misaligned: M, fallback: K, total: T}

    容错:
      - 整批 LLM 失败 → 该批 8 条用中性分 (score=5, is_topic_aligned=True, 保留)
      - 单条 JSON parse 失败 → 该条用中性分
      - 全部 fallback → 返回 (空 scored, stats with fallback 计数)
    """
    from agent.prompts_survey import build_fact_review_prompt
    from agent.tools.llm import _call_llm

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"
    from agent.tools.config import get_fact_review_batch_size
    batch_size = get_fact_review_batch_size()

    if not facts:
        return {}, {"aligned": 0, "misaligned": 0, "fallback": 0, "total": 0}

    logger.info(
        "[_review_facts_topic_alignment] topic=%r, fact %d 条, batch_size=%d, threshold=%d",
        topic[:50], len(facts), batch_size, threshold,
    )

    scored: dict[str, dict[str, Any]] = {}  # fact_id -> {score, is_topic_aligned, reason}
    stats = {"aligned": 0, "misaligned": 0, "fallback": 0, "total": len(facts)}

    # 用 fact_id 作 key, 如果 fact_id 缺失则用 (paper_id, claim 短 hash) 兜底
    def _fact_key(f: dict[str, Any], idx: int) -> str:
        fid = f.get("fact_id")
        if fid:
            return str(fid)
        pid = f.get("paper_id", "?")
        claim_hash = hash(str(f.get("claim", ""))[:50]) & 0xFFFF
        return f"{pid}:{idx}:{claim_hash}"

    # 10-02 并发: 批次之间互不依赖, worker 只负责"发请求 + 解析出 results"。
    # scored 的写入和 fallback 全部回主线程按原序做 —— 保序保证同一份输入
    # 的输出可复现(诊断日志里的分数分布直方图就是拿 scored 算的)。
    def _review_one(pair):
        idx, batch = pair
        prompt = build_fact_review_prompt(topic, batch)
        out = _call_llm(
            prompt,
            provider=_PROVIDER,
            model=_MODEL,
            enable_thinking=False,
            temperature=0.2,
            # 批大小从 8 提到 20 后, 每条 ~80 token + JSON overhead,
            # 固定 3000 会截断后半批 —— 截断的批整批 fallback 成中性分。
            # 与 _relevance_filter 的 max(4000, 400×n) 同一套做法。
            max_tokens=max(3000, 200 * len(batch)),
            timeout=120,
        )
        if out and "<think>" in out:
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
        results = _safe_json_loads_array(out or "")
        if results is None:
            raise ValueError(f"fact review batch JSON parse 失败: {(out or '')[:200]}")
        return results

    _slices = [(i, facts[i: i + batch_size]) for i in range(0, len(facts), batch_size)]
    _calls = run_ordered(_slices, _review_one, desc="fact_align")

    for idx, batch, res in zip([s[0] for s in _slices],
                                [s[1] for s in _slices], _calls, strict=True):
        batch_keys = [_fact_key(f, idx + j) for j, f in enumerate(batch)]
        if res.ok:
            results = res.value
            # 用 fact_id 匹配; LLM 可能没严格用我们的 fact_id, 用 paper_id + 顺序回退匹配
            for j, r in enumerate(results):
                if not isinstance(r, dict):
                    continue
                # 优先用 r["fact_id"]; 没有则按 batch 顺序回退
                fid = r.get("fact_id") or batch_keys[min(j, len(batch_keys) - 1)]
                score = r.get("score")
                if not isinstance(score, (int, float)):
                    score = 5
                align_raw = r.get("is_topic_aligned")
                is_aligned = int(score) >= threshold if align_raw is None else bool(align_raw)
                scored[fid] = {
                    "score": int(score),
                    "is_topic_aligned": is_aligned,
                    "reason": str(r.get("reason", ""))[:200],
                    "paper_id": r.get("paper_id", batch[min(j, len(batch) - 1)].get("paper_id", "?")),
                }
            # 漏评的用中性分
            for j, f in enumerate(batch):
                fk = batch_keys[j]
                if fk not in scored:
                    scored[fk] = {
                        "score": 5, "is_topic_aligned": True,  # fallback 默认对齐 (保守)
                        "reason": "[fallback] LLM 未评, 用中性分", "paper_id": f.get("paper_id", "?"),
                    }
        else:
            logger.warning("[_review_facts_topic_alignment] batch %d 失败 (整批保留): %s",
                           idx // batch_size, res.exc)
            for j, f in enumerate(batch):
                fk = batch_keys[j]
                if fk not in scored:
                    scored[fk] = {
                        "score": 5, "is_topic_aligned": True,
                        "reason": f"[batch_failed] {type(res.exc).__name__}",
                        "paper_id": f.get("paper_id", "?"),
                    }

    # 统计
    for s in scored.values():
        if s["reason"].startswith(("[fallback]", "[batch_failed]")):
            stats["fallback"] += 1
        elif s["is_topic_aligned"]:
            stats["aligned"] += 1
        else:
            stats["misaligned"] += 1

    return scored, stats
