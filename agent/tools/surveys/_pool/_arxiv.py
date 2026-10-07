"""
_pool._arxiv — 拆出自原 _pool.py (9-20 体检重构)

包含函数: _arxiv_search, _arxiv_search_batch
"""
from __future__ import annotations

import logging
from typing import Any

from .._paper_id import normalize_arxiv_id
from ._utils import _arxiv_author_name

logger = logging.getLogger(__name__)
def _arxiv_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """arxiv API 抓 k 篇, 返回 dict list."""
    import arxiv
    search = arxiv.Search(
        query=query,
        max_results=k,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    out: list[dict[str, Any]] = []
    for r in client.results(search):
        out.append({
            "paper_id": normalize_arxiv_id(r.entry_id),  # 10-04: 见 _paper_id
            # 10-07 加: 读 arXiv Atom feed 里的 <arxiv:doi>。已发表的论文这一栏
            # 就是期刊版的 DOI, 和 OpenAlex 侧的 doi **逐字相同** —— 于是同一篇
            # 的预印本和期刊版能落到同一个身份键上被折叠, **零额外 API 成本**。
            # 以前没读这个字段, 于是 `2110.03629`(arXiv) 和
            # `10.1103/physreva.107.042403`(OpenAlex) 被当成两篇不同的文献,
            # 各抽一遍 fact, 参考文献里出现重复条目。
            "doi": (getattr(r, "doi", "") or "").strip(),
            "arxiv_id": normalize_arxiv_id(r.entry_id),
            "title": r.title or "",
            "year": str(r.published.year) if r.published else "",
            "authors": ", ".join([_arxiv_author_name(a) for a in (r.authors or [])[:5]]),
            "abstract": r.summary or "",
            "keywords": ", ".join([_arxiv_author_name(a) for a in (r.authors or [])[:3]]),
        })
        if len(out) >= k:
            break
    return out


def _arxiv_search_batch(queries: list[str], k_per_query: int = 3) -> list[dict[str, Any]]:
    """
    多 query 批量 arxiv 召回 (9-09 user 拍板加).

    跟 _arxiv_search 的区别:
      - 单 query → 多 query (list of str)
      - 每 query 拉 k_per_query (默认 3, 避免 5 个 query × 5 = 25 调用过慢)
      - union + dedup by paper_id, 返回合并 list

    使用场景 (9-09 拍板, 未来 wire in):
      - Phase A 召回: 拿 LLM 扩展的多个学术 query (例: "quantum state verification" →
        ["self-testing quantum states", "state verification protocol",
         "device-independent certification", ...]), 一次 batch 召回, 比单 query 召回更聚焦
      - action_search_more: section_gaps 触发 search_more 时, gap queries 是 list, 可直接喂

    现有 _arxiv_search 不动 (单 query 场景仍可用, 跟 history 兼容)
    """
    import arxiv
    if not queries:
        return []
    client = arxiv.Client()
    seen: set = set()
    out: list[dict[str, Any]] = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        try:
            search = arxiv.Search(
                query=q,
                max_results=k_per_query,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for r in client.results(search):
                pid = normalize_arxiv_id(r.entry_id)   # 10-04: 旧格式要留 archive 前缀
                if pid in seen:
                    continue
                seen.add(pid)
                out.append({
                    "paper_id": pid,
                    # 10-07: 同 _arxiv_search, 读出 DOI 才能和 OpenAlex 版折叠
                    "doi": (getattr(r, "doi", "") or "").strip(),
                    "arxiv_id": pid,
                    "title": r.title or "",
                    "year": str(r.published.year) if r.published else "",
                    "authors": ", ".join([a.name for a in (r.authors or [])[:5]]),
                    "abstract": r.summary or "",
                    "keywords": ", ".join([a.name for a in (r.authors or [])[:3]]),
                    "_query_source": q[:80],  # debug 用: 这条 paper 是哪个 query 召回来的
                })
        except Exception as e:
            # 单 query 失败不阻塞整个 batch
            logger.warning("[_arxiv_search_batch] query=%r failed: %s", q[:50], e)
            continue
    logger.info(
        "[_arxiv_search_batch] %d queries × k=%d → union %d 篇 (deduped)",
        len(queries), k_per_query, len(out),
    )
    return out
