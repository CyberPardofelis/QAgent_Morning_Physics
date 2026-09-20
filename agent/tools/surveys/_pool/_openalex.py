"""
_pool._openalex — 拆出自原 _pool.py (9-20 体检重构)

包含函数: _openalex_aps_search, _openalex_iop_search, _openalex_work_to_pool_entry, _openalex_aps_search_batch, _openalex_iop_search_batch
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)
def _openalex_aps_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    APS 期刊搜 k 篇, 走 OpenAlex (跟 search_aps 同源, 但返 dict 列表而非 markdown).

    默认 quant-ph 主阵地: PRL + PRA + PRX Quantum + PRX + RMP.
    """
    from agent.tools.openalex import (
        openalex_search as _oa_search,
        resolve_source_ids as _oa_resolve,
        OPENALEX_APS_SOURCES as _OA_APS,
        OPENALEX_APS_DEFAULT_QUANT_PH as _OA_APS_DEFAULT,
    )
    source_ids = _oa_resolve(_OA_APS_DEFAULT, _OA_APS)
    if not source_ids:
        return []
    works = _oa_search(query=query, source_ids=source_ids, max_results=k, caller="survey_aps_recall")
    return [_openalex_work_to_pool_entry(w, "openalex_aps_api") for w in works]


def _openalex_iop_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    IOP 期刊搜 k 篇, 走 OpenAlex (跟 search_iop 同源, 但返 dict 列表而非 markdown).

    默认 quant-ph 主阵地: NJP + J. Phys. A + J. Phys. B + J. Opt.
    """
    from agent.tools.openalex import (
        openalex_search as _oa_search,
        resolve_source_ids as _oa_resolve,
        OPENALEX_IOP_SOURCES as _OA_IOP,
        OPENALEX_IOP_DEFAULT_QUANT_PH as _OA_IOP_DEFAULT,
    )
    source_ids = _oa_resolve(_OA_IOP_DEFAULT, _OA_IOP)
    if not source_ids:
        return []
    works = _oa_search(query=query, source_ids=source_ids, max_results=k, caller="survey_iop_recall")
    return [_openalex_work_to_pool_entry(w, "openalex_iop_api") for w in works]


def _openalex_work_to_pool_entry(work: dict, source_tag: str) -> Dict[str, Any]:
    """
    把 OpenAlex work 转成 pool entry 格式 (跟 arxiv fallback / summary_items 同构).

    跟 _arxiv_search 返回 dict 的 keys 对齐:
      paper_id / title / year / authors / abstract / keywords / _source

    跟 pool entry 的 keys 对齐 (含 _match_score):
      paper_id / title / year / journal / core_method / core_contrib / abstract /
      outlook / keywords / _match_score / _source

    关键差异:
      - paper_id 优先 DOI (稳定, 跟 arxiv ID 不冲突)
      - journal 标 'APS' / 'IOP' / 实际期刊名
      - authors 从 authorships[].author.display_name 抽, 跟 arxiv 一致
      - abstract 从 abstract_inverted_index 重构成字符串 (跟 OpenAlex 工具函数同)
    """
    from agent.tools.openalex import reconstruct_abstract as _oa_reconstruct
    from agent.tools.openalex import _format_authors as _oa_format_authors

    paper_id = _safe_paper_id(work)
    title = work.get("title") or work.get("display_name") or ""

    # 期刊名 (从 primary_location.source.display_name)
    primary_loc = work.get("primary_location") or {}
    src = primary_loc.get("source") or {}
    journal_name = src.get("display_name") or ""

    # 作者 (跟 arxiv 保持一致取 5 个)
    authors_str = _oa_format_authors(work.get("authorships") or [], max_show=5)

    # abstract 重组 (inverted index → 字符串)
    abstract = _oa_reconstruct(work.get("abstract_inverted_index")) or ""

    return {
        "paper_id": paper_id,
        "title": title,
        "year": str(work.get("publication_year") or ""),
        "journal": journal_name,
        "core_method": "",
        "core_contrib": "",
        "abstract": abstract,
        "outlook": "",
        "keywords": authors_str,    # 跟 arxiv fallback 同, 用 authors 当 keywords 占位
        "_match_score": 0,
        "_source": source_tag,
    }


def _openalex_aps_search_batch(queries: List[str], k_per_query: int = 3) -> List[Dict[str, Any]]:
    """
    多 query 批量 APS 召回 (跟 _arxiv_search_batch 同构, 给 search_more 用).
    """
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        try:
            papers = _openalex_aps_search(q, k_per_query)
            for p in papers:
                pid = p.get("paper_id", "")
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                p["_query_source"] = q[:80]
                out.append(p)
        except Exception as e:
            logger.warning("[_openalex_aps_search_batch] query=%r failed: %s", q[:50], e)
            continue
    logger.info(
        "[_openalex_aps_search_batch] %d queries × k=%d → union %d 篇 (deduped)",
        len(queries), k_per_query, len(out),
    )
    return out


def _openalex_iop_search_batch(queries: List[str], k_per_query: int = 3) -> List[Dict[str, Any]]:
    """
    多 query 批量 IOP 召回 (跟 _arxiv_search_batch 同构, 给 search_more 用).
    """
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        try:
            papers = _openalex_iop_search(q, k_per_query)
            for p in papers:
                pid = p.get("paper_id", "")
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                p["_query_source"] = q[:80]
                out.append(p)
        except Exception as e:
            logger.warning("[_openalex_iop_search_batch] query=%r failed: %s", q[:50], e)
            continue
    logger.info(
        "[_openalex_iop_search_batch] %d queries × k=%d → union %d 篇 (deduped)",
        len(queries), k_per_query, len(out),
    )
    return out
