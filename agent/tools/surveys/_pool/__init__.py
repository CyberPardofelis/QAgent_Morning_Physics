"""
_pool — 拆分子模块

原 _pool.py 拆为本目录子文件, 本 __init__.py 重新导出全部保持向后兼容。
模块分布:
- ._utils.py — ['_arxiv_author_name', '_safe_json_loads_array', '_extract_paper_ids', '_audit_section_paper_ids', '_safe_paper_id']
- ._build_pool.py — ['_build_candidate_pool']
- ._relevance_filter.py — ['_relevance_filter_pool']
- ._pool_audit.py — ['_audit_pool_with_retry']
- ._facts_review.py — ['_review_facts_topic_alignment']
- ._arxiv.py — ['_arxiv_search', '_arxiv_search_batch']
- ._openalex.py — ['_openalex_aps_search', '_openalex_iop_search', '_openalex_work_to_pool_entry', '_openalex_aps_search_batch', '_openalex_iop_search_batch']
- ._expand.py — ['ExpandConfig', 'ExpansionResult', 'expand_from_seeds', 'prune', 'select_frontier', 'topic_overlap']
"""

from ._arxiv import _arxiv_search, _arxiv_search_batch
from ._build_pool import _build_candidate_pool
from ._expand import (
    ExpandConfig,
    ExpansionResult,
    expand_from_seeds,
    prune,
    select_frontier,
    topic_overlap,
)
from ._facts_review import _review_facts_topic_alignment
from ._openalex import (
    _openalex_aps_search,
    _openalex_aps_search_batch,
    _openalex_iop_search,
    _openalex_iop_search_batch,
    _openalex_work_to_pool_entry,
)
from ._pool_audit import _audit_pool_with_retry
from ._relevance_filter import _relevance_filter_pool
from ._utils import (
    _PAPER_ID_ARXIV_RE,
    _PAPER_ID_PAPERS_MD_RE,
    # 9-20 加: 拆分时漏了 paper_id regex 常量 re-export
    _PAPER_ID_SIMPLE_RE,
    _arxiv_author_name,
    _audit_section_paper_ids,
    _extract_paper_ids,
    _identity_key,
    _safe_json_loads_array,
    _safe_paper_id,
    dedup_across_sources,
    dedup_existing,
)

__all__ = [
    "_arxiv_author_name",
    "_safe_json_loads_array",
    "_extract_paper_ids",
    "_audit_section_paper_ids",
    "_safe_paper_id",
    # 10-07 加: 跨来源去重(同一篇的 arXiv 预印本 <-> OpenAlex DOI 版)
    "_identity_key",
    "dedup_across_sources",
    "dedup_existing",
    "_PAPER_ID_SIMPLE_RE",
    "_PAPER_ID_ARXIV_RE",
    "_PAPER_ID_PAPERS_MD_RE",
    "_build_candidate_pool",
    "_relevance_filter_pool",
    "_audit_pool_with_retry",
    "_review_facts_topic_alignment",
    "_arxiv_search",
    "_arxiv_search_batch",
    "_openalex_aps_search",
    "_openalex_iop_search",
    "_openalex_work_to_pool_entry",
    "_openalex_aps_search_batch",
    "_openalex_iop_search_batch",
    # 9-30 引文图扩展层
    "ExpandConfig",
    "ExpansionResult",
    "expand_from_seeds",
    "prune",
    "select_frontier",
    "topic_overlap",
]
