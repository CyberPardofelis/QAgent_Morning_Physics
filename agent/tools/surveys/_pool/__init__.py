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
"""

from ._utils import _arxiv_author_name
from ._utils import _safe_json_loads_array
from ._utils import _extract_paper_ids
from ._utils import _audit_section_paper_ids
from ._utils import _safe_paper_id
from ._build_pool import _build_candidate_pool
from ._relevance_filter import _relevance_filter_pool
from ._pool_audit import _audit_pool_with_retry
from ._facts_review import _review_facts_topic_alignment
from ._arxiv import _arxiv_search
from ._arxiv import _arxiv_search_batch
from ._openalex import _openalex_aps_search
from ._openalex import _openalex_iop_search
from ._openalex import _openalex_work_to_pool_entry
from ._openalex import _openalex_aps_search_batch
from ._openalex import _openalex_iop_search_batch

__all__ = [
    "_arxiv_author_name",
    "_safe_json_loads_array",
    "_extract_paper_ids",
    "_audit_section_paper_ids",
    "_safe_paper_id",
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
    "_openalex_iop_search_batch"
]
