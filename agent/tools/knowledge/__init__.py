"""
knowledge — 拆分子模块

原 knowledge.py 拆为本目录子文件, 本 __init__.py 重新导出全部保持向后兼容。
模块分布:
- ._analyze.py — ['analyze_paper']
- ._query.py — ['query_knowledge', 'semantic_search', 'semantic_search_keywords']
- ._scan.py — ['scan_local_papers', 'migrate_json_to_db']
- ._index.py — ['index_paper_to_chroma', 'reindex_all', 'sync_recent_to_chroma']
- ._similar.py — ['similar_papers']
- ._utils.py — ['_safe_json_list']
"""

from ._analyze import analyze_paper
from ._index import index_paper_to_chroma, reindex_all, sync_recent_to_chroma
from ._query import query_knowledge, semantic_search, semantic_search_keywords
from ._scan import migrate_json_to_db, scan_local_papers
from ._similar import similar_papers
from ._utils import _safe_json_list

__all__ = [
    "analyze_paper",
    "query_knowledge",
    "semantic_search",
    "semantic_search_keywords",
    "scan_local_papers",
    "migrate_json_to_db",
    "index_paper_to_chroma",
    "reindex_all",
    "sync_recent_to_chroma",
    "similar_papers",
    "_safe_json_list"
]
