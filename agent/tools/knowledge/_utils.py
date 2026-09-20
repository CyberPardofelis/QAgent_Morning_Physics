"""
knowledge._utils — 拆出自原 knowledge.py (9-20 体检重构)

包含函数: _safe_json_list
"""
from __future__ import annotations

import json
import logging

from agent.tools.config import KNOWLEDGE_DIR, PAPER_DIR  # noqa: F401

logger = logging.getLogger(__name__)
def _safe_json_list(s) -> list:
    """解析 papers.db 的 keywords/authors 字段(可能是 JSON list 字符串)

    2026-09-02 P3 修:用顶层 import 的 `json` 名字(之前误用 `_json` 但只在 similar_papers
    函数体局部 import,导致 _safe_json_list 抛 NameError 被静默 except,fallback 返 [s])
    """
    if not s:
        return []
    if isinstance(s, list):
        return [str(x) for x in s]
    s = str(s).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            return [str(x) for x in json.loads(s)]
        except Exception:
            pass
    return [s]
