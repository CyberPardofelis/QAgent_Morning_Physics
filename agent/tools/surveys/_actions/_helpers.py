"""
surveys._actions._helpers — 拆出自原 _actions.py (9-20 体检重构)

包含函数: _friendly_error, _append_react_trace
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)
def _friendly_error(e: Exception, action: str) -> str:
    """把异常翻译成友好提示 (spec §4). step 1 简单实现, 后续 step 扩展查表."""
    err_type = type(e).__name__
    msg = str(e)[:200]
    return f"{action} 失败 ({err_type}): {msg}"


def _append_react_trace(ws, trace_entry: dict[str, Any]) -> None:
    """写 _react_trace.jsonl (spec §6). 失败不阻塞主流程."""
    try:
        trace_path = ws.path / "_react_trace.jsonl"
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("[survey_generate] _react_trace 写盘失败: %s", e)
