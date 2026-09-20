"""
surveys._actions._dispatch — 拆出自原 _actions.py (9-20 体检重构)

包含函数: _dispatch_action
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)
def _dispatch_action(decision: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """派发 action 到 5 个 handler (9-08 加 write_outline, 拆 write_section 前的结构生成)"""
    action = decision["action"]
    args = decision.get("args", {})
    if action == "search_more":
        yield from action_search_more(args, state, ws)
    elif action == "extract_facts":
        yield from action_extract_facts(args, state, ws)
    elif action == "write_outline":
        yield from action_write_outline(args, state, ws)
    elif action == "write_section":
        yield from action_write_section(args, state, ws)
    elif action == "finish":
        yield from action_finish(args, state, ws)
    else:
        yield f"⚠️ 未知 action: {action!r}\n"
