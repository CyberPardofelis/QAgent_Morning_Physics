"""
surveys._actions — ReAct 5 action + decision + finalize (拆分子模块)

原始 _actions.py (1143 行) 在 9-20 拆分为本目录 8 个子文件,
本 __init__.py 重新导出全部,保持向后兼容:
    from agent.tools.surveys._actions import action_write_outline  # OK

模块分布:
- ._helpers.py — ['_friendly_error', '_append_react_trace']
- ._decide.py — ['_decide_action', '_state_driven_decide']
- ._dispatch.py — ['_dispatch_action']
- ._search_more.py — ['action_search_more']
- ._extract_facts.py — ['action_extract_facts']
- ._write_outline.py — ['action_write_outline']
- ._write_section.py — ['action_write_section']
- ._finish.py — ['action_finish']
- ._finalize.py — ['_finalize_survey']
"""

from ._decide import _decide_action, _state_driven_decide
from ._dispatch import _dispatch_action
from ._extract_facts import action_extract_facts
from ._finalize import _finalize_survey
from ._finish import action_finish
from ._helpers import _append_react_trace, _friendly_error
from ._search_more import action_search_more
from ._write_outline import action_write_outline
from ._write_section import action_write_section

__all__ = [
    "_friendly_error",
    "_append_react_trace",
    "_decide_action",
    "_state_driven_decide",
    "_dispatch_action",
    "action_search_more",
    "action_extract_facts",
    "action_write_outline",
    "action_write_section",
    "action_finish",
    "_finalize_survey"
]
