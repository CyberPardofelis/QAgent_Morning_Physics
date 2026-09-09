"""
agent.tools.surveys — 综述库 (P1.7 重构后, 从原 2129 行 monolith 拆 5 文件)

模块结构:
- _crud          schema + 主题/章节 CRUD + health_check (UI 同步调)
- _pool          候选池构建 + paper_id 引用 + arxiv + relevance filter
- _actions       ReAct helpers + 5 个 action + finalize (LLM-heavy)
- survey_generate @register_tool 综述生成主入口 (L2)
- survey_audit    @register_tool 独立审计 (L0)

向后兼容:
- 旧 `from agent.tools.surveys import list_themes, ...` 仍然能跑 (本 __init__.py 重导出)
- 旧 `from agent.tools.surveys import survey_generate` 仍能跑
- 旧 `import agent.tools.surveys` 仍然触发 init_schema() (跟旧行为一致)
- 旧 symbol 路径 `_build_candidate_pool` / `_decide_action` 等下划线开头的也重导出
  (虽然 9-08 后这些是 _actions / _pool 内部 API, 外部不应用, 但旧测试可能直接 import)

UI 实际只用的 symbol (从 survey_workspace_dialog.py):
  list_themes, list_sections, get_section,
  create_theme, create_section, update_section, delete_section, delete_theme,
  health_check
"""
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 初始化 (模块 import 时建表 — 跟旧行为一致)
# ============================================================
try:
    from ._crud import init_schema
    init_schema()
except Exception as e:
    logger.warning("[surveys] init_schema 失败(非致命, 首次 save 时会重试): %s", e)


# ============================================================
# 重导出: CRUD (UI 用的)
# ============================================================
from ._crud import (
    SCHEMA_SQL,
    init_schema,
    # 主题 CRUD
    create_theme,
    list_themes,
    get_theme,
    update_theme,
    delete_theme,
    # 章节 CRUD
    create_section,
    list_sections,
    get_section,
    update_section,
    delete_section,
    reorder_section,
    # 健康检查
    health_check,
)


# ============================================================
# 重导出: 候选池/引用 (内部 API, 但旧测试可能直接用)
# ============================================================
from ._pool import (
    _safe_json_loads_array,
    _extract_paper_ids,
    _audit_section_paper_ids,
    _build_candidate_pool,
    _relevance_filter_pool,
    _arxiv_search,
    _arxiv_search_batch,
)


# ============================================================
# 重导出: 工具入口 (LLM 工具, @register_tool)
# ============================================================
from .survey_generate import survey_generate
from .survey_audit import survey_audit


# ============================================================
# 重导出: 内部 helpers (旧 import 路径兼容)
# ============================================================
from ._actions import (
    _friendly_error,
    _decide_action,
    _state_driven_decide,
    _dispatch_action,
    _append_react_trace,
    _finalize_survey,
    action_search_more,
    action_extract_facts,
    action_write_outline,
    action_write_section,
    action_finish,
)


__all__ = [
    # 公共 API (UI / 工具调用方用)
    "SCHEMA_SQL", "init_schema",
    "create_theme", "list_themes", "get_theme", "update_theme", "delete_theme",
    "create_section", "list_sections", "get_section", "update_section",
    "delete_section", "reorder_section",
    "health_check",
    "survey_generate", "survey_audit",
    # 内部 API (下划线开头, 旧测试可能直接 import)
    "_safe_json_loads_array", "_extract_paper_ids", "_audit_section_paper_ids",
    "_build_candidate_pool", "_relevance_filter_pool",
    "_arxiv_search", "_arxiv_search_batch",
    "_friendly_error", "_decide_action", "_state_driven_decide",
    "_dispatch_action", "_append_react_trace", "_finalize_survey",
    "action_search_more", "action_extract_facts", "action_write_outline",
    "action_write_section", "action_finish",
]
