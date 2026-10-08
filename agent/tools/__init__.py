"""
agent.tools — 工具集合（按子模块组织，注册到 ToolRegistry）

通过 import 这个包，所有 @register_tool 装饰的函数都会自动注册到 ToolRegistry。
外部 import 兼容：from agent.tools import scan_local_papers, generate_daily_summary_arxiv ...
"""

# ============================================================
# 显式导入每个工具子模块 → 触发 @register_tool 装饰器
# 同时把内部函数（get_paper_by_id / update_paper_llm_summary / update_full_paper）
# 显式 re-export 到 agent.tools，让 `from agent.tools import get_paper_by_id` 仍能工作
# ============================================================
# 这些 import 必须按以下顺序（被依赖的在前）
# knowledge 模块：触发 @register_tool
# 2026-09-09 P0.6: IOP 工具集 (新文件)
# 9-08 20:08 user 拍板: 加 surveys import, 触发 @register_tool(survey_generate, survey_audit)
# (之前 __init__.py 漏 import surveys, 装饰器没跑, survey_generate 一直不在 LLM 工具集里)
from . import (
    config,  # 路径/常量（不注册工具）
    iop,
    knowledge,
    llm,  # _call_llm 内部使用（不注册工具）
    pdf_utils,  # extract_text_from_pdf（不注册工具）
    # 2026-10-08 P0 加: P7 科研闭环。必须 import 才会跑 @register_tool,
    # 但这批工具 exposed=False, 所以只是注册进 registry、不会出现在
    # list_schemas() 里, LLM 看不见。评估集和测试走 ToolRegistry.get() 直接调。
    research,
    surveys,
)
from .aps import (
    download_aps_paper,
    fetch_aps_page_via_browser,
    generate_aps_daily_summary,
    parse_aps_papers_from_html,
    search_aps,
)

# arxiv / aps / utils 模块
from .arxiv import (
    analyze_today_topic_arxiv,
    download_arxiv_paper,
    generate_daily_summary_arxiv,
    search_arxiv,
)

# attachment_reader / search 模块
from .attachment_reader import (
    FILE_DIALOG_FILTER,
    MAX_CHARS,
    SUPPORTED_EXTS,
    compose_llm_user_content,
    extract_text,
    is_supported,
)

# database 模块：显式 import LLM 工具 + 内部函数
from .database import (
    get_paper_by_id,
    get_recent_papers,
    get_recent_papers_raw,
    init_db,  # UI 内部使用
    update_full_paper,  # UI 内部使用
    update_paper_comment,  # @register_tool
    update_paper_llm_summary,
)
from .iop import search_iop
from .knowledge import analyze_paper, migrate_json_to_db, query_knowledge, scan_local_papers

# llm / pdf_utils 内部接口
from .llm import _call_llm
from .memory import recall_memories, save_memory  # 通用记忆库工具
from .pdf_utils import chunk_text, extract_text_from_pdf
from .search import web_search
from .utils import (  # run_python 2026-09-04 停用,函数仍在 utils.py 但不导入(避免误用)
    JOURNAL_CONFIG,
    load_journal_config,
)

# knowledge 模块的 JINA/etc 常量（不重新 import，配置在 knowledge.py 里）
# 兼容旧代码 from agent.tools.knowledge import ...
# 上面已经 `from . import knowledge` + `from .knowledge import ...`，属性都能访问

# interests / interests.py 暂不注册到 LLM 工具（它是内部 API）
# from . import interests      # InterestModel（内部数据模型）


# ============================================================
# 显式 __all__：让 from agent.tools import * 行为可预期
# ============================================================
__all__ = [
    # 路径/配置
    "PAPER_DIR", "KNOWLEDGE_DIR", "INDEX_FILE", "VECTOR_DIR", "DB_PATH",
    "JOURNAL_CONFIG",
    "load_journal_config",
    "EMBEDDING_ENABLED", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL",
    # 知识库 / 论文
    "init_db", "get_paper_by_id", "update_paper_comment", "get_recent_papers",
    "get_recent_papers_raw", "update_paper_llm_summary", "update_full_paper",
    "analyze_paper", "query_knowledge", "scan_local_papers", "migrate_json_to_db",
    # arXiv / APS / IOP
    "search_arxiv", "download_arxiv_paper", "generate_daily_summary_arxiv",
    "analyze_today_topic_arxiv",
    "generate_aps_daily_summary", "download_aps_paper", "search_aps",
    "fetch_aps_page_via_browser", "parse_aps_papers_from_html",
    "search_iop",
    # 工具(2026-09-04:run_python 停用,不导出)
    # 搜索
    "web_search",
    # 记忆库
    "save_memory", "recall_memories",
    # 附件（UI 侧用）
    "extract_text", "compose_llm_user_content", "is_supported",
    "SUPPORTED_EXTS", "FILE_DIALOG_FILTER", "MAX_CHARS",
    # 内部
    "_call_llm", "extract_text_from_pdf", "chunk_text",
]


# ============================================================
# 包初始化：打印已注册的工具（debug 友好）
# ============================================================
def _log_registered_tools() -> None:
    """打印当前已注册的工具列表（用于启动调试）"""
    try:
        from agent.tools.registry import ToolRegistry
        names = ToolRegistry.list_names()
        # 只在非空时打印，避免干扰
        if names and len(names) > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[agent.tools] 注册了 {len(names)} 个 LLM 工具: {sorted(names)}")
    except Exception:
        # 避免初始化时任何异常影响主流程
        pass


_log_registered_tools()
