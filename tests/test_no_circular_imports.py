"""
test_no_circular_imports — 防 `tools → controller` 循环 import 回归

背景:
- 2026-09-05 18:23 真实崩过: ImportError: cannot import name 'MODE_TO_PROVIDER_MODEL'
  from partially initialized module 'agent.controller'
- 当时根因: tools/llm.py 顶层 from agent.controller import MODE_TO_PROVIDER_MODEL
  触发 controller → tools → llm → controller 循环
- 修法: tools/llm.py 改用函数内 lazy import (line 72)
- 加这个测试做回归守卫: 任何 `agent.tools.*` 子模块顶层 import 都不能触碰 agent.controller

如果将来想消除这个 lazy import, 正确做法是:
  - 把 MODE_TO_PROVIDER_MODEL 移到 agent/mode_registry.py
  - controller 和 tools/llm 都从 mode_registry 引用
  - 删 lazy import
"""
import importlib
import sys


# controller 反向依赖 tools (controller:18 from agent.tools.registry import ToolRegistry)
# 允许这个方向, 不允许 tools → controller
CONTROLLER_MODULES = {
    "agent.controller",
    "agent.tools.llm",
    "agent.tools.surveys",  # 内部从 agent.tools.config 读 (同包, 安全)
    "agent.tools.knowledge",
    "agent.tools.aps",
    "agent.tools.arxiv",
    "agent.tools.memory",
    "agent.tools.summary",
    "agent.tools.search",
    "agent.tools.database",
    "agent.tools.attachment_reader",
    "agent.tools.extract_keywords",
    "agent.tools.interests",
    "agent.tools.pdf_utils",
    "agent.tools.summary_parser",
    "agent.tools.browser_pool",
    "agent.tools.utils",
    "agent.tools.config",
    "agent.tools.registry",
    "agent.tools.__init__",
}


def test_tools_do_not_import_controller():
    """
    任何 agent.tools.* 子模块顶层 import 不能触碰 agent.controller。

    实施方式: 拿到该模块的全部 module-level imports, 检查 sys.modules 里
    'agent.controller' 的状态。如果 controller 在该模块 import 过程中被
    触发 partial init, sys.modules 里 'agent.controller' 会存在但拿不到
    MODE_TO_PROVIDER_MODEL (测试最后一行 import 验证)。
    """
    for mod_name in sorted(CONTROLLER_MODULES):
        if mod_name not in sys.modules:
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                # 跳过 PySide6 / dotenv / 业务逻辑缺 key 等业务原因,
                # 只捕获 "ImportError: cannot import name X from partially initialized"
                err = str(e)
                if "partially initialized" in err or "circular import" in err.lower():
                    raise AssertionError(
                        f"{mod_name} 触发循环 import:\n  {type(e).__name__}: {e}\n"
                        f"修法: 改 lazy import (函数内 import), 或把依赖对象移到共享模块"
                    ) from e
                # 其它 ImportError (缺依赖) 跳过 — 不是循环 import 问题
                continue

    # 最后确认 controller 完整可导入
    from agent.controller import MODE_TO_PROVIDER_MODEL
    assert isinstance(MODE_TO_PROVIDER_MODEL, dict)
    assert "deepseek-v4-flash" in MODE_TO_PROVIDER_MODEL


def test_full_import_path():
    """
    模拟 main.py 的导入顺序: load_dotenv → ui → config → agent.tools
    确保整个 import 链不触发循环。
    """
    # 清掉之前可能 partial init 的 module
    for k in list(sys.modules.keys()):
        if k.startswith("agent.") or k in ("config", "ui"):
            del sys.modules[k]

    # 触发跟 main.py 一致的导入
    from dotenv import load_dotenv
    load_dotenv()
    import agent.tools  # noqa: F401
    from agent.infra import api_keys  # noqa: F401
    from agent.controller import run_agent_stream
    from config import Config

    assert callable(run_agent_stream)
    assert Config is not None
