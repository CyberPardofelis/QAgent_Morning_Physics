"""
tests/conftest.py — QAgent 测试根 conftest (9-20 体检重构加)

职责:
1. 检测当前环境是否有可用的 QApplication (PySide6 GUI)
2. 没有时给带 @pytest.mark.ui 的测试自动 skip
3. 否则保持原行为

背景:
- 之前没有 conftest,UI 测试 (test_v7_*.py) 在 collection 阶段 import
  formula_renderer,formula_renderer 模块级创建 QWebEngineView 实例
- 无 GUI 环境 (Windows Server / CI / 容器) → Qt WebEngine 段错误 0xC0000005
- pytest 默认 collection abort,整轮测试直接挂掉
- 9-20 加这个 conftest + 给需要 QApplication 的测试加 @pytest.mark.ui

用法:
    @pytest.mark.ui
    def test_formula_renderer_renders_katex():
        ...
"""
import os
import sys

import pytest


# ============================================================
# 1. 检测 QApplication 是否能创建
# ============================================================
def _can_create_qapplication() -> bool:
    """尝试创建 QApplication 实例, 失败说明环境无 GUI"""
    try:
        # offscreen 平台插件让 Qt 不需要真实显示器 (CI / 容器友好)
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        return app is not None
    except Exception:
        return False


_QT_AVAILABLE = _can_create_qapplication()


# ============================================================
# 2. skip fixture: @pytest.mark.ui 测试在无 GUI 环境自动 skip
# ============================================================
@pytest.fixture(autouse=True)
def _skip_ui_without_qt(request):
    """
    autouse: 每个测试函数自动调用
    检查最近一个 marker 是不是 'ui', 如果是 + 当前环境没 QApplication → skip
    """
    marker = request.node.get_closest_marker("ui")
    if marker is not None and not _QT_AVAILABLE:
        pytest.skip(
            "@pytest.mark.ui 测试需要 QApplication, 当前环境无 GUI "
            "(PySide6/Qt 平台插件初始化失败)。设置 QT_QPA_PLATFORM=offscreen "
            "或跑在有显示器的环境。"
        )


# ============================================================
# 3. 显式 pytest 配置 (跟 pytest.ini 互补)
# ============================================================
def pytest_collection_modifyitems(config, items):
    """
    自动给以下测试加 @pytest.mark.ui marker:
    1. 路径含 ui/ (UI 模块测试)
    2. 文件名是 test_v*.py (UI smoke test, 历史版本: v4/v5/v6/v7)
    3. test_stream_render.py (流式渲染 UI 测试)
    4. test_no_circular_imports.py 跳过 (非 UI)

    这样 conftest 的 _skip_ui_without_qt fixture 自动 skip-on-no-qt
    """
    for item in items:
        path = str(item.fspath).replace("\\", "/")
        is_ui = False

        # 规则 1: 路径含 ui/
        if "/ui/" in path or path.endswith("/ui"):
            is_ui = True

        # 规则 2: test_v*.py (UI smoke 历史版本)
        fname = path.rsplit("/", 1)[-1]
        if fname.startswith("test_v") and fname.endswith(".py"):
            is_ui = True

        # 规则 3: 特定已知 UI 测试
        if fname in ("test_stream_render.py",):
            is_ui = True

        # 例外: 纯逻辑测试不加 ui marker
        if fname in ("test_no_circular_imports.py",):
            is_ui = False

        if is_ui and "ui" not in [m.name for m in item.iter_markers()]:
            item.add_marker(pytest.mark.ui)
