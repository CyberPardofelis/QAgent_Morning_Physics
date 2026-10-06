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
# 2.5 保护用户真实设置 (10-06 加)
# ============================================================
# 背景: tests/test_cost_guard.py 跑一次就把用户真实的 LLM 日计数器清零。
# 实测 4242 → 1(不是 0, 因为最后一个用例自己 increment 了一次)。
#
# **原以为有隔离, 其实从来没有**(踩坑记录):
#   `_isolated_cost_guard` 里那套 `QSettings.setDefaultFormat(IniFormat)` +
#   `setPath(IniFormat, UserScope, tmpdir)` 是**空操作**。Qt 的
#   `QSettings(org, app)` 双参构造**写死用 NativeFormat**, 不看
#   `setDefaultFormat()` —— 后者只对无参 `QSettings()` 生效。实测:
#       setDefaultFormat 之后 defaultFormat = IniFormat
#       但 get_settings().fileName() 仍是 \HKEY_CURRENT_USER\Software\...
#   所以 `guard.reset()` 一直直接写 Windows 注册表, 而且是**静默**写坏:
#   测试全绿, 用户的额度显示归零, 已花掉的次数再也看不到了。
#
# 修法: 不去改 `get_settings()`(那是生产代码, 两参构造改格式会影响真实用户),
# 而是在这里对**每个测试**快照/还原这两个键。测试可以随便改, 退出时原样写回。
@pytest.fixture(autouse=True)
def _protect_user_settings():
    """快照并还原用户真实设置里最容易被测试写坏的那几个键。

    目前只管 cost_guard 的两个键 —— 实测只有它们会被测试写坏。
    新增测试如果开始写别的用户设置, **必须在这里加键**, 否则同样会静默
    污染用户设置(这是本条存在的意义: 让"测试污染用户数据"变成一件
    在 conftest 里看得见的事, 而不是散落在各个测试文件里的隐形副作用)。
    """
    from agent.cost_guard import CostGuard
    from config.settings import get_settings

    guard_keys = (CostGuard.KEY_DATE, CostGuard.KEY_COUNT)
    s = get_settings()
    saved = {k: s.value(k, None) for k in guard_keys}
    try:
        yield
    finally:
        s = get_settings()          # 重新拿, 避免复用测试里被换掉的实例
        for k, v in saved.items():
            if v is None:
                s.remove(k)
            else:
                s.setValue(k, v)
        s.sync()                     # 不 sync 的话跨进程读到的还是旧值


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
