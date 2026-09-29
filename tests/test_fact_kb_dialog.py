"""
tests/test_fact_kb_dialog.py — 信息知识库 UI

2026-09-29 加。

用户明确要求(这是验收点, 不是风格偏好):
    **不要强制前置, 不要禁止切换到其他窗口。**
    即: 主对话框非模态, 新建/编辑用独立子对话框而不是 exec() 阻塞。

所以本测试重点不是"UI 好不好看", 而是**窗口模型**:
    - FactKBDialog.setModal(False) / Qt.NonModal
    - FactEditDialog 非模态, 且代码里没有 exec()
    - open_fact_kb 走 show() 而不是 exec()
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agent.tools import facts as F

REPO = Path(__file__).resolve().parent.parent
UI = REPO / "ui" / "fact_kb_dialog.py"
WIN = REPO / "ui" / "chat_window.py"


def _src():
    return UI.read_text(encoding="utf-8")


# ============================================================
# 窗口模型: 非模态(用户明确要求)
# ============================================================

def test_main_dialog_is_non_modal():
    from ui.fact_kb_dialog import FactKBDialog
    import inspect as _i
    src = _i.getsource(FactKBDialog.__init__)
    assert "setModal(False)" in src, "主对话框必须非模态"
    assert "Qt.NonModal" in src


def test_edit_dialog_is_non_modal():
    """编辑框也是独立非模态窗口, 不是 exec() 阻塞。"""
    src = _src()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "FactEditDialog"
    )
    init = next(
        n for n in fn.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    calls = [
        ast.unparse(n) for n in ast.walk(init)
        if isinstance(n, ast.Call)
    ]
    # 不能有 setModal(True) / WindowModal
    assert not any("setModal(True)" in c for c in calls), "编辑框不能是模态的"
    assert not any("WindowModal" in c for c in calls)


def test_no_exec_calls_in_fact_kb():
    """
    回归护栏: 整个文件里不能出现 .exec() —— 那是阻塞式模态调用。
    """
    src = _src()
    assert ".exec()" not in src, "不要用 exec() (阻塞且模态)"
    assert ".exec_()" not in src


def test_no_raise_or_activate():
    """不能强制前置窗口。"""
    src = _src()
    for bad in (".raise_()", ".activateWindow()", ".setActiveWindow("):
        assert bad not in src, f"不要用 {bad} 强制前置"


# ============================================================
# 菜单入口
# ============================================================

def test_menu_entry_exists():
    src = WIN.read_text(encoding="utf-8")
    assert 'QAction("信息知识库"' in src, "知识库菜单里要有'信息知识库'"
    assert "open_fact_kb" in src


def test_menu_entry_after_paper_kb_and_before_survey():
    """按用户描述"和论文摘要知识库之类的类似", 放在论文分析库之后。"""
    src = WIN.read_text(encoding="utf-8")
    i_paper = src.index('QAction("论文分析知识库"')
    i_fact = src.index('QAction("信息知识库"')
    i_survey = src.index('QAction("论文综述库"')
    assert i_paper < i_fact < i_survey


def test_opener_keeps_reference_and_shows_non_modal():
    """照现有模式: 存到 self 上保活(防 GC), 且用 show() 非模态。"""
    src = WIN.read_text(encoding="utf-8")
    i = src.index("def open_fact_kb")
    body = src[i:i + 900]
    assert "self._fact_kb_dialog = dialog" in body, "必须存到 self 上, 否则 show() 后被 GC 释放"
    assert "dialog.show()" in body
    assert "exec()" not in body
    assert "NonModal" in body


# ============================================================
# CRUD 覆盖
# ============================================================

def test_crud_actions_present():
    src = _src()
    for fn in ("_on_new", "_on_edit", "_on_add_source", "_on_delete"):
        assert f"def {fn}" in src, f"缺少 {fn}"


def test_uses_facts_module():
    src = _src()
    assert "from agent.tools import facts as F" in src
    for api in ("F.list_facts", "F.get_fact", "F.add_fact", "F.stats"):
        assert api in src, f"应调用 {api}"


def test_shows_multi_source_count():
    """多出处是 fact 库的核心价值, 列表和详情都要能看到。"""
    src = _src()
    assert "n_sources" in src
    assert "出处" in src


def test_detail_shows_grounded_warning():
    """未通过落地审查的出处要标出来。"""
    src = _src()
    assert "grounded" in src
    assert "未通过落地审查" in src


def test_filters_available():
    src = _src()
    assert "fact_type" in src and "source_type" in src and "subject" in src
    assert "仅多出处" in src


# ============================================================
# 真正跑一遍 CRUD (用隔离的库)
# ============================================================
# 标记: ui


@pytest.fixture
def kb(tmp_path, monkeypatch):
    from agent.infra import db as infra_db
    monkeypatch.setattr(infra_db, "_data_root", lambda: tmp_path)
    F.init_schema()
    return F


@pytest.fixture(scope="module")
def qapp():
    """UI 测试要一个 QApplication。conftest 里只有 skip 用的探测, 没有实例。"""
    import os
    import sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    return app


@pytest.mark.ui
def test_kb_dialog_lists_facts(kb, qapp):
    from ui.fact_kb_dialog import FactKBDialog
    for i in range(3):
        kb.add_fact(f"测试事实 {i}", "result", subject=f"对象{i}", source_id=f"P{i}")
    d = FactKBDialog()
    try:
        assert d.table.rowCount() == 3
        assert "3" in d.stats_label.text()
    finally:
        d.close()
        d.deleteLater()


@pytest.mark.ui
def test_kb_search_filters(kb, qapp):
    from ui.fact_kb_dialog import FactKBDialog
    kb.add_fact("苹果相关的事实", "result", subject="水果", source_id="P1")
    kb.add_fact("香蕉相关的事实", "result", subject="水果", source_id="P2")
    d = FactKBDialog()
    try:
        d.search_edit.setText("苹果")
        d._do_search()
        assert d.table.rowCount() == 1
    finally:
        d.close()
        d.deleteLater()


@pytest.mark.ui
def test_kb_detail_shows_all_sources(kb, qapp):
    from ui.fact_kb_dialog import FactKBDialog
    kb.add_fact("多来源命题", "result", subject="X", source_id="A")
    kb.add_fact("多来源命题", "result", subject="X", source_id="B")
    d = FactKBDialog()
    try:
        d.table.selectRow(0)
        txt = d.detail.toPlainText()
        assert "出处 (2 个)" in txt
        assert "A" in txt and "B" in txt
    finally:
        d.close()
        d.deleteLater()


@pytest.mark.ui
def test_kb_delete_removes_fact(kb, qapp, monkeypatch):
    from ui.fact_kb_dialog import FactKBDialog
    kb.add_fact("待删除", "result", source_id="A")
    d = FactKBDialog()
    try:
        d.table.selectRow(0)
        monkeypatch.setattr(
            "ui.fact_kb_dialog.QMessageBox.question", lambda *a, **k: QMessageBox.Yes
        )
        from PySide6.QtWidgets import QMessageBox
        d._on_delete()
        assert len(kb.list_facts()) == 0
    finally:
        d.close()
        d.deleteLater()


@pytest.mark.ui
def test_edit_dialog_validates_required_fields(qapp, monkeypatch):
    """内容与来源都不能为空 —— fact 必须自足且可溯源。"""
    from PySide6.QtWidgets import QMessageBox
    from ui.fact_kb_dialog import FactEditDialog
    warned = []
    monkeypatch.setattr(
        "ui.fact_kb_dialog.QMessageBox.warning",
        lambda self, t, m: warned.append(m) or QMessageBox.Ok,
    )
    d = FactEditDialog()
    try:
        d.content_edit.setPlainText("   ")          # 空白内容
        d._on_save()
        assert warned, "空内容应被拦下"
        warned.clear()
        d.content_edit.setPlainText("有内容了")
        d.source_id_edit.setText("  ")               # 缺来源
        d._on_save()
        assert warned, "缺来源应被拦下"
    finally:
        d.close()
        d.deleteLater()


@pytest.mark.ui
def test_edit_dialog_returns_result(kb, qapp):
    from ui.fact_kb_dialog import FactEditDialog
    d = FactEditDialog()
    try:
        d.content_edit.setPlainText("新事实内容")
        d.source_id_edit.setText("2401.12345")
        d._on_save()
        res = d.result()
        assert res and res["content"] == "新事实内容"
        assert res["source_id"] == "2401.12345"
    finally:
        d.close()
        d.deleteLater()


# ============================================================
# 列宽 (9-29 用户反馈: 更新时间太宽 / 核心对象太窄)
# ============================================================

@pytest.mark.ui
def test_column_widths_give_space_to_content_not_timestamp(kb, qapp):
    """
    回归: 原来 setStretchLastSection(True) 把剩余空间全给了"更新时间"列,
    真正需要空间的"内容"被挤扁。改成短列自适应 + 内容列拉伸。
    """
    from PySide6.QtWidgets import QHeaderView
    from ui.fact_kb_dialog import FactKBDialog
    d = FactKBDialog()
    try:
        h = d.table.horizontalHeader()
        assert h.sectionResizeMode(3) == QHeaderView.Stretch, "内容列应拉伸"
        assert h.sectionResizeMode(4) == QHeaderView.ResizeToContents, (
            "更新时间应按内容自适应, 不该抢空间"
        )
        assert h.sectionResizeMode(2) != QHeaderView.Stretch, (
            "核心对象不该拉伸(会把内容挤扁), 用 Interactive 给固定宽度"
        )
        assert d.table.columnWidth(2) >= 150, "核心对象列应有可用的默认宽度"
    finally:
        d.close()
        d.deleteLater()
