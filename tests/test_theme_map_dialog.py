"""
tests/test_theme_map_dialog.py — 主题地图 UI (10-05)

锁的是四件最容易悄悄坏掉的事:

1. **三层真的都在树上**。宏观主题 → 主题 → 核心对象 是用户 10-05 亲自纠正过
   的模型; 少一层 UI 不会报错, 只是"看起来没那么完整"。
2. **论文不是节点**。跨域边的 join 键是 source_id(论文) 不是 fact_id。写错时
   不报错、边数少两个数量级, 界面上就是"跨域关联那一页几乎是空的"。
3. **不静默截断**。graph_query 默认 limit=200, 真库 441 条边 —— 照默认走就是
   悄悄少给 241 条, 而用户看到的只是"有 200 条", 完全无从察觉。
4. **QSS 不覆盖全局字号**。QSS 里给 QDialog 写 font-size 会在绘制期盖掉
   QApplication 的全局字体并级联(PAgent 10-05 实测: 改设置里的字号, 菜单栏和
   树面板纹丝不动)。本项目 ui/styles.py 已经用 get_stylesheet() 读
   QApplication.font().pointSize() 处理好了, 所以窗口**只能**用它, 不能自己再
   写一份 font-size。

跑在**临时库**上(不碰 data/knowledge/facts.db), 但 schema 走生产 init_schema()
—— 改 schema 会被这些测试挡住。
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DLG = REPO / "ui" / "theme_map_dialog.py"
WIN = REPO / "ui" / "chat_window.py"


def _src():
    return DLG.read_text(encoding="utf-8")


# ============================================================
# 源码级约束(不需要 QApplication, 跑得快)
# ============================================================

def test_main_dialog_is_non_modal():
    """跟其它知识库窗口模型一致: 不阻塞主窗口, 边看结构边写综述。"""
    from ui.theme_map_dialog import ThemeMapDialog
    src = inspect.getsource(ThemeMapDialog.__init__)
    assert "setModal(False)" in src, "主对话框必须非模态"
    assert "Qt.NonModal" in src


def test_entry_is_mounted():
    """入口要挂在「知识库」菜单, 且存到 self 上防 GC(抄 fact_kb 的模式)。"""
    src = WIN.read_text(encoding="utf-8")
    assert 'QAction("主题地图"' in src, "知识库菜单里必须有「主题地图」"
    assert "triggered.connect(self.open_theme_map)" in src
    i = src.index("def open_theme_map")
    body = src[i:i + 900]
    assert "self._theme_map_dialog = dialog" in body, "必须存到 self 上, 否则 show() 后被 GC"
    assert "dialog.show()" in body
    assert "exec()" not in body, "不能用 exec(), 那是模态"
    assert "Qt.NonModal" in body


def test_dialog_does_not_hardcode_font_size():
    """字号只能来自 QApplication —— 见模块头第 4 条。"""
    src = _src()
    assert "font-size" not in src, (
        "QSS 里的 font-size 会覆盖 QApplication 全局字体并级联, "
        "窗口里不许自己写; 用 ui.styles.get_stylesheet()"
    )
    assert "get_stylesheet()" in src


def test_cross_limit_exceeds_query_default():
    """必须显式放大 limit, 否则 441 条边会被默认的 200 悄悄砍掉 241 条。"""
    from agent.infra import graph_query as gq
    from ui.theme_map_dialog import CROSS_LIMIT
    default = inspect.signature(gq.themes_sharing_papers).parameters["limit"].default
    assert CROSS_LIMIT > default, (
        f"CROSS_LIMIT={CROSS_LIMIT} 不大于 graph_query 默认 limit={default}, "
        f"界面会静默少显示一批跨域边"
    )


def test_truncation_is_reported_not_hidden():
    """截断了就必须写在界面上, 不能只在代码里 limit 一下就当没事发生。"""
    src = _src()
    assert "已截断" in src and "未截断" in src, "标签必须区分是否截断"


# ============================================================
# graph_query.facts_by_theme —— 三层树的叶子, 此前没有任何函数能按主题取
# ============================================================

def test_facts_by_theme_returns_all_themes(gdb):
    _seed()
    from agent.infra import graph_query as gq
    allm = gq.facts_by_theme()
    assert set(allm) == {"主题一", "主题二", "主题三", "游离主题"}
    assert len(allm["主题一"]) == 3


def test_facts_by_theme_filters_by_name(gdb):
    _seed()
    from agent.infra import graph_query as gq
    only = gq.facts_by_theme("主题三")
    assert set(only) == {"主题三"}, "传了名字就只能返回那一个主题"
    assert len(only["主题三"]) == 3


def test_facts_by_theme_limit_is_global_not_per_theme(gdb):
    """limit 是全局上限, 不是每主题上限。

    真库 182 个主题, "每主题 200" 就是 36000 行 —— 那不是保护是雪崩, 而且
    theme 数会随语料继续涨。

    断言选 limit=5 是因为它能真的分辨两种语义: 种子共 9 条已归属 fact, 按
    主题名排序后前 5 条必然跨**两个**主题(3 + 2), 所以总数必须是 5。若有人把它
    改成"每主题最多 1 条", 总数会变成 4, 这条就红 —— 用 limit=1 是分辨不出来的
    (两种语义都给 1)。
    """
    _seed()
    from agent.infra import graph_query as gq
    capped = gq.facts_by_theme(limit=5)
    assert sum(len(v) for v in capped.values()) == 5, (
        "总数应恰好等于全局 limit; 偏少说明被改成了按主题封顶"
    )
    assert len(capped) == 2, "前 5 条应跨两个主题(3+2), 只有一个说明是按主题切了"


def test_facts_by_theme_carries_source_count(gdb):
    """叶子要显示"几个出处", 多出处 fact 是这个库的核心价值。"""
    _seed()
    from agent.infra import graph_query as gq
    rows = gq.facts_by_theme("主题一")["主题一"]
    assert all("n_sources" in r and "fact_id" in r and "content" in r for r in rows)


# ============================================================
# 真跑一遍 UI (隔离库)
# ============================================================

@pytest.fixture(scope="module")
def qapp():
    import os
    import sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def gdb(tmp_path, monkeypatch):
    """把 facts.db 指到 tmp_path, 走生产 init_schema 建表。"""
    from agent.infra import db as infra_db
    monkeypatch.setattr(infra_db, "_data_root", lambda: tmp_path)
    from agent.tools import facts as F
    from agent.infra import theme_graph as tg
    F.init_schema()
    tg.init_schema()
    return infra_db


def _seed():
    """造一个能同时触发**所有**界面分支的最小图:

        宏观甲: 主题一(3条) 主题二(1条)
        宏观乙: 主题三(3条)
        (游离) 游离主题(2条)          ← 测 macro=None 那一组

        论文 P1 → 主题一/主题二/主题三     (三块切同一篇)
        论文 P2 → 主题一/主题三/游离主题
        论文 P4 → 主题一/主题三/游离主题
        论文 P9 → 谁都不属于                ← 测"未归入任何主题"的诚实提示

    边(join 键是论文)共 5 条:
        主题一-主题二 P1 · 主题二-主题三 P1 · 主题一-主题三 P1/P2/P4
        主题一-游离 P2/P4 · 主题三-游离 P2/P4
    其中 >=2 篇的边有 3 条, 于是 bridges() 吐出 2 个桥接主题
    (主题一 连 2 个方向, 主题三 连 1 个) —— 正好覆盖"跨接分支数"那一列。
    """
    from agent.infra import theme_graph as tg
    from agent.infra.db import connect
    from agent.tools import facts as F

    def add(content, paper):
        return F.add_fact(content=content, fact_type="result", subject="测试对象",
                          source_type="paper_abstract",
                          source_id=paper)["fact_id"]

    ids = {
        "t1a": add("主题一的事实甲来自第一篇", "P1"),
        "t1b": add("主题一的事实乙来自第二篇", "P2"),
        "t1c": add("主题一的事实丙来自第四篇", "P4"),
        "t2a": add("主题二的事实甲来自第一篇", "P1"),
        "t3a": add("主题三的事实甲来自第一篇", "P1"),
        "t3b": add("主题三的事实乙来自第二篇", "P2"),
        "t3c": add("主题三的事实丙来自第四篇", "P4"),
        "lo1": add("游离主题的事实甲来自第二篇", "P2"),
        "lo2": add("游离主题的事实乙来自第四篇", "P4"),
        "orphan": add("不属于任何主题的孤立事实", "P9"),
    }

    with connect("facts") as c:
        m1 = tg.upsert_macro(c, "宏观甲", "第一个宏观板块")
        m2 = tg.upsert_macro(c, "宏观乙", "第二个宏观板块")
        t1 = tg.upsert_theme(c, "主题一", "主题一的陈述", macro_id=m1)
        t2 = tg.upsert_theme(c, "主题二", "主题二的陈述", macro_id=m1)
        t3 = tg.upsert_theme(c, "主题三", "主题三的陈述", macro_id=m2)
        t4 = tg.upsert_theme(c, "游离主题", "还没归入宏观层", macro_id=None)
        tg.link_facts(c, t1, [ids["t1a"], ids["t1b"], ids["t1c"]])
        tg.link_facts(c, t2, [ids["t2a"]])
        tg.link_facts(c, t3, [ids["t3a"], ids["t3b"], ids["t3c"]])
        tg.link_facts(c, t4, [ids["lo1"], ids["lo2"]])
    return ids


def _macro_items(d):
    """顶层(宏观主题 / 游离组)与其直接子节点。"""
    root = d.tree.invisibleRootItem()
    return [(root.child(i),
             [root.child(i).child(j) for j in range(root.child(i).childCount())])
            for i in range(root.childCount())]


def _find_theme(d, name):
    """按名字取主题节点。

    不按位置取: `tree()` 是 `ORDER BY name`, 而中文按 Unicode 码位排("乙" U+4E59
    排在"甲" U+7532 前面), 位置跟肉眼预期的顺序相反 —— 一按位置取, 测试就变成
    在测排序而不是在测"这个主题的详情对不对"。
    """
    for _macro, themes in _macro_items(d):
        for it in themes:
            if it.text(0) == name:
                return it
    raise AssertionError(f"树上找不到主题 {name}")


def _visible_roots(d):
    root = d.tree.invisibleRootItem()
    return sum(1 for i in range(root.childCount()) if not root.child(i).isHidden())


@pytest.mark.ui
def test_tree_has_three_levels(gdb, qapp):
    """宏观 → 主题 → 核心对象, 一层都不能少。"""
    from ui.theme_map_dialog import ThemeMapDialog
    _seed()
    d = ThemeMapDialog()
    try:
        groups = _macro_items(d)
        # 顺序不测: tree() 是 ORDER BY name, 中文按码位排跟肉眼顺序相反
        assert {g[0].text(0) for g in groups} == {"宏观甲", "宏观乙", "未归入宏观主题"}
        assert sorted(len(g[1]) for g in groups) == [1, 1, 2]
        theme1 = _find_theme(d, "主题一")
        assert theme1.childCount() == 3, "主题下必须挂着它的核心对象"
        leaf = theme1.child(0)
        assert leaf.text(0).startswith("[result]"), "叶子是核心对象"
        assert "出处" in leaf.text(1)
    finally:
        d.deleteLater()


@pytest.mark.ui
def test_stats_report_unlinked_objects(gdb, qapp):
    """没归入任何主题的核心对象要**显示出来**, 不能让界面假装结构是完整的。"""
    from ui.theme_map_dialog import ThemeMapDialog
    _seed()
    d = ThemeMapDialog()
    try:
        txt = d.stats_label.text()
        assert "尚未归入任何主题" in txt, "孤立核心对象必须在界面上可见"
        assert "1 条" in txt
    finally:
        d.deleteLater()


@pytest.mark.ui
def test_theme_detail_shows_statement_and_cross_links(gdb, qapp):
    from ui.theme_map_dialog import ThemeMapDialog
    _seed()
    d = ThemeMapDialog()
    try:
        d.tree.setCurrentItem(_find_theme(d, "主题一"))
        txt = d.tree_detail.toPlainText()
        assert "主题一的陈述" in txt
        assert "跨域关联" in txt
        assert "主题三" in txt, "主题一跟主题三共享 P1/P2, 得列出来"
    finally:
        d.deleteLater()


@pytest.mark.ui
def test_fact_detail_shows_sources_and_owning_theme(gdb, qapp):
    from ui.theme_map_dialog import ThemeMapDialog
    _seed()
    d = ThemeMapDialog()
    try:
        d.tree.setCurrentItem(_find_theme(d, "主题一").child(0))
        txt = d.tree_detail.toPlainText()
        assert "主题一的事实甲" in txt
        assert "P1" in txt, "出处必须列出来"
        assert "所属主题" in txt and "主题一" in txt
    finally:
        d.deleteLater()


@pytest.mark.ui
def test_cross_tab_lists_every_edge(gdb, qapp):
    """五条边少一条就是 join 键写错了(写成 fact_id 会安静地少两个数量级)。"""
    from ui.theme_map_dialog import ThemeMapDialog
    _seed()
    d = ThemeMapDialog()
    try:
        assert d.pair_table.rowCount() == 5
        # >=2 篇共享的边有 3 条 → 主题一 连 2 个方向, 主题三 连 1 个
        assert d.bridge_table.rowCount() == 2
        assert d.bridge_table.item(0, 1).text() == "2"
    finally:
        d.deleteLater()


@pytest.mark.ui
def test_pair_detail_explains_how_paper_was_split(gdb, qapp):
    """跨域边那一页的价值不是"两主题相似", 而是"这篇论文被切成了哪几块"。"""
    from ui.theme_map_dialog import ThemeMapDialog
    _seed()
    d = ThemeMapDialog()
    try:
        d.pair_table.selectRow(0)
        txt = d.cross_detail.toPlainText()
        assert "共享论文" in txt
        assert "→" in txt, "要按主题拆出这篇论文的 fact 分布"
    finally:
        d.deleteLater()


@pytest.mark.ui
def test_bridge_detail_lists_connections(gdb, qapp):
    from ui.theme_map_dialog import ThemeMapDialog
    _seed()
    d = ThemeMapDialog()
    try:
        d.bridge_table.selectRow(0)
        txt = d.cross_detail.toPlainText()
        assert "桥接主题" in txt and "共享" in txt
    finally:
        d.deleteLater()


@pytest.mark.ui
def test_truncation_note_appears_when_limit_hit(gdb, qapp, monkeypatch):
    """把上限压到 1, 界面必须**自己说**被截断了。"""
    import ui.theme_map_dialog as mod
    from ui.theme_map_dialog import ThemeMapDialog
    _seed()
    monkeypatch.setattr(mod, "CROSS_LIMIT", 1)
    d = ThemeMapDialog()
    try:
        assert d.pair_table.rowCount() == 1
        assert "已截断" in d.cross_note.text()
    finally:
        d.deleteLater()


@pytest.mark.ui
def test_search_matches_fact_content_and_keeps_ancestors(gdb, qapp):
    """搜索框承诺能搜"核心对象", 就必须能按 fact 正文命中, 且父节点不能被藏掉
    (藏掉父节点 → 子节点跟着消失, 用户会觉得"搜不到")。"""
    from ui.theme_map_dialog import ThemeMapDialog
    _seed()
    d = ThemeMapDialog()
    try:
        d.search_edit.setText("孤立事实")
        assert _visible_roots(d) == 0, "孤立 fact 不属于任何主题, 不该在任何组里出现"
        d.search_edit.setText("主题三的事实丙")
        assert _visible_roots(d) == 1, "命中第三篇的 fact, 只有宏观乙那一组留着"
        d.search_edit.setText("主题一")
        assert _visible_roots(d) == 1
        d.search_edit.setText("")
        assert _visible_roots(d) == 3, "清空后必须全放回来"
    finally:
        d.deleteLater()


@pytest.mark.ui
def test_search_miss_hides_everything(gdb, qapp):
    from ui.theme_map_dialog import ThemeMapDialog
    _seed()
    d = ThemeMapDialog()
    try:
        d.search_edit.setText("ZZZ查无此物")
        assert _visible_roots(d) == 0
    finally:
        d.deleteLater()
