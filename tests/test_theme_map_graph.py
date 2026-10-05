"""
tests/test_theme_map_graph.py — 主题关联点线图 (10-05)

锁的是四件"不报错、只是悄悄不对"的事:

1. **默认一条边都不画**。441 条边全画出来是一团毛线, 什么都读不出来。用户
   明确要求"不点击某个节点的时候, 没有显示的关联"。
2. **点的关联数 == 查询层的度数**。这条是本轮真踩到的坑: `_fill_graph` 把
   graph_query 的 `theme_a/theme_b` 原样透传, 视图层读的是 `a/b` → 每条边
   都取不到端点 → 一条都不画。**而场景里 182 个节点照常在, 看起来完全正常**,
   只是关联功能整个是死的。所以这里不只测"有边", 而是逐个主题跟 graph_query
   的度数对账。
3. **宏观背景圆两两不重叠, 且每个主题都在自己那个圈里**。写反了就是"81 个
   主题掉到圈外 + 五个圈叠成一坨", 读图的人完全看不出板块。
4. **布局是确定性的**。位置每次都抖的话, 没法把上周的图和这周的图对比。

纯布局部分(`agent/infra/graph_layout.py`)不碰 Qt, 直接单测; 视图部分才要
QApplication。
"""
from __future__ import annotations

import itertools
import math

import pytest

from agent.infra import graph_layout as gl

# ============================================================
# 假数据(跟 tests/test_theme_map_dialog.py 的种子同构, 方便两边对照)
# ============================================================

def _nodes():
    return [
        {"name": "主题一", "macro": "宏观甲", "n_facts": 3},
        {"name": "主题二", "macro": "宏观甲", "n_facts": 1},
        {"name": "主题三", "macro": "宏观乙", "n_facts": 3},
        {"name": "游离主题", "macro": None, "n_facts": 2},
    ]


def _edges():
    return [
        {"a": "主题一", "b": "主题二", "weight": 1},
        {"a": "主题一", "b": "主题三", "weight": 3},
        {"a": "主题二", "b": "主题三", "weight": 1},
        {"a": "主题一", "b": "游离主题", "weight": 2},
        {"a": "主题三", "b": "游离主题", "weight": 2},
    ]


def _big_graph(n_macro=5, per_macro=8):
    """规模接近真库(真库 182 主题 / 441 边), 用来验性能和不变量。"""
    nodes = []
    for m in range(n_macro):
        for i in range(per_macro):
            nodes.append({"name": f"M{m}T{i}", "macro": f"宏观{m}",
                          "n_facts": 1 + (i * 7) % 40})
    edges = []
    for m in range(n_macro):
        for i in range(per_macro):
            j = (i + 1) % per_macro
            if j != i:
                edges.append({"a": f"M{m}T{i}", "b": f"M{m}T{j}", "weight": 1 + i % 4})
    for m in range(n_macro - 1):
        edges.append({"a": f"M{m}T0", "b": f"M{m + 1}T0", "weight": 5})
    return nodes, edges


# ============================================================
# 纯布局(不需要 QApplication)
# ============================================================

def test_every_node_gets_a_position():
    lay = gl.layout_themes(_nodes(), _edges())
    assert set(lay.pos) == {n["name"] for n in _nodes()}


def test_macro_circles_do_not_overlap():
    """背景圆叠在一起就等于没有分组 —— 这是本轮实测翻过的车。"""
    nodes, edges = _big_graph()
    lay = gl.layout_themes(nodes, edges)
    assert len(lay.macro_circles) == 5
    for (k1, (x1, y1, r1)), (k2, (x2, y2, r2)) in itertools.combinations(
            lay.macro_circles.items(), 2):
        gap = math.hypot(x1 - x2, y1 - y2) - (r1 + r2)
        assert gap >= -1.0, f"{k1} 与 {k2} 的背景圆重叠 {gap:.1f}px"


def test_themes_stay_inside_their_macro_circle():
    nodes, edges = _big_graph()
    lay = gl.layout_themes(nodes, edges)
    for n in nodes:
        c = lay.macro_circles.get(n["macro"])
        if c is None:
            continue
        x, y, r = c
        px, py = lay.pos[n["name"]]
        d = math.hypot(px - x, py - y)
        assert d <= r, f"{n['name']} 掉出 {n['macro']} 的圈外 {d - r:.1f}px"


def test_ungrouped_themes_get_no_circle():
    """没有宏观就没有背景圆: 硬画一个叫"未归类"的圈, 会让人以为那是个板块。"""
    lay = gl.layout_themes(_nodes(), _edges())
    assert None not in lay.macro_circles
    assert "游离主题" in lay.pos


def test_layout_is_deterministic():
    """位置每次都抖 → 没法把上周的图和这周的图对比。"""
    nodes, edges = _big_graph()
    a = gl.layout_themes(nodes, edges)
    b = gl.layout_themes(nodes, edges)
    for name in a.pos:
        assert a.pos[name] == pytest.approx(b.pos[name], abs=1e-9)


def test_coordinates_are_non_negative():
    lay = gl.layout_themes(*_big_graph())
    assert all(x >= 0 and y >= 0 for x, y in lay.pos.values())


def test_no_two_nodes_overlap():
    """任意两个主题的间距 ≥ 节点直径, 否则就是"几个点团在一起"。

    根因不是斥力不够, 而是斥力**方向**在两点靠近时趋零:
    `(delta/dist)·1/d²` 里 delta≈0 时单位向量也是 0, 1e12 乘 0 还是 0。
    真库有 33 个**没有任何边**的孤立主题(没有边引力, 只剩斥力 + 宏观引力),
    它们被吸到本组质心这个平衡点上叠成一堆 —— 实测最小间距 2.25px, 而节点
    直径最大 40.8px, 还有一对坐标完全相同。所以必须有一道显式的最小间距约束。
    """
    nodes, edges = _big_graph()
    lay = gl.layout_themes(nodes, edges)
    pos = lay.pos
    worst = min(math.dist(pos[a["name"]], pos[b["name"]])
                for a, b in itertools.combinations(nodes, 2))
    biggest = max(2 * (4.0 + 2.1 * math.sqrt(n["n_facts"])) for n in nodes)
    assert worst >= biggest, (
        f"最近的两个主题只隔 {worst:.1f}px, 而最大节点直径 {biggest:.1f}px —— "
        f"会有节点叠在一起"
    )


def test_min_gap_is_honoured_exactly():
    """min_gap 是最终像素, 而且要真的生效(不是摆设参数)。"""
    nodes, edges = _big_graph()
    for want in (30.0, 44.0):
        lay = gl.layout_themes(nodes, edges, min_gap=want)
        worst = min(math.dist(lay.pos[a["name"]], lay.pos[b["name"]])
                    for a, b in itertools.combinations(nodes, 2))
        assert worst >= want - 0.5, f"min_gap={want} 实际只做到 {worst:.1f}"


def test_isolated_themes_do_not_stack():
    """没有边的孤立主题最容易被叠死 —— 单独用"全是孤立节点"的图来验。"""
    nodes = [{"name": f"N{i}", "macro": "宏观甲", "n_facts": 5}
             for i in range(30)]
    lay = gl.layout_themes(nodes, [])          # 一条边都没有
    worst = min(math.dist(lay.pos[a["name"]], lay.pos[b["name"]])
                for a, b in itertools.combinations(nodes, 2))
    assert worst >= 10.0, f"30 个孤立主题叠到了 {worst:.1f}px"


@pytest.mark.parametrize("nodes,edges,expect_pos,expect_circles", [
    pytest.param([], [], 0, 0, id="empty"),
    pytest.param([{"name": "A", "macro": None}], [], 1, 0, id="no-edges"),
    pytest.param([{"name": "A", "macro": "M"}, {"name": "B", "macro": "M"}],
                 [], 2, 1, id="one-macro"),
    pytest.param([{"name": "A", "macro": "M"}],
                 [{"a": "A", "b": "GHOST", "weight": 1}], 1, 1, id="ghost-edge"),
    pytest.param([{"name": "A", "macro": None}],
                 [{"a": "A", "b": "A", "weight": 9}], 1, 0, id="self-edge"),
])
def test_degenerate_inputs_dont_crash(nodes, edges, expect_pos, expect_circles):
    """边指向不存在的节点 / 自环 / 空库: 宁可少画一条, 别让整个视图开不起来。"""
    lay = gl.layout_themes(nodes, edges)
    assert len(lay.pos) == expect_pos
    assert len(lay.macro_circles) == expect_circles


def test_theme_colors_are_distinct_and_adjacent_ones_differ():
    """一个主题一个颜色, 而且**相邻的**也得差得开。

    之前想过 `index // k` 分组取色 —— 那会让 182 个主题挤成十几种颜色,
    "一个主题一个颜色"直接作废。黄金角是为这个选的。
    """
    n = 182
    cols = [gl.theme_color(i) for i in range(n)]
    assert len(set(cols)) == n, "主题颜色有重复"

    def rgb(h):
        return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))

    close = [
        (i, i + 1) for i in range(n - 1)
        if math.dist(rgb(cols[i]), rgb(cols[i + 1])) < 90
    ]
    assert not close, f"相邻主题颜色太接近: {close[:5]}"


def test_macro_color_is_dimmer_than_theme_color():
    """圆是背景、节点是内容; 背景比内容亮就会盖住节点文字。"""
    def lum(h):
        r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
        return 0.299 * r + 0.587 * g + 0.114 * b
    assert lum(gl.macro_color(0)) < lum(gl.theme_color(0))


# ============================================================
# 视图(需要 QApplication)
# ============================================================

@pytest.fixture(scope="module")
def qapp():
    import os
    import sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def gview(qapp):
    from ui.theme_graph_view import ThemeGraphView
    v = ThemeGraphView()
    v.set_data(_nodes(), _edges())
    return v


def _visible_edges(v):
    return sum(1 for _a, _b, e in v._edge_items if e.isVisible())


@pytest.mark.ui
def test_nodes_are_themes_not_macros(gview):
    """节点是主题(4 个), 宏观主题只画成背景圆(2 个), 不占节点。"""
    assert len(gview._node_items) == 4
    assert len(gview._edge_items) == 5


@pytest.mark.ui
def test_default_shows_no_edges(gview):
    """核心要求: 不点就不画线, 否则图上密密麻麻全是线。"""
    assert _visible_edges(gview) == 0


@pytest.mark.ui
def test_select_theme_shows_only_its_own_edges(gview):
    gview.select_theme("主题一")
    # 主题一 连 主题二/主题三/游离主题 = 3 条
    assert _visible_edges(gview) == 3
    assert "主题一" in gview.info.text()


@pytest.mark.ui
def test_select_theme_shows_edges_on_either_endpoint(gview):
    """"以它为一端" —— 边是 (a,b), 谁在前面都要认。"""
    gview.select_theme("游离主题")
    assert _visible_edges(gview) == 2


@pytest.mark.ui
def test_clear_hides_everything(gview):
    gview.select_theme("主题一")
    gview.select_theme("")
    assert _visible_edges(gview) == 0


@pytest.mark.ui
def test_reselecting_same_theme_keeps_its_edges(gview):
    """回归: 曾经加过"再点一次同一节点就收起", 结果程序里连着选两次同一个主题
    (刷新后重选)会被静默收走。隐藏只走"点空白"这一条路。"""
    gview.select_theme("主题一")
    gview.select_theme("主题一")
    assert _visible_edges(gview) == 3


@pytest.mark.ui
def test_shown_edges_match_the_edge_list(gview):
    """逐个主题对账 —— 字段名对不上(theme_a vs a)时, 节点照常画出来,
    场景看着完全正常, 但一条边都没有。这个断言是唯一能抓住它的东西。"""
    edges = _edges()
    for name in (n["name"] for n in _nodes()):
        gview.select_theme(name)
        expect = sum(1 for e in edges if name in (e["a"], e["b"]))
        assert _visible_edges(gview) == expect, f"{name} 的关联数不对"


@pytest.mark.ui
def test_non_neighbors_are_dimmed(gview):
    """选中的主题和它的邻居保持清晰, 其余淡出 —— 保留上下文又不抢视线。"""
    gview.select_theme("主题一")
    gview.select_theme("")     # 先清一次, 再看 item 状态
    gview.select_theme("主题一")
    opac = {k: it.opacity() for k, it in gview._node_items.items()}
    assert opac["主题一"] == 1.0
    assert opac["主题三"] > 0.8, "邻居要保持清晰"
    assert opac["主题二"] > 0.8, "邻居要保持清晰"
    assert opac["游离主题"] > 0.8


@pytest.mark.ui
def test_labels_hidden_when_zoomed_out_shown_when_zoomed_in(gview):
    """182 个主题挤一屏时把名字都画出来就是一团糊。"""
    gview.select_theme("")
    gview._zoom_to_labels()          # scale 1.0
    assert all(it.data(1).isVisible() for it in gview._node_items.values())
    gview._view.scale(0.1, 0.1)      # 缩到看不清
    gview._refresh_label_visibility()
    assert not any(it.data(1).isVisible() for it in gview._node_items.values())


@pytest.mark.ui
def test_empty_data_shows_a_message_not_a_crash(qapp):
    from ui.theme_graph_view import ThemeGraphView
    v = ThemeGraphView()
    v.set_data([], [])
    assert "没有" in v.info.text()


def _click(scene, pos):
    """在场景坐标 pos 上"点"一下。

    合成的是 **QGraphicsSceneMouseEvent**, 不是 QMouseEvent: 真实点击由
    QGraphicsView 构造这个事件再交给 scene, 喂错类型会一路错到
    scenePos()/position() 取值上, 测出来的是假象。
    (也别改成往 QGraphicsView 塞 QMouseEvent —— offscreen 下 ScrollHandDrag
    会把进程打崩, 0xC0000005。)
    """
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtWidgets import QGraphicsSceneMouseEvent
    ev = QGraphicsSceneMouseEvent(QEvent.Type.MouseButtonPress)
    ev.setScenePos(QPointF(*pos))
    ev.setButton(Qt.MouseButton.LeftButton)
    ev.setButtons(Qt.MouseButton.LeftButton)
    scene.mousePressEvent(ev)


@pytest.mark.ui
def test_clicking_a_node_in_the_scene_selects_it(gview):
    """真实点选路径, 不走 select_theme() 直接调。标签是节点的子项, 所以
    点在文字上也必须能选中 —— 靠 scene 顺着 parent 往上找名字。"""
    from PySide6.QtWidgets import QGraphicsSimpleTextItem
    item = gview._node_items["主题一"]
    center = item.sceneBoundingRect().center()
    _click(gview._scene, (center.x(), center.y()))
    assert _visible_edges(gview) == 3, "点节点没反应 = 整个关联功能是死的"

    label = item.data(1)
    gview.select_theme("")
    gview._zoom_to_labels()            # 缩略时标签本来就不渲染, 点不到
    assert label.isVisible(), "放大后标签应该出现"
    assert label.parentItem() is item, "标签必须是节点的子项, scene 才能顺着往上找名字"

    # 不去抠"哪个标签在最上层"(邻居的标签 z 相同、绘制更晚, 常压在上面),
    # 断言的是真正要保证的事: 点在文字上, 选中的是**那个文字所属的主题**。
    tf = gview._view.transform()
    br = label.sceneBoundingRect()
    pt = (br.center().x(), br.center().y())
    top = gview._scene.itemAt(*pt, tf)
    assert isinstance(top, QGraphicsSimpleTextItem), "点上应该命中一个标签"
    owner = top.parentItem().data(0)
    _click(gview._scene, pt)
    expect = sum(1 for e in _edges() if owner in (e["a"], e["b"]))
    assert _visible_edges(gview) == expect, "点在标签文字上要能选中它所属的主题"


@pytest.mark.ui
def test_clicking_empty_area_hides_associations(gview):
    """用户明确要的手势: 点空白区域把关联收起来。"""
    gview.select_theme("主题一")
    assert _visible_edges(gview) == 3

    rect = gview._scene.sceneRect()
    tf = gview._view.transform()
    empty = None
    for fx in (0.5, 0.3, 0.7, 0.2, 0.8):
        for fy in (0.5, 0.3, 0.7, 0.2, 0.8):
            p = (rect.width() * fx, rect.height() * fy)
            if gview._scene.itemAt(*p, tf) is None:
                empty = p
                break
        if empty:
            break
    assert empty is not None, "场景里找不到空白点, 测试数据画得太满"
    _click(gview._scene, empty)
    assert _visible_edges(gview) == 0


@pytest.mark.ui
def test_clicking_inside_macro_circle_clears_associations(gview):
    """背景圆内部是"空白"而不是死区。

    点圆曾经**完全没反应**: 既不选主题也不收起关联。原因是当时靠
    `setAcceptedMouseButtons(Qt.NoButton)` 想让 itemAt 跳过它, 实测那个开关
    在这里不生效, itemAt 照样返回圆 —— 于是那一大片区域成了死区。
    """
    gview.select_theme("主题一")
    assert _visible_edges(gview) == 3

    circle = [i for i in gview._scene.items()
              if type(i).__name__ == "QGraphicsEllipseItem" and i.zValue() < 0][0]
    rect = circle.sceneBoundingRect()
    tf = gview._view.transform()
    spot = None
    r = rect.width() / 2
    c = rect.center()
    for frac in [i / 36 for i in range(36)]:
        ang = frac * 2 * math.pi
        p = (c.x() + r * 0.9 * math.cos(ang), c.y() + r * 0.9 * math.sin(ang))
        if gview._scene.itemAt(*p, tf) is circle:
            spot = p
            break
    assert spot is not None, "圆上找不到能被点到的位置"
    _click(gview._scene, spot)
    assert _visible_edges(gview) == 0, "点圆内部应该像点空白一样收起关联"


def test_dialog_mounts_graph_as_a_tab():
    """点线图要能从主题地图窗口里进去, 不是孤立的类。"""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "ui" / "theme_map_dialog.py").read_text(encoding="utf-8")
    assert 'self.tabs.addTab(self.graph_view, "关联图")' in src
    assert "self.graph_view.set_data(" in src
    # 字段名对齐: graph_query 出 theme_a/theme_b, 视图要 a/b
    assert '"a": e["theme_a"], "b": e["theme_b"]' in src
