"""
ui.theme_graph_view — 主题关联点线图视图 (10-05)

节点 = **主题**(不是宏观主题), 一个主题一个颜色。宏观主题画成一圈背景圆, 它
底下的主题聚在圈里。边 = 两个主题**共享的论文**, 线宽 ∝ 共享篇数。

**默认一条线都不画** —— 441 条边全画出来就是一团毛线, 什么都读不出来。所以:
    点某个主题 → 只画**以它为端点**的那些边, 并高亮它自己和邻居, 其余淡出
    点空白区域 → 全部收起
这是用户明确要的交互(原话: "不点击某个节点的时候, 没有显示的关联, 不然图上
密密麻麻的全是线了")。

**标签按缩放阈值显隐**: 182 个主题挤在一屏时把名字都画出来就是一团糊。缩略时
只留彩色圆点(靠 tooltip 和点选看名字), 放大到能读的字号再放出来。不这么做,
打开窗口第一眼看到的就是一堆认不出的色块。

**为什么自己画而不用现成图控件**: 只有 182 节点 / 441 边, QGraphicsScene 画得
动, 还能直接吃到项目那套 QSS 主题色; 引第三方图控件要多做一层样式桥接。
"""
from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from agent.infra import graph_layout as gl

NODE_R_BASE = 4.0
NODE_R_PER_SQRT = 2.1
LABEL_MAX_CHARS = 14
LABEL_ZOOM = 0.55        # 缩放到这个倍数以上才显示主题名
_ZOOM_STEP = 1.18
# 背景圆靠这个标记从"点中的 item"里排除(见 _Scene.mousePressEvent)
ROLE_BACKGROUND = 33


class _Scene(QGraphicsScene):
    """多两个信号: 点在空白处 / 点在某个主题上。"""

    node_clicked = Signal(str)
    background_clicked = Signal()

    def mousePressEvent(self, ev):
        views = self.views()
        # scenePos() 才对: 送到 QGraphicsScene.mousePressEvent 的是
        # QGraphicsSceneMouseEvent, 它有 scenePos() 没有 position();
        # 而 QMouseEvent 恰好相反(有 position() 没有 scenePos())。
        hit = self.itemAt(ev.scenePos(), views[0].transform()) if views else None
        # 背景圆自己过滤掉。**不能靠 acceptedMouseButtons**: 给它设
        # Qt.NoButton 之后实测 itemAt 照样返回它(那个"只返回接受鼠标的项"的
        # 开关在这里没生效), 于是点圆内部既不选主题也不收起关联, 是个死区。
        if hit is not None and hit.data(ROLE_BACKGROUND):
            hit = None
        if hit is None:
            self.background_clicked.emit()
        else:
            # 标签是节点的子项, 所以顺着 parent 往上找到带名字的那个
            it: QGraphicsItem | None = hit
            while it is not None:
                name = it.data(0)
                if isinstance(name, str) and name:
                    self.node_clicked.emit(name)
                    break
                it = it.parentItem()
        super().mousePressEvent(ev)


class ThemeGraphView(QWidget):
    """主题点线图。`set_data()` 喂数据, `select_theme()` 控关联显隐。"""

    node_selected = Signal(str)   # 主题名; 空串 = 什么都没选

    def __init__(self, parent=None):
        super().__init__(parent)
        self._node_items: dict[str, QGraphicsItem] = {}
        self._edge_items: list[tuple[str, str, QGraphicsItem]] = []
        self._selected: str = ""
        self._dim_opacity = 0.16

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        self.info = QLabel("点一个主题看它的关联; 点空白区域收起。")
        bar.addWidget(self.info, 1)
        fit_btn = QPushButton("适应窗口")
        fit_btn.clicked.connect(self.fit)
        zoom_btn = QPushButton("放大到能读")
        zoom_btn.clicked.connect(self._zoom_to_labels)
        bar.addWidget(fit_btn)
        bar.addWidget(zoom_btn)
        v.addLayout(bar)

        self._scene = _Scene()
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._view.setDragMode(QGraphicsView.ScrollHandDrag)
        self._view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.viewport().installEventFilter(self)
        self._scene.node_clicked.connect(self.select_theme)
        self._scene.background_clicked.connect(self.select_theme)
        v.addWidget(self._view, 1)

    # ============================================================
    # 数据
    # ============================================================

    def set_data(self, nodes: list[dict], edges: list[dict]) -> None:
        """nodes: `[{"name","macro","n_facts"}]`; edges: `[{"a","b","weight"}]`"""
        layout = gl.layout_themes(nodes, edges)
        self._scene.clear()
        self._node_items.clear()
        self._edge_items.clear()
        self._selected = ""

        if not layout.pos:
            self.info.setText("没有可画的主题(结构层还是空的)。")
            return

        n_by_name = {n["name"]: n.get("n_facts", 0) for n in nodes}

        # 背景圆先画, z 值最低, 不会盖住节点
        for mi, (macro, (cx, cy, r)) in enumerate(layout.macro_circles.items()):
            self._scene.addItem(self._make_macro_circle(macro, cx, cy, r, mi))
        for i, (name, (x, y)) in enumerate(layout.pos.items()):
            dot = self._make_node(name, x, y, n_by_name.get(name, 0), i)
            self._scene.addItem(dot)
            self._node_items[name] = dot
        for e in edges:
            a, b = e.get("a"), e.get("b")
            if a not in layout.pos or b not in layout.pos:
                continue
            # addLine 已经把线加进场景, 不要再 addItem(会刷一屏
            # "already been added to this scene" 警告)
            self._edge_items.append(
                (a, b, self._make_edge(a, b, e.get("weight", 1),
                                       layout.pos[a], layout.pos[b])))

        rect = QRectF(0, 0, layout.width, layout.height)
        self._scene.setSceneRect(rect)
        self._view.setSceneRect(rect)
        self.fit()

    def _make_macro_circle(self, macro, cx, cy, r, index) -> QGraphicsItem:
        color = QColor(gl.macro_color(index))
        item = QGraphicsEllipseItem(cx - r, cy - r, r * 2, r * 2)
        item.setZValue(-10)
        item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 26)))
        pen = QPen(color)
        pen.setWidthF(1.4)
        pen.setStyle(Qt.DashLine)
        item.setPen(pen)
        item.setToolTip(f"宏观主题: {macro}")
        item.setData(ROLE_BACKGROUND, True)   # 点它等于点空白 → 收起关联
        item.setAcceptedMouseButtons(Qt.NoButton)
        label = QGraphicsSimpleTextItem(macro)
        label.setBrush(QBrush(color.lighter(135)))
        f = QFont()
        f.setPointSizeF(11.0)
        f.setBold(True)
        label.setFont(f)
        label.setPos(cx, cy - r - 4)
        label.setParentItem(item)      # addToGroup 只在 QGraphicsItemGroup 上有
        return item

    def _make_node(self, name, x, y, n_facts, index) -> QGraphicsItem:
        r = NODE_R_BASE + NODE_R_PER_SQRT * math.sqrt(max(0, n_facts))
        color = QColor(gl.theme_color(index))
        dot = QGraphicsEllipseItem(x - r, y - r, r * 2, r * 2)
        dot.setZValue(5)
        dot.setBrush(QBrush(color))
        pen = QPen(QColor(20, 20, 20, 160))
        pen.setWidthF(0.8)
        dot.setPen(pen)
        dot.setData(0, name)
        dot.setToolTip(f"{name}\n核心对象 {n_facts} 条")

        text = QGraphicsSimpleTextItem(_ellipsis(name))
        text.setBrush(QBrush(color.lighter(115)))
        f = QFont()
        f.setPointSizeF(7.0)
        text.setFont(f)
        text.setPos(r + 3, -5)
        # 标签保持子项身份: _Scene 顺着 parent 往上找名字, 点文字 == 点节点。
        # (实测 Qt 的 itemAt 对接受 Qt.NoButton 的文本项照样返回它, 所以这里
        #  保持默认的鼠标处理就行, 不必特意去设任何值。)
        text.setVisible(False)          # 缩放到能读的字号再显示
        text.setParentItem(dot)
        dot.setData(1, text)
        return dot

    def _make_edge(self, a, b, weight, pa, pb) -> QGraphicsItem:
        w = min(5.0, 0.8 + 0.28 * max(1, int(weight or 1)))
        pen = QPen(QColor(150, 150, 160, 190))
        pen.setWidthF(w)
        line = self._scene.addLine(pa[0], pa[1], pb[0], pb[1], pen)
        line.setZValue(0)
        line.setVisible(False)          # 默认不画, 见模块头
        line.setToolTip(f"{a} ↔ {b}: 共享 {weight} 篇论文")
        return line

    # ============================================================
    # 关联显隐
    # ============================================================

    def select_theme(self, name: str | None = None) -> None:
        """选中一个主题 → 画以它为端点的边; 传空 → 全部收起。

        收起**只**由"点空白区域"触发(见 `_Scene.background_clicked`)。不给
        "再点一次同一节点也收起"加第二条路径: 曾经加过, 结果两条机制互相干扰
        —— 程序里连着选两次同一个主题(刷新后重选、对账遍历)会被当成"取消选择",
        静默把刚画出来的关联收了回去。隐藏路径只有一条, 才猜得对。
        """
        name = name or ""
        self._selected = name

        if not name:
            for _a, _b, e in self._edge_items:
                e.setVisible(False)
            for item in self._node_items.values():
                item.setOpacity(1.0)
                self._set_label_visible(item, None)
            self.info.setText("点一个主题看它的关联; 点空白区域收起。")
            self.node_selected.emit("")
            return

        neighbors: set[str] = set()
        shown = 0
        for a, b, e in self._edge_items:
            on = (a == name or b == name)
            e.setVisible(on)
            if on:
                shown += 1
                neighbors.add(b if a == name else a)

        for key, item in self._node_items.items():
            if key == name:
                item.setOpacity(1.0)
                self._set_label_visible(item, True)
            elif key in neighbors:
                item.setOpacity(0.95)
                self._set_label_visible(item, True)
            else:
                item.setOpacity(self._dim_opacity)
                self._set_label_visible(item, False)
        self.info.setText(f"【{name}】 关联 {shown} 个主题(共享论文) —— 点空白区域收起")
        self.node_selected.emit(name)

    def _set_label_visible(self, item: QGraphicsItem, on: bool | None) -> None:
        """标签显隐。`on=None` 表示"按当前缩放决定"。"""
        text = item.data(1)
        if text is not None:
            text.setVisible(self._current_scale() >= LABEL_ZOOM
                            if on is None else bool(on))

    # ============================================================
    # 缩放
    # ============================================================

    def _current_scale(self) -> float:
        try:
            return abs(float(self._view.transform().m11()))
        except Exception:
            return 1.0

    def _refresh_label_visibility(self) -> None:
        """按缩放刷一遍标签。正在看关联时不动 —— 那时的显隐是关联驱动的。"""
        if self._selected:
            return
        for item in self._node_items.values():
            self._set_label_visible(item, None)

    def fit(self) -> None:
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._refresh_label_visibility()

    def _zoom_to_labels(self) -> None:
        self._view.resetTransform()
        self._refresh_label_visibility()

    def eventFilter(self, obj, ev):
        # QGraphicsView 默认不响应滚轮缩放, 自己接 —— 否则"放大才显示名字"
        # 这个特性根本没有触发途径
        if ev.type() == QEvent.Type.Wheel and ev.angleDelta().y():
            step = _ZOOM_STEP if ev.angleDelta().y() > 0 else 1.0 / _ZOOM_STEP
            self._view.scale(step, step)
            self._refresh_label_visibility()
            return True
        return super().eventFilter(obj, ev)


def _ellipsis(s: str, n: int = LABEL_MAX_CHARS) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"
