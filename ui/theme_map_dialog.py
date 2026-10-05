"""
ui.theme_map_dialog — 主题地图 dialog (10-05)

职责:
- 把 `theme_macro → theme → fact` 三层结构画成一棵可点的树(这是**图**, 不是
  「论文综述库」那种"一次 run 的章节树" —— 主题与 workspace/run 无关, 累积生长)
- 点任一层出详情: 主题看陈述 + 跨域关联, 核心对象看内容 + 全部出处 + 所属主题
- 「跨域关联」页: 两个主题**共享论文**的边(441 条) + 桥接主题(同时和多个主题
  共享论文的节点)。这是"哪几块在互相借方法"的答案

跟其他知识库的区别:
    - 论文摘要知识库: 粒度是"扫过的论文"
    - 论文分析知识库: 粒度是"分析过的论文"
    - 论文综述库:     粒度是"**一次 run 的**主题 + 章节"(会随 run 变)
    - 信息知识库:     粒度是"事实", 一行一条
    - **主题地图:     粒度是"结构本身"** —— 主题是图谱节点, 论文不是节点(见
      theme_graph 模块头), 这里是唯一能把三层同时看全的地方

数据全部走 agent.infra.graph_query, 一次查完进内存(实测 5 个查询共 0.09s),
切页/筛选不重新查库。

**为什么不懒加载**: 1926 条归属一次建成 2113 个 QTreeWidgetItem 毫无压力,
而懒加载会引入"展开时才发现某主题没数据"这种更难查的问题 —— 省下的 20ms 不值。

**静默截断是这个窗口最该避免的事**: graph_query 的共享论文查询默认 limit=200,
真库有 441 条边。所以这里显式传 CROSS_LIMIT, 并且把真实条数写进标签,
让「只显示了前 N 条」这件事用户看得见, 而不是以为就这么多。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from agent.infra import graph_query as gq
from agent.tools import facts as F
from ui.styles import get_stylesheet
from ui.theme_graph_view import ThemeGraphView

# 跨域边真库 441 条, graph_query 默认只给 200 —— 显式放大并在标签里报真实条数
CROSS_LIMIT = 1000
_FACT_PREVIEW = 90          # 树叶子上显示多少字, 全文进 tooltip


class ThemeMapDialog(QDialog):
    """主题地图: 宏观主题 / 主题 / 核心对象 三层 + 跨域关联。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("主题地图")
        self.resize(1280, 800)
        self.setStyleSheet(get_stylesheet())
        # 非模态: 边看主题地图边切回主窗口写东西, 别把用户锁死
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)

        # 反向索引: fact_id -> 主题名列表(由 facts_by_theme 顺手反转, 不额外查库)
        self._fact_themes: dict[str, list[str]] = {}
        # 主题名 -> 与它共享论文的边
        self._edge_index: dict[str, list[dict]] = {}

        v = QVBoxLayout(self)
        self.stats_label = QLabel()
        v.addWidget(self.stats_label)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tree_tab(), "结构树")
        self.tabs.addTab(self._build_cross_tab(), "跨域关联")
        self.graph_view = ThemeGraphView()
        self.tabs.addTab(self.graph_view, "关联图")
        v.addWidget(self.tabs, 1)

        self._load()

    # ============================================================
    # 界面搭建
    # ============================================================

    def _build_tree_tab(self) -> QWidget:
        page = QWidget()
        pv = QVBoxLayout(page)

        top = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(
            "搜索宏观主题 / 主题 / 核心对象（回车筛选）"
        )
        self.search_edit.returnPressed.connect(self._apply_filter)
        self.search_edit.textChanged.connect(self._apply_filter)
        top.addWidget(self.search_edit, 3)
        btn_expand = QPushButton("展开全部")
        btn_expand.clicked.connect(self.tree_expand_all)
        btn_collapse = QPushButton("折叠到宏观")
        btn_collapse.clicked.connect(lambda: self.tree.collapseAll())
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self._load)
        top.addWidget(btn_expand)
        top.addWidget(btn_collapse)
        top.addWidget(btn_refresh)
        pv.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["结构(宏观主题 → 主题 → 核心对象)", "计数"])
        self.tree.setColumnWidth(0, 760)
        self.tree.setAlternatingRowColors(True)
        self.tree.itemSelectionChanged.connect(self._on_tree_select)
        split.addWidget(self.tree)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.tree_detail = QPlainTextEdit()
        self.tree_detail.setReadOnly(True)
        self.tree_detail.setPlaceholderText("选中左边任意一层, 这里显示详情。")
        rv.addWidget(self.tree_detail, 1)
        split.addWidget(right)
        split.setSizes([800, 480])
        pv.addWidget(split, 1)
        return page

    def _build_cross_tab(self) -> QWidget:
        page = QWidget()
        pv = QVBoxLayout(page)

        self.cross_note = QLabel()
        pv.addWidget(self.cross_note)

        split = QSplitter(Qt.Horizontal)
        left = QSplitter(Qt.Vertical)

        left.addWidget(self._table_block(
            "共享论文的主题对", ["主题 A", "主题 B", "共享论文", "论文"],
            "pair_table", [200, 200, 80, 320]))
        left.addWidget(self._table_block(
            "桥接主题(同时和多个主题共享论文)", ["主题", "跨接分支", "连接到"],
            "bridge_table", [200, 80, 520]))
        left.setSizes([420, 240])
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.cross_detail = QPlainTextEdit()
        self.cross_detail.setReadOnly(True)
        self.cross_detail.setPlaceholderText("选中左边任意一行, 这里显示详情。")
        rv.addWidget(self.cross_detail, 1)
        split.addWidget(right)
        split.setSizes([820, 460])
        pv.addWidget(split, 1)
        return page

    def _table_block(self, title, headers, attr, widths) -> QWidget:
        """一个标题 + 一张表的小块, 建好后挂到 self.<attr> 上。

        列宽显式给: 不在填完数据后调 resizeColumnsToContents() —— 那会连 Stretch
        段一起改, 同一份数据每次列宽都可能不一样, 没法截图对比。
        """
        box = QWidget()
        bv = QVBoxLayout(box)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.addWidget(QLabel(title))
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        for i, w in enumerate(widths):
            table.setColumnWidth(i, w)
        bv.addWidget(table, 1)
        setattr(self, attr, table)
        return box

    # ============================================================
    # 数据
    # ============================================================

    def _fetch(self) -> dict:
        """一次把界面要用的东西全查出来(实测 5 个查询共 0.09s)。

        刻意跟 `_load` 分开: `_load` 里的错误处理会弹 QMessageBox, 而 modal 框在
        没有事件循环的测试里是**挂死**而不是失败 —— 一条取数 bug 会变成一个
        超时的 CI, 而不是一条红色断言。取数逻辑单独可测, 弹框只是它的外衣。
        """
        return {
            "tree": gq.tree(),
            "overview": gq.overview(),
            "by_theme": gq.facts_by_theme(),
            "edges": gq.themes_sharing_papers(min_papers=1, limit=CROSS_LIMIT),
            "bridges": gq.bridges(min_papers=2, limit=CROSS_LIMIT),
        }

    def _load(self):
        try:
            d = self._fetch()
        except Exception as e:  # 库没建 / 被锁 / schema 旧
            QMessageBox.critical(self, "加载失败", str(e))
            self.stats_label.setText(f"加载失败: {e}")
            return
        tree = d["tree"]
        edges = d["edges"]
        bridges = d["bridges"]

        self._fact_themes = {}
        for theme, items in d["by_theme"].items():
            for it in items:
                self._fact_themes.setdefault(it["fact_id"], []).append(theme)

        self._edge_index = {}
        for e in edges:
            self._edge_index.setdefault(e["theme_a"], []).append(e)
            self._edge_index.setdefault(e["theme_b"], []).append(
                {"theme_a": e["theme_a"], "theme_b": e["theme_b"],
                 "n_papers": e["n_papers"], "papers": e["papers"],
                 "other": e["theme_a"]})
        for lst in self._edge_index.values():
            for e in lst:
                e.setdefault("other", e["theme_b"])

        self._fill_stats(d["overview"], tree, edges, bridges)
        self._fill_tree(tree, d["by_theme"])
        self._fill_cross(edges, bridges)
        self._fill_graph(tree, edges)
        self._apply_filter()

    def _fill_graph(self, tree, edges):
        """喂点线图。节点=主题(带所属宏观 + 核心对象数, 决定圆点大小), 边=共享论文。

        `tree` 里同一个主题名只可能出现一次(全局唯一键), 但仍然去个重 —— 万一
        将来有游离组和某个宏观组重名, 同一个节点被 addItem 两次会让点选行为变得
        不可预测(叠着的两个 item 抢点击)。
        """
        nodes: list[dict] = []
        seen: set[str] = set()
        for group in tree:
            for t in group["themes"]:
                name = t["name"]
                if name in seen:
                    continue
                seen.add(name)
                nodes.append({"name": name, "macro": group.get("macro"),
                              "n_facts": t.get("n_facts", 0)})
        # 字段名要对齐: graph_query 出的是 theme_a/theme_b, 视图层要的是 a/b。
        # 直接透传的话视图里 `e.get("a")` 全是 None, 一条边都画不出来 —— 而且
        # 场景里节点照常在, 看着"画成功了", 只是关联功能整个是死的。
        self.graph_view.set_data(nodes, [
            {"a": e["theme_a"], "b": e["theme_b"], "weight": e["n_papers"]}
            for e in edges
        ])

    def _fill_stats(self, overview, tree, edges, bridges):
        span = overview.get("paper_span", {})
        unlinked = overview.get("facts_total", 0) - overview.get("facts_linked", 0)
        self.stats_label.setText(
            f"宏观主题 {overview.get('macros', 0)} · 主题 {overview.get('themes', 0)} · "
            f"核心对象 {overview.get('facts_total', 0)} · 关系边 {overview.get('relations', 0)}"
            f"　│　论文 {span.get('total_papers', 0)}，其中横跨 ≥2 个主题的 "
            f"{span.get('multi_theme_papers', 0)} 篇"
            + (f"　│　⚠ 尚未归入任何主题的核心对象 {unlinked} 条" if unlinked else "")
        )
        self.cross_note.setText(
            f"跨域关联的边 = 两个主题**各自吃下了同一篇论文的不同侧面**(join 键是论文"
            f"不是 fact)。当前显示 {len(edges)} 条"
            + (f"(已达查询上限 {CROSS_LIMIT}, 库里更多, 已截断)"
               if len(edges) >= CROSS_LIMIT else "(未截断)")
            + f"；桥接主题 {len(bridges)} 个。这些是「哪几块在互相借方法」的答案。"
        )

    def _fill_tree(self, tree, by_theme):
        self.tree.clear()
        for group in tree:
            macro = group.get("macro")
            label = macro if macro else "未归入宏观主题"
            m = QTreeWidgetItem(self.tree, [label, f"{len(group['themes'])} 主题"])
            stmt = group.get("statement", "")
            m.setData(0, _role(), {"kind": "macro", "name": label,
                                   "statement": stmt,
                                   "search": f"{label} {stmt}",
                                   "themes": [t["name"] for t in group["themes"]]})
            m.setFirstColumnSpanned(False)
            for t in group["themes"]:
                name = t["name"]
                tstmt = t.get("statement", "")
                n_p = t.get("n_papers")
                cnt = f"{t.get('n_facts', 0)} 对象"
                if n_p is not None:
                    cnt += f" / {n_p} 论文"
                it = QTreeWidgetItem(m, [name, cnt])
                it.setData(0, _role(), {"kind": "theme", "name": name,
                                        "statement": tstmt,
                                        "search": f"{name} {tstmt}"})
                for f in by_theme.get(name, []):
                    content = f["content"] or ""
                    leaf = QTreeWidgetItem(it, [
                        f"[{f.get('fact_type') or 'fact'}] "
                        f"{content.replace(chr(10), ' ')[:_FACT_PREVIEW]}",
                        f"{f.get('n_sources', 0)} 出处"])
                    leaf.setToolTip(0, content)
                    # search 必须带上正文: 搜索框承诺能搜"核心对象", 不给内容
                    # 就只有主题名能命中, 那是界面在说谎
                    leaf.setData(0, _role(), {"kind": "fact",
                                              "fact_id": f["fact_id"],
                                              "search": content})
        # 默认只展开到宏观层: 1926 个叶节点全展开没必要, 要看再点
        self.tree.expandToDepth(0)

    def _fill_cross(self, edges, bridges):
        self.pair_table.setRowCount(len(edges))
        for r, e in enumerate(edges):
            self.pair_table.setItem(r, 0, QTableWidgetItem(e["theme_a"]))
            self.pair_table.setItem(r, 1, QTableWidgetItem(e["theme_b"]))
            self.pair_table.setItem(r, 2, QTableWidgetItem(str(e["n_papers"])))
            ps = ", ".join(e["papers"][:8])
            if len(e["papers"]) > 8:
                ps += f" …(共 {len(e['papers'])} 篇)"
            item = QTableWidgetItem(ps)
            item.setToolTip(", ".join(e["papers"]))
            self.pair_table.setItem(r, 3, item)
        self.pair_table.itemSelectionChanged.connect(self._on_pair_select)

        self.bridge_table.setRowCount(len(bridges))
        for r, b in enumerate(bridges):
            self.bridge_table.setItem(r, 0, QTableWidgetItem(b["theme"]))
            self.bridge_table.setItem(r, 1, QTableWidgetItem(str(b["n_branches"])))
            item = QTableWidgetItem(", ".join(b["connects"]))
            item.setToolTip(", ".join(b["connects"]))
            self.bridge_table.setItem(r, 2, item)
        self.bridge_table.itemSelectionChanged.connect(self._on_bridge_select)

    # ============================================================
    # 筛选
    # ============================================================

    def _apply_filter(self):
        needle = self.search_edit.text().strip().lower()
        if not needle:
            self._set_subtree_visible(self.tree.invisibleRootItem(), True)
            return
        self._filter_recursive(self.tree.invisibleRootItem(), needle)
        self.tree.expandAll()

    def _filter_recursive(self, item, needle) -> bool:
        """子树是否保留(自身命中 或 任一后代命中)。自身命中则整棵子树放出来。"""
        payload = item.data(0, _role()) or {}
        if needle in str(payload.get("search", "")).lower():
            self._set_subtree_visible(item, True)
            return True
        keep = False
        for i in range(item.childCount()):
            if self._filter_recursive(item.child(i), needle):
                keep = True
        item.setHidden(not keep)
        return keep

    def _set_subtree_visible(self, item, visible):
        item.setHidden(not visible)
        for i in range(item.childCount()):
            self._set_subtree_visible(item.child(i), visible)

    def tree_expand_all(self):
        self.tree.expandAll()

    # ============================================================
    # 详情
    # ============================================================

    def _on_tree_select(self):
        item = self.tree.currentItem()
        if item is None:
            return
        payload = item.data(0, _role()) or {}
        kind = payload.get("kind")
        try:
            if kind == "fact":
                self._show_fact(payload["fact_id"])
            elif kind == "theme":
                self._show_theme(payload["name"], payload.get("statement", ""))
            else:
                self._show_macro(payload)
        except Exception as e:
            self.tree_detail.setPlainText(f"读取详情失败: {e}")

    def _show_macro(self, payload):
        lines = [f"【宏观主题】{payload.get('name', '')}"]
        st = payload.get("statement", "")
        if st:
            lines += ["", st]
        lines += ["", f"下辖主题 {len(payload.get('themes', []))} 个:"]
        lines += [f"  · {n}" for n in payload.get("themes", [])]
        self.tree_detail.setPlainText("\n".join(lines))

    def _show_theme(self, name, statement):
        facts = gq.facts_by_theme(name).get(name, [])
        papers = gq.papers_of_theme(name)
        lines = [f"【主题】{name}"]
        if statement:
            lines += ["", statement]
        lines += ["", f"核心对象 {len(facts)} 条 · 论文 {len(papers)} 篇"]
        edges = self._edge_index.get(name, [])
        lines += ["", f"跨域关联 {len(edges)} 条(与其它主题共享论文):"]
        if not edges:
            lines.append("  (这条主题还没和任何主题共享过论文)")
        for e in sorted(edges, key=lambda x: -x["n_papers"])[:20]:
            ps = ", ".join(e["papers"][:6])
            if len(e["papers"]) > 6:
                ps += f" …(共 {len(e['papers'])})"
            lines.append(f"  · {e['other']}  共享 {e['n_papers']} 篇: {ps}")
        if len(edges) > 20:
            lines.append(f"  …另有 {len(edges) - 20} 条, 见「跨域关联」页")
        self.tree_detail.setPlainText("\n".join(lines))

    def _show_fact(self, fact_id):
        f = F.get_fact(fact_id)
        if not f:
            self.tree_detail.setPlainText(f"找不到 fact {fact_id}")
            return
        lines = [
            f"【核心对象】[{f.get('fact_type', '')}] {f.get('content', '')}",
            "",
            f"核心对象(subject): {f.get('subject', '') or '(未填)'}",
            f"fact_id: {f['id']}",
            f"所属主题: {', '.join(self._fact_themes.get(fact_id, [])) or '(无)'}",
            "",
            f"出处 {len(f.get('sources', []))} 个:",
        ]
        for i, s in enumerate(f.get("sources", []), 1):
            mark = "" if s.get("grounded") else "   ⚠ 未通过落地审查"
            lines.append(f"  [{i}] {s.get('source_type', '')} / "
                         f"{s.get('source_id', '')}{mark}")
        self.tree_detail.setPlainText("\n".join(lines))

    def _on_pair_select(self):
        r = self.pair_table.currentRow()
        if r < 0:
            return
        a = self.pair_table.item(r, 0)
        b = self.pair_table.item(r, 1)
        if not a or not b:
            return
        edge = next((e for e in self._edge_index.get(a.text(), [])
                     if e["other"] == b.text()), None)
        if edge is None:
            self.cross_detail.setPlainText("读取边失败")
            return
        lines = [
            f"【跨域关联】{a.text()}  ←共享论文→  {b.text()}",
            "",
            f"共同吃下 {edge['n_papers']} 篇论文:",
        ]
        # 这半句才是重点: 不是"两主题相似", 而是"同一篇论文被这两块切成了不同侧面"
        for pid in edge["papers"][:6]:
            parts = gq.themes_of_paper(pid)
            lines.append(f"  · {pid}")
            for p in parts:
                lines.append(f"      → {p['theme']}: {p['n_facts']} 条")
        if len(edge["papers"]) > 6:
            lines.append(f"  …另有 {len(edge['papers']) - 6} 篇")
        self.cross_detail.setPlainText("\n".join(lines))

    def _on_bridge_select(self):
        r = self.bridge_table.currentRow()
        if r < 0:
            return
        it = self.bridge_table.item(r, 0)
        if not it:
            return
        name = it.text()
        edges = self._edge_index.get(name, [])
        lines = [
            f"【桥接主题】{name}",
            "",
            f"同时和 {self.bridge_table.item(r, 1).text()} 个主题共享论文 —— "
            f"这个领域里它借用/被借了这么多方向, 是最值得单独看的一块。",
            "",
        ]
        for e in sorted(edges, key=lambda x: -x["n_papers"])[:30]:
            lines.append(f"  · {e['other']}  共享 {e['n_papers']} 篇论文")
        if len(edges) > 30:
            lines.append(f"  …另有 {len(edges) - 30} 条")
        self.cross_detail.setPlainText("\n".join(lines))


def _role():
    """QTreeWidget 的自定义 data role(集中一处, 免得两处各写一个数字)。"""
    return Qt.UserRole
