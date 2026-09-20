"""
ui.memory_viewer_dialog — 记忆库查看窗口（v3.1 加主题聚合 tab）

仿照 KnowledgeBaseDialog 结构（PySide6 QDialog + QTableWidget）。
支持：
- Tab 1「全部」：按 query / memory_type / min_importance 过滤 (v1+v2 功能)
- Tab 2「主题聚合」：按 metadata.extract.core_concepts[0].canonical 分组浏览
                    左: 概念树 (canonical + 数量 + category)
                    右: 选中概念的对应记忆列表 (同 7 列表格)
- 状态栏：总数 + embedder 状态 (vector/keyword-only)
- 双击记忆行：删除单条（带确认）

调用入口：chat_window.py 菜单"记忆库"
"""
import json
import logging
from collections import defaultdict
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.styles import get_stylesheet

logger = logging.getLogger(__name__)


# ============================================================
# 列定义
# ============================================================
COLUMNS = [
    ("id", "ID", 70),
    ("memory_type", "类型", 120),
    ("content", "内容", 360),
    ("importance", "重要性", 60),
    ("related_paper_id", "关联论文", 130),
    ("created_at", "创建时间", 140),
    ("access_count", "引用", 50),
]
COL_ID = 0
COL_TYPE = 1
COL_CONTENT = 2
COL_IMPORTANCE = 3
COL_PAPER = 4
COL_CREATED = 5
COL_ACCESS = 6


class MemoryViewerDialog(QDialog):
    """记忆库查看窗口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("记忆库")
        self.setMinimumSize(1100, 750)
        # 2026-09-02 改:用 get_stylesheet() 跟主题(切主题时刷新)
        self.setStyleSheet(get_stylesheet())
        self._processor = None  # lazy

        # ============================================================
        # 顶部：QTabWidget (全部 / 主题聚合)
        # ============================================================
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: 全部
        self.tab_all = QWidget()
        self.tabs.addTab(self.tab_all, "全部")
        self._build_all_tab(self.tab_all)

        # Tab 2: 主题聚合
        self.tab_group = QWidget()
        self.tabs.addTab(self.tab_group, "主题聚合")
        self._build_group_tab(self.tab_group)

        # 状态栏
        self.status = QStatusBar()
        layout.addWidget(self.status)

        # 初次加载
        self.load_all()
        self.load_concept_groups()

    # ============================================================
    # Tab 1: 全部 (原 v1+v2 功能)
    # ============================================================
    def _build_all_tab(self, parent_widget):
        v = QVBoxLayout(parent_widget)

        search_layout = QHBoxLayout()
        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("搜索记忆内容（自然语言 / 关键词均可）...")
        self.query_edit.returnPressed.connect(self.perform_search)

        self.type_combo = QComboBox()
        self.type_combo.addItem("全部类型", "")
        from agent.memory.schemas import MEMORY_TYPES
        for t in MEMORY_TYPES:
            self.type_combo.addItem(MEMORY_TYPES[t].split("（")[0] if "（" in MEMORY_TYPES[t] else t, t)

        self.min_imp_spin = QSpinBox()
        self.min_imp_spin.setRange(0, 100)
        self.min_imp_spin.setValue(0)
        self.min_imp_spin.setSuffix(" %")
        self.min_imp_spin.setToolTip("最低 importance (0-100)")

        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self.perform_search)

        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.load_all)

        search_layout.addWidget(QLabel("搜索:"))
        search_layout.addWidget(self.query_edit, 3)
        search_layout.addWidget(QLabel("类型:"))
        search_layout.addWidget(self.type_combo, 1)
        search_layout.addWidget(QLabel("重要性≥"))
        search_layout.addWidget(self.min_imp_spin, 0)
        search_layout.addWidget(search_btn)
        search_layout.addWidget(refresh_btn)
        search_layout.addStretch()
        v.addLayout(search_layout)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(len(COLUMNS))
        self.result_table.setHorizontalHeaderLabels([c[1] for c in COLUMNS])
        self.result_table.setColumnWidth(COL_ID, COLUMNS[COL_ID][2])
        self.result_table.setColumnWidth(COL_TYPE, COLUMNS[COL_TYPE][2])
        self.result_table.setColumnWidth(COL_CONTENT, COLUMNS[COL_CONTENT][2])
        self.result_table.setColumnWidth(COL_IMPORTANCE, COLUMNS[COL_IMPORTANCE][2])
        self.result_table.setColumnWidth(COL_PAPER, COLUMNS[COL_PAPER][2])
        self.result_table.setColumnWidth(COL_CREATED, COLUMNS[COL_CREATED][2])
        self.result_table.setColumnWidth(COL_ACCESS, COLUMNS[COL_ACCESS][2])
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.doubleClicked.connect(self.on_row_double_clicked)
        self.result_table.setSortingEnabled(True)
        v.addWidget(self.result_table)

    # ============================================================
    # Tab 2: 主题聚合 (v3.1 新功能)
    # ============================================================
    def _build_group_tab(self, parent_widget):
        v = QVBoxLayout(parent_widget)

        # 顶部说明
        tip = QLabel(
            "按 v3 抽取的核心概念 (metadata.extract.core_concepts[0].canonical) 分组。"
            "只有通过 LLM 抽取的记忆才会出现在这里。"
        )
        tip.setWordWrap(True)
        # 2026-09-02 改:用 role="muted" 走主题 token,不再硬编码 gray
        tip.setProperty("role", "muted")
        v.addWidget(tip)

        # 刷新按钮
        top = QHBoxLayout()
        refresh_group_btn = QPushButton("刷新主题")
        refresh_group_btn.clicked.connect(self.load_concept_groups)
        top.addWidget(refresh_group_btn)
        top.addStretch()
        self.group_stats_label = QLabel("")
        top.addWidget(self.group_stats_label)
        v.addLayout(top)

        # Splitter: 左 TreeWidget + 右 TableWidget
        splitter = QSplitter(Qt.Horizontal)
        v.addWidget(splitter, 1)

        # 左侧: 概念树
        self.concept_tree = QTreeWidget()
        self.concept_tree.setHeaderLabels(["概念 (canonical)", "类别", "数量"])
        self.concept_tree.setColumnWidth(0, 200)
        self.concept_tree.setColumnWidth(1, 130)
        self.concept_tree.setColumnWidth(2, 50)
        self.concept_tree.itemClicked.connect(self.on_concept_clicked)
        splitter.addWidget(self.concept_tree)

        # 右侧: 该概念对应的记忆表
        self.group_table = QTableWidget()
        self.group_table.setColumnCount(len(COLUMNS))
        self.group_table.setHorizontalHeaderLabels([c[1] for c in COLUMNS])
        self.group_table.setColumnWidth(COL_ID, COLUMNS[COL_ID][2])
        self.group_table.setColumnWidth(COL_TYPE, COLUMNS[COL_TYPE][2])
        self.group_table.setColumnWidth(COL_CONTENT, COLUMNS[COL_CONTENT][2])
        self.group_table.setColumnWidth(COL_IMPORTANCE, COLUMNS[COL_IMPORTANCE][2])
        self.group_table.setColumnWidth(COL_PAPER, COLUMNS[COL_PAPER][2])
        self.group_table.setColumnWidth(COL_CREATED, COLUMNS[COL_CREATED][2])
        self.group_table.setColumnWidth(COL_ACCESS, COLUMNS[COL_ACCESS][2])
        self.group_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.group_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.group_table.doubleClicked.connect(self.on_row_double_clicked)
        self.group_table.setSortingEnabled(True)
        splitter.addWidget(self.group_table)

        splitter.setSizes([350, 750])

    # ============================================================
    # 加载数据
    # ============================================================
    def _get_processor(self):
        if self._processor is None:
            from agent.memory.processor import MemoryProcessor
            self._processor = MemoryProcessor()
        return self._processor

    def load_all(self):
        """加载所有记忆（按 importance 排序）。"""
        memories = self._get_processor().list_all(limit=500)
        self._display(memories, mode="all", table=self.result_table)

    def perform_search(self):
        """按 query / type / min_imp 搜索。"""
        query = self.query_edit.text().strip()
        mtype = self.type_combo.currentData() or None
        min_imp = self.min_imp_spin.value() / 100.0 if self.min_imp_spin.value() > 0 else None

        try:
            if query:
                memories = self._get_processor().recall(
                    query=query, limit=200,
                    memory_type=mtype, min_importance=min_imp,
                )
                self._display(memories, mode=f"search:{query[:30]}", table=self.result_table)
            else:
                memories = self._get_processor().list_all(
                    memory_type=mtype, min_importance=min_imp, limit=200,
                )
                self._display(memories, mode=f"filter:type={mtype or '*'},imp>={min_imp or 0}", table=self.result_table)
        except Exception as e:
            logger.exception("[MemoryViewer] search failed")
            QMessageBox.critical(self, "检索失败", str(e))

    def _display(self, memories: list[dict[str, Any]], mode: str = "", table: QTableWidget | None = None):
        """填充表格 + 更新状态栏。"""
        if table is None:
            table = self.result_table
        table.setSortingEnabled(False)  # 关闭排序避免覆盖我们的顺序
        table.setRowCount(len(memories))
        for row, m in enumerate(memories):
            mid = m.get("id", "")[:8]
            mtype = m.get("memory_type") or "general"
            content = m.get("content", "")
            if len(content) > 120:
                content = content[:120] + "..."
            importance = m.get("importance", 0.5)
            paper = m.get("related_paper_id") or ""
            created_at = m.get("created_at", "")
            if created_at:
                # 格式化：'YYYY-MM-DD HH:MM:SS'
                created_at = created_at.replace("T", " ").split(".")[0]
            access = m.get("access_count", 0)

            table.setItem(row, COL_ID, QTableWidgetItem(mid))
            table.setItem(row, COL_TYPE, QTableWidgetItem(mtype))
            content_item = QTableWidgetItem(content)
            content_item.setToolTip(m.get("content", ""))  # 完整内容 hover 看
            table.setItem(row, COL_CONTENT, content_item)
            table.setItem(row, COL_IMPORTANCE, QTableWidgetItem(f"{importance:.2f}"))
            table.setItem(row, COL_PAPER, QTableWidgetItem(paper))
            table.setItem(row, COL_CREATED, QTableWidgetItem(created_at))
            table.setItem(row, COL_ACCESS, QTableWidgetItem(str(access)))

        table.setSortingEnabled(True)

        # 状态栏 (只在主表刷新时更新)
        if table is self.result_table:
            embedder = self._get_processor().embedder
            backend = embedder.get_backend_info()
            mode_str = "vector" if backend.get("enabled") else "keyword-only"
            dim = backend.get("dim", "?")
            self.status.showMessage(
                f"  共 {len(memories)} 条  |  当前视图: {mode}  |  检索模式: {mode_str} (dim={dim})  |  "
                f"双击删除"
            )

    # ============================================================
    # 主题聚合: 构建概念树
    # ============================================================
    def load_concept_groups(self):
        """遍历所有记忆, 按 metadata.extract.core_concepts[0].canonical 分组."""
        try:
            memories = self._get_processor().list_all(limit=1000)
        except Exception as e:
            logger.exception("[MemoryViewer] list_all failed")
            QMessageBox.critical(self, "加载失败", str(e))
            return

        # canonical -> {category, items: [mem, ...]}
        groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"category": "?", "items": []})
        no_extract_count = 0
        for m in memories:
            meta_raw = m.get("metadata")
            if not meta_raw:
                no_extract_count += 1
                continue
            # metadata 可能是 dict 或 json string (取决于 db_manager 怎么处理)
            if isinstance(meta_raw, str):
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    no_extract_count += 1
                    continue
            else:
                meta = meta_raw
            extract = meta.get("extract") if isinstance(meta, dict) else None
            if not extract:
                no_extract_count += 1
                continue
            concepts = extract.get("core_concepts") or []
            if not concepts:
                no_extract_count += 1
                continue
            # 取第一个 concept 作为"主题" (按重要性排, list 已经按 importance 排了, 所以第一个 = 最重要的记忆的主概念)
            primary = concepts[0]
            canonical = primary.get("canonical") or "?"
            category = primary.get("category") or "?"
            groups[canonical]["category"] = category
            groups[canonical]["items"].append(m)

        # 填 TreeWidget
        self.concept_tree.clear()
        # 按数量降序排
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]["items"]))
        for canonical, info in sorted_groups:
            count = len(info["items"])
            item = QTreeWidgetItem([canonical, info["category"], str(count)])
            item.setData(0, Qt.UserRole, info["items"])  # 存该组所有记忆
            self.concept_tree.addTopLevelItem(item)

        # 统计 label
        total_with_extract = sum(len(g["items"]) for g in groups.values())
        self.group_stats_label.setText(
            f"共 {len(groups)} 个概念 / {total_with_extract} 条已抽取记忆  |  {no_extract_count} 条未抽取 (v3 之前)"
        )

        # 清空右侧
        self._display([], mode="group:?", table=self.group_table)
        self.status.showMessage(
            f"  主题聚合: {len(groups)} 个概念, {no_extract_count} 条未抽取 (跑回填脚本)", 5000
        )

    def on_concept_clicked(self, item: QTreeWidgetItem, column: int):
        """点击概念 → 右侧显示该组记忆."""
        memories = item.data(0, Qt.UserRole) or []
        canonical = item.text(0)
        self._display(memories, mode=f"concept:{canonical}", table=self.group_table)
        self.status.showMessage(
            f"  概念「{canonical}」: {len(memories)} 条记忆"
        )

    # ============================================================
    # 行操作
    # ============================================================
    def on_row_double_clicked(self, index):
        """双击行 = 删除单条（带确认）。两个 tab 的表都连接到这里。"""
        # 找到是哪个 QTableWidget 触发的
        table: QTableWidget = index.model()  # type: ignore[assignment]
        if not isinstance(table, QTableWidget):
            table = self.result_table
        row = index.row()
        id_item = table.item(row, COL_ID)
        if not id_item:
            return
        short_id = id_item.text()
        content_item = table.item(row, COL_CONTENT)
        content_preview = content_item.text() if content_item else ""

        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除这条记忆？\n\nID: {short_id}...\n内容: {content_preview}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            memories = self._get_processor().list_all(limit=500)
            full_id = None
            for m in memories:
                if m.get("id", "").startswith(short_id):
                    full_id = m["id"]
                    break
            if not full_id:
                QMessageBox.warning(self, "删除失败", f"找不到 id={short_id} 的完整记录")
                return
            self._get_processor().delete(full_id)
            self.status.showMessage(f"  已删除 {short_id}...", 3000)
            # 刷新两个 tab
            if table is self.result_table:
                if self.query_edit.text().strip():
                    self.perform_search()
                else:
                    self.load_all()
            else:
                # 主题聚合 tab: 刷新树 + 右边表
                self.load_concept_groups()
        except Exception as e:
            logger.exception("[MemoryViewer] delete failed")
            QMessageBox.critical(self, "删除失败", str(e))
