"""
ui.memory_viewer_dialog — 记忆库查看窗口

仿照 KnowledgeBaseDialog 结构（PySide6 QDialog + QTableWidget）。
支持：
- 顶部搜索栏：按 query / memory_type / min_importance 过滤
- 表格列：id / type / content(前 100 字符) / importance / created / paper_id
- 状态栏：总数 + embedder 状态（vector/keyword-only）
- 双击行：删除单条（带确认）

调用入口：chat_window.py 菜单"记忆库"
"""
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox,
    QPushButton, QTableWidget, QTableWidgetItem, QMessageBox, QLabel,
    QSpinBox, QStatusBar,
)

from ui.styles import GLOBAL_STYLESHEET

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
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(GLOBAL_STYLESHEET)
        self._processor = None  # lazy

        # ============================================================
        # 顶部：搜索 + 过滤
        # ============================================================
        layout = QVBoxLayout(self)

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
        layout.addLayout(search_layout)

        # ============================================================
        # 中部：结果表格
        # ============================================================
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
        layout.addWidget(self.result_table)

        # ============================================================
        # 底部：状态栏
        # ============================================================
        self.status = QStatusBar()
        layout.addWidget(self.status)

        # 初次加载
        self.load_all()

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
        self._display(memories, mode="all")

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
                self._display(memories, mode=f"search:{query[:30]}")
            else:
                memories = self._get_processor().list_all(
                    memory_type=mtype, min_importance=min_imp, limit=200,
                )
                self._display(memories, mode=f"filter:type={mtype or '*'},imp>={min_imp or 0}")
        except Exception as e:
            logger.exception("[MemoryViewer] search failed")
            QMessageBox.critical(self, "检索失败", str(e))

    def _display(self, memories: List[Dict[str, Any]], mode: str = ""):
        """填充表格 + 更新状态栏。"""
        self.result_table.setSortingEnabled(False)  # 关闭排序避免覆盖我们的顺序
        self.result_table.setRowCount(len(memories))
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

            self.result_table.setItem(row, COL_ID, QTableWidgetItem(mid))
            self.result_table.setItem(row, COL_TYPE, QTableWidgetItem(mtype))
            content_item = QTableWidgetItem(content)
            content_item.setToolTip(m.get("content", ""))  # 完整内容 hover 看
            self.result_table.setItem(row, COL_CONTENT, content_item)
            self.result_table.setItem(row, COL_IMPORTANCE, QTableWidgetItem(f"{importance:.2f}"))
            self.result_table.setItem(row, COL_PAPER, QTableWidgetItem(paper))
            self.result_table.setItem(row, COL_CREATED, QTableWidgetItem(created_at))
            self.result_table.setItem(row, COL_ACCESS, QTableWidgetItem(str(access)))

        self.result_table.setSortingEnabled(True)

        # 状态栏
        embedder = self._get_processor().embedder
        backend = embedder.get_backend_info()
        mode_str = "vector" if backend.get("enabled") else "keyword-only"
        dim = backend.get("dim", "?")
        self.status.showMessage(
            f"  共 {len(memories)} 条  |  当前视图: {mode}  |  检索模式: {mode_str} (dim={dim})  |  "
            f"双击删除"
        )

    # ============================================================
    # 行操作
    # ============================================================
    def on_row_double_clicked(self, index):
        """双击行 = 删除单条（带确认）。"""
        row = index.row()
        id_item = self.result_table.item(row, COL_ID)
        if not id_item:
            return
        short_id = id_item.text()
        content_item = self.result_table.item(row, COL_CONTENT)
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

        # 找到完整 id
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
            # 刷新
            self.perform_search() if self.query_edit.text().strip() else self.load_all()
        except Exception as e:
            logger.exception("[MemoryViewer] delete failed")
            QMessageBox.critical(self, "删除失败", str(e))
