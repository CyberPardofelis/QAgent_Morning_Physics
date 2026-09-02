"""
ui.summary_kb_dialog — 论文摘要知识库 dialog(2026-09-02 P0.6 配套 UI)

职责:
- 列出 summary_items.db 全部条目(默认按 summary_date 倒序)
- 关键词/标题/作者 模糊搜索 + 语义检索(semantic_search_keywords)
- 双击行 → 弹详情(只读)

数据源:
- 抓文时(arxiv/aps)落库 → summary_items
- analyze_paper 完成后协同更新 keywords/core_method/core_contrib

跟论文分析知识库(KnowledgeBaseDialog)的区别:
- summary_kb: 粒度细(只存"扫过"的论文摘要), 来源多(arxiv_daily/aps_daily/reindex/manual)
- paper_kb: 粒度粗(精细分析过的论文), 只有 analyze_paper 走过才有
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox,
    QPushButton, QTableWidget, QTableWidgetItem, QMessageBox, QLabel, QTextEdit,
)
from ui.styles import get_stylesheet
from agent.tools.summary import list_summary_items, count_summary_items, get_summary_item
from agent.tools.knowledge import semantic_search_keywords


class SummaryKBDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("论文摘要知识库")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(get_stylesheet())
        self._search_mode = "all"  # all / keyword / semantic
        layout = QVBoxLayout(self)

        # ---- 状态条 ----
        status_layout = QHBoxLayout()
        self.status_label = QLabel("")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)
        self._update_status()

        # ---- 搜索区域 ----
        search_layout = QHBoxLayout()
        self.keyword_edit = QLineEdit()
        self.keyword_edit.setPlaceholderText("输入关键词 / 标题 / 模糊查询...")
        self.search_combo = QComboBox()
        self.search_combo.addItems(["全部", "关键词", "语义检索", "来源=arxiv_daily", "来源=aps_daily", "来源=reindex"])
        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self.perform_search)
        self.keyword_edit.returnPressed.connect(self.perform_search)

        search_layout.addWidget(QLabel("搜索:"))
        search_layout.addWidget(self.keyword_edit)
        search_layout.addWidget(self.search_combo)
        search_layout.addWidget(search_btn)
        search_layout.addStretch()
        layout.addLayout(search_layout)

        # ---- 结果表格 ----
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels([
            "论文ID", "标题", "来源", "日期", "已分析", "关键词"
        ])
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.doubleClicked.connect(self.open_detail_dialog)
        layout.addWidget(self.result_table)

        # 初始:列最近 100 条
        self.load_recent()

    def _update_status(self):
        """更新顶部状态条"""
        try:
            n = count_summary_items()
            self.status_label.setText(f"📊 summary_items.db 共 {n} 条")
        except Exception as e:
            self.status_label.setText(f"⚠ 状态查询失败: {e}")

    def load_recent(self):
        """加载最近的 100 条"""
        self._search_mode = "all"
        items = list_summary_items(limit=100)
        self.display_items(items, info="(最近 100 条)")

    def display_items(self, items, info=""):
        """把 summary_item 列表填到表格"""
        self.result_table.setRowCount(len(items))
        for row, it in enumerate(items):
            self.result_table.setItem(row, 0, QTableWidgetItem(it.get("paper_id", "")))
            self.result_table.setItem(row, 1, QTableWidgetItem((it.get("title") or "")[:80]))
            self.result_table.setItem(row, 2, QTableWidgetItem(it.get("source", "")))
            self.result_table.setItem(row, 3, QTableWidgetItem(it.get("summary_date", "")))
            in_db = "✅" if it.get("in_papers_db") else "—"
            self.result_table.setItem(row, 4, QTableWidgetItem(in_db))
            kw = (it.get("keywords") or "")[:60]
            self.result_table.setItem(row, 5, QTableWidgetItem(kw))
        self.result_table.resizeColumnsToContents()
        # 更新状态条
        self.status_label.setText(f"📊 显示 {len(items)} 条 {info}")

    def perform_search(self):
        """执行搜索"""
        keyword = self.keyword_edit.text().strip()
        mode = self.search_combo.currentText()

        try:
            if mode == "全部" or not keyword:
                self.load_recent()
                return

            if mode == "关键词":
                # SQL LIKE: paper_id / title / keywords
                items = list_summary_items(limit=500)
                kw_lower = keyword.lower()
                filtered = [
                    it for it in items
                    if kw_lower in (it.get("paper_id") or "").lower()
                    or kw_lower in (it.get("title") or "").lower()
                    or kw_lower in (it.get("keywords") or "").lower()
                    or kw_lower in (it.get("abstract") or "").lower()
                ]
                self.display_items(filtered, info=f"(关键词匹配 '{keyword}', {len(filtered)} 条)")
                return

            if mode == "语义检索":
                # 走 paper_keywords collection 语义检索
                md_result = semantic_search_keywords(query=keyword, top_k=30)
                # 解析输出,拿 paper_id 列表
                pids = []
                for line in md_result.split("\n"):
                    if "- ID: `" in line:
                        pid = line.split("- ID: `")[1].split("`")[0]
                        if pid not in pids:
                            pids.append(pid)
                if not pids:
                    QMessageBox.information(self, "提示", f"未找到匹配 '{keyword}' 的论文")
                    return
                # 用 paper_id 列表从 summary_items 拿详情
                items = list_summary_items(limit=500)
                pid_set = set(pids)
                filtered = [it for it in items if it.get("paper_id") in pid_set]
                # 按 pids 顺序排序
                filtered.sort(key=lambda it: pids.index(it.get("paper_id", "")) if it.get("paper_id") in pids else 999)
                self.display_items(filtered, info=f"(语义检索 '{keyword}', top {len(filtered)})")
                return

            if mode.startswith("来源="):
                src = mode.split("=")[1]
                items = list_summary_items(source=src, limit=500)
                self.display_items(items, info=f"(来源={src}, {len(items)} 条)")
                return

        except Exception as e:
            QMessageBox.critical(self, "搜索错误", f"搜索失败: {e}")
            return

    def open_detail_dialog(self, index):
        """双击行:弹详情 dialog(只读)"""
        row = index.row()
        pid_item = self.result_table.item(row, 0)
        if not pid_item:
            return
        paper_id = pid_item.text()
        # 找对应 summary_item
        # 用 list_summary_items 拿一个,然后按 paper_id 过滤
        items = list_summary_items(limit=1000)
        target = next((it for it in items if it.get("paper_id") == paper_id), None)
        if not target:
            QMessageBox.warning(self, "错误", f"找不到 {paper_id} 的详情")
            return
        # 弹详情
        dlg = SummaryItemDetailDialog(target, self)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.show()


class SummaryItemDetailDialog(QDialog):
    """summary_item 详情 dialog(只读)"""
    def __init__(self, item: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"摘要详情: {item.get('paper_id', '?')}")
        self.setMinimumSize(700, 500)
        self.setStyleSheet(get_stylesheet())
        layout = QVBoxLayout(self)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setMarkdown(self._render(item))
        layout.addWidget(text)

        # 关闭按钮
        from PySide6.QtWidgets import QDialogButtonBox
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    @staticmethod
    def _render(it: dict) -> str:
        parts = [
            f"# {it.get('title', '(无标题)')}",
            f"**论文 ID**: `{it.get('paper_id', '?')}`",
            f"**来源**: {it.get('source', '?')}",
            f"**日期**: {it.get('summary_date', '?')}",
        ]
        if it.get("authors"):
            parts.append(f"**作者**: {it['authors'][:200]}")
        if it.get("keywords"):
            parts.append(f"**关键词**: {it['keywords']}")
        if it.get("in_papers_db"):
            parts.append("**状态**: ✅ 已精细分析")
        if it.get("analysis_id"):
            parts.append(f"**关联 papers.id**: {it['analysis_id']}")
        if it.get("abstract"):
            parts.append("\n## 摘要")
            parts.append(it["abstract"][:1500])
        if it.get("core_method"):
            parts.append("\n## 核心方法")
            parts.append(it["core_method"][:1500])
        if it.get("core_contrib"):
            parts.append("\n## 核心贡献")
            parts.append(it["core_contrib"][:1500])
        return "\n\n".join(parts)
