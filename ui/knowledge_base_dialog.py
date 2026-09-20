from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from agent.tools import get_paper_by_id, get_recent_papers_raw, query_knowledge
from ui.paper_edit_dialog import PaperEditDialog
from ui.styles import get_stylesheet


class KnowledgeBaseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("论文知识库")
        self.setMinimumSize(1000, 700)
        # 2026-09-02 改:用 get_stylesheet() 跟主题
        self.setStyleSheet(get_stylesheet())
        # 主布局
        layout = QVBoxLayout(self)

        # ---- 搜索区域 ----
        search_layout = QHBoxLayout()
        self.keyword_edit = QLineEdit()
        self.keyword_edit.setPlaceholderText("输入搜索关键词...")
        self.search_combo = QComboBox()
        self.search_combo.addItems(["全文检索", "按标题", "按作者", "按主题"])
        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self.perform_search)

        search_layout.addWidget(QLabel("搜索:"))
        search_layout.addWidget(self.keyword_edit)
        search_layout.addWidget(self.search_combo)
        search_layout.addWidget(search_btn)
        search_layout.addStretch()
        layout.addLayout(search_layout)

        # ---- 结果表格 ----
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["论文ID", "标题", "分析时间"])
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.doubleClicked.connect(self.open_edit_dialog)
        layout.addWidget(self.result_table)
        self.load_recent_papers()

    def load_recent_papers(self):
        """加载最近的论文并显示在表格中"""
        recent = get_recent_papers_raw(15)  # UI 内部函数，返回 list of dict
        self.display_papers(recent)

    def display_papers(self, papers):
        """将论文列表填充到表格中，并格式化时间"""
        self.result_table.setRowCount(len(papers))
        for row, article in enumerate(papers):
            paper_id = article.get("paper_id", "")
            title = article.get("title", "")
            analyzed_at = article.get("analyzed_at", "")

            # 格式化时间：去掉 'T' 和毫秒部分
            formatted_time = self.format_analyzed_time(analyzed_at)

            self.result_table.setItem(row, 0, QTableWidgetItem(paper_id))
            self.result_table.setItem(row, 1, QTableWidgetItem(title))
            self.result_table.setItem(row, 2, QTableWidgetItem(formatted_time))
        self.result_table.resizeColumnsToContents()

    def format_analyzed_time(self, iso_time):
        """将 ISO 格式时间 'YYYY-MM-DDTHH:MM:SS.ffffff' 转换为 'YYYY-MM-DD HH:MM:SS'"""
        if not iso_time:
            return ""
        # 替换 'T' 为空格，并截断小数点及以后
        if 'T' in iso_time:
            base = iso_time.replace('T', ' ')
        else:
            base = iso_time
        # 去掉毫秒部分
        if '.' in base:
            base = base.split('.')[0]
        return base

    def perform_search(self):
        """执行搜索并更新表格"""
        keyword = self.keyword_edit.text().strip()
        search_type = self.search_combo.currentText()

        # 根据搜索类型调用 query_knowledge 函数
        # 你需要根据 search_type 来设置 query_knowledge 的参数
        # 例如，如果选择"按作者"，则调用 query_knowledge(author=keyword)
        # 这里以全文检索为例
        if not keyword:
            QMessageBox.warning(self, "提示", "请输入搜索关键词")
            return

        # 调用数据库查询函数（请根据你的实际函数调整）
        # from your_database_module import query_knowledge
        # results = query_knowledge(keyword=keyword)  # 这需要你的 query_knowledge 支持全文检索

        # --- 示例：直接使用 SQL 查询（假设你的 query_knowledge 已支持 FTS）---
        try:
            # 这是一个简化的调用，你需要根据你的实际函数来
            # 假设你的 query_knowledge 返回一个包含 dict 的列表
            results = query_knowledge(keyword=keyword)  # 这是你之前定义的函数
        except Exception as e:
            QMessageBox.critical(self, "数据库错误", f"查询失败：{e}")
            return

        # 更新表格
        self.result_table.setRowCount(len(results))
        for row, article in enumerate(results):
            self.result_table.setItem(row, 0, QTableWidgetItem(article.get("paper_id", "")))
            self.result_table.setItem(row, 1, QTableWidgetItem(article.get("title", "")))
            self.result_table.setItem(row, 2, QTableWidgetItem(article.get("analyzed_at", "")))
        self.result_table.resizeColumnsToContents()

    def open_edit_dialog(self, index):
        row = index.row()
        paper_id_item = self.result_table.item(row, 0)
        if not paper_id_item:
            return
        paper_id = paper_id_item.text()
        article = get_paper_by_id(paper_id)
        if not article:
            QMessageBox.warning(self, "错误", "无法获取论文详情")
            return

        dialog = PaperEditDialog(article, self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        # 当对话框关闭时（无论保存或取消），如果状态是 Accepted，则刷新列表
        dialog.finished.connect(lambda result: self.perform_search() if result == QDialog.Accepted else None)
        dialog.show()
