import json
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QTextEdit, QApplication,
    QTabWidget, QWidget, QHBoxLayout, QPushButton, QMessageBox
)
from PySide6.QtGui import QFont
from ui.styles import GLOBAL_STYLESHEET
from agent.tools import update_full_paper, update_paper_comment


class PaperEditDialog(QDialog):
    def __init__(self, article: dict, parent=None):
        super().__init__(parent)
        self.article = article
        self.paper_id = article.get("paper_id")
        self.setWindowTitle(f"论文详情 - {self.paper_id}")
        self.setMinimumSize(850, 750)
        self.setStyleSheet(GLOBAL_STYLESHEET)  # 应用全局样式

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 获取内容区域字体
        app_font = QApplication.font()
        base_size = app_font.pointSize()
        if base_size <= 0:
            base_size = 14
        content_font = QFont(app_font)
        content_font.setPointSize(base_size + 2)

        # 创建标签页控件
        tab_widget = QTabWidget()
        tab_widget.setDocumentMode(True)  # 更现代的标签页风格
        main_layout.addWidget(tab_widget)

        # ----- 标签页1：基本信息 -----
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        basic_layout.setSpacing(15)

        # 使用 QFormLayout 组织基本信息
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form_layout.setSpacing(12)

        # 论文ID（只读）
        id_label = QLabel(self.paper_id)
        id_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        id_label.setWordWrap(True)
        form_layout.addRow("论文ID:", id_label)

        # 标题（只读）
        title_label = QLabel(article.get("title", ""))
        title_label.setWordWrap(True)
        title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form_layout.addRow("标题:", title_label)

        # 作者（只读，逗号分隔）
        authors_str = ", ".join(article.get("authors", []))
        authors_label = QLabel(authors_str)
        authors_label.setWordWrap(True)
        authors_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form_layout.addRow("作者:", authors_label)

        # 链接（只读，超链接）
        link = article.get("link", "")
        link_label = QLabel()
        if link:
            link_label.setText(f'<a href="{link}">{link}</a>')
            link_label.setOpenExternalLinks(True)
        else:
            link_label.setText("无")
        link_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        form_layout.addRow("链接:", link_label)

        # 分析时间（只读，格式化）
        analyzed_time = article.get("analyzed_at", "")
        formatted_time = self.format_analyzed_time(analyzed_time)
        time_label = QLabel(formatted_time)
        time_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form_layout.addRow("分析时间:", time_label)

        # 关键词（可编辑，多行，每行一个）
        self.keywords_edit = QTextEdit()
        self.keywords_edit.setFont(content_font)  # 应用大字体
        self.keywords_edit.setPlainText("\n".join(article.get("keywords", [])))
        self.keywords_edit.setMinimumHeight(120)
        form_layout.addRow("关键词 (每行一个):", self.keywords_edit)

        # 研究主题（可编辑，多行）
        self.topic_edit = QTextEdit()
        self.topic_edit.setFont(content_font)
        self.topic_edit.setPlainText(article.get("research_topic", ""))
        self.topic_edit.setMinimumHeight(80)
        form_layout.addRow("研究主题:", self.topic_edit)

        basic_layout.addLayout(form_layout)
        basic_layout.addStretch()  # 顶起，防止控件过度拉伸

        # ----- 标签页2：总结 -----
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self.summary_edit = QTextEdit()
        self.summary_edit.setFont(content_font)
        self.summary_edit.setPlainText(article.get("summary", ""))
        self.summary_edit.setMinimumHeight(300)
        summary_layout.addWidget(QLabel("总结:"))
        summary_layout.addWidget(self.summary_edit)

        # ----- 标签页3：基本方法 -----
        method_tab = QWidget()
        method_layout = QVBoxLayout(method_tab)
        self.method_edit = QTextEdit()
        self.method_edit.setFont(content_font)
        self.method_edit.setPlainText(article.get("basic_method", ""))
        self.method_edit.setMinimumHeight(300)
        method_layout.addWidget(QLabel("基本方法:"))
        method_layout.addWidget(self.method_edit)

        # ----- 标签页4：主要结论 -----
        conclusion_tab = QWidget()
        conclusion_layout = QVBoxLayout(conclusion_tab)
        self.conclusion_edit = QTextEdit()
        self.conclusion_edit.setFont(content_font)
        self.conclusion_edit.setPlainText(article.get("main_conclusion", ""))
        self.conclusion_edit.setMinimumHeight(300)
        conclusion_layout.addWidget(QLabel("主要结论:"))
        conclusion_layout.addWidget(self.conclusion_edit)

        # ----- 标签页5：引用文献 -----
        cites_tab = QWidget()
        cites_layout = QVBoxLayout(cites_tab)
        cites_layout.setSpacing(20)

        # 重要引用
        cites_layout.addWidget(QLabel("重要引用 (每行一个):"))
        self.citations_edit = QTextEdit()
        self.citations_edit.setFont(content_font)
        # print("intro_cites type:", type(article.get("intro_cites")))
        # print("intro_cites content:", article.get("intro_cites"))
        #self.citations_edit.setPlainText("\n".join(article.get("citations", [])))
        citations_text = self.format_citations(article.get("citations", []))
        self.citations_edit.setPlainText(citations_text)
        self.citations_edit.setMinimumHeight(80)
        cites_layout.addWidget(self.citations_edit)

        # 引言引用
        cites_layout.addWidget(QLabel("引言引用 (每行一个):"))
        self.intro_cites_edit = QTextEdit()
        self.intro_cites_edit.setFont(content_font)
        # self.intro_cites_edit.setPlainText("\n".join(article.get("intro_cites", [])))
        intro_cites_text = self.format_citations(article.get("intro_cites", []))
        self.intro_cites_edit.setPlainText(intro_cites_text)
        self.intro_cites_edit.setMinimumHeight(160)
        cites_layout.addWidget(self.intro_cites_edit)

        # ----- 标签页6：展望 -----
        outlook_tab = QWidget()
        outlook_layout = QVBoxLayout(outlook_tab)
        self.outlook_edit = QTextEdit()
        self.outlook_edit.setFont(content_font)
        self.outlook_edit.setPlainText(article.get("outlook", ""))
        self.outlook_edit.setMinimumHeight(300)
        outlook_layout.addWidget(QLabel("展望:"))
        outlook_layout.addWidget(self.outlook_edit)

        # ----- 标签页7：用户评论 -----
        comments_tab = QWidget()
        comments_layout = QVBoxLayout(comments_tab)
        self.comments_edit = QTextEdit()
        self.comments_edit.setFont(content_font)
        self.comments_edit.setPlainText("\n".join(article.get("user_comments", [])))
        self.comments_edit.setMinimumHeight(350)
        comments_layout.addWidget(QLabel("用户评论 (每行一条):"))
        comments_layout.addWidget(self.comments_edit)

        # ----- 标签页8：LLM总结 -----
        llm_summary_tab = QWidget()
        llm_summary_layout = QVBoxLayout(llm_summary_tab)
        self.llm_summary_edit = QTextEdit()
        self.llm_summary_edit.setFont(content_font)  # 与其它标签页字体一致
        self.llm_summary_edit.setPlainText(article.get("llm_summary", ""))
        self.llm_summary_edit.setMinimumHeight(400)
        llm_summary_layout.addWidget(QLabel("LLM总结:"))
        llm_summary_layout.addWidget(self.llm_summary_edit)

        # 将所有标签页添加到 tab_widget
        tab_widget.addTab(basic_tab, "基本信息")
        tab_widget.addTab(summary_tab, "总结")
        tab_widget.addTab(method_tab, "基本方法")
        tab_widget.addTab(conclusion_tab, "主要结论")
        tab_widget.addTab(cites_tab, "引用文献")
        tab_widget.addTab(outlook_tab, "展望")
        tab_widget.addTab(comments_tab, "用户评论")
        tab_widget.addTab(llm_summary_tab, "LLM总结")

        # ----- 底部按钮区域 -----
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("保存修改")
        save_btn.setObjectName("saveBtn")
        save_btn.clicked.connect(self.save_changes)

        comment_btn = QPushButton("仅追加评论")
        comment_btn.setObjectName("commentBtn")
        comment_btn.clicked.connect(self.save_comment_only)

        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("cancelBtn")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(comment_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)

    def format_analyzed_time(self, iso_time):
        """将 ISO 时间格式化为 YYYY-MM-DD HH:MM:SS"""
        if not iso_time:
            return ""
        if 'T' in iso_time:
            base = iso_time.replace('T', ' ')
        else:
            base = iso_time
        if '.' in base:
            base = base.split('.')[0]
        return base

    def save_changes(self):
        """保存所有可编辑字段的修改"""
        try:
            updated = {
                "paper_id": self.paper_id,
                # 只读字段直接从原数据复制
                "title": self.article.get("title", ""),
                "authors": self.article.get("authors", []),
                "link": self.article.get("link", ""),
                "analyzed_at": self.article.get("analyzed_at", ""),
                "local_path": self.article.get("local_path", ""),
                # 可编辑字段从各个控件获取
                "keywords": [line.strip() for line in self.keywords_edit.toPlainText().split('\n') if line.strip()],
                "research_topic": self.topic_edit.toPlainText(),
                "basic_method": self.method_edit.toPlainText(),
                "main_conclusion": self.conclusion_edit.toPlainText(),
                "citations": [line.strip() for line in self.citations_edit.toPlainText().split('\n') if line.strip()],
                "intro_cites": [line.strip() for line in self.intro_cites_edit.toPlainText().split('\n') if line.strip()],
                "outlook": self.outlook_edit.toPlainText(),
                "summary": self.summary_edit.toPlainText(),
                "user_comments": [line.strip() for line in self.comments_edit.toPlainText().split('\n') if line.strip()],
            }
            updated["llm_summary"] = self.llm_summary_edit.toPlainText()
            update_full_paper(updated)  # 需要实现此函数
            QMessageBox.information(self, "成功", "论文信息已更新")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：{e}")

    def save_comment_only(self):
        """仅追加评论（复用原有 update_paper_comment 函数）"""
        new_comment = self.comments_edit.toPlainText().strip()
        if not new_comment:
            QMessageBox.warning(self, "提示", "请输入要追加的评论")
            return
        try:
            updated_article = update_paper_comment(self.paper_id, new_comment)
            # 更新评论显示
            comments = updated_article.get("user_comments", [])
            self.comments_edit.setPlainText("\n".join(comments))
            QMessageBox.information(self, "成功", "评论已追加")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"评论更新失败：{e}")

    def format_citations(self, cites):
        if not cites:
            return ""
        lines = []
        for cite in cites:
            # 处理 authors 字段：可能是列表、字符串或 None
            authors_raw = cite.get("authors", [])
            if isinstance(authors_raw, list):
                authors = ", ".join(authors_raw)
            elif isinstance(authors_raw, str):
                authors = authors_raw  # 直接使用字符串
            else:
                authors = ""

            year = cite.get("year", "")
            title = cite.get("title", "")
            journal = cite.get("journal", "")
            doi = cite.get("doi", "")

            line = f"{title}. {authors}. {year}. {journal}"
            if doi:
                line += f" DOI: {doi}"
            lines.append(line)
        return "\n\n".join(lines)
