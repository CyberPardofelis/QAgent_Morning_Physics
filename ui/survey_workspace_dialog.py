"""
ui.survey_workspace_dialog — 综述工作台 dialog(2026-09-02 P6 UI 准备)

设计(2026-09-02 用户拍板, 仿 WAgent 写作工作台简化):
- 去掉最外层 tab 层级, 最左固定为树状列表
- 树状列表(主题 + 章节二级)
- 中间 2 tab: 查看(QTextBrowser markdown 渲染) / 编辑(QPlainTextEdit)
- 右 AI 协作(简化:1 个 chat input + 流式输出)
- 顶栏: 当前主题 + 章节标题 + [+ 新建主题] [+ 新建章节] [保存] [删除]

数据源:
- surveys.db(agent.tools.surveys CRUD 已就位)
- AI 协作: 暂时调用 query_knowledge 工具(将来 P6 综述生成会接 research_engine)
"""
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem,
    QPlainTextEdit, QTextBrowser, QPushButton, QLabel, QInputDialog, QMessageBox,
    QTabWidget, QWidget, QMenu, QLineEdit, QDialogButtonBox, QComboBox,
)
from ui.styles import get_stylesheet
from agent.tools.surveys import (
    list_themes, list_sections, get_section,
    create_theme, create_section, update_section, delete_section, delete_theme,
    health_check,
)


class SurveyWorkspaceDialog(QDialog):
    """综述工作台 — 主 dialog"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("论文综述库")
        self.setMinimumSize(1300, 800)
        self.setStyleSheet(get_stylesheet())
        # 当前状态
        self.current_theme_id: int | None = None
        self.current_section_id: int | None = None
        self._dirty = False  # 章节正文是否有未保存修改

        self._setup_ui()
        self._refresh_tree()
        self._update_status()

    def _setup_ui(self):
        """UI 搭建: 顶栏 + 三栏 (左树 / 中 tab / 右 AI) + 状态条"""
        root_layout = QVBoxLayout(self)

        # ---- 顶栏 ----
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("当前主题:"))
        self.current_label = QLabel("(未选中)")
        self.current_label.setStyleSheet("font-weight: bold;")
        top_layout.addWidget(self.current_label, stretch=1)

        self.btn_new_theme = QPushButton("+ 新建主题")
        self.btn_new_theme.clicked.connect(self._on_new_theme)
        top_layout.addWidget(self.btn_new_theme)

        self.btn_new_section = QPushButton("+ 新建章节")
        self.btn_new_section.clicked.connect(self._on_new_section)
        self.btn_new_section.setEnabled(False)
        top_layout.addWidget(self.btn_new_section)

        self.btn_save = QPushButton("保存")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save.setEnabled(False)
        top_layout.addWidget(self.btn_save)

        self.btn_delete = QPushButton("删除")
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_delete.setEnabled(False)
        top_layout.addWidget(self.btn_delete)

        root_layout.addLayout(top_layout)

        # ---- 三栏 splitter ----
        splitter = QSplitter(Qt.Horizontal)

        # 左: 树状列表
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["主题 / 章节", "类型"])
        # 2026-09-05 P0 fix: 主题/章节列太窄, 类型列太宽
        # 用户实测报告: 左列 (主题/章节, 实际内容) 显示不全, 右列 (类型标签, intro/progress 等)
        # 占用太多宽度 (auto-stretch)
        # 修法: 显式设列宽, 让左列占 4/5, 右列固定 80px
        self.tree.setColumnCount(2)
        from PySide6.QtWidgets import QHeaderView
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # 主题/章节占主要宽度
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        self.tree.setColumnWidth(1, 80)  # 类型列固定 80px
        self.tree.setMinimumWidth(380)  # 总宽度从 220 提到 380 给主题列足够空间
        self.tree.itemSelectionChanged.connect(self._on_tree_select)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        splitter.addWidget(self.tree)

        # 中: QTabWidget(查看 / 编辑)
        self.center_tabs = QTabWidget()
        # 查看 tab
        self.view_browser = QTextBrowser()
        self.view_browser.setOpenExternalLinks(False)
        self.center_tabs.addTab(self.view_browser, "查看")
        # 编辑 tab
        self.edit_text = QPlainTextEdit()
        self.edit_text.setPlaceholderText("在此输入章节内容 (Markdown 格式)...")
        self.edit_text.textChanged.connect(self._on_edit_changed)
        self.center_tabs.addTab(self.edit_text, "编辑")
        splitter.addWidget(self.center_tabs)
        splitter.setStretchFactor(1, 3)

        # 右: AI 协作
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("AI 协作 (简化版)"))

        self.ai_output = QTextBrowser()
        self.ai_output.setPlaceholderText("AI 输出会显示在这里...")
        right_layout.addWidget(self.ai_output, stretch=3)

        ai_input_layout = QHBoxLayout()
        self.ai_input = QLineEdit()
        self.ai_input.setPlaceholderText("问 AI 关于当前章节 / 综述主题的问题...")
        self.ai_input.returnPressed.connect(self._on_ai_send)
        ai_input_layout.addWidget(self.ai_input, stretch=1)
        self.btn_ai_send = QPushButton("发送")
        self.btn_ai_send.clicked.connect(self._on_ai_send)
        ai_input_layout.addWidget(self.btn_ai_send)
        right_layout.addLayout(ai_input_layout)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(2, 2)

        root_layout.addWidget(splitter, stretch=1)

        # ---- 状态条 ----
        self.status_label = QLabel("")
        root_layout.addWidget(self.status_label)

    # ============================================================
    # 树状列表管理
    # ============================================================
    def _refresh_tree(self):
        """从 surveys.db 重新加载主题 + 章节"""
        self.tree.clear()
        themes = list_themes()
        for t in themes:
            t_item = QTreeWidgetItem([t.get("name", "(无标题)"), "主题"])
            t_item.setData(0, Qt.UserRole, ("theme", t["id"]))
            sections = list_sections(t["id"])
            for s in sections:
                s_item = QTreeWidgetItem([s.get("title", "(无标题)"), s.get("section_type", "")])
                s_item.setData(0, Qt.UserRole, ("section", s["id"]))
                t_item.addChild(s_item)
            self.tree.addTopLevelItem(t_item)
        self.tree.expandAll()
        self._update_status()

    def _update_status(self):
        try:
            h = health_check()
            self.status_label.setText(
                f"📊 surveys.db: {h.get('n_themes', 0)} 主题 / {h.get('n_sections', 0)} 章节"
            )
        except Exception as e:
            self.status_label.setText(f"⚠ 状态查询失败: {e}")

    def _on_tree_select(self):
        """选中节点变化时 → 加载章节到中间 tab"""
        items = self.tree.selectedItems()
        if not items:
            return
        node = items[0]
        kind, nid = node.data(0, Qt.UserRole)

        # 切换前先检查是否有未保存
        # 9-11 user 拍板: 查看 tab (index=0) 是只读浏览模式, 不弹未保存提示 (影响浏览体验)
        # 只有编辑 tab (index=1) 才检查 dirty 并弹 Save/Discard/Cancel
        is_view_tab = self.center_tabs.currentIndex() == 0
        if self._dirty and not is_view_tab:
            ret = QMessageBox.question(
                self, "未保存",
                "当前章节有未保存修改, 是否保存?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if ret == QMessageBox.Save:
                self._on_save()
            elif ret == QMessageBox.Cancel:
                return

        self._dirty = False

        if kind == "theme":
            t = next((t for t in list_themes() if t["id"] == nid), None)
            if not t:
                return
            self.current_theme_id = nid
            self.current_section_id = None
            self.current_label.setText(f"主题: {t['name']}")
            # 主题级操作: 显示主题描述
            self.view_browser.setMarkdown(
                f"# {t['name']}\n\n"
                f"{t.get('description') or '*（无描述）*'}"
            )
            self.edit_text.setPlainText(t.get("description") or "")
            self.btn_new_section.setEnabled(True)
            self.btn_delete.setEnabled(True)
            self.btn_save.setEnabled(False)  # 主题描述暂不通过这里编辑

        elif kind == "section":
            s = get_section(nid)
            if not s:
                return
            self.current_theme_id = s["theme_id"]
            self.current_section_id = nid
            theme = next((t for t in list_themes() if t["id"] == s["theme_id"]), None)
            theme_name = theme["name"] if theme else "?"
            self.current_label.setText(f"主题: {theme_name} / 章节: {s['title']}")
            content = s.get("content_md") or ""
            # 查看 tab: markdown 渲染
            self.view_browser.setMarkdown(
                f"# {s['title']}\n\n*类型: {s.get('section_type', '?')}*\n\n{content}"
            )
            # 编辑 tab: 纯文本
            self.edit_text.setPlainText(content)
            self.btn_save.setEnabled(True)
            self.btn_delete.setEnabled(True)
            self.btn_new_section.setEnabled(True)  # 允许在主题下新建章节

    def _on_tree_context_menu(self, pos):
        """右键菜单: 新建/删除"""
        items = self.tree.selectedItems()
        menu = QMenu(self)
        act_new_theme = menu.addAction("+ 新建主题")
        act_new_section = menu.addAction("+ 新建章节")
        act_delete = menu.addAction("删除")
        act_delete.setEnabled(bool(items))
        act_new_section.setEnabled(bool(items) and items[0].data(0, Qt.UserRole)[0] == "theme")
        action = menu.exec(self.tree.mapToGlobal(pos))
        if action == act_new_theme:
            self._on_new_theme()
        elif action == act_new_section:
            self._on_new_section()
        elif action == act_delete:
            self._on_delete()

    # ============================================================
    # 主题 / 章节 CRUD 按钮
    # ============================================================
    def _on_new_theme(self):
        name, ok = QInputDialog.getText(self, "新建主题", "主题名称:")
        if not ok or not name.strip():
            return
        try:
            tid = create_theme(name.strip())
            self._refresh_tree()
            self.status_label.setText(f"✓ 已新建主题: {name}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"新建主题失败: {e}")

    def _on_new_section(self):
        if not self.current_theme_id:
            QMessageBox.warning(self, "提示", "请先选中一个主题")
            return
        title, ok = QInputDialog.getText(self, "新建章节", "章节标题:")
        if not ok or not title.strip():
            return
        # 选 section_type
        type_options = ["intro", "progress", "method_compare", "gap", "outlook", "custom"]
        stype, ok = QInputDialog.getItem(
            self, "章节类型", "选择章节类型:",
            type_options, 0, False,
        )
        if not ok:
            return
        try:
            sid = create_section(self.current_theme_id, title.strip(), section_type=stype)
            self._refresh_tree()
            self.status_label.setText(f"✓ 已新建章节: {title}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"新建章节失败: {e}")

    def _on_save(self):
        """保存当前章节正文"""
        if not self.current_section_id:
            return
        content = self.edit_text.toPlainText()
        try:
            update_section(self.current_section_id, content_md=content)
            self._dirty = False
            # 刷新查看 tab
            s = get_section(self.current_section_id)
            self.view_browser.setMarkdown(
                f"# {s['title']}\n\n*类型: {s.get('section_type', '?')}*\n\n{content}"
            )
            self.status_label.setText(f"✓ 章节 {s['title']} 已保存")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def _on_delete(self):
        items = self.tree.selectedItems()
        if not items:
            return
        kind, nid = items[0].data(0, Qt.UserRole)
        if kind == "theme":
            ret = QMessageBox.question(
                self, "确认",
                f"删除主题会级联删所有章节, 确认?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return
            try:
                n = delete_theme(nid)
                self.status_label.setText(f"✓ 已删主题 + {n} 章节")
                self.current_theme_id = None
                self.current_section_id = None
                self.current_label.setText("(未选中)")
                self.view_browser.clear()
                self.edit_text.clear()
                self._refresh_tree()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {e}")
        elif kind == "section":
            ret = QMessageBox.question(
                self, "确认",
                "删除当前章节?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return
            try:
                delete_section(nid)
                self.status_label.setText(f"✓ 已删章节")
                self.current_section_id = None
                self._refresh_tree()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {e}")

    def _on_edit_changed(self):
        """编辑 tab 内容变化时标记 dirty"""
        if not self._dirty and self.current_section_id:
            self._dirty = True
            self.btn_save.setStyleSheet("background-color: #f0ad4e; color: white;")

    # ============================================================
    # AI 协作(简化版, 直接调 LLM 流式输出)
    # ============================================================
    def _on_ai_send(self):
        question = self.ai_input.text().strip()
        if not question:
            return
        self.ai_input.clear()
        # 简化: 把问题 + 当前章节 context 拼成 prompt 调 LLM
        # (将来 P6 综述生成会接 research_engine.py 的 ReAct 内核)
        context = ""
        if self.current_section_id:
            s = get_section(self.current_section_id)
            if s:
                context = f"\n\n[当前章节] {s['title']}:\n{s.get('content_md', '')[:500]}"
        prompt = question + context
        # 起后台线程调 LLM
        self.ai_output.append(f"\n\n> **你**: {question}\n")
        self.ai_output.append(f"\n**AI**: ")
        self._ai_thread = _AIWorker(prompt, self.ai_output)
        self._ai_thread.chunk.connect(self._on_ai_chunk)
        self._ai_thread.finished.connect(self._on_ai_done)
        self.btn_ai_send.setEnabled(False)
        self._ai_thread.start()

    def _on_ai_chunk(self, chunk: str):
        cursor = self.ai_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(chunk)
        self.ai_output.verticalScrollBar().setValue(self.ai_output.verticalScrollBar().maximum())

    def _on_ai_done(self, error: str):
        if error:
            self.ai_output.append(f"\n[错误: {error}]\n")
        self.btn_ai_send.setEnabled(True)
        self.status_label.setText("✓ AI 输出完毕")


class _AIWorker(QThread):
    """简化版 AI worker: 调 _call_llm(同步) + 切 chunk 输出(模拟流式)"""
    chunk = Signal(str)
    finished = Signal(str)  # error message, "" = success

    def __init__(self, prompt: str, output_widget=None):
        super().__init__()
        self.prompt = prompt
        self.output_widget = output_widget

    def run(self):
        try:
            from agent.tools.llm import _call_llm
            # 流式效果: 先 chunk 1 段(实际 LLM 调用是同步,这里只能分段)
            result = _call_llm(self.prompt, temperature=0.5, max_tokens=2000)
            # 切成 30 字符一段(模拟流式)
            for i in range(0, len(result), 30):
                self.chunk.emit(result[i:i+30])
                self.msleep(20)  # 20ms 间隔,看起来像流式
            self.finished.emit("")
        except Exception as e:
            self.finished.emit(str(e))
