# chat_window.py
import json
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, QSettings, QEvent
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QMenu,
    QPushButton, QTreeWidget, QTreeWidgetItem, QLabel, QComboBox, QInputDialog,
    QPlainTextEdit, QMessageBox, QScrollArea, QSizePolicy, QCheckBox
)
from agent.history_manager import Conversation
from ui.styles import GLOBAL_STYLESHEET
from ui.message_bubble_1 import MessageBubble
from ui.settings_dialog import SettingsDialog
from ui.group_prompt_dialog import GroupSystemPromptDialog
from ui.comparison_config_dialog import ComparisonConfigDialog
from ui.conversation_controller import ConversationController
from agent.prompts import SYSTEM_PROMPT
from ui.input_panel import InputPanel
from ui.chat_view import ChatView
from ui.tree_panel import ConversationTreePanel


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM科研助手: 早上好，物理学")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        self.controller = ConversationController(self)
        self._new_bubble_pending = False   # 标记需要创建新气泡

        self._setup_ui()
        self._connect_signals()
        self._load_settings()
        self.refresh_tree()
        self.add_chat_message("system",
                              "LLM科研助手已启动。请选择模式并开始对话。\n\n"
                              "发送 早上好 arxiv/pra/prl等 以获取对应平台新文章摘要.\n\n"
                              "发送 分析+文章DOI 以下载并分析对应文章，分析结果会保存至知识库。\n\n"
                              "在分析文章之后可以发送 保存评论：评论内容 以将文章评论保存到知识库\n\n"
                              "菜单栏中打开知识库，可以进行检索并打开相关文章的详情页面查看\n\n"
                              "在详情页面可以查看和修改总结信息、引用信息、评论和LLM详细总结等")

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── 左侧：对话树面板 ──
        self.tree_panel = ConversationTreePanel(self.controller.history_manager, self)
        splitter.addWidget(self.tree_panel)
        splitter.setStretchFactor(0, 1)

        # ── 右侧 ──
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("模式："))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["openai", "deepseek-v4-flash", "deepseek-v4-pro", "compare"])
        self.mode_combo.setCurrentIndex(1)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        top_layout.addWidget(self.mode_combo)

        self.think_checkbox = QCheckBox("思考模式")
        self.think_checkbox.setChecked(True)
        top_layout.addWidget(self.think_checkbox)
        top_layout.addSpacing(20)

        self.daily_summary_btn = QPushButton("每日总结")
        self.daily_summary_btn.clicked.connect(self.on_daily_summary_clicked)
        top_layout.addWidget(self.daily_summary_btn)
        top_layout.addStretch()
        right_layout.addLayout(top_layout)

        # 消息气泡区
        self.chat_view = ChatView()
        right_layout.addWidget(self.chat_view, stretch=1)

        # 输入面板
        self.input_panel = InputPanel()
        right_layout.addWidget(self.input_panel)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 3)

        # 菜单栏
        menubar = self.menuBar()
        settings_action = QAction("首选项", self)
        settings_action.triggered.connect(self.open_settings_dialog)
        menubar.addAction(settings_action)

        search_action = QAction("知识库", self)
        search_action.triggered.connect(self.open_knowledge_base)
        menubar.addAction(search_action)

        compare_config_action = QAction("对比模式设置", self)
        compare_config_action.triggered.connect(self.show_comparison_config_dialog)
        menubar.addAction(compare_config_action)

        icon_path = "resources/app_icon.ico"
        self.setWindowIcon(QIcon(icon_path))

    def _connect_signals(self):
        # Controller → UI
        self.controller.new_message.connect(self.add_chat_message)
        self.controller.stream_chunk.connect(self._on_stream_chunk_ui)
        self.controller.new_bubble.connect(self._on_new_bubble)
        self.controller.comparison_chunk.connect(self._add_comparison_chunk)
        self.controller.clear_chat.connect(self.chat_view.clear_chat)
        self.controller.scroll_to_bottom.connect(self.chat_view.scroll_to_bottom)
        self.controller.update_regenerate_button.connect(
            self.input_panel.regenerate_btn.setEnabled
        )
        self.controller.update_stop_button.connect(
            self.input_panel.stop_btn.setEnabled
        )
        self.controller.update_send_input.connect(
            self.input_panel.set_input_enabled
        )
        self.controller.conversation_switched.connect(self.on_conversation_switched)

        # 输入面板回调
        self.input_panel.send_clicked = self._on_send
        self.input_panel.stop_clicked = self.controller.stop_generation
        self.input_panel.regenerate_clicked = self._on_regenerate

        # 树面板回调
        self.tree_panel.on_conversation_selected = self.controller.switch_conversation
        self.tree_panel.on_create_conversation = self.controller.create_new_conversation
        self.tree_panel.on_edit_group_prompt = self.open_group_prompt_dialog
        self.tree_panel.on_delete_conversation = self.delete_conversation
        self.tree_panel.on_move_conversation = self._on_move_conversation

    def _on_move_conversation(self, conv_id, target_group_id):
        """移动对话后同步更新当前对话对象的 group_id"""
        if (self.controller.current_conversation and
                self.controller.current_conversation.conv_id == conv_id):
            self.controller.current_conversation.group_id = target_group_id

    def _on_new_bubble(self):
        self._new_bubble_pending = True

    def _on_stream_chunk_ui(self, role, chunk):
        if role != "assistant" or not chunk:
            return
        if self._new_bubble_pending:
            self.add_chat_message("assistant", chunk, isAppend=False)
            self._new_bubble_pending = False
        else:
            last_bubble = self.chat_view.find_last_assistant_bubble()
            if last_bubble:
                last_bubble.update_content(chunk)
                QTimer.singleShot(0, self.chat_view.scroll_if_needed)
            else:
                self.add_chat_message("assistant", chunk, isAppend=False)

    def _add_comparison_chunk(self, chunk):
        last_bubble = self.chat_view.find_last_assistant_bubble()
        if last_bubble and last_bubble.role == "assistant":
            last_bubble.update_content(chunk)
            QTimer.singleShot(0, self.chat_view.scroll_if_needed)
        else:
            self.add_chat_message("assistant", chunk, isAppend=False)

    def _on_send(self, text):
        mode = self.mode_combo.currentText()
        think = self.think_checkbox.isChecked()
        self.controller.send_message(text, mode, think)

    def _on_regenerate(self):
        mode = self.mode_combo.currentText()
        think = self.think_checkbox.isChecked()
        self.controller.regenerate_last_message(mode, think)

    # -------------------- 消息气泡（委托给 chat_view） --------------------
    def add_chat_message(self, role, text, isAppend=False):
        self.chat_view.add_chat_message(role, text, isAppend)

    # -------------------- 左侧树形（委托给 tree_panel） --------------------
    def refresh_tree(self):
        current_id = None
        if self.controller.current_conversation:
            current_id = self.controller.current_conversation.conv_id
        self.tree_panel.refresh_tree(current_conv_id=current_id)

    def on_conversation_switched(self):
        self.refresh_tree()
        if self.controller.current_conversation:
            self.tree_panel._highlight_conversation_in_tree(
                self.controller.current_conversation.conv_id
            )

    def create_new_group(self):
        self.tree_panel.create_new_group()

    # 以下方法保持原样，供 tree_panel 的右键菜单通过回调调用
    def delete_conversation(self, conv_id):
        self.controller.delete_conversation(conv_id)
        self.refresh_tree()

    # -------------------- 其他 --------------------
    def open_knowledge_base(self):
        from ui.knowledge_base_dialog import KnowledgeBaseDialog
        dialog = KnowledgeBaseDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.show()

    def open_settings_dialog(self):
        dialog = SettingsDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.default_prompt_changed.connect(self.on_default_prompt_changed)
        dialog.show()

    def on_default_prompt_changed(self, new_prompt):
        self.controller.default_system_prompt = new_prompt

    def open_group_prompt_dialog(self, group_id):
        groups = self.controller.history_manager.get_all_groups()
        group_name = next((g['name'] for g in groups if g['group_id'] == group_id), "未知分组")
        dialog = GroupSystemPromptDialog(group_id, group_name, self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.show()

    def show_comparison_config_dialog(self):
        dialog = ComparisonConfigDialog(self)
        if dialog.exec():
            self.controller._load_comparison_config()

    def on_mode_changed(self, mode):
        if mode == "compare":
            self.think_checkbox.setEnabled(False)
            self.think_checkbox.setToolTip("对比模式下，请使用「对比模式设置」分别配置各模型的思考开关")
        else:
            self.think_checkbox.setEnabled(True)
            self.think_checkbox.setToolTip("思考模式已开启")

    def on_daily_summary_clicked(self):
        settings = QSettings("CyberPardofelis", "LLM科研助手")
        selected = json.loads(settings.value("daily_summary_journals", '["arxiv"]'))
        if not selected:
            QMessageBox.information(self, "提示", "请先在设置中勾选需要每日总结的期刊")
            return
        group_name = "每日总结"
        groups = self.controller.history_manager.get_all_groups()
        group_id = next((g["group_id"] for g in groups if g["name"] == group_name), None)
        if group_id is None:
            group_id = self.controller.history_manager.create_group(group_name)
        today_str = datetime.now().strftime("%Y-%m-%d")
        conv = self.controller.history_manager.create_conversation(title=today_str, group_id=group_id)
        if hasattr(conv.created_at, 'isoformat'):
            conv.created_at = conv.created_at.isoformat()
            conv.updated_at = conv.updated_at.isoformat()
        self.controller.history_manager.save_conversation(conv)
        self.refresh_tree()
        self.controller.switch_conversation(conv.conv_id)
        lines = ["早上好 arxiv" if j == "arxiv" else f"早上好 {j}" for j in selected]
        self.input_panel.input_edit.setPlainText("\n".join(lines))
        self._on_send("\n".join(lines))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.chat_view.refresh_bubble_widths()

    def _load_settings(self):
        settings = QSettings("CyberPardofelis", "LLM科研助手")
        self.controller.default_system_prompt = settings.value("default_system_prompt", SYSTEM_PROMPT)
        self.controller._load_comparison_config()

    def closeEvent(self, event):
        if self.controller.current_conversation:
            self.controller.history_manager.save_conversation(self.controller.current_conversation)
        event.accept()