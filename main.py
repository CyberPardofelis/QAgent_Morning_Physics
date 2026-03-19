import sys
import traceback
import os
from datetime import datetime

try:
    # 原始导入和代码
    import json
    import html
    import sqlite3
    from datetime import datetime
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from agent.history_manager import HistoryManager, Conversation
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QSplitter, QPushButton, QTreeWidget, QTreeWidgetItem, QListWidget, QListWidgetItem,
        QLabel, QComboBox, QPlainTextEdit, QMenu, QMessageBox, QStackedWidget,
        QInputDialog, QApplication, QSizePolicy, QTableWidget, QTableWidgetItem, QTextEdit, QScrollArea
    )
    from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QPoint
    from PySide6.QtGui import QDesktopServices, QAction, QIcon, QFont, QFontInfo
    from agent.controller import run_agent_stream
    from agent.prompts import SYSTEM_PROMPT
    from agent.tools import (
        update_paper_comment, DB_PATH, query_knowledge, get_paper_by_id, get_db_connection, get_recent_papers,
        update_full_paper
    )

    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QCheckBox, QComboBox,
        QDialogButtonBox, QMessageBox, QGroupBox, QTabWidget
    )
    from PySide6.QtCore import QSettings,Qt,QPoint,QTimer,QUrl
    from QMarkdownWidget import QMView
    import config
    import yaml

except Exception:
    # 确定日志路径
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, 'error.log')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n--- {datetime.now()} ---\n")
        traceback.print_exc(file=f)
    # 重新抛出，让系统处理（可选）
    raise

def excepthook(exc_type, exc_value, exc_traceback):
    # 确定日志文件存放位置（与 exe 同目录或当前目录）
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, 'error.log')

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n--- {datetime.now()} ---\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    # 也可以打印到控制台（如果有）
    traceback.print_exception(exc_type, exc_value, exc_traceback)


sys.excepthook = excepthook


# ------ 全局样式 ------

GLOBAL_STYLESHEET = """
/* 全局字体和背景 */
QMainWindow, QDialog {
    background-color: #fafaff;
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    font-size: 14px;
}
/* 分割器样式 */
QSplitter::handle {
    background-color: #e0e4e9;
    width: 1px;
}
QSplitter::handle:hover {
    background-color: #b0b8c5;
}
/* 左侧边栏 */
QWidget#leftWidget {
    background-color: white;
    border-right: 1px solid #eaeef2;
}

QPushButton#newConvBtn:hover {
    background-color: #0b5ed7;
}
/* 对话列表 */
QListWidget {
    background-color: transparent;
    border: none;
    outline: none;
}
QListWidget::item {
    background-color: transparent;
    border-radius: 8px;
    padding: 10px;
    margin: 4px 8px;
    color: #1e293b;
}
QListWidget::item:selected {
    background-color: #e9f0ff;
    color: #0d6efd;
}
/* 通用 QComboBox 样式 */
QComboBox {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 20px;
    padding: 6px 16px;
    min-width: 120px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox::down-arrow {
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #5f6b7a;
    width: 0;
    height: 0;
    margin-right: 8px;
}
QComboBox:hover {
    border-color: #0d6efd;
}
QComboBox QAbstractItemView {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 8px;
    padding: 4px;
    outline: none;
    selection-background-color: #e9f0ff;
    selection-color: #0d6efd;
}
QComboBox QAbstractItemView::item {
    padding: 8px 12px;
    border-radius: 6px;
}
/* 通用 QLineEdit 样式（用于搜索框等） */
QLineEdit {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 20px;
    padding: 12px 16px;
    selection-background-color: #b2d3ff;
}
QLineEdit:focus {
    border-color: #0d6efd;
    outline: none;
}
/* 通用 QTextEdit 样式（用于多行输入） */
QTextEdit {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 14px;
    padding: 8px 12px;
    selection-background-color: #b2d3ff;
}
QTextEdit:focus {
    border-color: #0d6efd;
    outline: none;
}
/* 主页面文本输入样式 */
QPlainTextEdit {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 20px;
    padding: 8px 18px;
    selection-background-color: #b2d3ff;
}
/* 通用按钮样式 */
QPushButton {
    background-color: #0d6efd;
    color: white;
    border: none;
    border-radius: 14px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #0b5ed7;
}
QPushButton:pressed {
    background-color: #0a58ca;
}
QPushButton:disabled {
    background-color: #b6c2d0;
    color: #eef2f6;
}
/* 特殊按钮颜色覆盖 */
QPushButton#stopBtn {
    background-color: #dc3545;
}
QPushButton#stopBtn:hover {
    background-color: #bb2d3b;
}
QPushButton#regenerateBtn {
    background-color: #6c757d;
}
QPushButton#regenerateBtn:hover {
    background-color: #5c636a;
}
/* 表格样式 */
QTableWidget {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 12px;
    gridline-color: #e9ecef;
    selection-background-color: #e9f0ff;
    selection-color: #0d6efd;
}
QTableWidget::item {
    padding: 8px;
    border-bottom: 1px solid #f1f3f5;
}
QHeaderView::section {
    background-color: #f8f9fa;
    padding: 8px;
    border: none;
    font-weight: bold;
}
/* 滚动条样式（扁平圆角） */
QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 8px;
    border-radius: 4px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #c1c9d2;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #a0aab5;
}
QScrollBar:horizontal {
    border: none;
    background: transparent;
    height: 8px;
    border-radius: 4px;
    margin: 0px;
}
QScrollBar::handle:horizontal {
    background: #c1c9d2;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #a0aab5;
}
/* 菜单栏 */
QMenuBar {
    background-color: white;
    border-bottom: 1px solid #eaeef2;
}
QMenuBar::item {
    padding: 8px 12px;
    background: transparent;
    border-radius: 6px;
}
QMenuBar::item:selected {
    background-color: #e9f0ff;
}
QMenu {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 8px;
    padding: 6px;
}
QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 6px;
}
QMenu::item:selected {
    background-color: #e9f0ff;
    color: #0d6efd;
}
/* ---------- 扁平化标签页 ---------- */
QTabWidget::pane {
    border: 1px solid #dde1e6;
    border-radius: 12px;
    background: white;
    top: -1px;  /* 让 pane 与 tab 条对齐 */
}

QTabBar::tab {
    background-color: #f1f5f9;
    border: 1px solid #dde1e6;
    border-bottom: none;  /* 避免双边框 */
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 20px;
    margin-right: 2px;
    font-size: 13px;
    font-weight: 500;
    color: #1e293b;
}

QTabBar::tab:selected {
    background-color: white;
    border-bottom: 1px solid white;  /* 与 pane 背景融合 */
    color: #0d6efd;
}

QTabBar::tab:hover:!selected {
    background-color: #e9f0ff;
}

QTabBar::tab:first {
    margin-left: 8px;  /* 左侧留空 */
}

QTabBar::tab:last {
    margin-right: 8px; /* 右侧留空 */
}

/* 如果标签页数量少，可以固定高度 */
QTabBar::tab {
    min-height: 36px;
}
QTreeWidget {
    background-color: transparent;
    border: none;
    outline: none;
}
QTreeWidget::item {
    padding: 6px;
    border-radius: 6px;
}
QTreeWidget::item:selected {
    background-color: #e9f0ff;
    color: #0d6efd;
}
QTreeWidget::item:hover {
    background-color: #f1f5f9;
}
QTreeWidget::branch {
    background: transparent;
}
QTreeWidget::branch:has-children:!has-siblings:closed,
QTreeWidget::branch:closed:has-children:has-siblings {
    border-image: none;
    image: none;
}
QTreeWidget::branch:open:has-children:!has-siblings,
QTreeWidget::branch:open:has-children:has-siblings {
    border-image: none;
    image: none;
}
"""


class Worker(QThread):
    """支持流式输出的工作线程"""
    finished = Signal(str, str, list)          # 完整回复（或空），错误信息
    stream_chunk = Signal(str, str)      # 发送者，文本块

    def __init__(self, messages, mode):
        super().__init__()
        self.messages = messages
        self.mode = mode
        self._is_running = True

    def run(self):
        full_reply = ""
        try:
            for chunk in run_agent_stream(self.messages, mode=self.mode):
                if not self._is_running:
                    break
                if chunk:
                    full_reply += chunk
                    self.stream_chunk.emit("assistant", chunk)
            # 生成结束，self.messages 已被 run_agent_stream 更新
            self.finished.emit(full_reply, "", self.messages)
        except Exception as e:
            error_msg = f"执行出错：{e}\n{traceback.format_exc()}"
            self.finished.emit("", error_msg, self.messages)

    def stop(self):
        self._is_running = False


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM科研助手: 早上好，物理学")
        self.setMinimumSize(1200, 800)

        # ========== 全局样式表 ==========
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # 创建滚动区域
        self.scroll_area = QScrollArea()

        # 初始化历史管理器
        self.history_manager = HistoryManager()
        self.current_conversation = None  # 当前对话对象

        # 创建中央分割器
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ---------- 左侧对话列表（树形） ----------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # 新建对话按钮（保持不变）
        self.new_conv_btn = QPushButton("＋ 新建对话")
        self.new_conv_btn.setObjectName("newConvBtn")
        self.new_conv_btn.clicked.connect(self.create_new_conversation)
        left_layout.addWidget(self.new_conv_btn)

        # 新增分组按钮
        self.new_group_btn = QPushButton("＋ 新建分组")
        self.new_group_btn.setObjectName("newGroupBtn")
        self.new_group_btn.clicked.connect(self.create_new_group)
        left_layout.addWidget(self.new_group_btn)

        # 树形对话列表
        self.conv_tree = QTreeWidget()
        self.conv_tree.setHeaderHidden(True)
        self.conv_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.conv_tree.customContextMenuRequested.connect(self.show_tree_context_menu)
        self.conv_tree.itemClicked.connect(self.on_tree_item_clicked)
        left_layout.addWidget(self.conv_tree)

        splitter.addWidget(left_widget)
        splitter.setStretchFactor(0, 1)  # 左侧宽度可微调

        # ---------- 右侧聊天区域 ----------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # 顶部模式选择
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("模式："))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["openai", "deepseek", "compare"])
        self.mode_combo.setCurrentIndex(1)
        top_layout.addWidget(self.mode_combo)
        top_layout.addStretch()
        right_layout.addLayout(top_layout)

        # 聊天 WebView
        '''
        self.chat_view = QWebEngineView()
        self.setup_chat_html()
        self.page_loaded = False
        self.chat_view.loadFinished.connect(self.on_page_loaded)
        self.chat_view.page().navigationRequested.connect(self.on_navigation_requested)
        right_layout.addWidget(self.chat_view)
        # '''
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.NoFrame)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_content.setStyleSheet("background-color: #fafaff;")  # 例如浅灰色
        right_layout.addWidget(self.scroll_area)

        # 底部输入区
        bottom_layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("输入消息... (Enter发送，Shift+Enter换行)")
        self.input_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.input_edit.setMinimumHeight(self.line_height() * 2 + 20)
        self.input_edit.setMaximumHeight(self.line_height() * 8 + 10)
        # self.input_edit.textChanged.connect(self.adjust_input_height)
        self.send_btn = QPushButton("发送")
        self.send_btn.setObjectName("sendBtn")
        #self.send_btn.setFixedWidth(60)
        self.send_btn.clicked.connect(self.send_message)

        # 新增：中止和重新生成按钮，垂直排列
        self.stop_btn = QPushButton("中止")
        self.stop_btn.setObjectName("stopBtn")
        #self.stop_btn.setFixedWidth(80)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generation)

        self.regenerate_btn = QPushButton("重新生成")
        self.regenerate_btn.setObjectName("regenerateBtn")
        #self.regenerate_btn.setFixedWidth(80)
        self.regenerate_btn.setEnabled(False)
        self.regenerate_btn.clicked.connect(self.regenerate_message)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.regenerate_btn)

        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.send_btn)
        input_layout.addLayout(button_layout)  # 将垂直按钮组放到发送按钮右侧
        bottom_layout.addLayout(input_layout)
        right_layout.addLayout(bottom_layout)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 3)  # 右侧占更多空间

        # 添加菜单栏
        menubar = self.menuBar()
        #settings_menu = menubar.addMenu("设置")
        settings_action = QAction("首选项", self)
        settings_action.triggered.connect(self.open_settings_dialog)
        #settings_menu.addAction(settings_action)
        menubar.addAction(settings_action)

        # 添加“知识库”菜单
        search_action = QAction("知识库", self)
        search_action.triggered.connect(self.open_knowledge_base)
        menubar.addAction(search_action)

        # 事件过滤、状态等
        self.input_edit.installEventFilter(self)
        self.processing = False
        self.current_message_id = None
        self._last_ai_message_id = None

        # 其他原有属性
        self.input_edit.installEventFilter(self)
        self.processing = False
        self.current_message_id = None  # 用于流式追加消息
        self._last_ai_message_id = None

        # 加载历史对话列表（树形）
        self.refresh_tree()
        # 获取所有对话 ID（从 history_manager 缓存中获取）
        # all_conv_ids = list(self.history_manager.conversations.keys())
        '''
        if all_conv_ids:
            # 默认选中第一个对话（按更新时间排序，我们可以在 refresh_tree 后手动选中第一个）
            # 最简单的：直接切换到第一个对话
            first_conv_id = all_conv_ids[0]
            self.switch_conversation_by_id(first_conv_id)
        else:
            self.create_new_conversation()
        # '''
        self.current_conversation = None
        self.clear_chat()
        self.add_chat_message("system", "LLM科研助手已启动。请选择模式并开始对话。")

        self.update_regenerate_button_state()

        # ICON
        icon_path = "resources/app_icon.ico"  # 根据实际路径调整
        self.setWindowIcon(QIcon(icon_path))

    def open_knowledge_base(self):
        dialog = KnowledgeBaseDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.show()

    def stop_generation(self):
        """中止当前的生成"""
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.stop()
            # 中止后，worker 会触发 finished 信号，后续状态在 on_worker_finished 中恢复
            # 可以添加一条系统消息提示用户
            self.add_chat_message("system", "生成已中止")

    def regenerate_message(self):
        """重新生成最后一条用户消息的回复"""
        if self.processing:
            return
        if not self.current_conversation:
            return

        # 找到最后一条用户消息
        user_msg_index = -1
        for i in range(len(self.current_conversation.messages) - 1, -1, -1):
            if self.current_conversation.messages[i]["role"] == "user":
                user_msg_index = i
                break
        if user_msg_index == -1:
            QMessageBox.information(self, "提示", "没有用户消息可供重新生成")
            return

        # 如果最后一条是 AI 消息，则删除它
        if (len(self.current_conversation.messages) > user_msg_index + 1 and
                self.current_conversation.messages[-1]["role"] == "assistant"):
            self._remove_last_ai_message()

        # 构建消息列表（包含系统提示和当前对话所有消息）
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in self.current_conversation.messages:
            messages.append(msg.copy())
        messages = self.clean_messages_for_api(messages)

        # 启动生成
        self._run_with_messages(messages)

    def _remove_last_ai_message(self):
        """删除最后一条 AI 消息（UI 和历史）"""
        if not self.current_conversation:
            return
        if self.current_conversation.messages and self.current_conversation.messages[-1]["role"] == "assistant":
            # 从历史中移除
            self.current_conversation.messages.pop()
            self.current_conversation.updated_at = datetime.now().isoformat()
            # 从 UI 中移除最后一个 .ai 容器
            js = """
            (function() {
                var containers = document.querySelectorAll('.message-container.ai');
                if (containers.length > 0) {
                    containers[containers.length - 1].remove();
                    return true;
                }
                return false;
            })();
            """
            # self.chat_view.page().runJavaScript(js)
            self._last_ai_message_id = None  # 重置流式追加标记

    def _run_with_messages(self, messages):
        """使用给定的消息列表启动生成（公用方法）"""
        mode = self.mode_combo.currentText()
        self.processing = True
        self.send_btn.setEnabled(False)
        self.input_edit.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.regenerate_btn.setEnabled(False)
        self._last_ai_message_id = None   # 重置，使新消息作为独立气泡

        self.worker = Worker(messages, mode)
        self.worker.stream_chunk.connect(self.on_stream_chunk)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def refresh_conv_list(self):
        """刷新左侧对话列表"""
        self.conv_list.clear()
        conversations = self.history_manager.get_all_sorted()  # 返回元数据列表
        for meta in conversations:  # meta 是字典
            updated = datetime.fromisoformat(meta["updated_at"]).strftime("%m-%d %H:%M")
            display_text = f"{meta['title']}\n{updated}"
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, meta["conv_id"])
            self.conv_list.addItem(item)

    def create_new_conversation(self):
        """创建新对话并切换到它"""
        # 如果有正在进行的对话，先保存
        if self.current_conversation:
            self.history_manager.save_conversation(self.current_conversation)
        # 创建新对话
        conv = self.history_manager.create_conversation(title="新对话")
        # print("新建对话 ID:", conv.conv_id)
        # print("当前缓存中的对话 IDs:", list(self.history_manager.conversations.keys()))
        self.current_conversation = conv
        self.refresh_tree()
        # 选中新对话
        for i in range(self.conv_list.count()):
            if self.conv_list.item(i).data(Qt.UserRole) == conv.conv_id:
                self.conv_list.setCurrentRow(i)
                break
        # 清空聊天界面
        self.clear_chat()
        # 添加系统欢迎消息
        self.add_chat_message("system", "新对话已创建。")
        self.update_regenerate_button_state()

    def switch_conversation(self, item):
        """切换对话"""
        conv_id = item.data(Qt.UserRole)
        if not conv_id:
            return
        # 保存当前对话
        if self.current_conversation:
            self.history_manager.save_conversation(self.current_conversation)
        # 加载新对话
        conv = self.history_manager.get_conversation(conv_id)
        if conv:
            self.current_conversation = conv
            self.history_manager.set_current(conv_id)
            # 清空聊天界面并加载历史消息
            self.clear_chat()
            # 只显示用户、助手、系统消息，跳过工具消息
            for msg in conv.messages:
                if msg["role"] in ("user", "assistant", "system"):
                    self.add_chat_message(msg["role"], msg["content"])
            # 滚动到底部
            self.scroll_to_bottom()
        self.update_regenerate_button_state()

    def clear_chat(self):
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def scroll_to_bottom(self):
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def show_conv_context_menu(self, pos: QPoint):
        """右键菜单：重命名、删除"""
        item = self.conv_list.itemAt(pos)
        if not item:
            return
        conv_id = item.data(Qt.UserRole)
        conv = self.history_manager.get_conversation(conv_id)
        if not conv:
            return
        menu = QMenu()
        rename_action = QAction("重命名", self)
        delete_action = QAction("删除", self)
        menu.addAction(rename_action)
        menu.addAction(delete_action)
        action = menu.exec(self.conv_list.mapToGlobal(pos))
        if action == rename_action:
            self.rename_conversation(conv)
        elif action == delete_action:
            self.delete_conversation(conv, item)

    def rename_conversation(self, conv: Conversation):
        new_title, ok = QInputDialog.getText(self, "重命名对话", "输入新标题：",
                                             text=conv.title)
        if ok and new_title.strip():
            conv.title = new_title.strip()
            conv.updated_at = datetime.now().isoformat()
            self.history_manager.save_conversation(conv)
            self.refresh_tree()
            # 如果当前就是该对话，选中它
            if self.current_conversation and self.current_conversation.conv_id == conv.conv_id:
                self.current_conversation.title = conv.title

    def delete_conversation(self, conv: Conversation, item: QListWidgetItem):
        reply = QMessageBox.question(self, "确认删除", f"确定要删除对话“{conv.title}”吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.history_manager.delete_conversation(conv.conv_id)
            row = self.conv_list.row(item)
            self.conv_list.takeItem(row)
            if self.current_conversation and self.current_conversation.conv_id == conv.conv_id:
                # 当前对话被删除，切换到另一个
                if self.conv_list.count() > 0:
                    self.conv_list.setCurrentRow(0)
                    self.switch_conversation(self.conv_list.item(0))
                else:
                    self.create_new_conversation()

    '''
    def on_navigation_requested(self, request):
        """拦截导航，http/https链接在外部浏览器打开，其他放行"""
        url = request.url()
        if url.scheme().startswith('http'):
            QDesktopServices.openUrl(url)
            request.reject()  # 阻止在WebView内加载
        else:
            request.accept()
    # '''

    '''
    def setup_chat_html(self):
        """设置WebView的HTML内容，增强公式容错"""
        html = """
        <html>
        <head>
        <meta charset="utf-8">
        <!-- Markdown 解析库 -->
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <!-- MathJax 配置：启用noerrors扩展，处理轻微语法错误 -->
        <script>
        window.loadConversation = function(html) {
            document.getElementById('chat').innerHTML = html;
            window.scrollTo(0, document.body.scrollHeight);
        };
        MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                packages: {'[+]': ['noerrors']}  // 启用noerrors避免单个公式失败
            },
            options: {
                skipHtmlTags: ['noscript', 'style', 'textarea', 'pre', 'code'],
                ignoreHtmlClass: 'tex2jax_ignore',
                processHtmlClass: 'tex2jax_process'
            },
            loader: {load: ['[tex]/noerrors']},
            startup: {
                pageReady: () => {
                    console.log('MathJax 初始化完成');
                    window.mathJaxReady = true;
                }
            }
        };
        </script>
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
        <style>
            /* 整个滚动条区域 */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
                background: transparent;
            }
            /* 滚动条轨道 */
            ::-webkit-scrollbar-track {
                background: transparent;
                border-radius: 4px;
            }
            /* 滚动条滑块 */
            ::-webkit-scrollbar-thumb {
                background: #c1c9d2;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #a0aab5;
            }
            body {
                font-family: 'Segoe UI', '微软雅黑', Arial, sans-serif;
                background: #f5f7fa;
                margin: 0;
                padding: 20px;
            }
            #chat {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            .message-container {
                display: flex;
                margin-bottom: 4px;
            }
            .message-container.user {
                justify-content: flex-end;
            }
            .message-container.ai {
                justify-content: flex-start;
            }
            .message-container.system {
                justify-content: center;
            }
            .bubble {
                max-width: 70%;
                padding: 12px 18px;
                border-radius: 24px;
                word-wrap: break-word;
                white-space: normal;
                font-size: 14px;
                line-height: 1.5;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }
            .user .bubble {
                background-color: #0d6efd;
                color: white;
                border-bottom-right-radius: 4px;
            }
            .ai .bubble {
                background-color: white;
                color: #1e293b;
                border-bottom-left-radius: 4px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }
            .system .bubble {
                background-color: #e9ecef;
                color: #495057;
                font-size: 12px;
                border-radius: 16px;
                padding: 6px 14px;
                box-shadow: none;
            }
            pre {
                background: #f8f9fa;
                padding: 12px;
                border-radius: 12px;
                overflow-x: auto;
                border: 1px solid #e9ecef;
            }
            code {
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 13px;
                background: #f1f3f5;
                padding: 2px 6px;
                border-radius: 6px;
            }
            pre code {
                background: transparent;
                padding: 0;
            }
            .MathJax_SVG {
                display: inline-block;
            }
        </style>
        </head>
        <body>
        <div id="chat"></div>
        <script>
            // 轻量级修复常见公式格式问题
            function fixCommonMathErrors(text) {
                // 修复空上标： ^{}  ->  ^{}
                // 但空上标可能只是模型输出错误，直接移除可能导致语义变化，这里我们仅作示例
                // 更常见的错误：在align环境中误用了不完整的标记，我们暂时不做自动修改
                return text;
            }

            window.addMessage = function(sender, text, isAppend = false) {
                console.log('addMessage', sender, isAppend);
                // 如果isAppend为true，表示追加到最后一个同类型消息的气泡
                if (isAppend && sender === 'ai') {
                    // 获取最后一个.ai消息的气泡
                    const containers = document.querySelectorAll('.message-container.ai');
                    if (containers.length > 0) {
                        const lastContainer = containers[containers.length - 1];
                        const bubble = lastContainer.querySelector('.bubble');
                        // 将新文本追加到现有内容后（需要解析Markdown？但流式片段可能不完整）
                        // 简单追加文本，并用marked解析新片段？但可能破坏已有HTML
                        // 更稳妥：将新文本作为纯文本追加（不解析Markdown），或者重新解析整个内容
                        // 此处采用重新解析整个bubble内容（累积文本）
                        const currentText = bubble.getAttribute('data-fulltext') || '';
                        const newFullText = currentText + text;
                        bubble.setAttribute('data-fulltext', newFullText);
                        // 重新解析整个文本（会导致前面已渲染的公式重新渲染，但可接受）
                        let htmlContent;
                        if (typeof marked !== 'undefined') {
                            htmlContent = marked.parse(newFullText);
                        } else {
                            htmlContent = newFullText.replace(/&/g, '&amp;')
                                                     .replace(/</g, '&lt;')
                                                     .replace(/>/g, '&gt;')
                                                     .replace(/\\n/g, '<br>');
                        }
                        bubble.innerHTML = htmlContent;
                        // 触发MathJax重新渲染整个气泡
                        if (window.MathJax && MathJax.typesetPromise) {
                            MathJax.typesetPromise([bubble]).catch(err => console.error(err));
                        }
                        var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                        var windowHeight = window.innerHeight;
                        var documentHeight = document.documentElement.scrollHeight;
                        var distanceToBottom = documentHeight - (scrollTop + windowHeight);
                        if (distanceToBottom < 50) {
                            window.scrollTo(0, document.body.scrollHeight);
                        }
                        return true;
                    }
                }
                // 否则创建新消息
                let fullText = text;
                let htmlContent;
                if (typeof marked !== 'undefined') {
                    htmlContent = marked.parse(fullText);
                } else {
                    htmlContent = fullText.replace(/&/g, '&amp;')
                                          .replace(/</g, '&lt;')
                                          .replace(/>/g, '&gt;')
                                          .replace(/\\n/g, '<br>');
                }
                var container = document.createElement('div');
                container.className = 'message-container ' + sender;
                var bubble = document.createElement('div');
                bubble.className = 'bubble';
                bubble.setAttribute('data-fulltext', fullText);
                bubble.innerHTML = htmlContent;
                container.appendChild(bubble);
                document.getElementById('chat').appendChild(container);
                if (window.MathJax && MathJax.typesetPromise) {
                    MathJax.typesetPromise([bubble]).catch(err => console.error(err));
                }
                window.scrollTo(0, document.body.scrollHeight);
                return true;
            };
            console.log('addMessage 函数已定义');
        </script>
        </body>
        </html>
        """
        self.chat_view.setHtml(html)
    # '''

    '''
    def on_page_loaded(self, ok):
        if ok:
            self.page_loaded = True
            if self.current_conversation:
                self.clear_chat()
                for msg in self.current_conversation.messages:
                    if msg["role"] in ("user", "assistant", "system"):
                        self.add_chat_message(msg["role"], msg["content"])
        else:
            print("页面加载失败")
    # '''

    def line_height(self):
        fm = self.input_edit.fontMetrics()
        return fm.lineSpacing()

    def adjust_input_height(self):
        doc = self.input_edit.document()
        block_count = doc.blockCount()
        desired = block_count * self.line_height() + 10
        desired = max(self.input_edit.minimumHeight(), min(desired, self.input_edit.maximumHeight()))
        self.input_edit.setFixedHeight(desired)

    def eventFilter(self, obj, event):
        if obj == self.input_edit and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
                if event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier):
                    self.input_edit.insertPlainText("\n")
                    return True
                else:
                    self.send_message()
                    return True
        return super().eventFilter(obj, event)

    def clean_messages_for_api(self, messages: list) -> list:
        """清理消息列表：移除 reasoning_content，并修复/过滤无效的 tool 消息"""
        cleaned = []
        for i, msg in enumerate(messages):
            new_msg = msg.copy()

            # 检查 tool 消息的完整性
            if new_msg.get("role") == "tool":
                if "tool_call_id" not in new_msg:
                    print(f"!!! 发现损坏的 tool 消息 (索引 {i}):")
                    #print(json.dumps(new_msg, ensure_ascii=False, indent=2))
                    print("    临时将角色改为 assistant 并继续")
                    # 临时修复：将其视为 assistant 消息
                    new_msg["role"] = "assistant"
                    # 移除可能存在的 tool_calls 字段（如果有）
                    new_msg.pop("tool_calls", None)
            cleaned.append(new_msg)
        return cleaned

    def send_message(self):
        if self.processing:
            return
        text = self.input_edit.toPlainText().strip()
        if not text:
            return

        # 如果没有当前对话，自动创建一个
        if self.current_conversation is None:
            conv = self.history_manager.create_conversation(title="新对话", group_id=None)
            self.current_conversation = conv
            self.refresh_tree()
            self._highlight_conversation_in_tree(conv.conv_id)

        # 快捷指令处理（保持不变）
        if text == "早上好 arxiv":
            text = "早上好 arxiv，今天的新文章"
            self.input_edit.setPlainText(text)
        elif text == "近期文章 arxiv":
            text = "近期文章 arxiv，请整理一下"
            self.input_edit.setPlainText(text)

        self.input_edit.clear()
        self.adjust_input_height()
        self.add_chat_message("user", text)

        # 将用户消息添加到当前对话对象
        if self.current_conversation:
            self.current_conversation.add_message("user", text)

            # 构建完整的消息列表
            # 替换原来的循环
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in self.current_conversation.messages:
                messages.append(msg.copy())  # ✅ 复制完整消息，保留所有字段
            messages = self.clean_messages_for_api(messages)
        else:
            # 没有当前对话（不应发生）
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

        # 启动生成
        self._run_with_messages(messages)

    def on_stream_chunk(self, sender, chunk):
        # print(f"on_stream_chunk: sender={sender}, chunk='{chunk[:50]}'")  # 调试
        if sender != "assistant":
            return  # 忽略非助理消息（如 tool）

        if not chunk.strip():  # 过滤空内容
            return

        if self._last_ai_message_id is None:
            # 第一个块：创建新气泡
            self.add_chat_message("assistant", chunk, isAppend=False)
            self._last_ai_message_id = "assistant"  # 标记已有助理消息
            # 初始化历史记录
            if self.current_conversation:
                self.current_conversation.add_message("assistant", chunk)
        else:
            # 后续块：追加到最后一个助理气泡
            self.add_chat_message("assistant", chunk, isAppend=True)
            # 更新历史记录中的最后一条消息
            if self.current_conversation and self.current_conversation.messages:
                last_msg = self.current_conversation.messages[-1]
                if last_msg["role"] == "assistant":
                    last_msg["content"] += chunk
                    self.current_conversation.updated_at = datetime.now().isoformat()

    '''
    def append_to_last_ai_message(self, chunk):
        """向最后一条AI消息追加内容"""
        safe_chunk = json.dumps(chunk)
        js = f"""
        (function() {{
            const containers = document.querySelectorAll('.message-container.ai');
            if (containers.length > 0) {{
                const lastContainer = containers[containers.length - 1];
                const bubble = lastContainer.querySelector('.bubble');
                const currentText = bubble.getAttribute('data-fulltext') || '';
                const newText = currentText + {safe_chunk};
                bubble.setAttribute('data-fulltext', newText);
                let htmlContent;
                if (typeof marked !== 'undefined') {{
                    htmlContent = marked.parse(newText);
                }} else {{
                    htmlContent = newText.replace(/&/g, '&amp;')
                                         .replace(/</g, '&lt;')
                                         .replace(/>/g, '&gt;')
                                         .replace(/\\\\n/g, '<br>');
                }}
                bubble.innerHTML = htmlContent;
                if (window.MathJax && MathJax.typesetPromise) {{
                    MathJax.typesetPromise([bubble]).catch(err => console.error(err));
                }}
                // 自动滚动判断：如果距离底部小于50像素，则滚动到底部
                var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                var windowHeight = window.innerHeight;
                var documentHeight = document.documentElement.scrollHeight;
                var distanceToBottom = documentHeight - (scrollTop + windowHeight);
                if (distanceToBottom < 50) {{
                    window.scrollTo(0, document.body.scrollHeight);
                }}
                return true;
            }}
            return false;
        }})();
        """
        # self.chat_view.page().runJavaScript(js)
    # '''

    def on_worker_finished(self, reply, error, final_messages):
        self.processing = False
        self.send_btn.setEnabled(True)
        self.input_edit.setEnabled(True)
        self.stop_btn.setEnabled(False)  # 中止按钮恢复禁用
        self.update_regenerate_button_state()  # 更新重新生成按钮状态
        self.input_edit.setFocus()
        self._last_ai_message_id = None

        # 原有错误处理和保存逻辑保持不变 ...
        if error:
            QMessageBox.critical(self, "错误", error)
            self.add_chat_message("system", f"错误：{error}")
        else:
            # 更新对话
            if self.current_conversation:
                new_messages = [msg for msg in final_messages if msg["role"] != "system"]
                self.current_conversation.messages = new_messages

                self.current_conversation.updated_at = datetime.now().isoformat()
                if self.current_conversation.title == "新对话":
                    new_title = self.history_manager.generate_title_from_messages(self.current_conversation)
                    if new_title:
                        self.current_conversation.title = new_title
                self.history_manager.save_conversation(self.current_conversation)
                self.refresh_tree()

        last_bubble = self.find_last_assistant_bubble()
        if last_bubble:
            last_bubble.finalize()

        self._save_llm_summary(final_messages)

        self.worker.deleteLater()

    def update_regenerate_button_state(self):
        """根据当前状态更新重新生成按钮的可用性"""
        if self.processing:
            self.regenerate_btn.setEnabled(False)
        else:
            # 检查对话历史中是否有用户消息
            if self.current_conversation and any(msg['role'] == 'user' for msg in self.current_conversation.messages):
                self.regenerate_btn.setEnabled(True)
            else:
                self.regenerate_btn.setEnabled(False)

    def add_chat_message(self, role: str, text: str, isAppend: bool = False):
        # 过滤空文本（无论是新消息还是追加，空内容都不应处理）
        if not text or not text.strip():
            # print(f"add_chat_message: 忽略空消息 role={role}")
            return

        # print(f"add_chat_message: role={role}, text='{text[:50]}', isAppend={isAppend}")
        if isAppend and role == 'assistant':
            last_bubble = self.find_last_assistant_bubble()
            if last_bubble:
                last_bubble.update_content(text)
                # 延迟判断滚动，确保布局已更新
                QTimer.singleShot(0, self.scroll_if_needed)
                return

        # 创建新气泡
        bubble = MessageBubble(role, text)
        self.scroll_layout.addWidget(bubble)
        self.scroll_if_needed()

    def find_last_assistant_bubble(self):
        for i in range(self.scroll_layout.count() - 1, -1, -1):
            item = self.scroll_layout.itemAt(i)
            widget = item.widget()
            if widget and isinstance(widget, MessageBubble) and widget.role == 'assistant':
                return widget
        return None

    def _remove_last_ai_message(self):
        """删除最后一条 AI 消息（UI 和历史）"""
        if not self.current_conversation:
            return
        if self.current_conversation.messages and self.current_conversation.messages[-1]["role"] == "assistant":
            # 从历史中移除
            self.current_conversation.messages.pop()
            self.current_conversation.updated_at = datetime.now().isoformat()

        # 从 UI 中移除最后一个助理气泡
        last_bubble = self.find_last_assistant_bubble()
        if last_bubble:
            last_bubble.deleteLater()

    def closeEvent(self, event):
        if self.current_conversation:
            self.history_manager.save_conversation(self.current_conversation)
        event.accept()

    def delete_conversation(self, conv: Conversation, item: QListWidgetItem):
        reply = QMessageBox.question(self, "确认删除", f"确定要删除对话“{conv.title}”吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 1. 调用历史管理器删除（仅传递ID）
            self.history_manager.delete_conversation(conv.conv_id)

            # 2. 从列表控件中移除该项
            row = self.conv_list.row(item)
            self.conv_list.takeItem(row)

            # 3. 如果删除的是当前对话，则切换或新建
            if self.current_conversation and self.current_conversation.conv_id == conv.conv_id:
                self.current_conversation = None  # 关键：先清除当前对话引用
                if self.conv_list.count() > 0:
                    self.conv_list.setCurrentRow(0)
                    self.switch_conversation(self.conv_list.item(0))
                else:
                    self.create_new_conversation()

    def open_settings_dialog(self):
        dialog = SettingsDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)  # 关闭时自动销毁
        dialog.show()

    def _save_llm_summary(self, final_messages):
        """从 final_messages 中提取论文 ID 和最后一条助手消息，保存到数据库"""
        import json
        from agent.tools import update_paper_llm_summary

        # 1. 查找工具调用 analyze_paper 的结果，获取 paper_id
        paper_id = None
        for msg in final_messages:
            if msg.get("role") == "tool":
                # 尝试解析内容，假设它包含了 paper_id
                try:
                    content = msg.get("content", "")
                    if content:
                        result = json.loads(content)
                        if isinstance(result, dict) and "paper_id" in result:
                            paper_id = result["paper_id"]
                            break
                except:
                    continue

        if not paper_id:
            return

        # 检查是否已有 LLM 总结
        paper = get_paper_by_id(paper_id)
        if paper and paper.get("llm_summary"):
            print(f"论文 {paper_id} 已有 LLM 总结，跳过保存")
            return

        # 2. 获取最后一条 assistant 消息作为总结
        assistant_msgs = [msg for msg in final_messages if msg["role"] == "assistant"]
        if not assistant_msgs:
            return
        last_assistant = assistant_msgs[-1]["content"]

        # 3. 存入数据库
        try:
            update_paper_llm_summary(paper_id, last_assistant)
            print(f"LLM 总结已保存到论文 {paper_id}")
        except Exception as e:
            print(f"保存 LLM 总结失败: {e}")

    def refresh_tree(self):
        """刷新左侧树形结构"""
        # print("刷新树形结构")
        grouped = self.history_manager.get_conversations_by_group()
        groups_info = {g["group_id"]: g for g in self.history_manager.get_all_groups()}

        self.conv_tree.clear()

        # 1. 先为所有分组创建节点（即使没有对话）
        group_items = {}
        for group_id, group_meta in groups_info.items():
            group_item = QTreeWidgetItem(self.conv_tree)
            group_item.setText(0, group_meta['name'])
            group_item.setData(0, Qt.UserRole, f"group:{group_id}")
            group_items[group_id] = group_item

        # 2. 将对话添加到对应的分组节点下
        for group_id, convs in grouped.items():
            if group_id is None:
                # 未分组对话稍后处理
                continue
            if group_id in group_items:
                group_item = group_items[group_id]
                for conv_meta in convs:
                    self._add_conversation_item(group_item, conv_meta)

        # 3. 处理未分组对话
        ungrouped_convs = grouped.get(None, [])
        if ungrouped_convs:
            ungrouped_item = QTreeWidgetItem(self.conv_tree)
            ungrouped_item.setText(0, "未分组")
            ungrouped_item.setData(0, Qt.UserRole, "group:ungrouped")
            for conv_meta in ungrouped_convs:
                self._add_conversation_item(ungrouped_item, conv_meta)

        self.conv_tree.expandAll()

        self.conv_tree.expandAll()

    def _add_conversation_item(self, parent_item, conv_meta):
        # print("添加对话项:", conv_meta["conv_id"], conv_meta["title"])
        """在指定父项下添加一个对话项"""
        item = QTreeWidgetItem(parent_item)
        # 格式化时间
        updated = datetime.fromisoformat(conv_meta["updated_at"]).strftime("%m-%d %H:%M")
        display_text = f"{conv_meta['title']}\n{updated}"
        item.setText(0, display_text)
        item.setData(0, Qt.UserRole, f"conv:{conv_meta['conv_id']}")
        # 设置图标（可选）
        # item.setIcon(0, QIcon("resources/chat_icon.png"))

    def on_tree_item_clicked(self, item, column):
        """处理树项点击：如果是对话则切换，如果是分组则展开/折叠"""
        data = item.data(0, Qt.UserRole)
        if data and data.startswith("conv:"):
            conv_id = data.split(":", 1)[1]
            self.switch_conversation_by_id(conv_id)

    def switch_conversation_by_id(self, conv_id: str):
        """根据 ID 切换对话（由树点击或外部调用）"""
        if self.current_conversation and self.current_conversation.conv_id == conv_id:
            return  # 已经是当前对话
        # 保存当前
        if self.current_conversation:
            self.history_manager.save_conversation(self.current_conversation)
        conv = self.history_manager.get_conversation(conv_id)
        if conv:
            self.current_conversation = conv
            self.clear_chat()
            for msg in conv.messages:
                if msg["role"] in ("user", "assistant", "system") and msg.get("content", "").strip():
                    self.add_chat_message(msg["role"], msg["content"])
            # 强制处理事件，确保布局完成
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            self._retry_scroll_to_bottom(4)
        self.update_regenerate_button_state()

    def _retry_scroll_to_bottom(self, retries=4, interval=50):
        scrollbar = self.scroll_area.verticalScrollBar()
        if scrollbar.maximum() > 0:
            scrollbar.setValue(scrollbar.maximum())
        elif retries > 0:
            QTimer.singleShot(interval, lambda: self._retry_scroll_to_bottom(retries - 1, interval))

    def _highlight_conversation_in_tree(self, conv_id):
        """在树中选中对应的对话项"""

        def find_item(root):
            for i in range(root.childCount()):
                child = root.child(i)
                data = child.data(0, Qt.UserRole)
                if data and data == f"conv:{conv_id}":
                    return child
                found = find_item(child)
                if found:
                    return found
            return None

        item = find_item(self.conv_tree.invisibleRootItem())
        if item:
            self.conv_tree.setCurrentItem(item)

    def show_tree_context_menu(self, pos):
        """右键菜单：根据点击项类型显示不同菜单"""
        item = self.conv_tree.itemAt(pos)
        if not item:
            # 在空白处点击，显示全局菜单（新建分组等）
            menu = QMenu()
            new_group_action = QAction("新建分组", self)
            new_group_action.triggered.connect(self.create_new_group)
            menu.addAction(new_group_action)
            menu.exec(self.conv_tree.mapToGlobal(pos))
            return

        data = item.data(0, Qt.UserRole)
        if data and data.startswith("group:"):
            # 分组项
            group_id = data.split(":", 1)[1]
            menu = QMenu()
            rename_action = QAction("重命名分组", self)
            rename_action.triggered.connect(lambda: self.rename_group(group_id))
            delete_action = QAction("删除分组", self)
            delete_action.triggered.connect(lambda: self.delete_group(group_id))
            new_conv_action = QAction("在此分组新建对话", self)
            new_conv_action.triggered.connect(lambda: self.create_new_conversation(group_id))
            menu.addAction(rename_action)
            menu.addAction(delete_action)
            menu.addAction(new_conv_action)
            menu.exec(self.conv_tree.mapToGlobal(pos))
        elif data and data.startswith("conv:"):
            # 对话项
            conv_id = data.split(":", 1)[1]
            conv = self.history_manager.get_conversation(conv_id)
            if not conv:
                return
            menu = QMenu()
            rename_action = QAction("重命名对话", self)
            rename_action.triggered.connect(lambda: self.rename_conversation(conv))
            delete_action = QAction("删除对话", self)
            delete_action.triggered.connect(lambda: self.delete_conversation(conv, None))  # 注意 item 参数
            # 移动对话到其他分组
            move_menu = QMenu("移动到分组", self)
            # 获取所有分组
            groups = self.history_manager.get_all_groups()
            # 添加一个“未分组”选项
            move_to_ungrouped = QAction("未分组", self)
            move_to_ungrouped.triggered.connect(lambda: self.move_conversation(conv_id, None))
            move_menu.addAction(move_to_ungrouped)
            for group in groups:
                group_action = QAction(group["name"], self)
                group_action.triggered.connect(
                    lambda checked, gid=group["group_id"]: self.move_conversation(conv_id, gid))
                move_menu.addAction(group_action)
            menu.addMenu(move_menu)
            menu.addAction(rename_action)
            menu.addAction(delete_action)
            menu.exec(self.conv_tree.mapToGlobal(pos))

    def create_new_group(self):
        """新建分组（弹出输入框）"""
        name, ok = QInputDialog.getText(self, "新建分组", "输入分组名称：", text="新分组")
        if ok and name.strip():
            self.history_manager.create_group(name.strip())
            self.refresh_tree()

    def rename_group(self, group_id):
        """重命名分组"""
        current_name = self.history_manager.groups[group_id]["name"]
        new_name, ok = QInputDialog.getText(self, "重命名分组", "新名称：", text=current_name)
        if ok and new_name.strip():
            self.history_manager.rename_group(group_id, new_name.strip())
            self.refresh_tree()

    def delete_group(self, group_id):
        """删除分组（询问是否移动对话）"""
        # 简单实现：移动对话到未分组后删除分组
        reply = QMessageBox.question(self, "删除分组", f"确定删除分组？该分组下的对话将移至「未分组」。",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.history_manager.delete_group(group_id, move_to_group_id=None)
            self.refresh_tree()

    def move_conversation(self, conv_id, target_group_id):
        """移动对话到指定分组"""
        self.history_manager.move_conversation(conv_id, target_group_id)
        # 如果移动的是当前对话，同步更新 current_conversation 的 group_id
        if self.current_conversation and self.current_conversation.conv_id == conv_id:
            self.current_conversation.group_id = target_group_id
        self.refresh_tree()

    def create_new_conversation(self, group_id=None):
        """创建新对话，可指定分组"""
        # 保存当前对话
        if self.current_conversation:
            self.history_manager.save_conversation(self.current_conversation)
        conv = self.history_manager.create_conversation(title="新对话", group_id=group_id)
        self.current_conversation = conv
        self.refresh_tree()
        self.clear_chat()
        self.add_chat_message("system", "新对话已创建。")
        # 在树中选中新对话
        self._highlight_conversation_in_tree(conv.conv_id)
        self.update_regenerate_button_state()

    def delete_conversation(self, conv: Conversation, item: QTreeWidgetItem):
        """删除对话（item 参数在此处不需要，可改为直接用 conv_id）"""
        # 这里简化：直接使用 conv 参数
        reply = QMessageBox.question(self, "确认删除", f"确定要删除对话“{conv.title}”吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.history_manager.delete_conversation(conv.conv_id)
            self.refresh_tree()
            if self.current_conversation and self.current_conversation.conv_id == conv.conv_id:
                # 当前对话被删除，切换到另一个
                # 简单切换到第一个可用的对话
                all_convs = list(self.history_manager.conversations.keys())
                if all_convs:
                    self.switch_conversation_by_id(all_convs[0])
                else:
                    self.create_new_conversation()

    def should_auto_scroll(self) -> bool:
        scrollbar = self.scroll_area.verticalScrollBar()
        max_val = scrollbar.maximum()
        current_bottom = scrollbar.value()
        distance_to_bottom = max_val - current_bottom
        # 调试打印，可保留或删除
        # print(f"max={max_val}, bottom={current_bottom}, distance={distance_to_bottom}")
        return distance_to_bottom <= 20

    def scroll_if_needed(self):
        # print("scroll_if_needed called")  # 调试用，可删除
        if self.should_auto_scroll():
            # print("should auto scroll, scrolling...")
            self.scroll_to_bottom()

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumSize(700, 500)  # 稍微增大以容纳标签页
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # 主布局
        main_layout = QVBoxLayout(self)

        # 创建标签页
        tab_widget = QTabWidget()
        tab_widget.setDocumentMode(True)  # 扁平风格
        main_layout.addWidget(tab_widget)

        # ----- 标签页1：首选项 -----
        pref_tab = QWidget()
        pref_layout = QVBoxLayout(pref_tab)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        # 字体大小
        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems(["10px", "11px", "12px", "13px", "14px", "15px", "16px"])
        self.font_size_combo.setCurrentText("14px")
        form_layout.addRow("全局字体大小:", self.font_size_combo)

        # 通用 API 配置
        self.openai_api_key = QLineEdit()
        self.openai_api_key.setEchoMode(QLineEdit.Password)
        form_layout.addRow("OpenAI API Key:", self.openai_api_key)

        self.openai_model = QLineEdit()
        form_layout.addRow("OpenAI Model:", self.openai_model)

        self.deepseek_api_key = QLineEdit()
        self.deepseek_api_key.setEchoMode(QLineEdit.Password)
        form_layout.addRow("DeepSeek API Key:", self.deepseek_api_key)

        self.deepseek_model = QLineEdit()
        form_layout.addRow("DeepSeek Model:", self.deepseek_model)

        self.deepseek_chat_model = QLineEdit()
        form_layout.addRow("DeepSeek Chat Model:", self.deepseek_chat_model)

        # Embedding 开关（首选项版本）
        self.embedding_enabled_pref = QCheckBox()
        form_layout.addRow("启用 Embedding:", self.embedding_enabled_pref)

        # 添加一个说明标签（可选）
        form_layout.addRow("", QLabel("注：详细 Embedding 配置请前往「Embedding 设置」标签页"))

        pref_layout.addLayout(form_layout)
        pref_layout.addStretch()

        # ----- 标签页2：期刊设置 -----
        journal_tab = QWidget()
        journal_layout = QVBoxLayout(journal_tab)

        # 使用 QScrollArea 容纳期刊配置
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        journal_layout.addWidget(scroll_area)

        # 期刊配置内容部件
        journal_content = QWidget()
        scroll_area.setWidget(journal_content)
        self.journal_form_layout = QFormLayout(journal_content)
        self.journal_form_layout.setSpacing(10)

        # 存储每个期刊的输入控件
        self.journal_section_inputs = {}
        self._create_journal_inputs()  # 从 YAML 加载并创建输入框

        journal_layout.addStretch()

        # ----- 标签页3：Embedding 设置 -----
        embed_tab = QWidget()
        embed_layout = QVBoxLayout(embed_tab)

        embed_form = QFormLayout()
        embed_form.setSpacing(10)

        # Embedding 开关（Embedding 页面版本）
        self.embedding_enabled_embed = QCheckBox()
        embed_form.addRow("启用 Embedding:", self.embedding_enabled_embed)

        self.embedder_type = QComboBox()
        self.embedder_type.addItems(["local", "openai"])
        embed_form.addRow("Embedder 类型:", self.embedder_type)

        self.openai_embedding_model = QLineEdit()
        embed_form.addRow("OpenAI Embedding 模型:", self.openai_embedding_model)

        self.local_embedder_name = QLineEdit()
        embed_form.addRow("本地 Embedder 名称:", self.local_embedder_name)

        self.model_path = QLineEdit()
        embed_form.addRow("模型路径:", self.model_path)

        self.data_dir = QLineEdit()
        embed_form.addRow("数据目录:", self.data_dir)

        self.vector_dir = QLineEdit()
        embed_form.addRow("向量存储目录:", self.vector_dir)

        embed_layout.addLayout(embed_form)
        embed_layout.addStretch()

        # 将三个标签页添加到 tab_widget
        tab_widget.addTab(pref_tab, "首选项")
        tab_widget.addTab(journal_tab, "期刊设置")
        tab_widget.addTab(embed_tab, "Embedding 设置")

        # ----- 底部按钮 -----
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        # 加载现有配置
        self._load_config()
        self._load_journals_config()
        self._load_font_size()

        # 同步两个 Embedding 开关的状态（双向连接）
        self.embedding_enabled_pref.stateChanged.connect(
            lambda state: self.embedding_enabled_embed.setCheckState(state)
        )
        self.embedding_enabled_embed.stateChanged.connect(
            lambda state: self.embedding_enabled_pref.setCheckState(state)
        )

    def _load_font_size(self):
        """从 QSettings 加载字体大小设置"""
        settings = QSettings("CyberPardofelis", "LLM科研助手")
        font_size = settings.value("font_size", "10px")
        self.font_size_combo.setCurrentText(font_size)

    def _create_journal_inputs(self):
        """读取 YAML 文件，为每个期刊创建 section 输入框"""
        yaml_path = self._get_yaml_path()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                journals_data = yaml.safe_load(f)
            journals = journals_data.get('journals', {})
        except Exception as e:
            print(f"读取期刊配置失败: {e}")
            journals = {}

        # 清空原有布局（防止重复）
        while self.journal_form_layout.rowCount() > 0:
            self.journal_form_layout.removeRow(0)

        self.journal_section_inputs.clear()
        for journal_key, journal_info in journals.items():
            section_input = QLineEdit(journal_info.get('section', ''))
            self.journal_section_inputs[journal_key] = section_input
            self.journal_form_layout.addRow(f"{journal_key} section:", section_input)

    def _get_yaml_path(self):
        """获取 journals.yaml 的绝对路径"""
        # SettingsDialog 所在文件位于项目根目录，agent 是子目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, 'agent', 'journals.yaml')

    def _load_config(self):
        """从 QSettings 加载首选项和 Embedding 配置"""
        settings = QSettings("CyberPardofelis", "LLM科研助手")

        # 首选项
        self.openai_api_key.setText(settings.value("OPENAI_API_KEY", ""))
        self.openai_model.setText(settings.value("OPENAI_MODEL", ""))
        self.deepseek_api_key.setText(settings.value("DEEPSEEK_API_KEY", ""))
        self.deepseek_model.setText(settings.value("DEEPSEEK_MODEL", ""))
        self.deepseek_chat_model.setText(settings.value("DEEPSEEK_MODEL_CHAT", ""))

        # Embedding 开关（同时设置两个）
        enabled = settings.value("EMBEDDING_ENABLED", False)
        if isinstance(enabled, str):
            enabled = enabled.lower() in ["true", "1", "yes"]
        self.embedding_enabled_pref.setChecked(enabled)
        self.embedding_enabled_embed.setChecked(enabled)

        # Embedding 其他配置
        self.embedder_type.setCurrentText(settings.value("EMBEDDER_TYPE", "local"))
        self.openai_embedding_model.setText(settings.value("OPENAI_EMBEDDING_MODEL", ""))
        self.local_embedder_name.setText(settings.value("LOCAL_EMBEDDER_NAME", ""))
        self.model_path.setText(settings.value("MODEL_PATH", ""))
        self.data_dir.setText(settings.value("DATA_DIR", ""))
        self.vector_dir.setText(settings.value("VECTOR_DIR", ""))

    def _load_journals_config(self):
        """从 YAML 文件加载期刊 section 值到输入框"""
        yaml_path = self._get_yaml_path()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                journals_data = yaml.safe_load(f)
            journals = journals_data.get('journals', {})
            for journal_key, section_input in self.journal_section_inputs.items():
                if journal_key in journals:
                    section_input.setText(journals[journal_key].get('section', ''))
        except Exception as e:
            print(f"加载期刊配置失败: {e}")

    def _save_journals_config(self):
        """将用户修改的期刊 section 值写回 YAML 文件"""
        yaml_path = self._get_yaml_path()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                journals_data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            journals_data = {'journals': {}}

        if 'journals' not in journals_data:
            journals_data['journals'] = {}

        for journal_key, section_input in self.journal_section_inputs.items():
            if journal_key not in journals_data['journals']:
                journals_data['journals'][journal_key] = {}
            journals_data['journals'][journal_key]['section'] = section_input.text()

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(journals_data, f, allow_unicode=True, default_flow_style=False)

    def accept(self):
        """保存所有配置"""
        settings = QSettings("CyberPardofelis", "LLM科研助手")

        # 保存字体大小
        settings.setValue("font_size", self.font_size_combo.currentText())

        # 保存首选项 API 配置
        settings.setValue("OPENAI_API_KEY", self.openai_api_key.text())
        settings.setValue("OPENAI_MODEL", self.openai_model.text())
        settings.setValue("DEEPSEEK_API_KEY", self.deepseek_api_key.text())
        settings.setValue("DEEPSEEK_MODEL", self.deepseek_model.text())
        settings.setValue("DEEPSEEK_MODEL_CHAT", self.deepseek_chat_model.text())

        # 保存 Embedding 配置（开关以任一为准，两者已同步）
        settings.setValue("EMBEDDING_ENABLED", self.embedding_enabled_pref.isChecked())
        settings.setValue("EMBEDDER_TYPE", self.embedder_type.currentText())
        settings.setValue("OPENAI_EMBEDDING_MODEL", self.openai_embedding_model.text())
        settings.setValue("LOCAL_EMBEDDER_NAME", self.local_embedder_name.text())
        settings.setValue("MODEL_PATH", self.model_path.text())
        settings.setValue("DATA_DIR", self.data_dir.text())
        settings.setValue("VECTOR_DIR", self.vector_dir.text())

        # 保存期刊配置到 YAML
        self._save_journals_config()

        # 应用字体大小（需要重启，但可以提示）
        QMessageBox.information(self, "提示",
                                "配置已保存。部分更改（如字体）需要重启应用才能生效。")
        super().accept()


class KnowledgeBaseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("论文知识库")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(GLOBAL_STYLESHEET)
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
        recent = get_recent_papers(15)  # 调用新函数
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


class MessageBubble(QWidget):
    def __init__(self, role: str, markdown_text: str, parent=None):
        super().__init__(parent)
        self.role = role
        self.full_text = markdown_text
        self.is_final = False
        self.pending_text = ""  # 待追加的文本缓存
        self.update_timer = QTimer()
        self.update_timer.setInterval(750)
        self.update_timer.setSingleShot(True)  # 单次触发，每次有新文本时重启
        self.update_timer.timeout.connect(self._flush_pending)

        # 设置气泡整体样式（背景色、圆角、内边距），占满整个可用宽度
        if role == 'user':
            self.setStyleSheet("""
                background-color: #77aaff;
                border-radius: 10px;
                padding: 8px;
            """)
            alignment = Qt.AlignRight  # 整体靠右（背景从右侧开始）
            # 内容样式：文本左对齐，内容区块靠右（margin-left: auto），宽度限制80%
            content_css = """
                body {
                    background: transparent;
                    color: white;
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                    word-wrap: break-word;
                    max-width: 100%;
                    margin-left: auto;   /* 内容区块靠右 */
                    text-align: left;     /* 文本本身左对齐 */
                }
            """
        elif role == 'assistant':
            self.setStyleSheet("""
                background-color: white;
                border-radius: 10px;
                padding: 8px;
            """)
            alignment = Qt.AlignLeft  # 整体靠左
            content_css = """
                body {
                    background: transparent;
                    color: black;
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                    word-wrap: break-word;
                    max-width: 100%;
                    margin-right: auto;  /* 内容区块靠左 */
                    text-align: left;
                }
            """
        else:  # system
            self.setStyleSheet("""
                background-color: #e9ecef;
                color: #495057;
                border-radius: 10px;
                padding: 6px;
                font-size: 12px;
            """)
            alignment = Qt.AlignCenter
            content_css = """
                body {
                    background: transparent;
                    color: #495057;
                    margin: 0 auto;
                    padding: 0;
                    overflow: hidden;
                    word-wrap: break-word;
                    max-width: 100%;
                    text-align: center;
                }
            """

        # 布局：让 QMView 填充整个气泡
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(alignment)

        self.view = QMView()
        self.view.setMarkdown(markdown_text)

        # 注入内容样式 + 隐藏滚动条
        full_css = content_css + """
            ::-webkit-scrollbar {
                display: none;
            }
        """
        self.view.setHtmlStyle(full_css)

        # 启用自动尺寸（高度自适应，宽度由父布局控制）
        self.view.setAutoSize(True,max_width=900)

        layout.addWidget(self.view)
        # 确保在初始化后立即设置一次 Markdown
        self.view.setMarkdown(markdown_text)

    def update_content(self, new_text: str):
        """追加内容（流式更新）"""
        self.pending_text += new_text
        if not self.update_timer.isActive():
            self.update_timer.start()

    def _flush_pending(self):
        if self.pending_text:
            # 预处理：确保以 '##' 开头的行后面有换行符（简单示例）
            # 更完善的方案可以检测 Markdown 块边界
            text_to_add = self.pending_text
            # 如果当前 full_text 以标题开头且未换行，而新文本以换行开头，则可能解决问题
            # 此处不做复杂处理，仅合并后重新设置
            self.full_text += text_to_add
            self.view.setMarkdown(self.full_text)
            self.pending_text = ""

    def finalize(self):
        if self.is_final:
            return
        self.is_final = True

        # 强制重新解析 Markdown（先清空再设置）
        self.view.setMarkdown("")
        self.view.setMarkdown(self.full_text)

        # 触发重绘
        self.view.update()

        # 如果 QMView 基于 WebEngine，尝试触发 MathJax
        if hasattr(self.view, 'page'):
            self.view.page().runJavaScript("""
                if (window.MathJax) {
                    MathJax.typesetPromise();
                }
            """)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 读取字体大小设置
    settings = QSettings("CyberPardofelis", "LLM科研助手")
    font_size_str = settings.value("font_size", "10px")  # 默认10px
    try:
        # 提取数字部分，例如 "14px" -> 14
        font_size = int(font_size_str.replace('px', ''))
    except:
        font_size = 10

    # 设置全局字体
    font = app.font()
    font.setPointSize(font_size)
    app.setFont(font)

    settings = QSettings("CyberPardofelis", "LLM科研助手")
    if settings.contains("OPENAI_API_KEY"):
        config.OPENAI_API_KEY = settings.value("OPENAI_API_KEY")
    if settings.contains("OPENAI_MODEL"):
        config.OPENAI_MODEL = settings.value("OPENAI_MODEL")
    if settings.contains("DEEPSEEK_API_KEY"):
        config.DEEPSEEK_API_KEY = settings.value("DEEPSEEK_API_KEY")
    if settings.contains("DEEPSEEK_MODEL"):
        config.DEEPSEEK_MODEL = settings.value("DEEPSEEK_MODEL")
    if settings.contains("DEEPSEEK_MODEL_CHAT"):
        config.DEEPSEEK_MODEL_CHAT = settings.value("DEEPSEEK_MODEL_CHAT")
    if settings.contains("EMBEDDING_ENABLED"):
        val = settings.value("EMBEDDING_ENABLED")
        config.EMBEDDING_ENABLED = val in [True, "true", "1"]
    if settings.contains("EMBEDDER_TYPE"):
        config.EMBEDDER_TYPE = settings.value("EMBEDDER_TYPE")
    if settings.contains("OPENAI_EMBEDDING_MODEL"):
        config.OPENAI_EMBEDDING_MODEL = settings.value("OPENAI_EMBEDDING_MODEL")
    if settings.contains("LOCAL_EMBEDDER_NAME"):
        config.LOCAL_EMBEDDER_NAME = settings.value("LOCAL_EMBEDDER_NAME")
    if settings.contains("MODEL_PATH"):
        config.MODEL_PATH = settings.value("MODEL_PATH")
    if settings.contains("DATA_DIR"):
        config.DATA_DIR = settings.value("DATA_DIR")
    if settings.contains("VECTOR_DIR"):
        config.VECTOR_DIR = settings.value("VECTOR_DIR")

    window = ChatWindow()
    window.show()
    sys.exit(app.exec())