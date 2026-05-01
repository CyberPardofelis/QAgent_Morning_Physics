import json
import traceback
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, QUrl, QSettings, QPoint
from PySide6.QtGui import QDesktopServices, QAction, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QMenu,
    QPushButton, QTreeWidget, QTreeWidgetItem, QLabel, QComboBox, QInputDialog,
    QPlainTextEdit, QMessageBox, QScrollArea, QSizePolicy, QListWidgetItem
)
from agent.history_manager import HistoryManager, Conversation
from ui.styles import GLOBAL_STYLESHEET
from ui.worker import Worker
from ui.message_bubble import MessageBubble
from ui.settings_dialog import SettingsDialog
from ui.knowledge_base_dialog import KnowledgeBaseDialog
from ui.group_prompt_dialog import GroupSystemPromptDialog
from agent.tools import update_paper_comment, DB_PATH, query_knowledge, get_paper_by_id, get_db_connection, get_recent_papers, update_full_paper
from agent.prompts import SYSTEM_PROMPT
import config


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

        self.current_conversation = None
        self.clear_chat()
        self.add_chat_message("system",
                              "LLM科研助手已启动。请选择模式并开始对话。\n\n"
                              "发送 早上好 arxiv/pra/prl等 以获取对应平台新文章摘要.\n\n"
                              "发送 分析+文章DOI 以下载并分析对应文章，分析结果会保存至知识库。\n\n"
                              "在分析文章之后可以发送 保存评论：评论内容 以将文章评论保存到知识库\n\n"
                              "菜单栏中打开知识库，可以进行检索并打开相关文章的详情页面查看\n\n"
                              "在详情页面可以查看和修改总结信息、引用信息、评论和LLM详细总结等")

        self.update_regenerate_button_state()

        # 默认系统提示词
        settings = QSettings("CyberPardofelis", "LLM科研助手")
        self.default_system_prompt = settings.value("default_system_prompt", SYSTEM_PROMPT)

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
        system_prompt = self._get_current_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        for msg in self.current_conversation.messages:
            messages.append(msg.copy())
        messages = self.clean_messages_for_api(messages)

        # 启动生成
        self._run_with_messages(messages)

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
        system_prompt = self._get_current_system_prompt()
        # 将用户消息添加到当前对话对象
        if self.current_conversation:
            self.current_conversation.add_message("user", text)

            # 构建完整的消息列表

            messages = [{"role": "system", "content": system_prompt}]
            for msg in self.current_conversation.messages:
                messages.append(msg.copy())  # ✅ 复制完整消息，保留所有字段
            messages = self.clean_messages_for_api(messages)
            # print(messages)
        else:
            # 没有当前对话（不应发生）
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]

        # 启动生成
        self._run_with_messages(messages)

    def on_stream_chunk(self, sender, chunk):
        # print(f"on_stream_chunk: sender={sender}, chunk='{chunk[:50]}'")  # 调试
        if sender != "assistant":
            return  # 忽略非助理消息（如 tool）

        if chunk == "":
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
        if text == '':
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
        dialog.default_prompt_changed.connect(self.on_default_prompt_changed)
        dialog.show()

    def on_default_prompt_changed(self, new_prompt):
        self.default_system_prompt = new_prompt

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
        if self.current_conversation and self.current_conversation.conv_id in self.history_manager.conversations:
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
            set_prompt_action = QAction("设置系统提示词", self)
            set_prompt_action.triggered.connect(lambda: self.open_group_prompt_dialog(group_id))

            menu.addAction(rename_action)
            menu.addAction(delete_action)
            menu.addAction(new_conv_action)
            menu.addAction(set_prompt_action)
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
            settings = QSettings("CyberPardofelis", "LLM科研助手")
            settings.remove(f"group_system_prompt/{group_id}")
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
        return distance_to_bottom <= 50

    def scroll_if_needed(self):
        # print("scroll_if_needed called")  # 调试用，可删除
        if self.should_auto_scroll():
            # print("should auto scroll, scrolling...")
            self.scroll_to_bottom()

    def _get_current_system_prompt(self):
        """获取当前对话应使用的系统提示词"""
        if self.current_conversation and self.current_conversation.group_id:
            group_id = self.current_conversation.group_id
            settings = QSettings("CyberPardofelis", "LLM科研助手")
            group_prompt = settings.value(f"group_system_prompt/{group_id}")
            if group_prompt:
                return group_prompt
        return self.default_system_prompt

    def open_group_prompt_dialog(self, group_id):
        # 获取分组名称
        groups = self.history_manager.get_all_groups()
        group_name = next((g['name'] for g in groups if g['group_id'] == group_id), "未知分组")
        dialog = GroupSystemPromptDialog(group_id, group_name, self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.show()  # 非模态