import sys
import traceback
import os
import datetime

try:
    # 你的原始导入和代码
    import json
    from datetime import datetime
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from agent.history_manager import HistoryManager, Conversation
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPlainTextEdit, QPushButton, QComboBox, QMessageBox, QLabel, QSizePolicy,
        QSplitter, QListWidget, QListWidgetItem, QMenu, QInputDialog, QFrame
    )
    from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QPoint
    from PySide6.QtGui import QDesktopServices, QAction
    from agent.controller import run_agent_stream
    from agent.prompts import SYSTEM_PROMPT

except Exception:
    # 确定日志路径
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, 'error.log')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n--- {datetime.datetime.now()} ---\n")
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
        f.write(f"\n--- {datetime.datetime.now()} ---\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    # 也可以打印到控制台（如果有）
    traceback.print_exception(exc_type, exc_value, exc_traceback)


sys.excepthook = excepthook


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
                    self.stream_chunk.emit("ai", chunk)
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

        # ---------- 左侧对话列表 ----------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # 新建对话按钮
        self.new_conv_btn = QPushButton("＋ 新建对话")
        self.new_conv_btn.clicked.connect(self.create_new_conversation)
        left_layout.addWidget(self.new_conv_btn)

        # 对话列表
        self.conv_list = QListWidget()
        self.conv_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.conv_list.customContextMenuRequested.connect(self.show_conv_context_menu)
        self.conv_list.itemClicked.connect(self.switch_conversation)
        left_layout.addWidget(self.conv_list)

        splitter.addWidget(left_widget)
        splitter.setStretchFactor(0, 1)  # 左侧较小

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
        self.chat_view = QWebEngineView()
        self.setup_chat_html()
        self.page_loaded = False
        self.chat_view.loadFinished.connect(self.on_page_loaded)
        self.chat_view.page().navigationRequested.connect(self.on_navigation_requested)
        right_layout.addWidget(self.chat_view)

        # 底部输入区
        bottom_layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("输入消息... (Enter发送，Shift+Enter换行)")
        self.input_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.input_edit.setMinimumHeight(self.line_height() * 2 + 10)
        self.input_edit.setMaximumHeight(self.line_height() * 6 + 10)
        self.input_edit.textChanged.connect(self.adjust_input_height)
        self.send_btn = QPushButton("发送")
        self.send_btn.setFixedWidth(80)
        self.send_btn.clicked.connect(self.send_message)

        # 新增：中止和重新生成按钮，垂直排列
        self.stop_btn = QPushButton("中止")
        self.stop_btn.setFixedWidth(60)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generation)

        self.regenerate_btn = QPushButton("重新生成")
        self.regenerate_btn.setFixedWidth(60)
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
        splitter.setStretchFactor(1, 5)  # 右侧占更多空间

        # 其他原有属性
        self.input_edit.installEventFilter(self)
        self.processing = False
        self.current_message_id = None  # 用于流式追加消息
        self._last_ai_message_id = None

        # 加载历史对话列表
        self.refresh_conv_list()
        # 如果存在历史对话，加载最近的一个；否则新建一个
        if self.conv_list.count() > 0:
            self.conv_list.setCurrentRow(0)
            self.switch_conversation(self.conv_list.item(0))
        else:
            self.create_new_conversation()

        self.update_regenerate_button_state()

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
            self.chat_view.page().runJavaScript(js)
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
        conversations = self.history_manager.get_all_sorted()
        for conv in conversations:
            # 显示标题和最后更新时间（简短）
            updated = datetime.fromisoformat(conv.updated_at).strftime("%m-%d %H:%M")
            display_text = f"{conv.title}\n{updated}"
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, conv.conv_id)
            self.conv_list.addItem(item)

    def create_new_conversation(self):
        """创建新对话并切换到它"""
        # 如果有正在进行的对话，先保存
        if self.current_conversation:
            self.history_manager.save_conversation(self.current_conversation)
        # 创建新对话
        conv = self.history_manager.create_conversation(title="新对话")
        self.current_conversation = conv
        self.refresh_conv_list()
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
            for msg in conv.messages:
                self.add_chat_message(msg["role"], msg["content"])
            # 滚动到底部
            self.scroll_to_bottom()
        self.update_regenerate_button_state()

    def clear_chat(self):
        """清空 WebView 中的聊天内容"""
        js = "document.getElementById('chat').innerHTML = '';"
        self.chat_view.page().runJavaScript(js)

    def scroll_to_bottom(self):
        js = "window.scrollTo(0, document.body.scrollHeight);"
        self.chat_view.page().runJavaScript(js)

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
            self.refresh_conv_list()
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

    def on_navigation_requested(self, request):
        """拦截导航，http/https链接在外部浏览器打开，其他放行"""
        url = request.url()
        if url.scheme().startswith('http'):
            QDesktopServices.openUrl(url)
            request.reject()  # 阻止在WebView内加载
        else:
            request.accept()

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
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
        <style>
            body { font-family: '微软雅黑', Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }
            #chat { display: flex; flex-direction: column; }
            .message-container {
                display: flex;
                margin-bottom: 10px;
            }
            .message-container.user { justify-content: flex-end; }
            .message-container.ai { justify-content: flex-start; }
            .message-container.system { justify-content: center; }
            .bubble {
                max-width: 70%;
                padding: 10px 14px;
                border-radius: 18px;
                word-wrap: break-word;
                white-space: normal;
                font-size: 14px;
                line-height: 1.4;
            }
            .user .bubble {
                background-color: #007AFF;
                color: white;
                border-bottom-right-radius: 4px;
            }
            .ai .bubble {
                background-color: #E5E5EA;
                color: black;
                border-bottom-left-radius: 4px;
            }
            .system .bubble {
                background-color: #d0d0d0;
                color: #333;
                font-size: 12px;
                border-radius: 8px;
                padding: 4px 10px;
            }
            pre {
                background: #f0f0f0;
                padding: 8px;
                border-radius: 5px;
                overflow-x: auto;
            }
            code {
                font-family: Consolas, Monaco, 'Courier New', monospace;
                font-size: 13px;
            }
            .MathJax_SVG { display: inline-block; }
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

    def on_page_loaded(self, ok):
        if ok:
            self.page_loaded = True
            # 如果已经有当前对话，加载它的消息
            if self.current_conversation:
                self.clear_chat()
                for msg in self.current_conversation.messages:
                    self.add_chat_message(msg["role"], msg["content"])
            else:
                self.add_chat_message("system", "LLM科研助手已启动。请选择模式并开始对话。")
        else:
            print("页面加载失败")

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
        if not hasattr(self, '_last_ai_message_id') or self._last_ai_message_id is None:
            self.add_chat_message(sender, chunk, isAppend=False)
            self._last_ai_message_id = sender
            # 初始化最后一条消息内容
            if self.current_conversation:
                self.current_conversation.add_message("assistant", chunk)
        else:
            self.append_to_last_ai_message(chunk)
            # 更新内存中最后一条消息
            if self.current_conversation and self.current_conversation.messages:
                last_msg = self.current_conversation.messages[-1]
                if last_msg["role"] == "assistant":
                    last_msg["content"] += chunk
                    self.current_conversation.updated_at = datetime.now().isoformat()

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
        self.chat_view.page().runJavaScript(js)

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
            # 更新对话（原有代码不变）
            if self.current_conversation:
                new_messages = [msg for msg in final_messages if msg["role"] != "system"]
                self.current_conversation.messages = new_messages
                # ... 其余原有代码保持不变 ...
                self.current_conversation.updated_at = datetime.now().isoformat()
                if self.current_conversation.title == "新对话":
                    new_title = self.history_manager.generate_title_from_messages(self.current_conversation)
                    if new_title:
                        self.current_conversation.title = new_title
                self.history_manager.save_conversation(self.current_conversation)
                self.refresh_conv_list()

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

    def add_chat_message(self, sender, text, isAppend=False):
        """通过JavaScript添加消息（非流式情况使用）"""
        if not self.page_loaded:
            QTimer.singleShot(100, lambda: self.add_chat_message(sender, text, isAppend))
            return
        safe_sender = json.dumps(sender)
        safe_text = json.dumps(text)
        js = f"addMessage({safe_sender}, {safe_text}, {json.dumps(isAppend)});"
        self.chat_view.page().runJavaScript(js)
        self.update_regenerate_button_state()

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())