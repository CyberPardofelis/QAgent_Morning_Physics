"""
ChatView — 抄 SAgent 整段 (2026-09-01)。

保留 QAgent 自己的:
- chat_window / conversation_controller 调用的所有接口:add_chat_message /
  clear_chat / find_last_assistant_bubble / scroll_to_bottom / scroll_if_needed /
  refresh_bubble_widths / scroll_layout
- view.page().setUrl(about:blank) 兼容 (虽然 QTextBrowser 不需要,保留防御性)
"""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea

from ui.message_bubble_1 import MessageBubble


class ChatView(QWidget):
    """消息气泡视图"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.NoFrame)
        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("chatScrollContent")
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

    def clear_chat(self):
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def scroll_to_bottom(self):
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def scroll_if_needed(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        if scrollbar.maximum() - scrollbar.value() <= 50:
            self.scroll_to_bottom()

    def add_chat_message(self, role, text, isAppend=False, attachments=None):
        # v14:加非 system 角色气泡时(用户发消息 / 切新对话 → 新 user/assistant 气泡),
        # 自动删除启动时的 system 欢迎气泡(欢迎提示已阅)
        # 关键:必须放在 isAppend 分支**之前**!否则流式 chunk 的 isAppend=True 早退
        # return 走不到这里,welcome 气泡不消失
        if role != "system":
            self._remove_welcome_bubbles()

        if isAppend and role == "assistant":
            last_bubble = self.find_last_assistant_bubble()
            if last_bubble:
                last_bubble.update_content(text)
                self.scroll_if_needed()
                return

        bubble = MessageBubble(role, text, attachments=attachments)
        alignment = {'user': Qt.AlignRight, 'assistant': Qt.AlignLeft, 'system': Qt.AlignCenter}
        self.scroll_layout.addWidget(bubble, 0, alignment.get(role, Qt.AlignLeft))
        bubble.updateMaxWidth()
        self.scroll_if_needed()

    def _remove_welcome_bubbles(self):
        """v13:删除 system 角色欢迎气泡(用户切对话/发消息后消失)"""
        i = 0
        while i < self.scroll_layout.count():
            item = self.scroll_layout.itemAt(i)
            widget = item.widget() if item else None
            if isinstance(widget, MessageBubble) and widget.role == "system":
                self.scroll_layout.takeAt(i)
                widget.deleteLater()
                # 不增加 i(takeAt 后下一个 item 移到当前位置)
            else:
                i += 1

    def find_last_assistant_bubble(self):
        for i in range(self.scroll_layout.count() - 1, -1, -1):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, MessageBubble) and widget.role == "assistant":
                return widget
        return None

    def refresh_bubble_widths(self):
        for i in range(self.scroll_layout.count()):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, MessageBubble):
                widget.updateMaxWidth()
        self.scroll_content.updateGeometry()
