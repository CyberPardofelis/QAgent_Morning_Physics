# chat_view.py
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea

from ui.message_bubble_1 import MessageBubble


class ChatView(QWidget):
    """消息气泡视图 — 原样提取 ChatWindow 中气泡相关代码"""

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
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_content.setStyleSheet("background-color: #fafaff;")
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

    def add_chat_message(self, role, text, isAppend=False):
        # 空气泡（assistant 流式槽位）需要保留 → 单独放行
        if not text and role != "assistant":
            return
        if isAppend and role == "assistant" and text:
            last_bubble = self.find_last_assistant_bubble()
            if last_bubble:
                last_bubble.update_content(text)
                QTimer.singleShot(0, self.scroll_if_needed)
                return
        bubble = MessageBubble(role, text or "")
        alignment = {'user': Qt.AlignRight, 'assistant': Qt.AlignLeft, 'system': Qt.AlignCenter}
        self.scroll_layout.addWidget(bubble, 0, alignment.get(role, Qt.AlignLeft))
        bubble.updateMaxWidth()
        QTimer.singleShot(0, bubble.adjustSize)
        self.scroll_if_needed()

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