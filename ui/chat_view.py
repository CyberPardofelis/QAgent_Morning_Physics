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
        # 2026-09-02 v6 改:不调用 setStyleSheet() — 那会创建 widget 级 QSS,
        # 覆盖 app 的 QSS,导致气泡的 #bubble_user/assistant/system QSS 规则不生效
        # (用户报"对话气泡背景去掉了"的真根因 — debug 看到 bubble 实际渲染是纯黑)
        # 改靠 setAutoFillBackground(False) + QSS QScrollArea, QScrollArea > QWidget 规则
        self.scroll_area.setAutoFillBackground(False)
        self.scroll_area.viewport().setAutoFillBackground(False)
        # v18: scroll_content 显式 transparent — 也只用 setAutoFillBackground(False)
        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("chatScrollContent")
        self.scroll_content.setAutoFillBackground(False)
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

    def clear_chat(self):
        # v18 修复: takeAt + deleteLater 是异步删除,期间 widget 仍在 widget tree
        # (parent 还是 scroll_content)→ 仍然显示在旧位置,新加的 widget 跟它叠加。
        # 修法: 先 hide() + setParent(None) 立即从 widget tree 移除 + 显示,
        #       再 deleteLater() 释放内存。
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                w = child.widget()
                w.hide()  # ★ 立即隐藏,不让它继续显示
                w.setParent(None)  # ★ 立即从 widget tree 移除
                w.deleteLater()  # 然后异步释放内存

    def scroll_to_bottom(self):
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def scroll_if_needed(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        if scrollbar.maximum() - scrollbar.value() <= 50:
            self.scroll_to_bottom()

    def add_chat_message(self, role, text, isAppend=False, attachments=None):
        # v15:任何 role 都先 remove system 气泡(防堆叠)
        # 原因:create_new_conversation 会 emit new_message("system", "新对话已创建。"),
        # 这条 system 消息也走 add_chat_message。如果不先 remove,旧的 system 气泡
        # (欢迎提示)不会被清,新的 system 消息又叠加上去 → 切换对话后"不消失"
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
