# chat_view.py
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea

from ui.message_bubble_1 import MessageBubble


# B：视野检测间隔（毫秒）。不要太频繁（检测本身有开销），
# 也不要太慢（用户滚动后等几秒才 freeze/unfreeze，体感差）。
_FREEZE_CHECK_INTERVAL_MS = 3000
# B：超过这个数量的气泡才启动 freeze 定时器（避免短对话时无谓开销）
_FREEZE_MIN_BUBBLES = 5


class ChatView(QWidget):
    """消息气泡视图 — 原样提取 ChatWindow 中气泡相关代码"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_freeze_timer()
        self._freeze_enabled = False  # lazy：超过阈值才开启

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

    def _setup_freeze_timer(self):
        """B：定时检查气泡视野，滚出 viewport 的 view 释放 GPU 资源。
        定时器默认不启动，懒激活：超过 _FREEZE_MIN_BUBBLES 个气泡才开始检测。"""
        self._freeze_timer = QTimer(self)
        self._freeze_timer.setInterval(_FREEZE_CHECK_INTERVAL_MS)
        self._freeze_timer.timeout.connect(self._update_bubble_freeze_states)
        # 滚动时延后 300ms 触发一次（滚动停止后）
        self._scroll_check_timer = QTimer(self)
        self._scroll_check_timer.setSingleShot(True)
        self._scroll_check_timer.setInterval(300)
        self._scroll_check_timer.timeout.connect(self._update_bubble_freeze_states)
        self.scroll_area.verticalScrollBar().valueChanged.connect(
            self._on_scroll_changed
        )

    def _on_scroll_changed(self):
        """滚动时延后触发 freeze 检查（避免滚动期间频繁检测）"""
        if self._freeze_enabled:
            self._scroll_check_timer.start()

    def _maybe_enable_freeze(self):
        """气泡数超过阈值时启动 freeze 定时器"""
        if self._freeze_enabled:
            return
        if self.scroll_layout.count() >= _FREEZE_MIN_BUBBLES:
            self._freeze_enabled = True
            self._freeze_timer.start()
            # 启动时立即检查一次
            QTimer.singleShot(100, self._update_bubble_freeze_states)

    def _update_bubble_freeze_states(self):
        """
        B：遍历所有气泡，标记滚出视野的 view 为 frozen。
        视野判定：气泡的全局 y 范围与 scroll_area viewport 的 y 范围有无交集。
        """
        if not self._freeze_enabled:
            return
        if not self.scroll_area.isVisible():
            return
        viewport_top_in_content = self.scroll_area.verticalScrollBar().value()
        viewport_bottom = viewport_top_in_content + self.scroll_area.viewport().height()

        for i in range(self.scroll_layout.count()):
            widget = self.scroll_layout.itemAt(i).widget()
            if not isinstance(widget, MessageBubble):
                continue
            top = widget.y()
            bottom = top + widget.height()
            in_view = not (bottom <= viewport_top_in_content or top >= viewport_bottom)

            if not hasattr(widget, 'view') or widget.view is None:
                continue
            view = widget.view

            if in_view:
                if getattr(view, '_frozen', False):
                    view.unfreeze()
                    view._frozen = False
            else:
                if not getattr(view, '_frozen', False):
                    view.freeze()
                    view._frozen = True

    def clear_chat(self):
        """
        清空聊天区。重要：先显式释放 view 资源（避免 Chromium 端排队释放阻塞 UI），
        再 deleteLater 销毁 widget。
        """
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            widget = child.widget() if child else None
            if not widget:
                continue
            # 显式释放 view 资源（同步触发 Chromium 释放）
            if isinstance(widget, MessageBubble) and hasattr(widget, 'view') and widget.view is not None:
                view = widget.view
                # 取消 pending freeze 状态（即将被销毁）
                view._frozen = False
                try:
                    # 强制 Chromium 端同步释放页面（避免 deleteLater 排队）
                    view.page().setUrl(QUrl("about:blank"))
                except Exception:
                    pass
            widget.deleteLater()
        # 清空后重置 lazy 状态
        self._freeze_enabled = False
        if self._freeze_timer.isActive():
            self._freeze_timer.stop()

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
        # B：lazy — 气泡数超过阈值才启动 freeze 定时器（避免短对话时无谓开销）
        # 注意：这里**不**主动触发 freeze check。新气泡默认 visible，
        # 滚出视野的气泡由定时器（或滚动事件）自然处理。
        self._maybe_enable_freeze()
        # 注:v5 尝试的 ensure_loaded 懒加载已撤回(见 message_bubble_1.py 注释)
        # 撤回原因:view load 完前显示 → 黑色色块回归

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