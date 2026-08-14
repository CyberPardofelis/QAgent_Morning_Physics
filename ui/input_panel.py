# input_panel.py
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPlainTextEdit,
    QPushButton, QSizePolicy
)


class InputPanel(QWidget):
    """输入面板：输入框 + 发送/中止/重新生成按钮，真正修复高度自适应"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

        # ★ 预创建定时器
        self._height_timer = QTimer(self)
        self._height_timer.setSingleShot(True)
        self._height_timer.timeout.connect(self._do_adjust_height)

        # ★ 保存原始边界常量，用于计算钳位
        self._INPUT_MIN_HEIGHT = self.line_height() * 2 + 25
        self._INPUT_MAX_HEIGHT = self.line_height() * 8 + 25

        # 外部回调（由 ChatWindow 赋值）
        self.send_clicked = None
        self.stop_clicked = None
        self.regenerate_clicked = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        input_layout = QHBoxLayout()

        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("输入消息... (Enter发送，Shift+Enter换行)")
        self.input_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.input_edit.setFixedHeight(self.line_height() * 2 + 25)
        self.input_edit.installEventFilter(self)

        self.send_btn = QPushButton("发送")
        self.send_btn.setObjectName("sendBtn")

        self.stop_btn = QPushButton("中止")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)

        self.regenerate_btn = QPushButton("重新生成")
        self.regenerate_btn.setObjectName("regenerateBtn")
        self.regenerate_btn.setEnabled(False)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.regenerate_btn)

        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.send_btn)
        input_layout.addLayout(button_layout)
        layout.addLayout(input_layout)

    def _connect_signals(self):
        self.input_edit.textChanged.connect(self.adjust_input_height)
        self.send_btn.clicked.connect(self.on_send_clicked)
        self.stop_btn.clicked.connect(lambda: self.stop_clicked and self.stop_clicked())
        self.regenerate_btn.clicked.connect(
            lambda: self.regenerate_clicked and self.regenerate_clicked()
        )

    # ==================== 辅助 ====================

    def line_height(self):
        return self.input_edit.fontMetrics().lineSpacing()

    def _calc_visual_line_count(self):
        """★ 计算真实视觉行数（含自动换行）"""
        doc = self.input_edit.document()
        block = doc.begin()
        total = 0
        while block.isValid():
            layout = block.layout()
            if layout:
                total += layout.lineCount()
            block = block.next()
        return max(total, 1)

    # ==================== 高度自适应 ====================

    def adjust_input_height(self):
        """文本变化 → 延迟调整高度（防抖）"""
        self._height_timer.start(10)

    def _do_adjust_height(self):
        """★ 真正的高度调整：用视觉行数 + setFixedHeight"""
        line_count = self._calc_visual_line_count()
        line_spacing = self.line_height()
        desired = line_count * line_spacing + 30

        # 用保存的原始边界钳位
        desired = max(self._INPUT_MIN_HEIGHT, min(desired, self._INPUT_MAX_HEIGHT))

        # ★ 用 setFixedHeight 强制设置，父布局必须遵守
        self.input_edit.setFixedHeight(int(desired))

    # ==================== 事件过滤 ====================

    def eventFilter(self, obj, event):
        if obj == self.input_edit and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier):
                    self.input_edit.insertPlainText("\n")
                    return True
                else:
                    self.on_send_clicked()
                    return True
        return super().eventFilter(obj, event)

    # ==================== 行为 ====================

    def on_send_clicked(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        self.input_edit.clear()
        # ★ 清空后强制重置到最小高度
        self.input_edit.setFixedHeight(self._INPUT_MIN_HEIGHT)
        if self.send_clicked:
            self.send_clicked(text)

    def set_input_enabled(self, enabled):
        self.send_btn.setEnabled(enabled)
        self.input_edit.setEnabled(enabled)
