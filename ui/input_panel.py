# input_panel.py — 移植 WAgent 附件系统 + 保留 QAgent 高度自适应
#
# 结构（外层垂直布局）：
#   1. 附件 chip 横滚条（无附件时高度=0 完全隐藏）
#   2. + 按钮 | 输入框 | 发送按钮 / (中止 + 重新生成)  水平行
#
# 行为：
#   - Enter 发送 / Shift+Enter 换行（保留 QAgent 行为）
#   - 输入框高度随行数自动撑高（2~8 行，保留 QAgent 行为）
#   - 附件文本懒加载：on_send_clicked 时才 extract_text（读 PDF/Word/文本）
#   - 附件元数据（name/size/path）存历史；content 仅临时给 LLM
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer, QEvent, QSettings
from config.settings import get_settings
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPlainTextEdit,
    QPushButton, QSizePolicy, QToolButton, QScrollArea,
    QFileDialog, QLabel,
)

from agent.tools.attachment_reader import (
    SUPPORTED_EXTS, FILE_DIALOG_FILTER, is_supported, extract_text,
)


# ============================================================
# 附件数据结构 + 单个 chip 控件
# ============================================================
@dataclass
class AttachmentItem:
    """单个附件：路径 + 缓存的文本内容（懒加载）"""
    path: str
    text: Optional[str] = None          # 第一次发送时再读
    error: Optional[str] = None
    size_chars: int = 0

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def display_size(self) -> str:
        try:
            b = os.path.getsize(self.path)
        except OSError:
            return ""
        if b < 1024:
            return f"{b} B"
        if b < 1024 * 1024:
            return f"{b / 1024:.1f} KB"
        return f"{b / 1024 / 1024:.1f} MB"


class _AttachmentChip(QWidget):
    """单个附件 chip：图标 + 文件名 + 关闭 ×"""
    closed = None  # 回调：fn(self) -> None

    def __init__(self, item: AttachmentItem, parent=None):
        super().__init__(parent)
        self.item = item

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 4, 4)
        layout.setSpacing(6)

        self.icon_label = QLabel("📄")
        self.name_label = QLabel(self._truncate_name(item.name, 18))
        self.name_label.setObjectName("chipName")
        self.name_label.setToolTip(item.path)

        self.close_btn = QToolButton()
        self.close_btn.setText("×")
        self.close_btn.setObjectName("chipCloseBtn")
        self.close_btn.setCursor(Qt.PointingHandCursor)
        self.close_btn.setFixedSize(18, 18)
        self.close_btn.clicked.connect(lambda: self.closed and self.closed(self))

        layout.addWidget(self.icon_label)
        layout.addWidget(self.name_label)
        layout.addWidget(self.close_btn)

        self.setStyleSheet(
            "QWidget {"
            "  background-color: #f1f5f9;"
            "  border: 1px solid #e2e8f0;"
            "  border-radius: 14px;"
            "}"
        )
        if item.error:
            self.name_label.setProperty("state", "error")
            self.setToolTip(item.error)

    @staticmethod
    def _truncate_name(name: str, max_len: int) -> str:
        if len(name) <= max_len:
            return name
        stem, ext = os.path.splitext(name)
        keep = max_len - len(ext) - 1
        if keep <= 0:
            return name[:max_len]
        return stem[:keep] + "…" + ext


# ============================================================
# 输入面板主体
# ============================================================
class InputPanel(QWidget):
    """+ 按钮 / 输入框 / 发送/中止/重新生成；附件 chip 浮在输入框正上方"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ★ 预创建定时器 + 边界常量（必须在 _setup_ui 之前算出）
        self._height_timer = QTimer(self)
        self._height_timer.setSingleShot(True)
        self._height_timer.timeout.connect(self._do_adjust_height)

        from PySide6.QtWidgets import QPlainTextEdit as _QPT
        _tmp = _QPT()
        line_h = _tmp.fontMetrics().lineSpacing()
        _tmp.deleteLater()
        self._INPUT_MIN_HEIGHT = line_h * 2 + 25
        self._INPUT_MAX_HEIGHT = line_h * 8 + 25

        self._setup_ui()
        self._connect_signals()

        # 外部回调（由 ChatWindow 赋值）
        self.send_clicked = None          # fn(text: str, attachments: list) -> None
        self.stop_clicked = None           # fn() -> None
        self.regenerate_clicked = None    # fn() -> None
        self.attachment_error = None       # fn(msg: str) -> None

        # 附件列表
        self._attachments: List[AttachmentItem] = []

    # ---------------- UI ----------------
    def _setup_ui(self):
        # 外层垂直布局：上面 chip 区，中间 +按钮 / 输入框 / 按钮 的水平行
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # === 1. 附件条（chip 横滚区，初始 0 高度完全隐藏） ===
        self.attach_scroll = QScrollArea()
        self.attach_scroll.setObjectName("attachScroll")
        self.attach_scroll.setWidgetResizable(True)
        self.attach_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.attach_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.attach_scroll.setFixedHeight(0)
        self.attach_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._chip_container = QWidget()
        self._chip_container.setObjectName("attachChipContainer")
        self._chip_layout = QHBoxLayout(self._chip_container)
        self._chip_layout.setContentsMargins(2, 2, 2, 2)
        self._chip_layout.setSpacing(6)
        self._chip_layout.addStretch(1)
        self.attach_scroll.setWidget(self._chip_container)
        self._chip_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        outer.addWidget(self.attach_scroll)

        # === 2. 中间一行：+ 按钮 / 输入框 / 右侧按钮 ===
        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)

        # + 按钮
        self.attach_btn = QToolButton()
        self.attach_btn.setText("+")
        self.attach_btn.setObjectName("attachBtn")
        self.attach_btn.setCursor(Qt.PointingHandCursor)
        self.attach_btn.setToolTip("添加附件（支持 .docx / .pdf / .txt / .md）")
        self.attach_btn.setFixedSize(36, 36)
        self.attach_btn.clicked.connect(self._on_attach_clicked)

        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("输入消息... (Enter发送，Shift+Enter换行)")
        self.input_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.input_edit.setFixedHeight(self._INPUT_MIN_HEIGHT)
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

        input_layout.addWidget(self.attach_btn)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.send_btn)
        input_layout.addLayout(button_layout)
        outer.addLayout(input_layout)

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

    # ==================== 高度自适应（保留 QAgent 行为） ====================
    def adjust_input_height(self):
        self._height_timer.start(10)

    def _do_adjust_height(self):
        line_count = self._calc_visual_line_count()
        line_spacing = self.line_height()
        desired = line_count * line_spacing + 30
        desired = max(self._INPUT_MIN_HEIGHT, min(desired, self._INPUT_MAX_HEIGHT))
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

    # ==================== 附件相关 ====================
    def _on_attach_clicked(self):
        last_dir = self._last_attach_dir()
        files, _ = QFileDialog.getOpenFileNames(
            self, "添加附件", last_dir, FILE_DIALOG_FILTER
        )
        if not files:
            return
        try:
            self._save_last_attach_dir(os.path.dirname(files[0]))
        except Exception:
            pass

        added = 0
        for f in files:
            if not is_supported(f):
                self._notify_error(f"已跳过不支持的文件: {os.path.basename(f)}")
                continue
            self._attachments.append(AttachmentItem(path=f))
            self._add_chip(self._attachments[-1])
            added += 1

        if added:
            self._refresh_attach_bar_height()
            self.input_edit.setFocus()

    def _last_attach_dir(self) -> str:
        try:
            s = get_settings()
            d = s.value("last_attach_dir", "", type=str)
            return d or ""
        except Exception:
            return ""

    def _save_last_attach_dir(self, d: str):
        try:
            s = get_settings()
            s.setValue("last_attach_dir", d)
        except Exception:
            pass

    def _add_chip(self, item: AttachmentItem):
        chip = _AttachmentChip(item)
        chip.closed = self._on_chip_closed
        idx = self._chip_layout.count() - 1
        self._chip_layout.insertWidget(idx, chip)
        self._resize_chip_container()

    def _resize_chip_container(self):
        total = 0
        for i in range(self._chip_layout.count()):
            w = self._chip_layout.itemAt(i).widget()
            if w:
                total += w.sizeHint().width()
        total += self._chip_layout.spacing() * max(0, self._chip_layout.count() - 1)
        total += self._chip_layout.contentsMargins().left() + self._chip_layout.contentsMargins().right()
        self._chip_container.setMinimumWidth(total)

    def _on_chip_closed(self, chip: _AttachmentChip):
        try:
            self._attachments.remove(chip.item)
        except ValueError:
            pass
        chip.setParent(None)
        chip.deleteLater()
        self._refresh_attach_bar_height()
        self._resize_chip_container()

    def _refresh_attach_bar_height(self):
        if self._attachments:
            self.attach_scroll.setFixedHeight(40)
        else:
            self.attach_scroll.setFixedHeight(0)

    def clear_attachments(self):
        """清空所有附件（发送后由 InputPanel 自动调用）"""
        for chip in self._chip_container.findChildren(_AttachmentChip):
            chip.setParent(None)
            chip.deleteLater()
        self._attachments.clear()
        self._refresh_attach_bar_height()

    def _notify_error(self, msg: str):
        if self.attachment_error:
            try:
                self.attachment_error(msg)
                return
            except Exception:
                pass
        print(f"[InputPanel 附件] {msg}")

    # ==================== 行为 ====================
    def on_send_clicked(self):
        text = self.input_edit.toPlainText().strip()
        if not text and not self._attachments:
            return

        # 懒加载：发送时一次性读出所有附件文本
        attachments_meta: List[dict] = []
        had_error = False
        for item in self._attachments:
            try:
                item.text = extract_text(item.path)
                item.error = None
                item.size_chars = len(item.text)
            except Exception as e:
                item.error = str(e)
                had_error = True
                self._notify_error(f"读取附件失败: {item.name} — {e}")
                continue
            attachments_meta.append({
                "name": item.name,
                "size": item.display_size,
                "path": item.path,
                "content": item.text,
            })

        if had_error and not attachments_meta and not text:
            return  # 所有附件都失败 + 没文本 → 别发空消息

        # 清空输入 + 附件
        self.input_edit.clear()
        self.input_edit.setFixedHeight(self._INPUT_MIN_HEIGHT)
        self.clear_attachments()

        if self.send_clicked:
            # text 传原文（不入库），attachments 传元数据 + content
            # controller 负责把 content 临时拼到 LLM 消息，元数据落库
            self.send_clicked(text, attachments_meta)

    def set_input_enabled(self, enabled):
        self.send_btn.setEnabled(enabled)
        self.input_edit.setEnabled(enabled)
        self.attach_btn.setEnabled(enabled)
