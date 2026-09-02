"""
ui/theme_editor.py
==================
主题编辑器 dialog — 阶段 2: 用户自定义主题

设计:
- 顶部:基础主题(从 light/dark/paper 复制 token 初始值) + 主题名输入
- 中部:8 个核心 token (bg_main/bg_card/accent/accent_soft/text_primary/text_secondary/danger/success)
       每个一行: 标签 + 颜色按钮(点开 QColorDialog) + hex 输入
- 底部:实时预览区(用 derive_tokens 算出的 QSS 应用到小预览 widget) + 保存/删除/取消

数据结构:
- 编辑器只编辑 8 个 core token
- 保存时调 derive_tokens(core) 算 30+ 完整 token, 存到 data/themes.json
- 主题名会自动加 "user:" 前缀(跟 builtin 区分)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QPushButton,
    QLabel, QComboBox, QColorDialog, QMessageBox, QGroupBox, QWidget,
    QScrollArea, QSizePolicy, QInputDialog, QMenu, QFrame,
)

from ui.styles import (
    CORE_TOKEN_KEYS, LIGHT, DARK, PAPER,
    derive_tokens, get_default_core_tokens,
    register_custom_theme, list_themes, get_theme,
    get_current_theme_name, set_theme,
    upsert_user_theme, delete_user_theme,
    get_stylesheet,
)

logger = logging.getLogger(__name__)


# token 友好名 (UI 显示)
CORE_TOKEN_LABELS = {
    "bg_main":         "主背景 (大片背景)",
    "bg_card":         "次级背景 (详情/编辑区)",
    "accent":          "主强调色 (按钮/链接)",
    "accent_soft":     "主强调色浅 (选中背景)",
    "text_primary":    "主文字色",
    "text_secondary":  "次文字色 (标签/提示)",
    "danger":          "危险红 (删除按钮)",
    "success":         "成功绿 (成功按钮)",
}


def _is_valid_hex(s: str) -> bool:
    """检查是否是 #rrggbb 或 #rgb 格式"""
    if not s:
        return False
    s = s.strip()
    if not s.startswith("#"):
        s = "#" + s
    if re.match(r"^#[0-9a-fA-F]{6}$", s):
        return True
    if re.match(r"^#[0-9a-fA-F]{3}$", s):
        return True
    return False


def _normalize_hex(s: str) -> str:
    s = s.strip()
    if not s.startswith("#"):
        s = "#" + s
    # #abc -> #aabbcc
    if re.match(r"^#[0-9a-fA-F]{3}$", s):
        return "#" + s[1] * 2 + s[2] * 2 + s[3] * 2
    return s.lower()


class _ColorButton(QPushButton):
    """自绘颜色按钮: 显示一个色块, 点击调 QColorDialog"""

    color_changed = Signal(str)  # 新颜色 hex

    def __init__(self, color: str = "#888888", parent=None):
        super().__init__(parent)
        self._color = _normalize_hex(color) if _is_valid_hex(color) else "#888888"
        self.setFixedSize(48, 28)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("点击选颜色")
        self.clicked.connect(self._on_click)

    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        if _is_valid_hex(color):
            self._color = _normalize_hex(color)
            self.update()

    def _on_click(self):
        c = QColorDialog.getColor(
            QColor(self._color), self, "选颜色",
            QColorDialog.ShowAlphaChannel,
        )
        if c.isValid():
            self._color = c.name()
            self.update()
            self.color_changed.emit(self._color)

    def paintEvent(self, event):
        super().paintEvent(event)
        # 画色块 (留 4px padding)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.setPen(QPen(QColor("#888"), 1))
        painter.setBrush(QColor(self._color))
        painter.drawRoundedRect(rect, 3, 3)
        painter.end()


class _ColorRow(QWidget):
    """一行: 标签 + 颜色按钮 + hex 输入 + (恢复默认按钮)"""

    color_changed = Signal(str, str)  # token_name, new_color

    def __init__(self, token_name: str, initial_color: str, default_color: str, parent=None):
        super().__init__(parent)
        self.token_name = token_name
        self._default_color = _normalize_hex(default_color) if _is_valid_hex(default_color) else "#888888"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 颜色按钮
        self.color_btn = _ColorButton(initial_color)
        self.color_btn.color_changed.connect(self._on_btn_changed)
        layout.addWidget(self.color_btn)

        # hex 输入
        self.hex_edit = QLineEdit(initial_color)
        self.hex_edit.setMaximumWidth(90)
        self.hex_edit.editingFinished.connect(self._on_hex_changed)
        layout.addWidget(self.hex_edit)

        # token 名 (灰色)
        lbl = QLabel(f"({token_name})")
        lbl.setProperty("role", "muted")
        layout.addWidget(lbl)

        layout.addStretch()

    def color(self) -> str:
        return self.color_btn.color()

    def set_color(self, color: str) -> None:
        if not _is_valid_hex(color):
            return
        norm = _normalize_hex(color)
        self.color_btn.set_color(norm)
        self.hex_edit.setText(norm)

    def _on_btn_changed(self, color: str):
        self.hex_edit.setText(color)
        self.color_changed.emit(self.token_name, color)

    def _on_hex_changed(self):
        text = self.hex_edit.text()
        if not _is_valid_hex(text):
            # 还原
            self.hex_edit.setText(self.color_btn.color())
            return
        norm = _normalize_hex(text)
        self.color_btn.set_color(norm)
        self.color_changed.emit(self.token_name, norm)


class _PreviewWidget(QWidget):
    """小预览: 用当前编辑中的主题 token 渲染一个迷你窗口, 用户实时看效果"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tokens: Dict[str, str] = {}
        self.setMinimumHeight(140)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 标题
        title = QLabel("预览")
        title.setProperty("role", "title")
        layout.addWidget(title)

        # 主按钮
        btn_row = QHBoxLayout()
        primary = QPushButton("主按钮 (accent)")
        primary.setObjectName("previewAccent")
        btn_row.addWidget(primary)
        danger = QPushButton("删除 (danger)")
        danger.setProperty("kind", "danger")
        btn_row.addWidget(danger)
        success = QPushButton("成功 (success)")
        success.setProperty("kind", "success")
        btn_row.addWidget(success)
        muted = QPushButton("次要 (muted)")
        muted.setProperty("kind", "muted")
        btn_row.addWidget(muted)
        layout.addLayout(btn_row)

        # 卡片
        self.card = QLabel("次级卡片 (bg_card)")
        self.card.setObjectName("previewCard")
        self.card.setAlignment(Qt.AlignCenter)
        self.card.setMinimumHeight(40)
        layout.addWidget(self.card)

        # 文字演示
        text_row = QHBoxLayout()
        primary_text = QLabel("主文字")
        primary_text.setObjectName("previewPrimary")
        text_row.addWidget(primary_text)
        secondary_text = QLabel("次文字 (hint)")
        secondary_text.setObjectName("previewSecondary")
        text_row.addWidget(secondary_text)
        muted_text = QLabel("弱文字 (muted)")
        muted_text.setObjectName("previewMuted")
        text_row.addWidget(muted_text)
        text_row.addStretch()
        layout.addLayout(text_row)

        layout.addStretch()

    def update_tokens(self, tokens: Dict[str, str]):
        """根据 token dict 重写小预览的 QSS"""
        self._tokens = tokens
        # 用 QSS 模板渲染
        # 这里偷懒 — 直接 setStyleSheet 用几个核心 token 拼出最简 QSS
        # (避免拉整个 _QSS_TEMPLATE — 用户只需要看主色和卡片)
        qss = f"""
            QWidget {{
                background-color: {tokens.get('bg_main', '#fff')};
                color: {tokens.get('text_primary', '#000')};
            }}
            QPushButton#previewAccent {{
                background-color: {tokens.get('accent', '#888')};
                color: {tokens.get('text_on_accent', 'white')};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton[kind="danger"] {{
                background-color: {tokens.get('danger', '#dc3545')};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton[kind="success"] {{
                background-color: {tokens.get('success', '#28a745')};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton[kind="muted"] {{
                background-color: {tokens.get('neutral_gray', '#888')};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QLabel#previewCard {{
                background-color: {tokens.get('bg_card', '#eee')};
                border: 1px solid {tokens.get('border', '#ccc')};
                border-radius: 6px;
            }}
            QLabel#previewPrimary {{
                color: {tokens.get('text_primary', '#000')};
            }}
            QLabel#previewSecondary {{
                color: {tokens.get('text_secondary', '#666')};
            }}
            QLabel#previewMuted {{
                color: {tokens.get('text_muted', '#999')};
            }}
            QLabel[role="title"] {{
                color: {tokens.get('text_primary', '#000')};
                font-weight: bold;
            }}
            QLabel[role="muted"] {{
                color: {tokens.get('text_muted', '#999')};
            }}
        """
        self.setStyleSheet(qss)


class ThemeEditorDialog(QDialog):
    """主题编辑器 dialog"""

    theme_saved = Signal()  # 保存成功发出, 通知 settings_dialog 刷新下拉

    def __init__(self, parent=None, edit_theme_name: Optional[str] = None):
        """
        Args:
            edit_theme_name: 如果给定, 编辑现有主题; 否则新建
        """
        super().__init__(parent)
        self.setWindowTitle("主题编辑器" if edit_theme_name is None else f"编辑主题 - {edit_theme_name}")
        self.setMinimumSize(720, 620)
        # 解除父子关系
        self.setParent(None)

        self._editing_name = edit_theme_name
        self._rows: Dict[str, _ColorRow] = {}
        self._setup_ui()

        # 初始化 token
        if edit_theme_name is not None:
            existing = get_theme(edit_theme_name)
            if existing is not None:
                self._load_from_theme(existing.tokens)
                self.name_edit.setText(existing.display_name)
                self._update_preview()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ===== 顶部: 基础主题 + 主题名 =====
        top_group = QGroupBox("基础信息")
        top_layout = QFormLayout(top_group)

        # 基础主题 (复制 token 用) — 只在新建时显示
        if self._editing_name is None:
            self.based_combo = QComboBox()
            self.based_combo.addItem("浅色 (light)", "light")
            self.based_combo.addItem("深色 (dark)", "dark")
            self.based_combo.addItem("纸感 (paper)", "paper")
            self.based_combo.currentIndexChanged.connect(self._on_based_changed)
            top_layout.addRow("从基础主题复制:", self.based_combo)
        else:
            note = QLabel("编辑现有主题 (核心 token 已加载)")
            note.setProperty("role", "muted")
            top_layout.addRow(note)

        # 主题名
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("给主题起个名字 (例: 橄榄绿 / 暮色)")
        self.name_edit.setMaxLength(30)
        top_layout.addRow("主题名:", self.name_edit)

        layout.addWidget(top_group)

        # ===== 中部: 8 个核心 token =====
        colors_group = QGroupBox("核心配色 (8 个)")
        colors_layout = QVBoxLayout(colors_group)
        colors_layout.setSpacing(4)

        # 默认 token (从 light 拿)
        defaults = get_default_core_tokens()

        for token in CORE_TOKEN_KEYS:
            row = _ColorRow(
                token_name=token,
                initial_color=defaults[token],
                default_color=defaults[token],
            )
            row.color_changed.connect(self._on_color_changed)
            self._rows[token] = row

            # 横向: label + row
            r_layout = QHBoxLayout()
            label = QLabel(CORE_TOKEN_LABELS.get(token, token))
            label.setMinimumWidth(160)
            r_layout.addWidget(label)
            r_layout.addWidget(row)
            colors_layout.addLayout(r_layout)

        # 恢复全部默认按钮
        reset_row = QHBoxLayout()
        reset_row.addStretch()
        reset_btn = QPushButton("恢复全部默认")
        reset_btn.clicked.connect(self._reset_all)
        reset_row.addWidget(reset_btn)
        colors_layout.addLayout(reset_row)

        # 用 QScrollArea 包一下, 防 8 行太多
        scroll = QScrollArea()
        scroll.setWidget(colors_group)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, stretch=2)

        # ===== 底部: 预览 =====
        preview_group = QGroupBox("实时预览")
        preview_layout = QVBoxLayout(preview_group)
        self.preview = _PreviewWidget()
        preview_layout.addWidget(self.preview)
        layout.addWidget(preview_group, stretch=1)

        # ===== 按钮行 =====
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        if self._editing_name is not None and self._editing_name.startswith("user:"):
            del_btn = QPushButton("删除这个主题")
            del_btn.setProperty("kind", "danger")
            del_btn.clicked.connect(self._on_delete)
            btn_row.addWidget(del_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("保存")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)

    # ============================================================
    # 行为
    # ============================================================

    def _on_based_changed(self, _idx: int):
        """选基础主题时: 把 8 个 token 替换成对应 builtin 的 core token"""
        based = self.based_combo.currentData()
        if not based:
            return
        from ui.styles import get_theme as _get_theme
        t = _get_theme(based)
        if t is None:
            return
        for token in CORE_TOKEN_KEYS:
            self._rows[token].set_color(t.tokens[token])
        self._update_preview()

    def _on_color_changed(self, _token: str, _color: str):
        self._update_preview()

    def _reset_all(self):
        defaults = get_default_core_tokens()
        for token in CORE_TOKEN_KEYS:
            self._rows[token].set_color(defaults[token])
        self._update_preview()

    def _load_from_theme(self, full_tokens: Dict[str, str]):
        """从已有主题加载: full_tokens 是 30+ 完整 token, 提取 8 个核心"""
        for token in CORE_TOKEN_KEYS:
            if token in full_tokens:
                self._rows[token].set_color(full_tokens[token])

    def _current_core(self) -> Dict[str, str]:
        return {t: self._rows[t].color() for t in CORE_TOKEN_KEYS}

    def _update_preview(self):
        full = derive_tokens(self._current_core())
        self.preview.update_tokens(full)

    def _on_delete(self):
        if self._editing_name is None:
            return
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定删除主题「{self._editing_name}」?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        # 如果当前正在用, 切到 light
        if get_current_theme_name() == self._editing_name:
            set_theme("light", persist=True)
        delete_user_theme(self._editing_name)
        QMessageBox.information(self, "已删除", "主题已删除。")
        self.theme_saved.emit()
        self.accept()

    def _on_save(self):
        # 校验名称
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入主题名")
            return

        # 校验 8 个色值
        core = self._current_core()
        for token, color in core.items():
            if not _is_valid_hex(color):
                QMessageBox.warning(self, "提示", f"token {token} 的颜色无效: {color}")
                return

        # 派生完整 token
        full_tokens = derive_tokens(core)

        # 检查 name 冲突 (除了自己)
        existing_names = {t.name for t in list_themes()}
        if self._editing_name is None:
            target_name = f"user:{name}"
        else:
            # 编辑现有: name 可能改了
            target_name = self._editing_name
            if f"user:{name}" != target_name and f"user:{name}" in existing_names:
                QMessageBox.warning(self, "提示", f"主题名「{name}」已存在")
                return

        # 创建/更新 Theme
        theme = Theme(
            name=target_name,
            display_name=name,
            tokens=full_tokens,
            builtin=False,
        )
        try:
            upsert_user_theme(theme)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
            return

        # 立即应用
        set_theme(target_name, persist=True)

        QMessageBox.information(self, "已保存", f"主题「{name}」已保存并应用。\n重启 WAgent 仍然生效。")
        self.theme_saved.emit()
        self.accept()


# ============================================================
# 注册 Theme 类 (延迟, 避免 styles.py 导入循环)
# ============================================================
from ui.styles import Theme  # noqa: E402
