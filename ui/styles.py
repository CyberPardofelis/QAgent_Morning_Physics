# ui/styles.py
"""WAgent 全局样式 — 主题系统

阶段 1 (当前): 3 个预设主题(light / dark / paper) + 切换 API
阶段 2 (下一步): 用户自定义颜色主题 — 通过 register_custom_theme()

设计原则:
- 颜色 token 化: 同一语义用同一 token(如 accent=主蓝),不直接写 hex
- 添加新主题 = 新建一个 token dict,不用动 QSS 模板
- 兼容旧代码: `GLOBAL_STYLESHEET` 仍是 light 主题的渲染结果,9 个
  已有 `self.setStyleSheet(GLOBAL_STYLESHEET)` 的文件可以保持不动
  (但推荐改为 `self.setStyleSheet(get_stylesheet())` 以便主题切换时同步)
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path  # 9-20 加: 让 _themes_file() 函数签名里的 Path 类型注解可用

from PySide6.QtWidgets import QApplication

# ============================================================
# Theme 数据结构
# ============================================================

@dataclass(frozen=True)
class Theme:
    """单个主题

    Attributes:
        name: 内部 id, 如 "light" / "dark" / "paper" / "user:xxx"
        display_name: UI 显示名
        tokens: 颜色 token 字典, key 是 token 名, value 是 hex 颜色
        builtin: True=预设, False=用户自定义
    """
    name: str
    display_name: str
    tokens: dict[str, str]
    builtin: bool = True

    def get(self, token: str, default: str = "#000000") -> str:
        return self.tokens.get(token, default)


# ============================================================
# 3 个预设主题
# ============================================================

# ---------- LIGHT (现在这个,浅蓝白,专业感) ----------
LIGHT_TOKENS: dict[str, str] = {
    # 背景
    "bg_main": "#fafaff",
    "bg_panel": "white",
    "bg_card": "#e8eaf0",              # 详情/编辑区次级背景(比 main 略深形成层次)
    "bg_input": "white",
    "bg_table_header": "#f8f9fa",
    "bg_tab_unselected": "#f1f5f9",
    "bg_chip": "#f1f5f9",
    "bg_chip_close_hover": "#e2e8f0",
    "bg_disabled": "#b6c2d0",
    "bg_pressed": "#d0e2ff",
    "bg_hover": "#e9f0ff",
    # 边框
    "border": "#dde1e6",
    "border_strong": "#eaeef2",
    "border_pressed": "#e6eaef",
    "splitter": "#e0e4e9",
    "splitter_hover": "#b0b8c5",
    # 文字
    "text_primary": "#1e293b",
    "text_secondary": "#5f6b7a",
    "text_muted": "#94a3b8",
    "text_disabled": "#eef2f6",
    "text_on_accent": "white",
    # 主蓝
    "accent": "#0d6efd",
    "accent_hover": "#0b5ed7",
    "accent_pressed": "#0a58ca",
    "accent_soft": "#b2d3ff",
    # 危险红
    "danger": "#dc3545",
    "danger_hover": "#bb2d3b",
    # 成功绿
    "success": "#28a745",
    "success_hover": "#218838",
    # 中性灰
    "neutral_gray": "#6c757d",
    "neutral_gray_hover": "#5c636a",
    # 滚动条
    "scrollbar": "#c1c9d2",
    "scrollbar_hover": "#a0aab5",
    # 表格
    "table_grid": "#e9ecef",
    "table_row_divider": "#f1f3f5",
    # 消息气泡
    "bubble_user_bg": "#77aaff",
    "bubble_user_fg": "white",
    "bubble_assistant_bg": "white",
    "bubble_assistant_fg": "#1e293b",
    "bubble_system_bg": "#e9ecef",
    "bubble_system_fg": "#495057",
}

# ---------- DARK (深色基底,夜读/沉浸) ----------
DARK_TOKENS: dict[str, str] = {
    # 背景 — 深炭灰系
    "bg_main": "#1a1d23",         # 主背景
    "bg_panel": "#23262e",         # 面板(左栏/菜单/tab 底)
    "bg_card": "#2a2e36",          # 详情/编辑区次级背景(比 main 略浅形成层次)
    "bg_input": "#23262e",         # 输入框(比 card 略深,形成"凹陷"字段)
    "bg_table_header": "#2a2e36",
    "bg_tab_unselected": "#2a2e36",
    "bg_chip": "#2a2e36",
    "bg_chip_close_hover": "#3a3f48",
    "bg_disabled": "#3a3f48",
    "bg_pressed": "#1f3a6e",
    "bg_hover": "#1f3a6e",         # hover/选中 浅蓝深
    # 边框
    "border": "#3a3f48",
    "border_strong": "#2a2e36",
    "border_pressed": "#4a5058",
    "splitter": "#2a2e36",
    "splitter_hover": "#4a5058",
    # 文字
    "text_primary": "#e2e8f0",
    "text_secondary": "#a0aab5",
    "text_muted": "#6c7480",
    "text_disabled": "#4a5058",
    "text_on_accent": "white",
    # 主蓝(深色下用稍亮蓝,提升对比)
    "accent": "#3b82f6",
    "accent_hover": "#2563eb",
    "accent_pressed": "#1d4ed8",
    "accent_soft": "#1e3a5f",
    # 危险红
    "danger": "#ef4444",
    "danger_hover": "#dc2626",
    # 成功绿
    "success": "#22c55e",
    "success_hover": "#16a34a",
    # 中性灰
    "neutral_gray": "#6b7280",
    "neutral_gray_hover": "#9ca3af",
    # 滚动条
    "scrollbar": "#3a3f48",
    "scrollbar_hover": "#4a5058",
    # 表格
    "table_grid": "#2a2e36",
    "table_row_divider": "#2a2e36",
    # 消息气泡
    "bubble_user_bg": "#2c5db0",         # 蓝气泡(暗色下用更亮的蓝)
    "bubble_user_fg": "white",
    "bubble_assistant_bg": "#2a2e36",    # 助手气泡 = bg_card
    "bubble_assistant_fg": "#e2e8f0",
    "bubble_system_bg": "#3a3f48",       # 灰气泡(暗)
    "bubble_system_fg": "#a0aab5",
}

# ---------- PAPER (米黄底,稿纸感,适合写作) ----------
PAPER_TOKENS: dict[str, str] = {
    # 背景 — 米黄/纸色
    "bg_main": "#f4ecd8",
    "bg_panel": "#fbf6e9",
    "bg_card": "#ede4cc",          # 详情/编辑区(比 main 略深,米色卡片)
    "bg_input": "#fdf9ec",
    "bg_table_header": "#ede4cc",
    "bg_tab_unselected": "#ede4cc",
    "bg_chip": "#ede4cc",
    "bg_chip_close_hover": "#d8cdaf",
    "bg_disabled": "#d8cdaf",
    "bg_pressed": "#d8c8a0",
    "bg_hover": "#ede4cc",
    # 边框 — 暖灰/棕
    "border": "#d8cdaf",
    "border_strong": "#c9b88f",
    "border_pressed": "#b8a273",
    "splitter": "#c9b88f",
    "splitter_hover": "#a8946a",
    # 文字 — 暖深棕
    "text_primary": "#3d2f1a",
    "text_secondary": "#6b5635",
    "text_muted": "#8a7a5a",
    "text_disabled": "#b8a98a",
    "text_on_accent": "white",
    # 主蓝(纸感下用墨蓝,更和谐)
    "accent": "#1e4a7a",
    "accent_hover": "#163655",
    "accent_pressed": "#0f2a40",
    "accent_soft": "#c9d6e3",
    # 危险红
    "danger": "#a93226",
    "danger_hover": "#922b21",
    # 成功绿
    "success": "#5a7a3a",
    "success_hover": "#4a6a2a",
    # 中性灰
    "neutral_gray": "#8a7a5a",
    "neutral_gray_hover": "#6b5635",
    # 滚动条
    "scrollbar": "#c9b88f",
    "scrollbar_hover": "#a8946a",
    # 表格
    "table_grid": "#e0d4b0",
    "table_row_divider": "#e8dcc0",
    # 消息气泡
    "bubble_user_bg": "#3a6b8a",          # 墨蓝(纸感)
    "bubble_user_fg": "white",
    "bubble_assistant_bg": "#fbf6e9",     # 纸张色
    "bubble_assistant_fg": "#3d2f1a",
    "bubble_system_bg": "#d8cdaf",        # 米灰
    "bubble_system_fg": "#6b5635",
}


LIGHT = Theme(name="light", display_name="浅色", tokens=LIGHT_TOKENS, builtin=True)
DARK = Theme(name="dark", display_name="深色", tokens=DARK_TOKENS, builtin=True)
PAPER = Theme(name="paper", display_name="纸感", tokens=PAPER_TOKENS, builtin=True)

THEMES: dict[str, Theme] = {
    "light": LIGHT,
    "dark": DARK,
    "paper": PAPER,
}

DEFAULT_THEME = "light"


# ============================================================
# QSS 模板 (用 {token} 占位符)
# ============================================================

_QSS_TEMPLATE = """
/* === 主题: {theme_display_name} === */
/* 2026-09-02 加(抄 SAgent):所有没设 background 的 QWidget 默认 transparent,
   让 LLM API / 期刊设置 / 每日总结范围 等用 QScrollArea 包内容的 tab 自动透出
   dialog 的 bg_main,不再显示 Win 默认的白底
   - 关键: 这条 QWidget rule 必须在 QMainWindow / QDialog 块之前
     QMainWindow / QDialog 是 QWidget 子类, specificity 一样时后写者胜
     把 QWidget 放最前 → 后面的 QMainWindow 块(background-color bg_main)能覆盖它
   - 不会覆盖 QPushButton 等(QPushButton 自己的 rule specificity=1类 > QWidget 的 0) */
QWidget {{
    background-color: transparent;
}}
/* 全局字体和背景 */
QMainWindow, QDialog {{
    background-color: {bg_main};
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    /* 2026-09-02 加(抄 SAgent):{font_size_pt} 从 QApplication.font().pointSize() 读
       让 QSS 字号跟设置里的"全局字体大小"combobox 一致
       (之前 hardcode 14px → 用户改 16px 后 menu/tree widget 退回 system font 9pt,小一截) */
    font-size: {font_size_pt}pt;
    color: {text_primary};
}}
/* 兜底文字色(部分 widget 不继承父 QMainWindow 的 color,需要单独设) */
QCheckBox, QRadioButton, QLabel, QGroupBox, QMenu, QMenuBar, QTabBar, QTreeWidget, QListWidget, QTableWidget, QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QToolButton {{
    color: {text_primary};
}}
/* 2026-09-02 加:SpinBox / DoubleSpinBox 上下调整数值的输入框
   Win 默认是白底 + 黑字,切 dark 主题后白底刺眼
   走 {bg_input} token 跟主题,文字色 {text_primary} */
QSpinBox, QDoubleSpinBox {{
    background-color: {bg_input};
    color: {text_primary};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 22px;
    selection-background-color: {accent_soft};
    selection-color: {text_primary};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {accent};
}}
/* SpinBox 的上下按钮(up/down arrow)— 默认跟 widget 同色,Win 上常常看着跟底色不同 */
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {bg_card};
    border: none;
    width: 18px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {bg_hover};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: none;
    width: 8px;
    height: 8px;
    /* 用 border 画三角形: 不好看,fallback 到 system arrow */
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 6px solid {text_secondary};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: none;
    width: 8px;
    height: 8px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {text_secondary};
}}
/* 分割器样式 */
QSplitter::handle {{
    background-color: {splitter};
    width: 1px;
}}
QSplitter::handle:hover {{
    background-color: {splitter_hover};
}}
/* 左侧边栏 */
QWidget#leftWidget {{
    background-color: {bg_panel};
    border-right: 1px solid {border_strong};
}}

QPushButton#newConvBtn:hover {{
    background-color: {accent_hover};
}}
/* 对话列表 */
/* 2026-09-02 改:加 font-size: {font_size_pt}pt + color: {text_primary},跟全局字号同步
   之前没设 → Win 退回 system font 9pt,跟 settings 里 14/16 错位 */
QListWidget {{
    background-color: transparent;
    border: none;
    outline: none;
    color: {text_primary};
    font-size: {font_size_pt}pt;
}}
QListWidget::item {{
    background-color: transparent;
    border-radius: 8px;
    padding: 10px;
    margin: 4px 8px;
    color: {text_primary};
}}
QListWidget::item:selected {{
    background-color: {bg_hover};
    color: {accent};
}}
/* 通用 QComboBox 样式 */
/* 2026-09-02 改:加 font-size: {font_size_pt}pt,设置下拉/主题下拉等跟全局字号同步
   之前没设 → Win 退回 system font 9pt,跟 settings 里 14/16 错位 */
QComboBox {{
    background-color: {bg_input};
    border: 1px solid {border};
    border-radius: 20px;
    padding: 6px 16px;
    min-width: 120px;
    color: {text_primary};
    font-size: {font_size_pt}pt;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid {text_secondary};
    width: 0;
    height: 0;
    margin-right: 8px;
}}
QComboBox:hover {{
    border-color: {accent};
}}
QComboBox QAbstractItemView {{
    background-color: {bg_input};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 4px;
    outline: none;
    selection-background-color: {bg_hover};
    selection-color: {accent};
    /* 2026-09-02 加:下拉打开的项也跟全局字号 */
    font-size: {font_size_pt}pt;
}}
QComboBox QAbstractItemView::item {{
    padding: 8px 12px;
    border-radius: 6px;
    color: {text_primary};
}}
/* 通用 QLineEdit 样式(用于搜索框等) */
/* 2026-09-02 改:加 font-size: {font_size_pt}pt,跟全局字号同步 */
QLineEdit {{
    background-color: {bg_input};
    border: 1px solid {border};
    border-radius: 20px;
    padding: 12px 16px;
    selection-background-color: {accent_soft};
    color: {text_primary};
    font-size: {font_size_pt}pt;
}}
QLineEdit:focus {{
    border-color: {accent};
    outline: none;
}}
/* 通用 QTextEdit 样式(用于多行输入) */
QTextEdit {{
    background-color: {bg_input};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 8px 12px;
    selection-background-color: {accent_soft};
    color: {text_primary};
    font-size: {font_size_pt}pt;
}}
QTextEdit:focus {{
    border-color: {accent};
    outline: none;
}}
/* 主页面文本输入样式 */
QPlainTextEdit {{
    background-color: {bg_input};
    border: 1px solid {border};
    border-radius: 20px;
    padding: 8px 18px;
    selection-background-color: {accent_soft};
    color: {text_primary};
    font-size: {font_size_pt}pt;
}}
/* 通用按钮样式 */
QPushButton {{
    background-color: {accent};
    color: {text_on_accent};
    border: none;
    border-radius: 14px;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {accent_hover};
}}
QPushButton:pressed {{
    background-color: {accent_pressed};
}}
QPushButton:disabled {{
    background-color: {bg_disabled};
    color: {text_disabled};
}}
/* 特殊按钮颜色覆盖 */
/* 2026-09-02 v2 改:用 [state="stop"] dynamic property 替代 objectName="stopBtn"
   原因: 之前 objectName="stopBtn" 永远让按钮走 danger 红色,无法切回 idle 的 accent 蓝色
   现在 input_panel 根据状态 setProperty("state", "stop"/"send"),QSS 接管样式
   - state="stop"  → 红色 (streaming)
   - 无 state / state="send" → 走默认 QPushButton 规则 (accent 蓝, idle) */
QPushButton[state="stop"] {{
    background-color: {danger};
    color: {text_on_accent};
    border: none;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton[state="stop"]:hover {{
    background-color: {danger_hover};
    color: {text_on_accent};
}}
QPushButton[state="stop"]:disabled {{
    background-color: {bg_disabled};
    color: {text_disabled};
}}
/* 兼容旧 #stopBtn objectName (如果其他地方还在用),跟 [state="stop"] 同效 */
QPushButton#stopBtn {{
    background-color: {danger};
    color: {text_on_accent};
    border: none;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton#stopBtn:hover {{
    background-color: {danger_hover};
    color: {text_on_accent};
}}
QPushButton#regenerateBtn {{
    background-color: {neutral_gray};
    color: {text_on_accent};
    border: none;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton#regenerateBtn:hover {{
    background-color: {neutral_gray_hover};
    color: {text_on_accent};
}}
QPushButton#regenerateBtn:disabled {{
    background-color: {bg_disabled};
    color: {text_disabled};
}}
/* ---------- 按钮 kind 角色 (替代硬编码 setStyleSheet) ---------- */
QPushButton[kind="danger"] {{
    background-color: {danger};
    color: {text_on_accent};
    border: none;
    padding: 4px 10px;
}}
QPushButton[kind="danger"]:hover {{
    background-color: {danger_hover};
}}
QPushButton[kind="success"] {{
    background-color: {success};
    color: {text_on_accent};
    border: none;
    padding: 4px 10px;
}}
QPushButton[kind="success"]:hover {{
    background-color: {success_hover};
}}
QPushButton[kind="muted"] {{
    background-color: {neutral_gray};
    color: {text_on_accent};
    border: none;
    padding: 4px 10px;
}}
QPushButton[kind="muted"]:hover {{
    background-color: {neutral_gray_hover};
}}
/* ---------- 文字角色 (替代 color: #888 / gray / #555) ---------- */
QLabel[role="hint"] {{
    color: {text_secondary};
    background: transparent;
}}
QLabel[role="muted"] {{
    color: {text_muted};
    background: transparent;
}}
QLabel[role="title"] {{
    color: {text_primary};
    font-weight: bold;
}}
QLabel[role="chip-label"] {{
    color: {text_primary};
    font-weight: bold;
    padding: 2px 12px;
    background-color: {bg_chip};
    border-radius: 4px;
}}
QLabel[state="muted"] {{
    color: {text_muted};
    font-weight: bold;
}}
QLabel[state="warn"] {{
    color: #FFA726;
    font-weight: bold;
}}
/* ---------- 状态指示灯 (state="ok|warn|error") ---------- */
QWidget[state="ok"] {{
    background-color: {success};
    border-radius: 5px;
}}
QWidget[state="warn"] {{
    background-color: #FFA726;
    border-radius: 5px;
}}
QWidget[state="error"] {{
    background-color: {danger};
    border-radius: 5px;
}}
/* ---------- 消息气泡 (objectName="bubble_user/assistant/system") ---------- */
/* 2026-09-02 v6 改:用 #objectName 替代 [kind=...] attribute selector
   原因: PySide6 6.10 下 setProperty("kind", ...) 在 chat_view 嵌套上下文里 QSS 不匹配
         (debug 看到 palette bg = #000000, 气泡背景是黑色)
   改用 setObjectName + #objectName 选择器, 这是 Qt 5/6 最稳的 dynamic styling */
QWidget#bubble_user {{
    background-color: {bubble_user_bg};
    border-radius: 10px;
    border: none;
}}
QWidget#bubble_user QTextBrowser {{
    color: {bubble_user_fg};
    background: transparent;
    border: 0;
    outline: 0;
}}
QWidget#bubble_assistant {{
    background-color: {bubble_assistant_bg};
    border-radius: 10px;
    border: none;
}}
QWidget#bubble_assistant QTextBrowser {{
    color: {bubble_assistant_fg};
    background: transparent;
    border: 0;
    outline: 0;
}}
QWidget#bubble_system {{
    background-color: {bubble_system_bg};
    border-radius: 10px;
    border: none;
}}
QWidget#bubble_system QTextBrowser {{
    color: {bubble_system_fg};
    background: transparent;
    border: 0;
    outline: 0;
}}
/* ---------- 详情/编辑区卡片背景(给子 widget 用 objectName="#xxxCard") ---------- */
QWidget#detailCard {{
    background-color: {bg_card};
    border: 1px solid {border};
    border-radius: 8px;
}}
QWidget#chatScrollContent {{
    background-color: {bg_main};
}}
/* QScrollArea 默认 viewport 用 QPalette.Base(白色),跟主题冲突 — 强制用 main 背景 */
/* viewport 是 QScrollArea 的直接子 QWidget, 也要单独设 */
QScrollArea, QScrollArea > QWidget {{
    background-color: {bg_main};
    border: none;
}}
/* 2026-09-02 改(抄 SAgent):QGroupBox 背景改 transparent
   之前用 {bg_card} 在 light 主题显示浅灰底,跟 dialog {bg_main} 形成"灰色硬编码背景"
   (期刊设置 / 每日总结设置 / LLM API 等 tab 的 GroupBox 都受影响)
   改 transparent 后,GroupBox 跟 dialog 背景一致,视觉清爽,只保留边框作分区 */
QGroupBox {{
    background-color: transparent;
    border: 1px solid {border};
    border-radius: 6px;
    margin-top: 8px;
    padding: 10px 8px 8px 8px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: {text_primary};
}}
/* 表格样式 */
QTableWidget {{
    background-color: {bg_input};
    border: 1px solid {border};
    border-radius: 12px;
    gridline-color: {table_grid};
    selection-background-color: {bg_hover};
    selection-color: {accent};
    color: {text_primary};
}}
QTableWidget::item {{
    padding: 8px;
    border-bottom: 1px solid {table_row_divider};
}}
QHeaderView::section {{
    background-color: {bg_table_header};
    padding: 8px;
    border: none;
    font-weight: bold;
    color: {text_primary};
}}
/* 滚动条样式(扁平圆角) */
QScrollBar:vertical {{
    border: none;
    background: transparent;
    width: 8px;
    border-radius: 4px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {scrollbar};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {scrollbar_hover};
}}
QScrollBar:horizontal {{
    border: none;
    background: transparent;
    height: 8px;
    border-radius: 4px;
    margin: 0px;
}}
QScrollBar::handle:horizontal {{
    background: {scrollbar};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {scrollbar_hover};
}}
/* 菜单栏 */
/* 2026-09-02 改:加 font-size: {font_size_pt}pt,跟全局字号同步
   之前没设 → Win 退回 system font 9pt,跟 settings 里 14/16 错位 */
QMenuBar {{
    background-color: {bg_panel};
    border-bottom: 1px solid {border_strong};
    color: {text_primary};
    font-size: {font_size_pt}pt;
}}
QMenuBar::item {{
    padding: 8px 12px;
    background: transparent;
    border-radius: 6px;
    /* 2026-09-02 加:菜单项(主界面"设置"下拉里那两个选项)也跟全局字号 */
    font-size: {font_size_pt}pt;
}}
QMenuBar::item:selected {{
    background-color: {bg_hover};
}}
QMenu {{
    background-color: {bg_panel};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 6px;
    color: {text_primary};
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: 6px;
    /* 2026-09-02 加:下拉菜单项("设置"下拉里"首选项" / "期刊与论文")也跟全局字号 */
    font-size: {font_size_pt}pt;
}}
QMenu::item:selected {{
    background-color: {bg_hover};
    color: {accent};
}}
/* ---------- 扁平化标签页 ---------- */
/* 2026-09-02 改:QTabBar 自己设 {bg_main} 背景
   之前没设 → tab 之间空白处退到 system 颜色,Win 浅色主题下显示白块(用户报"左上角白色奇怪东西")
   现在 tab 之间空白处跟 dialog 背景一致
   + 固定高度,防止 tab 放不下时出现 1-2px 横滚条(用户第二轮反馈) */
QTabBar {{
    background-color: {bg_main};
    border: none;
    /* 2026-09-02 加:防止 tab 放不下时出现滚动条 */
    qproperty-usesScrollButtons: false;
    qproperty-drawBase: false;
    min-height: 0;
}}
/* 2026-09-02 加:兜底隐藏 QTabBar 任何可能出现的滚动元素 */
QTabBar QScrollBar:horizontal,
QTabBar QScrollBar:vertical {{
    height: 0px;
    width: 0px;
    background: transparent;
    border: none;
    margin: 0px;
}}
QTabBar QScrollBar::handle:horizontal,
QTabBar QScrollBar::handle:vertical {{
    background: transparent;
    min-width: 0;
    min-height: 0;
    height: 0;
    width: 0;
}}
QTabBar QScrollBar::add-line:horizontal,
QTabBar QScrollBar::sub-line:horizontal,
QTabBar QScrollBar::add-line:vertical,
QTabBar QScrollBar::sub-line:vertical,
QTabBar QScrollBar::add-page:horizontal,
QTabBar QScrollBar::sub-page:horizontal,
QTabBar QScrollBar::add-page:vertical,
QTabBar QScrollBar::sub-page:vertical {{
    background: none;
    border: none;
    width: 0;
    height: 0;
}}
QTabBar::scroller,
QTabBar::tear-left,
QTabBar::tear-right {{
    width: 0;
    height: 0;
    border: none;
    background: transparent;
}}
QTabBar QToolButton {{
    width: 0;
    height: 0;
    padding: 0;
    border: none;
}}
QTabWidget::pane {{
    border: 1px solid {border};
    border-top: none;             /* 2026-09-02 改:去 top border — 之前 top: -1px 配合 border
                                       在某些主题下会画出 1px 横条(用户报"首选项左上角横滚条")
                                       现在 top 无 border,1-2px 横条不再出现 */
    border-radius: 12px;
    background: {bg_card};
    top: 0;                       /* 2026-09-02 改:不再 -1px,跟 tab bar 齐平 */
}}

QTabBar::tab {{
    background-color: {bg_tab_unselected};
    border: 1px solid {border};
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 4px 16px;
    margin-right: 2px;
    /* 2026-09-02 加:跟全局字体大小同步(抄 SAgent) */
    font-size: {font_size_pt}pt;
    font-weight: 500;
    color: {text_primary};
}}

QTabBar::tab:selected {{
    background-color: {bg_input};
    border-bottom: 1px solid {bg_input};
    color: {accent};
}}

QTabBar::tab:hover:!selected {{
    background-color: {bg_hover};
}}

QTabBar::tab:first {{
    margin-left: 8px;
}}

QTabBar::tab:last {{
    margin-right: 8px;
}}

QTabBar::tab {{
    min-height: 24px;
}}
QTreeWidget {{
    background-color: transparent;
    border: none;
    outline: none;
    color: {text_primary};
    /* 2026-09-02 加:跟全局字号同步,左边对话树随 settings 改字号一起变大/小 */
    font-size: {font_size_pt}pt;
}}
/* 2026-09-02 加:左侧对话列表(QListWidget)同样处理 — 已被前面的 QListWidget 规则合并,留注释防重复 */
QListWidget {{
    /* 上面已经有 font-size 规则,这里不重复 */
}}
QTreeWidget::item {{
    padding: 6px;
    border-radius: 6px;
}}
QTreeWidget::item:selected {{
    background-color: {bg_hover};
    color: {accent};
}}
QTreeWidget::item:hover {{
    background-color: {bg_tab_unselected};
}}
QTreeWidget::branch {{
    background: transparent;
}}
QTreeWidget::branch:has-children:!has-siblings:closed,
QTreeWidget::branch:closed:has-children:has-siblings {{
    border-image: none;
    image: none;
}}
QTreeWidget::branch:open:has-children:!has-siblings,
QTreeWidget::branch:open:has-children:has-siblings {{
    border-image: none;
    image: none;
}}

/* ---------- 输入面板: + 附件按钮 ---------- */
/* + 号由 _AttachButton.paintEvent 自绘,这里只管背景/边框/圆角/hover */
QToolButton#attachBtn {{
    background-color: {bg_input};
    color: {accent};
    border: 1px solid {border};
    border-radius: 18px;
    padding: 0;
}}
QToolButton#attachBtn:hover {{
    background-color: {bg_hover};
    border-color: {accent};
    color: {accent_hover};
}}
QToolButton#attachBtn:pressed {{
    background-color: {bg_pressed};
}}
QToolButton#attachBtn:disabled {{
    background-color: {bg_tab_unselected};
    color: {bg_disabled};
    border-color: {border_pressed};
}}

/* ---------- 附件 chip 区 ---------- */
QScrollArea#attachScroll {{
    background: transparent;
    border: none;
}}
QScrollArea#attachScroll > QWidget > QWidget {{
    background: transparent;
}}
QWidget#attachChipContainer {{
    background: transparent;
}}

/* 2026-09-02 加:附件 chip 容器背景 + 边框 + 圆角(QAgent 的 _AttachmentChip 走这个)
   之前 QAgent 在 input_panel.py 里 hardcode #f1f5f9,切 dark 主题还是浅灰 → 改走 token */
QWidget#attachmentChip {{
    background-color: {bg_chip};
    border: 1px solid {border};
    border-radius: 14px;
}}

QLabel#chipName {{
    color: {text_primary};
    font-size: 13px;
    background: transparent;
}}
QLabel#chipName[state="error"] {{
    color: {danger};
}}
QToolButton#chipCloseBtn {{
    background: transparent;
    color: {text_muted};
    border: none;
    border-radius: 9px;
    /* 16px 是 × 关闭符的视觉大小,跟全局字体大小无关,保留 hardcode */
    font-size: 16px;
    font-weight: 400;
    padding: 0;
    qproperty-alignment: AlignCenter;
}}
QToolButton#chipCloseBtn:hover {{
    background-color: {bg_chip_close_hover};
    color: {text_primary};
}}

QScrollArea#attachScroll QScrollBar:horizontal {{
    height: 6px;
    background: transparent;
}}
QScrollArea#attachScroll QScrollBar::handle:horizontal {{
    background: {scrollbar};
    border-radius: 3px;
    min-width: 20px;
}}
QScrollArea#attachScroll QScrollBar::add-line:horizontal,
QScrollArea#attachScroll QScrollBar::sub-line:horizontal {{
    background: none;
    border: none;
    width: 0;
    height: 0;
}}
"""


# ============================================================
# 主题状态管理
# ============================================================

_current_theme: str = DEFAULT_THEME
_custom_themes: dict[str, Theme] = {}


def get_current_theme_name() -> str:
    return _current_theme


def get_theme(name: str) -> Theme | None:
    """获取主题(优先 builtin,再 custom)"""
    return THEMES.get(name) or _custom_themes.get(name)


def list_themes() -> list[Theme]:
    """所有主题:builtin 优先,custom 追加在后面"""
    return list(THEMES.values()) + list(_custom_themes.values())


def register_custom_theme(theme: Theme) -> None:
    """阶段 2: 注册一个用户自定义主题(替换已有同名 custom)"""
    if theme.builtin:
        raise ValueError("builtin 主题不能被 register_custom_theme 覆盖")
    _custom_themes[theme.name] = theme


def _get_current_font_pt() -> int:
    """2026-09-02 加(抄 SAgent):从 QApplication 读当前全局字号(point)。
    让 _QSS_TEMPLATE 里的 {font_size_pt} 跟 QApplication.setFont 同步。
    - 启动时 main.py 调 setFont(font) (setPointSize 从 Config.get_font_size() 读)
    - 这里读出实际 point size,默认 14
    - get_stylesheet() 渲染 QSS 时替换 {font_size_pt} 占位符
    - 这样所有 widget 的 font-size: {font_size_pt}pt 跟 settings 里 combobox 显示的字号一致
    """
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            pt = app.font().pointSize()
            if pt and pt > 0:
                return int(pt)
    except Exception:
        pass
    return 14


def get_stylesheet(theme_name: str | None = None) -> str:
    """渲染指定主题的 QSS。None = 用当前主题。"""
    name = theme_name or _current_theme
    theme = get_theme(name)
    if theme is None:
        theme = LIGHT
    return _QSS_TEMPLATE.format(
        theme_display_name=theme.display_name,
        font_size_pt=_get_current_font_pt(),
        **theme.tokens,
    )


def set_theme(name: str, app: QApplication | None = None,
              persist: bool = True) -> bool:
    """切换主题。

    Args:
        name: 主题 id(builtin 或 custom)
        app: QApplication 实例,None = 用 QApplication.instance()
        persist: True = 保存到 QSettings

    Returns: True=成功,False=主题不存在
    """
    global _current_theme
    if get_theme(name) is None:
        return False
    _current_theme = name
    if app is None:
        try:
            app = QApplication.instance()
        except Exception:
            app = None
    if app is not None:
        apply_to_all_windows(app, get_stylesheet(name))
        # 音乐 widget 联动: 用新色板重刷
        try:
            from audio.ui.player_widget import apply_music_palette
            apply_music_palette()
        except Exception:
            pass
    if persist:
        try:
            from config.settings import SettingsKeys, get_settings
            get_settings().setValue(SettingsKeys.UI_THEME, name)
        except Exception:
            pass
    return True


def apply_to_all_windows(app: QApplication, stylesheet: str) -> None:
    """给 app + 所有 topLevelWidgets 设 stylesheet(覆盖 widget 自己的 QSS)"""
    app.setStyleSheet(stylesheet)
    for w in app.topLevelWidgets():
        with contextlib.suppress(Exception):
            w.setStyleSheet(stylesheet)


def apply_current_theme(app: QApplication | None = None) -> None:
    """应用当前主题到 app(用于启动)

    1. 读 QSettings 里保存的主题,设到 _current_theme
    2. 给 app + 所有 topLevelWidgets 设 stylesheet
    """
    if app is None:
        try:
            app = QApplication.instance()
        except Exception:
            return
    if app is None:
        return
    # 先读用户保存的主题
    try:
        from config.settings import SettingsKeys, get_settings
        saved = get_settings().value(SettingsKeys.UI_THEME, DEFAULT_THEME)
        if get_theme(saved) is not None:
            global _current_theme
            _current_theme = saved
    except Exception:
        pass
    apply_to_all_windows(app, get_stylesheet())


# ============================================================
# 自定义主题: 8 个核心 token + 派生 25+ token
# ============================================================

# 用户在编辑器里能调的 8 个核心 token
CORE_TOKEN_KEYS = [
    "bg_main",          # 主背景(大区域)
    "bg_card",          # 次级背景(详情/编辑区)
    "accent",           # 主强调色(按钮/链接/选中)
    "accent_soft",      # 主强调色浅(选中背景/hover 浅)
    "text_primary",     # 主文字
    "text_secondary",   # 次文字(标签/提示)
    "danger",           # 危险红
    "success",          # 成功绿
]


def _shade(hex_color: str, factor: float) -> str:
    """调整颜色亮度。

    factor > 1.0  变浅 (e.g. 1.1 = 浅 10%)
    factor < 1.0  变深 (e.g. 0.85 = 深 15%)
    factor = 0.0  纯黑 (没意义,保护下限)
    返回 #rrggbb 格式。
    """
    if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
        return hex_color
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except ValueError:
        return hex_color
    if factor >= 1.0:
        t = min(factor - 1.0, 1.0)
        r = int(r + (255 - r) * t)
        g = int(g + (255 - g) * t)
        b = int(b + (255 - b) * t)
    else:
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _luminance(hex_color: str) -> float:
    """粗略亮度 (0.0 黑 ~ 1.0 白), 用于决定 accent 上的文字色。"""
    if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
        return 1.0
    try:
        r = int(hex_color[1:3], 16) / 255
        g = int(hex_color[3:5], 16) / 255
        b = int(hex_color[5:7], 16) / 255
    except ValueError:
        return 1.0
    return 0.299 * r + 0.587 * g + 0.114 * b


def derive_tokens(core: dict[str, str]) -> dict[str, str]:
    """从 8 个核心 token 派生 30+ 个完整 token dict。

    用于: 用户在编辑器里调 8 个核心色, 系统自动算出剩余 25+ token。
    返回的 dict 可直接喂给 Theme(tokens=...)。
    """
    bg_main = core["bg_main"]
    bg_card = core["bg_card"]
    accent = core["accent"]
    accent_soft = core["accent_soft"]
    text_primary = core["text_primary"]
    text_secondary = core["text_secondary"]
    danger = core["danger"]
    success = core["success"]

    # 决定 accent 上的文字色 (亮 accent -> 黑字, 暗 accent -> 白字)
    text_on_accent = "white" if _luminance(accent) < 0.6 else "black"

    # 决定 bg_panel/bg_input: 通常跟 card 同色
    bg_panel = bg_card
    bg_input = _shade(bg_card, 1.04)  # 比 card 略亮一点(在 dark 主题)或略深(在 light)

    out: dict[str, str] = {
        # 用户调
        "bg_main": bg_main,
        "bg_card": bg_card,
        "accent": accent,
        "accent_soft": accent_soft,
        "text_primary": text_primary,
        "text_secondary": text_secondary,
        "danger": danger,
        "success": success,
        # 派生
        "bg_panel": bg_panel,
        "bg_input": bg_input,
        "bg_table_header": _shade(bg_card, 0.97),  # 略深
        "bg_tab_unselected": _shade(bg_card, 0.97),
        "bg_chip": _shade(bg_card, 0.97),
        "bg_chip_close_hover": _shade(bg_card, 0.93),
        "bg_disabled": _shade(text_secondary, 1.2),
        "bg_pressed": _shade(accent_soft, 0.95),
        "bg_hover": accent_soft,
        "border": _shade(text_secondary, 1.3),
        "border_strong": _shade(text_secondary, 1.15),
        "border_pressed": _shade(accent, 1.2),
        "splitter": _shade(text_secondary, 1.2),
        "splitter_hover": _shade(text_secondary, 1.0),
        "text_muted": _shade(text_secondary, 1.2),
        "text_disabled": _shade(text_secondary, 1.4),
        "text_on_accent": text_on_accent,
        "accent_hover": _shade(accent, 0.88),
        "accent_pressed": _shade(accent, 0.78),
        "danger_hover": _shade(danger, 0.88),
        "success_hover": _shade(success, 0.88),
        "neutral_gray": text_secondary,
        "neutral_gray_hover": text_primary,
        "scrollbar": _shade(text_secondary, 1.3),
        "scrollbar_hover": _shade(text_secondary, 1.0),
        "table_grid": _shade(text_secondary, 1.4),
        "table_row_divider": _shade(bg_card, 0.96),
        # 气泡 (用现有 token 派生)
        "bubble_user_bg": accent,
        "bubble_user_fg": text_on_accent,
        "bubble_assistant_bg": bg_card,
        "bubble_assistant_fg": text_primary,
        "bubble_system_bg": _shade(bg_card, 0.94),
        "bubble_system_fg": text_secondary,
    }
    return out


def get_default_core_tokens() -> dict[str, str]:
    """从 light 主题拿 8 个核心 token 的默认值(用于"新建自定义主题"对话框初始化)"""
    return {k: LIGHT.tokens[k] for k in CORE_TOKEN_KEYS}


# ============================================================
# 持久化: data/themes.json
# ============================================================

def _themes_file() -> Path:
    return Path("data") / "themes.json"


def load_user_themes() -> None:
    """启动时调: 从 data/themes.json 加载用户自定义主题到 _custom_themes"""
    global _custom_themes
    _custom_themes.clear()
    fp = _themes_file()
    if not fp.exists():
        return
    try:
        import json
        data = json.loads(fp.read_text(encoding="utf-8"))
        for entry in data.get("themes", []):
            try:
                t = Theme(
                    name=entry["name"],
                    display_name=entry.get("display_name", entry["name"]),
                    tokens=entry["tokens"],
                    builtin=False,
                )
                _custom_themes[t.name] = t
            except Exception:
                continue
    except Exception:
        pass


def save_user_themes() -> None:
    """保存 _custom_themes 到 data/themes.json"""
    fp = _themes_file()
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    try:
        import json
        data = {
            "version": 1,
            "themes": [
                {"name": t.name, "display_name": t.display_name, "tokens": t.tokens}
                for t in _custom_themes.values()
            ],
        }
        fp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def delete_user_theme(name: str) -> bool:
    """删除一个用户自定义主题"""
    global _custom_themes
    if name not in _custom_themes:
        return False
    del _custom_themes[name]
    save_user_themes()
    return True


def upsert_user_theme(theme: Theme) -> None:
    """新建/更新一个用户自定义主题"""
    if theme.builtin:
        raise ValueError("builtin 主题不能作为 user theme 写入")
    if not theme.name.startswith("user:"):
        # 强制加 user: 前缀, 跟 builtin 区分
        theme = Theme(
            name=f"user:{theme.name}",
            display_name=theme.display_name,
            tokens=theme.tokens,
            builtin=False,
        )
    register_custom_theme(theme)
    save_user_themes()


# 兼容旧名: 默认 light 主题的渲染结果
# 9 个老文件用的 self.setStyleSheet(GLOBAL_STYLESHEET) 仍可工作
# 推荐改用 self.setStyleSheet(get_stylesheet()) 让主题切换时同步
GLOBAL_STYLESHEET = get_stylesheet(DEFAULT_THEME)
