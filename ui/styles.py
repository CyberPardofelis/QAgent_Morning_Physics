# ------ 全局样式 ------

GLOBAL_STYLESHEET = """
/* 全局字体和背景 */
QMainWindow, QDialog {
    background-color: #fafaff;
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    font-size: 14px;
}
/* 分割器样式 */
QSplitter::handle {
    background-color: #e0e4e9;
    width: 1px;
}
QSplitter::handle:hover {
    background-color: #b0b8c5;
}
/* 左侧边栏 */
QWidget#leftWidget {
    background-color: white;
    border-right: 1px solid #eaeef2;
}

QPushButton#newConvBtn:hover {
    background-color: #0b5ed7;
}
/* 对话列表 */
QListWidget {
    background-color: transparent;
    border: none;
    outline: none;
}
QListWidget::item {
    background-color: transparent;
    border-radius: 8px;
    padding: 10px;
    margin: 4px 8px;
    color: #1e293b;
}
QListWidget::item:selected {
    background-color: #e9f0ff;
    color: #0d6efd;
}
/* 通用 QComboBox 样式 */
QComboBox {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 20px;
    padding: 6px 16px;
    min-width: 120px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox::down-arrow {
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #5f6b7a;
    width: 0;
    height: 0;
    margin-right: 8px;
}
QComboBox:hover {
    border-color: #0d6efd;
}
QComboBox QAbstractItemView {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 8px;
    padding: 4px;
    outline: none;
    selection-background-color: #e9f0ff;
    selection-color: #0d6efd;
}
QComboBox QAbstractItemView::item {
    padding: 8px 12px;
    border-radius: 6px;
}
/* 通用 QLineEdit 样式（用于搜索框等） */
QLineEdit {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 20px;
    padding: 12px 16px;
    selection-background-color: #b2d3ff;
}
QLineEdit:focus {
    border-color: #0d6efd;
    outline: none;
}
/* 通用 QTextEdit 样式（用于多行输入） */
QTextEdit {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 14px;
    padding: 8px 12px;
    selection-background-color: #b2d3ff;
}
QTextEdit:focus {
    border-color: #0d6efd;
    outline: none;
}
/* 主页面文本输入样式 */
QPlainTextEdit {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 20px;
    padding: 8px 18px;
    selection-background-color: #b2d3ff;
}
/* 通用按钮样式 */
QPushButton {
    background-color: #0d6efd;
    color: white;
    border: none;
    border-radius: 14px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #0b5ed7;
}
QPushButton:pressed {
    background-color: #0a58ca;
}
QPushButton:disabled {
    background-color: #b6c2d0;
    color: #eef2f6;
}
/* 特殊按钮颜色覆盖 */
QPushButton#stopBtn {
    background-color: #dc3545;
}
QPushButton#stopBtn:hover {
    background-color: #bb2d3b;
}
QPushButton#regenerateBtn {
    background-color: #6c757d;
}
QPushButton#regenerateBtn:hover {
    background-color: #5c636a;
}
/* 表格样式 */
QTableWidget {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 12px;
    gridline-color: #e9ecef;
    selection-background-color: #e9f0ff;
    selection-color: #0d6efd;
}
QTableWidget::item {
    padding: 8px;
    border-bottom: 1px solid #f1f3f5;
}
QHeaderView::section {
    background-color: #f8f9fa;
    padding: 8px;
    border: none;
    font-weight: bold;
}
/* 滚动条样式（扁平圆角） */
QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 8px;
    border-radius: 4px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #c1c9d2;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #a0aab5;
}
QScrollBar:horizontal {
    border: none;
    background: transparent;
    height: 8px;
    border-radius: 4px;
    margin: 0px;
}
QScrollBar::handle:horizontal {
    background: #c1c9d2;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #a0aab5;
}
/* 菜单栏 */
QMenuBar {
    background-color: white;
    border-bottom: 1px solid #eaeef2;
}
QMenuBar::item {
    padding: 8px 12px;
    background: transparent;
    border-radius: 6px;
}
QMenuBar::item:selected {
    background-color: #e9f0ff;
}
QMenu {
    background-color: white;
    border: 1px solid #dde1e6;
    border-radius: 8px;
    padding: 6px;
}
QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 6px;
}
QMenu::item:selected {
    background-color: #e9f0ff;
    color: #0d6efd;
}
/* ---------- 扁平化标签页 ---------- */
QTabWidget::pane {
    border: 1px solid #dde1e6;
    border-radius: 12px;
    background: white;
    top: -1px;  /* 让 pane 与 tab 条对齐 */
}

QTabBar::tab {
    background-color: #f1f5f9;
    border: 1px solid #dde1e6;
    border-bottom: none;  /* 避免双边框 */
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 20px;
    margin-right: 2px;
    font-size: 13px;
    font-weight: 500;
    color: #1e293b;
}

QTabBar::tab:selected {
    background-color: white;
    border-bottom: 1px solid white;  /* 与 pane 背景融合 */
    color: #0d6efd;
}

QTabBar::tab:hover:!selected {
    background-color: #e9f0ff;
}

QTabBar::tab:first {
    margin-left: 8px;  /* 左侧留空 */
}

QTabBar::tab:last {
    margin-right: 8px; /* 右侧留空 */
}

/* 如果标签页数量少，可以固定高度 */
QTabBar::tab {
    min-height: 36px;
}
QTreeWidget {
    background-color: transparent;
    border: none;
    outline: none;
}
QTreeWidget::item {
    padding: 6px;
    border-radius: 6px;
}
QTreeWidget::item:selected {
    background-color: #e9f0ff;
    color: #0d6efd;
}
QTreeWidget::item:hover {
    background-color: #f1f5f9;
}
QTreeWidget::branch {
    background: transparent;
}
QTreeWidget::branch:has-children:!has-siblings:closed,
QTreeWidget::branch:closed:has-children:has-siblings {
    border-image: none;
    image: none;
}
QTreeWidget::branch:open:has-children:!has-siblings,
QTreeWidget::branch:open:has-children:has-siblings {
    border-image: none;
    image: none;
}
"""
