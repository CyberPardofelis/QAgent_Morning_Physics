# ui/message_bubble.py
"""
消息气泡组件 — 渲染单条消息（Markdown + MathJax）。

v2 改动：
- MarkdownView 不再由气泡内部创建，改为外部通过 set_view() 注入
  （由 WebEnginePool 统一管理，限制同时活跃的 WebEngine 实例数）
- 新增 _replace_view() 供 WebEnginePool 冻结/解冻时使用
- 保留流式追加 update_content() + finalize()
- 保留自适应宽度测量链
"""

from __future__ import annotations
import os
import markdown
import json
import re
import html as html_module
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QUrl, QEvent
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QSizePolicy, QLayout, QLabel,
)
from PySide6.QtWebEngineWidgets import QWebEngineView


# ============================================================
# HTML 模板
# ============================================================
BASE_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style id="custom-style"></style>
<style>
html,body{
    margin:0; padding:0; background:transparent; overflow:hidden;
}
#content{
    background:transparent;
    display: inline-block;
    white-space: normal;
    word-wrap: break-word;
    box-sizing: border-box;
    max-width: 100%;
}
</style>
<script>
window.MathJax = {
    tex: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        displayMath: [['$$','$$'], ['\\[','\\]']]
    },
    options: { renderActions: { addMenu: [] } },
    svg: { fontCache: 'global' }
};
</script>
<script type="text/javascript" async src="mathjax/tex-svg.js"></script>
</head><body><div id="content"></div></body></html>"""


# ============================================================
# MarkdownView — 渲染 Markdown + MathJax，自适应宽高
# ============================================================
class MarkdownView(QWebEngineView):
    MAX_HEIGHT = 26000

    _preloaded_engine = None

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._auto_size: bool = True
        self._content_html: str = ""
        self._base_loaded: bool = False
        self._pending_markdown: str = ""
        self._html_style: str = ""
        self._max_width: int = 1800

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.setHtml(BASE_HTML_TEMPLATE, QUrl.fromLocalFile(base_dir + '/'))
        self.loadFinished.connect(self._on_base_loaded)

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMinimumWidth(10)

    # ---- 预加载引擎 ----
    @staticmethod
    def preload_engine(parent=None):
        if MarkdownView._preloaded_engine is not None:
            return
        view = QWebEngineView()
        view.setAttribute(Qt.WA_DontShowOnScreen, True)
        view.setVisible(False)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        view.setHtml(BASE_HTML_TEMPLATE, QUrl.fromLocalFile(base_dir + '/'))
        view.loadFinished.connect(lambda ok: setattr(view, '_loaded', True))
        MarkdownView._preloaded_engine = view

    # ---- 基础加载完成 ----
    def _on_base_loaded(self, ok: bool):
        if not ok:
            return
        self._base_loaded = True
        self.page().setBackgroundColor(Qt.transparent)
        if self._html_style:
            self._inject_style(self._html_style)
        if self._pending_markdown:
            self._update_content()
            self._max_width = 1800
            QTimer.singleShot(0, self._remeasure_all)

    def _remeasure_all(self):
        self.setFixedWidth(self._max_width)
        QTimer.singleShot(80, self._measure_natural_width)

    # ---- 样式注入 ----
    def _inject_style(self, css: str):
        css_escaped = json.dumps(css)
        js = f"document.getElementById('custom-style').textContent = {css_escaped};"
        self.page().runJavaScript(js)

    def setHtmlStyle(self, css: str):
        self._html_style = css
        if self._base_loaded:
            self._inject_style(css)

    # ---- 自动尺寸开关 ----
    def setAutoSize(self, auto: bool, max_width: int = None):
        self._auto_size = auto
        if max_width:
            self.setMaximumWidth(max_width)

    # ---- 设置 Markdown 内容 ----
    def setMarkdown(self, text: str):
        html_body = self._markdown_to_html(text) if text else ""
        self._content_html = html_body
        if self._base_loaded:
            self._update_content()
            QTimer.singleShot(0, lambda: self.adjust_to_max_width(self._max_width))
        else:
            self._pending_markdown = html_body

    # ---- 将 HTML 注入页面 ----
    def _update_content(self):
        html_escaped = json.dumps(self._content_html)
        js = f"""
        document.getElementById('content').innerHTML = {html_escaped};
        if (window.MathJax &amp;&amp; MathJax.typesetPromise) {{
            MathJax.typesetPromise().then(() => {{}});
        }}
        """
        self.page().runJavaScript(js)

    # ---- 自适应宽度 ----
    def adjust_to_max_width(self, max_w: int):
        self._max_width = max_w
        self.setFixedWidth(max_w)
        QTimer.singleShot(50, self._measure_natural_width)

    def _measure_natural_width(self):
        js = """
        (function() {
            var el = document.getElementById('content');
            var oldWidth = el.style.width;
            var oldMaxWidth = el.style.maxWidth;
            el.style.width = 'auto';
            el.style.maxWidth = 'none';
            var natural = el.offsetWidth;
            el.style.width = oldWidth;
            el.style.maxWidth = oldMaxWidth;
            return natural;
        })()
        """
        self.page().runJavaScript(js, self._on_natural_width_measured)

    def _on_natural_width_measured(self, width):
        try:
            if hasattr(width, 'toInt'):
                natural_w = width.toInt()[0] if width.toInt()[1] else int(width.toFloat()[0])
            else:
                natural_w = int(float(width))
        except (ValueError, TypeError, AttributeError):
            natural_w = 0

        if natural_w <= 0:
            natural_w = self._max_width

        final_w = max(80, min(natural_w, self._max_width))
        self.setFixedWidth(final_w)
        self.updateGeometry()
        QTimer.singleShot(30, self._remeasure_height)

    # ---- 高度自适应 ----
    def _remeasure_height(self):
        if not self._auto_size:
            return
        self.page().runJavaScript("document.body.scrollHeight", self._apply_height)

    def _apply_height(self, height):
        if not self._auto_size:
            return
        try:
            h = int(height)
        except (ValueError, TypeError):
            return
        if h > self.MAX_HEIGHT:
            self.page().runJavaScript("document.body.style.overflow = 'auto';")
            self.setFixedHeight(self.MAX_HEIGHT)
        else:
            self.page().runJavaScript("document.body.style.overflow = 'hidden';")
            if h != self.height():
                self.setFixedHeight(h)

    # ---- Markdown → HTML ----
    def _markdown_to_html(self, text: str) -> str:
        placeholders = []

        def replace_display(m):
            placeholders.append(m.group(0))
            return f"\x00MATHHD{len(placeholders)-1}\x00"

        def replace_inline(m):
            placeholders.append(m.group(0))
            return f"\x00MATHIN{len(placeholders)-1}\x00"

        text = re.sub(r'\$\$(.+?)\$\$', replace_display, text, flags=re.DOTALL)
        text = re.sub(r'\$(.+?)\$', replace_inline, text)
        text = html_module.escape(text, quote=False)

        def restore_display(m):
            idx = int(m.group(1))
            return placeholders[idx]

        def restore_inline(m):
            idx = int(m.group(1))
            return placeholders[idx]

        text = re.sub(r'\x00MATHHD(\d+)\x00', restore_display, text)
        text = re.sub(r'\x00MATHIN(\d+)\x00', restore_inline, text)

        html = markdown.markdown(text, extensions=['fenced_code', 'tables'])
        html = html.replace('\x00MATHHD', '<span class="math display">').replace('\x00', '</span>')
        html = html.replace('\x00MATHIN', '<span class="math inline">')
        return html


# ============================================================
# MessageBubble — 消息气泡
# ============================================================
class MessageBubble(QWidget):
    """
    消息气泡组件。

    v2 改动：
    - 不再在 __init__ 中创建 MarkdownView，改为通过 set_view() 注入。
    - 新增 _replace_view() 供 WebEnginePool 冻结/解冻时调用。
    - 流式追加 update_content() + finalize() 保持兼容。
    """

    def __init__(self, role: str, markdown_text: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.role = role
        self.full_text = markdown_text
        self.is_final = False

        # 流式更新缓冲
        self._line_buffer: str = ""
        self._complete_lines: list[str] = []
        self._in_math_block: bool = False

        # 流式更新定时器（防抖）
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(100)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._flush_complete_lines)

        # ── 颜色配置 ──
        if role == 'user':
            self.setStyleSheet("background-color: #77aaff; border-radius: 10px;")
            self._text_color = "white"
        elif role == 'assistant':
            self.setStyleSheet("background-color: white; border-radius: 10px;")
            self._text_color = "black"
        else:  # system
            self.setStyleSheet("background-color: #e9ecef; border-radius: 10px;")
            self._text_color = "#495057"

        # ── 布局（空壳，视图由 set_view 注入） ──
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        # 视图稍后通过 set_view() 添加

        # ── 尺寸策略 ──
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._max_view_width: int = 1800
        self._top_level: Optional[QWidget] = None

        # ── 引用当前渲染视图（MarkdownView 或 FrozenBubble） ──
        self.view: Optional[QWidget] = None

        # 延迟到事件循环启动后首次计算宽度
        QTimer.singleShot(0, self._delayed_update_max_width)

    # ================================================================
    # 视图注入（WebEnginePool 调用）
    # ================================================================

    def set_view(self, view: QWidget) -> None:
        """
        设置气泡的渲染视图。

        由 WebEnginePool.acquire() 调用，注入 MarkdownView 或 FrozenBubble。

        Args:
            view: MarkdownView 实例（活跃渲染）或 FrozenBubble 实例（冻结替身）。
        """
        # 移除旧视图（如果有）
        if self.view:
            self._layout.removeWidget(self.view)
            # 不 deleteLate — MarkdownView 由池管理，FrozenBubble 由池清理

        self.view = view

        # 添加到布局
        self._layout.insertWidget(0, view)
        view.setVisible(True)
        view.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # 应用样式
        if isinstance(view, MarkdownView):
            if self.role == 'system':
                text_css = (
                    f"body {{ color: {self._text_color}; text-align: center; }} "
                    f"h1,h2,h3,h4,h5,h6 {{ color: {self._text_color} !important; }}"
                )
            else:
                text_css = (
                    f"body {{ color: {self._text_color}; }} "
                    f"h1,h2,h3,h4,h5,h6 {{ color: {self._text_color} !important; }}"
                )
            view.setHtmlStyle(text_css)
            view.setMarkdown(self.full_text)

        # 触发宽度更新
        QTimer.singleShot(0, self._delayed_update_max_width)

    def _replace_view(self, new_widget: QWidget) -> None:
        """
        替换气泡内的渲染视图（供 WebEnginePool 冻结/解冻时调用）。

        Args:
            new_widget: FrozenBubble（冻结时）或 MarkdownView（解冻时）。
        """
        if self.view is new_widget:
            return

        # 移除旧视图
        if self.view:
            self._layout.removeWidget(self.view)
            old_widget = self.view
            # MarkdownView 不 deleteLate（由池回收）
            # FrozenBubble 可以 deleteLate
            if not isinstance(old_widget, MarkdownView):
                old_widget.deleteLater()

        self.view = new_widget

        # 插入新视图
        self._layout.insertWidget(0, new_widget)
        new_widget.setVisible(True)
        new_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # 如果是 MarkdownView，同步内容
        if isinstance(new_widget, MarkdownView):
            if self.role == 'system':
                text_css = (
                    f"body {{ color: {self._text_color}; text-align: center; }} "
                    f"h1,h2,h3,h4,h5,h6 {{ color: {self._text_color} !important; }}"
                )
            else:
                text_css = (
                    f"body {{ color: {self._text_color}; }} "
                    f"h1,h2,h3,h4,h5,h6 {{ color: {self._text_color} !important; }}"
                )
            new_widget.setHtmlStyle(text_css)
            new_widget.setMarkdown(self.full_text)

        QTimer.singleShot(0, self._delayed_update_max_width)

    # ================================================================
    # 事件 &amp; 自适应宽度
    # ================================================================

    def showEvent(self, event):
        super().showEvent(event)
        top = self.window()
        if top and top != self._top_level:
            if self._top_level:
                self._top_level.removeEventFilter(self)
            top.installEventFilter(self)
            self._top_level = top
        QTimer.singleShot(0, self.updateMaxWidth)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.updateMaxWidth)

    def eventFilter(self, watched, event):
        if watched == self._top_level and event.type() == QEvent.Resize:
            self.updateMaxWidth()
        return super().eventFilter(watched, event)

    def _delayed_update_max_width(self):
        """首次计算最大宽度。"""
        parent = self.parent()
        area_w = 0
        if parent:
            area_w = parent.width()
        if not parent or area_w <= 0:
            top = self.window()
            area_w = top.width() * 0.6 if top else 800
        max_w = max(200, int(area_w * 0.8))
        self._max_view_width = max_w
        if isinstance(self.view, MarkdownView):
            self.view.adjust_to_max_width(max_w)

    def updateMaxWidth(self):
        """
        根据父容器宽度重新计算最大宽度。
        仅当 max_w 实际变化时才调用 view.adjust_to_max_width。
        """
        parent = self.parent()
        area_w = 0
        if parent:
            area_w = parent.width()
        if not parent or area_w <= 0:
            top = self.window()
            area_w = top.width() * 0.6 if top else 800
        max_w = max(200, int(area_w * 0.8))
        if max_w != self._max_view_width:
            self._max_view_width = max_w
            if isinstance(self.view, MarkdownView):
                self.view.adjust_to_max_width(max_w)

    # ================================================================
    # 流式更新（LLM 输出边收边渲染）
    # ================================================================

    def update_content(self, new_text: str):
        """
        追加新文本。

        改进：满 200 字符或遇到换行符时刷新，不再死等换行符。
        防止 LLM 连续输出无换行长段落时长时间不更新。
        """
        if self.is_final:
            return

        self._line_buffer += new_text

        # 满 200 字符 或 遇到换行符 → 刷新
        if len(self._line_buffer) >= 200 or '\n' in self._line_buffer:
            parts = self._line_buffer.split('\n')
            self._complete_lines.extend(parts[:-1])
            self._line_buffer = parts[-1]
            self._flush_complete_lines()

    def _flush_complete_lines(self):
        """将缓冲的完整行刷入 Markdown 渲染。"""
        if not self._complete_lines or self.is_final:
            return

        chunk = ''.join(self._complete_lines)
        self._complete_lines.clear()
        safe_chunk = self._preprocess(chunk)
        self.full_text += safe_chunk

        if isinstance(self.view, MarkdownView):
            self.view.setMarkdown(self.full_text)

        self.update_timer.stop()

    def _preprocess(self, text: str) -> str:
        """预处理：追踪数学公式块状态，标题后自动加空行。"""
        lines = text.split('\n')
        processed_lines = []
        in_math = self._in_math_block

        for line in lines:
            new_line = []
            i = 0
            while i < len(line):
                if line[i] == '$' and i + 1 < len(line) and line[i + 1] == '$':
                    new_line.append('$$')
                    in_math = not in_math
                    i += 2
                elif in_math and line[i] == '':
                    i += 1
                else:
                    new_line.append(line[i])
                    i += 1
            processed_lines.append(''.join(new_line))

        self._in_math_block = in_math

        # 标题后添加空行
        result = []
        for idx, line in enumerate(processed_lines):
            result.append(line)
            if line.lstrip().startswith('#'):
                if idx < len(processed_lines) - 1:
                    next_line = processed_lines[idx + 1]
                    if next_line.strip() and not next_line.lstrip().startswith('#'):
                        result.append('')
        return '\n'.join(result)

    def finalize(self):
        """
        完成流式更新。

        清空缓冲，重新渲染完整文本（确保 MathJax 正确处理所有公式）。
        """
        if self.is_final:
            return

        self.is_final = True

        if self._line_buffer:
            self._complete_lines.append(self._line_buffer + '\n')
            self._line_buffer = ""

        if self._complete_lines:
            self._flush_complete_lines()

        # 重新渲染完整文本（确保 MathJax 正确处理跨行公式）
        if isinstance(self.view, MarkdownView):
            self.view.setMarkdown("")
            self.view.setMarkdown(self.full_text)

    # ================================================================
    # 析构
    # ================================================================

    def __del__(self):
        self.update_timer.stop()


# ================================================================
# 缺少的 import（文件顶部补充）
# ================================================================
