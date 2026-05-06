import json
import re
import markdown
import os
import html as html_module
from PySide6.QtCore import Qt, QTimer, QUrl, QEvent
from PySide6.QtWidgets import QWidget, QHBoxLayout, QSizePolicy, QLayout
from PySide6.QtWebEngineWidgets import QWebEngineView

class MarkdownView(QWebEngineView):
    MAX_HEIGHT = 26000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._auto_size = True
        self._content_html = ""
        self._base_loaded = False
        self._pending_markdown = ""
        self._html_style = ""
        self._max_width = 1800   # 默认最大宽度

        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style id="custom-style"></style>
<style>html,body{margin:0;padding:0;background:transparent;overflow:hidden;word-wrap:break-word;}#content{background:transparent;}</style>
<script>
window.MathJax = {
    tex: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        displayMath: [['$$','$$'], ['\\[','\\]']]
    },
    options: {
        renderActions: {
            addMenu: []
        }
    },
    svg: {
        fontCache: 'global'
    }
};
</script>
<script type="text/javascript" async src="mathjax/tex-svg.js"></script>
</head><body><div id="content"></div></body></html>"""
        self.setHtml(base_html, QUrl.fromLocalFile(base_dir + '/'))
        self.loadFinished.connect(self._on_base_loaded)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMinimumWidth(10)

    def _on_base_loaded(self, ok):
        self._base_loaded = True
        self.page().setBackgroundColor(Qt.transparent)
        if self._html_style:
            self._inject_style(self._html_style)
        if self._pending_markdown:
            self._update_content()

    def _inject_style(self, css: str):
        css_escaped = json.dumps(css)
        js = f"document.getElementById('custom-style').textContent = {css_escaped};"
        self.page().runJavaScript(js)

    def setHtmlStyle(self, css: str):
        self._html_style = css
        if self._base_loaded:
            self._inject_style(css)

    def setAutoSize(self, auto: bool, max_width: int = None):
        self._auto_size = auto
        if max_width:
            self.setMaximumWidth(max_width)

    def setMarkdown(self, text: str):
        html_body = self._markdown_to_html(text)
        self._content_html = html_body
        if self._base_loaded:
            self._update_content()
        else:
            self._pending_markdown = html_body

    def _update_content(self):
        html_escaped = json.dumps(self._content_html)
        js = f"""
        document.getElementById('content').innerHTML = {html_escaped};
        if (window.MathJax && MathJax.typesetPromise) {{
            MathJax.typesetPromise().then(() => {{}});
        }}
        """
        self.page().runJavaScript(js)
        # 延长等待时间，给 MathJax 充分的渲染时间
        QTimer.singleShot(350, lambda: self.adjust_to_max_width(self._max_width))

    def adjust_to_max_width(self, max_w: int):
        """设置最大宽度限制，并重新测量内容尺寸"""
        if max_w != self._max_width:
            self.setMaximumWidth(max_w)
            self.setMinimumWidth(0)
        self._max_width = max_w
        # 临时设为最大宽度以测量真实宽度
        self.setFixedWidth(max_w)
        self.page().runJavaScript("document.body.scrollHeight", self._apply_height)

    def _apply_height(self, height):
        if not self._auto_size:
            return
        try:
            h = int(height)
        except (ValueError, TypeError):
            return
        # 超高时启用滚动
        if h > self.MAX_HEIGHT:
            self.page().runJavaScript("document.body.style.overflow = 'auto';")
            self.setFixedHeight(self.MAX_HEIGHT)
        else:
            self.page().runJavaScript("document.body.style.overflow = 'hidden';")
            # 只有高度确实变化才更新，避免循环
            if h != self.height():
                self.setFixedHeight(h)
        # 接着测量内容宽度
        self.page().runJavaScript(
            "document.getElementById('content').scrollWidth",
            self._apply_width
        )

    def _apply_width(self, width):
        try:
            w = int(width)
        except (ValueError, TypeError):
            w = 80
        max_w = self._max_width
        final_w = max(80, min(w, max_w))
        self.setFixedWidth(final_w)
        # 宽度变化可能导致高度改变（换行变化），延迟后重新测量高度
        QTimer.singleShot(100, self._remeasure_height)
        self.updateGeometry()

    def _remeasure_height(self):
        """宽度锁定后重新测量高度，确保高度与最终宽度匹配"""
        if not self._auto_size:
            return
        self.page().runJavaScript("document.body.scrollHeight", self._apply_height)

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


class MessageBubble(QWidget):
    def __init__(self, role: str, markdown_text: str, parent=None):
        super().__init__(parent)
        self.role = role
        self.full_text = markdown_text
        self.is_final = False
        self._line_buffer = ""
        self._complete_lines = []
        self.update_timer = QTimer()
        self.update_timer.setInterval(100)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._flush_complete_lines)
        self._in_math_block = False

        # 气泡背景与文本颜色
        if role == 'user':
            self.setStyleSheet("background-color: #77aaff; border-radius: 10px;")
            text_color = "white"
        elif role == 'assistant':
            self.setStyleSheet("background-color: white; border-radius: 10px;")
            text_color = "black"
        else:  # system
            self.setStyleSheet("background-color: #e9ecef; border-radius: 10px;")
            text_color = "#495057"

        # 创建 MarkdownView
        self.view = MarkdownView()
        self.view.setAutoSize(True)

        # 注入文本颜色
        if role == 'system':
            text_css = f"body {{ color: {text_color}; text-align: center; }} h1,h2,h3,h4,h5,h6 {{ color: {text_color} !important; }}"
        else:
            text_css = f"body {{ color: {text_color}; }} h1,h2,h3,h4,h5,h6 {{ color: {text_color} !important; }}"
        self.view.setHtmlStyle(text_css)
        self.view.setMarkdown(markdown_text)

        # 布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.view)

        # 关键：气泡尺寸完全由布局计算，自动跟随内容变化
        layout.setSizeConstraint(QLayout.SetFixedSize)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._max_view_width = 1800

        # 监听顶层窗口大小变化，保证对话区域缩放时气泡也更新
        self._top_level = None
        self._install_resize_filter()

        # 首次计算宽度限制并开始测量内容
        self.updateMaxWidth()

    def _install_resize_filter(self):
        """为顶层窗口安装事件过滤器，监听大小变化"""
        # 通过 parent 逐级找到顶层窗口，不一定在 __init__ 时已经有 parent
        # 所以放到 showEvent 或直接定时检测
        self._top_level = self.window()
        if self._top_level:
            self._top_level.installEventFilter(self)

    def showEvent(self, event):
        super().showEvent(event)
        # 确保顶层窗口的过滤器已安装
        if not self._top_level or self._top_level != self.window():
            self._top_level = self.window()
            if self._top_level:
                self._top_level.installEventFilter(self)
        # 显示时也更新一次宽度限制
        self.updateMaxWidth()

    def eventFilter(self, watched, event):
        # 顶层窗口大小变化时，重新计算80%限制
        if watched == self._top_level and event.type() == QEvent.Resize:
            self.updateMaxWidth()
        return super().eventFilter(watched, event)

    def updateMaxWidth(self):
        parent = self.parent()
        if parent:
            area_w = parent.width()
            max_w = max(200, int(area_w * 0.8))
        else:
            max_w = 1800
        if max_w != getattr(self, '_max_view_width', 0):
            self._max_view_width = max_w
            self.view.adjust_to_max_width(max_w)

    def update_content(self, new_text: str):
        if self.is_final:
            return
        for char in new_text:
            self._line_buffer += char
            if char == '\n':
                self._complete_lines.append(self._line_buffer)
                self._line_buffer = ""
        if self._complete_lines:
            self._flush_complete_lines()

    def _flush_complete_lines(self):
        if not self._complete_lines or self.is_final:
            return
        chunk = ''.join(self._complete_lines)
        self._complete_lines.clear()
        safe_chunk = self._preprocess(chunk)
        self.full_text += safe_chunk
        self.view.setMarkdown(self.full_text)
        # setMarkdown 会异步测量并调用 updateGeometry，无需额外操作
        self.update_timer.stop()

    def _preprocess(self, text: str) -> str:
        lines = text.split('\n')
        processed_lines = []
        in_math = self._in_math_block
        for line in lines:
            new_line = []
            i = 0
            while i < len(line):
                if line[i] == '$' and i+1 < len(line) and line[i+1] == '$':
                    new_line.append('$$')
                    in_math = not in_math
                    i += 2
                elif in_math and line[i] == '$':
                    i += 1
                else:
                    new_line.append(line[i])
                    i += 1
            processed_lines.append(''.join(new_line))
        self._in_math_block = in_math
        result = []
        for idx, line in enumerate(processed_lines):
            result.append(line)
            if line.lstrip().startswith('#'):
                if idx < len(processed_lines)-1:
                    next_line = processed_lines[idx+1]
                    if next_line.strip() and not next_line.lstrip().startswith('#'):
                        result.append('')
        return '\n'.join(result)

    def finalize(self):
        if self.is_final:
            return
        self.is_final = True
        if self._line_buffer:
            self._complete_lines.append(self._line_buffer + '\n')
            self._line_buffer = ""
        if self._complete_lines:
            self._flush_complete_lines()
        self.view.setMarkdown("")
        self.view.setMarkdown(self.full_text)