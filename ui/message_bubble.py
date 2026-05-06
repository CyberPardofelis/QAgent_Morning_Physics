import json
import re
import markdown
import os
import html as html_module
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtWidgets import QWidget, QHBoxLayout, QSizePolicy, QLayout
from PySide6.QtWebEngineWidgets import QWebEngineView


class MarkdownView(QWebEngineView):
    MAX_HEIGHT = 26000

    # 预加载用的静态变量
    _preloaded_engine = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._auto_size = True
        self._content_html = ""
        self._base_loaded = False
        self._pending_markdown = ""
        self._html_style = ""
        self._max_width = 1800

        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 基础 HTML 模板（外链 MathJax，稳定且可缓存）
        base_html = """<!DOCTYPE html>
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

    @staticmethod
    def preload_engine(parent=None):
        """在应用启动时调用：预加载一个隐藏的页面，预热浏览器引擎与 MathJax"""
        if MarkdownView._preloaded_engine is not None:
            return  # 已经预加载过
        view = QWebEngineView()
        view.setAttribute(Qt.WA_DontShowOnScreen, True)  # 完全隐藏
        view.setVisible(False)
        # 与正式视图使用相同的基础页面，确保 MathJax 被加载并缓存
        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_html = """<!DOCTYPE html>
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
</head><body><div id="content"> </div></body></html>"""
        view.setHtml(base_html, QUrl.fromLocalFile(base_dir + '/'))
        # 保存实例防止被回收，等待加载完成
        view.loadFinished.connect(lambda ok: setattr(view, '_loaded', True))
        MarkdownView._preloaded_engine = view

    def _on_base_loaded(self, ok):
        if not ok:
            return
        self._base_loaded = True
        self.page().setBackgroundColor(Qt.transparent)
        if self._html_style:
            self._inject_style(self._html_style)
        if self._pending_markdown:
            self._update_content()
            QTimer.singleShot(0, lambda: self.adjust_to_max_width(self._max_width))

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
            QTimer.singleShot(0, lambda: self.adjust_to_max_width(self._max_width))
        else:
            self._pending_markdown = html_body

    def _update_content(self):
        # 安全注入，json.dumps 保证任何 HTML 特殊字符都不会破坏 DOM
        html_escaped = json.dumps(self._content_html)
        js = f"""
        document.getElementById('content').innerHTML = {html_escaped};
        if (window.MathJax && MathJax.typesetPromise) {{
            MathJax.typesetPromise().then(() => {{}});
        }}
        """
        self.page().runJavaScript(js)

    def adjust_to_max_width(self, max_w: int):
        self._max_width = max_w
        self.setFixedWidth(max_w)
        QTimer.singleShot(0, self._measure_content_width)

    def _measure_content_width(self):
        js = """
        (function() {
            var el = document.getElementById('content');
            return parseFloat(el.offsetWidth) || parseFloat(el.scrollWidth) || 0;
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
        QTimer.singleShot(30, self._remeasure_height)   # 30ms 足够重排且肉眼看不出延迟

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

        # 气泡样式
        if role == 'user':
            self.setStyleSheet("background-color: #77aaff; border-radius: 10px;")
            text_color = "white"
        elif role == 'assistant':
            self.setStyleSheet("background-color: white; border-radius: 10px;")
            text_color = "black"
        else:
            self.setStyleSheet("background-color: #e9ecef; border-radius: 10px;")
            text_color = "#495057"

        self.view = MarkdownView()
        self.view.setAutoSize(True)

        if role == 'system':
            text_css = f"body {{ color: {text_color}; text-align: center; }} h1,h2,h3,h4,h5,h6 {{ color: {text_color} !important; }}"
        else:
            text_css = f"body {{ color: {text_color}; }} h1,h2,h3,h4,h5,h6 {{ color: {text_color} !important; }}"
        self.view.setHtmlStyle(text_css)
        self.view.setMarkdown(markdown_text)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.view)
        layout.setSizeConstraint(QLayout.SetFixedSize)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._max_view_width = 1800
        QTimer.singleShot(0, self._delayed_update_max_width)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.updateMaxWidth)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.updateMaxWidth)

    def _delayed_update_max_width(self):
        parent = self.parent()
        if parent:
            area_w = parent.width()
        if not parent or area_w <= 0:
            top = self.window()
            area_w = top.width() * 0.6 if top else 800
        max_w = max(200, int(area_w * 0.8))
        self._max_view_width = max_w
        self.view.adjust_to_max_width(max_w)

    def updateMaxWidth(self):
        parent = self.parent()
        if parent:
            area_w = parent.width()
        if not parent or area_w <= 0:
            top = self.window()
            area_w = top.width() * 0.6 if top else 800
        max_w = max(200, int(area_w * 0.8))
        if max_w != self._max_view_width:
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