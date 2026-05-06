import json
import re
import markdown
import os
import html as html_module
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtWidgets import QWidget, QHBoxLayout, QSizePolicy
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

    def _on_base_loaded(self, ok):
        if ok:
            self._base_loaded = True
            self.page().setBackgroundColor(Qt.transparent)
            # 注入之前缓存的样式
            if self._html_style:
                self._inject_style(self._html_style)
            if self._pending_markdown:
                self._update_content()

    # MarkdownView._on_base_loaded 测试修改
    def _on_base_loaded(self, ok):
        self._base_loaded = True  # 无论成败都认为基础已加载
        self.page().setBackgroundColor(Qt.transparent)
        if self._html_style:
            self._inject_style(self._html_style)
        if self._pending_markdown:
            self._update_content()

    def _inject_style(self, css: str):
        """通过 JS 设置 custom-style 的内容"""
        css_escaped = json.dumps(css)
        js = f"document.getElementById('custom-style').textContent = {css_escaped};"
        self.page().runJavaScript(js)

    def setHtmlStyle(self, css: str):
        """外部注入自定义 CSS 文本"""
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
        QTimer.singleShot(200, lambda: self.adjust_to_max_width(self._max_width))

    '''

    def _adjust_size(self):
        # 先获取高度，设置完后异步获取宽度并设置
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
            self.setFixedHeight(h)

        # 高度设置完成后，获取内容实际宽度（包含 padding）
        self.page().runJavaScript("document.body.scrollWidth", self._apply_width)

    def _apply_width(self, width):
        try:
            w = int(width)
        except (ValueError, TypeError):
            return
        # 计算最大允许宽度：父控件宽度的80%，至少200px
        parent = self.parent()
        if parent:
            max_w = max(200, int(parent.width() * 0.8))
        else:
            max_w = 800
        final_w = max(80, min(w, max_w))
        self.setFixedWidth(final_w)

    # '''

    def adjust_to_max_width(self, max_w: int):
        """外部调用：告知视图当前允许的最大宽度，并重新测量尺寸"""
        self._max_width = max_w
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
            self.setFixedHeight(h)

        # 宽度用 #content 测量，才能真正反映内容宽度
        self.page().runJavaScript(
            "document.getElementById('content').scrollWidth",
            self._apply_width
        )

    def _apply_width(self, width):
        try:
            w = int(width)
        except (ValueError, TypeError):
            return
        max_w = getattr(self, '_max_width', 1800)
        final_w = max(80, min(w, max_w))
        self.setFixedWidth(final_w)

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
            alignment = 'right'
        elif role == 'assistant':
            self.setStyleSheet("background-color: white; border-radius: 10px;")
            text_color = "black"
            alignment = 'left'
        else:  # system
            self.setStyleSheet("background-color: #e9ecef; border-radius: 10px;")
            text_color = "#495057"
            alignment = 'center'

        # 创建 MarkdownView
        self.view = MarkdownView()
        self.view.setAutoSize(True)

        # 注入文本颜色（标题继承）
        # 注入文本颜色，系统角色额外添加居中
        if role == 'system':
            text_css = f"body {{ color: {text_color}; text-align: center; }} h1,h2,h3,h4,h5,h6 {{ color: {text_color} !important; }}"
        else:
            text_css = f"body {{ color: {text_color}; }} h1,h2,h3,h4,h5,h6 {{ color: {text_color} !important; }}"
        self.view.setHtmlStyle(text_css)
        self.view.setMarkdown(markdown_text)

        # 布局与对齐
        # 布局：直接放入 view，不再用 stretch 控制内部对齐
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.view)

        # 尺寸策略：允许气泡随内容缩小，不水平拉伸
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setAttribute(Qt.WA_StyledBackground, True)  # 确保背景色显示

        self._max_view_width = 1800
        # 首次调整大小
        self.updateMaxWidth()  # 直接调用公共方法

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_view_max_width()

    def _update_view_max_width(self):
        # 获取对话区域宽度（父控件 scroll_content 的宽度）
        parent = self.parent()
        if parent:
            area_w = parent.width()
            max_w = max(200, int(area_w * 0.8))
        else:
            max_w = 1800
        if max_w != self._max_view_width:
            self._max_view_width = max_w
            self.view.adjust_to_max_width(max_w)
            QTimer.singleShot(0, self._bubble_adjust)

    def _bubble_adjust(self):
        self.adjustSize()
        self.updateGeometry()

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
            QTimer.singleShot(0, self._bubble_adjust)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateMaxWidth()   # 改用公共方法

    # 修改 _flush_complete_lines 方法
    def _flush_complete_lines(self):
        if not self._complete_lines or self.is_final:
            return
        chunk = ''.join(self._complete_lines)
        self._complete_lines.clear()
        safe_chunk = self._preprocess(chunk)
        self.full_text += safe_chunk
        self.view.setMarkdown(self.full_text)
        self.updateMaxWidth()   # 新增：更新宽度
        self.update_timer.stop()

    # 修改 finalize 方法
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
        self.updateMaxWidth()   # 新增：最终内容渲染后调整宽度