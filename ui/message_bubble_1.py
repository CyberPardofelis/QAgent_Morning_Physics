import json
import re
import markdown
import os
import html as html_module
from PySide6.QtCore import Qt, QTimer, QUrl, QEvent
from PySide6.QtWidgets import QWidget, QHBoxLayout, QSizePolicy, QLayout
from PySide6.QtWebEngineWidgets import QWebEngineView


# ============================================================
# HTML 模板（提取为常量，避免重复）
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._auto_size = True
        self._content_html = ""
        self._base_loaded = False
        self._pending_markdown = ""
        self._html_style = ""
        self._max_width = 2500    # 当前允许的最大宽度

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.setHtml(BASE_HTML_TEMPLATE, QUrl.fromLocalFile(base_dir + '/'))
        self.loadFinished.connect(self._on_base_loaded)

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMinimumWidth(10)

    # ---- 预加载引擎（静态方法） ----
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

    # ---- 基础加载完成回调 ----
    def _on_base_loaded(self, ok):
        if not ok:
            return
        self._base_loaded = True
        self.page().setBackgroundColor(Qt.transparent)
        if self._html_style:
            self._inject_style(self._html_style)
        if self._pending_markdown:
            self._update_content()
            # ★ 修复1：不再使用 adjust_to_max_width（可能提前返回）
            # 直接触发完整的测量流程：先展开到最大宽度，再测量自然宽度
            self._max_width = 2500  # 重置为默认
            QTimer.singleShot(0, self._remeasure_all)

    def _remeasure_all(self):
        """完整测量流程：展开→等待渲染→测量宽高"""
        # 先展开视图让内容重排
        self.setFixedWidth(self._max_width)
        # 给 MathJax 和 DOM 重排时间
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
        html_body = self._markdown_to_html(text)
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
        if (window.MathJax && MathJax.typesetPromise) {{
            MathJax.typesetPromise().then(() => {{}});
        }}
        """
        self.page().runJavaScript(js)

    # ================================================================
    # ★ 修复后的自适应宽度方法（合并第一版和第二版的优点）
    # ================================================================
    def adjust_to_max_width(self, max_w: int):
        """★ 修复2：移除提前返回逻辑，确保总是触发测量"""
        self._max_width = max_w
        self.setFixedWidth(max_w)
        QTimer.singleShot(50, self._measure_natural_width)

    def _delayed_update_max_width(self):
        if not self.view._base_loaded:
            # ★ 修复3：页面未就绪时重试
            QTimer.singleShot(100, self._delayed_update_max_width)
            return

    def _measure_natural_width(self):
        """测量内容的自然宽度（不受父容器约束）"""
        js = """
        (function() {
            var el = document.getElementById('content');
            var oldWidth = el.style.width;
            var oldMaxWidth = el.style.maxWidth;
            // 临时移除宽度约束，让元素按内容自然伸展
            el.style.width = 'auto';
            el.style.maxWidth = 'none';
            // 强制同步重排后读取宽度
            var natural = el.offsetWidth;  // offsetWidth 触发重排
            // 恢复原样式
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
            # 测量失败时使用最大宽度
            natural_w = self._max_width

        # 最终宽度：不低于 80px，不超过 _max_width
        final_w = max(80, min(natural_w, self._max_width))
        self.setFixedWidth(final_w)
        self.updateGeometry()

        # 宽度稳定后重新测量高度
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

    # ---- Markdown → HTML 转换 ----
    def _markdown_to_html(self, text: str) -> str:
        placeholders = []

        def replace_display(m):
            placeholders.append(m.group(0))
            return f"\x00MATHHD{len(placeholders)-1}\x00"

        def replace_inline(m):
            placeholders.append(m.group(0))
            return f"\x00MATHIN{len(placeholders)-1}\x00"

        # 保护 LaTeX 数学公式不被 HTML escape 破坏
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
# MessageBubble — 消息气泡（渲染 Markdown 内容）
# ============================================================
class MessageBubble(QWidget):
    def __init__(self, role: str, markdown_text: str, parent=None):
        super().__init__(parent)
        self.role = role
        self.full_text = markdown_text
        self.is_final = False
        self._line_buffer = ""
        self._complete_lines = []
        self._in_math_block = False

        # 流式更新定时器（防抖）
        self.update_timer = QTimer()
        self.update_timer.setInterval(100)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._flush_complete_lines)

        # 气泡背景色 & 文字颜色
        if role == 'user':
            self.setStyleSheet("background-color: #77aaff; border-radius: 10px;")
            text_color = "white"
        elif role == 'assistant':
            self.setStyleSheet("background-color: white; border-radius: 10px;")
            text_color = "black"
        else:
            self.setStyleSheet("background-color: #e9ecef; border-radius: 10px;")
            text_color = "#495057"

        # Markdown 渲染视图
        self.view = MarkdownView()
        self.view.setAutoSize(True)

        if role == 'system':
            text_css = f"body {{ color: {text_color}; text-align: center; }} h1,h2,h3,h4,h5,h6 {{ color: {text_color} !important; }}"
        else:
            text_css = f"body {{ color: {text_color}; }} h1,h2,h3,h4,h5,h6 {{ color: {text_color} !important; }}"
        self.view.setHtmlStyle(text_css)
        self.view.setMarkdown(markdown_text)

        # 布局 — ★ 修复：移除 SetFixedSize，改用默认约束
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.view)
        # 移除 layout.setSizeConstraint(QLayout.SetFixedSize)
        # 改用默认行为，让父布局可以自由调整气泡大小

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._max_view_width = 1800
        self._top_level = None  # 用于事件过滤器的顶层窗口

        # 延迟到事件循环启动后首次计算宽度
        QTimer.singleShot(0, self._delayed_update_max_width)

    # ---- 显示事件：安装父窗口事件过滤器 ----
    def showEvent(self, event):
        super().showEvent(event)
        # ★ 修复：安装顶层窗口的事件过滤器，捕获所有 Resize 事件
        top = self.window()
        if top and top != self._top_level:
            if self._top_level:
                self._top_level.removeEventFilter(self)
            top.installEventFilter(self)
            self._top_level = top
        QTimer.singleShot(0, self.updateMaxWidth)

    # ---- 尺寸变化事件 ----
    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.updateMaxWidth)

    # ---- ★ 修复：事件过滤器（捕获顶层窗口 Resize） ----
    def eventFilter(self, watched, event):
        if watched == self._top_level and event.type() == QEvent.Resize:
            self.updateMaxWidth()
        return super().eventFilter(watched, event)

    # ---- 首次延迟更新（用于 __init__ 后） ----
    def _delayed_update_max_width(self):
        """首次计算最大宽度（无条件执行）"""
        parent = self.parent()
        if parent:
            area_w = parent.width()
        if not parent or area_w <= 0:
            top = self.window()
            area_w = top.width() * 0.6 if top else 800
        max_w = max(200, int(area_w * 0.8))
        self._max_view_width = max_w
        self.view.adjust_to_max_width(max_w)

    # ---- 更新最大宽度（带变化检测） ----
    def updateMaxWidth(self):
        """
        根据父容器宽度重新计算最大宽度。
        仅当 max_w 实际变化时才调用 view.adjust_to_max_width。
        """
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

    # ---- 流式更新（追加新文本） ----
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
        """预处理文本：处理数学公式跨行、标题后自动加空行"""
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
                elif in_math and line[i] == '$':
                    i += 1
                else:
                    new_line.append(line[i])
                    i += 1
            processed_lines.append(''.join(new_line))
        self._in_math_block = in_math

        # 标题后添加空行（确保 Markdown 正确解析）
        result = []
        for idx, line in enumerate(processed_lines):
            result.append(line)
            if line.lstrip().startswith('#'):
                if idx < len(processed_lines) - 1:
                    next_line = processed_lines[idx + 1]
                    if next_line.strip() and not next_line.lstrip().startswith('#'):
                        result.append('')
        return '\n'.join(result)

    # ---- 完成流式更新 ----
    def finalize(self):
        if self.is_final:
            return
        self.is_final = True
        if self._line_buffer:
            self._complete_lines.append(self._line_buffer + '\n')
            self._line_buffer = ""
        if self._complete_lines:
            self._flush_complete_lines()
        # 重新渲染完整文本（确保 MathJax 正确处理）
        self.view.setMarkdown("")
        self.view.setMarkdown(self.full_text)
        # ★ 修复：finalize 中不再有死代码
        QTimer.singleShot(1000, self.updateMaxWidth)
        QTimer.singleShot(2000, self.updateMaxWidth)
        QTimer.singleShot(4000, self.updateMaxWidth)