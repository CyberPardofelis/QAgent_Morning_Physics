"""
MessageBubble + MarkdownView（v3 — 稳定渲染版）

修复历史问题：
1. 流式过程中反复调整宽度 → "闪来闪去最后才落到适合的大小"
2. MathJax typeset 触发布局重排 → 又触发新一轮测量
3. showEvent / resizeEvent / 父窗口 resize 多个事件链重复触发 updateMaxWidth
4. update_content 累积整段再 setMarkdown，O(n²) 性能

设计目标（保留 QWebEngineView 以支持 MathJax 公式渲染）：
- 流式过程：宽度锁住（用容器宽度的 75% 一次设置，不重测自然宽度）
- finalize 时：才 typeset 一次（避免流式过程多次 typeset 引起的重排）
- MathJax typeset：自己写 setTimeout-based 合并，避免每 chunk 都触发
- resize 事件：防抖 80ms，只调宽度不重排
- showEvent：只触发一次

向后兼容：
- MessageBubble(role, markdown_text, parent=...) 签名不变
- update_content(new_text) / finalize() / updateMaxWidth() / adjustSize() 不变
- find_last_assistant_bubble() 通过 widget.role 检查（self.role 已保留）
"""
import json
import re
import markdown
import os
import html as html_module
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, QEvent, QUrl, QSize
from PySide6.QtGui import QTextOption, QDesktopServices
from PySide6.QtWidgets import QWidget, QHBoxLayout, QSizePolicy
from PySide6.QtWebEngineWidgets import QWebEngineView


# ============================================================
# HTML 模板（v4：放到外部文件，让 Chromium HTTP 缓存）
# ============================================================
# 模板文件路径（相对 message_bubble_1.py 同目录的 resources/）
_BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_BUBBLE_TEMPLATE_PATH = _BASE_DIR / "resources" / "bubble_template.html"
_BUBBLE_TEMPLATE_URL = QUrl.fromLocalFile(str(_BUBBLE_TEMPLATE_PATH))


# ============================================================
# MarkdownView — 渲染 Markdown + MathJax（稳定版）
# ============================================================



# ============================================================
# MarkdownView — 渲染 Markdown + MathJax（稳定版）
# ============================================================
class MarkdownView(QWebEngineView):
    MAX_HEIGHT = 26000
    MIN_WIDTH = 80
    # 圆角视觉缓冲（垂直）：view 高度上额外加的像素，气泡最终高 = view_h + 2*MARGIN_V
    BUBBLE_V_PADDING = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._auto_size = True
        self._content_html = ""
        self._base_loaded = False
        self._pending_markdown = ""
        self._html_style = ""
        self._max_width = 0
        self._is_streaming = False  # 流式过程中不 typeset

        # A2+C 流式增量渲染状态
        self._stream_rendered_text = ""
        self._stream_pending = ""

        # v4：改用本地 file:// URL，让 Chromium HTTP 缓存复用 BASE 模板
        # （之前 setHtml(..., data:URL) 每个 view 都要重新 parse + 重新加载 MathJax，
        #  导致切对话时 5-10 个 view 串行启动 → 3 秒白屏）
        self.load(_BUBBLE_TEMPLATE_URL)
        self.loadFinished.connect(self._on_base_loaded)

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        # 关键：QWebEngineView 默认 sizeHint 在 PySide6 下不可靠（返回 0×0）
        # 用 Expanding 让 layout 给 view 完整空间（fill bubble 内容区）
        # 不依赖 sizeHint
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # 不设 setMinimumWidth（让它 fill layout 给的空间）
        # 高度 setMinimumHeight(40) 兜底，避免初始高度为 0
        self.setMinimumHeight(40)

        # 高度变化时通过 JS 回调通知（防抖 200ms，在 MathJax typeset 完成后）
        self._height_callback_pending = False
        self._last_reported_height = 0
        self._last_reported_width = 0
        # 自然宽度（finalize 后由 JS 测得）
        self._natural_width = 0
        # 防止自然宽度测量并发 / 自动重试
        self._natural_width_measuring = False
        self._natural_width_retry_count = 0

    def _on_base_loaded(self, ok):
        if not ok:
            return
        self._base_loaded = True
        self.page().setBackgroundColor(Qt.transparent)
        if self._html_style:
            self._inject_style(self._html_style)
        if self._pending_markdown:
            self._update_content()
        # load 完后立即测一次高度（避免初始 0px）
        QTimer.singleShot(100, self.reportHeight)

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
            self._max_width = max_width
            self.setMaximumWidth(max_width)

    def setMarkdown(self, text: str):
        """
        设置 Markdown。
        注意：不会重置 _is_streaming，调用方需自己管（如 endStreaming）。
        """
        html_body = self._markdown_to_html(text)
        # 内容未变 → 跳过（避免重复 setHtml）
        if html_body == self._content_html:
            return
        self._content_html = html_body
        if self._base_loaded:
            self._update_content()
        else:
            self._pending_markdown = html_body

    def beginStreaming(self):
        """标记进入流式状态（流式追加不 typeset，节省大量计算）"""
        self._is_streaming = True

    def endStreaming(self):
        """
        结束流式：触发一次 typeset 让 MathJax 渲染公式，然后报告高度。
        调一次即可，不要每 chunk 调。
        """
        self._is_streaming = False
        # 触发 typeset（不立即，setTimeout 300ms 合并）
        self.page().runJavaScript("window.qgTypeset && window.qgTypeset();")
        # 重新 setHtml（带 typeset 触发）
        if self._base_loaded:
            self._update_content()

    def _update_content(self):
        html_escaped = json.dumps(self._content_html)
        # setHtml 后 qgTypeset 由 endStreaming 统一触发
        # 但 base_loaded 第一次 setHtml 时需要立即 typeset
        js = f"""
        document.getElementById('content').innerHTML = {html_escaped};
        if ({str(self._is_streaming).lower()}) {{
            // 流式中：等待 endStreaming 统一 typeset
        }} else if (window.MathJax && MathJax.typesetPromise) {{
            window.qgTypeset && window.qgTypeset();
        }}
        """
        self.page().runJavaScript(js)

    def setMaxBubbleWidth(self, max_w: int):
        """
        设置气泡最大宽度（容器宽度的 80%）。
        - 总是 setMaximumWidth 作为上限
        - 通知 parent bubble，由 bubble 决定 setFixedWidth
        """
        if max_w == self._max_width:
            return
        self._max_width = max_w
        self.setMaximumWidth(max_w)
        # 通知 parent bubble
        parent = self.parent()
        if parent and hasattr(parent, 'setMaxBubbleWidth'):
            parent.setMaxBubbleWidth(max_w)

    def reportHeight(self):
        """外部触发高度报告（防抖 200ms，避免流式时频繁更新）"""
        if not self._auto_size:
            return
        if self._height_callback_pending:
            return
        self._height_callback_pending = True
        QTimer.singleShot(200, self._do_report_height)

    def _do_report_height(self):
        self._height_callback_pending = False
        # 测 #content 的高度（不是 body），避免 body 撑满 viewport 后的循环依赖
        self.page().runJavaScript(
            "document.getElementById('content') ? document.getElementById('content').scrollHeight : 0",
            self._on_height_measured
        )

    def reportNaturalWidth(self):
        """
        外部触发自然宽度测量。
        - 如果 _base_loaded 未就绪 → 200ms 后重试
        - 防止并发测量
        """
        if self._natural_width_measuring:
            return
        if not self._base_loaded:
            QTimer.singleShot(200, self.reportNaturalWidth)
            return
        self._natural_width_measuring = True
        # 重置重试计数（每次新调用都重置）
        self._natural_width_retry_count = 0
        QTimer.singleShot(250, self._do_measure_natural_width)

    def _do_measure_natural_width(self):
        """
        测 #content 的真正自然宽度。
        关键：position:absolute + white-space:nowrap + left:-9999px
        把元素取出文档流，避免被 view 的 setMaximumWidth 限制，
        也避免 white-space:normal 自动 wrap。
        """
        js = """
        (function() {
            var el = document.getElementById('content');
            if (!el) return 0;
            var saved = {
                position: el.style.position,
                left: el.style.left,
                top: el.style.top,
                maxWidth: el.style.maxWidth,
                width: el.style.width,
                whiteSpace: el.style.whiteSpace,
                display: el.style.display
            };
            el.style.position = 'absolute';
            el.style.left = '-9999px';
            el.style.top = '0';
            el.style.maxWidth = 'none';
            el.style.width = 'auto';
            el.style.whiteSpace = 'nowrap';
            // 强制 inline-block 拿 offsetWidth
            var computed = window.getComputedStyle(el).display;
            if (computed === 'inline') {
                el.style.display = 'inline-block';
            }
            var w = el.offsetWidth;
            // 还原
            el.style.position = saved.position;
            el.style.left = saved.left;
            el.style.top = saved.top;
            el.style.maxWidth = saved.maxWidth;
            el.style.width = saved.width;
            el.style.whiteSpace = saved.whiteSpace;
            el.style.display = saved.display;
            return w;
        })()
        """
        self.page().runJavaScript(js, self._on_natural_width_measured)

    def _on_natural_width_measured(self, width):
        self._natural_width_measuring = False
        try:
            if hasattr(width, 'toInt'):
                ok, n = width.toInt()
                natural_w = int(n) if ok else int(width.toFloat()[0])
            else:
                natural_w = int(float(width))
        except (ValueError, TypeError, AttributeError):
            natural_w = 0
        if natural_w <= 0:
            # 测量失败（content 还没渲染好 / JS 异常）→ 重试，最多 3 次
            if self._natural_width_retry_count < 3:
                self._natural_width_retry_count += 1
                QTimer.singleShot(400, self.reportNaturalWidth)
            return
        self._natural_width = natural_w
        # 通知 parent bubble
        parent = self.parent()
        if parent and hasattr(parent, 'setNaturalWidth'):
            parent.setNaturalWidth(natural_w)

    def _on_height_measured(self, height):
        try:
            h = int(height)
        except (ValueError, TypeError):
            return
        if h == self._last_reported_height:
            return
        self._last_reported_height = h
        if h > self.MAX_HEIGHT:
            self.page().runJavaScript("document.body.style.overflow = 'auto';")
            self.setFixedHeight(self.MAX_HEIGHT)
        else:
            self.page().runJavaScript("document.body.style.overflow = 'hidden';")
            # 加 BUBBLE_V_PADDING 给圆角留视觉缓冲（顶部+底部）
            self.setFixedHeight(max(40, h + self.BUBBLE_V_PADDING))
        # 通知父气泡
        parent = self.parent()
        if parent and hasattr(parent, "_on_view_resized"):
            parent._on_view_resized()

    # 不重写 sizeHint / minimumSizeHint：
    # QWebEngineView 在 PySide6 下重写这两个会引发 native crash
    # （PySide6 内部多次调用 sizeHint，可能带 QSizeF 参数，Python 重写易触发 ACCESS_VIOLATION）
    # 宽度由父 MessageBubble 通过 setFixedWidth / setMaximumWidth 直接控制

    def _markdown_to_html(self, text: str) -> str:
        """Markdown → HTML，保护 LaTeX 公式不被破坏"""
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
        # 还原占位符
        for i, p in enumerate(placeholders):
            html = html.replace(f"\x00MATHHD{i}\x00", p, 1).replace(f"\x00MATHIN{i}\x00", p, 1)
        return html

    # ============================================================
    # B：生命周期冻结（不可见时释放 GPU/内存）
    # ============================================================
    def freeze(self) -> None:
        """
        冻结页面：释放 Chromium 端的 GPU 资源，DOM 状态保留。
        滚回视野调 unfreeze() 恢复（无需重新渲染）。
        """
        try:
            from PySide6.QtWebEngineCore import QWebEnginePage
            self.page().setLifecycleState(QWebEnginePage.LifecycleState.Frozen)
        except (ImportError, AttributeError):
            # PySide6 < 6.2 没有 setLifecycleState
            self.setVisible(False)
            self.page().setUrl(QUrl("about:blank"))

    def unfreeze(self) -> None:
        """恢复冻结的页面。"""
        try:
            from PySide6.QtWebEngineCore import QWebEnginePage
            self.page().setLifecycleState(QWebEnginePage.LifecycleState.Active)
        except (ImportError, AttributeError):
            self.setVisible(True)

    # ============================================================
    # A2 + C：流式增量渲染
    # ============================================================
    def append_stream_chunk(self, full_text: str) -> None:
        """
        流式增量追加。每 chunk 调用一次。
        - 切稳定段（\n\n），转 markdown → HTML（公式保留为原文）
        - runJavaScript append 到 #content（不替换 innerHTML，不重排已存在节点）
        - **不** typeset
        """
        if not self._base_loaded:
            self._pending_markdown = full_text
            return

        if full_text.startswith(self._stream_rendered_text):
            new_text = full_text[len(self._stream_rendered_text):]
        else:
            new_text = full_text
            self._stream_pending = ""
        if not new_text:
            return
        self._stream_rendered_text = full_text
        self._stream_pending += new_text

        stable, self._stream_pending = self._split_stable(self._stream_pending)
        if not stable:
            return

        html = self._stream_markdown_to_html(stable)
        if not html.strip():
            return
        self.page().runJavaScript(
            f"window.qgStreamAppend && window.qgStreamAppend({json.dumps(html)});"
        )

    def end_streaming(self) -> None:
        """流式结束：处理剩余 pending + 一次性 typeset。"""
        if not self._base_loaded:
            return
        rest = self._stream_pending
        self._stream_pending = ""
        self._stream_rendered_text = ""
        rest_html = self._markdown_to_html(rest) if rest.strip() else ""
        self.page().runJavaScript(
            f"window.qgStreamFinalize && window.qgStreamFinalize({json.dumps(rest_html)});"
        )

    def _split_stable(self, pending: str) -> tuple:
        """按 \n\n 切稳定段；未闭合的代码块 / display 公式不切。"""
        if pending.count('```') % 2 == 1:
            return "", pending
        if pending.count('$$') % 2 == 1:
            return "", pending
        idx = pending.rfind('\n\n')
        if idx == -1:
            return "", pending
        return pending[:idx + 2], pending[idx + 2:]

    def _stream_markdown_to_html(self, text: str) -> str:
        """
        流式阶段：转 markdown。公式 $...$ / $$...$$ 保留为原文
        （markdown 不处理 $，MathJax typeset 时再处理）。
        """
        text = html_module.escape(text, quote=False)
        return markdown.markdown(text, extensions=['fenced_code', 'tables'])


# ============================================================
# MessageBubble — 消息气泡
# ============================================================
class MessageBubble(QWidget):
    """
    消息气泡（稳定版）：
    - 宽度由 setMaxBubbleWidth 一次设置，流式过程不重测
    - 流式过程：appendMarkdown 增量渲染，**不 typeset**（省 MathJax 计算）
    - finalize：endStreaming 触发一次 typeset，然后报告最终高度
    - resize 事件：防抖 80ms
    """
    # 气泡最小宽度（防止内容太短时气泡过窄；system 角色尤其需要）
    MIN_BUBBLE_WIDTH = 100
    # 边距（与 QHBoxLayout.setContentsMargins(5,5,5,5) 对应）
    MARGIN_H = 5
    MARGIN_V = 5
    # 圆角视觉缓冲（约半个字；防止最后一字被 border-radius 削掉 / 意外换行）
    BUBBLE_W_PADDING = 12  # 水平（左 + 右）
    BUBBLE_V_PADDING = 2   # 垂直（上 + 下）

    def __init__(self, role: str, markdown_text: str = "", attachments=None, parent=None):
        super().__init__(parent)
        self.role = role
        self.attachments = attachments or []
        if self.attachments:
            markdown_text = self._append_attachments_marker(markdown_text)
        self.full_text = markdown_text
        self.is_final = False

        # 流式状态：空内容 → 流式模式；非空内容 → 直接渲染模式
        self._is_streaming = not bool(markdown_text)
        self._natural_width = 0  # view 测出的内容自然宽度（finalize 后填）

        # 背景色
        if role == 'user':
            self.setStyleSheet("background-color: #77aaff; border-radius: 10px; border: none;")
            text_color = "white"
        elif role == 'assistant':
            self.setStyleSheet("background-color: white; border-radius: 10px; border: none;")
            text_color = "black"
        else:
            self.setStyleSheet("background-color: #e9ecef; border-radius: 10px; border: none;")
            text_color = "#495057"

        # 防抖 resize
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(80)
        self._resize_timer.timeout.connect(self._do_resize)

        # Markdown 视图
        self.view = MarkdownView(self)
        if role == 'system':
            text_css = f"body {{ color: {text_color}; text-align: center; }} h1,h2,h3,h4,h5,h6 {{ color: {text_color} !important; }}"
        else:
            text_css = f"body {{ color: {text_color}; }} h1,h2,h3,h4,h5,h6 {{ color: {text_color} !important; }}"
        self.view.setHtmlStyle(text_css)
        if markdown_text:
            self.view.setMarkdown(markdown_text)
        elif self._is_streaming:
            self.view.beginStreaming()

        # 布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(self.view)
        # sizePolicy: 横向 Maximum（可以扩展到 maxSize，但更倾向 sizeHint）
        #            纵向 Preferred（按内容高度）
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._max_view_width = 0  # 0 表示未设置，showEvent 后再算
        self._top_level = None
        # _update_max_width 推迟到 showEvent（那时 parent 已 layout，width 准确）
        # 非流式场景（markdown_text 非空）→ 渲染完后调 view.reportNaturalWidth 测紧凑尺寸
        if not self._is_streaming:
            # 留够时间让 base HTML 加载 + content 渲染 + MathJax typeset
            QTimer.singleShot(500, self.view.reportNaturalWidth)

    def _append_attachments_marker(self, text: str) -> str:
        if not self.attachments:
            return text
        names = " · ".join(f"📎 {a.get('name', '?')}" for a in self.attachments)
        sizes = " · ".join(a.get("size", "") for a in self.attachments if a.get("size"))
        suffix = f"> {names}  ({sizes})" if sizes and sizes.strip(" ·") else f"> {names}"
        return f"{text}\n\n{suffix}" if text else suffix

    def _bubble_width_for_text(self, text_w: int) -> int:
        """
        把 JS 测出的"内容自然宽度"换算成气泡宽度。
        加上：2 * MARGIN_H（QHBoxLayout 水平边距） + BUBBLE_W_PADDING（圆角缓冲）
        """
        if text_w <= 0:
            return 0
        return text_w + 2 * self.MARGIN_H + self.BUBBLE_W_PADDING

    def setMaxBubbleWidth(self, max_w: int):
        """
        外部调：设置气泡最大宽度（容器宽度的 80%）。
        - _max_view_width 不变就不动
        - 流式过程：固定 max_w（避免宽度抖动）
        - 已 finalize / 非流式：min(natural+padding, max_w)，不低于 MIN_BUBBLE_WIDTH
        - 如果自然宽度还没测到，主动触发一次测量（避免一直顶着 80%）
        """
        if max_w == self._max_view_width:
            return
        self._max_view_width = max_w
        # view 的 max 也更新
        self.view.setMaxBubbleWidth(max_w)
        # 自身也设 max（防止 setFixedWidth 之外的场景超界）
        self.setMaximumWidth(max_w)
        # 决定实际 fixed width
        if self._is_streaming:
            # 流式中：锁最大宽度
            self.setFixedWidth(max_w)
        else:
            # 已 finalize / 非流式加载：用自然宽度（如已知且 < max_w）
            natural_w = getattr(self, '_natural_width', 0)
            if natural_w and natural_w > 0:
                bubble_w = self._bubble_width_for_text(natural_w)
                w = max(self.MIN_BUBBLE_WIDTH, min(bubble_w, max_w))
                self.setFixedWidth(w)
            else:
                # 自然宽度还没测出来 → 暂用 max_w，等 setNaturalWidth 回调再缩
                self.setFixedWidth(max_w)
                # 主动触发一次测量（如果 view 还没在测）
                if hasattr(self.view, 'reportNaturalWidth'):
                    QTimer.singleShot(50, self.view.reportNaturalWidth)

    def setNaturalWidth(self, natural_w: int):
        """
        外部调：告知 view 内容的自然宽度。
        - 流式过程：缓存 _natural_width，不缩（避免抖动）
        - finalize 后 / 非流式：min(text+padding, max_w)，不低于 MIN_BUBBLE_WIDTH
        """
        if natural_w <= 0:
            return
        self._natural_width = natural_w
        if self._is_streaming:
            return
        if self._max_view_width > 0:
            bubble_w = self._bubble_width_for_text(natural_w)
            w = max(self.MIN_BUBBLE_WIDTH, min(bubble_w, self._max_view_width))
            self.setFixedWidth(w)

    def _update_max_width(self):
        """计算最大宽度（基于容器宽度，最大 80%）"""
        parent = self.parent()
        available = 0
        if parent:
            available = parent.width()
        if available <= 0:
            top = self.window()
            available = top.width() if top else 800
        # 用户期望：最大 80% 容器宽度（自适应内容；最宽不超过 80%）
        max_w = max(self.view.MIN_WIDTH, min(int(available * 0.80), 1800))
        self.setMaxBubbleWidth(max_w)

    def updateMaxWidth(self):
        """兼容旧接口：触发宽度更新（防抖）"""
        self._resize_timer.start()

    def _do_resize(self):
        self._update_max_width()
        # 不调 adjustSize（避免反复 layout）
        self.updateGeometry()

    def _on_view_resized(self):
        """view 高度变化时通知"""
        self.updateGeometry()

    # ---------- 流式更新 ----------
    def update_content(self, new_text: str):
        """
        流式追加内容（A2 + C 实现）：
        - 第一次 update_content 时进入流式状态
        - 追加到 self.full_text，调 view.append_stream_chunk（**增量追加**到 DOM，**不**重新替换 innerHTML）
        - 触发一次 reportHeight（200ms 防抖）
        - 宽度保持 _max_view_width，不根据内容调整（避免流式中闪）
        """
        if self.is_final or not new_text:
            return

        # 第一次 update_content → 标记流式
        if not self._is_streaming:
            self._is_streaming = True
            self.view.beginStreaming()

        self.full_text = (self.full_text or "") + new_text
        # ✨ A2 + C：增量追加到 DOM，不替换 innerHTML
        self.view.append_stream_chunk(self.full_text)
        # 触发一次高度报告（防抖 200ms）
        self.view.reportHeight()

    def finalize(self):
        """
        流式结束（A2 + C 实现）：
        - view.end_streaming()：把剩余 pending 一次性解析 + 追加 + 一次性 typeset
        - 报告最终高度
        - 测 view 的自然宽度（JS 测 offsetWidth），让 bubble 缩到紧凑尺寸
        """
        if self.is_final:
            return
        self.is_final = True
        self._is_streaming = False
        # ✨ A2 + C：end_streaming 一次性处理剩余 + typeset
        self.view.end_streaming()
        # 报告最终高度（typeset 前后各一次）
        QTimer.singleShot(100, self.view.reportHeight)
        QTimer.singleShot(800, self.view.reportHeight)
        # 测 view 自然宽度（多次重试，覆盖 MathJax 渲染节奏）
        QTimer.singleShot(400, self.view.reportNaturalWidth)
        QTimer.singleShot(1200, self.view.reportNaturalWidth)
        QTimer.singleShot(2500, self.view.reportNaturalWidth)
        # finalize 后：_is_streaming=False，会按 _natural_width 收缩
        # 这里显式触发一次 _update_max_width，让 bubble 立即按当前状态决定宽度
        QTimer.singleShot(50, self._update_max_width)

    # ---------- 事件过滤（防抖） ----------
    def eventFilter(self, watched, event):
        if watched == self._top_level and event.type() == QEvent.Resize:
            self._resize_timer.start()
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        super().showEvent(event)
        top = self.window()
        if top and top != self._top_level:
            if self._top_level:
                self._top_level.removeEventFilter(self)
            top.installEventFilter(self)
            self._top_level = top
        QTimer.singleShot(0, self._update_max_width)
