"""
MessageBubble + MarkdownView（v9 — 抄 SAgent 简化 + QAgent 公式扩展 + QAgent 特色功能补回）

v9 状态 = v8 (整段抄 SAgent) + 3 个 QAgent 特色功能补回:
- MessageBubble hardcode 气泡颜色 (QAgent 原本的 user 蓝/assistant 白/system 灰)
- MarkdownView setHtmlStyle CSS 累积 (QAgent 原本的 text_color 文字色)
- MarkdownView.setMarkdown detect inline + display 公式 (QAgent 公式渲染)

外部接口保持 (chat_window / conversation_controller / chat_view 调用):
- MessageBubble: role, is_final, full_text, view, _frozen
  update_content / finalize / setMaxBubbleWidth / setNaturalWidth
  setHtmlStyle / _on_view_resized
- MarkdownView: setMarkdown / setHtml / freeze / unfreeze
  setAutoSize / setMaxBubbleWidth / reportHeight / reportNaturalWidth
  (beginStreaming / append_stream_chunk / endStreaming 旧接口占位)
"""
import re
from PySide6.QtWidgets import QWidget, QHBoxLayout, QTextBrowser, QSizePolicy
from PySide6.QtCore import Qt, QTimer, QEvent, QSize
from PySide6.QtGui import QTextOption, QDesktopServices


# ============================================================
# MarkdownView — 抄 SAgent (QTextBrowser 自动 layout)
# + QAgent 扩展: setHtml 公式路径 + setHtmlStyle CSS 累积
# ============================================================
class MarkdownView(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QTextBrowser.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setLineWrapMode(QTextBrowser.WidgetWidth)
        self.setStyleSheet("background: transparent; border: none; margin: 0px; padding: 0px;")
        self.document().setDocumentMargin(0)
        # 2026-09-04 修:关掉 QTextBrowser 的 setOpenLinks(默认 True)
        # 默认行为:点链接 → 触发 anchorClicked(我们用 QDesktopServices 开外部浏览器)
        #   **同时** QTextBrowser 内部 navigate 到那个 URL,导致气泡内容被替换成
        #   abstract 页/404 页面文本。
        # 修法:setOpenLinks(False) 后,只走 anchorClicked 信号,内部不 navigate。
        self.setOpenLinks(False)
        self.anchorClicked.connect(lambda url: QDesktopServices.openUrl(url))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        # 文档布局变化 → 自动 updateGeometry
        self.document().documentLayout().documentSizeChanged.connect(self._onDocumentSizeChanged)

        self.right_margin = 0
        self._html_style = ""  # v9 补:QAgent 特色 — CSS 累积
        self._apply_document_margins()
        self._apply_stylesheet()

    def _apply_document_margins(self):
        doc = self.document()
        root = doc.rootFrame()
        fmt = root.frameFormat()
        fmt.setLeftMargin(0)
        fmt.setRightMargin(self.right_margin)
        root.setFrameFormat(fmt)

    def _apply_stylesheet(self):
        """v9 补:合并 _html_style + markdown_with_formulas.DEFAULT_BODY_CSS"""
        from ui.markdown_with_formulas import DEFAULT_BODY_CSS
        full_css = DEFAULT_BODY_CSS + '\n' + (self._html_style or '')
        self.document().setDefaultStyleSheet(full_css)

    def setHtmlStyle(self, css: str):
        """v9 补:QAgent 特色 — 累积 CSS(MessageBubble.__init__ 调一次给文字色)"""
        if css and css not in self._html_style:
            self._html_style = (self._html_style + '\n' + css) if self._html_style else css
        self._apply_stylesheet()

    def setMarkdown(self, text: str):
        """
        QAgent 扩展:自动 detect 公式走 setHtml 路径。
        - 有公式 ($...$ 或 $$...$$) → setHtml 走 markdown_with_formulas
        - 无公式 → super().setMarkdown() 走原生 (快)
        """
        if not text:
            super().setMarkdown("")
            self._apply_document_margins()
            return
        # v9 修复:detect inline + display 两种公式
        if re.search(r'\$[^$\n]+\$|\$\$[\s\S]+?\$\$', text):
            html = self._render_html(text)
            self.setHtml(html)
        else:
            super().setMarkdown(text)
        self._apply_document_margins()

    def setHtml(self, html: str):
        """直接 setHtml (QAgent 扩展)"""
        super().setHtml(html)
        self._apply_document_margins()
        # v18.1: 强制公式图片按 PNG 物理像素显示
        # 关键: 必须用 QTextImageFormat 重新构造(用 toImageFormat 副本 setWidth 写回不算,
        # 副本被 setCharFormat 时会丢失 image 类型标志)
        self._force_formula_image_size()

    def _force_formula_image_size(self):
        """
        v18 公式图片强制按 PNG 物理像素显示 — 三重保险之一
        - 关键: QTextImageFormat 是按值传递的!改 fragment.charFormat() 返回的
          fmt 副本,doc 内部 layout 用的 fmt 不变。
          必须用 QTextCursor.setCharFormat 写回 doc 才会生效。
        - 单位: setWidth 接受 1:1 像素(实测,不是 pt)
        - v18.1: 用 setProperty(ImageWidth/ImageHeight) 直接设 QTextLength property
          (不用 img_fmt.setWidth/setHeight — 那可能被 toImageFormat 转换丢失)
        """
        import os as _os
        from PIL import Image as _PILImage
        from PySide6.QtGui import QTextCursor, QTextCharFormat, QTextLength, QTextFormat, QTextImageFormat
        try:
            doc = self.document()
            cursor = QTextCursor(doc)
            block = doc.begin()
            while block.isValid():
                it = block.begin()
                while not it.atEnd():
                    frag = it.fragment()
                    if frag.isValid():
                        char_fmt = frag.charFormat()
                        if char_fmt.isImageFormat():
                            raw = char_fmt.toImageFormat().name()
                            if raw:
                                if raw.startswith('file:///'):
                                    local = raw[len('file:///'):]
                                elif raw.startswith('file://'):
                                    local = raw[len('file://'):]
                                else:
                                    local = raw
                                local = local.replace('/', _os.sep)
                                try:
                                    with _PILImage.open(local) as p:
                                        w_px, h_px = p.size
                                except Exception:
                                    w_px, h_px = 0, 0
                                if w_px > 0 and h_px > 0:
                                    # v18.9 撤回 (2026-09-01 17:31) → 恢复 v18.8 状态
                                    # 原因: v18.9 (PIL Lanczos 4x 预降采样 + 1:1 物理显示) 实测 user 反馈"更模糊了"
                                    #   - PIL Lanczos 对 4x 激进 downsample + 稀疏字符 组合产生 ringing 伪影
                                    #   - 1:1 物理显示 + DPI 缩放链上 Qt 再做一次缩放,叠加反而更差
                                    #   - v18.8 (Qt bilinear 4x 实时缩放) 是当前最稳状态
                                    # 恢复: SCALE = 1/4 + setWidth(80 fixed) + setHeight 按 PNG 宽高比算
                                    #   - 字符 visual 跟 v18.4 接近(简单公式 19.7, 多行公式 111.8)
                                    #   - 长公式字符 visual 从 v18.7 之前的 7.8 提升到 23-30
                                    #   - 多行公式不被压(按 PNG 物理宽高比)
                                    # 历史参考:
                                    #   v18.4: setWidth=80 fixed + setHeight 按比例 → user 接受 "scale 没问题"
                                    #   v18.5/v18.6: 标准化 PNG 高 100 + setWidth=w*0.5 → user 报"又太大/又压扁"否决
                                    #   v18.7: 强制 inline + setWidth=80 fixed + setHeight 按比例 → user 接受
                                    #   v18.8: SCALE 1/4 去掉 hardcode 80 → 解决"宽度统一"问题 (v18.7 setWidth=80 fixed 已被回退,见 v18.8 注释)
                                    #   v18.9: PIL Lanczos + 1:1 → 失败撤回
                                    # 当前状态(回退 v18.8): SCALE 1/4 + 按比例 (user 接受过的 v18.7 状态)
                                    SCALE = 1.0 / 4.0
                                    new_fmt = QTextImageFormat()
                                    new_fmt.setName(raw)
                                    new_fmt.setWidth(w_px * SCALE)
                                    new_fmt.setHeight(h_px * SCALE)
                                    cursor.setPosition(frag.position())
                                    cursor.setPosition(
                                        frag.position() + frag.length(),
                                        QTextCursor.KeepAnchor
                                    )
                                    cursor.setCharFormat(new_fmt)
                    it += 1
                block = block.next()
            # v18.1: 强制 doc 重新算 idealWidth
            self.document().setTextWidth(-1)
        except Exception as e:
            import sys
            print(f"[MarkdownView] _force_formula_image_size 失败: {e}", file=sys.stderr)

    def _render_html(self, md_text: str) -> str:
        """markdown → 完整 HTML 文档(含公式图片)"""
        from ui.markdown_with_formulas import md_to_html_with_formulas, wrap_html
        body = md_to_html_with_formulas(md_text)
        return wrap_html(body)

    def _onDocumentSizeChanged(self, size):
        self.updateGeometry()

    def sizeHint(self):
        doc = self.document()
        current_width = self.viewport().width()
        if current_width > 0:
            doc.setTextWidth(-1)
            ideal_width = doc.idealWidth()
            doc.setTextWidth(current_width)
        else:
            ideal_width = doc.idealWidth()
        width = int(ideal_width) + self.right_margin + 26
        height = int(doc.size().height()) + 18
        return QSize(width, height)

    def minimumSizeHint(self):
        return self.sizeHint()

    def resizeEvent(self, event):
        if event.size().width() != event.oldSize().width():
            self.document().setTextWidth(self.viewport().width())
        super().resizeEvent(event)

    # ---------- 兼容旧接口 ----------
    def beginStreaming(self):
        pass

    def append_stream_chunk(self, full_text: str):
        pass

    def endStreaming(self):
        pass

    def setAutoSize(self, auto: bool, max_width: int = None):
        if max_width:
            self.setMaximumWidth(max_width)

    def setMaxBubbleWidth(self, max_w: int):
        self.setMaximumWidth(max_w)

    def reportHeight(self):
        self.updateGeometry()

    def reportNaturalWidth(self):
        self.updateGeometry()

    def freeze(self) -> None:
        self.setVisible(False)

    def unfreeze(self) -> None:
        self.setVisible(True)


# ============================================================
# MessageBubble — 抄 SAgent + QAgent 特色补回
# ============================================================
class MessageBubble(QWidget):
    def __init__(self, role: str, markdown_text: str = "", attachments=None, parent=None):
        super().__init__(parent)
        self.role = role
        self.attachments = attachments or []
        if self.attachments:
            markdown_text = self._append_attachments_marker(markdown_text)
        self.full_text = markdown_text
        self.is_final = False
        self._line_buffer = ""
        self._complete_lines = []
        self._in_math_block = False
        self._frozen = False
        self._max_view_width = 800

        self.update_timer = QTimer()
        self.update_timer.setInterval(100)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._flush_complete_lines)

        # 2026-09-02 v6 改:用 setObjectName("bubble_user/assistant/system") 替代 setProperty("kind", ...)
        # 原因: 之前用 dynamic property + [kind=...] QSS attribute selector 在 chat_view
        # 上下文中不生效 (debug 看到 palette bg = #000000 而不是 #77aaff, QSS 规则没被匹配)
        # 改用 objectName + #objectName 选择器 — 这是 Qt 5/6 最稳的 dynamic styling 方式
        # 对应 QSS: QWidget#bubble_user { background-color: {bubble_user_bg}; ... }
        if role == 'user':
            self.setObjectName("bubble_user")
            text_color = "white"
        elif role == 'assistant':
            self.setObjectName("bubble_assistant")
            text_color = "black"
        else:
            self.setObjectName("bubble_system")
            text_color = "#495057"
        # objectName 已设, #objectName QSS 规则会接管, 不需要 WA_StyledBackground
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.view = MarkdownView(self)
        self.view.setStyleSheet("background: transparent;")
        if markdown_text:
            self.view.setMarkdown(markdown_text)
        # 2026-09-02 修:系统消息(欢迎/启动提示)文字不居中 —
        # body { text-align: center } CSS 在 QTextBrowser 不稳;
        # widget 级 setAlignment 会被后续 setMarkdown 重置,
        # 所以**必须在 setMarkdown 之后**再设。同时加文档级 setDefaultTextOption 兜底
        # (后者对 QTextDocument 走 HTML setHtml 路径更稳,前者对 setMarkdown 路径稳)。
        if role == 'system':
            self.view.setAlignment(Qt.AlignCenter)
            from PySide6.QtGui import QTextOption as _QTextOption
            self.view.document().setDefaultTextOption(_QTextOption(Qt.AlignCenter))
        # v9 补:setHtmlStyle 给 view 文字色(QAgent 原本特色)
        if role == 'system':
            text_css = f"body {{ color: {text_color}; text-align: center; }} h1,h2,h3,h4,h5,h6 {{ color: {text_color} !important; }}"
        else:
            text_css = f"body {{ color: {text_color}; }} h1,h2,h3,h4,h5,h6 {{ color: {text_color} !important; }}"
        self.view.setHtmlStyle(text_css)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.view)

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._max_width = 800
        self._top_level = None
        QTimer.singleShot(0, self._update_max_width)

    def _append_attachments_marker(self, text: str) -> str:
        if not self.attachments:
            return text
        names = " · ".join(
            f"📎 {a.get('name', '?')}" for a in self.attachments
        )
        sizes = " · ".join(
            a.get("size", "") for a in self.attachments if a.get("size")
        )
        if sizes and sizes.strip(" ·"):
            suffix = f"> {names}  ({sizes})"
        else:
            suffix = f"> {names}"
        if text:
            return f"{text}\n\n{suffix}"
        return suffix

    # ---------- 兼容旧接口 ----------
    def updateMaxWidth(self):
        self._update_max_width()

    def _on_view_resized(self):
        pass

    def setMaxBubbleWidth(self, max_w: int):
        if max_w == self._max_view_width:
            return
        self._max_view_width = max_w
        self._max_width = max_w
        self.view.setMaximumWidth(max_w)
        self.layout().activate()
        self.adjustSize()

    def setNaturalWidth(self, natural_w: int):
        pass

    # ---------- 核心:动态更新最大宽度 ----------
    def _update_max_width(self):
        # v18.1: 找 QAbstractScrollArea 类型的 parent(或更上层)的 viewport
        # 之前用 hasattr(parent, 'viewport') 不靠谱 — QWidget 也可能有 viewport 属性
        # 改用 isinstance(parent, QAbstractScrollArea)
        from PySide6.QtWidgets import QAbstractScrollArea
        parent = self.parent()
        available_width = 800
        while parent:
            if isinstance(parent, QAbstractScrollArea):
                available_width = parent.viewport().width()
                break
            parent = parent.parent()
        if available_width <= 0:
            window = self.window()
            if window:
                available_width = window.width()
            else:
                available_width = 800

        max_width = max(40, min(int(available_width * 0.80), 2000))
        if max_width != self._max_width:
            self._max_width = max_width
            self.view.setMaximumWidth(max_width)
            self.layout().activate()
            self.adjustSize()

    def resizeEvent(self, event):
        QTimer.singleShot(0, self._update_max_width)
        super().resizeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._update_max_width)
        top = self.window()
        if top and top != self._top_level:
            if self._top_level:
                self._top_level.removeEventFilter(self)
            top.installEventFilter(self)
            self._top_level = top

    def eventFilter(self, watched, event):
        if watched == self._top_level and event.type() == QEvent.Resize:
            self._update_max_width()
        return super().eventFilter(watched, event)

    # ---------- 流式追加 (抄 SAgent) ----------
    def update_content(self, new_text: str):
        if self.is_final:
            return
        for ch in new_text:
            self._line_buffer += ch
            if ch == '\n':
                self._complete_lines.append(self._line_buffer)
                self._line_buffer = ""
        if self._complete_lines:
            self._flush_complete_lines()
        else:
            if not self.update_timer.isActive():
                self.update_timer.start()

    def replace_content(self, new_text: str):
        self.full_text = new_text
        self.view.setMarkdown(new_text)
        self._apply_alignment_if_system()  # 2026-09-02 加:setMarkdown 会重置 alignment
        self.is_final = True
        self._line_buffer = ""
        self._complete_lines = []
        self.update_timer.stop()

    def set_content(self, new_text: str):
        self.full_text = new_text
        self.view.setMarkdown(new_text)
        self._apply_alignment_if_system()  # 2026-09-02 加:setMarkdown 会重置 alignment
        self.update_timer.stop()

    def _flush_complete_lines(self):
        if self.is_final:
            return
        if not self._complete_lines and self._line_buffer:
            self._complete_lines.append(self._line_buffer)
            self._line_buffer = ""
        if not self._complete_lines:
            return
        chunk = ''.join(self._complete_lines)
        self._complete_lines.clear()
        safe_chunk = self._preprocess(chunk)
        self.full_text += safe_chunk
        self.view.setMarkdown(self.full_text)
        self._apply_alignment_if_system()  # 2026-09-02 加:setMarkdown 会重置 alignment
        self.update_timer.stop()

    def _apply_alignment_if_system(self):
        """2026-09-02 加:setMarkdown 会重置 QTextBrowser 的 alignment,system 气泡需每帧重设"""
        if self.role == 'system':
            self.view.setAlignment(Qt.AlignCenter)

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
                if idx < len(processed_lines)-1 and processed_lines[idx+1].strip():
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
        self.view.setMarkdown(self.full_text)
        self._apply_alignment_if_system()  # 2026-09-02 加:setMarkdown 会重置 alignment
        QTimer.singleShot(100, self._update_max_width)
