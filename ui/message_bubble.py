from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget, QHBoxLayout
from QMarkdownWidget import QMView

class MessageBubble(QWidget):
    def __init__(self, role: str, markdown_text: str, parent=None):
        super().__init__(parent)
        self.role = role
        self.full_text = markdown_text
        self.is_final = False
        self._line_buffer = ""          # 当前正在构建的行（不含换行符）
        self._complete_lines = []        # 已完成的行队列（每行包含换行符）
        self.update_timer = QTimer()
        self.update_timer.setInterval(100)  # 300ms 刷新一次
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._flush_complete_lines)
        self._in_math_block = False

        # 设置气泡整体样式（背景色、圆角、内边距），占满整个可用宽度
        if role == 'user':
            self.setStyleSheet("""
                        background-color: #77aaff;
                        border-radius: 10px;
                        padding: 8px;
                    """)
            alignment = Qt.AlignRight  # 整体靠右（背景从右侧开始）
            # 内容样式：文本左对齐，内容区块靠右（margin-left: auto），宽度限制80%
            content_css = """
                        body {
                            background: transparent;
                            color: white;
                            margin: 0;
                            padding: 0;
                            overflow: hidden;
                            word-wrap: break-word;
                            max-width: 100%;
                            margin-left: auto;   /* 内容区块靠右 */
                            text-align: left;     /* 文本本身左对齐 */
                        }
                    """
        elif role == 'assistant':
            self.setStyleSheet("""
                        background-color: white;
                        border-radius: 10px;
                        padding: 8px;
                    """)
            alignment = Qt.AlignLeft  # 整体靠左
            content_css = """
                        body {
                            background: transparent;
                            color: black;
                            margin: 0;
                            padding: 0;
                            overflow: hidden;
                            word-wrap: break-word;
                            max-width: 100%;
                            margin-right: auto;  /* 内容区块靠左 */
                            text-align: left;
                        }
                    """
        else:  # system
            self.setStyleSheet("""
                        background-color: #e9ecef;
                        color: #495057;
                        border-radius: 10px;
                        padding: 6px;
                        font-size: 12px;
                    """)
            alignment = Qt.AlignCenter
            content_css = """
                        body {
                            background: transparent;
                            color: #495057;
                            margin: 0 auto;
                            padding: 0;
                            overflow: hidden;
                            word-wrap: break-word;
                            max-width: 100%;
                            text-align: center;
                        }
                    """
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(alignment)

        self.view = QMView()
        self.view.setMarkdown(markdown_text)

        # 注入样式：显式设置标题样式 + 隐藏滚动条
        full_css = content_css + """
                    ::-webkit-scrollbar {
                        display: none;
                    }
                    h1, h2, h3, h4, h5, h6 {
                        font-weight: bold !important;
                        font-size: 1.5em !important;
                        color: inherit !important;
                    }
                """
        self.view.setHtmlStyle(full_css)
        self.view.setAutoSize(True, max_width=900)

        layout.addWidget(self.view)

    def update_content(self, new_text: str):
        """按字符处理，构建完整的行，并保留换行符"""
        if self.is_final:
            return
        for char in new_text:
            self._line_buffer += char
            if char == '\n':
                # print("发现换行！")
                # 遇到换行符，将当前行（包含换行符）存入队列
                self._complete_lines.append(self._line_buffer)
                self._line_buffer = ""   # 清空缓冲区，准备下一行

        # 如果有完成的行，立即刷新或启动定时器
        if self._complete_lines:
            if len(self._complete_lines) >= 1:   # 积累2行立即刷新，增强实时感
                self._flush_complete_lines()
            elif not self.update_timer.isActive():
                self.update_timer.start()

    def _flush_complete_lines(self):
        """刷新所有完成的行"""
        if not self._complete_lines or self.is_final:
            return

        # 将队列中的行拼接成一个字符串（它们已经自带换行符）
        chunk = ''.join(self._complete_lines)
        self._complete_lines.clear()

        # print(f"[刷新] 完整行: {repr(chunk)}")  # 调试输出

        # 预处理（处理数学块）
        safe_chunk = self._preprocess(chunk)
        self.full_text += safe_chunk

        # 全量渲染（QMView 会重新解析整个文本）
        self.view.setMarkdown(self.full_text)
        self.update_timer.stop()

    def _preprocess(self, text: str) -> str:
        """预处理：处理数学块，标题后插入空行"""
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

        # 标题后插入空行
        result = []
        for idx, line in enumerate(processed_lines):
            result.append(line)
            if line.lstrip().startswith('#'):
                if idx < len(processed_lines) - 1:
                    next_line = processed_lines[idx + 1]
                    if next_line.strip() and not next_line.lstrip().startswith('#'):
                        result.append('')  # 插入空行，join后变成\n\n
        return '\n'.join(result)

    def finalize(self):
        if self.is_final:
            return
        self.is_final = True

        # 处理最后一行（可能没有换行符）
        if self._line_buffer:
            # 最后一行没有换行符，我们主动加一个
            self._complete_lines.append(self._line_buffer + '\n')
            self._line_buffer = ""

        if self._complete_lines:
            self._flush_complete_lines()

        # 最终全量渲染（确保格式正确）
        self.view.setMarkdown("")
        self.view.setMarkdown(self.full_text)
