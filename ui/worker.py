# worker.py — 流式 Worker（QAgent 适配 WAgent 范式）
#
# 关键点：
# 1. 转发所有 chunk（包括 \x1cNEW_BUBBLE）给 controller；不自己处理切气泡
# 2. signal 全部带 conv_id，让 chat_window 能按当前 conv 过滤
# 3. finished 同时给 final_messages 和 bubble_list（保留旧 QAgent 接口，
#    ConversationController 用 bubble_list 落库；chat_window 用它兜底）
import traceback

from PySide6.QtCore import QThread, Signal

from agent.controller import run_agent_stream

NEW_BUBBLE = "\x1cNEW_BUBBLE"


class Worker(QThread):
    # (conv_id, full_reply, error, final_messages, bubble_list)
    finished = Signal(str, str, str, list, list)
    # (conv_id, chunk)  —— chunk 可以是文本或 \x1cNEW_BUBBLE
    stream_chunk = Signal(str, str)

    def __init__(self, conv_id, messages, mode, extra_params=None):
        super().__init__()
        self.conv_id = conv_id or ""
        self.messages = messages
        self.mode = mode
        self.extra_params = extra_params or {}
        self._is_running = True

    def run(self):
        full_reply = ""
        bubbles = []           # 备用：所有按 NEW_BUBBLE 切出来的段（流式落库用）
        current_bubble = ""
        try:
            for chunk in run_agent_stream(self.messages, mode=self.mode, **self.extra_params):
                if not self._is_running:
                    break
                if chunk == NEW_BUBBLE:
                    # 切气泡信号：直接转发给 controller，让 chat_window 决定是否开新气泡
                    if current_bubble:
                        bubbles.append(current_bubble)
                        current_bubble = ""
                    self.stream_chunk.emit(self.conv_id, NEW_BUBBLE)
                elif chunk:
                    current_bubble += chunk
                    full_reply += chunk
                    self.stream_chunk.emit(self.conv_id, chunk)
            # 收尾
            if current_bubble:
                bubbles.append(current_bubble)
            self.finished.emit(self.conv_id, full_reply, "", self.messages, bubbles)
        except Exception as e:
            error_msg = f"执行出错：{e}\n{traceback.format_exc()}"
            self.finished.emit(self.conv_id, "", error_msg, self.messages, bubbles)

    def stop(self):
        self._is_running = False
