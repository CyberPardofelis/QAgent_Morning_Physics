# worker.py (完整修改后)
import traceback
from PySide6.QtCore import QThread, Signal
from agent.controller import run_agent_stream

class Worker(QThread):
    # 增加第三个参数：bubble_list（存储每个独立气泡的内容）
    finished = Signal(str, str, list, list)  # full_reply, error, final_messages, bubble_list
    stream_chunk = Signal(str, str)

    def __init__(self, messages, mode, extra_params=None):
        super().__init__()
        self.messages = messages
        self.mode = mode
        self.extra_params = extra_params or {}
        self._is_running = True

    def run(self):
        full_reply = ""
        bubbles = []          # 存储每个气泡的完整内容
        current_bubble = ""   # 当前正在累积的气泡内容
        try:
            for chunk in run_agent_stream(self.messages, mode=self.mode, **self.extra_params):
                if not self._is_running:
                    break
                if chunk == "\x1cNEW_BUBBLE":
                    # 收到分隔符，结束当前气泡
                    if current_bubble:
                        bubbles.append(current_bubble)
                        current_bubble = ""
                else:
                    current_bubble += chunk
                    full_reply += chunk
                    self.stream_chunk.emit("assistant", chunk)
            # 最后一个气泡
            if current_bubble:
                bubbles.append(current_bubble)
            # 传递气泡列表给主窗口
            self.finished.emit(full_reply, "", self.messages, bubbles)
        except Exception as e:
            error_msg = f"执行出错：{e}\n{traceback.format_exc()}"
            self.finished.emit("", error_msg, self.messages, [])

    def stop(self):
        self._is_running = False