import traceback
from PySide6.QtCore import QThread, Signal
from agent.controller import run_agent_stream


class Worker(QThread):
    """支持流式输出的工作线程"""
    finished = Signal(str, str, list)          # 完整回复（或空），错误信息
    stream_chunk = Signal(str, str)      # 发送者，文本块

    def __init__(self, messages, mode):
        super().__init__()
        self.messages = messages
        self.mode = mode
        self._is_running = True

    def run(self):
        full_reply = ""
        try:
            for chunk in run_agent_stream(self.messages, mode=self.mode):
                if not self._is_running:
                    break
                if chunk:
                    full_reply += chunk
                    # ===== 新增：打印原始 chunk 的完整内容和长度 =====
                    # print(f"[RAW CHUNK] length={len(chunk)}, repr={repr(chunk)}")
                    self.stream_chunk.emit("assistant", chunk)
            # 生成结束，self.messages 已被 run_agent_stream 更新
            self.finished.emit(full_reply, "", self.messages)
        except Exception as e:
            error_msg = f"执行出错：{e}\n{traceback.format_exc()}"
            self.finished.emit("", error_msg, self.messages)

    def stop(self):
        self._is_running = False