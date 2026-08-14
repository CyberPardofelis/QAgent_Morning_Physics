import json
from datetime import datetime
from PySide6.QtCore import QObject, Signal, QSettings
from agent.history_manager import HistoryManager
from ui.worker import Worker
from agent.tools import get_paper_by_id, update_paper_llm_summary
from agent.prompts import SYSTEM_PROMPT
import config


class ConversationController(QObject):
    new_message = Signal(str, str)          # role, text
    stream_chunk = Signal(str, str)         # role, chunk (流式)
    new_bubble = Signal()                   # 强制创建新气泡
    comparison_chunk = Signal(str)          # 对比模式专用
    worker_finished = Signal(object, str, list, list)
    comparison_finished = Signal(object, str, list)
    clear_chat = Signal()
    scroll_to_bottom = Signal()
    update_regenerate_button = Signal(bool)
    update_stop_button = Signal(bool)
    update_send_input = Signal(bool)
    conversation_switched = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_manager = HistoryManager()
        self.current_conversation = None
        self.processing = False
        self.worker = None
        self._last_ai_message_id = None
        self._comparison_index = 0
        self._comparison_params = []
        self._comparison_messages = []
        self._comparison_last_bubble = None

        settings = QSettings("CyberPardofelis", "LLM科研助手")
        self.default_system_prompt = settings.value("default_system_prompt", SYSTEM_PROMPT)
        self._load_comparison_config()

    def _load_comparison_config(self):
        settings = QSettings("CyberPardofelis", "LLM科研助手")
        try:
            self.compare_config_a = json.loads(settings.value("comparison/config_a",
                '{"model":"deepseek-flash","thinking":true,"temperature":0.3,"top_p":0.9,"frequency_penalty":0.0,"presence_penalty":0.0}'))
            self.compare_config_b = json.loads(settings.value("comparison/config_b",
                '{"model":"deepseek-pro","thinking":true,"temperature":0.3,"top_p":0.9,"frequency_penalty":0.0,"presence_penalty":0.0}'))
        except:
            self.compare_config_a = {"model": "deepseek-flash", "thinking": True, "temperature": 0.3, "top_p": 0.9,
                                     "frequency_penalty": 0.0, "presence_penalty": 0.0}
            self.compare_config_b = {"model": "deepseek-pro", "thinking": True, "temperature": 0.3, "top_p": 0.9,
                                     "frequency_penalty": 0.0, "presence_penalty": 0.0}

    def _get_current_system_prompt(self):
        if self.current_conversation and self.current_conversation.group_id:
            group_id = self.current_conversation.group_id
            settings = QSettings("CyberPardofelis", "LLM科研助手")
            group_prompt = settings.value(f"group_system_prompt/{group_id}")
            if group_prompt:
                return group_prompt
        return self.default_system_prompt

    def _clean_messages_for_api(self, messages):
        cleaned = []
        for msg in messages:
            new_msg = msg.copy()
            if new_msg.get("role") == "tool" and "tool_call_id" not in new_msg:
                new_msg["role"] = "assistant"
                new_msg.pop("tool_calls", None)
            cleaned.append(new_msg)
        return cleaned

    def _get_model_name_by_mode(self, mode):
        if mode == "openai":
            return config.OPENAI_MODEL
        return mode

    def _resolve_model_name(self, model_type):
        if model_type == "openai":
            return config.OPENAI_MODEL
        return model_type

    def _save_llm_summary(self, final_messages):
        paper_id = None
        for msg in final_messages:
            if msg.get("role") == "tool":
                try:
                    content = msg.get("content", "")
                    if content:
                        result = json.loads(content)
                        if isinstance(result, dict) and "paper_id" in result:
                            paper_id = result["paper_id"]
                            break
                except:
                    continue
        if not paper_id:
            return
        paper = get_paper_by_id(paper_id)
        if paper and paper.get("llm_summary"):
            return
        assistant_msgs = [msg for msg in final_messages if msg["role"] == "assistant"]
        if not assistant_msgs:
            return
        last_assistant = assistant_msgs[-1]["content"]
        try:
            update_paper_llm_summary(paper_id, last_assistant)
        except Exception as e:
            print(f"保存 LLM 总结失败: {e}")

    # -------------------- 核心 --------------------
    def send_message(self, text, mode, think_checked):
        if self.processing:
            return
        text = text.strip()
        if not text:
            return

        if self.current_conversation is None:
            conv = self.history_manager.create_conversation(title="新对话", group_id=None)
            self.current_conversation = conv
            self.conversation_switched.emit()

        # 快捷指令
        if text == "早上好 arxiv":
            text = "早上好 arxiv，今天的新文章"
        elif text == "近期文章 arxiv":
            text = "近期文章 arxiv，请整理一下"

        self.new_message.emit("user", text)
        self.current_conversation.add_message("user", text)

        system_prompt = self._get_current_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        for msg in self.current_conversation.messages:
            messages.append(msg.copy())
        messages = self._clean_messages_for_api(messages)

        # 关键修复：告知 UI 创建新气泡
        self.new_bubble.emit()

        if mode == "compare":
            self._run_comparison(messages)
        else:
            model_name = self._get_model_name_by_mode(mode)
            extra_params = {
                "model": model_name,
                "temperature": 0.3,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "enable_thinking": think_checked,
                "reasoning_effort": "high" if think_checked else None
            }
            self._run_single_model(messages, mode, extra_params)

    def _run_single_model(self, messages, mode, extra_params):
        self.processing = True
        self.update_send_input.emit(False)
        self.update_stop_button.emit(True)
        self.update_regenerate_button.emit(False)

        self.worker = Worker(messages, mode, extra_params=extra_params)
        self.worker.stream_chunk.connect(self._on_stream_chunk)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_stream_chunk(self, sender, chunk):
        if chunk == "\x1cNEW_BUBBLE":
            # 工具调用产生的新气泡
            self.new_bubble.emit()
            self._last_ai_message_id = None
            return
        if sender != "assistant" or chunk == "":
            return

        if self._last_ai_message_id is None:
            self.stream_chunk.emit("assistant", chunk)
            self._last_ai_message_id = "assistant"
            if self.current_conversation:
                self.current_conversation.add_message("assistant", chunk)
        else:
            self.stream_chunk.emit("assistant", chunk)
            if self.current_conversation and self.current_conversation.messages:
                last_msg = self.current_conversation.messages[-1]
                if last_msg["role"] == "assistant":
                    last_msg["content"] += chunk
                    self.current_conversation.updated_at = datetime.now().isoformat()

    def _on_worker_finished(self, reply, error, final_messages, bubble_list):
        self.processing = False
        self.update_send_input.emit(True)
        self.update_stop_button.emit(False)
        self.update_regenerate_button.emit(True)
        self._last_ai_message_id = None

        if error:
            self.new_message.emit("system", f"错误：{error}")
        else:
            if self.current_conversation:
                if bubble_list:
                    user_indices = [i for i, msg in enumerate(self.current_conversation.messages) if msg["role"] == "user"]
                    if user_indices:
                        last_user_idx = user_indices[-1]
                        self.current_conversation.messages = self.current_conversation.messages[:last_user_idx + 1]
                    for bubble in bubble_list:
                        self.current_conversation.add_message("assistant", bubble)
                    self.current_conversation.updated_at = datetime.now().isoformat()
                    if self.current_conversation.title == "新对话":
                        new_title = self.history_manager.generate_title_from_messages(self.current_conversation)
                        if new_title:
                            self.current_conversation.title = new_title
                    self.history_manager.save_conversation(self.current_conversation)
                else:
                    new_messages = [msg for msg in final_messages if msg["role"] != "system"]
                    self.current_conversation.messages = new_messages
                    self.current_conversation.updated_at = datetime.now().isoformat()
                    if self.current_conversation.title == "新对话":
                        new_title = self.history_manager.generate_title_from_messages(self.current_conversation)
                        if new_title:
                            self.current_conversation.title = new_title
                    self.history_manager.save_conversation(self.current_conversation)

        self._save_llm_summary(final_messages)
        self.worker_finished.emit(reply, error, final_messages, bubble_list)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    # -------------------- 对比模式 --------------------
    def _run_comparison(self, messages):
        if self.processing:
            return
        self.processing = True
        self.update_send_input.emit(False)
        self.update_stop_button.emit(True)
        self.update_regenerate_button.emit(False)

        params_a = {
            "model": self._resolve_model_name(self.compare_config_a["model"]),
            "temperature": self.compare_config_a["temperature"],
            "top_p": self.compare_config_a["top_p"],
            "frequency_penalty": self.compare_config_a["frequency_penalty"],
            "presence_penalty": self.compare_config_a["presence_penalty"],
            "enable_thinking": self.compare_config_a["thinking"],
            "reasoning_effort": "high" if self.compare_config_a["thinking"] else None
        }
        params_b = {
            "model": self._resolve_model_name(self.compare_config_b["model"]),
            "temperature": self.compare_config_b["temperature"],
            "top_p": self.compare_config_b["top_p"],
            "frequency_penalty": self.compare_config_b["frequency_penalty"],
            "presence_penalty": self.compare_config_b["presence_penalty"],
            "enable_thinking": self.compare_config_b["thinking"],
            "reasoning_effort": "high" if self.compare_config_b["thinking"] else None
        }

        self.new_message.emit("system", "⚖️ 对比模式开始，将先后展示两个模型的回复...")
        self._comparison_index = 0
        self._comparison_params = [params_a, params_b]
        self._comparison_messages = messages.copy()
        self._run_next_comparison()

    def _run_next_comparison(self):
        if self._comparison_index >= len(self._comparison_params):
            self.processing = False
            self.update_send_input.emit(True)
            self.update_stop_button.emit(False)
            self.update_regenerate_button.emit(True)
            if self.current_conversation:
                self.history_manager.save_conversation(self.current_conversation)
            return

        # 每个模型回复前也发送新气泡信号
        self.new_bubble.emit()

        model_label = "模型 A" if self._comparison_index == 0 else "模型 B"
        self.new_message.emit("system", f"🤖 {model_label} 开始生成...")
        extra_params = self._comparison_params[self._comparison_index]
        msgs_copy = [msg.copy() for msg in self._comparison_messages]
        self.worker = Worker(msgs_copy, "custom", extra_params=extra_params)
        self.worker.stream_chunk.connect(self._on_comparison_chunk)
        self.worker.finished.connect(lambda reply, error, final_messages, _: self._on_comparison_finished(reply, error, final_messages))
        self.worker.start()

    def _on_comparison_chunk(self, sender, chunk):
        if chunk == "\x1cNEW_BUBBLE":
            self.new_bubble.emit()
            return
        prefix = "🔵 模型 A: " if self._comparison_index == 0 else "🟢 模型 B: "
        if self._comparison_last_bubble is None:
            self.comparison_chunk.emit(prefix + chunk)
            self._comparison_last_bubble = True
            if self.current_conversation:
                self.current_conversation.add_message("assistant", prefix + chunk)
        else:
            self.comparison_chunk.emit(chunk)
            if self.current_conversation and self.current_conversation.messages:
                last_msg = self.current_conversation.messages[-1]
                if last_msg["role"] == "assistant":
                    last_msg["content"] += chunk

    def _on_comparison_finished(self, reply, error, final_messages):
        if error:
            self.new_message.emit("system", f"对比模型生成出错: {error}")
        else:
            self.comparison_finished.emit(reply, error, final_messages)
        self._comparison_index += 1
        self._comparison_last_bubble = None
        self._run_next_comparison()

    # -------------------- 对话管理 --------------------
    def regenerate_last_message(self, mode, think_checked):
        if self.processing or not self.current_conversation:
            return
        user_idx = -1
        for i in range(len(self.current_conversation.messages) - 1, -1, -1):
            if self.current_conversation.messages[i]["role"] == "user":
                user_idx = i
                break
        if user_idx == -1:
            self.new_message.emit("system", "没有用户消息可供重新生成")
            return
        if self.current_conversation.messages and self.current_conversation.messages[-1]["role"] == "assistant":
            self.current_conversation.messages.pop()
        system_prompt = self._get_current_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        for msg in self.current_conversation.messages:
            messages.append(msg.copy())
        messages = self._clean_messages_for_api(messages)

        # 重新生成也需要强制新气泡
        self.new_bubble.emit()

        if mode == "compare":
            while self.current_conversation.messages and self.current_conversation.messages[-1]["role"] == "assistant":
                self.current_conversation.messages.pop()
            self._run_comparison(messages)
        else:
            model_name = self._get_model_name_by_mode(mode)
            extra_params = {
                "model": model_name,
                "temperature": 0.3,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "enable_thinking": think_checked,
                "reasoning_effort": "high" if think_checked else None
            }
            self._run_single_model(messages, mode, extra_params)

    def stop_generation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.new_message.emit("system", "生成已中止")

    def switch_conversation(self, conv_id):
        if self.current_conversation and self.current_conversation.conv_id == conv_id:
            return
        if self.current_conversation:
            self.history_manager.save_conversation(self.current_conversation)
        conv = self.history_manager.get_conversation(conv_id)
        if conv:
            self.current_conversation = conv
            self.clear_chat.emit()
            for msg in conv.messages:
                if msg["role"] in ("user", "assistant", "system") and msg.get("content", "").strip():
                    self.new_message.emit(msg["role"], msg["content"])
            self.conversation_switched.emit()
            self.update_regenerate_button.emit(any(m["role"] == "user" for m in conv.messages))

    def create_new_conversation(self, group_id=None):
        if self.current_conversation:
            self.history_manager.save_conversation(self.current_conversation)
        conv = self.history_manager.create_conversation(title="新对话", group_id=group_id)
        self.current_conversation = conv
        self.clear_chat.emit()
        self.new_message.emit("system", "新对话已创建。")
        self.conversation_switched.emit()
        self.update_regenerate_button.emit(False)

    def delete_conversation(self, conv_id):
        if conv_id == self.current_conversation.conv_id:
            all_convs = list(self.history_manager.conversations.keys())
            if all_convs and all_convs[0] != conv_id:
                self.switch_conversation(all_convs[0])
            else:
                self.create_new_conversation()
        self.history_manager.delete_conversation(conv_id)

    def get_history_manager(self):
        return self.history_manager