"""
ui.conversation_controller — QAgent 适配 WAgent 对话隔离范式

变更点（相对旧版）：
1. 所有 signal 都带 conv_id → chat_window 按 current_conv_id 过滤
2. 引入 _active_workers 字典 + _busy_conversation_ids 集合
3. 流式开始前预分配一条空 assistant 消息 + 触发 new_bubble（让 UI 准备好"流式槽"）
4. worker.finished 时触发 bubble_finalize(conv_id) → 通知 UI 调 MessageBubble.finalize()
5. 用 self.sender() 拿到是哪个 worker 在回调
"""
import logging  # noqa: E402

logger = logging.getLogger(__name__)


import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

from PySide6.QtCore import QObject, Signal  # noqa: E402

import config  # noqa: E402
from agent.history_manager import HistoryManager  # noqa: E402
from agent.prompts import SYSTEM_PROMPT  # noqa: E402
from agent.tools import get_paper_by_id, update_paper_llm_summary  # noqa: E402
from config.settings import get_settings  # noqa: E402
from ui.worker import NEW_BUBBLE, Worker  # noqa: E402


class ConversationController(QObject):
    # ---- 所有带 conv_id 的 signal（chat_window 按 current_conv_id 过滤）----
    new_message = Signal(str, str, str)        # (conv_id, role, text)
    stream_chunk = Signal(str, str)            # (conv_id, chunk)  —— chunk 可以是 \x1cNEW_BUBBLE
    new_bubble = Signal(str)                   # (conv_id)  —— 预分配/换气泡
    bubble_finalize = Signal(str)              # (conv_id)  —— 流式结束，让 UI 调 MessageBubble.finalize()
    comparison_chunk = Signal(str, str)        # (conv_id, chunk)
    clear_chat = Signal()                      # 清空（无 conv_id）
    scroll_to_bottom = Signal(str)             # (conv_id)
    update_regenerate_button = Signal(bool)
    update_stop_button = Signal(bool)
    update_send_input = Signal(bool)
    conversation_switched = Signal()           # 切换完成

    # ---- 旧 worker_finished 保留兼容（部分代码可能还连着）----
    worker_finished = Signal(object, str, list, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_manager = HistoryManager()
        self.current_conversation = None
        # 旧单一槽位（保留兼容，新代码用 _active_workers）
        self.processing = False
        self.worker = None
        self._last_ai_message_id = None

        # 新：per-conv worker 跟踪
        self._active_workers = {}               # worker -> Conversation
        self._busy_conversation_ids = set()
        self._comparison_index = 0
        self._comparison_params = []
        # 2026-09-07 P0 fix: 流式节流 save 时间戳 (每 30s 存一次, 防段错误丢数据)
        self._last_stream_save_ts = 0.0
        self._comparison_modes = []
        self._comparison_messages = []
        self._comparison_target_conv = None
        self._comparison_last_bubble = None

        settings = get_settings()
        self.default_system_prompt = settings.value("default_system_prompt", SYSTEM_PROMPT)
        self._load_comparison_config()

    def _load_comparison_config(self):
        settings = get_settings()
        try:
            self.compare_config_a = json.loads(settings.value("comparison/config_a",
                '{"model":"deepseek-flash","thinking":true,"temperature":0.3,"top_p":0.9,"frequency_penalty":0.0,"presence_penalty":0.0}'))
            self.compare_config_b = json.loads(settings.value("comparison/config_b",
                '{"model":"deepseek-pro","thinking":true,"temperature":0.3,"top_p":0.9,"frequency_penalty":0.0,"presence_penalty":0.0}'))
        except Exception:
            self.compare_config_a = {"model": "deepseek-flash", "thinking": True, "temperature": 0.3, "top_p": 0.9,
                                     "frequency_penalty": 0.0, "presence_penalty": 0.0}
            self.compare_config_b = {"model": "deepseek-pro", "thinking": True, "temperature": 0.3, "top_p": 0.9,
                                     "frequency_penalty": 0.0, "presence_penalty": 0.0}

    # ============================================================
    # 工具
    # ============================================================
    def _get_current_system_prompt(self):
        if self.current_conversation and self.current_conversation.group_id:
            group_id = self.current_conversation.group_id
            settings = get_settings()
            group_prompt = settings.value(f"group_system_prompt/{group_id}")
            if group_prompt:
                return group_prompt
        return self.default_system_prompt

    def _clean_messages_for_api(self, messages):
        # 2026-09-04 加:对 assistant 消息的 content 跑一次 <think> 标签剥离
        # 背景:之前老对话(MiniMax-M3 thinking 漏 <think> 标签)的 content
        # 字段里直含 thinking 文本,送回 LLM 会让 LLM "看到" 自己的旧 thinking
        # 污染下一轮回答。剥离后再送 LLM。
        from agent.controller import _strip_think_tags_from_text
        cleaned = []
        for msg in messages:
            new_msg = msg.copy()
            if new_msg.get("role") == "tool" and "tool_call_id" not in new_msg:
                new_msg["role"] = "assistant"
                new_msg.pop("tool_calls", None)
            # 2026-09-04 改:对所有 role 走一遍剥离(用户/工具/system 也可能含
            # 泄漏的 thinking, 不限 assistant)
            content = new_msg.get("content")
            if isinstance(content, str) and content:
                stripped = _strip_think_tags_from_text(content)
                if stripped != content:
                    new_msg["content"] = stripped
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
                except Exception:
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

    # ============================================================
    # 附件 → LLM 消息拼接（懒加载 + 避免拉起 llm）
    # ============================================================
    _compose_llm_user_content = None  # 懒加载

    @classmethod
    def _llm_compose(cls, text: str, attachments: list) -> str:
        """
        把附件文本拼到 user 消息后面（仅本轮发 LLM，不入库）。
        attachments: [{name, size, path, content}, ...]  —— content 来自 InputPanel 懒加载
        """
        if cls._compose_llm_user_content is None:
            # 绕开 agent.tools.__init__ 的级联 import（避免拉起 llm.py）
            import importlib.util as _ilu
            import os as _os
            _p = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)),
                "..", "agent", "tools", "attachment_reader.py",
            )
            _spec = _ilu.spec_from_file_location("_att_reader", _p)
            _m = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_m)
            cls._compose_llm_user_content = _m.compose_llm_user_content
        return cls._compose_llm_user_content(text, attachments)

    def _update_ui_for_busy(self, conv_id):
        """刷新输入区按钮的可用状态（per-conv）"""
        busy = conv_id in self._busy_conversation_ids
        if self.current_conversation and self.current_conversation.conv_id == conv_id:
            self.update_send_input.emit(not busy)
            self.update_stop_button.emit(busy)
            has_user = any(m["role"] == "user" for m in self.current_conversation.messages)
            self.update_regenerate_button.emit(has_user and not busy)

    # ============================================================
    # 发送消息
    # ============================================================
    def send_message(self, text, mode, think_checked, attachments=None):
        text = (text or "").strip()
        attachments = attachments or []
        if not text and not attachments:
            return

        if self.current_conversation is None:
            conv = self.history_manager.create_conversation(title="新对话", group_id=None)
            self.current_conversation = conv
            self.conversation_switched.emit()

        if self.current_conversation.conv_id in self._busy_conversation_ids:
            self.new_message.emit(self.current_conversation.conv_id, "system", "当前对话正在生成，请等待完成")
            return

        # 快捷指令
        if text == "早上好 arxiv":
            text = "早上好 arxiv，今天的新文章"
        elif text == "近期文章 arxiv":
            text = "近期文章 arxiv，请整理一下"

        # 用户消息（持久化时只存元数据）
        conv_id = self.current_conversation.conv_id
        attachments_meta = [
            {"name": a.get("name", ""), "size": a.get("size", ""), "path": a.get("path", "")}
            for a in attachments
        ]
        self.new_message.emit(conv_id, "user", text)
        self.current_conversation.add_message("user", text, attachments=attachments_meta)

        # 预分配空气泡（流式槽）—— 跟 WAgent 一样
        self.current_conversation.add_message("assistant", "")
        self.new_bubble.emit(conv_id)

        system_prompt = self._get_current_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        # ★ 拼 user 消息时把附件 content 临时展开（不入库，content 只走 LLM 路径）
        for i, msg in enumerate(self.current_conversation.messages):
            if msg["role"] == "user" and i == len(self.current_conversation.messages) - 2 and attachments:
                # 找到刚刚 add 的那条 user 消息 → 展开附件
                content = self._llm_compose(text=text, attachments=attachments)
                messages.append({"role": "user", "content": content})
            else:
                messages.append(msg.copy())
        messages = self._clean_messages_for_api(messages)

        if mode == "compare":
            self._run_comparison(messages, self.current_conversation)
        else:
            model_name = self._get_model_name_by_mode(mode)
            extra_params = {
                "model": model_name,
                "temperature": 0.3,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "enable_thinking": think_checked,
                "reasoning_effort": "high" if think_checked else None,
            }
            self._run_single_model(messages, mode, extra_params, self.current_conversation)

    # ============================================================
    # 单模型
    # ============================================================
    def _run_single_model(self, messages, mode, extra_params, target_conv):
        conv_id = target_conv.conv_id
        self._busy_conversation_ids.add(conv_id)
        self._update_ui_for_busy(conv_id)

        worker = Worker(conv_id, messages, mode, extra_params=extra_params)
        self._active_workers[worker] = target_conv
        worker.stream_chunk.connect(self._on_stream_chunk)
        worker.finished.connect(self._on_worker_finished)
        # 兼容旧单一槽位
        self.worker = worker
        self.processing = True
        worker.start()

    def _on_stream_chunk(self, conv_id, chunk):
        """Worker.stream_chunk (conv_id, chunk) 回调"""
        # sender 校验：双保险（防止跨 conv 误投）
        worker = self.sender()
        target_conv = self._active_workers.get(worker)
        if target_conv is not None and target_conv.conv_id != conv_id:
            return  # signal 带的 conv_id 跟 worker 登记的不一致 → 丢弃

        if chunk == NEW_BUBBLE:
            # 切气泡：把当前累积内容冻结到 conv.messages[-1]，并通知 UI 开新气泡
            self.new_bubble.emit(conv_id)
            return
        if not chunk:
            return

        if target_conv is None:
            return  # 找不到 conv（理论不应发生）

        # 追加到当前对话最后一条 assistant 消息
        if target_conv.messages and target_conv.messages[-1]["role"] == "assistant":
            target_conv.messages[-1]["content"] += chunk
            target_conv.updated_at = datetime.now().isoformat()
        else:
            target_conv.add_message("assistant", chunk)
        self.stream_chunk.emit(conv_id, chunk)

        # 2026-09-07 P0 fix: 流式过程中节流 save (每 30s 一次)
        # 之前: 段错误 (0xC0000005/0xC00000FD) 崩进程时 _on_worker_finished 不跑 → save_conversation 不调
        # → 用户整段对话没存盘, 看起来"对话消失"
        # 修法: 流式每 30s 强制 save 一次, 段错误最多丢 30s 数据
        now_ts = time.monotonic()
        if now_ts - self._last_stream_save_ts > 30:
            try:
                self.history_manager.save_conversation(target_conv)
                self._last_stream_save_ts = now_ts
            except Exception as e:
                logger.warning("[controller] 流式节流 save 失败: %s", e)

    def _on_worker_finished(self, conv_id, full_reply, error, final_messages, bubble_list):
        """Worker.finished (conv_id, full_reply, error, final_messages, bubble_list) 回调"""
        worker = self.sender()
        target_conv = self._active_workers.pop(worker, None)
        if target_conv is None:
            return
        # 二次校验 conv_id 一致
        if target_conv.conv_id != conv_id:
            return

        if conv_id in self._busy_conversation_ids:
            self._busy_conversation_ids.remove(conv_id)

        # 错误：通知 UI（系统消息）并 finalize
        if error:
            self.new_message.emit(conv_id, "system", f"错误：{error}")
            self.bubble_finalize.emit(conv_id)
        else:
            # 正常：把 bubble_list 里"切气泡"产生的多段内容落到 conv.messages
            if bubble_list and len(bubble_list) > 1:
                # 切掉最后一条累积的 assistant 消息（流式累的全文），换成 bubble_list 的多段
                user_indices = [i for i, m in enumerate(target_conv.messages) if m["role"] == "user"]
                user_indices[-1] if user_indices else -1
                # 删掉最后一个 assistant（流式累的）
                while target_conv.messages and target_conv.messages[-1]["role"] == "assistant":
                    target_conv.messages.pop()
                for b in bubble_list:
                    target_conv.add_message("assistant", b)
            # 不管切不切气泡，最后一条 assistant 消息内容 == full_reply（流式累的），
            # 上面的 bubble_list 处理已经覆盖了"多段"的情况
            target_conv.updated_at = datetime.now().isoformat()
            if target_conv.title == "新对话":
                new_title = self.history_manager.generate_title_from_messages(target_conv)
                if new_title:
                    target_conv.title = new_title
            self.history_manager.save_conversation(target_conv)

        # 让 UI 收尾（finalize 最后一个气泡 → 触发 typeset + 收缩宽度）
        self.bubble_finalize.emit(conv_id)

        if self.current_conversation and self.current_conversation.conv_id == conv_id:
            self.scroll_to_bottom.emit(conv_id)
        self._update_ui_for_busy(conv_id)

        # 旧 worker_finished 兼容
        self.worker_finished.emit(full_reply, error or "", final_messages, bubble_list)

        if worker:
            worker.deleteLater()
        if self.worker is worker:
            self.worker = None
            self.processing = False

    # ============================================================
    # 对比模式
    # ============================================================
    def _run_comparison(self, messages, target_conv):
        conv_id = target_conv.conv_id
        params_a = {
            "temperature": self.compare_config_a["temperature"],
            "top_p": self.compare_config_a["top_p"],
            "frequency_penalty": self.compare_config_a["frequency_penalty"],
            "presence_penalty": self.compare_config_a["presence_penalty"],
            "enable_thinking": self.compare_config_a["thinking"],
            "reasoning_effort": "high" if self.compare_config_a["thinking"] else None,
        }
        params_b = {
            "temperature": self.compare_config_b["temperature"],
            "top_p": self.compare_config_b["top_p"],
            "frequency_penalty": self.compare_config_b["frequency_penalty"],
            "presence_penalty": self.compare_config_b["presence_penalty"],
            "enable_thinking": self.compare_config_b["thinking"],
            "reasoning_effort": "high" if self.compare_config_b["thinking"] else None,
        }

        self._busy_conversation_ids.add(conv_id)
        self._update_ui_for_busy(conv_id)
        self.new_message.emit(conv_id, "system", "⚖️ 对比模式开始，将先后展示两个模型的回复...")
        self._comparison_index = 0
        self._comparison_params = [params_a, params_b]
        self._comparison_modes = [self._resolve_model_name(self.compare_config_a["model"]),
                                  self._resolve_model_name(self.compare_config_b["model"])]
        self._comparison_messages = messages
        self._comparison_target_conv = target_conv
        self._comparison_last_bubble = None
        self._run_next_comparison()

    def _run_next_comparison(self):
        if self._comparison_index >= len(self._comparison_params):
            conv = self._comparison_target_conv
            if conv and conv.conv_id in self._busy_conversation_ids:
                self._busy_conversation_ids.remove(conv.conv_id)
            if conv:
                self.history_manager.save_conversation(conv)
                self.bubble_finalize.emit(conv.conv_id)
            if self.current_conversation and self.current_conversation.conv_id == conv.conv_id:
                self._update_ui_for_busy(conv.conv_id)
            return

        target_conv = self._comparison_target_conv
        # 每个模型前预分配空气泡
        target_conv.add_message("assistant", "")
        self.new_bubble.emit(target_conv.conv_id)

        model_label = "模型 A" if self._comparison_index == 0 else "模型 B"
        self.new_message.emit(target_conv.conv_id, "system", f"🤖 {model_label} 开始生成...")

        mode = self._comparison_modes[self._comparison_index]
        extra_params = self._comparison_params[self._comparison_index]
        msgs_copy = [msg.copy() for msg in self._comparison_messages]

        worker = Worker(target_conv.conv_id, msgs_copy, mode, extra_params=extra_params)
        self._active_workers[worker] = target_conv
        worker.stream_chunk.connect(self._on_comparison_chunk)
        worker.finished.connect(self._on_comparison_finished)
        worker.start()

    def _on_comparison_chunk(self, conv_id, chunk):
        worker = self.sender()
        target_conv = self._active_workers.get(worker)
        if target_conv is None or target_conv.conv_id != conv_id:
            return

        if chunk == NEW_BUBBLE:
            self.new_bubble.emit(conv_id)
            return
        if not chunk:
            return

        prefix = "🔵 模型 A: " if self._comparison_index == 0 else "🟢 模型 B: "
        if self._comparison_last_bubble is None:
            full_chunk = prefix + chunk
            if target_conv.messages and target_conv.messages[-1]["role"] == "assistant":
                target_conv.messages[-1]["content"] += full_chunk
            else:
                target_conv.add_message("assistant", full_chunk)
            self._comparison_last_bubble = True
        else:
            if target_conv.messages and target_conv.messages[-1]["role"] == "assistant":
                target_conv.messages[-1]["content"] += chunk
            else:
                target_conv.add_message("assistant", chunk)
        self.comparison_chunk.emit(conv_id, chunk if self._comparison_last_bubble else prefix + chunk)

    def _on_comparison_finished(self, conv_id, full_reply, error, final_messages, bubble_list):
        worker = self.sender()
        target_conv = self._active_workers.pop(worker, None)
        if target_conv is None:
            return
        if target_conv.conv_id != conv_id:
            return

        if error:
            self.new_message.emit(conv_id, "system", f"对比模型生成出错: {error}")
        if target_conv:
            self.bubble_finalize.emit(conv_id)
            self.history_manager.save_conversation(target_conv)
        if worker:
            worker.deleteLater()
        self._comparison_index += 1
        self._comparison_last_bubble = None
        self._run_next_comparison()

    # ============================================================
    # 停止 / 重新生成
    # ============================================================
    def stop_generation(self):
        if self.current_conversation is None:
            return
        cur_id = self.current_conversation.conv_id
        stopped = False
        for w, conv in list(self._active_workers.items()):
            if conv.conv_id == cur_id:
                if w.isRunning():
                    w.stop()
                    stopped = True
                break
        if stopped:
            self.new_message.emit(cur_id, "system", "生成已中止")

    def regenerate_last_message(self, mode, think_checked):
        if self.processing or not self.current_conversation:
            return
        if self.current_conversation.conv_id in self._busy_conversation_ids:
            self.new_message.emit(self.current_conversation.conv_id, "system", "当前对话正在生成，请等待完成")
            return
        # 删掉所有尾部 assistant
        while self.current_conversation.messages and self.current_conversation.messages[-1]["role"] == "assistant":
            self.current_conversation.messages.pop()
        # 找最后一条 user
        user_idx = -1
        for i in range(len(self.current_conversation.messages) - 1, -1, -1):
            if self.current_conversation.messages[i]["role"] == "user":
                user_idx = i
                break
        if user_idx == -1:
            self.new_message.emit(self.current_conversation.conv_id, "system", "没有用户消息可供重新生成")
            return
        last_user_text = self.current_conversation.messages[user_idx]["content"]
        # 不直接复用 send_message（send_message 会再次写 user 历史）—— 走专门路径
        self._regenerate_with(last_user_text, mode, think_checked)

    def _regenerate_with(self, text, mode, think_checked):
        """重新生成：保留历史 user 不重复 add，只重发 assistant 段"""
        if self.current_conversation is None:
            return
        conv = self.current_conversation
        conv_id = conv.conv_id
        # 预分配空气泡
        conv.add_message("assistant", "")
        self.new_bubble.emit(conv_id)

        system_prompt = self._get_current_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conv.messages:
            messages.append(msg.copy())
        messages = self._clean_messages_for_api(messages)

        if mode == "compare":
            self._run_comparison(messages, conv)
        else:
            model_name = self._get_model_name_by_mode(mode)
            extra_params = {
                "model": model_name,
                "temperature": 0.3,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "enable_thinking": think_checked,
                "reasoning_effort": "high" if think_checked else None,
            }
            self._run_single_model(messages, mode, extra_params, conv)

    # ============================================================
    # 对话管理
    # ============================================================
    def switch_conversation(self, conv_id):
        if self.current_conversation and self.current_conversation.conv_id == conv_id:
            return
        if self.current_conversation:
            self.history_manager.save_conversation(self.current_conversation)
        conv = self.history_manager.get_conversation(conv_id)
        if conv:
            self.current_conversation = conv
            self.clear_chat.emit()
            # v18:切对话不复读历史里的 system 消息 — user 要求
            # "切对话 = system 消息都消失"(无论是启动欢迎气泡还是历史的错误/对比提示)
            # 仍然复读 user + assistant
            # 2026-09-04 加:复读前剥 <think> 标签,避免老 thinking 污染气泡
            # (用户报"重启后看到 LLM 旧 thinking 整段显示")
            from agent.controller import _strip_think_tags_from_text
            for msg in conv.messages:
                if msg["role"] in ("user", "assistant") and msg.get("content", "").strip():
                    content = msg["content"]
                    stripped = _strip_think_tags_from_text(content)
                    self.new_message.emit(conv_id, msg["role"], stripped)
            self.conversation_switched.emit()
            busy = conv_id in self._busy_conversation_ids
            has_user = any(m["role"] == "user" for m in conv.messages)
            self.update_send_input.emit(not busy)
            self.update_stop_button.emit(busy)
            self.update_regenerate_button.emit(has_user and not busy)

    def create_new_conversation(self, group_id=None):
        if self.current_conversation:
            self.history_manager.save_conversation(self.current_conversation)
        conv = self.history_manager.create_conversation(title="新对话", group_id=group_id)
        self.current_conversation = conv
        self.clear_chat.emit()
        # v16:删掉"新对话已创建。"system 消息 emit — 切新对话不应有 system 气泡显示
        # (user 要求 system 欢迎气泡"切对话就消失",所以 create_new_conversation 不能新建 system 消息)
        # 注意:错误 system 消息(self.new_message.emit(..., "system", f"错误:{error}"))
        #       仍然保留 — 错误提示 user 需要看到
        self.conversation_switched.emit()
        self.update_regenerate_button.emit(False)
        self.update_send_input.emit(True)
        self.update_stop_button.emit(False)

    def delete_conversation(self, conv_id):
        # 中止该 conv 的 worker
        for w, conv in list(self._active_workers.items()):
            if conv.conv_id == conv_id:
                if w.isRunning():
                    w.stop()
                self._active_workers.pop(w, None)
                if conv_id in self._busy_conversation_ids:
                    self._busy_conversation_ids.remove(conv_id)
                break
        # 切走
        if self.current_conversation and conv_id == self.current_conversation.conv_id:
            all_convs = list(self.history_manager.conversations.keys())
            if all_convs and all_convs[0] != conv_id:
                self.switch_conversation(all_convs[0])
            else:
                self.create_new_conversation()
        self.history_manager.delete_conversation(conv_id)
        self._update_ui_for_busy(self.current_conversation.conv_id if self.current_conversation else "")

    def get_history_manager(self):
        return self.history_manager
