# agent/history_manager.py
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import hashlib

HISTORY_DIR = Path("data/conversations")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

class Conversation:
    """单个对话的数据模型"""
    def __init__(self, conv_id: str = None, title: str = "新对话"):
        self.conv_id = conv_id or self._generate_id()
        self.title = title
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.messages = []  # 格式: [{"role": "user"/"assistant"/"system", "content": "..."}]

    def _generate_id(self) -> str:
        """生成唯一ID，基于时间戳和随机数"""
        import time
        import random
        timestamp = int(time.time() * 1000)
        rand = random.randint(1000, 9999)
        return f"{timestamp}-{rand}"

    def add_message(self, role: str, content: str):
        """添加消息并更新更新时间"""
        self.messages.append({"role": role, "content": content})
        self.updated_at = datetime.now().isoformat()

    def update_last_message(self, content: str):
        """更新最后一条消息的内容（用于流式追加）"""
        if self.messages and self.messages[-1]["role"] in ("assistant", "user"):
            self.messages[-1]["content"] = content
            self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "conv_id": self.conv_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages
        }

    @classmethod
    def from_dict(cls, data: dict):
        conv = cls(data["conv_id"], data["title"])
        conv.created_at = data["created_at"]
        conv.updated_at = data["updated_at"]
        conv.messages = data["messages"]
        return conv


class HistoryManager:
    """管理所有对话历史"""
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
        self.current_id: Optional[str] = None
        self.load_all()

    def _get_file_path(self, conv_id: str) -> Path:
        return HISTORY_DIR / f"{conv_id}.json"

    def load_all(self):
        """加载所有对话历史"""
        self.conversations.clear()
        for file in HISTORY_DIR.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conv = Conversation.from_dict(data)
                    self.conversations[conv.conv_id] = conv
            except Exception as e:
                print(f"加载对话历史失败 {file}: {e}")

    def save_conversation(self, conv: Conversation):
        """保存单个对话"""
        file_path = self._get_file_path(conv.conv_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conv.to_dict(), f, ensure_ascii=False, indent=2)

    def delete_conversation(self, conv_id: str):
        print(f"尝试删除对话 {conv_id}")
        if conv_id in self.conversations:
            del self.conversations[conv_id]
        file_path = self._get_file_path(conv_id)
        print(f"文件路径: {file_path}")
        if file_path.exists():
            file_path.unlink()
            print(f"文件已删除")
        else:
            print(f"文件不存在，可能已删除")

    def create_conversation(self, title: str = "新对话") -> Conversation:
        """创建新对话"""
        conv = Conversation(title=title)
        self.conversations[conv.conv_id] = conv
        self.current_id = conv.conv_id
        self.save_conversation(conv)
        return conv

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        return self.conversations.get(conv_id)

    def set_current(self, conv_id: str):
        self.current_id = conv_id

    def get_all_sorted(self) -> List[Conversation]:
        """按更新时间倒序返回所有对话"""
        return sorted(self.conversations.values(),
                     key=lambda x: x.updated_at, reverse=True)

    def generate_title_from_messages(self, conv: Conversation) -> str:
        """根据对话内容自动生成标题（简单规则：取第一条用户消息的前10字）"""
        for msg in conv.messages:
            if msg["role"] == "user":
                title = msg["content"][:20].strip()
                if len(title) > 10:
                    title = title[:10] + "..."
                return title
        return "新对话"