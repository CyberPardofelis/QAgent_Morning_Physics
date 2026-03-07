# agent/history_manager.py
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

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
        self.conversations: Dict[str, dict] = {}  # 只存储元数据
        self.current_id: Optional[str] = None
        self.index_path = HISTORY_DIR / "index.json"
        self._load_index()

    def _load_index(self):
        """加载索引文件（快速）"""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # data 应为 [{"conv_id": "...", "title": "...", "updated_at": "..."}, ...]
                    for item in data:
                        self.conversations[item["conv_id"]] = item
            except Exception as e:
                print(f"加载索引失败: {e}，将重建索引")
                self._rebuild_index()
        else:
            self._rebuild_index()

    def _rebuild_index(self):
        """扫描所有对话文件，重建索引（通常在文件损坏或首次运行时调用）"""
        self.conversations.clear()
        for file in HISTORY_DIR.glob("*.json"):
            if file.name == "index.json":
                continue
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 只提取元数据
                    meta = {
                        "conv_id": data["conv_id"],
                        "title": data["title"],
                        "updated_at": data["updated_at"]
                    }
                    self.conversations[data["conv_id"]] = meta
            except Exception as e:
                print(f"重建索引时跳过文件 {file}: {e}")
        self._save_index()

    def _save_index(self):
        """保存索引到文件"""
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(list(self.conversations.values()), f, ensure_ascii=False, indent=2)

    def create_conversation(self, title="新对话") -> Conversation:
        conv = Conversation(title=title)
        # 保存完整对话内容到文件
        self.save_conversation(conv)
        # 更新索引
        self.conversations[conv.conv_id] = {
            "conv_id": conv.conv_id,
            "title": conv.title,
            "updated_at": conv.updated_at
        }
        self._save_index()
        return conv

    def save_conversation(self, conv: Conversation):
        """保存完整对话内容"""
        file_path = HISTORY_DIR / f"{conv.conv_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conv.to_dict(), f, ensure_ascii=False, indent=2)
        # 更新索引中的元数据（如果存在）
        if conv.conv_id in self.conversations:
            self.conversations[conv.conv_id].update({
                "title": conv.title,
                "updated_at": conv.updated_at
            })
        else:
            self.conversations[conv.conv_id] = {
                "conv_id": conv.conv_id,
                "title": conv.title,
                "updated_at": conv.updated_at
            }
        self._save_index()

    def delete_conversation(self, conv_id: str):
        """删除对话"""
        if conv_id in self.conversations:
            del self.conversations[conv_id]
        file_path = HISTORY_DIR / f"{conv_id}.json"
        if file_path.exists():
            file_path.unlink()
        self._save_index()

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """按需加载完整对话内容"""
        file_path = HISTORY_DIR / f"{conv_id}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return Conversation.from_dict(data)
            except Exception as e:
                print(f"加载对话 {conv_id} 失败: {e}")
        return None

    def get_all_sorted(self) -> List[dict]:
        """返回元数据列表，按更新时间排序"""
        return sorted(self.conversations.values(),
                      key=lambda x: x["updated_at"], reverse=True)

    def _get_file_path(self, conv_id: str) -> Path:
        return HISTORY_DIR / f"{conv_id}.json"

    '''
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
    # '''

    def set_current(self, conv_id: str):
        self.current_id = conv_id

    def generate_title_from_messages(self, conv: Conversation) -> str:
        """根据对话内容自动生成标题（简单规则：取第一条用户消息的前10字）"""
        for msg in conv.messages:
            if msg["role"] == "user":
                title = msg["content"][:20].strip()
                if len(title) > 10:
                    title = title[:10] + "..."
                return title
        return "新对话"