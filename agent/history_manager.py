# agent/history_manager.py
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

HISTORY_DIR = Path("data/conversations")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = HISTORY_DIR / "conversations.db"

class Conversation:
    """单个对话的数据模型"""
    def __init__(self, conv_id: str = None, title: str = "新对话", group_id: str = None):
        self.conv_id = conv_id or self._generate_id()
        self.title = title
        self.group_id = group_id  # 所属分组ID，None 表示未分组
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.messages = []  # 格式: [{"role": "user"/"assistant"/"system", "content": "..."}]

    def _generate_id(self) -> str:
        import time
        import random
        timestamp = int(time.time() * 1000)
        rand = random.randint(1000, 9999)
        return f"{timestamp}-{rand}"

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.updated_at = datetime.now().isoformat()

    def update_last_message(self, content: str):
        if self.messages and self.messages[-1]["role"] in ("assistant", "user"):
            self.messages[-1]["content"] = content
            self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "conv_id": self.conv_id,
            "title": self.title,
            "group_id": self.group_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages
        }

    @classmethod
    def from_dict(cls, data: dict):
        conv = cls(data["conv_id"], data["title"], data.get("group_id"))
        conv.created_at = data["created_at"]
        conv.updated_at = data["updated_at"]
        conv.messages = data["messages"]
        return conv


class Group:
    """分组数据模型"""
    def __init__(self, group_id: str = None, name: str = "新分组"):
        self.group_id = group_id or self._generate_id()
        self.name = name
        self.created_at = datetime.now().isoformat()

    def _generate_id(self) -> str:
        import time
        import random
        timestamp = int(time.time() * 1000)
        rand = random.randint(1000, 9999)
        return f"group-{timestamp}-{rand}"

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: dict):
        group = cls(data["group_id"], data["name"])
        group.created_at = data["created_at"]
        return group


class HistoryManager:
    """管理对话和分组，使用 SQLite 存储"""
    def __init__(self):
        self._init_db()
        self._load_cache()

    def _get_connection(self):
        return sqlite3.connect(str(DB_PATH))

    def _init_db(self):
        with self._get_connection() as conn:
            c = conn.cursor()
            # 分组表
            c.execute('''
                CREATE TABLE IF NOT EXISTS groups (
                    group_id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT
                )
            ''')
            # 对话表
            c.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    conv_id TEXT PRIMARY KEY,
                    group_id TEXT,
                    title TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    FOREIGN KEY (group_id) REFERENCES groups(group_id) ON DELETE SET NULL
                )
            ''')
            # 消息表
            c.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id TEXT,
                    role TEXT,
                    content TEXT,
                    message_data TEXT,
                    created_at TEXT,
                    FOREIGN KEY (conv_id) REFERENCES conversations(conv_id) ON DELETE CASCADE
                )
            ''')
            conn.commit()

    def _load_cache(self):
        """加载所有分组和对话元数据到内存（用于快速显示）"""
        self.groups: Dict[str, dict] = {}      # group_id -> {name, created_at}
        self.conversations: Dict[str, dict] = {}  # conv_id -> {title, group_id, updated_at}
        with self._get_connection() as conn:
            c = conn.cursor()
            # 加载分组
            c.execute("SELECT group_id, name, created_at FROM groups")
            for row in c.fetchall():
                self.groups[row[0]] = {"name": row[1], "created_at": row[2]}
            # 加载对话元数据
            c.execute("SELECT conv_id, group_id, title, updated_at FROM conversations")
            for row in c.fetchall():
                self.conversations[row[0]] = {
                    "group_id": row[1],
                    "title": row[2],
                    "updated_at": row[3]
                }

    # ---------- 分组操作 ----------
    def create_group(self, name: str = "新分组") -> Group:
        group = Group(name=name)
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("INSERT INTO groups (group_id, name, created_at) VALUES (?, ?, ?)",
                      (group.group_id, group.name, group.created_at))
            conn.commit()
        self.groups[group.group_id] = {"name": group.name, "created_at": group.created_at}
        return group

    def rename_group(self, group_id: str, new_name: str):
        if group_id not in self.groups:
            return
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("UPDATE groups SET name = ? WHERE group_id = ?", (new_name, group_id))
            conn.commit()
        self.groups[group_id]["name"] = new_name

    def delete_group(self, group_id: str, move_to_group_id: str = None):
        """删除分组，如果 move_to_group_id 指定，则将该分组下的对话移动到目标分组，否则移动到 None（未分组）"""
        with self._get_connection() as conn:
            c = conn.cursor()
            # 移动对话
            c.execute("UPDATE conversations SET group_id = ? WHERE group_id = ?",
                      (move_to_group_id, group_id))
            # 删除分组
            c.execute("DELETE FROM groups WHERE group_id = ?", (group_id,))
            conn.commit()
        # 更新缓存
        if group_id in self.groups:
            del self.groups[group_id]
        for conv_id, meta in self.conversations.items():
            if meta["group_id"] == group_id:
                meta["group_id"] = move_to_group_id

    def get_all_groups(self) -> List[dict]:
        """返回所有分组信息（用于左侧列表）"""
        return [{"group_id": gid, **info} for gid, info in self.groups.items()]

    # ---------- 对话操作 ----------
    def create_conversation(self, title="新对话", group_id: str = None) -> Conversation:
        conv = Conversation(title=title, group_id=group_id)
        self.save_conversation(conv)
        return conv

    def save_conversation(self, conv: Conversation):
        # 将 group_id 统一为 None 或有效字符串
        group_id = conv.group_id if conv.group_id else None
        with self._get_connection() as conn:
            c = conn.cursor()
            # 插入或替换对话元数据
            c.execute('''
                INSERT OR REPLACE INTO conversations (conv_id, group_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (conv.conv_id, group_id, conv.title, conv.created_at, conv.updated_at))
            # 删除该对话的所有现有消息
            c.execute("DELETE FROM messages WHERE conv_id = ?", (conv.conv_id,))
            # 插入所有消息
            for msg in conv.messages:
                msg_json = json.dumps(msg, ensure_ascii=False)
                c.execute('''
                    INSERT INTO messages (conv_id, role, content, message_data, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (conv.conv_id, msg.get("role", ""), msg.get("content", ""), msg_json, datetime.now().isoformat()))
            conn.commit()
        # 更新缓存
        self.conversations[conv.conv_id] = {
            "group_id": group_id,
            "title": conv.title,
            "updated_at": conv.updated_at
        }

    def delete_conversation(self, conv_id: str):
        print(f"准备删除对话 {conv_id}")
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM conversations WHERE conv_id = ?", (conv_id,))
            conn.commit()
            # 检查是否删除成功
            c.execute("SELECT COUNT(*) FROM conversations WHERE conv_id = ?", (conv_id,))
            count = c.fetchone()[0]
            print(f"删除后 conversations 表中该 ID 剩余记录数: {count}")
        if conv_id in self.conversations:
            del self.conversations[conv_id]
            print("已从缓存中删除")
        else:
            print("缓存中未找到该 ID")

    def move_conversation(self, conv_id: str, target_group_id: str):
        """移动对话到指定分组（target_group_id 可为 None 表示未分组）"""
        if conv_id not in self.conversations:
            return
        # 将假值转换为 None
        if not target_group_id:
            target_group_id = None
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("UPDATE conversations SET group_id = ? WHERE conv_id = ?",
                      (target_group_id, conv_id))
            conn.commit()
        self.conversations[conv_id]["group_id"] = target_group_id

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        with self._get_connection() as conn:
            c = conn.cursor()
            # 获取对话元数据
            c.execute("SELECT group_id, title, created_at, updated_at FROM conversations WHERE conv_id = ?", (conv_id,))
            row = c.fetchone()
            if not row:
                return None
            group_id, title, created_at, updated_at = row
            # 获取所有消息
            c.execute("SELECT message_data FROM messages WHERE conv_id = ? ORDER BY id", (conv_id,))
            msg_rows = c.fetchall()
            messages = [json.loads(row[0]) for row in msg_rows]

        conv = Conversation(conv_id, title, group_id)
        conv.created_at = created_at
        conv.updated_at = updated_at
        conv.messages = messages
        return conv

    def get_conversations_by_group(self) -> Dict[Optional[str], List[dict]]:
        result = {}
        for conv_id, meta in self.conversations.items():
            group_id = meta["group_id"]
            # 将假值（None, False, 0, "" 等）统一为 None
            if not group_id:
                group_id = None
            if group_id not in result:
                result[group_id] = []
            result[group_id].append({"conv_id": conv_id, **meta})
        for group_id in result:
            result[group_id].sort(key=lambda x: x["updated_at"], reverse=True)
        return result

    def generate_title_from_messages(self, conv: Conversation) -> str:
        for msg in conv.messages:
            if msg["role"] == "user":
                title = msg["content"][:20].strip()
                if len(title) > 10:
                    title = title[:10] + "..."
                return title
        return "新对话"