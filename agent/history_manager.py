"""
agent.history_manager — 对话历史持久化（参考 WAgent 重构版）

要点：
- 用稳定 msg_id 做 UPSERT，替代 DELETE+INSERT（长对话性能大幅提升）
- BEGIN IMMEDIATE 事务包裹：中途失败自动 ROLLBACK
- 增量删除：只删不在当前 msg_ids 集合中的旧消息
- 附件元数据：add_message 接受 attachments=[]，持久化时只存元数据（不存附件文本）
- 兼容 QAgent 原接口：Conversation / Group / HistoryManager 类签名不变
"""
import json
import logging
import random
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# 数据库位置：保持与原 QAgent 一致（兼容旧数据）
HISTORY_DIR = Path("data/conversations")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = HISTORY_DIR / "conversations.db"


# ============================================================
# Conversation / Group 数据模型
# ============================================================
class Conversation:
    """单个对话的数据模型"""

    def __init__(
        self,
        conv_id: str = None,
        title: str = "新对话",
        group_id: str = None,
        msg_ids: list[str] | None = None,
    ):
        self.conv_id = conv_id or self._generate_id()
        self.title = title
        self.group_id = group_id
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        # 消息列表：[{"role": ..., "content": ..., "attachments"?: [...元数据]}, ...]
        self.messages: list[dict] = []
        # 稳定 msg_id：每次 add_message 生成一个 UUID；save 时 UPSERT 用
        self._msg_ids: list[str] = list(msg_ids) if msg_ids else []

    @staticmethod
    def _generate_id() -> str:
        timestamp = int(time.time() * 1000)
        rand = random.randint(1000, 9999)
        return f"{timestamp}-{rand}"

    def add_message(self, role: str, content: str, attachments: list = None) -> None:
        """
        新增消息：同步分配稳定 msg_id

        attachments（可选）：[{name, size, path, content?}, ...]
        注意：这里只持久化元数据（name/size/path），content 只在 LLM 调用时临时用
        """
        msg: dict = {"role": role, "content": content}
        if attachments:
            # 脱敏：丢弃 content 字段，只保留元数据
            sanitized = []
            for a in attachments:
                sanitized.append({
                    "name": a.get("name", ""),
                    "size": a.get("size", ""),
                    "path": a.get("path", ""),
                })
            msg["attachments"] = sanitized
        self.messages.append(msg)
        self._msg_ids.append(uuid.uuid4().hex)
        self.updated_at = datetime.now().isoformat()
        self._sync_msg_ids()

    def update_last_message(self, content: str) -> None:
        """直接覆盖最后一条消息的内容（msg_id 不变）"""
        if self.messages and self.messages[-1]["role"] in ("assistant", "user"):
            self.messages[-1]["content"] = content
            self.updated_at = datetime.now().isoformat()

    def _sync_msg_ids(self) -> None:
        """兜底：外部直接 .messages.append() 时补齐 msg_id"""
        while len(self._msg_ids) < len(self.messages):
            self._msg_ids.append(uuid.uuid4().hex)
        if len(self._msg_ids) > len(self.messages):
            self._msg_ids = self._msg_ids[:len(self.messages)]

    def to_dict(self) -> dict:
        self._sync_msg_ids()
        return {
            "conv_id": self.conv_id,
            "title": self.title,
            "group_id": self.group_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "msg_ids": self._msg_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        # 兼容旧数据：可能没有 msg_ids 字段
        conv = cls(
            conv_id=data["conv_id"],
            title=data["title"],
            group_id=data.get("group_id"),
            msg_ids=data.get("msg_ids"),
        )
        conv.created_at = data["created_at"]
        conv.updated_at = data["updated_at"]
        conv.messages = data["messages"]
        if not conv._msg_ids:
            conv._msg_ids = [uuid.uuid4().hex for _ in conv.messages]
        else:
            conv._sync_msg_ids()
        return conv


class Group:
    """分组数据模型"""

    def __init__(self, group_id: str = None, name: str = "新分组"):
        self.group_id = group_id or self._generate_id()
        self.name = name
        self.created_at = datetime.now().isoformat()

    @staticmethod
    def _generate_id() -> str:
        timestamp = int(time.time() * 1000)
        rand = random.randint(1000, 9999)
        return f"group-{timestamp}-{rand}"

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Group":
        group = cls(data["group_id"], data["name"])
        group.created_at = data["created_at"]
        return group


# ============================================================
# HistoryManager
# ============================================================
class HistoryManager:
    """管理对话和分组，使用 SQLite 存储（UPSERT + 稳定 msg_id）"""

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

            # 消息表（msg_id 是稳定 ID，用于 UPSERT 替代 DELETE+INSERT）
            c.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id TEXT NOT NULL,
                    msg_id TEXT NOT NULL,
                    role TEXT,
                    content TEXT,
                    message_data TEXT,
                    created_at TEXT,
                    FOREIGN KEY (conv_id) REFERENCES conversations(conv_id) ON DELETE CASCADE
                )
            ''')
            # 一次性 schema 补全：老库没有 msg_id 列时补上
            c.execute("PRAGMA table_info(messages)")
            msg_columns = {col[1] for col in c.fetchall()}
            if "msg_id" not in msg_columns:
                c.execute("ALTER TABLE messages ADD COLUMN msg_id TEXT")
            # 唯一索引（save 时 UPSERT 的依据）
            c.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_conv_msg
                ON messages(conv_id, msg_id)
            """)
            conn.commit()

    def _load_cache(self):
        """加载所有分组和对话元数据到内存（用于快速显示）"""
        self.groups: dict[str, dict] = {}
        self.conversations: dict[str, dict] = {}

        with self._get_connection() as conn:
            c = conn.cursor()

            c.execute("SELECT group_id, name, created_at FROM groups")
            for row in c.fetchall():
                self.groups[row[0]] = {
                    "name": row[1],
                    "created_at": row[2],
                }

            c.execute("SELECT conv_id, group_id, title, updated_at FROM conversations")
            for row in c.fetchall():
                self.conversations[row[0]] = {
                    "group_id": row[1],
                    "title": row[2],
                    "updated_at": row[3],
                }

    # ---------- 分组操作 ----------
    def create_group(self, name: str = "新分组") -> Group:
        group = Group(name=name)
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO groups (group_id, name, created_at) VALUES (?, ?, ?)",
                (group.group_id, group.name, group.created_at),
            )
            conn.commit()
        self.groups[group.group_id] = {
            "name": group.name,
            "created_at": group.created_at,
        }
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
        """删除分组，如果 move_to_group_id 指定，则将该分组下的对话移动到目标分组"""
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute(
                "UPDATE conversations SET group_id = ? WHERE group_id = ?",
                (move_to_group_id, group_id),
            )
            c.execute("DELETE FROM groups WHERE group_id = ?", (group_id,))
            conn.commit()
        if group_id in self.groups:
            del self.groups[group_id]
        for conv_id, meta in self.conversations.items():
            if meta["group_id"] == group_id:
                meta["group_id"] = move_to_group_id

    def get_all_groups(self) -> list[dict]:
        return [{"group_id": gid, **info} for gid, info in self.groups.items()]

    # ---------- 对话操作 ----------
    def create_conversation(self, title: str = "新对话", group_id: str = None) -> Conversation:
        conv = Conversation(title=title, group_id=group_id)
        self.save_conversation(conv)
        return conv

    def save_conversation(self, conv: Conversation):
        """
        保存对话。优化点：
        1. 显式 BEGIN IMMEDIATE 事务：中途失败自动 ROLLBACK
        2. UPSERT 替代 DELETE+INSERT：用稳定 msg_id 做 INSERT...ON CONFLICT，
           不变的 message 不会再重写磁盘，长对话性能显著提升
        3. 增量删除：只 DELETE 不在当前 msg_ids 集合中的旧消息
        """
        group_id = conv.group_id if conv.group_id else None
        # 兜底对齐：外部若直接 .messages.append() 而没 add_message，这里补齐 msg_ids
        conv._sync_msg_ids()

        now_iso = datetime.now().isoformat()
        with self._get_connection() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                c = conn.cursor()

                # 1. upsert 对话元数据
                c.execute('''
                    INSERT OR REPLACE INTO conversations
                    (conv_id, group_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (conv.conv_id, group_id, conv.title, conv.created_at, conv.updated_at))

                # 2. upsert 所有消息
                for msg, msg_id in zip(conv.messages, conv._msg_ids):
                    msg_json = json.dumps(msg, ensure_ascii=False)
                    c.execute('''
                        INSERT INTO messages (conv_id, msg_id, role, content, message_data, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(conv_id, msg_id) DO UPDATE SET
                            role = excluded.role,
                            content = excluded.content,
                            message_data = excluded.message_data
                    ''', (conv.conv_id, msg_id,
                          msg.get("role", ""), msg.get("content", ""),
                          msg_json, now_iso))

                # 3. 增量删除：移除已经不在当前对话里的旧消息
                if conv._msg_ids:
                    placeholders = ",".join("?" * len(conv._msg_ids))
                    c.execute(
                        f"DELETE FROM messages WHERE conv_id = ? AND msg_id NOT IN ({placeholders})",
                        [conv.conv_id, *conv._msg_ids],
                    )
                else:
                    c.execute("DELETE FROM messages WHERE conv_id = ?", (conv.conv_id,))

                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        # 更新内存缓存
        self.conversations[conv.conv_id] = {
            "group_id": group_id,
            "title": conv.title,
            "updated_at": conv.updated_at,
        }

    def delete_conversation(self, conv_id: str):
        logger.info("删除对话 %s", conv_id)
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM conversations WHERE conv_id = ?", (conv_id,))
            conn.commit()
        if conv_id in self.conversations:
            del self.conversations[conv_id]

    def move_conversation(self, conv_id: str, target_group_id: str):
        """移动对话到指定分组（target_group_id 可为 None 表示未分组）"""
        if conv_id not in self.conversations:
            return
        if not target_group_id:
            target_group_id = None
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute(
                "UPDATE conversations SET group_id = ? WHERE conv_id = ?",
                (target_group_id, conv_id),
            )
            conn.commit()
        self.conversations[conv_id]["group_id"] = target_group_id

    def get_conversation(self, conv_id: str) -> Conversation | None:
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT group_id, title, created_at, updated_at FROM conversations WHERE conv_id = ?",
                (conv_id,),
            )
            row = c.fetchone()
            if not row:
                return None
            group_id, title, created_at, updated_at = row

            # 同时取 msg_id 和 message_data，确保 save 时 UPSERT 命中稳定 ID
            c.execute(
                "SELECT msg_id, message_data FROM messages WHERE conv_id = ? ORDER BY id",
                (conv_id,),
            )
            msg_rows = c.fetchall()
            msg_ids = [r[0] for r in msg_rows]
            messages = [json.loads(r[1]) for r in msg_rows]

        conv = Conversation(conv_id, title, group_id, msg_ids=msg_ids)
        conv.created_at = created_at
        conv.updated_at = updated_at
        conv.messages = messages
        return conv

    def get_conversations_by_group(self) -> dict[str | None, list[dict]]:
        result: dict[str | None, list[dict]] = {}
        for conv_id, meta in self.conversations.items():
            group_id = meta["group_id"]
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
