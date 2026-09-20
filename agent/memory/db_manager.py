"""
agent.memory.db_manager — SQLite memories 表管理

通过 agent/infra/db.py 的 connect("memory") 访问。
表设计：
- id              TEXT PK (uuid4)
- scope           TEXT DEFAULT 'default'  (未来 per-function 用)
- content         TEXT NOT NULL
- memory_type     TEXT (user_preference/research_fact/paper_note/reflection/task_state/general)
- importance      REAL DEFAULT 0.5
- related_paper_id TEXT (可空，关联 papers 表)
- source_conv_id  TEXT (可空，关联 conversations 表)
- created_at      TEXT NOT NULL (ISO 8601)
- last_accessed   TEXT (可空)
- access_count    INTEGER DEFAULT 0
- metadata        TEXT (JSON)

所有 CRUD 走新连接（与 WAgent memory/db_manager.py 一致，线程安全）。
"""
import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Optional

from .schemas import DEFAULT_SCOPE, validate_metadata

logger = logging.getLogger(__name__)


class MemoryDBManager:
    """记忆库 SQLite 管理器（单例）。"""

    _instance: Optional["MemoryDBManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._init_schema()
        self._initialized = True

    def _init_schema(self) -> None:
        """建表（idempotent）。通过 agent/infra/db.py 连接。"""
        from agent.infra.db import connect
        with connect("memory") as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL DEFAULT 'default',
                    content TEXT NOT NULL,
                    memory_type TEXT,
                    importance REAL DEFAULT 0.5,
                    related_paper_id TEXT,
                    source_conv_id TEXT,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope);
                CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_mem_paper ON memories(related_paper_id);
                CREATE INDEX IF NOT EXISTS idx_mem_importance ON memories(importance DESC);
                CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(created_at DESC);
            """)
        logger.info("[MemoryDB] 初始化完成 (scope=default)")

    # ============================================================
    # CRUD
    # ============================================================
    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())

    def insert(
        self,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        scope: str = DEFAULT_SCOPE,
        related_paper_id: str | None = None,
        source_conv_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> str:
        """
        插入一条记忆。返回 memory_id。

        metadata 会 JSON 序列化存到 metadata 字段。
        """
        if not content or not content.strip():
            raise ValueError("content 不能为空")
        if not (0.0 <= importance <= 1.0):
            raise ValueError(f"importance 必须在 [0, 1]，实际 {importance}")
        if metadata is not None and not validate_metadata(metadata):
            raise ValueError("metadata 必须是可 JSON 序列化的 dict (< 10KB)")

        mid = memory_id or self.generate_id()
        now = datetime.now().isoformat()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)

        from agent.infra.db import connect
        with connect("memory") as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, scope, content, memory_type, importance,
                    related_paper_id, source_conv_id, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid, scope, content, memory_type, importance,
                    related_paper_id, source_conv_id, now, meta_json,
                ),
            )
        logger.debug("[MemoryDB] 插入: id=%s type=%s importance=%.2f", mid[:8], memory_type, importance)
        return mid

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """按 id 查一条。返回 dict 或 None。"""
        from agent.infra.db import connect
        with connect("memory") as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_by_ids(self, memory_ids: list[str]) -> list[dict[str, Any]]:
        """按 id 批量查。保持输入顺序，去掉 None。"""
        if not memory_ids:
            return []
        from agent.infra.db import connect
        placeholders = ",".join(["?"] * len(memory_ids))
        with connect("memory") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})",
                memory_ids,
            ).fetchall()
        id_to_row = {r["id"]: self._row_to_dict(r) for r in rows}
        return [id_to_row[mid] for mid in memory_ids if mid in id_to_row]

    def list_recent(
        self,
        scope: str | None = None,
        memory_type: str | None = None,
        related_paper_id: str | None = None,
        min_importance: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        按条件查（不带向量搜索）。
        按 importance DESC, created_at DESC 排序。
        """
        conditions = []
        params: list[Any] = []
        if scope is not None:
            conditions.append("scope = ?")
            params.append(scope)
        if memory_type is not None:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        if related_paper_id is not None:
            conditions.append("related_paper_id = ?")
            params.append(related_paper_id)
        if min_importance is not None:
            conditions.append("importance >= ?")
            params.append(min_importance)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        from agent.infra.db import connect
        with connect("memory") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE {where_clause}
                ORDER BY importance DESC, created_at DESC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def search_keyword(
        self,
        keyword: str,
        scope: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """关键词 LIKE 搜索（content 字段）。"""
        if not keyword:
            return []
        from agent.infra.db import connect
        conditions = ["content LIKE ?"]
        params: list[Any] = [f"%{keyword}%"]
        if scope is not None:
            conditions.append("scope = ?")
            params.append(scope)
        params.append(limit)
        where_clause = " AND ".join(conditions)
        with connect("memory") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE {where_clause}
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_importance(self, memory_id: str, new_importance: float) -> bool:
        """更新 importance。"""
        if not (0.0 <= new_importance <= 1.0):
            raise ValueError(f"importance 必须在 [0, 1]，实际 {new_importance}")
        from agent.infra.db import connect
        with connect("memory") as conn:
            cursor = conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (new_importance, memory_id),
            )
            return cursor.rowcount > 0

    def update_content(self, memory_id: str, new_content: str) -> bool:
        """更新 content（用户编辑）。"""
        if not new_content or not new_content.strip():
            raise ValueError("content 不能为空")
        from agent.infra.db import connect
        with connect("memory") as conn:
            cursor = conn.execute(
                "UPDATE memories SET content = ? WHERE id = ?",
                (new_content, memory_id),
            )
            return cursor.rowcount > 0

    def touch(self, memory_id: str) -> None:
        """更新 last_accessed + access_count += 1。"""
        now = datetime.now().isoformat()
        from agent.infra.db import connect
        with connect("memory") as conn:
            conn.execute(
                """
                UPDATE memories
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
                """,
                (now, memory_id),
            )

    def delete(self, memory_id: str) -> bool:
        """硬删除。"""
        from agent.infra.db import connect
        with connect("memory") as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            return cursor.rowcount > 0

    def delete_by_ids(self, memory_ids: list[str]) -> int:
        """批量硬删除。"""
        if not memory_ids:
            return 0
        from agent.infra.db import connect
        placeholders = ",".join(["?"] * len(memory_ids))
        with connect("memory") as conn:
            cursor = conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                memory_ids,
            )
            return cursor.rowcount

    # ============================================================
    # 统计
    # ============================================================
    def count(self, scope: str | None = None, memory_type: str | None = None) -> int:
        conditions = []
        params: list[Any] = []
        if scope is not None:
            conditions.append("scope = ?")
            params.append(scope)
        if memory_type is not None:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        from agent.infra.db import connect
        with connect("memory") as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM memories WHERE {where_clause}",
                params,
            ).fetchone()
        return row[0] if row else 0

    def get_statistics(self, scope: str | None = None) -> dict[str, Any]:
        """统计信息（用于 UI / debug）。"""
        from agent.infra.db import connect
        with connect("memory") as conn:
            conn.row_factory = sqlite3.Row
            params: list[Any] = []
            where = ""
            if scope is not None:
                where = " WHERE scope = ?"
                params = [scope]
            total = conn.execute(
                f"SELECT COUNT(*) AS n FROM memories{where}", params
            ).fetchone()["n"]
            by_type_rows = conn.execute(
                f"""
                SELECT memory_type, COUNT(*) AS n, AVG(importance) AS avg_imp
                FROM memories{where}
                GROUP BY memory_type
                """,
                params,
            ).fetchall()
            by_paper = conn.execute(
                f"""
                SELECT related_paper_id, COUNT(*) AS n
                FROM memories{where}
                WHERE related_paper_id IS NOT NULL
                GROUP BY related_paper_id
                ORDER BY n DESC
                LIMIT 10
                """,
                params,
            ).fetchall()
        return {
            "total": total,
            "by_type": {
                r["memory_type"] or "general": {
                    "count": r["n"],
                    "avg_importance": r["avg_imp"] or 0.0,
                }
                for r in by_type_rows
            },
            "top_papers": [
                {"paper_id": r["related_paper_id"], "count": r["n"]}
                for r in by_paper
            ],
        }

    # ============================================================
    # Helper
    # ============================================================
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        meta = d.get("metadata")
        if meta and isinstance(meta, str):
            try:
                d["metadata"] = json.loads(meta)
            except json.JSONDecodeError:
                d["metadata"] = {}
        else:
            d["metadata"] = {}
        return d


# ============================================================
# 全局单例
# ============================================================
def get_db_manager() -> MemoryDBManager:
    """获取全局 MemoryDBManager 单例。"""
    return MemoryDBManager()
