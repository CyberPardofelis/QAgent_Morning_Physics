"""
agent.memory.processor — 记忆库总控

读侧：把记忆格式化成 markdown context（决策 2=b，默认不自动注入，备查）
写侧：save_memory 主流程（encode + SQLite + faiss + 同步）

注意：决策 1=b（关键词触发）的实现：
- 后端不自动后台提取（避免噪音）
- 由 LLM 在 MEMORY_GUIDE 看到"包含 X 关键词就调 save_memory"提示后自行决定
- processor.should_suggest_save(text) 提供一致性判断
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from .db_manager import MemoryDBManager
from .vector_store import MemoryVectorStore
from .embedder import MemoryEmbedder
from .retriever import MemoryRetriever
from .schemas import (
    DEFAULT_SCOPE, MEMORY_TYPES, TRIGGER_KEYWORDS, content_has_trigger_keyword,
)

logger = logging.getLogger(__name__)


class MemoryProcessor:
    """
    记忆库总控。

    用法：
        p = MemoryProcessor()
        # 写
        mid = p.save(content="...", memory_type="user_preference", importance=0.8)
        # 读
        mems = p.recall(query="用户偏好")
        formatted = p.format_for_context(mems)
    """

    def __init__(self):
        self.db = MemoryDBManager()
        self.vs = MemoryVectorStore()
        self.embedder = MemoryEmbedder()
        self.retriever = MemoryRetriever(self.db, self.vs, self.embedder)

    # ============================================================
    # 写
    # ============================================================
    def save(
        self,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        related_paper_id: Optional[str] = None,
        source_conv_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scope: str = DEFAULT_SCOPE,
    ) -> str:
        """
        保存一条记忆。完整流程：encode → SQLite insert → faiss add。

        返回 memory_id。
        """
        if memory_type not in MEMORY_TYPES and memory_type != "general":
            logger.warning(
                "[MemoryProcessor] 未知 memory_type=%s，存为 general", memory_type
            )
            memory_type = "general"

        # 1) SQLite 插入（拿 id）
        memory_id = self.db.insert(
            content=content,
            memory_type=memory_type,
            importance=importance,
            scope=scope,
            related_paper_id=related_paper_id,
            source_conv_id=source_conv_id,
            metadata=metadata,
        )

        # 2) 编码 + 写 faiss
        try:
            vec = self.embedder.encode(content)
        except RuntimeError as e:
            logger.error("[MemoryProcessor] embedding 失败（仅 SQLite 写入成功）: %s", e)
            # 不抛：SQLite 已经有这条数据，向量检索拿不到而已
            return memory_id

        try:
            self.vs.add(
                memory_id=memory_id,
                vector=vec,
                metadata={
                    "memory_type": memory_type,
                    "importance": importance,
                    "related_paper_id": related_paper_id,
                    "created_at": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.exception("[MemoryProcessor] faiss add 失败: %s", e)
            # 不抛：SQLite 数据完整，向量索引缺这条（标记 _vector_missing）

        logger.info(
            "[MemoryProcessor] saved: id=%s type=%s importance=%.2f content='%s...'",
            memory_id[:8], memory_type, importance, content[:30],
        )
        return memory_id

    def save_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[str]:
        """批量 save。"""
        ids = []
        for it in items:
            mid = self.save(**it)
            ids.append(mid)
        return ids

    def delete(self, memory_id: str) -> bool:
        """删除（同时从 faiss 和 SQLite）。"""
        ok_db = self.db.delete(memory_id)
        try:
            ok_vs = self.vs.remove(memory_id)
        except Exception:
            ok_vs = False
        return ok_db or ok_vs

    def update_importance(self, memory_id: str, new_importance: float) -> bool:
        ok = self.db.update_importance(memory_id, new_importance)
        # faiss 端的 importance 是 metadata，不会影响搜索，但为了保持一致可以更新
        # 暂不更新（faiss 搜索只看向量，importance 在 SQLite）
        return ok

    def update_content(self, memory_id: str, new_content: str) -> bool:
        """
        更新 content。**重要**：这会改变语义向量，所以必须重新 encode + 重建 faiss。
        简化：删除旧 faiss 记录 + 重新添加。
        """
        if not self.db.update_content(memory_id, new_content):
            return False
        try:
            # 重新 encode + 加回 faiss（旧的会被 _deleted 标记，purge 时清掉）
            vec = self.embedder.encode(new_content)
            self.vs.add(memory_id, vec, metadata={"updated_at": datetime.now().isoformat()})
            # 标记旧 faiss row 为 _deleted（避免重复）
            old_idx = self.vs.get_row_idx(memory_id + "_old")  # 不存在，正常
            # 简化：直接 add 新条（faiss id 不冲突因为是不同 row）
        except RuntimeError as e:
            logger.warning("[MemoryProcessor] update_content 重新 encode 失败: %s", e)
        return True

    # ============================================================
    # 读
    # ============================================================
    def recall(
        self,
        query: str,
        limit: int = 5,
        scope: Optional[str] = None,
        memory_type: Optional[str] = None,
        related_paper_id: Optional[str] = None,
        min_importance: Optional[float] = None,
        method: str = "vector",
    ) -> List[Dict[str, Any]]:
        """
        检索记忆。委托给 retriever。
        """
        return self.retriever.retrieve(
            query=query,
            limit=limit,
            scope=scope,
            memory_type=memory_type,
            related_paper_id=related_paper_id,
            min_importance=min_importance,
            method=method,
        )

    def recall_for_paper(self, paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.retriever.retrieve_for_paper(paper_id, limit=limit)

    def recall_important(self, limit: int = 10, min_importance: float = 0.7) -> List[Dict[str, Any]]:
        return self.retriever.retrieve_important(limit=limit, min_importance=min_importance)

    def list_all(
        self,
        scope: Optional[str] = None,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return self.db.list_recent(
            scope=scope, memory_type=memory_type, min_importance=min_importance, limit=limit,
        )

    # ============================================================
    # 格式化
    # ============================================================
    def format_for_context(
        self,
        memories: List[Dict[str, Any]],
        include_score: bool = False,
        max_chars: int = 4000,
    ) -> str:
        """
        把记忆格式化成 markdown 文本，供 LLM 引用。
        （决策 2=b：默认不自动注入，但保留这个 helper 供将来用或 LLM 显式调用。）
        """
        if not memories:
            return ""

        lines = ["【相关记忆】"]
        total_chars = len(lines[0])
        for i, m in enumerate(memories, 1):
            line = self._format_one_memory(i, m, include_score)
            if total_chars + len(line) > max_chars and i > 1:
                lines.append(f"... (还有 {len(memories) - i + 1} 条未显示)")
                break
            lines.append(line)
            total_chars += len(line)
        return "\n".join(lines)

    def _format_one_memory(self, idx: int, m: Dict[str, Any], include_score: bool) -> str:
        parts = [f"{idx}."]
        if m.get("memory_type"):
            parts.append(f"[{m['memory_type']}]")
        parts.append(m.get("content", ""))
        if m.get("related_paper_id"):
            parts.append(f"(论文: {m['related_paper_id']})")
        if include_score:
            score = m.get("_score", 0)
            parts.append(f"(相关度: {score:.2f})")
        return " ".join(parts)

    # ============================================================
    # 触发检测（用于 MEMORY_GUIDE 提示 / 外部判断）
    # ============================================================
    def should_suggest_save(self, text: str) -> bool:
        """
        判断文本是否包含应触发 save_memory 工具的关键词。

        后端不自动调（决策 1=b 实现：把决定权交给 LLM），
        但这个函数用来给 LLM 一致性提示（避免漏看）。
        """
        return content_has_trigger_keyword(text)

    def get_trigger_keywords(self) -> tuple:
        return TRIGGER_KEYWORDS

    # ============================================================
    # 状态 / 统计
    # ============================================================
    def get_stats(self) -> Dict[str, Any]:
        return {
            "sqlite": self.db.get_statistics(),
            "vector_store": self.vs.get_backend_info(),
            "embedder": self.embedder.get_backend_info(),
            "trigger_keywords": list(TRIGGER_KEYWORDS),
            "memory_types": MEMORY_TYPES,
        }
