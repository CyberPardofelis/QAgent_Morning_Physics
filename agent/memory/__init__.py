"""
agent.memory — 通用记忆库（参考 WAgent 设计，适配 QAgent 任务场景）

模块结构：
- schemas          路径常量 + 记忆类型枚举 + 元数据 schema
- embedder         包装 QAgent 现有 BGE Embedder（CPU 优先）
- vector_store     faiss-cpu 向量存储（独立于 paper 向量库）
- db_manager       SQLite memories 表（CRUD + 统计）
- retriever        混合检索（向量 + 关键词）
- processor        总控：写侧触发策略 + 读侧格式化

设计要点（与 WAgent 角色记忆库的关键差异）：
- 范围：通用 scope='default'（未来支持 per-function scope，不区分角色）
- Embedding：复用 QAgent BGE-large-en-v1.5 (1024 dim)，不引入新模型
- 向量库：faiss-cpu (IndexFlatIP)，不依赖 GPU
- 触发：决策 1=b — 显式 save_memory 工具 + LLM 看到关键词自动调
- 读侧：决策 2=b — 不自动注入，LLM 想用就调 recall_memories 工具
- 记忆类型：user_preference / research_fact / paper_note / reflection / task_state
"""
from .schemas import (
    VECTOR_DIR,
    MEMORY_FAISS_PATH,
    MEMORY_META_PATH,
    MEMORY_TYPES,
    DEFAULT_SCOPE,
    DEFAULT_EMBEDDING_DIM,
    validate_metadata,
    TRIGGER_KEYWORDS,
    content_has_trigger_keyword,
)
from .embedder import (
    MemoryEmbedder,
    get_embedder,
    encode_text,
    encode_texts,
)
from .vector_store import MemoryVectorStore, get_vector_store
from .db_manager import MemoryDBManager, get_db_manager
from .retriever import MemoryRetriever
from .processor import MemoryProcessor

__all__ = [
    # 常量
    "VECTOR_DIR", "MEMORY_FAISS_PATH", "MEMORY_META_PATH",
    "MEMORY_TYPES", "DEFAULT_SCOPE", "DEFAULT_EMBEDDING_DIM",
    "TRIGGER_KEYWORDS", "content_has_trigger_keyword",
    # 类 / 工厂
    "MemoryEmbedder", "get_embedder", "encode_text", "encode_texts",
    "MemoryVectorStore", "get_vector_store",
    "MemoryDBManager", "get_db_manager",
    "MemoryRetriever", "MemoryProcessor",
]
