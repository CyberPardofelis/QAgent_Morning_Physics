"""
agent.vector_db — chroma 持久化向量库（论文 chunk 用，P0 新增 2026-09-02）

设计要点（参考 WAgent / memory 模块 / ROADMAP §0.4）：
- 单例 + lazy import：第一次调时才 import chromadb，避免启动 5s+ 延迟
- 用本地 bge-large-en-v1.5 当 embedding function：
  避免 chroma 默认 80MB onnx 下载（国内网络下不动）
  默认路径走 D:\\pythonProject\\BGE_Large_en_v1.5（用户已下好）
- 线程安全：chromadb PersistentClient 自带文件锁，collection 操作可重入
- 4 个核心方法：add_paper_chunks / search / get_paper / delete_paper
- metadata 清洗：chroma 不支持 list/nested，所有值必须可序列化到 str/int/float/bool

跟老 vector_store.py（faiss）的关系：
- 老 vector_store.py 保留为孤儿代码，标记 deprecated，不删
- 本模块独立路径 data/chroma_db/，跟老 faiss index/ 互不干扰
- P0.5 完成后，老 faiss 索引才考虑清掉
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 全局单例（lazy）
# ============================================================
_chroma_client = None
_chroma_embedding_fn = None
_init_lock = threading.Lock()


def _make_embedding_function():
    """
    用本地 bge-large-en-v1.5 当 embedding function。

    为什么不用 chroma 默认:
    - chroma 默认会下载 all-MiniLM-L6-v2 (80MB) 到 ~/.cache/chroma/onnx_models/
    - 国内网络下 80MB 要 30+ 分钟
    - 用户本地已下好 bge-large-en-v1.5(配置 MODEL_PATH 指向)
    - 1024 维(vs 默认 384 维), 向量质量更好

    失败回退: 返回 None, chroma 会用默认(让代码至少能跑)

    2026-09-05 加: 检查 Config.get_embedding_enabled(), False 时不加载 BGE 直接返 None
    (之前 EMBEDDING_ENABLED=False 也会加载 BGE, 跟 PyQt6 多线程竞争偶发 0xC0000005 段错误)
    """
    global _chroma_embedding_fn
    if _chroma_embedding_fn is not None:
        return _chroma_embedding_fn
    # P0 段错误绕开: 用户在 settings 显式关掉 BGE 时, 不加载模型
    # 2026-09-05 加: 在 BGE 路径之前 return None, 防止 PyTorch 加载
    # 实测 Config.get_embedding_enabled() 可能仍 True (用户开了), 需双重检查
    try:
        from config import Config
        embedding_enabled = Config.get_embedding_enabled()
        print(f"[vector_db] _make_embedding_function called, embedding_enabled={embedding_enabled}", flush=True)
        if not embedding_enabled:
            logger.info("[vector_db] EMBEDDING_ENABLED=False, 跳过 BGE 加载, 用 chroma 默认")
            print("[vector_db] EMBEDDING_ENABLED=False, 返 None, 走 chroma 默认", flush=True)
            return None
    except Exception as e:
        print(f"[vector_db] check error: {e}", flush=True)
        pass
    # P0 加: 即使 QSettings=True, 也检查 env / 显式禁用标记
    # 防止用户开了 QSettings 但实际仍崩的场景
    try:
        import os
        env_disabled = os.getenv("DISABLE_LOCAL_EMBEDDING", "").lower() in ("true", "1", "yes")
        if env_disabled:
            print("[vector_db] DISABLE_LOCAL_EMBEDDING=true env 设置, 跳过 BGE", flush=True)
            return None
    except Exception:
        pass
    try:
        from chromadb.utils import embedding_functions

        from config import Config
        model_path = Config.get_chroma_embedding_model_path()
        if not model_path or not Path(model_path).exists():
            logger.warning(
                "[vector_db] 本地 embedding model 路径不存在: %s。"
                "回退到 chroma 默认(会触发 80MB onnx 下载, 可能很慢)",
                model_path,
            )
            return None
        # 2026-09-07 P0 加: 第 4 道守护 — 加载 BGE 前再检查一次
        # 用户实测: embedding_enabled=False 走到 print 但 BGE 仍被加载
        # 可能原因: try/except 中 except pass 跳过 return, 或 chromadb 内部独立路径
        # 强制最后一道: 不管上面怎么走, 加载前必须确认
        try:
            from config import Config as _Cfg
            if not _Cfg.get_embedding_enabled():
                print("[vector_db] 第 4 道守护: embedding_enabled=False, 强制不走 BGE", flush=True)
                return None
        except Exception:
            pass
        logger.info("[vector_db] 加载本地 embedding model: %s", model_path)
        print(f"[vector_db] 即将加载 BGE: {model_path}", flush=True)
        fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_path,
            device="cpu",  # chroma 内置,默认 CPU
        )
        _chroma_embedding_fn = fn
        return fn
    except Exception as e:
        logger.warning("[vector_db] 本地 embedding function 加载失败: %s", e)
        return None


def get_chroma_client():
    """
    全局 chroma PersistentClient 单例（lazy 初始化）。
    第一次调时：import chromadb + 创建 client + 探测 embedding function。
    后续调时直接返回缓存对象。
    """
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
    with _init_lock:
        if _chroma_client is not None:
            return _chroma_client
        import chromadb

        from config import Config
        persist_dir = Config.get_chroma_path()
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        logger.info("[vector_db] 创建 chroma PersistentClient, path=%s", persist_dir)
        _chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        return _chroma_client


# ============================================================
# metadata 清洗
# ============================================================
def _clean_metadata_value(v: Any) -> Any:
    """
    chroma metadata 值只能是 str / int / float / bool。
    把 list / dict / None 转成 str 或合理替代。
    """
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        # 逗号分隔字符串（避开 chroma 不支持 list）
        return ",".join(str(x) for x in v) if v else ""
    if isinstance(v, dict):
        # 嵌套 dict 序列化（虽然会丢结构，但至少不报错）
        import json
        return json.dumps(v, ensure_ascii=False)
    return str(v)


# ============================================================
# VectorDB 主类
# ============================================================
class VectorDB:
    """
    chroma collection 的薄包装。4 个核心方法 + count。

    用法：
        db = VectorDB()  # 默认 "papers" collection
        db.add_paper_chunks("2401.12345", ["chunk 1", "chunk 2"],
                            metadata={"title": "...", "year": 2024})
        results = db.search("quantum error correction", top_k=5)
        chunks = db.get_paper("2401.12345")
        n = db.delete_paper("2401.12345")
    """

    def __init__(self, collection_name: str | None = None):
        from config import Config
        self.collection_name = collection_name or Config.get_chroma_collection()
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            client = get_chroma_client()
            fn = _make_embedding_function()
            kwargs = {"metadata": {"hnsw:space": "cosine"}}
            if fn is not None:
                kwargs["embedding_function"] = fn
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                **kwargs,
            )
        return self._collection

    def add_paper_chunks(
        self,
        paper_id: str,
        chunks: list[str],
        metadata: dict[str, Any] | None = None,
        chunk_ids: list[str] | None = None,
    ) -> int:
        """
        批量加论文 chunks。返回写入条数。

        - chunks: 切分后的文本列表
        - metadata: 论文级 metadata（每条 chunk 都一样），比如 {title, authors, year, journal, ...}
        - chunk_ids: 可选，自定义 id；留空时自动用 f"{paper_id}-chunk-{i}"
        """
        if not chunks:
            return 0
        # 准备 metadata（每条 chunk 一份，但都是同一份 paper 级 metadata）
        meta_base = dict(metadata or {})
        meta_base["paper_id"] = paper_id
        clean_meta = [
            {k: _clean_metadata_value(v) for k, v in meta_base.items()}
            for _ in chunks
        ]
        # chunk ids
        if chunk_ids is None:
            chunk_ids = [f"{paper_id}-chunk-{i}" for i in range(len(chunks))]
        # 幂等：先删同 paper_id 的旧 chunks（避免重复入库）
        # 注意：delete + add 不是原子的，但 chroma collection 加锁，并发场景下
        # 同 paper_id 同时 add 也不会真出问题（后到的会被覆盖）
        try:
            self.delete_paper(paper_id)
        except Exception:
            pass  # 库里没有这个 paper, 删不动是正常的
        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            metadatas=clean_meta,
        )
        logger.info(
            "[vector_db] add_paper_chunks: paper=%s chunks=%d collection=%s",
            paper_id, len(chunks), self.collection_name,
        )
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        语义检索。返回列表，每条含 {id, document, metadata, distance}。
        where 走 chroma metadata filter（如 {"year": 2024} 或 {"source": "arxiv"}）。
        """
        kwargs = {"query_texts": [query], "n_results": top_k}
        if where:
            kwargs["where"] = where
        try:
            res = self.collection.query(**kwargs)
        except Exception as e:
            logger.exception("[vector_db] search 失败: %s", e)
            return []
        results = []
        if not res or not res.get("ids") or not res["ids"][0]:
            return results
        for i, doc_id in enumerate(res["ids"][0]):
            results.append({
                "id": doc_id,
                "document": (res["documents"][0][i] if res.get("documents") else ""),
                "metadata": (res["metadatas"][0][i] if res.get("metadatas") else {}),
                "distance": (res["distances"][0][i] if res.get("distances") else None),
            })
        return results

    def get_paper(self, paper_id: str) -> list[dict[str, Any]]:
        """按 paper_id 拿所有 chunks（不重新 embed）。"""
        try:
            res = self.collection.get(where={"paper_id": paper_id})
        except Exception as e:
            logger.exception("[vector_db] get_paper 失败: %s", e)
            return []
        out = []
        for i, doc_id in enumerate(res.get("ids", [])):
            out.append({
                "id": doc_id,
                "document": (res["documents"][i] if res.get("documents") else ""),
                "metadata": (res["metadatas"][i] if res.get("metadatas") else {}),
            })
        return out

    def delete_paper(self, paper_id: str) -> int:
        """按 paper_id 删所有 chunks。返回删除条数。"""
        try:
            existing = self.collection.get(where={"paper_id": paper_id})
        except Exception as e:
            logger.exception("[vector_db] delete_paper 查询失败: %s", e)
            return 0
        n = len(existing.get("ids", []))
        if n:
            self.collection.delete(where={"paper_id": paper_id})
        logger.info("[vector_db] delete_paper: paper=%s deleted=%d", paper_id, n)
        return n

    def count(self) -> int:
        return self.collection.count()


# ============================================================
# 便捷单例（与 memory 模块风格一致）
# ============================================================
def get_vector_db() -> VectorDB:
    """获取默认 collection 的 VectorDB 实例（lazy）。"""
    return VectorDB()


# ============================================================
# 健康检查
# ============================================================
def health_check() -> dict[str, Any]:
    """
    给 settings_dialog "数据状态" tab 用。
    返回 {exists, path, collection, count, embedding_model, error?}
    """
    from config import Config
    info: dict[str, Any] = {
        "path": Config.get_chroma_path(),
        "collection": Config.get_chroma_collection(),
        "embedding_model": Config.get_chroma_embedding_model_path(),
        "exists": False,
        "count": 0,
    }
    chroma_path = Path(info["path"])
    if not chroma_path.exists():
        return info
    info["exists"] = True
    try:
        # 不实例化 collection（避免触发 embedding 加载）
        # 只看 chroma sqlite 文件
        sqlite_file = chroma_path / "chroma.sqlite3"
        info["sqlite_exists"] = sqlite_file.exists()
        info["sqlite_size_kb"] = sqlite_file.stat().st_size / 1024 if sqlite_file.exists() else 0
    except Exception as e:
        info["error"] = str(e)
    return info
