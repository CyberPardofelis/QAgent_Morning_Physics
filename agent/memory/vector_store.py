"""
agent.memory.vector_store — faiss-cpu 向量存储（独立于 paper 向量库）

设计：
- 用 IndexFlatIP（精确内积搜索），normalize 后 = 余弦相似度
- 独立路径 data/vector_store/memories.faiss + memories_meta.pkl
- 线程安全：faiss-cpu 索引在 build 后是 read-only safe，但 add 需要重建
  → 我们的模式：启动加载 + 偶尔 add（save_memory 时），用 lock 串行化
- metadata 存 list[dict]（用 pickle），顺序与 faiss index id 一一对应
- CPU 优先：完全用 faiss-cpu，不依赖 GPU
"""
import logging
import pickle
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np

from .schemas import MEMORY_FAISS_PATH, MEMORY_META_PATH, DEFAULT_EMBEDDING_DIM

logger = logging.getLogger(__name__)


# 软依赖 faiss（如果未安装就报错，不静默）
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError as e:
    _FAISS_AVAILABLE = False
    _FAISS_IMPORT_ERROR = str(e)


class MemoryVectorStore:
    """
    faiss-cpu 索引 + pickle metadata list。

    索引结构：
    - faiss.IndexFlatIP(dim)：精确搜索，无 ANN 加速
    - metadata: list[dict]，长度 = faiss 索引 ntotal
    - id_map: dict[memory_id -> faiss_idx]，记录 SQLite id 到 faiss 内部 row 的映射
    """

    _instance: Optional["MemoryVectorStore"] = None
    _lock = threading.Lock()

    def __new__(cls, dim: Optional[int] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # 第一次创建时如果没传 dim，从 embedder 自动读
                    if dim is None:
                        try:
                            from .embedder import get_embedder
                            dim = get_embedder().get_dim()
                        except Exception:
                            dim = DEFAULT_EMBEDDING_DIM
                    instance = super().__new__(cls)
                    instance._initialized = False
                    instance._dim_arg = dim
                    cls._instance = instance
        return cls._instance

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM):
        if self._initialized:
            return
        if not _FAISS_AVAILABLE:
            raise RuntimeError(
                f"faiss 未安装，无法创建 MemoryVectorStore。错误: {_FAISS_IMPORT_ERROR}。"
                f"请 pip install faiss-cpu（QAgent requirements.txt 已锁 faiss-cpu==1.13.2）"
            )

        # dim 优先用 __new__ 时读到的 _dim_arg（从 embedder 来的）
        dim = getattr(self, "_dim_arg", None) or dim
        self.dim = dim
        self._write_lock = threading.Lock()  # 写操作串行化
        self._meta: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}

        # 加载已有索引
        if MEMORY_FAISS_PATH.exists() and MEMORY_META_PATH.exists():
            try:
                self.index = faiss.read_index(str(MEMORY_FAISS_PATH))
                with open(MEMORY_META_PATH, "rb") as f:
                    data = pickle.load(f)
                # 兼容旧版 data 格式（纯 list）
                if isinstance(data, list):
                    self._meta = data
                elif isinstance(data, dict) and "meta" in data:
                    self._meta = data.get("meta", [])
                    self._id_to_idx = data.get("id_to_idx", {})
                else:
                    self._meta = []
                # 重建 id_to_idx（如果 meta 已加载但 map 缺失）
                if not self._id_to_idx:
                    for i, m in enumerate(self._meta):
                        mid = m.get("id")
                        if mid:
                            self._id_to_idx[mid] = i
                logger.info(
                    "[MemoryVectorStore] 加载完成: %d 条记忆, dim=%d",
                    self.index.ntotal, self.dim
                )
            except Exception as e:
                logger.warning("[MemoryVectorStore] 加载失败，从空开始: %s", e)
                self.index = faiss.IndexFlatIP(dim)
                self._meta = []
                self._id_to_idx = {}
        else:
            self.index = faiss.IndexFlatIP(dim)
            logger.info("[MemoryVectorStore] 新建索引, dim=%d", self.dim)

        self._initialized = True

    # ============================================================
    # 写
    # ============================================================
    def add(
        self,
        memory_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        添加一条向量。

        返回 faiss 内部 row idx。
        """
        if vector.ndim != 1:
            raise ValueError(f"vector 必须是 1D, 实际 shape={vector.shape}")
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"vector dim={vector.shape[0]} 与 index dim={self.dim} 不匹配。"
                f"如果换 embedding model 了，需要 delete data/vector_store/memories.* 重建。"
            )

        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        # 归一化（IndexFlatIP 当向量都归一化时 = 余弦相似度）
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v = v / norm

        with self._write_lock:
            row_idx = self.index.ntotal
            self.index.add(v)
            meta = dict(metadata or {})
            meta["id"] = memory_id
            self._meta.append(meta)
            self._id_to_idx[memory_id] = row_idx
            self._save()
            return row_idx

    def add_batch(
        self,
        items: List[Tuple[str, np.ndarray, Dict[str, Any]]],
    ) -> List[int]:
        """
        批量添加。items = [(memory_id, vector, metadata), ...]
        """
        if not items:
            return []

        # 先全部 encode 校验
        for mid, vec, _ in items:
            if vec.ndim != 1:
                raise ValueError(f"vector {mid} 不是 1D")
            if vec.shape[0] != self.dim:
                raise ValueError(f"vector {mid} dim 不匹配")

        vectors = np.stack([np.asarray(v, dtype=np.float32) for _, v, _ in items])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        vectors = vectors / norms

        with self._write_lock:
            start_idx = self.index.ntotal
            self.index.add(vectors)
            row_indices = list(range(start_idx, start_idx + len(items)))
            for (mid, _, meta), row_idx in zip(items, row_indices):
                m = dict(meta or {})
                m["id"] = mid
                self._meta.append(m)
                self._id_to_idx[mid] = row_idx
            self._save()
            return row_indices

    # ============================================================
    # 读
    # ============================================================
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Tuple[int, float]]:
        """
        向量搜索。

        返回 [(row_idx, similarity), ...] 按相似度降序。
        exclude_ids: 排除某些 memory_id（避免重复返回刚写的）。
        """
        if self.index.ntotal == 0:
            return []

        v = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v = v / norm

        # 多取一些再 filter
        k_search = min(top_k * 2 + len(exclude_ids or []), self.index.ntotal)
        D, I = self.index.search(v, k_search)

        results: List[Tuple[int, float]] = []
        for dist, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:  # faiss 用 -1 表示无效
                continue
            # 检查 exclude
            if exclude_ids and self._meta[idx].get("id") in exclude_ids:
                continue
            results.append((idx, float(dist)))
            if len(results) >= top_k:
                break
        return results

    def get_metadata(self, row_idx: int) -> Optional[Dict[str, Any]]:
        """按 faiss 内部 row idx 拿 metadata。"""
        if 0 <= row_idx < len(self._meta):
            return dict(self._meta[row_idx])
        return None

    def get_vector(self, row_idx: int) -> Optional[np.ndarray]:
        """
        拿单条向量的副本（用于 MMR 等需要候选向量的场景）。
        返回 1D ndarray 或 None（row_idx 越界 / 已被 tombstone）。
        """
        if row_idx < 0 or row_idx >= self.index.ntotal:
            return None
        try:
            v = self.index.reconstruct(int(row_idx))
            return np.asarray(v, dtype=np.float32).copy()
        except Exception as e:
            logger.warning("[MemoryVectorStore] reconstruct %d 失败: %s", row_idx, e)
            return None

    def get_row_idx(self, memory_id: str) -> Optional[int]:
        """按 memory_id 拿 faiss row idx。"""
        return self._id_to_idx.get(memory_id)

    def remove(self, memory_id: str) -> bool:
        """
        从索引删除一条。
        faiss-cpu 不支持单条 delete（要重建），所以这里用 lazy tombstone：
        - 从 metadata/id_map 移除（后续 search 拿不到）
        - faiss index 留着（等累积到一定量再 compact）
        """
        with self._write_lock:
            row_idx = self._id_to_idx.pop(memory_id, None)
            if row_idx is None:
                return False
            if 0 <= row_idx < len(self._meta):
                self._meta[row_idx]["_deleted"] = True
            self._save()
            return True

    def purge_deleted(self) -> int:
        """
        真正从 faiss 索引里清掉标记 _deleted 的（重建 index）。
        默认不主动调，由用户/管理命令触发。
        """
        with self._write_lock:
            alive_meta = [m for m in self._meta if not m.get("_deleted")]
            if len(alive_meta) == len(self._meta):
                return 0  # 无可清理

            # 重建 faiss index（没有原始 vector，只能重新 encode 或接受清空）
            # 简化：清空索引，meta 留着待 reindex（不主动 encode，避免大块 IO）
            # 调用方负责 reindex
            new_index = faiss.IndexFlatIP(self.dim)
            self.index = new_index
            self._meta = alive_meta
            self._id_to_idx = {m["id"]: i for i, m in enumerate(alive_meta) if m.get("id")}
            self._save()
            return len(self._meta)

    # ============================================================
    # 状态
    # ============================================================
    def count(self) -> int:
        """活跃记忆数（排除 _deleted）。"""
        return sum(1 for m in self._meta if not m.get("_deleted"))

    def total(self) -> int:
        """faiss 索引里总条数（含 _deleted）。"""
        return self.index.ntotal

    def is_empty(self) -> bool:
        return self.count() == 0

    def all_active_ids(self) -> List[str]:
        """所有活跃 memory_id（用于重索引/调试）。"""
        return [m["id"] for m in self._meta if not m.get("_deleted") and m.get("id")]

    # ============================================================
    # 持久化
    # ============================================================
    def _save(self) -> None:
        """落盘。内部调用（已持锁）。"""
        try:
            MEMORY_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(MEMORY_FAISS_PATH))
            with open(MEMORY_META_PATH, "wb") as f:
                pickle.dump(
                    {"meta": self._meta, "id_to_idx": self._id_to_idx},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as e:
            logger.exception("[MemoryVectorStore] 落盘失败: %s", e)
            # 不抛出，存储失败不应让上层崩

    def get_backend_info(self) -> dict:
        """用于 debug / UI 显示。"""
        return {
            "type": "faiss-cpu",
            "index": "IndexFlatIP",
            "dim": self.dim,
            "total": self.total(),
            "active": self.count(),
            "faiss_path": str(MEMORY_FAISS_PATH),
            "meta_path": str(MEMORY_META_PATH),
        }


# ============================================================
# 全局单例
# ============================================================
def get_vector_store() -> MemoryVectorStore:
    """获取全局 MemoryVectorStore 单例。dim 自动从 embedder 读。"""
    from .embedder import get_embedder
    dim = get_embedder().get_dim()
    # 注意：单例以首次调用时的 dim 为准。如果换 embedding model，需要
    # 1) delete data/vector_store/memories.* 2) 重启 app
    return MemoryVectorStore(dim=dim)
