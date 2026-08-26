"""
agent.memory.relation_vector_store — v3.2: relation 级 faiss 向量索引

设计:
- 跟 MemoryVectorStore 平行, 但存的是"每条 relation 的谓语向量"
- metadata: {rid, memory_id, subj_canonical, pred, obj_canonical, polarity, idx_in_memory}
- 用途: _recall_by_pred_semantic (谓语语义召回) / _recall_by_subj_pred_obj (主谓宾三元组召回)
- 独立路径 data/vector_store/relations.faiss + relations_meta.pkl
- CPU 优先: IndexFlatIP, 不依赖 GPU
- 线程安全: lock + lazy tombstone (跟 MemoryVectorStore 一样的策略)

跟 MemoryVectorStore 的区别:
- 存的是 relation (每条记忆可能有 1-5 条 relation)
- 提供按 (subj, pred, obj) 字符串匹配的 search_by_subj_pred_obj
- 单例 (跟 MemoryVectorStore 平行, 互不干扰)
"""
import logging
import pickle
import threading
from typing import List, Dict, Optional, Any

import numpy as np

from .schemas import RELATION_FAISS_PATH, RELATION_META_PATH, DEFAULT_EMBEDDING_DIM

logger = logging.getLogger(__name__)


# 软依赖 faiss
try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError as e:
    _FAISS_AVAILABLE = False
    _FAISS_IMPORT_ERROR = str(e)


class RelationVectorStore:
    """
    关系级向量存储: 每条 relation 一行, 存谓语 embedding.
    """

    _instance: Optional["RelationVectorStore"] = None
    _lock = threading.Lock()

    def __new__(cls, dim: Optional[int] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
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
                f"faiss 未安装，无法创建 RelationVectorStore。错误: {_FAISS_IMPORT_ERROR}。"
            )

        # dim 优先用 __new__ 时读到的 _dim_arg
        dim = getattr(self, "_dim_arg", None) or dim
        self.dim = dim
        self._write_lock = threading.Lock()
        self._meta: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}

        # 加载已有索引
        if RELATION_FAISS_PATH.exists() and RELATION_META_PATH.exists():
            try:
                self.index = faiss.read_index(str(RELATION_FAISS_PATH))
                with open(RELATION_META_PATH, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, list):
                    self._meta = data
                elif isinstance(data, dict) and "meta" in data:
                    self._meta = data.get("meta", [])
                    self._id_to_idx = data.get("id_to_idx", {})
                else:
                    self._meta = []
                if not self._id_to_idx:
                    for i, m in enumerate(self._meta):
                        rid = m.get("rid")
                        if rid:
                            self._id_to_idx[rid] = i
                logger.info(
                    "[RelationVectorStore] 加载完成: %d 条 relation, dim=%d",
                    self.index.ntotal, self.dim,
                )
            except Exception as e:
                logger.warning("[RelationVectorStore] 加载失败，从空开始: %s", e)
                self.index = faiss.IndexFlatIP(dim)
                self._meta = []
                self._id_to_idx = {}
        else:
            self.index = faiss.IndexFlatIP(dim)
            logger.info("[RelationVectorStore] 新建索引, dim=%d", self.dim)

        # [v3.2 修复] dim 变化时重建 (单例 + embedder 切换会触发)
        if self.index.d != self.dim:
            logger.warning(
                "[RelationVectorStore] dim 不一致: index=%d, expected=%d, 重建",
                self.index.d, self.dim,
            )
            self.index = faiss.IndexFlatIP(self.dim)
            self._meta = []
            self._id_to_idx = {}
            self._save()

        self._initialized = True

    # ============================================================
    # 写
    # ============================================================
    def add_relation(
        self,
        memory_id: str,
        subj_canonical: str,
        pred: str,
        obj_canonical: str,
        pred_vector: np.ndarray,
        polarity: str = "support",
        idx_in_memory: int = 0,
    ) -> Optional[int]:
        """
        加一条 relation 向量.

        返回 faiss row idx, 失败返回 None.
        """
        rid = f"{memory_id}#{idx_in_memory}"  # relation 唯一 id

        if pred_vector.ndim != 1:
            logger.warning("[RelationVectorStore] pred_vector 不是 1D, 跳过: shape=%s", pred_vector.shape)
            return None
        if pred_vector.shape[0] != self.dim:
            logger.warning(
                "[RelationVectorStore] pred_vector dim=%d != index dim=%d, 跳过",
                pred_vector.shape[0], self.dim,
            )
            return None

        v = np.asarray(pred_vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v = v / norm
        else:
            return None

        metadata = {
            "rid": rid,
            "memory_id": memory_id,
            "subj_canonical": subj_canonical,
            "pred": pred,
            "obj_canonical": obj_canonical,
            "polarity": polarity,
            "idx_in_memory": idx_in_memory,
        }

        with self._write_lock:
            row_idx = self.index.ntotal
            self.index.add(v)
            self._meta.append(metadata)
            self._id_to_idx[rid] = row_idx
            self._save()
            return row_idx

    def add_relations_batch(
        self,
        memory_id: str,
        relations: List[Dict[str, Any]],
        pred_vectors: List[np.ndarray],
    ) -> List[int]:
        """
        批量加. relations = [{"subj_canonical", "pred", "obj_canonical", "polarity"}, ...]
        pred_vectors = [vec1, vec2, ...] 顺序对应
        """
        if not relations:
            return []
        if len(relations) != len(pred_vectors):
            logger.warning(
                "[RelationVectorStore] relations(%d) != pred_vectors(%d), 跳过",
                len(relations), len(pred_vectors),
            )
            return []

        # 校验所有向量
        valid_idx = []
        vectors = []
        metas = []
        for i, (r, vec) in enumerate(zip(relations, pred_vectors)):
            if vec is None or vec.ndim != 1 or vec.shape[0] != self.dim:
                continue
            v = np.asarray(vec, dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm < 1e-9:
                continue
            valid_idx.append(i)
            vectors.append(v / norm)
            metas.append({
                "rid": f"{memory_id}#{i}",
                "memory_id": memory_id,
                "subj_canonical": r.get("subj_canonical", ""),
                "pred": r.get("pred", ""),
                "obj_canonical": r.get("obj_canonical", ""),
                "polarity": r.get("polarity", "support"),
                "idx_in_memory": i,
            })

        if not vectors:
            return []

        with self._write_lock:
            start_idx = self.index.ntotal
            vec_matrix = np.stack(vectors)
            self.index.add(vec_matrix)
            row_indices = list(range(start_idx, start_idx + len(vectors)))
            for m, row_idx in zip(metas, row_indices):
                self._meta.append(m)
                self._id_to_idx[m["rid"]] = row_idx
            self._save()
            return row_indices

    # ============================================================
    # 读
    # ============================================================
    def search_by_pred_semantic(
        self,
        query_pred_vector: np.ndarray,
        top_k: int = 5,
        sim_threshold: float = 0.9,
        exclude_memory_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        按谓语语义相似度召回 (路 2: 只匹配谓语).

        参数:
            query_pred_vector  谓语 embedding
            top_k              最多返回多少条
            sim_threshold      相似度下限 (默认 0.9)
            exclude_memory_ids 排除某些 memory_id

        返回:
            [{"memory_id", "subj_canonical", "pred", "obj_canonical", "similarity", "row_idx"}, ...]
        """
        if self.index.ntotal == 0:
            return []

        v = np.asarray(query_pred_vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return []
        v = v / norm

        # 多取再 filter
        k_search = min(top_k * 3, self.index.ntotal)
        D, I = self.index.search(v, k_search)

        results = []
        for dist, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            if dist < sim_threshold:
                continue
            if 0 <= idx < len(self._meta):
                meta = dict(self._meta[idx])
                if meta.get("_deleted"):
                    continue
                mid = meta.get("memory_id")
                if exclude_memory_ids and mid in exclude_memory_ids:
                    continue
                meta["similarity"] = float(dist)
                meta["row_idx"] = int(idx)
                results.append(meta)
                if len(results) >= top_k:
                    break
        return results

    def search_by_subj_pred_obj(
        self,
        subj_canonical: str,
        query_pred_vector: np.ndarray,
        obj_canonical: str,
        top_k: int = 5,
        sim_threshold: float = 0.9,
        exclude_memory_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        按主+宾相同 + 谓语语义相似召回 (路 1).

        实现: 全表扫描 (subj, obj) 字符串匹配, 再用 sim 过滤
        (relation 索引里不存 SQL 字段, 用内存扫, 量小可接受)
        """
        if self.index.ntotal == 0:
            return []

        v = np.asarray(query_pred_vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return []
        v_norm = v / norm

        scored = []
        for i, m in enumerate(self._meta):
            if m.get("_deleted"):
                continue
            if m.get("subj_canonical") != subj_canonical:
                continue
            if m.get("obj_canonical") != obj_canonical:
                continue
            mid = m.get("memory_id")
            if exclude_memory_ids and mid in exclude_memory_ids:
                continue
            # 拿向量算 sim
            try:
                cand_vec = self.index.reconstruct(i)
            except Exception:
                continue
            cand_vec = np.asarray(cand_vec, dtype=np.float32)
            cand_norm = np.linalg.norm(cand_vec)
            if cand_norm < 1e-9:
                continue
            sim = float(np.dot(v_norm, cand_vec / cand_norm))
            if sim < sim_threshold:
                continue
            m_copy = dict(m)
            m_copy["similarity"] = sim
            m_copy["row_idx"] = i
            scored.append(m_copy)

        scored.sort(key=lambda x: -x["similarity"])
        return scored[:top_k]

    def get_metadata(self, row_idx: int) -> Optional[Dict[str, Any]]:
        if 0 <= row_idx < len(self._meta):
            return dict(self._meta[row_idx])
        return None

    def get_vector(self, row_idx: int) -> Optional[np.ndarray]:
        if row_idx < 0 or row_idx >= self.index.ntotal:
            return None
        try:
            v = self.index.reconstruct(int(row_idx))
            return np.asarray(v, dtype=np.float32).copy()
        except Exception:
            return None

    # ============================================================
    # 删除
    # ============================================================
    def remove_by_memory_id(self, memory_id: str) -> int:
        """
        删除一个 memory 的所有 relations (lazy tombstone).
        """
        removed = 0
        with self._write_lock:
            for m in self._meta:
                if m.get("memory_id") == memory_id and not m.get("_deleted"):
                    m["_deleted"] = True
                    rid = m.get("rid")
                    if rid:
                        self._id_to_idx.pop(rid, None)
                    removed += 1
            if removed:
                self._save()
        return removed

    # ============================================================
    # 状态
    # ============================================================
    def count_active(self) -> int:
        return sum(1 for m in self._meta if not m.get("_deleted"))

    def total(self) -> int:
        return self.index.ntotal

    def is_empty(self) -> bool:
        return self.count_active() == 0

    # ============================================================
    # 持久化
    # ============================================================
    def _save(self) -> None:
        try:
            RELATION_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(RELATION_FAISS_PATH))
            with open(RELATION_META_PATH, "wb") as f:
                pickle.dump(
                    {"meta": self._meta, "id_to_idx": self._id_to_idx},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as e:
            logger.exception("[RelationVectorStore] 落盘失败: %s", e)

    def get_backend_info(self) -> dict:
        return {
            "type": "faiss-cpu",
            "index": "IndexFlatIP",
            "dim": self.dim,
            "total": self.total(),
            "active": self.count_active(),
            "faiss_path": str(RELATION_FAISS_PATH),
            "meta_path": str(RELATION_META_PATH),
        }


# ============================================================
# 全局单例
# ============================================================
def get_relation_vector_store() -> RelationVectorStore:
    """获取全局 RelationVectorStore 单例. dim 自动从 embedder 读."""
    return RelationVectorStore()
