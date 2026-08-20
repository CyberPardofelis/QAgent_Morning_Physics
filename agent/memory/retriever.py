"""
agent.memory.retriever — 混合检索（向量 + 关键词 + 衰减）

设计：
- 主路径：向量检索（语义匹配）
- fallback：向量结果不足时，关键词检索补
- 排序：向量 sim + importance 权重，叠加 time decay
- 过滤：scope / memory_type / related_paper_id

为什么需要 importance + decay？
- 长期来看，"记住" 的东西应该比"试试看" 的东西更早被检索到
- 但太老的"记住"也应该自然衰减（让用户忘记不重要的细节）
- 公式: final_score = sim * (α * importance + (1-α) * 1) * exp(-decay_rate * days)
"""
import logging
import math
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

from .db_manager import MemoryDBManager
from .vector_store import MemoryVectorStore
from .embedder import MemoryEmbedder

logger = logging.getLogger(__name__)


# 中文停用词（简版；做关键词分词时过滤）
_STOPWORDS_ZH: Set[str] = {
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "我们", "你们", "他们",
    "这个", "那个", "什么", "怎么", "为什么", "然后", "就是", "可以", "觉得", "认为",
    "这", "那", "吗", "吧", "呢", "啊", "嗯", "哦", "好", "是", "不", "没",
    "的", "了", "和", "与", "或", "也", "都", "就", "还", "才",
}


class MemoryRetriever:
    """
    记忆检索器。

    用法：
        r = MemoryRetriever()
        results = r.retrieve(query="用户偏好 LaTeX 公式", limit=5)
    """

    def __init__(self, db: Optional[MemoryDBManager] = None, vs: Optional[MemoryVectorStore] = None,
                 embedder: Optional[MemoryEmbedder] = None):
        self.db = db or MemoryDBManager()
        self.vs = vs or MemoryVectorStore()
        self.embedder = embedder or MemoryEmbedder()

    # ============================================================
    # 主入口
    # ============================================================
    def retrieve(
        self,
        query: str,
        limit: int = 5,
        scope: Optional[str] = None,
        memory_type: Optional[str] = None,
        related_paper_id: Optional[str] = None,
        min_importance: Optional[float] = None,
        min_similarity: float = 0.0,
        method: str = "vector",
        importance_weight: float = 0.3,
        decay_rate: float = 0.0,  # 默认不衰减（v1 简化，未来按需开）
    ) -> List[Dict[str, Any]]:
        """
        检索记忆。

        参数：
            query           用户查询文本（会 encode）
            limit           返回条数
            scope/memory_type/related_paper_id 过滤
            min_importance  importance 下限
            min_similarity  向量相似度下限
            method          "vector" / "keyword" / "hybrid"
            importance_weight  最终分数 = sim * (w*importance + (1-w)*1)
            decay_rate      时间衰减率（每日 e^(-rate) 倍衰减）

        返回：
            记忆列表（带 _score 字段），按分数降序
        """
        if not query or not query.strip():
            return []

        # 应用 scope 默认值
        if scope is None:
            scope = "default"

        # 应用 related_paper_id 过滤（SQL 层做）
        if related_paper_id is not None:
            # 强相关 paper → 直接走 SQL 拿全部
            results = self.db.list_recent(
                scope=scope,
                memory_type=memory_type,
                related_paper_id=related_paper_id,
                min_importance=min_importance,
                limit=limit,
            )
            for r in results:
                r["_score"] = 1.0
                r["_method"] = "paper_id_exact"
            return results

        if method == "keyword":
            return self._keyword_retrieve(
                query, limit, scope, memory_type, min_importance,
            )
        if method == "hybrid":
            return self._hybrid_retrieve(
                query, limit, scope, memory_type, min_importance,
                importance_weight, decay_rate,
            )
        # 默认 vector
        return self._vector_retrieve(
            query, limit, scope, memory_type, min_importance,
            min_similarity, importance_weight, decay_rate,
        )

    # ============================================================
    # 向量检索
    # ============================================================
    def _vector_retrieve(
        self,
        query: str,
        limit: int,
        scope: Optional[str],
        memory_type: Optional[str],
        min_importance: Optional[float],
        min_similarity: float,
        importance_weight: float,
        decay_rate: float,
    ) -> List[Dict[str, Any]]:
        # 1) 编码查询
        try:
            q_vec = self.embedder.encode(query)
        except RuntimeError as e:
            logger.warning("[MemoryRetriever] embedding 失败: %s", e)
            # fallback 到关键词
            return self._keyword_retrieve(
                query, limit, scope, memory_type, min_importance,
            )

        # 2) 向量搜索
        # 多取一些（后面要按 importance / SQL 过滤再截断）
        k_search = min(max(limit * 3, 30), self.vs.total() or 1)
        hits = self.vs.search(q_vec, top_k=k_search)

        if not hits:
            # fallback 关键词
            return self._keyword_retrieve(
                query, limit, scope, memory_type, min_importance,
            )

        # 3) 拿 metadata + 查 SQLite 拿完整信息
        candidate_ids = []
        candidate_sim = {}  # row_idx -> sim
        for row_idx, sim in hits:
            meta = self.vs.get_metadata(row_idx)
            if not meta:
                continue
            mid = meta.get("id")
            if not mid:
                continue
            candidate_ids.append(mid)
            candidate_sim[mid] = sim

        if not candidate_ids:
            return []

        memories = self.db.list_by_ids(candidate_ids)

        # 4) 应用过滤 + 算分
        scored: List[Dict[str, Any]] = []
        for mem in memories:
            mid = mem["id"]
            if scope is not None and mem.get("scope") != scope:
                continue
            if memory_type is not None and mem.get("memory_type") != memory_type:
                continue
            if min_importance is not None and mem.get("importance", 0) < min_importance:
                continue

            sim = candidate_sim.get(mid, 0.0)
            if sim < min_similarity:
                continue

            imp = mem.get("importance", 0.5) or 0.5
            imp_factor = importance_weight * imp + (1.0 - importance_weight) * 1.0
            score = sim * imp_factor

            if decay_rate > 0:
                score = self._apply_decay(score, mem.get("created_at", ""), decay_rate)

            mem["_score"] = score
            mem["_similarity"] = sim
            mem["_importance"] = imp
            mem["_method"] = "vector"
            scored.append((score, mem))

        # 5) 按分数排序截断
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [m for _, m in scored[:limit]]

        # 6) 更新访问记录
        for mem in results:
            try:
                self.db.touch(mem["id"])
            except Exception as e:
                logger.debug("[MemoryRetriever] touch 失败: %s", e)

        return results

    # ============================================================
    # 关键词检索
    # ============================================================
    def _keyword_retrieve(
        self,
        query: str,
        limit: int,
        scope: Optional[str],
        memory_type: Optional[str],
        min_importance: Optional[float],
    ) -> List[Dict[str, Any]]:
        keywords = self._extract_keywords(query)
        if not keywords:
            keywords = [query[:30]]  # fallback

        # 合并结果（多关键词命中加分）
        id_to_score: Dict[str, float] = {}
        for kw in keywords[:5]:  # 最多 5 个关键词
            results = self.db.search_keyword(kw, scope=scope, limit=limit * 2)
            for mem in results:
                if memory_type is not None and mem.get("memory_type") != memory_type:
                    continue
                if min_importance is not None and mem.get("importance", 0) < min_importance:
                    continue
                mid = mem["id"]
                id_to_score[mid] = id_to_score.get(mid, 0) + 1.0

        # 取 top
        sorted_ids = sorted(id_to_score.keys(), key=lambda x: id_to_score[x], reverse=True)[:limit]
        memories = self.db.list_by_ids(sorted_ids)
        for mem in memories:
            mem["_score"] = id_to_score[mem["id"]]
            mem["_similarity"] = 0.0
            mem["_method"] = "keyword"
        return memories

    # ============================================================
    # 混合检索
    # ============================================================
    def _hybrid_retrieve(
        self,
        query: str,
        limit: int,
        scope: Optional[str],
        memory_type: Optional[str],
        min_importance: Optional[float],
        importance_weight: float,
        decay_rate: float,
    ) -> List[Dict[str, Any]]:
        # 多取一些
        vec_results = self._vector_retrieve(
            query, limit * 2, scope, memory_type, min_importance,
            min_similarity=0.0, importance_weight=importance_weight, decay_rate=decay_rate,
        )
        kw_results = self._keyword_retrieve(
            query, limit * 2, scope, memory_type, min_importance,
        )

        # 归一化合并
        vec_scores = {m["id"]: m.get("_score", 0) for m in vec_results}
        kw_scores = {}
        for m in kw_results:
            kw_scores[m["id"]] = m.get("_score", 0) / max(1.0, max(kw_results and [k.get("_score", 1) for k in kw_results] or [1.0]))

        id_to_mem: Dict[str, Dict[str, Any]] = {}
        for m in vec_results:
            id_to_mem[m["id"]] = m
        for m in kw_results:
            if m["id"] not in id_to_mem:
                id_to_mem[m["id"]] = m

        final_scores = {}
        for mid in id_to_mem:
            v = vec_scores.get(mid, 0.0)
            k = kw_scores.get(mid, 0.0)
            # 向量权重 0.7, 关键词权重 0.3
            final_scores[mid] = v * 0.7 + k * 0.3

        # 排序
        sorted_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:limit]
        memories = self.db.list_by_ids(sorted_ids)
        for mem in memories:
            mem["_score"] = final_scores[mem["id"]]
            mem["_method"] = "hybrid"
        return memories

    # ============================================================
    # 按类型查（不基于 query）
    # ============================================================
    def retrieve_by_type(
        self,
        memory_type: str,
        scope: Optional[str] = None,
        limit: int = 10,
        min_importance: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return self.db.list_recent(
            scope=scope, memory_type=memory_type, min_importance=min_importance, limit=limit
        )

    def retrieve_important(
        self,
        scope: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.7,
    ) -> List[Dict[str, Any]]:
        return self.db.list_recent(
            scope=scope, min_importance=min_importance, limit=limit
        )

    def retrieve_for_paper(
        self,
        paper_id: str,
        scope: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        return self.db.list_recent(
            scope=scope, related_paper_id=paper_id, limit=limit
        )

    # ============================================================
    # 辅助
    # ============================================================
    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """从文本中提取关键词（过滤停用词 + 短词）。"""
        words = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9_]+", text)
        keywords = [w for w in words if len(w) >= 2 and w not in _STOPWORDS_ZH]
        return keywords[:8]

    @staticmethod
    def _apply_decay(score: float, created_at: str, decay_rate: float) -> float:
        """时间衰减：score *= exp(-decay_rate * days)"""
        if not created_at or decay_rate <= 0:
            return score
        try:
            created = datetime.fromisoformat(created_at)
            days = (datetime.now() - created).total_seconds() / 86400
            if days <= 0:
                return score
            return score * math.exp(-decay_rate * days)
        except Exception:
            return score
