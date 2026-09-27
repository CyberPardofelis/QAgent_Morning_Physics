"""
agent.memory.processor — 记忆库总控

读侧：把记忆格式化成 markdown context（决策 2=b，默认不自动注入，备查）
写侧：save_memory 主流程（encode + SQLite + faiss + 同步）

注意：决策 1=b（关键词触发）的实现：
- 后端不自动后台提取（避免噪音）
- 由 LLM 在 MEMORY_GUIDE 看到"包含 X 关键词就调 save_memory"提示后自行决定
- processor.should_suggest_save(text) 提供一致性判断
"""
import logging  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Any  # noqa: E402

from .db_manager import MemoryDBManager  # noqa: E402
from .embedder import MemoryEmbedder  # noqa: E402
from .retriever import MemoryRetriever  # noqa: E402
from .schemas import (  # noqa: E402
    DEFAULT_SCOPE,
    MEMORY_TYPES,
    TRIGGER_KEYWORDS,
    content_has_trigger_keyword,
)
from .vector_store import MemoryVectorStore  # noqa: E402

logger = logging.getLogger(__name__)
from .extractor import extract as _llm_extract  # noqa: E402
from .judger import judge as _judge_triples  # noqa: E402


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
    # ============================================================
    # 去重检测（save 前调用）
    # ============================================================
    def check_duplicate(
        self,
        content: str,
        sim_merge: float = 0.99,
        sim_similar: float = 0.75,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        检查要保存的内容是否与库内已有记忆重复。

        判定策略 (v3.2 三路召回 + judger):
          1. LLM 抽取三元组 → judger 判定 (same_fact / opposite / related_* / unrelated)
          2. 三路召回候选 (每路都去重合并):
             - 路 0: 整体 embedding sim top-3 (embedding 不可用时跳过)
             - 路 1: 主+宾相同 + 谓语语义 sim>0.9 (embedding 不可用时跳过, 最多 5)
             - 路 2: 仅谓语语义 sim>0.9 (embedding 不可用时跳过, 最多 5)
             - 路 3: concept_recall 降级 (SQL LIKE 按 canonical, 不依赖 embedding, 最多 top_k*3)
          3. embedding similarity 兜底

        返回:
            {
                "verdict": "duplicate" | "opposite" | "similar" | "unique",
                "top_match": {memory_dict, similarity} | None,
                "all_matches": [{memory_dict, similarity, _via}, ...] (deduped),
                "judge_verdict": "...",
                "judge_reason": "..."
            }

        判定:
            - same_fact → verdict="duplicate" (合并)
            - opposite → verdict="opposite" (不要合并, 保留对立)
            - related_* → verdict="similar" (提示用户)
            - unrelated → verdict 由 embedding/concept_recall 决定
        """
        # Step 1: LLM 抽取
        new_extract = None
        try:
            new_extract = _llm_extract(content)
        except Exception as e:
            logger.debug("[MemoryProcessor] extract 异常: %s", e)

        # Step 2: 多路召回
        all_matches: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # ---- 路 0: 整体 embedding 召回 ----
        try:
            vec = self.embedder.encode(content)
        except RuntimeError as e:
            logger.debug("[MemoryProcessor] check_duplicate embedding 失败: %s", e)
            vec = None
        if vec is not None:
            try:
                hits = self.vs.search(vec, top_k=top_k)
                if hits:
                    ids = [self.vs.get_metadata(ri).get("id") for ri, _ in hits if self.vs.get_metadata(ri)]
                    sim_map = {self.vs.get_metadata(ri).get("id"): sim for ri, sim in hits if self.vs.get_metadata(ri)}
                    memories = self.db.list_by_ids([mid for mid in ids if mid])
                    for m in memories:
                        mid = m["id"]
                        if mid in seen_ids:
                            continue
                        seen_ids.add(mid)
                        all_matches.append({"memory": m, "similarity": sim_map.get(mid, 0.0), "_via": "embedding"})
                    logger.debug("[MemoryProcessor] check_duplicate 路 0 (embedding) 命中 %d", len(hits))
            except Exception as e:
                logger.warning("[MemoryProcessor] check_duplicate faiss 失败: %s", e)

        # ---- 路 1 + 路 2: 按 entry 召回 (谓语向量) ----
        # 跟路 0 共享 embedder 状态: 如果路 0 失败, 这里也跳过
        relations_for_recall: list[dict[str, Any]] = []
        if new_extract:
            relations_for_recall = new_extract.get("relations", []) or []
        if vec is not None and relations_for_recall:
            try:
                from .relation_vector_store import get_relation_vector_store
                rvs = get_relation_vector_store()
                entry_candidates: dict[str, dict[str, Any]] = {}  # memory_id -> {memory, similarity, _via}
                for rel in relations_for_recall:
                    subj = rel.get("subj_canonical", "")
                    pred = rel.get("pred", "")
                    obj = rel.get("obj_canonical", "")
                    if not pred:
                        continue
                    # 编码谓语
                    try:
                        pred_vec = self.embedder.encode(pred)
                    except RuntimeError:
                        continue

                    # 路 1: 主+宾相同 + 谓语相似
                    if subj and obj:
                        l1_hits = rvs.search_by_subj_pred_obj(
                            subj_canonical=subj,
                            query_pred_vector=pred_vec,
                            obj_canonical=obj,
                            top_k=5,
                            sim_threshold=0.9,
                        )
                        for h in l1_hits:
                            mid = h.get("memory_id")
                            if not mid or mid in seen_ids:
                                continue
                            # 拿完整 memory
                            mem = self.db.get(mid)
                            if not mem:
                                continue
                            seen_ids.add(mid)
                            entry_candidates[mid] = {
                                "memory": mem,
                                "similarity": h["similarity"],
                                "_via": "subj_pred_obj",
                            }
                            if len(entry_candidates) >= 5:
                                break
                    if len(entry_candidates) >= 5:
                        break

                    # 路 2: 仅谓语语义
                    l2_hits = rvs.search_by_pred_semantic(
                        query_pred_vector=pred_vec,
                        top_k=5,
                        sim_threshold=0.9,
                    )
                    for h in l2_hits:
                        mid = h.get("memory_id")
                        if not mid or mid in seen_ids:
                            continue
                        mem = self.db.get(mid)
                        if not mem:
                            continue
                        seen_ids.add(mid)
                        entry_candidates[mid] = {
                            "memory": mem,
                            "similarity": h["similarity"],
                            "_via": "pred_semantic",
                        }
                        if len(entry_candidates) >= 5:
                            break
                all_matches.extend(entry_candidates.values())
                logger.debug("[MemoryProcessor] check_duplicate 路 1+2 (entry) 命中 %d", len(entry_candidates))
            except Exception as e:
                logger.debug("[MemoryProcessor] check_duplicate entry 召回失败: %s", e)

        # ---- 路 3: concept_recall 降级 (SQL LIKE 按 canonical, 不依赖 embedding) ----
        # 当路 0/1/2 没召回时 (embedding 不可用) 自动生效
        if not all_matches and new_extract:
            try:
                from .retriever import MemoryRetriever
                retriever = MemoryRetriever()
                concept_recalls = retriever._concept_recall(content, scope=None, limit=top_k * 3)
                for m in concept_recalls:
                    mid = m["id"]
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                    all_matches.append({"memory": m, "similarity": 0.0, "_via": "concept_recall"})
                logger.debug("[MemoryProcessor] check_duplicate 路 3 (concept_recall) 命中 %d", len(concept_recalls))
            except Exception as e:
                logger.warning("[MemoryProcessor] check_duplicate concept_recall 失败: %s", e)

        # Step 3: judger 逐个判断
        # 9-27 修: 记录"实际被判 same_fact/opposite 的那条", 供 top_match 使用。
        # 之前直接返回 all_matches[0]（embedding 相似度最高的那条）, 跟 judge 结论不是同一条,
        # 会把 LLM 导向错误的 update_memory 目标。
        judge_verdict = "unrelated"
        judge_reason = ""
        judged_match: dict[str, Any] | None = None
        for match in all_matches:
            existing_extract = (match["memory"].get("metadata") or {}).get("extract") or {}
            if not existing_extract:
                continue
            verdict, reason = _judge_triples(
                new_extract or {"core_concepts": [], "relations": [], "negation": False},
                existing_extract,
            )
            match["judge_verdict"] = verdict
            match["judge_reason"] = reason
            if verdict in ("same_fact", "opposite"):
                judge_verdict = verdict
                judge_reason = reason
                judged_match = match
                break

        # Step 4: 综合 verdict
        if judge_verdict in ("same_fact", "opposite"):
            return {
                "verdict": "duplicate" if judge_verdict == "same_fact" else "opposite",
                # top_match = 真正被 judge 判定的条目（不是相似度最高的那条）
                "top_match": judged_match,
                "all_matches": all_matches,
                "judge_verdict": judge_verdict,
                "judge_reason": judge_reason,
            }

        # 回落到 similarity 判断
        if not all_matches:
            return {
                "verdict": "unique",
                "top_match": None,
                "all_matches": [],
                "judge_verdict": judge_verdict,
                "judge_reason": judge_reason,
            }
        top_sim = all_matches[0]["similarity"]
        if top_sim >= sim_merge:
            verdict = "duplicate"
        elif top_sim >= sim_similar:
            verdict = "similar"
        else:
            verdict = "unique"
        return {
            "verdict": verdict,
            "top_match": all_matches[0],
            "all_matches": all_matches,
            "judge_verdict": judge_verdict,
            "judge_reason": judge_reason,
        }

    def save(
        self,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        related_paper_id: str | None = None,
        source_conv_id: str | None = None,
        metadata: dict[str, Any] | None = None,
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
            # 注意：不能 return, enrich 还要跑（enrich 跟 embedding 独立）
            vec = None

        if vec is not None:
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

        # 调 LLM 抽取元数据 (后台, 不报错)
        try:
            self._enrich_with_extract(memory_id, content)
        except Exception as e:
            logger.debug("[MemoryProcessor] enrich 错误: %s", e)

        logger.info(
            "[MemoryProcessor] saved: id=%s type=%s importance=%.2f content='%s...'",
            memory_id[:8], memory_type, importance, content[:30],
        )
        return memory_id

    def save_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[str]:
        """批量 save。"""
        ids = []
        for it in items:
            mid = self.save(**it)
            ids.append(mid)
        return ids

    def _enrich_with_extract(self, memory_id: str, content: str) -> None:
        """
        使用 LLM 抽取元数据 (核心概念 / 三元组 / 否定) 追加到 metadata 字段.
        仅在抽取成功时更新, 不覆盖现有 metadata.
        """
        if not content:
            return
        try:
            ext = _llm_extract(content)
        except Exception as e:
            logger.debug("[MemoryProcessor] extract 调用异常: %s", e)
            return
        if not ext:
            return
        if not ext.get("core_concepts") and not ext.get("relations"):
            return
        # 读现有 metadata, 合并
        existing = self.db.get(memory_id)
        if not existing:
            return
        meta = existing.get("metadata") or {}
        meta["extract"] = ext
        meta["extracted_at"] = __import__("datetime").datetime.now().isoformat()
        # 写回 SQLite
        try:
            from agent.infra.db import connect
            with connect("memory") as conn:
                conn.execute(
                    "UPDATE memories SET metadata = ? WHERE id = ?",
                    (__import__("json").dumps(meta, ensure_ascii=False), memory_id),
                )
        except Exception as e:
            logger.warning("[MemoryProcessor] update metadata 失败: %s", e)

        # [v3.2] 把每条 relation 的 pred 向量存进 relation 索引, 供 _recall_by_pred 召回
        # 注意: embedding 不可用时跳过 (跟主向量库一致, 不报错)
        relations = ext.get("relations", []) or []
        if relations:
            try:
                from .relation_vector_store import get_relation_vector_store
                rvs = get_relation_vector_store()
                pred_vectors = []
                for r in relations:
                    pred = r.get("pred", "")
                    if not pred:
                        pred_vectors.append(None)
                        continue
                    try:
                        vec = self.embedder.encode(pred)
                        pred_vectors.append(vec)
                    except RuntimeError:
                        pred_vectors.append(None)
                # 只 add 不为 None 的 (add_relations_batch 内部会过滤)
                rvs.add_relations_batch(memory_id, relations, pred_vectors)
            except Exception as e:
                logger.debug("[MemoryProcessor] 存 relation 向量失败: %s", e)

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
            self.vs.get_row_idx(memory_id + "_old")  # 不存在，正常
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
        scope: str | None = None,
        memory_type: str | None = None,
        related_paper_id: str | None = None,
        min_importance: float | None = None,
        method: str = "vector",
    ) -> list[dict[str, Any]]:
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

    def recall_for_paper(self, paper_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return self.retriever.retrieve_for_paper(paper_id, limit=limit)

    def recall_important(self, limit: int = 10, min_importance: float = 0.7) -> list[dict[str, Any]]:
        return self.retriever.retrieve_important(limit=limit, min_importance=min_importance)

    def list_all(
        self,
        scope: str | None = None,
        memory_type: str | None = None,
        min_importance: float | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self.db.list_recent(
            scope=scope, memory_type=memory_type, min_importance=min_importance, limit=limit,
        )

    # ============================================================
    # 格式化
    # ============================================================
    def format_for_context(
        self,
        memories: list[dict[str, Any]],
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

    def _format_one_memory(self, idx: int, m: dict[str, Any], include_score: bool) -> str:
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
    def get_stats(self) -> dict[str, Any]:
        return {
            "sqlite": self.db.get_statistics(),
            "vector_store": self.vs.get_backend_info(),
            "embedder": self.embedder.get_backend_info(),
            "trigger_keywords": list(TRIGGER_KEYWORDS),
            "memory_types": MEMORY_TYPES,
        }
