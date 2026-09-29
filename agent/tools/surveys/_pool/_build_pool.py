"""
_pool._build_pool — 拆出自原 _pool.py (9-20 体检重构)

包含函数: _build_candidate_pool
"""
from __future__ import annotations

import logging
import re
from typing import Any

from agent.infra.console import safe_print  # 控制台 cp936 编不了 emoji 时不崩
from agent.tools.config import (
    get_chroma_sim_threshold,
    get_min_pool_papers,
    get_relevance_threshold,
)

from ._openalex import _openalex_aps_search, _openalex_iop_search
from ._pool_audit import _audit_pool_with_retry
from ._utils import _arxiv_author_name

logger = logging.getLogger(__name__)

def _build_candidate_pool(
    topic: str,
    max_pool: int = 50,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Phase A: 候选池构建 (9-09 升级: 三道防线, 9-09 19:30 user 拍板返 status dict)

    Returns:
        (pool, build_status)  ← 9-09 改返 tuple, 调用方能区分 arxiv 失败原因
        - build_status["arxiv_status"]:
            "skipped"     pool 已够大, 没调 arxiv
            "0_results"   arxiv 调用成功但没返回 paper (常因中文 topic)
            "exception:TypeError: msg"   arxiv 抛异常 (网络/解析/库问题)
            "success"     arxiv 拉到了 paper

    三路召回, union + 去重:
      1. summary_items.db 关键词匹配 (title/abstract/core_method/core_contrib/keywords/outlook)
         - 关键词从 topic 切词 + 标题/摘要 token
         - 9-08 修: 不再过度依赖 keywords 字段 (很多核心论文 keywords 缺失)
      2. chroma 向量检索 (如果 vector_db 可用)
         - 9-08 新增: 走语义相似度, 召回 summary_items 关键词匹配漏掉的论文
         - 9-09 升级: 用相似度阈值 (默认 0.97, env QAGENT_SURVEY_CHROMA_SIM_THRESHOLD) 代替 top_k
           远距离论文不进入 pool, 防"quantum state verification" 召进 "quantum state tomography"
      3. arxiv API 兜底
         - 候选池 < 最小阈值 时, 拉 arxiv 补足 (9-08 user 拍板 抽 env QAGENT_SURVEY_MIN_POOL, 默认 30)

    9-08 user 拍板: 让 survey 工具能查看到"摘要数据库"更多内容, 不要只依赖 keywords 字段
    9-09 user 拍板: 加 per-paper LLM 相关性评分 (0-10) 精筛, 解决"邻近子领域误召"问题
                    (env QAGENT_SURVEY_RELEVANCE_THRESHOLD, 默认 7; 设 0 = 关闭)

    整体流程: 3 路 recall → union → relevance 精筛 → 按相关性排序 → 限 max_pool
    """
    # 候选池最小阈值 (9-08 抽 env, 默认 30, 充裕 = min+10)
    _MIN_POOL = get_min_pool_papers()
    from agent.tools.summary import list_summary_items

    # 9-08 修: 拆成多路召回, 每路各取 top, 然后 union + 去重
    seen_ids: set = set()
    pool_dict: dict[str, dict[str, Any]] = {}  # paper_id -> pool entry

    def add_pool_entry(entry: dict[str, Any]) -> None:
        """union + 去重"""
        pid = entry.get("paper_id", "")
        if not pid or pid in seen_ids:
            return
        seen_ids.add(pid)
        pool_dict[pid] = entry

    # 关键词切分 (中文 + 英文)
    keywords = [kw.strip() for kw in re.split(r"[\s,，、。;；:：]+", topic) if len(kw.strip()) >= 2]
    if not keywords:
        keywords = [topic]

    # ========== 1. summary_items 关键词匹配 ==========
    try:
        raw = list_summary_items(limit=max_pool * 4)  # 多拿点
        if raw:
            def kw_score(item: dict[str, Any]) -> int:
                # 9-08 修: 关键词字段权重降低, title + abstract 权重高
                score = 0
                for field, w in [
                    ("title", 3),
                    ("abstract", 2),
                    ("core_method", 2),
                    ("core_contrib", 2),
                    ("keywords", 1),       # 9-08 降权 (用户实测很多核心论文 keywords 缺失)
                    ("outlook", 1),
                ]:
                    v = (item.get(field) or "").lower()
                    for kw in keywords:
                        if kw.lower() in v:
                            score += w
                return score

            scored = [(kw_score(it), it) for it in raw]
            scored = [(s, it) for s, it in scored if s > 0]
            scored.sort(key=lambda x: -x[0])
            for score, it in scored[:max_pool]:
                add_pool_entry({
                    "paper_id": it.get("paper_id", ""),
                    "title": it.get("title", ""),
                    "year": (it.get("summary_date") or "")[:4] if it.get("summary_date") else "",
                    "journal": it.get("source", ""),
                    "core_method": it.get("core_method", ""),
                    "core_contrib": it.get("core_contrib", ""),
                    "abstract": it.get("abstract", ""),
                    "outlook": it.get("outlook", ""),
                    "keywords": it.get("keywords", ""),
                    "_match_score": score,
                    "_source": "summary_items_keyword",
                })
    except Exception as e:
        logger.warning("[_build_candidate_pool] summary_items 召回失败: %s", e)

    # ========== 2. chroma 向量检索 (9-09 升级: 相似度阈值) ==========
    # 9-08 user 拍板: 修复底层索引, 让 survey 能查看到更多数据库信息
    # 9-09 user 拍板: 改 top_k 为相似度阈值 (默认 0.97), 防"邻近子领域"误召
    #   - sim 0.97 = distance 0.03 (chroma cosine distance)
    #   - 比"quantum state verification" vs "quantum state tomography" 在 embedding 空间更近
    #     的论文全踢掉, 只留真正主题的
    #   - 用 top_k = max(50, max_pool*2) 多拉点, 再用阈值卡
    try:
        from agent.vector_db import get_vector_db
        vdb = get_vector_db()
        if vdb and vdb.count() > 0:
            sim_threshold = get_chroma_sim_threshold()  # 默认 0.97
            dist_threshold = 1.0 - sim_threshold  # 默认 0.03
            # 多拉候选, 阈值过滤后会少; 留 2x 余量
            chroma_top_k = max(50, max_pool * 2)
            chroma_results = vdb.search(topic, top_k=chroma_top_k)
            kept = 0
            for r in chroma_results:
                meta = r.get("metadata", {}) or {}
                pid = meta.get("paper_id") or r.get("id", "")
                if not pid:
                    continue
                distance = r.get("distance", 1.0) or 1.0
                # 9-09 升级: 相似度阈值过滤 (cosine distance > dist_threshold 跳过)
                if distance > dist_threshold:
                    continue
                kept += 1
                add_pool_entry({
                    "paper_id": pid,
                    "title": meta.get("title", ""),
                    "year": str(meta.get("year", "")),
                    "journal": meta.get("journal") or meta.get("source", ""),
                    "core_method": meta.get("core_method", ""),
                    "core_contrib": meta.get("core_contrib", ""),
                    "abstract": (r.get("document") or "")[:1000],  # chroma 文档截短
                    "outlook": meta.get("outlook", ""),
                    "keywords": meta.get("keywords", ""),
                    "_match_score": int((1 - distance) * 100) if distance is not None else 0,
                    "_source": "chroma_vector",
                })
            logger.info(
                "[_build_candidate_pool] chroma: top_k=%d, sim>=%.2f, kept=%d",
                chroma_top_k, sim_threshold, kept,
            )
    except Exception as e:
        # chroma 不可用是正常的 (项目可选), 不影响主流程
        logger.debug("[_build_candidate_pool] chroma 召回跳过: %s", e)

    # ========== 3. APS / IOP 召回 (9-18 加: 跨多源通用论文搜索) ==========
    # 之前只走 arxiv 兜底, 漏了 quant-ph 主阵地 (PRL / NJP 等) 大量论文.
    # 现在: APS + IOP + arxiv 三路外部召回 (并行), pool 不再卡死在 arxiv 一家.
    # APS / IOP 走 OpenAlex (跟 search_aps / search_iop 同源), 优先 quant-ph 主阵地.
    # 9-18 user 拍板: "survey 工具应该用搜索工具搜索, 不要总用 arxiv".
    external_status: dict[str, str] = {}  # 每路状态: skipped / 0_results / success / exception:...

    # ========== 3a. APS (OpenAlex, PRL/PRA/PRXQuantum/PRX/RMP) ==========
    if len(pool_dict) < _MIN_POOL:
        # 9-18: 优先 APS 召回. 一次拉足 min_pool 篇, 不要稀稀拉拉 5 篇.
        need = max(_MIN_POOL, max_pool) - len(pool_dict)
        aps_k = min(need, 30)  # OpenAlex 单次 max 200, 限 30 避免太慢
        safe_print(
            f"[_build_candidate_pool] APS 召回 start  topic={topic[:60]!r}  k={aps_k}",
            flush=True,
        )
        try:
            aps_papers = _openalex_aps_search(topic, aps_k)
            aps_added = 0
            for p in aps_papers:
                add_pool_entry(p)
                aps_added += 1
            external_status["aps"] = "success" if aps_added > 0 else "0_results"
            safe_print(
                f"[_build_candidate_pool] APS 召回 done  +{aps_added} 篇  pool={len(pool_dict)}",
                flush=True,
            )
        except Exception as e:
            external_status["aps"] = f"exception:{type(e).__name__}:{str(e)[:80]}"
            safe_print(
                f"[_build_candidate_pool] APS 召回 ❌ {external_status['aps']}",
                flush=True,
            )
            logger.warning("[_build_candidate_pool] APS 召回失败: %s", e)
    else:
        external_status["aps"] = "skipped"
        safe_print(f"[_build_candidate_pool] APS 召回 skipped  (pool 已够大 {len(pool_dict)}>={_MIN_POOL})", flush=True)

    # ========== 3b. IOP (OpenAlex, NJP/J.Phys.A/J.Phys.B/J.Opt.) ==========
    if len(pool_dict) < _MIN_POOL:
        need = max(_MIN_POOL, max_pool) - len(pool_dict)
        iop_k = min(need, 30)
        safe_print(
            f"[_build_candidate_pool] IOP 召回 start  topic={topic[:60]!r}  k={iop_k}",
            flush=True,
        )
        try:
            iop_papers = _openalex_iop_search(topic, iop_k)
            iop_added = 0
            for p in iop_papers:
                add_pool_entry(p)
                iop_added += 1
            external_status["iop"] = "success" if iop_added > 0 else "0_results"
            safe_print(
                f"[_build_candidate_pool] IOP 召回 done  +{iop_added} 篇  pool={len(pool_dict)}",
                flush=True,
            )
        except Exception as e:
            external_status["iop"] = f"exception:{type(e).__name__}:{str(e)[:80]}"
            safe_print(
                f"[_build_candidate_pool] IOP 召回 ❌ {external_status['iop']}",
                flush=True,
            )
            logger.warning("[_build_candidate_pool] IOP 召回失败: %s", e)
    else:
        external_status["iop"] = "skipped"
        safe_print(f"[_build_candidate_pool] IOP 召回 skipped  (pool 已够大 {len(pool_dict)}>={_MIN_POOL})", flush=True)

    # ========== 3c. arxiv API 兜底 (9-08: 最小阈值走 env QAGENT_SURVEY_MIN_POOL, 默认 30) ==========
    # 9-18: 从 "唯一外部源" 降级为 "第三路兜底". APS / IOP 都没命中时才调 arxiv.
    if len(pool_dict) < _MIN_POOL:
        arxiv_status = "0_results"  # 默认 0 results, 拉到了再改 success
        safe_print(
            f"[_build_candidate_pool] arxiv 兜底 start  topic={topic[:60]!r}  pool 还差 "
            f"{_MIN_POOL - len(pool_dict)} 篇",
            flush=True,
        )
        try:
            import arxiv
            need = max(_MIN_POOL, max_pool) - len(pool_dict)
            search = arxiv.Search(
                query=topic,
                max_results=min(need, 20),  # 限 20 篇避免太慢
                sort_by=arxiv.SortCriterion.Relevance,
            )
            client = arxiv.Client()
            arxiv_added = 0
            for r in client.results(search):
                pid = r.entry_id.split("/")[-1].split("v")[0]
                if pid in seen_ids:
                    continue
                add_pool_entry({
                    "paper_id": pid,
                    "title": r.title or "",
                    "year": str(r.published.year) if r.published else "",
                    "journal": "arxiv",
                    "core_method": "",
                    "core_contrib": (r.summary or "")[:200],
                    "abstract": r.summary or "",
                    "outlook": "",
                    "keywords": ", ".join([_arxiv_author_name(a) for a in (r.authors or [])[:3]]),
                    "_match_score": 0,
                    "_source": "arxiv_api_fallback",
                })
                arxiv_added += 1
                if len(pool_dict) >= max(_MIN_POOL, max_pool):
                    break
            if arxiv_added > 0:
                arxiv_status = "success"
            external_status["arxiv"] = arxiv_status
            safe_print(
                f"[_build_candidate_pool] arxiv 兜底 done  +{arxiv_added} 篇  pool={len(pool_dict)}",
                flush=True,
            )
        except Exception as e:
            arxiv_status = f"exception:{type(e).__name__}:{str(e)[:100]}"
            external_status["arxiv"] = arxiv_status
            safe_print(
                f"[_build_candidate_pool] arxiv 兜底 ❌ {arxiv_status}",
                flush=True,
            )
            logger.warning("[_build_candidate_pool] arxiv 兜底失败: %s", e)
    else:
        arxiv_status = "skipped"  # pool 已够大, 没调 arxiv
        external_status["arxiv"] = "skipped"
        safe_print(f"[_build_candidate_pool] arxiv 兜底 skipped  (pool 已够大 {len(pool_dict)}>={_MIN_POOL})", flush=True)

    # ========== 4. 9-18 加: audit retry loop (严格筛选 + 自动重搜) ==========
    # 9-18 user 拍板: 在构建候选池的时候就严格审核, 不通过的 paper 全部踢掉 (不再 rescue),
    #                 pool 不够 min_pool 时自动 retry (换 query / 换 source), 直到找到足够多合格论文.
    # audit 结论写到 build_status["audit_state"], 供 survey_generate 塞进 state 传给 ReAct.
    audit_state = _audit_pool_with_retry(
        topic=topic,
        pool_dict=pool_dict,
        min_pool=_MIN_POOL,
        max_pool=max_pool,
        relevance_threshold=get_relevance_threshold(),
        print_prefix="[_build_candidate_pool]",
    )
    pool_dict = audit_state["kept_dict"]

    # 排序: _relevance_score 降序 (relevance 跑过后) > _match_score (未跑) > source 优先级
    # 9-18 加: APS / IOP / arxiv 三路外部源都进 source_priority 排序
    source_priority = {
        "chroma_vector": 0,
        "summary_items_keyword": 1,
        "openalex_aps_api": 2,
        "openalex_iop_api": 3,
        "arxiv_api_fallback": 4,
    }
    def _sort_key(x: dict[str, Any]) -> tuple:
        rel = x.get("_relevance_score")
        if rel is not None:
            return (-rel, 0, source_priority.get(x.get("_source", ""), 9))
        return (-x.get("_match_score", 0), 1, source_priority.get(x.get("_source", ""), 9))

    pool = sorted(pool_dict.values(), key=_sort_key)
    # 限 max_pool
    # 9-18: build_status 扩字段, 仍保留 arxiv_status 向后兼容, 新增 audit_state 给 ReAct 看到
    return pool[:max_pool], {
        "arxiv_status": arxiv_status,
        "external_status": external_status,
        "audit_state": audit_state["summary"],
    }
