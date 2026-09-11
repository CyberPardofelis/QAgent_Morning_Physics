"""
agent.tools.surveys._pool — 候选池构建 (Phase A) + 引用审计 + LLM relevance 精筛

从原 surveys.py 抽出 (P1.7 重构):
- paper_id regex (3 种格式)
- _safe_json_loads_array (LLM JSON 容错 parse, 4 段策略)
- _extract_paper_ids / _audit_section_paper_ids (引用审计)
- _build_candidate_pool (3 路召回: summary_items + chroma + arxiv fallback, 9-08 多源)
- _relevance_filter_pool (per-paper LLM 0-10 评分精筛, 9-09 user 拍板加)
- _arxiv_search / _arxiv_search_batch (单 query / 多 query 批量召回)
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional

from agent.tools.config import (
    get_min_pool_papers,
    get_chroma_sim_threshold,
    get_relevance_threshold,
    get_relevance_batch_size,
)

logger = logging.getLogger(__name__)


# ============================================================
# paper_id 抽取 (3 种格式, plan §3.1 + P6.2 LLM 实际行为)
# ============================================================
# 1) 简洁:  [2609.10003] / [0607734]              ← plan §3.1 prompt 强约束
# 2) arXiv:  arXiv:2609.10003  / arXiv 0607734
# 3) 链接:   papers/2609.10003.md  (从行内 markdown 链接 url 抽)
# 注意: plan §3.1 反例里 LLM 偶尔只写完整链接不写简洁形式, 必须 3 种都识别
# arXiv 编号两种格式: 新版 XXXX.XXXXX (如 2401.12345) / 旧版 XXXXXXX (如 0607734, 2007 年前)
_PAPER_ID_SIMPLE_RE = re.compile(r"\[(\d{4,5}\.\d{4,5}|\d{7})\]")
_PAPER_ID_ARXIV_RE = re.compile(r"arXiv[:\s]+(\d{4,5}\.\d{4,5}|\d{7})")
_PAPER_ID_PAPERS_MD_RE = re.compile(r"papers/(\d{4,5}\.\d{4,5}|\d{7})\.md")


def _safe_json_loads_array(text: str) -> Optional[list]:
    """
    容错 parse JSON 数组 (LLM 经常在 closing `"}]` 前就停, P6.2 实测多次踩)。

    策略 (逐级):
      1. 直接 json.loads
      2. 补 '"}]' / ']' 等简单收尾
      3. truncate 到最后一个完整 `{}` 之前, 补 `]`
      4. regex 抽所有单层完整 {...} 对象重建数组

    返 None 表示解析失败 (不是空数组)
    """
    if not text:
        return None
    s = text.strip()
    # 1. 直接
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):  # 偶尔 LLM 返单 object 不返 array
            return [v]
    except (json.JSONDecodeError, SyntaxError):
        pass
    # 2. 补 "}]
    for tail in ['"}]', '"]', '"', '}', ']']:
        try:
            v = json.loads(s.rstrip(",\n ") + tail)
            if isinstance(v, list):
                return v
        except (json.JSONDecodeError, SyntaxError):
            pass
    # 3. truncate 到最后一个 } (P6.2 4b 案例: 末尾是 "y" 缺 }, 截掉)
    last_brace = s.rfind('}')
    if last_brace > 0:
        try:
            v = json.loads(s[: last_brace + 1] + ']')
            if isinstance(v, list):
                return v
        except (json.JSONDecodeError, SyntaxError):
            pass
    # 4. regex 抽所有单层完整 {...} 对象 (不允许嵌套 {})
    obj_re = re.compile(r'\{[^{}]*\}', re.DOTALL)
    objs = obj_re.findall(s)
    if objs:
        try:
            v = json.loads("[" + ",".join(objs) + "]")
            if isinstance(v, list):
                return v
        except (json.JSONDecodeError, SyntaxError):
            pass
    return None


def _extract_paper_ids(text: str) -> List[str]:
    """从 markdown 文本抽所有 paper_id 引用, 3 种格式都识别, 去重保序"""
    if not text:
        return []
    seen = set()
    out = []
    # 顺序很重要: 简洁形式先 (用户期望), 然后 arXiv: 形式, 然后链接 url
    for pattern in (_PAPER_ID_SIMPLE_RE, _PAPER_ID_ARXIV_RE, _PAPER_ID_PAPERS_MD_RE):
        for m in pattern.finditer(text):
            pid = m.group(1)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(pid)
    return out


def _audit_section_paper_ids(content_md: str, pool_ids: set) -> Dict[str, List[str]]:
    """候选池内/外分桶 (plan §3.2 覆盖度审计核心)"""
    cites = _extract_paper_ids(content_md or "")
    in_pool = []
    out_pool = []
    for c in cites:
        if c in pool_ids:
            in_pool.append(c)
        else:
            out_pool.append(c)
    return {
        "in_pool": sorted(set(in_pool)),
        "out_of_pool": sorted(set(out_pool)),
        "total_cites": len(cites),
    }


# ============================================================
# Phase A: 候选池构建 (9-09 升级: 三道防线)
# ============================================================
def _build_candidate_pool(
    topic: str,
    max_pool: int = 50,
) -> "tuple[List[Dict[str, Any]], Dict[str, str]]":
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
    pool_dict: Dict[str, Dict[str, Any]] = {}  # paper_id -> pool entry

    def add_pool_entry(entry: Dict[str, Any]) -> None:
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
            def kw_score(item: Dict[str, Any]) -> int:
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

    # ========== 3. arxiv API 兜底 (9-08: 最小阈值走 env QAGENT_SURVEY_MIN_POOL, 默认 30) ==========
    if len(pool_dict) < _MIN_POOL:
        arxiv_status = "0_results"  # 默认 0 results, 拉到了再改 success
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
                    "keywords": ", ".join([a.text for a in (r.authors or [])[:3]]),
                    "_match_score": 0,
                    "_source": "arxiv_api_fallback",
                })
                arxiv_added += 1
                if len(pool_dict) >= max(_MIN_POOL, max_pool):
                    break
            if arxiv_added > 0:
                arxiv_status = "success"
        except Exception as e:
            arxiv_status = f"exception:{type(e).__name__}:{str(e)[:100]}"
            logger.warning("[_build_candidate_pool] arxiv 兜底失败: %s", e)
    else:
        arxiv_status = "skipped"  # pool 已够大, 没调 arxiv

    # ========== 4. 9-09 user 拍板: per-paper LLM 相关性评分精筛 ==========
    # 解决"邻近子领域误召" (例: 量子态验证召进 30 篇层析/密钥分发)
    # chroma 阈值 + summary_items 关键词都是召回层粗筛, 仍有邻域 paper 漏过
    # 这里 LLM 看 paper 实情 (core_method + core_contrib) 打 0-10 分, 阈值 (默认 7) 踢出
    relevance_threshold = get_relevance_threshold()
    if relevance_threshold > 0 and pool_dict:
        try:
            pool_dict = _relevance_filter_pool(topic, pool_dict, relevance_threshold)
        except Exception as e:
            logger.warning("[_build_candidate_pool] relevance filter 失败 (非致命, 用原 pool): %s", e)

    # 排序: _relevance_score 降序 (relevance 跑过后) > _match_score (未跑) > source 优先级
    source_priority = {"chroma_vector": 0, "summary_items_keyword": 1, "arxiv_api_fallback": 2}
    def _sort_key(x: Dict[str, Any]) -> tuple:
        rel = x.get("_relevance_score")
        if rel is not None:
            return (-rel, 0, source_priority.get(x.get("_source", ""), 9))
        return (-x.get("_match_score", 0), 1, source_priority.get(x.get("_source", ""), 9))

    pool = sorted(pool_dict.values(), key=_sort_key)
    # 限 max_pool
    return pool[:max_pool], {"arxiv_status": arxiv_status}


# ============================================================
# per-paper LLM 相关性评分精筛 (9-09 user 拍板加)
# ============================================================
def _relevance_filter_pool(
    topic: str,
    pool_dict: Dict[str, Dict[str, Any]],
    threshold: int,
) -> Dict[str, Dict[str, Any]]:
    """
    per-paper LLM 相关性评分精筛 (9-09 user 拍板加).

    输入: pool_dict (paper_id -> pool entry)
    处理: 5 篇/批 (env 可调) 调 LLM, 每篇 0-10 打分
    输出: pool_dict, score < threshold 的 paper 剔除, 其余加 _relevance_score 字段

    容错:
      - 整批 LLM 失败 → 该批 5 篇保留 (降级, 不剔除)
      - 单篇 JSON parse 失败 → 该篇用 score=5 (中性) 保留
      - 全部过滤后剩 0 篇 → 抛 ValueError, 调用方 fallback 用原 pool
    """
    from agent.tools.llm import _call_llm
    from agent.prompts_survey import build_relevance_score_prompt

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"
    batch_size = get_relevance_batch_size()

    items = list(pool_dict.values())
    if not items:
        return pool_dict

    logger.info(
        "[_relevance_filter_pool] topic=%r, 候选池 %d 篇, batch_size=%d, threshold=%d",
        topic[:50], len(items), batch_size, threshold,
    )

    scored: Dict[str, Dict[str, Any]] = {}  # paper_id -> {score, reason, is_about_topic, is_adjacent_field}

    # 批量化
    for i in range(0, len(items), batch_size):
        batch = items[i: i + batch_size]
        try:
            prompt = build_relevance_score_prompt(topic, batch)
            out = _call_llm(
                prompt,
                provider=_PROVIDER,
                model=_MODEL,
                enable_thinking=False,
                temperature=0.2,
                max_tokens=2000,    # 5 篇 × ~100 token/篇 + JSON overhead
                timeout=120,
            )
            # 剥 <think>
            if out and "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
            # 容错 parse
            results = _safe_json_loads_array(out or "")
            if results is None:
                raise ValueError(f"relevance batch JSON parse 失败: {(out or '')[:200]}")
            # 写入
            for r in results:
                if not isinstance(r, dict):
                    continue
                pid = r.get("paper_id", "")
                if not pid:
                    continue
                score = r.get("score")
                if not isinstance(score, (int, float)):
                    score = 5  # 中性 fallback
                # 9-09 严判: 提取 is_about_topic (跟 score 冗余但更清晰, 缺则从 score 回填)
                about_raw = r.get("is_about_topic")
                if about_raw is None:
                    is_about = int(score) >= 8
                else:
                    is_about = bool(about_raw)
                scored[pid] = {
                    "score": int(score),
                    "reason": str(r.get("reason", ""))[:200],
                    "is_about_topic": is_about,
                    "is_adjacent_field": bool(r.get("is_adjacent_field", False)),
                }
            # 漏评的 (LLM 没返) 用中性分保留
            for p in batch:
                pid = p.get("paper_id", "")
                if pid and pid not in scored:
                    scored[pid] = {
                        "score": 5, "reason": "[fallback] LLM 未评, 用中性分",
                        "is_about_topic": False, "is_adjacent_field": False,
                    }
        except Exception as e:
            # 整批失败: 整批 5 篇用中性分保留 (降级, 不剔除)
            logger.warning("[_relevance_filter_pool] batch %d 失败 (整批保留): %s", i // batch_size, e)
            for p in batch:
                pid = p.get("paper_id", "")
                if pid and pid not in scored:
                    scored[pid] = {
                        "score": 5, "reason": f"[batch_failed] {type(e).__name__}",
                        "is_about_topic": False, "is_adjacent_field": False,
                    }

    # 把评分写回 pool_dict
    for pid, p in pool_dict.items():
        s = scored.get(pid, {})
        p["_relevance_score"] = s.get("score", 5)
        p["_relevance_reason"] = s.get("reason", "")
        p["_is_about_topic"] = s.get("is_about_topic", False)
        p["_is_adjacent_field"] = s.get("is_adjacent_field", False)

    # 9-09 严判: 双重过滤 — score >= threshold **AND** is_about_topic = True
    # is_about_topic 是 LLM 显式给的二元判断, 比单纯 score 阈值更可靠
    # 邻近子领域 paper (is_about_topic=False) 一律踢掉, 即便 score=7
    kept = {
        pid: p for pid, p in pool_dict.items()
        if p["_relevance_score"] >= threshold and p["_is_about_topic"]
    }

    # 9-09 优化: 全部 paper 都标 [fallback] / [batch_failed] → LLM 整体失败, 不应用任何过滤
    all_fallback = all(
        p.get("_relevance_reason", "").startswith(("[fallback]", "[batch_failed]"))
        for p in pool_dict.values()
    )
    if all_fallback:
        logger.warning(
            "[_relevance_filter_pool] 所有 paper 都是 LLM fallback/batch_failed 标记, "
            "跳过 relevance 过滤, 原样返回 (LLM 整体不可用, 让 chroma/summary_items/arxiv 兜底)"
        )
        return dict(pool_dict)

    if len(kept) < 5 and len(kept) > 0:
        # 兜底: 取剩余的按 score 倒序, 补足到 5 篇 (避免下游 "候选池为空" 假阳)
        before_rescue = len(kept)
        remaining = sorted(
            (p for pid, p in pool_dict.items() if pid not in kept),
            key=lambda x: (-x["_relevance_score"], not x["_is_about_topic"]),  # 优先 is_about_topic=True
        )
        for p in remaining[:5 - len(kept)]:
            kept[p["paper_id"]] = p
        logger.warning(
            "[_relevance_filter_pool] threshold=%d 严判后剩 %d 篇 < 5, 兜底补足到 %d 篇 (含邻近子领域, 审计时会标 ❌)",
            threshold, before_rescue, len(kept),
        )

    if not kept:
        # 全部 0 分 → 抛错, 调用方 fallback 用原 pool
        raise ValueError("relevance filter 把所有 paper 踢光 (可能主题太偏或 LLM 全失败)")

    n_filtered = len(pool_dict) - len(kept)
    if n_filtered > 0:
        logger.info(
            "[_relevance_filter_pool] 过滤掉 %d 篇 (score<%d 或 is_about_topic=False), 保留 %d 篇",
            n_filtered, threshold, len(kept),
        )

    return kept


# ============================================================
# per-fact 主题对齐审查 (9-09 user 拍板加, mechanism B)
# ============================================================
# 跟 _relevance_filter_pool 配套, 但作用于 fact 而非 paper
# 位置: action_extract_facts 末尾 (Step 2 完成后, Step 3 write_outline 之前)
#
# 设计:
#   - generator, 每批 yield 一次进度 (避免用户看到 30s 静默)
#   - 8 条/批 (config 可调), 1 次 LLM = 1 个 batch 的 JSON 数组
#   - 不删除非对齐 fact, 只打标 (audit 用, 落 _facts.jsonl + _finalize_survey 段)
#   - LLM 整体失败 → 所有 fact 标 fallback, 不报错 (让用户看到原始 fact 列表)
def _review_facts_topic_alignment(
    topic: str,
    facts: List[Dict[str, Any]],
    threshold: int,
) -> "tuple[Dict[str, Dict[str, Any]], Dict[str, int]]":
    """
    per-fact 主题对齐审查 (9-09 user 拍板加).

    输入:
      - topic: 综述主题
      - facts: list of fact dict (来自 state["facts"], 每条至少有 fact_id/paper_id/claim)
      - threshold: >= threshold 算主题对齐, < threshold 算邻近子领域

    输出:
      - scored_facts: dict[fact_id] -> {score, is_topic_aligned, reason}
      - stats: {aligned: N, misaligned: M, fallback: K, total: T}

    容错:
      - 整批 LLM 失败 → 该批 8 条用中性分 (score=5, is_topic_aligned=True, 保留)
      - 单条 JSON parse 失败 → 该条用中性分
      - 全部 fallback → 返回 (空 scored, stats with fallback 计数)
    """
    from agent.tools.llm import _call_llm
    from agent.prompts_survey import build_fact_review_prompt

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"
    from agent.tools.config import get_fact_review_batch_size
    batch_size = get_fact_review_batch_size()

    if not facts:
        return {}, {"aligned": 0, "misaligned": 0, "fallback": 0, "total": 0}

    logger.info(
        "[_review_facts_topic_alignment] topic=%r, fact %d 条, batch_size=%d, threshold=%d",
        topic[:50], len(facts), batch_size, threshold,
    )

    scored: Dict[str, Dict[str, Any]] = {}  # fact_id -> {score, is_topic_aligned, reason}
    stats = {"aligned": 0, "misaligned": 0, "fallback": 0, "total": len(facts)}

    # 用 fact_id 作 key, 如果 fact_id 缺失则用 (paper_id, claim 短 hash) 兜底
    def _fact_key(f: Dict[str, Any], idx: int) -> str:
        fid = f.get("fact_id")
        if fid:
            return str(fid)
        pid = f.get("paper_id", "?")
        claim_hash = hash(str(f.get("claim", ""))[:50]) & 0xFFFF
        return f"{pid}:{idx}:{claim_hash}"

    for i in range(0, len(facts), batch_size):
        batch = facts[i: i + batch_size]
        batch_keys = [_fact_key(f, i + j) for j, f in enumerate(batch)]
        try:
            prompt = build_fact_review_prompt(topic, batch)
            out = _call_llm(
                prompt,
                provider=_PROVIDER,
                model=_MODEL,
                enable_thinking=False,
                temperature=0.2,
                max_tokens=3000,    # 8 条 × ~80 token/条 + JSON overhead
                timeout=120,
            )
            if out and "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
            results = _safe_json_loads_array(out or "")
            if results is None:
                raise ValueError(f"fact review batch JSON parse 失败: {(out or '')[:200]}")
            # 用 fact_id 匹配; LLM 可能没严格用我们的 fact_id, 用 paper_id + 顺序回退匹配
            for j, r in enumerate(results):
                if not isinstance(r, dict):
                    continue
                # 优先用 r["fact_id"]; 没有则按 batch 顺序回退
                fid = r.get("fact_id") or batch_keys[min(j, len(batch_keys) - 1)]
                score = r.get("score")
                if not isinstance(score, (int, float)):
                    score = 5
                align_raw = r.get("is_topic_aligned")
                if align_raw is None:
                    is_aligned = int(score) >= threshold
                else:
                    is_aligned = bool(align_raw)
                scored[fid] = {
                    "score": int(score),
                    "is_topic_aligned": is_aligned,
                    "reason": str(r.get("reason", ""))[:200],
                    "paper_id": r.get("paper_id", batch[min(j, len(batch) - 1)].get("paper_id", "?")),
                }
            # 漏评的用中性分
            for j, f in enumerate(batch):
                fk = batch_keys[j]
                if fk not in scored:
                    scored[fk] = {
                        "score": 5, "is_topic_aligned": True,  # fallback 默认对齐 (保守)
                        "reason": "[fallback] LLM 未评, 用中性分", "paper_id": f.get("paper_id", "?"),
                    }
        except Exception as e:
            logger.warning("[_review_facts_topic_alignment] batch %d 失败 (整批保留): %s", i // batch_size, e)
            for j, f in enumerate(batch):
                fk = batch_keys[j]
                if fk not in scored:
                    scored[fk] = {
                        "score": 5, "is_topic_aligned": True,
                        "reason": f"[batch_failed] {type(e).__name__}", "paper_id": f.get("paper_id", "?"),
                    }

    # 统计
    for s in scored.values():
        if s["reason"].startswith(("[fallback]", "[batch_failed]")):
            stats["fallback"] += 1
        elif s["is_topic_aligned"]:
            stats["aligned"] += 1
        else:
            stats["misaligned"] += 1

    return scored, stats


# ============================================================
# arxiv API 抓取
# ============================================================
def _arxiv_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """arxiv API 抓 k 篇, 返回 dict list."""
    import arxiv
    search = arxiv.Search(
        query=query,
        max_results=k,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    out: List[Dict[str, Any]] = []
    for r in client.results(search):
        out.append({
            "paper_id": r.entry_id.split("/")[-1].split("v")[0],  # 去 /abs/ 前缀和版本号
            "title": r.title or "",
            "year": str(r.published.year) if r.published else "",
            "authors": ", ".join([a.name for a in (r.authors or [])[:5]]),
            "abstract": r.summary or "",
            "keywords": ", ".join([a.name for a in (r.authors or [])[:3]]),
        })
        if len(out) >= k:
            break
    return out


def _arxiv_search_batch(queries: List[str], k_per_query: int = 3) -> List[Dict[str, Any]]:
    """
    多 query 批量 arxiv 召回 (9-09 user 拍板加).

    跟 _arxiv_search 的区别:
      - 单 query → 多 query (list of str)
      - 每 query 拉 k_per_query (默认 3, 避免 5 个 query × 5 = 25 调用过慢)
      - union + dedup by paper_id, 返回合并 list

    使用场景 (9-09 拍板, 未来 wire in):
      - Phase A 召回: 拿 LLM 扩展的多个学术 query (例: "quantum state verification" →
        ["self-testing quantum states", "state verification protocol",
         "device-independent certification", ...]), 一次 batch 召回, 比单 query 召回更聚焦
      - action_search_more: section_gaps 触发 search_more 时, gap queries 是 list, 可直接喂

    现有 _arxiv_search 不动 (单 query 场景仍可用, 跟 history 兼容)
    """
    import arxiv
    if not queries:
        return []
    client = arxiv.Client()
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        try:
            search = arxiv.Search(
                query=q,
                max_results=k_per_query,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for r in client.results(search):
                pid = r.entry_id.split("/")[-1].split("v")[0]
                if pid in seen:
                    continue
                seen.add(pid)
                out.append({
                    "paper_id": pid,
                    "title": r.title or "",
                    "year": str(r.published.year) if r.published else "",
                    "authors": ", ".join([a.name for a in (r.authors or [])[:5]]),
                    "abstract": r.summary or "",
                    "keywords": ", ".join([a.name for a in (r.authors or [])[:3]]),
                    "_query_source": q[:80],  # debug 用: 这条 paper 是哪个 query 召回来的
                })
        except Exception as e:
            # 单 query 失败不阻塞整个 batch
            logger.warning("[_arxiv_search_batch] query=%r failed: %s", q[:50], e)
            continue
    logger.info(
        "[_arxiv_search_batch] %d queries × k=%d → union %d 篇 (deduped)",
        len(queries), k_per_query, len(out),
    )
    return out
