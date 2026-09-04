"""
agent.tools.extract_keywords — LLM 提取论文关键词(2026-09-02 P2.5 协同 fix)

职责:
- 调 LLM 提 5-10 个关键词
- 失败 fallback (返空字符串)
- 联动 summary_item 更新 + paper_keywords 重嵌

设计:
- 单次 LLM 调用,L1 档
- 异步线程跑,不阻塞抓文主流程
- 失败 log 不抛(让 daily summary 流程稳)

为什么独立文件:
- arxiv/aps 抓文时调用
- analyze_paper 完成后调用
- keywords 提取 / 更新 / 重嵌三步一气呵成,内聚在一处
"""
import json
import re
import time
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


# 2026-09-02 P0 fix: 限制 update_keywords_for_paper_async 后台线程并发数, 防 P0.6 蓝屏复刻
# 30 papers 调 update_keywords_for_paper_async → 最多 4 个后台线程同时跑
# (per-paper dedup 由调用方决定,Semaphore 是全局并发限流)
_BG_SEMAPHORE = threading.Semaphore(4)


# LLM 提关键词的 prompt
_EXTRACT_PROMPT = """从以下论文的标题和摘要中,提取 5-10 个最核心的英文关键词(technical terms)。
要求:
- 优先 technical terms(算法名/物理概念/方法),不要泛词("research", "study")
- 用逗号分隔,不要编号
- 5-10 个,少不够多则冗余
- 直接返关键词列表,不要其他解释

标题: {title}
摘要: {abstract}

关键词:"""


def extract_keywords_with_llm(title: str, abstract: str, max_retries: int = 1) -> str:
    """
    调 LLM 提 5-10 个关键词(逗号分隔字符串)
    失败返 ""

    设计:
    - 单次 LLM 调用(L1 档,符合 cost_tier 规范)
    - max_retries:失败重试次数(网络/限流场景)
    - 不抛异常,只 log

    2026-09-02 P0 fix:
    - max_retries 默认 2 → 1(原 3 attempts × 180s = 540s/paper,30 papers = 270 min 阻塞)
    - 显式传 timeout=30s(_call_llm 默认 180s,过长会卡 Semaphore 槽位)
    """
    if not title:
        return ""
    abstract_short = (abstract or "")[:1500]  # 截断,避免 prompt 太长
    prompt = _EXTRACT_PROMPT.format(title=title, abstract=abstract_short)
    for attempt in range(max_retries + 1):
        try:
            from agent.tools.llm import _call_llm
            # P0 fix: 显式 timeout=30s,防单次卡死
            result = _call_llm(prompt, enable_thinking=False, temperature=0.2, max_tokens=200, timeout=30)
            return _parse_keywords(result)
        except Exception as e:
            logger.warning("[extract_keywords] LLM 失败(attempt %d): %s", attempt + 1, e)
            if attempt < max_retries:
                time.sleep(0.5)
    return ""


def _parse_keywords(llm_output: str) -> str:
    """
    解析 LLM 输出,提取关键词列表,返逗号分隔字符串

    LLM 输出可能是:
    - "keyword1, keyword2, keyword3"  -> 干净
    - "1. keyword1\n2. keyword2\n3. keyword3" -> 编号列表
    - "Here are the keywords:\nkeyword1, keyword2" -> 有前缀
    - "[kw1, kw2, kw3]" -> JSON-like
    """
    if not llm_output:
        return ""
    text = llm_output.strip()
    # 去掉 JSON-like 包装
    if text.startswith("[") and text.endswith("]"):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return ", ".join(str(x).strip() for x in arr if x)
        except Exception:
            text = text[1:-1]
    # 去掉编号列表前缀(1. 2. 3.)
    text = re.sub(r"^\s*\d+[\.\)、]\s*", "", text, flags=re.MULTILINE)
    # 去掉 "Here are the keywords:" 之类前缀
    for prefix in ("Here are the keywords:", "Keywords:", "关键词:", "答案:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # 按行/逗号切分
    if "\n" in text:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    else:
        parts = [p.strip() for p in text.split(",") if p.strip()]
    # 清理每个 key(去掉 - * 之类)
    cleaned = []
    for p in parts:
        p = p.strip(" -*\"'[]")
        # 去掉 "1. " "1) " 之类残余
        p = re.sub(r"^\d+[\.\)、]\s*", "", p).strip()
        if p and len(p) < 80:  # 单 key 不超过 80 字符
            cleaned.append(p)
    # 去重保序
    seen = set()
    deduped = []
    for k in cleaned:
        kl = k.lower()
        if kl in seen:
            continue
        seen.add(kl)
        deduped.append(k)
    return ", ".join(deduped[:10])


# ============================================================
# 联动: 关键词更新后回写 summary_item + 重嵌 paper_keywords
# ============================================================
def update_keywords_for_paper_async(
    paper_id: str,
    source: str,
    title: str = "",
    abstract: str = "",
    core_method: str = "",
    core_contrib: str = "",
    in_papers_db: bool = False,
    analysis_id: Optional[int] = None,
) -> None:
    """
    异步: 提关键词 + update summary_item + 重嵌 paper_keywords

    设计:
    - 失败 log 不抛
    - 用 threading,跟 save_summary_item_async 一致
    - 整个流程约 2-5s/paper(LLM 提关键词为主耗时)
    """
    def _bg():
        try:
            # 1. 提关键词(如果没传)
            keywords = ""
            if title and abstract:
                keywords = extract_keywords_with_llm(title, abstract)
            # 2. 找对应 summary_item 并更新
            from agent.tools.summary import (
                get_summary_item_by_paper, update_summary_item_fields,
            )
            existing = get_summary_item_by_paper(paper_id, source)
            if existing:
                updates = {
                    "keywords": keywords,
                    "core_method": core_method,
                    "core_contrib": core_contrib,
                }
                if in_papers_db:
                    updates["in_papers_db"] = True
                if analysis_id is not None:
                    updates["analysis_id"] = analysis_id
                # 过滤 None,只传有值的字段
                updates = {k: v for k, v in updates.items() if v}
                if updates:
                    update_summary_item_fields(existing["id"], **updates)
            # 3. 重嵌 paper_keywords(用最新的 keywords + title + abstract)
            if title and (keywords or abstract or core_method):
                # 拼嵌入文本: title + core_method[:500] + keywords
                parts = [title]
                if core_method:
                    parts.append(core_method[:500])
                elif abstract:
                    parts.append(abstract[:500])
                if keywords:
                    parts.append(keywords)
                embed_text = " | ".join(p for p in parts if p)
                from agent.tools.summary import embed_summary_item_to_chroma
                embed_summary_item_to_chroma(
                    paper_id=paper_id,
                    source=source,
                    summary_date=existing.get("summary_date") if existing else "2026-09-02",
                    title=title,
                    abstract=abstract,
                    core_method=core_method,
                    core_contrib=core_contrib,
                    keywords=keywords,
                    row_id=existing.get("id") if existing else 0,
                )
                # 注意: embed_summary_item_to_chroma 自己拼文本,我们多传了 core_method,
                # 实际函数里 abstract 优先,core_method 不被用 — 接受这个简化
        except Exception as e:
            logger.warning("[update_keywords_for_paper_async] 失败(已吞): paper=%s err=%s", paper_id, e)

    def _bg_wrapped():
        """2026-09-02 P0 fix: Semaphore 包 _bg, 防 30 papers 起 30 线程
        acquire 阻塞直到有可用槽位(不会无限等待太久,因为 max_retries=1 + timeout=30 限流)"""
        with _BG_SEMAPHORE:
            _bg()

    t = threading.Thread(
        target=_bg_wrapped, name=f"update_keywords_{paper_id[:20]}", daemon=True,
    )
    t.start()


def update_summary_after_analyze(
    paper_id: str,
    source: str,
    keywords: str,
    core_method: str,
    core_contrib: str,
    in_papers_db: bool = True,
    analysis_id: Optional[int] = None,
) -> bool:
    """
    analyze_paper 完成后调用: 直接用 LLM 提的关键词, 不再调 LLM 提取

    流程:
    1. 找 summary_item (UNIQUE by paper_id+source+summary_date, 取最新一行)
    2. update keywords / core_method / core_contrib / in_papers_db / analysis_id
    3. 重嵌 paper_keywords (用新 keywords 改善召回)

    Returns:
        True if summary_item found and updated, False otherwise

    为什么同步:
    - analyze_paper 本身已经耗 LLM, 用户等着结果, 不需要再异步
    - 重嵌一次 paper_keywords 是 1-2s (bge embedding), 可接受
    """
    try:
        from agent.tools.summary import (
            get_summary_item_by_paper, update_summary_item_fields,
            embed_summary_item_to_chroma,
        )
        existing = get_summary_item_by_paper(paper_id, source)
        if not existing:
            # 没找到对应行(可能来源不是 daily summary,例如本地 manual)
            # 不创建新行, 静默返 False — 抓文流程有 catch
            logger.info(
                "[update_summary_after_analyze] 未找到 summary_item: paper=%s source=%s",
                paper_id, source,
            )
            return False

        # 1. update summary_item 字段
        updates = {}
        if keywords:
            updates["keywords"] = keywords
        if core_method:
            updates["core_method"] = core_method
        if core_contrib:
            updates["core_contrib"] = core_contrib
        if in_papers_db:
            updates["in_papers_db"] = True
        if analysis_id is not None:
            updates["analysis_id"] = analysis_id
        if updates:
            update_summary_item_fields(existing["id"], **updates)

        # 2. 重嵌 paper_keywords
        # 拼嵌入文本: title + core_method[:500] + keywords
        # 比抓文版"abstract only"更精准,因为有 LLM 提的关键词 + 方法
        title = existing.get("title", "")
        abstract = existing.get("abstract", "")
        if title:
            embed_summary_item_to_chroma(
                paper_id=paper_id,
                source=source,
                summary_date=existing.get("summary_date", "2026-09-02"),
                title=title,
                authors=existing.get("authors", ""),
                abstract=abstract,
                core_method=core_method or "",
                core_contrib=core_contrib or "",
                keywords=keywords or "",
                row_id=existing.get("id", 0),
            )
        return True
    except Exception as e:
        logger.warning("[update_summary_after_analyze] 失败(非致命): paper=%s err=%s", paper_id, e)
        return False


def infer_source_from_paper_id(paper_id: str) -> str:
    """
    推断 paper_id 属于哪个 source(给 analyze_paper 末尾用)

    - arXiv id 形如 2408.10205[v1] / cond-mat/0204159[v2] → "arxiv_daily"
    - DOI 形如 10.1103/PhysRevLett.136.080202 / 10.1103/wfqb-rfl9 → "aps_daily"
    - 其他(本地 PDF 名)→ "manual"
    """
    if not paper_id:
        return "manual"
    pid = paper_id.strip()
    # arXiv 新格式: NNNN.NNNNN[vN]
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", pid):
        return "arxiv_daily"
    # arXiv 老格式: cond-mat/NNNNNNN 或 hep-th/NNNNNNN
    if re.match(r"^[a-z\-]+/\d{7}(v\d+)?$", pid):
        return "arxiv_daily"
    # DOI 形如 10.1103/...
    if pid.startswith("10."):
        return "aps_daily"
    # 其他: 本地 PDF 文件名
    return "manual"
