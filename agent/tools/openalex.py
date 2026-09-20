"""
agent.tools.openalex — 共享 OpenAlex API 客户端 (2026-09-09 新增)

背景:
- APS (journals.aps.org) 和 IOP (iopscience.iop.org) 都加 hCaptcha / Cloudflare 反爬,
  本地直接 requests 拿不到搜索结果(实测:APS 403, IOP 200 但返回 captcha challenge)
- 现有 browser_pool 走 Playwright,慢(5-30s)且占用浏览器实例
- OpenAlex 是免费的学术元数据索引, 完整覆盖 APS / IOP / arxiv / Nature / Science ...
  1.3 亿+ works, 通过 API 拿 DOI / 标题 / 作者 / 机构 / 摘要 / PDF URL (OA)
- 用 `filter=primary_location.source.id:S<id>` 精确限定到某本期刊

参考: https://docs.openalex.org/

提供:
- OPENALEX_APS_SOURCES / OPENALEX_IOP_SOURCES : 期刊 short_name -> OpenAlex source ID 映射
- resolve_source_ids(short_names) : 把 ["prl", "pra"] 转成 ["S24807848", "S164566984"]
- openalex_search(query, source_ids, max_results) : 主搜索入口, 返回 list[dict]
- reconstruct_abstract(inverted_index) : OpenAlex 摘要是 "word -> positions" 格式, 还原成完整字符串
- format_paper_markdown(work) : 统一 markdown 格式 (跟 search_arxiv 对齐)
- safe_doi(doi_url) : "https://doi.org/10.1103/..." -> "10.1103/..."

约定:
- 不调 LLM, 纯本地 API 客户端, 0 token
- 错误捕获: 网络 / 解析 / 401(需 polite pool email) 都 raise, 调用方决定 fallback
"""
import logging
import re
import time
from typing import Dict, List, Optional, Iterable
from urllib.parse import unquote, urlparse

import requests

logger = logging.getLogger(__name__)


# ============================================================
# OpenAlex source ID 字典 — APS / IOP 期刊
# ============================================================
# 字段: short_name (lowercase, 跟 journals.yaml 一致) -> {
#   "id": "S..." (OpenAlex source ID, 不带 https://openalex.org/ 前缀),
#   "name": 完整期刊名,
#   "issn_l": linking ISSN,
#   "publisher": 出版商,
# }
#
# 维护说明:
#   1) ID 是稳定 URL ID, OpenAlex 不变
#   2) 新增期刊: 查 https://api.openalex.org/sources?search=<name>&per_page=3
#   3) 同名但不同 ISSN 的旧/新刊: 保留最新, 旧的丢 (OpenAlex 自己也用最新)
#
# APS 期刊 (American Physical Society)
OPENALEX_APS_SOURCES: Dict[str, Dict[str, str]] = {
    "prl":  {"id": "S24807848",  "name": "Physical Review Letters",        "issn_l": "0031-9007", "publisher": "American Physical Society"},
    "pra":  {"id": "S164566984", "name": "Physical Review A",               "issn_l": "0556-2791", "publisher": "American Physical Society"},
    "prb":  {"id": "S6791298",   "name": "Physical Review B",               "issn_l": "1098-0121", "publisher": "American Physical Society"},
    "prc":  {"id": "S4210168441","name": "Physical Review C",               "issn_l": "0556-2813", "publisher": "American Physical Society"},
    "prd":  {"id": "S4210238307","name": "Physical Review D",               "issn_l": "2470-0010", "publisher": "American Physical Society"},
    "pre":  {"id": "S35412551",  "name": "Physical Review E",               "issn_l": "1539-3755", "publisher": "American Physical Society"},
    "prx":  {"id": "S137042341", "name": "Physical Review X",               "issn_l": "2160-3308", "publisher": "American Physical Society"},
    "prxquantum": {"id": "S4210195673", "name": "PRX Quantum",             "issn_l": "2691-3399", "publisher": "American Physical Society"},
    "prapplied":  {"id": "S2764453006", "name": "Physical Review Applied", "issn_l": "2331-7019", "publisher": "American Physical Society"},
    "prresearch": {"id": "S4210240247", "name": "Physical Review Research","issn_l": "2643-1564", "publisher": "American Physical Society"},
    "prmaterials":{"id": "S2786256796", "name": "Physical Review Materials","issn_l": "2475-9953","publisher": "American Physical Society"},
    "rmp":  {"id": "S73260565",  "name": "Reviews of Modern Physics",       "issn_l": "0034-6861", "publisher": "American Physical Society"},
}

# APS 默认 quant-ph 相关期刊集 (user 拍板: 默认 PRL + PRA + PRX Quantum + PRX + RMP, 这些是 quant-ph 主阵地)
OPENALEX_APS_DEFAULT_QUANT_PH: List[str] = ["prl", "pra", "prxquantum", "prx", "rmp"]


# IOP 期刊 (IOP Publishing)
OPENALEX_IOP_SOURCES: Dict[str, Dict[str, str]] = {
    "njp":      {"id": "S148121101", "name": "New Journal of Physics",                              "issn_l": "1367-2630", "publisher": "IOP Publishing"},
    "jphysa":   {"id": "S4210167992","name": "Journal of Physics A: Mathematical and General",     "issn_l": "0305-4470", "publisher": "Institute of Physics"},
    "jphysb":   {"id": "S108417319", "name": "Journal of Physics B: Atomic, Molecular and Optical Physics", "issn_l": "0953-4075", "publisher": "IOP Publishing"},
    "jphysd":   {"id": "S111927887", "name": "Journal of Physics D: Applied Physics",             "issn_l": "0022-3727", "publisher": "Institute of Physics"},
    "jphyscm":  {"id": "S190924190", "name": "Journal of Physics: Condensed Matter",              "issn_l": "0953-8984", "publisher": "IOP Publishing"},
    "jopt":     {"id": "S128936867", "name": "Journal of Optics",                                  "issn_l": "0150-536X", "publisher": "IOP Publishing"},
    "jphysconf":{"id": "S4210187594","name": "Journal of Physics Conference Series",              "issn_l": "1742-6588", "publisher": "IOP Publishing"},
}

# IOP 默认 quant-ph 相关期刊集 (NJP + J. Phys. A 是主阵地, J. Phys. B + J. Opt. 是邻域)
OPENALEX_IOP_DEFAULT_QUANT_PH: List[str] = ["njp", "jphysa", "jphysb", "jopt"]


# ============================================================
# DOI / URL 工具
# ============================================================
def safe_doi(doi_or_url: Optional[str]) -> str:
    """OpenAlex 的 doi 字段是 'https://doi.org/10.1103/xxx', 转成纯 '10.1103/xxx'"""
    if not doi_or_url:
        return ""
    s = doi_or_url.strip()
    if s.startswith("https://doi.org/") or s.startswith("http://doi.org/"):
        return s.split("doi.org/", 1)[-1]
    if s.startswith("doi.org/"):
        return s.split("doi.org/", 1)[-1]
    return s


def reconstruct_abstract(inverted_index) -> str:
    """
    OpenAlex 摘要格式: {"word": [position, position, ...], ...}
    还原成完整字符串:  按 position 排序, 拼接

    Args:
        inverted_index: OpenAlex API 返回的 abstract_inverted_index 字段 (dict)
                        也可能是 None / 缺失
    Returns:
        完整摘要字符串, 缺失时返 ""
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    # 收集 (position, word) 配对
    pairs = []
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for p in positions:
            try:
                pairs.append((int(p), word))
            except (TypeError, ValueError):
                continue
    if not pairs:
        return ""
    pairs.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pairs)


# ============================================================
# OpenAlex 搜索主入口
# ============================================================
_OPENALEX_BASE = "https://api.openalex.org"
_DEFAULT_TIMEOUT = 30
_RETRY_ON_5XX = 2
_USER_AGENT = "QAgent/1.0 (https://github.com/qagent; mailto:qagent@example.com)"


def _build_session() -> requests.Session:
    """requests Session, 设 polite pool email (OpenAlex 推荐, 避免被限流到 second-tier)"""
    s = requests.Session()
    s.headers.update({
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
    })
    return s


def resolve_source_ids(
    short_names: Iterable[str],
    registry: Dict[str, Dict[str, str]],
) -> List[str]:
    """
    把 short_name list -> OpenAlex source ID list.

    跳过未识别的 short_name (不抛错, 调用方根据返回值长度判断是否全部命中)
    """
    ids = []
    for n in short_names:
        key = (n or "").strip().lower()
        info = registry.get(key)
        if info:
            ids.append(info["id"])
        else:
            logger.warning("[openalex] 未识别的 short_name: %r (registry: %s)", n, list(registry.keys())[:5])
    return ids


def openalex_search(
    query: str,
    *,
    source_ids: Optional[List[str]] = None,
    max_results: int = 20,
    from_year: int = 0,
    sort: str = "relevance_score:desc",
    session: Optional[requests.Session] = None,
    caller: str = "openalex_search",     # 9-18 加: 调用方标签 (search_aps/search_iop), 方便控制台追踪
) -> List[dict]:
    """
    OpenAlex works 搜索.

    Args:
        query: 自由文本查询 (OpenAlex 用 fulltext.search + title_and_abstract.search 一起)
        source_ids: 限定到这些 source ID (OR 关系, 用 | 分隔)
                    None = 不限 source (全网搜)
        max_results: 最多拿多少条 (1-200, OpenAlex 单页 max 200)
        from_year: >= this year (0 = 不过滤)
        sort: relevance_score:desc (默认) | cited_by_count:desc | publication_date:desc
        session: 复用 session (避免每工具都建连接)
        caller: 调用方标签, 控制台前缀. 默认 'openalex_search'.

    Returns:
        list[dict]: OpenAlex works (每个含 id, doi, title, authorships, primary_location, abstract_inverted_index, ...)

    Raises:
        ValueError: 入参不合法 (query 空 / sort 不支持)
        RuntimeError: 网络错误 / 非 2xx / 解析失败. msg 前缀带分类标签:
          - "[timeout]"     requests Timeout, 网络慢/被墙
          - "[conn]"        ConnectionError, DNS 失败 / 断网
          - "[http 4xx]"    query 写错 (字段名错 / 编码问题), 不重试
          - "[http 5xx]"    OpenAlex 服务端错, 已重试 _RETRY_ON_5XX 次
          - "[json]"        返回非 JSON
          - "[unknown]"     兜底
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")

    query = query.strip()
    max_results = max(1, min(int(max_results), 200))
    sess = session or _build_session()

    # 构造 filter
    filters = []
    if source_ids:
        # OpenAlex 允许多个 source_id OR 过滤: primary_location.source.id:S1|S2|...
        filters.append("primary_location.source.id:" + "|".join(source_ids))
    if from_year > 0:
        filters.append(f"publication_year:>={int(from_year)}")
    if sort and sort != "relevance_score:desc":
        # sort: relevance_score:desc | cited_by_count:desc | publication_date:desc
        if sort not in ("relevance_score:desc", "cited_by_count:desc", "publication_date:desc"):
            raise ValueError(f"不支持的 sort: {sort}")
        filters.append(f"sort:{sort}")

    params = {
        "search": query,
        "per_page": max_results,
        # 拉足够字段, 避免二次调用
        # 字段跟 OpenAlex 实际支持一致, 错了会被服务端拒绝 (400)
        "select": ",".join([
            "id", "doi", "title", "display_name",
            "publication_year", "publication_date",
            "authorships", "primary_location", "best_oa_location",
            "abstract_inverted_index", "cited_by_count",
            "open_access", "type",
        ]),
    }
    if filters:
        params["filter"] = ",".join(filters)

    url = f"{_OPENALEX_BASE}/works"

    # 9-18 加: 入口 print, 关键节点 (URL/参数/命中数) 全部走 stdout (flush=True)
    # 让用户控制台能直接看到, 不必翻日志
    short_query = (query[:50] + "...") if len(query) > 50 else query
    print(
        f"[{caller}] openalex_search start  query={short_query!r}  "
        f"source_ids={len(source_ids or [])}个  max_results={max_results}  "
        f"from_year={from_year or '-'}",
        flush=True,
    )

    last_err = None
    for attempt in range(_RETRY_ON_5XX + 1):
        try:
            t0 = time.time()
            r = sess.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
            dt = time.time() - t0
            if r.status_code >= 500:
                print(
                    f"[{caller}] HTTP {r.status_code} (5xx, attempt={attempt+1}/{_RETRY_ON_5XX+1})  "
                    f"dt={dt:.1f}s",
                    flush=True,
                )
                raise RuntimeError(f"[http 5xx] OpenAlex HTTP {r.status_code} (attempt {attempt+1})")
            if r.status_code != 200:
                # 4xx 一般是 query 写错 (例: 字段名错) — 不重试
                print(
                    f"[{caller}] HTTP {r.status_code} (4xx, query 写错?)  "
                    f"body={r.text[:200]!r}",
                    flush=True,
                )
                raise RuntimeError(
                    f"[http 4xx] OpenAlex HTTP {r.status_code}: {r.text[:300]}"
                )
            try:
                data = r.json()
            except ValueError as e:
                print(f"[{caller}] 返回非 JSON: {r.text[:200]!r}", flush=True)
                raise RuntimeError(f"[json] OpenAlex 返回非 JSON: {e}") from e
            results = data.get("results") or []
            logger.info(
                "[openalex] query=%r source_ids=%d results=%d  dt=%.1fs  total=%s",
                query[:60], len(source_ids or []), len(results), dt,
                data.get("meta", {}).get("count"),
            )
            print(
                f"[{caller}] OK  HTTP 200  dt={dt:.1f}s  "
                f"results={len(results)}  total={data.get('meta', {}).get('count', '?')}",
                flush=True,
            )
            return results
        except requests.Timeout as e:
            last_err = e
            print(
                f"[{caller}] timeout (>{_DEFAULT_TIMEOUT}s) attempt={attempt+1}/{_RETRY_ON_5XX+1}",
                flush=True,
            )
            if attempt < _RETRY_ON_5XX:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(f"[timeout] OpenAlex 搜索超时 (>{_DEFAULT_TIMEOUT}s, 重试 {_RETRY_ON_5XX} 次): {e}") from last_err
        except requests.ConnectionError as e:
            # DNS 失败 / 断网 / 防火墙 — 重试无意义, 直接 raise
            print(f"[{caller}] connection error: {e}", flush=True)
            raise RuntimeError(f"[conn] OpenAlex 网络连不上: {e}") from e
        except RuntimeError as e:
            # 上面 5xx/4xx/json 抛的 RuntimeError, 区分处理
            msg = str(e)
            if msg.startswith("[http 5xx]"):
                last_err = e
                if attempt < _RETRY_ON_5XX:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"[http 5xx] OpenAlex 服务端连续错 {_RETRY_ON_5XX+1} 次: {e}") from last_err
            # 4xx / json — 不重试, 直接上抛
            raise


# ============================================================
# 论文转 markdown (跟 search_arxiv 输出格式对齐)
# ============================================================
def _format_authors(authorships, max_show: int = 5) -> str:
    if not authorships:
        return ""
    names = []
    for a in authorships[:max_show]:
        ad = a.get("author") or {}
        nm = ad.get("display_name") or ""
        if nm:
            names.append(nm)
    if not names:
        return ""
    s = ", ".join(names)
    if len(authorships) > max_show:
        s += " 等"
    return s


def _format_published(work: dict) -> str:
    """优先 publication_date, 缺失时降级 publication_year"""
    d = work.get("publication_date") or ""
    if d:
        return d
    y = work.get("publication_year")
    return str(y) if y else ""


def _extract_pdf_url(work: dict) -> Optional[str]:
    """OA paper: best_oa_location.pdf_url / primary_location.pdf_url"""
    for key in ("best_oa_location", "primary_location"):
        loc = work.get(key) or {}
        pdf = loc.get("pdf_url")
        if pdf:
            return pdf
    return None


def format_paper_markdown(work: dict, idx: int) -> str:
    """
    把 OpenAlex work 渲染成跟 search_arxiv 风格一致的 markdown 块.

    例:
        1. **[Experimental QEC](https://doi.org/10.1103/...)**
           - 作者: Cory, Price, Maas 等
           - 发布日期: 1998-09-18
           - 期刊: Physical Review Letters  (cited: 2345)
           - 摘要: Quantum error correction is required to ...
           - [📄 PDF (OA)]() (if OA)
    """
    title = (work.get("title") or work.get("display_name") or "(无标题)").strip()
    if not title:
        title = "(无标题)"
    doi = safe_doi(work.get("doi") or "")
    abs_url = f"https://doi.org/{doi}" if doi else (work.get("id") or "")

    authors = _format_authors(work.get("authorships") or [])
    published = _format_published(work)
    cited = work.get("cited_by_count") or 0
    src = ((work.get("primary_location") or {}).get("source") or {})
    journal_name = src.get("display_name") or ""
    is_oa = bool(src.get("is_oa"))

    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    if abstract and len(abstract) > 400:
        abstract = abstract[:400].rstrip() + "..."

    lines = [f"{idx}. **[{title}]({abs_url})**"]
    if authors:
        lines.append(f"   - 作者: {authors}")
    if published:
        lines.append(f"   - 发布日期: {published}")
    if journal_name:
        meta = journal_name
        if cited:
            meta += f"  (cited: {cited})"
        lines.append(f"   - 期刊: {meta}")
    if abstract:
        lines.append(f"   - 摘要: {abstract}")
    if is_oa:
        pdf = _extract_pdf_url(work)
        if pdf:
            lines.append(f"   - [📄 PDF (OA)]({pdf})")
    return "\n".join(lines)
