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
import os
import time
from collections.abc import Iterable

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
OPENALEX_APS_SOURCES: dict[str, dict[str, str]] = {
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
OPENALEX_APS_DEFAULT_QUANT_PH: list[str] = ["prl", "pra", "prxquantum", "prx", "rmp"]


# IOP 期刊 (IOP Publishing)
OPENALEX_IOP_SOURCES: dict[str, dict[str, str]] = {
    "njp":      {"id": "S148121101", "name": "New Journal of Physics",                              "issn_l": "1367-2630", "publisher": "IOP Publishing"},
    "jphysa":   {"id": "S4210167992","name": "Journal of Physics A: Mathematical and General",     "issn_l": "0305-4470", "publisher": "Institute of Physics"},
    "jphysb":   {"id": "S108417319", "name": "Journal of Physics B: Atomic, Molecular and Optical Physics", "issn_l": "0953-4075", "publisher": "IOP Publishing"},
    "jphysd":   {"id": "S111927887", "name": "Journal of Physics D: Applied Physics",             "issn_l": "0022-3727", "publisher": "Institute of Physics"},
    "jphyscm":  {"id": "S190924190", "name": "Journal of Physics: Condensed Matter",              "issn_l": "0953-8984", "publisher": "IOP Publishing"},
    "jopt":     {"id": "S128936867", "name": "Journal of Optics",                                  "issn_l": "0150-536X", "publisher": "IOP Publishing"},
    "jphysconf":{"id": "S4210187594","name": "Journal of Physics Conference Series",              "issn_l": "1742-6588", "publisher": "IOP Publishing"},
}

# IOP 默认 quant-ph 相关期刊集 (NJP + J. Phys. A 是主阵地, J. Phys. B + J. Opt. 是邻域)
OPENALEX_IOP_DEFAULT_QUANT_PH: list[str] = ["njp", "jphysa", "jphysb", "jopt"]


# ============================================================
# DOI / URL 工具
# ============================================================
def safe_doi(doi_or_url: str | None) -> str:
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
# 9-30: 原来写死 qagent@example.com —— 那是**假邮箱**, OpenAlex 的 polite pool
# 要求真实联系方式, 用假邮箱等于没申请, 拿到的就是较紧的配额(实测撞 429:
# "Anonymous search is temporarily rate-limited ... use a free API key")。
# 改成可配置: 设 OPENALEX_MAILTO=你的真实邮箱 即可进 polite pool。
_DEFAULT_mailto = "qagent@example.com"
_mailto = (os.getenv("OPENALEX_MAILTO", "").strip() or _DEFAULT_mailto)
_USER_AGENT = f"QAgent/1.0 (mailto:{_mailto})"
# 429 的退避上限(秒)。OpenAlex 会在 429 响应里给 retryAfter, 优先用它。
_RETRY_ON_429_CAP = 90
# 429 限流重试次数(与 5xx 分开计数: 限流等一会就好, 5xx 是服务端抽风)
_RETRY_ON_429 = 3


def _openalex_identity_note() -> str | None:
    """没配真实邮箱时提醒一次。

    9-30 更正: 原来这里写的是"占位邮箱拿不到 polite pool 的宽松配额"——
    那是错的, 已被实测证伪。三种 UA (无 mailto / 占位邮箱 / 格式合法邮箱)
    返回的 X-RateLimit-Limit 都是 1000 credits, 单次消耗也都是 10, 没有差别。
    官方文档对 polite pool 的说法是"响应时间更一致"以及"出问题时能联系到你",
    没提任何额度差异。
    填真实邮箱的真实价值: 万一我们脚本跑飞被限流, 官方能写信找到人, 而不是直接把
    IP 掐了。属于礼貌, 不是提速。
    """
    if _mailto == _DEFAULT_mailto:
        return (
            "OpenAlex 用的是占位邮箱 qagent@example.com。填真实邮箱(OPENALEX_MAILTO)不增加额度, "
            "只是万一本项目脚本跑飞被限流时, 官方能联系到人而不是直接掐 IP。"
        )
    return None


# 429 提示只完整打一次, 之后缩成一行短提示。
# 撞限流时一次跑动里可能有几十次 429(每个 query 一次), 每次都刷三行长文
# 会把控制台埋掉 —— 真实跑动里实测一次 APS 召回就刷了两遍完整文案。
_IDENTITY_NOTE_SHOWN = False


def _identity_hint() -> str:
    global _IDENTITY_NOTE_SHOWN
    note = _openalex_identity_note()
    if not note:
        return ""
    if not _IDENTITY_NOTE_SHOWN:
        _IDENTITY_NOTE_SHOWN = True
        return f"  提示: {note}"
    return "  (提示见首次 429; 配 OPENALEX_MAILTO 不增加额度, 只是便于官方联系)"


def _get_with_backoff_on_429(sess, url: str, params: dict, caller: str):
    """发一次 GET, 遇 429 退避重试。

    单独抽出来而不是塞在 openalex_search 的重试循环里, 是因为两者预算不同:
    - 5xx / timeout 是"服务端抽风", 共享外层那个 _RETRY_ON_5XX 次预算
    - 429 是"限流", 等一会就好, 每次外层尝试都应该有**自己完整**的 _RETRY_ON_429 次预算

    之前 429 分支写在共用循环里, 有两个后果:
    1) 想要 4 次尝试(_RETRY_ON_429=3), 实际被 _RETRY_ON_5XX+1=3 次的循环上限截断
    2) 最后一次走 `continue` 而循环已到尾 -> 直接掉出函数**隐式返回 None**,
       调用方拿到 None 当候选池用, 表现就是 Phase A 召回阶段"卡住不动也不报错"

    9-30: 之前只对 5xx 重试, 429 直接当 4xx 放弃 —— 撞上限流这一轮召回就全废了。
    """
    for i in range(_RETRY_ON_429 + 1):
        r = sess.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
        if r.status_code != 429:
            return r
        # OpenAlex 会在 429 响应里给 Retry-After(秒), 优先按它退避
        ra = 0.0
        try:
            ra = float((r.headers or {}).get("Retry-After") or 0)
        except (TypeError, ValueError):
            ra = 0.0
        wait = max(2.0, min(ra or (5.0 * (i + 1)), _RETRY_ON_429_CAP))
        print(
            f"[{caller}] HTTP 429 限流, 退避 {wait:.0f}s "
            f"(attempt={i + 1}/{_RETRY_ON_429 + 1})"
            + _identity_hint(),
            flush=True,
        )
        if i >= _RETRY_ON_429:
            raise RuntimeError(
                f"[http 429] OpenAlex 持续限流, 已退避重试 {_RETRY_ON_429} 次: {r.text[:200]}"
            )
        time.sleep(wait)
    # 理论不可达 (循环最后一次必定 raise), 留一句兜底防将来改动漏出 None
    raise RuntimeError("[http 429] OpenAlex 限流, 退避重试循环异常结束")


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
    registry: dict[str, dict[str, str]],
) -> list[str]:
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
    source_ids: list[str] | None = None,
    max_results: int = 20,
    from_year: int = 0,
    sort: str = "relevance_score:desc",
    session: requests.Session | None = None,
    caller: str = "openalex_search",     # 9-18 加: 调用方标签 (search_aps/search_iop), 方便控制台追踪
) -> list[dict]:
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
          - "[http 429]"    限流, 已按 Retry-After 退避重试 _RETRY_ON_429 次仍失败
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
        # 9-30 修: 原来写 publication_year:>=2019, OpenAlex 把 ">=2019" 当年份值解析,
        # 回 400 "Value for param publication_year must be a number"。
        # publication_year 只接受单个年份(=2019), 要"某年及以后"得用日期型下界字段。
        filters.append(f"from_publication_date:{int(from_year)}-01-01")
    if sort and sort != "relevance_score:desc":
        # sort: relevance_score:desc | cited_by_count:desc | publication_date:desc
        if sort not in ("relevance_score:desc", "cited_by_count:desc", "publication_date:desc"):
            raise ValueError(f"不支持的 sort: {sort}")

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
    if sort and sort != "relevance_score:desc":
        # 9-30 修: 原来错把 sort 拼进 filter 串 -> 发出 filter=sort:cited_by_count:desc,
        # OpenAlex 把 "sort" 当成一个 filter **字段名** 解析, 回 400
        # "sort is not a valid field"。sort 必须是与 filter 平级的顶层查询参数。
        # 后果: 只要不用默认的 relevance 排序(比如按被引量挑高质量论文), 召回必 400。
        params["sort"] = sort

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
            r = _get_with_backoff_on_429(sess, url, params, caller)
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

    # 兜底: 上面每条分支都以 raise 或 return 结束, 正常走不到这里。
    # 但如果将来有人加了新的 except 却忘了 raise, 函数会隐式返回 None,
    # 调用方 `for w in openalex_search(...)` 会炸在迭代上, 报错点离真凶很远。
    raise last_err or RuntimeError("[unknown] OpenAlex 搜索重试循环异常结束")


# ============================================================
# 引文图遍历 (9-30)
# ------------------------------------------------------------
# 为什么值得单独一节:
#   OpenAlex 是 credit 计费的, 不同原语代价差 10 倍, 而引文图是最便宜的那个。
#   实测 (X-RateLimit-Credits-Used 响应头):
#     search= 关键词检索          10 credits / 次  (per_page 不计费, 3 条和 200 条同价)
#     cites:  / cited_by: / doi:   1 credit  / 次  (per_page 给满 200 条)
#   也就是说: 关键词检索只适合用来"拿种子", 真正的产能必须走引文图。
#
# ★★★ 方向语义的坑 (实测确认, 名字和实际相反的地方有两个) ★★★
#   cites:W123           -> 前向: 引用了 W123 的论文
#   referenced_works:W123-> 前向: 同上, 是 cites 的**别名**, 返回完全一致
#   cited_by:W123        -> 后向: W123 引用的论文 (即 W123 的参考文献)
#
#   "cited_by" 听着像前向, 实际是后向;
#   "referenced_works" 听着像后向, 实际是前向。
#   多 seed 用 | 连接是 **OR** 语义 (实测: 两个不相关领域的 seed 合并后
#   返回 25 条而非 0 条, 不是 AND)。
#
#   验证方法(唯一无歧义): 看返回的论文**自己的** referenced_works 里有没有 seed。
#   别用"结果是否落在 seed 的 referenced_works 列表里"来判断 cites: ——
#   0% 是前向遍历的**正确**表现, 用错判据会误判成"接口坏了"。
# ============================================================

# 字段跟 openalex_search 对齐, 外加 cited_by_count / type:
#   - type 用于 type:review 综述通道, 免得客户端再查一次
#   - cited_by_count 用于 frontier 截断时按影响力排序
_GRAPH_SELECT = ",".join([
    "id", "doi", "title", "display_name",
    "publication_year", "publication_date",
    "authorships", "primary_location", "best_oa_location",
    "abstract_inverted_index", "cited_by_count",
    "open_access", "type",
])

_MAX_GRAPH_SEEDS = 50
_MAX_GRAPH_PER_PAGE = 200


def _normalize_oa_id(x: str) -> str:
    """'https://openalex.org/W123' / 'W123' -> 'W123'。空值返回 ''。"""
    s = str(x or "").strip()
    if not s:
        return ""
    return s.rstrip("/").rsplit("/", 1)[-1]


def openalex_cites(
    works: list[str],
    per_page: int = _MAX_GRAPH_PER_PAGE,
    reviews_only: bool = False,
    caller: str = "openalex_cites",
) -> list[dict]:
    """前向扩展: 引用了这些论文的论文 (谁在用它)。

    ⚠️ 方向别搞反: 这是**前向**。想拿某篇论文的参考文献请用 openalex_references()。

    Args:
        works: seed 论文 id 列表 (W123 或完整 URL 都行), 用 | 做 OR
        reviews_only: 叠 type:review, 只返回综述
        per_page: 1~200, **不计费**, 默认给满

    Returns:
        list[dict]: works; 出参同 openalex_search (多一个 type 字段)
    """
    return _openalex_graph(
        "cites", works, per_page=per_page,
        reviews_only=reviews_only, caller=caller,
    )


def openalex_references(
    works: list[str],
    per_page: int = _MAX_GRAPH_PER_PAGE,
    reviews_only: bool = False,
    caller: str = "openalex_references",
) -> list[dict]:
    """后向扩展: 这些论文引用的论文 (它的前人工作 / 参考文献)。

    ⚠️ 这个用 `cited_by:` 过滤 —— 名字听着像前向, **实际是后向**。
       `referenced_works:` 才是前向的(且与 cites: 等价), 那是 OpenAlex 的命名反直觉。

    对综述特别有用: 一篇综述的参考文献就是它认为该引的全部基座工作,
    比关键词检索准得多, 而且 1 credit 就能拿满 200 条。
    """
    return _openalex_graph(
        "cited_by", works, per_page=per_page,
        reviews_only=reviews_only, caller=caller,
    )


def openalex_batch_by_doi(
    dois: list[str],
    caller: str = "openalex_batch_doi",
) -> dict[str, dict]:
    """按 DOI 批量取 works, 返回 {规范化DOI: work}。

    同样 1 credit 拿满一整批 (实测 129 个 DOI 一起查也是 1 credit),
    适合把 Crossref 免费拿到的参考文献列表回填成带被引量/期刊名的完整记录。
    查不到的 DOI 直接不出现在返回里, 不报错。
    """
    cleaned: list[str] = []
    for d in dois or []:
        s = str(d or "").strip()
        if not s:
            continue
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if s.lower().startswith(prefix):
                s = s[len(prefix):]
                break
        if s:
            cleaned.append(s.lower())
    if not cleaned:
        return {}

    out: dict[str, dict] = {}
    for i in range(0, len(cleaned), _MAX_GRAPH_SEEDS):
        chunk = cleaned[i:i + _MAX_GRAPH_SEEDS]
        params = {
            "filter": "doi:" + "|".join(chunk),
            "per_page": len(chunk),
            "select": _GRAPH_SELECT,
        }
        try:
            items = _oa_get_results(params, caller, f"doi 批量 {i}~{i + len(chunk)}")
        except RuntimeError as e:
            logging.getLogger(__name__).warning("[%s] doi 批量失败: %s", caller, e)
            continue
        for w in items or []:
            d = (w.get("doi") or "").lower()
            for prefix in ("https://doi.org/", "http://doi.org/"):
                if d.startswith(prefix):
                    d = d[len(prefix):]
            if d:
                out[d] = w
    return out


def _openalex_graph(
    key: str,
    works: list[str],
    per_page: int = _MAX_GRAPH_PER_PAGE,
    reviews_only: bool = False,
    caller: str = "openalex_cites",
) -> list[dict]:
    """引文图遍历的统一实现。key 只有 "cites" / "cited_by" 两种语义。"""
    seeds: list[str] = []
    for w in works or []:
        n = _normalize_oa_id(w)
        if n and n not in seeds:
            seeds.append(n)
    if not seeds:
        return []
    if len(seeds) > _MAX_GRAPH_SEEDS:
        # 静默截断会让用户以为"扩了全部", 这里明确打个折
        logging.getLogger(__name__).info(
            "[%s] seed %d 个, 截断到 %d 个 (单次上限)", caller, len(seeds), _MAX_GRAPH_SEEDS)
        seeds = seeds[:_MAX_GRAPH_SEEDS]

    filters = [f"{key}:" + "|".join(seeds)]
    if reviews_only:
        filters.append("type:review")

    per_page = max(1, min(int(per_page), _MAX_GRAPH_PER_PAGE))
    params = {
        "filter": ",".join(filters),
        "per_page": per_page,
        "select": _GRAPH_SELECT,
    }
    short = f"{seeds[0]}...{seeds[-1]}" if len(seeds) > 2 else ",".join(seeds)
    print(
        f"[{caller}] 引文图 {key}({'前向' if key == 'cites' else '后向'})  "
        f"seeds={len(seeds)}  per_page={per_page}  reviews_only={reviews_only}",
        flush=True,
    )
    results = _oa_get_results(params, caller, f"引文图 {key} (seeds={short})")
    return results


def _oa_get_results(params: dict, caller: str, label: str) -> list[dict]:
    """发一次裸 GET (带 429 退避), 返回 results 列表。引文图/批量回填共用。"""
    sess = _build_session()
    url = f"{_OPENALEX_BASE}/works"
    r = _get_with_backoff_on_429(sess, url, params, caller)
    if r.status_code != 200:
        raise RuntimeError(
            f"[http {r.status_code}] OpenAlex {label} 失败: {r.text[:200]}"
        )
    try:
        results = (r.json() or {}).get("results") or []
    except ValueError as e:
        raise RuntimeError(f"[json] OpenAlex {label} 返回非 JSON: {e}") from e
    print(
        f"[{caller}] {label} → {len(results)} 篇  "
        f"cost={r.headers.get('X-RateLimit-Credits-Used')}  "
        f"rem={r.headers.get('X-RateLimit-Remaining')}",
        flush=True,
    )
    return results


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


def _extract_pdf_url(work: dict) -> str | None:
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
