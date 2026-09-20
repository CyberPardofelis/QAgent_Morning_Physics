import os
import re
import logging
import requests
from bs4 import BeautifulSoup
from agent.tools.config import PAPER_DIR
from agent.tools.llm import _call_llm
from agent.tools.utils import JOURNAL_CONFIG
from agent.tools.registry import register_tool
from agent.tools.browser_pool import get_browser_pool
from agent.tools.summary import save_summary_item
from agent.tools.summary_parser import parse_summary_markdown
# 2026-09-09 P0.6: 走 OpenAlex API 搜 APS (避免 journals.aps.org 403 反爬)
# 9-18 加: 顺手导入 _DEFAULT_TIMEOUT / _RETRY_ON_5XX, 错误信息能告诉用户具体配置
from agent.tools.openalex import (
    openalex_search as _openalex_search,
    format_paper_markdown as _openalex_format_paper,
    resolve_source_ids as _resolve_openalex_source_ids,
    OPENALEX_APS_SOURCES as _OPENALEX_APS_SOURCES,
    OPENALEX_APS_DEFAULT_QUANT_PH as _OPENALEX_APS_DEFAULT_QUANT_PH,
    _DEFAULT_TIMEOUT as _OPENALEX_TIMEOUT,
    _RETRY_ON_5XX as _OPENALEX_RETRY,
)

logger = logging.getLogger(__name__)

# ============================================================
# 2026-09-03 接回 LLM 总结字段落库(用户拍板):
# 之前 2026-09-02 P0.9 彻底回退只走了"抓文 + LLM 总结 return 文本",没存任何 LLM 产出。
# 用户原话:"LLM 总结完了就完了?" —— 应该把 LLM 输出的 核心贡献/方法/意义/关键词 都入库。
# 关键安全约束(防 P0.6 蓝屏复刻):
#   - **零后台线程**:主流程同步解析 markdown + 同步 save_summary_item()(走同步版,不走 _async)
#   - **30 篇 paper → 30 次顺序 SQLite INSERT** → 毫秒级,绝对安全
#   - **不调嵌入**:LLM 总结只入 summary_items.db,不入 chromadb
#     (chroma P0.5 嵌入仍由 analyze_paper 走 section-aware chunk 那条线,跟这条解耦)
# ============================================================


def fetch_aps_page_via_browser(journal: str = 'prl'):
    """
    通用APS期刊页面抓取函数。
    Args:
        journal: 期刊标识，如 'prl', 'pra'，对应配置文件中的键。
    """
    import time as _t
    _t0 = _t.time()
    print(f"[aps] fetch_aps_page_via_browser start: journal={journal}", flush=True)

    config = JOURNAL_CONFIG.get(journal)
    if not config:
        print(f"[aps] ! 不支持的期刊: {journal} 耗时 {_t.time()-_t0:.1f}s", flush=True)
        raise ValueError(f"不支持的期刊: {journal}")

    url = config['url_template'].format(journal=journal, section=config['section'])
    link_selector = config['link_selector']
    print(f"[aps]   url={url}  selector={link_selector}", flush=True)

    pool = get_browser_pool()
    print(f"[aps]   拿 browser_pool, 准备 acquire_page (firefox)...", flush=True)
    try:
        with pool.acquire_page(
            browser_type="firefox",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
        ) as page:
            print(f"[aps]   acquire_page ok 耗时 {_t.time()-_t0:.1f}s, 开始 goto", flush=True)
            page.set_default_timeout(60000)
            page.goto(url, wait_until="domcontentloaded")
            print(f"[aps]   goto ok 耗时 {_t.time()-_t0:.1f}s, 准备 wait_for_selector({link_selector}, 15s)", flush=True)
            page.wait_for_selector(link_selector, timeout=15000)
            print(f"[aps]   wait_for_selector ok 耗时 {_t.time()-_t0:.1f}s, 拿 page.content()", flush=True)
            html = page.content()
            print(f"[aps]   page.content() 完成 ({len(html)} 字符)  累计耗时 {_t.time()-_t0:.1f}s", flush=True)
            return html
    except Exception as e:
        print(f"[aps] ! 浏览器抓取失败: {e}  累计耗时 {_t.time()-_t0:.1f}s", flush=True)
        raise Exception(f"浏览器自动化抓取失败: {str(e)}")


def parse_aps_papers_from_html(html: str) -> list:
    soup = BeautifulSoup(html, 'html.parser')
    papers = []
    article_cards = soup.find_all('div', class_='browse-result-card')

    for idx, card in enumerate(article_cards):
        try:
            paper = {}

            link_tag = card.find('a', class_='default-link-no-flex', href=True)
            if not link_tag:
                continue
            relative_link = link_tag['href']
            paper['link'] = 'https://journals.aps.org' + relative_link
            paper['title'] = link_tag.get_text().strip()
            if not paper['title']:
                continue

            doi_match = re.search(r'10\.1103/[\w\.-]+', relative_link)
            if not doi_match:
                continue
            paper['doi'] = doi_match.group()
            paper['id'] = paper['doi']

            info_paragraphs = card.find_all('p', class_='t-sm-regular text-text-regular-neutral')
            paper['authors'] = info_paragraphs[0].get_text().strip() if len(info_paragraphs) > 0 else ""
            if len(info_paragraphs) > 1:
                date_text = info_paragraphs[1].get_text().strip()
                date_match = re.search(r'Published (.+)$', date_text)
                paper['published_date'] = date_match.group(1) if date_match else date_text
            else:
                paper['published_date'] = ""

            summary_div = card.find('div', class_='summary')
            if summary_div:
                abstract_p = summary_div.find('p')
                paper['abstract'] = abstract_p.get_text().strip() if abstract_p else ""
            else:
                paper['abstract'] = ""

            papers.append(paper)
        except Exception:
            continue

    return papers


@register_tool(name="generate_aps_daily_summary", is_streaming=False)
def generate_aps_daily_summary(journal: str = 'prl') -> str:
    """
    通过浏览器自动化抓取 APS 最新论文，调用 LLM 生成总结。

    controller 会识别这是"直接输出"工具，单独分气泡渲染。

    2026-09-03: LLM 总结里的 核心贡献/方法/意义/关键词 会同步落库到 summary_items.db
    (零后台线程,主流程同步解析 + 同步 save_summary_item)
    """
    config = JOURNAL_CONFIG.get(journal)
    if not config:
        return f"❌ 不支持的期刊: {journal}"

    try:
        html = fetch_aps_page_via_browser(journal)
        papers = parse_aps_papers_from_html(html)
        print(f"[aps] 解析到 {len(papers)} 篇 papers", flush=True)
        if not papers:
            return f"未找到 {config['name']} 论文列表，请稍后重试。"

        # 2026-09-03: 抓文元数据同步落库(title/authors/abstract),不等 LLM
        # 走 save_summary_item() 同步版(不走 _async),绝对安全
        from datetime import date as _date
        today = _date.today().isoformat()
        for p in papers:
            try:
                save_summary_item(
                    paper_id=p['id'],
                    source='aps_daily',
                    summary_date=today,
                    title=p['title'],
                    authors=p.get('authors', ''),
                    abstract=p.get('abstract', ''),
                )
            except Exception as e:
                logger.warning("[aps] 抓文元数据落库失败(已吞): paper=%s err=%s", p['id'], e)

        papers_text = ""
        for i, p in enumerate(papers, 1):
            papers_text += f"{i}. **{p['title']}**\n"
            papers_text += f"   - ID: `{p['id']}` | [PRL链接]({p['link']})\n"
            papers_text += f"   - 作者: {p.get('authors', 'N/A')[:150]}...\n"
                # 2026-09-04:abstract 从 300 缩到 180(20 篇 × 180 = 3.6k chars,
            # 之前 300 → 6k chars,MiniMax-M3 跑 12k prompt 撞 180s timeout)
            papers_text += f"   - 摘要: {p.get('abstract', 'N/A')[:180]}...\n\n"

        prompt = f"""
        你现在是一个科研助手。以下是从 {config['name']} 期刊抓取的"量子信息科学与技术"类别最新论文列表（共 {len(papers)} 篇）。
        请先分类，然后按照类别为**每一篇论文**生成一份详细的阅读笔记，按照以下格式输出：

        #### 论文序号. 论文标题（超链接）
        - **作者**：列出所有作者（如果超过5位，列出前3位加"等"）
        - **DOI**：列出文章ID（使用 [DOI 链接](https://doi.org/)）
        - **发表日期**：（日期）
        - **核心贡献**：用 2-3 句话概括论文的主要创新点
        - **方法**：简述使用的方法或技术路线（100字左右）
        - **意义与展望**：该工作的潜在影响或后续研究方向（80字左右）
        - **关键词**：5-8 个英文关键词，逗号分隔（用于后续检索/分类；不要翻译，保持英文原词）

        注意事项：
        - 使用清晰的 Markdown 标题分级。
        - 确保每篇论文都单独列出，不要合并。
        - 输出长度没有限制，请详尽分析。

        论文列表：
        {papers_text}
        """
        # 2026-09-04 用户拍板:daily summary 类任务(批量格式化 + 轻度摘要)强制不开思考
        # 原因:60+ 篇 × 思考 token = 2-5x 延迟 + 2-3x 成本,质量提升有限
        # analyze_paper 那种单篇深读场景保留"跟 chat 走",需要时再显式开思考
        # 2026-09-04 加:timeout 提到 600s(20 篇 × 180 字 = 3.6k prompt,有些慢模型
        # 跑完整总结需要 > 180s,MiniMax-M3 12k prompt 实测撞过 180s timeout)
        summary_md = _call_llm(prompt, enable_thinking=False, timeout=600)

        # 2026-09-03: 同步解析 LLM 输出 → 补 核心贡献/方法/意义/关键词
        # 主流程同步 save,不创建后台线程(防 P0.6 蓝屏)
        # 2026-09-04 加:把已抓取的 papers 传进去,parser 在 LLM 漏掉 DOI 时
        # 用 title 回退匹配(用户实测 20 条 APS 缺字段的根因)
        _persist_parsed_summary_fields(
            parsed=parse_summary_markdown(summary_md, fetched_papers=papers),
            source='aps_daily',
            summary_date=today,
        )

        return summary_md
    except Exception as e:
        return f"❌ 抓取或总结失败: {str(e)}"


def _persist_parsed_summary_fields(parsed, source: str, summary_date: str) -> None:
    """2026-09-03: 把解析出的 LLM 字段同步 UPDATE 到 summary_item 已存在的行

    为什么 UPDATE 而非 INSERT:
    - 抓文时已 INSERT 一行(只有 title/authors/abstract)
    - LLM 出来后再补 core_method/core_contrib/keywords/outlook
    - UNIQUE(paper_id, source, summary_date) 已存在,直接 UPDATE

    关键(2026-09-03 P0 防 P0.6 蓝屏复刻):
    - **单连接批 UPDATE**:30 篇用 1 个事务做完,不开 30 个连接
      (开 30 个连接 + FTS5 触发器会触发 "database disk image is malformed")
    - **不走 _async**:同步主流程,绝对安全
    - **失败不抛**:单条失败吞掉,主流程不受影响
    """
    if not parsed:
        return
    valid = [ps for ps in parsed if ps.is_valid()]
    if not valid:
        return
    saved = 0
    try:
        from agent.infra.db import connect
        with connect("summary_items") as conn:
            for ps in valid:
                try:
                    cur = conn.execute(
                        """
                        UPDATE summary_item
                        SET core_method = COALESCE(NULLIF(?, ''), core_method),
                            core_contrib = COALESCE(NULLIF(?, ''), core_contrib),
                            keywords = COALESCE(NULLIF(?, ''), keywords),
                            outlook = COALESCE(NULLIF(?, ''), outlook)
                        WHERE paper_id = ? AND source = ? AND summary_date = ?
                        """,
                        (ps.core_method, ps.core_contrib, ps.keywords, ps.outlook,
                         ps.paper_id, source, summary_date),
                    )
                    if cur.rowcount > 0:
                        saved += 1
                except Exception as e:
                    logger.warning("[aps→db] 单条 UPDATE 失败(已吞): paper=%s err=%s",
                                   ps.paper_id, e)
    except Exception as e:
        logger.warning("[aps→db] 批 UPDATE 失败(已吞): %s", e)
    if saved:
        print(f"[aps→db] ✓ 补 LLM 字段完成: {saved}/{len(valid)} 条", flush=True)


def _download_with_playwright(doi: str, save_dir: str, filename: str = None) -> str:
    """
    2026-09-04 改:用 `requests` 直接拿 APS 论文 PDF,不再走 Playwright。

    之前用 Playwright 模拟点击 `<a download>` 触发下载,**根本触发不了**:
    - APS 的 PDF endpoint 用 `Content-Disposition: inline`(在浏览器里打开)
    - 浏览器不触发 download 事件,只是导航到 PDF URL
    - Playwright 的 `expect_download()` 等不到事件,被 cancel
    - 工具报"已下载"但实际什么也没下,用户报"工具只显示执行完成,没有报错信息"

    现在的流程:
    1. GET `https://doi.org/{doi}` → redirect 到 `https://journals.aps.org/{journal}/abstract/{doi}`
       (顺便从最终 URL 拿到 journal code,如 prl/pra/...)
    2. 把 `/abstract/` 换成 `/pdf/` 拿到 PDF 直链
    3. GET PDF URL,设合理 headers(Referer + User-Agent)
    4. 校验返回 application/pdf,把 body 写盘

    优点:稳、无浏览器依赖、能区分 404 / 订阅墙 / 真实 PDF。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 2026-09-04 加:容错 — LLM 经常把 DOI 简写(漏 `10.1103/` 前缀),
    # 自动补全常见格式;但要识别"已经是 URL"的情况
    doi = doi.strip()
    if doi:
        if doi.startswith("https://doi.org/") or doi.startswith("http://doi.org/"):
            # 已经是 URL,直接抽 DOI 部分
            doi = doi.split("doi.org/")[-1]
        elif not doi.startswith("10.1103/"):
            # 看起来像短 DOI(prefix 漏了),自动补
            doi = f"10.1103/{doi}"
    print(f"[aps-dl] 处理后的 DOI: {doi!r}", flush=True)

    # 1) 走 doi.org 拿真实 journal + final url
    abs_url = f"https://doi.org/{doi}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,text/html,application/xhtml+xml,*/*;q=0.8",
    }
    print(f"[aps-dl] GET {abs_url}", flush=True)
    r0 = requests.get(abs_url, allow_redirects=True, timeout=30, headers=headers)
    final_url = r0.url
    print(f"[aps-dl] final URL: {final_url}  status={r0.status_code}", flush=True)
    if r0.status_code != 200:
        raise RuntimeError(
            f"无法访问 DOI 页面(可能是无效 DOI): HTTP {r0.status_code} ({final_url})"
        )

    # 从 final URL 抽 journal code(/prl/abstract/xxx → prl)
    # 兜底:URL 不是 /abstract/ 形态就 raise(不强行猜 journal)
    if "/abstract/" not in final_url:
        raise RuntimeError(
            f"无法从 DOI 推断期刊(redirect 后 URL 形态不符): {final_url}"
        )
    journal = final_url.split("/abstract/")[0].rsplit("/", 1)[-1]
    pdf_url = final_url.replace("/abstract/", "/pdf/")

    # 2) GET PDF
    pdf_headers = dict(headers)
    pdf_headers["Referer"] = final_url
    print(f"[aps-dl] GET {pdf_url}", flush=True)
    r1 = requests.get(pdf_url, allow_redirects=True, timeout=60, headers=pdf_headers)
    print(
        f"[aps-dl] PDF status={r1.status_code}  "
        f"Content-Type={r1.headers.get('content-type')}  "
        f"size={len(r1.content)} bytes",
        flush=True,
    )

    # 3) 校验
    if r1.status_code == 404:
        raise RuntimeError(f"APS 返回 404: {pdf_url} (DOI {doi} 不存在或已撤稿)")
    if r1.status_code == 403:
        raise RuntimeError(
            f"APS 返回 403: {pdf_url} (需要机构订阅,免费用户无法下载)"
        )
    if r1.status_code != 200:
        raise RuntimeError(
            f"APS 返回 HTTP {r1.status_code}: {pdf_url}"
        )
    if not r1.content:
        raise RuntimeError(f"PDF 内容为空: {pdf_url}")
    if r1.content[:4] != b"%PDF":
        # APS 有时候对订阅墙返 HTML 而不是 PDF
        snippet = r1.content[:200].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"返回内容不是 PDF(可能是订阅墙 HTML): {pdf_url}\n"
            f"前 200 字符: {snippet}"
        )

    # 4) 写盘(filename 缺省用 doi 末尾作文件名)
    if filename is None:
        # URL 末尾的 id(去掉 .pdf 后缀,再加 .pdf)
        last = pdf_url.rsplit("/", 1)[-1]
        if last.lower().endswith(".pdf"):
            last = last[:-4]
        filename = f"{journal}_{last}.pdf"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "wb") as f:
        f.write(r1.content)
    return filepath


@register_tool(name="download_aps_paper", is_streaming=False, cost_tier="L0")
def download_aps_paper(doi: str) -> str:
    """下载 APS 期刊论文 PDF 到 PAPER_DIR，返回本地路径"""
    save_dir = str(PAPER_DIR)
    os.makedirs(save_dir, exist_ok=True)
    try:
        path = _download_with_playwright(doi, save_dir)
        return f"✅ 已下载：{path}  ({os.path.getsize(path)} bytes)"
    except Exception as e:
        return f"❌ 下载失败: {str(e)}"


# ============================================================
# 2026-09-09 P0.6: search_aps — APS 期刊关键词搜索
# 走 OpenAlex API (避免 journals.aps.org 反爬)
# 默认覆盖 quant-ph 主阵地: PRL + PRA + PRX Quantum + PRX + RMP
# ============================================================
@register_tool(name="search_aps", is_streaming=False, cost_tier="L0")
def search_aps(
    query: str,
    max_results: int = 10,
    journals: str = "",
) -> str:
    """
    在 APS 期刊中按关键词搜索论文（走 OpenAlex, 不调 LLM）。

    Args:
        query: 自由文本查询, e.g. "quantum error correction" / "topological insulator"
        max_results: 最多返回多少篇 (1-200, 默认 10)
        journals: 逗号分隔的期刊 short_name, 限定搜索范围
                  可选: prl, pra, prb, prc, prd, pre, prx, prxquantum, prapplied, prresearch, prmaterials, rmp
                  空 = 默认 quant-ph 集 (prl, pra, prxquantum, prx, rmp)

    Returns:
        Markdown 格式的论文列表, 含 DOI 链接 / 作者 / 日期 / 期刊 / 摘要 / PDF (OA)

    触发场景:
      - 用户说 "搜一下 APS 上 X 主题的论文" / "PRL 有没有 X 的文章" / "在 PRA 找 X"
      - 用户说 "search APS for X" / "find PRL paper on X"
    """
    if not query or not query.strip():
        print("[search_aps] ❌ query 为空", flush=True)
        return "❌ query 不能为空"

    query = query.strip()
    max_results = max(1, min(int(max_results), 200))

    # 解析 journals 参数
    if journals and journals.strip():
        wanted = [j.strip() for j in journals.split(",") if j.strip()]
    else:
        wanted = _OPENALEX_APS_DEFAULT_QUANT_PH
    source_ids = _resolve_openalex_source_ids(wanted, _OPENALEX_APS_SOURCES)
    if not source_ids:
        print(f"[search_aps] ❌ 没有任何合法期刊 short_name (传入={wanted!r})", flush=True)
        return (
            f"❌ APS 搜索失败: 没有任何合法期刊 short_name\n"
            f"- 传入: {wanted}\n"
            f"- 可用 short_name: {list(_OPENALEX_APS_SOURCES.keys())}\n"
            f"- 提示: 用逗号分隔, 例: 'prl,pra,prxquantum'"
        )

    # 哪些 short_name 落空了 (没匹配到)
    resolved_short = set(_OPENALEX_APS_SOURCES.keys())
    unmatched = [w for w in wanted if w.lower() not in resolved_short]

    short_query = (query[:60] + "...") if len(query) > 60 else query
    print(
        f"[search_aps] start  query={short_query!r}  journals={wanted}  "
        f"max_results={max_results}  source_ids={len(source_ids)}个",
        flush=True,
    )

    try:
        results = _openalex_search(
            query=query,
            source_ids=source_ids,
            max_results=max_results,
            caller="search_aps",     # 9-18 加: 控制台标签用 search_aps, 方便追踪
        )
    except ValueError as e:
        # 入参不合法 (query 空/sort 不支持 — 上面已防, 这里兜底)
        print(f"[search_aps] ❌ 入参错: {e}", flush=True)
        return f"❌ APS 搜索失败 (入参): {e}"
    except RuntimeError as e:
        # 9-18 加: openalex 已经按 [timeout]/[conn]/[http 4xx]/[http 5xx]/[json] 分类了
        # 这里按分类给用户具体建议
        msg = str(e)
        if msg.startswith("[timeout]"):
            return (
                f"❌ APS 搜索超时 (OpenAlex)\n"
                f"- 网络慢或被墙, >{_OPENALEX_TIMEOUT}s 没响应\n"
                f"- 建议: 1) 检查网络/VPN  2) 稍后重试  3) max_results 调小"
            )
        if msg.startswith("[conn]"):
            return (
                f"❌ APS 搜索失败: 网络连不上 OpenAlex\n"
                f"- 可能是: DNS 失败 / 断网 / 防火墙\n"
                f"- 建议: 1) 检查网络  2) 试 web_search 兜底"
            )
        if msg.startswith("[http 4xx]"):
            return (
                f"❌ APS 搜索被 OpenAlex 拒绝 (HTTP 4xx)\n"
                f"- 通常是 query 写法有问题或触发了 OpenAlex 限流\n"
                f"- 详情: {msg}\n"
                f"- 建议: 1) 简化 query  2) 去掉特殊字符  3) 等几秒重试"
            )
        if msg.startswith("[http 5xx]"):
            return (
                f"❌ APS 搜索失败: OpenAlex 服务端连续错 (HTTP 5xx)\n"
                f"- OpenAlex 自己挂了, 已重试 {_OPENALEX_RETRY} 次\n"
                f"- 建议: 1) 稍后重试  2) 试 web_search 兜底"
            )
        if msg.startswith("[json]"):
            return (
                f"❌ APS 搜索失败: OpenAlex 返回非 JSON\n"
                f"- 服务端可能挂了 / 限流\n"
                f"- 详情: {msg}"
            )
        # 兜底
        return f"❌ APS 搜索失败 (OpenAlex): {msg}"

    if not results:
        print(
            f"[search_aps] ⚠ 0 results  query={short_query!r}  journals={wanted}",
            flush=True,
        )
        return (
            f"未找到匹配 '{query}' 的 APS 论文\n"
            f"- 搜索范围: {', '.join(wanted)}\n"
            f"- 建议: 换个关键词, 或放宽 journals (例: 空 = 默认 PRL+PRA+PRXQuantum+PRX+RMP)"
        )

    journal_label = ", ".join(wanted)
    header_lines = [
        f"找到 {len(results)} 篇 APS 论文 (query='{query}', 期刊: {journal_label}):\n",
    ]
    body = "\n\n".join(_openalex_format_paper(w, i) for i, w in enumerate(results, 1))
    footer = ""
    if unmatched:
        footer = f"\n\n⚠ 未识别的 short_name: {unmatched} (可用的: {list(_OPENALEX_APS_SOURCES.keys())})"
    print(
        f"[search_aps] done  返回 {len(results)} 篇  journals={wanted}",
        flush=True,
    )
    return "\n".join(header_lines) + body + footer

