import logging
import re

import arxiv
import requests
from bs4 import BeautifulSoup

from agent.tools.config import PAPER_DIR
from agent.tools.llm import _call_llm
from agent.tools.registry import register_tool
from agent.tools.summary import save_summary_item
from agent.tools.summary_parser import parse_summary_markdown

logger = logging.getLogger(__name__)


# 9-18 加: 自定义 User-Agent 防 arxiv 反爬 (406 Not Acceptable).
# arxiv 库默认 User-Agent 是 "python-requests/2.32.5", 经常被 arxiv 限流拒掉.
# 实测: 用 Mozilla/5.0 + 标识性 UA 能拿到 200 OK, 默认 UA 拿到 406.
# arxiv.Client 没有 user_agent 参数, 必须创建后 patch _session.headers
_ARXIV_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
    "QAgent/1.0 (research; mailto:research@example.com)"   # 标识性 UA + 联系邮箱 (arxiv 推荐)
)


def _make_arxiv_client(num_retries: int = 5, page_size: int = 50, delay_seconds: float = 3.0) -> "arxiv.Client":
    """
    创建 arxiv.Client 并 patch User-Agent (9-18 加, 防 arxiv 406 反爬).

    用法:
        client = _make_arxiv_client()
        results = list(client.results(arxiv.Search(query=...)))
    """
    client = arxiv.Client(
        page_size=page_size,
        delay_seconds=delay_seconds,
        num_retries=num_retries,
    )
    # 9-18: arxiv 库没有 user_agent 参数, 必须 patch _session.headers
    client._session.headers["User-Agent"] = _ARXIV_USER_AGENT
    return client


def _download_pdf_with_fallback(arxiv_id: str, pdf_path) -> bool:
    """
    下载 arxiv PDF, 主用 arxiv 库, fallback 到直接 PDF URL.

    9-18 加: arxiv 库的 download_pdf() 经常返 HTTP 406 Not Acceptable
    (User-Agent 被拒), 这里先 arxiv 库, 失败后用 requests 直接 GET PDF URL.

    Returns:
        True = 成功, False = 失败 (raise 让上层报)
    """
    # 路径 1: arxiv 库
    try:
        client = _make_arxiv_client(num_retries=3)
        paper = next(client.results(arxiv.Search(id_list=[arxiv_id])))
        paper.download_pdf(filename=str(pdf_path))
        if pdf_path.exists() and pdf_path.stat().st_size > 1024:   # 至少 1KB (PDF magic number 一般 > 1KB)
            return True
    except Exception as e:
        logger.warning("[download_arxiv_paper] arxiv 库下载失败 (%s): %s", type(e).__name__, e)

    # 路径 2: 直接 GET PDF URL (arxiv.org/pdf/<id>.pdf)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        resp = requests.get(
            pdf_url,
            headers={"User-Agent": _ARXIV_USER_AGENT},
            timeout=60,
            stream=True,
        )
        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("application/pdf"):
            with open(pdf_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            if pdf_path.exists() and pdf_path.stat().st_size > 1024:
                print(f"[download_arxiv_paper] fallback PDF URL 下载成功: {pdf_url}", flush=True)
                return True
        else:
            logger.warning(
                "[download_arxiv_paper] fallback PDF URL 失败: HTTP %s, Content-Type=%s",
                resp.status_code, resp.headers.get("Content-Type"),
            )
    except Exception as e:
        logger.warning("[download_arxiv_paper] fallback PDF URL 异常: %s", e)

    return False

# ============================================================
# 2026-09-03 接回 LLM 总结字段落库(用户拍板):
# 跟 aps.py 同步 — 抓文元数据同步落库,LLM 总结后补 核心贡献/方法/意义/关键词。
# **零后台线程**(防 P0.6 蓝屏复刻):
#   - 抓文 → 同步 save_summary_item(只存 title/authors/abstract)
#   - LLM 总结 → 同步解析 → 同步 UPDATE 已存在行
#   - 不走 _async,不调嵌入
#
# 2026-09-04 用户拍板:只总结 new submission,跳过 cross 和 replacement
# 原因:cross/replacement 量也很大,但不属于"今天的 quant-ph 新工作",总结价值低
# 方案:继续用 listing 页抓(arxiv 已经做好日期切分),只解析 "New submissions" 那个 dl 块
# 备选:arxiv API(`cat:quant-ph` + `primary_category==quant-ph` 过滤)需要自己算日期范围 + 分页
#       + 限流,得不偿失,listing 页一行 HTTP 就够
# ============================================================


# ============================================================
# 内部：抓取并解析 arXiv "今日新论文" 列表
# ============================================================
_ARXIV_LISTING_URL = "https://arxiv.org/list/quant-ph/new"


def _fetch_arxiv_listing(include_authors: bool = True) -> list:
    """
    抓取 arXiv quant-ph "今日 new submission" 列表并解析。

    2026-09-04 改:只拿 "New submissions" 块,跳过 "Cross submissions" 和 "Replacement submissions"
    - listing 页结构:1 个日期 h3 + 3 个 h3 分类标题(Each followed by a dl)
      - "New submissions (showing X of Y entries)"     ← 只拿这个
      - "Cross submissions (showing X of Y entries)"   ← 跳过
      - "Replacement submissions (showing X of Y entries)" ← 跳过
    - 解析 dl 块内所有 <dt> + <dd> 配对

    Returns:
        list of dict: [{'id', 'title', 'abstract', 'link', 'authors'?}, ...]

    Raises:
        RuntimeError: 抓取或解析失败,或找不到 "New submissions" 块
    """
    import time as _t
    _t0 = _t.time()
    print(f"[arxiv] _fetch_arxiv_listing start: url={_ARXIV_LISTING_URL} include_authors={include_authors}", flush=True)
    response = requests.get(_ARXIV_LISTING_URL, timeout=30)
    print(f"[arxiv]   requests.get 状态码={response.status_code} 耗时 {_t.time()-_t0:.1f}s, 解析 HTML ({len(response.text)} 字符)", flush=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # 2026-09-04 找 "New submissions" h3 所标注的 dl
    # (h3 是前一个 dl 的 label,不是后一个)
    new_dl = _find_section_dl(soup, "New submissions")
    if new_dl is None:
        raise RuntimeError("找不到 'New submissions' 块(arxiv 页面结构可能变了)")

    # 统计其它 section(用于日志)
    cross_dl = _find_section_dl(soup, "Cross submissions")
    repl_dl = _find_section_dl(soup, "Replacement submissions")
    cross_count = len(cross_dl.find_all('dt')) if cross_dl else 0
    repl_count = len(repl_dl.find_all('dt')) if repl_dl else 0

    papers = []
    dts = new_dl.find_all('dt')
    dds = new_dl.find_all('dd')
    for dt, dd in zip(dts, dds, strict=False):
        arxiv_id_tag = dt.find('a', title='Abstract')
        if arxiv_id_tag:
            arxiv_id = arxiv_id_tag['href'].split('/')[-1]
        else:
            match = re.search(r'arXiv:(\d+\.\d+)', dt.get_text())
            arxiv_id = match.group(1) if match else "unknown"

        title_tag = dd.find('div', class_='list-title')
        title = title_tag.get_text().replace('Title:', '').strip() if title_tag else ''

        abstract_tag = dd.find('p', class_='mathjax')
        abstract = abstract_tag.get_text().strip() if abstract_tag else ''

        paper = {
            'id': arxiv_id,
            'title': title,
            'abstract': abstract,
            'link': f"https://arxiv.org/abs/{arxiv_id}",
        }

        if include_authors:
            authors_tag = dd.find('div', class_='list-authors')
            if authors_tag:
                authors_text = authors_tag.get_text().replace('Authors:', '').strip()
                author_list = [a.strip() for a in authors_text.split(',') if a.strip()]
                if len(author_list) > 5:
                    paper['authors'] = ', '.join(author_list[:3]) + ' 等'
                else:
                    paper['authors'] = ', '.join(author_list)
            else:
                paper['authors'] = "未知"

        papers.append(paper)
    print(
        f"[arxiv] 解析到 {len(papers)} 篇 (new) + 跳过 cross={cross_count} + replacement={repl_count}  累计耗时 {_t.time()-_t0:.1f}s",
        flush=True,
    )
    return papers


def _find_section_dl(soup, section_label: str):
    """2026-09-04 加:在 listing 页里找 h3 文本以 section_label 开头的那个 h3 所标注的 dl

    关键:arxiv listing 页的 h3 是**它前面那个 dl 的 label**,不是后面的 dl!
    实际 DOM 结构(2026-09-04 实测):
      <h3>Showing new listings for Friday, 4 September 2026</h3>     ← 页头,无对应 dl
      <dl id="articles">...66 个 new submission dt/dd...</dl>
      <h3>New submissions (showing 66 of 66 entries)</h3>             ← 这是上面 dl 的 label
      <dl id="articles">...18 个 cross dt/dd...</dl>
      <h3>Cross submissions (showing 18 of 18 entries)</h3>           ← 这是上面 dl 的 label
      <dl id="articles">...68 个 replacement dt/dd...</dl>
      <h3>Replacement submissions (showing 68 of 68 entries)</h3>     ← 这是上面 dl 的 label

    所以要找 h3 对应的 dl,要用 `find_previous('dl')`,不是 `find_next('dl')`!
    """
    for h3 in soup.find_all('h3'):
        if h3.get_text(strip=True).startswith(section_label):
            return h3.find_previous('dl')
    return None


def _format_arxiv_links(text: str) -> str:
    """把 LLM 总结里的 arXiv:NNNN.NNNNN / (NNNN.NNNNN) 转成 Markdown 超链接"""
    text = re.sub(r'arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)', r'arXiv:[\1](https://arxiv.org/abs/\1)', text)
    text = re.sub(r'\((\d{4}\.\d{4,5}(?:v\d+)?)\)', r'([\1](https://arxiv.org/abs/\1))', text)
    return text


# ============================================================
# search_arxiv: 在 arXiv 上搜索论文
# 9-18 加: summary_only 参数 (默认 True), 防止 LLM 把完整 markdown 复制贴到 UI 污染对话窗口
# ============================================================
@register_tool(name="search_arxiv", is_streaming=False)
def search_arxiv(
    query: str,
    max_results: int = 10,
    summary_only: bool = True,   # 9-18 加: 默认只返简短摘要 (ID/标题/作者/日期), 不含 200 字符摘要全文
) -> str:
    """
    搜索 arXiv。

    Args:
        query: 关键词, 英文
        max_results: 1-50, 默认 10
        summary_only:
          - True (默认): 只返 ID + 标题 + 作者 + 日期, **不含摘要全文**
            适合: LLM 只想知道"有哪些论文", 避免 markdown 污染 UI
          - False: 完整版, 加 200 字符摘要 + 链接
            适合: LLM 想要摘要内容辅助决策

    返回 Markdown 格式论文列表, LLM 自己决定怎么用.
    """
    if not query or not query.strip():
        print("[search_arxiv] ❌ query 为空", flush=True)
        return "❌ query 不能为空"
    query = query.strip()
    max_results = max(1, min(int(max_results), 50))

    short_query = (query[:60] + "...") if len(query) > 60 else query
    print(
        f"[search_arxiv] start  query={short_query!r}  max_results={max_results}  summary_only={summary_only}",
        flush=True,
    )

    # 9-18 加: 异常分类 + 友好错误. arxiv 库的常见失败:
    try:
        # 9-18 加: 自定义 User-Agent 防 arxiv 反爬 (406 Not Acceptable).
        client = _make_arxiv_client(num_retries=5)
        search = arxiv.Search(query=query, max_results=max_results)
        papers = list(client.results(search))
    except arxiv.HTTPError as e:
        print(f"[search_arxiv] ❌ arxiv HTTPError: {e}", flush=True)
        return (
            f"❌ arXiv 搜索失败 (HTTP 错)\n"
            f"- 详情: {e}\n"
            f"- 建议: 1) 检查网络  2) 等几秒重试  3) arxiv 可能临时限流"
        )
    except arxiv.UnexpectedEmptyPageError as e:
        print(f"[search_arxiv] ❌ arxiv UnexpectedEmptyPageError: {e}", flush=True)
        return (
            "❌ arXiv 搜索失败: 中途返空页\n"
            "- 通常是 arxiv 服务短暂不稳  \n"
            "- 建议: 1) 稍后重试  2) max_results 调小"
        )
    except arxiv.ArxivError as e:
        print(f"[search_arxiv] ❌ arxiv ArxivError: {e}", flush=True)
        return f"❌ arXiv 搜索失败 (ArxivError): {e}"
    except Exception as e:
        print(f"[search_arxiv] ❌ 异常: {type(e).__name__}: {e}", flush=True)
        return (
            f"❌ arXiv 搜索失败 ({type(e).__name__})\n"
            f"- 详情: {e}\n"
            f"- 建议: 1) 检查网络  2) query 是不是太复杂, 简化试试"
        )

    if not papers:
        print(f"[search_arxiv] ⚠ 0 results  query={short_query!r}", flush=True)
        return (
            f"未找到匹配 '{query}' 的 arxiv 论文\n"
            f"- 建议: 1) 简化 query  2) 用英文关键词  3) max_results 调大"
        )

    # 9-18 加: summary_only 模式 - 只返 ID/标题/作者/日期, 不返摘要全文
    # 防止 LLM 把 200 字符摘要摆出来污染对话窗口
    output = f"找到 {len(papers)} 篇论文：\n\n"
    for idx, paper in enumerate(papers, 1):
        arxiv_id = paper.entry_id.split('/')[-1]
        title = paper.title or "(无标题)"
        from agent.tools.surveys._pool import _arxiv_author_name
        authors = [_arxiv_author_name(a) for a in (paper.authors or [])[:3]]
        authors = [a for a in authors if a]
        author_str = ', '.join(authors) + ('等' if len(paper.authors) > 3 else '') if authors else '(未知作者)'
        published = paper.published.strftime("%Y-%m-%d") if paper.published else "(未知日期)"

        if summary_only:
            # 简短版: 只一行核心信息, 让 LLM 自己组织回复
            output += f"{idx}. **[{title}**](https://arxiv.org/abs/{arxiv_id}) — `{arxiv_id}`\n"
            output += f"   - 作者: {author_str}\n"
            output += f"   - 发布: {published}\n\n"
        else:
            # 完整版: 含 200 字符摘要
            summary = (paper.summary or "")[:200].replace('\n', ' ') + '...'
            output += f"{idx}. **[**{title}**](https://arxiv.org/abs/{arxiv_id})**\n"
            output += f"   - 作者: {author_str}\n"
            output += f"   - 发布日期: {published}\n"
            output += f"   - 摘要: {summary}\n\n"
    print(f"[search_arxiv] done  返回 {len(papers)} 篇  summary_only={summary_only}", flush=True)
    return output


# ============================================================
# download_arxiv_paper: 下载 arXiv 论文 PDF
# 9-18 加: 自定义 User-Agent 防 406, fallback 到直接 PDF URL
# ============================================================
@register_tool(name="download_arxiv_paper", is_streaming=False, cost_tier="L0")
def download_arxiv_paper(arxiv_id: str) -> str:
    """根据 arXiv ID 下载论文 PDF 到 PAPER_DIR，返回本地路径"""
    # 2026-09-04 加:容错 — LLM 经常传 "arXiv:2501.12345" 或 "https://arxiv.org/abs/2501.12345" 形式
    # 抽纯 ID
    arxiv_id = arxiv_id.strip()
    import re as _re
    m = _re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', arxiv_id)
    if not m:
        return (
            f"❌ 无效的 arXiv ID: {arxiv_id!r}。"
            f"需要类似 '2501.12345' 或 'arXiv:2501.12345' 或 'https://arxiv.org/abs/2501.12345' 的格式"
        )
    arxiv_id = m.group(1)
    pdf_path = PAPER_DIR / f"{arxiv_id}.pdf"

    print(f"[download_arxiv_paper] start  id={arxiv_id}  → {pdf_path}", flush=True)

    # 检查是否已下载
    if pdf_path.exists() and pdf_path.stat().st_size > 1024:
        size = pdf_path.stat().st_size
        print(f"[download_arxiv_paper] 已存在  {size} bytes, 跳过", flush=True)
        return f"✅ 已下载(缓存):{pdf_path}  ({size} bytes)"

    # 9-18 加: 先试 arxiv 库, 失败 fallback 到 requests 直 GET PDF URL
    try:
        ok = _download_pdf_with_fallback(arxiv_id, pdf_path)
        if ok:
            size = pdf_path.stat().st_size
            print(f"[download_arxiv_paper] done  {size} bytes", flush=True)
            return f"✅ 已下载:{pdf_path}  ({size} bytes)"
        # 都失败: 友好提示 + 替代方案
        print(f"[download_arxiv_paper] ❌ 两条路径都失败  id={arxiv_id}", flush=True)
        return (
            f"❌ 下载 arXiv 论文失败: {arxiv_id}\n"
            f"- 主路径 (arxiv 库) + fallback (PDF URL) 都拒 (可能是 arxiv 反爬 406)\n"
            f"- 替代方案:\n"
            f"  - 改用 analyze_paper(arxiv_id='{arxiv_id}'): 不下载 PDF, 走 arxiv API 拉元数据/摘要, 一样能入库\n"
            f"  - 等几分钟再试 (arxiv 限流有时效)\n"
            f"  - 手动从 https://arxiv.org/pdf/{arxiv_id}.pdf 浏览器下载, 放到 {PAPER_DIR}"
        )
    except Exception as e:
        print(f"[download_arxiv_paper] ❌ 异常 {type(e).__name__}: {e}", flush=True)
        return (
            f"❌ 下载 arXiv 论文失败: {type(e).__name__}: {e}\n"
            f"- 替代: analyze_paper(arxiv_id='{arxiv_id}') 走 API 拉元数据, 不依赖 PDF"
        )


# ============================================================
# generate_daily_summary_arxiv: arXiv 量子物理每日总结
# ============================================================
@register_tool(name="generate_daily_summary_arxiv", is_streaming=False)
def generate_daily_summary_arxiv() -> str:
    """
    抓取 arXiv quant-ph 最新论文页面，解析每篇论文的 ID、标题、摘要，
    调用 LLM 生成分类总结，并确保亮点论文包含 arXiv ID 和超链接。

    返回完整 Markdown 总结。controller 会识别这是"直接输出"工具，单独分气泡渲染。

    2026-09-03: LLM 总结里的 核心贡献/方法/意义/关键词 同步落库到 summary_items.db
    (零后台线程,主流程同步解析 + 同步 save/update)
    """
    try:
        papers = _fetch_arxiv_listing(include_authors=True)
    except Exception as e:
        return f"❌ 抓取或解析 arXiv 页面失败: {e}"

    if not papers:
        return "未找到论文列表，请稍后重试。"

    # 2026-09-03: 抓文元数据同步落库(title/authors/abstract)
    from datetime import date as _date
    today = _date.today().isoformat()
    for p in papers:
        try:
            save_summary_item(
                paper_id=p['id'],
                source='arxiv_daily',
                summary_date=today,
                title=p['title'],
                authors=p.get('authors', ''),
                abstract=p.get('abstract', ''),
            )
        except Exception as e:
            logger.warning("[arxiv] 抓文元数据落库失败(已吞): paper=%s err=%s", p['id'], e)

    papers_text = ""
    for i, p in enumerate(papers, 1):
        papers_text += f"{i}. **{p['title']}**\n"
        papers_text += f"   - 作者: {p.get('authors', '未知')}\n"
        papers_text += f"   - ID: `{p['id']}` | [arXiv链接]({p['link']})\n"
        # 2026-09-04:abstract 从 300 缩到 180(66 篇 × 180 = 12k chars,
        # 之前 300 → 20k chars 撞 180s timeout)
        papers_text += f"   - 摘要: {p.get('abstract', 'N/A')[:180]}...\n\n"

    prompt = f"""
    你现在是一个科研助手。以下是从 arXiv.org quant-ph 抓取的**今日 new submission**列表（共 {len(papers)} 篇,只含 primary category 是 quant-ph 的新论文,不含 cross-list 和 replacement)。

    请按研究方向分类总结，**对每篇论文都生成笔记**（不需要挑选亮点），并确保描述时**包含该论文的 arXiv ID 和超链接**（使用格式 `[标题](链接)`）。

    **任务要求：**
    1. **整体概览**：一句话说明今天 new submission 总数。
    2. **分类总结**：根据标题和摘要，将论文分为合理的研究方向（例如：量子算法与纠错、量子模拟、量子信息理论、实验与测量技术、量子机器学习等）。
    3. **格式要求**：
        ### 研究方向
        #### 论文序号. 论文标题（超链接）
        - **作者**：列出所有作者（如果超过5位，列出前3位加"等"）
        - **arXiv ID**：[id]
        - **核心贡献**：用 2-3 句话概括论文的主要创新点
        - **方法**：简述使用的方法或技术路线（50字左右）
        - **意义与展望**：该工作的潜在影响或后续研究方向（40字左右）
        - **关键词**：5-8 个英文关键词，逗号分隔（用于后续检索/分类；不要翻译，保持英文原词）
    4. **输出格式**：使用清晰 Markdown 格式（### 小标题、- 列表等）。总字数在50000字符以内。

    论文列表：
    {papers_text}
    """

    try:
        # 2026-09-04 用户拍板:批量 daily summary 强制不开思考
        # 66+ 篇 × 思考 token = 慢+贵,格式化任务不需要
        # 2026-09-04 加:timeout 提到 600s(防 MiniMax 等慢模型 12k+ prompt 撞 timeout)
        summary = _call_llm(prompt, enable_thinking=False, timeout=600)
        # 2026-09-03: 同步解析 + 同步补 LLM 字段到 summary_item
        # 2026-09-04 加:把已抓取的 papers 传进去,parser 在 LLM 漏掉 DOI/arXiv ID 时
        # 用 title 回退匹配(用户实测 89% arxiv 缺字段的根因)
        _persist_parsed_summary_fields(
            parsed=parse_summary_markdown(summary, fetched_papers=papers),
            source='arxiv_daily',
            summary_date=today,
        )
        return _format_arxiv_links(summary)
    except Exception as e:
        return f"❌ 生成总结失败: {e}"


def _persist_parsed_summary_fields(parsed, source: str, summary_date: str) -> None:
    """2026-09-03: 把 LLM 解析出的字段 UPDATE 到已存在的 summary_item 行

    跟 aps.py 同款(两边各保留一份,避免互相 import)
    关键:单连接批 UPDATE,不开 30 个连接(防 FTS5 触发器在多连接下损坏索引)
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
                    logger.warning("[arxiv→db] 单条 UPDATE 失败(已吞): paper=%s err=%s",
                                   ps.paper_id, e)
    except Exception as e:
        logger.warning("[arxiv→db] 批 UPDATE 失败(已吞): %s", e)
    if saved:
        print(f"[arxiv→db] ✓ 补 LLM 字段完成: {saved}/{len(valid)} 条", flush=True)


# ============================================================
# analyze_today_topic_arxiv: arXiv 某主题详细分析
# ============================================================
@register_tool(name="analyze_today_topic_arxiv", is_streaming=False)
def analyze_today_topic_arxiv(topic: str) -> str:
    """
    针对当天 arXiv quant-ph 新论文中某个主题的论文进行详细分析。
    """
    try:
        papers = _fetch_arxiv_listing(include_authors=False)
    except Exception as e:
        return f"❌ 抓取或解析 arXiv 页面失败: {e}"

    if not papers:
        return "当天没有新论文。"

    papers_text = ""
    for i, p in enumerate(papers, 1):
        papers_text += f"{i}. [{p['title']}]({p['link']})\n"
        # 2026-09-04:abstract 从 500 缩到 200(虽然篇数少,但质量比冗长好)
        papers_text += f"   摘要: {p.get('abstract', 'N/A')[:200]}...\n\n"

    prompt = f"""
    你现在是一个科研助手。以下是今天 arXiv"量子物理"类别的新论文列表（共 {len(papers)} 篇）。
    用户希望针对主题 **"{topic}"** 的论文进行详细分析。

    请执行以下任务：
    1. 从列表中筛选出与主题 "{topic}" 相关的论文（根据标题和摘要判断）。
    2. 对筛选出的论文，逐一详细介绍：
       - 论文标题（使用Markdown链接，格式为 `[标题](链接)`）
       - 核心贡献（用2-3句话概括）
       - 主要方法或结论
    3. 如果筛选出的论文数量为0，请明确说明。
    4. 输出格式使用Markdown，确保清晰易读。

    论文列表：
    {papers_text}
    """

    try:
        # 主题分析也是批量筛选+格式化,不开思考
        return _call_llm(prompt, enable_thinking=False, timeout=300)
    except Exception as e:
        return f"❌ 生成主题分析失败: {e}"
