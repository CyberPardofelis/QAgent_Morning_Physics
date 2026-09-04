import json
import re
import logging
import arxiv
import requests
from bs4 import BeautifulSoup
from agent.tools.config import PAPER_DIR
from agent.tools.llm import _call_llm
from agent.tools.registry import register_tool
from agent.tools.summary import save_summary_item
from agent.tools.summary_parser import parse_summary_markdown

logger = logging.getLogger(__name__)

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
    for dt, dd in zip(dts, dds):
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
# ============================================================
@register_tool(name="search_arxiv", is_streaming=False)
def search_arxiv(query: str, max_results: int = 10) -> str:
    """搜索 arXiv，返回带 Markdown 链接的文本摘要"""
    try:
        client = arxiv.Client(num_retries=5)
        search = arxiv.Search(query=query, max_results=max_results)
        papers = list(client.results(search))
    except Exception as e:
        return f"❌ 搜索 arXiv 失败: {e}"

    if not papers:
        return "未找到相关论文。"

    output = f"找到 {len(papers)} 篇论文：\n\n"
    for idx, paper in enumerate(papers, 1):
        arxiv_id = paper.entry_id.split('/')[-1]
        title = paper.title
        authors = [a.name for a in paper.authors[:3]]
        author_str = ', '.join(authors) + ('等' if len(paper.authors) > 3 else '')
        published = paper.published.strftime("%Y-%m-%d")
        summary = paper.summary[:200].replace('\n', ' ') + '...'
        # 生成 Markdown 链接，点击后会在外部浏览器打开
        output += f"{idx}. **[**{title}**](https://arxiv.org/abs/{arxiv_id})**\n"
        output += f"   - 作者: {author_str}\n"
        output += f"   - 发布日期: {published}\n"
        output += f"   - 摘要: {summary}\n\n"
    return output


# ============================================================
# download_arxiv_paper: 下载 arXiv 论文 PDF
# ============================================================
@register_tool(name="download_arxiv_paper", is_streaming=False, cost_tier="L0")
def download_arxiv_paper(arxiv_id: str) -> str:
    """根据 arXiv ID 下载论文到 PAPER_DIR，返回本地路径"""
    try:
        client = arxiv.Client()
        paper = next(client.results(arxiv.Search(id_list=[arxiv_id])))
        pdf_path = PAPER_DIR / f"{arxiv_id}.pdf"
        paper.download_pdf(filename=str(pdf_path))
        return f"✅ 已下载：{pdf_path}"
    except Exception as e:
        return f"❌ 下载 arXiv 论文失败: {e}"


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
        papers_text += f"   - 摘要: {p['abstract'][:300]}...\n\n"

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
        summary = _call_llm(prompt, enable_thinking=False)
        # 2026-09-03: 同步解析 + 同步补 LLM 字段到 summary_item
        _persist_parsed_summary_fields(
            parsed=parse_summary_markdown(summary),
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
        papers_text += f"   摘要: {p['abstract'][:500]}...\n\n"

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
        return _call_llm(prompt, enable_thinking=False)
    except Exception as e:
        return f"❌ 生成主题分析失败: {e}"
