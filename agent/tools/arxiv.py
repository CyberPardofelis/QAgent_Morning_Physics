import json
import re
import arxiv
import requests
from bs4 import BeautifulSoup
from agent.tools.config import PAPER_DIR
from agent.tools.llm import _call_llm
from agent.tools.registry import register_tool
# 2026-09-02 P0.6 加:抓文后异步落库
from agent.tools.summary import save_summary_item_async
# 2026-09-02 协同 fix:抓文后 LLM 提关键词 + update summary_item + 重嵌 paper_keywords
from agent.tools.extract_keywords import update_keywords_for_paper_async


# ============================================================
# 内部：抓取并解析 arXiv "今日新论文" 列表
# ============================================================
_ARXIV_LISTING_URL = "https://arxiv.org/list/quant-ph/new"


def _fetch_arxiv_listing(include_authors: bool = True) -> list:
    """
    抓取 arXiv quant-ph "今日新论文" 列表并解析。

    Returns:
        list of dict: [{'id', 'title', 'abstract', 'link', 'authors'?}, ...]

    Raises:
        RuntimeError: 抓取或解析失败
    """
    import time as _t
    _t0 = _t.time()
    print(f"[arxiv] _fetch_arxiv_listing start: url={_ARXIV_LISTING_URL} include_authors={include_authors}", flush=True)
    response = requests.get(_ARXIV_LISTING_URL, timeout=30)
    print(f"[arxiv]   requests.get 状态码={response.status_code} 耗时 {_t.time()-_t0:.1f}s, 解析 HTML ({len(response.text)} 字符)", flush=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    papers = []
    dts = soup.find_all('dt')
    dds = soup.find_all('dd')
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
    print(f"[arxiv] 解析到 {len(papers)} 篇 papers  累计耗时 {_t.time()-_t0:.1f}s", flush=True)
    return papers


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
    """
    try:
        papers = _fetch_arxiv_listing(include_authors=True)
    except Exception as e:
        return f"❌ 抓取或解析 arXiv 页面失败: {e}"

    if not papers:
        return "未找到论文列表，请稍后重试。"

    # 2026-09-02 P0.6:异步落库(不阻塞 LLM 总结,失败不抛)
    for p in papers:
        save_summary_item_async(
            paper_id=p["id"],
            source="arxiv_daily",
            title=p["title"],
            authors=p.get("authors"),
            abstract=p.get("abstract"),
        )
        # 2026-09-02 协同:后台线程 LLM 提关键词 + update + 重嵌(独立线程,跟 save 并行)
        update_keywords_for_paper_async(
            paper_id=p["id"],
            source="arxiv_daily",
            title=p["title"],
            abstract=p.get("abstract", ""),
        )

    papers_text = ""
    for i, p in enumerate(papers, 1):
        papers_text += f"{i}. **{p['title']}**\n"
        papers_text += f"   - 作者: {p.get('authors', '未知')}\n"
        papers_text += f"   - ID: `{p['id']}` | [arXiv链接]({p['link']})\n"
        papers_text += f"   - 摘要: {p['abstract'][:300]}...\n\n"

    prompt = f"""
    你现在是一个科研助手。以下是从 arXiv.org 抓取的"量子物理（quant-ph）"类别最新论文列表（共 {len(papers)} 篇）。
    请按研究方向分类总结，每个类别挑选 **4-5 篇** 亮点论文，并确保在描述亮点时**包含该论文的 arXiv ID 和超链接**（使用格式 `[标题](链接)`）。

    **任务要求：**
    1. **整体概览**：一句话说明今天新上榜论文总数。
    2. **分类总结**：根据标题和摘要，将论文分为合理的研究方向（例如：量子算法与纠错、量子模拟、量子信息理论、实验与测量技术、量子机器学习等）。
    3. **格式要求**：
        ### 研究方向
        #### 论文序号. 论文标题（超链接）
        - **作者**：列出所有作者（如果超过5位，列出前3位加"等"）
        - **arXiv ID**：[id]
        - **核心贡献**：用 2-3 句话概括论文的主要创新点
        - **方法**：简述使用的方法或技术路线（50字左右）
        - **意义与展望**：该工作的潜在影响或后续研究方向（40字左右）
    4. **输出格式**：使用清晰 Markdown 格式（### 小标题、- 列表等）。总字数在50000字符以内。

    论文列表：
    {papers_text}
    """

    try:
        summary = _call_llm(prompt)
        return _format_arxiv_links(summary)
    except Exception as e:
        return f"❌ 生成总结失败: {e}"


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
        return _call_llm(prompt)
    except Exception as e:
        return f"❌ 生成主题分析失败: {e}"
