import arxiv
import requests
from bs4 import BeautifulSoup
from .config import PAPER_DIR
from .llm import _call_llm

# ----------------------------
# 新增：arXiv 论文检索（返回带链接的文本）
# ----------------------------
def search_arxiv(query: str, max_results: int = 10):

    q = query.lower()

    """搜索arXiv，返回带Markdown链接的文本摘要"""
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results)
    papers = list(client.results(search))
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
        # 生成Markdown链接，点击后会在外部浏览器打开（由UI的导航拦截实现）
        output += f"{idx}. **[**{title}**](https://arxiv.org/abs/{arxiv_id})**\n"
        output += f"   - 作者: {author_str}\n"
        output += f"   - 发布日期: {published}\n"
        output += f"   - 摘要: {summary}\n\n"
    return output

# ----------------------------
# 新增：下载arXiv论文PDF
# ----------------------------
def download_arxiv_paper(arxiv_id: str):
    """根据arXiv ID下载论文到PAPER_DIR"""
    client = arxiv.Client()
    paper = next(client.results(arxiv.Search(id_list=[arxiv_id])))
    pdf_path = PAPER_DIR / f"{arxiv_id}.pdf"
    paper.download_pdf(filename=str(pdf_path))
    return str(pdf_path)

# ----------------------------
# 新增：生成 arXiv 量子物理每日总结
# ----------------------------
def generate_daily_summary_arxiv():
    """
    抓取 arXiv quant-ph 最新论文页面，解析每篇论文的ID、标题、摘要，
    调用LLM生成分类总结，并确保亮点论文包含arXiv ID和超链接。
    """
    url = "https://arxiv.org/list/quant-ph/new"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # arXiv 新论文列表通常包含在 <dl> 标签中，每条记录由 <dt>（元数据）和 <dd>（摘要）组成
        papers = []
        dts = soup.find_all('dt')  # 每个 <dt> 对应一篇论文的编号和链接
        dds = soup.find_all('dd')  # 每个 <dd> 对应一篇论文的标题、作者、摘要

        # 确保数量匹配
        for dt, dd in zip(dts, dds):
            # 提取 arXiv ID
            arxiv_id_tag = dt.find('a', title='Abstract')
            if arxiv_id_tag:
                # 链接形如 /abs/2602.20171
                arxiv_id = arxiv_id_tag['href'].split('/')[-1]
            else:
                # 备选：从文本中提取 [1] arXiv:2602.20171
                text = dt.get_text()
                import re
                match = re.search(r'arXiv:(\d+\.\d+)', text)
                arxiv_id = match.group(1) if match else "unknown"

            # 提取标题和摘要
            title_tag = dd.find('div', class_='list-title')
            title = title_tag.get_text().replace('Title:', '').strip() if title_tag else ''

            abstract_tag = dd.find('p', class_='mathjax')
            abstract = abstract_tag.get_text().strip() if abstract_tag else ''

            papers.append({
                'id': arxiv_id,
                'title': title,
                'abstract': abstract,
                'link': f"https://arxiv.org/abs/{arxiv_id}"
            })

        if not papers:
            return "未找到论文列表，请稍后重试。"

        # 将论文列表格式化为易读的文本，包含ID、标题、摘要
        papers_text = ""
        for i, p in enumerate(papers, 1):
            papers_text += f"{i}. **{p['title']}**\n"
            papers_text += f"   - ID: `{p['id']}` | [arXiv链接]({p['link']})\n"
            papers_text += f"   - 摘要: {p['abstract'][:300]}...\n\n"

    except Exception as e:
        return f"❌ 抓取或解析 arXiv 页面失败: {str(e)}"

    # 构建提示词，要求LLM输出包含ID和链接的亮点
    prompt = f"""
    你现在是一个科研助手。以下是从 arXiv.org 抓取的“量子物理（quant-ph）”类别最新论文列表（共 {len(papers)} 篇）。
    请按研究方向分类总结，每个类别挑选 **4-5 篇** 亮点论文，并确保在描述亮点时**包含该论文的 arXiv ID 和超链接**（使用格式 `[标题](链接)`）。

    **任务要求：**
    1. **整体概览**：一句话说明今天新上榜论文总数。
    2. **分类总结**：根据标题和摘要，将论文分为合理的研究方向（例如：量子算法与纠错、量子模拟、量子信息理论、实验与测量技术、量子机器学习等）。
    3. **格式要求**：
        #### 论文序号. 论文标题（超链接）
        - **作者**：列出所有作者（如果超过5位，列出前3位加“等”）
        - **arXiv ID**：[id]
        - **核心贡献**：用 2-3 句话概括论文的主要创新点
        - **方法**：简述使用的方法或技术路线（50字左右）
        - **意义与展望**：该工作的潜在影响或后续研究方向（40字左右）
    4. **输出格式**：使用清晰 Markdown 格式（### 小标题、- 列表等）。总字数在50000字符以内。

    论文列表：
    {papers_text}
    """

    def _add_arxiv_links(text: str) -> str:
        import re
        # 处理 arXiv:ID 格式
        text = re.sub(r'arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)', r'arXiv:[\1](https://arxiv.org/abs/\1)', text)
        # 处理 (ID) 格式
        text = re.sub(r'\((\d{4}\.\d{4,5}(?:v\d+)?)\)', r'([\1](https://arxiv.org/abs/\1))', text)
        return text

    summary = _call_llm(prompt)
    summary = _add_arxiv_links(summary)
    # print(summary)
    return summary

def analyze_today_topic_arxiv(topic: str):
    """
    针对当天 arXiv quant-ph 新论文中某个主题的论文进行详细分析。
    """
    url = "https://arxiv.org/list/quant-ph/new"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # 解析论文列表（与每日总结相同）
        papers = []
        dts = soup.find_all('dt')
        dds = soup.find_all('dd')
        for dt, dd in zip(dts, dds):
            arxiv_id_tag = dt.find('a', title='Abstract')
            if arxiv_id_tag:
                arxiv_id = arxiv_id_tag['href'].split('/')[-1]
            else:
                text = dt.get_text()
                import re
                match = re.search(r'arXiv:(\d+\.\d+)', text)
                arxiv_id = match.group(1) if match else "unknown"
            title_tag = dd.find('div', class_='list-title')
            title = title_tag.get_text().replace('Title:', '').strip() if title_tag else ''
            abstract_tag = dd.find('p', class_='mathjax')
            abstract = abstract_tag.get_text().strip() if abstract_tag else ''
            papers.append({
                'id': arxiv_id,
                'title': title,
                'abstract': abstract,
                'link': f"https://arxiv.org/abs/{arxiv_id}"
            })
    except Exception as e:
        return f"❌ 抓取或解析 arXiv 页面失败: {str(e)}"

    if not papers:
        return "当天没有新论文。"

    # 将论文列表格式化为文本
    papers_text = ""
    for i, p in enumerate(papers, 1):
        papers_text += f"{i}. [{p['title']}]({p['link']})\n"
        papers_text += f"   摘要: {p['abstract'][:500]}...\n\n"  # 摘要适当放长

    prompt = f"""
    你现在是一个科研助手。以下是今天 arXiv“量子物理”类别的新论文列表（共 {len(papers)} 篇）。
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

    analysis = _call_llm(prompt)
    # print(analysis)
    return analysis