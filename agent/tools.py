import os
import subprocess
import tempfile
import pickle
import faiss
import numpy as np
from PyPDF2 import PdfReader
import re
from collections import Counter

from agent.local_embedder import LocalEmbedder
from config import DATA_DIR, VECTOR_DIR

import arxiv
import json
import hashlib
from datetime import datetime
from pathlib import Path
import requests  # arxiv可能已包含，但下载可能需要
# 可能需要 PyMuPDF 或 fitz
import fitz  # PyMuPDF

import requests
from bs4 import BeautifulSoup
import time

from agent.local_embedder import LocalEmbedder
from config import DATA_DIR, VECTOR_DIR, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from openai import OpenAI
from config import INTERESTED_TOPICS

# ----------------------------
# 路径配置
# ----------------------------
DATA_DIR = Path("data")
PAPER_DIR = DATA_DIR / "papers"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
INDEX_FILE = KNOWLEDGE_DIR / "index.json"
os.makedirs(PAPER_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# ----------------------------
# 全局索引缓存（原有FAISS）
# ----------------------------
_index = None
_metadata = None
INDEX_FILE_FAISS = os.path.join(VECTOR_DIR, "faiss.index")
META_FILE = os.path.join(VECTOR_DIR, "meta.pkl")

# ----------------------------
# 工具函数：提取PDF文本（支持两种库）
# ----------------------------
def extract_text_from_pdf(filepath, method="fitz"):
    """提取PDF文本，method可选 'fitz' 或 'pypdf2'"""
    if method == "fitz":
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    else:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text

# ----------------------------
# 工具函数：文本分块
# ----------------------------
def chunk_text(text, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ----------------------------
# 判断索引是否存在
# ----------------------------
def index_exists():
    return os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)


# ----------------------------
# 构建新索引
# ----------------------------
def build_index():
    print("Building new vector index (first time, may take a while)...")

    os.makedirs(VECTOR_DIR, exist_ok=True)

    all_chunks = []
    metadata = []

    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DATA_DIR, filename)
            print(f"Indexing {filename}...")
            text = extract_text_from_pdf(path)
            chunks = chunk_text(text)

            for chunk in chunks:
                all_chunks.append(chunk)
                metadata.append({
                    "source": filename,
                    "content": chunk
                })

    if not all_chunks:
        raise ValueError("No PDF documents found in DATA_DIR.")

    embeddings = LocalEmbedder.embed(all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print("Index built successfully.")

    return index, metadata


# ----------------------------
# 加载或获取索引（延迟加载）
# ----------------------------
def get_index():
    global _index, _metadata

    if _index is not None:
        return _index, _metadata

    print("索引文件路径:", INDEX_FILE_FAISS)
    print("元数据路径:", META_FILE)
    print("文件是否存在:", os.path.exists(INDEX_FILE_FAISS), os.path.exists(META_FILE))

    if index_exists():
        print("Loading existing vector index...")
        _index = faiss.read_index(INDEX_FILE_FAISS)  # 如果这行报错，说明文件可能损坏
        with open(META_FILE, "rb") as f:
            _metadata = pickle.load(f)
        return _index, _metadata

    # 如果不存在，构建新索引
    _index, _metadata = build_index()
    return _index, _metadata


# ----------------------------
# 文献检索工具
# ----------------------------
def search_papers(query: str, top_k=5):
    index, metadata = get_index()

    query_embedding = LocalEmbedder.embed([query])

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        if i < len(metadata):
            item = metadata[i]
            results.append(
                f"[Source: {item['source']}]\n{item['content']}\n"
            )

    return "\n\n---\n\n".join(results)


# ----------------------------
# Python 执行工具
# ----------------------------
def run_python(code: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(code.encode())
        filename = f.name

    result = subprocess.run(
        ["python", filename],
        capture_output=True,
        text=True,
        encoding="utf-8",  # 强制用 utf-8
        errors="replace"  # 或者遇到非法字符替换成 ?
    )

    return (result.stdout or "") + (result.stderr or "")

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
def download_paper(arxiv_id: str):
    """根据arXiv ID下载论文到PAPER_DIR"""
    client = arxiv.Client()
    paper = next(client.results(arxiv.Search(id_list=[arxiv_id])))
    pdf_path = PAPER_DIR / f"{arxiv_id}.pdf"
    paper.download_pdf(filename=str(pdf_path))
    return str(pdf_path)

# ----------------------------
# 内部函数：调用LLM提取结构化信息
# ----------------------------
def _call_llm(prompt: str) -> str:
    """使用DeepSeek模型（非流式）生成分析结果"""
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000
    )
    return response.choices[0].message.content

# ----------------------------
# 新增：分析论文并保存到知识库
# ----------------------------
def analyze_paper(paper_id: str, pdf_path: str = None, user_comment: str = ""):
    """
    分析论文（arXiv ID或本地PDF），提取结构化信息，保存JSON到知识库。
    若提供有效的pdf_path，则从PDF提取文本；否则从arXiv获取摘要。
    无论是否已有记录，都会重新生成所有字段（覆盖原有文件）。
    """
    json_path = KNOWLEDGE_DIR / f"{paper_id}.json"

    # 获取论文文本（优先本地PDF，否则arXiv）
    text = None
    if pdf_path is None:
        # 尝试在默认目录查找同名PDF
        potential_pdf = PAPER_DIR / f"{paper_id}.pdf"
        if potential_pdf.exists():
            pdf_path = str(potential_pdf)
            print(f"自动找到本地PDF: {pdf_path}")

    if pdf_path and os.path.exists(pdf_path):
        print(f"从本地PDF提取文本: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)[:5000]
    else:
        # 尝试从arXiv获取摘要
        if pdf_path and not os.path.exists(pdf_path):
            print(f"警告：提供的PDF路径不存在: {pdf_path}，尝试从arXiv获取摘要")
        try:
            client = arxiv.Client()
            paper = next(client.results(arxiv.Search(id_list=[paper_id])))
            text = paper.summary
            pdf_path = None  # 不保存无效路径
        except (arxiv.HTTPError, StopIteration) as e:
            raise ValueError(f"arXiv ID '{paper_id}' 无效或无法访问: {e}")

    if not text:
        raise ValueError("无法获取论文内容，请提供有效的PDF路径或arXiv ID")

    # 调用LLM生成结构化信息（原有逻辑不变）
    prompt = f"""
    Analyze the following paper text and return a **valid JSON object** with these fields:
    - title: the paper title
    - authors: list of authors
    - keywords: list of keywords
    - research_topic: main topic or subject
    - main_conclusion: the primary conclusion
    - citations: list of cited paper IDs (arXiv IDs or DOIs if available)
    - summary: a one-paragraph summary

    Paper text:
    {text}

    Output ONLY the JSON object, no other text.
    """
    result_str = _call_llm(prompt)

    try:
        result = json.loads(result_str)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', result_str, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError(f"无法解析LLM输出为JSON: {result_str}")

    # 补充元数据
    result["paper_id"] = paper_id
    result["analyzed_at"] = datetime.now().isoformat()
    # 将 user_comments 初始化为列表（便于后续追加）
    result["user_comments"] = [user_comment] if user_comment else []
    if pdf_path and os.path.exists(pdf_path):
        result["local_path"] = pdf_path

    # 保存JSON（覆盖）
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 更新索引
    _update_index(paper_id)
    return result


def update_paper_comment(paper_id: str, user_comment: str):
    """
    更新已有分析记录的 user_comments 字段（追加模式或覆盖模式，此处以追加为例）。
    如果分析记录不存在，则抛出异常。
    """
    json_path = KNOWLEDGE_DIR / f"{paper_id}.json"
    if not json_path.exists():
        raise ValueError(f"论文 {paper_id} 的分析记录不存在，无法更新评论。请先运行 analyze_paper 进行分析。")

    # 读取现有记录
    with open(json_path, "r", encoding="utf-8") as f:
        article = json.load(f)

    # 确保 user_comments 是列表（方便追加历史评论）
    if "user_comments" not in article or not isinstance(article["user_comments"], list):
        # 如果原来是字符串，转换为列表
        old_comment = article.get("user_comments", "")
        article["user_comments"] = [old_comment] if old_comment else []

    # 追加新评论（可根据需要改为覆盖）
    if user_comment:
        article["user_comments"].append(user_comment)

    # 更新时间戳
    article["analyzed_at"] = datetime.now().isoformat()

    # 写回文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(article, f, ensure_ascii=False, indent=2)

    return article


def _update_index(paper_id: str):
    """记录已分析的论文（用于增量更新）"""
    index = {}
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            index = json.load(f)
    index[paper_id] = datetime.now().isoformat()
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f)

# ----------------------------
# 新增：扫描本地PDF文件夹，增量分析新论文
# ----------------------------
def scan_local_papers():
    """扫描PAPER_DIR，分析新出现的PDF（增量更新）"""
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {}

    new_papers = []
    for pdf_file in PAPER_DIR.glob("*.pdf"):
        paper_id = pdf_file.stem  # 假设文件名是arXiv ID
        if paper_id not in index:
            print(f"分析新论文: {paper_id}")
            result = analyze_paper(paper_id, pdf_path=str(pdf_file))
            new_papers.append(result)
        else:
            print(f"跳过已分析: {paper_id}")
    return new_papers

# ----------------------------
# 新增：在知识库中检索文章
# ----------------------------
def query_knowledge(keyword: str = None, topic: str = None, author: str = None):
    """
    在知识库JSON文件中搜索匹配的文章。
    返回匹配的文章列表（每个文章为dict）。
    """
    results = []
    for json_file in KNOWLEDGE_DIR.glob("*.json"):
        if json_file.name == "index.json":
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            article = json.load(f)
        if keyword:
            if keyword.lower() in article.get("summary", "").lower() or \
               keyword.lower() in article.get("title", "").lower():
                results.append(article)
                continue
        if topic:
            if topic.lower() in article.get("research_topic", "").lower():
                results.append(article)
                continue
        if author:
            if any(author.lower() in a.lower() for a in article.get("authors", [])):
                results.append(article)
                continue
    return results


# ----------------------------
# 调用 LLM 生成摘要（非流式）
# ----------------------------
def _call_llm(prompt: str) -> str:
    """调用 DeepSeek 模型，添加超时和错误处理"""
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        timeout=60
    )
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=5000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM 调用错误: {e}")
        return ""  # 返回空字符串，避免后续处理崩溃

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
    请按研究方向分类总结，每个类别挑选 **2-3 篇** 亮点论文，并确保在描述亮点时**包含该论文的 arXiv ID 和超链接**（使用格式 `[标题](链接)`）。

    **任务要求：**
    1. **整体概览**：一句话说明今天新上榜论文总数。
    2. **分类总结**：根据标题和摘要，将论文分为合理的研究方向（例如：量子算法与纠错、量子模拟、量子信息理论、实验与测量技术、量子机器学习等）。
     3. **亮点提取**：每个分类下列出 **2-3 篇** 亮点论文。**每篇亮点必须使用以下格式：**
       - `[标题](arxiv:ID)`：一句核心贡献说明
       **确保每个亮点单独一行，不要有多余换行。**
       例如:[QSolver: A Quantum Constraint Solver](arXiv:2602.20171)：首个针对量子程序的约束求解框架，用于确保程序正确性。
    4. **输出格式**：使用清晰 Markdown 格式（### 小标题、- 列表等）。

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
    return analysis

