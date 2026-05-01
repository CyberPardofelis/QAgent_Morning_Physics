import os
import json
import arxiv
from datetime import datetime
from pathlib import Path
from .config import PAPER_DIR, KNOWLEDGE_DIR, INDEX_FILE
from .database import get_db_connection, get_paper_by_id
from .pdf_utils import extract_text_from_pdf
from .llm import _call_llm

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
    - link: The online link of the paper. If not available, please record the arXiv ID or DOI number.
    - keywords: list of keywords, about 6 to 10 keywords
    - research_topic: main topic or subject
    - basic_method: Technical methods and innovative points, about 100-400 words
    - main_conclusion: the primary conclusion and mathematical expression, about 100-300 words
   - citations: list of important citations in the method section. Each citation should be an object with the following fields: 
        * title: the title of the cited paper
        * authors: list of authors
        * year: publication year (as integer)
        * journal: journal name or venue
        * doi: DOI if available (optional)
    - intro_cites: list of important citations in the introduction section. Each citation should have the same structure as in 'citations'.
    - outlook: the meaning, impact and outlook of this paper
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

    '''
    # 保存JSON（覆盖）
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 更新索引
    _update_index(paper_id)
    # '''

    # ---------- 存入数据库 ----------
    conn = get_db_connection()
    c = conn.cursor()

    # 将列表字段转为 JSON 字符串
    authors_json = json.dumps(result.get("authors", []), ensure_ascii=False)
    keywords_json = json.dumps(result.get("keywords", []), ensure_ascii=False)
    citations_json = json.dumps(result.get("citations", []), ensure_ascii=False)
    intro_cites_json = json.dumps(result.get("intro_cites", []), ensure_ascii=False)
    user_comments_json = json.dumps(result.get("user_comments", []), ensure_ascii=False)

    # 插入或替换主表记录
    c.execute('''
        INSERT OR REPLACE INTO papers (
            paper_id, title, authors, link, keywords, research_topic,
            basic_method, main_conclusion, citations, intro_cites,
            outlook, summary, local_path, user_comments, analyzed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result["paper_id"],
        result.get("title", ""),
        authors_json,
        result.get("link", ""),
        keywords_json,
        result.get("research_topic", ""),
        result.get("basic_method", ""),
        result.get("main_conclusion", ""),
        citations_json,
        intro_cites_json,
        result.get("outlook", ""),
        result.get("summary", ""),
        result.get("local_path", ""),
        user_comments_json,
        result["analyzed_at"]
    ))

    # 插入或更新 FTS 表（通过 rowid 关联主表）
    # 注意：FTS5 会自动与主表同步（如果使用了 content 选项），但此处我们显式插入以确保索引更新
    # 我们构造一个用于 FTS 的字符串，将 keywords 数组转为空格分隔
    keywords_str = ' '.join(result.get("keywords", []))
    comments_str = ' '.join(result.get("user_comments", []))
    c.execute('''
        INSERT INTO papers_fts (rowid, title, research_topic, basic_method, main_conclusion, outlook, summary, keywords, user_comments_text)
        VALUES (last_insert_rowid(), ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result.get("title", ""),
        result.get("research_topic", ""),
        result.get("basic_method", ""),
        result.get("main_conclusion", ""),
        result.get("outlook", ""),
        result.get("summary", ""),
        keywords_str,
        comments_str
    ))

    conn.commit()
    conn.close()

    print(f"论文 {paper_id} 已存入数据库")

    return result

def query_knowledge(keyword: str = None, topic: str = None, author: str = None):
    """
    在知识库中搜索匹配的文章。
    支持全文关键词搜索，以及按主题、作者过滤。
    返回匹配的文章列表（每个文章为dict）。
    """
    conn = get_db_connection()
    c = conn.cursor()

    # 构建查询条件
    conditions = []
    params = []

    # 全文检索：使用 FTS 的 MATCH 语法
    if keyword:
        # 在 FTS 表中搜索所有字段（我们构建了一个包含所有文本字段的 FTS 表）
        # 注意：FTS 的 MATCH 默认搜索所有已索引列
        conditions.append("p.rowid IN (SELECT rowid FROM papers_fts WHERE papers_fts MATCH ?)")
        params.append(keyword)

    # 按主题过滤（research_topic 精确匹配或模糊匹配？这里用 LIKE，也可以考虑 FTS）
    if topic:
        conditions.append("p.research_topic LIKE ?")
        params.append(f"%{topic}%")

    # 按作者过滤（authors 是 JSON 数组，需要特殊处理）
    if author:
        # SQLite 没有直接查询 JSON 数组元素的内置函数，但可以使用 json_each 或 LIKE 简单处理
        # 这里用 JSON1 扩展的 json_each 进行精确作者匹配（需要 SQLite 编译了 JSON1）
        # 为简化，使用 LIKE 模糊匹配（可能误匹配）
        conditions.append("p.authors LIKE ?")
        params.append(f"%{author}%")

    # 构建最终 SQL
    base_sql = "SELECT p.* FROM papers p"
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)
    else:
        where_clause = ""

    sql = base_sql + where_clause
    rows = c.execute(sql, params).fetchall()
    conn.close()

    # 转换为字典并解析 JSON 字段
    results = []
    for row in rows:
        paper = dict(row)
        for field in ['authors', 'keywords', 'citations', 'intro_cites', 'user_comments']:
            if paper.get(field):
                try:
                    parsed = json.loads(paper[field])
                    # 如果解析后还是字符串（说明可能是双重 JSON），则再次解析
                    if isinstance(parsed, str):
                        parsed = json.loads(parsed)
                    paper[field] = parsed
                except:
                    paper[field] = []
            else:
                paper[field] = []
        results.append(paper)
    return results

# ----------------------------
# （可选）迁移现有 JSON 文件到数据库
# ----------------------------
def migrate_json_to_db():
    """将 KNOWLEDGE_DIR 下所有 .json 文件（除了 index.json）导入数据库"""
    conn = get_db_connection()
    c = conn.cursor()

    for json_file in KNOWLEDGE_DIR.glob("*.json"):
        if json_file.name == "index.json":
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        paper_id = data.get("paper_id", json_file.stem)   # 兼容旧文件可能无 paper_id
        # 检查是否已存在，若存在可跳过或覆盖，这里采用覆盖
        # 准备字段（与 analyze_paper 类似）
        authors_json = json.dumps(data.get("authors", []), ensure_ascii=False)
        keywords_json = json.dumps(data.get("keywords", []), ensure_ascii=False)
        citations_json = json.dumps(data.get("citations", []), ensure_ascii=False)
        intro_cites_json = json.dumps(data.get("intro_cites", []), ensure_ascii=False)
        user_comments_json = json.dumps(data.get("user_comments", []), ensure_ascii=False)

        c.execute('''
            INSERT OR REPLACE INTO papers (
                paper_id, title, authors, link, keywords, research_topic,
                basic_method, main_conclusion, citations, intro_cites,
                outlook, summary, local_path, user_comments, analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            paper_id,
            data.get("title", ""),
            authors_json,
            data.get("link", ""),
            keywords_json,
            data.get("research_topic", ""),
            data.get("basic_method", ""),
            data.get("main_conclusion", ""),
            citations_json,
            intro_cites_json,
            data.get("outlook", ""),
            data.get("summary", ""),
            data.get("local_path", ""),
            user_comments_json,
            data.get("analyzed_at", datetime.now().isoformat())
        ))

        # 插入 FTS
        keywords_str = ' '.join(data.get("keywords", []))
        comments_str = ' '.join(data.get("user_comments", []))
        c.execute('''
            INSERT INTO papers_fts (rowid, title, research_topic, basic_method, main_conclusion, outlook, summary, keywords, user_comments_text)
            VALUES (last_insert_rowid(), ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get("title", ""),
            data.get("research_topic", ""),
            data.get("basic_method", ""),
            data.get("main_conclusion", ""),
            data.get("outlook", ""),
            data.get("summary", ""),
            keywords_str,
            comments_str
        ))

    conn.commit()
    conn.close()
    print("JSON 文件迁移完成。")

# ----------------------------
# 增量扫描本地论文
# ----------------------------
def scan_local_papers():
    """扫描 PAPER_DIR，分析新出现的 PDF（增量更新）"""
    conn = get_db_connection()
    existing_ids = set(row['paper_id'] for row in conn.execute("SELECT paper_id FROM papers").fetchall())
    conn.close()

    new_papers = []
    for pdf_file in PAPER_DIR.glob("*.pdf"):
        paper_id = pdf_file.stem
        if paper_id not in existing_ids:
            print(f"分析新论文: {paper_id}")
            result = analyze_paper(paper_id, pdf_path=str(pdf_file))
            new_papers.append(result)
        else:
            print(f"跳过已分析: {paper_id}")
    return new_papers