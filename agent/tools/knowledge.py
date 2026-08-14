import os
import json
import re
import arxiv
from datetime import datetime
from pathlib import Path
from agent.tools.config import PAPER_DIR, KNOWLEDGE_DIR
from agent.tools.database import get_db_connection, get_paper_by_id, save_paper_record, deserialize_json_fields
from agent.tools.pdf_utils import extract_text_from_pdf
from agent.tools.llm import _call_llm
from agent.tools.registry import register_tool


# ============================================================
# LLM 工具：analyze_paper
# ============================================================
@register_tool(name="analyze_paper", is_streaming=False)
def analyze_paper(paper_id: str, pdf_path: str = "", user_comment: str = "") -> str:
    """
    分析论文（arXiv ID 或本地 PDF），提取结构化信息，保存到知识库。
    若提供有效的 pdf_path，则从 PDF 提取文本；否则从 arXiv 获取摘要。
    无论是否已有记录，都会重新生成所有字段（覆盖原有文件）。
    """
    # 兼容 None（LLM 可能不传）
    pdf_path = pdf_path or None
    user_comment = user_comment or ""

    # 获取论文文本（优先本地 PDF，否则 arXiv）
    text = None
    if pdf_path is None:
        potential_pdf = PAPER_DIR / f"{paper_id}.pdf"
        if potential_pdf.exists():
            pdf_path = str(potential_pdf)

    if pdf_path and os.path.exists(pdf_path):
        try:
            text = extract_text_from_pdf(pdf_path)[:5000]
        except Exception as e:
            return f"❌ 读取 PDF 失败: {e}"
    else:
        if pdf_path and not os.path.exists(pdf_path):
            pass  # 走 arXiv 兜底
        try:
            client = arxiv.Client()
            paper = next(client.results(arxiv.Search(id_list=[paper_id])))
            text = paper.summary
            pdf_path = None
        except (arxiv.HTTPError, StopIteration) as e:
            return f"❌ arXiv ID '{paper_id}' 无效或无法访问: {e}"

    if not text:
        return "❌ 无法获取论文内容，请提供有效的 PDF 路径或 arXiv ID"

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

    try:
        result_str = _call_llm(prompt)
    except Exception as e:
        return f"❌ LLM 分析失败: {e}"

    try:
        result = json.loads(result_str)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', result_str, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return f"❌ 无法解析 LLM 输出为 JSON: {result_str[:200]}..."
        else:
            return f"❌ 无法解析 LLM 输出为 JSON: {result_str[:200]}..."

    # 补充元数据
    result["paper_id"] = paper_id
    result["analyzed_at"] = datetime.now().isoformat()
    result["user_comments"] = [user_comment] if user_comment else []
    if pdf_path and os.path.exists(pdf_path):
        result["local_path"] = pdf_path

    # 存入数据库
    try:
        save_paper_record(result)
    except Exception as e:
        return f"❌ 存入数据库失败: {e}"

    return (
        f"✅ 论文分析完成\n\n"
        f"- 论文 ID: `{result['paper_id']}`\n"
        f"- 标题: {result.get('title', 'N/A')}\n"
        f"- 研究方向: {result.get('research_topic', 'N/A')}\n"
        f"- 关键词: {', '.join(result.get('keywords', []))}\n\n"
        f"已存入知识库。可继续基于此论文与用户讨论。"
    )


# ============================================================
# LLM 工具：query_knowledge
# ============================================================
@register_tool(name="query_knowledge", is_streaming=False)
def query_knowledge(keyword: str = "", topic: str = "", author: str = "") -> str:
    """
    在知识库中搜索匹配的文章。
    返回 Markdown 格式的论文列表。
    """
    conn = get_db_connection()
    c = conn.cursor()

    conditions = []
    params = []

    if keyword:
        conditions.append("p.rowid IN (SELECT rowid FROM papers_fts WHERE papers_fts MATCH ?)")
        params.append(keyword)
    if topic:
        conditions.append("p.research_topic LIKE ?")
        params.append(f"%{topic}%")
    if author:
        conditions.append("p.authors LIKE ?")
        params.append(f"%{author}%")

    base_sql = "SELECT p.* FROM papers p"
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)
    else:
        where_clause = ""
    sql = base_sql + where_clause
    rows = c.execute(sql, params).fetchall()
    conn.close()

    if not rows:
        return "未找到匹配的论文。"

    results = [deserialize_json_fields(dict(row)) for row in rows]

    lines = [f"找到 {len(results)} 篇匹配论文：\n"]
    for i, p in enumerate(results, 1):
        title = p.get("title", "(无标题)")
        paper_id = p.get("paper_id", "?")
        research_topic = p.get("research_topic", "")
        authors_list = p.get("authors", [])
        authors_str = ", ".join(authors_list[:3]) + (" 等" if len(authors_list) > 3 else "")
        keywords_list = p.get("keywords", [])
        keywords_str = ", ".join(keywords_list[:5])

        lines.append(f"{i}. **{title}**")
        lines.append(f"   - ID: `{paper_id}`")
        if authors_str:
            lines.append(f"   - 作者: {authors_str}")
        if research_topic:
            lines.append(f"   - 方向: {research_topic}")
        if keywords_str:
            lines.append(f"   - 关键词: {keywords_str}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# LLM 工具：scan_local_papers
# ============================================================
@register_tool(name="scan_local_papers", is_streaming=False)
def scan_local_papers() -> str:
    """扫描 PAPER_DIR，分析新出现的 PDF（增量更新）"""
    conn = get_db_connection()
    existing_ids = set(row['paper_id'] for row in conn.execute("SELECT paper_id FROM papers").fetchall())
    conn.close()

    new_papers = []
    skipped = 0
    for pdf_file in PAPER_DIR.glob("*.pdf"):
        paper_id = pdf_file.stem
        if paper_id in existing_ids:
            skipped += 1
            continue
        result = analyze_paper(paper_id, pdf_path=str(pdf_file))
        if result.startswith("✅"):
            new_papers.append(paper_id)
        else:
            return f"扫描中止：分析 {paper_id} 失败\n{result}"

    if not new_papers and not skipped:
        return "扫描完成：本地目录中暂无 PDF 文件。"
    if not new_papers:
        return f"扫描完成：跳过已分析的 {skipped} 篇，没有新论文。"

    lines = [f"扫描完成：新增分析 {len(new_papers)} 篇（跳过 {skipped} 篇）：\n"]
    for pid in new_papers:
        lines.append(f"- `{pid}`")
    return "\n".join(lines)


# ============================================================
# 内部使用：migrate_json_to_db（一次性数据迁移）
# ============================================================
def migrate_json_to_db():
    """将 KNOWLEDGE_DIR 下所有 .json 文件（除了 index.json）导入数据库"""
    for json_file in KNOWLEDGE_DIR.glob("*.json"):
        if json_file.name == "index.json":
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "paper_id" not in data:
            data["paper_id"] = json_file.stem
        if "analyzed_at" not in data:
            data["analyzed_at"] = datetime.now().isoformat()
        save_paper_record(data)
