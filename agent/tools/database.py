import sqlite3
import json
from datetime import datetime
from agent.tools.config import DB_PATH
from agent.tools.registry import register_tool


# ----------------------------
# 辅助函数：获取数据库连接（每次调用都新建连接，避免多线程问题）
# ----------------------------
def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ----------------------------
# 数据库初始化
# ----------------------------
def init_db():
    """创建数据库表（如果不存在）"""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # 主表：存储所有结构化字段，并将 user_comments 存为 JSON 字符串
    c.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,               -- JSON 数组字符串
            link TEXT,
            keywords TEXT,               -- JSON 数组字符串
            research_topic TEXT,
            basic_method TEXT,
            main_conclusion TEXT,
            citations TEXT,              -- JSON 数组字符串
            intro_cites TEXT,             -- JSON 数组字符串
            outlook TEXT,
            summary TEXT,
            local_path TEXT,
            user_comments TEXT,           -- JSON 数组字符串
            analyzed_at TEXT
        )
    ''')

    # FTS5 虚拟表
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
            title,
            research_topic,
            basic_method,
            main_conclusion,
            outlook,
            summary,
            keywords,
            user_comments_text,
            content=papers
        )
    ''')

    # 尝试添加 llm_summary 列（如果尚未添加）
    try:
        c.execute("ALTER TABLE papers ADD COLUMN llm_summary TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


# ============================================================
# KnowledgeBaseDialog 内部使用的查询函数（不是 LLM 工具，直接返回 dict/list）
# ============================================================
def get_paper_by_id(paper_id: str):
    """根据 ID 获取论文完整信息（转换为字典），供 UI 直接使用"""
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    conn.close()
    if not row:
        return None
    paper = dict(row)
    for field in ['authors', 'keywords', 'citations', 'intro_cites', 'user_comments']:
        if paper.get(field):
            try:
                parsed = json.loads(paper[field])
                while isinstance(parsed, str):
                    parsed = json.loads(parsed)
                paper[field] = parsed
            except Exception:
                paper[field] = []
        else:
            paper[field] = []
    return paper


# ============================================================
# LLM 工具：update_paper_comment
# ============================================================
@register_tool(name="update_paper_comment", is_streaming=False)
def update_paper_comment(paper_id: str, user_comment: str) -> str:
    """
    更新已有记录的 user_comments 字段（追加模式）。
    如果记录不存在，抛出异常（工具层捕获后 yield 错误消息）。
    """
    conn = get_db_connection()
    c = conn.cursor()

    row = c.execute("SELECT user_comments FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if not row:
        conn.close()
        return f"❌ 论文 {paper_id} 的分析记录不存在，无法更新评论。请先运行 analyze_paper 进行分析。"

    comments = json.loads(row['user_comments']) if row['user_comments'] else []
    if not isinstance(comments, list):
        comments = [comments] if comments else []

    if user_comment:
        comments.append(user_comment)

    c.execute('''
        UPDATE papers SET user_comments = ?, analyzed_at = ?
        WHERE paper_id = ?
    ''', (json.dumps(comments, ensure_ascii=False), datetime.now().isoformat(), paper_id))

    comments_str = ' '.join(comments)
    c.execute('''
            UPDATE papers_fts SET user_comments_text = ?
            WHERE rowid = (SELECT rowid FROM papers WHERE paper_id = ?)
        ''', (comments_str, paper_id))

    conn.commit()
    conn.close()
    return f"✅ 评论已更新至论文 {paper_id}，现有 {len(comments)} 条评论"


# ============================================================
# LLM 工具：get_recent_papers
# ============================================================
@register_tool(name="get_recent_papers", is_streaming=False)
def get_recent_papers(limit: int = 15) -> str:
    """
    获取最近分析的论文列表（按时间倒序），用于回顾最近看过的文章。
    返回 Markdown 格式的列表。
    """
    conn = get_db_connection()
    rows = conn.execute('''
        SELECT paper_id, title, analyzed_at
        FROM papers
        ORDER BY analyzed_at DESC
        LIMIT ?
    ''', (limit,)).fetchall()
    conn.close()

    if not rows:
        return "知识库中还没有论文记录。"

    lines = [f"最近分析的 {len(rows)} 篇论文：\n"]
    for i, row in enumerate(rows, 1):
        title = row['title'] or '(无标题)'
        paper_id = row['paper_id']
        analyzed_at = row['analyzed_at'] or ''
        lines.append(f"{i}. **{title}**")
        lines.append(f"   - ID: `{paper_id}`")
        lines.append(f"   - 分析时间: {analyzed_at}")
        lines.append("")
    return "\n".join(lines)


# ============================================================
# 内部使用：UI 调用的同步函数（不是 LLM 工具）
# ============================================================
def get_recent_papers_raw(limit=15):
    """UI 内部用的，返回 list of dict（KnowledgeBaseDialog 直接用）"""
    conn = get_db_connection()
    rows = conn.execute('''
        SELECT paper_id, title, analyzed_at
        FROM papers
        ORDER BY analyzed_at DESC
        LIMIT ?
    ''', (limit,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_paper_llm_summary(paper_id: str, summary: str):
    """将对话中生成的 LLM 总结保存到论文记录（UI 内部使用）"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE papers SET llm_summary = ? WHERE paper_id = ?", (summary, paper_id))
    conn.commit()
    conn.close()


def update_full_paper(updated_data):
    """更新论文完整信息到数据库（UI 内部使用）"""
    conn = get_db_connection()
    c = conn.cursor()

    authors_json = json.dumps(updated_data["authors"], ensure_ascii=False)
    keywords_json = json.dumps(updated_data["keywords"], ensure_ascii=False)
    citations_json = json.dumps(updated_data["citations"], ensure_ascii=False)
    intro_cites_json = json.dumps(updated_data["intro_cites"], ensure_ascii=False)
    user_comments_json = json.dumps(updated_data["user_comments"], ensure_ascii=False)

    c.execute('''
        UPDATE papers SET
            title=?, authors=?, link=?, keywords=?, research_topic=?,
            basic_method=?, main_conclusion=?, citations=?, intro_cites=?,
            outlook=?, summary=?, local_path=?, user_comments=?, analyzed_at=?, llm_summary=?
        WHERE paper_id=?
    ''', (
        updated_data["title"], authors_json, updated_data["link"], keywords_json,
        updated_data["research_topic"], updated_data["basic_method"],
        updated_data["main_conclusion"], citations_json, intro_cites_json,
        updated_data["outlook"], updated_data["summary"], updated_data["local_path"],
        user_comments_json, updated_data["analyzed_at"], updated_data.get("llm_summary", ""),
        updated_data["paper_id"]
    ))

    keywords_str = ' '.join(updated_data["keywords"])
    comments_str = ' '.join(updated_data["user_comments"])
    c.execute('''
        UPDATE papers_fts SET
            title=?, research_topic=?, basic_method=?, main_conclusion=?,
            outlook=?, summary=?, keywords=?, user_comments_text=?
        WHERE rowid = (SELECT rowid FROM papers WHERE paper_id=?)
    ''', (
        updated_data["title"], updated_data["research_topic"], updated_data["basic_method"],
        updated_data["main_conclusion"], updated_data["outlook"], updated_data["summary"],
        keywords_str, comments_str, updated_data["paper_id"]
    ))
    conn.commit()
    conn.close()
