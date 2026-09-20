import contextlib
import json
import sqlite3
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
    with contextlib.suppress(sqlite3.OperationalError):
        c.execute("ALTER TABLE papers ADD COLUMN llm_summary TEXT")

    conn.commit()
    conn.close()


# ============================================================
# KnowledgeBaseDialog 内部使用的查询函数（不是 LLM 工具，直接返回 dict/list）
# ============================================================

# 论文表中以 JSON 字符串存储的列表字段
JSON_LIST_FIELDS = ('authors', 'keywords', 'citations', 'intro_cites', 'user_comments')


def deserialize_json_fields(paper: dict) -> dict:
    """
    把 papers 表里以 JSON 字符串存的列表字段（authors / keywords / citations 等）
    反序列化为 Python 对象。解析失败则置为空列表。
    会就地修改 paper 并返回。
    """
    for field in JSON_LIST_FIELDS:
        if paper.get(field):
            try:
                parsed = json.loads(paper[field])
                # 兜底：有些旧数据会被双重编码
                while isinstance(parsed, str):
                    parsed = json.loads(parsed)
                paper[field] = parsed
            except Exception:
                paper[field] = []
        else:
            paper[field] = []
    return paper


def get_paper_by_id(paper_id: str):
    """根据 ID 获取论文完整信息（转换为字典），供 UI 直接使用"""
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return deserialize_json_fields(dict(row))


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
@register_tool(name="get_recent_papers", is_streaming=False, cost_tier="L0")
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


def save_paper_record(data: dict) -> None:
    """
    把单篇论文的完整字段写入 papers + papers_fts（覆盖式）。

    必填 key: paper_id
    可选 key: title, authors(list), link, keywords(list), research_topic,
             basic_method, main_conclusion, citations(list), intro_cites(list),
             outlook, summary, local_path, user_comments(list), analyzed_at, llm_summary

    行为：
    - INSERT OR REPLACE 进 papers（按 paper_id 去重）
    - 同时同步 papers_fts（FTS5 external content 表需手动同步）
    - 如果 data 里没带 llm_summary，**保留已有值**（避免 analyze_paper 重复分析时
      清空用户通过 update_paper_llm_summary 写入的对话总结）
    """
    conn = get_db_connection()
    c = conn.cursor()

    # 保留已有 llm_summary（除非调用方显式覆盖）
    if "llm_summary" not in data:
        row = c.execute(
            "SELECT llm_summary FROM papers WHERE paper_id = ?",
            (data["paper_id"],),
        ).fetchone()
        data["llm_summary"] = row["llm_summary"] if row and row["llm_summary"] else ""

    authors_json = json.dumps(data.get("authors", []) or [], ensure_ascii=False)
    keywords_json = json.dumps(data.get("keywords", []) or [], ensure_ascii=False)
    citations_json = json.dumps(data.get("citations", []) or [], ensure_ascii=False)
    intro_cites_json = json.dumps(data.get("intro_cites", []) or [], ensure_ascii=False)
    user_comments_json = json.dumps(data.get("user_comments", []) or [], ensure_ascii=False)

    c.execute('''
        INSERT OR REPLACE INTO papers (
            paper_id, title, authors, link, keywords, research_topic,
            basic_method, main_conclusion, citations, intro_cites,
            outlook, summary, local_path, user_comments, analyzed_at, llm_summary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data["paper_id"],
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
        data.get("analyzed_at", ""),
        data.get("llm_summary", ""),
    ))

    # INSERT OR REPLACE 走 DELETE+INSERT 路径，rowid 会变，必须重新拿
    rowid = c.lastrowid

    keywords_str = " ".join(data.get("keywords") or [])
    comments_str = " ".join(data.get("user_comments") or [])
    c.execute('''
        INSERT OR REPLACE INTO papers_fts (rowid, title, research_topic, basic_method, main_conclusion, outlook, summary, keywords, user_comments_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        rowid,
        data.get("title", ""),
        data.get("research_topic", ""),
        data.get("basic_method", ""),
        data.get("main_conclusion", ""),
        data.get("outlook", ""),
        data.get("summary", ""),
        keywords_str,
        comments_str,
    ))

    conn.commit()
    conn.close()


def update_full_paper(updated_data):
    """更新论文完整信息到数据库（UI 内部使用）— 现在是 save_paper_record 的薄封装"""
    save_paper_record(updated_data)
