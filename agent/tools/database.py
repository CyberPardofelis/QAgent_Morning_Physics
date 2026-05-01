import sqlite3
import json
from datetime import datetime
from .config import DB_PATH

# ----------------------------
# 辅助函数：获取数据库连接（每次调用都新建连接，避免多线程问题）
# ----------------------------
def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row   # 便于按列名访问
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

    # FTS5 虚拟表：包含所有需要全文检索的字段
    # 注意：FTS5 表通过 rowid 与主表关联，此处使用 content 选项指向主表（可选，但推荐）
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
            title,
            research_topic,
            basic_method,
            main_conclusion,
            outlook,
            summary,
            keywords,           -- 虽然 keywords 是数组，但我们会将其转为空格分隔的字符串存入 FTS
            user_comments_text,
            content=papers       -- 可选，关联主表以便使用 "content=" 选项，但需要主表存在且列名匹配
        )
    ''')

    # 尝试添加 llm_summary 列（如果尚未添加）
    try:
        c.execute("ALTER TABLE papers ADD COLUMN llm_summary TEXT")
    except sqlite3.OperationalError:
        # 列已存在，忽略
        pass

    conn.commit()
    conn.close()

# ----------------------------
# 获取论文信息
# ----------------------------
def get_paper_by_id(paper_id: str):
    """根据 ID 获取论文完整信息（转换为字典）"""
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    conn.close()
    if not row:
        return None
    # 将 Row 对象转为 dict，并解析 JSON 字段
    paper = dict(row)
    for field in ['authors', 'keywords', 'citations', 'intro_cites', 'user_comments']:
        if paper.get(field):
            try:
                parsed = json.loads(paper[field])
                # 如果解析后还是字符串，说明可能被双重 JSON 编码，继续解析直到不再是字符串
                while isinstance(parsed, str):
                    parsed = json.loads(parsed)
                paper[field] = parsed
            except:
                paper[field] = []
        else:
            paper[field] = []
    return paper

# ----------------------------
# 更新评论函数
# ----------------------------
def update_paper_comment(paper_id: str, user_comment: str):
    """
    更新已有记录的 user_comments 字段（追加模式）。
    如果记录不存在，抛出异常。
    """
    conn = get_db_connection()
    c = conn.cursor()

    # 获取现有记录
    row = c.execute("SELECT user_comments FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError(f"论文 {paper_id} 的分析记录不存在，无法更新评论。请先运行 analyze_paper 进行分析。")

    # 解析现有评论列表
    comments = json.loads(row['user_comments']) if row['user_comments'] else []
    if not isinstance(comments, list):
        comments = [comments] if comments else []

    # 追加新评论
    if user_comment:
        comments.append(user_comment)

    # 更新数据库
    c.execute('''
        UPDATE papers SET user_comments = ?, analyzed_at = ?
        WHERE paper_id = ?
    ''', (json.dumps(comments, ensure_ascii=False), datetime.now().isoformat(), paper_id))

    comments_str = ' '.join(comments)  # 拼接所有评论
    c.execute('''
            UPDATE papers_fts SET user_comments_text = ?
            WHERE rowid = (SELECT rowid FROM papers WHERE paper_id = ?)
        ''', (comments_str, paper_id))

    conn.commit()
    conn.close()

    # 返回完整记录（可选）
    return get_paper_by_id(paper_id)

def update_paper_llm_summary(paper_id: str, summary: str):
    """将对话中生成的 LLM 总结保存到论文记录"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE papers SET llm_summary = ? WHERE paper_id = ?", (summary, paper_id))
    conn.commit()
    conn.close()

def get_recent_papers(limit=15):
    """获取最近分析的论文，按 analyzed_at 倒序"""
    conn = get_db_connection()
    rows = conn.execute('''
        SELECT paper_id, title, analyzed_at
        FROM papers
        ORDER BY analyzed_at DESC
        LIMIT ?
    ''', (limit,)).fetchall()
    conn.close()
    # 转换为字典列表
    return [dict(row) for row in rows]

def update_full_paper(updated_data):
    """更新论文完整信息到数据库（包括主表和FTS表）"""
    conn = get_db_connection()
    c = conn.cursor()

    # 将列表字段转为 JSON 字符串
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

    # 更新 FTS 表（假设已有 user_comments_text 列）
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


