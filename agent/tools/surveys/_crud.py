"""
agent.tools.surveys._crud — 主题/章节 CRUD + 健康检查 (UI 同步调, 0 LLM)

从原 surveys.py 抽出 (P1.7 重构):
- 30+ 个函数从 2129 行 monolith 拆出
- 本文件包含: SCHEMA_SQL + init_schema + 主题 CRUD + 章节 CRUD + health_check
- 同步 sqlite3, 不调 LLM, 不依赖 faiss/chroma
"""
import logging
import sqlite3
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# ============================================================
# Schema (ROADMAP_NEXT §9.2 简化版)
# ============================================================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS survey_theme (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    description  TEXT DEFAULT '',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS survey_section (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_id     INTEGER NOT NULL,
    section_type TEXT NOT NULL DEFAULT 'intro',  -- intro / progress / method_compare / gap / outlook / custom
    title        TEXT NOT NULL,
    content_md   TEXT DEFAULT '',
    paper_ids    TEXT DEFAULT '',                -- 逗号分隔
    order_idx    INTEGER DEFAULT 0,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (theme_id) REFERENCES survey_theme(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_survey_section_theme ON survey_section(theme_id, order_idx);
"""


def init_schema() -> None:
    """建表 (幂等, 可多次调用)"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.executescript(SCHEMA_SQL)
        # 启用外键 (级联删除)
        conn.execute("PRAGMA foreign_keys = ON")
    logger.info("[surveys] surveys.db schema 初始化完成")


# ============================================================
# 主题 CRUD
# ============================================================
def create_theme(name: str, description: str = "") -> int:
    """新建主题, 返新 id"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        cur = conn.execute(
            "INSERT INTO survey_theme (name, description) VALUES (?, ?)",
            (name, description),
        )
        return cur.lastrowid


def list_themes() -> List[Dict[str, Any]]:
    """列所有主题"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM survey_theme ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_theme(theme_id: int) -> Optional[Dict[str, Any]]:
    """按 id 拿主题"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM survey_theme WHERE id = ?", (theme_id,)).fetchone()
    return dict(row) if row else None


def update_theme(theme_id: int, name: str = "", description: str = "") -> None:
    """更新主题 (只更新非空字段)"""
    from agent.infra.db import connect
    sets, params = [], []
    if name:
        sets.append("name = ?")
        params.append(name)
    if description:
        sets.append("description = ?")
        params.append(description)
    if not sets:
        return
    sets.append("updated_at = CURRENT_TIMESTAMP")
    params.append(theme_id)
    with connect("surveys") as conn:
        conn.execute(f"UPDATE survey_theme SET {', '.join(sets)} WHERE id = ?", params)


def delete_theme(theme_id: int) -> int:
    """删除主题 (级联删所有章节), 返删除章节数"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        n_sections = conn.execute(
            "SELECT COUNT(*) FROM survey_section WHERE theme_id = ?", (theme_id,),
        ).fetchone()[0]
        conn.execute("DELETE FROM survey_theme WHERE id = ?", (theme_id,))
    return n_sections


# ============================================================
# 章节 CRUD
# ============================================================
def create_section(
    theme_id: int,
    title: str,
    section_type: str = "intro",
    content_md: str = "",
    paper_ids: str = "",
) -> int:
    """新建章节, 自动算 order_idx = max + 1"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        max_idx = conn.execute(
            "SELECT COALESCE(MAX(order_idx), -1) FROM survey_section WHERE theme_id = ?",
            (theme_id,),
        ).fetchone()[0]
        cur = conn.execute(
            """INSERT INTO survey_section (theme_id, section_type, title, content_md, paper_ids, order_idx)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (theme_id, section_type, title, content_md, paper_ids, max_idx + 1),
        )
        # 同步更新主题的 updated_at
        conn.execute(
            "UPDATE survey_theme SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (theme_id,),
        )
        return cur.lastrowid


def list_sections(theme_id: int) -> List[Dict[str, Any]]:
    """列某主题下的所有章节 (按 order_idx 排序)"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM survey_section WHERE theme_id = ? ORDER BY order_idx",
            (theme_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_section(section_id: int) -> Optional[Dict[str, Any]]:
    """按 id 拿章节"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM survey_section WHERE id = ?", (section_id,)).fetchone()
    return dict(row) if row else None


def update_section(
    section_id: int,
    title: str = "",
    section_type: str = "",
    content_md: Optional[str] = None,  # None = 不传, "" = 清空
    paper_ids: str = "",
) -> None:
    """更新章节 (只更新传入的字段, None 表示跳过)"""
    from agent.infra.db import connect
    sets, params = [], []
    if title:
        sets.append("title = ?")
        params.append(title)
    if section_type:
        sets.append("section_type = ?")
        params.append(section_type)
    if content_md is not None:  # None=不传, ""=清空
        sets.append("content_md = ?")
        params.append(content_md)
    if paper_ids:
        sets.append("paper_ids = ?")
        params.append(paper_ids)
    if not sets:
        return
    sets.append("updated_at = CURRENT_TIMESTAMP")
    params.append(section_id)
    with connect("surveys") as conn:
        conn.execute(f"UPDATE survey_section SET {', '.join(sets)} WHERE id = ?", params)
        # 同步更新所属主题的 updated_at
        conn.execute(
            """UPDATE survey_theme SET updated_at = CURRENT_TIMESTAMP
               WHERE id = (SELECT theme_id FROM survey_section WHERE id = ?)""",
            (section_id,),
        )


def delete_section(section_id: int) -> None:
    """删除章节"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.execute("DELETE FROM survey_section WHERE id = ?", (section_id,))


def reorder_section(section_id: int, new_order: int) -> None:
    """调整章节顺序"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.execute(
            "UPDATE survey_section SET order_idx = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_order, section_id),
        )


# ============================================================
# 健康检查
# ============================================================
def health_check() -> Dict[str, Any]:
    from agent.infra.db import path_of
    p = path_of("surveys")
    info = {
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else 0,
    }
    if p.exists():
        info["n_themes"] = len(list_themes())
        # 总章节数
        from agent.infra.db import connect
        with connect("surveys") as conn:
            info["n_sections"] = conn.execute("SELECT COUNT(*) FROM survey_section").fetchone()[0]
    else:
        info["n_themes"] = 0
        info["n_sections"] = 0
    return info
