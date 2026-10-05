"""
agent.infra.theme_graph — 知识结构图谱: 宏观主题 / 主题 / 核心对象 三层 (10-05)

**模型(用户 10-05 亲自纠正过一次, 这里写死防止再犯)**

    L1  theme_macro     宏观主题
    L2  theme           主题 = **图谱的节点**
    L2.5 theme_paradigm 主题内部的技术路线
    L3  fact            核心对象(在 facts.db 里, 已是全局内容哈希去重)

    theme_fact         主题 ↔ 核心对象, **多对多**

**论文不是节点。** 论文是 fact 的出处属性(fact_source.source_id), 主题↔论文
的边是通过 fact 涌现出来的多对多, 不独立存储 —— 独立存储会立刻和 fact 层的
归属不同步(一篇论文的 10 条 fact 分到 7 个主题时, 论文边到底是 1 条还是 7 条
没法回答)。需要时用 theme_fact ⋈ fact_source 查出来。

**主题与 workspace/run 无关。** 这条是重点, 之前错过一次:
  - `theme` / `theme_macro` / `theme_relation` 上**没有 run_id**, 主题不属于任何一次运行
  - 唯一带 run 语义的是 `theme_fact.asserted_by`, 它是**溯源**(这条归属最早是谁
    断言的), 不是归属。删掉它图谱照样成立
  - 因此结构是**累积生长**的: 新 fact 归入已有主题或新建主题, **从不删除**。
    这跟"每轮重画一张图"是两种东西, 也是为什么不需要覆盖/替换语义

**为什么加在 facts.db 而不是另开一个库**: 主题层是 fact 库的视图, 跨库就没法用
外键, ON DELETE CASCADE 也就没了。放一起时"删一条 fact → 它的所有主题归属自动
消失"是数据库保证的, 不靠应用层自觉。

幂等性: 所有写入都是 upsert / INSERT OR IGNORE, 同一批 fact 跑两遍结果一致。
这正是崩溃续跑需要的性质 —— checkpoint 说"themes 阶段做完了", 那重跑必须是
同一条语句而不是又生成一套新主题。
"""
from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- ==================== L1 宏观主题 ====================
CREATE TABLE IF NOT EXISTS theme_macro (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,     -- 全局唯一 = 跨 run 复用的键
    statement   TEXT DEFAULT '',
    notes       TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==================== L2 主题(图谱的节点) ====================
CREATE TABLE IF NOT EXISTS theme (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    macro_id    INTEGER REFERENCES theme_macro(id) ON DELETE SET NULL,
    name        TEXT NOT NULL UNIQUE,     -- 全局唯一: 同名主题在不同 run 里是**同一个**
    statement   TEXT DEFAULT '',
    notes       TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==================== L2.5 技术路线 ====================
CREATE TABLE IF NOT EXISTS theme_paradigm (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_id    INTEGER NOT NULL REFERENCES theme(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    statement   TEXT DEFAULT '',
    UNIQUE(theme_id, name)
);

-- ==================== L3 归属: 主题 ↔ 核心对象(多对多) ====================
CREATE TABLE IF NOT EXISTS theme_fact (
    theme_id    INTEGER NOT NULL REFERENCES theme(id) ON DELETE CASCADE,
    fact_id     TEXT    NOT NULL REFERENCES fact(id)  ON DELETE CASCADE,
    asserted_by TEXT DEFAULT '',          -- **溯源**: 首个断言此归属的 topic_hash
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (theme_id, fact_id)       -- 幂等键
);

-- ==================== 主题之间的横向边 ====================
CREATE TABLE IF NOT EXISTS theme_relation (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    src_theme   INTEGER NOT NULL REFERENCES theme(id) ON DELETE CASCADE,
    dst_theme   INTEGER NOT NULL REFERENCES theme(id) ON DELETE CASCADE,
    note        TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(src_theme, dst_theme)          -- 同一对主题只留一条边
);

CREATE INDEX IF NOT EXISTS idx_theme_macro_id    ON theme(macro_id);
CREATE INDEX IF NOT EXISTS idx_theme_fact_fid    ON theme_fact(fact_id);
CREATE INDEX IF NOT EXISTS idx_theme_fact_tid    ON theme_fact(theme_id);
CREATE INDEX IF NOT EXISTS idx_theme_paradigm_t  ON theme_paradigm(theme_id);
CREATE INDEX IF NOT EXISTS idx_theme_rel_dst     ON theme_relation(dst_theme);
"""


def _conn():
    """facts.db 连接。主题层和 fact 层必须同库, 否则外键失效。"""
    from agent.infra.db import connect
    return connect("facts")


def init_schema() -> None:
    """建表(幂等)。fact / fact_source 由 facts 模块自己建, 这里只补主题层。"""
    with _conn() as conn:
        conn.executescript(SCHEMA_SQL)
    logger.info("[theme_graph] 主题层 schema 初始化完成")


# ============================================================
# 归一化 —— 决定"同名即同一主题"的关键
# ============================================================

def norm_theme_name(s: str) -> str:
    """主题名归一。只有归一后相同才认为是**同一个主题**。

    为什么需要: 同一次跑里模型自己就可能给出"张量网络表示"和"张量网络表示法"
    两种写法, 不归一就会在图里变成两个节点, 而它们本来该合并。
    归一只动**空白和标点**, 不动字词 —— 动字词的"智能归一"会把本来不同的两个
    主题糊成一个, 那比多一个节点危险得多。
    """
    return re.sub(r"[\s、,，。;；:：!！?？()（）\[\]【】\"'’“”]+", "", s or "").strip().lower()


# ============================================================
# 写
# ============================================================

def upsert_macro(conn, name: str, statement: str = "", notes: str = "") -> int:
    """按名字取/建宏观主题, 顺带更新 statement。返回 macro id。"""
    n = (name or "").strip()
    if not n:
        raise ValueError("宏观主题名不能为空")
    row = conn.execute("SELECT id FROM theme_macro WHERE name = ?", (n,)).fetchone()
    if row:
        if statement or notes:
            conn.execute(
                "UPDATE theme_macro SET statement = COALESCE(NULLIF(?,''), statement), "
                "notes = COALESCE(NULLIF(?,''), notes), "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (statement or "", notes or "", row[0]))
        return row[0]
    cur = conn.execute(
        "INSERT INTO theme_macro (name, statement, notes) VALUES (?,?,?)",
        (n, statement or "", notes or ""))
    return int(cur.lastrowid)


def upsert_theme(conn, name: str, statement: str = "", notes: str = "",
                 macro_id: int | None = None) -> int:
    """按名字取/建主题。**同名即同一主题** —— 这是跨 run 累积的关键。"""
    n = (name or "").strip()
    if not n:
        raise ValueError("主题名不能为空")
    row = conn.execute("SELECT id FROM theme WHERE name = ?", (n,)).fetchone()
    if row:
        sets, params = [], []
        if statement:
            sets.append("statement = ?")
            params.append(statement)
        if notes:
            sets.append("notes = ?")
            params.append(notes)
        if macro_id:
            sets.append("macro_id = ?")
            params.append(macro_id)
        if sets:
            params.append(row[0])
            conn.execute(
                f"UPDATE theme SET {', '.join(sets)}, updated_at = CURRENT_TIMESTAMP "
                f"WHERE id = ?", params)
        return row[0]
    cur = conn.execute(
        "INSERT INTO theme (name, statement, notes, macro_id) VALUES (?,?,?,?)",
        (n, statement or "", notes or "", macro_id))
    return int(cur.lastrowid)


def link_facts(conn, theme_id: int, fact_ids: list[str],
               asserted_by: str = "") -> int:
    """主题 ← 核心对象。**幂等**: 同一 (theme, fact) 重复写入不报错、不重复。

    `asserted_by` 只记第一次是谁断言的(INSERT OR IGNORE 天然保留首次写入的值),
    它是溯源不是归属 —— 见模块头。
    """
    rows = [(theme_id, fid, asserted_by or "") for fid in fact_ids if fid]
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR IGNORE INTO theme_fact (theme_id, fact_id, asserted_by) "
        "VALUES (?,?,?)", rows)
    return len(rows)


def add_relation(conn, src_theme_id: int, dst_theme_id: int, note: str = "") -> int:
    """主题之间的横向边。**没有 type 字段** —— 10-05 删掉了那个噪声标签。"""
    conn.execute(
        "INSERT INTO theme_relation (src_theme, dst_theme, note) VALUES (?,?,?) "
        "ON CONFLICT(src_theme, dst_theme) DO UPDATE SET note = excluded.note",
        (src_theme_id, dst_theme_id, note or ""))
    row = conn.execute(
        "SELECT id FROM theme_relation WHERE src_theme = ? AND dst_theme = ?",
        (src_theme_id, dst_theme_id)).fetchone()
    return int(row[0]) if row else 0


def upsert_paradigm(conn, theme_id: int, name: str, statement: str = "") -> int:
    conn.execute(
        "INSERT INTO theme_paradigm (theme_id, name, statement) VALUES (?,?,?) "
        "ON CONFLICT(theme_id, name) DO UPDATE SET statement = excluded.statement",
        (theme_id, (name or "").strip(), statement or ""))
    row = conn.execute(
        "SELECT id FROM theme_paradigm WHERE theme_id = ? AND name = ?",
        (theme_id, (name or "").strip())).fetchone()
    return int(row[0]) if row else 0


# ============================================================
# 读
# ============================================================

def get_theme_id(conn, name: str) -> int | None:
    row = conn.execute("SELECT id FROM theme WHERE name = ?", ((name or "").strip(),)).fetchone()
    return int(row[0]) if row else None


def all_themes(conn) -> list[dict[str, Any]]:
    """全部主题(带宏观归属 + 归属 fact 数)。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT t.id, t.name, t.statement, t.notes, t.macro_id,
               m.name AS macro_name,
               (SELECT COUNT(*) FROM theme_fact tf WHERE tf.theme_id = t.id) AS n_facts,
               (SELECT COUNT(DISTINCT fs.source_id)
                  FROM theme_fact tf
                  JOIN fact_source fs ON fs.fact_id = tf.fact_id
                 WHERE tf.theme_id = t.id) AS n_papers
          FROM theme t
          LEFT JOIN theme_macro m ON m.id = t.macro_id
         ORDER BY n_facts DESC, t.name
    """).fetchall()
    out = [dict(r) for r in rows]
    conn.row_factory = None
    return out


def stats(conn) -> dict[str, Any]:
    def one(sql, *a):
        r = conn.execute(sql, a).fetchone()
        return r[0] if r else 0
    return {
        "macros": one("SELECT COUNT(*) FROM theme_macro"),
        "themes": one("SELECT COUNT(*) FROM theme"),
        "paradigms": one("SELECT COUNT(*) FROM theme_paradigm"),
        "links": one("SELECT COUNT(*) FROM theme_fact"),
        "relations": one("SELECT COUNT(*) FROM theme_relation"),
        "facts_linked": one("SELECT COUNT(DISTINCT fact_id) FROM theme_fact"),
        "facts_total": one("SELECT COUNT(*) FROM fact"),
        "themes_without_fact": one(
            "SELECT COUNT(*) FROM theme t WHERE NOT EXISTS "
            "(SELECT 1 FROM theme_fact tf WHERE tf.theme_id = t.id)"),
        "facts_unlinked": one(
            "SELECT COUNT(*) FROM fact f WHERE NOT EXISTS "
            "(SELECT 1 FROM theme_fact tf WHERE tf.fact_id = f.id)"),
        "multi_theme_facts": one(
            "SELECT COUNT(*) FROM (SELECT fact_id FROM theme_fact "
            "GROUP BY fact_id HAVING COUNT(DISTINCT theme_id) > 1)"),
    }
