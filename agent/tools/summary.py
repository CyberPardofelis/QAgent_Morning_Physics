"""
agent.tools.summary — 论文摘要落库（P0.6 新增 2026-09-02, P2.5 加嵌入联动 2026-09-02）

职责:
- 抓文后(generate_daily_summary_arxiv / generate_aps_daily_summary / topic_search / manual)落库
- 走 ROADMAP_NEXT §9.1 schema,UNIQUE(paper_id, source, summary_date) 自动 dedup
- 异步调用:失败不阻塞抓文,只 log
- 提供 list / get / count 接口给 UI / P2.5 嵌入脚本用
- **P2.5**: 落库后后台线程顺手嵌入 paper_keywords collection(L0,纯本地)

跟 papers.db 的区别(ROADMAP_NEXT §ADR-N2):
- papers.db: 精细分析过(LLM 提取 method/contrib/keywords),粒度粗
- summary_items.db: 扫过就行(只存 title/authors/abstract),粒度细
- 两者独立,summary_items 体量 > papers 10-100x

P2.5: paper_keywords 独立 collection,语义检索"找和 X 贡献最像的论文"
- 抓文后自动嵌入(本文件 _bg 线程里)
- 也可手动跑 scripts/summary_items_to_chroma.py 全量嵌入
"""
import logging
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# ============================================================
# Schema（ROADMAP_NEXT §9.1）
# ============================================================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS summary_item (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id      TEXT NOT NULL,
    source        TEXT NOT NULL,        -- 'arxiv_daily' / 'aps_daily' / 'topic_search' / 'manual' / 'reindex'
    summary_date  DATE NOT NULL,
    title         TEXT NOT NULL,
    authors       TEXT,
    abstract      TEXT,
    core_method   TEXT,
    core_contrib  TEXT,
    keywords      TEXT,                 -- 逗号分隔
    in_papers_db  BOOLEAN DEFAULT 0,    -- 是否已精细分析
    analysis_id   INTEGER,              -- 关联 papers.id
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(paper_id, source, summary_date)
);

CREATE INDEX IF NOT EXISTS idx_summary_date  ON summary_item(summary_date);
CREATE INDEX IF NOT EXISTS idx_summary_source ON summary_item(source);
CREATE INDEX IF NOT EXISTS idx_summary_in_papers ON summary_item(in_papers_db);

-- 2026-09-03 加:FTS5 全文检索
-- 不用 external content(content='summary_item'),因为:
--   - external content + triggers + WAL + 多连接 = 索引状态错乱(malformed)
--   - 改成 FTS5 自带存储,触发器更新数据,稳得多
-- 代价:FTS5 多存一份文本(对 summary_item 这种小数据可接受)
-- 2026-09-03 加上 outlook(用户原话:"意义与展望"应该可搜)
CREATE VIRTUAL TABLE IF NOT EXISTS summary_item_fts USING fts5(
    title, abstract, core_method, core_contrib, keywords, outlook,
    tokenize='unicode61 remove_diacritics 2'
);
"""


# 2026-09-03 加:outlook 列(LLM 总结"意义与展望") + FTS5 触发器(自动同步)
SCHEMA_MIGRATION_SQL = """
-- outlook: LLM "意义与展望"
-- 用 try/except 模式,首次加后 ALTER 失败(列已存在)就忽略
"""


def _safe_alter(conn, sql: str) -> None:
    """ALTER TABLE 失败不抛(列已存在)"""
    try:
        conn.execute(sql)
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
            return
        raise


def _ensure_outlook_column(conn) -> None:
    """2026-09-03: 加 outlook 列(若已存在则跳过)"""
    row = conn.execute(
        "SELECT 1 FROM pragma_table_info('summary_item') WHERE name='outlook'"
    ).fetchone()
    if not row:
        conn.execute("ALTER TABLE summary_item ADD COLUMN outlook TEXT")
        logger.info("[summary] 已为 summary_item 添加 outlook 列")


def _ensure_fts_triggers(conn) -> None:
    """2026-09-03: FTS5 触发器(INSERT/UPDATE/DELETE 时同步 summary_item_fts)
    非 external content 模式,触发器直接写数据进 FTS5,稳。

    2026-09-03 P0 fix:
    - 之前用 external content + 'delete' 特殊语法,跟 WAL + 多连接冲突,索引 malformed
    - 改成自带存储的 FTS5,触发器 INSERT/DELETE 实体行,UPDATE 拆 delete+insert
    """
    conn.executescript("""
    CREATE TRIGGER IF NOT EXISTS summary_item_ai AFTER INSERT ON summary_item BEGIN
        INSERT INTO summary_item_fts(rowid, title, abstract, core_method, core_contrib, keywords, outlook)
        VALUES (new.id, new.title, new.abstract, new.core_method, new.core_contrib, new.keywords, new.outlook);
    END;
    CREATE TRIGGER IF NOT EXISTS summary_item_ad AFTER DELETE ON summary_item BEGIN
        DELETE FROM summary_item_fts WHERE rowid = old.id;
    END;
    CREATE TRIGGER IF NOT EXISTS summary_item_au AFTER UPDATE ON summary_item BEGIN
        DELETE FROM summary_item_fts WHERE rowid = old.id;
        INSERT INTO summary_item_fts(rowid, title, abstract, core_method, core_contrib, keywords, outlook)
        VALUES (new.id, new.title, new.abstract, new.core_method, new.core_contrib, new.keywords, new.outlook);
    END;
    """)


def _backfill_fts(conn) -> None:
    """2026-09-03: 如果 summary_item_fts 是空的但 summary_item 有数据,全量回填一次
    触发器只能管新增,旧数据得手动 backfill
    """
    cur = conn.execute("SELECT COUNT(*) FROM summary_item_fts")
    if cur.fetchone()[0] > 0:
        return
    cur = conn.execute("SELECT COUNT(*) FROM summary_item")
    if cur.fetchone()[0] == 0:
        return
    conn.execute("""
        INSERT INTO summary_item_fts(rowid, title, abstract, core_method, core_contrib, keywords, outlook)
        SELECT id, title, abstract, core_method, core_contrib, keywords, outlook FROM summary_item
    """)
    logger.info("[summary] FTS5 已 backfill 旧 summary_item 数据")


def _drop_legacy_external_fts(conn) -> None:
    """2026-09-03 P0 fix: 检测并删除旧的 FTS5(任何 schema 不匹配的版本)
    旧版 1:external-content(content='summary_item') + triggers + WAL + 多连接 = malformed
    旧版 2:non-external 但缺 outlook 列(2026-09-03 早期版本,用户需求 outlook 也要可搜)
    """
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='summary_item_fts'"
        ).fetchone()
    except sqlite3.OperationalError:
        row = None
    if not row or not row["sql"]:
        return
    sql = row["sql"]
    is_external = "content=" in sql.lower()
    has_outlook = "outlook" in sql.lower()
    if is_external or not has_outlook:
        logger.warning("[summary] 检测到旧版 FTS5 (external=%s, has_outlook=%s),删除重建",
                       is_external, has_outlook)
        conn.executescript("""
            DROP TRIGGER IF EXISTS summary_item_ai;
            DROP TRIGGER IF EXISTS summary_item_ad;
            DROP TRIGGER IF EXISTS summary_item_au;
            DROP TABLE IF EXISTS summary_item_fts;
        """)
        # 重建在 SCHEMA_SQL 里跑(幂等 CREATE),这里不重复


def init_schema() -> None:
    """建表(幂等,可多次调用) + 跑迁移"""
    from agent.infra.db import connect
    with connect("summary_items") as conn:
        # 先检查并删除旧版 external-content FTS5(P0 fix)
        _drop_legacy_external_fts(conn)
        conn.executescript(SCHEMA_SQL)
        _ensure_outlook_column(conn)
        _ensure_fts_triggers(conn)
        _backfill_fts(conn)
    logger.info("[summary] summary_items.db schema 初始化完成")


# ============================================================
# 初始化(模块 import 时建表)
# ============================================================
try:
    init_schema()
except Exception as e:
    # 启动时 init_schema 失败不致命(可能是 PySide6 没启动,connect 需要 QSettings 之外的 context)
    logger.warning("[summary] init_schema 失败(非致命,首次 save 时会重试): %s", e)


# ============================================================
# 异步写入(后台线程)
# ============================================================
_async_lock = threading.Lock()
_async_inflight = set()  # paper_id 集合,防止同一 paper 短时间内重复触发
# 2026-09-02 P0 fix: 限制 save_summary_item_async 后台线程并发数,防 P0.6 蓝屏复刻
# 30 papers 调 save_summary_item_async → 最多 4 个后台线程同时跑
# (per-paper dedup 仍由 _async_inflight 守,Semaphore 是全局并发限流)
_SAVE_SEMAPHORE = threading.Semaphore(4)


def save_summary_item_async(
    paper_id: str,
    source: str,
    title: str,
    authors: Optional[str] = None,
    abstract: Optional[str] = None,
    summary_date: Optional[str] = None,
    core_method: Optional[str] = None,
    core_contrib: Optional[str] = None,
    keywords: Optional[str] = None,
    outlook: Optional[str] = None,   # 2026-09-03 加
) -> None:
    """异步落库(后台线程)

    设计:
    - 不阻塞调用方(抓文流程)
    - 失败只 log,不抛
    - 用 in_flight set 短时间 dedup(避免同 paper 短时间内多次触发)
    - 用 threading 而非 asyncio:agent 工具是同步的,简化
    - 2026-09-02 P0 fix: _bg 受 Semaphore(4) 限流, 防 30 papers 起 30 线程 → 蓝屏
    """
    if not paper_id or not source or not title:
        logger.warning("[summary] save_summary_item_async 缺关键字段: paper_id=%s source=%s title=%s",
                        paper_id, source, title)
        return
    if summary_date is None:
        summary_date = date.today().isoformat()

    # 同 paper 短时间 dedup(避免同源短时间内多次触发)
    with _async_lock:
        if paper_id in _async_inflight:
            logger.debug("[summary] 已在 inflight, 跳过: paper=%s", paper_id)
            return
        _async_inflight.add(paper_id)

    def _bg():
        try:
            row_id = save_summary_item(
                paper_id=paper_id, source=source, summary_date=summary_date,
                title=title, authors=authors, abstract=abstract,
                core_method=core_method, core_contrib=core_contrib,
                keywords=keywords, outlook=outlook,
            )
            # 2026-09-02 P2.5: 落库成功后,顺手嵌入 paper_keywords collection(L0,纯本地)
            # row_id == -1 表示 UNIQUE 冲突没插,跳过嵌入(旧条目已有)
            if row_id > 0:
                embed_summary_item_to_chroma(
                    paper_id=paper_id, source=source, summary_date=summary_date,
                    title=title, authors=authors, abstract=abstract,
                    core_method=core_method, core_contrib=core_contrib,
                    keywords=keywords, row_id=row_id,
                )
        except Exception as e:
            logger.warning("[summary] 异步落库失败(已吞): paper=%s err=%s", paper_id, e)
        finally:
            with _async_lock:
                _async_inflight.discard(paper_id)

    def _bg_wrapped():
        """2026-09-02 P0 fix: Semaphore 包 _bg, 限流后台线程数
        30 papers 调 save → 最多 4 个同时跑 + 其余在 acquire 阻塞等槽位
        单次 _bg 通常 < 1s(SQLite insert + 一次 chromadb add),不会卡死 Semaphore"""
        with _SAVE_SEMAPHORE:
            _bg()

    t = threading.Thread(target=_bg_wrapped, name=f"summary_save_{paper_id[:20]}", daemon=True)
    t.start()


def save_summary_item(
    paper_id: str,
    source: str,
    title: str,
    summary_date: Optional[str] = None,
    authors: Optional[str] = None,
    abstract: Optional[str] = None,
    core_method: Optional[str] = None,
    core_contrib: Optional[str] = None,
    keywords: Optional[str] = None,
    outlook: Optional[str] = None,   # 2026-09-03 加:LLM "意义与展望"
) -> int:
    """同步落库,返回 row id (新增) 或 -1 (UNIQUE 冲突)

    UNIQUE(paper_id, source, summary_date) 冲突时忽略(用 INSERT OR IGNORE)
    """
    if summary_date is None:
        summary_date = date.today().isoformat()
    from agent.infra.db import connect
    with connect("summary_items") as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO summary_item (
                paper_id, source, summary_date, title, authors, abstract,
                core_method, core_contrib, keywords, outlook
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (paper_id, source, summary_date, title, authors, abstract,
             core_method, core_contrib, keywords, outlook),
        )
        if cur.lastrowid == 0:
            # UNIQUE 冲突,没插入
            logger.debug("[summary] UNIQUE 冲突, 跳过: paper=%s source=%s date=%s",
                         paper_id, source, summary_date)
            return -1
        return cur.lastrowid


# ============================================================
# 查询接口(给 UI / P2.5 / smoke 用)
# ============================================================
def list_summary_items(
    source: Optional[str] = None,
    since_date: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """列 summary_item 记录

    Args:
        source: 按 source 过滤(如 "arxiv_daily")
        since_date: 按 summary_date >= since_date 过滤
        limit: 最多返回多少条
        offset: 跳过前 N 条(分页用,2026-09-03 加)
    """
    from agent.infra.db import connect
    sql = "SELECT * FROM summary_item WHERE 1=1"
    params: list = []
    if source:
        sql += " AND source = ?"
        params.append(source)
    if since_date:
        sql += " AND summary_date >= ?"
        params.append(since_date)
    sql += " ORDER BY summary_date DESC, id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with connect("summary_items") as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_summary_items_fts(
    keyword: str,
    source: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """2026-09-03 加:FTS5 全文检索 summary_item

    走 summary_item_fts MATCH,匹配 title/abstract/core_method/core_contrib/keywords
    自动加 '*' 前缀支持模糊匹配(e.g. "量子" → "量子*")
    """
    if not keyword or not keyword.strip():
        return []
    from agent.infra.db import connect
    # 加 * 支持部分匹配 + 转义双引号(FTS5 syntax)
    fts_query = keyword.strip().replace('"', '""') + "*"
    sql = """
        SELECT s.* FROM summary_item s
        JOIN summary_item_fts f ON s.id = f.rowid
        WHERE summary_item_fts MATCH ?
    """
    params: list = [fts_query]
    if source:
        sql += " AND s.source = ?"
        params.append(source)
    sql += " ORDER BY rank LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with connect("summary_items") as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            # FTS5 syntax 错误(用户输入特殊字符),降级到 LIKE
            logger.warning("[summary→fts] MATCH 失败,降级 LIKE: %s", e)
            sql2 = """
                SELECT * FROM summary_item
                WHERE (title LIKE ? OR abstract LIKE ? OR core_method LIKE ?
                       OR core_contrib LIKE ? OR keywords LIKE ?)
            """
            like_pat = f"%{keyword}%"
            params2 = [like_pat] * 5
            if source:
                sql2 += " AND source = ?"
                params2.append(source)
            sql2 += " ORDER BY summary_date DESC, id DESC LIMIT ? OFFSET ?"
            params2.extend([limit, offset])
            rows = conn.execute(sql2, params2).fetchall()
    return [dict(r) for r in rows]


def get_summary_item(item_id: int) -> Optional[Dict[str, Any]]:
    from agent.infra.db import connect
    with connect("summary_items") as conn:
        row = conn.execute("SELECT * FROM summary_item WHERE id = ?", (item_id,)).fetchone()
    return dict(row) if row else None


def get_summary_item_by_paper(paper_id: str, source: str) -> Optional[Dict[str, Any]]:
    from agent.infra.db import connect
    with connect("summary_items") as conn:
        row = conn.execute(
            "SELECT * FROM summary_item WHERE paper_id=? AND source=? ORDER BY id DESC LIMIT 1",
            (paper_id, source),
        ).fetchone()
    return dict(row) if row else None


def count_summary_items(source: Optional[str] = None) -> int:
    """统计条数(给 UI 显示 / smoke 用)"""
    from agent.infra.db import connect
    sql = "SELECT COUNT(*) FROM summary_item"
    params: list = []
    if source:
        sql += " WHERE source = ?"
        params.append(source)
    with connect("summary_items") as conn:
        return conn.execute(sql, params).fetchone()[0]


def update_summary_item_fields(
    item_id: int,
    core_method: Optional[str] = None,
    core_contrib: Optional[str] = None,
    keywords: Optional[str] = None,
    in_papers_db: Optional[bool] = None,
    analysis_id: Optional[int] = None,
) -> None:
    """更新 LLM 提取的字段(analyze_paper 完成后调)

    只更新非 None 字段,允许部分更新。
    """
    sets: list = []
    params: list = []
    if core_method is not None:
        sets.append("core_method = ?")
        params.append(core_method)
    if core_contrib is not None:
        sets.append("core_contrib = ?")
        params.append(core_contrib)
    if keywords is not None:
        sets.append("keywords = ?")
        params.append(keywords)
    if in_papers_db is not None:
        sets.append("in_papers_db = ?")
        params.append(1 if in_papers_db else 0)
    if analysis_id is not None:
        sets.append("analysis_id = ?")
        params.append(analysis_id)
    if not sets:
        return
    params.append(item_id)
    from agent.infra.db import connect
    with connect("summary_items") as conn:
        conn.execute(
            f"UPDATE summary_item SET {', '.join(sets)} WHERE id = ?",
            params,
        )


# ============================================================
# P2.5: 嵌入到 paper_keywords collection (L0,纯本地)
# ============================================================
def embed_summary_item_to_chroma(
    paper_id: str,
    source: str,
    summary_date: str,
    title: str,
    authors: Optional[str] = None,
    abstract: Optional[str] = None,
    core_method: Optional[str] = None,
    core_contrib: Optional[str] = None,
    keywords: Optional[str] = None,
    row_id: int = 0,
) -> bool:
    """
    把一条 summary_item 嵌入到 chroma paper_keywords collection。

    设计:
    - **纯本地**:走本地 bge-large-en-v1.5,无 LLM 调用(L0 档)
    - **幂等**:用 f"summary_item-{paper_id}::{source}" 作 chroma id,重复调用覆盖(不重复)
    - **失败不抛**:失败只 log,抓文主流程不中断
    - **被 save_summary_item_async 落库成功后自动调用**

    为什么独立 collection(ROADMAP_NEXT §ADR-N3):
    - 跟 paper chunks metadata schema 不同(一条 summary_item 一条 chroma record)
    - 跨 collection 检索时不会污染 filter
    """
    try:
        from agent.vector_db import get_chroma_client, _make_embedding_function
        client = get_chroma_client()
        fn = _make_embedding_function()
        kwargs = {
            "name": "paper_keywords",
            "metadata": {"hnsw:space": "cosine"},
        }
        if fn is not None:
            kwargs["embedding_function"] = fn
        collection = client.get_or_create_collection(**kwargs)
    except Exception as e:
        logger.warning("[summary→chroma] 创建 collection 失败(非致命): %s", e)
        return False

    # 拼接嵌入文本
    parts = [title or ""]
    if abstract:
        parts.append(abstract[:500])
    if keywords:
        parts.append(keywords)
    text = " | ".join(p for p in parts if p)
    if not text.strip():
        logger.debug("[summary→chroma] 空文本,跳过: paper=%s", paper_id)
        return False

    meta = {
        "paper_id": paper_id,
        "source": source,
        "summary_date": summary_date,
        "title": title or "",
        "authors": authors or "",
        "keywords": keywords or "",
        "in_papers_db": False,  # 抓文时不知道,先 False,后续 analyze_paper 完成后 update
        "table": "summary_item",  # ROADMAP §10 约定
        "row_id": int(row_id or 0),
    }

    cid = f"summary_item-{paper_id}::{source}"
    try:
        # 用 update 而非 add:chromadb 0.4+ 用 update 覆盖
        # 但 update 要求 id 已存在;新条目用 add
        # 保险做法:delete + add(单条,快)
        try:
            collection.delete(ids=[cid])
        except Exception:
            pass  # 不存在是正常的
        collection.add(ids=[cid], documents=[text], metadatas=[meta])
        logger.info(
            "[summary→chroma] ✓ 嵌入 paper=%s source=%s → paper_keywords (text_len=%d)",
            paper_id, source, len(text),
        )
        return True
    except Exception as e:
        logger.warning("[summary→chroma] 嵌入失败(已吞): paper=%s err=%s", paper_id, e)
        return False


# ============================================================
# 健康检查
# ============================================================
def health_check() -> Dict[str, Any]:
    from agent.infra.db import path_of
    p = path_of("summary_items")
    return {
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else 0,
        "count": count_summary_items() if p.exists() else 0,
    }
