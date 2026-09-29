"""
统一的 SQLite 连接管理（参考 WAgent 实现）。

设计目标：
- 集中管理：连接配置（WAL、外键、busy_timeout）只写一次
- 不做运行时迁移：schema 演进由各模块自己的 init_schema() 负责（幂等的 CREATE TABLE IF NOT EXISTS）
- 健康检查：health_check() 供诊断和状态栏使用

QAgent 数据布局（保留现有文件位置以兼容旧数据）：
- data/conversations/conversations.db  → 对话历史
- data/knowledge/papers.db            → 论文知识库
- data/memory/memories.db             → 通用记忆库（v1，2026-08-20 新增）
- data/knowledge/summary_items.db     → 论文摘要库（2026-09-02 P0.6）
- data/knowledge/surveys.db           → 综述库（2026-09-02 P6 准备，UI 先上线）
"""
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 数据库注册表
# ============================================================
# 各数据库文件名（保留现有位置，兼容旧数据）
DB_FILES: dict[str, str] = {
    "conversations": "conversations/conversations.db",   # 对话历史
    "papers":        "knowledge/papers.db",              # 论文知识库
    "memory":        "memory/memories.db",               # 通用记忆库
    # 2026-09-02 P0.6 加:summary_items 独立库(粒度比 papers 细,只存"扫过"的论文摘要)
    # ROADMAP_NEXT §ADR-N2:避免和 papers.db 混用,语义不同
    "summary_items": "knowledge/summary_items.db",       # 论文摘要(抓文后落)
    # 2026-09-02 加:surveys 综述库(主题 + 章节树状结构 + 内容 markdown)
    # ROADMAP_NEXT §9.2 schema 简化版(暂不接 ReAct,UI 先建空库,等 P6 自动填)
    "surveys":       "knowledge/surveys.db",             # 综述库(主题/章节/正文)
    # 2026-09-29 加:facts 事实库(跨来源的知识底座)
    # ADR: fact 与 source 分表 —— 一条 fact 可有多个出处(论文A/论文B/对话…),
    # 这一点是 fact 作为"知识型数据库"的核心价值(合并证据), 单表做不到。
    "facts":         "knowledge/facts.db",               # 原子事实 + 多来源出处
}


def _data_root() -> Path:
    return Path("data")


# ============================================================
# 连接管理
# ============================================================
@contextmanager
def connect(name: str) -> Iterator[sqlite3.Connection]:
    """
    统一的 SQLite 连接 context manager。

    特性：
    - PRAGMA foreign_keys = ON：启用外键约束
    - PRAGMA journal_mode = WAL：提升并发读性能
    - PRAGMA busy_timeout = 30000：写入冲突时最多等 30 秒
    - 自动 commit / rollback
    - 异常时自动 close

    用法：
        with connect("conversations") as conn:
            conn.execute("SELECT ...")
    """
    if name not in DB_FILES:
        raise ValueError(f"Unknown database: {name!r}. Known: {list(DB_FILES.keys())}")
    path = _data_root() / DB_FILES[name]
    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def path_of(name: str) -> Path:
    """获取数据库文件路径（不打开连接）"""
    if name not in DB_FILES:
        raise ValueError(f"Unknown database: {name!r}")
    return _data_root() / DB_FILES[name]


# ============================================================
# 健康检查
# ============================================================
def health_check() -> dict[str, dict[str, Any]]:
    """
    检查所有数据库状态。供诊断 / 状态栏 / 启动校验使用。

    返回 {db_name: {path, exists, size_bytes, table_count, writable, error?}}
    """
    results: dict[str, dict[str, Any]] = {}
    for name in DB_FILES:
        p = path_of(name)
        info: dict[str, Any] = {
            "path": str(p),
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "table_count": 0,
            "writable": False,
        }
        if not p.exists():
            # 还没建过的库不算错误
            results[name] = info
            continue
        try:
            with connect(name) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                ).fetchone()
                info["table_count"] = row[0] if row else 0
                info["writable"] = True
        except Exception as e:
            info["error"] = str(e)
        results[name] = info
    return results


def all_paths() -> dict[str, Path]:
    """返回所有库的路径（用于备份 / 调试）"""
    return {name: path_of(name) for name in DB_FILES}
