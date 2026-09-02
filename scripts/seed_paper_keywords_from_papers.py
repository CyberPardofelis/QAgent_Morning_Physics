"""
scripts.seed_paper_keywords_from_papers — P2.5 一次性回填脚本（2026-09-02）

职责:
- 读 papers.db 全部 30+ 篇已分析论文
- 写入 summary_items.db (source='reindex'), 复用抓文落库机制
- 后续 summary_items_to_chroma.py 会自动嵌入 paper_keywords

为什么:
- 真实场景下 paper_keywords 来自抓文(arxiv_daily / aps_daily), 但当前 0 条
- papers.db 已有 30 篇高质量元数据(标题/作者/关键词/方法), 值得先回填验证
- 让 P2.5 立刻可用, 不用等下次抓文

跑法(QAgent 根目录):
    venv/Scripts/python.exe scripts/seed_paper_keywords_from_papers.py
    # 然后:
    venv/Scripts/python.exe scripts/summary_items_to_chroma.py
"""
import os
import sys
import sqlite3
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seed_paper_keywords")


def _list_json_fields(s: str) -> list:
    """papers.db 的 authors / keywords 字段可能是 JSON list 字符串, 拆成 list"""
    if not s:
        return []
    s = s.strip()
    if s.startswith("["):
        try:
            import json
            return [str(x) for x in json.loads(s)]
        except Exception:
            pass
    return [s]


def main():
    from agent.tools.summary import (
        save_summary_item, count_summary_items, list_summary_items,
    )
    from config import Config
    from agent.infra.db import _data_root  # noqa: F401  保留以防 monkey patch

    knowledge_dir = Path("data/knowledge")
    papers_db = knowledge_dir / "papers.db"
    if not papers_db.exists():
        logger.error("papers.db 不存在: %s", papers_db)
        return 1

    # 1. 读 papers.db
    conn = sqlite3.connect(str(papers_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM papers").fetchall()
    conn.close()
    logger.info("读到 %d 篇 papers.db 论文", len(rows))

    if not rows:
        logger.warning("papers.db 是空的, 跳过")
        return 0

    # 2. 逐篇写入 summary_items.db (source='reindex')
    n_added = 0
    n_skipped = 0
    for r in rows:
        row = dict(r)
        pid = row.get("paper_id")
        if not pid:
            continue
        # 已有同 source 条目? 跳过(避免重复)
        existing = [it for it in list_summary_items(source="reindex", limit=10000)
                    if it.get("paper_id") == pid]
        if existing:
            n_skipped += 1
            continue

        # 构造 summary_item 字段
        title = row.get("title", "") or ""
        authors_list = _list_json_fields(row.get("authors", ""))
        authors_str = ", ".join(authors_list)
        kw_list = _list_json_fields(row.get("keywords", ""))
        keywords_str = ", ".join(kw_list)
        # abstract 拼自 basic_method + main_conclusion + summary(如果没 abstract 字段)
        method = row.get("basic_method", "") or ""
        conclusion = row.get("main_conclusion", "") or ""
        summ = row.get("summary", "") or ""
        # abstract = method 前 500 字符(主要语义信息)
        abstract = method[:500] if method else summ[:500]
        core_method = method[:1000] if method else ""
        core_contrib = conclusion[:1000] if conclusion else ""

        # analyzed_at 拆日期
        analyzed_at = row.get("analyzed_at", "") or ""
        summary_date = analyzed_at[:10] if analyzed_at else "2026-09-02"

        new_id = save_summary_item(
            paper_id=pid,
            source="reindex",
            summary_date=summary_date,
            title=title,
            authors=authors_str,
            abstract=abstract,
            core_method=core_method,
            core_contrib=core_contrib,
            keywords=keywords_str,
        )
        if new_id > 0:
            n_added += 1
            # 设置 in_papers_db=True + analysis_id(暂时用 rowid 不准, 留空)
            from agent.tools.summary import update_summary_item_fields
            update_summary_item_fields(new_id, in_papers_db=True)
        else:
            n_skipped += 1

    total = count_summary_items()
    logger.info("=" * 50)
    logger.info("✓ 回填完成:")
    logger.info("  新增: %d 条", n_added)
    logger.info("  跳过(已存在): %d 条", n_skipped)
    logger.info("  summary_items.db 现有: %d 条", total)
    logger.info("=" * 50)
    logger.info("下一步: venv/Scripts/python.exe scripts/summary_items_to_chroma.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
