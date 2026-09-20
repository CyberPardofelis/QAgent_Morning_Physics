"""
knowledge._index — 拆出自原 knowledge.py (9-20 体检重构)

包含函数: index_paper_to_chroma, reindex_all, sync_recent_to_chroma
"""
from __future__ import annotations  # noqa: E402

import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

# ROOT 路径常量 (原 knowledge.py 顶部)
ROOT = Path(__file__).resolve().parent.parent.parent

from agent.tools import config as _config  # noqa: E402
from agent.tools.config import KNOWLEDGE_DIR, PAPER_DIR  # noqa: E402, F401
from agent.tools.registry import register_tool  # noqa: E402

logger = logging.getLogger(__name__)
@register_tool(name="index_paper_to_chroma", is_streaming=False, cost_tier="L0")
def index_paper_to_chroma(paper_id: str) -> str:
    """
    把单篇论文切分后索引到 chroma papers collection(纯本地,无 LLM 调用)。
    数据源:papers.db 元数据 + data/papers/{paper_id}.pdf。
    幂等:同 paper_id 先删后插。
    """
    import sqlite3

    from agent.pdf_chunker import chunk_by_section
    from agent.vector_db import VectorDB

    # 1. 读 papers.db 元数据
    db_path = Path(_config.KNOWLEDGE_DIR) / "papers.db"
    if not db_path.exists():
        return f"❌ papers.db 不存在: {db_path}"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    conn.close()
    if not row:
        return f"❌ 论文 {paper_id} 不在 papers.db,先用 analyze_paper 分析"

    paper = dict(row)

    # 2. 找 PDF 路径
    local = paper.get("local_path", "")
    pdf_path = None
    if local:
        local_norm = local.replace("\\", "/")
        p = Path(local_norm) if Path(local_norm).is_absolute() else (ROOT / local_norm)
        if p.exists():
            pdf_path = p
    if pdf_path is None:
        for ext in (".pdf",):
            p = PAPER_DIR / f"{paper_id}{ext}"
            if p.exists():
                pdf_path = p
                break
    if pdf_path is None:
        return f"❌ 找不到 PDF: data/papers/{paper_id}.pdf"

    # 3. 准备 metadata
    authors_raw = paper.get("authors", "")
    if isinstance(authors_raw, str) and authors_raw.startswith("["):
        try:
            authors_list = json.loads(authors_raw)
            authors_str = ", ".join(str(a) for a in authors_list)
        except Exception:
            authors_str = authors_raw
    else:
        authors_str = str(authors_raw) if authors_raw else ""
    keywords_raw = paper.get("keywords", "")
    if isinstance(keywords_raw, str) and keywords_raw.startswith("["):
        try:
            kw_list = json.loads(keywords_raw)
            keywords_str = ", ".join(str(k) for k in kw_list)
        except Exception:
            keywords_str = keywords_raw
    else:
        keywords_str = str(keywords_raw) if keywords_raw else ""
    try:
        year_int = int(paper.get("analyzed_at", "0")[:4] or 0)
    except Exception:
        year_int = 0
    base_meta = {
        "title": paper.get("title", "") or "",
        "authors": authors_str,
        "year": year_int,
        "journal": paper.get("link", "") or "",
        "keywords": keywords_str,
        "research_topic": paper.get("research_topic", "") or "",
        "basic_method": paper.get("basic_method", "") or "",
        "summary": paper.get("summary", "") or "",
        "source": "arxiv",
    }

    # 4. 切分
    chunks = chunk_by_section(
        pdf_path=str(pdf_path),
        paper_id=paper_id,
        title=base_meta["title"],
        year=year_int,
        journal=base_meta["journal"],
        source="arxiv",
    )
    if not chunks:
        return f"❌ {paper_id} 切出 0 chunks(PDF 可能损坏或为空)"

    # 5. 写 chroma
    texts = [c.text for c in chunks]
    metadatas = []
    for c in chunks:
        m = dict(base_meta)
        m.update({
            "paper_id": paper_id,  # 2026-09-02 v2 修:必须写 paper_id,get_paper 靠这个查
            "section": c.section,
            "section_idx": c.section_idx,
            "chunk_idx": c.chunk_idx,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "has_equation": c.has_equation,
            "has_table": c.has_table,
        })
        metadatas.append(m)
    chunk_ids = [f"{paper_id}-chunk-{i:04d}" for i in range(len(chunks))]
    db = VectorDB()
    db.delete_paper(paper_id)
    db.collection.add(ids=chunk_ids, documents=texts, metadatas=metadatas)

    return f"✅ {paper_id} 已索引: {len(chunks)} chunks 入 chroma papers collection"


@register_tool(name="reindex_all", is_streaming=False, cost_tier="L0")
def reindex_all(limit: int = 0) -> str:
    """
    全量重建 chroma papers collection。
    读 papers.db 全部论文, 用 P0.5 chunker 切, 写入 chroma(幂等 delete+add)。
    限流:每 10 篇 sleep 0.5s 防 CPU 100%。
    Args:
        limit: 最多重建多少篇(0=全部,给测试/部分重建用)
    """
    import sqlite3
    import time

    from agent.pdf_chunker import chunk_by_section
    from agent.vector_db import VectorDB

    db_path = Path(_config.KNOWLEDGE_DIR) / "papers.db"
    if not db_path.exists():
        return f"❌ papers.db 不存在: {db_path}"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT paper_id, local_path FROM papers").fetchall()
    conn.close()
    if not rows:
        return "papers.db 是空的,没东西可重建"

    if limit > 0:
        rows = rows[:limit]

    db = VectorDB()
    n_papers = 0
    n_chunks_total = 0
    n_fail = 0
    for i, row in enumerate(rows, 1):
        pid = row["paper_id"]
        local = (row["local_path"] or "").replace("\\", "/")
        pdf_path = None
        if local:
            p = Path(local) if Path(local).is_absolute() else (ROOT / local)
            if p.exists():
                pdf_path = p
        if pdf_path is None:
            for ext in (".pdf",):
                p = _config.PAPER_DIR / f"{pid}{ext}"
                if p.exists():
                    pdf_path = p
                    break
        if pdf_path is None:
            n_fail += 1
            continue
        try:
            chunks = chunk_by_section(
                pdf_path=str(pdf_path), paper_id=pid,
                title="", year=0, journal="", source="arxiv",
            )
            if chunks:
                texts = [c.text for c in chunks]
                chunk_ids = [f"{pid}-chunk-{j:04d}" for j in range(len(chunks))]
                # metadata 用 chunk 自带字段 + 简单 paper 级字段
                metadatas = [
                    {
                        "paper_id": pid,
                        "section": c.section,
                        "section_idx": c.section_idx,
                        "chunk_idx": c.chunk_idx,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "has_equation": c.has_equation,
                        "has_table": c.has_table,
                        "title": "", "year": 0, "journal": "", "source": "arxiv",
                    } for c in chunks
                ]
                db.delete_paper(pid)
                db.collection.add(ids=chunk_ids, documents=texts, metadatas=metadatas)
                n_papers += 1
                n_chunks_total += len(chunks)
        except Exception as e:
            n_fail += 1
            logger.warning("[reindex_all] %s 失败: %s", pid, e)

        if i % 10 == 0:
            time.sleep(0.5)

    return (
        f"✅ reindex_all 完成\n\n"
        f"- 成功: {n_papers} 篇\n"
        f"- 失败: {n_fail} 篇\n"
        f"- 共写入: {n_chunks_total} chunks\n"
        f"- chroma papers collection 现状: {db.count()}"
    )


@register_tool(name="sync_recent_to_chroma", is_streaming=False, cost_tier="L0")
def sync_recent_to_chroma(days: int = 7, limit: int = 20) -> str:
    """
    兜底:扫最近 N 天的 summary_items,对尚未入 chroma 的 paper 走 P0.5 chunker 切 + 入库。
    适用:抓文后 analyze 失败/summary_items 落库但 chroma 没数据。
    Args:
        days: 查最近 N 天的 summary_item(默认 7)
        limit: 最多处理多少条(默认 20, 防单次调用太慢)
    """
    import time
    from datetime import date, timedelta

    from agent.pdf_chunker import chunk_by_section
    from agent.tools.summary import list_summary_items
    from agent.vector_db import VectorDB

    since_date = (date.today() - timedelta(days=days)).isoformat()
    items = list_summary_items(since_date=since_date, limit=limit)
    if not items:
        return f"最近 {days} 天没有 summary_item 落库(等抓文或手动加)"

    db = VectorDB()
    n_indexed = 0
    n_skipped = 0
    n_fail = 0
    for it in items:
        pid = it["paper_id"]
        # 检查是否已在 chroma
        existing = db.get_paper(pid)
        if existing:
            n_skipped += 1
            continue
        # 找 PDF
        pdf_path = None
        for ext in (".pdf",):
            p = PAPER_DIR / f"{pid}{ext}"
            if p.exists():
                pdf_path = p
                break
        if pdf_path is None:
            n_fail += 1
            continue
        try:
            chunks = chunk_by_section(
                pdf_path=str(pdf_path), paper_id=pid,
                title=it.get("title", "") or "", year=0, journal="", source="arxiv",
            )
            if chunks:
                texts = [c.text for c in chunks]
                chunk_ids = [f"{pid}-chunk-{i:04d}" for i in range(len(chunks))]
                metadatas = [
                    {
                        "paper_id": pid, "title": it.get("title", ""),
                        "section": c.section, "section_idx": c.section_idx,
                        "chunk_idx": c.chunk_idx, "page_start": c.page_start,
                        "page_end": c.page_end, "has_equation": c.has_equation,
                        "has_table": c.has_table, "year": 0, "journal": "",
                        "keywords": it.get("keywords", ""), "source": "arxiv",
                    } for c in chunks
                ]
                db.delete_paper(pid)
                db.collection.add(ids=chunk_ids, documents=texts, metadatas=metadatas)
                n_indexed += 1
        except Exception as e:
            n_fail += 1
            logger.warning("[sync_recent] %s 失败: %s", pid, e)
        if (n_indexed + n_fail) % 5 == 0:
            time.sleep(0.3)

    return (
        f"✅ sync_recent_to_chroma 完成\n\n"
        f"- 时间窗: 最近 {days} 天\n"
        f"- 处理 summary_items: {len(items)} 条\n"
        f"- 索引成功: {n_indexed} 篇\n"
        f"- 跳过(已在库): {n_skipped} 篇\n"
        f"- 失败: {n_fail} 篇\n"
        f"- chroma papers collection 现状: {db.count()}"
    )
