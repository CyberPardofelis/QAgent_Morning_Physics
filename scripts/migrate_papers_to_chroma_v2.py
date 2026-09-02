"""
scripts.migrate_papers_to_chroma_v2 — P1 全量迁移 + P0.5 chunk 策略（2026-09-02）

职责:
- 读 papers.db 全部 30+ 论文元数据
- 对每篇:用 P0.5 agent.pdf_chunker 切 PDF(段落级,带 section metadata)
- 写 chroma `papers` collection(Step 1 P0 已就位,本地 bge 嵌入)
- 幂等:同 paper_id 删旧 chunk 再 add
- 限流:每 10 篇 sleep 1s 避免 CPU 飙高
- 进度:用 print 实时打印(避免 tqdm import 复杂)
- 完成:输出 "迁移 N 篇,共 M chunks,耗时 Xs"

跟 P1 原始 ROADMAP 的差异:
- 走 P0.5 section-aware chunk 策略(不是 500 token 无脑切)
- metadata 加 section / section_idx / has_equation / has_table 字段
- papers.db 元数据自动填充(title / authors / year / journal / keywords / research_topic / basic_method / summary)

跑法(QAgent 根目录):
    venv/Scripts/python.exe scripts/migrate_papers_to_chroma_v2.py [--dry-run] [--limit N]
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate_v2")


def load_papers_metadata() -> list:
    """从 papers.db 读全部论文元数据

    Returns:
        list of dict, 每条 {paper_id, title, authors, year, journal, ...}
    """
    import sqlite3
    db_path = ROOT / "data" / "knowledge" / "papers.db"
    if not db_path.exists():
        logger.warning("papers.db 不存在: %s", db_path)
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT * FROM papers")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _authors_to_str(authors_raw) -> str:
    """authors 字段可能是 JSON 数组字符串或纯字符串,统一成逗号分隔"""
    if not authors_raw:
        return ""
    if isinstance(authors_raw, str):
        # 尝试 JSON 解析
        s = authors_raw.strip()
        if s.startswith("["):
            try:
                lst = json.loads(s)
                return ", ".join(str(x) for x in lst)
            except Exception:
                return s  # 解析失败,直接用
        return s
    if isinstance(authors_raw, list):
        return ", ".join(str(x) for x in authors_raw)
    return str(authors_raw)


def _keywords_to_str(keywords_raw) -> str:
    """keywords 字段逗号分隔字符串(避开 chroma 不支持 list)"""
    return _authors_to_str(keywords_raw)  # 同一逻辑


def _extract_year(paper: dict, local_path: Path) -> int:
    """从 paper.analyzed_at 或 arxiv id 前缀提取年份"""
    # 1. 优先 analyzed_at
    analyzed_at = paper.get("analyzed_at", "")
    if analyzed_at and len(analyzed_at) >= 4:
        try:
            return int(analyzed_at[:4])
        except Exception:
            pass
    # 2. arxiv id 前缀(2401 → 2024, 2512 → 2025)
    pid = paper.get("paper_id", "")
    if len(pid) >= 2 and pid[:2].isdigit():
        yy = int(pid[:2])
        if 90 <= yy <= 99:
            return 1900 + yy
        if 0 <= yy <= 30:
            return 2000 + yy
    # 3. 文件名
    if local_path.exists():
        import datetime
        mtime = local_path.stat().st_mtime
        return datetime.datetime.fromtimestamp(mtime).year
    return 0


def _resolve_local_path(paper: dict) -> Path:
    """拿 PDF 本地路径(papers.db 的 local_path 是相对路径)"""
    local = paper.get("local_path", "")
    if local:
        # windows 路径可能是 "data\\papers\\xxx.pdf",统一分隔符
        local_norm = local.replace("\\", "/")
        p = ROOT / local_norm
        if p.exists():
            return p
    # fallback: 用 paper_id 拼文件名
    pid = paper.get("paper_id", "")
    if pid:
        for ext in (".pdf",):
            p = ROOT / "data" / "papers" / f"{pid}{ext}"
            if p.exists():
                return p
    return None


def migrate_one_paper(paper: dict, db, dry_run: bool = False) -> int:
    """迁移单篇论文。返 chunk 数(0=失败)"""
    from agent.pdf_chunker import chunk_by_section
    pid = paper.get("paper_id", "")
    pdf_path = _resolve_local_path(paper)
    if not pid:
        return 0
    if pdf_path is None:
        logger.warning("跳过 %s: PDF 找不到", pid)
        return 0

    # 准备 paper 级 metadata
    year = _extract_year(paper, pdf_path)
    journal_str = paper.get("link", "") or ""
    if "aps.org" in journal_str or "10.1103" in journal_str:
        # 从 link 推断期刊
        if "/PRL/" in journal_str or "PhysRevLett" in journal_str:
            journal = "PRL"
        elif "/PRA/" in journal_str or "PhysRevA" in journal_str:
            journal = "PRA"
        elif "/PRB/" in journal_str or "PhysRevB" in journal_str:
            journal = "PRB"
        else:
            journal = "APS"
    elif "arxiv.org" in journal_str:
        journal = "arXiv"
    else:
        journal = paper.get("journal", "") or ""

    base_meta = {
        "title": paper.get("title", ""),
        "authors": _authors_to_str(paper.get("authors", "")),
        "year": int(year) if year else 0,
        "journal": journal,
        "keywords": _keywords_to_str(paper.get("keywords", "")),
        "research_topic": paper.get("research_topic", "") or "",
        "basic_method": paper.get("basic_method", "") or "",
        "summary": paper.get("summary", "") or "",
        "source": "arxiv",
    }

    # 切 PDF
    try:
        chunks = chunk_by_section(
            pdf_path=str(pdf_path),
            paper_id=pid,
            title=base_meta["title"],
            year=year,
            journal=journal,
            source="arxiv",
        )
    except Exception as e:
        logger.warning("切分 %s 失败: %s", pid, e)
        return 0

    if not chunks:
        logger.warning("跳过 %s: 没切出 chunk", pid)
        return 0

    if dry_run:
        logger.info("[DRY-RUN] %s: %d chunks", pid, len(chunks))
        return len(chunks)

    # 准备 chroma 添加数据(每个 chunk 合并 paper 级 + chunk 级 metadata)
    texts = [c.text for c in chunks]
    metadatas = []
    for c in chunks:
        m = dict(base_meta)
        m.update({
            "paper_id": pid,  # 2026-09-02 v2 修:get_paper 靠这个查
            "section": c.section,
            "section_idx": c.section_idx,
            "chunk_idx": c.chunk_idx,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "has_equation": c.has_equation,
            "has_table": c.has_table,
        })
        # 清理 None 值(chroma 不喜欢)
        m = {k: ("" if v is None else v) for k, v in m.items()}
        metadatas.append(m)
    # 2026-09-02 v2 修:用全局 index,避免跨 section 重复
    # 之前用 c.chunk_idx,跨 section 会重复(每个 section 自己的 chunk_idx 从 0 开始)
    # chroma 要求 collection 内 IDs 全局唯一
    chunk_ids = [f"{pid}-chunk-{i:04d}" for i in range(len(chunks))]

    # 幂等:先 delete 旧 chunk,再 add 完整 chunk 级 metadata
    db.delete_paper(pid)
    db.collection.add(ids=chunk_ids, documents=texts, metadatas=metadatas)
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="P1 全量迁移 + P0.5 chunk 策略")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="最多迁多少篇(0=全部)")
    parser.add_argument("--throttle-every", type=int, default=10, help="每 N 篇 sleep 1s")
    args = parser.parse_args()

    # 1. 读 papers.db
    papers = load_papers_metadata()
    if args.limit:
        papers = papers[:args.limit]
    logger.info("读到 %d 篇论文元数据", len(papers))
    if not papers:
        return 0

    # 2. 准备 chroma
    if not args.dry_run:
        from agent.vector_db import VectorDB
        db = VectorDB(collection_name="papers")
        logger.info("chroma collection 现状: %d chunks", db.count())
    else:
        db = None

    # 3. 逐篇迁移
    t0 = time.time()
    n_papers = 0
    n_chunks_total = 0
    n_fail = 0
    for i, paper in enumerate(papers, 1):
        pid = paper.get("paper_id", "?")
        try:
            n = migrate_one_paper(paper, db, dry_run=args.dry_run)
            if n == 0:
                n_fail += 1
            else:
                n_papers += 1
                n_chunks_total += n
            logger.info("[%d/%d] %s: %d chunks (累计 %d 篇, %d chunks)",
                        i, len(papers), pid, n, n_papers, n_chunks_total)
        except Exception as e:
            n_fail += 1
            logger.exception("[%d/%d] %s 失败: %s", i, len(papers), pid, e)

        # 限流
        if not args.dry_run and i % args.throttle_every == 0:
            time.sleep(1.0)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    if args.dry_run:
        logger.info("[DRY-RUN] 完成: 试迁 %d 篇,共 %d chunks (含失败), 耗时 %.1fs",
                    len(papers), n_chunks_total, elapsed)
    else:
        logger.info("✓ 迁移完成: 成功 %d 篇, 失败 %d 篇, 共 %d chunks, 耗时 %.1fs",
                    n_papers, n_fail, n_chunks_total, elapsed)
        if db is not None:
            logger.info("  chroma papers collection 总数: %d", db.count())
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
