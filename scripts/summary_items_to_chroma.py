"""
scripts.summary_items_to_chroma — P2.5 summary_item 嵌入到 paper_keywords collection

职责:
- 扫 summary_items.db 全部条目
- 拼接 title + abstract + keywords → 用本地 bge-large-en-v1.5 编码
- 写 chroma `paper_keywords` collection(独立于 `papers` collection)
- 幂等:用 paper_id + source 作为 chroma id 一部分,重复跑不重复入库
- 加 --rebuild 选项(delete + add)用于切分策略升级

为什么独立 collection(ROADMAP_NEXT §ADR-N3 / §10):
- metadata schema 不同(summary_items 跟 paper chunks 粒度不同)
- 跨 collection 检索时不会污染 filter
- chroma id 约定(§10): f"{table}-{row_id}",metadata 冗余 table + row_id 方便反查

跑法(QAgent 根目录):
    venv/Scripts/python.exe scripts/summary_items_to_chroma.py [--limit N] [--dry-run] [--rebuild]
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# 让脚本能 import QAgent 模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("summary_items_to_chroma")

# 2026-09-02 P2.5 锁定的 collection 名
PAPER_KEYWORDS_COLLECTION = "paper_keywords"


def build_embedding_text(item: dict) -> str:
    """
    拼接嵌入文本:title + abstract(截断) + keywords(如果有)

    为什么这样:
    - title: 论文核心,必入
    - abstract: 即使没 keywords,也有足够语义信息
    - keywords: 抓文时 LLM 提取的关键词(当前可能空,但 v2 会升级)
    - 全部用 | 分隔,避免短文本拼接成超长串
    """
    parts = [item.get("title", "")]
    abstract = item.get("abstract") or ""
    if abstract:
        parts.append(abstract[:500])  # 截断,避免太长
    keywords = item.get("keywords") or ""
    if keywords:
        parts.append(keywords)
    return " | ".join(p for p in parts if p)


def build_metadata(item: dict) -> dict:
    """
    chroma metadata(扁平化,所有值必须可序列化)

    字段说明(参考 ROADMAP_NEXT §9.1 + §10):
    - paper_id / source / summary_date: 唯一标识
    - title / authors / keywords: 检索召回时展示
    - in_papers_db / analysis_id: 关联 papers.db(已精细分析过)
    - table / row_id: ROADMAP_NEXT §10 约定,chroma id 反查用
    """
    return {
        "paper_id": item["paper_id"],
        "source": item["source"],
        "summary_date": item["summary_date"],
        "title": item.get("title", ""),
        "authors": item.get("authors") or "",
        "keywords": item.get("keywords") or "",
        "in_papers_db": bool(item.get("in_papers_db")),
        "table": "summary_item",  # ROADMAP §10 约定
        "row_id": int(item.get("id", 0) or 0),
    }


def make_chunk_id(item: dict) -> str:
    """
    chroma id 格式: f"summary_item-{paper_id}::{source}"

    为什么不只用 paper_id:
    - 同一 paper 可能从多个 source 落库(arxiv_daily + reindex)
    - 唯一性靠 (paper_id, source) 联合,跟 sqlite UNIQUE 一致
    - 跟 papers collection 区分(papers 用 f"{pid}-chunk-NNNN")
    """
    return f"summary_item-{item['paper_id']}::{item['source']}"


def main():
    parser = argparse.ArgumentParser(
        description="P2.5 summary_item 嵌入到 chroma paper_keywords collection",
    )
    parser.add_argument("--limit", type=int, default=0, help="最多嵌入多少条(0=全部)")
    parser.add_argument("--dry-run", action="store_true", help="只打印,不入库")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="先 delete 整个 collection 再 add(用于切分策略升级,默认增量)",
    )
    parser.add_argument(
        "--collection",
        default=PAPER_KEYWORDS_COLLECTION,
        help=f"chroma collection 名(默认 {PAPER_KEYWORDS_COLLECTION})",
    )
    args = parser.parse_args()

    # 1. 读 summary_items
    from agent.tools.summary import list_summary_items, health_check
    h = health_check()
    logger.info("summary_items.db 状态: %s", h)
    if h["count"] == 0:
        logger.warning("summary_items.db 是空的, 没有要嵌入的。先跑一次抓文(generate_daily_summary_arxiv)或用 _seed_paper_keywords.py 回填。")
        return 0

    items = list_summary_items(limit=args.limit if args.limit else 100000)
    logger.info("读出 %d 条 summary_item", len(items))

    if args.dry_run:
        for it in items[:5]:
            text = build_embedding_text(it)
            logger.info("[DRY-RUN] paper=%s source=%s text_len=%d",
                        it["paper_id"], it["source"], len(text))
        logger.info("[DRY-RUN] 完成, 不入库")
        return 0

    # 2. 拿 chroma client
    from agent.vector_db import get_chroma_client, _make_embedding_function
    client = get_chroma_client()
    fn = _make_embedding_function()

    # 3. rebuild 模式: 先 delete 整个 collection
    if args.rebuild:
        try:
            client.delete_collection(args.collection)
            logger.info("[rebuild] 已 delete collection: %s", args.collection)
        except Exception as e:
            logger.warning("[rebuild] delete_collection 失败(可能不存在): %s", e)

    # 4. 拿 collection(不存在则创建)
    kwargs = {"name": args.collection, "metadata": {"hnsw:space": "cosine"}}
    if fn is not None:
        kwargs["embedding_function"] = fn
    collection = client.get_or_create_collection(**kwargs)

    # 5. 批量嵌入
    n_added = 0
    n_skipped = 0
    BATCH = 50  # chroma collection.add 单次 ≤ 41666,这里分批便于错误隔离
    batch_ids = []
    batch_docs = []
    batch_metas = []

    def flush():
        nonlocal n_added
        if not batch_ids:
            return
        try:
            collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
            n_added += len(batch_ids)
            logger.debug("[flush] +%d (累计 %d)", len(batch_ids), n_added)
        except Exception as e:
            logger.exception("[flush] 失败: %s", e)
        finally:
            batch_ids.clear()
            batch_docs.clear()
            batch_metas.clear()

    for it in items:
        text = build_embedding_text(it)
        if not text.strip():
            logger.debug("跳过空文本: paper=%s", it["paper_id"])
            n_skipped += 1
            continue
        meta = build_metadata(it)
        cid = make_chunk_id(it)
        batch_ids.append(cid)
        batch_docs.append(text)
        batch_metas.append(meta)
        if len(batch_ids) >= BATCH:
            flush()

    flush()  # 收尾

    logger.info("=" * 50)
    logger.info("✓ 嵌入完成:")
    logger.info("  新增: %d 条", n_added)
    logger.info("  跳过(空文本): %d 条", n_skipped)
    logger.info("  collection: %s", args.collection)
    logger.info("  当前总数: %d", collection.count())
    logger.info("=" * 50)
    return n_added


if __name__ == "__main__":
    sys.exit(main() or 0)
