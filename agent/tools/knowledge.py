import os
import json
import re
import arxiv
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
# 2026-09-02 P2 改:用模块属性访问,允许测试 patch(原来 from import 是值绑定,改不了)
from agent.tools import config as _config
from agent.tools.config import PAPER_DIR, KNOWLEDGE_DIR  # noqa: F401 保留旧 import 兼容
from agent.tools.database import get_db_connection, get_paper_by_id, save_paper_record, deserialize_json_fields
from agent.tools.pdf_utils import extract_text_from_pdf
from agent.tools.llm import _call_llm
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)


# ============================================================
# LLM 工具：analyze_paper
# ============================================================
@register_tool(name="analyze_paper", is_streaming=False)
def analyze_paper(paper_id: str, pdf_path: str = "", user_comment: str = "") -> str:
    """
    分析论文（arXiv ID 或本地 PDF），提取结构化信息，保存到知识库。
    若提供有效的 pdf_path，则从 PDF 提取文本；否则从 arXiv 获取摘要。
    无论是否已有记录，都会重新生成所有字段（覆盖原有文件）。
    """
    # 兼容 None（LLM 可能不传）
    pdf_path = pdf_path or None
    user_comment = user_comment or ""

    # 获取论文文本（优先本地 PDF，否则 arXiv）
    text = None
    if pdf_path is None:
        potential_pdf = PAPER_DIR / f"{paper_id}.pdf"
        if potential_pdf.exists():
            pdf_path = str(potential_pdf)

    if pdf_path and os.path.exists(pdf_path):
        try:
            # 2026-09-04 改:
            # - 不再 [:5000] 粗暴截断(用户报"analyze_paper JSON 解析失败,
            #   LLM 拿不到全文,补只能 run_python 跑 PyMuPDF 绕过" 已确认)
            # - pdf_utils 默认走 fitz(PyMuPDF)排版还原好,数字/公式完整
            # - 上限 60K 字符(约 15K token),既给 LLM 全文又不超 prompt
            # - 单页失败不连累整文(pdf_utils 已加 try/except)
            text = extract_text_from_pdf(pdf_path, max_chars=60000)
            if not text or not text.strip():
                return (
                    f"❌ PDF 文本提取为空({pdf_path}),可能扫描件无 OCR 层。"
                    f"请先跑 OCR 或手动提供 arXiv ID 走摘要路径。"
                )
        except FileNotFoundError as e:
            return f"❌ {e}"
        except RuntimeError as e:
            return f"❌ 读取 PDF 失败: {e}"
        except Exception as e:
            return f"❌ 读取 PDF 失败(未分类错误): {e}"
    else:
        if pdf_path and not os.path.exists(pdf_path):
            pass  # 走 arXiv 兜底
        try:
            client = arxiv.Client()
            paper = next(client.results(arxiv.Search(id_list=[paper_id])))
            text = paper.summary
            pdf_path = None
        except (arxiv.HTTPError, StopIteration) as e:
            return f"❌ arXiv ID '{paper_id}' 无效或无法访问: {e}"

    if not text:
        return "❌ 无法获取论文内容，请提供有效的 PDF 路径或 arXiv ID"

    prompt = f"""
    Analyze the following paper text and return a **valid JSON object** with these fields:
    - title: the paper title
    - authors: list of authors
    - link: The online link of the paper. If not available, please record the arXiv ID or DOI number.
    - keywords: list of keywords, about 6 to 10 keywords
    - research_topic: main topic or subject
    - basic_method: Technical methods and innovative points, about 100-400 words
    - main_conclusion: the primary conclusion and mathematical expression, about 100-300 words
   - citations: list of important citations in the method section. Each citation should be an object with the following fields: 
        * title: the title of the cited paper
        * authors: list of authors
        * year: publication year (as integer)
        * journal: journal name or venue
        * doi: DOI if available (optional)
    - intro_cites: list of important citations in the introduction section. Each citation should have the same structure as in 'citations'.
    - outlook: the meaning, impact and outlook of this paper
    - summary: a one-paragraph summary

    Paper text:
    {text}

    Output ONLY the JSON object, no other text.
    """

    try:
        result_str = _call_llm(prompt)
    except Exception as e:
        return f"❌ LLM 分析失败: {e}"

    try:
        result = json.loads(result_str)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', result_str, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return f"❌ 无法解析 LLM 输出为 JSON: {result_str[:200]}..."
        else:
            return f"❌ 无法解析 LLM 输出为 JSON: {result_str[:200]}..."

    # 补充元数据
    result["paper_id"] = paper_id
    result["analyzed_at"] = datetime.now().isoformat()
    result["user_comments"] = [user_comment] if user_comment else []
    if pdf_path and os.path.exists(pdf_path):
        result["local_path"] = pdf_path

    # 存入数据库
    try:
        save_paper_record(result)
    except Exception as e:
        return f"❌ 存入数据库失败: {e}"

    # 2026-09-02 P2.1:analyze_paper 完成后,自动切 PDF + 入 chroma(双重作用:分析 + 入库)
    # 走 P0.5 section-aware chunker + P0.5 vector_db,纯本地(L0 路径,不再调 LLM)
    indexed_msg = ""
    if pdf_path and os.path.exists(pdf_path):
        try:
            from agent.pdf_chunker import chunk_by_section
            from agent.vector_db import VectorDB
            # 准备 paper 级 metadata
            keywords = result.get("keywords", [])
            if isinstance(keywords, list):
                keywords_str = ", ".join(str(k) for k in keywords)
            else:
                keywords_str = str(keywords)
            authors_raw = result.get("authors", [])
            if isinstance(authors_raw, list):
                authors_str = ", ".join(str(a) for a in authors_raw)
            else:
                authors_str = str(authors_raw)
            base_meta = {
                "title": result.get("title", "") or "",
                "authors": authors_str,
                "year": int(result.get("year", 0) or 0),
                "journal": result.get("journal", "") or "",
                "keywords": keywords_str,
                "research_topic": result.get("research_topic", "") or "",
                "basic_method": result.get("basic_method", "") or "",
                "summary": result.get("summary", "") or "",
                "source": "arxiv",
            }
            # 切 PDF
            chunks = chunk_by_section(
                pdf_path=pdf_path,
                paper_id=paper_id,
                title=base_meta["title"],
                year=base_meta["year"],
                journal=base_meta["journal"],
                source="arxiv",
            )
            if chunks:
                # 构造 chunk_ids(全局唯一)
                texts = [c.text for c in chunks]
                metadatas = []
                for c in chunks:
                    m = dict(base_meta)
                    m.update({
                        "paper_id": paper_id,  # 2026-09-02 v2 修:get_paper 靠这个查
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
                # 写 chroma(delete + add 幂等)
                db = VectorDB()
                db.delete_paper(paper_id)
                db.collection.add(ids=chunk_ids, documents=texts, metadatas=metadatas)
                indexed_msg = f"\n\n📥 已同步到 chroma papers collection({len(chunks)} chunks,可被 `semantic_search` 检索)"
        except Exception as e:
            logger.warning("[analyze_paper] 同步 chroma 失败(非致命,已吞): %s", e)
            indexed_msg = f"\n\n⚠ chroma 同步失败(不影响主流程): {e}"

    # 2026-09-02 协同 fix: analyze_paper 完成后, 把 LLM 提的 keywords / core_method / core_contrib
    # 写回 summary_items.db (推断 source), 并重嵌 paper_keywords
    # 让"已分析的论文"也能在 semantic_search_keywords / similar_papers 工具中被召回
    summary_updated = False
    try:
        from agent.tools.extract_keywords import (
            update_summary_after_analyze, infer_source_from_paper_id,
        )
        src = infer_source_from_paper_id(paper_id)
        kw_list = result.get("keywords", [])
        kw_str = ", ".join(str(k) for k in kw_list) if isinstance(kw_list, list) else str(kw_list)
        method = result.get("basic_method", "") or ""
        contrib = result.get("main_conclusion", "") or ""
        # analysis_id 暂时不填(没有 papers.id 的 rowid 暴露, 后续可加)
        summary_updated = update_summary_after_analyze(
            paper_id=paper_id,
            source=src,
            keywords=kw_str,
            core_method=method[:1000],
            core_contrib=contrib[:1000],
            in_papers_db=True,
            analysis_id=None,
        )
    except Exception as e:
        logger.warning("[analyze_paper] 协同更新 summary_item 失败(非致命): %s", e)

    return (
        f"✅ 论文分析完成\n\n"
        f"- 论文 ID: `{result['paper_id']}`\n"
        f"- 标题: {result.get('title', 'N/A')}\n"
        f"- 研究方向: {result.get('research_topic', 'N/A')}\n"
        f"- 关键词: {', '.join(result.get('keywords', []))}\n"
        f"{indexed_msg}\n"
        f"已存入知识库。可继续基于此论文与用户讨论。"
    )


# ============================================================
# LLM 工具：query_knowledge
# ============================================================
@register_tool(name="query_knowledge", is_streaming=False, cost_tier="L0")
def query_knowledge(keyword: str = "", topic: str = "", author: str = "") -> str:
    """
    在知识库中搜索匹配的文章。
    返回 Markdown 格式的论文列表。
    """
    conn = get_db_connection()
    c = conn.cursor()

    conditions = []
    params = []

    if keyword:
        conditions.append("p.rowid IN (SELECT rowid FROM papers_fts WHERE papers_fts MATCH ?)")
        params.append(keyword)
    if topic:
        conditions.append("p.research_topic LIKE ?")
        params.append(f"%{topic}%")
    if author:
        conditions.append("p.authors LIKE ?")
        params.append(f"%{author}%")

    base_sql = "SELECT p.* FROM papers p"
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)
    else:
        where_clause = ""
    sql = base_sql + where_clause
    rows = c.execute(sql, params).fetchall()
    conn.close()

    if not rows:
        return "未找到匹配的论文。"

    results = [deserialize_json_fields(dict(row)) for row in rows]

    lines = [f"找到 {len(results)} 篇匹配论文：\n"]
    for i, p in enumerate(results, 1):
        title = p.get("title", "(无标题)")
        paper_id = p.get("paper_id", "?")
        research_topic = p.get("research_topic", "")
        authors_list = p.get("authors", [])
        authors_str = ", ".join(authors_list[:3]) + (" 等" if len(authors_list) > 3 else "")
        keywords_list = p.get("keywords", [])
        keywords_str = ", ".join(keywords_list[:5])

        lines.append(f"{i}. **{title}**")
        lines.append(f"   - ID: `{paper_id}`")
        if authors_str:
            lines.append(f"   - 作者: {authors_str}")
        if research_topic:
            lines.append(f"   - 方向: {research_topic}")
        if keywords_str:
            lines.append(f"   - 关键词: {keywords_str}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# LLM 工具：scan_local_papers
# ============================================================
@register_tool(name="scan_local_papers", is_streaming=False)
def scan_local_papers() -> str:
    """扫描 PAPER_DIR，分析新出现的 PDF（增量更新）"""
    conn = get_db_connection()
    existing_ids = set(row['paper_id'] for row in conn.execute("SELECT paper_id FROM papers").fetchall())
    conn.close()

    new_papers = []
    skipped = 0
    for pdf_file in _config.PAPER_DIR.glob("*.pdf"):
        paper_id = pdf_file.stem
        if paper_id in existing_ids:
            skipped += 1
            continue
        result = analyze_paper(paper_id, pdf_path=str(pdf_file))
        if result.startswith("✅"):
            new_papers.append(paper_id)
        else:
            return f"扫描中止：分析 {paper_id} 失败\n{result}"

    if not new_papers and not skipped:
        return "扫描完成：本地目录中暂无 PDF 文件。"
    if not new_papers:
        return f"扫描完成：跳过已分析的 {skipped} 篇，没有新论文。"

    lines = [f"扫描完成：新增分析 {len(new_papers)} 篇（跳过 {skipped} 篇）：\n"]
    for pid in new_papers:
        lines.append(f"- `{pid}`")
    return "\n".join(lines)


# ============================================================
# LLM 工具：semantic_search（P1 新增 2026-09-02）
# 走 P0.5 section-aware chunk + P0 chroma 向量库
# 成本档 L0：纯本地语义检索，0 次 LLM 调用
# ============================================================
@register_tool(name="semantic_search", is_streaming=False, cost_tier="L0")
def semantic_search(
    query: str,
    top_k: int = 5,
    year: int = 0,
    journal: str = "",
    section: str = "",
    topic_keyword: str = "",
) -> str:
    """
    基于语义相似度检索知识库(P0 chroma + P0.5 section-aware chunk)。

    Args:
        query: 自然语言查询, e.g. "量子纠错码阈值"
        top_k: 返回多少条 chunk
        year: 过滤年份(0 = 不过滤)
        journal: 过滤期刊(arXiv / PRL / PRA 等,空 = 不过滤)
        section: 过滤 section 类别(introduction / method / result / discussion),
                 空 = 不过滤
        topic_keyword: 关键词(在 keywords 字段里模糊匹配),空 = 不过滤

    Returns:
        Markdown 格式的 chunk 列表,每条带论文标题 / section / 摘要片段 / distance
    """
    try:
        from agent.vector_db import VectorDB
    except Exception as e:
        return f"❌ 加载向量库失败: {e}。先跑 `scripts/migrate_papers_to_chroma_v2.py` 初始化。"

    # 构造 chroma where filter
    where: Dict[str, Any] = {}
    if year:
        where["year"] = int(year)
    if journal:
        where["journal"] = journal
    if section:
        # section 用 contains 操作(chroma 的 $contains)
        where["section"] = {"$contains": section}
    if topic_keyword:
        where["keywords"] = {"$contains": topic_keyword}

    db = VectorDB()
    try:
        # 多取一些(因不同 chunk 来自同一篇要去重)
        results = db.search(query, top_k=max(top_k * 3, 15), where=where or None)
    except Exception as e:
        logger.exception("semantic_search 失败: %s", e)
        return f"❌ 检索失败: {e}"

    if not results:
        return f"未找到匹配 '{query}' 的论文 chunk(可放宽过滤条件重试)。"

    # 同一论文多个 chunk 合并展示(选距离最小的)
    by_paper: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        pid = r["metadata"].get("paper_id", "?")
        by_paper.setdefault(pid, []).append(r)

    paper_list = []
    for pid, chunks in by_paper.items():
        # 选距离最小的 chunk 作为代表
        chunks.sort(key=lambda c: c.get("distance") or 999)
        top_chunk = chunks[0]
        m = top_chunk["metadata"]
        paper_list.append({
            "paper_id": pid,
            "title": m.get("title", "(无标题)"),
            "authors": m.get("authors", ""),
            "year": m.get("year", 0),
            "journal": m.get("journal", ""),
            "section": m.get("section", ""),
            "distance": top_chunk.get("distance"),
            "snippet": (top_chunk.get("document") or "")[:400],
            "n_chunks": len(chunks),
        })

    # 按 distance 排序
    paper_list.sort(key=lambda p: p["distance"] or 999)

    # 截到 top_k 个 paper
    paper_list = paper_list[:top_k]

    lines = [f"找到 {len(paper_list)} 篇相关论文(query='{query}'):\n"]
    for i, p in enumerate(paper_list, 1):
        lines.append(f"{i}. **{p['title']}**")
        if p["authors"]:
            au = p["authors"]
            if len(au) > 80:
                au = au[:80] + "..."
            lines.append(f"   - 作者: {au}")
        if p["year"] or p["journal"]:
            yj = []
            if p["year"]:
                yj.append(str(p["year"]))
            if p["journal"]:
                yj.append(p["journal"])
            lines.append(f"   - 期刊/年份: {' / '.join(yj)}")
        lines.append(f"   - ID: `{p['paper_id']}`")
        if p["section"]:
            lines.append(f"   - 命中章节: {p['section']} (共 {p['n_chunks']} chunks)")
        if p["distance"] is not None:
            lines.append(f"   - 相似度: {1 - p['distance']:.3f} (distance={p['distance']:.4f})")
        if p["snippet"]:
            lines.append(f"   - 摘要片段: {p['snippet']}...")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# P2.2: index_paper_to_chroma — 单篇索引到 chroma（纯本地, L0）
# 适用场景: 论文已 analyze 完, 补索引(或者没 analyze 但想快速入 chroma)
# ============================================================
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


# ============================================================
# P2.3: reindex_all — 全量重建 chroma papers collection（纯本地, L0）
# 适用场景: 切分策略升级(P0.5)、embedding model 切换、需要全量刷新
# 走 agent.pdf_chunker 切 + chroma 写入
# ============================================================
@register_tool(name="reindex_all", is_streaming=False, cost_tier="L0")
def reindex_all(limit: int = 0) -> str:
    """
    全量重建 chroma papers collection。
    读 papers.db 全部论文, 用 P0.5 chunker 切, 写入 chroma(幂等 delete+add)。
    限流:每 10 篇 sleep 0.5s 防 CPU 100%。
    Args:
        limit: 最多重建多少篇(0=全部,给测试/部分重建用)
    """
    import time
    import sqlite3
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


# ============================================================
# P2.4: sync_recent_to_chroma — 兜底最近未索引的论文（纯本地, L0）
# 适用场景: 抓文后 analyze 失败, summary_items 落库但 chroma 没数据
# 走 summary_items.db 查最近 N 天, 对还没入 chroma 的 paper 走 P0.5 chunker 切 + 入
# ============================================================
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
    from agent.tools.summary import list_summary_items
    from agent.pdf_chunker import chunk_by_section
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


# ============================================================
# P2.5: semantic_search_keywords — 关键词/摘要级语义检索（纯本地, L0）
# 走 chroma paper_keywords collection(跟 papers chunks 独立)
# 适用场景: "找和'拓扑码'贡献最像的论文" / 召回最新抓文 / 跨 collection 检索
# 跟 semantic_search 区别: 这个粒度 = 1 篇 1 条 record(summary_item 粒度),
#                          那个粒度 = 1 篇 N 条 chunks(section-aware)
# ============================================================
@register_tool(name="semantic_search_keywords", is_streaming=False, cost_tier="L0")
def semantic_search_keywords(
    query: str,
    top_k: int = 5,
    source: str = "",
    since_date: str = "",
    in_papers_db_only: bool = False,
) -> str:
    """
    基于语义相似度检索 summary_items 库(走 P2.5 paper_keywords collection)。

    适用场景(跟 P1 semantic_search 互补):
    - "找和'拓扑码'贡献最像的论文" → 召回粒度=论文级, 不分 chunk
    - 跨 collection 检索: 抓文但未 analyze 的论文也能召回
    - 按 source / since_date 过滤

    Args:
        query: 自然语言查询, e.g. "topological code fault tolerance"
        top_k: 返回多少条 record(每条 = 1 篇 summary_item)
        source: 按 source 过滤("arxiv_daily" / "aps_daily" / "reindex" 等,空=不过滤)
        since_date: 按 summary_date >= since_date 过滤(ISO date, 空=不过滤)
        in_papers_db_only: 只返回已精细分析过的(summary_item.in_papers_db=True)

    Returns:
        Markdown 格式的 summary_item 列表, 每条带 paper_id / title / source / date / distance
    """
    try:
        from agent.vector_db import get_chroma_client, _make_embedding_function
    except Exception as e:
        return f"❌ 加载向量库失败: {e}。先跑 `scripts/summary_items_to_chroma.py` 初始化。"

    # 拿 collection(paper_keywords 独立于 papers)
    try:
        client = get_chroma_client()
        fn = _make_embedding_function()
        kwargs = {"name": "paper_keywords", "metadata": {"hnsw:space": "cosine"}}
        if fn is not None:
            kwargs["embedding_function"] = fn
        collection = client.get_or_create_collection(**kwargs)
        if collection.count() == 0:
            return (
                "❌ paper_keywords collection 是空的。"
                "先跑 `scripts/summary_items_to_chroma.py` 把 summary_item 嵌入, 或抓文触发自动嵌入。"
            )
    except Exception as e:
        logger.exception("semantic_search_keywords 拿 collection 失败: %s", e)
        return f"❌ 拿 collection 失败: {e}"

    # 构造 chroma where filter
    where: Dict[str, Any] = {}
    if source:
        where["source"] = source
    if since_date:
        where["summary_date"] = {"$gte": since_date}
    if in_papers_db_only:
        where["in_papers_db"] = True

    # 检索
    query_kwargs = {"query_texts": [query], "n_results": min(top_k, collection.count())}
    if where:
        query_kwargs["where"] = where
    try:
        res = collection.query(**query_kwargs)
    except Exception as e:
        # P0 紧急: chroma where filter 类型错误(用户实测 $gte 拿到 string 失败)
        # 降级: 去掉 where 重试 (忽略 source/since_date/in_papers_db 过滤)
        logger.warning("semantic_search_keywords 带 where 失败 (%s), 降级去掉过滤重试", e)
        try:
            query_kwargs_no_where = {"query_texts": [query], "n_results": min(top_k, collection.count())}
            res = collection.query(**query_kwargs_no_where)
            # 报告里说明降级
            res_note = f"⚠️ 过滤条件 {where} 类型不兼容 chroma, 已降级为无过滤检索\n\n"
        except Exception as e2:
            logger.exception("semantic_search_keywords 降级也失败: %s", e2)
            return f"❌ 检索失败: {e}; 降级也失败: {e2}"
    else:
        res_note = ""

    if not res or not res.get("ids") or not res["ids"][0]:
        return f"未找到匹配 '{query}' 的 summary_item(可放宽过滤条件重试)。"

    # 9-09 user 拍板: 只返总数, 不贴 paper_id / title / snippet / 相似度 等具体结果
    # 完整信息: 内部 res 变量仍有 (调试时 print(res) 可见), 但不返给 LLM/用户
    # LLM 需要详情时, 调 `query_knowledge(paper_id)` 按 id 单篇查; 不在 search 工具里展开
    count = len(res["ids"][0])
    return f"{res_note}找到 {count} 条相关 summary_item(query='{query}')。"


# ============================================================
# P3: similar_papers — 找相似论文（字段匹配优先 + chroma 3-5 补充, 纯本地 L0）
# 设计(2026-09-02 用户拍板):
#   字段匹配 top N(同 keywords / 同 authors / 同 research_topic 打分)
#   + chroma paper_keywords 召回 M 篇(语义近邻,补多样性)
#   合并去重,字段匹配优先,语义补充后排
#
# 为什么字段匹配优先:
#   - 用户主诉"找相似"= 找"同方向同方法"的论文
#   - 字段(同 keywords / 同 authors / 同 topic)是高信号证据,比纯语义更精准
#   - 语义召回作为多样性补充(避免"全是一模一样方向的论文")
#
# 字段打分(总分越高越相关):
#   - keywords 重合 N 个: +N(精确匹配,高信号)
#   - authors 重合 N 个: +0.5*N(中等信号,合著者可能撞方向)
#   - research_topic 子串命中: +0.3(粗略,避免完全无关)
# ============================================================
@register_tool(name="similar_papers", is_streaming=False, cost_tier="L0")
def similar_papers(
    paper_id: str,
    top_k: int = 10,
    n_field: int = 5,
    n_semantic: int = 4,
) -> str:
    """
    找相似论文: 字段匹配优先 + chroma 关键词召回补充(纯本地, 无 LLM)。

    字段匹配(同 keywords / 同 authors / 同 research_topic 打分排序):
      - keywords 重合 1 个 = 1.0 分(高信号)
      - authors 重合 1 个 = 0.5 分(中等信号)
      - research_topic 子串命中 = 0.3 分(粗略)
    chroma 召回(走 paper_keywords collection 找语义近邻, 补多样性):
      - 用 source paper 的 title + keywords 作为 query
      - 拿 top n_semantic 篇
    合并去重, 字段匹配优先, 语义补充后排, 截到 top_k。

    Args:
        paper_id: 源论文 ID
        top_k: 返回多少条
        n_field: 字段匹配拿多少条(0 = 全用语义)
        n_semantic: chroma 拿多少条(0 = 全用字段)

    Returns:
        Markdown 格式的相似论文列表, 每条带 paper_id / title / 命中字段 / 相似度
    """
    import json as _json
    db_path = Path(_config.KNOWLEDGE_DIR) / "papers.db"
    if not db_path.exists():
        return f"❌ papers.db 不存在: {db_path}"

    # 1. 读 source paper
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    src_row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if not src_row:
        conn.close()
        return f"❌ 论文 {paper_id} 不在 papers.db, 先用 analyze_paper 分析"
    src = dict(src_row)
    src_kw = _safe_json_list(src.get("keywords", ""))
    src_authors = _safe_json_list(src.get("authors", ""))
    src_topic = (src.get("research_topic") or "").strip().lower()
    src_title = src.get("title", "")
    conn.close()

    # 2. 字段匹配: 拿所有 papers 计算分数
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    all_rows = conn.execute("SELECT paper_id, title, keywords, authors, research_topic FROM papers WHERE paper_id != ?", (paper_id,)).fetchall()
    conn.close()

    candidates = []  # (score, paper_id, title, matched_fields, breakdown)
    for r in all_rows:
        d = dict(r)
        if d["paper_id"] == paper_id:
            continue
        pid2 = d["paper_id"]
        kw2 = _safe_json_list(d.get("keywords", ""))
        au2 = _safe_json_list(d.get("authors", ""))
        topic2 = (d.get("research_topic") or "").strip().lower()

        score = 0.0
        matched = []
        # keywords 重合
        kw_overlap = set(k.lower() for k in src_kw) & set(k.lower() for k in kw2)
        if kw_overlap:
            kw_score = float(len(kw_overlap))
            score += kw_score
            matched.append(f"keywords×{len(kw_overlap)}({','.join(list(kw_overlap)[:3])})")
        # authors 重合
        au_overlap = set(src_authors) & set(au2)
        if au_overlap:
            score += 0.5 * len(au_overlap)
            matched.append(f"authors×{len(au_overlap)}")
        # research_topic 子串命中
        topic_score = 0.0
        if src_topic and topic2:
            if src_topic == topic2:
                topic_score = 1.0
            elif src_topic in topic2 or topic2 in src_topic:
                topic_score = 0.3
        if topic_score > 0:
            score += topic_score
            matched.append(f"topic({topic_score})")
        if score > 0:
            candidates.append({
                "paper_id": pid2,
                "title": d.get("title", ""),
                "score": score,
                "matched": matched,
                "source": "field",
            })

    # 字段匹配排序取 top n_field
    candidates.sort(key=lambda c: -c["score"])
    field_top = candidates[:n_field]
    field_pids = {c["paper_id"] for c in field_top}

    # 3. chroma 召回(用 source paper 的 title + keywords 作为 query)
    semantic_top = []
    if n_semantic > 0:
        try:
            from agent.vector_db import get_chroma_client, _make_embedding_function
            client = get_chroma_client()
            fn = _make_embedding_function()
            kwargs = {"name": "paper_keywords", "metadata": {"hnsw:space": "cosine"}}
            if fn is not None:
                kwargs["embedding_function"] = fn
            coll = client.get_or_create_collection(**kwargs)
            if coll.count() > 0:
                # 拼 query: title + keywords
                q_parts = [src_title]
                if src_kw:
                    q_parts.append(", ".join(src_kw[:8]))
                query_text = " | ".join(q for q in q_parts if q)
                # 多取一些,后面去重
                n_fetch = n_semantic + len(field_pids) + 5
                res = coll.query(query_texts=[query_text], n_results=min(n_fetch, coll.count()))
                if res and res.get("ids") and res["ids"][0]:
                    for i, cid in enumerate(res["ids"][0]):
                        meta = res["metadatas"][0][i] if res.get("metadatas") else {}
                        dist = res["distances"][0][i] if res.get("distances") else None
                        pid2 = meta.get("paper_id", "?")
                        if pid2 == paper_id or pid2 in field_pids:
                            continue  # 去重
                        semantic_top.append({
                            "paper_id": pid2,
                            "title": meta.get("title", ""),
                            "score": 1 - (dist or 1),  # 转相似度
                            "matched": [f"semantic({1 - (dist or 1):.3f})"],
                            "source": "semantic",
                        })
        except Exception as e:
            logger.warning("similar_papers chroma 召回失败(非致命): %s", e)
    semantic_top = semantic_top[:n_semantic]
    semantic_pids = {c["paper_id"] for c in semantic_top}

    # 4. 合并: 字段匹配优先, 语义补充
    merged = field_top + [c for c in semantic_top if c["paper_id"] not in field_pids]
    merged = merged[:top_k]

    if not merged:
        return f"未找到 {paper_id} 的相似论文(同 keywords/authors/topic 没有命中, chroma 也无近邻)。"

    # 5. 渲染 Markdown
    lines = [f"找到 {len(merged)} 篇与 `{paper_id}` 相似的论文:\n"]
    lines.append(f"**源论文**: {src_title[:80]}")
    if src_kw:
        lines.append(f"  - 关键词: {', '.join(src_kw[:5])}")
    lines.append("")
    for i, c in enumerate(merged, 1):
        src_tag = "🔍 字段" if c["source"] == "field" else "🧬 语义"
        lines.append(f"{i}. [{src_tag}] **{c['title'][:80]}**")
        lines.append(f"   - ID: `{c['paper_id']}`")
        lines.append(f"   - 命中: {', '.join(c['matched'])}")
        if c["source"] == "field":
            lines.append(f"   - 字段匹配分: {c['score']:.2f}")
        else:
            lines.append(f"   - 语义相似度: {c['score']:.3f}")
        lines.append("")
    return "\n".join(lines)


# ============================================================
# 内部 helper: 安全解析 JSON list 字符串
# ============================================================
def _safe_json_list(s) -> list:
    """解析 papers.db 的 keywords/authors 字段(可能是 JSON list 字符串)

    2026-09-02 P3 修:用顶层 import 的 `json` 名字(之前误用 `_json` 但只在 similar_papers
    函数体局部 import,导致 _safe_json_list 抛 NameError 被静默 except,fallback 返 [s])
    """
    if not s:
        return []
    if isinstance(s, list):
        return [str(x) for x in s]
    s = str(s).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            return [str(x) for x in json.loads(s)]
        except Exception:
            pass
    return [s]


# ============================================================
# 内部使用：migrate_json_to_db（一次性数据迁移）
# ============================================================
def migrate_json_to_db():
    """将 KNOWLEDGE_DIR 下所有 .json 文件（除了 index.json）导入数据库"""
    for json_file in _config.KNOWLEDGE_DIR.glob("*.json"):
        if json_file.name == "index.json":
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "paper_id" not in data:
            data["paper_id"] = json_file.stem
        if "analyzed_at" not in data:
            data["analyzed_at"] = datetime.now().isoformat()
        save_paper_record(data)
