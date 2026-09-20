"""
knowledge._analyze — 拆出自原 knowledge.py (9-20 体检重构)

包含函数: analyze_paper
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime

import arxiv

from agent.tools.config import KNOWLEDGE_DIR, PAPER_DIR  # noqa: F401
from agent.tools.database import save_paper_record
from agent.tools.llm import _call_llm
from agent.tools.pdf_utils import extract_text_from_pdf
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)
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
            infer_source_from_paper_id,
            update_summary_after_analyze,
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
