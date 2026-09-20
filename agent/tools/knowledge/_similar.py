"""
knowledge._similar — 拆出自原 knowledge.py (9-20 体检重构)

包含函数: similar_papers
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from agent.tools import config as _config
from agent.tools.config import KNOWLEDGE_DIR, PAPER_DIR  # noqa: F401
from agent.tools.registry import register_tool

from ._utils import _safe_json_list

logger = logging.getLogger(__name__)
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
            from agent.vector_db import _make_embedding_function, get_chroma_client
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
