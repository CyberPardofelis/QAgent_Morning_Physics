"""
knowledge._query — 拆出自原 knowledge.py (9-20 体检重构)

包含函数: query_knowledge, semantic_search, semantic_search_keywords
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

from agent.tools.config import KNOWLEDGE_DIR, PAPER_DIR  # noqa: F401
from agent.tools.database import deserialize_json_fields, get_db_connection
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)
@register_tool(name="query_knowledge", is_streaming=False, cost_tier="L0")
def query_knowledge(keyword: str = "", topic: str = "", author: str = "") -> str:
    """
    在知识库中搜索匹配的文章。
    返回 Markdown 格式的论文列表。

    9-18 加: 控制台 print + 异常友好化.
    之前: FTS5 query 写错会抛 sqlite3.OperationalError, 直接 stacktrace 给用户看.
    现在: 4 类异常分别处理 (DB 不存在 / 表 schema 旧 / FTS5 query 错 / 其他).
    """
    short_kw = (keyword[:50] + "...") if len(keyword) > 50 else keyword
    short_topic = (topic[:50] + "...") if len(topic) > 50 else topic
    short_author = (author[:50] + "...") if len(author) > 50 else author
    print(
        f"[query_knowledge] start  keyword={short_kw!r}  topic={short_topic!r}  author={short_author!r}",
        flush=True,
    )

    if not (keyword or topic or author):
        print("[query_knowledge] ⚠ 三个参数都为空, 全表扫描", flush=True)
        return (
            "⚠ query_knowledge: keyword / topic / author 至少传一个\n"
            "- 三个都空 = 全表扫描, 慢且容易超长\n"
            "- 推荐: keyword (FTS5 全文匹配, 最快) 或 topic (research_topic LIKE) 或 author (authors LIKE)"
        )

    # 9-18 加: 容错包住 DB 操作, 4 类异常分别给具体建议
    try:
        conn = get_db_connection()
    except sqlite3.DatabaseError as e:
        msg = str(e)
        if "no such table" in msg or "unable to open" in msg:
            print(f"[query_knowledge] ❌ 数据库文件缺失/损坏: {e}", flush=True)
            return (
                f"❌ query_knowledge 失败: 知识库 DB 缺失或损坏\n"
                f"- 详情: {e}\n"
                f"- 建议: 1) 先跑 scan_local_papers 初始化库  2) 重新分析几篇论文"
            )
        print(f"[query_knowledge] ❌ 数据库连接失败: {e}", flush=True)
        return f"❌ query_knowledge 失败: DB 连接错 ({type(e).__name__}): {e}"

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

    # 9-18 加: FTS5 query 语法错误单独捕获
    # (FTS5 对特殊字符很敏感: 含 : / " / ^ 等会抛 OperationalError "fts5: syntax error")
    try:
        rows = c.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        msg = str(e)
        conn.close()
        print(f"[query_knowledge] ❌ SQL 执行失败: {msg}", flush=True)
        if "fts5" in msg.lower() or "syntax error" in msg.lower():
            return (
                f"❌ query_knowledge 失败: FTS5 query 语法错\n"
                f"- keyword={keyword!r} 触发了 FTS5 解析错误\n"
                f"- 详情: {msg}\n"
                f"- 建议: 1) 简化 keyword, 去掉 : / \" / ^ / ( / ) / + / - / \"  "
                f"2) 改用 topic 或 author 参数 (LIKE 模糊匹配, 不挑字符)"
            )
        if "no such column" in msg or "no such table" in msg:
            return (
                f"❌ query_knowledge 失败: 表结构不匹配\n"
                f"- 详情: {msg}\n"
                f"- 建议: 重新初始化知识库 (跑 scan_local_papers 或 analyze_paper)"
            )
        return f"❌ query_knowledge 失败 (SQL): {msg}"
    except Exception as e:
        conn.close()
        print(f"[query_knowledge] ❌ 查询异常: {type(e).__name__}: {e}", flush=True)
        return f"❌ query_knowledge 失败 ({type(e).__name__}): {e}"

    conn.close()

    if not rows:
        # 9-18 加: 0 结果时给具体建议, 不要直接 "未找到匹配的论文"
        print(
            f"[query_knowledge] ⚠ 0 results  keyword={short_kw!r}  topic={short_topic!r}  author={short_author!r}",
            flush=True,
        )
        return (
            f"未找到匹配论文 (keyword={keyword!r}, topic={topic!r}, author={author!r})\n"
            f"- 建议: 1) 放宽关键词 (例: 只传一个参数)  2) 用 `query_knowledge(keyword=\"quantum\")` 测试库是否为空  "
            f"3) 没数据先跑 scan_local_papers / analyze_paper"
        )

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

    print(f"[query_knowledge] done  返回 {len(results)} 篇", flush=True)
    return "\n".join(lines)


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
    where: dict[str, Any] = {}
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
    by_paper: dict[str, list[dict[str, Any]]] = {}
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
        from agent.vector_db import _make_embedding_function, get_chroma_client
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
    where: dict[str, Any] = {}
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
