"""
agent.tools.surveys — 综述库 CRUD(2026-09-02 P6 准备 + UI 上线)

职责:
- 主题(theme) + 章节(section) 二级树状结构
- 章节正文 markdown 内容(查看 tab 渲染 / 编辑 tab 纯文本)
- P6 综述生成会自动调 create_theme + create_section
- UI 现在就能用:手动建主题/章节/编辑

ROADMAP_NEXT §9.2 schema 简化版(等 P6 实施时再补 source_survey / superseded_by 等字段):
- survey_theme(id, name, description, created_at)
- survey_section(id, theme_id, section_type, title, content_md, paper_ids, order_idx, created_at, updated_at)

设计:
- 同步 CRUD(操作快,UI 用同步调用)
- 主题名 + 章节标题 唯一性不强制(允许同主题下多个"intro" 章节)
- 删除主题级联删章节
- 编辑正文:用 content_md 字段(markdown)
"""
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from agent.tools.config import get_min_pool_papers

logger = logging.getLogger(__name__)


# ============================================================
# Schema (ROADMAP_NEXT §9.2 简化版)
# ============================================================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS survey_theme (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    description  TEXT DEFAULT '',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS survey_section (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    theme_id     INTEGER NOT NULL,
    section_type TEXT NOT NULL DEFAULT 'intro',  -- intro / progress / method_compare / gap / outlook / custom
    title        TEXT NOT NULL,
    content_md   TEXT DEFAULT '',
    paper_ids    TEXT DEFAULT '',                -- 逗号分隔
    order_idx    INTEGER DEFAULT 0,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (theme_id) REFERENCES survey_theme(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_survey_section_theme ON survey_section(theme_id, order_idx);
"""


def init_schema() -> None:
    """建表(幂等,可多次调用)"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.executescript(SCHEMA_SQL)
        # 启用外键(级联删除)
        conn.execute("PRAGMA foreign_keys = ON")
    logger.info("[surveys] surveys.db schema 初始化完成")


# ============================================================
# 初始化(模块 import 时建表)
# ============================================================
try:
    init_schema()
except Exception as e:
    logger.warning("[surveys] init_schema 失败(非致命,首次 save 时会重试): %s", e)


# ============================================================
# 主题 CRUD
# ============================================================
def create_theme(name: str, description: str = "") -> int:
    """新建主题,返新 id"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        cur = conn.execute(
            "INSERT INTO survey_theme (name, description) VALUES (?, ?)",
            (name, description),
        )
        return cur.lastrowid


def list_themes() -> List[Dict[str, Any]]:
    """列所有主题"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM survey_theme ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_theme(theme_id: int) -> Optional[Dict[str, Any]]:
    """按 id 拿主题"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM survey_theme WHERE id = ?", (theme_id,)).fetchone()
    return dict(row) if row else None


def update_theme(theme_id: int, name: str = "", description: str = "") -> None:
    """更新主题(只更新非空字段)"""
    from agent.infra.db import connect
    sets, params = [], []
    if name:
        sets.append("name = ?")
        params.append(name)
    if description:
        sets.append("description = ?")
        params.append(description)
    if not sets:
        return
    sets.append("updated_at = CURRENT_TIMESTAMP")
    params.append(theme_id)
    with connect("surveys") as conn:
        conn.execute(f"UPDATE survey_theme SET {', '.join(sets)} WHERE id = ?", params)


def delete_theme(theme_id: int) -> int:
    """删除主题(级联删所有章节),返删除章节数"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        n_sections = conn.execute(
            "SELECT COUNT(*) FROM survey_section WHERE theme_id = ?", (theme_id,),
        ).fetchone()[0]
        conn.execute("DELETE FROM survey_theme WHERE id = ?", (theme_id,))
    return n_sections


# ============================================================
# 章节 CRUD
# ============================================================
def create_section(
    theme_id: int,
    title: str,
    section_type: str = "intro",
    content_md: str = "",
    paper_ids: str = "",
) -> int:
    """新建章节,自动算 order_idx = max + 1"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        max_idx = conn.execute(
            "SELECT COALESCE(MAX(order_idx), -1) FROM survey_section WHERE theme_id = ?",
            (theme_id,),
        ).fetchone()[0]
        cur = conn.execute(
            """INSERT INTO survey_section (theme_id, section_type, title, content_md, paper_ids, order_idx)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (theme_id, section_type, title, content_md, paper_ids, max_idx + 1),
        )
        # 同步更新主题的 updated_at
        conn.execute(
            "UPDATE survey_theme SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (theme_id,),
        )
        return cur.lastrowid


def list_sections(theme_id: int) -> List[Dict[str, Any]]:
    """列某主题下的所有章节(按 order_idx 排序)"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM survey_section WHERE theme_id = ? ORDER BY order_idx",
            (theme_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_section(section_id: int) -> Optional[Dict[str, Any]]:
    """按 id 拿章节"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM survey_section WHERE id = ?", (section_id,)).fetchone()
    return dict(row) if row else None


def update_section(
    section_id: int,
    title: str = "",
    section_type: str = "",
    content_md: Optional[str] = None,  # 2026-09-02 修:用 None 表示"不传",空字符串表示"清空"
    paper_ids: str = "",
) -> None:
    """更新章节(只更新传入的字段,None 表示跳过)"""
    from agent.infra.db import connect
    sets, params = [], []
    if title:
        sets.append("title = ?")
        params.append(title)
    if section_type:
        sets.append("section_type = ?")
        params.append(section_type)
    if content_md is not None:  # None=不传, ""=清空
        sets.append("content_md = ?")
        params.append(content_md)
    if paper_ids:
        sets.append("paper_ids = ?")
        params.append(paper_ids)
    if not sets:
        return
    sets.append("updated_at = CURRENT_TIMESTAMP")
    params.append(section_id)
    with connect("surveys") as conn:
        conn.execute(f"UPDATE survey_section SET {', '.join(sets)} WHERE id = ?", params)
        # 同步更新所属主题的 updated_at
        conn.execute(
            """UPDATE survey_theme SET updated_at = CURRENT_TIMESTAMP
               WHERE id = (SELECT theme_id FROM survey_section WHERE id = ?)""",
            (section_id,),
        )


def delete_section(section_id: int) -> None:
    """删除章节"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.execute("DELETE FROM survey_section WHERE id = ?", (section_id,))


def reorder_section(section_id: int, new_order: int) -> None:
    """调整章节顺序"""
    from agent.infra.db import connect
    with connect("surveys") as conn:
        conn.execute(
            "UPDATE survey_section SET order_idx = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_order, section_id),
        )


# ============================================================
# 健康检查
# ============================================================
def health_check() -> Dict[str, Any]:
    from agent.infra.db import path_of
    p = path_of("surveys")
    info = {
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else 0,
    }
    if p.exists():
        info["n_themes"] = len(list_themes())
        # 总章节数
        from agent.infra.db import connect
        with connect("surveys") as conn:
            info["n_sections"] = conn.execute("SELECT COUNT(*) FROM survey_section").fetchone()[0]
    else:
        info["n_themes"] = 0
        info["n_sections"] = 0
    return info


# ============================================================
# P6.2 综述生成(LLM 工具)— 2026-09-05
# ============================================================
# 6 段 pipeline: 候选池 → fact 提取 → 大纲 → 分章节生成 → 落库 → audit
# 共享 prompt 模板 agent.prompts_survey + 工作区 agent.infra.survey_workspace

import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator

from agent.tools.registry import register_tool
from agent.prompts_survey import (
    build_fact_prompt,
    build_outline_prompt,
    build_section_prompt,
    SECTION_REF_QUOTAS,            # 9-08 修: action_write_outline 用它校验 section_type
)
from agent.infra.survey_workspace import SurveyWorkspace, hash_topic


# 抽 paper_id 的 regex(3 种格式, plan §3.1 + P6.2 LLM 实际行为)
# 1) 简洁:  [2609.10003] / [0607734]              ← plan §3.1 prompt 强约束
# 2) arXiv:  arXiv:2609.10003  / arXiv 0607734
# 3) 链接:   papers/2609.10003.md  (从行内 markdown 链接 url 抽)
# 注意: plan §3.1 反例里 LLM 偶尔只写完整链接不写简洁形式, 必须 3 种都识别
# arXiv 编号两种格式: 新版 XXXX.XXXXX (如 2401.12345) / 旧版 XXXXXXX (如 0607734, 2007 年前)
_PAPER_ID_SIMPLE_RE = re.compile(r"\[(\d{4,5}\.\d{4,5}|\d{7})\]")
_PAPER_ID_ARXIV_RE = re.compile(r"arXiv[:\s]+(\d{4,5}\.\d{4,5}|\d{7})")
_PAPER_ID_PAPERS_MD_RE = re.compile(r"papers/(\d{4,5}\.\d{4,5}|\d{7})\.md")


def _safe_json_loads_array(text: str) -> Optional[list]:
    """
    容错 parse JSON 数组(LLM 经常在 closing `"}]` 前就停, P6.2 实测多次踩)。
    策略(逐级):
      1. 直接 json.loads
      2. 补 '"}]' / ']' 等简单收尾
      3. truncate 到最后一个完整 `{}` 之前, 补 `]`
      4. regex 抽所有单层完整 {...} 对象重建数组
    返 None 表示解析失败(不是空数组)
    """
    if not text:
        return None
    s = text.strip()
    # 1. 直接
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):  # 偶尔 LLM 返单 object 不返 array
            return [v]
    except (json.JSONDecodeError, SyntaxError):
        pass
    # 2. 补 "}]
    for tail in ['"}]', '"]', '"', '}', ']']:
        try:
            v = json.loads(s.rstrip(",\n ") + tail)
            if isinstance(v, list):
                return v
        except (json.JSONDecodeError, SyntaxError):
            pass
    # 3. truncate 到最后一个 } (P6.2 4b 案例: 末尾是 "y" 缺 }, 截掉)
    last_brace = s.rfind('}')
    if last_brace > 0:
        try:
            v = json.loads(s[: last_brace + 1] + ']')
            if isinstance(v, list):
                return v
        except (json.JSONDecodeError, SyntaxError):
            pass
    # 4. regex 抽所有单层完整 {...} 对象(不允许嵌套 {})
    obj_re = re.compile(r'\{[^{}]*\}', re.DOTALL)
    objs = obj_re.findall(s)
    if objs:
        try:
            v = json.loads("[" + ",".join(objs) + "]")
            if isinstance(v, list):
                return v
        except (json.JSONDecodeError, SyntaxError):
            pass
    return None


def _extract_paper_ids(text: str) -> List[str]:
    """从 markdown 文本抽所有 paper_id 引用,3 种格式都识别,去重保序"""
    if not text:
        return []
    seen = set()
    out = []
    # 顺序很重要: 简洁形式先(用户期望), 然后 arXiv: 形式, 然后链接 url
    for pattern in (_PAPER_ID_SIMPLE_RE, _PAPER_ID_ARXIV_RE, _PAPER_ID_PAPERS_MD_RE):
        for m in pattern.finditer(text):
            pid = m.group(1)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(pid)
    return out


def _audit_section_paper_ids(content_md: str, pool_ids: set) -> Dict[str, List[str]]:
    """候选池内/外分桶(plan §3.2 覆盖度审计核心)"""
    cites = _extract_paper_ids(content_md or "")
    in_pool = []
    out_pool = []
    for c in cites:
        if c in pool_ids:
            in_pool.append(c)
        else:
            out_pool.append(c)
    return {
        "in_pool": sorted(set(in_pool)),
        "out_of_pool": sorted(set(out_pool)),
        "total_cites": len(cites),
    }


def _build_candidate_pool(topic: str, max_pool: int = 50) -> List[Dict[str, Any]]:
    """
    Phase A: 候选池构建 (9-08 多源召回版)

    三路召回, union + 去重:
      1. summary_items.db 关键词匹配 (title/abstract/core_method/core_contrib/keywords/outlook)
         - 关键词从 topic 切词 + 标题/摘要 token
         - 9-08 修: 不再过度依赖 keywords 字段 (很多核心论文 keywords 缺失)
      2. chroma 向量检索 (如果 vector_db 可用)
         - 9-08 新增: 走语义相似度, 召回 summary_items 关键词匹配漏掉的论文
      3. arxiv API 兜底
         - 候选池 < 最小阈值 时, 拉 arxiv 补足 (9-08 user 拍板 抽 env QAGENT_SURVEY_MIN_POOL, 默认 30)

    9-08 user 拍板: 让 survey 工具能查看到"摘要数据库"更多内容, 不要只依赖 keywords 字段
    """
    # 候选池最小阈值 (9-08 抽 env, 默认 30, 充裕 = min+10)
    _MIN_POOL = get_min_pool_papers()
    from agent.tools.summary import list_summary_items

    # 9-08 修: 拆成多路召回, 每路各取 top, 然后 union + 去重
    seen_ids: set = set()
    pool_dict: Dict[str, Dict[str, Any]] = {}  # paper_id -> pool entry

    def add_pool_entry(entry: Dict[str, Any]) -> None:
        """union + 去重"""
        pid = entry.get("paper_id", "")
        if not pid or pid in seen_ids:
            return
        seen_ids.add(pid)
        pool_dict[pid] = entry

    # 关键词切分 (中文 + 英文)
    keywords = [kw.strip() for kw in re.split(r"[\s,，、。;；:：]+", topic) if len(kw.strip()) >= 2]
    if not keywords:
        keywords = [topic]

    # ========== 1. summary_items 关键词匹配 ==========
    try:
        raw = list_summary_items(limit=max_pool * 4)  # 多拿点
        if raw:
            def kw_score(item: Dict[str, Any]) -> int:
                # 9-08 修: 关键词字段权重降低, title + abstract 权重高
                score = 0
                for field, w in [
                    ("title", 3),
                    ("abstract", 2),
                    ("core_method", 2),
                    ("core_contrib", 2),
                    ("keywords", 1),       # 9-08 降权 (用户实测很多核心论文 keywords 缺失)
                    ("outlook", 1),
                ]:
                    v = (item.get(field) or "").lower()
                    for kw in keywords:
                        if kw.lower() in v:
                            score += w
                return score

            scored = [(kw_score(it), it) for it in raw]
            scored = [(s, it) for s, it in scored if s > 0]
            scored.sort(key=lambda x: -x[0])
            for score, it in scored[:max_pool]:
                add_pool_entry({
                    "paper_id": it.get("paper_id", ""),
                    "title": it.get("title", ""),
                    "year": (it.get("summary_date") or "")[:4] if it.get("summary_date") else "",
                    "journal": it.get("source", ""),
                    "core_method": it.get("core_method", ""),
                    "core_contrib": it.get("core_contrib", ""),
                    "abstract": it.get("abstract", ""),
                    "outlook": it.get("outlook", ""),
                    "keywords": it.get("keywords", ""),
                    "_match_score": score,
                    "_source": "summary_items_keyword",
                })
    except Exception as e:
        logger.warning("[_build_candidate_pool] summary_items 召回失败: %s", e)

    # ========== 2. chroma 向量检索 (9-08 新增) ==========
    # 9-08 user 拍板: 修复底层索引, 让 survey 能查看到更多数据库信息
    # chroma 索引里的论文 metadata 通常比 summary_items 更全, 走语义相似度召回
    try:
        from agent.vector_db import get_vector_db
        vdb = get_vector_db()
        if vdb and vdb.count() > 0:
            # 用 topic 调, 多取点 (top_k=min+10), 让 union 后能到 最小阈值+
            chroma_results = vdb.search(topic, top_k=min(_MIN_POOL + 10, max_pool))
            for r in chroma_results:
                meta = r.get("metadata", {}) or {}
                pid = meta.get("paper_id") or r.get("id", "")
                if not pid:
                    continue
                distance = r.get("distance", 1.0) or 1.0
                add_pool_entry({
                    "paper_id": pid,
                    "title": meta.get("title", ""),
                    "year": str(meta.get("year", "")),
                    "journal": meta.get("journal") or meta.get("source", ""),
                    "core_method": meta.get("core_method", ""),
                    "core_contrib": meta.get("core_contrib", ""),
                    "abstract": (r.get("document") or "")[:1000],  # chroma 文档截短
                    "outlook": meta.get("outlook", ""),
                    "keywords": meta.get("keywords", ""),
                    "_match_score": int((1 - distance) * 100) if distance is not None else 0,
                    "_source": "chroma_vector",
                })
    except Exception as e:
        # chroma 不可用是正常的 (项目可选), 不影响主流程
        logger.debug("[_build_candidate_pool] chroma 召回跳过: %s", e)

    # ========== 3. arxiv API 兜底 (9-08: 最小阈值走 env QAGENT_SURVEY_MIN_POOL, 默认 30) ==========
    if len(pool_dict) < _MIN_POOL:
        try:
            import arxiv
            need = max(_MIN_POOL, max_pool) - len(pool_dict)
            search = arxiv.Search(
                query=topic,
                max_results=min(need, 20),  # 限 20 篇避免太慢
                sort_by=arxiv.SortCriterion.Relevance,
            )
            client = arxiv.Client()
            for r in client.results(search):
                pid = r.entry_id.split("/")[-1].split("v")[0]
                if pid in seen_ids:
                    continue
                add_pool_entry({
                    "paper_id": pid,
                    "title": r.title or "",
                    "year": str(r.published.year) if r.published else "",
                    "journal": "arxiv",
                    "core_method": "",
                    "core_contrib": (r.summary or "")[:200],
                    "abstract": r.summary or "",
                    "outlook": "",
                    "keywords": ", ".join([a.text for a in (r.authors or [])[:3]]),
                    "_match_score": 0,
                    "_source": "arxiv_api_fallback",
                })
                if len(pool_dict) >= max(_MIN_POOL, max_pool):
                    break
        except Exception as e:
            logger.warning("[_build_candidate_pool] arxiv 兜底失败: %s", e)

    # 排序: match_score 降序, 同分按 source 优先级 (chroma > summary_items > arxiv)
    source_priority = {"chroma_vector": 0, "summary_items_keyword": 1, "arxiv_api_fallback": 2}
    pool = sorted(
        pool_dict.values(),
        key=lambda x: (
            -x.get("_match_score", 0),
            source_priority.get(x.get("_source", ""), 9),
        ),
    )
    # 限 max_pool
    return pool[:max_pool]


@register_tool(name="survey_generate", is_streaming=True, cost_tier="L2")
def survey_generate(
    topic: str,
    max_steps: int = 20,
    max_pool: int = 50,                # 9-08 17:00 加: 候选池上限, 默认 50 (LLM 可显式传更大值)
    dry_run: bool = False,
) -> Iterator[str]:
    """
    生成综述 (P6.5 ReAct 改写) — L2 档, 流式 yield 进度 + 章节内容.

    ReAct 主循环:
      Step 0: 初始候选池 (Phase A, 0 LLM)
      Step 1..max_steps: LLM 观察 state → 决策 action → 派发 → yield obs
        action: search_more / extract_facts / write_section / finish
      收尾: theme + sections 落 surveys.db, _audit.md, 最终 markdown

    Args:
        topic: 综述主题, 例 "量子纠缠度量计算"
        max_steps: ReAct 循环上限 (默认 20, spec §7.1)
        dry_run: True = 不调 LLM, 只走 Phase A 候选池构建

    Yields:
        流式进度 + 章节内容, chat stream_chunk signal 消费
        (is_streaming=True 标记, agent.py 派发层识别为 generator)
    """
    if not topic or not topic.strip():
        yield "❌ topic 不能为空\n"
        return

    topic = topic.strip()

    # 0. workspace 准备
    topic_hash = hash_topic(topic)
    ws = SurveyWorkspace(topic_hash, topic=topic)
    ws.set_status("init", 0, 0, f"topic={topic}")

    yield f"🔍 开始综述: {topic}\n"

    # 1. Step 0: Phase A 初始候选池 (0 LLM, summary_items.db + chroma + arxiv fallback, 9-08 多源召回)
    pool = _build_candidate_pool(topic, max_pool=max_pool)
    if not pool:
        ws.set_status("failed", 0, 0, "pool empty")
        yield (
            f"❌ 候选池为空\n"
            f"- topic: {topic!r}\n"
            f"- 主题词在 summary_items.db 里没匹配到任何 paper, arxiv API fallback 也失败\n"
            f"- 建议: 换个主题词, 或先跑 `抓文 → analyze_paper` 准备数据\n"
        )
        return
    ws.save_pool(pool)
    yield f"📚 初始候选池: {len(pool)} 篇\n\n"

    # state 初始化
    # 9-08 user 拍板: 加 outline 字段, 把"写章节"拆成 write_outline + write_section 两个 action
    # outline_sections: list of {title, section_type, sub_points, is_transition}
    # 9-08 user 拍板: 加 section_gaps 字段, LLM 写章节时如果觉得还缺文章, 会在 markdown 末尾
    #   标注 `<!-- QAgent: missing_queries=[...] -->`  →  system 解析后存到 section_gaps
    #   → action_finish 看到 section_gaps 不空就拒绝结束, ReAct 回到 search_more 补
    state: Dict[str, Any] = {
        "topic": topic,
        "topic_hash": topic_hash,
        "pool": pool,
        "facts": [],
        "sections": [],
        "outline": None,                # 综述大纲原始 LLM 输出 (str)
        "outline_sections": [],         # 大纲结构化: [{title, section_type, sub_points, is_transition}, ...]
        "section_gaps": [],             # 章节缺漏: [{section_idx, title, queries: [...]}]
        "unused_pool_papers": [],       # 9-08 user 拍板: 候选池中未被任何章节引用的 paper_ids
                                        # build_react_prompt 提示 LLM, 让写章节时能补充引用
        "errors": [],
        "step": 0,
        "finish_requested": False,
        "max_steps": max_steps,
    }

    if dry_run:
        ws.save_audit(
            f"# dry_run audit\n- topic: {topic}\n- pool_size: {len(pool)}\n- workspace: {ws.path}\n"
        )
        yield (
            f"## survey_generate dry_run\n\n"
            f"- topic: {topic}\n"
            f"- 候选池: {len(pool)}\n"
            f"- workspace: `{ws.path}`\n"
            f"- 跳过 ReAct 循环 (dry_run=True)\n"
        )
        return

    # 2. ReAct 主循环
    while state["step"] < state["max_steps"] and not state["finish_requested"]:
        state["step"] += 1
        yield f"\n--- Step {state['step']} ---\n"

        # 2.1 LLM 决策 (step 1 用 mock, 后续 step 接真实 _call_llm + parse_decision)
        decision = _decide_action(state)

        # 9-08 17:25 user 拍板: 主循环只显示"调用了 X 工具", 不显示 thought (LLM 内部想法) 和 args (query 等详情)
        yield f"➡️ 调用工具: {decision['action']}\n"

        # 2.2 派发 action (try/except 包裹, 失败塞 state["errors"], 不中断)
        try:
            for chunk in _dispatch_action(decision, state, ws):
                if chunk:
                    yield chunk
        except Exception as e:
            friendly = _friendly_error(e, decision["action"])
            state["errors"].append({
                "step": state["step"],
                "action": decision["action"],
                "msg": friendly,
            })
            yield f"👁️ 观察: ❌ {friendly}\n"

        # 2.3 trace 落盘 (spec §6 _react_trace.jsonl)
        _append_react_trace(ws, {
            "step": state["step"],
            "thought": decision["thought"],
            "action": decision["action"],
            "args": decision.get("args", {}),
            "n_errors": len(state["errors"]),
        })

    # 3. 收尾 (step 1 stub — 后续 step 5 落 surveys.db + audit + 最终 markdown)
    final = _finalize_survey(state, ws)
    yield final


# ============================================================
# P6.5 ReAct helpers (step 1 骨架 — action stub + mock decision)
# 业务实现将在 step 2-5 逐步替换 action_* stub 内容
# ============================================================

def _friendly_error(e: Exception, action: str) -> str:
    """把异常翻译成友好提示 (spec §4). step 1 简单实现, 后续 step 扩展查表."""
    err_type = type(e).__name__
    msg = str(e)[:200]
    return f"{action} 失败 ({err_type}): {msg}"


def _decide_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ReAct LLM 真决策 (step 6 真实实现).

    1. 调 minimax LLM 拿 decision (build_react_prompt)
    2. parse JSON {thought, action, args}
    3. 失败 fallback 到 _state_driven_decide (state 规则保底)

    9-08 19:30 加: finish_rejected 时直接走 state 规则 (不调 LLM, 不浪费 LLM 调用,
    因为 finish 刚被拒, 已经知道下一步该调什么, 没必要再问 LLM)
    """
    from agent.tools.llm import _call_llm
    from agent.prompts_survey import build_react_prompt

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    fallback = _state_driven_decide(state)

    # 9-08 19:30: finish 刚被拒 → 直接用 state 规则 + forced action hint
    if state.get("finish_rejected"):
        forced = state.pop("_forced_action_hint", None)
        if forced:
            action, args = forced
            return {
                "thought": f"[finish 被拒自动执行] {action}({args})",
                "action": action,
                "args": args,
                "_source": "auto_from_finish_reject",
            }
        # 没 hint 就走 state 规则
        return fallback

    try:
        prompt = build_react_prompt(state)
        out = _call_llm(
            prompt,
            provider=_PROVIDER,
            model=_MODEL,
            enable_thinking=False,
            temperature=0.3,
            max_tokens=1000,
            timeout=60,
        )
        if out and "<think>" in out:
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
        m = re.search(r"\{[\s\S]*\}", out or "")
        if not m:
            raise ValueError("LLM 输出无 JSON")
        decision = json.loads(m.group(0))
        action = decision.get("action", "")
        if action not in ("search_more", "extract_facts", "write_outline", "write_section", "finish"):
            raise ValueError(f"未知 action: {action!r}")
        if not isinstance(decision.get("args"), dict):
            decision["args"] = {}
        decision.setdefault("thought", "")
        decision["_source"] = "llm"  # 标记是 LLM 决策, 不是 fallback
        return decision
    except Exception as e:
        # LLM 决策失败 -> fallback state 规则 (spec §1.3 容错)
        return {
            "thought": f"[fallback] LLM 决策失败 ({type(e).__name__}: {str(e)[:80]}), 走 state 规则",
            "action": fallback["action"],
            "args": fallback["args"],
            "_source": "fallback",
        }


def _state_driven_decide(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    state-driven fallback 规则 (LLM 决策失败时用, 也是测试兜底).

    9-08 user 拍板的固定候选池规则:
      - 候选池 < 20  → search_more (硬规则, 优先)
      - 候选池 ≥ 40 且 fact < 5  → extract_facts (硬规则, 优先)
      - 20 ≤ pool < 40  → LLM 自由(中间区不强制, 让 LLM 按主题判断)
    9-08 user 拍板的拆分:
      - 没 outline → write_outline (硬规则, 写章节前必须先生成大纲)
      - 有 outline 且未写完所有章节 → write_section(section_idx)
      - 写完所有 outline 章节 → finish
    9-08 user 拍板的缺漏反馈:
      - section_gaps 不空 → search_more(用 gap 的 query 扩候选池, 优先)
      - search_more 完成后 → write_section 重写该 gap 章节 (section_idx=gap.section_idx)
    """
    pool_size = len(state["pool"])
    fact_count = len(state["facts"])
    section_count = len(state["sections"])
    outline_sections = state.get("outline_sections", [])
    section_gaps = state.get("section_gaps", []) or []

    # ========== 候选池硬规则 (9-08 抽 env QAGENT_SURVEY_MIN_POOL, 默认 30; 充裕 = min+10) ==========
    _MIN_POOL = get_min_pool_papers()
    if pool_size < _MIN_POOL:
        return {"action": "search_more", "args": {"query": state["topic"], "k": 5}}
    if pool_size >= _MIN_POOL + 10 and fact_count < 5:
        return {
            "action": "extract_facts",
            "args": {"paper_ids": [p["paper_id"] for p in state["pool"][:5]]},
        }

    # ========== 缺漏优先: section_gaps 不空 → 补文章 ==========
    # 用最新 gap 的第一个 query 调 search_more 扩池
    # (如果有多个 gap, 一次只补一个, 避免 step 浪费)
    if section_gaps:
        first_gap = section_gaps[0]
        gap_queries = first_gap.get("queries", [])
        if gap_queries:
            return {
                "action": "search_more",
                "args": {"query": gap_queries[0], "k": 5},
            }
        # gap 没 query, 退化: 重写该 section_idx 章节 (让 LLM 重新评估)
        return {
            "action": "write_section",
            "args": {"section_idx": first_gap.get("section_idx", 0)},
        }

    # ========== 大纲规则: 写章节前必须先 write_outline ==========
    if not outline_sections:
        return {"action": "write_outline", "args": {}}

    # ========== 写章节: 按 outline_sections 顺序, 用 section_idx 标识 ==========
    if section_count < len(outline_sections):
        return {
            "action": "write_section",
            "args": {"section_idx": section_count},  # 0-based, 下一章
        }

    # ========== 收尾 ==========
    return {"action": "finish", "args": {}}


def _dispatch_action(decision: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """派发 action 到 5 个 handler (9-08 加 write_outline, 拆 write_section 前的结构生成)"""
    action = decision["action"]
    args = decision.get("args", {})
    if action == "search_more":
        yield from action_search_more(args, state, ws)
    elif action == "extract_facts":
        yield from action_extract_facts(args, state, ws)
    elif action == "write_outline":
        yield from action_write_outline(args, state, ws)
    elif action == "write_section":
        yield from action_write_section(args, state, ws)
    elif action == "finish":
        yield from action_finish(args, state, ws)
    else:
        yield f"⚠️ 未知 action: {action!r}\n"


def _append_react_trace(ws, trace_entry: Dict[str, Any]) -> None:
    """写 _react_trace.jsonl (spec §6). 失败不阻塞主流程."""
    try:
        trace_path = ws.path / "_react_trace.jsonl"
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("[survey_generate] _react_trace 写盘失败: %s", e)


# ---- 4 个 action stub (step 2-5 逐步替换为真实实现) ----

def action_search_more(args: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """
    search_more action (step 2 真实实现):
      1. arxiv API 抓 k 篇
      2. 每篇调 minimax LLM 跑单篇 summarize (4 字段)
      3. 写 summary_items.db (source='survey_search')
      4. 加进 state['pool']

    单篇失败不阻塞 batch (spec §7.2 D).
    """
    from agent.tools.summary import save_summary_item
    from agent.tools.llm import _call_llm
    from agent.prompts_survey import build_summarize_prompt

    query = args.get("query") or state["topic"]
    k = args.get("k", 5)
    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    # 9-08 17:20 user 拍板: 不显示具体论文内容 (paper_id/标题/row_id) , 只显示高层动作
    # 之前 yield 太多细节: "📥 arxiv 返回 N 篇, 开始单篇 summarize"、"  [i/N] summarize xxx: title..."、
    # "    ✅ 入库 (row_id=123)" 等 — 全是污染, 去掉
    yield "🔎 去找论文了\n"

    # 1. arxiv API 抓 (silent)
    try:
        raw_papers = _arxiv_search(query, k)
    except Exception as e:
        friendly = _friendly_error(e, "search_more.arxiv")
        state["errors"].append({
            "step": state["step"], "action": "search_more.arxiv",
            "msg": friendly,
        })
        yield f"❌ 找论文失败: {friendly}\n"
        return

    if not raw_papers:
        yield "⚠️ 没找到新论文\n"
        yield f"👁️ 候选池 {len(state['pool'])} 篇 (无新增)\n"
        return

    # 2. 单篇 summarize + 写库 (silent, 不 yield 单篇进度)
    new_pool = []
    for i, p in enumerate(raw_papers, 1):
        paper_id = p["paper_id"]
        try:
            # 调 LLM 抽 4 字段
            sum_prompt = build_summarize_prompt(
                paper_id=paper_id,
                title=p.get("title", ""),
                authors=p.get("authors", ""),
                abstract=p.get("abstract", ""),
            )
            out = _call_llm(
                sum_prompt,
                provider=_PROVIDER,
                model=_MODEL,
                enable_thinking=False,
                temperature=0.2,
                max_tokens=10000,   # 9-08 user 拍板: 上限 10000
                timeout=180,
            )
            # 剥 <think>
            if out and "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
            # parse JSON (容错)
            m = re.search(r"\{[\s\S]*\}", out or "")
            if not m:
                raise ValueError(f"summarize LLM 输出无 JSON: {(out or '')[:200]}")
            try:
                summary_dict = json.loads(m.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"summarize JSON parse 失败: {e}")

            # 写 summary_items.db (source='survey_search', spec §7.2 C)
            save_summary_item(
                paper_id=paper_id,
                source="survey_search",
                title=p.get("title", ""),
                authors=p.get("authors", ""),
                abstract=p.get("abstract", ""),
                core_method=summary_dict.get("core_method", ""),
                core_contrib=summary_dict.get("core_contrib", ""),
                keywords=summary_dict.get("keywords", ""),
                outlook=summary_dict.get("outlook", ""),
            )
            # 加进 state['pool'], 沿用 Phase A 候选池 dict 格式
            new_pool.append({
                "paper_id": paper_id,
                "title": p.get("title", ""),
                "year": str(p.get("year", "")),
                "journal": "arxiv",
                "core_method": summary_dict.get("core_method", ""),
                "core_contrib": summary_dict.get("core_contrib", ""),
                "abstract": p.get("abstract", ""),
                "outlook": summary_dict.get("outlook", ""),
                "keywords": summary_dict.get("keywords", ""),
                "_source": "arxiv_api_via_survey_search",
            })
        except Exception as e:
            # 单篇失败不阻塞 batch (spec §7.2 D), 但不 yield 错误详情
            state["errors"].append({
                "step": state["step"], "action": "search_more.summarize",
                "paper": paper_id, "msg": _friendly_error(e, "search_more.summarize"),
            })
            # 静默继续

    state["pool"].extend(new_pool)
    # 9-08 17:20 user 拍板: 只显示"找到 N 篇" 这种高层总结
    yield f"👁️ 找到 {len(new_pool)} 篇, 累计候选池 {len(state['pool'])} 篇\n"


def _arxiv_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """arxiv API 抓 k 篇, 返回 dict list."""
    import arxiv
    search = arxiv.Search(
        query=query,
        max_results=k,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    out: List[Dict[str, Any]] = []
    for r in client.results(search):
        out.append({
            "paper_id": r.entry_id.split("/")[-1].split("v")[0],  # 去 /abs/ 前缀和版本号
            "title": r.title or "",
            "year": str(r.published.year) if r.published else "",
            "authors": ", ".join([a.name for a in (r.authors or [])[:5]]),
            "abstract": r.summary or "",
            "keywords": ", ".join([a.name for a in (r.authors or [])[:3]]),
        })
        if len(out) >= k:
            break
    return out


def action_extract_facts(args: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """
    extract_facts action (step 3 真实实现):
      - 对每篇 paper 调 minimax LLM 抽 3-5 条 fact (JSON 数组)
      - 复用 P6 build_fact_prompt + _safe_json_loads_array 容错
      - fact 写 workspace/_facts.jsonl, 加 state['facts']
      - 单篇失败不阻塞 batch
    """
    from agent.tools.llm import _call_llm
    from agent.prompts_survey import build_fact_prompt

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    # 默认抽 state['pool'] 全部; 也可指定 paper_ids 子集
    target_ids = args.get("paper_ids")
    if target_ids:
        targets = [p for p in state["pool"] if p["paper_id"] in target_ids]
    else:
        targets = list(state["pool"])

    if not targets:
        yield "⚠️ extract_facts: 0 篇目标 (paper_ids 不在 pool 中)\n"
        yield f"👁️ 观察: extract_facts 0 篇, 累计 facts={len(state['facts'])} 条\n"
        return

    yield f"📝 extract_facts: 抽 {len(targets)} 篇, 调 {_PROVIDER}/{_MODEL}\n"

    for i, paper in enumerate(targets, 1):
        paper_id = paper["paper_id"]
        # 9-08 17:20 user 拍板: 不显示具体 fact_id / 单篇进度
        try:
            fact_prompt = build_fact_prompt(
                paper_id=paper_id,
                title=paper.get("title", ""),
                year=str(paper.get("year", "")),
                journal=paper.get("journal", ""),
                core_method=paper.get("core_method", ""),
                core_contrib=paper.get("core_contrib", ""),
                abstract=paper.get("abstract", ""),
                outlook=paper.get("outlook", ""),
            )
            out = _call_llm(
                fact_prompt,
                provider=_PROVIDER,
                model=_MODEL,
                enable_thinking=False,
                temperature=0.2,
                max_tokens=10000,   # 9-08 user 拍板: 上限 10000
                timeout=180,
            )
            # 剥 <think>
            if out and "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
            # 容错 parse JSON 数组
            m = re.search(r"\[[\s\S]*", out or "")
            if not m:
                raise ValueError(f"LLM 输出无 JSON 数组: {(out or '')[:200]}")
            facts = _safe_json_loads_array(m.group(0))
            if facts is None:
                raise ValueError(f"JSON parse 失败 (含容错): {(out or '')[:200]}")
            # 落 _facts.jsonl
            for f in facts:
                if not isinstance(f, dict):
                    continue
                f.setdefault("paper_id", paper_id)
                ws.append_fact(f)
                state["facts"].append(f)
        except Exception as e:
            # 单篇失败不阻塞 batch, 不 yield 详情
            state["errors"].append({
                "step": state["step"], "action": "extract_facts",
                "paper": paper_id, "msg": _friendly_error(e, "extract_facts"),
            })

    # 9-08 17:20 user 拍板: 只显示"抽了 N 条 fact" 高层
    yield f"👁️ 抽完 {len(targets)} 篇, 累计 {len(state['facts'])} 条 fact\n"


def action_write_outline(args: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """
    write_outline action (9-08 新增, user 拍板拆分 write_section):
      - 调 minimax LLM 走 build_outline_prompt 生成综述大纲(5 段固定: intro/progress/method_compare/gap/outlook)
      - 解析 JSON: {sections: [{title, section_type, sub_points, is_transition}, ...]}
      - 存 state["outline"] (raw) + state["outline_sections"] (结构化 list)
      - 落 workspace/_outline.json (SurveyWorkspace.save_outline 已有)
      - outline_sections 锁住后, write_section 按 section_idx 顺序写

    大纲生成失败不重试(LLM 调用本身就贵, 失败就让 LLM 下一轮重选 action)
    """
    from agent.tools.llm import _call_llm
    from agent.prompts_survey import build_outline_prompt

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    # 已存在大纲 → 跳过(避免 LLM 重复调)
    if state.get("outline_sections"):
        # 9-08 17:25 改: silent 跳过, 不 yield 啰嗦
        return

    yield f"📋 生成综述大纲\n"

    try:
        prompt = build_outline_prompt(topic=state["topic"], papers=state["pool"])
        out = _call_llm(
            prompt,
            provider=_PROVIDER,
            model=_MODEL,
            enable_thinking=False,
            temperature=0.3,
            max_tokens=10000,    # 9-08 user 拍板: 上限 10000
            timeout=180,
        )
    except Exception as e:
        friendly = _friendly_error(e, "write_outline")
        state["errors"].append({
            "step": state["step"], "action": "write_outline",
            "msg": friendly,
        })
        yield f"  ❌ LLM 失败: {friendly}\n"
        return

    # 剥 <think>
    if out and "<think>" in out:
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()

    # parse JSON {sections: [...]}
    parsed: Optional[Dict[str, Any]] = None
    if out:
        # 1) 直接
        try:
            v = json.loads(out.strip())
            if isinstance(v, dict) and "sections" in v:
                parsed = v
        except (json.JSONDecodeError, SyntaxError):
            pass
        # 2) regex 抓最大 {...}
        if parsed is None:
            m = re.search(r"\{[\s\S]*\}", out)
            if m:
                try:
                    v = json.loads(m.group(0))
                    if isinstance(v, dict) and "sections" in v:
                        parsed = v
                except (json.JSONDecodeError, SyntaxError):
                    pass
        # 3) 修常见尾巴
        if parsed is None:
            for tail in ['"}]', '"}', '}', ']']:
                try:
                    s = out.strip().rstrip(",\n ") + tail
                    v = json.loads(s)
                    if isinstance(v, dict) and "sections" in v:
                        parsed = v
                        break
                except (json.JSONDecodeError, SyntaxError):
                    pass

    if not parsed or not isinstance(parsed.get("sections"), list) or not parsed["sections"]:
        state["errors"].append({
            "step": state["step"], "action": "write_outline",
            "msg": f"JSON parse 失败: {(out or '')[:200]}",
        })
        yield f"  ❌ 大纲 JSON parse 失败, 下一轮重试\n"
        return

    # 规范化 sections
    # 9-08 17:00 user 拍板: section_type 容错 + 至少 5 段
    #   - LLM 偶尔拼写错: "Method-Compare" / "GAP" / "out look" → 强制 normalize 到 5 段合法值
    #   - LLM 偷懒只返 2 段 → 补到 5 段 (intro/progress/method_compare/gap/outlook)
    VALID_TYPES = {"intro", "progress", "method_compare", "gap", "outlook"}

    def _normalize_sec_type(raw: str) -> Optional[str]:
        """拼写/分隔符容错, 返回合法 section_type 或 None"""
        s = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not s:                          # 9-08 修: 空字符串不参与匹配 (避免 "" in "outlook" 误命中)
            return None
        # 直接命中
        if s in VALID_TYPES:
            return s
        # 模糊匹配
        for v in VALID_TYPES:
            if v in s or s in v:
                return v
        # 关键词 fallback
        if "intro" in s or "background" in s or "introduction" in s:
            return "intro"
        if "progress" in s or "timeline" in s or "milestone" in s:
            return "progress"
        if "method" in s or "compare" in s or "comparison" in s:
            return "method_compare"
        if "gap" in s or "limit" in s or "challenge" in s:
            return "gap"
        if "outlook" in s or "future" in s or "direction" in s:
            return "outlook"
        return None

    normalized = []
    seen_types: set = set()
    for sec in parsed["sections"]:
        if not isinstance(sec, dict):
            continue
        title = (sec.get("title") or "").strip()
        sec_type = _normalize_sec_type(sec.get("section_type", ""))
        if not title or not sec_type:
            continue  # 跳过非法 section (避免污染 outline)
        # 9-08 防重复: 同一 section_type 出现多次, 只保留第一个, 后续的去重
        if sec_type in seen_types:
            continue
        seen_types.add(sec_type)
        sub_points = sec.get("sub_points") or []
        if not isinstance(sub_points, list):
            sub_points = [str(sub_points)]
        sub_points = [str(sp).strip() for sp in sub_points if str(sp).strip()]
        if not sub_points:
            sub_points = ["(待定)"]
        is_transition = str(sec.get("is_transition", "no")).strip().lower()
        normalized.append({
            "title": title,
            "section_type": sec_type,
            "sub_points": sub_points,
            "is_transition": "yes" if is_transition in ("yes", "true", "1") else "no",
        })

    # 9-08 17:00 user 拍板: LLM 偷懒只返 < 5 段时, 用 default 补足到 5 段
    DEFAULT_OUTLINE = [
        {"title": "引言", "sec_type": "intro", "sub_points": ["背景与动机", "研究问题", "本综述范围"], "is_transition": "yes"},
        {"title": "关键进展时间线", "sec_type": "progress", "sub_points": ["早期突破", "中期优化", "近期进展"], "is_transition": "no"},
        {"title": "方法分类与对比", "sec_type": "method_compare", "sub_points": ["按机制分类", "按性能对比", "适用场景"], "is_transition": "no"},
        {"title": "研究空白与不足", "sec_type": "gap", "sub_points": ["技术瓶颈", "实验限制", "理论未解"], "is_transition": "no"},
        {"title": "未来方向与展望", "sec_type": "outlook", "sub_points": ["可迁移方向", "开放问题", "标准化展望"], "is_transition": "no"},
    ]
    if len(normalized) < 5:
        existing = {s["section_type"] for s in normalized}
        for default in DEFAULT_OUTLINE:
            if default["sec_type"] not in existing:
                normalized.append({
                    "title": default["title"],
                    "section_type": default["sec_type"],
                    "sub_points": default["sub_points"],
                    "is_transition": default["is_transition"],
                })
                existing.add(default["sec_type"])
                if len(normalized) >= 5:
                    break
        # 9-08 17:25 删: 补足信息是 internal, 不 yield

    if not normalized:
        state["errors"].append({
            "step": state["step"], "action": "write_outline",
            "msg": "规范化后 sections 为空 (全部非法, section_type 不在 5 段列表)",
        })
        # 9-08 17:25 删: 错误时只 yield 一次, 简短
        yield "❌ 大纲生成失败\n"
        return

    # 存 state + 落盘 (silent, 9-08 17:25 删了 💾 落盘 yield)
    state["outline"] = out or ""
    state["outline_sections"] = normalized
    try:
        ws.save_outline({"sections": normalized, "raw": out or ""})
    except Exception as e:
        logger.warning("[write_outline] save_outline 失败: %s", e)

    # 9-08 17:25 改: 大纲 yield 简洁版, 只显示章节数 + 标题列表
    yield f"✅ 大纲 {len(normalized)} 章\n"
    for i, sec in enumerate(normalized):
        yield f"  {i+1}. [{sec['section_type']}] {sec['title']}\n"
    # 注: 完整大纲(含子点)落 workspace/_outline.json + _finalize_survey 报告里展示, 不在主循环啰嗦 yield
    # (之前加了一堆 📑 === 综述结构 === 大段 markdown 把主循环进度淹没了, 已回退)


def action_write_section(args: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """
    write_section action (step 4 真实实现):
      - 调 minimax LLM 写单个章节 markdown
      - 复用 P6 build_section_prompt + _audit_section_paper_ids
      - 自动重写 (P6.3): out_of_pool > 0 时加约束重试, 最多 2 次
      - 章节落 surveys.db (create_section, paper_ids 逗号分隔)
      - step 4 用非流式 LLM 拿完整 markdown 再 yield (真流式要 _call_llm_stream, 留 P+)
    """
    from agent.tools.llm import _call_llm
    from agent.prompts_survey import build_section_prompt

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    # 9-08 user 拍板: 章节内容必须基于 outline (先 write_outline 再 write_section)
    # title / section_type / sub_points 全部从 outline_sections[section_idx] 拿
    # 旧的 args.title / args.sub_points 字段废弃, 但允许 LLM 老调用通过 (会忽略)
    outline_sections = state.get("outline_sections", [])
    if not outline_sections:
        yield (
            f"❌ write_section: 大纲未生成, 必须先 write_outline (state['outline_sections'] 空)\n"
            f"👁️ 观察: 跳到下一轮, LLM 应选 write_outline\n"
        )
        state["errors"].append({
            "step": state["step"], "action": "write_section",
            "msg": "大纲未生成, write_section 被拒",
        })
        return

    # section_idx 默认 = 已写章节数 (下一章)
    section_idx = args.get("section_idx")
    if section_idx is None:
        section_idx = len(state["sections"])
    if not isinstance(section_idx, int) or section_idx < 0 or section_idx >= len(outline_sections):
        yield (
            f"❌ write_section: section_idx={section_idx} 越界 "
            f"(outline 共 {len(outline_sections)} 章, 已写 {len(state['sections'])} 章)\n"
        )
        state["errors"].append({
            "step": state["step"], "action": "write_section",
            "msg": f"section_idx={section_idx} 越界",
        })
        return

    outline_sec = outline_sections[section_idx]
    title = outline_sec["title"]
    sec_type = outline_sec["section_type"]
    sub_points = outline_sec["sub_points"]
    outline_section_md = "\n".join(f"- {sp}" for sp in sub_points)

    yield f"✍️ 写章节 [{sec_type}] {title}\n"

    # theme: 第一次创建, 后续复用 (silent, 9-08 17:25 删了 📌 新建 theme yield)
    if "theme_id" not in state:
        state["theme_id"] = create_theme(
            name=state["topic"],
            description=f"P6.5 survey_generate 自动创建 at {datetime.now().isoformat(timespec='seconds')}",
        )
    theme_id = state["theme_id"]

    pool_ids = {p["paper_id"] for p in state["pool"]}

    # 自动重写 loop (P6.3: 最多 2 次, spec §3.3)
    MAX_REWRITE = 2
    sec_md = ""
    bucket = {"in_pool": [], "out_of_pool": [], "total_cites": 0}
    for rewrite_i in range(MAX_REWRITE + 1):
        try:
            section_prompt = build_section_prompt(
                topic=state["topic"],
                section_type=sec_type,
                section_title=title,
                outline_section=outline_section_md,
                papers=state["pool"],
            )
            if rewrite_i > 0:
                # 重写时加约束: 只能用候选池内 paper_id (silent, 9-08 17:25 删了 🔁 重写 yield)
                extra_constraint = (
                    f"\n\n【P6.3 自动重写 {rewrite_i}/{MAX_REWRITE}】"
                    f"\n⚠️ 你上一次的输出引用了候选池外的 paper_id ({bucket['out_of_pool']}),"
                    f"\n这会被自动审计标 ❌ (幻觉风险). 现在重写, **只能用候选池内**的 paper_id:"
                    f"\n候选池 paper_id 列表: {sorted(pool_ids)}"
                    f"\n请删除/替换所有候选池外引用, 然后再 output 完整章节 markdown。"
                )
                section_prompt = section_prompt + extra_constraint

            sec_md = _call_llm(
                section_prompt,
                provider=_PROVIDER,
                model=_MODEL,
                enable_thinking=False,
                temperature=0.4,
                max_tokens=10000,  # 9-08 user 拍板: 上限 10000
                timeout=240,
            )
        except Exception as e:
            # 异常不重试 (silent, 9-08 17:25 删了 ❌ LLM 失败 yield)
            sec_md = f"⚠️ 章节生成失败: {_friendly_error(e, 'write_section')}\n\n(占位章节: {title})"
            state["errors"].append({
                "step": state["step"], "action": "write_section",
                "msg": _friendly_error(e, "write_section"),
            })
            break

        # 剥 <think>
        if sec_md and "<think>" in sec_md:
            sec_md = re.sub(r"<think>.*?</think>", "", sec_md, flags=re.DOTALL).strip()

        # 审计引用
        bucket = _audit_section_paper_ids(sec_md, pool_ids)
        if not bucket["out_of_pool"] or rewrite_i >= MAX_REWRITE:
            break
        # out_of_pool > 0 且还能重写

    # ========== 9-08 17:45 user 反馈: 章节内容为空 / 过短检测 ==========
    # user 实测 "第 2 章 progress 实际是空的——LLM 写了大标题但没正文" (#48)
    # 兜底: 如果 LLM 写的章节内容 < 200 字符 (排除纯标题/纯# 后), 标记空章节 + 阻断
    # 9-08 18:15 升级: 不再 silent 标 gaps, 而是完整阻断 — 不 yield 章节内容 + 不落 db +
    #   不 append sections (章节数自然少 1) + 标 section_gaps + 立即 return
    # 这样 action_finish 看到: ① 章节数少 1 (拒) + ② section_gaps 不空 (拒) → 双保险
    # user 也看不到"空章节被 yield 出去" 的污染
    sec_md_clean = (sec_md or "").strip()
    effective_lines = [
        ln for ln in sec_md_clean.split("\n")
        if ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith("---")
    ]
    effective_content = "\n".join(effective_lines).strip()
    if len(effective_content) < 200:
        # 9-08 18:15 升级: 阻断 (return), 防止空章节内容被 yield / 落 db / 算章节数
        state["section_gaps"].append({
            "section_idx": section_idx,
            "title": title,
            "queries": [f"章节 '{title}' 内容过短 ({len(effective_content)} 字符), 需要重写并补充 800-1500 字内容"],
        })
        state["errors"].append({
            "step": state["step"], "action": "write_section",
            "msg": f"章节 '{title}' 内容过短: {len(effective_content)} 字符 (要求 ≥ 200)",
            "effective_content_preview": effective_content[:100],
        })
        yield f"⚠️ 章节 '{title}' 内容过短 ({len(effective_content)} 字符), 触发重写\n"
        return  # 阻断: 不 yield 章节内容 / 不落 db / 不 append sections / 不算章节数

    # ========== 9-08 user 拍板: 解析 missing_queries 标注 ==========
    # LLM 写章节时如果觉得还缺某些方向的文献, 会在 markdown 末尾加:
    #   `<!-- QAgent: missing_queries=["BB84 原始协议", "MDI-QKD"] -->`
    # system 抓出来后存 state["section_gaps"], 并从 sec_md 删掉标注 (避免污染正文)
    # action_finish 看到 section_gaps 不空 → 拒绝 finish → 回到 search_more 循环
    _MISSING_Q_RE = re.compile(
        r'<!--\s*QAgent:\s*missing_queries\s*=\s*\[(.*?)\]\s*-->',
        re.DOTALL,
    )
    m_missing = _MISSING_Q_RE.search(sec_md or "")
    if m_missing:
        # 解析 query 列表 (简单 split, 容忍单引号/双引号)
        raw_queries = m_missing.group(1)
        queries: List[str] = []
        try:
            import ast as _ast
            parsed = _ast.literal_eval(f"[{raw_queries}]")
            if isinstance(parsed, list):
                queries = [str(q).strip() for q in parsed if str(q).strip()]
        except Exception:
            # 退化: 手动 split
            queries = [
                q.strip().strip('"').strip("'").strip()
                for q in raw_queries.split(",")
                if q.strip()
            ]
        # 删掉标注 (避免污染正文)
        sec_md = _MISSING_Q_RE.sub("", sec_md).rstrip() + "\n"
        if queries:
            gap_entry = {
                "section_idx": section_idx,
                "title": title,
                "queries": queries,
            }
            state["section_gaps"].append(gap_entry)
            # 9-08 17:25 删: 不在主循环 yield 缺漏详情 (queries 是具体内容, 污染显示)
            # state['section_gaps'] 仍存, action_finish 会检查

    # 章节内容整体 yield (review 用, 9-08 17:25 保留)
    yield f"\n--- [{sec_type}] {title} ---\n"
    for line in sec_md.split("\n"):
        yield line + "\n"
    yield "--- 章节结束 ---\n"

    # 落 surveys.db (silent, 9-08 17:25 删了 💾 落盘 yield)
    try:
        sec_id = create_section(
            theme_id=theme_id,
            section_type=sec_type,
            title=title,
            content_md=sec_md or "",
            paper_ids=",".join(bucket["in_pool"]),
        )
        state["sections"].append({
            "section_id": sec_id,
            "section_idx": section_idx,
            "title": title,
            "type": sec_type,
            "in_pool": bucket["in_pool"],
            "out_of_pool": bucket["out_of_pool"],
        })
    except Exception as e:
        friendly = _friendly_error(e, "create_section")
        state["errors"].append({
            "step": state["step"], "action": "write_section.create_section",
            "msg": friendly,
        })

    # 写完章节后清掉对应 idx 的 gap (silent, 9-08 17:25 删了 ✅ 清理 gap yield)
    state["section_gaps"] = [
        g for g in state.get("section_gaps", [])
        if g.get("section_idx") != section_idx
    ]

    # 覆盖度 audit → 传递给写作阶段 (silent, 9-08 17:25 删了 📊 覆盖度 audit yield 列 paper_ids)
    # 但 state['unused_pool_papers'] 仍更新, build_react_prompt 给 LLM 看
    all_cited: set = set()
    for r in state.get("sections", []):
        all_cited.update(r.get("in_pool", []))
    all_cited.update(bucket.get("in_pool", []))
    unused = sorted(pool_ids - all_cited)
    state["unused_pool_papers"] = unused
    # 不 yield 详情, 避免 paper_ids 列表污染显示

    # 9-08 17:25 改: 高层 yield 只显示"章节 X/Y 完成"
    yield f"👁️ 章节 {len(state['sections'])}/{len(outline_sections)} 完成\n"


def action_finish(args: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """
    finish action (9-08 19:30 三重校验 + 自动打回执行):
      1. 章节数校验: section_count < len(outline_sections) → 拒绝, 自动触发 write_section
      2. 缺漏校验: state["section_gaps"] 不空 → 拒绝, 自动触发对应工具
      3. 全部通过 → 真正 finish (state["finish_requested"] = True)

    9-08 19:30 user 拍板:
      - 0 引用候选 不算"打回重写" 条件 (删 9-08 18:13 的 unused 校验)
      - finish 被拒要 **自动** 调用相关工具去解决, 不等用户说
        (实现: 标记 state["finish_rejected"]=True, 主循环下一轮 _decide_action 不调 LLM, 走 state 规则)
    """
    outline_sections = state.get("outline_sections", []) or []
    section_count = len(state.get("sections", []))
    gaps = state.get("section_gaps", []) or []

    # ========== 1. 章节数校验 ==========
    if outline_sections and section_count < len(outline_sections):
        missing_idx = list(range(section_count, len(outline_sections)))
        missing_titles = ", ".join(
            f"'{outline_sections[i].get('title', '?')}'" for i in missing_idx[:5]
        )
        more = f" 等 {len(missing_idx)} 章" if len(missing_idx) > 5 else ""
        # 9-08 19:30 拍板: finish 被拒要自动执行相关工具 (write_section)
        state["finish_rejected"] = True
        state["_forced_action_hint"] = ("write_section", {"section_idx": section_count})
        yield (
            f"❌ finish 被拒: 章节未写完! "
            f"已写 {section_count}/{len(outline_sections)} 章, "
            f"还差 {len(missing_idx)} 章 ({missing_titles}{more})\n"
            f"👁️ 自动触发: write_section(section_idx={section_count})\n"
        )
        return

    # ========== 2. 缺漏校验 ==========
    if gaps:
        gap_titles = ", ".join(f"'{g.get('title', '?')}'" for g in gaps[:5])
        # 9-08 19:30 拍板: 自动执行 — 优先 search_more (用 gap query), 没 query 就 write_section
        first_gap = gaps[0]
        gap_queries = first_gap.get("queries", [])
        if gap_queries and "内容过短" not in gap_queries[0]:
            # 缺漏是 search_more 类的, 自动调 search_more
            state["finish_rejected"] = True
            state["_forced_action_hint"] = ("search_more", {"query": gap_queries[0], "k": 5})
            yield (
                f"❌ finish 被拒: 还有 {len(gaps)} 个章节缺漏/空章节未补 ({gap_titles} 等)\n"
                f"👁️ 自动触发: search_more(query='{gap_queries[0]}')\n"
            )
        else:
            # 缺漏是"内容过短" 或没 query, 自动调 write_section 重写
            target_idx = first_gap.get("section_idx", section_count)
            state["finish_rejected"] = True
            state["_forced_action_hint"] = ("write_section", {"section_idx": target_idx})
            yield (
                f"❌ finish 被拒: 还有 {len(gaps)} 个章节缺漏/空章节未补 ({gap_titles} 等)\n"
                f"👁️ 自动触发: write_section(section_idx={target_idx}) 重写缺漏章节\n"
            )
        return

    # ========== 3. 全部通过 → 真正 finish ==========
    state["finish_rejected"] = False
    state["finish_requested"] = True
    yield (
        f"👁️ 观察: ✅ 收尾确认 (章节 {section_count}/{len(outline_sections)} 已写完, 无缺漏) → 🏁\n"
    )


def _finalize_survey(state: Dict[str, Any], ws) -> str:
    """
    收尾 (step 5 真实实现):
      - 算 audit: 候选池 0 引用论文
      - 落 workspace/_audit.md
      - 落 surveys.db theme (已经在 write_section 第一次创建)
      - yield 最终 markdown 报告
    """
    topic = state["topic"]
    pool = state["pool"]
    sections = state["sections"]
    pool_ids = {p["paper_id"] for p in pool}

    # 覆盖度 audit
    cited_ids: set = set()
    for r in sections:
        cited_ids.update(r.get("in_pool", []))
    uncited = pool_ids - cited_ids

    audit_lines = [
        f"# 覆盖度审计 — {topic}",
        "",
        f"- workspace: `{ws.path}`",
        f"- 候选池: {len(pool)}",
        f"- 抽 fact: {len(state['facts'])}",
        f"- 章节: {len(sections)} (theme_id={state.get('theme_id', '?')})",
        f"- ReAct steps: {state['step']}",
        f"- 错误数: {len(state['errors'])}",
        "",
        "## 每章节引用统计",
        "",
        "| 章节 | 类型 | 候选池内 | 候选池外(幻觉) |",
        "|---|---|---|---|",
    ]
    for r in sections:
        audit_lines.append(
            f"| {r['title']} | {r['type']} | "
            f"{len(r['in_pool'])} | {len(r['out_of_pool'])} |"
        )
    audit_lines.extend([
        "",
        f"## ⚠️ 候选池 0 引用论文 ({len(uncited)} / {len(pool_ids)} = "
        f"{len(uncited)*100//max(1,len(pool_ids))}%)",
        "",
    ])
    for pid in sorted(uncited):
        p = next((x for x in pool if x["paper_id"] == pid), None)
        title = (p or {}).get("title", "?")[:80]
        audit_lines.append(f"- `{pid}`: {title}")
    audit_md = "\n".join(audit_lines)
    ws.save_audit(audit_md)
    ws.set_status(
        "done", 1, 1,
        f"theme_id={state.get('theme_id', '?')}, sections={len(sections)}, audit=ok",
    )

    # 最终 markdown 报告
    report = [
        f"## survey_generate 完成 — {topic}",
        "",
        f"- workspace: `{ws.path}`",
        f"- theme_id: `{state.get('theme_id', '?')}` (surveys.db survey_theme.id)",
        f"- 候选池: {len(pool)} 篇",
        f"- 抽 fact: {len(state['facts'])} 条",
        f"- 章节: {len(sections)} 段",
        f"- ReAct steps: {state['step']}",
        f"- 错误数: {len(state['errors'])}",
        f"- 覆盖度审计: `{ws.path}/_audit.md`",
        "",
    ]

    # ========== 9-08 user 拍板: 综述结构先展示, 再展开 ==========
    # 用 outline_sections 列出"完整大纲", 让 user 在收尾报告里先看结构, 再看章节引用
    outline_sections = state.get("outline_sections", [])
    if outline_sections:
        report.extend([
            "## 综述结构 (大纲)",
            "",
        ])
        for i, sec in enumerate(outline_sections):
            sps = " / ".join(sec.get("sub_points", []) or ["(无)"])
            report.append(f"**{i+1}. [{sec.get('section_type', '?')}] {sec.get('title', '?')}**  ({'过渡' if sec.get('is_transition') == 'yes' else '实质'})")
            report.append(f"   - 子点: {sps}")
        report.append("")

    report.extend([
        "### 章节 (section_id, 落库 surveys.db)",
        "",
    ])
    for r in sections:
        in_pids_str = ", ".join(f"[[{pid}]]" for pid in r['in_pool']) or "(无)"
        out_pids_str = ", ".join(f"[[{pid}]]" for pid in r['out_of_pool']) or "(无)"
        report.append(
            f"- `{r['section_id']}` [{r['type']}] {r['title']}\n"
            f"  - 候选池内引用 ({len(r['in_pool'])}): {in_pids_str}\n"
            f"  - 候选池外引用/幻觉 ({len(r['out_of_pool'])}): {out_pids_str}"
        )
    report.append("")
    report.append(f"### 候选池 0 引用 ({len(uncited)} / {len(pool_ids)}):")
    if uncited:
        for pid in sorted(uncited):
            p = next((x for x in pool if x["paper_id"] == pid), None)
            title = (p or {}).get("title", "?")[:80]
            report.append(f"- `[[{pid}]]`: {title}")
    else:
        report.append("  (无, 所有候选池 paper 都被引用过)")
    if state["errors"]:
        report.append("")
        report.append(f"### ⚠️ 错误 ({len(state['errors'])} 条)")
        for e in state["errors"][:10]:  # 最多列 10 条
            report.append(f"- step {e.get('step', '?')} [{e.get('action', '?')}] {e.get('msg', '?')}")
        if len(state["errors"]) > 10:
            report.append(f"- ... 还有 {len(state['errors']) - 10} 条")
    report.append("")
    report.append(f"### 审计详情")
    report.append(f"见 `{ws.path}/_audit.md`")

    return "\n".join(report)


# ============================================================
# P6.4 survey_audit 独立工具 — 对已有 survey 跑覆盖度审计(不重生成)
# ============================================================
@register_tool(name="survey_audit", is_streaming=False, cost_tier="L0")
def survey_audit(theme_id: int, topic: str = "") -> str:
    """
    对已有 survey 跑覆盖度审计(P6.4, 0 LLM, L0 档)。

    不重生成章节, 只对 surveys.db 已落库的 theme + sections 跑:
      - 每章节候选池内/外引用统计
      - 总引用次数 / 唯一 paper_id 数
      - 候选池 0 引用列表 (从 workspace _pool.json 读, 找不到则跳过)
      - 候选池外引用 (幻觉风险) 列表

    Args:
        theme_id: survey_theme.id
        topic: 可选, 用来从 workspace data/surveys/working/{hash}/_pool.json 找候选池
               不传则只跑章节级审计 (无候选池覆盖度)

    Returns:
        markdown 审计报告
    """
    # 1. 读 theme
    theme = get_theme(theme_id)
    if not theme:
        return f"❌ 找不到 theme_id={theme_id} 的 survey"

    # 2. 读 sections
    secs = list_sections(theme_id)
    if not secs:
        return f"❌ theme_id={theme_id} 没任何 section"

    # 3. 找候选池 (从 workspace _pool.json)
    pool: List[Dict[str, Any]] = []
    pool_topic = topic or theme.get("name", "")
    if pool_topic:
        from agent.infra.survey_workspace import hash_topic
        ws_path = Path("data/surveys/working") / hash_topic(pool_topic)
        pool_file = ws_path / "_pool.json"
        if pool_file.exists():
            try:
                pool = json.loads(pool_file.read_text(encoding="utf-8")).get("papers", [])
            except Exception:
                pass

    pool_ids = {p["paper_id"] for p in pool}

    # 4. 每章节 audit
    section_audit = []
    all_in_pool: set = set()
    all_out_pool: set = set()
    for sec in secs:
        content = sec.get("content_md", "")
        bucket = _audit_section_paper_ids(content, pool_ids)
        section_audit.append({
            "section_id": sec.get("id"),
            "title": sec.get("title", ""),
            "type": sec.get("section_type", ""),
            "in_pool": bucket["in_pool"],
            "out_of_pool": bucket["out_of_pool"],
        })
        all_in_pool.update(bucket["in_pool"])
        all_out_pool.update(bucket["out_of_pool"])

    # 5. 报告
    report = [
        f"## survey_audit — theme_id={theme_id}",
        "",
        f"- topic: {theme.get('name', '?')}",
        f"- description: {theme.get('description', '')[:100]}",
        f"- 章节数: {len(secs)}",
        f"- 候选池: {len(pool)} 篇" + ("" if pool else " (没找到 workspace, 候选池覆盖度审计跳过)"),
        "",
        "## 每章节引用统计",
        "",
        "| section_id | 章节 | 类型 | 候选池内 | 候选池外(幻觉) |",
        "|---|---|---|---|---|",
    ]
    for r in section_audit:
        report.append(
            f"| {r['section_id']} | {r['title'][:40]} | {r['type']} | "
            f"{len(r['in_pool'])} | {len(r['out_of_pool'])} |"
        )

    report.extend([
        "",
        f"## 总计",
        f"- 唯一 paper_id 引用: {len(all_in_pool)} 个 (候选池内), {len(all_out_pool)} 个 (候选池外, 幻觉风险)",
    ])

    if pool:
        uncited = pool_ids - all_in_pool
        coverage = (len(all_in_pool) * 100 // max(1, len(pool_ids)))
        report.extend([
            f"- 候选池覆盖度: {coverage}% ({len(all_in_pool)} / {len(pool_ids)})",
            "",
            f"## ⚠️ 候选池 0 引用论文 ({len(uncited)} / {len(pool_ids)}):",
            "",
        ])
        for pid in sorted(uncited):
            p = next((x for x in pool if x["paper_id"] == pid), None)
            title = (p or {}).get("title", "?")[:80]
            report.append(f"- `[[{pid}]]`: {title}")

    if all_out_pool:
        report.extend([
            "",
            f"## ❌ 候选池外引用 (幻觉风险) ({len(all_out_pool)} 个):",
            "",
        ])
        for pid in sorted(all_out_pool):
            report.append(f"- `[[{pid}]]`")

    return "\n".join(report)


