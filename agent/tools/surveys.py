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
from typing import List, Dict, Any, Optional

from agent.tools.registry import register_tool
from agent.prompts_survey import (
    build_fact_prompt,
    build_outline_prompt,
    build_section_prompt,
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


def _build_candidate_pool(topic: str, max_pool: int = 40) -> List[Dict[str, Any]]:
    """
    Phase A: 候选池构建(MVP 简化版)
    - 走 summary_items.db(0 LLM, 快)
    - 关键词匹配 title/abstract/core_method/core_contrib/keywords/outlook
    - 候选池上限 max_pool

    P6.3 优化: 走 semantic_search + semantic_search_keywords LLM 召回, 跟关键词过滤 hybrid

    P6.4 增强 (2026-09-05): summary_items.db 没匹配时 (用户实测 "entanglement measure"
    关键词匹配 0 条 → 工具早返回) 自动 fallback 到 arxiv API 抓 top-K 论文
    """
    from agent.tools.summary import list_summary_items

    # 多拿点,database 端过滤
    raw = list_summary_items(limit=max_pool * 3)
    if not raw:
        raw = []

    # 简单关键词匹配(topic 切词, 任一命中)
    keywords = [kw.strip() for kw in re.split(r"[\s,，、。;；]+", topic) if len(kw.strip()) >= 2]
    if not keywords:
        keywords = [topic]  # 兜底

    def match(item: Dict[str, Any]) -> int:
        """返回命中关键词数, 0 = 不匹配"""
        score = 0
        for field in ("title", "abstract", "core_method", "core_contrib", "keywords", "outlook"):
            v = (item.get(field) or "").lower()
            for kw in keywords:
                if kw.lower() in v:
                    score += 1
        return score

    scored = [(match(it), it) for it in raw]
    scored = [(s, it) for s, it in scored if s > 0]
    scored.sort(key=lambda x: -x[0])  # 高分优先

    pool = []
    for score, it in scored[:max_pool]:
        pool.append({
            "paper_id": it.get("paper_id", ""),
            "title": it.get("title", ""),
            "year": (it.get("summary_date") or "")[:4] if it.get("summary_date") else "",
            "journal": it.get("source", ""),
            "core_method": it.get("core_method", ""),
            "core_contrib": it.get("core_contrib", ""),
            "abstract": it.get("abstract", ""),
            "outlook": it.get("outlook", ""),
            "keywords": it.get("keywords", ""),
            "_match_score": score,  # 调试用
        })

    # P6.4 增强: summary_items 关键词匹配空 → 自动 fallback 到 arxiv API
    # 用户实测 2026-09-05: 跑 "entanglement measure" 综述, summary_items.db 全是 Test Paper,
    # 关键词匹配 0 条 → 工具早返回 "❌ 候选池为空", 综述生成失败
    if not pool:
        try:
            import arxiv
            search = arxiv.Search(
                query=topic,
                max_results=min(max_pool, 10),  # 限 10 篇避免太慢
                sort_by=arxiv.SortCriterion.Relevance,
            )
            client = arxiv.Client()
            arxiv_pool = []
            for r in client.results(search):
                arxiv_pool.append({
                    "paper_id": r.entry_id.split("/")[-1].split("v")[0],  # 去掉 /abs/ 前缀和版本号
                    "title": r.title or "",
                    "year": str(r.published.year) if r.published else "",
                    "journal": "arxiv",
                    "core_method": "",  # arxiv API 不给 core_method
                    "core_contrib": (r.summary or "")[:200],  # 用 abstract 截短
                    "abstract": r.summary or "",
                    "outlook": "",
                    "keywords": ", ".join([a.text for a in (r.authors or [])[:3]]),  # 用作者当关键词
                    "_match_score": 0,  # 标记为 arxiv 召回
                    "_source": "arxiv_api_fallback",
                })
                if len(arxiv_pool) >= max_pool:
                    break
            if arxiv_pool:
                pool = arxiv_pool
        except Exception as e:
            # arxiv API 失败 (网络/timeout), 不影响主流程, 让 pool 保持空
            # Phase A 仍会返 "❌ 候选池为空" 但用户能看到根因
            pass

    return pool


@register_tool(name="survey_generate", is_streaming=False, cost_tier="L2")
def survey_generate(
    topic: str,
    max_pool: int = 40,
    max_steps: int = 10,
    dry_run: bool = False,
) -> str:
    """
    生成综述(P6 主工具, L2 档)。

    6 段 pipeline:
      Phase A: 候选池构建 (0 LLM, summary_items.db 关键词匹配)
      Phase B: (本轮跳过) 抓新 + analyze_paper
      Phase C: fact 提取 (每篇 1 次 LLM, 3-5 fact/paper)
      Phase D: 大纲生成 (1 次 LLM, 5 章节)
      Phase E: 章节生成 (每章 1 次 LLM, 落 surveys.db)
      Phase F: 覆盖度审计 (0 LLM, 写 _audit.md)

    Args:
        topic: 综述主题, 例 "量子纠错码近期进展"
        max_pool: 候选池上限(默认 40)
        max_steps: 暂未使用(预留 P4 ReAct 共享)
        dry_run: True = 不调 LLM, 只构建候选池(用于 UI 快速测)

    Returns:
        markdown 报告, 含 surveys.db section_id 列表 + 覆盖度 audit
    """
    if not topic or not topic.strip():
        return "❌ topic 不能为空"

    topic = topic.strip()

    # 0. workspace 准备
    topic_hash = hash_topic(topic)
    ws = SurveyWorkspace(topic_hash, topic=topic)
    ws.set_status("init", 0, 0, f"topic={topic}")

    # 1. Phase A: 候选池
    pool = _build_candidate_pool(topic, max_pool=max_pool)
    if not pool:
        ws.set_status("failed", 0, 0, "pool empty")
        return (
            f"❌ 候选池为空\n"
            f"- topic: {topic!r}\n"
            f"- 主题词在 summary_items.db 里没匹配到任何 paper\n"
            f"- 建议: 换个主题词, 或先跑 `抓文 → analyze_paper` 准备数据"
        )
    ws.save_pool(pool)
    ws.set_status("pool_built", len(pool), max_pool, f"pool={len(pool)}")

    if dry_run:
        ws.save_audit(f"# dry_run audit\n- topic: {topic}\n- pool_size: {len(pool)}\n- workspace: {ws.path}\n")
        return (
            f"## survey_generate dry_run\n\n"
            f"- topic: {topic}\n"
            f"- 候选池: {len(pool)}\n"
            f"- workspace: `{ws.path}`\n"
            f"- 跳过 LLM 调用 (dry_run=True)\n\n"
            f"### 候选池前 5 条:\n\n"
            + "\n".join(
                f"- `{p['paper_id']}`: {p.get('title', '?')[:80]}"
                for p in pool[:5]
            )
        )

    # 2. Phase C: fact 提取(每篇 1 次 LLM)
    # P6.0 调子验证发现 deepseek 推理模型在 enable_thinking=False 下,LLM 思考/答案
    # 都进 reasoning_content, fallback 出来的是 thinking 不是 answer。
    # survey_generate 强制走 minimax-m3 (走主界面 chat mode 容易撞这 bug)
    from agent.tools.llm import _call_llm

    _LLM_PROVIDER = "minimax"
    _LLM_MODEL = "MiniMax-M3"
    _LLM_TEMPERATURE = 0.3

    ws.set_status("fact_extraction", 0, len(pool), "")
    all_facts: List[Dict[str, Any]] = []
    for i, paper in enumerate(pool, 1):
        fact_prompt = build_fact_prompt(
            paper_id=paper["paper_id"],
            title=paper.get("title", ""),
            year=str(paper.get("year", "")),
            journal=paper.get("journal", ""),
            core_method=paper.get("core_method", ""),
            core_contrib=paper.get("core_contrib", ""),
            abstract=paper.get("abstract", ""),
            outlook=paper.get("outlook", ""),
        )
        try:
            out = _call_llm(
                fact_prompt,
                provider=_LLM_PROVIDER,
                model=_LLM_MODEL,
                enable_thinking=False,
                temperature=0.2,
                max_tokens=4000,  # P6.2 修: 2000 会被 min-max 截断(JSON 缺 "}]")
                timeout=120,
            )
        except Exception as e:
            ws.set_status(
                "fact_extraction", i, len(pool),
                f"⚠️ {paper['paper_id']} 失败: {type(e).__name__}: {e}",
            )
            continue
        # 剥 <think>
        if out and "<think>" in out:
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
        # parse JSON(容错: LLM 经常缺 '"}]' 收尾)
        m = re.search(r"\[[\s\S]*", out or "")
        if not m:
            ws.set_status("fact_extraction", i, len(pool), f"⚠️ {paper['paper_id']} 无 JSON")
            continue
        facts = _safe_json_loads_array(m.group(0))
        if facts is None:
            ws.set_status(
                "fact_extraction", i, len(pool),
                f"⚠️ {paper['paper_id']} JSON parse 失败 (含容错)",
            )
            continue
        # 落 _facts.jsonl
        for f in facts:
            if not isinstance(f, dict):
                continue
            f.setdefault("paper_id", paper["paper_id"])
            ws.append_fact(f)
            all_facts.append(f)
        ws.set_status(
            "fact_extraction", i, len(pool),
            f"{paper['paper_id']} → {len(facts)} facts",
        )

    # 3. Phase D: 大纲
    ws.set_status("outline", 0, 1, "")
    outline_prompt = build_outline_prompt(topic, pool)
    try:
        outline_raw = _call_llm(
            outline_prompt,
            provider=_LLM_PROVIDER,
            model=_LLM_MODEL,
            enable_thinking=False,
            temperature=0.3,
            max_tokens=3000,  # P6.2 修: 2000 不够 5 章节 outline
            timeout=120,
        )
    except Exception as e:
        ws.set_status("failed", 0, 0, f"outline failed: {e}")
        return f"❌ 大纲生成失败: {type(e).__name__}: {e}"
    if outline_raw and "<think>" in outline_raw:
        outline_raw = re.sub(r"<think>.*?</think>", "", outline_raw, flags=re.DOTALL).strip()
    m = re.search(r"\{[\s\S]*\}", outline_raw or "")
    if not m:
        ws.set_status("failed", 0, 0, "outline no JSON")
        return "❌ 大纲 LLM 输出不是 JSON 对象, 检查 prompt 或模型"
    try:
        outline = json.loads(m.group(0))
    except Exception as e:
        ws.set_status("failed", 0, 0, f"outline JSON parse: {e}")
        return f"❌ 大纲 JSON parse 失败: {e}"
    ws.save_outline(outline)
    sections = outline.get("sections", [])
    ws.set_status("outline", 1, 1, f"sections={len(sections)}")
    if not sections:
        return "❌ 大纲 LLM 输出 sections 字段为空"

    # 4. Phase E: 章节生成(每章 1 次 LLM + 落库)
    ws.set_status("section_generation", 0, len(sections), "")
    theme_id = create_theme(
        name=topic,
        description=f"P6 survey_generate 自动创建 at {datetime.now().isoformat(timespec='seconds')}",
    )
    section_results: List[Dict[str, Any]] = []
    pool_ids = {p["paper_id"] for p in pool}

    for j, sec in enumerate(sections, 1):
        sec_type = (sec.get("section_type") or "intro").strip()
        sec_title = sec.get("title") or f"section {j}"
        # 兼容 LLM 偶尔写 sub_point 单数
        sub_points = sec.get("sub_points") or sec.get("sub_point") or []
        if isinstance(sub_points, list):
            outline_section_md = "\n".join(f"- {sp}" for sp in sub_points)
        else:
            outline_section_md = str(sub_points)

        ws.set_status(
            "section_generation", j - 1, len(sections),
            f"[{sec_type}] {sec_title}",
        )

        section_prompt = build_section_prompt(
            topic=topic,
            section_type=sec_type,
            section_title=sec_title,
            outline_section=outline_section_md,
            papers=pool,
        )

        # P6.3 强化: 最多自动重写 2 次 (out_of_pool > 0 时)
        sec_md = ""
        bucket = {"in_pool": [], "out_of_pool": [], "total_cites": 0}
        MAX_REWRITE = 2
        for rewrite_i in range(MAX_REWRITE + 1):  # 0 = 首次, 1-2 = 重写
            try:
                sec_md = _call_llm(
                    section_prompt,
                    provider=_LLM_PROVIDER,
                    model=_LLM_MODEL,
                    enable_thinking=False,
                    temperature=0.4,
                    max_tokens=6000,  # P6.2 修: 4000 会截断 progress/method_compare 章节
                    timeout=180,
                )
            except Exception as e:
                sec_md = f"⚠️ 章节生成失败: {type(e).__name__}: {e}\n\n(占位章节: {sec_title})"
                break  # 异常不重试
            if sec_md and "<think>" in sec_md:
                sec_md = re.sub(r"<think>.*?</think>", "", sec_md, flags=re.DOTALL).strip()

            bucket = _audit_section_paper_ids(sec_md, pool_ids)
            if not bucket["out_of_pool"] or rewrite_i >= MAX_REWRITE:
                if rewrite_i > 0:
                    ws.set_status(
                        "section_generation", j - 1, len(sections),
                        f"[{sec_type}] {sec_title} 重写 {rewrite_i} 次后 out_of_pool={len(bucket['out_of_pool'])}",
                    )
                break  # 干净或达上限, 跳出
            # out_of_pool > 0 且还能重写: 加约束再试一次
            extra_constraint = (
                f"\n\n【P6.3 自动重写 {rewrite_i + 1}/{MAX_REWRITE}】"
                f"\n⚠️ 你上一次的输出引用了候选池外的 paper_id ({bucket['out_of_pool']}),"
                f"\n这会被自动审计标 ❌ (幻觉风险). 现在重写, **只能用候选池内**的 paper_id:"
                f"\n候选池 paper_id 列表: {sorted(pool_ids)}"
                f"\n请删除/替换所有候选池外引用, 然后再 output 完整章节 markdown。"
            )
            section_prompt = section_prompt + extra_constraint
            ws.set_status(
                "section_generation", j - 1, len(sections),
                f"[{sec_type}] {sec_title} out_of_pool={len(bucket['out_of_pool'])} → 自动重写 {rewrite_i + 1}",
            )

        # 落 surveys.db(注意: paper_ids 字段是 "逗号分隔" 字符串, plan §9.2 简化版)
        # order_idx 由 create_section 自动算 max+1, 不传
        sec_id = create_section(
            theme_id=theme_id,
            section_type=sec_type,
            title=sec_title,
            content_md=sec_md or "",
            paper_ids=",".join(bucket["in_pool"]),
        )
        section_results.append({
            "section_id": sec_id,
            "title": sec_title,
            "type": sec_type,
            "in_pool": bucket["in_pool"],
            "out_of_pool": bucket["out_of_pool"],
        })
        ws.set_status(
            "section_generation", j, len(sections),
            f"[{sec_type}] {sec_title} done (引用 {len(bucket['in_pool'])} 内, {len(bucket['out_of_pool'])} 外)",
        )

    # 5. Phase F: audit
    ws.set_status("audit", 0, 1, "")
    audit_lines = [
        f"# 覆盖度审计 — {topic}",
        "",
        f"- workspace: `{ws.path}`",
        f"- 候选池: {len(pool)}",
        f"- 提取 fact: {len(all_facts)}",
        f"- 章节: {len(section_results)} (theme_id={theme_id})",
        "",
        "## 每章节引用统计",
        "",
        "| 章节 | 类型 | 候选池内 | 候选池外(幻觉) |",
        "|---|---|---|---|",
    ]
    cited_ids: set = set()
    for r in section_results:
        audit_lines.append(
            f"| {r['title']} | {r['type']} | "
            f"{len(r['in_pool'])} | {len(r['out_of_pool'])} |"
        )
        cited_ids.update(r["in_pool"])
    uncited = pool_ids - cited_ids
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
        f"theme_id={theme_id}, sections={len(section_results)}, audit=ok",
    )

    # 6. 返 markdown 报告
    report = [
        f"## survey_generate 完成 — {topic}",
        "",
        f"- workspace: `{ws.path}`",
        f"- theme_id: `{theme_id}` (surveys.db survey_theme.id)",
        f"- 候选池: {len(pool)} 篇",
        f"- 提取 fact: {len(all_facts)} 条",
        f"- 章节: {len(section_results)} 段",
        f"- 覆盖度审计: `data/surveys/working/{topic_hash}/_audit.md`",
        "",
        "### 章节(section_id, 落库 surveys.db)",
        "",
    ]
    for r in section_results:
        # 2026-09-05 P0 fix: paper_id 用 [[xxx]] 双括号格式, 视觉上明确"引用 token"
        # 之前用 `xxx` 容易跟普通字符混淆
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


