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


@register_tool(name="survey_generate", is_streaming=True, cost_tier="L2")
def survey_generate(
    topic: str,
    max_steps: int = 20,
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

    # 1. Step 0: Phase A 初始候选池 (0 LLM, summary_items.db + arxiv fallback)
    pool = _build_candidate_pool(topic)
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
    state: Dict[str, Any] = {
        "topic": topic,
        "topic_hash": topic_hash,
        "pool": pool,
        "facts": [],
        "sections": [],
        "outline": None,                # 综述大纲原始 LLM 输出 (str)
        "outline_sections": [],         # 大纲结构化: [{title, section_type, sub_points, is_transition}, ...]
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

        yield f"💭 思考: {decision['thought']}\n"
        args_str = json.dumps(decision.get("args", {}), ensure_ascii=False)
        yield f"➡️ 行动: {decision['action']}({args_str})\n"

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
    """
    from agent.tools.llm import _call_llm
    from agent.prompts_survey import build_react_prompt

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    fallback = _state_driven_decide(state)

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
    """
    pool_size = len(state["pool"])
    fact_count = len(state["facts"])
    section_count = len(state["sections"])
    outline_sections = state.get("outline_sections", [])

    # ========== 候选池硬规则 ==========
    if pool_size < 20:
        return {"action": "search_more", "args": {"query": state["topic"], "k": 5}}
    if pool_size >= 40 and fact_count < 5:
        return {
            "action": "extract_facts",
            "args": {"paper_ids": [p["paper_id"] for p in state["pool"][:5]]},
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

    yield f"🔎 搜索 arxiv: query={query!r}, k={k}\n"

    # 1. arxiv API 抓
    try:
        raw_papers = _arxiv_search(query, k)
    except Exception as e:
        friendly = _friendly_error(e, "search_more.arxiv")
        state["errors"].append({
            "step": state["step"], "action": "search_more.arxiv",
            "msg": friendly,
        })
        yield f"❌ arxiv 搜索失败: {friendly}\n"
        return

    if not raw_papers:
        yield "⚠️ arxiv 返回 0 篇, 候选池无新增\n"
        yield f"👁️ 观察: search_more 0 篇新增, 累计 pool={len(state['pool'])} 篇\n"
        return

    yield f"📥 arxiv 返回 {len(raw_papers)} 篇, 开始单篇 summarize (调 {_PROVIDER}/{_MODEL})\n"

    # 2. 单篇 summarize + 写库
    new_pool = []
    for i, p in enumerate(raw_papers, 1):
        paper_id = p["paper_id"]
        yield f"  [{i}/{len(raw_papers)}] summarize {paper_id}: {p.get('title', '?')[:60]}...\n"
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
            row_id = save_summary_item(
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
            if row_id == -1:
                yield f"    ⚠️ UNIQUE 冲突 (paper={paper_id}), 跳过\n"
            else:
                yield f"    ✅ 入库 (row_id={row_id})\n"

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
            # 单篇失败不阻塞 batch (spec §7.2 D)
            friendly = _friendly_error(e, "search_more.summarize")
            state["errors"].append({
                "step": state["step"], "action": "search_more.summarize",
                "paper": paper_id, "msg": friendly,
            })
            yield f"    ❌ {friendly}\n"

    state["pool"].extend(new_pool)
    yield (
        f"👁️ 观察: 抓 {len(raw_papers)} 篇, 成功 {len(new_pool)} 篇入库, "
        f"累计 pool={len(state['pool'])} 篇, errors={len(state['errors'])}\n"
    )


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
        yield f"  [{i}/{len(targets)}] {paper_id} 抽 fact...\n"
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
            yield f"    ✅ 抽 {len(facts)} 条 fact\n"
        except Exception as e:
            # 单篇失败不阻塞 batch
            friendly = _friendly_error(e, "extract_facts")
            state["errors"].append({
                "step": state["step"], "action": "extract_facts",
                "paper": paper_id, "msg": friendly,
            })
            yield f"    ❌ {friendly}\n"

    yield f"👁️ 观察: 抽 {len(targets)} 篇, 累计 facts={len(state['facts'])} 条, errors={len(state['errors'])}\n"


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
        yield (
            f"⚠️ write_outline: 大纲已存在 ({len(state['outline_sections'])} 章), 跳过\n"
            f"👁️ 观察: outline_sections={len(state['outline_sections'])}, sections={len(state['sections'])}\n"
        )
        return

    yield f"📋 write_outline: 生成综述大纲, 调 {_PROVIDER}/{_MODEL}\n"

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
    normalized = []
    for sec in parsed["sections"]:
        if not isinstance(sec, dict):
            continue
        title = (sec.get("title") or "").strip()
        sec_type = (sec.get("section_type") or "").strip().lower()
        if not title or sec_type not in SECTION_REF_QUOTAS:
            continue  # 跳过非法 section (避免污染 outline)
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

    if not normalized:
        state["errors"].append({
            "step": state["step"], "action": "write_outline",
            "msg": "规范化后 sections 为空 (全部非法, section_type 不在 5 段列表)",
        })
        yield f"  ❌ 大纲规范化失败, section_type 必须从 intro/progress/method_compare/gap/outlook 选\n"
        return

    # 存 state + 落盘
    state["outline"] = out or ""
    state["outline_sections"] = normalized
    try:
        ws.save_outline({"sections": normalized, "raw": out or ""})
    except Exception as e:
        logger.warning("[write_outline] save_outline 失败: %s", e)

    yield f"  ✅ 大纲生成: {len(normalized)} 章\n"
    for i, sec in enumerate(normalized):
        yield f"    {i}. [{sec['section_type']}] {sec['title']} ({len(sec['sub_points'])} 子点)\n"
    yield f"💾 落 workspace/_outline.json\n"
    yield (
        f"👁️ 观察: outline_sections={len(normalized)}, "
        f"pool={len(state['pool'])}, facts={len(state['facts'])}, sections={len(state['sections'])}\n"
    )


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

    yield f"✍️ write_section: [{sec_type}] {title} (section_idx={section_idx}/{len(outline_sections)-1})\n"

    # theme: 第一次创建, 后续复用
    if "theme_id" not in state:
        state["theme_id"] = create_theme(
            name=state["topic"],
            description=f"P6.5 survey_generate 自动创建 at {datetime.now().isoformat(timespec='seconds')}",
        )
        yield f"  📌 新建 theme (theme_id={state['theme_id']})\n"
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
                # 重写时加约束: 只能用候选池内 paper_id
                extra_constraint = (
                    f"\n\n【P6.3 自动重写 {rewrite_i}/{MAX_REWRITE}】"
                    f"\n⚠️ 你上一次的输出引用了候选池外的 paper_id ({bucket['out_of_pool']}),"
                    f"\n这会被自动审计标 ❌ (幻觉风险). 现在重写, **只能用候选池内**的 paper_id:"
                    f"\n候选池 paper_id 列表: {sorted(pool_ids)}"
                    f"\n请删除/替换所有候选池外引用, 然后再 output 完整章节 markdown。"
                )
                section_prompt = section_prompt + extra_constraint
                yield f"  🔁 out_of_pool={len(bucket['out_of_pool'])} → 重写 {rewrite_i}/{MAX_REWRITE}\n"

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
            # 异常不重试
            sec_md = f"⚠️ 章节生成失败: {_friendly_error(e, 'write_section')}\n\n(占位章节: {title})"
            state["errors"].append({
                "step": state["step"], "action": "write_section",
                "msg": _friendly_error(e, "write_section"),
            })
            yield f"  ❌ LLM 失败: {_friendly_error(e, 'write_section')}\n"
            break

        # 剥 <think>
        if sec_md and "<think>" in sec_md:
            sec_md = re.sub(r"<think>.*?</think>", "", sec_md, flags=re.DOTALL).strip()

        # 审计引用
        bucket = _audit_section_paper_ids(sec_md, pool_ids)
        if not bucket["out_of_pool"] or rewrite_i >= MAX_REWRITE:
            if rewrite_i > 0:
                yield f"  ✅ 重写 {rewrite_i} 次后 out_of_pool={len(bucket['out_of_pool'])}\n"
            break
        # out_of_pool > 0 且还能重写

    # 章节内容整体 yield (非 token 流式, spec §11 开放问题 3)
    yield f"\n--- [{sec_type}] {title} ---\n"
    for line in sec_md.split("\n"):
        yield line + "\n"
    yield "--- 章节结束 ---\n"

    # 落 surveys.db
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
            "title": title,
            "type": sec_type,
            "in_pool": bucket["in_pool"],
            "out_of_pool": bucket["out_of_pool"],
        })
        yield (
            f"  💾 落 surveys.db (section_id={sec_id}, "
            f"引用 {len(bucket['in_pool'])} 内, {len(bucket['out_of_pool'])} 外)\n"
        )
    except Exception as e:
        friendly = _friendly_error(e, "create_section")
        state["errors"].append({
            "step": state["step"], "action": "write_section.create_section",
            "msg": friendly,
        })
        yield f"  ❌ 落库失败: {friendly}\n"

    yield f"👁️ 观察: write_section 完成, 已写 {len(state['sections'])} 章, errors={len(state['errors'])}\n"


def action_finish(args: Dict[str, Any], state: Dict[str, Any], ws) -> Iterator[str]:
    """finish action — 标记收尾, 主循环下次 break"""
    state["finish_requested"] = True
    yield "👁️ 观察: 🏁 收尾中...\n"


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


