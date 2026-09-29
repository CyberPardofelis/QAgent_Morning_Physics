"""
agent.tools.surveys._organize — fact 优先的综述组织 (9-29 新增)

核心主张 (来自用户洞察):
    真实的人写综述是**以信息为锚点**, 不是以文章为锚点。
    顺序是: 收集 fact → **把 fact 归入主题、看它们之间什么关系** → 围绕主题写叙述。
    中间那步(主题归类)才是智识劳动本身; 直接"给一堆论文写个章节"必然变成论文摘要。

因此这里补上被跳过的一步, 把它变成一个**有产物**的阶段:

    Phase B  organize_facts_into_themes()  → _thematic_map.json
             输入: 全部 fact
             输出: 主题列表, 每个主题有
                     - statement   该主题的**领域判断**(不是"哪些论文做了什么")
                     - fact_ids    归属 fact(只给 fact_id, 不给 paper_id)
                     - relations   与其他主题的关系(support/tension/evolution/contrast)
                     - notes       主题内部的逻辑线索
             外加: narrative_arc(推荐叙述顺序) + gaps(文献覆盖不足处)

    Phase C  write_theme_narrative()        → 按主题写叙述
             锚点是主题, fact 是证据。引文由 fact_id 机械映射 —— 模型碰不到 paper_id。

与旧做法的关键区别 —— 允许什么 / 禁止什么:

    禁止: 引入 fact 里没有的**经验内容**(数字、装置、平台、方法名)
          ← 旧版综述的编造引用全部出在这里
    要求: 做**分析**(fact 之间的关系、分类、对照、张力、领域为何这样演化)
          ← 这正是"综述区别于论文摘要"的东西, 旧 prompt 把它一起禁掉了,
            结果综述既短又单薄

模型只输出 fact_id, 一律经 _fact_key 校验; 校验不过的 id 直接丢弃,
不会被换算成假引用(见 _render_citations)。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# Phase B: 主题地图
# ============================================================

THEME_MAP_PROMPT = """你是一位资深研究者在通读一批文献**笔记**后, 整理领域结构。

下面是从文献中抽取的原子事实笔记(每条带一个 fact_id)。你的任务**不是**总结这些
论文, 而是把这些事实**组织成一个领域的结构** —— 也就是回答: "这个领域由哪几个
子问题构成, 它们之间是什么关系"。

【铁律】
1. **只能使用下面给出的 fact_id**。不要发明 id。
2. 每个主题的 statement 必须是一句**领域判断**(对领域的认识/分类/趋势/矛盾),
   而不是"A 等人提出 X, B 等人改进了 Y"这种论文列举。
3. 关系类型只能用这四种: support(互相支持) / tension(存在张力或矛盾) /
   evolution(演化/前后代关系) / contrast(适用范围或前提不同)。
4. 如果某几条 fact 其实说的是同一件事的不同侧面, 把它们放**同一个主题**而不是拆开。
5. 宁可 4-6 个扎实的主题, 也不要 12 个碎主题。覆盖面不足的地方写进 gaps, 不要硬凑。

【fact 笔记】{facts_block}

【输出格式】严格 JSON, 不要 markdown 包裹:
{{
  "themes": [
    {{
      "name": "主题名(12 字以内)",
      "statement": "一句领域判断, 不是论文列举",
      "fact_ids": ["fact_id_1", "fact_id_2"],
      "notes": "这个主题内部的逻辑线索(2-3 句, 说明这些 fact 如何串起来)",
      "relations": [
        {{"to": "另一个主题名", "type": "support|tension|evolution|contrast", "note": "一句话"}}
      ]
    }}
  ],
  "narrative_arc": ["建议先讲的主题名", "...", "..."],
  "gaps": ["文献覆盖不足、值得后续补充的方向"]
}}

只输出 JSON。"""


def _facts_block(facts: list[dict[str, Any]], paper_titles: dict[str, str]) -> str:
    out = []
    for f in facts:
        fid = f.get("fact_id", "?")
        ftype = f.get("fact_type", "?")
        claim = (f.get("claim") or "").strip()
        out.append(f"[{fid}] ({ftype}) {claim}")
        # 原文只留给需要时的核对, 这里不展开以免挤占结构化任务
    return "\n".join(out)


def organize_facts_into_themes(
    facts: list[dict[str, Any]],
    paper_titles: dict[str, str] | None = None,
    provider: str = "minimax",
    model: str = "MiniMax-M3",
    temperature: float = 0.4,
    max_themes: int = 8,
) -> dict[str, Any]:
    """
    Phase B: 把 fact 组织成主题地图。

    返回 {"themes": [...], "narrative_arc": [...], "gaps": [...], "warnings": [...]}
    校验不通过的 fact_id 会被丢弃并记进 warnings(不静默)。
    """
    from agent.tools.llm import _call_llm

    paper_titles = paper_titles or {}
    if not facts:
        return {"themes": [], "narrative_arc": [], "gaps": [], "warnings": ["没有 fact"]}

    valid_ids = {f.get("fact_id") for f in facts if f.get("fact_id")}
    prompt = THEME_MAP_PROMPT.format(facts_block=_facts_block(facts, paper_titles))

    out = _call_llm(prompt, provider=provider, model=model,
                    enable_thinking=False, temperature=temperature,
                    max_tokens=8000, timeout=300)
    if not out:
        return {"themes": [], "narrative_arc": [], "gaps": [], "warnings": ["LLM 返回空"]}
    if "<think>" in out:
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()

    import json as _json
    parsed = None
    m = re.search(r"\{[\s\S]*\}", out)
    if m:
        try:
            v = _json.loads(m.group(0))
            if isinstance(v, dict) and "themes" in v:
                parsed = v
        except Exception:
            parsed = None
    if parsed is None:
        return {"themes": [], "narrative_arc": [], "gaps": [],
                "warnings": [f"主题地图 JSON 解析失败: {out[:200]}"]}

    warnings: list[str] = []
    themes: list[dict[str, Any]] = []
    seen_theme: set[str] = set()
    for t in parsed.get("themes") or []:
        if not isinstance(t, dict):
            continue
        name = (t.get("name") or "").strip()
        if not name:
            warnings.append("主题缺 name, 已跳过")
            continue
        if name in seen_theme:
            warnings.append(f"主题名重复: {name}")
            continue
        seen_theme.add(name)
        # fact_id 校验 —— 模型编的 id 直接丢
        raw_ids = t.get("fact_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        kept, bad = [], 0
        for fid in raw_ids:
            fid = str(fid).strip()
            if fid in valid_ids:
                kept.append(fid)
            else:
                bad += 1
        if bad:
            warnings.append(f"主题「{name}」有 {bad} 个无效 fact_id, 已丢弃")
        if not kept:
            warnings.append(f"主题「{name}」无有效 fact, 已跳过")
            continue
        rels = []
        for r in t.get("relations") or []:
            if isinstance(r, dict) and (r.get("to") or "").strip():
                rels.append({
                    "to": (r.get("to") or "").strip(),
                    "type": (r.get("type") or "support").strip(),
                    "note": (r.get("note") or "").strip(),
                })
        themes.append({
            "name": name,
            "statement": (t.get("statement") or "").strip(),
            "fact_ids": kept,
            "notes": (t.get("notes") or "").strip(),
            "relations": rels,
        })

    if len(themes) > max_themes:
        warnings.append(f"主题数 {len(themes)} 超过上限 {max_themes}, 已截断")
        themes = themes[:max_themes]

    assigned = {fid for t in themes for fid in t["fact_ids"]}
    unused = sorted(valid_ids - assigned)
    if unused:
        warnings.append(f"{len(unused)} 条 fact 未被任何主题覆盖")

    return {
        "themes": themes,
        "narrative_arc": [x for x in (parsed.get("narrative_arc") or []) if isinstance(x, str)],
        "gaps": [x for x in (parsed.get("gaps") or []) if isinstance(x, str)],
        "warnings": warnings,
    }


# ============================================================
# Phase C: 按主题写叙述
# ============================================================

THEME_WRITE_PROMPT = """你在为一个综述的「{theme_name}」小节写正文。

【本小节的领域判断】{theme_statement}

【小节内部的逻辑线索】
{theme_notes}

【与其他小节的关系】
{theme_relations}

【可用事实】(只列本小节的 fact, 带 fact_id)
{facts_block}

【写什么】
- 围绕上面的**领域判断**组织叙述, 而不是逐篇介绍论文。
- 【必须做】在事实之间建立联系: 它们如何互相支持、在什么条件下结论相反、
  技术路线为何沿着这个方向演化、哪些是共识哪些是分歧。
  这一层分析就是综述区别于论文摘要的地方, 缺了它就只是摘要。
- 【禁止】引入上面没有的**经验内容** —— 不要补充装置平台、测量方法名、
  具体数字或论文里没有提到的技术名称。你没有其他信息来源。
  (允许做分析/比较/归纳/解释因果; 不允许凭空提供事实。)
- 【引用】每个陈述性句子末尾标 [[fact_id]], 只能取上面列出的。
  没有 fact 支撑的话不要说。
- 中文, 学术综述文体, 直接输出正文, 不要标题, 不要前后缀。

写 {n_para} 个自然段。"""


def _render_citations(text: str, fact2pid: dict[str, str]) -> str:
    """
    fact_id → paper_id 机械映射。模型给不了 paper_id, 编造的 fact_id 直接丢标注。
    """
    def sub(m):
        pids = []
        for part in m.group(1).split(","):
            fid = part.strip()
            pid = fact2pid.get(fid)
            if pid and pid not in pids:
                pids.append(pid)
        return "".join(f"[{p}]" for p in pids)

    text = re.sub(r"\[\[([^[\]]+)\]\]", sub, text)
    text = re.sub(r"\[\s*\]", "", text)
    # 合并连续重复的同一引用
    text = re.sub(r"(\[[^\]\s]+\])(?:\s*\1)+", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


_SENTENCE_SPLIT = re.compile(r"(?<=[。!?；;])\s+")


def _drop_unattributed(text: str) -> tuple[str, int]:
    """
    丢掉**没有任何 fact 标注**的句子 —— 无出处的断言正是要避免的东西。
    比整块丢弃温和: 模型部分配合时仍能保住大部分正文。
    """
    tag = re.compile(r"\[\[[^[\]]+\]\]")
    kept, dropped = [], 0
    for para in text.split("\n\n"):
        lines = []
        for line in para.split("\n"):
            if not line.strip() or tag.search(line):
                lines.append(line)
            else:
                dropped += 1
        c = "\n".join(lines).strip()
        if c:
            kept.append(c)
    return "\n\n".join(kept), dropped


def write_theme_narrative(
    theme: dict[str, Any],
    facts: list[dict[str, Any]],
    provider: str = "minimax",
    model: str = "MiniMax-M3",
    n_para: int = 3,
    temperature: float = 0.4,
    max_retry: int = 1,
) -> dict[str, Any]:
    """
    Phase C: 围绕一个主题写叙述。返回
        {"text": str, "used_fact_ids": [...], "dropped_sentences": int, "warnings": [...]}
    """
    from agent.tools.llm import _call_llm

    fmap = {f.get("fact_id"): f for f in facts if f.get("fact_id")}
    theme_facts = [fmap[fid] for fid in theme.get("fact_ids", []) if fid in fmap]
    if not theme_facts:
        return {"text": "", "used_fact_ids": [], "dropped_sentences": 0,
                "warnings": ["主题没有可用的 fact"]}

    fact2pid = {f["fact_id"]: f.get("paper_id", "") for f in theme_facts}
    lines = []
    for f in theme_facts:
        claim = (f.get("claim") or "").strip()
        quote = (f.get("support_quote") or "").strip()
        lines.append(f"[{f['fact_id']}] ({f.get('fact_type','?')}) {claim}")
        if quote:
            lines.append(f"    原文: {quote[:180]}")
    facts_block = "\n".join(lines)

    rels = theme.get("relations") or []
    rel_txt = "\n".join(
        f"- 与「{r.get('to')}」是 {r.get('type')} 关系: {r.get('note')}"
        for r in rels if r.get("to")
    ) or "(无)"

    prompt = THEME_WRITE_PROMPT.format(
        theme_name=theme.get("name", ""),
        theme_statement=theme.get("statement", "") or "(未给出)",
        theme_notes=theme.get("notes", "") or "(无)",
        theme_relations=rel_txt,
        facts_block=facts_block,
        n_para=n_para,
    )

    warnings: list[str] = []
    out = ""
    for attempt in range(max_retry + 1):
        if attempt:
            prompt += (
                "\n\n【重要 — 上一版没有任何 fact 标注, 已被作废】\n"
                "每个陈述性句子的句末**必须**跟 [[fact_id]] 标注, id 只能取上面列出的。\n"
                "没有 fact 支撑的话就不要写那一句。"
            )
        out = _call_llm(prompt, provider=provider, model=model,
                        enable_thinking=False, temperature=temperature,
                        max_tokens=8000, timeout=300)
        if not out:
            continue
        if "<think>" in out:
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        out = out.replace("```markdown", "").replace("```", "").strip()
        if re.search(r"\[\[", out):
            break
    else:
        warnings.append(f"主题「{theme.get('name')}」未能建立 fact 对应, 已跳过")

    if not out:
        return {"text": "", "used_fact_ids": [], "dropped_sentences": 0, "warnings": warnings}

    kept, dropped = _drop_unattributed(out)
    text = _render_citations(kept, fact2pid)
    used = sorted({x.strip() for m in re.findall(r"\[\[([^[\]]+)\]\]", kept)
                   for x in m.split(",") if x.strip() in fact2pid})
    if dropped:
        warnings.append(f"{dropped} 句因无 fact 标注被剔除")
    return {"text": text, "used_fact_ids": used,
            "dropped_sentences": dropped, "warnings": warnings}


# ============================================================
# 组装
# ============================================================

def render_survey(
    theme_map: dict[str, Any],
    narratives: dict[str, dict[str, Any]],
    title: str = "",
    subtitle: str = "",
) -> str:
    """把主题地图 + 各主题叙述拼成综述 markdown。"""
    parts: list[str] = []
    if title:
        parts.append(f"# {title}")
    if subtitle:
        parts.append(f"> {subtitle}")
    arc = theme_map.get("narrative_arc") or []
    order = [t["name"] for t in theme_map.get("themes", [])]
    if arc:
        rank = {n: i for i, n in enumerate(arc)}
        order = sorted(order, key=lambda n: (rank.get(n, len(rank)), order.index(n)))
    for name in order:
        theme = next((t for t in theme_map["themes"] if t["name"] == name), None)
        if not theme:
            continue
        nar = narratives.get(name) or {}
        text = (nar.get("text") or "").strip()
        if not text:
            continue
        head = f"## {name}"
        if theme.get("statement"):
            head += f"\n\n*{theme['statement']}*"
        parts.append(head + "\n\n" + text)
    gaps = theme_map.get("gaps") or []
    if gaps:
        parts.append("## 文献覆盖的不足\n\n" + "\n".join(f"- {g}" for g in gaps))
    return "\n\n".join(parts) + "\n"


def save_thematic_map(ws, theme_map: dict[str, Any]) -> None:
    """主题地图落盘(可编辑, 像大纲一样)。"""
    import time as _t
    p = ws.path / "_thematic_map.json"
    payload = dict(theme_map)
    payload["_saved_at"] = _t.strftime("%Y-%m-%dT%H:%M:%S")
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_thematic_map(ws) -> dict[str, Any] | None:
    p = ws.path / "_thematic_map.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
