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

from agent.prompts_survey import _tokenize

logger = logging.getLogger(__name__)


# ============================================================
# Phase B: 主题地图
# ============================================================

def cluster_by_subject(
    facts: list[dict[str, Any]],
    min_count: int | None = None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """
    按 subject 把 fact 聚成簇。

    为什么这一步有用 (9-29):
        Phase B 原来把**全部** fact 平铺给 LLM 让它自己归纳主题。那是"让模型
        凭印象总结语料结构", 换个时间跑结果就会飘。

        先按 subject 聚簇, 等于先给出一个**客观的语料分布** —— 哪些对象被
        反复讨论、各有多少条 fact。LLM 是在真实分布上归纳, 不是凭印象。
        而且 subject 来自 fact 抽取阶段, 跨论文对齐过(同一对象用同一字符串),
        所以"这个领域分成哪几块"这件事本身就成了可复用的资产,
        下次写同主题综述不用重新归纳。

        **subject 是提示, 不是硬约束** —— LLM 仍可跨 subject 组合主题
        (真实的领域结构经常横跨几个对象), 只是不被单点分布带偏。

    min_count: 少于这个条数的 subject 并入"其他"。**自适应语料规模** ——
        9-27 实测: 10 篇主题各异的论文, 多数 subject 本就只出现 1 次,
        固定 min_count=2 会把 29/67 条全丢进"其他", 分布失去意义。
        语料越小, 门槛越低(小到只剩 3-4 篇时, 出现 1 次也值得单独看)。
    Returns: [(subject, facts), ...] 按簇大小降序。
    """
    n = len(facts)
    if min_count is None:
        # 语料小的时候(几十条), 多数 subject 本来就只出现 1 次 ——
        # 那不是噪声, 那就是"这几篇论文各讲各的"这个事实本身。
        # 门槛太高会把整个分布压成一个"其他", Phase B 就看不到结构了。
        min_count = 1 if n < 120 else (2 if n < 300 else 3)

    buckets: dict[str, list[dict[str, Any]]] = {}
    no_subject: list[dict[str, Any]] = []
    bad_subject: list[dict[str, Any]] = []
    for f in facts:
        subj = (f.get("subject") or "").strip()
        if not subj:
            no_subject.append(f)
            continue
        # 9-27: subject 校验不通过的(自指/超长/过短)不适合当聚类键 ——
        # 放进簇里会污染分布, 单独归一类并标出来
        if f.get("_subject_ok", True) is False:
            bad_subject.append(f)
            continue
        buckets.setdefault(subj, []).append(f)

    if not buckets:
        return [("其他/未归类", list(facts))] if facts else []

    big = [(s, fs) for s, fs in buckets.items() if len(fs) >= min_count]
    small = [f for s, fs in buckets.items() if len(fs) < min_count for f in fs]
    big.sort(key=lambda t: (-len(t[1]), t[0]))
    # 9-27: 没有 subject / subject 不合格的都要进"其他", 否则 histogram 的总数
    # 跟实际 fact 数对不上, LLM 会困惑("你说 39 条, 我看到 65 条")
    small.extend(no_subject)
    small.extend(bad_subject)
    if small:
        big.append(("其他/未归类", small))
    return big


def _subject_histogram(clusters, top_k: int = 15) -> str:
    """
    把 subject 聚类渲染成给 LLM 看的分布表。

    只列前 top_k 个簇, 其余汇总成一行 —— 40 多个只出现 1 次的对象全列出来
    会把 prompt 撑爆, 而它们对"这个领域由哪几块构成"这个问题贡献不大。
    但**总数必须守恒**(汇总行写明被折叠的簇数与 fact 数), 否则 LLM 看到的
    数字和它手里的 fact 数对不上, 会困惑。
    """
    if not clusters:
        return "(fact 没有 subject 字段, 无法按对象聚类)"
    lines = []
    total = sum(len(fs) for _s, fs in clusters)
    for subj, fs in clusters[:top_k]:
        n_papers = len({f.get("paper_id") for f in fs})
        lines.append(f"- {subj}: {len(fs)} 条 fact, 来自 {n_papers} 篇不同论文")
    rest = clusters[top_k:]
    if rest:
        n_rest_facts = sum(len(fs) for _s, fs in rest)
        lines.append(
            f"- (其余 {len(rest)} 个只覆盖少量 fact 的对象, 合计 {n_rest_facts} 条, 已折叠)"
        )
    lines.append(f"(共 {total} 条 fact, {len(clusters)} 个对象簇)")
    return "\n".join(lines)


THEME_MAP_PROMPT = """你是一位资深研究者在通读一批文献**笔记**后, 整理领域结构。

下面是从文献中抽取的原子事实笔记(每条带一个 fact_id)。你的任务**不是**总结这些
论文, 而是把这些事实**组织成一个领域的结构** —— 也就是回答: "这个领域由哪几个
子问题构成, 它们之间是什么关系"。

【语料分布 — 客观数据, 先看这个】
下面的簇是按 fact 的 `subject`(核心对象)自动聚的, 反映了这批文献**实际**在讨论什么。
{n_subject_histogram}

**这是提示, 不是硬约束**: 真实的领域结构经常横跨几个对象(比如"随机测量"和
"快照"常被合在一起讨论), 你可以跨簇组合主题。但它比"凭印象总结"可靠 ——
如果你归纳出的主题和这个分布严重脱节, 说明你漏读了材料。

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



def flag_subject_variants(
    facts: list[dict[str, Any]],
    min_overlap: float = 0.6,
) -> list[dict[str, Any]]:
    """
    找出**可能**是同一个对象的 subject 变体, 打标记但**不合并**。

    背景 (9-29):
        subject 靠 LLM 在抽取时自己填, 没有跨篇校验。结果就是同一对象出现
        多种叫法 —— "快照" / "classical shadow" / "经典阴影" / "阴影层析"
        会被 `cluster_by_subject` 拆成 4 个簇, Phase B 看到的是虚假的分布。

        确定性手段抓不全(同义是语义问题), 但能抓明显的:
        token 重叠高 / 一方是另一方的子串。抓到的**只报告不合并** ——
        与 fact 库"只追加不合并"一致, 合并判断错了不可逆。

    在每条 fact 上写:
        _subject_variant_group  组 id(同一组共用), 无变体则不写
        _subject_canonical      组内最长的那个 subject(作为"看起来最规范"的代表)

    Returns: [{group, canonical, members: [{subject, n_facts}]}, ...]
    """
    buckets: dict[str, list[dict]] = {}
    for f in facts:
        s = (f.get("subject") or "").strip()
        if s and f.get("_subject_ok", True):
            buckets.setdefault(s, []).append(f)
    if len(buckets) < 2:
        return []

    names = list(buckets.keys())
    # 并查集: 把"可能同一对象"的 subject 归到一组
    parent: dict[str, str] = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if _subjects_may_be_same(a, b, min_overlap):
                union(a, b)

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)

    out: list[dict[str, Any]] = []
    for _root, members in groups.items():
        if len(members) < 2:
            continue
        # 9-27 实测修正: canonical 取**最短**的, 不是最长的。
        # 之前选最长, 结果把"经典阴影与直接量子测量硬件对比"(一整个句子)
        # 当成了规范名 —— 那是句子不是对象名。
        # 短的反而更像"这个名字本身"。
        canonical = min(members, key=lambda m: (len(m), m))
        out.append({
            "group": _root,
            "canonical": canonical,
            "members": [{"subject": m, "n_facts": len(buckets[m])} for m in members],
            "note": "疑似同一对象的多种叫法, 建议统一后再聚类",
        })
    # 稳定排序: 按组大小+名字, 避免 union-find 的 root 顺序影响输出顺序
    out.sort(key=lambda g: (-len(g["members"]), g["group"]))
    # 组 id 重编号成稳定序号, **并把 fact 上的标记同步改成新 id**
    for i, g in enumerate(out):
        g["group"] = f"subjvar_{i}"
        canon = g["canonical"]
        for m in [x["subject"] for x in g["members"]]:
            for f in buckets[m]:
                f["_subject_variant_group"] = g["group"]
                f["_subject_canonical"] = canon
    return out


def _subjects_may_be_same(a: str, b: str, min_overlap: float) -> bool:
    """
    两个 subject 是否可能是同一对象。

    9-27 实测踩到的坑: 最初用 token 重叠率 `inter / min(len)` 判定,
    结果中文被严重误判 —— "经典阴影" / "费米子经典阴影" / "经典阴影层析"
    共享"经典/阴影/层析"这几个单字 token, 重叠率轻松过 0.6, 于是
    20 个明显不同的 subject 被并成了一组, Phase B 拿到的是虚假分布。

    改成两条**更严**的判据(满足其一即可):
      a) 一方是另一方的子串, 且较短方长度 >= 4
         ("经典阴影" ⊂ "经典阴影层析" -> 判为变体; 排除 "X" ⊂ 任何 X 的噪声)
      b) token 重叠率高 **且** 较短方的 token 几乎完全被包含
         (overlap >= 阈值, 且 inter / min >= 0.9 —— 几乎完全一致才认为同义)
    """
    al, bl = a.lower(), b.lower()
    if al == bl:
        return False          # 同一个桶, 不算变体
    short, long = (al, bl) if len(al) <= len(bl) else (bl, al)
    # 判据 a: 子串, 但要求短侧像个完整对象名(>=4 字符)
    if len(short) >= 4 and short in long:
        return True
    # 判据 b: token 几乎完全重合(而不是"共享几个常用字")
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    smaller = min(len(ta), len(tb))
    if smaller == 0:
        return False
    containment = inter / smaller
    return containment >= 0.9 and (inter / max(len(ta), len(tb))) >= min_overlap


def _facts_block(facts: list[dict[str, Any]], paper_titles: dict[str, str]) -> str:
    out = []
    for f in facts:
        fid = f.get("fact_id", "?")
        ftype = f.get("fact_type", "?")
        claim = (f.get("claim") or "").strip()
        out.append(f"[{fid}] ({ftype}) {claim}")
        # 原文只留给需要时的核对, 这里不展开以免挤占结构化任务
    return "\n".join(out)


def _parse_theme_json(raw: str) -> dict[str, Any] | None:
    """
    从模型输出里抠出主题地图 JSON。

    9-29 实测踩到的三种情况都要兜住:
      1. 正常: 纯 JSON
      2. ```json ... ``` 包裹
      3. 前面一大段 <think> 思考, 答案被 max_tokens 截断在 think 里
         —— 此时先剥 think 得到空串, 必须回退到**原始串**里找 JSON;
         还有一种是把 JSON 写在 <think> 和 </think> 之间(思考与答案混排)。
    """
    import json as _json

    def _try(s: str):
        if not s:
            return None
        # 注意两个坑:
        #  - 被 max_tokens 截断时可能**连右括号都没有**, 所以不能只匹配完整的 {...}
        #  - ```json ... ``` 包裹时尾部有反引号, 不能直接 loads 整段
        i = s.find("{")
        if i < 0:
            return None
        body = s[i:]
        j = body.rfind("}")
        candidates = [body] if j < 0 else [body[:j + 1], body]
        for cand in candidates:
            try:
                v = _json.loads(cand)
            except Exception:
                continue
            if isinstance(v, dict) and isinstance(v.get("themes"), list) and v["themes"]:
                return v
        # 补右括号 / 补尾引号再试(截断恢复)
        for tail in ("}", "}}", '"}}', '"}}}', '"]}}', '"', ']', '"}]', '"}}]}'):
            try:
                v = _json.loads(body + tail)
            except Exception:
                continue
            if isinstance(v, dict) and isinstance(v.get("themes"), list) and v["themes"]:
                return v
        return None

    # 1) 剥 think 后
    stripped = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    for cand in (stripped, raw or ""):
        got = _try(cand)
        if got:
            return got
    # 2) <think> ... JSON ... </think> 混排: 取每个 think 块内部
    for block in re.findall(r"<think>([\s\S]*?)</think>", raw or ""):
        got = _try(block)
        if got:
            return got
    return None


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
    # 9-29: 先按 subject 聚簇, 把客观分布给 LLM 做归纳的锚。
    # 比"平铺全部 fact 让模型自己总结"稳 —— 那是凭印象, 换次跑就飘。
    clusters = cluster_by_subject(facts)
    prompt = THEME_MAP_PROMPT.format(
        n_subject_histogram=_subject_histogram(clusters),
        facts_block=_facts_block(facts, paper_titles),
    )

    # 9-29 实测: MiniMax 即使 enable_thinking=False 也会输出大段 <think>,
    # 那部分**照样吃 max_tokens**。8000 会被思考吃光, JSON 答案被截断 -> 解析失败。
    # 主题地图是要"想清楚"的活, 给足预算。
    out = _call_llm(prompt, provider=provider, model=model,
                    enable_thinking=False, temperature=temperature,
                    max_tokens=20000, timeout=600)
    if not out:
        return {"themes": [], "narrative_arc": [], "gaps": [], "warnings": ["LLM 返回空"]}

    parsed = _parse_theme_json(out)
    if parsed is None:
        return {"themes": [], "narrative_arc": [], "gaps": [],
                "warnings": [f"主题地图 JSON 解析失败 (返回 {len(out)} 字符): {out[-300:]}"]}

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
    chunk_size: int = 8,
) -> dict[str, Any]:
    """
    Phase C: 围绕一个主题写叙述。返回
        {"text": str, "used_fact_ids": [...], "dropped_sentences": int, "warnings": [...]}

    9-29 实测补充: 一个主题的 fact 数可能到 30+ 条。一次性全塞进 prompt 时
    模型会完全忘了标注 fact_id(返回 0 字符)。所以**主题内也分批**:
    每批 chunk_size 条写若干段, 拼起来。分批后每个小批的标注服从率显著提高。
    """
    from agent.tools.llm import _call_llm

    fmap = {f.get("fact_id"): f for f in facts if f.get("fact_id")}
    theme_facts = [fmap[fid] for fid in theme.get("fact_ids", []) if fid in fmap]
    if not theme_facts:
        return {"text": "", "used_fact_ids": [], "dropped_sentences": 0,
                "warnings": ["主题没有可用的 fact"]}

    warnings: list[str] = []
    all_used: list[str] = []
    paras: list[str] = []
    dropped_total = 0
    failed_batches = 0

    batches = [theme_facts[i:i + chunk_size] for i in range(0, len(theme_facts), chunk_size)]
    for bi, batch in enumerate(batches, 1):
        fact2pid = {f["fact_id"]: f.get("paper_id", "") for f in batch}
        lines = []
        for f in batch:
            claim = (f.get("claim") or "").strip()
            quote = (f.get("support_quote") or "").strip()
            subj = (f.get("subject") or "").strip()
            head = f"[{f['fact_id']}]"
            if subj:
                head += f" <对象: {subj}>"
            head += f" ({f.get('fact_type','?')})"
            lines.append(head + " " + claim)
            if quote:
                lines.append(f"    原文: {quote[:180]}")
        facts_block = "\n".join(lines)

        prompt = THEME_WRITE_PROMPT.format(
            theme_name=theme.get("name", ""),
            theme_statement=theme.get("statement", "") or "(未给出)",
            theme_notes=theme.get("notes", "") or "(无)",
            theme_relations=(
                "\n".join(f"- 与「{r.get('to')}」是 {r.get('type')} 关系: {r.get('note')}"
                          for r in (theme.get("relations") or []) if r.get("to"))
                or "(无)"
            ),
            facts_block=facts_block,
            n_para=n_para if len(batches) == 1 else max(2, n_para // 2),
        )

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
                            max_tokens=12000, timeout=420)
            if not out:
                continue
            if "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
            out = out.replace("```markdown", "").replace("```", "").strip()
            if re.search(r"\[\[", out):
                break
        else:
            failed_batches += 1
            warnings.append(
                f"主题「{theme.get('name')}」第 {bi}/{len(batches)} 批未能建立 fact 对应"
            )
            continue

        kept, dropped = _drop_unattributed(out)
        dropped_total += dropped
        if kept:
            paras.append(_render_citations(kept, fact2pid))
        all_used.extend(
            x.strip() for m in re.findall(r"\[\[([^[\]]+)\]\]", kept)
            for x in m.split(",") if x.strip() in fact2pid
        )

    if failed_batches == len(batches) and batches:
        warnings.append(f"主题「{theme.get('name')}」全部批次均未建立 fact 对应, 已跳过")
        return {"text": "", "used_fact_ids": [], "dropped_sentences": dropped_total,
                "warnings": warnings}

    if dropped_total:
        warnings.append(f"{dropped_total} 句因无 fact 标注被剔除")
    if len(batches) > 1:
        warnings.append(f"主题含 {len(theme_facts)} 条 fact, 分 {len(batches)} 批生成")

    return {"text": "\n\n".join(paras), "used_fact_ids": sorted(set(all_used)),
            "dropped_sentences": dropped_total, "warnings": warnings}


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
