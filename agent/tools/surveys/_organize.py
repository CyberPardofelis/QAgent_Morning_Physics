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
                     - relations   与其他主题的关系(只有一句具体的话, **不分类**)
                     - notes       主题内部的逻辑线索
             外加: narrative_arc(推荐叙述顺序) + gaps(文献覆盖不足处)

    Phase B1.5 cluster_themes_into_macros() → 同一文件里的 macros 字段
             把主题再往上聚一层: 宏观主题 -> 主题 -> 范式 -> 命题 -> fact。
             这是"知识结构地图"里的第三层, 以前只到主题就停了。

    Phase B2  condense_facts_by_paradigm()  → _paradigms.json + _claims.json
             主题 -> 技术路线 -> 命题。范式层以前只活在内存里(10-05 起落盘)。

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

from agent.prompts_survey import (
    CLAIM_CONDENSE_PROMPT,
    THEME_WRITE_FROM_CLAIMS_PROMPT,
    _tokenize,
)

from ._parallel import run_ordered

logger = logging.getLogger(__name__)

# 9-30: 主题地图的输出量随 fact 数线性增长, 顶到 max_tokens 就截断 -> JSON 解析失败。
# 实测 303 条 fact 时返回 59349 字符, 远超 20000 max_tokens 对应的量。
THEME_MAP_MAX_TOKENS = 32000

# 10-05: 分块参数, 从 backfill_graph.py 搬过来(那边有完整的实测记录)。
# **按字符封顶, 不按条数**: 实测同是 140 条的块, prompt 能从 3360 到 13194 字符
# (claim 长度差异极大), 而 13194 那次**卡死 43 分钟**、timeout=600 形同虚设。
# 决定输出和稳定性的是 prompt 大小, 不是条数。8000 是实测安全线。
THEME_CHUNK_CHARS = 8000
THEME_CHUNK_FACTS = 60          # 双重保险: 极长的 claim 也不会把块撑爆


def split_fact_chunks(
        facts: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """按 claim 总字符数切块, 同时受条数上限约束。

    为什么生产路径要分块而不是抽样(10-05 改): 原来是"fact 数 > 200 就按 subject
    均匀抽样到 200", 被抽掉的 fact **永远不会被任何主题覆盖**。实测
    `669cfc28a53e` 有 309 条 fact, 硬上限静默丢掉 35% —— 而那正是要总结的目标
    数据。抽样只在"一次调用吃下全部 fact"这个旧前提下才成立; 现在主题是全局
    累积的, 分块跑完把主题按名合并即可, 一条 fact 都不丢。
    """
    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    n = 0
    for f in facts:
        L = len(f.get("claim") or "") + 40      # 40 是 [fact_id] (type) 的开销
        if cur and (n + L > THEME_CHUNK_CHARS or len(cur) >= THEME_CHUNK_FACTS):
            chunks.append(cur)
            cur, n = [], 0
        cur.append(f)
        n += L
    if cur:
        chunks.append(cur)
    return chunks


def _merge_chunk_results(
        results: list[dict[str, Any]], max_themes: int) -> dict[str, Any]:
    """把多个块的主题结果合并成一张图。

    - 同名主题**并集** fact_ids(同名即同一主题, 这与 theme 表按 name upsert 一致)
    - relations 按 (to, note) 去重后合并
    - narrative_arc / gaps 逐块拼接去重 —— 它们本来就是全局性的
    - warnings 全部保留: 某块失败/截断必须让用户看见
    """
    by_name: dict[str, dict[str, Any]] = {}
    arc: list[str] = []
    gaps: list[str] = []
    warnings: list[str] = []
    for r in results:
        warnings.extend(r.get("warnings") or [])
        for x in r.get("narrative_arc") or []:
            if x not in arc:
                arc.append(x)
        for x in r.get("gaps") or []:
            if x not in gaps:
                gaps.append(x)
        for t in r.get("themes") or []:
            name = t["name"]
            cur = by_name.get(name)
            if cur is None:
                by_name[name] = {
                    "name": name,
                    "statement": t.get("statement") or "",
                    "fact_ids": list(dict.fromkeys(t.get("fact_ids") or [])),
                    "notes": t.get("notes") or "",
                    "relations": list(t.get("relations") or []),
                }
                continue
            # 同名: 并 fact_ids, 关系去重合并
            cur["fact_ids"] = list(dict.fromkeys(
                cur["fact_ids"] + (t.get("fact_ids") or [])))
            seen = {(x["to"], x.get("note", "")) for x in cur["relations"]}
            for rel in t.get("relations") or []:
                if (rel["to"], rel.get("note", "")) not in seen:
                    seen.add((rel["to"], rel.get("note", "")))
                    cur["relations"].append(rel)
            if len(t.get("statement") or "") > len(cur["statement"]):
                cur["statement"] = t["statement"]

    themes = list(by_name.values())
    if len(themes) > max_themes:
        # 砍主题 = 砍 fact, 属于**有损**操作: 必须说清楚丢了多少条 fact。
        dropped = sum(len(t["fact_ids"]) for t in themes[max_themes:])
        warnings.append(
            f"主题数 {len(themes)} 超过上限 {max_themes}, 已截断, "
            f"被截掉的主题涉及 {dropped} 条 fact(这些 fact 不会出现在主题地图里)")
        themes = themes[:max_themes]
    return {"themes": themes, "narrative_arc": arc, "gaps": gaps, "warnings": warnings}


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
3. 关系**不分类, 只写清楚是什么关系**。不要给关系贴任何类别标签(任何从
   固定小词表里挑一个的标签都不要) —— 那几个词换成任何一对主题都成立,
   等于没信息。要写的是这句话在**具体**说清什么: 哪个前提被哪个结果挑战了、
   哪条路线接替了哪条路线的哪个瓶颈、哪两个结论在哪个条件下相反。
   带上具体的对象和条件。写不出具体的就别写这条关系。
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
        {{"to": "另一个主题名", "note": "具体说清什么关系(带对象和条件), 不写关系类型"}}
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


def _parse_theme_json(raw: str, key: str = "themes") -> dict[str, Any] | None:
    """
    从模型输出里抠出主题地图 JSON。`key` = 要找的顶层非空列表键
    ("themes" 主题层 / "macros" 宏观主题层), 截断抢救那套逻辑两边共用。

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
            if isinstance(v, dict) and isinstance(v.get(key), list) and v[key]:
                return v
        # 补右括号 / 补尾引号再试(截断恢复)
        for tail in ("}", "}}", '"}}', '"}}}', '"]}}', '"', ']', '"}]', '"}}]}'):
            try:
                v = _json.loads(body + tail)
            except Exception:
                continue
            if isinstance(v, dict) and isinstance(v.get(key), list) and v[key]:
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


def _salvage_themes(body: str) -> list[dict[str, Any]] | None:
    """从**被截断**的主题地图输出里抢救出已经写完整的主题对象。

    10-05 补。这是从范式层照搬的教训 —— 10-03 就因为范式 JSON 被 max_tokens
    截断而白跑 2 个主题, 范式层因此有了 `_salvage_paradigms`, **但主题层一直没
    有**。后果实测: 一次 34 个 workspace 的回填里 5 个(约 360 条 fact)因为
    "结构写对了但在 max_tokens 处被截断"而整块作废, 一条主题都没产出。

    为什么 `_parse_theme_json` 的"补右括号"救不了: 它只能修**最后一个**对象
    缺尾巴的情况; 而实际失败是最后一个对象**只写了一半**(JSON 字符串没闭合),
    补任何数量的右括号都拼不出合法 JSON。

    两个必须遵守的点(来自 `_salvage_paradigms` 的实测教训):
      - 不能只认 depth==0: 输出被截断时外层 `{"themes": [` 是不平衡的, 里面
        每个主题对象都处在 depth>=1, 只认顶层会把它们全漏掉(这是第一版抢救
        逻辑的 bug, 对着真实失败样本才发现)
      - 必须跟踪字符串状态: statement 里出现 `{` `}` 或转义引号时, 不跟踪就会
        把字符串内容当成结构符号, 括号深度彻底算错
    """
    import json as _json

    out: list[dict[str, Any]] = []
    stack: list[int] = []
    in_str = False
    esc = False
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            try:
                obj = _json.loads(body[start:i + 1])
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("themes"), list):
                out.extend(x for x in obj["themes"] if isinstance(x, dict))
            elif isinstance(obj, dict) and obj.get("name") and "fact_ids" in obj:
                out.append(obj)
    # 长得像主题 = 有 name 且 fact_ids 存在(哪怕是空数组, 也说明写完了)
    return [o for o in out if (o.get("name") or "").strip() and "fact_ids" in o] or None


def _is_llm_error_response(text: str) -> str | None:
    """判断 `_call_llm` 返回的是**错误提示**还是模型输出, 是则返回错误摘要。

    10-05 加。原因: `_call_llm` 失败时**不抛异常, 而是返回一段 markdown 错误
    提示字符串**(见 llm.py 的 "失败时为带错误提示的字符串")。于是调用方会拿
    这段错误提示去当 JSON 解析, 解析失败, 然后报"疑似被 max_tokens 截断" ——
    **诊断完全错了**, 而且折半重试是最糟的补救: 提示词变短对 API 错误毫无帮助,
    只是把一次瞬时故障放大成连续 3 次失败。

    实测踩到: 34 个 workspace 的回填里 5 个"一条主题都没产出", 日志全是
    "返回 228 字符, 疑似被 max_tokens 截断"。手抄同一份 prompt 单发一次完全
    正常(4099 字符 / 5 个主题) —— 证明不是数据问题也不是模型问题, 是瞬时 API
    故障被误诊成了截断。

    错误串由 `llm_clients.classify_llm_error` 生成, 统一带 emoji 前缀
    (💰 余额不足 / ❌ 鉴权失败 / ❌ 超时 / 未设置 key 等)。
    """
    t = (text or "").strip()
    if not t:
        return None
    head = t[:80]
    for marker in ("💰", "❌", "未设置", "创建 client 失败", "未知 provider"):
        if marker in head:
            return t[:200].replace("\n", " ")
    return None


def _theme_map_retry_loop(
    facts: list[dict[str, Any]],
    clusters: list[tuple[str, list[dict[str, Any]]]] | list,
    paper_titles: dict[str, str],
    provider: str, model: str, temperature: float,
):
    """主题地图的调用+重试循环。返回 (facts, prompt, parsed, salvaged, last_err, n_api_err)。

    抽出来只为把 `organize_facts_into_themes` 的复杂度压回 30 以下(10-05 加了
    API 错误识别和截断抢救两个分支之后超了), 逻辑一行没改。

    **两条重试路径的补救方式相反, 这是本函数最要紧的地方**:
      - **API 错误**: 原样重试, **不折半**。故障跟提示词长度无关, 折半只会把
        一次瞬时故障放大成连续 3 次失败。实测 5 个 workspace 就是这么全灭的,
        日志里还全是误导人的"疑似被 max_tokens 截断"(那段其实是错误提示字符串,
        因为 `_call_llm` 失败时不抛异常, 见 llm.py)。
      - **截断/解析失败**: 折半重试, 并先抢救一次已写完整的主题。
    """
    from agent.tools.llm import _call_llm

    prompt = THEME_MAP_PROMPT.format(
        n_subject_histogram=_subject_histogram(clusters),
        facts_block=_facts_block(facts, paper_titles),
    )
    parsed = None
    salvaged: list[dict[str, Any]] | None = None
    last_err = ""
    n_api_err = 0
    n_now = len(facts)
    for _ in range(3):                     # 最多折半重试 2 次
        # 10-05: timeout 从 600 降到 240。实测 3789 字符的调用**卡了 6 分钟
        # 不返回**, 而同一批里 9458 字符的 154 秒就跑完了 —— 不是 prompt 大小
        # 问题, 是瞬时挂起。更糟的是 OpenAI SDK 默认 max_retries=2, 客户端超时
        # 不是"最多 600 秒"而是 600×3, 再乘本循环 3 次 = 理论最坏 90 分钟。
        # 240s 足够覆盖实测最长的一次(170s), 又能让挂起快速失败。
        # ⚠ 改这个数之前要记住: 170s 是 **M3.1 flash** 的实测值, M3 明显更慢。
        #   本函数默认 model="MiniMax-M3", 首次投产时务必盯一次真实耗时。
        out = _call_llm(prompt, provider=provider, model=model,
                        enable_thinking=False, temperature=temperature,
                        max_tokens=THEME_MAP_MAX_TOKENS, timeout=240)
        api_err = _is_llm_error_response(out)
        if api_err:
            n_api_err += 1
            last_err = f"LLM 调用失败: {api_err}"
            print(f"[organize] {last_err}; 原样重试", flush=True)
            continue
        if not out:
            last_err = "LLM 返回空"
        else:
            parsed = _parse_theme_json(out)
            if parsed is not None:
                break
            # 10-05: 解析不过先抢救一次。截断是这条路径上最常见的失败模式,
            # 前面几个主题往往**已经写完整了**, 整块丢掉等于那一轮白跑
            # (范式层 10-03 就有同样的教训, 见 _salvage_paradigms)。
            salvaged = _salvage_themes(out)
            if salvaged:
                print(f"[organize] 主题地图 JSON 被截断, 抢救出 "
                      f"{len(salvaged)} 个完整主题 (返回 {len(out)} 字符)",
                      flush=True)
                break
            last_err = (f"主题地图 JSON 解析失败 (返回 {len(out)} 字符, "
                        f"疑似被 max_tokens={THEME_MAP_MAX_TOKENS} 截断)")
        n_now = n_now // 2
        print(f"[organize] {last_err}; 折半到 {n_now} 条重试", flush=True)
        if n_now < 40:
            break
        facts = facts[:n_now]
        prompt = THEME_MAP_PROMPT.format(
            n_subject_histogram=_subject_histogram(clusters),
            facts_block=_facts_block(facts, paper_titles),
        )
    return facts, prompt, parsed, salvaged, last_err, n_api_err


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

    **10-05 改: 超量时"分块"而不是"抽样"。**
    原来是 `fact 数 > THEME_MAP_MAX_FACTS(200) 就按 subject 均匀抽样到 200`,
    被抽掉的 fact 永远不会被任何主题覆盖 —— 实测 `669cfc28a53e` 有 309 条,
    静默丢掉 35%, 而那正是要总结的目标数据。现在按字符切块(见
    split_fact_chunks), 每块独立归纳, 最后按主题名合并(_merge_chunk_results):
    一条 fact 都不丢, 代价是超量时多次调用。
    """
    paper_titles = paper_titles or {}
    if not facts:
        return {"themes": [], "narrative_arc": [], "gaps": [], "warnings": ["没有 fact"]}

    chunks = split_fact_chunks(facts)
    if len(chunks) > 1:
        print(f"[organize] {len(facts)} 条 fact 切成 {len(chunks)} 块"
              f"(每块 <= {THEME_CHUNK_CHARS} 字符 / {THEME_CHUNK_FACTS} 条), "
              f"逐块归纳后按主题名合并 —— 一条 fact 都不丢", flush=True)
        results = []
        for i, ch in enumerate(chunks, 1):
            print(f"[organize]   块 {i}/{len(chunks)}: {len(ch)} 条", flush=True)
            results.append(_organize_one_chunk(
                ch, paper_titles=paper_titles, provider=provider, model=model,
                temperature=temperature, max_themes=max_themes))
        merged = _merge_chunk_results(results, max_themes)
        return merged
    return _organize_one_chunk(
        chunks[0], paper_titles=paper_titles, provider=provider, model=model,
        temperature=temperature, max_themes=max_themes)


def _organize_one_chunk(
    facts: list[dict[str, Any]],
    paper_titles: dict[str, str] | None = None,
    provider: str = "minimax",
    model: str = "MiniMax-M3",
    temperature: float = 0.4,
    max_themes: int = 8,
) -> dict[str, Any]:
    """单块(调用量在 max_tokens 之内)的归纳。10-05 从 organize_facts_into_themes
    抽出来, 让上层能分块调用; 逻辑一行没改。"""
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
    parsed = None
    salvaged: list[dict[str, Any]] | None = None
    last_err = ""
    n_api_err = 0
    facts, prompt, parsed, salvaged, last_err, n_api_err = _theme_map_retry_loop(
        facts, clusters, paper_titles, provider, model, temperature)
    if parsed is None:
        if not salvaged:
            if n_api_err:
                # 全程都是 API 故障, 没有一次真的拿到模型输出 —— 说清楚,
                # 别让用户以为是数据太大或 prompt 写坏了
                return {"themes": [], "narrative_arc": [], "gaps": [],
                        "warnings": [f"{last_err} (连续 {n_api_err} 次 API 故障, "
                                     f"不是输出被截断; 重跑即可)"]}
            return {"themes": [], "narrative_arc": [], "gaps": [], "warnings": [last_err]}
        # 抢救出来的只有 themes, narrative_arc / gaps 必然残缺(它们排在数组后面)
        parsed = {"macros": None, "themes": salvaged,
                  "narrative_arc": [], "gaps": []}

    warnings: list[str] = []
    if salvaged:
        # 「宁可吵, 不可静默」: 抢救成功不等于没损失 —— narrative_arc 和 gaps
        # 排在 themes 数组后面, 截断时必然残缺, 写综述的顺序和覆盖空白会少一块
        warnings.append(
            f"主题地图 JSON 被 max_tokens 截断, 已抢救出 {len(salvaged)} 个主题; "
            f"叙述顺序(narrative_arc)和覆盖空白(gaps)丢失")
    themes: list[dict[str, Any]] = []
    seen_theme: set[str] = set()
    _pending_rel_check: list[tuple[str, list[str]]] = []
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
        # 10-05: **不再存 type**。support/tension/evolution/contrast 换成任何
        # 一对主题都成立 —— 实测 68 条关系边里 type 只有 4 个取值, 而同一
        # 取值下 note 说的东西毫无共性(6 条 tension 实际是三类完全不同的关系
        # 混标)。标签不携带信息, 具体的那句话才携带。现在只留 to + note。
        # 兼容: 磁盘上已有的旧 _thematic_map.json 里 type 还在, 读侧原样
        # 忽略即可(没有代码按 type 分支), 所以不需要数据迁移。
        rels = []
        n_blank_note = 0
        for r in t.get("relations") or []:
            if isinstance(r, dict) and (r.get("to") or "").strip():
                note = (r.get("note") or "").strip()
                if not note:
                    n_blank_note += 1
                rels.append({
                    "to": (r.get("to") or "").strip(),
                    "note": note,
                })
        if n_blank_note:
            # 「宁可吵, 不可静默」: 去掉 type 之后, 一条 note 为空的关系边就
            # 什么都不剩了。留着它是结构、丢掉它会掩盖问题 —— 所以说出来。
            warnings.append(
                f"主题「{name}」有 {n_blank_note} 条关联没有具体说明(只有指向没有内容)")
        # 指向校验**必须等所有主题都进 seen_theme 之后再做**: 关系可以指向
        # 在 JSON 里排在后面才出现的主题, 在循环内查会把它误报成"不存在"。
        _pending_rel_check.append((name, [r["to"] for r in rels]))
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

    # 关系指向校验(见 _pending_rel_check 处的注释): 只警告, **不删边** ——
    # 模型写错主题名是常见现象, 直接删会让结构凭空少一块, 而这里只要让
    # 用户看见"有 N 条边指向了不存在的名字"就够了。指向自身不算错。
    n_bad_rel = 0
    for src, tos in _pending_rel_check:
        bad = sorted({t for t in tos if t not in seen_theme and t != src})
        if bad:
            n_bad_rel += len(bad)
            warnings.append(f"主题「{src}」关联指向了不存在的主题: {'、'.join(bad)}")
    if n_bad_rel:
        print(f"[organize] 关系边校验: {n_bad_rel} 条指向了不存在的主题名(保留边, 仅告警)",
              flush=True)

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
# Phase B1.5: 宏观主题层 (10-05 加)
# ============================================================

MACRO_THEME_PROMPT = """你在为一份文献综述整理**上位结构**。

下面已经归纳好了一批主题(每个主题 = 一个子问题方向, 附有它的领域判断和 fact 数)。
你的任务是把这些主题**再往上聚一层**, 也就是回答: "这个领域由哪几个更大的板块
构成, 每个板块里包含下面哪几个主题"。

【已经归纳出的主题】
{themes_block}
{existing_block}
【怎么聚】
1. 按**研究对象/技术路线所属的板块**聚, 不是按"哪几个主题看起来像"。
   一个宏观主题至少要含 2 个主题; 只装 1 个主题的板块没有意义, 别拆。
2. **不要平分**。如果 5 个主题里 4 个明显同属一个板块, 就是 1 大 + 1 小,
   不要为了"平均"硬凑成 2-3 个。
3. 板块数量控制在 {n_macro_hint} 个左右。宁可 2 个大板块, 不要 6 个小分组。
4. 每个板块的 statement 写**这个板块在解决什么问题、为什么它们该放一起**,
   不是把下面主题的名字串一遍。
5. 板块之间如果有一条值得记的关系, 写进 cross_notes, 说清楚具体是什么关系
   (带对象和条件), **不要给关系贴类别标签**。

【输出格式】严格 JSON, 不要 markdown 包裹:
{{
  "macros": [
    {{
      "name": "宏观板块名(14 字以内)",
      "statement": "这个板块在解决什么问题, 为什么这些主题属于同一块",
      "theme_names": ["主题名1", "主题名2"],
      "notes": "板块内部的推进逻辑(2-3 句)",
      "cross_notes": ["与另一个板块之间的具体关系(可留空数组)"]
    }}
  ]
}}

theme_names 只能从上面给出的主题名里原样挑, 不能改写、不能新造。
每个主题应当恰好归入一个板块; 实在归不进去的可以不列。
只输出 JSON。"""


def _salvage_macros(body: str) -> list[dict[str, Any]] | None:
    """从**被截断**的宏观主题输出里抢救出已经写完整的板块对象。

    `_parse_theme_json` 的"补右括号"救不了这种截断: 它只能修**最后一个**
    对象缺尾巴的情况, 而实际失败是"最后一个对象只写了一半"—— 补任何数量的
    右括号都拼不回一个合法 JSON。必须像 `_paradigm._salvage_paradigms` 那样
    用括号栈扫出**所有平衡的对象**, 逐个 loads, 前面的完整对象照收。

    两个坑(都来自 _salvage_paradigms 的实测教训):
      - 不能只认 depth==0: 输出被截断时外层 `{"macros": [` 是不平衡的,
        里面每个板块对象都处在 depth>=1, 只认顶层会把它们全漏掉。
      - 必须跟踪字符串状态: statement 里出现 `{` `}` 或转义引号时,
        不跟踪就会把字符串内容当成结构符号, 括号深度彻底算错。
    """
    out: list[dict[str, Any]] = []
    stack: list[int] = []
    in_str = False
    esc = False
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            try:
                obj = json.loads(body[start:i + 1])
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("macros"), list):
                out.extend(x for x in obj["macros"] if isinstance(x, dict))
            elif isinstance(obj, dict) and obj.get("theme_names") is not None:
                out.append(obj)
    # 长得像板块 = 有 name 且 theme_names 非空
    return [o for o in out
            if (o.get("name") or "").strip() and o.get("theme_names")]


def _macro_prompt_parts(
    themes: list[dict[str, Any]],
    names: list[str],
    max_macros: int,
    existing_macros: list[dict[str, Any]] | None,
) -> tuple[str, int]:
    """返回 (existing_block, n_macro_hint)。空板块时块是空串。

    独立成函数而不是塞在 `cluster_themes_into_macros` 里: 那条链已经有解析、
    抢救、重试三条分支, 再叠块拼装就把 C901 顶爆了(实测 34 > 30)。
    """
    ex_lines: list[str] = []
    for m in existing_macros or []:
        nm = (m.get("name") or "").strip()
        if not nm:
            continue
        st = (m.get("statement") or "").strip()
        ex_lines.append(f"- 「{nm}」(已收编 {m.get('n_themes', 0)} 个主题)"
                        + (f": {st}" if st else ""))
    if not ex_lines:
        return "", max(2, min(max_macros, len(names) // 2))
    block = (
        "\n【已有的宏观板块 —— 优先归入这些, 不要造同类的新板块】\n"
        + "\n".join(ex_lines)
        + f"\n下面 {len(ex_lines)} 个板块里能收下这些主题的就收下, "
          "**name 原样照抄**才算复用; 只有当某几个主题确实不属于任何"
          "已有板块时才新建, 新名字不要和已有板块近义。"
          "已有板块里原有的主题不在这次输入里, 不用管, "
          "也不要因为看不到它们就不敢复用该板块。\n")
    # 增量模式下"复用几个 + 新建几个"才是有意义的量; 直接拿 max_macros 去问
    # 会诱导模型凭空造满 5 个同类板块。
    return block, max(1, min(max_macros, len(names) // 3))


def cluster_themes_into_macros(
    themes: list[dict[str, Any]],
    provider: str = "minimax",
    model: str = "MiniMax-M3",
    max_macros: int = 5,
    existing_macros: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """主题 -> **宏观主题**。返回 (macros, warnings)。

    10-05 加。这是"知识结构地图"的第三层: 以前流水线到主题就停了
    (宏观主题 → 主题 → 范式 → 命题 → fact 四层里的第二层是终点)。

    为什么是**对主题再聚**而不是对 fact 重聚:
      主题层已经做完"哪些 fact 说的是一件事"的判断, 那是这块语料里最难的一步。
      再对 fact 跑一遍聚类, 等于把已经付过学费的判断推翻重来, 而且会得到
      与主题层**不一致**的边界(两层对不上, 图谱就自相矛盾了)。
      宏观层要回答的是另一个问题 —— 主题之间的上位关系, 它的输入本来就该是主题。

    输入只有主题名 + 领域判断 + fact 数, **不喂 fact 正文**: 这一层不看细节,
    只看主题的归属关系。1 次调用, 输出很小(实测 6-8 个主题 -> 2-4 个板块),
    相比主题层的 20000+ max_tokens 预算基本可以忽略。

    **`existing_macros`(10-06 加) = 增量归类模式**。宏观主题是**全局累积**的:
    一次全量重聚看着干净, 实则有两个硬伤 ——
      1. **prompt 撑爆**: 209 个主题全塞进去 = 19351 字符, timeout=300 秒
         必然超时(实测 6575 字符就要 160 秒)。而且主题数只增不减, 越用越炸。
      2. **已有结构被推翻**: 上一轮定好的板块边界, 这轮换个措辞就变了。
    所以默认只喂**未归类**的主题, 并把已有板块名列给它, 让它优先收编 ——
    prompt 小两个数量级, 且历史结构只增不改。

    失败不抛异常, 返回 ([], warnings) —— 宏观层是加法, 缺了它单层结构照样能用。
    """
    from agent.tools.llm import _call_llm

    warnings: list[str] = []
    names = [(t.get("name") or "").strip() for t in themes]
    names = [n for n in names if n]
    if len(names) < 2:
        # 一个主题谈不上"再聚一层", 强行聚只会得到 1:1 的空壳结构
        return [], ([f"只有 {len(names)} 个主题, 不做宏观聚类"] if names else ["没有主题"])

    known = set(names)
    lines = []
    for t in themes:
        nm = (t.get("name") or "").strip()
        if not nm:
            continue
        st = (t.get("statement") or "").strip() or "(无领域判断)"
        lines.append(f"- 「{nm}」({len(t.get('fact_ids') or [])} 条 fact): {st}")

    existing_block, n_hint = _macro_prompt_parts(
        themes=themes, names=names, max_macros=max_macros,
        existing_macros=existing_macros)
    ex_names = {(m.get("name") or "").strip() for m in (existing_macros or [])
                if (m.get("name") or "").strip()}

    prompt = MACRO_THEME_PROMPT.format(
        themes_block="\n".join(lines),
        n_macro_hint=n_hint,
        existing_block=existing_block,
    )

    parsed = None
    salvaged: list[dict[str, Any]] | None = None
    last_err = ""
    n_api_err = 0
    for _attempt in range(2):                # 最多重试 1 次, 宏观层不是关键路径
        out = _call_llm(prompt, provider=provider, model=model,
                        enable_thinking=False, temperature=0.3,
                        # 10-05: 必须和主题层一样给 32000。第一版这里写的 8000,
                        # 结果模型把 22868 字符全花在 <think> 上、正文一个字没出
                        # (llm.py 打的是 "content 空 → reasoning_content"),
                        # 于是解析失败。这正是本文件 9-29 就记过的坑:
                        # enable_thinking=False **挡不住** MiniMax 输出 think,
                        # 那部分照样吃 max_tokens。宏观层要"想清楚"得更久
                        # (184 个主题一起判断归属), 预算不该比主题层紧。
                        max_tokens=THEME_MAP_MAX_TOKENS, timeout=300)
        # 同主题层: 错误提示串当 JSON 解析, 会误诊成"截断"。见 _is_llm_error_response
        api_err = _is_llm_error_response(out)
        if api_err:
            n_api_err += 1
            last_err = f"LLM 调用失败: {api_err}"
            print(f"[macro] {last_err}; 原样重试", flush=True)
            continue
        if not out:
            last_err = "LLM 返回空"
        else:
            parsed = _parse_theme_json(out, key="macros")
            if parsed is not None:
                break
            # 解析不过先抢救一次 —— 板块少一两个不该让这一整层作废
            salvaged = _salvage_macros(out)
            if salvaged:
                warnings.append(
                    f"宏观主题 JSON 被截断(返回 {len(out)} 字符), "
                    f"抢救出 {len(salvaged)} 个完整板块")
                break
            last_err = f"宏观主题 JSON 解析失败 (返回 {len(out)} 字符, 疑似截断)"
        print(f"[macro] {last_err}", flush=True)
    if parsed is None:
        if not salvaged:
            if n_api_err:
                return [], [f"{last_err} (连续 {n_api_err} 次 API 故障, "
                            f"不是输出被截断; 重跑即可)"]
            return [], [f"宏观主题层失败: {last_err}"]
        parsed = {"macros": salvaged}

    macros: list[dict[str, Any]] = []
    seen_macro: set[str] = set()
    taken: dict[str, str] = {}              # 主题名 -> 所属宏观主题名
    for m in parsed.get("macros") or []:
        if not isinstance(m, dict):
            continue
        name = (m.get("name") or "").strip()
        if not name:
            warnings.append("宏观板块缺 name, 已跳过")
            continue
        if name in seen_macro:
            warnings.append(f"宏观板块名重复: {name}")
            continue
        raw = m.get("theme_names") or []
        if isinstance(raw, str):
            raw = [raw]
        kept, bad, dup = [], 0, []
        for tn in raw:
            tn = str(tn).strip()
            if tn not in known:
                bad += 1
            elif tn in kept:
                dup.append(tn)
            else:
                kept.append(tn)
        if bad:
            warnings.append(f"宏观板块「{name}」有 {bad} 个 theme_names 不是已有主题名, 已丢弃")
        if dup:
            warnings.append(f"宏观板块「{name}」重复列入主题: {'、'.join(sorted(set(dup)))}")
        if not kept:
            warnings.append(f"宏观板块「{name}」没有有效主题, 已跳过")
            continue
        if len(kept) == 1 and name not in ex_names:
            # 装 1 个主题的板块不构成"上位结构", 但也不是模型的错(语料就
            # 这么散)。保留并说明, 不静默丢。
            #
            # **复用已有板块的不能这么判**(10-06): 增量模式下 kept 只是这批
            # 未归类主题里落在该板块的那几个, 板块里原有的几十个主题根本不在
            # 输入里。按 kept 数告警会把正常的复用报成"1:1 空壳", 是纯误报。
            warnings.append(f"宏观板块「{name}」只含 1 个主题, 不构成上位聚类")
        elif name in ex_names:
            warnings.append(f"宏观板块「{name}」复用已有板块, 这次新增 {len(kept)} 个主题")
        for tn in kept:
            if tn in taken and taken[tn] != name:
                warnings.append(f"主题「{tn}」被多个宏观板块收编(『{taken[tn]}』和『{name}』), 以后者为准")
            taken[tn] = name
        seen_macro.add(name)
        macros.append({
            "name": name,
            "statement": (m.get("statement") or "").strip(),
            "notes": (m.get("notes") or "").strip(),
            "theme_names": kept,
            # 复用已有板块时, 调用方必须知道: 它的 statement/notes 会被
            # upsert 覆盖, 而原有的几十个主题并不在 kept 里。不标出来的话
            # 调用方只能把"改了名字的板块"和"新建的板块"同等对待。
            "reused": name in ex_names,
            "cross_notes": [x.strip() for x in (m.get("cross_notes") or [])
                            if isinstance(x, str) and x.strip()],
        })

    if not macros:
        return [], ["宏观主题层没产出任何板块"]
    if len(macros) > max_macros:
        warnings.append(f"宏观板块数 {len(macros)} 超过上限 {max_macros}, 已截断")
        macros = macros[:max_macros]

    orphan = [n for n in names if n not in taken]
    if orphan:
        # 不硬塞给某个板块 —— 硬塞会造出一条假的归属关系。明确说出来。
        warnings.append(f"{len(orphan)} 个主题未归入任何宏观板块: {'、'.join(orphan)}")
    return macros, warnings


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


# 句子切分 —— 9-27 实测修正
# 原来要求句末标点后必须有空白("(?<=[。!?；;])\s+")。但**中文句号后面没有空格**,
# 于是这个正则对中文文本**从来没真正分过句** —— 整个段落被当成一句,
# 结果是"含一个编造引用 → 整段全丢"。
# 现在不要求空白, 但排除右引号/右括号, 避免把「……。」切断。
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])(?![」』”’\"'\)\]】》])")
_ANNOT_RE = re.compile(r"\[\[([^[\]]+)\]\]")


def _render_citations_or_drop(text: str, fact2pid: dict[str, str]) -> tuple[str, int]:
    """
    fact_id → paper_id 机械映射; **引不出出处的句子直接丢掉**。

    9-27 实测发现的漏洞: 原来模型编造 fact_id 时, 标注被去掉但**句子留下了** ——
    结果就是"看起来通顺但没有任何出处的断言", 正是用户最早指出的 v1 毛病。
    现在只要一段里的标注**全部解析不出来**, 整段视为无出处, 丢弃。

    Returns: (保留的文本, 被丢弃的句子数)
    """
    def _resolve(raw: str) -> tuple[str, bool]:
        """返回 (替换成的引用串, 是否全部解析失败)"""
        pids, unresolved = [], 0
        for part in raw.split(","):
            fid = part.strip()
            pid = fact2pid.get(fid)
            if pid and pid not in pids:
                pids.append(pid)
            else:
                unresolved += 1
        return "".join(f"[{p}]" for p in pids), unresolved > 0 and not pids

    kept_paras, dropped = [], 0
    for para in text.split("\n\n"):
        kept_lines = []
        for line in para.split("\n"):
            if not line.strip():
                kept_lines.append(line)
                continue
            # 句子级处理: 一句里可能既有合法引用又有编造的 id,
            # 只要**至少一个**标注能解析就保留(把解析不出的部分丢掉即可);
            # 全部解析不出才丢整句。
            kept_sents = []
            for sent in _SENT_SPLIT.split(line):
                if not sent.strip():
                    continue
                if not _ANNOT_RE.search(sent):
                    dropped += 1          # 本来就没标注(过渡句也走这里)
                    continue
                resolved_any = [False]
                all_good = [True]

                def _sub(m, _ok=resolved_any, _all=all_good):
                    rep, all_bad = _resolve(m.group(1))
                    _ok[0] = _ok[0] or not all_bad
                    if all_bad:
                        _all[0] = False
                    return rep
                new_sent = _ANNOT_RE.sub(_sub, sent)
                # 严格: 只要有一个标注解析不出, 整句就不可信 —— 因为把编造的
                # 引用抹掉会留下"看起來通顺但没有出处"的半截断言, 那正是要避免的
                if not resolved_any[0] or not all_good[0]:
                    dropped += 1
                    continue
                kept_sents.append(new_sent)
            if kept_sents:
                kept_lines.append(" ".join(kept_sents))
        cleaned = "\n".join(kept_lines).strip()
        if cleaned:
            kept_paras.append(cleaned)
    body = "\n\n".join(kept_paras)
    body = re.sub(r"\[\s*\]", "", body)
    body = re.sub(r"(\[[^\]\s]+\])(?:\s*\1)+", r"\1", body)   # 合并连续重复引用
    return re.sub(r"[ \t]{2,}", " ", body).strip(), dropped


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

    # 10-02 并发: 每批的 prompt / 请求 / 解析互相独立。
    # 顺序是硬约束 —— paras 的先后就是最终段落的排版顺序。
    # 所以并发只跑"拿 out", 剔引用/渲染/append 回主线程按原序做。
    def _write_one(batch: list[dict[str, Any]]) -> tuple[str, bool]:
        """纯函数: 组 prompt + 调 LLM(带重试) + 剥壳。返回 (正文, 是否建立 fact 对应)。"""
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
                # 10-05: 不再输出"是 support 关系"这种标签 —— 它对写作没有任何
                # 指示作用, 只把 note 那句具体的话挤到后面去。只给那句话。
                "\n".join(f"- 与「{r.get('to')}」: {r.get('note')}"
                          for r in (theme.get("relations") or [])
                          if r.get("to") and r.get("note"))
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
                return out, True
        return "", False

    for bi, (batch, res) in enumerate(
            zip(batches, run_ordered(batches, _write_one, desc="theme_write"), strict=True), 1):
        fact2pid = {f["fact_id"]: f.get("paper_id", "") for f in batch}
        if not res.ok:
            failed_batches += 1
            warnings.append(
                f"主题「{theme.get('name')}」第 {bi}/{len(batches)} 批异常: {res.exc}"
            )
            continue
        out, ok = res.value
        if not ok:
            failed_batches += 1
            warnings.append(
                f"主题「{theme.get('name')}」第 {bi}/{len(batches)} 批未能建立 fact 对应"
            )
            continue

        kept, dropped = _drop_unattributed(out)
        body, unresolved_dropped = _render_citations_or_drop(kept, fact2pid)
        dropped_total += dropped + unresolved_dropped
        if body:
            paras.append(body)
        all_used.extend(
            x.strip() for m in _ANNOT_RE.findall(kept)
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
# 命题层 (9-30 加) —— 综述"综"的缺失环节
# ============================================================
# 9-30 用户实测反馈: "综述很长、看上去到处都是引用, 但引文实际很稀疏,
# 因为一篇文章提取的fact不少……综述当中试图把每个小的fact都引一下,
# 而无论是否其实它们在讲同一件事。"
#
# 实测证据(40 篇那次): 同一篇 10.1103/revmodphys.93.045003 在**一段话里被引 5 次**,
# 下一段整段只引一篇重复 3 次。295 条 fact → 逐条成句 → 逐条挂同一个引用。
# 这不是综述, 是"把论文摘要抄一遍并标上出处"。
#
# 缺的一环: fact(原子事实) 和 句子 之间应该有 **命题(claim)** 这一层。
#     fact ──[合并讲同一件事的]──> claim ──[成文]──> 句子
# 一条 claim 可以由多篇论文共同支持, 成文时写成**一句话挂多个引用** ——
# 这正是用户要的"一句话就表达了一个或多个文章的结果"。
#
# 用序号而不是 fact_id 喂给模型: fact_id 是内容哈希(几十字符), 20 条一批
# 光 id 就占掉大半 prompt, 且模型抄不对长哈希。喂 [1]..[N] 让它回索引。


def condense_facts_into_claims(
    facts: list[dict[str, Any]],
    provider: str = "minimax",
    model: str = "MiniMax-M3",
    batch_size: int = 24,
    temperature: float = 0.2,
    min_per_claim: int = 1,
) -> list[dict[str, Any]]:
    """把讲同一件事的多条 fact 合并成一条命题。

    Args:
        facts: [{fact_id, claim, subject, paper_id, fact_type}, ...]
        batch_size: 每批喂给 LLM 多少条
        min_per_claim: 合并率下限 —— 若模型几乎不合并(全 1 条 1 命题),
            说明这批 fact 本来就各说各话, 保留原样不强凑

    Returns:
        [{claim_id, statement, fact_ids, paper_ids, theme_id}, ...]
        未能合并的 fact 会各自成为一条 claim(保证不丢内容)
    """
    from agent.tools.llm import _call_llm

    if not facts:
        return []

    claims: list[dict[str, Any]] = []
    seq = 0
    # 9-30 实测修正: **按 subject 聚簇后再分批**, 不能顺序切。
    # 顺序切的问题: 讲同一件事的两条 fact(往往来自不同论文)隔了几十条,
    # 落进不同批次, 模型永远没机会把它们放一起 —— 实测 629 条 fact 顺序切,
    # 合并率只有 1.32:1, 且合并出来的几乎全是"同一篇的多个细节"。
    # 同一 subject 内部的 fact 更可能是不同论文对同一件事的独立表述。
    #
    # 10-02 并发: 每个 subject 聚簇的 prompt / 请求 / 解析完全独立。
    # **顺序是硬约束** —— 下面 `seq` 是全局递增的 claim_id 计数器, claim_id 决定
    # 正文引用编号, 顺序一变整篇的引用编号就漂。所以并发只跑"拿 groups",
    # 分配 claim_id 和 append 一律回主线程按簇的原序做。
    def _condense_one(cluster: list[dict[str, Any]]) -> list[tuple[list[int], str]]:
        """纯函数: 组 prompt + 调 LLM + 解析成 groups。不碰 seq / claims。"""
        lines = []
        pid_index: dict[str, int] = {}

        def _pid_label(pids: list[str], _idx=pid_index) -> str:
            out = []
            for p in pids:
                if not p:
                    continue
                if p not in _idx:
                    _idx[p] = len(_idx) + 1
                out.append(str(_idx[p]))
            return ",".join(out) if out else "-"

        for i, f in enumerate(cluster, 1):
            subj = (f.get("subject") or "").strip()
            head = f"[{i}] <p{_pid_label([p for p in (f.get('paper_ids') or [f.get('paper_id') or '']) if p])}>"
            if subj:
                head += f" <{subj}>"
            lines.append(head + " " + (f.get("claim") or "").strip())
        prompt = CLAIM_CONDENSE_PROMPT.format(
            n=len(cluster), facts_block="\n".join(lines))
        groups: list[tuple[list[int], str]] = []
        for _ in range(2):          # 一次重试, 提高 JSON 服从率
            try:
                out = _call_llm(prompt, provider=provider, model=model,
                                enable_thinking=False, temperature=temperature,
                                # 9-30: 6000 -> 3000。输出里真正有用的只有
                                # fact_idx 列表 + 一句 statement; MiniMax 即便
                                # enable_thinking=False 也会吐<think> 且照样吃额度,
                                # 给太多就一路生成到上限(实测每批 16000 字符)。
                                max_tokens=3000, timeout=300)
            except Exception as e:
                logger.warning("[condense_claims] batch 失败: %s", e)
                out = ""
            if out and "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
            groups = _parse_claim_groups(out, len(cluster))
            if groups:
                break
        if not groups:
            groups = [([i], "") for i in range(1, len(cluster) + 1)]   # 解析失败 -> 不强凑

        # 覆盖检查: LLM 可能漏掉某些 fact, 漏的单独成条, 绝不能静默丢
        covered = {i for g, _s in groups for i in g}
        for i in range(1, len(cluster) + 1):
            if i not in covered:
                groups.append(([i], ""))
        return groups

    clusters = _batch_by_subject(facts, batch_size)
    _condensed = run_ordered(clusters, _condense_one, desc="condense_claims")
    for cluster, res in zip(clusters, _condensed, strict=True):
        if res.ok:
            groups = res.value
        else:
            logger.warning("[condense_claims] 簇失败, 退回逐条: %s", res.exc)
            groups = [([i], "") for i in range(1, len(cluster) + 1)]

        for g, stmt in groups:
            members = [cluster[i - 1] for i in g if 1 <= i <= len(cluster)]
            if not members:
                continue
            seq += 1
            pids: list[str] = []
            for m in members:
                pid = m.get("paper_id") or ""
                if pid and pid not in pids:
                    pids.append(pid)
            # 9-30: 优先用**模型合成的 statement**, 拿不到才回退到拼接 fact 原文。
            # 原来无论模型写没写都拼接原文 —— 模型那句概括白写(付了 token 就被扔),
            # 而且命题层本来该做的"综合"就没了, 这也是合并率上不去的原因之一。
            statement = (stmt or "").strip() or " ".join(
                (m.get("claim") or "").strip() for m in members)
            claims.append({
                "claim_id": f"c{seq}",
                "statement": statement,
                # 10-02: 去重后的代表 fact 带着 fact_ids(同篇同 subject 的全部),
                # 这里要展开, 否则覆盖度审计和论文覆盖统计会漏掉被收掉的那些。
                "fact_ids": [fid for m in members
                             for fid in (m.get("fact_ids") or [m["fact_id"]])
                             if fid],
                "paper_ids": pids,
                "theme_id": members[0].get("theme_id"),
                "n_facts": sum(m.get("n_facts", 1) for m in members),
            })

    merged = sum(1 for c in claims if c["n_facts"] > 1)
    cross = sum(1 for c in claims if len(c.get("paper_ids") or []) > 1)
    ratio = len(claims) / max(len(facts), 1)
    print(
        f"[condense_claims] {len(facts)} 条 fact -> {len(claims)} 条命题 "
        f"(合并 {merged} 条, 其中**跨论文** {cross} 条, 压缩比 {ratio:.2f})",
        flush=True,
    )
    return claims


def _batch_by_subject(
    facts: list[dict[str, Any]],
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    """按 subject 聚簇, 簇内再分批(不足一批的相邻簇合并填满)。

    目的: 让"不同论文讲同一件事"的 fact 落进同一批。顺序切做不到这点。
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for f in facts:
        key = (f.get("subject") or "").strip() or "∅"
        buckets.setdefault(key, []).append(f)
    # 大簇优先, 便于填满批次
    order = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    out: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for _key, members in order:
        for m in members:
            cur.append(m)
            if len(cur) >= batch_size:
                out.append(cur)
                cur = []
    if cur:
        out.append(cur)
    return out


_DETAIL_TAIL_CHARS = 220


def _dedupe_same_paper_same_subject(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同一篇论文、同一 subject 的多条 fact -> 一条(细节并进 claim 尾部)。

    10-02 实测 (v60g, 241 条 usable fact):
        不同 subject 163 个, 但**只有 4 个 subject 跨论文**, 落在跨论文 subject
        里的 fact 只有 17 条 (7%) —— 也就是 93% 的 fact 的 subject 只对应一篇论文。
        把这样的 fact 批量喂给合并器, 模型看到的是"一篇论文的各个细节",
        合并它们是摘要不是综合, 所以模型**正确地**不合并(实测簇级压缩比 1.01)。

        于是命题 ≈ 1:1 对应论文, 成文时一句只能挂一个引用 —— 这才是
        "引文看着有、实际很稀疏"的根因, 不是引用编号机制的问题。

    这里机械地把"同一篇论文讲的同一件事"收成一条, 不指望模型做 ——
    模型两次实测都拒绝合并同论文 fact, 所以这事只能在这儿做。

    fact_ids 合并保留, 所以覆盖度审计/论文覆盖统计不受影响。
    """
    order: list[str] = []
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for f in facts:
        key = (f.get("paper_id") or "", (f.get("subject") or "").strip())
        if key not in by:
            by[key] = []
            order.append(key)
        by[key].append(f)

    out: list[dict[str, Any]] = []
    for key in order:
        group = by[key]
        if len(group) == 1:
            out.append(group[0])
            continue
        rep = dict(group[0])
        ids = [g.get("fact_id") for g in group if g.get("fact_id")]
        extra = "；".join((g.get("claim") or "").strip() for g in group[1:] if g.get("claim"))
        base = (rep.get("claim") or "").strip()
        rep["claim"] = f"{base}（同一工作的其他侧面: {extra}）"[:len(base) + _DETAIL_TAIL_CHARS] \
            if extra else base
        rep["fact_ids"] = ids
        rep["n_facts"] = len(group)
        out.append(rep)
    return out


_CLAIM_GROUP_RE = re.compile(r"\{[^{}]*\}")


def _parse_claim_groups(raw: str, n: int) -> list[tuple[list[int], str]]:
    """从 LLM 输出里解析分组。返回 [(fact_idx 列表, statement)], statement 可为空串。

    9-30 修: 原来只返回 idx 列表, **模型写的 statement 被整个丢掉**, 后面又用
    fact 原文拼接替代 —— 等于模型每组白写一句概括(付了 token), 换回来的还是
    原始 fact 拼接。既浪费输出, 又丢掉了"综合"本身, 这正是合并率上不去的原因之一。

    两种格式都收(JSON 对象 / 纯下标数组), 模型的 JSON 服从率不是 100%。
    """
    if not raw:
        return []
    txt = raw.replace("```json", "").replace("```", "").strip()
    out: list[tuple[list[int], str]] = []

    def _idxs(part: str) -> list[int]:
        v = [int(x) for x in re.findall(r"\d+", part)]
        return sorted({i for i in v if 1 <= i <= n})

    # 1) JSON 对象: {"fact_idx": [1,3], "statement": "..."}
    for chunk in _CLAIM_GROUP_RE.findall(txt):
        idxs = _idxs(chunk)
        if not idxs:
            continue
        sm = re.search(r'"statement"\s*:\s*"([^"]*)"', chunk)
        out.append((idxs, (sm.group(1).strip() if sm else "")))
    if out:
        return out

    # 2) 多组块 [[1,3],[2,5]] —— 这种格式没有 statement, 留空让调用方兜底
    for block in re.findall(r"\[[\d\s,\[\]]{3,}\]", txt):
        for part in re.split(r"\]\s*,\s*\[", block.strip("[]")):
            idxs = _idxs(part)
            if idxs:
                out.append((idxs, ""))
    return out


# ============================================================
# 命题层: 按主题合并 + 从命题成文
# ============================================================
def condense_facts_by_theme(
    facts: list[dict[str, Any]],
    theme_map: dict[str, Any],
    provider: str = "minimax",
    model: str = "MiniMax-M3",
    batch_size: int = 24,
    max_claims_per_theme: int = 60,
) -> list[dict[str, Any]]:
    """按主题分别合并 fact 成命题。

    为什么分主题而不是一次性全合并: 跨主题的 fact 本来就不可能讲同一件事
    (它们被 Phase B 分到不同主题就是因为讲的不是一回事), 混在一起喂给模型
    只会浪费 token 并增加错合并风险。主题内合并, 命中率最高。

    主题里 fact 太少(< 3 条)时合并没意义, 直接各自成条。
    """
    fmap = {f.get("fact_id"): f for f in facts if f.get("fact_id")}
    all_claims: list[dict[str, Any]] = []
    _global_seq = [0]   # 跨主题递增, 保证 q1/q2/... 全局不撞号
    n_condensed_facts = 0        # 真正进了合并器的 fact 数(不含主题地图没覆盖的)
    for theme in theme_map.get("themes", []):
        tfs = [fmap[fid] for fid in (theme.get("fact_ids") or []) if fid in fmap]
        if not tfs:
            continue
        name = theme.get("name", "")
        if len(tfs) < 3:
            for i, f in enumerate(tfs, 1):
                all_claims.append({
                    "claim_id": f"{name}#{i}", "statement": (f.get("claim") or "").strip(),
                    "fact_ids": [f["fact_id"]], "paper_ids": [f.get("paper_id") or ""],
                    "theme_id": name, "n_facts": 1, "theme_name": name,
                })
            continue
        # 10-02: 同篇同 subject 先机械收成一条(见 _dedupe_same_paper_same_subject
        # 的实测数据), 否则模型面对的全是"一篇论文的各个细节", 不会合并。
        n_condensed_facts += len(tfs)
        tfs = _dedupe_same_paper_same_subject(tfs)
        batch_claims = condense_facts_into_claims(
            tfs, provider=provider, model=model, batch_size=batch_size)
        # 10-03 兜底: 范式层失败退回这条路时, 同一篇论文会在**多个 subject** 下
        # 各出一条命题, 于是成文时同一篇被连引 8 次(实测「二维网络变分优化」)。
        # _dedupe_same_paper_same_subject 挡不住这种(它们 subject 不同),
        # 所以这里再加一道**同主题内同论文**合并: 一篇论文在一个主题里只出一条
        # 命题, 其余作为该命题的 fact 支撑。
        batch_claims = _merge_same_paper_within_theme(batch_claims)
        # 9-30 修: claim_id 原来拼成 f"{name}#{cid}"。这个前缀**把标注语法搞脆了** ——
        # 模型看到 [张量网络与拟设#c1], 写出来多半是 [[c1]], 解析不上 -> 整句被丢,
        # 实测 4 批成文全部 0 字符。claim_id 是要让模型在句末**照抄**的, 必须好写。
        # 改成全局递增的短 id, 跨主题不撞号(所以要维护一个全局 seq)。
        for c in batch_claims:
            c["theme_name"] = name
            _global_seq[0] += 1
            c["claim_id"] = f"q{_global_seq[0]}"
        all_claims.extend(batch_claims)
    return all_claims


def _merge_same_paper_within_theme(
    claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """一个主题内, 同一篇论文只保留一条命题(最长的那条), 其余合并进去。

    10-03 实测: 范式层对部分主题失败退回旧路线时, 「二维网络变分优化」那 4262 字里
    同一篇论文被连引 8 次。`_dedupe_same_paper_same_subject` 挡不住 —— 那些 fact
    的 subject 各不相同, 只是碰巧来自同一篇论文。
    """
    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for c in claims:
        pids = [p for p in (c.get("paper_ids") or []) if p]
        if len(pids) != 1:
            # 多篇支撑的命题已经是"综合", 天然不会被重复引 —— 原样留
            key = f"_multi_{len(order)}"
            best[key] = c
            order.append(key)
            continue
        pid = pids[0]
        cur = best.get(pid)
        if cur is None:
            best[pid] = c
            order.append(pid)
            continue
        # 同一篇的第二条: 并进第一条, 保留信息更全的那条 statement
        merged_ids = list(dict.fromkeys(
            (cur.get("fact_ids") or []) + (c.get("fact_ids") or [])))
        if len(c.get("statement") or "") > len(cur.get("statement") or ""):
            cur["statement"] = c["statement"]
        cur["fact_ids"] = merged_ids
        cur["n_facts"] = sum(int(x.get("n_facts", 1)) for x in (cur, c))
    return [best[k] for k in order if best.get(k)]


def condense_facts_by_paradigm(
    facts: list[dict[str, Any]],
    theme_map: dict[str, Any],
    provider: str = "minimax",
    model: str = "MiniMax-M3",
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """主题 -> **范式** -> 命题。返回 (claims, paradigms)。

    这是引文密度的正解。实测(10-03, 110 条真实 fact, 主题「纯态张量网络表示」):
    归属后 MPS 5 篇 / PEPS 5 篇 / MERA 2 篇, 写出来均引 **12.0/句**
    (原路线 0.94/句)。

    为什么不再用 `condense_facts_by_theme`(主题 -> LLM 合并):
    fact 的 subject 是"按论文取的标签", 实测 163 个 subject 只有 4 个跨论文,
    93% 的 fact 其 subject 只对应一篇论文 —— LLM 面对"一篇论文的各个细节"
    正确地不合并(簇级压缩比 1.01), 于是命题 ≈1:1 对应论文。
    范式层把"该按什么归类"这件事从 fact 自带的标签里拿出来, 交给模型按
    **技术路线**重新判断, 再用向量质心把 fact 落到路线上(见 _paradigm)。

    范式层任何一步失败(模型没吐出 themes / 编码不可用)都返回 ([], {}),
    调用方退回 `condense_facts_by_theme` —— 宁可密度低, 不能丢内容。

    10-05: 返回值从 `list[claims]` 改成 `(claims, paradigms)`, paradigms 是
    `{主题名: [范式, ...]}` 供调用方落盘。以前范式只活在内存里、用完就拍扁进
    claim 的 statement 字符串, 于是磁盘上**从来没有任何地方存过"技术路线"
    这一层** —— 实测 10-03 走范式路线的 workspace 里 155 条命题没有一条带范式
    签名。而 `fact -> 范式 -> 主题` 的多归属结构(一条 fact 可以落进多个范式,
    见 `_paradigm._seed`)正是"一篇论文同时落在多个主题里"的来源, 不落盘就等于
    这个结构不存在。
    """
    from ._paradigm import assign, build_paradigms

    empty: tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]] = ([], {})
    fmap = {f["fact_id"]: f for f in facts if f.get("fact_id")}
    if not fmap:
        return empty
    paras = build_paradigms(list(fmap.values()), theme_map, provider, model)
    if not paras:
        print("[paradigm] 范式层没产出, 退回主题合并", flush=True)
        return empty

    all_claims: list[dict[str, Any]] = []
    used_paradigms: dict[str, list[dict[str, Any]]] = {}
    seq = 0
    # 10-03: 范式 statement 截到 120 字。这不是省 token, 是**压输入**:
    # 思维链长度随输入复杂度增长 —— 同样的成文调用, statement 是 146 字时
    # max_tokens=2000 正常出正文, 换成 2-3 句的长 statement 就全被 think 吃光
    # (实测返回 4795 字符、0 正文)。范式的**身份由 name 承担**, statement 只需
    # 交代机制和局限, 细节由写作模型自己补。
    for theme in theme_map.get("themes", []):
        name = (theme.get("name") or "").strip()
        plist = paras.get(name) or []
        if not plist:
            continue
        tfs = [fmap[i] for i in (theme.get("fact_ids") or []) if i in fmap]
        if not tfs:
            continue
        groups = assign(tfs, plist)
        for pn, g in groups.items():
            if not g:
                continue
            meta = next((p for p in plist if p["name"] == pn), {})
            pids = sorted({f.get("paper_id") or "" for f in g if f.get("paper_id")})
            seq += 1
            stmt = (meta.get("statement") or "").strip()[:120]
            all_claims.append({
                "claim_id": f"q{seq}",
                "statement": f"{meta.get('name', '')}: {stmt}".strip(),
                "fact_ids": [f["fact_id"] for f in g if f.get("fact_id")],
                "paper_ids": pids,
                "theme_id": name,
                "n_facts": len(g),
                "theme_name": name,
                # 10-05: 命题记着自己来自哪条范式。这是"主题 -> 范式 -> 命题"三层
                # 结构里中间那层的连接键, 落盘后才是可用的层级, 而不是三段平铺。
                "paradigm": pn,
            })
            # 10-05: 记录这个主题下实际**吃到 fact** 的范式(assign 可能把某条范式
            # 全部判给 margin 门槛掉了, 那就不该出现在结构里)
            used_paradigms.setdefault(name, []).append({
                "name": pn,
                "statement": stmt,
                "keywords": list(meta.get("keywords") or []),
                "n_facts": len(g),
                "fact_ids": [f["fact_id"] for f in g if f.get("fact_id")],
                "paper_ids": pids,
                "claim_id": f"q{seq}",
            })
    if not all_claims:
        print("[paradigm] 归属后没有命题, 退回主题合并", flush=True)
    return all_claims, used_paradigms


def write_theme_narrative_from_claims(
    theme: dict[str, Any],
    claims: list[dict[str, Any]],
    provider: str = "minimax",
    model: str = "MiniMax-M3",
    max_retry: int = 2,
    chunk_size: int = 12,
    target_chars: int = 900,
    max_claims: int = 60,
) -> dict[str, Any]:
    """Phase C (命题版): 围绕一个主题, 用命题写宏观叙述。

    与 fact 版的关键差别:
      - 写作单元从"一条 fact 一句话"变成"一条命题一句话"
      - **允许多个命题写进同一句**(逗号分隔标注) —— 这就是"一句话好几个引用"
      - 句末标注 `[[claim_id]]`, 渲染时展开成该命题所有出处的引用
      - 给了篇幅目标, 逼模型往"宏观描述"写而不是逐条罗列
    """
    from agent.tools.llm import _call_llm

    if not claims:
        return {"text": "", "used_claim_ids": [], "dropped_sentences": 0,
                "warnings": ["该主题没有可用命题"], "used_fact_ids": []}

    # 9-30: 写之前再筛一道。合并层压不下来的时候靠这里兜 ——
    # 实测 629 条 fact 合并后仍 476 条命题, 全写出来又变回几万字。
    # 优先选**跨论文支持**的命题, 写出来天然就是"一句话几个引用"。
    if len(claims) > max_claims:
        n_before = len(claims)
        claims = select_claims_for_writing(claims, max_claims)
        multi = sum(1 for c in claims if len(c.get("paper_ids") or []) > 1)
        print(
            f"[write_claims] 主题「{theme.get('name')}」命题 {n_before} -> 选 {len(claims)} "
            f"(其中跨论文 {multi} 条) 写",
            flush=True,
        )

    warnings: list[str] = []
    used_claim_ids: list[str] = []
    used_fact_ids: list[str] = []
    paras: list[str] = []
    dropped_total = 0
    failed = 0

    batches = [claims[i:i + chunk_size] for i in range(0, len(claims), chunk_size)]

    # 10-02 并发: 每批命题的 prompt / 请求 / 解析互相独立。
    # 顺序是硬约束 —— paras 顺序 = 最终段落排版, used_claim_ids 也按批序累积。
    # 并发只跑"拿 out", 渲染引用 / append 回主线程按原序做。
    def _write_one(_bi: int, batch: list[dict[str, Any]]) -> tuple[str, bool]:
        """纯函数: 组 prompt + 调 LLM(带重试) + 剥壳。返回 (正文, 是否建立命题对应)。"""
        lines = []
        for c in batch:
            src = (f"（{len(c['paper_ids'])} 篇）" if len(c.get("paper_ids") or []) > 1 else "")
            lines.append(f"[{c['claim_id']}]{src} {c.get('statement', '')}")
        block = "\n".join(lines)
        per_batch_chars = max(200, target_chars // max(1, len(batches)))

        prompt = THEME_WRITE_FROM_CLAIMS_PROMPT.format(
            theme_name=theme.get("name", ""),
            theme_statement=theme.get("statement", "") or "(未给出)",
            theme_notes=theme.get("notes", "") or "(无)",
            claims_block=block,
            n_claims=len(batch),
            target_chars=per_batch_chars,
        )

        out = ""
        for attempt in range(max_retry + 1):
            if attempt:
                prompt += (
                    "\n\n【重要 — 上一版没有任何命题标注, 已被作废】\n"
                    "每句话末尾**必须**跟 [[claim_id]] 标注, 只能取上面列出的。\n"
                    "没有命题支撑的话就不要写那一句。"
                )
            try:
                out = _call_llm(prompt, provider=provider, model=model,
                                enable_thinking=False, temperature=0.4,
                                # 9-30: 8000 -> 2000(实测每批只活下来约 245 字符)。
                                # 10-03: 2000 -> 6000。范式级命题的 statement 更长
                                # (120 字), 模型的思维链跟着变长, 2000 经常被 think
                                # 吃光 -> 返回 5000+ 字符却 0 正文(实测连续 3 次)。
                                # 注意这里跟**分类任务**相反: 写作是自由文本,
                                # think 长度和输出结构没有绑定, 加预算是安全的;
                                # 分类那种要吐结构化 JSON 的任务才是一加就废
                                # (3000 能过、5000 起 100% think、JSON 归零)。
                                max_tokens=6000, timeout=420)
            except Exception as e:
                logger.warning("[write_claims] batch %d 失败: %s", _bi, e)
                out = ""
            if not out:
                continue
            if "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
            out = out.replace("```markdown", "").replace("```", "").strip()
            if re.search(r"\[\[", out):
                return out, True
        return "", False

    _written = run_ordered(
        list(enumerate(batches, 1)), lambda pair: _write_one(pair[0], pair[1]),
        desc="write_claims",
    )
    for bi, (batch, res) in enumerate(zip(batches, _written, strict=True), 1):
        claim2pids = {c["claim_id"]: [p for p in c.get("paper_ids", []) if p]
                      for c in batch}
        if not res.ok:
            failed += 1
            warnings.append(
                f"主题「{theme.get('name')}」第 {bi}/{len(batches)} 批异常: {res.exc}"
            )
            continue
        out, ok = res.value
        if not ok:
            failed += 1
            warnings.append(f"主题「{theme.get('name')}」第 {bi}/{len(batches)} 批未能建立命题对应")
            continue

        kept, dropped = _drop_unattributed(out)
        body, unres = _render_claim_citations_or_drop(kept, claim2pids)
        dropped_total += dropped + unres
        if body:
            paras.append(body)
        for m in _ANNOT_RE.findall(kept):
            for cid in m.split(","):
                cid = cid.strip()
                if cid in claim2pids:
                    used_claim_ids.append(cid)

    if failed == len(batches) and batches:
        warnings.append(f"主题「{theme.get('name')}」全部批次均未建立命题对应, 已跳过")
        return {"text": "", "used_claim_ids": [], "used_fact_ids": [],
                "dropped_sentences": dropped_total, "warnings": warnings}

    if dropped_total:
        warnings.append(f"{dropped_total} 句因无命题标注被剔除")
    if len(batches) > 1:
        warnings.append(f"主题含 {len(claims)} 条命题, 分 {len(batches)} 批生成")

    cmap = {c["claim_id"]: c for c in claims}
    for cid in used_claim_ids:
        used_fact_ids.extend(cmap.get(cid, {}).get("fact_ids", []))

    return {"text": "\n\n".join(paras), "used_claim_ids": sorted(set(used_claim_ids)),
            "used_fact_ids": sorted(set(used_fact_ids)),
            "dropped_sentences": dropped_total, "warnings": warnings}


def _render_claim_citations_or_drop(text: str, claim2pids: dict[str, list[str]]
                                   ) -> tuple[str, int]:
    """`[[c1]]` -> `[pid1][pid2][...]`; 引不出出处的句子直接丢。

    与 fact 版同规则(严格): 一句里只要有一个标注解析不出, 整句不可信 ——
    抹掉标注留句子 = 制造"看起来通顺但无出处"的断言。
    """
    kept_paras, dropped = [], 0
    for para in (text or "").split("\n\n"):
        kept_lines = []
        for line in para.split("\n"):
            if not line.strip():
                kept_lines.append(line)
                continue
            kept_sents = []
            for sent in _SENT_SPLIT.split(line):
                if not sent.strip():
                    continue
                if not _ANNOT_RE.search(sent):
                    dropped += 1
                    continue
                ok = [False]
                bad = [False]

                def _sub(m, _ok=ok, _bad=bad):
                    pids, unresolved = [], 0
                    for part in m.group(1).split(","):
                        cid = part.strip()
                        got = claim2pids.get(cid) or []
                        if got:
                            for pid in got:
                                if pid not in pids:
                                    pids.append(pid)
                        else:
                            unresolved += 1
                    if not pids:
                        _bad[0] = True
                        return ""
                    _ok[0] = True
                    return "".join(f"[{p}]" for p in pids)

                new_sent = _ANNOT_RE.sub(_sub, sent)
                if bad[0]:
                    dropped += 1
                    continue
                kept_sents.append(new_sent)
            if kept_sents:
                kept_lines.append("".join(kept_sents))
        if any(x.strip() for x in kept_lines):
            kept_paras.append("\n".join(kept_lines))
    return "\n\n".join(kept_paras), dropped


def select_claims_for_writing(
    claims: list[dict[str, Any]],
    max_claims: int,
    max_per_paper: int = 3,
) -> list[dict[str, Any]]:
    """按"综述价值"挑命题, 而不是一股脑全写。

    排序依据(这个顺序是刻意的):
      1. **跨论文支持**的优先 —— 一条命题挂 3 篇, 写出来就是"一句话三个引用",
         这正是综述该有的样子; 只挂 1 篇的命题写出来还是逐篇罗列
      2. 覆盖的 fact 多 -> 信息量大

    为什么必须限: 9-30 实测 629 条 fact 合并后仍有 476 条命题, 拿 476 条命题去
    写 6000 字的综述, 比例完全不对(上一版 27000 字/40 篇, 现在要 6000 字)。
    命题层压不下来的时候, 只能在这里选 —— 选多论文支持的, 天然就会又短又"综"。

    max_per_paper: **同一篇论文最多贡献几条命题**。
    9-30 实测踩的坑: 60 条命题里好几条引同一篇, 结果 2242 字符只覆盖 **11 篇**,
    广度根本不够(用户要"一二百篇引 3 万字")。加了这个配额后, 同样的 60 条
    至少能覆盖 20 篇 —— 篇幅不变, 广度翻倍。
    """
    def key(c: dict[str, Any]):
        return (len(c.get("paper_ids") or []), c.get("n_facts", 1),
                len(c.get("fact_ids") or []))

    ordered = sorted(claims, key=key, reverse=True)
    picked: list[dict[str, Any]] = []
    paper_count: dict[str, int] = {}

    # 第一轮: 遵守每篇配额, 尽量铺开覆盖面
    for c in ordered:
        if len(picked) >= max_claims:
            break
        pids = [p for p in (c.get("paper_ids") or []) if p]
        if not pids:
            continue
        if max(paper_count.get(p, 0) for p in pids) >= max_per_paper:
            continue
        picked.append(c)
        for p in pids:
            paper_count[p] = paper_count.get(p, 0) + 1

    # 第二轮: 配额用完了还有名额, 放宽限制补满。
    # 按"这条能新增几篇没被引过的论文"排序 —— 上一版是无脑按原顺序补,
    # 结果配额被架空(实测 60 条还是单篇最多 7 条, 跟没加配额一样)。
    if len(picked) < max_claims:
        chosen = {id(c) for c in picked}
        covered = {p for c in picked for p in (c.get("paper_ids") or []) if p}
        rest = [c for c in ordered if id(c) not in chosen]
        rest.sort(key=lambda c: -len({p for p in (c.get("paper_ids") or [])
                                      if p and p not in covered}))
        for c in rest:
            if len(picked) >= max_claims:
                break
            picked.append(c)
            covered |= {p for p in (c.get("paper_ids") or []) if p}

    return picked[: max(0, max_claims)]


# ============================================================
# 组装
# ============================================================
# 9-30 修: 正文引用格式与文末参考文献表对不上。
# 实测(40 篇那次): `_render_citations_or_drop` 把 fact_id 映射成 `[{paper_id}]`,
# 而 OpenAlex 论文的 paper_id 就是 DOI 串, 于是正文渲染成
# `[10.1103/revmodphys.93.045003]`; `build_reference_list` 却按首次出现顺序编号成
# `[1]`..`[30]`。**两套格式永远对不上** —— 正文里一个数字引用都没有, 文末的编号
# 没有任何一条在正文被引用过。之前"引用编号连续无越界"那个结论就是这么查错的。
#
# 修法: 正文照旧先渲染成 `[paper_id]`, 全文拼完后**一次性编号化**,
# 同一张映射表交给 build_reference_list —— 一处编号, 两边天然一致。
# 引用形如 [10.1103/revmodphys.93.045003] / [arXiv:2401.01234] / [2301.01234]
# 不去猜 DOI 的位数(10.1103 是点前 2 位, 有的又是 4 位), 改成"先抓方括号内容,
# 再判断像不像标识符" —— 宁可漏判也不能把正常的 [1] 编号误当成 paper_id。
_BRACKET_RE = re.compile(r"\[([^\[\]\n]{3,120})\]")
# 三种 paper_id 形态都要认:
#   10.1103/revmodphys.93.045003   DOI(OpenAlex/APS 来源)
#   arXiv:2401.01234              带前缀(少见)
#   2301.01234                    裸 arXiv ID —— _arxiv.py 的 paper_id 就是这个形态,
#                                  漏了它参考文献编号会跟正文对不上
#   quant-ph/0003004              旧格式 arXiv(带 archive 前缀) —— 10-04 补, 漏了它
#                                  旧格式论文既不编号也不进文献表, 而正文里挂着
#                                  [quant-ph/0003004], 覆盖度审计还报它 0 引用
#   W2741809807 / OA-<md5>        无 DOI 时的 OpenAlex 兜底 id —— 10-04 补, 同上:
#                                  _safe_paper_id 的第 3/4 优先级会产出这两种形态,
#                                  之前 _IDISH_RE 完全不认, 于是它们逃出参考文献表
_IDISH_RE = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/|doi:)?\d{2,}\.\d{4,}/\S+$"   # DOI
    r"|^arxiv[:/]\S+$"                                              # arXiv:xxx
    r"|^\d{4}\.\d{4,5}(?:v\d+)?$"                                   # 裸 arXiv 新格式
    r"|^[a-z][a-z0-9-]*(?:\.[A-Za-z]{2})?/\d{7}$"                   # 旧格式 arXiv
    r"|^W\d+$"                                                      # OpenAlex work id
    r"|^OA-[0-9a-f]{8,}$",                                          # OpenAlex 标题 hash
    re.I,
)


def _looks_like_paper_id(s: str) -> bool:
    return bool(_IDISH_RE.match(s.strip()))


def renumber_citations(text: str) -> tuple[str, dict[str, int]]:
    """把正文里的 `[paper_id]` 换成 `[1] [2] ...`, 按首次出现顺序编号。

    同一句里同一篇被引多次时折叠成一次 —— 之前"一段里同一篇被引 5 次"的
    观感, 在编号化这一层就能压掉一半(合并命题层是另一半)。

    Returns:
        (编号后的正文, {paper_id: 编号})
        没被引用的论文不会进映射, build_reference_list 也就不会收它们。
    """
    pid2num: dict[str, int] = {}

    def _sub(m: re.Match) -> str:
        pid = m.group(1).strip()
        if pid.isdigit() or not _looks_like_paper_id(pid):
            return m.group(0)                  # 已经是编号的 / 不是标识符的, 别动
        if pid not in pid2num:
            pid2num[pid] = len(pid2num) + 1
        return f"\x00{pid2num[pid]}\x00"

    new_text = _BRACKET_RE.sub(_sub, text or "")
    # 临时标记换成 [n] 形式
    new_text = re.sub(r"\x00(\d+)\x00", lambda m: f"[{m.group(1)}]", new_text)
    # 句内重复的同一编号只留第一次。
    # 必须**按句**做而不是只匹配相邻: 实测症状是
    #   "A [1] B [1] C [1] D [1] E [1]" —— 中间隔着文字, 只匹配相邻折不掉。
    out_lines = []
    for line in new_text.split("\n"):
        out_sents = []
        for sent in _SENT_SPLIT.split(line):
            seen_in_sent: set[str] = set()

            def _dedup(m: re.Match, _s=seen_in_sent) -> str:
                num = m.group(1)
                if num in _s:
                    return ""            # 这一句里已经引过这篇了
                _s.add(num)
                return m.group(0)

            out_sents.append(re.sub(r"\[(\d+)\]", _dedup, sent))
        out_lines.append("".join(out_sents))
    new_text = "\n".join(out_lines)
    # 折叠后可能留下多余空格, 收一下
    new_text = re.sub(r"[ \t]{2,}", " ", new_text)
    return new_text, pid2num


def build_reference_list(
    text: str,
    pool: list[dict[str, Any]],
    title: str = "参考文献",
    pid2num: dict[str, int] | None = None,
) -> str:
    """
    按正文**实际出现顺序**生成编号参考文献表。

    9-29 用户要求: "引用的论文虽然在正文里有 doi, 但最后还是希望有一个文献列表"。

    只收录正文真正引用到的 paper_id(按首次出现顺序编号), 不掺没引用的 ——
    未引用的论文会在 Phase B 另列为"文献覆盖的空白", 不该出现在参考文献里。

    9-30: 传 pid2num 时用它自己的编号(正文已被 renumber_citations 编号化),
    不再从正文重新解析 —— 之前"正文渲染成 [DOI]、文末编号成 [1]"两套格式
    对不上, 编号对不上就等于没有引用。
    """
    by_id = {p.get("paper_id"): p for p in pool}
    if pid2num:
        order = list(pid2num)
        numbers = dict(pid2num)
    else:
        order = []
        for pid in _extract_cited_ids(text):
            if pid not in order:
                order.append(pid)
        numbers = {pid: i for i, pid in enumerate(order, 1)}
    if not order:
        return ""
    lines = [f"## {title}", ""]
    for pid in sorted(order, key=lambda x: numbers.get(x, 99999)):
        i = numbers.get(pid, 0)
        p = by_id.get(pid) or {}
        bits = []
        if p.get("title"):
            bits.append(str(p["title"]).strip())
        meta = []
        if p.get("journal"):
            meta.append(str(p["journal"]))
        if p.get("year"):
            meta.append(str(p["year"]))
        if meta:
            bits.append("(" + ", ".join(meta) + ")")
        lines.append(f"[{i}] {''.join(bits)}  `{pid}`")
    return "\n".join(lines) + "\n"


def _extract_cited_ids(text: str) -> list[str]:
    """从正文抽 paper_id, 按首次出现顺序去重。"""
    from ._pool._utils import _extract_paper_ids
    out: list[str] = []
    for pid in _extract_paper_ids(text or ""):
        if pid not in out:
            out.append(pid)
    return out


def render_survey(
    theme_map: dict[str, Any],
    narratives: dict[str, dict[str, Any]],
    title: str = "",
    subtitle: str = "",
    pool: list[dict[str, Any]] | None = None,
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
    body_parts: list[str] = []
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
        body_parts.append(head + "\n\n" + text)
    gaps = theme_map.get("gaps") or []
    if gaps:
        body_parts.append("## 文献覆盖的空白\n\n" + "\n".join(f"- {g}" for g in gaps))
    # 10-06 加: 覆盖空白回补的记录。
    # **只报这一轮真的判过、真的搜过的那些空白原文**(theme_map.gap_fill_rounds),
    # 绝不把上一轮的判定贴到这一轮重新生成的 gaps 上 —— 实测那样会 32 条全部
    # 标成"判定补不上", 包括刚搜成功的那几条。看着权威但全错, 比静默更糟。
    for r in theme_map.get("gap_fill_rounds") or []:
        acted = r.get("acted") or []
        not_acted = r.get("judged_not_acted") or []
        body_parts.append(
            f"## 覆盖空白回补(第 {r.get('round', '?')} 轮)\n\n"
            f"针对上一轮归纳出的 {r.get('n_judged', '?')} 条空白打分后, "
            f"回补 {len(acted)} 条, 新增 {r.get('n_new_papers', 0)} 篇论文 / "
            f"{r.get('n_new_facts', 0)} 条 fact。"
            + (f"\n\n判定为补不上、本轮没去搜的 {len(not_acted)} 条: "
               + "\n".join(f"- {g}" for g in not_acted) if not_acted else "")
            + "\n\n本轮实际检索的空白:\n"
            + "\n".join(
                f"- {a.get('score', '?')} 分 · {a.get('gap', '')}\n"
                f"  → 检索式 `{a.get('query', '')}`"
                for a in acted))
    # 9-30: 正文先拼好(此时引用还是 [paper_id]), **一次性编号化**,
    # 同一张映射表交给参考文献表 —— 之前两边各自编号, 格式对不上。
    body, pid2num = renumber_citations("\n\n".join(body_parts))
    if pool:
        refs = build_reference_list(body, pool, pid2num=pid2num)
        if refs:
            body = body + "\n\n" + refs
    body_parts = [body]
    if body_parts:
        parts.append("\n\n".join(body_parts))
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
