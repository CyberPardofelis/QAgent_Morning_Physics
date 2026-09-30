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

from agent.prompts_survey import (
    CLAIM_CONDENSE_PROMPT,
    THEME_WRITE_FROM_CLAIMS_PROMPT,
    _tokenize,
)

logger = logging.getLogger(__name__)

# 9-30: 主题地图的输出量随 fact 数线性增长, 顶到 max_tokens 就截断 -> JSON 解析失败。
# 实测 303 条 fact 时返回 59349 字符, 远超 20000 max_tokens 对应的量。
THEME_MAP_MAX_FACTS = 200
THEME_MAP_MAX_TOKENS = 32000


def _sample_facts_for_themes(
    facts: list[dict[str, Any]],
    clusters: list[tuple[str, list[dict[str, Any]]]] | list,
    cap: int,
) -> list[dict[str, Any]]:
    """按 subject 聚簇**均匀**抽样到 cap 条。

    为什么均匀而不是取前 cap 条: 主题地图的作用是给出"这个领域的结构"，
    尾部 subject 往往正是那些**小众但有代表性**的分支(比如"费米子张量网络"),
    直接截尾会把它们整个丢掉, 主题结构就缺了一块。
    均匀抽样保证每个 subject 都有代表。
    """
    if len(facts) <= cap:
        return list(facts)
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    # 第一轮: 每个聚簇至少取 1 条(有 fact_id 的)
    for _name, members in clusters:
        for m in members:
            fid = m.get("fact_id")
            if fid and fid not in seen:
                seen.add(fid)
                picked.append(m)
                break
        if len(picked) >= cap:
            return picked[:cap]
    # 第二轮: 按聚簇轮流补齐到 cap。
    # 起始索引必须是 1 —— 第一轮已经取过每个聚簇的 members[0] 了,
    # 从 0 开始会全部命中 seen, added 恒为 False 直接 break(实测只抽出 5 条)。
    round_no = 1
    while len(picked) < cap and round_no < 500:
        added = False
        for _name, members in clusters:
            if round_no < len(members):
                m = members[round_no]
                fid = m.get("fact_id")
                if fid and fid not in seen:
                    seen.add(fid)
                    picked.append(m)
                    added = True
                    if len(picked) >= cap:
                        break
        if not added:
            break
        round_no += 1
    return picked[:cap]


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

    # 9-30 加固: 主题地图的**输出**随 fact 数线性增长, 超过 max_tokens 就被截断,
    # JSON 解析必失败。实测 60 篇那次: 303 条 fact -> 25K prompt -> 返回 59349 字符,
    # 撞上 20000 max_tokens 上限, 主题地图解析失败, 下游(命题层/成文)全断,
    # 整轮白跑 40 分钟。40 篇那次(295 条)只是**侥幸**没过线。
    #
    # 三管齐下:
    #   1) fact 数封顶 —— 主题地图要的是"有代表性的分布"而不是穷尽所有 fact,
    #      超出的按 subject 聚簇均匀抽样, 保证每个 subject 都有代表
    #   2) max_tokens 提到 32000
    #   3) 解析失败时**自动折半重试**, 而不是整轮报废
    if len(facts) > THEME_MAP_MAX_FACTS:
        facts = _sample_facts_for_themes(facts, clusters, THEME_MAP_MAX_FACTS)
        clusters = cluster_by_subject(facts)
        print(
            f"[organize] fact 数超上限 {THEME_MAP_MAX_FACTS}, 按 subject 聚簇均匀抽样到 "
            f"{len(facts)} 条 (再多的输出会撑爆 max_tokens 导致 JSON 截断)",
            flush=True,
        )

    prompt = THEME_MAP_PROMPT.format(
        n_subject_histogram=_subject_histogram(clusters),
        facts_block=_facts_block(facts, paper_titles),
    )

    # 9-29 实测: MiniMax 即使 enable_thinking=False 也会输出大段 <think>,
    # 那部分**照样吃 max_tokens**。8000 会被思考吃光, JSON 答案被截断 -> 解析失败。
    # 主题地图是要"想清楚"的活, 给足预算。
    parsed = None
    last_err = ""
    n_now = len(facts)
    for _ in range(3):                     # 最多折半重试 2 次
        out = _call_llm(prompt, provider=provider, model=model,
                        enable_thinking=False, temperature=temperature,
                        max_tokens=THEME_MAP_MAX_TOKENS, timeout=600)
        if not out:
            last_err = "LLM 返回空"
        else:
            parsed = _parse_theme_json(out)
            if parsed is not None:
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
    if parsed is None:
        return {"themes": [], "narrative_arc": [], "gaps": [], "warnings": [last_err]}

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
    for cluster in _batch_by_subject(facts, batch_size):
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
        groups: list[list[int]] = []
        for _ in range(2):          # 一次重试, 提高 JSON 服从率
            try:
                out = _call_llm(prompt, provider=provider, model=model,
                                enable_thinking=False, temperature=temperature,
                                max_tokens=6000, timeout=300)
            except Exception as e:
                logger.warning("[condense_claims] batch 失败: %s", e)
                out = ""
            if out and "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
            groups = _parse_claim_groups(out, len(cluster))
            if groups:
                break
        if not groups:
            groups = [[i] for i in range(1, len(cluster) + 1)]   # 解析失败 -> 不强凑

        # 覆盖检查: LLM 可能漏掉某些 fact, 漏的单独成条, 绝不能静默丢
        covered = {i for g in groups for i in g}
        for i in range(1, len(cluster) + 1):
            if i not in covered:
                groups.append([i])

        for g in groups:
            members = [cluster[i - 1] for i in g if 1 <= i <= len(cluster)]
            if not members:
                continue
            seq += 1
            pids: list[str] = []
            for m in members:
                pid = m.get("paper_id") or ""
                if pid and pid not in pids:
                    pids.append(pid)
            statement = " ".join((m.get("claim") or "").strip() for m in members)
            claims.append({
                "claim_id": f"c{seq}",
                "statement": statement,
                "fact_ids": [m["fact_id"] for m in members if m.get("fact_id")],
                "paper_ids": pids,
                "theme_id": members[0].get("theme_id"),
                "n_facts": len(members),
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


_CLAIM_GROUP_RE = re.compile(r"\{[^{}]*\}")


def _parse_claim_groups(raw: str, n: int) -> list[list[int]]:
    """从 LLM 输出里解析 [[1,3,5], [2], ...] 或 [{"fact_idx":[1,3]}, ...]。
    两种格式都收 —— 模型的 JSON 服从率不是 100%, 宽松点比整批丢掉强。"""
    if not raw:
        return []
    txt = raw.replace("```json", "").replace("```", "").strip()

    groups: list[list[int]] = []
    # 优先按 JSON 对象分组 {"fact_idx": [1,3]}
    for chunk in _CLAIM_GROUP_RE.findall(txt):
        idxs = [int(x) for x in re.findall(r"\d+", chunk)]
        idxs = [i for i in idxs if 1 <= i <= n]
        if idxs:
            groups.append(sorted(set(idxs)))
    if groups:
        return groups

    # 整块形如 [[1, 3], [2, 5]] 或 [[1, 2, 3]]。
    # 关键: 不能用 `\[\[...\]\]` 那种"内层排除方括号"的写法 —— 组间就是靠
    # "], [" 断开的, 排除方括号会导致整块匹配不上(实测返回空)。
    # 改成先抓"只由数字/空白/逗号/方括号组成"的整块, 剥掉外层再按 "], [" 拆组。
    for block in re.findall(r"\[[\d\s,\[\]]{3,}\]", txt):
        inner = block.strip()[1:-1].strip()      # 剥最外层方括号
        for part in re.split(r"\]\s*,\s*\[", inner):
            idxs = [int(x) for x in re.findall(r"\d+", part)]
            idxs = [i for i in idxs if 1 <= i <= n]
            if idxs:
                groups.append(sorted(set(idxs)))
    return groups


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
        batch_claims = condense_facts_into_claims(
            tfs, provider=provider, model=model, batch_size=batch_size)
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
    for bi, batch in enumerate(batches, 1):
        claim2pids = {c["claim_id"]: [p for p in c.get("paper_ids", []) if p]
                      for c in batch}
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
                                max_tokens=8000, timeout=420)
            except Exception as e:
                logger.warning("[write_claims] batch %d 失败: %s", bi, e)
                out = ""
            if not out:
                continue
            if "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
            out = out.replace("```markdown", "").replace("```", "").strip()
            if re.search(r"\[\[", out):
                break
        else:
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
_IDISH_RE = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/|doi:)?\d{2,}\.\d{4,}/\S+$"   # DOI
    r"|^arxiv[:/]\S+$"                                              # arXiv:xxx
    r"|^\d{4}\.\d{4,5}(?:v\d+)?$",                                  # 裸 arXiv ID
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
