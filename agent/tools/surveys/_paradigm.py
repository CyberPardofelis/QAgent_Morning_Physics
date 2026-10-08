"""
agent.tools.surveys._paradigm — 综述的"范式层"(10-03)

要解决的问题
------------
引文密度上不去, 根因不在写作 prompt, 在**命题层没在合并**。
实测(v60g, 241 条可用 fact): 163 个 subject 里只有 4 个跨论文,
93% 的 fact 其 subject 只对应一篇论文 —— 于是命题 ≈1:1 对应论文,
成文时一句只能挂一个引用(实测均引 0.94/句)。

正确的作用点是**范式**: 论文各自不同是必然的(不然发不出来), 但它们落在
不同的**技术路线**上, 而每种路线天然有多篇论文支撑:

    纯态张量网络表示
    ├── MPS 表示     12 篇   用链状张量表示一维多体波函数
    ├── PEPS 表示    11 篇   MPS 的二维格点推广
    ├── MERA          7 篇   树状结构, 显式编码多尺度纠缠
    ├── 等距张量网络态  5 篇   施加等距约束以高效收缩
    └── 对称性张量网络  5 篇   用表示论把对称性嵌入局域张量

一类做法 = 一条命题 = 挂上这一类的全部论文。实测成文后均引 **12.0/句**。

为什么归属不能靠关键词
----------------------
试过了, 撑不住。范式级关键词恰好是**对象名**(PEPS / MPS / 面积律), 跟 fact 的
subject 同词面, 所以能对上; 拆到子面层级后关键词变成**结论性描述**
("二维面积律纠缠"), 而 fact 的 subject 是对象名("PEPS"), 词面根本不重合 ——
MPS 的侧面全丢, 21 篇只剩 11 篇(实测 10-02)。

也不能让模型自己点名 fact: 实测 4 次全灭 —— 只要输出里出现"编号 -> 类别"的
映射, 模型就一定要先"核对"这个分配, 然后陷进组合记账(它会算 `+5+6+10=`,
**自己还加错**), 预算全烧在 think 里, JSON 一个字节没有。给短序号、说漏掉
没关系、max_tokens 从 4000 加到 24000, 全都无效。

所以: **关键词播种 + 向量质心重新归属**。
    1. LLM 只出"有哪些范式"(名字 + 一句话 + 关键词), 不碰 fact 编号 —— 这步稳
    2. 关键词给每个范式播种一批 fact(>=1 个关键词命中)
    3. 种子数够的范式算质心, 拿质心把全部 fact 重新归属(带 margin 门槛)
    4. 分不出 / 种子不足的 fact 不归属 —— 主题地图本来就容得下未覆盖的 fact

实测(110 条真实 fact, 主题「纯态张量网络表示」, bge-m3):
    质心归属 margin=0.04 -> 覆盖 87%, 论文分布 MPS 5 篇 / PEPS 5 篇 / MERA 2 篇,
    落在"每句挂 3-6 个引用"的健康区
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ._embed import encode

logger = logging.getLogger(__name__)

# 范式层 LLM 调用: 只出名字/说明/关键词。max_tokens 必须给**小** ——
# 10-02 实测 3000 能成, 5000 就开始"想太多"烧光预算反而不输出 JSON。
PARADIGM_MAX_TOKENS = 3000
PARADIGM_RETRIES = 2

# 归属参数(10-03 实测调出来的)
SEED_MIN_HITS = 1      # 播种: 命中几个关键词算种子
# 10-03: 3 -> 2。种子只有 1 条时质心就是那一条自己, 会被自己的范式全吸走;
# 2 条以上质心才有意义。实测这一改把覆盖率从 38% 提上去, 精度基本不变
# (margin 仍守着边界, 没动)。
SEED_MIN_COUNT = 2
ASSIGN_MARGIN = 0.04   # 领先次优多少才认这个归属
PARADIGM_PROMPT = """你要为一个文献综述的一个主题做技术路线细分, 输出严格 JSON。

【为什么需要这一层】
论文各自不同是必然的(不然发不出来), 但它们落在不同的**技术路线**上,
而每种路线天然有多篇论文支撑。综述的一类做法配一组论文, 这样一句话才能挂多个引用。

【粒度】
这个主题是「{theme_name}」。你要给出的是**具体有哪几条技术路线**,
每条路线 2-5 个, 不要更细(更细就退化成"一篇论文的各个细节"了)。

【fact 笔记】(只读, **不要在输出里引用编号**)
{facts_block}

【铁律】
1. **不要统计、不要核对、不要验算任何数量。** 把预算用来做判断。
   (实测教训: 让模型逐条核对覆盖, 它会陷进去反复数, 预算全烧光, JSON 一个字写不出。)
2. **不要输出 fact_id 或任何编号。**
3. **不要数字符、不要检查关键词长短。** 关键词直接写出来就行。
   (实测教训: prompt 里给了 "术语1" 这种占位示例, 模型会去数它几个字符,
   数完嫌长又想改, 反复重数, 预算全烧在 think 里 —— 范式 JSON 一个字没输出。)
4. 每条路线给 3-5 个关键词。
5. 关键词要挑"能把这批 fact 认出来的"对象名/术语, 不要写结论性描述。
   (反例"纠缠结构的多尺度分析" —— 这是结论, fact 文本里不会出现;
    正例"矩阵乘积态" "正则形式" "Vidal" "一维链" —— 这些 fact 里真出现)
6. 写完立刻输出, 不要回头检查。

【输出格式】严格 JSON, 不要 markdown:
{{"paradigms": [
  {{"name": "路线名",
    "statement": "2-3 句: 是什么 / 核心机制 / 解决什么或有什么局限",
    "keywords": ["矩阵乘积态", "正则形式", "Vidal", "一维链"]}}
]}}

只输出 JSON。"""


def _norm(s: str) -> str:
    return re.sub(r"[\s、,，。;；:：()（）\[\]【】]+", "", s or "")


def _salvage_paradigms(body: str) -> list[dict[str, Any]] | None:
    """从**被截断**的输出里抢救出已经写完整的范式对象。

    10-03 实测: 7 个主题里 2 个的范式调用把结构**完全正确**地写完了大半,
    只是在 max_tokens 处被截断(输出 9637 / 8351 字符, 尾部停在某个 statement
    中间)。整块丢掉等于这个主题白跑 —— 而丢掉的主题会退回旧路径,
    退回的主题恰恰暴露出"同一篇论文连引 8 次"的老毛病。

    做法: 用栈扫出所有**括号平衡**的对象, 逐个 json.loads, 只收"长得像范式"的
    (有 name + keywords)。前面的完整对象照收, 残缺的那个解析失败自然丢掉。

    注意**不能只在 depth==0 记录**: 输出被截断时外层 `{"paradigms": [` 是不平衡的,
    于是里面每个范式对象都处在 depth>=1, 只认顶层会把它们全漏掉 —— 这正是
    2026-10-03 第一版抢救逻辑的 bug, 对着真实失败样本才发现。
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
            if isinstance(obj, dict) and isinstance(obj.get("paradigms"), list):
                out.extend(x for x in obj["paradigms"] if isinstance(x, dict))
            elif isinstance(obj, dict) and obj.get("keywords") is not None:
                out.append(obj)
    if not out:
        return None
    res, seen = [], set()
    for p in out:
        pn = (p.get("name") or "").strip()
        kws = [k.strip() for k in (p.get("keywords") or []) if k and k.strip()]
        if not (pn and kws) or (pn, tuple(kws)) in seen:
            continue
        seen.add((pn, tuple(kws)))
        res.append({"name": pn, "statement": (p.get("statement") or "").strip(),
                    "keywords": kws})
    return res or None


def _parse(raw: str, theme_name: str) -> list[dict[str, Any]] | None:
    """解析单主题的范式 JSON。模型常把结构写成 {"themes":[...]} 或直接 {"paradigms":[...]}。"""
    body = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    i = body.find("{")
    if i < 0:
        return None
    for cand in (body[i:], body[i:] + "}", body[i:] + '"}', body[i:] + "}}",
                 body[i:] + "}]", body[i:] + "}}}"):
        try:
            v = json.loads(cand)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        themes = v.get("themes")
        if isinstance(themes, list) and themes:
            v = themes[0]
            if not isinstance(v, dict):
                continue
        ps = v.get("paradigms")
        if isinstance(ps, list) and ps:
            out = []
            for p in ps:
                if not isinstance(p, dict):
                    continue
                pn = (p.get("name") or "").strip()
                kws = [k.strip() for k in (p.get("keywords") or []) if k and k.strip()]
                if pn and kws:
                    out.append({"name": pn,
                                "statement": (p.get("statement") or "").strip(),
                                "keywords": kws})
            if out:
                return out
    # 严格解析失败 -> 抢救已写完整的部分
    salvaged = _salvage_paradigms(body[i:])
    if salvaged:
        logger.warning("[paradigm] 输出被截断, 抢救出 %d 条完整范式", len(salvaged))
    return salvaged


def build_paradigms(
    usable: list[dict[str, Any]],
    theme_map: dict[str, Any],
    provider: str = "minimax",
    model: str = "MiniMax-M3",
) -> dict[str, list[dict[str, Any]]]:
    """{主题名: [范式, ...]}。失败返回空 dict —— 调用方退回原路线。

    **按主题拆成 N 次调用**, 不是一次调用出全部主题。
    实测(10-03, 6 主题 / 256 条 fact): 合成一次调用时 prompt 11586 字符,
    模型写出结构完全正确的 JSON 但**在 3000 token 处被截断**(输出 8245 字符,
    数组断在中途) -> 解析失败。而且这类结构化输出**不能靠调大 max_tokens 解决**:
    实测 4000/6000/8000 全是 100% think、JSON 0 字节(见模块文档)。
    拆开之后每次只出 2-5 条范式, 输出量小且稳; 6 次调用互不依赖, 顺便并发。
    """
    from agent.tools.llm import _call_llm

    from ._parallel import run_ordered

    fmap = {f["fact_id"]: f for f in usable if f.get("fact_id")}
    jobs: list[tuple[str, list[dict[str, Any]]]] = []
    for t in theme_map.get("themes", []):
        tfs = [fmap[i] for i in (t.get("fact_ids") or []) if i in fmap]
        if len(tfs) >= 2:
            jobs.append(((t.get("name") or "").strip(), tfs[:40]))
    if not jobs:
        return {}

    def _one(job):
        name, tfs = job
        body = "\n".join(
            f"- <{(f.get('subject') or '').strip()}> {(f.get('claim') or '').strip()}"
            for f in tfs
        )
        prompt = PARADIGM_PROMPT.format(theme_name=name, facts_block=body)
        for _attempt in range(PARADIGM_RETRIES):
            try:
                out = _call_llm(prompt, provider=provider, model=model,
                                enable_thinking=False, temperature=0.2,
                                max_tokens=PARADIGM_MAX_TOKENS, timeout=420)
            except Exception as e:
                logger.warning("[paradigm] 主题「%s」调用失败: %s", name, e)
                return name, None
            got = _parse(out, name)
            if got:
                return name, got
            logger.warning("[paradigm] 主题「%s」没解析出(输出 %d 字符)尾部: %s",
                           name, len(out or ""), (out or "")[-200:])
        return name, None

    out: dict[str, list[dict[str, Any]]] = {}
    for name, ps in (r.value if r.ok else (None, None)
                     for r in run_ordered(jobs, _one, desc="paradigm")):
        if name and ps:
            out[name] = ps
    return out


def _seed(facts: list[dict[str, Any]], paradigms: list[dict[str, Any]]
          ) -> dict[str, set[int]]:
    """关键词播种: 返回 {路线名: fact 下标集合}。"""
    seed: dict[str, set[int]] = {p["name"]: set() for p in paradigms}
    for i, f in enumerate(facts):
        hay = _norm(f"{f.get('subject') or ''} {f.get('claim') or ''}")
        for p in paradigms:
            hits = sum(1 for k in p["keywords"] if _norm(k) and _norm(k) in hay)
            if hits >= SEED_MIN_HITS:
                seed[p["name"]].add(i)
    return seed


def assign(
    facts: list[dict[str, Any]],
    paradigms: list[dict[str, Any]],
    margin: float = ASSIGN_MARGIN,
) -> dict[str, list[dict[str, Any]]]:
    """{路线名: [fact, ...]}。编码失败时退回纯关键词播种。"""
    if not paradigms:
        return {}
    seed = _seed(facts, paradigms)

    def _by_seed() -> dict[str, list[dict[str, Any]]]:
        return {n: [facts[i] for i in sorted(idxs)]
                for n, idxs in seed.items() if idxs}

    texts = [(f.get("claim") or f.get("subject") or "").strip() for f in facts]
    vec = encode(texts) if any(len(v) >= SEED_MIN_COUNT for v in seed.values()) else None
    if vec is None or len(vec) != len(facts):
        logger.info("[paradigm] 编码不可用(%d 条 fact), 退回关键词播种",
                    len(facts))
        return _by_seed()

    import numpy as np
    names, cents = [], []
    for p in paradigms:
        idxs = seed.get(p["name"]) or set()
        if len(idxs) < SEED_MIN_COUNT:
            continue          # 种子太少, 质心不可信 -> 这个范式不参与归属
        c = vec[list(idxs)].mean(0)
        n = float(np.linalg.norm(c))
        if n < 1e-6:
            continue
        names.append(p["name"])
        cents.append(c / n)
    if not names:
        return _by_seed()

    sim = vec @ np.array(cents).T
    order = np.argsort(-sim, axis=1)
    out: dict[str, list[dict[str, Any]]] = {n: [] for n in names}
    for i in range(len(facts)):
        best = int(order[i][0])
        if sim[i, best] <= 0:
            continue
        if len(names) >= 2 and float(sim[i, best] - sim[i, int(order[i][1])]) < margin:
            continue          # 两条路线难分 -> 不归属
        out[names[best]].append(facts[i])
    return {n: v for n, v in out.items() if v}


__all__ = ["ASSIGN_MARGIN", "assign", "build_paradigms"]
