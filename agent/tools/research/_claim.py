"""
agent.tools.research._claim — 命题规范化 + 证据绑定 + 报告渲染 (P7 阶段 2)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §8 条款 1/3

================================ 这一层是干什么的 ================================
P0/P1 把"单条命题验证"做通了(命题 → 代码 → 沙箱 → verdict → 证据)。
P2 要把它变成一个**研究任务**: 多个命题、多轮尝试、一份能给人看的报告。

而"能给人看"这件事的难点不在排版, 在于**防裸断言**:
  报告里每一句结论都必须挂着 ev_XXXX 证据编号, 挂不上的一律降级进"未验证"区。

所以本模块提供三件事:
  1. normalize_claim()  命题规范化 + **可判定性判定**(§8 条款 3)
  2. render_report()     报告渲染 —— 证据绑定规则在这里**编码**, 不靠 prompt 叮嘱
  3. audit_claims()      独立复核(0 LLM), 抓"宣称 proved 但证据只是 partial"这类翻车

**第 3 件事为什么不能省**: prompt 已经在 §8 里要求 LLM 标 VERDICT 了, 但 prompt 是软约束。
审计是硬约束 —— 它只读磁盘上的证据文件, 不看 LLM 说了什么。
这是 §0 里"验证器说了算"原则的最后一环。

历史:
  - 2026-10-08 P7.2 首版
"""
from __future__ import annotations

import re
from typing import Any

# 命题状态: 比 verdict 多一个 undecidable(不可判定, 是正当产物不是失败)
CLAIM_STATES: tuple[str, ...] = (
    "pending",      # 还没验
    "proved",
    "refuted",
    "partial",
    "failed",
    "inconclusive",
    "undecidable",  # 建模阶段就判定"这条没法写成可判定断言" (§8 条款 3)
)

#: 只有这三种 verdict 算"有结论"(partial/undecidable 都不算, 必须带范围限定说话)
CONCLUSIVE: tuple[str, ...] = ("proved", "refuted")

#: 结论区允许出现的判定, 其余一律降级
_REPORTABLE: tuple[str, ...] = ("proved", "refuted", "partial", "inconclusive")

#: 量词在报告/流式回显里的中文标签。"对所有" 和 "存在" 的结论正好相反,
#: 所以标签必须一眼分得清, 不能缩成字母。
_QUANT_LABEL: dict[str, str] = {
    "forall": "∀ 对**所有** n 成立",
    "exists": "∃ **存在** n 使其成立",
    "none": "∄ **不存在** n 使其成立",
}

# ~~UNDECIDABLE_MARKERS~~ 已于 P4.5 第三轮删除(关键词黑名单, 见上方 1b-2 说明段)。
# 保留它就等于保留一条"漏得比不拦还危险"的防线: 真实翻车里最刁钻的几道
# (「最小的不存在的素数」「n>100000 的阶乘」「哥德巴赫猜想成立吗」)
# 一个标记词都不含, 全都放行。

_EVIDENCE_REF = re.compile(r"\bev_\d{4}\b")


# ============================================================
# 1b. 量词 —— P4 实跑打出来的补丁
# ============================================================
# 为什么单独一节:
#   P4 实跑(2026-10-08)里,「对所有正整数 n, n^2 小于 2n」这个**明确的假命题**
#   (n=2 时 4<4 就已经假了)被判成了 proved, 同一任务里还并存 refuted。
#   根因是"用户问题 → 命题"这段没有量词校验: 题面里的"对所有"在拆解时被丢掉,
#   生成的验证代码对着一个"存在 n 使 n²<2n"在跑 —— 代码没错, 命题被换了。
#
# 量词错一个, 结论整个反过来。所以它必须是一个**独立字段**, 不能藏在中文措辞里。

#: 合法量词。刻意用穷举而不是开放字符串 —— LLM 编一个新值时直接当成缺失处理。
QUANTIFIERS: tuple[str, ...] = ("forall", "exists", "none")

#: 量词标记词。**顺序有意义**: "不存在" 必须排在 "存在" 前面, 否则
#: "不存在这样的 n" 会被判成 exists, 结论正好反掉。
_QUANT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("none", (
        "不存在", "无解", "永不成立", "没有反例",
        "there is no", "no such", "never", "none of",
    )),
    ("forall", (
        "对所有", "对任意", "对每个", "每个", "任意", "全部", "凡是", "一律", "恒成立",
        "∀", "for all", "for every", "every", "always", "all ",
    )),
    ("exists", (
        "存在", "某个", "至少一个", "至少存在", "找得到", "能找到一个",
        "∃", "there exists", "exists", "some ",
    )),
)

#: ★ 疑问句必须**先于** none 判定。踩过的坑: "有没有 n 使 P" 的后两个字就是
#: "没有" —— 一旦 "没有" 进了 none 的标记表, 疑问句(exists)会被判成 none,
#: 结论正好反掉。宁可单独处理, 也不靠标记表排序解决。
_QUESTION_MARKERS: tuple[str, ...] = ("有没有", "是否有", "有没有可能", "存在吗")

#: 否定陈述: "没有 n 使它成立" 这类。限定"没有"后面紧跟使/满足/成立/存在,
#: 才算 none —— 光一个"没有"太宽(见上面 _QUESTION_MARKERS 的坑)。
_NEGATIVE_RE = re.compile(r"没有[^。;；?？!！]{0,14}?(使|满足|成立|存在|符合)")


def detect_quantifier(text: str) -> str | None:
    """
    从自然语言里抽量词。抽不到就返回 None(**不猜**)。

    为什么要留 None: 命题文本里可能根本没有量词("sin^2+cos^2 = 1" 对所有实数成立,
    写成公式时量词是隐含的)。猜一个出来会制造假冲突。
    """
    low = (text or "").lower()
    # 疑问句优先: "有没有 n 使 P" 是问存在性, 不是否定陈述
    for m in _QUESTION_MARKERS:
        if m in low:
            return "exists"
    if _NEGATIVE_RE.search(low):
        return "none"
    for name, markers in _QUANT_PATTERNS:
        for m in markers:
            if m in low:
                return name
    return None


def quantifier_conflict(question: str, claim_text: str, explicit: str = "") -> tuple[str, str, str] | None:
    """
    检查拆出来的命题有没有把问题的量词改掉。改了就是误读, 返回 (问题量词, 命题量词, 说明)。

    ★ 这是 P4 实跑逼出来的一条。orig 失败现场:
        question = "对所有正整数 n, n^2 小于 2n 吗"
        claim    = "存在正整数 n 使得 n^2 小于 2n"     ← 量词被换掉了
        结果     = proved(对着存在版验的), 同时还有 refuted —— 全错

    取 explicit(LLM 自己填的 quantifier 字段)优先; 没填才从文本抽。
    两边都抽不到就返回 None —— 宁可漏检也不能瞎报错, 误报会让人把这个告警当噪声忽略。
    """
    q = detect_quantifier(question)
    c = (explicit or "").strip().lower() or detect_quantifier(claim_text)
    if not q or c not in QUANTIFIERS or q == c:
        return None
    return (q, c, f"量词冲突: 原问题是 `{q}`, 拆出的命题是 `{c}` —— 结论会整个反过来")


# ============================================================
# 1b-2. ~~范围 / 阈值检查~~ —— 已于 P4.5 第三轮删除
# ============================================================
# 这里原来有两道防线, 现在**故意留空**:
#   scope_conflict()    从文本正则解析范围上下界, 比对有没有缩到问题之外
#   threshold_conflict() 从文本正则解析"至少 N 组"这类阈值, 比对有没有被加码
#
# 删掉的理由不是它们没抓到问题(它们确实抓到了 trap_04 和 goldbach), 而是:
#   1. 正则不懂语义。"存在至少 1000 组素数对"里哪个数字是"原问题没要求的限定",
#      靠模式只能猜 —— 换个说法("不少于一千组")就漏。
#   2. 它们在错误的层。这些检查都在问"这条命题长得对不对",
#      而真正该问的是"**这条命题该不该被验证**", 那是流程问题。
#      用模式匹配去解流程问题, 天生就是打补丁。
#   3. 阈值黑名单只会越长, 加到没人敢动为止。
#
# 现在由 _precheck.run_precheck() 在建模后、验证前做**整道闸门**:
# 让 LLM 逐条做语义对照("这和原问题是同一件事吗" + "它本身可判定吗"),
# 不该验的根本不进验证循环。那是"该验才验", 而不是"出了事再拦"。
#
# 判据: **"结论对不对"可以问 LLM, "结论有没有证据"必须问机器。**
# 所以量词作为结构化字段、证据强制、0 LLM 审计这三条留在代码层不动 ——


# ~~question_is_undecidable()~~ 已于 P4.5 第三轮删除。
# 它靠 UNDECIDABLE_MARKERS 关键词黑名单判断问题是否不可判定, 两个致命问题:
#   1. 黑名单只会越长 —— "优美/有趣/最优算法" 之后必然要加"最小的""不存在的""超算力",
#      加到某个长度就没人敢动它。
#   2. **漏得比不拦还危险**。真实翻车里最刁钻的几道一个标记词都不含:
#      「最小的不存在的素数」「存在 n>100000 使 n!+1 是素数」(超算力)、
#      「哥德巴赫猜想成立吗」(开放问题) —— 全部放行。
# 现在这些交给 _precheck 闸门: 问 LLM"它本身可判定吗/是不是同一件事", 是语义判断。



# ============================================================
# 1. 命题规范化
# ============================================================
def normalize_claim(
    claim_text: str,
    scope: str = "",
    how: str = "",
    *,
    quantifier: str = "",
    question: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """
    把一条(可能很粗糙的)命题整理成标准 spec。

    判定不可判定的两条依据:
      a. 显式标记 —— 建模阶段自己填 how='undecidable'。
       这是模型的结构化输出, 不是我们对文本的猜测, 可以信。
       (原来还有一条"文本命中关键词黑名单"的启发式, 已删 —— 见 1b-2)
      b. 缺 scope —— 完全没有变量域的命题基本没法写成断言, 但**不直接判死**,
         只标 warning 让调用方决定(用户可能就是要验一个不带范围的简单命题)

    Args:
        quantifier: LLM 自己填的量词字段(forall/exists/none)。留空就从文本抽。
        question: **原始研究问题**。给了才能查出量词被偷换(见 quantifier_conflict)。

    Returns:
        {claim_text, scope, how, quantifier, state, decidable, warn, claim_id, ...}
    """
    from ._provenance import claim_id_for

    text = (claim_text or "").strip()
    scope_s = (scope or "").strip()
    how_s = (how or "").strip().lower()

    warn: list[str] = []
    if not text:
        warn.append("命题为空")
    if not scope_s:
        warn.append(
            "未给变量域/范围 —— 验证代码只能自己拍范围, 容易验的不是你想验的那个。"
            "建议补 scope(如 'n=1..500' / 'n<=8 的所有树')"
        )

    # ---- 量词(P4 加) ----
    q_explicit = (quantifier or "").strip().lower()
    if q_explicit and q_explicit not in QUANTIFIERS:
        # LLM 编了个新值 —— 别报错, 当成没填, 走文本抽取
        warn.append(f"quantifier={q_explicit!r} 不是合法值(应为 {'/'.join(QUANTIFIERS)}), 改从文本抽取")
        q_explicit = ""
    q_final = q_explicit or detect_quantifier(text) or ""
    # ★ 这里只剩量词一道。范围和阈值曾经也在这拦, 但那两道是拿正则匹配自然语言,
    #   换个说法就漏, 且属于"出了事再拦"的思路 —— 已删, 改由 _precheck 闸门在
    #   建模后统一判断(见上方 1b-2 的说明段)。
    conflict = quantifier_conflict(question, text, q_explicit) if question else None
    if conflict:
        q_from, q_to, msg = conflict
        # 量词被偷换是硬伤: 命题验得再对, 验的也不是用户问的那个。
        # 判 undecidable —— 让它进"需人判断"区, 而不是白跑一轮验证。
        warn.append(msg)
        return {
            "claim_text": text,
            "scope": scope_s,
            "how": how_s or "symbolic",
            "quantifier": q_to,
            "quantifier_from_question": q_from,
            "decidable": False,
            "state": "undecidable",
            "warn": warn,
            "claim_id": extra.pop("claim_id", None) or claim_id_for(text, scope_s),
            **extra,
        }

    # 不可判定的判定**不在这里做**。
    # 原来是"没有 scope + 文本命中 UNDECIDABLE_MARKERS 关键词" —— 已删, 因为:
    #   1. 关键词黑名单只会越长;
    #   2. 而"这条到底该不该验"是语义问题, 由 _precheck 闸门问 LLM 更准。
    # 这里只认一个**显式**信号: 建模阶段自己说 how='undecidable'。
    # 显式信号是模型的结构化输出, 不是我们对文本的猜测 —— 可以信。
    undecidable = how_s == "undecidable"

    # claim_id 允许调用方显式传入(从 _specs.jsonl 重建时要沿用原 id, 不能重算,
    # 否则审计时新旧 id 对不上, 证据会被误判成孤儿)。
    # ⚠️ 必须在这里 pop 掉: 否则 **extra 会在返回 dict 的最后覆盖掉刚算好的 claim_id。
    explicit_id = extra.pop("claim_id", None)

    return {
        "claim_text": text,
        "scope": scope_s,
        "how": how_s or "symbolic",
        "quantifier": q_final,
        "decidable": not undecidable,
        "state": "undecidable" if undecidable else "pending",
        "warn": warn,
        "claim_id": explicit_id or claim_id_for(text, scope_s),
        **extra,
    }


def dedupe_claims(specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """
    去重(按 claim_id, 内容哈希 = claim_text + scope)。

    ⚠️ **只去掉完全相同的重复**(同文本同范围)。不同范围**不合并**:
    n<=10 无反例 和 n<=20 查出反例 是两条不同的科研结论,
    合成一条就等于抹掉了"反例出现在 11~20 之间"这个最有价值的信息。
    claim_id 含 scope 正是为此(见 _provenance.claim_id_for)。

    Returns:
        (去重后的 specs, 去掉的重数)
    """
    seen: dict[str, dict[str, Any]] = {}
    dropped = 0
    for i, s in enumerate(specs):
        cid = s.get("claim_id") or f"__anon_{i}"
        if cid in seen:
            dropped += 1
            continue
        seen[cid] = s
    return list(seen.values()), dropped


# ============================================================
# 2. 报告渲染 —— 证据绑定规则在这里编码
# ============================================================
def render_report(
    specs: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    question: str = "",
    env: dict[str, str] | None = None,
) -> str:
    """
    渲染最终报告。

    ★ **核心纪律**: 结论区里每条判定都带 `ev_XXXX` 编号。
      拿不到证据的命题**不会**出现在结论区, 而是被归到"未验证区"。
      这条规则是代码强制的, 不是 prompt 里的一句叮嘱。

    Args:
        specs: 规范化后的命题列表
        evidence: 该任务全部证据(_evidence.jsonl)
        question: 原始问题
        env: 环境指纹
    """
    by_claim: dict[str, list[dict[str, Any]]] = {}
    for ev in evidence:
        by_claim.setdefault(ev.get("claim_id") or "", []).append(ev)

    # 结论区 / 未验证区
    concluded: list[tuple[dict, str, list[dict]]] = []
    unverified: list[tuple[dict, str]] = []

    for spec in specs:
        cid = spec.get("claim_id") or ""
        evs = by_claim.get(cid, [])
        # 取该命题**最新**的判定为准, 但证据全带着(能看出反复验证的历史)
        best = _latest_verdict(evs)
        if best and best[0] in _REPORTABLE:
            concluded.append((spec, best[0], evs))
        elif spec.get("decidable", True):
            # ⚠️ 不可判定的**不进未验证区** —— 它有独立的"需人判断"分区,
            #    两边都列会让读者以为是两条不同的东西(实测报告里重复出现)。
            #    未验证区只装"本来可验但这次没验出结果"的命题。
            unverified.append((spec, best[0] if best else spec.get("state", "pending")))

    out: list[str] = []
    out.append("# 科研验证报告\n")
    if question:
        out.append(f"**问题**: {question}\n")

    # ---- 环境指纹: 没它证据不可复现 ----
    if env:
        out.append("**环境**: " + " | ".join(f"{k}={v}" for k, v in env.items()) + "\n")

    # ---- 统计 ----
    counts: dict[str, int] = {}
    for ev in evidence:
        v = ev.get("verdict", "?")
        counts[v] = counts.get(v, 0) + 1
    if counts:
        out.append("**判定统计**: " + " | ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "\n")

    # ================= 结论区 =================
    out.append("\n## 结论\n")
    if not concluded:
        out.append("(无。任何命题都没达到可写进结论区的判定门槛。)\n")
    for spec, verdict, evs in concluded:
        refs = ", ".join(e.get("id", "?") for e in evs)
        out.append(f"### {verdict.upper()} — {spec['claim_text']}\n")
        if verdict in CONCLUSIVE:
            out.append(f"- 证据: `{refs}`\n")
        else:
            # partial / inconclusive 必须带范围限定, 这是 §8 条款 5
            scope = spec.get("scope") or "(范围未声明)"
            out.append(
                f"- 证据: `{refs}` — **这不是完整结论**: "
                f"仅在 {scope} 范围内检查通过, 不能外推。\n"
            )
        # 反复验证过就把历史摊开(结论翻车要看得见)
        if len(evs) > 1:
            hist = " → ".join(e.get("verdict", "?") for e in evs)
            out.append(f"- ⚠ 该命题被验证了 {len(evs)} 次: {hist}\n")
        if spec.get("scope"):
            out.append(f"- 范围: {spec['scope']}\n")
        # ★ 量词回显(P4 加): "对所有" 和 "存在" 的结论正好相反, 报告里必须看得见
        if spec.get("quantifier"):
            out.append(f"- 量词: {_QUANT_LABEL.get(spec['quantifier'], spec['quantifier'])}\n")
        for w in spec.get("warn") or []:
            out.append(f"- ⚠️ {w}\n")
        out.append("")

    # ================= 未验证区 =================
    out.append("\n## 未验证区\n")
    out.append("> 这些命题**没有**达到结论门槛。不要把它们当结论引用。\n")
    if not unverified:
        out.append("(空)\n")
    for spec, verdict in unverified:
        out.append(f"- [{verdict}] {spec['claim_text']}")
        if spec.get("scope"):
            out.append(f"  — 范围: {spec['scope']}")
        out.append("\n")

    # ================= 人工判断区 =================
    undec = [s for s in specs if not s.get("decidable", True)]
    if undec:
        out.append("\n## 需人判断\n")
        out.append("> 建模阶段判定为不可自动验证, 需要数学家判断:\n")
        for s in undec:
            out.append(f"- {s['claim_text']}\n")
            if s.get("scope"):
                out.append(f"  — 范围: {s['scope']}\n")
            if s.get("quantifier"):
                out.append(f"  — 量词: {_QUANT_LABEL.get(s['quantifier'], s['quantifier'])}\n")
            # ★ 必须把 warn 打出来(P4 加): 用户只看到"需人判断"是不知道原因的。
            #   典型的就是量词冲突 —— "你问的是『对所有』, 我拆出来的是『存在』",
            #   这句话不落到报告里, 误读就永远看不见。
            for w in s.get("warn") or []:
                out.append(f"  — ⚠️ {w}\n")
            out.append("\n")

    return "".join(out)


def _latest_verdict(evs: list[dict[str, Any]]) -> tuple[str, dict] | None:
    """取最新一条证据的判定; 没有有效判定返回 None"""
    for ev in reversed(evs):
        v = ev.get("verdict")
        if v and v != "untested":
            return v, ev
    return None


# ============================================================
# 3. 审计(0 LLM) —— 硬约束, 不看 LLM 说了什么
# ============================================================
def audit_claims(
    specs: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    report_md: str = "",
) -> dict[str, Any]:
    """
    独立复核: 只读证据文件, 抓翻车。**不调 LLM**。

    抓六类问题:
      1. 宣称 proved 但证据链里没有 proved 类证据 (LLM 吹了)
      2. 结论前后不一致 (同一 claim_id 多次验证, verdict 互斥)
      3. 裸断言 (报告结论区里有句子没挂 ev_ 编号)
      4. 证据孤儿 (evidence.claim_id 指向不存在的 spec)
      5. proved 判定来源可疑 (标 proved 但证据 kind 是 citation/数值抽查)
      6. refuted 但拿不出反例 (P4 加, 见下)

    Returns:
        {ok, issues: [{level, code, claim_id, detail}], n_claims, n_evidence}
    """
    issues: list[dict[str, Any]] = []

    def add(level: str, code: str, detail: str, claim_id: str = "") -> None:
        issues.append({"level": level, "code": code, "detail": detail, "claim_id": claim_id})

    spec_ids = {s.get("claim_id") for s in specs if s.get("claim_id")}

    # --- 2. 同一命题判定互斥 ---
    by_claim: dict[str, list[dict]] = {}
    for ev in evidence:
        cid = ev.get("claim_id") or ""
        if cid:
            by_claim.setdefault(cid, []).append(ev)

    for cid, evs in by_claim.items():
        verdicts = {e.get("verdict") for e in evs}
        # proved 与 refuted 同时出现 = 结论打架, 必须人工看
        if "proved" in verdicts and "refuted" in verdicts:
            add("error", "verdict_conflict",
                f"同一命题既被证成又被反驳: proved/refuted 同时出现"
                f"({len(evs)} 条证据) —— 环境变了还是命题写错了?", cid)
        elif "proved" in verdicts and "partial" in verdicts:
            add("info", "proved_with_partial",
                "有 proved 也有 partial 证据 —— 若 partial 的范围更小属正常, 否则可疑", cid)

    # --- 1 & 5. 结论必须有对应证据, 且 proved 的来源要可信 ---
    for spec in specs:
        cid = spec.get("claim_id") or ""
        if not spec.get("decidable", True):
            continue
        evs = by_claim.get(cid, [])
        verdicts = {e.get("verdict") for e in evs}

        if "proved" in verdicts:
            proved_evs = [e for e in evs if e.get("verdict") == "proved"]
            # 5. proved 必须来自计算, 不能来自纯引用
            weak = [e for e in proved_evs if e.get("kind") == "citation"]
            if weak and len(weak) == len(proved_evs):
                add("error", "proved_from_citation",
                    "proved 判定只有 citation 类证据 —— 文献引用不是证明", cid)
            # 1. 报告里写了 PROVED 但磁盘没有 proved 证据
            if report_md:
                claim_line = _find_claim_line(report_md, spec.get("claim_text", ""))
                if claim_line and "PROVED" in claim_line and not proved_evs:
                    add("error", "proved_without_evidence",
                        "报告宣称 PROVED, 但证据文件里没有对应证据", cid)

        # 6. ★ refuted 必须拿得出反例(P4 加)
        #    动机: P4 实跑里"哥德巴赫猜想成立吗"被判 refuted, 而它拆出来的
        #    弱哥德巴赫(三素数之和)**本来就是被证明了的定理** —— 也就是说那个
        #    refuted 根本就是错的。错在哪? 拿不出反例, 只是代码顺手打了个
        #    VERDICT: REFUTED。一个"证明某猜想为假"的结论, 如果给不出具体的
        #    见证值, 它和"证明某猜想为真"一样不可信。
        refuted_evs = [e for e in evs if e.get("verdict") == "refuted"]
        for ev in refuted_evs:
            out = (ev.get("stdout") or "").strip()
            if not out:
                add("error", "refuted_without_counterexample",
                    "判了 refuted 却没有任何反例输出 —— 见证值都没有, "
                    "凭什么说它假?", cid)
            elif len(out) < 5:
                add("warn", "thin_counterexample",
                    f"反例输出过短({out!r}), 建议给出可复现的具体见证值", cid)

        # 7. ★ proved 也必须有实质支撑 —— 反过来一样不可信(P4.5 第二轮加)
        #    动机: P4.5 实跑里「最小的不存在的素数是多少」被判 proved, 理由是
        #    "不存在 m >= 1 使得 n = 2^m - 1 是素数且 n < 2" ——
        #    这是一条**空洞真命题**: 它为真只因为约束弱到几乎没有反例空间,
        #    而且偷换成了 2^m-1。stdout 空 = 只证明了一个空集。
        for ev in [e for e in evs if e.get("verdict") == "proved"]:
            if not (ev.get("stdout") or "").strip():
                add("error", "proved_without_support",
                    "判了 proved 却没有任何输出 —— 空集为真不等于命题被证明", cid)

        # 3. 裸断言: 结论区出现 PROVED/REFUTED 但**整个结论块**都没有 ev_ 编号。
        #    ⚠️ 必须按"块"判定而不是按行 —— 本模块 render_report 产出的正常格式是
        #       `### PROVED — 命题` 换行后才是 `- 证据: ev_0001`。
        #       只看 ### 单行会把**所有正常结论**全判成裸断言(2026-10-08 实测踩到)。
        if report_md:
            for blk in _conclusion_blocks(report_md):
                if ("PROVED" in blk["header"] or "REFUTED" in blk["header"]) \
                        and not _EVIDENCE_REF.search(blk["body"]):
                    add("error", "bare_assertion",
                        f"结论块没有挂 ev_ 编号: {blk['header'].strip()[:70]}", cid)

    # --- 4. 孤儿证据 ---
    for ev in evidence:
        cid = ev.get("claim_id") or ""
        if cid and cid not in spec_ids:
            add("warn", "orphan_evidence",
                f"证据 {ev.get('id')} 的 claim_id={cid} 在命题列表里找不到", cid)

    # 汇总
    errors = [i for i in issues if i["level"] == "error"]
    warns = [i for i in issues if i["level"] == "warn"]
    return {
        "ok": not errors,
        "issues": issues,
        "n_errors": len(errors),
        "n_warnings": len(warns),
        "n_claims": len(specs),
        "n_evidence": len(evidence),
        "summary": (
            f"审计通过: {len(specs)} 条命题 / {len(evidence)} 条证据, 无硬伤"
            if not errors else
            f"审计发现 {len(errors)} 处硬伤 / {len(warns)} 处警告 —— 见 issues"
        ),
    }


def _find_claim_line(report_md: str, claim_text: str) -> str:
    """在报告里找到某命题对应的结论行"""
    if not claim_text:
        return ""
    head = claim_text[:24]
    for line in report_md.splitlines():
        if line.startswith("### ") and head and head in line:
            return line
    return ""


def _conclusion_blocks(report_md: str) -> list[dict[str, str]]:
    """
    把报告切成结论块: 一个 `### 标题` + 它后面到下一个 `###`/`##` 之间的正文。

    为什么要按块而不是按行: 证据编号写在标题**下面**的列表项里
    (`- 证据: ev_0001`), 只看标题行会把正常结论全判成裸断言。
    """
    blocks: list[dict[str, str]] = []
    header = ""
    body: list[str] = []

    def flush() -> None:
        if header:
            blocks.append({"header": header, "body": "\n".join(body)})

    for line in report_md.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            flush()
            header = line
            body = []
        elif header:
            body.append(line)
    flush()
    return blocks
