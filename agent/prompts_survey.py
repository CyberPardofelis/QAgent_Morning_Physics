"""
agent.prompts_survey — 综述生成(P6)专用 LLM prompt 模板

设计: 跟 ROADMAP_NEXT §7 / SURVEY_GENERATION_PLAN.md §3.1 对齐。
每个 prompt 是模块级常量 + build_xxx_prompt(...) 工厂函数(动态注入候选池/章节)。

约束总览(每个 prompt 末尾都重申):
  1. 事实性陈述句必须带 [paper_id]
  2. 过渡/总结/铺垫句不带 [paper_id]
  3. 不得引用候选池外的 paper_id
  4. 不得编造方法/数字/实验结果
  5. 行内 markdown 链接格式: [作者 年份, arXiv:xxxx.xxxxx](papers/xxxx.xxxxx.md)

9-27 重要修正 — 写作阶段必须拿到**真实素材**:
    之前 build_section_prompt 只接收 papers, format_pool 把每篇压成
    "title[:100] | method[:80] | contrib[:80]" 约 260 字符, **abstract 被整个丢掉**。
    而 extract_facts 对每篇论文花了 1 次 LLM 抽出的原子 fact
    (含 support_quote 原文摘录) 根本没进写作阶段 —— 全仓只有审计报告读它。
    等于"花 50+ 次 LLM 精读, 然后只把书目卡片表喂给写作模型",
    而 prompt 却要求它写出 5-8 个领域判断 + 每点 2-4 个引用 + 1500-2500 字。
    输入信息量和输出要求差两个数量级, 模型只能编或写空话 —— 这就是"显得傻"的根因。

    现在: 章节写作优先使用 state["facts"], 并按 sub_point 做相关度选取,
    只把最相关的论文连同其 fact 送进 prompt（信息量大幅提升, token 反而可控）。

不依赖外部 LLM 客户端,本模块只负责"prompt 文本构造",不调 API。
调用方(P6.2 survey_generate 工具)负责:
  1. 调 _call_llm(prompt)
  2. parse 输出
  3. 走 post-parse 校验(候选池外引用 → ❌)

历史:
  - 2026-09-05: 3 个 prompt 模板首版(P6.0 阶段)
"""
import re
from typing import Any

# ============================================================
# 共享 prompt 片段
# ============================================================
# 9-27: 这段"pool 是采样,不是领域全集"(RAG 局部最优防御)之前在
# SURVEY_OUTLINE_PROMPT 和 SURVEY_SECTION_PROMPT 里**逐字重复了 710 字符**。
# 重复的代价: 改一处忘另一处 → 两边规则悄悄漂移; 且每次请求多烧一遍 token。
# 现在抽成常量, 由两个 build_*_prompt 注入 —— 单一真相来源。

POOL_SAMPLING_RULE = """【候选池是采样, 不是领域全集】(RAG 局部最优防御)
- 候选池是从外部检索系统按相关性排序取的 {pool_size} 篇, **是检索采样, 不是领域全部论文**
- 你写任何"领域判断"前, 先问自己: 这个判断是基于 **pool 内统计**, 还是基于 **领域共识**?
  - 基于 pool 内统计 → 写"在本综述检索范围内, X 算法占主导" (前置 pool 数量 + 引用佐证)
  - 基于领域共识 → **必须**有 ≥ K (K=5) 篇 pool 内 paper + 多源(arxiv+APS+IOP)+ 时间窗口覆盖; 否则改写为 pool 内统计表述
- ❌ 反例: pool 含 30 篇 Gilbert + 2 篇 Hamilton → "Gilbert 已成为本领域主流算法" 错;
- ✅ 改写: "在本综述检索的 {pool_size} 篇 paper 中, Gilbert 路径占 30 篇 (94%), Hamilton/变分/Path Integral 等方向各有 1-2 篇相关工作, 本综述受检索覆盖限制未充分展开 — 主流判断应保留给领域共识层 (教科书/综述)"
- claim 范围限定: 凡是"主流/共识/领域普遍/已成定式"的措辞, 必须有"基于 {pool_size} 篇 paper 的检索"或"在本综述范围内"前置; 直接说"X 是主流"会被审计标 ⚠️ RAG 局部最优"""


def _get_min_pool() -> int:
    """
    候选池最小阈值 (9-08 抽 env 后用).
    函数内 lazy import agent.tools.config: 避开 prompts_survey → surveys → prompts_survey 循环依赖。
    (surveys.py 顶层 import prompts_survey, 所以 prompts_survey 不能顶层 import agent.tools.*)
    """
    from agent.tools.config import get_min_pool_papers
    return get_min_pool_papers()


# ============================================================
# 候选池注入格式(三个 prompt 共享)
# ============================================================
def format_pool(papers: list[dict[str, Any]]) -> str:
    """
    把候选池格式化成 LLM 友好的多行文本。

    每行: - {paper_id} | {title} | {core_method} | {core_contrib}
    paper_id 是 LLM 唯一可引用的标识;title 用作链接文本;core_method/contrib 提供上下文。

    Args:
        papers: 候选池 paper 列表,每条至少有 {paper_id, title, core_method, core_contrib}
                可选 {year, authors, journal}

    Returns:
        多行字符串,每行一篇 paper。空候选池返回 "(空候选池)"
    """
    if not papers:
        return "(空候选池)"
    lines = []
    for p in papers:
        pid = p.get("paper_id", "?")
        title = (p.get("title") or "").strip().replace("|", "/")
        method = (p.get("core_method") or "").strip().replace("|", "/")
        contrib = (p.get("core_contrib") or "").strip().replace("|", "/")
        year = p.get("year") or "?"
        journal = p.get("journal") or ""
        prefix = f"- {pid} ({year}"
        if journal:
            prefix += f", {journal}"
        prefix += ")"
        # 单行不超过 ~200 chars,过长截断
        def trunc(s: str, n: int) -> str:
            return s if len(s) <= n else s[: n - 1] + "…"
        lines.append(
            f"{prefix}: {trunc(title, 100)} | method: {trunc(method, 80)} | "
            f"contrib: {trunc(contrib, 80)}"
        )
    return "\n".join(lines)


# ============================================================
# 相关度选取 + 带 fact 的渲染 (9-27)
# ============================================================
# 纯 L0 实现（不调 LLM），用词面重叠把最相关的论文挑出来。
# 目的是: 章节 prompt 不再把整个 50 篇候选池无差别倒给模型,
# 而是给"这一章真正相关的 N 篇 + 它们被精读出的 fact"。

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _tokenize(text: str) -> set[str]:
    """
    中英混合分词: 英文/数字按词, 中文按字。

    中文没有空格, 按词切需要词典; 这里退化成"单字 + 相邻字对",
    对"表面码/阈值"这类术语的重合判定已经足够用。
    """
    if not text:
        return set()
    low = text.lower()
    tokens = set(_TOKEN_RE.findall(low))
    cjk = _CJK_RE.findall(low)
    tokens.update(cjk)
    tokens.update(a + b for a, b in zip(cjk, cjk[1:], strict=False))
    return tokens


def _paper_text(p: dict[str, Any], facts: list[dict[str, Any]] | None) -> str:
    """一篇论文用于相关度比对的全部文本。"""
    parts = [
        p.get("title", "") or "",
        p.get("core_method", "") or "",
        p.get("core_contrib", "") or "",
        p.get("keywords", "") or "",
    ]
    for f in facts or []:
        parts.append(f.get("claim", "") or "")
        parts.append(f.get("support_quote", "") or "")
    return " ".join(parts)


# ============================================================
# fact 规范化 (9-29)
# ============================================================
# 注意: 这里**不做**"claim 术语是否出现在 support_quote"的确定性检查。
# 实测 181 条真实 fact 的 claim 是中文、原文是英文, token 覆盖率中位数 0.021,
# 这种检查跨语言根本不成立(改查英文术语则被残留标注污染, 且中文编造细节看不见)。
# 语义层面的"有没有超出原文"由 _fact_audit.audit_fact_grounding 负责(LLM, 分批)。
# 本文件只做**确定无争议**的清理: 去掉模型自写的 paper_id 标注。


def strip_paper_id_from_claim(claim: str) -> str:
    """
    去掉 claim 里模型自写的 [paper_id] / [paper_id: xxx] / [2209.02093] 之类标注。

    9-29: paper_id 现在由 fact_id 机械映射, claim 里再带一份既是冗余,
    也是污染源 —— 实测 91% 的 claim 内嵌了这种标注, 且格式五花八门
    ([2209.02093] / [paper_id: 10.1103/xxx]), 抽取器根本解析不出来。
    """
    if not claim:
        return ""
    s = re.sub(r"\[\s*paper_id\s*[:：]\s*[^\]]*\]", "", claim, flags=re.I)
    s = re.sub(r"\[(?:\d{4}\.\d{4,5}(?:v\d+)?|\d{7})\]", "", s)
    s = re.sub(r"\[10\.\d{4,9}/[^\]]*\]", "", s)
    return re.sub(r"\s{2,}", " ", s).strip(" ，,;；")


# ============================================================
# subject 校验 (9-29)
# ============================================================
# subject 是 Phase B 聚类的依据, 一旦被自指词/超长句/空值污染,
# "跨论文对齐"就失效, 聚类会碎成一堆没意义的簇。
# 这里做**确定性**检查(不调 LLM): 打标记, 不自动改写 ——
# 改写 subject 等于替模型做判断, 出错很难回溯。

_SUBJECT_GENERIC = (
    "本文", "该文", "该方法", "该工作", "该研究", "该论文", "本工作", "方法一", "方法二",
    "我们", "作者", "本文方法", "此方法", "上述方法", "提出的方法", "新方法",
    "the method", "this method", "our method", "the approach", "this work",
)
_SUBJECT_TOO_SHORT = 3
_SUBJECT_TOO_LONG = 40


def _check_subject(subject: str) -> dict[str, Any]:
    """
    检查 subject 是否可用于跨论文对齐。

    返回 {"_subject_ok": bool, "_subject_issues": [str, ...]}
    """
    issues: list[str] = []
    s = (subject or "").strip()
    if not s:
        issues.append("subject 为空 —— 聚类时会落进'其他', 失去区分度")
    else:
        low = s.lower()
        for g in _SUBJECT_GENERIC:
            if g in low:
                issues.append(f"含自指/泛指词 {g!r} —— 跨论文无法对齐")
                break
        # 长度按"字符"算: 英文 3 词大概 10+ 字符, 中文 2-6 词约 6-20 字符
        if len(s) < _SUBJECT_TOO_SHORT:
            issues.append(f"过短({len(s)} 字符) —— 大概率不是完整对象名")
        if len(s) > _SUBJECT_TOO_LONG:
            issues.append(f"过长({len(s)} 字符) —— 应该是句子而不是对象名")
        if s.endswith(("。", ".", "，", ",", "；", ";")):
            issues.append("以标点结尾 —— 不是干净的���象名")
    return {"_subject_ok": not issues, "_subject_issues": issues}


def clean_fact(f: dict[str, Any], paper_id: str, idx: int) -> dict[str, Any]:
    """
    规范化一条抽取结果: 去 paper_id 标注 + 加质检标记 + 校验 subject。

    标记:
        _grounded            落地审查是否通过(默认 True, 由 _fact_audit 覆盖)
        _unsupported_terms   超出原文的术语(写作时应回避)
        _self_referential    claim 带"本文/该工作/作者"等自指(去语境化失败)
        _subject_ok/_subject_issues  subject 是否可用于跨论文对齐
    """
    f = dict(f)
    # 9-30 修: 原来用 setdefault —— 只在 key **不存在**时才写入。而模型基本总会
    # 带上 paper_id 字段(多篇合抽时甚至可能填成批内序号 "4"), 于是 setdefault 什么
    # 都不做, 调用方传进来的**权威值被无视**, fact 就挂到了错的论文上。
    # 实测: 51 篇合抽, 落盘一堆 paper_id="4" 的 fact。
    # 调用方(抽取循环)知道真实 paper_id, 这里必须**强制覆盖**。
    f["paper_id"] = paper_id
    f["fact_id"] = f"{paper_id}::{idx}"
    # subject: 这条 fact 讨论的核心对象(跨论文对齐用, Phase B 聚类的依据)
    subj = (f.get("subject") or "").strip()
    f["subject"] = subj[:60]
    f.update(_check_subject(f["subject"]))
    f["local_fact_id"] = f.pop("fact_id_local", None) or f.get("local_fact_id")
    claim = strip_paper_id_from_claim(f.get("claim", "") or "")
    f["claim"] = claim
    # _grounded 默认 True; 由 _fact_audit.audit_fact_grounding 事后覆盖。
    # (确定性词表检查在跨语言下不成立, 已移除 — 见上方注释)
    f.setdefault("_unsupported_terms", [])
    f.setdefault("_grounded", True)
    f["_self_referential"] = any(
        s in claim for s in ("本文", "该工作", "该文", "该研究", "该论文", "本工作", "作者", "我们")
    )
    return f


def select_relevant_papers(
    papers: list[dict[str, Any]],
    query_text: str,
    facts_by_paper: dict[str, list[dict[str, Any]]] | None = None,
    limit: int = 24,
) -> list[dict[str, Any]]:
    """
    按 query_text(章节标题 + sub_points) 对论文做词面相关度排序, 取前 limit 篇。

    打分 = 交集大小 / sqrt(len(论文 token 数)) —— 轻度惩罚长文,
    避免一篇摘要特别长的 paper 光靠词多就挤掉更相关的。

    没有 facts_by_paper 时也能用(退化成只按 title/method/contrib 匹配)。

    返回 [(paper, score), ...] 按分数降序, 分数相同时保持原候选池顺序
    (候选池本身已按 _relevance_score 排过, 稳定排序能保住那个优先级)。
    """
    if not papers:
        return []
    if len(papers) <= limit:
        return list(papers)

    facts_by_paper = facts_by_paper or {}
    q_tokens = _tokenize(query_text)
    if not q_tokens:
        return list(papers[:limit])

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, p in enumerate(papers):
        p_tokens = _tokenize(_paper_text(p, facts_by_paper.get(p.get("paper_id"))))
        if not p_tokens:
            score = 0.0
        else:
            overlap = len(q_tokens & p_tokens)
            score = overlap / (len(p_tokens) ** 0.5) if overlap else 0.0
        scored.append((score, idx, p))

    # 分数降序; 同分按原顺序(稳定)
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [p for _s, _i, p in scored[:limit]]


def _is_usable_fact(f: dict[str, Any]) -> bool:
    """
    fact 能不能进写作素材。

    9-27: action_extract_facts 的 fact 主题对齐审查 (_review_facts_topic_alignment)
    会把 `_is_topic_aligned=False` 打在邻近子领域的 fact 上。之前这个标记只进
    审计报告, 写作阶段照单全收 —— 等于那 20+ 次 LLM 调用白花了。

    审查关闭时 (get_fact_review_threshold()==0) 不写这个字段, 视为可用,
    行为与之前一致。
    9-29 再加一层: `_grounded=False` —— 确定性质检(fact_grounding_problems)
    判定 claim 里的术语没出现在 support_quote 中, 即该 claim 超出原文。
    实测"离子阱/离子附加能级/避免辅助离子"就属此类 —— 论文摘要只提了
    ion trap 和 nonorthogonal, 抽取器自行补全成完整机制。这类 fact 是
    **编造的**, 绝不能进写作素材。
    """
    return not (
        f.get("_is_topic_aligned", True) is False
        or f.get("_grounded", True) is False
    )


def format_papers_with_facts(
    papers: list[dict[str, Any]],
    facts_by_paper: dict[str, list[dict[str, Any]]] | None = None,
    max_facts_per_paper: int = 4,
) -> tuple[str, int]:
    """
    渲染"论文 + 它被精读出的 fact"。

    这是写作阶段真正该看到的东西: 每条 fact 带 claim 和 support_quote 原文摘录,
    比之前 260 字符的书目卡片表信息量高一个数量级, 且能直接支撑"带 [paper_id]
    的事实性陈述"这个硬约束。

    没有可用 fact 的论文退化为带 abstract 截断的普通条目(比之前多给摘要);
    如果它被 fact 审查判为邻近子领域, 会额外标注提醒谨慎引用。

    返回 (渲染文本, 被丢弃的 misaligned fact 条数)。
    """
    if not papers:
        return "(空候选池)", 0
    facts_by_paper = facts_by_paper or {}
    dropped = 0

    def trunc(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[: n - 1] + "…"

    lines: list[str] = []
    for p in papers:
        pid = p.get("paper_id", "?")
        title = trunc((p.get("title") or "").strip().replace("|", "/"), 120)
        year = p.get("year") or "?"
        journal = p.get("journal") or ""
        head = f"- {pid} ({year}" + (f", {journal}" if journal else "") + f"): {title}"

        all_facts = facts_by_paper.get(pid) or []
        facts = [f for f in all_facts if _is_usable_fact(f)]
        dropped += len(all_facts) - len(facts)

        if facts:
            lines.append(head)
            for f in facts[:max_facts_per_paper]:
                ftype = f.get("fact_type", "fact")
                claim = trunc((f.get("claim") or "").strip().replace("\n", " "), 220)
                quote = trunc((f.get("support_quote") or "").strip().replace("\n", " "), 160)
                if not claim:
                    continue
                line = f"    · [{ftype}] {claim}"
                if quote:
                    line += f"  (原文: {quote})"
                lines.append(line)
        else:
            method = trunc((p.get("core_method") or "").strip().replace("|", "/"), 100)
            contrib = trunc((p.get("core_contrib") or "").strip().replace("|", "/"), 100)
            abstract = trunc((p.get("abstract") or "").strip().replace("\n", " "), 300)
            extra = f" | method: {method}" if method else ""
            extra += f" | contrib: {contrib}" if contrib else ""
            if all_facts and not facts:
                # 全部 fact 都被判为邻近子领域
                extra += "  ⚠ 邻近子领域, 谨慎引用"
            lines.append(head + extra)
            if abstract:
                lines.append(f"    · 摘要: {abstract}")
    return "\n".join(lines), dropped


# ============================================================
# 1. SURVEY_FACT_PROMPT — 从单篇 paper 提 3-5 个 fact(Phase C)
# ============================================================
SURVEY_FACT_PROMPT = """你是文献综述的事实抽取器, 任务:对**单篇论文**做原子事实抽取。

【输入论文】
- paper_id: {paper_id}
- title: {title}
- authors: {authors}
- year: {year}
- journal: {journal}
- abstract: {abstract}
- core_method: {core_method}
- core_contrib: {core_contrib}
- outlook: {outlook}

【你的产出会被放进综述正文, 在那里"本文/该工作/作者"会失去指代对象】
所以每条事实必须是**自足**的、去语境化的陈述。

【铁律 — 违反即为错误】
1. **只允许复述【论文原文】里出现过的内容**。
   禁止"补充"你觉得合理但原文没写的细节 —— 尤其是:
   - 实验平台/装置(离子阱、超导、NMR、光晶格、中性原子…)。原文没写平台就不要写。
   - 具体机制("利用 X 的附加能级实现 Y"、"通过 Z 消除 W")。原文没写机制就不要写。
   - 数字、年份、人名。原文没有的不要出现。
   看到摘要里出现两个相关词就自行补全成一条因果链, 是最严重的错误。
2. **claim 必须用中文写**(专有名词可保留英文缩写, 如 SIC-POVM、classical shadow),
   而且主语必须是方法/概念/现象本身, 不是"本文/该工作/作者/我们"。
   ❌ "本文提出稳定化解析延拓框架"        (读者不知道谁提出的)
   ✅ "稳定化解析延拓(SAC)框架从多项式估计中重构密度矩阵的非多项式解析函数"
   ⚠️ 不要因为"只许复述原文"就把 claim 整句写成英文 —— 那是逐句复制 support_quote,
   不是抽取。support_quote 才是"原文照抄"的那一份, claim 是中文转述。
3. **claim 里不要写 [paper_id]、不要写编号**。出处由系统按 fact_id 机械关联, 不需要你写。
   写错反而会污染引用。
4. **support_quote 必须是原文的完整相关句**(可以多句), 且必须**包含 claim 的关键术语**。
   support_quote 用来核对 claim 有没有超出原文 —— 如果它撑不起 claim, 说明 claim 编了。
5. **数量: 4-12 条,由这篇论文的信息量自己决定**。
   这���综述篇幅的直接来源 —— 下游写作阶段只能看到 fact,
   fact 少 → 综述短。实测 10 篇论文每篇只抽 5 条时, 全文只有 9000 字符,
   远不够一篇综述; 每篇抽 10 条时篇幅大致翻倍。
   ⚠️ 但**宁可少提也不要编**: 抽不够就少抽, 后续有落地审查会兜住编造,
   但审查只能剔除、不能补写 —— 被剔除的部分就是白丢的篇幅。
   所以理想是"每条都站得住", 而不是"凑够 N 条"。
   ⚠️ 信息量小的短文/综述性短文抽 4 条就够, **不要为凑数把一句话拆成三条**。

【fact 类型】(每条必须标一个)
  - method:      用了什么方法/算法/框架
  - result:      实验/理论结果(数字、性能、复杂度)
  - conclusion:  核心结论
  - limitation:  局限/不足/未解决问题
  - hypothesis:  假设/猜想

【subject 字段 — 很重要】
subject = 这条事实讨论的**核心对象**(方法名/概念/现象), 用**可跨论文对齐的短形式**,
2-6 个词, 尽量用领域通用叫法而不是本文的自造术语。
  ✅ "SIC-POVM 态层析" / "随机测量快照" / "经典阴影" / "张量网络态层析"
  ❌ "本文方法一" / "该框架" / "我们提出的技术"
理由: 系统会按 subject 把来自不同论文的 fact 聚成簇, 再据此归纳综述的主题结构。
subject 越对齐, 跨论文的聚类越准 —— 这直接决定综述的章节划分质量。
如果两条 fact 讲的是同一个对象, subject 必须写成**同一个字符串**。

【输出格式】严格 JSON 数组, 不要 markdown 代码块包裹, 不要解释:
[
  {{"subject": "<核心对象, 2-6 词, 可跨论文对齐>",
   "fact_type": "method",
   "claim": "<去语境化、自足、只在原文出现过的陈述>",
   "support_quote": "<原文完整相关句>"}}
]

只输出 JSON。"""


def build_fact_prompt(
    paper_id: str,
    title: str = "",
    authors: str = "",
    year: str = "",
    journal: str = "",
    abstract: str = "",
    core_method: str = "",
    core_contrib: str = "",
    outlook: str = "",
) -> str:
    """
    构造单篇 paper 的 fact 提取 prompt。

    缺字段传 "" 即可(模板里有占位符),LLM 对空字段容忍。
    """
    return SURVEY_FACT_PROMPT.format(
        paper_id=paper_id,
        title=title or "(无标题)",
        authors=authors or "(未知作者)",
        year=year or "?",
        journal=journal or "(未知期刊)",
        abstract=abstract or "(无摘要)",
        core_method=core_method or "(未提取)",
        core_contrib=core_contrib or "(未提取)",
        outlook=outlook or "(未提取)",
    )


# ============================================================
# 2. SURVEY_OUTLINE_PROMPT — 从候选池生成章节大纲(Phase D)
# ============================================================
SURVEY_OUTLINE_PROMPT = """你是综述章节规划助手,任务:基于候选池,生成**claim-centric** 的结构化大纲。

【综述主题】: {topic}

【候选池】(共 {pool_size} 篇,paper_id 唯一):
{pool_text}

【写作视角 — claim-centric】
**综述的核心是"领域判断/分类/趋势",不是"罗列 paper 摘要"。**
- 每个 sub_point = 1 个**领域宏观判断/分类/趋势/分类法**, 用完整陈述句表达, 不是"问题清单"
- 后续章节写作时, paper 是"佐证", 是"判断的证据", 不是"段落的主角"
- 大纲的丰富度直接决定章节能不能展开 — sub_point 越具体, 章节越有支撑, 引用越能挂上

【章节类型与顺序】(固定 5 段,从上到下,不要漏)
  1. intro: 背景 / 研究问题 / 本文综述范围(0-2 引用)
  2. progress: 关键进展时间线 / 里程碑 + 演化驱动力 (8-15 引用, 核心章节)
  3. method_compare: 方法分类法 + trade-off 对比 + 适用场景(6-12 引用,核心章节)
  4. gap: 当前研究空白 / 不足 / 反例(3-5 引用)
  5. outlook: 未来方向 / 开放问题 / 跨领域迁移(0-2 引用,基于 gap 推断)

【每章应包含的"宏观内容类型"】(5 段各有侧重)
  - intro:    研究背景 / 已有综述覆盖范围 / 本综述的切入角度 / 全文组织结构
  - progress: 历史脉络(早期→中期→近期) / 转折点(为什么 X 替代了 Y) / 当前共识 / 演化的内在驱动力
  - method_compare: 分类法(按 X 维度怎么分) / 各类的 trade-off 对比 / 适用场景 / 性能/复杂度对比
  - gap:     公开问题清单(技术 / 实验 / 理论) / 各问题的解决难度 / 现有反例 / 为什么没解决
  - outlook: 短期可做(1-2 年) / 长期方向(5 年+) / 跨领域迁移机会 / 标准化展望

【每章输出 schema】
- title: 章节标题(中文,15 字以内)
- section_type: 5 选 1(intro / progress / method_compare / gap / outlook)
- sub_points: **5-8 个宏观要点**,每个要点是 1 句完整的"领域判断/分类/趋势"陈述句
  ⚠️ 字段名严格用 sub_points(复数),任何 chapter 都不能写成 sub_point(单数)
  ⚠️ sub_point 数量必须 5-8 个, 这是章节展开度的硬下限 (防模型偷懒)
- is_transition: yes / no
    - yes = 该章主要是铺垫/过渡,不强求引用(给 intro 的部分子节)
    - no = 该章必须有事实引用(给 progress / method_compare / gap / outlook)

【sub_point 措辞示例】(错误写法会让模型敷衍)
  ❌ 错误 (问题式, 让 LLM 想敷衍):
     "X 方法的工作原理"
     "Y 方法有什么优点"
     "该领域未来如何发展"
  ✅ 正确 (判断式, 逼 LLM 写出来):
     "X 方法通过 Y 机制将错误率压到 <1%, 已成为容错量子计算主流路径"
     "Y 在大规模系统中遭遇性能瓶颈, 主要因为 Z 开销随节点数线性增长"
     "跨领域迁移方面, A 领域的 W 技术有可能解决 B 领域长期未解的 V 问题"

{pool_sampling_rule}
- sub_point 措辞同理: 避免"X 是领域普遍现象", 写"在本综述检索范围内, X 表现出..."

【强制约束】
1. 必须**严格按上面 5 段顺序**输出,不能合并章节,不能改顺序
2. paper_id 只能从候选池里选,不能编造
3. 引用配额按"章节类型"软约束:超过上下限会被审计标 ⚠️
4. method_compare 的 sub_points 应该按"机制/性能/场景"维度组织,不要按时间组织(时间组织是 progress 的事)
5. **sub_points 数量 5-8 个**: 大纲生成后会被审计, < 5 个 sub_point 的章节会在 action_finish 被拒
6. 大纲生成后, **不允许**直接给"X 算法是本领域主流"这种无前置的判断; 必须按 §pool-采样 改写为有前置限定的 claim

【输出格式】(严格 JSON 对象,不要 markdown 代码块包裹,不要解释)
{{
  "sections": [
    {{"title": "...", "section_type": "intro", "sub_points": ["...", "...", "...", "...", "..."], "is_transition": "yes"}},
    {{"title": "...", "section_type": "progress", "sub_points": ["...", "...", "...", "...", "...", "..."], "is_transition": "no"}}
  ]
}}

只输出 JSON,不要前缀"以下是...",不要后缀"希望对你有帮助"。"""


def build_outline_prompt(topic: str, papers: list[dict[str, Any]]) -> str:
    """
    构造综述大纲生成 prompt。

    papers: 候选池 paper 列表(同 format_pool 的输入)
    """
    return SURVEY_OUTLINE_PROMPT.format(
        topic=topic or "(未指定主题)",
        pool_size=len(papers),
        pool_text=format_pool(papers),
        pool_sampling_rule=POOL_SAMPLING_RULE.format(pool_size=len(papers)),
    )


# ============================================================
# 3. SURVEY_SECTION_PROMPT — 单章生成 markdown 正文(Phase E)
# ============================================================
SURVEY_SECTION_PROMPT = """你是综述章节写作助手,任务:基于候选池 + 章节大纲,生成 1 个章节的 markdown 正文。

【综述主题】: {topic}
【本章节类型】: {section_type}
【本章节标题】: {section_title}
【本章节大纲】(含 5-8 个 sub_point 宏观判断, 必须逐条展开):
{outline_section}

【候选池】({pool_note}, paper_id 唯一):
{pool_text}

【怎么用上面的素材】(重要)
- 每条 `·` 开头的是从该论文里**预抽出来的原子 fact**,带 `claim`(结论) 和 `原文:`(support_quote 出处)。
- 写事实性陈述时, **优先直接使用这些 fact 的 claim**, 并在句末挂上该行的 paper_id。
  这样既满足"事实性陈述必须带 [paper_id]"的硬约束, 又不会凭空编造数字/结果。
- `原文:` 括号里是论文原句, 可据此判断该 fact 的可信度。
- 论文行末的 `摘要:` 只在**没有**预抽 fact 时出现, 仅作背景参考, 不要从中编造数字。
- **没在上面出现的论文, 不要引用它的 paper_id** —— 你没看到它的内容, 引用即幻觉。

【写作风格 — claim-centric (最关键)】
1. **章节是讲"领域判断/分类/趋势",不是讲"哪些 paper 各自做了啥"**
   - ❌ 错误 (paper-centric 罗列): 每段以 paper 为单位, "...A 提出了 X [paper1]...B 提出了 Y [paper2]...C 改进了 Z [paper3]" — 这是罗列 paper 摘要
   - ✅ 正确 (claim-centric 论述): 每段**先抛出该段的领域判断/分类/趋势 (claim)**, 再用 2-4 个 paper_id 佐证
     - claim 写法: 完整陈述句, 不是问题 (例: "X 方法已成为 Y 场景的主流路径, 关键改进点是 Z")
     - 佐证写法: claim 后挂 [paper_id], 注明"哪几篇贡献了哪个具体点/数字/结果"
2. **典型结构** (claim-first 视角):
   - 开篇 1-2 段: 本章核心判断的预告 (1 段) + 本章将向读者展示哪些宏观结论 (1 段)
   - 主体 N 段 (按 sub_point 展开): 每段 = 1 个 sub_point 的展开
     - 段首: 抛出该 sub_point 的领域判断/分类/趋势 (claim, 1-2 句)
     - 段中: 解释该判断的依据/机制/数据
     - 段末: 挂 2-4 个 paper_id, 注明哪几篇贡献了哪个具体点/数字/结果
   - 收尾 1 段: 本章宏观判断总结 + 跟下一章的衔接
3. **长度要求**:
   - 单章 markdown **1500-2500 字** (sub_points × 3-4 段/子点, 每段平均 200-300 字)
   - 5-8 个 sub_point 决定章节下界 1500 字; 写得再短会被审计标 ⚠️
4. **每个 sub_point 至少配 2-4 个 paper_id 引用** (关键事实/数字/结果 处必带), 引用的 paper 是"佐证", 不是"中心"
5. **不要复述 paper 摘要**: 不要写 "A 提出了 X, 它的核心方法是 Y", 写 "X 已成为主流路径 (A [p1], B [p2], C [p3] 都在此方向), 关键改进点在于 ..."

{pool_sampling_rule}
- 章节段落写法同理: 避免"X 是领域普遍现象", 写"在本综述检索范围内, X 表现出..."

【强制约束】(违反会被自动审计标 ❌,请严格遵守)
1. 事实性陈述句末尾**必须**带 [paper_id] 简洁形式, 格式: "...发现 X 提升了 Y% [paper_id]"
   ⚠️ **关键**: 简洁 [paper_id] 形式必带, 完整 markdown 链接 `[作者 年份, arXiv:xxx](papers/xxx.md)` 是**补充**, 不能替代
2. 过渡/总结/铺垫/动机类句子**不要**带 [paper_id](这是关键,过渡句污染是常见错误)
3. 不得引用候选池外的 paper_id(候选池外引用 = 幻觉,自动审计会**自动重写该章节**,浪费一次 LLM 调用)
4. 不得编造方法/数字/实验结果;不知道就跳过,不要凑
5. 行内 markdown 链接格式: [作者 年份, arXiv:xxxx.xxxxx](papers/xxxx.xxxxx.md)
   例: "拓扑码在 [Bombin 2006, arXiv:0607734](papers/0607734.md) 中被提出"
6. 简洁 [paper_id] + 完整链接**同时写**最理想, 例: "Bombin 2006 提出拓扑码 [0607734](papers/0607734.md) [Bombin 2006, arXiv:0607734](papers/0607734.md) 中被提出"
7. 不带引用 [paper_id] 的行内链接不要写(纯链接没意义)
8. markdown 格式: 标题用 ##, 列表用 -, 不嵌套列表超过 2 层
9. **禁止**直接给"X 算法是本领域主流 / Y 是领域共识"这种无前置的判断; 必须按 §pool-采样 改写为有前置限定的 claim, 否则审计标 ⚠️ RAG 局部最优

【P6.3 自动重写机制】
- 每个章节生成后, 自动审计会检测候选池外引用 (out_of_pool)
- 如果 out_of_pool > 0, 系统会自动用"只允许候选池内 paper_id"作为额外约束重写该章节 (最多 2 次)
- 第 2 次重写后仍 out_of_pool > 0, 接受并落库 (覆盖度审计报告里标 ⚠️)

【"claim-centric 视角 vs paper 罗列" 对比示例】

  ❌ paper 罗列 (错误, paper 是主角):
    "A 提出了表面码 [p1]。B 改进了表面码 [p2]。C 提出了 XZZX 码 [p3]。D 比较了多种码 [p4]。"
    (把多篇 paper 各开一段, 章节读起来像 "list of paper summaries")

  ✅ claim-centric (正确, 判断是主角, paper 是佐证):
    "表面码自 Fowler 2012 展示低于 1% 逻辑错误率以来, 已成为容错量子计算的主流路径 [p1]。其核心思路是用二维网格上的物理比特编码单个逻辑比特, 通过周期测量稳定子算子检测错误 [p1][p2]。在偏置噪声场景下, Bravyi 等指出传统表面码阈值显著退化, 由此提出 XZZX 码并展示了 ~10x 的阈值改进 [p3]。近期工作进一步把表面码扩展到子系统码, 在硬件开销和纠错能力间取得新平衡 [p4]。"
    (以"机制演化"为线索, 多篇 paper 串成一个连贯论述, 关键事实点 cite)

【本章节引用配额】(软约束, 超过会被审计标 ⚠️)
  {ref_quota}

【缺漏反馈机制】(关键)
- 写完本章节后, **主动扫一遍 sub_points, 看哪个 sub_point 的引用支撑偏少 (< 2 个 paper_id)**:
  - 如果发现某些 sub_point 引用不足或缺方向性文献, 必须在 markdown 末尾加 HTML 注释标注, **系统会自动捕获并回到 search_more 循环补文章**:
    `<!-- QAgent: missing_queries=["BB84 原始协议", "MDI-QKD 实验", "诱骗态方法"] -->`
  - **主动声明缺失是更专业的综述态度**, 写这个标注**不会扣分**, 反而让综述更扎实
  - **不写 ≠ 章节完整**: 即使你觉得章节"看起来完成", 也要自检每个 sub_point 至少有 2 个 paper_id 引用, 否则应当标注
- **query 应该是搜索关键词, 不是 paper_id** (系统会用 query 调 arxiv 扩候选池)
- **query 应该精准**: 1-5 个, 每个 query 是一类文献的关键词 (例: "BB84 protocol original", "MDI-QKD experiment", "decoy-state method")
- **注意**: 标注必须是合法 HTML 注释格式, 且在 markdown 末尾 (其他内容前/中会被审计标 ❌)

【输出】(直接 markdown 章节正文, 不要前缀"以下是...",不要后缀总结,不要"```markdown" 包裹)
"""


# 章节类型 → 引用配额(覆盖度审计的软约束参考)
# 9-08 23:50 user 拍板: claim-centric 视角下, sub_points 涨到 5-8 个,
# 章节长度 1500-2500 字, 引用配额同步上调 (progress/method_compare 核心章节翻倍)
SECTION_REF_QUOTAS = {
    "intro":          "0-2 个引用,主要讲背景和综述范围",
    "progress":       "8-15 个引用,核心进展章节 (5-8 sub_point × 2-3 引用/sub_point)",
    "method_compare": "6-12 个引用,按机制/性能/场景组织 (5-8 sub_point × 2-3 引用/sub_point)",
    "gap":            "3-5 个引用,以反例为主, 支撑「研究空白」判断",
    "outlook":        "0-2 个引用,基于 gap 推断",
}


def build_section_prompt(
    topic: str,
    section_type: str,
    section_title: str,
    outline_section: str,
    papers: list[dict[str, Any]],
    section_type_safe: str | None = None,
    facts: list[dict[str, Any]] | None = None,
    papers_limit: int = 24,
) -> str:
    """
    构造单章生成 prompt。

    section_type_safe: 强校验后的小写 section_type;若不传则用 section_type.lower()。

    facts (9-27): extract_facts 抽出的原子 fact。这是写作阶段真正该用的素材 ——
        之前它根本没进来, 写作模型只看到 260 字符/篇的书目卡片表, 却要写
        5-8 个领域判断 + 每点 2-4 个引用。传了就走"带 fact 的渲染 + 相关度选取"。

    papers_limit: 送给模型的论文条数上限(按 sub_point 相关度选取)。
        不传 facts 时也生效, 避免把 50 篇无差别倒进 prompt。
    """
    safe_type = (section_type_safe or section_type or "").strip().lower()
    if safe_type not in SECTION_REF_QUOTAS:
        safe_type = "intro"  # 兜底,不让 LLM 看到陌生章节类型

    # 有 fact → 新路径: 按本章标题 + sub_points 选相关论文, 连同 fact 渲染
    if facts:
        facts_by_paper: dict[str, list[dict[str, Any]]] = {}
        for f in facts:
            pid = f.get("paper_id")
            if pid:
                facts_by_paper.setdefault(pid, []).append(f)
        query = f"{section_title or ''}\n{outline_section or ''}"
        selected = select_relevant_papers(
            papers, query, facts_by_paper, limit=papers_limit,
        )
        pool_text, dropped = format_papers_with_facts(selected, facts_by_paper)
        drop_note = f",另有 {dropped} 条邻近子领域 fact 已剔除" if dropped else ""
        pool_note = (
            f"共 {len(papers)} 篇,下面是按本章节相关度选出的 {len(selected)} 篇"
            f"(含每篇预抽的原子 fact{drop_note})"
        )
    else:
        selected = select_relevant_papers(
            papers, f"{section_title or ''}\n{outline_section or ''}",
            limit=papers_limit,
        )
        pool_text = format_pool(selected)
        pool_note = f"共 {len(papers)} 篇,下面是按本章节相关度选出的 {len(selected)} 篇"

    return SURVEY_SECTION_PROMPT.format(
        topic=topic or "(未指定主题)",
        section_type=safe_type,
        section_title=section_title or "(无标题)",
        outline_section=outline_section or "(无大纲)",
        pool_size=len(papers),
        pool_text=pool_text,
        pool_note=pool_note,
        pool_sampling_rule=POOL_SAMPLING_RULE.format(pool_size=len(papers)),
        ref_quota=SECTION_REF_QUOTAS[safe_type],
    )


# ============================================================
# P6.5 search_more 用的单篇 summarize prompt
# (复用 analyze_paper 风格, 但简化到 4 字段, 适配 summary_items.db)
# ============================================================
def build_summarize_prompt(
    paper_id: str,
    title: str,
    authors: str = "",
    abstract: str = "",
) -> str:
    """
    单篇 paper summarize prompt — 写进 summary_items.db (source='survey_search')

    输出 4 字段 JSON:
      - core_method: 技术方法 + 创新点, 50-200 词
      - core_contrib: 主要贡献 + 结果, 50-150 词
      - keywords: 3-8 关键词 (逗号分隔字符串)
      - outlook: 意义与展望, 30-100 词
    """
    return f"""Analyze the following paper and return a **valid JSON object** with these fields:

- core_method: Technical methods and innovative points, about 50-200 words (中文或英文都可, 但要完整)
- core_contrib: Main contribution and result, about 50-150 words
- keywords: list of 3-8 keywords (output as comma-separated string, e.g. "quantum entanglement, entropy, fidelity")
- outlook: significance and outlook, about 30-100 words

Paper ID: {paper_id}
Title: {title}
Authors: {authors}
Abstract: {abstract[:3000]}

Output ONLY the JSON object, no other text, no markdown fence."""


# ============================================================
# P6.5 ReAct 决策 prompt (step 6, LLM 真选 action)
# ============================================================
def build_react_prompt(state: dict[str, Any]) -> str:
    """
    ReAct 决策 prompt — 把当前 state + 候选池 paper_ids 序列化给 LLM, 让它自己选下一步 action.

    LLM 必须返回严格 JSON: {thought, action, args}
    """
    pool_size = len(state["pool"])
    fact_count = len(state["facts"])
    section_count = len(state["sections"])
    error_count = len(state["errors"])
    step = state["step"]
    max_steps = state["max_steps"]
    topic = state["topic"]
    section_titles = [s["title"] for s in state["sections"]]

    # ============ 9-08 抽 env: 固定候选池规则 ============
    # 硬规则(写进 prompt 给 LLM 看, fallback 也走这套):
    #   - 候选池 < 最小阈值: 优先 search_more (env QAGENT_SURVEY_MIN_POOL, 默认 30)
    #   - 候选池 ≥ 最小阈值+10 且 fact 不足: extract_facts
    #   - 最小阈值 ≤ pool < 最小阈值+10: LLM 自由判断(用户没指定)
    # 加 outline: 9-08 user 拍板, 把"写章节"拆成 write_outline + write_section,
    # 顺序: 先 write_outline, 再按 outline 章节写
    POOL_MIN = _get_min_pool()             # 9-08 抽 env, 默认 30 (之前 user 拍板硬编码 30)
    POOL_ENOUGH = POOL_MIN + 10            # 充裕门槛 = 最小 + 10, 保持原有 +10 关系

    # 关键: 把候选池 paper_ids 列给 LLM, 避免编 paper_id (P6.5 step 6 fix)
    # 显示窗口联动 POOL_MIN (9-08 抽 env 后调整): LLM 至少能看到跟最小阈值等量的 paper_id,
    # 避免候选池达标 (>= 最小) 但 LLM 看不到全部 id, 出现"自创 id" 的边角问题。
    # 池子 < 最小阈值时 (search_more 阶段) 仍按池子实际大小显示。
    pool_pids = [p.get("paper_id", "?") for p in state["pool"][:min(POOL_MIN, pool_size)]]
    pool_ids_block = "\n".join(f"  - {pid}" for pid in pool_pids)
    if len(state["pool"]) > min(POOL_MIN, pool_size):
        pool_ids_block += f"\n  ... (共 {pool_size} 篇, 上面是前 {min(POOL_MIN, pool_size)} 个 paper_id)"
    has_outline = bool(state.get("outline_sections"))

    # ============ 9-08 user 拍板: 缺漏反馈 section_gaps ============
    # LLM 写章节时如果觉得还缺文章, 会在 markdown 末尾加
    # `<!-- QAgent: missing_queries=[...] -->` 标注, system 解析后塞进 state["section_gaps"]
    # finish 看到 section_gaps 不空 → 拒绝结束, 让 LLM 回到 search_more 循环补
    section_gaps = state.get("section_gaps", [])
    if section_gaps:
        gaps_block = "\n".join(
            f"  - 章节 '{g.get('title', '?')}' 缺: {g.get('queries', [])}"
            for g in section_gaps
        )
    else:
        gaps_block = "  (无)"

    # ============ 9-08 17:00 user 拍板: 未引用 paper 反馈 ============
    # 审核发现的"候选池 0 引用"传给 LLM, 不能再重复犯错
    unused_papers = state.get("unused_pool_papers", []) or []
    if unused_papers:
        unused_preview = ", ".join(unused_papers[:15])
        more = f" 等 {len(unused_papers)} 篇" if len(unused_papers) > 15 else ""
        unused_block = f"  - {unused_preview}{more}"
    else:
        unused_block = "  (无, 所有候选池 paper 都已被引用)"

    # ============ 9-18 加: pool audit 报告 (audit retry loop 的结论) ============
    # 9-18 user 拍板: audit 结论要打通到 ReAct 主循环 LLM, 让它知道候选池质量.
    # - pool_meets_target=False 时, LLM 应该优先 search_more 调方向
    # - rounds 里能看到每轮审计通过/拒绝了多少, 知道是否还能再扩
    audit = state.get("pool_audit", {}) or {}
    audit_block = "  (未跑 audit)"
    if audit:
        meets = audit.get("pool_meets_target", False)
        final = audit.get("final_pool_size", 0)
        target = audit.get("target_min_pool", 0)
        rounds = audit.get("rounds", [])
        max_retries = audit.get("max_retries", 0)
        rounds_summary = " | ".join(
            f"round{r.get('round', '?')}: +{r.get('after', 0)}/{r.get('before', 0)} ok={r.get('ok', False)}"
            for r in rounds
        ) or "(无)"
        status = "✅ 满足" if meets else f"⚠️ 不足 (已重试 {max_retries} 次仍只 {final}/{target})"
        audit_block = (
            f"  - 状态: {status}\n"
            f"  - 最终池大小: {final}/{target}\n"
            f"  - audit 轮次: {rounds_summary}\n"
            f"  - 提示: 'meets_target=False' 时, 优先 search_more 调整方向 (用更精确的关键词, "
            f"或传 'sources': ['aps','iop','arxiv'] 跨源)"
        )

    return f"""你是综述生成 ReAct 决策者. 当前状态:

- 主题: {topic}
- 候选池: {pool_size} 篇 (硬规则阈值: 最小 {POOL_MIN}, 充裕 {POOL_ENOUGH})
- 提取 fact: {fact_count} 条
- 已写章节 ({section_count}): {section_titles if section_titles else "(无)"}
- 综述大纲: {"已有 (" + str(len(state.get("outline_sections", []))) + " 章)" if has_outline else "(无, 必须先 write_outline)"}
  {f"⚠️ 大纲只有 {len(state.get('outline_sections', []))} 段, 少于 5 段必被 action_finish 拒绝! 重新 write_outline 出 5 段" if has_outline and len(state.get("outline_sections", [])) < 5 else ""}
- 章节缺漏 (section_gaps, {len(section_gaps)} 个待补):
{gaps_block}
- ⚠️ 候选池 0 引用 paper ({len(unused_papers)} 篇, 后续章节请尽量引用):
{unused_block}
- 🔍 候选池 audit 报告 (9-18 加, _build_candidate_pool 的审核结论):
{audit_block}
- 错误数: {error_count}
- ReAct step: {step}/{max_steps}

候选池 paper_id 列表 (**严格使用这里的 id, 不要编造**):
{pool_ids_block}

可选 action (选一个, 严格 JSON 输出):
1. search_more {{"query": "...", "k": 5}} — 扩候选池 (默认走 arxiv, 9-18 加: 可传 "sources": ["arxiv","aps","iop"] 多源)
2. extract_facts {{"paper_ids": ["上面的真 id"]}} — 抽 fact
3. write_outline {{}} — 生成综述大纲(章节结构), 必须在 write_section 之前
4. write_section {{"section_idx": <int>}} — 按 outline 写第 section_idx 章(0-based); 0=intro, 1=progress, ...
5. finish {{}} — 收尾

search_more 多源 (9-18 user 拍板):
  - {{"query": "...", "k": 5, "sources": ["arxiv"]}} — 默认, 只走 arxiv
  - {{"query": "...", "k": 5, "sources": ["aps"]}} — 只走 APS (PRL/PRA/PRX Quantum 等, 走 OpenAlex)
  - {{"query": "...", "k": 5, "sources": ["arxiv","aps","iop"]}} — 跨多源并行 (通用搜索)
  - {{"query": "...", "k": 5, "source": "all"}} — 字符串简写, 等价 ["arxiv","aps","iop"]
  - 候选池 < {POOL_MIN} 且扩不出论文时, 优先传 "sources": ["aps","iop","arxiv"] 跨源

【固定候选池规则】(user 9-08 拍板, 硬规则)
- 候选池 < {POOL_MIN} → 优先 search_more(候选池太小, 综述地基不稳)
- 候选池 ≥ {POOL_ENOUGH} 且 fact < 5 → 优先 extract_facts(候选池够大, 缺的是精读)
- 候选池在 [{POOL_MIN}, {POOL_ENOUGH}) 区间 → LLM 自由判断(用户没指定, 走"启发式")

【综述生成顺序】(user 9-08 拍板, 拆分 write_section)
- 必须先 write_outline 生成大纲, 再 write_section 按章节写
- write_outline 只调一次, 调完就锁住 state["outline_sections"]
- write_section 用 section_idx 指定写第几章(0-based, 按 outline_sections 顺序)
- 写完所有 outline 章节后 → finish

启发式 (20-39 中间区或边界情况可参考):
- fact_count < 5 且 pool 充裕 → extract_facts
- 写完所有 outline 章节 → finish
- outline 章节用完不要继续 write_section, 直接 finish
- 9-08 缺漏处理: 如果 section_gaps 有待补 query, 优先用这些 query 调 search_more 扩候选池, 扩完后再视情况 extract_facts / 重新 write_section
- search_more 的 query 应该是 "BB84 协议" 这种关键词, 不要传 paper_id (system 会调 arxiv 拉新文章)

请返回严格 JSON (不要 markdown fence, 不要其它 text):
{{"thought": "为什么这样选 (一句话)", "action": "action_name", "args": {{...}}}}"""


# ============================================================
# 4. SURVEY_RELEVANCE_SCORE_PROMPT — per-paper 相关性评分 (2026-09-09 加, 9-09 严判版)
# ============================================================
# 9-09 user 实测 + 严判升级:
#   - 9-09 初版: 量子态验证综述召回了 30+ 篇层析/密钥分发论文, 粗筛阈值 7 不够严
#   - 9-09 严判版: 阈值 7→8, 新增 is_about_topic 字段
#     is_about_topic=True 才是综述主引, 邻近子领域 paper 即便 score=7 也踢掉
#
# 设计要点:
#   - 5 篇/批 (config 可调), 1 次 LLM = 1 个 batch 的 JSON 数组输出
#   - 显式强调"X 本身" vs "X 的子方向" vs "邻近子领域" 三档
#   - is_about_topic bool: 跟 score 冗余但 LLM 表达更清晰, 严判字段
#   - is_adjacent_field bool: 跟 mechanism B fact review 配套, audit 用
SURVEY_RELEVANCE_SCORE_PROMPT = """你是综述主题相关性严判助手。

【综述主题】: {topic}

【任务】: 对下面 {batch_size} 篇论文, **每篇独立打分** (0-10, 整数), 评估"该论文核心贡献是不是 X 本身 (而不是邻近子领域)"。

【打分尺度 — 严判】
  10 = 完美命中, 论文核心就是 X
  9  = 核心, 论文专门讲 X
  8  = 强相关, X 是论文主方向之一 (但论文还有其他方向)
  7  = 边缘相关, X 是论文的某子方向/应用/扩展
  5-6 = 邻近子领域, 共享术语/方法但论文核心不是 X
  3-4 = 误召, 看起来像但完全是另一领域
  1-2 = 完全不相关
  0   = 无关 (无 abstract / 完全无关)

【特别注意 — 严判】
**"是 X 的子方向" 跟 "是 X 本身" 要严格区分**:
  - "X 的子方向" (score 7-8): 论文仍在 X 大伞下, 但有自己独立内容
  - "邻近子领域" (score 5-6): 论文不在 X 大伞下, 只是共享术语/方法
  - "误召" (score 3-4): 看起来像但完全是另一领域

经典反例 (严判):
  - "quantum state verification" (yes/no 验证) ≠ "quantum state tomography" (完整密度矩阵重建)
    → tomography paper 即便共享 "quantum state" 术语, score 应 <= 6, is_about_topic=False
  - "容错量子计算" ≠ "量子模拟"
    → 量子模拟 paper 共享"量子"但核心不同, score 应 <= 6
  - "transformer 架构" ≠ "BERT 预训练"
    → BERT paper 核心是预训练, 不是架构, score 应 <= 6
不要因为 title/abstract 含主题词就给高分 — 必须看 core_method + core_contrib 的实质内容。

【关键字段 - is_about_topic】(严判新增)
新增 is_about_topic 字段 (bool), 判定规则:
  - is_about_topic = true: score >= 8 (论文核心就是 X 本身, 综述主引)
  - is_about_topic = false: score <= 7 (论文只是 X 的子方向或邻近子领域, 不应作为综述主引)
  - threshold 8 严判: 只保留 is_about_topic = true 的 paper 进 pool
  - 误召 paper (score <= 6) 必须 is_about_topic = false

【论文列表】(共 {batch_size} 篇):
{papers_block}

【输出】(严格 JSON 数组, {batch_size} 个对象, 不要 markdown 包裹, 不要前缀解释)
[
  {{"paper_id": "<id>", "score": <int 0-10>, "is_about_topic": <bool>, "is_adjacent_field": <bool>, "reason": "1 句判断依据"}},
  ...
]
只输出 JSON 数组, 不要其他文字。"""


def build_relevance_score_prompt(topic: str, papers_batch: list) -> str:
    """
    构造 per-paper 相关性评分 prompt (5 篇/批, batch_size 由 config 决定).

    papers_batch: list of pool entry dict, 每条至少有 {paper_id, title, core_method, core_contrib}
                  可选 {abstract, year, journal} (缺字段容忍)
    """
    from agent.tools.config import get_relevance_batch_size
    batch_size = get_relevance_batch_size()

    lines = []
    for i, p in enumerate(papers_batch, 1):
        pid = p.get("paper_id", "?")
        title = (p.get("title") or "").strip()
        method = (p.get("core_method") or "").strip()
        contrib = (p.get("core_contrib") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        year = p.get("year", "") or ""
        journal = p.get("journal", "") or ""

        # 单论文 block: paper_id 必带, 其他缺则空字符串
        block_lines = [f"{i}. paper_id={pid}"]
        if title:
            block_lines.append(f"   title: {title[:200]}")
        if year or journal:
            block_lines.append(f"   venue: {year}{', ' + journal if journal else ''}")
        if method:
            block_lines.append(f"   core_method: {method[:200]}")
        if contrib:
            block_lines.append(f"   core_contrib: {contrib[:200]}")
        if abstract:
            block_lines.append(f"   abstract: {abstract[:400]}")
        lines.append("\n".join(block_lines))

    papers_block = "\n\n".join(lines)

    return SURVEY_RELEVANCE_SCORE_PROMPT.format(
        topic=topic or "(未指定主题)",
        batch_size=batch_size,
        papers_block=papers_block,
    )


# ============================================================
# 5. SURVEY_FACT_REVIEW_PROMPT — per-fact 主题对齐审查 (2026-09-09 加, mechanism B)
# ============================================================
# 9-09 user 拍板: pre-outline review 阶段, 对每条 fact 做主题对齐审查
# 跟 mechanism A (paper 严判) 配套 — paper 严判只过滤 pool, 但 pool 内的 fact
# 仍可能是从邻近子领域 paper 抽出来的"挂羊头卖狗肉" 内容
#
# 设计要点:
#   - 8 条/批 (fact 比 paper 短, 可更多), 1 次 LLM = 1 batch JSON 数组
#   - fact 来自哪篇 paper 跟评分无关 — 只看 fact 内容
#   - is_topic_aligned bool 跟 score 冗余但更清晰
#   - 不删除非对齐 fact, 只打标 (审计用, 落 _facts.jsonl + _audit.md)
SURVEY_FACT_REVIEW_PROMPT = """你是综述 fact 主题对齐审核助手。

【综述主题】: {topic}

【任务】: 对每条 fact, 判断"该 fact 是否直接服务于综述主题 X 本身 (而不是邻近子领域)"。

【打分尺度 — 严判】(跟 paper 严判对齐)
  10 = fact 直接讲 X 本身, 必引
  9  = fact 强相关 X 的关键子方向/方法
  7-8 = fact 是 X 的扩展/应用/对比 (但仍可作为综述引用)
  5-6 = fact 弱相关, 共享术语但讲邻近子领域
  3-4 = fact 邻近子领域, 不应作为 X 综述核心引用
  1-2 = fact 跟 X 完全无关

【特别注意】
- fact 来自哪篇 paper 跟评分无关 — 只看 fact 内容本身
- "挂羊头卖狗肉" 严判: fact 可能是从邻近子领域 paper 抽出来的, 看起来相关但其实在讲别的
- "X 的方法在 Y 领域的应用" 算 7-8 (应用/对比, 仍可引)
- "Y 领域自己的方法, 跟 X 仅共享术语" 算 3-4 (邻近, 不应引)

【关键字段 - is_topic_aligned】(新增)
新增 is_topic_aligned 字段 (bool), 判定规则:
  - is_topic_aligned = true: score >= 5 (fact 跟 X 直接相关, 综述可引)
  - is_topic_aligned = false: score <= 4 (fact 邻近子领域, audit 标 ❌)
  - threshold 5 默认 (env QAGENT_SURVEY_FACT_REVIEW_THRESHOLD 可调)

【fact 列表】(共 {batch_size} 条):
{facts_block}

【输出】(严格 JSON 数组, {batch_size} 个对象, 不要 markdown 包裹, 不要前缀解释)
[
  {{"fact_id": "<id>", "paper_id": "<id>", "score": <int 0-10>, "is_topic_aligned": <bool>, "reason": "1 句"}},
  ...
]
只输出 JSON 数组, 不要其他文字。"""


def build_fact_review_prompt(topic: str, facts_batch: list) -> str:
    """
    构造 per-fact 主题对齐审查 prompt (8 条/批, batch_size 由 config 决定).

    facts_batch: list of fact dict, 每条至少有 {fact_id, paper_id, claim, fact_type}
    """
    from agent.tools.config import get_fact_review_batch_size
    batch_size = get_fact_review_batch_size()

    lines = []
    for i, f in enumerate(facts_batch, 1):
        fid = f.get("fact_id", f"f{i}")
        pid = f.get("paper_id", "?")
        claim = (f.get("claim") or "").strip()
        ftype = f.get("fact_type", "?")
        support = (f.get("support_quote") or "").strip()

        # 单 fact block
        block_lines = [f"{i}. fact_id={fid} | paper_id={pid} | fact_type={ftype}"]
        if claim:
            block_lines.append(f"   claim: {claim[:250]}")
        if support:
            block_lines.append(f"   support_quote: {support[:150]}")
        lines.append("\n".join(block_lines))

    facts_block = "\n\n".join(lines)

    return SURVEY_FACT_REVIEW_PROMPT.format(
        topic=topic or "(未指定主题)",
        batch_size=batch_size,
        facts_block=facts_block,
    )


# ============================================================
# 模块自检(可独立 import 测试,不调 LLM)
# ============================================================
if __name__ == "__main__":
    # 简单 sanity: 能构造出来
    sample = [
        {
            "paper_id": "2401.12345",
            "title": "Surface code under biased noise",
            "year": 2024,
            "journal": "PRX",
            "core_method": "XZZX surface code with tailored bias",
            "core_contrib": "10x threshold improvement under bias-100 noise",
        },
        {
            "paper_id": "1809.10727",
            "title": "Subsystem codes",
            "year": 2018,
            "journal": "PRX",
            "core_method": "Pauli-based subsystem codes",
            "core_contrib": "New subsystem framework for biased noise",
        },
    ]
    print("=== build_outline_prompt ===")
    print(build_outline_prompt("量子纠错码近期进展", sample)[:400], "...")
    print()
    print("=== build_section_prompt (progress) ===")
    print(
        build_section_prompt(
            topic="量子纠错码近期进展",
            section_type="progress",
            section_title="关键进展时间线",
            outline_section="- 早期突破 (2010-2015)\n- 中期优化 (2015-2020)\n- 近期进展 (2020+)",
            papers=sample,
        )[:400],
        "...",
    )
    print()
    print("=== build_fact_prompt ===")
    print(build_fact_prompt(paper_id="2401.12345", title="Surface code", core_method="XZZX")[:300], "...")


# ============================================================
# 单段重写 prompt (9-27)
# ============================================================
SURVEY_PARAGRAPH_REWRITE_PROMPT = """你是综述章节的**局部重写**助手。

用户对已写好的某一章不满意, 但只要求重写其中 {n_target} 段, 其余段落保持原样。

【综述主题】: {topic}
【本章节类型】: {section_type}
【本章节标题】: {section_title}
【本章节大纲】(这一章要覆盖的 sub_point):
{outline_section}

【当前章节全文】(用 [第 N 段] 标注了每段, 标了"要重写"的是目标段)
{current_structure}

【本次只重写这些段】: 第 {target_list} 段

【可用素材】(从候选池预抽的原子 fact, 带 claim 和原文出处)
{pool_text}

【上下文】(目标段的前后段落, 保持不变 —— 新写的段落必须与它们衔接)
{context}

【任务】
1. 只输出**第 {target_list} 段**的替换正文, 不要输出其它段落, 不要输出解释。
2. 多个目标段时, 按段号顺序输出, 每个目标段独立成一块, 块之间空一行。
3. 新段落必须**无缝衔接**上下文: 承接上一段的论述, 并为下一段铺垫。
4. 事实性陈述必须挂 [paper_id], 优先使用上面素材里的 fact claim 与数字。
5. **不要输出 markdown 标题**(## ...), 也不要输出 [第 N 段] 这类标记。
6. 保持与上下文相同的语言(中文)和文体。

【输出】(直接给替换段落正文, 不要前缀"以下是...",不要代码块包裹)
"""


def build_paragraph_rewrite_prompt(
    topic: str,
    section_type: str,
    section_title: str,
    outline_section: str,
    current_structure: str,
    context: str,
    target_idxs: list[int],
    papers: list[dict[str, Any]],
    facts: list[dict[str, Any]] | None = None,
    user_instruction: str = "",
    papers_limit: int = 16,
) -> str:
    """构造单段重写 prompt。素材选取范围比整章小(默认 16 篇), 因为只写几段。"""
    safe_type = (section_type or "intro").strip().lower()
    if safe_type not in SECTION_REF_QUOTAS:
        safe_type = "intro"

    facts_by_paper: dict[str, list[dict[str, Any]]] = {}
    for f in facts or []:
        pid = f.get("paper_id")
        if pid:
            facts_by_paper.setdefault(pid, []).append(f)

    if facts:
        query = f"{section_title or ''}\n{outline_section or ''}"
        selected = select_relevant_papers(
            papers, query, facts_by_paper, limit=papers_limit,
        )
        pool_text, dropped = format_papers_with_facts(selected, facts_by_paper)
        drop_note = f"(另有 {dropped} 条邻近子领域 fact 已剔除)" if dropped else ""
        pool_text = f"(共 {len(papers)} 篇候选池, 下面是按本段相关度选出的 {len(selected)} 篇{drop_note})\n{pool_text}"
    else:
        selected = select_relevant_papers(
            papers, f"{section_title or ''}\n{outline_section or ''}", limit=papers_limit,
        )
        pool_text = f"(共 {len(papers)} 篇候选池, 下面 {len(selected)} 篇)\n{format_pool(selected)}"

    target_list = ", ".join(str(i) for i in target_idxs)
    instr = f"\n【用户额外要求】\n{user_instruction.strip()}\n" if user_instruction.strip() else ""

    return SURVEY_PARAGRAPH_REWRITE_PROMPT.format(
        topic=topic or "(未指定主题)",
        section_type=safe_type,
        section_title=section_title or "(无标题)",
        outline_section=outline_section or "(无大纲)",
        current_structure=current_structure,
        context=context or "(无上下文, 这是本章开头或结尾)",
        target_list=target_list,
        n_target=len(target_idxs),
        pool_text=pool_text,
    ) + instr

# ============================================================
# 命题合并 (2026-09-30 加)
# ------------------------------------------------------------
# 背景: 综述缺"综"这一环。40 篇实测: 同一篇论文在**一段话里被引 5 次**,
# 295 条 fact 逐条成句、逐条挂同一个引用 —— 看着满篇引用, 实际引文密度极稀疏。
# 这一层把"讲同一件事"的 fact 合并成命题, 一条命题可由多篇论文共同支持,
# 成文时写成一句话挂多个引用。
#
# 关键约束(来自 9-30 踩坑): 宁可**少合并**也不能错合并。
# 错合并 = 把两件不同的事说成一件, 综述就出现事实错误, 比漏合并严重得多。
CLAIM_CONDENSE_PROMPT = """你要把学术论文抽取出的原子事实(fact)合并成"命题"。

【合并的意义是什么】
一份综述的价值不在于把每篇论文的每个细节都罗列一遍, 而在于**用较少的语言描述一个问题的宏观情况**。
真正的"综"是: **多篇论文说的是同一件事**, 那就写成一句话, 一次挂上这几篇的引用。

【最重要的一条】
**合并的价值在于跨论文。** 每条 fact 后面标了它来自哪几篇论文(形如 p3 / p7,13):
  - 几条 fact 来自**不同**论文、但讲的是**同一个结论** → 这才是该合并的情形, 合并后一句挂多篇引用
  - 几条 fact **全都来自同一篇**论文 → 合并它们对引用密度**没有任何帮助**, 保持分开

举例:
  p3: "PEPS 纠缠熵在临界点呈对数标度"
  p7: "PEPS 纠缠熵的标度行为与 CFT 一致"
  p13:"数值确认 PEPS 在临界点具有对数纠缠标度"
  → 三条来自三篇不同的论文, 讲同一件事 → 合并成 1 条命题, 挂 3 篇引用 ✓

  p3: "该方法用梯度法优化"
  p3: "收缩误差用截断控制"
  p3: "计算复杂度随键维平方增长"
  → 都来自同一篇 → **不要合并**, 三条各自成命题, 各自挂 p3

【不要合并的情形】
  - 讲的是**不同的技术点/结论**, 哪怕同属一个大主题
      反例: "MPS 用于一维链" 与 "PEPS 用于二维格点"
      反例: "iTEBD 算一维演化" 与 "MPS 求基态能量"
  - 结论互相矛盾或互相否定
  - 只是共享术语、没有实质共识

判断不了就不合并 —— 合并错会产生事实错误, 比漏合并严重。

【fact 列表】(p 后面是来源论文编号)
{facts_block}

【输出格式】
严格 JSON, 不要 markdown 包裹, 不要额外解释:
{{"groups": [{{"fact_idx": [1, 3, 7], "statement": "一句话概括, 要能独立成立"}}, {{"fact_idx": [2], "statement": "..."}}]}}

要求:
- fact_idx 只能用上面列出的编号(1 到 {n})
- **每一条 fact 都要出现在某一组里**, 漏掉的会被系统单独成条
- statement 用中文, **不超过 40 字**, 一句话说清这件事是什么, 不要堆细节
- 每组一条 statement"""

# ============================================================
# 命题版主题成文 (2026-09-30 加)
# ------------------------------------------------------------
# 与 fact 版的根本差别: 写作单元是"命题"不是"fact"。
# 允许多个命题写进同一句 -> 一句话挂多个引用 -> 综述的"综"。
# 事实版实测: 同一篇论文在一段话里被引 5 次, 因为它那 5 条 fact 各自成句。
THEME_WRITE_FROM_CLAIMS_PROMPT = """你要围绕一个主题写一段**综述式**叙述。

【这一节的写作单元是"命题", 不是单条事实】
下面每条命题可能由**多篇论文共同支持**(标注里写了"（N 篇）")。
命题已经是把"讲同一件事"的多条事实合并后的结果。

【最重要的一条: 一句话可以挂多个引用】
如果两三件事本来就属于同一个更大的判断, 就把它们写进**同一句话**,
句末用逗号分隔标注, 例如:
    二维纠缠的刻画依赖面序参量, 面积定律的适用范围由此确定 [[c1, c3]], 该结论在多套张量表示下保持一致 [[c2]]。

**不要把每条命题拆成独立的一两句话。** 那正是"把摘要抄一遍"的写法。
综述的价值在于用**较少的语言**描述一个问题的**宏观情况**。

【本节目标篇幅】约 {target_chars} 字符, 覆盖 {n_claims} 条命题。

【覆盖要求 —— 这条最容易做错】
**{n_claims} 条命题必须全部写到, 一条都不能丢。**
篇幅要压, 靠的是**把几条命题写进同一句话**(句末逗号分隔多个标注),
**不是靠少写**。丢命题 = 丢引用 = 综述覆盖变窄。

实测踩过的坑: 给了 60 条命题, 模型只写出 14 句、只碰到 11 篇论文,
剩下 49 条命题被静默丢弃 —— 篇幅是短了, 但引用广度全没了。
写得短和覆盖得全不冲突: 一句话挂 3~5 个标注就同时做到了。

【写作要求】
- 中文
- 先给这个主题**是什么、为什么重要**, 再讲具体进展
- 讲清楚不同做法之间的**关系**: 谁补了谁的短板, 谁在什么条件下更优, 它们指向同一个什么结论
- 只写上面命题覆盖得到的内容, **不要引入命题里没有的经验内容**(数字/装置/方法名/人名)
- 宏观为主, 细节为辅

【主题】{theme_name}
【主题判断】{theme_statement}
【内部线索】{theme_notes}

【命题列表】
{claims_block}

【输出】
正文 markdown, 段落之间空一行。每个陈述性句子句末跟 [[claim_id]] 标注
(多个命题用逗号分隔), id 只能取上面列出的。不要输出标题, 不要输出标题以外的 markdown 标记。"""
