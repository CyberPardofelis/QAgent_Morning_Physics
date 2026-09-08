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

不依赖外部 LLM 客户端,本模块只负责"prompt 文本构造",不调 API。
调用方(P6.2 survey_generate 工具)负责:
  1. 调 _call_llm(prompt)
  2. parse 输出
  3. 走 post-parse 校验(候选池外引用 → ❌)

历史:
  - 2026-09-05: 3 个 prompt 模板首版(P6.0 阶段)
"""
from typing import List, Dict, Any, Optional


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
def format_pool(papers: List[Dict[str, Any]]) -> str:
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
# 1. SURVEY_FACT_PROMPT — 从单篇 paper 提 3-5 个 fact(Phase C)
# ============================================================
SURVEY_FACT_PROMPT = """你是文献综述助手,任务:对**单篇论文**做事实提取,产出一组原子 fact。

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

【任务】
从上述论文中提取 **3-5 个** 原子 fact(每条 fact 是 1 句事实性陈述),用于后续综述章节生成引用。

【fact 类型】(每条 fact 必须标一个):
  - method: 用了什么方法/算法/框架
  - result: 实验/理论结果(数字 / 性能 / 复杂度)
  - conclusion: 核心结论(该论文最想说的)
  - limitation: 局限/不足/未解决问题
  - hypothesis: 假设/猜想(论文里"假设 X 成立"那种)

【强制约束】
1. 数量: 3-5 条,不多不少(中位数 4,根据信息密度自行调整)
2. claim 末尾**必须**带 [paper_id](用上面给的 paper_id,不要编)
3. support_quote 必须是**原文摘录**(可短可拼接),**不要改写或意译**;找不到原文就降级到 abstract 原句
4. 不要编造方法/数字/实验结果,信息不够就少提,不要凑数
5. 不输出论文的"复述",只提取"可被引用的原子事实"

【输出格式】(严格 JSON 数组,不要 markdown 代码块包裹,不要解释)
[
  {{"fact_id": "f1", "fact_type": "method", "claim": "Smith 等提出用 surface code 做容错, [paper_id]", "support_quote": "We propose a surface code architecture for fault-tolerant quantum computation, [paper_id]"}},
  {{"fact_id": "f2", "fact_type": "result", "claim": "...", "support_quote": "..."}}
]

只输出 JSON,不要前缀解释,不要后缀总结。"""


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
SURVEY_OUTLINE_PROMPT = """你是综述章节规划助手,任务:基于候选池,生成结构化大纲。

【综述主题】: {topic}

【候选池】(共 {pool_size} 篇,paper_id 唯一):
{pool_text}

【章节类型与顺序】(固定 5 段,从上到下,不要漏)
  1. intro: 背景 / 研究问题 / 本文综述范围(0-2 引用)
  2. progress: 关键进展时间线 / 里程碑(5-10 引用,核心章节)
  3. method_compare: 方法分类与对比(3-8 引用,按机制/性能/适用场景分组)
  4. gap: 当前研究空白 / 不足(1-3 引用,以"反例"为主)
  5. outlook: 未来方向 / 开放问题(0-2 引用,基于 gap 推断)

【每章输出 schema】
- title: 章节标题(中文,15 字以内)
- section_type: 5 选 1(intro / progress / method_compare / gap / outlook)
- sub_points: 2-4 个子点,每个子点是 1 句"该章节需要回答的问题"
  ⚠️ 字段名严格用 sub_points(复数),任何 chapter 都不能写成 sub_point(单数)
- is_transition: yes / no
    - yes = 该章主要是铺垫/过渡,不强求引用(给 intro 的部分子节)
    - no = 该章必须有事实引用(给 progress / method_compare / gap / outlook)

【强制约束】
1. 必须**严格按上面 5 段顺序**输出,不能合并章节,不能改顺序
2. paper_id 只能从候选池里选,不能编造
3. 引用配额按"章节类型"软约束:超过上下限会被审计标 ⚠️
4. method_compare 的 sub_points 应该按"机制/性能/场景"维度组织,不要按时间组织(时间组织是 progress 的事)

【输出格式】(严格 JSON 对象,不要 markdown 代码块包裹,不要解释)
{{
  "sections": [
    {{"title": "...", "section_type": "intro", "sub_points": ["...", "..."], "is_transition": "yes"}},
    {{"title": "...", "section_type": "progress", "sub_points": ["...", "...", "..."], "is_transition": "no"}}
  ]
}}

只输出 JSON,不要前缀"以下是...",不要后缀"希望对你有帮助"。"""


def build_outline_prompt(topic: str, papers: List[Dict[str, Any]]) -> str:
    """
    构造综述大纲生成 prompt。

    papers: 候选池 paper 列表(同 format_pool 的输入)
    """
    return SURVEY_OUTLINE_PROMPT.format(
        topic=topic or "(未指定主题)",
        pool_size=len(papers),
        pool_text=format_pool(papers),
    )


# ============================================================
# 3. SURVEY_SECTION_PROMPT — 单章生成 markdown 正文(Phase E)
# ============================================================
SURVEY_SECTION_PROMPT = """你是综述章节写作助手,任务:基于候选池 + 章节大纲,生成 1 个章节的 markdown 正文。

【综述主题】: {topic}
【本章节类型】: {section_type}
【本章节标题】: {section_title}
【本章节大纲】:
{outline_section}

【候选池】(共 {pool_size} 篇,paper_id 唯一):
{pool_text}

【9-08 user 拍板 写作风格 — 最关键】
1. **章节是讲"领域/方向的内容、机制、进展",不是讲"哪些 paper 各自做了啥"**
   - 错误: 每段以 paper 为单位, "...A 提出了 X [paper1]...B 提出了 Y [paper2]...C 改进了 Z [paper3]" — 这是罗列 paper 摘要
   - 正确: 以"机制/思路/技术路线"为线索, 把多篇 paper 串成一个连贯论述, 在关键事实点上 cite [paper_id] 佐证
2. **典型结构**:
   - 开篇 1-2 段: 这个章节要回答的科学/工程问题是什么, 为什么重要, 整体思路
   - 主体 N 段 (按 sub_point 展开): 每段先讲清一个机制/技术/思路, 再用多篇 paper 佐证"哪几篇贡献了哪个具体点/数字/结果"
   - 收尾 1 段: 本章小结 + 跟下一章的衔接
3. **长度要求**: 单章 markdown **800-1500 字** (sub_points × 2-3 段/子点), 内容要饱满, 不要惜字如金
4. **每个 sub_point 至少配 2-4 个 paper_id 引用** (关键事实/数字/结果 处必带), 引用的 paper 是"佐证", 不是"中心"
5. **不要复述 paper 摘要**: 不要写 "A 提出了 X, 它的核心方法是 Y", 写 "X 已成为主流路径 (A [p1], B [p2], C [p3] 都在此方向), 关键改进点在于 ..."

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

【P6.3 自动重写机制】
- 每个章节生成后, 自动审计会检测候选池外引用 (out_of_pool)
- 如果 out_of_pool > 0, 系统会自动用"只允许候选池内 paper_id"作为额外约束重写该章节 (最多 2 次)
- 第 2 次重写后仍 out_of_pool > 0, 接受并落库 (覆盖度审计报告里标 ⚠️)

【"领域叙述 vs paper 罗列" 对比示例】

  ❌ 罗列 paper 摘要 (错误):
    "A 提出了表面码 [p1]。B 改进了表面码 [p2]。C 提出了 XZZX 码 [p3]。D 比较了多种码 [p4]。"
    (把多篇 paper 各开一段, 章节读起来像 "list of paper summaries")

  ✅ 领域叙述 + paper 佐证 (正确):
    "表面码自 Fowler 2012 展示低于 1% 逻辑错误率以来, 已成为容错量子计算的主流路径 [p1]。其核心思路是用二维网格上的物理比特编码单个逻辑比特, 通过周期测量稳定子算子检测错误 [p1][p2]。在偏置噪声场景下, Bravyi 等指出传统表面码阈值显著退化, 由此提出 XZZX 码并展示了 ~10x 的阈值改进 [p3]。近期工作进一步把表面码扩展到子系统码, 在硬件开销和纠错能力间取得新平衡 [p4]。"
    (以"机制演化"为线索, 多篇 paper 串成一个连贯论述, 关键事实点 cite)

【本章节引用配额】(软约束, 超过会被审计标 ⚠️)
  {ref_quota}

【9-08 user 拍板 — 缺漏反馈机制】(关键)
- 如果你写完本章节后, **发现还缺某些方向的直接文献** (例如: "本章还缺 BB84 原始文献"、"需要 MDI-QKD 实验证据"、"缺少诱骗态方法" 等)
- 你可以在 markdown 正文末尾加一行 HTML 注释标注, **系统会自动捕获并回到 search_more 循环补文章**:
  `<!-- QAgent: missing_queries=["BB84 原始协议", "MDI-QKD 实验", "诱骗态方法"] -->`
- 写这个标注 = 你"诚实承认"本章覆盖不完整, **不会扣分**, 反而让综述更扎实
- 不写 = 章节完整 (你确认本章节已覆盖 sub_points 所需的所有证据)
- **query 应该是搜索关键词, 不是 paper_id** (系统会用 query 调 arxiv 扩候选池)
- **query 应该精准**: 1-5 个, 每个 query 是一类文献的关键词 (例: "BB84 protocol original", "MDI-QKD experiment", "decoy-state method")
- **注意**: 标注必须是合法 HTML 注释格式, 且在 markdown 末尾 (其他内容前/中会被审计标 ❌)

【输出】(直接 markdown 章节正文, 不要前缀"以下是...",不要后缀总结,不要"```markdown" 包裹)
"""


# 章节类型 → 引用配额(覆盖度审计的软约束参考)
SECTION_REF_QUOTAS = {
    "intro":          "0-2 个引用,主要讲背景和综述范围",
    "progress":       "5-10 个引用,核心进展章节",
    "method_compare": "3-8 个引用,按机制/性能/场景组织",
    "gap":            "1-3 个引用,以反例为主",
    "outlook":        "0-2 个引用,基于 gap 推断",
}


def build_section_prompt(
    topic: str,
    section_type: str,
    section_title: str,
    outline_section: str,
    papers: List[Dict[str, Any]],
    section_type_safe: Optional[str] = None,
) -> str:
    """
    构造单章生成 prompt。

    section_type_safe: 强校验后的小写 section_type;若不传则用 section_type.lower()。
    """
    safe_type = (section_type_safe or section_type or "").strip().lower()
    if safe_type not in SECTION_REF_QUOTAS:
        safe_type = "intro"  # 兜底,不让 LLM 看到陌生章节类型
    return SURVEY_SECTION_PROMPT.format(
        topic=topic or "(未指定主题)",
        section_type=safe_type,
        section_title=section_title or "(无标题)",
        outline_section=outline_section or "(无大纲)",
        pool_size=len(papers),
        pool_text=format_pool(papers),
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
def build_react_prompt(state: Dict[str, Any]) -> str:
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
    fact_low = fact_count < 5  # fact 不足的硬阈值

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
- 错误数: {error_count}
- ReAct step: {step}/{max_steps}

候选池 paper_id 列表 (**严格使用这里的 id, 不要编造**):
{pool_ids_block}

可选 action (选一个, 严格 JSON 输出):
1. search_more {{"query": "...", "k": 5}} — 扩候选池
2. extract_facts {{"paper_ids": ["上面的真 id"]}} — 抽 fact
3. write_outline {{}} — 生成综述大纲(章节结构), 必须在 write_section 之前
4. write_section {{"section_idx": <int>}} — 按 outline 写第 section_idx 章(0-based); 0=intro, 1=progress, ...
5. finish {{}} — 收尾

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
