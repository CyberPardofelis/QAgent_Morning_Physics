"""
test_prompts_survey — `prompts_survey` 模板 + builder 单测

覆盖:
  - "pool 是采样,不是领域全集" 段 (RAG 局部最优防御) 两个 prompt 都注入,
    且来自同一共享常量
  - 占位符注入正确 (no raw {xxx} 残留)
  - claim-centric 段落保留 (回归检查, 断言语义而非时间戳)
  - 反例文本必须出现在 prompt 里 (LLM 视觉锚点)
  - prompt 正文不得含迭代时间戳 (否则 prompt 永远无法演进)

为什么重要: prompts_survey 是 survey_generate 工具的"灵魂", 改坏一个字符
就让整轮综述产出降级。回退会让 LLM 不再写"claim-centric"或不再避开"主流/共识"型断言,
触发 RAG 局部最优陷阱 (pool ≠ 领域全集)。

9-27 修正说明: 本文件此前把 prompt 里的迭代时间戳(`assert "9-08 23:50" in prompt`)
写成了断言契约, 导致 prompt 正文完全无法清理。已全部改为语义断言。
"""
from agent import prompts_survey
from agent.prompts_survey import (
    POOL_SAMPLING_RULE,
    SECTION_REF_QUOTAS,
    SURVEY_OUTLINE_PROMPT,
    SURVEY_SECTION_PROMPT,
    build_fact_prompt,
    build_outline_prompt,
    build_section_prompt,
    build_summarize_prompt,
)


# ============================================================
# 测试数据 (跟模块自检同源)
# ============================================================
SAMPLE_PAPERS = [
    {
        "paper_id": "2401.12345",
        "title": "Surface code under biased noise",
        "year": "2024",
        "journal": "PRX",
        "core_method": "XZZX surface code with tailored bias",
        "core_contrib": "10x threshold improvement under bias-100 noise",
        "abstract": "We propose a biased-noise surface code variant.",
    },
    {
        "paper_id": "1809.10727",
        "title": "Subsystem codes",
        "year": "2018",
        "journal": "PRX",
        "core_method": "Pauli-based subsystem codes",
        "core_contrib": "New subsystem framework for biased noise",
    },
]

SAMPLE_OUTLINE_SECTION = (
    "- 早期突破 (2010-2015)\n"
    "- 中期优化 (2015-2020)\n"
    "- 近期进展 (2020+)\n"
    "- 转折驱动力\n"
    "- 未来 5 年方向"
)


# ============================================================
# A 改动验证 — "pool 是采样" 段必须出现在两个 prompt
# ============================================================
class TestPoolSamplingWarning:
    """候选池是采样、不是领域全集 (RAG 局部最优防御) —— 两个 prompt 都必须带

    9-27: 这段原本在两个模板里**逐字重复 710 字符**。重复的代价是改一处忘另一处
    → 两边规则悄悄漂移。现在抽成共享常量 POOL_SAMPLING_RULE, 由两个
    build_*_prompt 注入。本组测试锁定"两边都在"且"内容一致"。
    """

    def test_outline_prompt_includes_pool_sampling_warning(self):
        """大纲 prompt 必须有"pool 是采样"段"""
        prompt = build_outline_prompt(topic="量子纠错码进展", papers=SAMPLE_PAPERS)
        assert "不是领域全部论文" in prompt
        assert "RAG 局部最优" in prompt
        assert "Gilbert" in prompt  # 反例文本(作为视觉锚点)
        assert "领域共识" in prompt  # 关键二元区分

    def test_section_prompt_includes_pool_sampling_warning(self):
        """章节 prompt 必须有"pool 是采样"段"""
        prompt = build_section_prompt(
            topic="量子纠错码进展",
            section_type="progress",
            section_title="关键进展时间线",
            outline_section=SAMPLE_OUTLINE_SECTION,
            papers=SAMPLE_PAPERS,
        )
        assert "不是领域全部论文" in prompt
        assert "RAG 局部最优" in prompt
        assert "Gilbert" in prompt
        assert "领域共识" in prompt

    def test_both_prompts_share_one_source_of_truth(self):
        """
        回归: 两个 prompt 里的"pool 是采样"段必须来自同一个常量, 内容一致。
        之前是两份复制粘贴的副本 —— 改一处忘另一处就会漂移。
        """
        expected = POOL_SAMPLING_RULE.format(pool_size=len(SAMPLE_PAPERS))
        outline_prompt = build_outline_prompt(topic="X", papers=SAMPLE_PAPERS)
        section_prompt = build_section_prompt(
            topic="X", section_type="progress", section_title="T",
            outline_section=SAMPLE_OUTLINE_SECTION, papers=SAMPLE_PAPERS,
        )
        assert expected in outline_prompt, "大纲 prompt 未注入共享片段"
        assert expected in section_prompt, "章节 prompt 未注入共享片段"

    def test_templates_use_placeholder_not_literal(self):
        """模板里应是占位符, 实际文本由 build 函数注入 —— 单一真相来源。"""
        assert "{pool_sampling_rule}" in prompts_survey.SURVEY_OUTLINE_PROMPT
        assert "{pool_sampling_rule}" in prompts_survey.SURVEY_SECTION_PROMPT
        # 模板本身不应残留已展开的正文
        assert "Gilbert" not in prompts_survey.SURVEY_OUTLINE_PROMPT
        assert "Gilbert" not in prompts_survey.SURVEY_SECTION_PROMPT

    def test_pool_size_injected_into_pool_sampling_warning(self):
        """pool_size 实际值必须进入新段 (不能仍是占位符 {pool_size})"""
        prompt_3 = build_outline_prompt(topic="X", papers=SAMPLE_PAPERS)
        assert "{pool_size}" not in prompt_3, "占位符未替换!"
        assert "2 篇" in prompt_3  # SAMPLE_PAPERS 有 2 篇

        prompt_s_3 = build_section_prompt(
            topic="X", section_type="progress",
            section_title="T", outline_section=SAMPLE_OUTLINE_SECTION,
            papers=SAMPLE_PAPERS,
        )
        assert "{pool_size}" not in prompt_s_3
        assert "2 篇" in prompt_s_3


# ============================================================
# 占位符注入正确性 (回归检查 — 任何 .format() 字段都不能残留)
# ============================================================
class TestPlaceholderInjection:
    """所有 .format() 占位符都必须被注入, 不残留 {xxx}"""

    def test_outline_prompt_no_residual_placeholders(self):
        prompt = build_outline_prompt(topic="量子纠错码进展", papers=SAMPLE_PAPERS)
        # 占位符都已替换
        for ph in ["{topic}", "{pool_size}", "{pool_text}"]:
            assert ph not in prompt, f"{ph} 未替换"

    def test_section_prompt_no_residual_placeholders(self):
        prompt = build_section_prompt(
            topic="T1", section_type="progress",
            section_title="T2", outline_section=SAMPLE_OUTLINE_SECTION,
            papers=SAMPLE_PAPERS,
        )
        for ph in [
            "{topic}", "{section_type}", "{section_title}",
            "{outline_section}", "{pool_size}", "{pool_text}", "{ref_quota}",
        ]:
            assert ph not in prompt, f"{ph} 未替换"

    def test_fact_prompt_no_residual_placeholders(self):
        prompt = build_fact_prompt(
            paper_id="2401.12345",
            title="Surface code",
            authors="Smith et al.",
            year="2024",
            journal="PRX",
            abstract="We propose ...",
            core_method="XZZX",
            core_contrib="10x improvement",
            outlook="future direction",
        )
        for ph in [
            "{paper_id}", "{title}", "{authors}", "{year}",
            "{journal}", "{abstract}", "{core_method}",
            "{core_contrib}", "{outlook}",
        ]:
            assert ph not in prompt, f"{ph} 未替换"

    def test_summarize_prompt_includes_paper_id_and_title(self):
        prompt = build_summarize_prompt(
            paper_id="2401.12345",
            title="Surface code",
            authors="Smith et al.",
            abstract="We propose X.",
        )
        # summarize prompt 是 f-string, 不用 .format(); 但 paper_id/title 必现
        assert "2401.12345" in prompt
        assert "Surface code" in prompt
        # 输出 schema 4 字段必须提到
        for field in ["core_method", "core_contrib", "keywords", "outlook"]:
            assert field in prompt

    def test_section_prompt_ref_quota_per_type(self):
        """不同章节类型的 ref_quota 必须独立切换"""
        for st in ["intro", "progress", "method_compare", "gap", "outlook"]:
            prompt = build_section_prompt(
                topic="T", section_type=st,
                section_title="T", outline_section=SAMPLE_OUTLINE_SECTION,
                papers=SAMPLE_PAPERS,
            )
            assert SECTION_REF_QUOTAS[st] in prompt, f"{st} 的 ref_quota 未注入"


# ============================================================
# 旧段落保留 (回归 — 9-22 A 改动不能误删已有约束)
# ============================================================
class TestOldConstraintsPreserved:
    """claim-centric 视角 / 强制约束 / 反例段落必须保留

    9-27 修正: 原来这里断言 `assert "9-08 23:50" in prompt` —— 把 prompt 里的
    迭代时间戳当成了契约。这条测试让 prompt 永远无法清理(改任何措辞都会红),
    属于负资产。现在改为断言**语义**, 时间戳标记已从 prompt 正文移除。
    """

    def test_outline_prompt_preserves_claim_centric_paragraph(self):
        prompt = build_outline_prompt(topic="X", papers=SAMPLE_PAPERS)
        assert "claim-centric" in prompt
        # 语义: 要求判断式陈述句, 而不是问题式罗列
        assert "判断式" in prompt or "判断/分类/趋势" in prompt
        assert "问题式" in prompt  # 反例(❌ 错误写法)必须保留

    def test_section_prompt_preserves_claim_centric_paragraph(self):
        prompt = build_section_prompt(
            topic="X", section_type="progress",
            section_title="T", outline_section=SAMPLE_OUTLINE_SECTION,
            papers=SAMPLE_PAPERS,
        )
        assert "claim-centric" in prompt
        assert "paper 罗列" in prompt  # 反例段仍在

    def test_prompts_have_no_iteration_timestamps(self):
        """
        回归: prompt 正文里不应再出现迭代时间戳/拍板标记。
        这些是开发过程的考古记录, 对模型零价值, 却让 prompt 无法演进。
        """
        import re

        pat = re.compile(r"\b9-\d{2}\b|\b\d{1,2}-\d{1,2} (?:user|拍板)|user 拍板")
        for name in (
            "SURVEY_OUTLINE_PROMPT", "SURVEY_SECTION_PROMPT", "SURVEY_FACT_PROMPT",
            "SURVEY_RELEVANCE_SCORE_PROMPT", "SURVEY_FACT_REVIEW_PROMPT",
            "POOL_SAMPLING_RULE",
        ):
            text = getattr(prompts_survey, name)
            hits = pat.findall(text)
            assert not hits, f"{name} 仍含迭代时间戳/拍板标记: {hits}"

    def test_section_prompt_preserves_mandatory_rules(self):
        """章节强制约束 1-9 都在"""
        prompt = build_section_prompt(
            topic="X", section_type="progress",
            section_title="T", outline_section=SAMPLE_OUTLINE_SECTION,
            papers=SAMPLE_PAPERS,
        )
        # 关键硬约束(子集, 抽样检查)
        assert "[paper_id]" in prompt  # 强制引用
        assert "不得引用候选池外的 paper_id" in prompt  # 候选池外幻觉防御
        assert "不得编造方法/数字/实验结果" in prompt  # 数字幻觉防御

    def test_outline_prompt_preserves_five_section_types(self):
        """大纲 prompt 必须有完整 5 段类型"""
        prompt = build_outline_prompt(topic="X", papers=SAMPLE_PAPERS)
        for st in ["intro", "progress", "method_compare", "gap", "outlook"]:
            assert st in prompt, f"5 段章节类型 {st} 缺失"


# ============================================================
# 边界 / 容错
# ============================================================
class TestEdgeCases:
    """边界 / 兜底"""

    def test_outline_prompt_empty_topic(self):
        """空 topic 兜底成 (未指定主题), 不抛"""
        prompt = build_outline_prompt(topic="", papers=SAMPLE_PAPERS)
        assert "(未指定主题)" in prompt

    def test_section_prompt_unknown_section_type_fallback(self):
        """未知 section_type 兜底成 intro, 不抛"""
        prompt = build_section_prompt(
            topic="X", section_type="totally_unknown_type",
            section_title="T", outline_section=SAMPLE_OUTLINE_SECTION,
            papers=SAMPLE_PAPERS,
        )
        # 兜底引用 intro 的 ref_quota
        assert SECTION_REF_QUOTAS["intro"] in prompt

    def test_section_prompt_pool_size_zero_no_break(self):
        """空候选池也能构造 prompt (不抛)"""
        prompt = build_section_prompt(
            topic="X", section_type="intro",
            section_title="T", outline_section="(无大纲)",
            papers=[],
        )
        assert "0 篇" in prompt  # 空池子, 0 篇仍出现


# ============================================================
# sanity: SURVEY_OUTLINE_PROMPT / SURVEY_SECTION_PROMPT 模板本身有 RAG 防御段
# (即使 builder 改动也不能丢失 — 因为新段就在这里)
# ============================================================
class TestRawTemplateIntegrity:
    """直接验证模板常量引用了共享段占位符, 防止 builder 重写把注入点挪走

    9-27: 该段已抽成 POOL_SAMPLING_RULE 常量, 模板里只留 {pool_sampling_rule}。
    """

    def test_outline_template_has_pool_sampling_placeholder(self):
        assert "{pool_sampling_rule}" in SURVEY_OUTLINE_PROMPT
        assert "{pool_sampling_rule}" not in build_outline_prompt(
            topic="X", papers=SAMPLE_PAPERS
        ), "占位符必须已被替换"

    def test_section_template_has_pool_sampling_placeholder(self):
        assert "{pool_sampling_rule}" in SURVEY_SECTION_PROMPT
        assert "{pool_sampling_rule}" not in build_section_prompt(
            topic="X", section_type="progress", section_title="T",
            outline_section=SAMPLE_OUTLINE_SECTION, papers=SAMPLE_PAPERS,
        ), "占位符必须已被替换"
