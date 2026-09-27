"""
tests/test_survey_section_materials.py — 综述写作阶段素材供给的单元测试

9-27 修复配套测试。

背景 (综述"显得傻"的根因):
    extract_facts 对候选池每篇论文花 1 次 LLM 抽出原子 fact(含 support_quote
    原文摘录), 但 build_section_prompt 的签名里**没有 facts 参数** ——
    全仓 grep 确认, state["facts"] 只出现在计数、审计报告、对齐打分三处,
    没有一处流进写作阶段。

    与此同时 format_pool 把每篇论文压成
    "title[:100] | method[:80] | contrib[:80]" ≈ 260 字符, **abstract 被整个丢掉**。

    于是: 花 50+ 次 LLM 精读的内容被丢弃, 写作模型拿到的是一份书目卡片表,
    而 prompt 要求它写 5-8 个领域判断 + 每点 2-4 个引用 + 1500-2500 字。
    输入信息量和输出要求差两个数量级 → 只能编或写空话。

本测试锁定修复后的行为:
  - build_section_prompt 真的接收并使用 facts
  - fact 的 claim 与 support_quote 都进了 prompt
  - 有 fact 时没有把 abstract 丢掉
  - 相关度选取能在超量候选池里挑出与本章节相关的论文
  - 选不出来/无 fact 时能安全退化
"""
from __future__ import annotations

from agent.prompts_survey import (
    _tokenize,
    build_section_prompt,
    format_papers_with_facts,
    select_relevant_papers,
)


# ============================================================
# 测试数据
# ============================================================

def _pool(n: int = 30):
    return [
        {
            "paper_id": f"p{i:02d}",
            "title": f"Unrelated topic number {i} about miscellaneous subject {i}",
            "year": "2020",
            "journal": "arxiv",
            "core_method": f"method {i}",
            "core_contrib": f"contribution {i}",
            "abstract": f"abstract {i}",
            "keywords": "",
        }
        for i in range(n)
    ]


FACTS = [
    {
        "paper_id": "pA",
        "fact_type": "result",
        "claim": "XZZX code achieves 10x threshold improvement under bias-100 noise",
        "support_quote": "We demonstrate a 10x improvement in the threshold under bias-100.",
    },
    {
        "paper_id": "pB",
        "fact_type": "method",
        "claim": "Pauli-based subsystem codes reduce the overhead by three times",
        "support_quote": "Our subsystem framework reduces overhead by a factor of three.",
    },
    {
        "paper_id": "pC",
        "fact_type": "limitation",
        "claim": "Surface code threshold degrades severely under strong bias",
        "support_quote": "The threshold degrades significantly as the bias increases.",
    },
]

RELEVANT_PAPERS = [
    {
        "paper_id": "pA", "title": "Biased noise surface code", "year": "2024",
        "journal": "PRX", "core_method": "XZZX", "core_contrib": "10x threshold",
        "abstract": "surface code under bias noise",
    },
    {
        "paper_id": "pB", "title": "Subsystem codes", "year": "2021",
        "journal": "PRX", "core_method": "Pauli", "core_contrib": "3x overhead",
        "abstract": "subsystem code overhead",
    },
    {
        "paper_id": "pC", "title": "Threshold degradation", "year": "2020",
        "journal": "PRL", "core_method": "noise", "core_contrib": "degrades",
        "abstract": "threshold under bias",
    },
]

OUTLINE = (
    "- biased noise 下阈值退化\n"
    "- XZZX 码的阈值改进\n"
    "- subsystem code 的开销优势"
)


def _build(papers, facts=None, limit=24, section_type="progress"):
    return build_section_prompt(
        topic="量子纠错码",
        section_type=section_type,
        section_title="关键进展",
        outline_section=OUTLINE,
        papers=papers,
        facts=facts,
        papers_limit=limit,
    )


# ============================================================
# 核心回归: fact 真的进了 prompt
# ============================================================

def test_facts_are_included_in_section_prompt():
    """回归核心: extract_facts 的产出必须真的流进写作阶段。"""
    prompt = _build(RELEVANT_PAPERS, facts=FACTS)
    for f in FACTS:
        assert f["claim"] in prompt, f"fact claim 未进入 prompt: {f['claim'][:40]}"


def test_support_quotes_are_included():
    """support_quote 原文摘录也要给 —— 模型据此判断 fact 可信度。"""
    prompt = _build(RELEVANT_PAPERS, facts=FACTS)
    for f in FACTS:
        assert f["support_quote"] in prompt


def test_prompt_never_blanked_placeholder():
    prompt = _build(RELEVANT_PAPERS, facts=FACTS)
    assert "{pool_text}" not in prompt
    assert "{pool_note}" not in prompt
    assert "{facts}" not in prompt
    assert "{pool_size}" not in prompt


def test_paper_ids_present_for_citation():
    prompt = _build(RELEVANT_PAPERS, facts=FACTS)
    for pid in ("pA", "pB", "pC"):
        assert pid in prompt, f"paper_id {pid} 应出现在 prompt 里供引用"


def test_backward_compatible_without_facts():
    """不传 facts 时行为不变(向后兼容) —— 现有调用方/测试不受影响。"""
    prompt = _build(RELEVANT_PAPERS, facts=None)
    assert "pA" in prompt
    assert "{pool_note}" not in prompt


def test_empty_facts_falls_back_safely():
    prompt = _build(RELEVANT_PAPERS, facts=[])
    assert "pA" in prompt
    assert "{pool_note}" not in prompt


# ============================================================
# abstract 不再被丢掉
# ============================================================

def test_abstract_available_when_no_facts_for_paper():
    """
    之前 format_pool 把 abstract 整个丢弃, 论文只剩 260 字符卡片。
    没有 fact 的论文至少要给摘要。
    """
    papers = [{
        "paper_id": "pZ", "title": "T", "year": "2020", "journal": "arxiv",
        "core_method": "M", "core_contrib": "C", "abstract": "这是一个很具体的摘要内容用于验证",
    }]
    out = format_papers_with_facts(papers, {})
    assert "这是一个很具体的摘要内容用于验证" in out


def test_fact_rendering_includes_type_and_quote():
    out = format_papers_with_facts(RELEVANT_PAPERS, {"pA": [FACTS[0]]})
    assert "[result]" in out
    assert "10x threshold improvement" in out
    assert "10x improvement in the threshold" in out


def test_fact_rendering_caps_facts_per_paper():
    many = [{"paper_id": "pA", "fact_type": "result", "claim": f"claim {i}", "support_quote": ""}
            for i in range(20)]
    out = format_papers_with_facts(RELEVANT_PAPERS, {"pA": many}, max_facts_per_paper=3)
    assert out.count("[result]") == 3


def test_fact_rendering_empty_papers():
    assert format_papers_with_facts([], {}) == "(空候选池)"


def test_fact_rendering_skips_blank_claims():
    out = format_papers_with_facts(RELEVANT_PAPERS, {"pA": [{"claim": "   ", "fact_type": "x"}]})
    assert "·" not in out.split("\n")[1]  # 该论文不应渲染出空的 fact 行


# ============================================================
# 相关度选取
# ============================================================

def test_tokenize_handles_cjk_and_latin():
    toks = _tokenize("surface code 表面码 threshold")
    assert "surface" in toks
    assert "code" in toks
    assert "表" in toks          # 中文字
    assert "表面" in toks         # 中文相邻字对


def test_tokenize_empty():
    assert _tokenize("") == set()
    assert _tokenize(None) == set()


def test_select_returns_all_when_under_limit():
    papers = _pool(5)
    assert len(select_relevant_papers(papers, "anything", None, limit=24)) == 5


def test_select_caps_at_limit():
    papers = _pool(30)
    got = select_relevant_papers(papers, "miscellaneous", None, limit=10)
    assert len(got) == 10


def test_select_prefers_relevant_papers():
    """
    核心: 候选池远超上限时, 必须把与本章节相关的挑出来, 而不是随机取前 N。
    构造 30 篇无关 + 3 篇强相关, 章节大纲讲 surface code。
    """
    papers = _pool(30) + list(RELEVANT_PAPERS)
    facts_by_paper = {"pA": [FACTS[0]], "pB": [FACTS[1]], "pC": [FACTS[2]]}
    got = select_relevant_papers(
        papers, f"关键进展\n{OUTLINE}", facts_by_paper, limit=6,
    )
    got_ids = {p["paper_id"] for p in got}
    assert {"pA", "pB", "pC"} <= got_ids, (
        f"强相关论文应被优先选中, 实际选中: {sorted(got_ids)}"
    )


def test_select_uses_facts_text_for_matching():
    """
    一篇标题毫不相关、但 fact 讲 surface code 的论文, 也应被选中 ——
    说明打分确实读了 fact 内容, 不是只看标题。
    """
    papers = _pool(30) + [{
        "paper_id": "hidden", "title": "zzz qqq unrelated", "year": "2024",
        "journal": "", "core_method": "", "core_contrib": "", "abstract": "",
    }]
    got = select_relevant_papers(
        papers, f"关键进展\n{OUTLINE}", {"hidden": [FACTS[0]]}, limit=3,
    )
    assert "hidden" in {p["paper_id"] for p in got}


def test_select_stable_ordering_on_tie():
    """同分时保持候选池原顺序(候选池已按 relevance 排过, 稳定排序能保住优先级)。"""
    papers = _pool(10)
    got = select_relevant_papers(papers, "zzz qqq xxx", None, limit=5)
    assert [p["paper_id"] for p in got] == ["p00", "p01", "p02", "p03", "p04"]


def test_select_empty_query_returns_head():
    papers = _pool(30)
    got = select_relevant_papers(papers, "", None, limit=5)
    assert [p["paper_id"] for p in got] == ["p00", "p01", "p02", "p03", "p04"]


def test_select_empty_papers():
    assert select_relevant_papers([], "x", None, limit=5) == []


def test_select_handles_paper_missing_fields():
    """论文缺字段不能崩。"""
    papers = [{"paper_id": "pX"}, {"paper_id": "pY", "title": "surface code"}]
    got = select_relevant_papers(papers, "surface code", None, limit=1)
    assert len(got) == 1


# ============================================================
# token 预算: 选取让 prompt 大小可控
# ============================================================

def test_selection_keeps_prompt_bounded():
    """
    50 篇池子全带 fact 时, prompt 不应无界膨胀 ——
    选取把论文数压到 papers_limit。
    """
    papers = _pool(50)
    facts = [
        {"paper_id": p["paper_id"], "fact_type": "result",
         "claim": f"claim for {p['paper_id']} " + "x" * 200,
         "support_quote": "q" * 150}
        for p in papers
    ]
    prompt = _build(papers, facts=facts, limit=20)
    assert prompt.count("·") <= 20 * 5  # 每篇最多 4 条 fact + 可能摘要行
    assert "{pool_note}" not in prompt


def test_pool_note_states_selection():
    prompt = _build(_pool(30), facts=None, limit=10)
    assert "选出的 10 篇" in prompt
    assert "共 30 篇" in prompt
