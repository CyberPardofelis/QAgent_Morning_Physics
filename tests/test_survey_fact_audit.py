"""
tests/test_survey_fact_audit.py — fact 落地审查 + 抽取规范化

9-29 新增。

要解决的问题 (实测):
    上一版综述里"离子阱 / 利用每个离子内的附加能级 / 无需辅助离子"这组细节,
    论文摘要**根本没有** —— 原文只提了 ion trap 和 nonorthogonal,
    抽取器看到两个词就自行补全成一条完整机制。写作阶段只是照抄 fact,
    所以写作 prompt 怎么约束都拦不住。**编造源头在 fact 抽取阶段。**

为什么用 LLM 审查而不是确定性词表检查:
    实测 181 条真实 fact 的 claim 是**中文**、原文是**英文**,
    token 覆盖率中位数 0.021 —— 词面重叠检查对跨语言根本不成立。
    改查"英文术语"则会被残留的 [paper_id] 文本污染, 且中文编造细节看不见。
    所以确定性检查已被移除(见 test_no_deterministic_grounding_check),
    语义判断交给 _fact_audit。
"""
from __future__ import annotations

import inspect
import json

import pytest

from agent.prompts_survey import clean_fact, strip_paper_id_from_claim
from agent.tools.surveys import _fact_audit as A
from agent.tools.surveys._actions import _extract_facts as ef


# ============================================================
# 去 paper_id 标注 (确定性、可验证)
# ============================================================

@pytest.mark.parametrize("raw,expected", [
    ("方法很好 [2209.02093] 结束", "方法很好 结束"),
    ("方法 [paper_id: 10.1103/xyz] 好", "方法 好"),
    ("方法 [10.1103/revmodphys.90.035006] 好", "方法 好"),
    ("方法 [2111.08018] 好", "方法 好"),
    ("方法 [0607734] 好", "方法 好"),
    ("方法 [Paper_ID：10.1103/abc] 好", "方法 好"),
])
def test_strip_paper_id_variants(raw, expected):
    assert strip_paper_id_from_claim(raw) == expected


def test_strip_keeps_real_content():
    """不能把正常内容误删。"""
    s = "SIC-POVM 方法在 ion trap 上实现 [10.1103/xyz] ，样本复杂度为 O(k)"
    out = strip_paper_id_from_claim(s)
    assert "SIC-POVM" in out and "ion trap" in out and "O(k)" in out
    assert "[10.1103/xyz]" not in out


def test_strip_empty():
    assert strip_paper_id_from_claim("") == ""
    assert strip_paper_id_from_claim(None) == ""


# ============================================================
# clean_fact
# ============================================================

def test_clean_fact_assigns_unique_id_and_keeps_paper():
    f = clean_fact({"fact_type": "method", "claim": "c", "support_quote": "q"}, "pA", 3)
    assert f["fact_id"] == "pA::3"
    assert f["paper_id"] == "pA"
    assert f["fact_type"] == "method"


def test_clean_fact_strips_paper_id_from_claim():
    f = clean_fact(
        {"claim": "结论成立 [2401.12345]", "support_quote": "q"}, "pA", 0,
    )
    assert "[2401.12345]" not in f["claim"]


def test_clean_fact_flags_self_reference():
    """claim 带"本文/作者"= 去语境化失败, 要标出来。"""
    f = clean_fact({"claim": "本文提出一种新方法", "support_quote": "q"}, "pA", 0)
    assert f["_self_referential"] is True
    g = clean_fact({"claim": "随机测量方法可降低采样开销", "support_quote": "q"}, "pA", 1)
    assert g["_self_referential"] is False


def test_clean_fact_defaults_grounded_true():
    """_grounded 由审查器事后覆盖, 抽取阶段默认 True(保守放行)。"""
    f = clean_fact({"claim": "c", "support_quote": "q"}, "pA", 0)
    assert f["_grounded"] is True
    assert f["_unsupported_terms"] == []


# ============================================================
# 确定性词表检查已被移除
# ============================================================

def test_no_deterministic_grounding_check():
    """
    回归护栏: 别把那个跨语言不成立的词表检查加回来。
    实测覆盖率中位数 0.021(中文 claim vs 英文原文), 加回来只会制造误报。
    """
    from agent import prompts_survey as P
    assert not hasattr(P, "fact_grounding_problems"), (
        "跨语言下确定性词表检查会大量误报, 语义审查已由 _fact_audit 承担"
    )


# ============================================================
# 审查器
# ============================================================

class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt, **_kw):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


@pytest.fixture
def patched(monkeypatch):
    def _set(responses):
        fake = _FakeLLM(responses)
        import agent.tools.llm as lm
        monkeypatch.setattr(lm, "_call_llm", fake)
        return fake
    return _set


def _facts():
    return [
        {"fact_id": "pA::0", "paper_id": "pA", "claim": "诚实的 claim", "support_quote": "real quote"},
        {"fact_id": "pA::1", "paper_id": "pA", "claim": "编造的 claim", "support_quote": "real quote"},
    ]


# 审查器现在必须逐字引用原文英文片段, 且片段要能在 source 里找到
# (机械校验 _verify_evidence), 否则即使 LLM 说 grounded 也会被否掉。
AUDIT_OK = json.dumps([
    {"id": "pA::0", "grounded": True, "evidence": "real quote", "unsupported": ""},
    {"id": "pA::1", "grounded": False, "evidence": "",
     "unsupported": "原文只提到 ion trap, claim 自行补出离子附加能级"},
])


def test_audit_marks_ungrounded(patched):
    patched([AUDIT_OK])
    facts = _facts()
    st = A.audit_fact_grounding(facts, {"pA": "this abstract contains a real quote inside"})

    assert facts[0]["_grounded"] is True
    assert facts[1]["_grounded"] is False
    assert "离子附加能级" in facts[1]["_unsupported_terms"][0]
    assert st["grounded"] == 1 and st["ungrounded"] == 1 and st["audited"] == 2


def test_audit_prompt_contains_source_and_claim(patched):
    """审查必须同时看到 claim、出处、论文原文 —— 只看 claim 判不了。"""
    fake = patched([AUDIT_OK])
    A.audit_fact_grounding(_facts(), {"pA": "ORIGINAL ABSTRACT HERE"})
    p = fake.prompts[0]
    assert "ORIGINAL ABSTRACT HERE" in p
    assert "编造的 claim" in p
    assert "pA::0" in p and "pA::1" in p


def test_audit_prompt_warns_about_the_real_fabrication_pattern(patched):
    """
    护栏: prompt 必须显式点名"平台 + 自行补出实现机制"这个已实测到的编造形态,
    否则审查器会漏掉它(这是最容易被放过的一种编造)。
    """
    fake = patched([AUDIT_OK])
    A.audit_fact_grounding(_facts(), {"pA": "src"})
    p = fake.prompts[0]
    assert "ion trap" in p and "附加能级" in p


def test_audit_batch_keeps_uncovered_conservative(patched):
    """审查器漏掉的 id 不应被误判为编造 —— 保守放行。"""
    patched([json.dumps([{"id": "pA::0", "grounded": True, "evidence": "real quote"}])])
    facts = _facts()
    A.audit_fact_grounding(facts, {"pA": "a real quote lives here"})
    assert facts[0].get("_grounded") is not False
    assert facts[1].get("_grounded") is not False, "审查器没覆盖到就不该判编造"


def test_audit_survives_llm_failure(patched):
    """审查器本身挂掉时, 不应把 fact 全判成编造。"""
    import agent.tools.llm as lm
    import agent.prompts_survey  # noqa: F401

    def boom(*a, **k):
        raise RuntimeError("network down")

    patched([])
    lm._call_llm = boom
    facts = _facts()
    st = A.audit_fact_grounding(facts, {"pA": "a real quote lives here"})
    assert facts[0].get("_grounded") is not False
    assert facts[1].get("_grounded") is not False
    assert st["failed"] == 2


def test_audit_handles_garbage_response(patched):
    patched(["完全不是 JSON"])
    facts = _facts()
    st = A.audit_fact_grounding(facts, {"pA": "src"})
    assert st["failed"] == 2
    assert facts[0].get("_grounded") is not False


def test_audit_handles_think_prefix(patched):
    patched(["<think>先看一遍</think>" + AUDIT_OK])
    facts = _facts()
    A.audit_fact_grounding(facts, {"pA": "src"})
    assert facts[1]["_grounded"] is False


def test_audit_batches_large_sets():
    """超过 batch_size 的 fact 要分批, 且不重复/不遗漏。"""
    many = [{"fact_id": f"pA::{i}", "paper_id": "pA", "claim": f"c{i}", "support_quote": "q"}
            for i in range(20)]
    got = []

    def fake_batch(facts, source, provider, model):
        got.extend(f["fact_id"] for f in facts)
        return {f["fact_id"]: {"grounded": True, "unsupported": ""} for f in facts}

    orig = A._audit_batch
    try:
        A._audit_batch = fake_batch
        st = A.audit_fact_grounding(many, {"pA": "s"})
    finally:
        A._audit_batch = orig
    assert got == [f"pA::{i}" for i in range(20)]
    assert st["audited"] == 20


def test_audit_empty_input():
    st = A.audit_fact_grounding([], {})
    assert st["audited"] == 0


# ============================================================
# 接入抽取流程
# ============================================================

def test_extract_facts_runs_grounding_audit():
    src = inspect.getsource(ef.action_extract_facts)
    assert "audit_fact_grounding" in src, "抽取结束必须跑一次落地审查"
    assert "clean_fact" in src


def test_extract_facts_uses_clean_fact():
    src = inspect.getsource(ef.action_extract_facts)
    assert "from agent.prompts_survey import clean_fact" in src


# ============================================================
# 写作阶段会剔除未通过审查的 fact
# ============================================================

def test_writing_excludes_ungrounded_facts():
    """
    核心闭环: 被判编造的 fact 绝不能进写作素材。
    (否则审查只是摆设 —— 抽取阶段拦住了, 写作却又用了)
    """
    from agent.prompts_survey import _is_usable_fact
    ok = {"_grounded": True}
    bad = {"_grounded": False, "_unsupported_terms": ["离子附加能级"]}
    misaligned = {"_is_topic_aligned": False}
    assert _is_usable_fact(ok) is True
    assert _is_usable_fact(bad) is False
    assert _is_usable_fact(misaligned) is False
    assert _is_usable_fact({}) is True, "无标记(旧数据/审查关闭)应放行"


def test_audit_rejects_fabricated_evidence(patched):
    """
    核心护栏: 光靠 LLM 说 grounded: true 是不够的。

    实测问题: 纯 LLM 判断会把明显编造放过去 —— 源文确实有 "ion trap" 和
    "nonorthogonal", 审查器看到就信, 忽略了 claim 自行补出的"离子附加能级"。

    现在的防线是**机械校验**: 审查器必须逐字引用原文英文片段, 引不出来
    (或引了原文没有的词) 就判编造 —— 哪怕它自己说 grounded。
    """
    fake_grounded = json.dumps([
        {"id": "pA::0", "grounded": True, "evidence": "additional atomic level"},
    ])
    patched([fake_grounded])
    facts = [{"fact_id": "pA::0", "paper_id": "pA", "claim": "c", "support_quote": "q"}]
    src = {"pA": "we demonstrate a scalable QST approach using a single measurement setting"}
    A.audit_fact_grounding(facts, src)
    assert facts[0]["_grounded"] is False, (
        "引用了原文没有的片段, 应当判编造 —— 这正是实测被放过的那类"
    )


def test_audit_accepts_verbatim_evidence(patched):
    """引用片段确实在原文里 -> 放行, 并把 evidence 记下来备查。"""
    patched([json.dumps([
        {"id": "pA::0", "grounded": True, "evidence": "single measurement setting"},
    ])])
    facts = [{"fact_id": "pA::0", "paper_id": "pA", "claim": "c", "support_quote": "q"}]
    src = {"pA": "we demonstrate a scalable QST approach using a single measurement setting"}
    A.audit_fact_grounding(facts, src)
    assert facts[0]["_grounded"] is True
    assert facts[0].get("_evidence") == "single measurement setting"


def test_verify_evidence_normalizes_case_and_punct():
    from agent.tools.surveys._fact_audit import _verify_evidence
    src = "We demonstrate a scalable QST approach, using a single measurement setting."
    assert _verify_evidence("single measurement setting", src) is True
    assert _verify_evidence("SINGLE, MEASUREMENT setting!", src) is True
    assert _verify_evidence("additional atomic level", src) is False
    assert _verify_evidence("", src) is False


# ============================================================
# build_fact_prompt 真的把论文内容填进去了
# ============================================================
# 9-29 实测踩的坑: 重写 SURVEY_FACT_PROMPT 时把带占位符的「输入论文」整段删掉了,
# 模型收到空任务直接回"您没有附上论文原文", 抽取结果 0 条。
# 原有测试只断言模板里有占位符, 没断言 build_fact_prompt 真的填了内容 ——
# 这里补上。


def test_fact_prompt_contains_paper_abstract():
    from agent.prompts_survey import build_fact_prompt
    out = build_fact_prompt(
        paper_id="2401.12345", title="Surface code under bias",
        abstract="A VERY DISTINCTIVE ABSTRACT STRING",
        core_method="XZZX", core_contrib="10x",
    )
    assert "A VERY DISTINCTIVE ABSTRACT STRING" in out, "摘要必须真的填进 prompt"
    assert "Surface code under bias" in out
    assert "2401.12345" in out
    assert "XZZX" in out


def test_fact_prompt_has_no_unfilled_placeholders():
    from agent.prompts_survey import build_fact_prompt
    out = build_fact_prompt(paper_id="p", title="t", abstract="a")
    for ph in ("{paper_id}", "{title}", "{abstract}", "{core_method}", "{core_contrib}"):
        assert ph not in out, f"占位符 {ph} 未被替换"


def test_fact_prompt_demands_chinese_claim():
    """
    回归: 加了"只许复述原文"铁律后, 模型把 claim 整句写成英文 support_quote 的复制。
    prompt 必须明确 claim 是中文转述, support_quote 才是照抄的那份。
    """
    from agent.prompts_survey import SURVEY_FACT_PROMPT
    assert "必须用中文写" in SURVEY_FACT_PROMPT
    assert "不要因为" in SURVEY_FACT_PROMPT and "整句写成英文" in SURVEY_FACT_PROMPT


# ============================================================
# 10-02 跨论文合批
# ============================================================
# 要修的问题 (实测 v60e): 旧实现先按 paper_id 分组、再在组内按 batch_size 切。
# fact 配额只有 4-12 条/篇, 永远切不满一批 —— batch_size 写 8 还是 20 结果都一样,
# 退化成"每篇一次调用": 49 篇 = 47 次, 311 条 fact 实测 86 次两级审查, 占整轮 ~28%。
# 改成跨论文合批 + 原文字符预算, 预期降到 ~32 次。


def _mk(n_papers, n_facts, src_len=800):
    facts = [{"fact_id": f"p{i}::{j}", "paper_id": f"p{i}", "claim": "c", "support_quote": "q"}
             for i in range(n_papers) for j in range(n_facts)]
    src = {f"p{i}": "x" * src_len for i in range(n_papers)}
    return facts, src


def test_plan_batches_actually_merges_papers():
    """
    核心回归: 49 篇 x 6 条 fact 必须合成 ~15 批, 而不是 49 批。
    这条断言直接对应 10-02 实测的 47 次调用。
    """
    facts, src = _mk(49, 6)
    batches = A._plan_batches(facts, src, 20)
    assert len(batches) == 15, f"294 条 / 20 = 15 批, 实得 {len(batches)}"
    assert len(batches) < 49, "必须显著少于论文数, 否则等于没合批"
    assert sum(len(b) for b in batches) == len(facts), "不能漏 fact"


def test_plan_batches_respects_batch_size():
    facts, src = _mk(10, 9)          # 90 条
    for b in A._plan_batches(facts, src, 20):
        assert len(b) <= 20, "批大小上限失效"


def test_plan_batches_respects_source_budget():
    """原文总字符是闸门 —— 合批后 prompt 会变长, 不能失控。"""
    facts = [{"fact_id": f"p{i}::0", "paper_id": f"p{i}", "claim": "c", "support_quote": "q"}
             for i in range(10)]
    src = {f"p{i}": "x" * A.PER_PAPER_SOURCE_CHARS for i in range(10)}
    batches = A._plan_batches(facts, src, 20, source_budget=3 * A.PER_PAPER_SOURCE_CHARS)
    assert len(batches) == 4, "每批 3 篇 -> 10 篇切 4 批(3+3+3+1)"
    for b in batches:
        papers = {f["paper_id"] for f in b}
        total = sum(min(len(src[p]), A.PER_PAPER_SOURCE_CHARS) for p in papers)
        assert total <= 3 * A.PER_PAPER_SOURCE_CHARS


def test_plan_batches_keeps_same_paper_contiguous():
    """同一篇的 fact 应聚成连续一段(原文只贴一次), 不能被别的篇插在中间。"""
    facts, src = _mk(5, 4)
    for b in A._plan_batches(facts, src, 6):
        pids = [f["paper_id"] for f in b]
        runs = [p for i, p in enumerate(pids) if i == 0 or pids[i - 1] != p]
        assert len(runs) == len(set(pids)), f"批内同一篇被拆成多段: {pids}"


def test_plan_batches_oversized_single_paper_still_audited():
    """
    一篇本身超预算时必须整篇合进同一批, 不能被预算拆成 N 次调用
    (拆了就等于把同一段原文重复贴 N 遍, 白烧钱); 更不能因为超预算跳过审查。
    """
    facts = [{"fact_id": f"pA::{i}", "paper_id": "pA", "claim": "c", "support_quote": "q"}
             for i in range(5)]
    batches = A._plan_batches(facts, {"pA": "x" * 99999}, 20, source_budget=10)
    assert len(batches) == 1, "同一篇的后续 fact 不新增原文, 不该被预算切开"
    assert sum(len(b) for b in batches) == 5


def test_audit_merges_multiple_papers_into_one_call(patched):
    """两篇论文的 fact 必须合成一次调用。"""
    fake = patched([json.dumps([
        {"id": "pA::0", "grounded": True, "evidence": "one dimensional chain"},
        {"id": "pB::0", "grounded": True, "evidence": "kagome"},
    ])])
    facts = [
        {"fact_id": "pA::0", "paper_id": "pA", "claim": "c", "support_quote": "q"},
        {"fact_id": "pB::0", "paper_id": "pB", "claim": "c", "support_quote": "q"},
    ]
    src = {"pA": "we study a one dimensional chain", "pB": "we report a kagome model"}
    A.audit_fact_grounding(facts, src)
    assert len(fake.prompts) == 1, "两篇应合成一次调用"


def test_audit_checks_evidence_against_the_facts_own_paper(patched):
    """
    合批带来的新风险: 一次 prompt 里有多篇原文, 机械校验如果拿"批内任意一篇"
    去比对, 审查器抄了别的论文的证据也会被放行 —— 这是合批引入的漏判口子。
    """
    patched([json.dumps([
        {"id": "pA::0", "grounded": True, "evidence": "chiral spin liquid"},
        {"id": "pB::0", "grounded": True, "evidence": "chiral spin liquid"},
    ])])
    facts = [
        {"fact_id": "pA::0", "paper_id": "pA", "claim": "c", "support_quote": "q"},
        {"fact_id": "pB::0", "paper_id": "pB", "claim": "c", "support_quote": "q"},
    ]
    src = {
        "pA": "we study matrix product states on a one dimensional chain",
        "pB": "we report a chiral spin liquid phase in a kagome model",
    }
    A.audit_fact_grounding(facts, src)
    assert facts[0]["_grounded"] is False, "证据抄自 pB, pA 里没有 -> 必须判编造"
    assert facts[1]["_grounded"] is True


def test_audit_prompt_labels_each_paper_block_and_fact(patched):
    """合批后 prompt 必须标出每篇原文的归属, 否则审查器分不清证据该从哪篇抄。"""
    fake = patched([json.dumps([
        {"id": "pA::0", "grounded": False, "evidence": "", "unsupported": "x"},
        {"id": "pB::0", "grounded": False, "evidence": "", "unsupported": "x"},
    ])])
    facts = [
        {"fact_id": "pA::0", "paper_id": "pA", "claim": "c", "support_quote": "q"},
        {"fact_id": "pB::0", "paper_id": "pB", "claim": "c", "support_quote": "q"},
    ]
    A.audit_fact_grounding(facts, {"pA": "SRC_A_TEXT", "pB": "SRC_B_TEXT"})
    p = fake.prompts[0]
    assert "paper_id = pA" in p and "paper_id = pB" in p
    assert "SRC_A_TEXT" in p and "SRC_B_TEXT" in p
    assert "所属论文: pA" in p and "所属论文: pB" in p
    assert "不许从别的论文里抄" in p, "必须显式禁止跨篇抄证据"


# ============================================================
# fact 主题对齐: 批大小默认值 + max_tokens 跟着批走
# ============================================================
# 10-02 实测: get_fact_review_batch_size() 的 docstring 和告警文案都写"默认 20",
# 代码三处却都 return 8。311 条 fact 走了 39 批 = ceil(311/8), 铁证。


def test_fact_review_batch_size_default_is_20(monkeypatch):
    import agent.tools.config as C
    monkeypatch.delenv("QAGENT_SURVEY_FACT_REVIEW_BATCH_SIZE", raising=False)
    assert C._resolve_survey_fact_review_batch_size() == 20, "无 env 时应回默认 20"
    monkeypatch.setenv("QAGENT_SURVEY_FACT_REVIEW_BATCH_SIZE", "8")
    assert C._resolve_survey_fact_review_batch_size() == 8, "显式 env 应优先"
    monkeypatch.setenv("QAGENT_SURVEY_FACT_REVIEW_BATCH_SIZE", "999")
    assert C._resolve_survey_fact_review_batch_size() == 20, "越界值应回落到默认值 20"


def test_fact_review_max_tokens_scales_with_batch(monkeypatch):
    """
    批提到 20 后 max_tokens 必须跟着涨 —— 固定 3000 会截断后半批,
    截断的批走 fallback 整批变中性分, 等于白花钱。
    """
    import agent.tools.llm as lm
    from agent.tools.surveys._pool import _facts_review as FR

    captured = {}

    def fake(prompt, **kw):
        captured["max_tokens"] = kw.get("max_tokens")
        captured["n"] = prompt.count("fact_id=")
        return "[]"

    monkeypatch.setattr(lm, "_call_llm", fake)
    facts = [{"fact_id": f"f{i}", "paper_id": "pA", "claim": "c", "fact_type": "method"}
             for i in range(20)]
    FR._review_facts_topic_alignment("topic", facts, threshold=5)
    assert captured["n"] == 20, "20 条应进一次调用"
    assert captured["max_tokens"] >= 4000, (
        f"max_tokens 没跟上批大小: {captured['max_tokens']}"
    )
