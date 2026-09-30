"""
test_claim_layer.py — 命题层 + 引用编号化 (9-30)

对应用户 9-30 的实测反馈:
  "综述很长、看上去到处都是引用, 但引文实际很稀疏……综述当中试图把每个小的
   fact 都引一下, 而无论是否其实它们在讲同一件事"
  "一句话就表达了一个或多个文章的结果, 而不是去详尽地介绍一堆文章"

以及实测发现的引用格式 bug: 正文渲染成 [DOI], 文末编号成 [1], 两套格式对不上。
"""
from __future__ import annotations

import importlib
import sys

import pytest


def _mod(name: str):
    if name not in sys.modules:
        importlib.import_module(name)
    return sys.modules[name]


OG = _mod("agent.tools.surveys._organize")


# ============================================================
# 引用编号化
# ============================================================

def test_doi_in_body_becomes_number():
    """9-30 核心 bug: 正文是 [DOI], 文末是 [1], 永远对不上。"""
    text = "某结论 [10.1103/revmodphys.93.045003] 另一结论 [10.1038/s41567-020-01234]。"
    new, m = OG.renumber_citations(text)
    assert "[1]" in new and "[2]" in new
    assert "10.1103" not in new, "DOI 不该残留在正文"
    assert m == {"10.1103/revmodphys.93.045003": 1, "10.1038/s41567-020-01234": 2}


def test_bare_arxiv_id_numbered():
    """arxiv 来源的 paper_id 就是裸 ID(2301.01234), 漏了它编号也对不上。"""
    new, m = OG.renumber_citations("结论 [2301.01234] 另 [2405.99999]")
    assert new == "结论 [1] 另 [2]"
    assert set(m) == {"2301.01234", "2405.99999"}


def test_same_paper_in_one_sentence_folded():
    """
    用户原话: "综述当中试图把每个小的 fact 都引一下"。
    同一句里同一篇被引 5 次是那个症状的典型表现 —— 编号化这一层就该压掉。
    """
    text = ("A [10.1103/x] B [10.1103/x] C [10.1103/x] D [10.1103/x] E [10.1103/x]。")
    new, m = OG.renumber_citations(text)
    assert new.count("[1]") == 1, f"同句同篇应折叠成一次, 实得 {new!r}"


def test_existing_numbers_untouched():
    """已经是编号的 [1] 不能被当成 paper_id 再编号。"""
    new, m = OG.renumber_citations("见 [1] 与 [2]。")
    assert new == "见 [1] 与 [2]。"
    assert m == {}


def test_non_id_brackets_untouched():
    """普通方括号(参数、枚举)不能被误改。"""
    new, m = OG.renumber_citations("参数 [0, 1] 与 [tensor] 保持")
    assert new == "参数 [0, 1] 与 [tensor] 保持"
    assert m == {}


def test_body_and_reference_list_share_numbering():
    """最关键: 正文编号与文末条目编号必须一致。"""
    body, m = OG.renumber_citations(
        "结论 [10.1103/revmodphys.93.045003] 支持 [10.1103/scipostphyslectnotes.8]")
    pool = [
        {"paper_id": "10.1103/revmodphys.93.045003", "title": "MPS/PEPS", "journal": "RMP", "year": "2015"},
        {"paper_id": "10.1103/scipostphyslectnotes.8", "title": "Anthology", "journal": "SciPost", "year": "2022"},
    ]
    refs = OG.build_reference_list(body, pool, pid2num=m)
    assert "[1] MPS/PEPS" in refs
    assert "[2] Anthology" in refs
    # 正文出现的每个编号, 文末都要有对应条目
    import re
    body_nums = {int(x) for x in re.findall(r"\[(\d+)\]", body)}
    ref_nums = {int(x) for x in re.findall(r"^\[(\d+)\]", refs, flags=re.M)}
    assert body_nums == ref_nums == {1, 2}, f"正文 {body_nums} vs 文末 {ref_nums}"


def test_uncited_paper_not_in_reference_list():
    """没被引用的不该进参考文献(9-29 原有要求)。"""
    body, m = OG.renumber_citations("只有 [10.1103/aaa] 这篇")
    pool = [
        {"paper_id": "10.1103/aaa", "title": "A"},
        {"paper_id": "10.1103/bbb", "title": "B"},
    ]
    refs = OG.build_reference_list(body, pool, pid2num=m)
    assert "A" in refs and "B" not in refs


# ============================================================
# 命题合并: 分组解析
# ============================================================

@pytest.mark.parametrize("raw,expect", [
    ('{"groups": [{"fact_idx": [1, 3]}, {"fact_idx": [2]}]}', [[1, 3], [2]]),
    ('[[1, 3], [2]]', [[1, 3], [2]]),
    ('[[1,2],[4,5,6]]', [[1, 2], [4, 5, 6]]),
    ('   [[1, 2]]   ', [[1, 2]]),
    ("```json\n[[1,2]]\n```", [[1, 2]]),
])
def test_parse_claim_groups(raw, expect):
    assert OG._parse_claim_groups(raw, 10) == expect


def test_parse_claim_groups_filters_out_of_range():
    """模型编造越界序号时要丢掉, 否则会 IndexError 或错配 fact。"""
    assert OG._parse_claim_groups("[[1, 99], [2]]", 3) == [[1], [2]]


def test_parse_claim_groups_empty():
    assert OG._parse_claim_groups("", 5) == []
    assert OG._parse_claim_groups("模型拒绝回答", 5) == []


def test_looks_like_paper_id():
    assert OG._looks_like_paper_id("10.1103/revmodphys.93.045003")
    assert OG._looks_like_paper_id("2301.01234")
    assert OG._looks_like_paper_id("arXiv:2401.01234")
    assert not OG._looks_like_paper_id("1")
    assert not OG._looks_like_paper_id("tensor")
    assert not OG._looks_like_paper_id("0, 1")


# ============================================================
# 命题引用渲染
# ============================================================

def test_claim_renders_all_its_papers():
    """
    核心诉求: "一句话就表达了一个或多个文章的结果"。
    一条命题挂 3 篇 -> 渲染出 3 个引用。
    """
    text = "PEPS 描述二维格点 [[c1]]。"
    c2p = {"c1": ["p1", "p2", "p3"]}
    out, dropped = OG._render_claim_citations_or_drop(text, c2p)
    assert dropped == 0
    assert out == "PEPS 描述二维格点 [p1][p2][p3]。"


def test_multiple_claims_in_one_sentence():
    """一句话挂多个命题 -> 多组引用, 这就是引用密度提升的来源。"""
    text = "面积定律成立 [[c1, c3]] 且与标度维数相关 [[c2]]。"
    c2p = {"c1": ["p1"], "c2": ["p2"], "c3": ["p3"]}
    out, dropped = OG._render_claim_citations_or_drop(text, c2p)
    assert dropped == 0
    assert out == "面积定律成立 [p1][p3] 且与标度维数相关 [p2]。", out


def test_unknown_claim_drops_whole_sentence():
    """严格: 编造的 claim_id -> 整句丢。留句子=制造无出处的断言。"""
    text = "正常句 [[c1]]。编造句 [[cX]]。"
    out, dropped = OG._render_claim_citations_or_drop(
        text, {"c1": ["p1"]})
    assert dropped == 1
    assert "编造句" not in out
    assert "正常句" in out


def test_unannotated_sentence_dropped():
    text = "有标注 [[c1]]。没标注的过渡句。"
    out, dropped = OG._render_claim_citations_or_drop(text, {"c1": ["p1"]})
    assert dropped == 1
    assert "没标注" not in out


def test_empty_claims_map_drops_everything():
    out, dropped = OG._render_claim_citations_or_drop("任何句子 [[c1]]", {})
    assert dropped == 1 and out == ""


# ============================================================
# 按主题合并
# ============================================================

def test_condense_by_theme_small_theme_no_merge(monkeypatch):
    """主题里 fact 太少(<3)时不该硬合并 —— 合并没意义。"""
    OG_m = _mod("agent.tools.surveys._organize")
    facts = [{"fact_id": f"f{i}", "claim": f"c{i}", "paper_id": f"p{i}"} for i in range(2)]
    tmap = {"themes": [{"name": "T", "fact_ids": ["f0", "f1"]}]}
    # 不该调 LLM
    monkeypatch.setattr(OG_m, "condense_facts_into_claims",
                        lambda *a, **k: pytest.fail("fact 太少时不该调 LLM"))
    claims = OG_m.condense_facts_by_theme(facts, tmap)
    assert len(claims) == 2
    assert all(c["n_facts"] == 1 for c in claims)


def test_condense_by_theme_scopes_to_theme(monkeypatch):
    """不同主题的 fact 不该混在一起合并(Phase B 分主题正是因为讲的不是一回事)。"""
    OG_m = _mod("agent.tools.surveys._organize")
    seen_batches = []

    def _fake(tfs, **kw):
        seen_batches.append([f["fact_id"] for f in tfs])
        return [{"claim_id": "c1", "statement": "s", "fact_ids": [f["fact_id"] for f in tfs],
                 "paper_ids": ["p"], "theme_id": "", "n_facts": len(tfs)}]

    monkeypatch.setattr(OG_m, "condense_facts_into_claims", _fake)
    facts = [{"fact_id": f"f{i}", "claim": "c", "paper_id": "p"} for i in range(6)]
    tmap = {"themes": [{"name": "A", "fact_ids": ["f0", "f1", "f2"]},
                       {"name": "B", "fact_ids": ["f3", "f4", "f5"]}]}
    claims = OG_m.condense_facts_by_theme(facts, tmap)
    assert seen_batches == [["f0", "f1", "f2"], ["f3", "f4", "f5"]], "两个主题应分开调用"
    assert {c["theme_name"] for c in claims} == {"A", "B"}


def test_claim_ids_are_unique_across_themes(monkeypatch):
    """claim_id 要带主题前缀, 否则两个主题的 c1 会撞。"""
    OG_m = _mod("agent.tools.surveys._organize")
    monkeypatch.setattr(
        OG_m, "condense_facts_into_claims",
        lambda tfs, **kw: [{"claim_id": "c1", "statement": "s",
                            "fact_ids": [f["fact_id"] for f in tfs],
                            "paper_ids": ["p"], "theme_id": "", "n_facts": len(tfs)}])
    facts = [{"fact_id": f"f{i}", "claim": "c", "paper_id": "p"} for i in range(6)]
    tmap = {"themes": [{"name": "A", "fact_ids": ["f0", "f1", "f2"]},
                       {"name": "B", "fact_ids": ["f3", "f4", "f5"]}]}
    claims = OG_m.condense_facts_by_theme(facts, tmap)
    ids = [c["claim_id"] for c in claims]
    assert len(ids) == len(set(ids)), f"claim_id 撞了: {ids}"

# ============================================================
# 主题地图的缩放性 (2026-09-30 加)
# ============================================================
def test_theme_map_caps_cover_every_subject(monkeypatch):
    """
    9-30 实测事故: 60 篇/303 条 fact 时, 主题地图输出 59349 字符, 撞上
    max_tokens=20000 被截断 -> JSON 解析失败 -> 整轮白跑 40 分钟。
    40 篇那次(295 条)只是侥幸没过线。
    """
    OG = _mod("agent.tools.surveys._organize")
    assert OG.THEME_MAP_MAX_TOKENS >= 32000, "截断过的上限不够, 必须加高"
    assert OG.THEME_MAP_MAX_FACTS <= 220, "fact 上限要能在输出预算内写完"

    facts = [{"fact_id": f"f{i}", "subject": f"s{i%5}", "claim": "c", "paper_id": "p"}
             for i in range(300)]
    clusters = OG.cluster_by_subject(facts)
    s = OG._sample_facts_for_themes(facts, clusters, OG.THEME_MAP_MAX_FACTS)
    assert len(s) == OG.THEME_MAP_MAX_FACTS
    assert len({f["fact_id"] for f in s}) == len(s), "不能重复"
    subs = {f["subject"] for f in s}
    assert subs == {f"s{i}" for i in range(5)}, f"每个 subject 都要有代表, 实得 {subs}"


def test_sampler_keeps_rare_small_subjects(monkeypatch):
    """
    均匀抽样而不是截尾: 小众 subject 往往正是那些有代表性的分支
    (比如"费米子张量网络"), 截尾会把整个主题结构挖掉一块。
    """
    OG = _mod("agent.tools.surveys._organize")
    facts = [{"fact_id": f"g{i}", "subject": "big", "claim": "c", "paper_id": "p"}
             for i in range(280)]
    facts += [{"fact_id": f"x{i}", "subject": f"rare{j}", "claim": "c", "paper_id": "p"}
              for j in range(4) for i in range(5)]
    clusters = OG.cluster_by_subject(facts)
    s = OG._sample_facts_for_themes(facts, clusters, 200)
    subs = {f["subject"] for f in s}
    assert {"rare0", "rare1", "rare2", "rare3"} <= subs, f"小众 subject 被丢了, 实得 {subs}"


def test_sampler_passthrough_under_cap():
    OG = _mod("agent.tools.surveys._organize")
    facts = [{"fact_id": f"f{i}", "subject": "s", "claim": "c", "paper_id": "p"}
             for i in range(50)]
    out = OG._sample_facts_for_themes(facts, OG.cluster_by_subject(facts), 200)
    assert len(out) == 50, "不够 cap 时不该动"


def test_sampler_fills_cap_not_just_one_per_cluster():
    """
    回归: 第二轮起始索引写错成 0, 会全部命中 seen -> added 恒 False -> break,
    300 条只抽出 5 条(每簇一条), cap 完全没达到。
    """
    OG = _mod("agent.tools.surveys._organize")
    facts = [{"fact_id": f"f{i}", "subject": f"s{i%5}", "claim": "c", "paper_id": "p"}
             for i in range(300)]
    s = OG._sample_facts_for_themes(facts, OG.cluster_by_subject(facts), 200)
    assert len(s) == 200, f"应抽满 200, 实得 {len(s)}"

# ============================================================
# 命题层第二轮修正 (9-30 实测驱动)
# ============================================================
def test_batch_by_subject_concentrates_same_subject():
    """
    9-30 实测: 顺序切批时, 讲同一件事的两条 fact(常来自不同论文)隔了几十条,
    落进不同批次, 模型永远没机会合并 —— 合并率只有 1.32:1, 且合并出来的
    几乎全是"同一篇论文的多个细节"(挂 2+ 篇的命题只占 1%)。
    按 subject 聚簇分批, 让不同论文对同一件事的表述落进同一批。
    """
    OG = _mod("agent.tools.surveys._organize")
    facts = [{"fact_id": f"f{i}", "subject": f"s{i%4}", "claim": "c", "paper_id": f"p{i%9}"}
             for i in range(60)]
    batches = OG._batch_by_subject(facts, 24)
    assert len(batches) == 3 and [len(b) for b in batches] == [24, 24, 12]
    from collections import Counter
    # 顺序切: 每批 4 个 subject 各 6 条(完全打散, 同一 subject 最集中也就 6 条)
    seq = [facts[i:i + 24] for i in range(0, 60, 24)]
    seq_max = max(Counter(f["subject"] for f in b).most_common(1)[0][1] for b in seq)
    # 聚簇切: 同一 subject 在某一批里应该显著更集中
    clu_max = max(Counter(f["subject"] for f in b).most_common(1)[0][1] for b in batches)
    assert clu_max > seq_max, f"聚簇后集中度应高于顺序切: 聚簇 {clu_max} vs 顺序 {seq_max}"


def test_batch_by_subject_never_loses_facts():
    OG = _mod("agent.tools.surveys._organize")
    facts = [{"fact_id": f"f{i}", "subject": f"s{i%7}", "claim": "c", "paper_id": "p"}
             for i in range(53)]
    got = [f["fact_id"] for b in OG._batch_by_subject(facts, 24) for f in b]
    assert sorted(got) == sorted(f["fact_id"] for f in facts)
    assert len(got) == len(set(got)), "不能重复"


def test_select_claims_prefers_cross_paper():
    """
    综述价值排序: 跨论文支持的命题写出来就是"一句话几个引用"。
    只挂 1 篇的写出来仍是逐篇罗列 —— 那正是要治的病。
    """
    OG = _mod("agent.tools.surveys._organize")
    claims = [
        {"claim_id": "single", "paper_ids": ["p1"], "n_facts": 9, "fact_ids": ["x"]},
        {"claim_id": "cross3", "paper_ids": ["p1", "p2", "p3"], "n_facts": 3, "fact_ids": ["y"]},
        {"claim_id": "cross2", "paper_ids": ["p1", "p2"], "n_facts": 2, "fact_ids": ["z"]},
    ]
    got = [c["claim_id"] for c in OG.select_claims_for_writing(claims, 2)]
    assert got == ["cross3", "cross2"], f"应优先跨论文的, 实得 {got}"
    # 不足上限时原样返回
    assert len(OG.select_claims_for_writing(claims, 99)) == 3
    assert OG.select_claims_for_writing(claims, 0) == []


def test_claim_prompt_tells_model_where_facts_come_from():
    """
    9-30 实测修正: 合并出来的几乎全是"同一篇的多个细节"(挂 2+ 篇只占 1%)。
    根因之一: prompt 没说清"合并的价值在于跨论文", 而且**没告诉模型
    每条 fact 来自哪几篇** —— 它看不见, 合并同论文的细节当然毫无阻力。
    """
    from agent.prompts_survey import CLAIM_CONDENSE_PROMPT as P
    assert "跨论文" in P, "必须点明合并的价值在于跨论文"
    assert "p" in P and "来源论文" in P, "必须告诉模型每条 fact 的来源论文"
    # 同论文合并要明确劝阻
    assert "全都来自同一篇" in P or "都来自同一篇" in P


def test_condense_prompt_passes_paper_labels(monkeypatch):
    """喂给模型的 fact 行里必须带 <pN> 标记, 且编号在批内自洽。"""
    OG = _mod("agent.tools.surveys._organize")
    seen = {}

    def _fake_llm(prompt, **kw):
        seen["prompt"] = prompt
        return '{"groups": [{"fact_idx": [1, 2], "statement": "s"}]}'

    monkeypatch.setattr("agent.tools.llm._call_llm", _fake_llm)
    facts = [
        {"fact_id": "f1", "claim": "c1", "subject": "s", "paper_id": "P_A"},
        {"fact_id": "f2", "claim": "c2", "subject": "s", "paper_id": "P_B"},
    ]
    out = OG.condense_facts_into_claims(facts)
    assert out and len(out) == 1
    assert set(out[0]["paper_ids"]) == {"P_A", "P_B"}
    assert "<p1>" in seen["prompt"] and "<p2>" in seen["prompt"], \
        f"fact 行要带来源标记, 实得:\n{seen['prompt'][:400]}"
