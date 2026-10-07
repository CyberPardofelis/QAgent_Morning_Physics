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
import pathlib
import re
import sys

import pytest


def _mod(name: str):
    if name not in sys.modules:
        importlib.import_module(name)
    return sys.modules[name]


OG = _mod("agent.tools.surveys._organize")


def _org_src() -> str:
    """_organize.py 的源码文本 —— 有些断言要看**代码怎么写的**, 不看它的行为。

    行为断言有个盲区: 打印文案、注释里的承诺这些东西, 改了不影响任何返回值,
    而它们恰恰是对用户说错话的地方(10-07 那句「一条 fact 都不丢」就是)。
    """
    import pathlib

    return pathlib.Path(OG.__file__).read_text(encoding="utf-8")


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
# 9-30: 返回值改成 (indices, statement) 二元组 —— statement 会被真正使用,
# 原来模型写的概括被整个丢掉, 换回 fact 原文拼接, 白付 token 又丢掉"综合"本身。

def test_parse_claim_groups_keeps_statement():
    """
    9-30 修的核心: 模型为每组合成的 statement 之前被整个丢弃。
    """
    raw = '{"groups": [{"fact_idx": [1, 3], "statement": "两篇都指出临界点对数标度"}]}'
    assert OG._parse_claim_groups(raw, 10) == [([1, 3], "两篇都指出临界点对数标度")]


@pytest.mark.parametrize("raw,expect", [
    ('{"groups": [{"fact_idx": [1, 3]}, {"fact_idx": [2]}]}', [([1, 3], ""), ([2], "")]),
    ('[[1, 3], [2]]', [([1, 3], ""), ([2], "")]),
    ('[[1,2],[4,5,6]]', [([1, 2], ""), ([4, 5, 6], "")]),
    ('   [[1, 2]]   ', [([1, 2], "")]),
    ("```json\n[[1,2]]\n```", [([1, 2], "")]),
])
def test_parse_claim_groups(raw, expect):
    assert OG._parse_claim_groups(raw, 10) == expect


def test_parse_claim_groups_filters_out_of_range():
    """模型编造越界序号时要丢掉, 否则会 IndexError 或错配 fact。"""
    assert OG._parse_claim_groups("[[1, 99], [2]]", 3) == [([1], ""), ([2], "")]


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
    # 10-02: paper_id 必须各不相同 —— 同篇同 subject 的 fact 会被
    # _dedupe_same_paper_same_subject 先收成一条(见下面那个测试),
    # 全给同一篇的话这个用例测的就不是"主题隔离"了。
    facts = [{"fact_id": f"f{i}", "claim": "c", "paper_id": f"p{i}"} for i in range(6)]
    tmap = {"themes": [{"name": "A", "fact_ids": ["f0", "f1", "f2"]},
                       {"name": "B", "fact_ids": ["f3", "f4", "f5"]}]}
    claims = OG_m.condense_facts_by_theme(facts, tmap)
    assert seen_batches == [["f0", "f1", "f2"], ["f3", "f4", "f5"]], "两个主题应分开调用"
    assert {c["theme_name"] for c in claims} == {"A", "B"}


def test_dedup_collapses_same_paper_same_subject():
    """
    10-02 核心护栏。

    实测 (v60g, 241 条 usable fact): 163 个 subject 里只有 4 个跨论文,
    93% 的 fact 其 subject 只对应一篇论文。把它们批量喂给合并器, 模型面对的
    是"一篇论文的各个细节", 合并它们是摘要不是综合 —— 模型**正确地**不合并
    (实测簇级压缩比 1.01)。所以这一步只能机械做, 不能指望模型。
    """
    OG_m = _mod("agent.tools.surveys._organize")
    facts = [
        {"fact_id": "a1", "paper_id": "P", "subject": "折叠算法", "claim": "用折叠算法做收缩"},
        {"fact_id": "a2", "paper_id": "P", "subject": "折叠算法", "claim": "截断误差可控"},
        {"fact_id": "a3", "paper_id": "P", "subject": "折叠算法", "claim": "键长可调"},
        {"fact_id": "b1", "paper_id": "Q", "subject": "折叠算法", "claim": "另一篇也做折叠"},
    ]
    out = OG_m._dedupe_same_paper_same_subject(facts)
    assert len(out) == 2, "同篇同 subject 的 3 条应收成 1 条"
    assert {f["paper_id"] for f in out} == {"P", "Q"}
    p = next(f for f in out if f["paper_id"] == "P")
    assert p["fact_ids"] == ["a1", "a2", "a3"], "覆盖度审计靠这个, 不能丢"
    assert p["n_facts"] == 3
    assert "截断误差可控" in p["claim"] and "键长可调" in p["claim"], "细节要并进 claim"


def test_dedup_keeps_different_papers_apart():
    """不同论文的同 subject **不能**被收在一起 —— 那才是真正的跨论文合并。"""
    OG_m = _mod("agent.tools.surveys._organize")
    facts = [
        {"fact_id": "a1", "paper_id": "P", "subject": "iPEPS 变分优化", "claim": "x"},
        {"fact_id": "b1", "paper_id": "Q", "subject": "iPEPS 变分优化", "claim": "y"},
    ]
    assert len(OG_m._dedupe_same_paper_same_subject(facts)) == 2


def test_dedup_keeps_different_subjects_of_same_paper_apart():
    """同一篇论文的不同 subject 是不同工作, 不该并成一条。"""
    OG_m = _mod("agent.tools.surveys._organize")
    facts = [
        {"fact_id": "a1", "paper_id": "P", "subject": "iTEBD 算法", "claim": "x"},
        {"fact_id": "a2", "paper_id": "P", "subject": "MPS 基态", "claim": "y"},
    ]
    assert len(OG_m._dedupe_same_paper_same_subject(facts)) == 2


def test_claim_layer_preserves_all_fact_ids_after_dedup(monkeypatch):
    """去重后进命题层的 fact_ids 必须全都在, 否则覆盖度审计会漏。"""
    OG_m = _mod("agent.tools.surveys._organize")
    monkeypatch.setattr(
        OG_m, "condense_facts_into_claims",
        lambda tfs, **kw: [{"claim_id": "c1", "statement": "s",
                            "fact_ids": [fid for f in tfs
                                         for fid in f.get("fact_ids", [f["fact_id"]])],
                            "paper_ids": ["P"], "theme_id": "", "n_facts": len(tfs)}])
    facts = [{"fact_id": f"f{i}", "paper_id": "P", "subject": "S", "claim": f"c{i}"}
             for i in range(5)]
    tmap = {"themes": [{"name": "T", "fact_ids": [f"f{i}" for i in range(5)]}]}
    claims = OG_m.condense_facts_by_theme(facts, tmap)
    assert set(claims[0]["fact_ids"]) == {f"f{i}" for i in range(5)}


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
# 主题地图的缩放性 (2026-09-30 加, 10-05 改)
# ============================================================
def test_theme_map_does_not_drop_facts():
    """
    9-30 实测事故: 60 篇/303 条 fact 时, 主题地图输出 59349 字符, 撞上
    max_tokens=20000 被截断 -> JSON 解析失败 -> 整轮白跑 40 分钟。

    10-05 的修法换了方向: 原来是"超 200 条就均匀抽样到 200"(被抽掉的 fact
    永远不被任何主题覆盖, 实测 309 条静默丢 35%), 现在按字符**分块**逐块归纳
    再按主题名合并。这条测试锁住新不变量: **一条 fact 都不能少**。
    """
    OG = _mod("agent.tools.surveys._organize")
    assert OG.THEME_MAP_MAX_TOKENS >= 32000, "截断过的上限不够, 必须加高"
    # claim 长度刻意拉开差异: 按条数封顶的旧写法就是这么翻车的
    # (同 140 条, prompt 3360~13194 字符, 大的那次卡死 43 分钟)
    facts = [{"fact_id": f"f{i}", "subject": f"s{i%5}",
              "claim": "短" if i % 3 else "长" * 120, "paper_id": "p"}
             for i in range(300)]
    chunks = OG.split_fact_chunks(facts)
    assert len(chunks) > 1, "300 条应该被切开"
    seen = [f["fact_id"] for ch in chunks for f in ch]
    assert sorted(seen) == sorted(f["fact_id"] for f in facts), \
        "分块不能丢 fact 也不能重复"
    for ch in chunks:
        # 不给余量: 上限就是上限。留 +200 容差的话, "只按条数切"这个退化
        # (13194 字符那次卡死 43 分钟的根因) 照样能过 —— 10-05 反向验证实测到。
        n = sum(len(f["claim"]) for f in ch) + 40 * len(ch)
        assert n <= OG.THEME_CHUNK_CHARS, f"块超字符上限: {n} > {OG.THEME_CHUNK_CHARS}"
        assert len(ch) <= OG.THEME_CHUNK_FACTS


def test_chunks_split_by_chars_not_by_count():
    """极端长度分布: 200 条极短的 fact 必须**按字符**切开, 而不是按条数一把梭。

    10-05 踩过: 同样是 140 条, prompt 能从 3360 到 13194 字符(claim 长度差异
    极大), 而 13194 那次卡死 43 分钟。决定输出和稳定性的是 prompt 大小。
    """
    OG = _mod("agent.tools.surveys._organize")
    facts = [{"fact_id": f"f{i}", "subject": "s", "claim": "长" * 400, "paper_id": "p"}
             for i in range(200)]
    chunks = OG.split_fact_chunks(facts)
    # 每条开销 = 400 字 + 40 字([fact_id] (type) 那部分), 所以每块装
    # 8000/440 = 18 条 -> 200/18 = 12 块。**关键是块数由字符决定**:
    # 若只按条数切(上限 60), 会得到 4 块 x 26400 字符, 那正是卡死 43 分钟的形态。
    assert len(chunks) == 12, f"应按字符切成 12 块, 实得 {len(chunks)}"
    for ch in chunks:
        n = sum(len(f["claim"]) for f in ch) + 40 * len(ch)
        assert n <= OG.THEME_CHUNK_CHARS
    # 反证: 只按条数切会是什么样
    by_count = [facts[i:i + OG.THEME_CHUNK_FACTS]
                for i in range(0, len(facts), OG.THEME_CHUNK_FACTS)]
    naive_max = max(sum(len(f["claim"]) for f in c) for c in by_count)
    assert naive_max > OG.THEME_CHUNK_CHARS * 2, \
        "对照组本身就超预算, 说明'按条数'确实是病根"


def test_split_fact_chunks_single_chunk_when_small():
    OG = _mod("agent.tools.surveys._organize")
    facts = [{"fact_id": f"f{i}", "subject": "s", "claim": "c", "paper_id": "p"}
             for i in range(50)]
    assert len(OG.split_fact_chunks(facts)) == 1, "不够一块时不该切"


def test_organize_entry_point_chunks_instead_of_sampling(monkeypatch):
    """入口超量时必须**分块**, 不能抽样(10-05 的核心修复)。

    10-05 之前是 `len(facts) > 200 -> _sample_facts_for_themes(...)`,
    实测 309 条静默丢掉 35%。

    10-07 修: 这条测试原来还传了 `max_themes=1000`, **把截断分支变成了死代码**
    —— 于是它压根没走过它想守护的那段路径, 却在给"一条 fact 都不丢"发绿灯。
    现在断言的是**切块这一步**不丢 fact(合并之后的取舍由下面那条测),
    因为「切块不丢」和「合并后不丢」是两件不同的事, 混在一条里就等于没测。
    """
    OG = _mod("agent.tools.surveys._organize")
    calls = []

    def fake_chunk(facts, paper_titles=None, provider="minimax", model="MiniMax-M3",
                   temperature=0.4, max_themes=8, chunk_idx=1, n_chunks=1,
                   total_facts=0, skeleton=None):
        calls.append([f["fact_id"] for f in facts])
        ids = [f["fact_id"] for f in facts]
        return {"themes": [{"name": f"T{i}", "statement": "", "fact_ids": [ids[i]],
                            "notes": "", "relations": []} for i in range(len(ids))],
                "narrative_arc": [], "gaps": [], "warnings": []}

    monkeypatch.setattr(OG, "_organize_one_chunk", fake_chunk)
    facts = [{"fact_id": f"f{i}", "subject": f"s{i%5}", "claim": "长" * 100,
              "paper_id": "p"} for i in range(300)]
    OG.organize_facts_into_themes(facts)          # ← 不再传 max_themes=1000
    assert len(calls) > 1, f"应当分块调用, 实际只调了 {len(calls)} 次"
    seen = [fid for batch in calls for fid in batch]
    assert sorted(seen) == sorted(f["fact_id"] for f in facts), \
        f"切块这步就丢了 fact: 300 -> {len(seen)}"


def test_merge_does_not_let_early_chunks_take_all_the_slots(monkeypatch):
    """合并后截断时**不能**让前面的块通吃名额(10-07 的核心修复)。

    这条模拟的是**生产实测形态**: 主题名块块不同, 同名合并命中率 0。
    原来的 `themes[:max_themes]` 因为 `by_name` 插入序 = 块序, 于是前两块的
    8 个主题把名额占满, 第 3 块往后整块归零 —— 实测中性原子综述
    422 条 fact 只有 23% 进了地图(块1 95% / 块2 67% / 块3-8 全部 0%)。

    判据是"**每一块都要有 fact 活下来**", 而不是"活下来的 fact 多" ——
    后者在"前两块刚好很大"时也会绿, 那正是原来那个 bug 的形态。
    """
    OG = _mod("agent.tools.surveys._organize")
    N_CHUNKS, PER_CHUNK, N_THEMES_PER_CHUNK = 8, 60, 4
    results = []
    for c in range(N_CHUNKS):
        results.append({
            # **块块不同的主题名** —— 复刻生产的 0% 命中, 不要让测试优待它
            "themes": [
                {"name": f"块{c}主题{i}", "statement": "", "notes": "",
                 "fact_ids": [f"c{c}f{i}_{j}" for j in range(PER_CHUNK // N_THEMES_PER_CHUNK)],
                 "relations": []}
                for i in range(N_THEMES_PER_CHUNK)
            ],
            "narrative_arc": [], "gaps": [], "warnings": [],
        })

    out = OG._merge_chunk_results(results, max_themes=8)
    alive_chunks = set()
    for t in out["themes"]:
        for fid in t["fact_ids"]:
            alive_chunks.add(int(fid[1:].split("f")[0]))
    assert len(out["themes"]) == 8
    assert alive_chunks == set(range(N_CHUNKS)), \
        (f"只有块 {sorted(alive_chunks)} 的 fact 活下来, 块 "
         f"{sorted(set(range(N_CHUNKS)) - alive_chunks)} 整块归零了 —— "
         f"这就是 10-06 实测到的 23% 覆盖")


def test_merge_warns_how_many_facts_were_dropped_and_from_which_chunks():
    """丢了多少、丢哪几块, 必须写进 warnings —— 不然用户看不出来。"""
    OG = _mod("agent.tools.surveys._organize")
    results = [{
        "themes": [{"name": f"块{c}主题{i}", "statement": "", "notes": "",
                    "fact_ids": [f"c{c}f{i}_{j}" for j in range(15)],
                    "relations": []} for i in range(4)],
        "narrative_arc": [], "gaps": [], "warnings": [],
    } for c in range(8)]

    out = OG._merge_chunk_results(results, max_themes=8)
    joined = " ".join(out["warnings"])
    assert any("条 fact" in w for w in out["warnings"]), \
        f"截断了却没有报丢了多少 fact: {out['warnings']}"
    # 断言具体形态 `块N=`, 不能只查有没有"块"字 —— 写成"若干块"照样绿(反向验证 M4 踩过)
    assert re.search(r"块\d+=\d+条", joined), \
        f"告警里没有按块的具体分布: {out['warnings']}"
    # 8 块各丢 45 条 = 360, 轮转下应该是均匀的
    assert "块1=45条" in joined and "块8=45条" in joined, \
        f"轮转下每块丢的应该一样多, 实得: {joined}"


def test_merge_drops_dangling_relations_to_truncated_themes():
    """被砍掉的主题会留下悬空 relation, 那些名字会被带进 Phase C 的 prompt。

    指向 `主题3` 而不是 `主题0`: 轮转第 1 轮就把 8 个名额分完了(每块取
    `主题0`), `主题1/2/3` 全被砍 —— 所以 relation 指向它们才是真悬空。
    """
    OG = _mod("agent.tools.surveys._organize")
    results = [{
        "themes": [{"name": f"块{c}主题{i}", "statement": "", "notes": "",
                    "fact_ids": [f"c{c}f{i}_{j}" for j in range(15)],
                    "relations": [{"to": f"块{c}主题3", "note": "n"}]}
                   for i in range(4)],
        "narrative_arc": [], "gaps": [], "warnings": [],
    } for c in range(8)]

    out = OG._merge_chunk_results(results, max_themes=8)
    alive = {t["name"] for t in out["themes"]}
    assert alive == {f"块{c}主题0" for c in range(8)}, \
        f"轮转第 1 轮应该只留每块的主题0, 实得 {sorted(alive)}"
    for t in out["themes"]:
        for rel in t["relations"]:
            assert rel["to"] in alive, \
                f"主题 {t['name']} 的 relation 指向了已被截掉的主题 {rel['to']}"
    assert any("被截掉" in w for w in out["warnings"]), \
        f"剔了悬空 relation 却没有报出来: {out['warnings']}"


def test_merge_keeps_relations_to_never_defined_themes():
    """**不能**把"模型提到但没输出成主题"的那种 to 也删掉。

    真实产出里模型经常写 relation 时 `to` 写了个自己没落成主题的名字。那条
    关系本身是有信息的, 跟"指向被截断主题"是两回事 —— 10-07 第一版把两者
    一起删了, 把 `test_merge_chunk_results_unions_fact_ids_by_theme_name` 弄红了。
    """
    OG = _mod("agent.tools.surveys._organize")
    out = OG._merge_chunk_results([{
        "themes": [{"name": "A", "statement": "", "notes": "",
                    "fact_ids": ["f1"], "relations": [{"to": "B", "note": "n"}]}],
        "narrative_arc": [], "gaps": [], "warnings": [],
    }], max_themes=8)
    assert len(out["themes"][0]["relations"]) == 1, \
        "B 从来没被定义成主题, 这条关系不该被当成悬空删掉"


def test_runtime_print_does_not_promise_losslessness():
    """那句**直接对用户 print** 的话不许无条件宣称"一条 fact 都不丢"。

    10-06 实测就是被它骗的: 运行时对用户说"逐块归纳后按主题名合并 ——
    一条 fact 都不丢", 实际只有 23% 的 fact 进了地图。**主动的错误保证比
    沉默更糟** —— 它让用户以为安全, 根本不会去追查。

    用 ast 精确定位那个 print 调用: 全文件扫文本的话, 本函数自己的 docstring
    里那句「一条 fact 都不丢」会把自己判红(踩过)。
    """
    import ast

    src = _org_src()
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            continue
        seg = ast.get_source_segment(src, node) or ""
        if "逐块归纳" not in seg:
            continue
        assert "主题数超上限" in seg or "会丢" in seg, \
            f"运行时 print 仍然无条件宣称不丢: {seg[:200]}"
        return
    pytest.fail("找不到那句切块的运行时 print —— 它被删了? 要看一眼为什么")


def _code_only(path) -> str:
    """只保留**代码**: 剥掉注释和所有字符串字面量(含 docstring)。

    扫源码文本时必须走这个 —— 否则本文件 docstring 里为了说明问题而引用的
    旧代码片段会把自己判红(这条已经踩了两次: 一次是 test_no_test_disables,
    一次是 test_entry_test_does_not_pass)。
    """
    import io
    import tokenize

    src = pathlib.Path(path).read_text(encoding="utf-8")
    skip = {tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE,
            tokenize.INDENT, tokenize.DEDENT}
    return " ".join(tok.string for tok in
                    tokenize.generate_tokens(io.StringIO(src).readline)
                    if tok.type not in skip)


def _me_file():
    """本测试文件自己的路径 —— 下面两条用它来盯"不许再开后门"。"""
    return pathlib.Path(__file__).resolve().parent / "test_claim_layer.py"


def test_no_test_disables_the_truncation_path():
    """**任何测试都不许把截断分支变成死代码**。

    `organize_facts_into_themes(facts, max_themes=1000)` 就是干这个的: 它让
    `_merge_chunk_results` 里那段截断代码永远不执行, 于是"一条 fact 都不丢"
    在两种人为优停下必然成立(截断关掉 + mock 跨块同名 100% 合并), 测试给假
    声明发了绿灯, 而生产在丢 77%。这条盯的就是那个绿灯。
    """
    me = _me_file()
    assert "max_themes = 1000" not in _code_only(me), \
        "这个文件里又出现了 max_themes=1000 —— 截断分支被重新变成死代码"


def test_entry_test_does_not_pass_a_custom_max_themes():
    """入口那条测试必须走**生产默认**的 max_themes(=8), 不许自己传。"""
    # 去掉空格再比: _code_only 用空格拼 token, 源码里的 `f(a, b)` 会变成 `f ( a , b )`
    code = _code_only(_me_file()).replace(" ", "")
    assert "organize_facts_into_themes(facts)" in code, \
        "入口测试没有用默认 max_themes 调用"
    assert "organize_facts_into_themes(facts,max_themes" not in code, \
        "入口测试又自己传了 max_themes —— 生产默认是 8"


def test_merge_chunk_results_unions_fact_ids_by_theme_name():
    """同名主题必须并集 fact_ids —— 分块会把同一个主题劈到两块里。"""
    OG = _mod("agent.tools.surveys._organize")
    merged = OG._merge_chunk_results([
        {"themes": [{"name": "A", "statement": "s1", "fact_ids": ["f1", "f2"],
                     "notes": "", "relations": [{"to": "B", "note": "n1"}]}],
         "narrative_arc": ["a1"], "gaps": ["g1"], "warnings": ["w1"]},
        {"themes": [{"name": "A", "statement": "更长的陈述", "fact_ids": ["f2", "f3"],
                     "notes": "", "relations": [{"to": "B", "note": "n1"},
                                                {"to": "C", "note": "n2"}]}],
         "narrative_arc": ["a1", "a2"], "gaps": ["g2"], "warnings": ["w2"]},
    ], max_themes=8)
    assert [t["name"] for t in merged["themes"]] == ["A"]
    a = merged["themes"][0]
    assert a["fact_ids"] == ["f1", "f2", "f3"], f"同名主题没并集: {a['fact_ids']}"
    assert a["statement"] == "更长的陈述", "该保留信息量更大的陈述"
    assert len(a["relations"]) == 2, f"关系应去重后合并: {a['relations']}"
    assert merged["narrative_arc"] == ["a1", "a2"]
    assert merged["gaps"] == ["g1", "g2"]
    assert merged["warnings"] == ["w1", "w2"], "块的告警必须全部保留"


def test_merge_chunk_results_truncation_says_how_many_facts_lost():
    """砍主题 = 砍 fact, 属于有损操作 —— 必须说清丢了多少条, 不能只说"已截断"。"""
    OG = _mod("agent.tools.surveys._organize")
    merged = OG._merge_chunk_results([
        {"themes": [{"name": f"T{i}", "statement": "", "fact_ids": [f"f{i}"],
                     "notes": "", "relations": []} for i in range(5)],
         "narrative_arc": [], "gaps": [], "warnings": []},
    ], max_themes=2)
    assert len(merged["themes"]) == 2
    assert any("3 条 fact" in w for w in merged["warnings"]), \
        f"截断告警必须报出丢失的 fact 数: {merged['warnings']}"

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
