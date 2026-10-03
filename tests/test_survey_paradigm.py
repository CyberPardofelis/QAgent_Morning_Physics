"""
tests/test_survey_paradigm.py — 范式层的单元测试 (10-03)

三块都是"端到端跑一遍才知道坏没坏、但其实不该依赖 LLM"的地方:
  1. `_salvage_paradigms` —— 输出被 max_tokens 截断时抢救已完整的部分
  2. `_merge_same_paper_within_theme` —— 退回旧路径时的兜底去重
  3. `assign` 的归属门槛(margin / 最少种子数 / 编码不可用时降级)

第 3 块用假向量驱动, 目的是把"质心 + 门槛"的逻辑**钉死**:
端到端实测里同一份数据跑四次覆盖 37%/51%/49%/37%, 波动很大,
所以门槛的行为必须能被离线断言, 不能只靠跑一次看数字。
"""
from __future__ import annotations

import json

import sys

import pytest


def _P():
    """**调用时**取当前的 _paradigm 模块。

    10-03 踩过: 同套件里有测试 reload 了模块, 于是模块顶层 import 到的 PA/OG
    变成了旧对象 —— 而 condense_facts_by_paradigm 内部的函数级 import 取的是
    reload 后的新对象。补丁打在旧对象上完全无效, 于是测试真调了一次 LLM(58 秒)。
    所以这里和补丁处都走 sys.modules 现取。
    """
    import importlib

    name = "agent.tools.surveys._paradigm"
    return sys.modules.get(name) or importlib.import_module(name)


def _O():
    import importlib

    name = "agent.tools.surveys._organize"
    return sys.modules.get(name) or importlib.import_module(name)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """任何测试都**不许**真发 LLM 请求。

    10-03 踩过: 补丁打在了旧的模块对象上(同套件里有测试 reload 了 _paradigm),
    而 `condense_facts_by_paradigm` 的函数内 import 取的是新对象, 补丁失效 ->
    真的调了 MiniMax 58 秒, 还把 4 条 fact 归成 1 个范式(真实范式就一个)。
    测试打网络 = CI 不稳 + 离线跑不了 + 烧配额, 一律在这里堵死。
    """
    def _boom(*a, **k):
        raise AssertionError("测试里不许真调 LLM —— 补丁没生效?")

    monkeypatch.setattr("agent.tools.llm._call_llm", _boom)


def _patch_paradigm(monkeypatch, name, value):
    """按**调用时**的模块对象打补丁。

    `condense_facts_by_paradigm` 内部是 `from ._paradigm import ...`，
    同套件里若有测试 reload 过模块，模块顶部的 `PA` 就成了旧对象，补丁无效。
    这里每次都从 sys.modules 现取，和函数内 import 拿到的是同一个对象。
    """
    monkeypatch.setattr(_P(), name, value)


# ============================================================
# 1. 截断抢救
# ============================================================

def _para(name, kws, stmt="说明"):
    return {"name": name, "statement": stmt, "keywords": list(kws)}


def test_salvage_recovers_complete_paradigms_from_truncated_output():
    """
    10-03 实测: 7 个主题里 2 个的范式调用把结构**完全正确**地写完了大半,
    只是在 max_tokens 处被截断(尾部停在某个 statement 中间)。
    整块丢掉 = 这个主题白跑 = 退回旧路径 = 暴露出"同一篇连引 8 次"。
    """
    full = {"paradigms": [
        _para("路线甲", ["A1", "A2"]),
        _para("路线乙", ["B1", "B2"]),
        _para("路线丙", ["C1"]),
    ]}
    s = json.dumps(full, ensure_ascii=False)
    truncated = s[:s.rindex('{"name": "路线丙"')] + '{"name": "路线丙", "state'
    got = _P()._salvage_paradigms(truncated)
    names = [p["name"] for p in got]
    assert "路线甲" in names and "路线乙" in names
    assert "路线丙" not in names, "残缺对象不该被收进来"


def test_salvage_returns_none_when_first_object_is_broken():
    assert _P()._salvage_paradigms('{"paradigms": [{"name": "x", "st') is None


def test_salvage_handles_wrapped_in_themes():
    """模型有时会多包一层 {"themes":[{"paradigms":[...]}]}。"""
    got = _P()._salvage_paradigms(
        json.dumps({"themes": [{"paradigms": [_para("路线", ["K1", "K2"])]}]},
                   ensure_ascii=False))
    assert got and got[0]["name"] == "路线"


def test_salvage_ignores_objects_without_keywords():
    got = _P()._salvage_paradigms(json.dumps(
        {"paradigms": [{"name": "没关键词", "statement": "x"}]}, ensure_ascii=False))
    assert got is None


def test_parse_falls_back_to_salvage():
    """_parse 严格解析失败时要走抢救, 而不是直接返回 None。"""
    good = json.dumps({"paradigms": [_para("甲", ["A"]), _para("乙", ["B"])]},
                      ensure_ascii=False)
    broken = good[:good.rindex('{"name": "乙"')] + '{"name": "乙", "x'
    got = _P()._parse(broken, "某主题")
    assert got and [p["name"] for p in got] == ["甲"]


def test_parse_handles_clean_output_without_salvage():
    raw = ('<think>先想想</think>{"paradigms": ['
           + json.dumps(_para("甲", ["A", "B"]), ensure_ascii=False) + "]}")
    got = _P()._parse(raw, "某主题")
    assert got and got[0]["keywords"] == ["A", "B"]


# ============================================================
# 2. 兜底去重: 同主题内同一篇论文只出一条命题
# ============================================================

def _claim(cid, pids, statement="短", n_facts=1):
    return {"claim_id": cid, "statement": statement, "paper_ids": list(pids),
            "fact_ids": [f"{cid}-f{i}" for i in range(n_facts)], "n_facts": n_facts}


def test_merge_same_paper_collapses_duplicate_claims():
    """
    10-03 实测: 范式层失败退回旧路线时, 「二维网络变分优化」4262 字里同一篇
    被连引 8 次 —— 那些 fact 的 subject 各不相同, 所以
    `_dedupe_same_paper_same_subject` 挡不住, 得在命题层再挡一道。
    """
    claims = [
        _claim("c1", ["P1"], "关于 X 的细节一"),
        _claim("c2", ["P1"], "关于 X 的细节二"),
        _claim("c3", ["P1"], "关于 X 的细节三"),
        _claim("c4", ["P2"], "另一篇"),
    ]
    got = _O()._merge_same_paper_within_theme(claims)
    assert [c["paper_ids"] for c in got] == [["P1"], ["P2"]]
    assert got[0]["fact_ids"] == ["c1-f0", "c2-f0", "c3-f0"], "事实不能丢"
    assert got[0]["n_facts"] == 3


def test_merge_keeps_the_longer_statement():
    got = _O()._merge_same_paper_within_theme(
        [_claim("c1", ["P1"], "短"), _claim("c2", ["P1"], "明显更长的那句")])
    assert got[0]["statement"] == "明显更长的那句"


def test_merge_never_touches_multi_paper_claims():
    """多篇支撑的命题已经是"综合", 天然不会被重复引, 必须原样保留。"""
    claims = [_claim("c1", ["P1", "P2"]), _claim("c2", ["P1", "P2"])]
    assert _O()._merge_same_paper_within_theme(claims) == claims


def test_merge_preserves_order():
    claims = [_claim("c1", ["P1"]), _claim("c2", ["P2"]), _claim("c3", ["P1"]),
              _claim("c4", ["P3"])]
    got = _O()._merge_same_paper_within_theme(claims)
    assert [c["claim_id"] for c in got] == ["c1", "c2", "c4"]


# ============================================================
# 3. 归属: 播种 -> 质心 -> margin
# ============================================================

def _fact(fid, pid, text):
    return {"fact_id": fid, "paper_id": pid, "subject": text, "claim": text}


def _vec(groups, idx_groups, n):
    """给每条 fact 一个单位向量, 同组的方向相同, 组间正交。"""
    import numpy as np
    v = np.zeros((n, len(groups)), dtype="float32")
    for i, g in enumerate(idx_groups):
        v[i, groups.index(g)] = 1.0
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_assign_groups_facts_by_centroid(monkeypatch):
    facts = [
        _fact("f1", "P1", "MPS 矩阵乘积态"),
        _fact("f2", "P2", "MPS 一维链"),
        _fact("f3", "P3", "PEPS 二维格点"),
        _fact("f4", "P4", "PEPS 面积律"),
    ]
    paras = [{"name": "MPS", "statement": "", "keywords": ["MPS"]},
             {"name": "PEPS", "statement": "", "keywords": ["PEPS"]}]
    vec = _vec(["a", "b"], ["a", "a", "b", "b"], 4)
    _patch_paradigm(monkeypatch, "encode", lambda t, **k: vec)
    got = _P().assign(facts, paras)
    assert {f["fact_id"] for f in got["MPS"]} == {"f1", "f2"}
    assert {f["fact_id"] for f in got["PEPS"]} == {"f3", "f4"}


def test_assign_drops_ambiguous_facts(monkeypatch):
    """
    领先次优不足 margin 的 fact **不归属** —— 宁可少写也不能挂错范式,
    挂错的代价是"这句话挂着不相干的引用"。
    """
    import numpy as np
    facts = [_fact(f"f{i}", f"P{i}", "MPS") for i in range(1, 5)]
    paras = [{"name": "A", "statement": "", "keywords": ["MPS"]},
             {"name": "B", "statement": "", "keywords": ["MPS"]}]
    # 前两条指向 A, 后两条指向 B, 但各自与另一个几乎一样近 -> 全部不归属
    vec = np.array([[1.0, 0.0], [1.0, 0.0], [0.5, 0.87], [0.5, 0.87]], dtype="float32")
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    _patch_paradigm(monkeypatch, "encode", lambda t, **k: vec)
    got = _P().assign(facts, paras, margin=0.30)
    assert got == {}, "分不出来的就该一个都不归属"


def test_assign_skips_paradigm_with_too_few_seeds(monkeypatch):
    """种子只有 1 条时质心就是它自己, 会被自己的范式全吸走 —— 宁可不建。"""
    facts = [_fact("f1", "P1", "MPS"), _fact("f2", "P2", "MPS"),
             _fact("f3", "P3", "PEPS"), _fact("f4", "P4", "PEPS"),
             _fact("f5", "P5", "MERA")]
    paras = [{"name": "MPS", "statement": "", "keywords": ["MPS", "MPS"]},
             {"name": "PEPS", "statement": "", "keywords": ["PEPS", "PEPS"]},
             {"name": "MERA", "statement": "", "keywords": ["MERA", "MERA"]}]
    vec = _vec(["a", "b", "c"], ["a", "a", "b", "b", "c"], 5)
    _patch_paradigm(monkeypatch, "encode", lambda t, **k: vec)
    got = _P().assign(facts, paras)
    assert "MERA" not in got, "只有 1 条种子的范式不该参与归属"
    assert {f["fact_id"] for f in got.get("MPS", [])} == {"f1", "f2"}


def test_assign_falls_back_to_seeds_when_encode_unavailable(monkeypatch):
    """编码失败(模型没装/子进程崩了)必须退回关键词播种, 不能整体丢内容。"""
    facts = [_fact("f1", "P1", "MPS 一维"), _fact("f2", "P2", "MPS 一维链")]
    paras = [{"name": "MPS", "statement": "", "keywords": ["MPS", "一维"]},
             {"name": "PEPS", "statement": "", "keywords": ["PEPS", "二维"]}]
    _patch_paradigm(monkeypatch, "encode", lambda t, **k: None)
    got = _P().assign(facts, paras)
    assert {f["fact_id"] for f in got["MPS"]} == {"f1", "f2"}
    assert "PEPS" not in got, "没有 fact 命中的范式不该出现"


def test_assign_empty_inputs():
    assert _P().assign([], []) == {}
    assert _P().assign([_fact("f", "P", "x")], []) == {}


# ============================================================
# 4. 接线护栏
# ============================================================

def test_condense_by_paradigm_returns_empty_when_no_paradigms(monkeypatch):
    """范式层任何一步失败都要返回空, 让调用方退回旧路线 —— 不能抛。"""
    _patch_paradigm(monkeypatch, "build_paradigms", lambda *a, **k: {})
    facts = [{"fact_id": "f1", "paper_id": "P1", "claim": "c"}]
    tmap = {"themes": [{"name": "T", "fact_ids": ["f1"]}]}
    assert _O().condense_facts_by_paradigm(facts, tmap) == []


def test_condense_by_paradigm_makes_one_claim_per_paradigm(monkeypatch):
    # 关键词给**两个**且 fact 里都出现: 不依赖 SEED_MIN_HITS 这个模块全局,
    # 阈值是 1 还是 2 都能过(全量跑时若有别的测试动过它, 单词关键词的版本会翻车)。
    _patch_paradigm(monkeypatch, "build_paradigms", lambda *a, **k: {
        "T": [{"name": "MPS", "statement": "链状表示", "keywords": ["MPS", "一维"]},
              {"name": "PEPS", "statement": "二维推广", "keywords": ["PEPS", "二维"]}]})
    facts = [
        {"fact_id": "f1", "paper_id": "P1", "claim": "c1", "subject": "MPS 一维链"},
        {"fact_id": "f2", "paper_id": "P2", "claim": "c2", "subject": "MPS 一维基态"},
        {"fact_id": "f3", "paper_id": "P3", "claim": "c3", "subject": "PEPS 二维格点"},
        {"fact_id": "f4", "paper_id": "P4", "claim": "c4", "subject": "PEPS 二维面积律"},
    ]
    tmap = {"themes": [{"name": "T", "fact_ids": ["f1", "f2", "f3", "f4"]}]}
    _patch_paradigm(monkeypatch, "encode", lambda t, **k: None)   # 退回播种
    got = _O().condense_facts_by_paradigm(facts, tmap)
    assert len(got) == 2, f"应一类范式一条命题, 实际 {len(got)} 条: {got}"
    mps = next(c for c in got if "MPS" in c["statement"])
    assert mps["paper_ids"] == ["P1", "P2"], "一条范式 = 一条跨论文命题"
    assert set(mps["fact_ids"]) == {"f1", "f2"}


def test_paradigm_statement_is_truncated():
    """
    10-03: 思维链长度随输入复杂度增长 —— statement 146 字时成文正常,
    换成 2-3 句长 statement 就全被 think 吃光(连续 3 次 0 正文)。
    所以这里截到 120 字: 范式的身份由 name 承担, statement 只需交代机制和局限。
    """
    import inspect as _i
    src = _i.getsource(_O().condense_facts_by_paradigm)
    assert "[:120]" in src, "statement 必须截断, 否则成文会被 think 吃光"


def test_organize_phase_uses_paradigm_layer():
    import inspect as _i
    src = _i.getsource(_O().condense_facts_by_paradigm)
    assert "build_paradigms" in src and "assign" in src
