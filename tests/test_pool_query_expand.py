"""
tests/test_pool_query_expand.py — Phase A 的 LLM 查询扩展 (10-06)

**为什么这个文件存在**
10-06 实测: topic="neutral atom quantum computing", OpenAlex 全库 8 万+ 篇,
本地语料 247 篇里 0 篇相关, 最终候选池只有 **39 篇**。根因是三路外部召回
全部拿 topic 原词搜, 每路 k<=30。检索面就那么大, 跟领域实际大小无关。

这些测试锁的是"别把查询扩展改回单 query":
  - 扩展失败时**必须响亮**, 不能静默回退(静默的话用户看到的还是 39 篇,
    根本不知道本该有 6 条检索式没跑起来)
  - 检索式**必须是英文**(arxiv / OpenAlex 都是英文 API, 中文检索式等于没搜)
  - 扩展是**补充不是替换** —— 扩展挂了不能连原来能搜的都搜不到

测试取向: 用**真实跑出来的扩展结果**做表征测试(见 _REAL_* 常量), 不用编出来的
理想结构 —— 理想结构正好躲开真实数据里已经出现过的失败模式(模型爱在 queries
里混一条中文、或者把原主题原样抄一遍当"细分方向")。
"""
from __future__ import annotations

import importlib

import pytest

QE = importlib.import_module("agent.tools.surveys._pool._query_expand")
BUILD = importlib.import_module("agent.tools.surveys._pool._build_pool")
CFG = importlib.import_module("agent.tools.config")

# 真实跑出来的(topic="neutral atom quantum computing", MiniMax-M3, 23s)
_REAL_QUERIES = [
    "reconfigurable optical tweezer arrays neutral atom quantum computing",
    "Rydberg blockade two-qubit entangling gate fidelity neutral atom",
    "quantum error correction logical qubit surface code neutral atom array",
    "mid-circuit measurement non-destructive imaging qubit reset neutral atom",
    "coherent atom transport optical tweezer conveyor belt scalable neutral atom architecture",
    "Rydberg atom analog quantum simulation Ising spin Hamiltonian tweezer",
]
_REAL_DIRS = [
    "光学镊子阵列与可重构原子排布",
    "Rydberg阻塞与纠缠门",
    "中性原子量子纠错与逻辑比特",
    "中间电路测量与原子复用",
    "相干原子输运与大规模控制",
    "中性原子模拟量子计算",
]


def _fake_llm(payload, box=None):
    """返回一个可塞进 monkeypatch 的 _call_llm。box 用来抓 prompt。"""
    import json as _json

    def _fn(prompt, **kw):
        if box is not None:
            box["prompt"] = prompt
            box["kw"] = kw
        return payload if isinstance(payload, str) else _json.dumps(payload, ensure_ascii=False)
    return _fn


# ============================================================
# 检索式筛选: 英文 / 重复 / 原样抄主题
# ============================================================

def test_pick_queries_drops_chinese_only():
    """纯中文检索式必须被扔掉 —— 喂给 arxiv/OpenAlex 等于没搜。

    这正是中文 topic 候选池 0 篇的成因(见 survey_generate 里的中文提示),
    扩展自己不能再犯一遍。
    """
    parsed = {"sub_directions": [], "queries": [
        {"q": "中性原子量子计算"},          # 纯中文
        {"q": "neutral atom quantum computing"},   # 英文, 保留
    ]}
    got, _ = QE._pick_queries(parsed, 6)
    assert got == ["neutral atom quantum computing"]


def test_pick_queries_drops_single_latin_token():
    """只有一个拉丁词("quantum")的不算检索式 —— 召回面等于整个量子领域。"""
    parsed = {"queries": [{"q": "quantum"}, {"q": "Rydberg blockade gate"}]}
    got, _ = QE._pick_queries(parsed, 6)
    assert got == ["Rydberg blockade gate"]


def test_pick_queries_drops_digits_and_one_letter_tokens():
    """`a1 b2 c3` 这种每个 token 都只有 1 个拉丁字母的也要挡掉。

    规则是 `[A-Za-z]{2,}` —— 连续 2 个以上拉丁字母才算一个真词。
    之前写成 `a1 b2 c3 d4` 当"应该通过"的样例, 实际它一个真词都没有,
    反而证明规则是对的: 测��本身写错了。
    """
    parsed = {"queries": [{"q": "a1 b2 c3 d4"}, {"q": "Rydberg blockade"}]}
    got, _ = QE._pick_queries(parsed, 6)
    assert got == ["Rydberg blockade"]


def test_pick_queries_accepts_bare_string_array():
    """模型偶尔直接给字符串数组而不是对象数组 —— 容错, 不该整批丢。"""
    parsed = {"queries": ["Rydberg blockade gate neutral atom", "erasure conversion atom"]}
    got, _ = QE._pick_queries(parsed, 6)
    assert got == ["Rydberg blockade gate neutral atom", "erasure conversion atom"]


def test_pick_queries_dedups():
    parsed = {"queries": [{"q": "Rydberg gate neutral atom"}] * 3}
    got, _ = QE._pick_queries(parsed, 6)
    assert got == ["Rydberg gate neutral atom"], "重复检索式会让同一批结果反复进池"


def test_pick_queries_respects_n_limit():
    parsed = {"queries": [{"q": f"query number {i} neutral atom"} for i in range(12)]}
    got, _ = QE._pick_queries(parsed, 4)
    assert len(got) == 4


def test_pick_queries_keeps_real_output_intact():
    """表征测试: 真实跑出来的 6 条必须一条不少地留下。

    这是防回归的关键 —— 筛选规则加严到把真结果也筛掉, 这条会红。
    """
    parsed = {"queries": [{"q": q} for q in _REAL_QUERIES]}
    got, _ = QE._pick_queries(parsed, 6)
    assert got == _REAL_QUERIES


# ============================================================
# expand_queries 的失败与成功路径
# ============================================================

def test_expand_ok_returns_queries_and_dirs(monkeypatch):
    box = {}
    payload = {
        "sub_directions": [{"name": n, "why": "w"} for n in _REAL_DIRS],
        "queries": [{"direction": n, "q": q}
                    for n, q in zip(_REAL_DIRS, _REAL_QUERIES)],
    }
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fake_llm(payload, box), raising=False)
    r = QE.expand_queries("neutral atom quantum computing", n=6)
    assert r["ok"] is True
    assert r["queries"] == _REAL_QUERIES
    assert [d["name"] for d in r["sub_directions"]] == _REAL_DIRS
    # prompt 必须带主题, 而且要**明确要求检索式是英文**
    assert "neutral atom quantum computing" in box["prompt"]
    assert "必须是英文" in box["prompt"], \
        "prompt 必须要求检索式是英文 —— arxiv/OpenAlex 都是英文 API, \
给中文等于没搜(中文 topic 召回 0 篇就是这么来的)"


def test_expand_fails_loudly_on_exception(monkeypatch):
    """LLM 挂了必须 ok=False + 说清原因, **不能抛出去把 Phase A 带崩**。"""
    def _boom(*a, **kw):
        raise RuntimeError("socket closed")
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _boom, raising=False)
    r = QE.expand_queries("neutral atom quantum computing", n=6)
    assert r["ok"] is False
    assert r["queries"] == [], "失败时不能留下半截检索式去污染召回"
    assert "socket closed" in r["error"]


def test_expand_fails_on_empty_return(monkeypatch):
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", lambda p, **kw: "", raising=False)
    r = QE.expand_queries("t", n=6)
    assert r["ok"] is False and r["error"]


def test_expand_fails_on_unparseable(monkeypatch):
    """返回一大段没有 JSON 的文本(比如全被 think 吃掉) —— 报出来, 别静默。"""
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", lambda p, **kw: "<think>想了很久</think>",
                        raising=False)
    r = QE.expand_queries("t", n=6)
    assert r["ok"] is False and r["error"]


def test_expand_fails_when_only_chinese_queries(monkeypatch):
    """模型只给了中文检索式 -> 必须 ok=False。

    静默接受的话, 后面 batch 召回拿着中文去搜 arxiv, 一篇都回不来, 而用户
    看到的是"扩展跑了, 但召回没涨", 根本查不到是哪一步坏的。
    """
    payload = {"sub_directions": [{"name": "甲", "why": "w"}],
               "queries": [{"q": "中性原子 量子计算"}, {"q": "光镊 阵列"}]}
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fake_llm(payload), raising=False)
    r = QE.expand_queries("neutral atom quantum computing", n=6)
    assert r["ok"] is False
    assert "英文" in r["error"]


def test_expand_empty_topic_short_circuits(monkeypatch):
    """空主题不该花 LLM 调用。"""
    def _boom(*a, **kw):
        raise AssertionError("空主题时不该调 LLM")
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _boom, raising=False)
    r = QE.expand_queries("   ", n=6)
    assert r["ok"] is False


def test_expand_strips_think_and_fence(monkeypatch):
    """<think> 包裹 + ```json 围栏是真实见过的输出形态, 必须能剥。

    **think 块里必须带花括号** —— 这才是剥离成为关键路径的原因:
    后面找 JSON 用的是贪婪的 `\\{.*\\}`, 如果 think 里有 `{`, 不剥离的话
    正则会从 think 里那个 `{` 一路吃到最后的 `}`, 拼出一段非法 JSON 直接解析
    失败。think 里没有花括号时, 贪婪匹配也能歪打正着找到正确的 JSON, 剥离与否
    都能过 —— 那种用例测不到任何东西。
    """
    import json as _json
    fence = chr(96) * 3
    real = _json.dumps(
        {"sub_directions": [{"name": "甲", "why": "w"}],
         "queries": [{"q": "Rydberg gate neutral atom"}]}, ensure_ascii=False)
    raw = ('<think> 假设 {"x": 1} 这样</think>' + chr(10)
           + fence + "json" + chr(10) + real + chr(10) + fence)
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", lambda p, **kw: raw, raising=False)
    r = QE.expand_queries("t", n=6)
    assert r["ok"] is True
    assert r["queries"] == ["Rydberg gate neutral atom"]


# ============================================================
# 接线: build_status 必须把检索式和细分方向交出去
# ============================================================

def test_build_status_carries_query_expansion():
    """build_status 里必须有 query_expansion, 且**值真的来自 qexp**。

    只断言 `"sub_directions"` 这个字符串出现过是不够的: 把它改成写死 `[]`
    键还在, 测试照样绿, 而下游拿到的是空骨架。断言必须落在**取值表达式**上。
    """
    import pathlib
    src = pathlib.Path(BUILD.__file__).read_text(encoding="utf-8")
    assert '"query_expansion"' in src
    assert '"queries": _qexp_queries' in src, "检索式必须原样交出去"
    assert '"sub_directions": qexp.get("sub_directions") or []' in src, \
        "细分方向必须真的从 qexp 取, 不能写死空列表"


def test_build_status_keeps_backward_compat_fields():
    """老消费方还在读 arxiv_status / external_status / audit_state, 不能删。"""
    src = __import__("pathlib").Path(
        BUILD.__file__).read_text(encoding="utf-8")
    for k in ('"arxiv_status"', '"external_status"', '"audit_state"'):
        assert k in src, f"build_status 丢了向后兼容字段 {k}"


def test_expanded_sources_have_sort_priority():
    """三个新增 source 必须有**各不相同、且排在单 query 源之后**的优先级。

    只断言"键存在"是不够的: 三个值都写成 9 时代码照样跑, 扩展召回的论文却会
    跟没排上号的一起被顶到 max_pool 之外。数字本身才是护栏。
    """
    import re as _re
    import pathlib
    src = pathlib.Path(BUILD.__file__).read_text(encoding="utf-8")
    prio = {}
    for s in ("aps_batch_expanded_query", "iop_batch_expanded_query",
              "arxiv_batch_expanded_query", "openalex_aps_api",
              "openalex_iop_api", "arxiv_api_fallback"):
        m = _re.search(rf'"{s}":\s*(\d+)', src)
        assert m, f"{s} 没在 source_priority 里给数值"
        prio[s] = int(m.group(1))
    exp = [prio["aps_batch_expanded_query"], prio["iop_batch_expanded_query"],
           prio["arxiv_batch_expanded_query"]]
    assert len(set(exp)) == 3, f"三个扩展源的优先级必须互不相同, 实际 {exp}"
    for s, v in (("aps_batch_expanded_query", prio["aps_batch_expanded_query"]),
                 ("arxiv_batch_expanded_query", prio["arxiv_batch_expanded_query"])):
        assert v > prio["openalex_aps_api"], (
            f"{s}({v}) 必须排在单 query 的 openalex_aps_api"
            f"({prio['openalex_aps_api']})**之后** —— 同等相关性下原词召回更贴题")


def test_single_query_recall_uses_raw_topic_only():
    """单 query 三路(3a/3b/3c)必须继续用原 topic —— 扩展是**补充**。

    把它们也换成扩展词等于放弃"扩展挂了也能照常跑"这条退路。
    """
    src = __import__("pathlib").Path(
        BUILD.__file__).read_text(encoding="utf-8")
    assert "_openalex_aps_search(topic," in src
    assert "_openalex_iop_search(topic," in src
    assert "query=topic" in src


def test_batch_pass_never_drops_other_sources(monkeypatch):
    """一路 batch 召回抛异常, 另外两路的结果必须还在。

    单源失败就整个放弃, 等于把"扩展召回"变成"扩展把主路径也拖垮"。
    """
    import agent.tools.surveys._pool._build_pool as bp

    def _aps_batch(queries, k):
        return [{"paper_id": "10.1/aps1", "title": "A", "journal": "PRL"}]

    def _iop_batch(queries, k):
        raise RuntimeError("IOP down")

    def _arxiv_batch(queries, k):
        return [{"paper_id": "10.1/arxiv1", "title": "B", "journal": "arxiv"}]

    monkeypatch.setattr(bp, "_openalex_aps_search_batch", _aps_batch)
    monkeypatch.setattr(bp, "_openalex_iop_search_batch", _iop_batch)
    monkeypatch.setattr(bp, "_arxiv_search_batch", _arxiv_batch)
    # 审查和后处理都打成空操作 —— 这里只验 batch 段本身
    monkeypatch.setattr(bp, "_audit_pool_with_retry",
                        lambda **kw: {"kept_dict": kw["pool_dict"], "summary": {}})
    monkeypatch.setattr(bp, "get_relevance_threshold", lambda: 0)
    # 本地召回和单 query 外部源全部清空, 逼着代码只走 batch 段
    # 注意: list_summary_items 是**函数内 import** 的, 只能打在源模块上
    import agent.tools.summary as summary_mod
    monkeypatch.setattr(summary_mod, "list_summary_items", lambda **kw: [])
    monkeypatch.setattr(bp, "_openalex_aps_search", lambda q, k=0: [])
    monkeypatch.setattr(bp, "_openalex_iop_search", lambda q, k=0: [])
    monkeypatch.setattr(bp, "get_min_pool_papers", lambda: 10**9)
    monkeypatch.setattr(bp, "expand_queries",
                        lambda topic, n=6: {"queries": ["q1 neutral atom"],
                                             "sub_directions": [], "ok": True,
                                             "error": ""})
    # chroma 在测试环境没有索引, 直接让 get_vector_db 返 None
    import sys
    vdb = sys.modules.get("agent.vector_db")
    if vdb is None:
        vdb = type(sys)("agent.vector_db")
        sys.modules["agent.vector_db"] = vdb
    monkeypatch.setattr(vdb, "get_vector_db", lambda: None, raising=False)

    pool, st = bp._build_candidate_pool("neutral atom quantum computing", max_pool=50)
    srcs = {p.get("_source") for p in pool}
    assert "aps_batch_expanded_query" in srcs, "APS 那一路的结果丢了"
    assert "arxiv_batch_expanded_query" in srcs, "arxiv 那一路的结果也丢了"
    assert st["external_status"]["iop_batch"].startswith("exception:"), \
        "IOP 挂掉必须如实记进 external_status, 不能悄悄消失"


# ============================================================
# 配置开关
# ============================================================

def test_query_expand_switch_off_returns_zero(monkeypatch):
    """QAGENT_SURVEY_QUERY_EXPAND=0 时 n=0, 调用方据此跳过。"""
    monkeypatch.setattr(CFG, "_SURVEY_QUERY_EXPAND_ENABLED", False)
    assert CFG.get_query_expand_n() == 0


def test_query_expand_n_respects_bound():
    """n 必须夹在 1..12 之间 —— 太多会让召回变噪, 太少又拆不出方向。

    这条只能**读源码里的字面量**: 上下界是在 import 时由
    `_resolve_survey_expand_int` 算成模块级常量的, 之后再改源码, 已 import 的
    值不会跟着变 —— 从 `get_query_expand_n()` 上断言是测不到的(踩过)。
    """
    import pathlib
    src = pathlib.Path(CFG.__file__).read_text(encoding="utf-8")
    assert '"QAGENT_SURVEY_QUERY_EXPAND_N", 6, 1, 12)' in src, \
        "n 的合法区间必须是 (1, 12) —— 放开会灌进一堆噪点, 每篇还要过 LLM 审查"
    assert '"QAGENT_SURVEY_QUERY_EXPAND_K", 4, 1, 20)' in src, \
        "k 的合法区间必须是 (1, 20)"


def test_query_expand_resolve_out_of_range_falls_back_to_default(monkeypatch):
    """越界的 env 值**回落到默认值**, 不是夹到边界。

    `_resolve_survey_expand_int` 的语义是 `lo <= v <= hi ? v : default` ——
    越界就当这个变量没设。比夹取更安全: 用户写 `N=999` 时得到默认的 6,
    而不是"我以为会少拆一点结果拆了 12 个方向"。

    (一开始我把这里写成"夹到边界", 断言 999→12, 实际是 999→6。
     改的是测试, 不是实现 —— 实现的行为是对的。)
    """
    for raw, want in (("999", 6), ("0", 6), ("abc", 6), ("6", 6), ("10", 10)):
        monkeypatch.setenv("QAGENT_SURVEY_QUERY_EXPAND_N", raw)
        got = CFG._resolve_survey_expand_int(
            "QAGENT_SURVEY_QUERY_EXPAND_N", 6, 1, 12)
        assert got == want, f"env={raw!r} 应回落到 {want}, 实际 {got}"
    monkeypatch.delenv("QAGENT_SURVEY_QUERY_EXPAND_N", raising=False)
    assert CFG._resolve_survey_expand_int(
        "QAGENT_SURVEY_QUERY_EXPAND_N", 6, 1, 12) == 6


@pytest.mark.parametrize("n,k", [(6, 4), (8, 3), (1, 1)])
def test_expand_accepts_varargs(monkeypatch, n, k):
    box = {}
    payload = {"sub_directions": [{"name": "甲", "why": "w"}],
               "queries": [{"q": f"neutral atom query {i}"} for i in range(n)]}
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fake_llm(payload, box), raising=False)
    r = QE.expand_queries("neutral atom quantum computing", n=n)
    assert r["ok"] is True
    assert str(n) in box["prompt"], "prompt 里要告诉模型要拆几个方向"
