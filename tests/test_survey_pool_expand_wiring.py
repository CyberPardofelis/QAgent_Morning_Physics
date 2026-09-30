"""
test_survey_pool_expand_wiring.py — 引文图扩展接进 _build_candidate_pool 的行为 (9-30)

只测"接线"这一层: 扩展在什么条件下跑、跑多少、失败怎么办。
扩展算法本身在 test_openalex_graph_expand.py 里测。
"""
from __future__ import annotations

import importlib
import sys

import pytest


def _mod(name: str):
    if name not in sys.modules:
        importlib.import_module(name)
    return sys.modules[name]


BP = _mod("agent.tools.surveys._pool._build_pool")
EX = _mod("agent.tools.surveys._pool._expand")
CFG = _mod("agent.tools.config")


def _oa_entry(wid, title="Paper", abstract="quantum error correction", cited=10,
              source="openalex_aps_api"):
    return {
        "paper_id": f"10.1/{wid.lower()}", "title": title, "year": "2020",
        "journal": "Phys Rev Lett", "abstract": abstract,
        "_openalex_id": wid, "cited_by_count": cited, "type": "article",
        "_source": source,
    }


@pytest.fixture
def wired(monkeypatch):
    """把 _build_candidate_pool 的四路召回/审查全部换成假实现, 只留扩展这一路可控。

    ⚠️ arxiv 也必须 mock —— 真实实现是 import arxiv 后直接打网络,
       不 mock 的话这个单测会真联网 (实测拖到 40 秒, 而且结果不确定)。
    """
    import arxiv as real_arxiv
    import agent.tools.summary as summary_mod
    import agent.vector_db as vdb_mod

    calls = {"expand": 0, "cfg": None}

    monkeypatch.setattr(summary_mod, "list_summary_items", lambda limit=0: [])
    monkeypatch.setattr(vdb_mod, "get_vector_db", lambda: None)
    monkeypatch.setattr(BP, "_audit_pool_with_retry", lambda **kw: {
        "kept_dict": dict(kw["pool_dict"]),
        "summary": {"kept": len(kw["pool_dict"]), "rounds": 0},
    })

    class _EmptyResults:
        def __iter__(self):
            return iter(())

    monkeypatch.setattr(
        real_arxiv, "Client",
        lambda *a, **k: type("C", (), {"results": lambda self, s: _EmptyResults()})(),
    )

    # 池子初始为空 -> APS/IOP 一定会触发, 正好当种子来源
    monkeypatch.setattr(BP, "_openalex_aps_search",
                        lambda q, k=5: [_oa_entry("W1"), _oa_entry("W2")])
    monkeypatch.setattr(BP, "_openalex_iop_search", lambda q, k=5: [])

    def _cfg(**over):
        base = {"enabled": True, "rounds": 1, "breadth": 5,
                "max_add": 3, "max_requests": 6, "reviews": True}
        base.update(over)
        calls["cfg"] = base
        monkeypatch.setattr(BP, "get_survey_expand", lambda: dict(base))

    _cfg()
    calls["_set_cfg"] = _cfg

    def _expand(seeds, config=None, topic=""):
        calls["expand"] += 1
        # 真实管线里 source 由 _openalex_work_to_pool_entry 打:
        #   f"openalex_{direction}_api" -> openalex_forward_api / openalex_backward_api
        entries = [
            _oa_entry(f"WF{i}", cited=i, source="openalex_forward_api") for i in range(5)
        ] + [
            _oa_entry(f"WB{i}", cited=100 + i, source="openalex_backward_api") for i in range(5)
        ]
        return EX.ExpansionResult(
            entries=entries,
            stats={"rounds": 1, "requests": 2, "dropped": {"seen": 1},
                   "stopped_reason": "达到 max_rounds 上限(收敛)", "total": len(entries)},
        )

    monkeypatch.setattr(BP, "expand_from_seeds", _expand)
    return calls


# ============================================================
# 开关
# ============================================================

def test_disabled_by_env(wired, monkeypatch):
    wired["_set_cfg"](enabled=False)
    pool, st = BP._build_candidate_pool("quantum error correction", max_pool=10)
    assert st["external_status"]["expand"] == "disabled"
    assert wired["expand"] == 0, "关了就不该调扩展"


def test_max_add_zero_skips(wired):
    wired["_set_cfg"](max_add=0)
    _, st = BP._build_candidate_pool("quantum error correction", max_pool=10)
    assert st["external_status"]["expand"] == "max_add=0"
    assert wired["expand"] == 0


# ============================================================
# 种子
# ============================================================

def test_no_openalex_seed_skips(wired, monkeypatch):
    """arxiv 那路进来的条目没有 OpenAlex id, 全靠 APS/IOP 才有种子。"""
    monkeypatch.setattr(BP, "_openalex_aps_search", lambda q, k=5: [])
    monkeypatch.setattr(BP, "_openalex_iop_search", lambda q, k=5: [])
    _, st = BP._build_candidate_pool("quantum error correction", max_pool=10)
    assert st["external_status"]["expand"] == "no_seed"
    assert wired["expand"] == 0


# ============================================================
# 正常路径
# ============================================================

def test_expansion_runs_and_reports(wired):
    pool, st = BP._build_candidate_pool("quantum error correction", max_pool=50)
    es = st["external_status"]["expand"]
    assert es.startswith("success"), es
    assert wired["expand"] == 1


def test_injection_capped_at_max_add(wired):
    """
    关键约束: 扩展可能出几百篇, 但下游 audit 是逐篇 LLM,
    全塞进去瓶颈就从 API 配额变成 token 成本。所以必须限量注入。
    """
    wired["_set_cfg"](max_add=3)
    pool, st = BP._build_candidate_pool("quantum error correction", max_pool=50)
    n_added = int(st["external_status"]["expand"].split("+")[1].split("/")[0])
    assert n_added <= 3, f"注入了 {n_added} 篇, 超过 max_add=3"


def test_expanded_entries_land_in_pool(wired):
    wired["_set_cfg"](max_add=5)
    pool, st = BP._build_candidate_pool("quantum error correction", max_pool=50)
    srcs = {e.get("_source") for e in pool}
    assert "openalex_forward_api" in srcs or "openalex_backward_api" in srcs, (
        f"扩展来的条目没进池, 现有 source: {srcs}")


def test_config_values_forwarded(wired, monkeypatch):
    """rounds/breadth 必须真的传进 ExpandConfig, 不是只在 cfg 里躺着。"""
    seen = {}
    real = EX.ExpandConfig

    def _capture(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(BP, "ExpandConfig", _capture)
    wired["_set_cfg"](rounds=3, breadth=7, max_requests=9, reviews=False)
    BP._build_candidate_pool("quantum error correction", max_pool=50)
    assert seen["max_rounds"] == 3
    assert seen["forward_breadth"] == 7 and seen["backward_breadth"] == 7
    assert seen["max_requests"] == 9
    assert seen["reviews_only"] is False


def test_topic_forwarded(wired, monkeypatch):
    """剪枝要主题词, 没传就等于剪不了。"""
    seen = {}

    def _expand(seeds, config=None, topic=""):
        seen["topic"] = topic
        return EX.ExpansionResult(entries=[], stats={
            "rounds": 1, "requests": 1, "dropped": {},
            "stopped_reason": "x", "total": 0})

    monkeypatch.setattr(BP, "expand_from_seeds", _expand)
    BP._build_candidate_pool("kirigami origami metasurface", max_pool=50)
    assert seen["topic"] == "kirigami origami metasurface"


# ============================================================
# 失败处理
# ============================================================

def test_expansion_failure_does_not_break_pipeline(wired, monkeypatch, capsys):
    def _boom(*a, **k):
        raise RuntimeError("[http 429] 限流")
    monkeypatch.setattr(BP, "expand_from_seeds", _boom)
    pool, st = BP._build_candidate_pool("quantum error correction", max_pool=50)
    assert st["external_status"]["expand"].startswith("exception:")
    assert "429" in capsys.readouterr().out, "扩展失败必须打到控制台"
    assert pool, "扩展挂了不该把整个候选池搞空"


# ============================================================
# 配置解析
# ============================================================

def test_expand_config_defaults():
    cfg = CFG.get_survey_expand()
    for k in ("enabled", "rounds", "breadth", "max_add", "max_requests", "reviews"):
        assert k in cfg, f"缺配置项 {k}"
    assert cfg["enabled"] is True


@pytest.mark.parametrize("raw,expect", [
    ("0", 0), ("3", 3), ("999", 40), ("abc", 40), ("-1", 40),
])
def test_resolver_clamps_and_falls_back(monkeypatch, raw, expect):
    """越界/非法值回落到默认, 而不是带着脏配置去烧 credit。"""
    f = CFG._resolve_survey_expand_int
    monkeypatch.setenv("QAGENT_SURVEY_EXPAND_MAX_ADD", raw)
    assert f("QAGENT_SURVEY_EXPAND_MAX_ADD", 40, 0, 200) == expect


def test_resolver_unset_uses_default(monkeypatch):
    monkeypatch.delenv("QAGENT_SURVEY_EXPAND_MAX_ADD", raising=False)
    assert CFG._resolve_survey_expand_int("QAGENT_SURVEY_EXPAND_MAX_ADD", 40, 0, 200) == 40


# ============================================================
# 排序优先级不能被新 source 挤掉
# ============================================================

def test_source_priority_ordering_unchanged():
    """
    新增的引文图 source 排在 APS/IOP 之后、arxiv 之前, 且老 source 的相对顺序原样保留:
    chroma_vector < summary_items_keyword < openalex_aps_api < openalex_iop_api
    （数字越小越靠前）。

    用 AST 抓 source_priority 这个真正的 dict, 不做源码文本 index ——
    文本搜会匹配到函数别处的同名字符串(chroma_vector 在 chroma 召回块里也出现过),
    那个坑我踩过一次。
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(BP._build_candidate_pool))
    d = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "source_priority":
                    d = ast.literal_eval(node.value)
    assert d is not None, "没抓到 source_priority"
    assert d["chroma_vector"] < d["summary_items_keyword"]
    assert d["summary_items_keyword"] < d["openalex_aps_api"]
    assert d["openalex_aps_api"] < d["openalex_iop_api"]
    # 新 source 插在 IOP 之后、arxiv 之前
    assert d["openalex_iop_api"] < d["openalex_forward_api"] < d["openalex_backward_api"]
    assert d["openalex_backward_api"] < d["arxiv_api_fallback"]
