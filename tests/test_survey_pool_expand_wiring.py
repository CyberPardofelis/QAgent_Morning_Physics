"""
test_survey_pool_expand_wiring.py — 引文图扩展接线 (9-30)

9-30 改版: 扩展从 _build_candidate_pool 移到 _audit_pool_with_retry 的重试轮里,
**种子是已经过审的论文**。原因见 _pool_audit._expand_from_audited 的 docstring。

这里测两处接线:
  A. _pool_audit._expand_from_audited —— 种子来源 / 词法门控 / 开关 / 失败
  B. _build_candidate_pool —— 召回层的词法门控
"""
from __future__ import annotations

import importlib
import sys

import pytest


def _mod(name: str):
    if name not in sys.modules:
        importlib.import_module(name)
    return sys.modules[name]


PA_NAME = "agent.tools.surveys._pool._pool_audit"
BP_NAME = "agent.tools.surveys._pool._build_pool"
EX_NAME = "agent.tools.surveys._pool._expand"
TM_NAME = "agent.tools.surveys._pool._topic_match"
CFG_NAME = "agent.tools.config"


def _entry(wid, title="Paper", abstract="tensor network states for quantum simulation",
           cited=10, source="openalex_aps_api", topic_hit=True):
    return {
        "paper_id": f"10.1/{wid.lower()}", "title": title, "year": "2020",
        "journal": "Phys Rev X", "abstract": abstract,
        "core_method": abstract[:100], "core_contrib": abstract[:200],
        "_openalex_id": wid, "cited_by_count": cited, "type": "article",
        "_source": source,
        # 测试用: 决定词法门控放不放行
        "_off_topic": not topic_hit,
    }


@pytest.fixture
def cfg(monkeypatch):
    """装一个可控的扩展配置。"""
    def _set(**over):
        base = {"enabled": True, "rounds": 2, "breadth": 20,
                "max_add": 40, "max_requests": 12, "reviews": False}
        base.update(over)
        monkeypatch.setattr(_mod(PA_NAME), "get_survey_expand", lambda: dict(base))
        return base
    return _set


@pytest.fixture
def gate(monkeypatch):
    """让词法门控可控 —— 否则测试要靠真实标题能不能命中主题短语, 太脆。"""
    def _set(off_topic_rejected=True):
        orig = _mod(TM_NAME).qualifies
        monkeypatch.setattr(
            _mod(PA_NAME), "qualifies",
            lambda topic, text: (not off_topic_rejected) or ("off-topic" not in text),
        )
        return orig
    return _set


# ============================================================
# A1. 种子来源: 必须是过审的论文
# ============================================================

def test_seeds_come_from_audited_dict(cfg, gate):
    """
    核心断言: 扩展的种子就是传进来的 audited(= all_kept)那批,
    不是原始召回结果。这是 9-30 user 拍板的改动。
    """
    cfg()
    gate()
    audited = {"p1": _entry("W1"), "p2": _entry("W2")}
    seen = {"p1", "p2"}
    got = []
    orig = _mod(PA_NAME).expand_from_seeds
    _mod(PA_NAME).expand_from_seeds = lambda seeds, **kw: (got.extend(seeds), _mod(EX_NAME).ExpansionResult(
        entries=[], stats={"rounds": 1, "requests": 1, "dropped": {},
                          "stopped_reason": "x", "total": 0}))[1]
    try:
        _mod(PA_NAME)._expand_from_audited("tensor network quantum simulation", audited, seen)
    finally:
        _mod(PA_NAME).expand_from_seeds = orig
    assert len(got) == 2
    assert {e["_openalex_id"] for e in got} == {"W1", "W2"}


def test_audited_without_openalex_id_is_skipped(cfg, gate, capsys):
    """过审论文里没有 W-id 的没法当种子 —— 应该跳过而不是报错。"""
    cfg()
    gate()
    audited = {"p1": {"paper_id": "x", "title": "no id here",
                      "abstract": "tensor network"}}
    out = _mod(PA_NAME)._expand_from_audited("tensor network quantum simulation", audited, set())
    assert out == []
    assert "skipped" in capsys.readouterr().out


# ============================================================
# A2. 开关
# ============================================================

def test_disabled(cfg, gate):
    cfg(enabled=False)
    assert _mod(PA_NAME)._expand_from_audited("t", {"p": _entry("W1")}, set()) == []


def test_max_add_zero(cfg, gate):
    cfg(max_add=0)
    assert _mod(PA_NAME)._expand_from_audited("t", {"p": _entry("W1")}, set()) == []


def test_empty_audited(cfg, gate):
    cfg()
    assert _mod(PA_NAME)._expand_from_audited("t", {}, set()) == []


# ============================================================
# A3. 词法门控: 扩展出来的也要剪
# ============================================================

def test_expansion_candidates_go_through_gate(cfg, gate, monkeypatch):
    """扩展出的候选不能直接进池, 也要过词法门控 —— 否则又变成灌噪声。"""
    cfg()
    gate()
    pa = _mod("agent.tools.surveys._pool._pool_audit")
    monkeypatch.setattr(pa, "expand_from_seeds", lambda seeds, **kw: _mod(EX_NAME).ExpansionResult(
        entries=[_entry("WA"), _entry("WB", abstract="off-topic unrelated paper")],
        stats={"rounds": 1, "requests": 1, "dropped": {},
               "stopped_reason": "x", "total": 2}))
    out = _mod(PA_NAME)._expand_from_audited("tensor network quantum simulation",
                                  {"p": _entry("W1")}, set())
    ids = {e["_openalex_id"] for e in out}
    assert "WA" in ids
    assert "WB" not in ids, "过不了门控的扩展候选必须被剪掉"


def test_gate_rejects_all_yields_empty(cfg, gate, monkeypatch):
    cfg()
    gate()
    pa = _mod("agent.tools.surveys._pool._pool_audit")
    monkeypatch.setattr(pa, "expand_from_seeds", lambda seeds, **kw: _mod(EX_NAME).ExpansionResult(
        entries=[_entry("WA", abstract="off-topic unrelated paper")],
        stats={"rounds": 1, "requests": 1, "dropped": {},
               "stopped_reason": "x", "total": 1}))
    assert _mod(PA_NAME)._expand_from_audited("t", {"p": _entry("W1")}, set()) == []


def test_max_add_limits_injection(cfg, gate, monkeypatch):
    cfg(max_add=3)
    gate()
    pa = _mod("agent.tools.surveys._pool._pool_audit")
    monkeypatch.setattr(pa, "expand_from_seeds", lambda seeds, **kw: _mod(EX_NAME).ExpansionResult(
        entries=[_entry(f"WX{i}", cited=i) for i in range(20)],
        stats={"rounds": 1, "requests": 1, "dropped": {},
               "stopped_reason": "x", "total": 20}))
    out = _mod(PA_NAME)._expand_from_audited("tensor network quantum simulation",
                                  {"p": _entry("W1")}, set())
    assert len(out) <= 3


def test_dedups_against_seen(cfg, gate, monkeypatch):
    cfg()
    gate()
    pa = _mod("agent.tools.surveys._pool._pool_audit")
    monkeypatch.setattr(pa, "expand_from_seeds", lambda seeds, **kw: _mod(EX_NAME).ExpansionResult(
        entries=[_entry("WDUP")],
        stats={"rounds": 1, "requests": 1, "dropped": {},
               "stopped_reason": "x", "total": 1}))
    out = _mod(PA_NAME)._expand_from_audited("t", {"p": _entry("W1")}, {"10.1/wdup"})
    assert out == [], "已在 seen 里的不能重复采纳"


# ============================================================
# A4. 失败不炸
# ============================================================

def test_expansion_failure_returns_empty_not_raise(cfg, gate, monkeypatch, capsys):
    cfg()
    gate()
    pa = _mod("agent.tools.surveys._pool._pool_audit")

    def _boom(*a, **k):
        raise RuntimeError("[http 429] 限流")
    monkeypatch.setattr(pa, "expand_from_seeds", _boom)
    out = _mod(PA_NAME)._expand_from_audited("t", {"p": _entry("W1")}, set())
    assert out == []
    assert "429" in capsys.readouterr().out, "失败必须打控制台, 不能静默"


# ============================================================
# B. 召回层词法门控
# ============================================================

def test_build_pool_uses_phrase_gate(monkeypatch):
    """
    召回层用短语准入而不是"任一单词命中"。
    9-30 实测: 旧逻辑 247 篇本地库命中 218 篇 (88%), 噪声原样进池每篇烧一次 LLM 审查。
    """
    import inspect
    src = inspect.getsource(_mod(BP_NAME)._build_candidate_pool)
    assert "qualifies(topic" in src, "召回层应走词法门控"
    assert "expand_from_seeds" not in src, "扩展不该再出现在审查之前"


def test_build_pool_no_longer_expands_before_audit():
    """扩展已从 _build_candidate_pool 移除, 只在审查重试轮里。"""
    import inspect
    src = inspect.getsource(_mod(BP_NAME)._build_candidate_pool)
    for bad in ("ExpandConfig", "select_frontier", "get_survey_expand"):
        assert bad not in src, f"{bad} 不该再出现在审查前的召回层"


# ============================================================
# C. 配置
# ============================================================

def test_expand_config_defaults():
    cfg = _mod(CFG_NAME).get_survey_expand()
    for k in ("enabled", "rounds", "breadth", "max_add", "max_requests", "reviews"):
        assert k in cfg
    assert cfg["enabled"] is True
    assert cfg["reviews"] is False, "综述通道默认关(会把原始研究论文筛掉)"


@pytest.mark.parametrize("raw,expect", [("0", 0), ("3", 3), ("999", 40), ("abc", 40), ("-1", 40)])
def test_resolver_clamps(monkeypatch, raw, expect):
    monkeypatch.setenv("QAGENT_SURVEY_EXPAND_MAX_ADD", raw)
    assert _mod(CFG_NAME)._resolve_survey_expand_int("QAGENT_SURVEY_EXPAND_MAX_ADD", 40, 0, 200) == expect
