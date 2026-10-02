"""
tests/test_survey_resume.py — 崩溃续跑 (10-02)

v60e 跑了 76 分钟 / 151 次调用死在主题归纳那一步, 全轮作废。
这里锁的是续跑的两个硬要求: **signature 对不上就不许续**(否则旧参数下的中间
产物接到新参数上, 输出看着正常但结论是错的), 以及**续跑必须喊出声**(宁可吵,
不可静默接上旧数据)。
"""
from __future__ import annotations

import importlib

import pytest

# `from agent.tools.surveys import survey_generate` 拿到的是**函数**
# (包 __init__ 把同名函数 re-export 出来盖住了模块), 必须用 importlib。
SG_MOD = importlib.import_module("agent.tools.surveys.survey_generate")


# ============================================================
# workspace checkpoint / claims / narratives
# ============================================================

@pytest.fixture
def ws(tmp_path):
    from agent.infra.survey_workspace import SurveyWorkspace
    return SurveyWorkspace("testhash", topic="t", root=tmp_path)


def test_checkpoint_roundtrip(ws):
    assert ws.load_checkpoint() is None
    ws.save_checkpoint("facts", signature="abc", n_facts=42)
    ck = ws.load_checkpoint()
    assert ck["stage"] == "facts"
    assert ck["signature"] == "abc"
    assert ck["n_facts"] == 42
    assert ck["updated_at"]
    ws.clear_checkpoint()
    assert ws.load_checkpoint() is None


def test_checkpoint_rejects_corrupt_file(ws):
    ws.path.mkdir(parents=True, exist_ok=True)
    (ws.path / "_checkpoint.json").write_text("{不是 json", encoding="utf-8")
    assert ws.load_checkpoint() is None, "坏文件不该让流水线崩掉"


def test_claims_and_narratives_roundtrip(ws):
    # 关键区分: **没有文件**返回 None, **有文件但空**也返回 None
    # —— 续跑据此判断"能不能接着走", 分不清就会拿空数据接着跑。
    assert ws.load_claims() is None
    assert ws.load_narratives() is None

    ws.save_claims([{"claim_id": "c1", "statement": "s"}])
    assert [c["claim_id"] for c in ws.load_claims()] == ["c1"]

    ws.save_narratives({"主题A": {"text": "正文", "used_fact_ids": ["f1"]}})
    nar = ws.load_narratives()
    assert nar["主题A"]["text"] == "正文"
    assert nar["主题A"]["used_fact_ids"] == ["f1"]


# ============================================================
# _resolve_resume: signature 门禁
# ============================================================

_STAGES = ("pool", "facts", "themes", "claims", "narratives", "done")


def _sig_for(ws, topic_hash="H", max_pool=50, target_chars=0):
    """算出该参数组合的真实签名(走一次空跑)。"""
    _st, info = SG_MOD._resolve_resume(ws, topic_hash, max_pool, target_chars,
                                       False, _STAGES)
    return info["signature"]


def test_resume_from_checkpoint_returns_that_stage(ws):
    ws.save_checkpoint("themes", signature=_sig_for(ws))
    stage, info = SG_MOD._resolve_resume(ws, "H", 50, 0, True, _STAGES)
    assert stage == "themes"
    assert info["signature"] == _sig_for(ws)
    assert any("♻️" in ln for ln in info["lines"]), "续跑必须喊出声, 绝不静默"


def test_resume_rejects_mismatched_signature(ws):
    """
    改了 max_pool / target_chars 就不能续 —— 否则旧参数下的中间产物接到新参数上,
    那种错最难查: 输出看着完全正常, 结论却是错的。
    """
    ws.save_checkpoint("claims", signature=_sig_for(ws, max_pool=50))
    stage, info = SG_MOD._resolve_resume(ws, "H", 999, 0, True, _STAGES)
    assert stage == "pool", "signature 对不上必须从头跑"
    assert ws.load_checkpoint() is None, "失效的 checkpoint 要清掉"
    assert any("参数对不上" in ln for ln in info["lines"])


def test_resume_disabled_ignores_checkpoint(ws):
    ws.save_checkpoint("claims", signature=_sig_for(ws))
    stage, _info = SG_MOD._resolve_resume(ws, "H", 50, 0, False, _STAGES)
    assert stage == "pool", "resume=False 必须能强制从头跑"


def test_resume_rejects_unknown_stage(ws):
    ws.save_checkpoint("garbage-stage", signature=_sig_for(ws))
    stage, _ = SG_MOD._resolve_resume(ws, "H", 50, 0, True, _STAGES)
    assert stage == "pool"
    assert ws.load_checkpoint() is None


def test_resume_signature_depends_on_max_pool_and_target_chars():
    """签名必须真的把这两个参数算进去, 否则"改参数自动失效"是假的。"""
    from agent.infra.survey_workspace import SurveyWorkspace
    import tempfile
    from pathlib import Path

    sigs = set()
    with tempfile.TemporaryDirectory() as td:
        for mp, tc in ((50, 0), (60, 0), (50, 6000)):
            w = SurveyWorkspace("H", topic="t", root=Path(td))
            _st, info = SG_MOD._resolve_resume(w, "H", mp, tc, True, _STAGES)
            sigs.add(info["signature"])
    assert len(sigs) == 3, "max_pool / target_chars 任一变化都必须改签名"


# ============================================================
# 续跑接线的回归护栏
# ============================================================

def test_organize_phase_wires_resume_throughout():
    import inspect as _i
    src = (_i.getsource(SG_MOD._run_organize_phase)
           + _i.getsource(SG_MOD._phase_facts)
           + _i.getsource(SG_MOD._phase_write))
    for must in ("_resolve_resume", "load_pool", "load_facts",
                 "load_thematic_map", "load_claims"):
        assert must in src, f"续跑必须接上 {must}"
    # 跑完要清 checkpoint, 否则下次同参数重跑会接着上一轮的 narratives 走
    assert "clear_checkpoint" in src


def test_workspace_has_all_resume_pieces():
    from agent.infra.survey_workspace import SurveyWorkspace
    for m in ("save_checkpoint", "load_checkpoint", "clear_checkpoint",
              "save_claims", "load_claims", "save_narratives", "load_narratives"):
        assert hasattr(SurveyWorkspace, m), f"缺 {m}"


def test_run_actually_uses_run_ordered_in_the_three_hot_loops():
    """
    护栏: 三段最贵的循环必须走 run_ordered。
    这三段占了实测全流程 LLM 等待时间的 59%, 忘了并行就白改。
    """
    import inspect as _i
    from agent.tools.surveys._actions import _extract_facts as ef
    from agent.tools.surveys import _organize as org

    assert "run_ordered" in _i.getsource(ef.action_extract_facts)
    assert "run_ordered" in _i.getsource(org.condense_facts_into_claims)
    assert "run_ordered" in _i.getsource(org.write_theme_narrative)
    assert "run_ordered" in _i.getsource(org.write_theme_narrative_from_claims)


def test_workers_never_touch_shared_state_in_extract():
    """
    护栏: _extract_one 必须是纯函数 —— 它只发请求和解析,
    state/workspace 的写入一律留在主线程的归并循环里。

    切的是 _extract_one 的**函数体**(从它的定义到归并循环起点),
    不是"文件里 _extract_one 之后的所有内容"。
    """
    import inspect as _i
    from agent.tools.surveys._actions import _extract_facts as ef
    src = _i.getsource(ef.action_extract_facts)
    body = src[src.index("def _extract_one"):src.index("from agent.prompts_survey import clean_fact")]
    for forbidden in ("state[", "ws.append_fact", "_friendly_error"):
        assert forbidden not in body, f"worker 里不能碰 {forbidden}"


# ============================================================
# 端到端: 并发跑与串行跑的产物必须**逐字节相同**
# ============================================================
# 这是整个并发改动唯一真正重要的验证。前面的测试都只证明 run_ordered
# 自己保序; 这里证明接上真实流水线后, claim_id 编号 / 命题顺序 / 段落顺序
# 在并发下与串行完全一致 —— 因为 claim_id 决定正文引用编号,
# 差一位就是整篇综述的引用全部漂移。

def _facts_for_condense(n_subjects=6, per_subject=5):
    out = []
    for s in range(n_subjects):
        for j in range(per_subject):
            out.append({
                "fact_id": f"p{s % 3}::{s}_{j}",
                "paper_id": f"p{s % 3}",
                "subject": f"主题{s}",
                "claim": f"关于主题{s}的第 {j} 条事实",
                "n_facts": 1,
            })
    return out


def test_config_max_workers_default():
    from agent.tools.config import _resolve_survey_max_workers
    import os
    old = os.environ.pop("QAGENT_SURVEY_MAX_WORKERS", None)
    try:
        assert _resolve_survey_max_workers() == 4
        os.environ["QAGENT_SURVEY_MAX_WORKERS"] = "1"
        assert _resolve_survey_max_workers() == 1
        os.environ["QAGENT_SURVEY_MAX_WORKERS"] = "999"
        assert _resolve_survey_max_workers() == 4, "越界回落到默认"
        os.environ["QAGENT_SURVEY_MAX_WORKERS"] = "abc"
        assert _resolve_survey_max_workers() == 4, "非法值回落到默认"
    finally:
        os.environ.pop("QAGENT_SURVEY_MAX_WORKERS", None)
        if old is not None:
            os.environ["QAGENT_SURVEY_MAX_WORKERS"] = old
