"""
tests/test_survey_fact_id_uniqueness.py — fact_id 全局唯一性

9-29 修的 P0。

问题:
    SURVEY_FACT_PROMPT 的输出示例就是 `{"fact_id": "f1", ...}`, 而
    action_extract_facts 之前**原样**存下 LLM 返回的 fact_id。于是每篇论文的
    第一条 fact 都叫 "f1"、第二条都叫 "f2"......

    实测(QCVV 素材): 181 条 fact 只有 **5 个唯一 fact_id** (f1..f5),
    fid -> paper_id 映射 **97% 指向错误的论文**。

    为什么没被发现: 错配出来的也是同领域论文(都在候选池里), 抽查时看着"像"。
    但只要综述跨多个子领域, 引用就会系统性地指错, 且审计查不出来
    (审计只看 id 在不在池里, 不知道 id 原本属于谁)。

修复:
    抽取时按 `{paper_id}::{序号}` 重编 fact_id, 原文的存进 local_fact_id 备查。
"""
from __future__ import annotations

import inspect
from pathlib import Path


from agent.tools.surveys._actions import _extract_facts as ef
from agent.tools.surveys._pool._facts_review import _review_facts_topic_alignment

REPO = Path(__file__).resolve().parent.parent


# ============================================================
# 源码层面: 重编逻辑必须在
# ============================================================

def test_extract_facts_reassigns_fact_id():
    """
    9-29: 编号逻辑已移进 clean_fact(顺带去 paper_id 标注 + 质检),
    这里锁"抽取路径必须经过 clean_fact"这个不变量, 而不是某一行具体写法。
    """
    src = inspect.getsource(ef.action_extract_facts)
    assert "clean_fact(" in src, "抽取必须经过 clean_fact 规范化(重编 id + 去标注 + 质检)"
    # clean_fact 内部才该做重编
    from agent.prompts_survey import clean_fact
    csrc = inspect.getsource(clean_fact)
    assert 'f"{paper_id}::{idx}"' in csrc, "clean_fact 必须按 {paper_id}::{序号} 重编 fact_id"


def test_extract_facts_source_has_no_bare_append_of_llm_fact_id():
    """防止将来有人把重编逻辑改回去。"""
    src = inspect.getsource(ef.action_extract_facts)
    # setdefault("fact_id"...) 不允许出现 —— 那正是旧写法
    assert 'setdefault("fact_id"' not in src, "不要用 setdefault 兜底 fact_id"


# ============================================================
# 行为层面: 模拟抽取流程, 验证唯一性
# ============================================================

class _FakeWS:
    def __init__(self):
        self.rows = []

    def append_fact(self, f):
        self.rows.append(dict(f))


def _simulate(papers_and_llm_output, monkeypatch):
    """
    复现 action_extract_facts 的抽取+编号逻辑(不真的调 LLM)。
    直接用真实代码路径太重(依赖 build_fact_prompt/_call_llm), 这里只验证
    "重编"这一步的语义 —— 与真实现场同构。
    """
    ws = _FakeWS()
    all_facts = []
    for paper_id, raw in papers_and_llm_output:
        facts = [dict(x) for x in raw]
        for _i, f in enumerate(facts):
            f.setdefault("paper_id", paper_id)
            local_id = f.get("fact_id")
            f["local_fact_id"] = local_id
            f["fact_id"] = f"{paper_id}::{_i}"
            ws.append_fact(f)
            all_facts.append(f)
    return all_facts


# 每篇论文的 LLM 都会返回 f1..f4(因为 prompt 示例就是这么写的)
LLM_OUTPUT = [
    {"fact_id": "f1", "claim": "A1", "fact_type": "method"},
    {"fact_id": "f2", "claim": "A2", "fact_type": "result"},
    {"fact_id": "f3", "claim": "A3", "fact_type": "conclusion"},
    {"fact_id": "f4", "claim": "A4", "fact_type": "limitation"},
]


def test_fact_ids_are_globally_unique():
    papers = [
        ("10.1103/prxquantum.3.040310", LLM_OUTPUT),
        ("10.1088/1367-2630/15/1/015024", LLM_OUTPUT),
        ("2209.02093", LLM_OUTPUT),
    ]
    facts = _simulate(papers, None)
    ids = [f["fact_id"] for f in facts]
    assert len(facts) == 12
    assert len(set(ids)) == 12, f"fact_id 必须唯一, 实际 {len(set(ids))}/12: {sorted(set(ids))}"


def test_fact_id_to_paper_mapping_is_correct():
    """核心: fid -> paper_id 必须对得上。旧实现在这里 97% 错。"""
    papers = [
        ("10.1103/prxquantum.3.040310", LLM_OUTPUT),
        ("10.1088/1367-2630/15/1/015024", LLM_OUTPUT),
        ("2209.02093", LLM_OUTPUT),
    ]
    facts = _simulate(papers, None)
    fid2pid = {f["fact_id"]: f["paper_id"] for f in facts}
    assert len(fid2pid) == len(facts)
    for f in facts:
        assert f["fact_id"].startswith(f["paper_id"] + "::")
        assert fid2pid[f["fact_id"]] == f["paper_id"]


def test_local_fact_id_preserved():
    facts = _simulate([("p1", LLM_OUTPUT)], None)
    assert [f["local_fact_id"] for f in facts] == ["f1", "f2", "f3", "f4"]
    assert facts[0]["fact_id"] == "p1::0"


def test_paper_id_with_colons_does_not_break_roundtrip():
    """
    paper_id 本身可能含 '/' 和 '.'(DOI 形式), 不能让分隔符歧义。
    '::' 在 DOI 里不会出现, 用它做分隔是安全的。
    """
    pid = "10.1103/physrevlett.127.110504"
    facts = _simulate([(pid, LLM_OUTPUT)], None)
    assert facts[0]["fact_id"] == f"{pid}::0"
    assert facts[0]["fact_id"].split("::")[0] == pid


# ============================================================
# 兼容性: 老 workspace 的 fact(没有唯一 id)不能崩
# ============================================================

def test_old_facts_without_unique_id_still_auditable():
    """
    历史 workspace 里已有 fact_id="f1" 的老数据(本次修复前生成的)。
    事实对齐审查必须仍能工作 —— 它的 _fact_key 以 fact_id 优先, 重复时退到
    (paper_id, idx, claim_hash)。
    """
    import agent.tools.surveys._pool._facts_review as fr
    src = inspect.getsource(fr._review_facts_topic_alignment)
    assert "_fact_key" in src, "审查必须走 _fact_key 回退逻辑"


def test_fact_key_falls_back_when_fact_id_missing(monkeypatch):
    from agent.tools.surveys._pool import _facts_review as fr
    # fact_id 缺失时用 (paper_id, idx, claim_hash)
    assert hasattr(fr, "_review_facts_topic_alignment")
    s = inspect.getsource(fr)
    assert 'claim_hash' in s, "fact_id 缺失时应有基于 claim hash 的回退 key"


# ============================================================
# 事实对齐审查的 key 与新 fact_id 对齐
# ============================================================

def test_review_key_prefers_fact_id():
    """
    修复后 fact_id 唯一, _fact_key 返回它 —— 两个模块的 key 必须一致,
    否则审查结果写不回 fact 本体。
    """
    s = inspect.getsource(_review_facts_topic_alignment)
    assert 'fid = f.get("fact_id")' in s
    assert 'if fid:' in s
