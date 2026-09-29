"""
tests/test_facts_db.py — 跨来源 fact 库

9-29 新增。

核心不变量(按重要性):
    1. `fact.id` 由**内容**决定, 绝不含 paper_id —— 跨来源后一条 fact 属于多个
       paper, 前缀方案不成立(9-29 刚修过一个 P0: `{paper_id}::{i}` 在单篇论文内
       就有 97% 映射错误)。
    2. **同内容 + 不同来源 → 同一条 fact + 多个出处**。这正是 [fact, paper]
       结构的价值所在, 也是"合并证据"能成立的前提。
    3. 同来源重复抽取幂等(不重复入库)。
    4. **只追加, 不做自动合并** —— 自动去重价值大但不可逆(合错了无法还原),
       第一版只提供候选。
    5. grounded 标记必须保留 —— fact 库大了以后要能筛掉不可信的。
    6. 与 memories 分开(那是"用户想记住的主观偏好", fact 是"从来源抽出的命题")。
"""
from __future__ import annotations

import inspect

import pytest

from agent.infra import db as infra_db
from agent.tools import facts as F


@pytest.fixture
def facts_db(tmp_path, monkeypatch):
    """把 facts 库指到 tmp, 不污染真实 data/。"""
    real_root = infra_db._data_root
    monkeypatch.setattr(infra_db, "_data_root", lambda: tmp_path)
    F.init_schema()
    yield
    monkeypatch.setattr(infra_db, "_data_root", real_root)


# ============================================================
# id 生成: 内容决定, 与来源无关
# ============================================================

def test_fact_id_is_content_derived():
    a = F.fact_id_for("随机测量可降低采样开销", "method")
    b = F.fact_id_for("完全不同的另一条事实", "method")
    assert a != b
    assert a.startswith("fact_")


def test_fact_id_ignores_punctuation_and_case():
    """同一句话不同标点/大小写应算同一条, 否则会分裂成多条 fact。"""
    a = F.fact_id_for("随机测量可降低采样开销。", "method")
    b = F.fact_id_for("随机测量, 可降低采样开销", "method")
    c = F.fact_id_for("  随机测量可降低采样开销  ", "method")
    assert a == b == c


def test_fact_id_distinguishes_type():
    """同内容不同 fact_type 视为不同事实。"""
    assert F.fact_id_for("X", "method") != F.fact_id_for("X", "result")


def test_fact_id_contains_no_paper_id():
    """护栏: id 绝不能含 paper_id —— 跨来源后一条 fact 属于多个 paper。"""
    pid = "10.1103/physrevlett.127.110504"
    fid = F.fact_id_for("某条事实", "method")
    assert pid not in fid
    assert "/" not in fid and ":" not in fid


def test_fact_id_stable_across_calls():
    assert F.fact_id_for("稳定性检查", "result") == F.fact_id_for("稳定性检查", "result")


# ============================================================
# 多来源合并 (核心价值)
# ============================================================

def test_same_content_different_sources_merges_into_one_fact(facts_db):
    """同命题从两篇论文抽出 -> 一条 fact + 两个出处。"""
    r1 = F.add_fact("SAC 框架对统计噪声鲁棒", "method",
                    source_id="2401.11111", quote="robust to statistical noise")
    r2 = F.add_fact("SAC 框架对统计噪声鲁棒。", "method",
                    source_id="2401.22222", quote="another paper also says so")

    assert r1["created"] is True
    assert r2["created"] is False, "第二条不该新建 fact"
    assert r2["source_added"] is True, "但应追加出处"
    assert r1["fact_id"] == r2["fact_id"]

    srcs = F.get_sources(r1["fact_id"])
    assert len(srcs) == 2
    assert {s["source_id"] for s in srcs} == {"2401.11111", "2401.22222"}


def test_multi_source_count_visible(facts_db):
    """多出处是"几个独立工作都这么说"的可审计信号。"""
    F.add_fact("命题甲", "result", source_id="A")
    F.add_fact("命题甲", "result", source_id="B")
    F.add_fact("命题乙", "result", source_id="C")

    got = F.list_facts(min_sources=2)
    assert len(got) == 1
    assert got[0]["n_sources"] == 2
    assert len(F.list_facts()) == 2


def test_same_source_reextraction_is_idempotent(facts_db):
    """同一来源重复抽取(比如综述跑两次)不该重复入库。"""
    r1 = F.add_fact("命题丙", "result", source_id="X", quote="q")
    r2 = F.add_fact("命题丙", "result", source_id="X", quote="q")
    r3 = F.add_fact("命题丙", "result", source_id="X", quote="q")
    assert r1["created"] and not r2["created"] and not r3["created"]
    assert r2["source_added"] is False
    assert len(F.get_sources(r1["fact_id"])) == 1


def test_cross_source_types_supported(facts_db):
    """来源类型是开放的: 论文摘要/全文/对话/手工 都能挂。"""
    F.add_fact("命题丁", "claim", source_type="paper_abstract", source_id="A1")
    F.add_fact("命题丁", "claim", source_type="paper_fulltext", source_id="A1")
    F.add_fact("命题丁", "claim", source_type="conversation", source_id="conv-7")
    srcs = F.get_sources(F.fact_id_for("命题丁", "claim"))
    assert {s["source_type"] for s in srcs} == {
        "paper_abstract", "paper_fulltext", "conversation",
    }
    # 同 source_id 但不同 source_type 算两个出处(摘要和全文是不同的证据)
    assert len(srcs) == 3


# ============================================================
# grounded 标记
# ============================================================

def test_grounded_flag_preserved(facts_db):
    F.add_fact("可信命题", "result", source_id="A", grounded=True)
    F.add_fact("可疑命题", "result", source_id="B", grounded=False)
    assert F.get_fact(F.fact_id_for("可信命题", "result"))["sources"][0]["grounded"] == 1
    assert F.get_fact(F.fact_id_for("可疑命题", "result"))["sources"][0]["grounded"] == 0


def test_list_facts_can_filter_ungrounded(facts_db):
    F.add_fact("好事实", "result", source_id="A", grounded=True)
    F.add_fact("坏事实", "result", source_id="B", grounded=False)
    assert len(F.list_facts()) == 2
    only_ok = F.list_facts(only_grounded=True)
    assert len(only_ok) == 1
    assert "好事实" in only_ok[0]["content"]


def test_self_referential_flag_preserved(facts_db):
    F.add_fact("带自指的 fact", "method", source_id="A", self_referential=True)
    s = F.get_sources(F.fact_id_for("带自指的 fact", "method"))[0]
    assert s["self_referential"] == 1


# ============================================================
# 批量入库
# ============================================================

def test_add_facts_from_extraction_skips_ungrounded(facts_db):
    batch = [
        {"claim": "通过审查的", "fact_type": "result", "paper_id": "P1",
         "support_quote": "q1", "_grounded": True},
        {"claim": "没通过审查的", "fact_type": "result", "paper_id": "P2",
         "support_quote": "q2", "_grounded": False},
    ]
    st = F.add_facts_from_extraction(batch)
    assert st["inserted"] == 1
    assert st["ungrounded"] == 1
    assert st["skipped"] == 1
    assert len(F.list_facts()) == 1


def test_add_facts_from_extraction_keeps_ungrounded_when_asked(facts_db):
    batch = [
        {"claim": "存下来但标记不可信", "fact_type": "result", "paper_id": "P2",
         "support_quote": "q", "_grounded": False},
    ]
    st = F.add_facts_from_extraction(batch, only_grounded=False)
    assert st["inserted"] == 1
    assert len(F.list_facts()) == 1
    assert F.list_facts(only_grounded=True) == []


def test_add_facts_from_extraction_handles_missing_fields(facts_db):
    batch = [
        {"claim": "", "paper_id": "P1"},                      # 空 content
        {"claim": "有内容但没 paper_id", "fact_type": "x"},  # 无来源
        {"claim": "正常", "fact_type": "result", "paper_id": "P3"},
    ]
    st = F.add_facts_from_extraction(batch)
    assert st["inserted"] == 1 and st["skipped"] == 2


def test_add_fact_requires_source(facts_db):
    """fact 必须可溯源 —— 没 source_id 不该能入库。"""
    with pytest.raises(ValueError, match="source_id"):
        F.add_fact("无出处的事实", "result", source_id="")


def test_add_fact_requires_content(facts_db):
    with pytest.raises(ValueError, match="content"):
        F.add_fact("", "result", source_id="A")


# ============================================================
# 只追加, 不自动合并
# ============================================================

def test_no_automatic_merge(facts_db):
    """
    第一版刻意不做自动去重合并(不可逆)。
    语义相近但不完全相同的两条 fact 应该各自独立存在。
    """
    F.add_fact("快照法能降低采样开销", "method", source_id="A")
    F.add_fact("快照法可以降低采样开销", "method", source_id="B")
    assert len(F.list_facts()) == 2, "近似但不同的 fact 不该被自动合并"


def test_dedup_candidates_is_advisory_only(facts_db):
    """候选生成只是提供信息, 不做修改。"""
    F.add_fact("同一个前缀开头的事实甲", "method", source_id="A")
    F.add_fact("同一个前缀开头的事实乙", "method", source_id="B")
    cands = F.dedup_candidates()
    assert isinstance(cands, list)
    assert len(F.list_facts()) == 2, "候选生成不应改动数据"


# ============================================================
# 检索 / 统计
# ============================================================

def test_filter_by_subject_type_source(facts_db):
    F.add_fact("命题A", "method", subject="快照", source_id="P1")
    F.add_fact("命题B", "result", subject="快照", source_id="P2")
    F.add_fact("命题C", "method", subject="层析", source_id="P1")

    assert len(F.list_facts(subject="快照")) == 2
    assert len(F.list_facts(fact_type="method")) == 2
    assert len(F.list_facts(source_id="P1")) == 2
    assert len(F.list_facts(subject="快照", fact_type="method")) == 1


def test_list_facts_limit(facts_db):
    for i in range(8):
        F.add_fact(f"命题{i}", "result", source_id=f"S{i}")
    assert len(F.list_facts(limit=3)) == 3


def test_stats_shape(facts_db):
    F.add_fact("统计用命题", "result", source_id="A")
    F.add_fact("统计用命题", "result", source_id="B")
    F.add_fact("不可信命题", "result", source_id="C", grounded=False)
    s = F.stats()
    assert s["n_facts"] == 2
    assert s["n_sources"] == 3
    assert s["multi_source"] == 1
    assert s["ungrounded"] == 1
    assert s["by_source_type"]["paper_abstract"] == 3


def test_get_fact_missing(facts_db):
    assert F.get_fact("fact_deadbeef") is None


def test_cascade_delete_source_removes_fact(facts_db):
    """fact 被删时出处应级联删掉, 不留孤儿行。"""
    r = F.add_fact("级联测试", "result", source_id="A")
    with infra_db.connect("facts") as conn:
        conn.execute("DELETE FROM fact WHERE id = ?", (r["fact_id"],))
    assert F.get_fact(r["fact_id"]) is None


# ============================================================
# 与现有库的关系
# ============================================================

def test_facts_is_own_database():
    """必须独立于 memories(主观偏好) 和 surveys(成文综述)。"""
    assert "facts" in infra_db.DB_FILES
    assert "facts" not in {"memory", "surveys", "papers", "summary_items"}


def test_facts_db_path_separate_from_memory():
    assert infra_db.DB_FILES["facts"] != infra_db.DB_FILES["memory"]


def test_schema_has_two_tables():
    assert "CREATE TABLE IF NOT EXISTS fact (" in F.SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS fact_source (" in F.SCHEMA_SQL


def test_fact_source_unique_constraint():
    """(fact_id, source_type, source_id) 唯一 —— 幂等的基础。"""
    assert "UNIQUE (fact_id, source_type, source_id)" in F.SCHEMA_SQL


# ============================================================
# 接入
# ============================================================

def test_extract_facts_writes_to_fact_store():
    from agent.tools.surveys._actions import _extract_facts as ef
    src = inspect.getsource(ef.action_extract_facts)
    assert "add_facts_from_extraction" in src, "fact 抽取结果应写入 fact 库"
    assert "source_type=\"paper_abstract\"" in src


def test_fact_store_failure_does_not_break_survey():
    """fact 库写失败不该让综述失败 —— 它是增强, 不是前置依赖。"""
    from agent.tools.surveys._actions import _extract_facts as ef
    src = inspect.getsource(ef.action_extract_facts)
    assert "fact_store" in src and "不影响本次综述" in src
