"""
tests/test_memory_check_duplicate.py — MemoryProcessor.check_duplicate 单元测试

9-27 P0 修复配套测试。

背景:
    check_duplicate 的 judger 循环在第一个判为 same_fact/opposite 的条目就 break,
    但返回的 top_match 之前是 all_matches[0] —— 那是 embedding 相似度最高的那条,
    跟 judger 实际判定的不是同一条。工具层会把这个 top_match 的内容/ID 交给 LLM
    并指示它调 update_memory, 等于引导模型合并到错误记录上（数据丢失路径）。

本测试用假 embedder / 假 vector_store / 假 db 隔离外部依赖,
覆盖"judge 判定项 == top_match"这条不变量。
"""
from __future__ import annotations

import numpy as np
import pytest

from agent.memory import processor as proc_mod
from agent.memory.processor import MemoryProcessor


# ============================================================
# 假对象
# ============================================================

class _FakeEmbedder:
    def __init__(self, vector=(1.0, 0.0)):
        self._v = np.asarray(vector, dtype=np.float32)

    def encode(self, text, *args, **kwargs):
        return self._v


class _FakeVectorStore:
    """按传入顺序返回 row_idx（faiss 是按相似度降序返回的）。"""

    def __init__(self, rows):
        # rows: list[(row_idx, memory_id, similarity)]
        self._rows = rows

    def search(self, query_vector, top_k=10, exclude_ids=None):
        return [(ri, sim) for ri, _mid, sim in self._rows[:top_k]]

    def get_metadata(self, row_idx):
        for ri, mid, _sim in self._rows:
            if ri == row_idx:
                return {"id": mid}
        return None


class _FakeDB:
    def __init__(self, memories_by_id):
        self._by_id = memories_by_id

    def list_by_ids(self, memory_ids):
        # 跟真实实现一致：保持输入顺序
        return [self._by_id[mid] for mid in memory_ids if mid in self._by_id]

    def get(self, memory_id):
        return self._by_id.get(memory_id)


def _make_processor(rows, memories_by_id):
    """绕过 __init__ 构造, 避免真实 DB / faiss / embedding 模型。"""
    p = object.__new__(MemoryProcessor)
    p.embedder = _FakeEmbedder()
    p.vs = _FakeVectorStore(rows)
    p.db = _FakeDB(memories_by_id)
    p.retriever = None
    return p


def _mem(mid, extract):
    return {"id": mid, "content": f"content of {mid}", "metadata": {"extract": extract}}


# ============================================================
# 测试
# ============================================================

def test_top_match_is_the_judged_item_not_the_highest_similarity(monkeypatch):
    """
    回归: 相似度最高的那条被判 unrelated, 第二条被判 same_fact。
    top_match 必须是第二条（被 judge 判定的那条），不是第一条。
    """
    # extract 里 relations 为空 → 路 1/2 不触发；all_matches 非空 → 路 3 不触发
    monkeypatch.setattr(
        proc_mod, "_llm_extract",
        lambda content: {"core_concepts": ["c"], "relations": [], "negation": False},
    )

    def fake_judge(new_extract, existing_extract):
        # existing_extract["marker"] 决定判定结果
        if existing_extract.get("marker") == "SAME":
            return "same_fact", "同一个事实"
        return "unrelated", "不相干"

    monkeypatch.setattr(proc_mod, "_judge_triples", fake_judge)

    rows = [
        (0, "high_sim_but_unrelated", 0.99),   # 相似度最高
        (1, "lower_sim_but_same_fact", 0.80),  # 真正重复
    ]
    memories = {
        "high_sim_but_unrelated": _mem(
            "high_sim_but_unrelated", {"marker": "UNRELATED", "core_concepts": [], "relations": []},
        ),
        "lower_sim_but_same_fact": _mem(
            "lower_sim_but_same_fact", {"marker": "SAME", "core_concepts": [], "relations": []},
        ),
    }

    p = _make_processor(rows, memories)
    result = p.check_duplicate("我要保存的内容", top_k=3)

    assert result["verdict"] == "duplicate"
    assert result["judge_verdict"] == "same_fact"
    # 关键断言: top_match 必须是真正被 judge 判定为 same_fact 的那条
    assert result["top_match"] is not None
    assert result["top_match"]["memory"]["id"] == "lower_sim_but_same_fact", (
        "top_match 必须是实际被 judge 判定的条目, 而不是 embedding 相似度最高的那条"
    )
    # all_matches 顺序仍按相似度, 供 UI 展示
    assert [m["memory"]["id"] for m in result["all_matches"]] == [
        "high_sim_but_unrelated",
        "lower_sim_but_same_fact",
    ]


def test_opposite_verdict_also_returns_judged_item(monkeypatch):
    """verdict="opposite" 分支同样必须返回被判定的那条。"""
    monkeypatch.setattr(
        proc_mod, "_llm_extract",
        lambda content: {"core_concepts": [], "relations": [], "negation": False},
    )

    def fake_judge(new_extract, existing_extract):
        if existing_extract.get("marker") == "OPPOSITE":
            return "opposite", "互相矛盾"
        return "unrelated", ""

    monkeypatch.setattr(proc_mod, "_judge_triples", fake_judge)

    rows = [(0, "sim_high_unrelated", 0.99), (1, "sim_low_opposite", 0.70)]
    memories = {
        "sim_high_unrelated": _mem(
            "sim_high_unrelated", {"marker": "UNRELATED", "core_concepts": [], "relations": []},
        ),
        "sim_low_opposite": _mem(
            "sim_low_opposite", {"marker": "OPPOSITE", "core_concepts": [], "relations": []},
        ),
    }

    p = _make_processor(rows, memories)
    result = p.check_duplicate("新内容", top_k=3)

    assert result["verdict"] == "opposite"
    assert result["top_match"]["memory"]["id"] == "sim_low_opposite"


def test_first_item_judged_same_fact_still_returns_it(monkeypatch):
    """当相似度最高的那条本身就被判 same_fact, top_match 仍是它（不回归）。"""
    monkeypatch.setattr(
        proc_mod, "_llm_extract",
        lambda content: {"core_concepts": [], "relations": [], "negation": False},
    )
    monkeypatch.setattr(
        proc_mod, "_judge_triples",
        lambda new_e, old_e: ("same_fact", "重复"),
    )

    rows = [(0, "m0", 0.99), (1, "m1", 0.80)]
    memories = {
        "m0": _mem("m0", {"core_concepts": [], "relations": []}),
        "m1": _mem("m1", {"core_concepts": [], "relations": []}),
    }

    p = _make_processor(rows, memories)
    result = p.check_duplicate("新内容", top_k=3)

    assert result["verdict"] == "duplicate"
    assert result["top_match"]["memory"]["id"] == "m0"


def test_no_matches_returns_unique_and_none(monkeypatch):
    """没有任何候选时, top_match 必须是 None（不是空 dict / 越界）。"""
    monkeypatch.setattr(
        proc_mod, "_llm_extract",
        lambda content: {"core_concepts": [], "relations": [], "negation": False},
    )
    monkeypatch.setattr(
        proc_mod, "_judge_triples",
        lambda new_e, old_e: pytest.fail("没有候选时不应调用 judger"),
    )

    p = _make_processor([], {})  # faiss 空
    result = p.check_duplicate("新内容", top_k=3)

    assert result["verdict"] == "unique"
    assert result["top_match"] is None
    assert result["all_matches"] == []


def test_similarity_fallback_path_uses_highest_sim(monkeypatch):
    """
    judge 全是 unrelated 时, 走 similarity 兜底 — 此时 top_match 应是相似度最高的那条。
    确认修复没有把兜底路径也改错。
    """
    monkeypatch.setattr(
        proc_mod, "_llm_extract",
        lambda content: {"core_concepts": [], "relations": [], "negation": False},
    )
    monkeypatch.setattr(
        proc_mod, "_judge_triples",
        lambda new_e, old_e: ("unrelated", ""),
    )

    rows = [(0, "a", 0.96), (1, "b", 0.80)]
    memories = {
        "a": _mem("a", {"core_concepts": [], "relations": []}),
        "b": _mem("b", {"core_concepts": [], "relations": []}),
    }

    p = _make_processor(rows, memories)
    result = p.check_duplicate("新内容", sim_merge=0.99, sim_similar=0.75, top_k=3)

    assert result["judge_verdict"] == "unrelated"
    assert result["verdict"] == "similar"  # 0.96 >= 0.75 但 < 0.99
    assert result["top_match"]["memory"]["id"] == "a"


def test_matches_without_extract_are_skipped_by_judger(monkeypatch):
    """没有 extract 的候选不应被 judge, 也不应成为 top_match。"""
    monkeypatch.setattr(
        proc_mod, "_llm_extract",
        lambda content: {"core_concepts": [], "relations": [], "negation": False},
    )

    def fake_judge(new_extract, existing_extract):
        if existing_extract.get("marker") == "SAME":
            return "same_fact", "重复"
        return "unrelated", ""

    monkeypatch.setattr(proc_mod, "_judge_triples", fake_judge)

    rows = [(0, "no_extract", 0.99), (1, "has_extract_same", 0.80)]
    memories = {
        "no_extract": _mem("no_extract", {"marker": "SAME", "core_concepts": [], "relations": []}),
        "has_extract_same": _mem(
            "has_extract_same", {"marker": "SAME", "core_concepts": [], "relations": []},
        ),
    }

    p = _make_processor(rows, memories)
    result = p.check_duplicate("新内容", top_k=3)

    # 第一条有 extract → 会被 judge 判 same_fact, 所以 top_match 是它
    assert result["top_match"]["memory"]["id"] == "no_extract"
