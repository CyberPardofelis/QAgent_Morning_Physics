"""
tests/test_memory_concept_aware.py — concept_aware 默认值的单元测试

9-27 P0 修复配套测试。

背景:
    recall_memories 注册为 cost_tier="L0"（项目约定 L0 = 0 次 LLM 调用）,
    但 retriever.retrieve() 的 concept_aware 默认是 True,
    而 concept_aware=True 会先调 _concept_recall -> _extractor_extract
    发一次完整 LLM 请求。

    结果: 标着 L0 的工具每次检索都在发 LLM, 成本守卫和工具分档全失真。
    记忆查看器 (ui/memory_viewer_dialog.py perform_search) 也走同一条路径,
    等于在 GUI 线程上发网络请求。

修复: concept_aware 默认改为 False, 并在工具 schema 上暴露为显式可选参数。
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from agent.memory.processor import MemoryProcessor
from agent.memory.retriever import MemoryRetriever
from agent.tools.memory import recall_memories


# ============================================================
# 假对象
# ============================================================

class _StubEmbedder:
    def encode(self, text, *a, **k):
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _StubVS:
    def __init__(self, rows):
        self._rows = rows

    def search(self, vec, top_k=10, exclude_ids=None):
        return [(ri, sim) for ri, _mid, sim in self._rows[:top_k]]

    def total(self):
        return len(self._rows)

    def get_metadata(self, row_idx):
        for ri, mid, _s in self._rows:
            if ri == row_idx:
                return {"id": mid}
        return None

    def get_vector(self, row_idx):
        return None


class _StubDB:
    def __init__(self, memories):
        self._memories = memories

    def list_by_ids(self, ids):
        return [self._memories[i] for i in ids if i in self._memories]

    def touch_batch(self, ids):
        pass

    def touch(self, mid):
        pass

    def list_recent(self, **kw):
        return []
    def search_keyword(self, kw, scope=None, limit=10):
        return []


def _make_memories(n=3):
    return {
        f"m{i}": {"id": f"m{i}", "content": f"c{i}", "importance": 0.5,
                  "metadata": {}, "created_at": "2026-01-01T00:00:00",
                  "scope": "default", "memory_type": "general"}
        for i in range(n)
    }


def _make_retriever(monkeypatch, concept_hits=None):
    """retriever + 记录 _concept_recall 调用次数。"""
    calls = []

    def fake_concept_recall(self, query, scope=None, limit=50):
        calls.append(query)
        return list(concept_hits or [])

    monkeypatch.setattr(MemoryRetriever, "_concept_recall", fake_concept_recall)

    memories = _make_memories()
    db = _StubDB(memories)
    vs = _StubVS([(i, f"m{i}", 1.0 - i * 0.1) for i in range(3)])
    r = MemoryRetriever(db=db, vs=vs, embedder=_StubEmbedder())
    return r, calls


# ============================================================
# retriever.retrieve 默认不发 LLM
# ============================================================

def test_retrieve_default_does_not_call_concept_recall(monkeypatch):
    """默认参数下 concept_aware=False, 不应触发 concept_recall（即不发 LLM）。"""
    r, calls = _make_retriever(monkeypatch)
    r.retrieve(query="用户偏好", limit=3)
    assert calls == [], "默认不应调 _concept_recall（那会发 LLM 请求）"


def test_retrieve_explicit_concept_aware_does_call_it(monkeypatch):
    """显式开启时仍走 concept_recall（功能保留, 只是不再默认打开）。"""
    r, calls = _make_retriever(monkeypatch, concept_hits=[{"id": "m0"}])
    r.retrieve(query="用户偏好", limit=3, concept_aware=True)
    assert len(calls) == 1


@pytest.mark.parametrize("method", ["vector", "keyword", "hybrid"])
def test_retrieve_default_no_concept_for_all_methods(monkeypatch, method):
    """三条检索路径 (vector/keyword/hybrid) 默认都不发 LLM。"""
    r, calls = _make_retriever(monkeypatch)
    r.retrieve(query="测试", limit=3, method=method)
    assert calls == [], f"method={method} 默认不应触发 concept_recall"


def test_retrieve_signature_default_is_false():
    """签名层面锁住默认值, 防止被无意改回 True。"""
    sig = inspect.signature(MemoryRetriever.retrieve)
    assert sig.parameters["concept_aware"].default is False


def test_processor_recall_signature_default_is_false():
    sig = inspect.signature(MemoryProcessor.recall)
    assert sig.parameters["concept_aware"].default is False


def test_processor_recall_passes_concept_aware_through(monkeypatch):
    """processor 必须把 concept_aware 透传给 retriever, 不能吞掉。"""
    seen = {}

    class _FakeRetriever:
        def retrieve(self, **kw):
            seen.update(kw)
            return []

    p = object.__new__(MemoryProcessor)
    p.retriever = _FakeRetriever()

    p.recall(query="q", concept_aware=True)
    assert seen["concept_aware"] is True

    p.recall(query="q")
    assert seen["concept_aware"] is False


# ============================================================
# 工具层
# ============================================================

def test_recall_memories_signature_has_concept_aware_default_false():
    sig = inspect.signature(recall_memories)
    assert "concept_aware" in sig.parameters
    assert sig.parameters["concept_aware"].default is False


def test_recall_memories_schema_exposes_concept_aware():
    """LLM 看到的 schema 必须包含这个参数, 否则改默认值对模型不可见。"""
    from agent.tool_definitions import TOOL_SCHEMAS

    props = TOOL_SCHEMAS["recall_memories"]["function"]["parameters"]["properties"]
    assert "concept_aware" in props
    assert props["concept_aware"]["type"] == "boolean"
    assert props["concept_aware"]["default"] is False


def test_recall_memories_docstring_warns_about_llm_cost():
    """docstring 是 LLM 读到的契约, 必须写明开启会多一次 LLM 请求。"""
    doc = recall_memories.__doc__ or ""
    assert "LLM" in doc
    assert "concept_aware" in doc
