"""
tests/test_memory_touch_batch.py — 记忆读路径 N+1 touch 修复的单元测试

9-27 P0 修复配套测试。

背景:
    MemoryRetriever._vector_retrieve 在返回结果前对每条记忆调一次 db.touch(),
    而 db.touch() 每次都 connect() 开一条新连接 + 一次 commit。
    记忆查看器一次搜索 limit=200, 就是 200 条连接 + 200 次 commit 全压在读路径上,
    还持有 WAL 写锁, 跟 worker 线程的写互相阻塞 (busy_timeout=30000)。

修复: db.touch_batch() 单连接 executemany, retriever 改调它。
"""
from __future__ import annotations

import contextlib

import numpy as np
import pytest

from agent.infra import db as infra_db
from agent.memory.db_manager import MemoryDBManager


# ============================================================
# connect() 假对象: 记录开了多少连接 / 执行了多少条
# ============================================================

class _FakeCursor:
    def __init__(self, log, ids):
        self._log = log
        self._ids = ids
        self.rowcount = len(ids)

    def executemany(self, sql, seq):
        self._log["executemany_calls"] += 1
        self._log["params"] = list(seq)
        return self

    def execute(self, sql, params=()):
        self._log["execute_calls"] += 1
        self._log["params"] = list(params)
        return self


class _FakeConn:
    def __init__(self, log):
        self._log = log

    def execute(self, sql, params=()):
        self._log["execute_calls"] += 1
        return _FakeCursor(self._log, [])

    def executemany(self, sql, seq):
        seq = list(seq)
        self._log["executemany_calls"] += 1
        self._log["batches"] = self._log.get("batches", 0) + 1
        self._log["params"] = seq
        return _FakeCursor(self._log, [p[1] for p in seq])

    def commit(self):
        self._log["commits"] += 1

    def rollback(self):
        pass

    def close(self):
        self._log["closes"] += 1


@pytest.fixture
def counted_connect(monkeypatch):
    """把 agent.infra.db.connect 换成计数假实现。"""
    log = {"connections": 0, "execute_calls": 0, "executemany_calls": 0,
           "commits": 0, "closes": 0, "batches": 0, "params": []}

    @contextlib.contextmanager
    def fake_connect(name):
        log["connections"] += 1
        conn = _FakeConn(log)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    monkeypatch.setattr(infra_db, "connect", fake_connect)
    return log


def _make_db_manager():
    """绕过 __init__（会建表 + 走单例）。"""
    return object.__new__(MemoryDBManager)


# ============================================================
# touch_batch
# ============================================================

def test_touch_batch_uses_single_connection(counted_connect):
    """N 条记忆 → 1 条连接, 1 次 commit。"""
    db = _make_db_manager()
    ids = [f"m{i}" for i in range(200)]

    n = db.touch_batch(ids)

    assert counted_connect["connections"] == 1, "N 条记忆只应开 1 条连接"
    assert counted_connect["commits"] == 1
    assert counted_connect["executemany_calls"] == 1, "应走 executemany, 不是逐条 execute"
    assert counted_connect["execute_calls"] == 0
    assert n == 200


def test_touch_batch_sends_one_param_pair_per_id(counted_connect):
    db = _make_db_manager()
    db.touch_batch(["a", "b", "c"])
    params = counted_connect["params"]
    assert len(params) == 3
    assert [p[1] for p in params] == ["a", "b", "c"]
    # 第一个参数是统一的 now 时间戳
    assert len({p[0] for p in params}) == 1


def test_touch_batch_dedupes_ids(counted_connect):
    """同一 id 只应计一次 access_count。"""
    db = _make_db_manager()
    db.touch_batch(["a", "b", "a", "a", "b"])
    params = counted_connect["params"]
    assert [p[1] for p in params] == ["a", "b"]


def test_touch_batch_preserves_order(counted_connect):
    db = _make_db_manager()
    db.touch_batch(["z", "y", "x"])
    assert [p[1] for p in counted_connect["params"]] == ["z", "y", "x"]


def test_touch_batch_empty_opens_no_connection(counted_connect):
    db = _make_db_manager()
    assert db.touch_batch([]) == 0
    assert counted_connect["connections"] == 0


def test_touch_batch_all_blank_opens_no_connection(counted_connect):
    db = _make_db_manager()
    assert db.touch_batch(["", None]) == 0
    assert counted_connect["connections"] == 0


def test_touch_delegates_to_touch_batch(counted_connect):
    """单条 touch() 现在走批量实现 —— 行为不变, 但也只开 1 条连接。"""
    db = _make_db_manager()
    db.touch("only-one")
    assert counted_connect["connections"] == 1
    assert counted_connect["executemany_calls"] == 1
    assert counted_connect["params"][0][1] == "only-one"


# ============================================================
# retriever 侧: 走 touch_batch 而不是逐条 touch
# ============================================================

class _RecordingDB:
    """记录被调用的方法名。"""

    def __init__(self, memories):
        self._memories = memories
        self.touch_batch_calls = []
        self.touch_calls = []

    def list_by_ids(self, ids):
        return [self._memories[i] for i in ids if i in self._memories]

    def touch_batch(self, ids):
        self.touch_batch_calls.append(list(ids))

    def touch(self, mid):
        self.touch_calls.append(mid)


class _StubVS:
    def __init__(self, rows):
        self._rows = rows  # [(row_idx, memory_id, sim)]

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


class _StubEmbedder:
    def encode(self, text, *a, **k):
        return np.asarray([1.0, 0.0], dtype=np.float32)


@pytest.mark.ui
def test_vector_retrieve_uses_touch_batch(monkeypatch):
    """读路径只调一次 touch_batch, 不再逐条 touch。"""
    from agent.memory import retriever as retr_mod
    from agent.memory.retriever import MemoryRetriever

    monkeypatch.setattr(
        retr_mod.MemoryRetriever, "_concept_recall",
        lambda self, *a, **k: [],  # 关掉 concept 分支, 隔离本测试
    )

    memories = {
        f"m{i}": {"id": f"m{i}", "content": f"c{i}", "importance": 0.5,
                  "metadata": {}, "created_at": "2026-01-01T00:00:00",
                  "scope": "global", "memory_type": "general"}
        for i in range(10)
    }
    db = _RecordingDB(memories)
    vs = _StubVS([(i, f"m{i}", 1.0 - i * 0.01) for i in range(10)])

    r = MemoryRetriever(db=db, vs=vs, embedder=_StubEmbedder())
    results = r._vector_retrieve(
        query="测试",
        limit=5,
        scope=None,
        memory_type=None,
        min_importance=None,
        min_similarity=0.0,
        importance_weight=0.3,
        decay_rate=0.0,
        mmr_lambda=0.7,
        concept_bonus_ids=None,
        concept_bonus=0.2,
    )

    assert len(results) == 5
    assert db.touch_calls == [], "不应再逐条调 touch()"
    assert len(db.touch_batch_calls) == 1, "应只调一次 touch_batch"
    assert len(db.touch_batch_calls[0]) == 5
