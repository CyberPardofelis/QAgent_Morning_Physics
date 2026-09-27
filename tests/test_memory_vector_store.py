"""
tests/test_memory_vector_store.py — MemoryVectorStore tombstone 生命周期单元测试

9-27 P0 修复配套测试。

背景:
    MemoryVectorStore 用 lazy tombstone 做删除（faiss-cpu 不支持单条 delete）:
    remove()/replace() 只在 metadata 上打 _deleted, faiss 行留着。
    但 search() 从来不看这个标记 —— tombstone 完全是摆设。后果:

    1. update_content 走 vs.add() 追加新 row 并覆盖 _id_to_idx, 旧 row 仍可命中,
       同一个 memory_id 对应两行, 检索时旧内容挤掉新内容（孤儿副本）
    2. remove() 之后向量仍在搜索结果里
    3. purge_deleted() 重建的是**空**索引却保留 alive_meta, meta 与 index 失同步

本测试用 object.__new__ 绕过单例 + 落盘, 直接测检索/替换/清理语义。
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from agent.memory import vector_store as vs_mod

pytestmark = pytest.mark.skipif(
    not vs_mod._FAISS_AVAILABLE, reason="faiss 未安装, 跳过 MemoryVectorStore 测试"
)

faiss = vs_mod.faiss


def _make_store(dim: int = 4, path_patches: bool = True):
    """构造一个不落盘、不走单例的 MemoryVectorStore。"""
    store = object.__new__(vs_mod.MemoryVectorStore)
    store.dim = dim
    store._write_lock = threading.Lock()
    store._meta = []
    store._id_to_idx = {}
    store._deleted_count = 0
    store.index = faiss.IndexFlatIP(dim)
    # 落盘是磁盘副作用, 测试里直接短路
    store._save = lambda: None
    return store


def _vec(*vals):
    return np.asarray(vals, dtype=np.float32)


# ============================================================
# search(): 跳过 tombstone
# ============================================================

def test_search_skips_tombstoned_rows():
    """remove() 之后, search() 不应再返回该条。"""
    s = _make_store()
    s.add("m1", _vec(1, 0, 0, 0), {})
    s.add("m2", _vec(0, 1, 0, 0), {})

    q = _vec(1, 0, 0, 0)
    before = {s.get_metadata(ri)["id"] for ri, _ in s.search(q, top_k=10)}
    assert before == {"m1", "m2"}

    assert s.remove("m1") is True

    after = {s.get_metadata(ri)["id"] for ri, _ in s.search(q, top_k=10)}
    assert "m1" not in after, "tombstone 的行不应被 search 返回"
    assert after == {"m2"}


def test_search_window_widens_with_tombstones():
    """
    tombstone 会被过滤, 检索窗口必须放大, 否则 top_k 填不满。
    造 6 条 tombstone(跟 query 几乎一致, 排在最前) + 3 条存活(排后面), top_k=3 仍拿满 3 条。
    """
    s = _make_store()
    for i in range(6):
        # dead 与 query 几乎一样 → 会排在最前, 如果不被 tombstone 过滤就会挤掉 live
        s.add(f"dead{i}", _vec(1.0, 0.01 * i, 0, 0), {})
    for i in range(3):
        s.add(f"live{i}", _vec(0.3, 1, 0, 0), {})
    for i in range(6):
        assert s.remove(f"dead{i}") is True

    q = _vec(1, 0, 0, 0)
    hits = s.search(q, top_k=3)
    assert len(hits) == 3, f"top_k=3 应拿满 3 条, 实际 {len(hits)}"
    assert {s.get_metadata(ri)["id"] for ri, _ in hits} == {"live0", "live1", "live2"}


def test_search_guards_out_of_bounds_row():
    """_meta 与 index 不同步时, 越界 row 应被跳过而不是 IndexError。"""
    s = _make_store()
    s.add("m1", _vec(1, 0, 0, 0), {})
    # 故意让 _meta 比 index 短
    s._meta = []
    hits = s.search(_vec(1, 0, 0, 0), top_k=5)
    assert hits == []


# ============================================================
# replace(): 不留孤儿副本
# ============================================================

def test_replace_tombstones_old_row_and_keeps_new():
    """replace() 后: 旧 row 标 _deleted, 新 row 生效, id_map 指向新 row。"""
    s = _make_store()
    s.add("m1", _vec(1, 0, 0, 0), {"v": 1})

    new_idx = s.replace("m1", _vec(0, 1, 0, 0), {"v": 2})

    assert s.get_row_idx("m1") == new_idx
    assert s._deleted_count == 1
    assert s._meta[new_idx].get("_deleted") is None
    old_row = 0
    assert s._meta[old_row].get("_deleted") is True


def test_replace_result_is_searchable_and_old_is_not():
    """
    核心回归: 同一个 memory_id 只有一条有效向量。
    replace 之前的行为是 add —— 旧 row 仍可命中, 旧内容会挤掉新内容。
    """
    s = _make_store()
    s.add("m1", _vec(1, 0, 0, 0), {})
    old_row = 0

    # 换成跟旧内容正交的新内容
    s.replace("m1", _vec(0, 1, 0, 0), {})

    # 旧 row 必须不在结果里（注意: faiss 仍会返回相似度 0 的行, 所以断言的是 row 本身）
    hits_old_q = s.search(_vec(1, 0, 0, 0), top_k=10)
    assert old_row not in {ri for ri, _ in hits_old_q}, "旧向量 row 不应再被搜到"

    # 新 row 可搜到, 且 metadata 正确
    hits_new_q = s.search(_vec(0, 1, 0, 0), top_k=10)
    new_rows = {ri for ri, _ in hits_new_q}
    assert s.get_row_idx("m1") in new_rows
    assert s.get_metadata(s.get_row_idx("m1"))["id"] == "m1"

    # m1 在结果里只出现一次
    ids = [s.get_metadata(ri)["id"] for ri, _ in hits_new_q]
    assert ids.count("m1") == 1


def test_replace_on_unknown_id_behaves_like_add():
    """replace() 一个不存在的 id = 普通 add, 不报错。"""
    s = _make_store()
    idx = s.replace("brand_new", _vec(1, 0, 0, 0), {})
    assert s.get_row_idx("brand_new") == idx
    assert s._deleted_count == 0
    assert len(s.search(_vec(1, 0, 0, 0), top_k=5)) == 1


def test_add_preserves_existing_behavior():
    """add() 语义不变: 纯追加, 不动已有行（replace 才 tombstone）。"""
    s = _make_store()
    s.add("m1", _vec(1, 0, 0, 0), {})
    s.add("m1", _vec(0, 1, 0, 0), {})  # 故意重复 add
    assert s._deleted_count == 0
    assert s.index.ntotal == 2


# ============================================================
# purge_deleted(): 正确重建
# ============================================================

def test_purge_deleted_rebuilds_with_vectors_intact():
    """
    回归: 旧实现建**空**索引却保留 alive_meta, meta 与 index 失同步。
    正确行为: 存活行的向量必须还在, 且仍可被搜到。
    """
    s = _make_store()
    s.add("keep1", _vec(1, 0, 0, 0), {"tag": "a"})
    s.add("dead1", _vec(0, 1, 0, 0), {"tag": "b"})
    s.add("keep2", _vec(0, 0, 1, 0), {"tag": "c"})

    s.remove("dead1")
    n = s.purge_deleted()

    assert n == 1, "应清掉 1 行"
    assert s.index.ntotal == 2
    assert s._deleted_count == 0
    assert [m["id"] for m in s._meta] == ["keep1", "keep2"]
    assert s.get_row_idx("keep1") == 0
    assert s.get_row_idx("keep2") == 1
    assert s.get_row_idx("dead1") is None

    # 关键: 存活行的向量还在, 搜索正常
    hits = s.search(_vec(0, 0, 1, 0), top_k=5)
    top_id = s.get_metadata(hits[0][0])["id"]
    assert top_id == "keep2", "purge 后 keep2 应仍是最佳匹配"
    # meta 与 index 行数必须一致（修复前 meta=2 / index=0, 彻底失同步）
    assert len(s._meta) == s.index.ntotal

    # get_vector 跟 index 不再失同步
    v = s.get_vector(s.get_row_idx("keep1"))
    assert v is not None
    np.testing.assert_allclose(np.abs(v), np.abs(_vec(1, 0, 0, 0)), atol=1e-5)


def test_purge_deleted_returns_zero_when_nothing_deleted():
    s = _make_store()
    s.add("m1", _vec(1, 0, 0, 0), {})
    assert s.purge_deleted() == 0
    assert s.index.ntotal == 1


def test_purge_deleted_handles_all_deleted():
    """全部 tombstone 时 purge 不应炸, 结果是空索引。"""
    s = _make_store()
    s.add("m1", _vec(1, 0, 0, 0), {})
    s.add("m2", _vec(0, 1, 0, 0), {})
    s.remove("m1")
    s.remove("m2")

    n = s.purge_deleted()
    assert n == 2
    assert s.index.ntotal == 0
    assert s._meta == []
    assert s.search(_vec(1, 0, 0, 0), top_k=5) == []


# ============================================================
# 归一化 / 维度校验（add / replace / add_batch 共用）
# ============================================================

def test_replace_validates_dimension():
    s = _make_store(dim=4)
    s.add("m1", _vec(1, 0, 0, 0), {})
    with pytest.raises(ValueError, match="dim"):
        s.replace("m1", _vec(1, 0, 0), {})


def test_replace_validates_ndim():
    s = _make_store(dim=4)
    with pytest.raises(ValueError, match="1D"):
        s.replace("m1", np.zeros((2, 4), dtype=np.float32), {})


def test_normalize_produces_unit_vector():
    s = _make_store(dim=4)
    v = s._normalize("x", _vec(3, 4, 0, 0))
    np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-6)


def test_count_excludes_tombstones():
    s = _make_store()
    s.add("m1", _vec(1, 0, 0, 0), {})
    s.add("m2", _vec(0, 1, 0, 0), {})
    assert s.count() == 2
    s.remove("m1")
    assert s.count() == 1
    assert s.total() == 2
    assert s.all_active_ids() == ["m2"]
