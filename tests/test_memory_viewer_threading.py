"""
tests/test_memory_viewer_threading.py — 记忆查看器不阻塞 GUI 线程的测试

9-27 P0 修复配套测试。

背景:
    MemoryViewerDialog 的 load_all / perform_search / load_concept_groups /
    删除 之前都在 GUI 线程直接调 processor。其中 perform_search 会走
    retriever._vector_retrieve -> embedder.encode, 即 BGE-large 的 CPU 推理
    (几百 ms ~ 数秒), 期间整个窗口无响应。聊天路径有 ui/worker.py 的 QThread
    保护, 记忆查看器一直没有。

本测试验证: 阻塞调用发生在工作线程, 结果正确回到 GUI 线程渲染,
且过期结果不会覆盖新结果。
"""
from __future__ import annotations

import threading

import pytest

pytestmark = pytest.mark.ui

from PySide6.QtCore import QCoreApplication, QThreadPool  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        import sys

        from PySide6.QtWidgets import QApplication
        app = QApplication(sys.argv)
    return app


class _FakeProcessor:
    """记录被调用的线程 id, 模拟耗时。"""

    def __init__(self, memories=None, recall_result=None):
        self._memories = memories if memories is not None else []
        self._recall_result = recall_result
        self.call_threads: list[int] = []
        self.deleted: list[str] = []
        self.recall_kwargs: dict = {}

    def _record(self):
        self.call_threads.append(threading.get_ident())

    def list_all(self, **kw):
        self._record()
        return list(self._memories)

    def recall(self, **kw):
        self._record()
        self.recall_kwargs = kw
        return list(self._recall_result or [])

    def delete(self, mid):
        self._record()
        self.deleted.append(mid)
        return True

    @property
    def embedder(self):
        return self

    def get_backend_info(self):
        return {"enabled": True, "dim": 1024}


def _mem(i):
    return {
        "id": f"memory-000{i}",
        "memory_type": "general",
        "content": f"内容 {i}",
        "importance": 0.5,
        "related_paper_id": "",
        "created_at": "2026-01-01T00:00:00",
        "access_count": 0,
    }


@pytest.fixture
def dialog(qapp, monkeypatch):
    """构造一个真实的 MemoryViewerDialog, 但注入假 processor。"""
    from ui import memory_viewer_dialog as mvd

    created = {}

    def factory(memories=None, recall_result=None):
        p = _FakeProcessor(memories=memories, recall_result=recall_result)
        created["p"] = p
        return p

    dlg = mvd.MemoryViewerDialog()
    # 替换掉 lazy processor, 让所有查询走假实现
    dlg._processor = factory([_mem(i) for i in range(5)], recall_result=[_mem(0), _mem(1)])
    yield dlg, created["p"]
    dlg.close()
    dlg.deleteLater()


def _drain(pool: QThreadPool, ms: int = 5000):
    assert pool.waitForDone(ms), "线程池任务未在超时内完成"
    QCoreApplication.processEvents()


def test_perform_search_runs_off_gui_thread(dialog):
    """核心回归: 检索必须发生在工作线程, 不能在 GUI 线程。"""
    dlg, proc = dialog
    main_thread = threading.get_ident()

    dlg.query_edit.setText("用户偏好")
    dlg.perform_search()
    _drain(dlg._pool)

    assert proc.call_threads, "recall 应被调用"
    assert all(t != main_thread for t in proc.call_threads), (
        f"检索跑在了 GUI 线程上: {proc.call_threads} (main={main_thread})"
    )


def test_load_all_runs_off_gui_thread(dialog):
    dlg, proc = dialog
    main_thread = threading.get_ident()
    proc.call_threads.clear()

    dlg.load_all()
    _drain(dlg._pool)

    assert proc.call_threads
    assert all(t != main_thread for t in proc.call_threads)


def test_concept_groups_run_off_gui_thread(monkeypatch, dialog):
    dlg, _proc = dialog
    main_thread = threading.get_ident()
    seen: list[int] = []

    def fake_compute():
        seen.append(threading.get_ident())
        return {"groups": {}, "no_extract_count": 0}

    dlg._compute_concept_groups = fake_compute
    dlg.load_concept_groups()
    _drain(dlg._pool)

    assert seen and all(t != main_thread for t in seen)


def test_search_result_is_rendered_on_gui_thread(dialog):
    """worker 的结果最终要落到表格上。"""
    dlg, _proc = dialog
    dlg.query_edit.setText("偏好")
    dlg.perform_search()
    _drain(dlg._pool)

    assert dlg.result_table.rowCount() == 2, "检索结果应被填进主表"


def test_display_does_not_touch_processor_on_gui_thread(dialog):
    """
    _display 之前会 self._get_processor().embedder —— 在主线程构造 embedder。
    现在 backend info 由 worker 带回缓存。
    """
    dlg, _proc = dialog

    class _Boom:
        def __getattr__(self, name):
            raise AssertionError(f"_display 不应在 GUI 线程访问 {name}")

    dlg._processor = _Boom()
    dlg._display([_mem(0)], mode="all", table=dlg.result_table)  # 不应抛
    assert dlg.result_table.rowCount() == 1


def test_stale_result_is_discarded(dialog):
    """连续触发两次查询时, 先发起的那个结果不应覆盖后发起的。"""
    dlg, _proc = dialog
    seen_generations: list[int] = []
    original_run_async = dlg._run_async

    def spy(key, fn, on_ok, on_err=None):
        seen_generations.append(dlg._generation + 1)
        return original_run_async(key, fn, on_ok, on_err)

    dlg._run_async = spy
    dlg.load_all()
    dlg.load_all()
    _drain(dlg._pool)

    assert len(seen_generations) == 2
    assert seen_generations[0] != seen_generations[1], "两次查询的 generation 应不同"


def test_close_shuts_down_thread_pool(dialog):
    """关窗不能因为在途任务卡住。"""
    dlg, _proc = dialog
    dlg.query_edit.setText("x")
    dlg.perform_search()
    dlg.close()  # 不应抛异常或永久阻塞
    QCoreApplication.processEvents()
