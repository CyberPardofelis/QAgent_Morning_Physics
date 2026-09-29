"""
tests/test_console_hardening.py — 控制台编码兜底

9-29 P0。

问题: 项目里大量 `print("... ❌ ...")`, Windows 默认控制台 cp936 编不了这些
字符, print 上去直接 UnicodeEncodeError **打断整个业务流程**。
实测 `survey_generate` 在 APS 召回失败时就崩在 `_build_pool.py` 的一行 print 上。
"""
from __future__ import annotations

import io
import sys

import pytest

from agent.infra import console as C


class _FakeStream:
    def __init__(self, encoding="cp936", can_reconfigure=True, fail=False):
        self.encoding = encoding
        self._can = can_reconfigure
        self._fail = fail
        self.written = []
        self.reconfigured = None

    def reconfigure(self, **kw):
        if self._fail or not self._can:
            raise ValueError("cannot reconfigure")
        self.reconfigured = kw
        return None

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass


# ============================================================
# harden_stream
# ============================================================

def test_harden_stream_sets_errors_replace():
    """核心: errors 必须是 replace —— 默认 strict 遇错就抛。"""
    st = _FakeStream()
    assert C.harden_stream(st, "stdout") is True
    assert st.reconfigured == {"errors": "replace"}


def test_harden_stream_survives_failure():
    """改不了配置也不该抛(比如流被换成了不支持的)。"""
    st = _FakeStream(fail=True)
    assert C.harden_stream(st, "stdout") is False
    st2 = _FakeStream(can_reconfigure=False)
    assert C.harden_stream(st2, "stdout") is False


def test_harden_stream_handles_none():
    assert C.harden_stream(None, "stdout") is False


# ============================================================
# safe_print
# ============================================================

def test_safe_print_normal_path(monkeypatch):
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    C.safe_print("普通输出", "第二段")
    assert "普通输出" in out.getvalue() and "第二段" in out.getvalue()


def test_safe_print_never_raises_on_emoji(monkeypatch):
    """
    回归: 模拟 cp936 编码不了 ✅ 的情形, safe_print 必须静默降级而不是抛。
    原先 print("... ❌ ...") 在 cp936 上直接 UnicodeEncodeError 崩掉业务流程。
    """
    class _StrictStream(io.StringIO):
        encoding = "cp936"

        def write(self, s):
            s.encode("cp936")          # 严格模式: 编不了就抛
            return super().write(s)

    monkeypatch.setattr(sys, "stdout", _StrictStream())
    C.safe_print("召回失败 ❌ 降级 ⚠️")     # 关键: 不能抛
    C.safe_print("✅ 成功 🔍 搜索")


def test_safe_print_never_raises_when_stream_broken(monkeypatch):
    class _Broken:
        encoding = "cp936"
        def write(self, s):
            raise RuntimeError("stream closed")
        def flush(self):
            raise RuntimeError("stream closed")
        def reconfigure(self, **kw):
            raise RuntimeError("nope")

    monkeypatch.setattr(sys, "stdout", _Broken())
    C.safe_print("随便什么 ❌")          # 也不能抛


def test_safe_print_cjk_still_works(monkeypatch):
    """cp936 能编中文, 正常输出不该被破坏。"""
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    C.safe_print("候选池 40 篇")
    assert "候选池 40 篇" in out.getvalue()


# ============================================================
# _printable
# ============================================================

def test_printable_detects_cp936_gap(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _FakeStream(encoding="cp936"))
    assert C._printable("A") is True
    assert C._printable("候") is True          # cp936 支持中文
    assert C._printable("❌") is False        # cp936 不支持


def test_printable_allows_emoji_on_utf8(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _FakeStream(encoding="utf-8"))
    assert C._printable("❌") is True


# ============================================================
# 生产代码: 关键 print 已换成 safe_print
# ============================================================

def test_survey_pool_uses_safe_print():
    """
    回归护栏: 崩溃点 _build_pool.py 的 print 必须走 safe_print。
    (那行 print 在 APS 召回失败的分支上, 实测把 survey_generate 整个打断了)
    """
    import inspect

    from agent.tools.surveys._pool import _build_pool
    src = inspect.getsource(_build_pool)
    assert "from agent.infra.console import safe_print" in src
    assert not any(
        line.strip().startswith("print(") for line in src.splitlines()
    ), "_build_pool 不该再有裸 print"


@pytest.mark.parametrize("mod_path", [
    "agent.tools.surveys._pool._pool_audit",
    "agent.tools.arxiv",
    "agent.tools.aps",
    "agent.tools.iop",
])
def test_other_modules_use_safe_print(mod_path):
    import importlib
    import inspect
    m = importlib.import_module(mod_path)
    src = inspect.getsource(m)
    if "print(" not in src and "safe_print(" not in src:
        pytest.skip(f"{mod_path} 无 print")
    assert not any(
        line.strip().startswith("print(") for line in src.splitlines()
    ), f"{mod_path} 还有裸 print"


def test_main_installs_console_hardening():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    assert "_harden_console()" in src
    assert "from agent.infra.console import install" in src
