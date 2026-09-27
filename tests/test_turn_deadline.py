"""
tests/test_turn_deadline.py — turn 级 deadline 的单元测试

9-27 P0 修复配套测试。

背景:
    controller 之前只有**单次** LLM 调用的 timeout(默认 600s), 工具执行完全没有
    超时, 整个 turn 的墙上时间没有上界。最坏情况: 综述工具 max_pool=200 时
    extract_facts 串行发 200 次 LLM(每次 timeout=180s), 全部发生在**一个**
    controller 轮次里 —— 计费为 1, 耗时可达数小时, 且无法中断。

修复: turn 预算(Config.get_turn_deadline_seconds, 默认 1800s)
  - 每轮开始前检查预算, 超了就优雅收尾
  - 单次调用的 timeout 压到剩余预算以内, 避免一次调用吃光整个预算
  - 超预算时不扣费、不发请求
"""
from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from agent import controller as ctrl
from config import Config

REPO = Path(__file__).resolve().parent.parent


# ============================================================
# Config 项
# ============================================================

def test_turn_deadline_default_is_1800(monkeypatch):
    monkeypatch.setattr(Config, "_settings", staticmethod(lambda: _FakeSettings({}, 1800)))
    assert Config.get_turn_deadline_seconds() == 1800


class _FakeSettings:
    def __init__(self, data, default):
        self._data = data
        self._default = default

    def value(self, key, default=None, type=None):
        return self._data.get(key, default)

    def setValue(self, key, value):
        self._data[key] = value


def test_turn_deadline_floor_is_60(monkeypatch):
    """小于 60s 的配置会被抬到 60s —— 否则连一次长 prompt 都跑不完。"""
    monkeypatch.setattr(Config, "_settings", staticmethod(lambda: _FakeSettings({"turn_deadline_seconds": 5}, 1800)))
    assert Config.get_turn_deadline_seconds() == 60


def test_turn_deadline_handles_garbage(monkeypatch):
    monkeypatch.setattr(Config, "_settings", staticmethod(lambda: _FakeSettings({"turn_deadline_seconds": "abc"}, 1800)))
    assert Config.get_turn_deadline_seconds() == 1800


def test_turn_deadline_setter_clamps():
    Config.set_turn_deadline_seconds(30)
    assert Config.get_turn_deadline_seconds() == 60
    Config.set_turn_deadline_seconds(900)
    assert Config.get_turn_deadline_seconds() == 900
    # 还原, 别污染真实 QSettings
    Config.set_turn_deadline_seconds(1800)


# ============================================================
# 结构性: 预算检查的顺序
# ============================================================

def _run_agent_stream_ast():
    tree = ast.parse((REPO / "agent/controller.py").read_text(encoding="utf-8"))
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_agent_stream"
    )


def test_budget_check_precedes_charge():
    """
    超预算的轮次不应再扣费 —— 预算检查必须排在 check_and_increment 之前。
    """
    fn = _run_agent_stream_ast()
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.While))

    order: list[str] = []
    for stmt in loop.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Compare):
                for c in ast.walk(node):
                    if isinstance(c, ast.Name) and c.id == "_remaining":
                        order.append("budget")
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name in {"check_and_increment", "charge"}:
                    order.append("charge")

    assert "budget" in order, "while 循环里应有 turn 预算检查"
    assert "charge" in order
    assert order.index("budget") < order.index("charge"), (
        f"预算检查必须在计费之前, 实际顺序: {order}"
    )


def test_effective_timeout_is_clamped_to_budget():
    """单次调用的 timeout 必须被压到剩余预算以内。"""
    src = (REPO / "agent/controller.py").read_text(encoding="utf-8")
    assert "_effective_timeout" in src
    assert "min(timeout, max(_budget_left, 10))" in src, (
        "应把 timeout 压到剩余预算(_budget_left)以内"
    )


# ============================================================
# 行为: 预算耗尽时优雅收尾
# ============================================================

class _FakeStream:
    """最小 stream 对象: 没有 tool_calls, 没有 delta.content。"""

    def __iter__(self):
        return iter([])


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _FakeStream()


class _FakeClient:
    def __init__(self):
        self.chat = type("_Chat", (), {"completions": _FakeCompletions()})()


@pytest.fixture
def budget_exhausted(monkeypatch):
    """
    造一个"一进循环预算就耗尽"的场景。

    deadline 算在 run_agent_stream 入口, 所以把 get_turn_deadline_seconds
    压到最小 60s, 再把 time.monotonic 推进 61s 假装已经过了 61 秒。
    """
    monkeypatch.setattr(Config, "get_turn_deadline_seconds", staticmethod(lambda: 60))
    client = _FakeClient()
    monkeypatch.setattr(
        ctrl, "_resolve_client_and_model", lambda mode: (client, "m", "deepseek")
    )
    monkeypatch.setattr(ctrl, "resolve_legacy_model", lambda m, t, r="high": (m, t, r))
    monkeypatch.setattr(ctrl, "get_cost_guard", lambda: _NeverChargeGuard())
    monkeypatch.setattr(ctrl.ToolRegistry, "list_schemas_filtered", staticmethod(lambda t: []))

    real_monotonic = time.monotonic
    state = {"t": 1000.0}
    monkeypatch.setattr(ctrl.time, "monotonic", lambda: state["t"])
    # 第一次读(算 deadline)返回 1000, 之后都返回 1000+61 → 预算已耗尽
    calls = {"n": 0}

    def fake_monotonic():
        calls["n"] += 1
        return 1000.0 if calls["n"] == 1 else 1061.0

    monkeypatch.setattr(ctrl.time, "monotonic", fake_monotonic)
    assert real_monotonic is not None
    return client


class _NeverChargeGuard:
    """如果走到计费就说明顺序错了。"""

    def check_and_increment(self, ceiling=None):
        raise AssertionError("预算已耗尽, 不应再计费")


def test_exhausted_budget_stops_before_any_llm_call(budget_exhausted):
    client = budget_exhausted

    out = list(ctrl.run_agent_stream([{"role": "user", "content": "hi"}]))

    assert client.chat.completions.calls == 0, "预算耗尽时不应发出 LLM 请求"
    joined = "".join(out)
    assert "时间预算" in joined or "已停止" in joined
    assert ctrl.NEW_BUBBLE in out, "收尾要发 NEW_BUBBLE 让 UI 正确关闭气泡"
