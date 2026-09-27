"""
tests/test_llm_cost_accounting.py — LLM 调用计费口径的单元测试

9-27 P0 修复配套测试。

背景:
    项目里有 5 个真实的 chat.completions.create 调用点, 之前只有 3 个计费:
      1. controller.run_agent_stream 的 ReAct 轮   → 计费 ✓
      2. controller._stream_openai_simple          → 计费 ✓
      3. controller.call_model                     → 计费 ✓
      4. agent/tools/llm.py 的 _call_llm (所有工具内部调用) → **不计费** ✗
      5. agent/memory/extractor.py 的抽取调用      → **不计费** ✗

    另外 controller.py 在 run_agent_stream 开头还预扣了一次, 第一轮被重复计费。

    实际后果: survey_generate 一次运行通过 _call_llm 发 100+ 次请求,
    守卫只看见约 50 次 —— ceiling 约束不住真实开销, 成本面板失真。

本测试:
  - AST 层面保证每个 create() 调用点所在函数都接了 charge
  - 行为层面验证 _call_llm 真的扣费、且被阻断时不发请求
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from agent import controller as ctrl_mod
from agent import cost_guard as cg_mod
from agent.tools import llm as tools_llm

REPO = Path(__file__).resolve().parent.parent


# ============================================================
# 找所有真实的 LLM 请求点
# ============================================================

def _iter_source_files():
    for rel in ("agent/controller.py", "agent/tools/llm.py", "agent/memory/extractor.py"):
        yield REPO / rel


def _find_create_calls(tree: ast.AST) -> list[ast.Call]:
    """找出所有 client.chat.completions.create(...) 调用节点。"""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute) or f.attr != "create":
            continue
        v = f.value
        # 匹配 xxx.chat.completions.create
        if isinstance(v, ast.Attribute) and v.attr == "completions":
            found.append(node)
    return found


def _enclosing_function_names(tree: ast.Module, target: ast.Call) -> set[str]:
    """返回包含该调用的一串嵌套函数名。"""
    names: set[str] = set()

    def visit(node, stack):
        for child in ast.iter_child_nodes(node):
            new_stack = stack
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new_stack = stack | {child.name}
            if child is target:
                names.update(new_stack)
            visit(child, new_stack)

    visit(tree, set())
    return names


CHARGE_NAMES = {"check_and_increment", "charge", "cost_guard_charge", "_charge"}


def _function_has_charge(fn: ast.AST) -> bool:
    """函数体内(含嵌套)是否出现 check_and_increment / charge 调用。"""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in CHARGE_NAMES:
                return True
            if isinstance(f, ast.Name) and f.id in {"charge", "cost_guard_charge", "_charge"}:
                return True
    return False


# ============================================================
# 结构性保证: 每个请求点都计费
# ============================================================

def test_every_llm_request_site_is_charged():
    """
    每个 chat.completions.create 所在的函数链上必须有一次 charge。
    """
    missing: list[str] = []
    total_sites = 0
    for path in _iter_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in _find_create_calls(tree):
            total_sites += 1
            fn_names = _enclosing_function_names(tree, call)
            # 找到这些名字对应的函数节点, 看有没有 charge
            charged = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in fn_names:
                    if _function_has_charge(node):
                        charged = True
                        break
            if not charged:
                missing.append(f"{path.name}: {sorted(fn_names)}")

    assert total_sites >= 5, f"应至少找到 5 个 LLM 请求点, 实际 {total_sites}"
    assert not missing, (
        "以下 LLM 请求点所在函数没有计费 (cost_guard 绕过了):\n  " + "\n  ".join(missing)
    )


def test_no_duplicate_precharge_in_run_agent_stream():
    """
    回归: run_agent_stream 开头不能再预扣一次 ——
    第一轮会被 while 循环里的 charge 重复计费。
    """
    tree = ast.parse((REPO / "agent/controller.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_agent_stream"
    )

    # 找到 while True 循环体
    while_nodes = [n for n in ast.walk(fn) if isinstance(n, ast.While)]
    assert while_nodes, "run_agent_stream 应有 while 循环"

    # 循环体之前的顶层语句里不应再有 charge
    loop = while_nodes[0]
    body_start = fn.body.index(loop) if loop in fn.body else len(fn.body)
    pre_loop = fn.body[:body_start]
    for stmt in pre_loop:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name in CHARGE_NAMES:
                    pytest.fail(
                        "run_agent_stream 在 while 循环之前仍有计费调用, 第一轮会被重复扣费"
                    )


def test_max_rounds_check_precedes_charge():
    """
    回归: MAX_ROUNDS 检查必须排在 charge 之前。
    之前顺序反了 —— 被 MAX_ROUNDS 拦下的那一轮仍然扣了费。
    """
    tree = ast.parse((REPO / "agent/controller.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_agent_stream"
    )
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.While))

    order: list[str] = []
    for stmt in loop.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Compare):
                for c in ast.walk(node):
                    if isinstance(c, ast.Name) and c.id == "MAX_ROUNDS":
                        order.append("max_rounds")
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name in CHARGE_NAMES:
                    order.append("charge")

    assert "max_rounds" in order, "while 循环里应仍保留 MAX_ROUNDS 检查"
    assert "charge" in order, "while 循环里应仍有计费"
    assert order.index("max_rounds") < order.index("charge"), (
        f"MAX_ROUNDS 检查必须在 charge 之前, 实际顺序: {order}"
    )


# ============================================================
# charge() 收口函数
# ============================================================

def test_charge_helper_delegates_to_singleton():
    assert callable(cg_mod.charge)
    src = inspect.getsource(cg_mod.charge)
    assert "get_cost_guard" in src and "check_and_increment" in src


def test_charge_uses_configured_ceiling(monkeypatch):
    guard = cg_mod.CostGuard()
    monkeypatch.setattr(cg_mod, "get_cost_guard", lambda: guard)

    seen = []

    def fake_check(ceiling=None):
        seen.append(ceiling)
        return True, ""

    monkeypatch.setattr(guard, "check_and_increment", fake_check)
    cg_mod.charge()
    cg_mod.charge(ceiling=7)
    assert seen == [None, 7]


# ============================================================
# 行为层面: _call_llm 真的计费, 且被阻断时不发请求
# ============================================================

class _FakeMsg:
    def __init__(self, content):
        self.content = content
        self.reasoning_content = None


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content="ok"):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _FakeResp()


class _FakeClient:
    def __init__(self):
        self.chat = type("_Chat", (), {"completions": _FakeCompletions()})()


@pytest.fixture
def fake_llm(monkeypatch):
    """把 _call_llm 的 client / provider 解析 / 计费都替换掉。"""
    client = _FakeClient()
    monkeypatch.setattr(tools_llm, "get_client", lambda p: client)
    monkeypatch.setattr(tools_llm, "_resolve_chat_mode_params", lambda: ("deepseek", "m", False))
    monkeypatch.setattr(tools_llm, "resolve_legacy_model", lambda m, t, r="high": (m, t, r))
    monkeypatch.setitem(tools_llm.PROVIDERS, "deepseek", type(
        "P", (), {"default_model": "m", "display_name": "DeepSeek", "api_key": "k", "base_url": "u"}
    )())
    return client


def test_call_llm_charges_once_per_call(monkeypatch, fake_llm):
    """每调一次 _call_llm 就扣一次费。"""
    charges = []

    def fake_charge(ceiling=None):
        charges.append(ceiling)
        return True, ""

    monkeypatch.setattr(tools_llm, "cost_guard_charge", fake_charge)

    for _ in range(3):
        tools_llm._call_llm("prompt", provider="deepseek", model="m", enable_thinking=False)

    assert len(charges) == 3, "3 次 _call_llm 应扣 3 次费"


def test_call_llm_blocked_by_guard_does_not_call_api(monkeypatch, fake_llm):
    """被 cost_guard 阻断时不能发出请求。"""
    monkeypatch.setattr(
        tools_llm, "cost_guard_charge", lambda ceiling=None: (False, "❌ 已达上限")
    )

    out = tools_llm._call_llm("prompt", provider="deepseek", model="m", enable_thinking=False)

    assert fake_llm.chat.completions.calls == 0, "被阻断时不应发出 LLM 请求"
    assert "已达上限" in out
    assert "未发出 LLM 请求" in out


def test_call_llm_warns_but_proceeds(monkeypatch, fake_llm):
    """80% 警告时仍然放行。"""
    monkeypatch.setattr(
        tools_llm, "cost_guard_charge", lambda ceiling=None: (True, "⚠ 今日已用 45/50")
    )

    out = tools_llm._call_llm("prompt", provider="deepseek", model="m", enable_thinking=False)

    assert fake_llm.chat.completions.calls == 1, "80% 警告不应阻断"
    assert out == "ok"


def test_extractor_is_charged(monkeypatch):
    """记忆抽取的 LLM 调用也要计费。"""
    from agent.memory import extractor as ex_mod

    src = inspect.getsource(ex_mod._call_llm_extract)
    assert "charge" in src, "extractor._call_llm_extract 应调用 cost_guard.charge"


def test_extractor_blocked_returns_none(monkeypatch):
    """被阻断时返回 None, 不发请求。"""
    from agent.memory import extractor as ex_mod

    class _BoomClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise AssertionError("被阻断时不应发出 LLM 请求")

    monkeypatch.setattr("agent.llm_clients.get_client", lambda p: _BoomClient())
    monkeypatch.setattr(
        "agent.cost_guard.charge", lambda ceiling=None: (False, "❌ 已达上限")
    )

    assert ex_mod._call_llm_extract("some content") is None
