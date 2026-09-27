"""
tests/test_system_prompt_wiring.py — system prompt 组装 + schema 一致性测试

9-27 P0 修复配套测试。

背景:
    agent/prompts.py 里的 build_system_prompt() / BASE_RULES / TOOL_GUIDE 是
    组装最终 system prompt 的唯一入口, 但它**零调用点** —— 实际发出去的是裸的
    SYSTEM_PROMPT(837 字符)。

    后果: <think> 标签协议(整个流式层依赖)、"DIRECT_OUTPUT 工具不要输出 prose"、
    save_memory 触发词表、工具选择决策树 —— 一条都没进过 prompt。
    controller 里那套 DIRECT_OUTPUT 散文丢弃 hack 正是为了绕开这个缺失而写的。

    同时 TOOL_GUIDE 相对 tool_definitions.py 已经漂移, 里面有几处会让 LLM
    必然调用失败的指令。直接接上而不先修漂移 = 把坏指令喂给模型。

本测试保证:
  - build_system_prompt 真的被接上了, 且内容真的进了发出去的 system 消息
  - 分组 prompt 是"额外指令"而不是整体替换
  - TOOL_GUIDE 里提到的工具/参数在 schema 里都真实存在
"""
from __future__ import annotations

import re

import pytest

from agent.prompts import (
    BASE_RULES,
    REASONING_NATIVE_NOTE,
    REASONING_UNKNOWN_NOTE,
    SYSTEM_PROMPT,
    THINK_TAG_RULE,
    TOOL_GUIDE,
    _thinking_rule_for,
    build_system_prompt,
)
from agent.tool_definitions import TOOL_SCHEMAS


# ============================================================
# build_system_prompt 本身
# ============================================================

def test_build_system_prompt_contains_all_parts():
    out = build_system_prompt(base="MY-BASE", group_rules="MY-GROUP", mode="minimax-m3")
    assert "MY-BASE" in out
    assert BASE_RULES in out
    assert THINK_TAG_RULE in out
    assert TOOL_GUIDE in out
    assert "MY-GROUP" in out
    assert "当前分组额外指令" in out


def test_build_system_prompt_includes_current_time():
    out = build_system_prompt(base="x")
    assert "当前系统时间" in out
    assert re.search(r"\d{4}年\d{2}月\d{2}日", out), "应包含当前日期"


def test_build_system_prompt_omits_group_section_when_empty():
    out = build_system_prompt(base="x", group_rules="   ")
    assert "当前分组额外指令" not in out


# ============================================================
# 真正接上了吗
# ============================================================

def test_build_system_prompt_has_a_call_site():
    """
    回归: build_system_prompt 必须有调用点。
    之前它是零调用点的死函数 —— 整个 BASE_RULES/TOOL_GUIDE 都没发出去过。
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    # 只扫源码目录, 别扫 venv / 数据目录(否则会遍历上万文件)
    skip_dirs = {"venv", "__pycache__", "tests", "data", ".git", "docs", "logs", "resources"}
    hits = []
    for top in ("agent", "ui", "config"):
        for path in (repo / top).rglob("*.py"):
            if set(path.parts) & skip_dirs:
                continue
            try:
                src = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if "build_system_prompt(" in src and "def build_system_prompt" not in src:
                hits.append(path.relative_to(repo).as_posix())
    assert hits, "build_system_prompt 仍然没有任何调用点（死函数）"


def test_conversation_controller_uses_build_system_prompt():
    from ui import conversation_controller as cc

    src = cc.ConversationController._get_current_system_prompt.__doc__ or ""
    assert "build_system_prompt" in src


# ============================================================
# 分组 prompt 是"额外指令"而非整体替换
# ============================================================

class _FakeConv:
    def __init__(self, group_id):
        self.group_id = group_id


class _FakeSettings:
    def __init__(self, data):
        self._data = data

    def value(self, key, default=None, type=None):
        return self._data.get(key, default)

    def setValue(self, key, value):
        self._data[key] = value


def _make_controller(monkeypatch, group_id, group_prompt):
    from PySide6.QtCore import QObject

    from ui import conversation_controller as cc

    # ConversationController 继承 QObject, 必须走 QObject.__new__(cls, ...)
    ctrl = QObject.__new__(cc.ConversationController)
    ctrl.default_system_prompt = "DEFAULT-PROMPT-MARKER"
    ctrl.current_conversation = _FakeConv(group_id)
    monkeypatch.setattr(cc, "get_settings", lambda: _FakeSettings(
        {"group_system_prompt/g1": group_prompt} if group_prompt else {}
    ))
    return ctrl


def test_group_prompt_does_not_replace_base(monkeypatch):
    """
    回归: 设了分组 prompt 后, 基础 prompt + BASE_RULES + TOOL_GUIDE 必须还在。
    之前分组 prompt 会整体顶掉 default_system_prompt —— 用户一设分组就丢掉
    <think> 协议等所有默认规则。
    """
    ctrl = _make_controller(monkeypatch, "g1", "分组的额外要求")
    out = ctrl._get_current_system_prompt()

    assert "DEFAULT-PROMPT-MARKER" in out
    assert "分组的额外要求" in out
    assert "当前分组额外指令" in out
    assert BASE_RULES in out
    assert TOOL_GUIDE in out


def test_no_group_prompt_still_gets_rules(monkeypatch):
    ctrl = _make_controller(monkeypatch, None, None)
    out = ctrl._get_current_system_prompt()
    assert "DEFAULT-PROMPT-MARKER" in out
    assert BASE_RULES in out
    assert TOOL_GUIDE in out


# ============================================================
# 关键规则真的进了 prompt
# ============================================================

@pytest.mark.parametrize("needle", [
    "save_memory",             # 记忆写入触发
    "DIRECT_OUTPUT",           # 每日总结工具的输出规则
])
def test_critical_rules_are_present(needle):
    out = build_system_prompt(base=SYSTEM_PROMPT)
    assert needle in out, f"关键规则 {needle!r} 应出现在 system prompt 里"


# ============================================================
# thinking 协议必须按 provider 分发 (9-27 修正)
# ============================================================
# 背景: QAgent 的 thinking 有两条通道 —
#   1. 原生 reasoning 字段 (deepseek/openai/多数 OpenAI 兼容端点):
#      API 把推理放 reasoning_content, 与 content 分开。controller 静默累加、
#      从不 yield 给 UI, 所以模型什么都不用做。
#   2. 混入 content (只有 MiniMax M3): 靠 <think> 标签 + 状态机剥离。
#
# 如果对所有 provider 都要求 <think> 标签, deepseek/openai 会白白多吐几千
# token 的字面量标签 —— 它们的推理根本不走 content。

def test_minimax_gets_think_tag_protocol():
    """MiniMax 的 thinking 混在 content 里, 必须要求 <think> 标签。"""
    out = build_system_prompt(base="x", mode="minimax-m3")
    assert THINK_TAG_RULE in out
    assert "必须包在" in out


@pytest.mark.parametrize("mode", ["deepseek-v4-pro", "deepseek-v4-flash", "openai"])
def test_native_reasoning_providers_must_not_be_asked_for_tags(mode):
    """
    回归: 原生 reasoning 通道的 provider 不能被要求输出 <think> 标签。
    它们应该被明确告知"不需要标记"。
    """
    out = build_system_prompt(base="x", mode=mode)
    assert THINK_TAG_RULE not in out, f"{mode} 不该收到 <think> 标签协议"
    assert REASONING_NATIVE_NOTE in out
    assert "不要**输出字面量" in out, "应明确禁止输出字面量 <think> 标签"


@pytest.mark.parametrize("mode", ["custom", None])
def test_unknown_provider_gets_neutral_note(mode):
    """未知端点不臆测协议, 只说"不需要标记"。"""
    out = build_system_prompt(base="x", mode=mode)
    assert THINK_TAG_RULE not in out
    assert REASONING_NATIVE_NOTE not in out
    assert REASONING_UNKNOWN_NOTE in out


def test_base_rules_are_provider_neutral():
    """BASE_RULES 本身不应含任何 provider 特定的 thinking 指令。"""
    assert "<think>" not in BASE_RULES
    assert "MiniMax" not in BASE_RULES
    assert "deepseek" not in BASE_RULES.lower()


@pytest.mark.parametrize("provider,expected", [
    ("minimax", THINK_TAG_RULE),
    ("deepseek", REASONING_NATIVE_NOTE),
    ("openai", REASONING_NATIVE_NOTE),
    ("local_qwen", REASONING_UNKNOWN_NOTE),
    (None, REASONING_UNKNOWN_NOTE),
])
def test_thinking_rule_dispatch(provider, expected):
    assert _thinking_rule_for(provider) == expected


def test_explicit_provider_overrides_mode():
    """provider 显式传参时优先于 mode。"""
    out = build_system_prompt(base="x", mode="deepseek-v4-pro", provider="minimax")
    assert THINK_TAG_RULE in out


def test_only_minimax_modes_map_to_tag_protocol():
    """
    遍历 controller 里所有已注册的 mode, 确保只有 minimax 系拿到标签协议。
    这是防止以后新增 mode 时静默拿错规则的护栏。
    """
    from agent.controller import MODE_TO_PROVIDER_MODEL

    for mode in MODE_TO_PROVIDER_MODEL:
        out = build_system_prompt(base="x", mode=mode)
        provider = MODE_TO_PROVIDER_MODEL[mode][0]
        if provider == "minimax":
            assert THINK_TAG_RULE in out, f"{mode} (minimax) 应收到 <think> 协议"
        else:
            assert THINK_TAG_RULE not in out, f"{mode} ({provider}) 不应收到 <think> 协议"


def test_conversation_controller_passes_mode_to_builder():
    """两个调用点都必须把 mode 传下去, 否则全部退化成"未知 provider"。"""
    from ui import conversation_controller as cc

    src = cc.ConversationController._get_current_system_prompt.__doc__ or ""
    assert "mode" in src

    import ast
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    tree = ast.parse((repo / "ui/conversation_controller.py").read_text(encoding="utf-8"))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_get_current_system_prompt"
    ]
    assert calls, "找不到 _get_current_system_prompt 调用点"
    for call in calls:
        args = {a.id for a in call.args if isinstance(a, ast.Name)}
        assert "mode" in args, "调用 _get_current_system_prompt 时必须传 mode"


# ============================================================
# TOOL_GUIDE 不再漂移: 提到的参数必须真实存在
# ============================================================

def _schema_props(name):
    return set(TOOL_SCHEMAS[name]["function"]["parameters"].get("properties", {}))


def test_tool_guide_mentions_only_real_tools():
    """TOOL_GUIDE 里点名的工具必须都在 schema 里注册过。"""
    mentioned = set(re.findall(r"`([a-z_][a-z0-9_]{3,})\(", TOOL_GUIDE))
    known = set(TOOL_SCHEMAS.keys())
    unknown = {
        t for t in mentioned
        if t not in known and not t.startswith(("system", "content", "metadata"))
    }
    assert not unknown, f"TOOL_GUIDE 提到了未注册的工具: {sorted(unknown)}"


@pytest.mark.parametrize("tool,param", [
    ("recall_memories", "mmr_lambda"),   # 之前指导 LLM 传这个, 但 handler 没这个参数 → 必错
    ("query_knowledge", "paper_id"),     # 之前指导 LLM 传这个, 但 handler 不接受   → 必错
    ("semantic_search_keywords", "year"),        # 漂移
    ("semantic_search_keywords", "topic_keyword"),  # 漂移
])
def test_tool_guide_does_not_teach_nonexistent_params(tool, param):
    """
    回归: TOOL_GUIDE 不能指导 LLM 传 schema 里不存在的参数 ——
    那会让每次调用都必然报"参数错误"，白烧一整轮 ReAct。
    """
    props = _schema_props(tool)
    assert param not in props, f"{tool} 实际没有 {param} 参数"
    # 并且 guide 里不能再教用它
    pattern = rf"`{tool}\([^`]*\b{param}\b"
    assert not re.search(pattern, TOOL_GUIDE), (
        f"TOOL_GUIDE 仍在指导 LLM 传 {tool}({param}=...), 该参数并不存在"
    )


def test_recall_memories_guide_lists_real_params():
    """recall_memories 的说明应覆盖 schema 里真实的可选参数。"""
    props = _schema_props("recall_memories")
    for p in props:
        assert p in TOOL_GUIDE, f"TOOL_GUIDE 未提及 recall_memories 的 {p} 参数"


def test_save_memory_guide_does_not_hardcode_threshold():
    """
    回归: 阈值常量(SIM_MERGE_THRESHOLD 等)会调整, guide 里不能写死数字 ——
    之前写死 0.95, 实际常量是 0.99, LLM 会按错误阈值预判。
    """
    assert "sim >= 0.95" not in TOOL_GUIDE
    assert "SIM_MERGE_THRESHOLD = 0.99" not in TOOL_GUIDE
    # 应改为"以工具实际返回的提示为准"
    assert "以 `save_memory` 工具实际返回的提示为准" in TOOL_GUIDE


def test_save_memory_docstring_does_not_hardcode_threshold():
    """同样约束 tool docstring（LLM 直接读它）。"""
    from agent.tools.memory import save_memory

    doc = save_memory.__doc__ or ""
    assert "相似度 >= 0.95" not in doc
    assert "SIM_MERGE_THRESHOLD" in doc
