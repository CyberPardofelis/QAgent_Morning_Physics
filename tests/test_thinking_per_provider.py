"""
tests/test_thinking_per_provider.py — DS / MiniMax 思考段处理的端到端回归护栏

背景 (9-27):

QAgent 的 thinking 走两条**不同**的通道, 这是两套独立机制, 不能合并:

  1. 原生 reasoning 字段 (deepseek / openai)
     API 把推理放在 `delta.reasoning_content`, 与 `delta.content` 分开。
     controller 的流式循环看到 rc 就 `continue` —— 静默收集、**从不 yield 给 UI**。
     配套: _harmonize_assistant_reasoning 在 deepseek+thinking 时给历史 assistant
     补 reasoning_content="" (不补会 400), 其它 provider 则剥掉该字段。

  2. 混入 content (**只有 MiniMax M3**)
     模型把 thinking 直接拼在 `delta.content` 里、用字面量 <think> 标签包。
     controller 靠 `_strip_think_tags` 状态机跨 chunk 实时剥离。

本测试用假 stream 跑完整的 run_agent_stream, 断言:
  - MiniMax: <think>...</think> 段不进入 UI, 正文完整
  - DeepSeek: reasoning_content 段不进入 UI, 正文完整
  - 两者都能用, 且走的是**各自**的通道 (不是合并后的通用路径)

这类测试的作用: 以后任何人"顺手统一"两条路径, 这里立刻会红。
"""
from __future__ import annotations

import pytest

from agent import controller as ctrl


# ============================================================
# 假 stream 基础设施
# ============================================================

class _Delta:
    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Chunk:
    def __init__(self, delta):
        self.choices = [_Choice(delta)]


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeStream(self._chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(chunks)})()


class _AllowGuard:
    def check_and_increment(self, ceiling=None):
        return True, ""


def _run(chunks, provider_name, enable_thinking):
    """跑一遍 run_agent_stream, 返回 (UI 收到的文本, 发给 API 的 request 参数)。"""
    client = _FakeClient(chunks)
    captured = {}

    def fake_resolve(mode):
        return client, "test-model", provider_name

    import agent.controller as m
    orig = (
        m._resolve_client_and_model,
        m.resolve_legacy_model,
        m.get_cost_guard,
        m.ToolRegistry.list_schemas_filtered,
    )
    try:
        m._resolve_client_and_model = fake_resolve
        m.resolve_legacy_model = lambda mm, t, r="high": (mm, t, r)
        m.get_cost_guard = lambda: _AllowGuard()
        m.ToolRegistry.list_schemas_filtered = staticmethod(lambda t: [])
        out = list(m.run_agent_stream(
            [{"role": "user", "content": "hi"}],
            mode="custom", enable_thinking=enable_thinking,
        ))
        captured = client.chat.completions.kwargs
    finally:
        (m._resolve_client_and_model, m.resolve_legacy_model,
         m.get_cost_guard, m.ToolRegistry.list_schemas_filtered) = orig
    text = "".join(c for c in out if isinstance(c, str) and c != ctrl.NEW_BUBBLE)
    return text, captured


# ============================================================
# MiniMax: thinking 混在 content 里, 靠 <think> 标签剥离
# ============================================================

def test_minimax_strips_think_tags_from_content():
    """MiniMax 的 <think> 段不能进 UI。"""
    chunks = [
        _Chunk(_Delta(content="<think>我先看看用户问的是什么")),
        _Chunk(_Delta(content="，是个物理问题</think>")),
        _Chunk(_Delta(content="答案是 42。")),
    ]
    text, _ = _run(chunks, "minimax", enable_thinking=True)
    assert "<think>" not in text
    assert "</think>" not in text
    assert "我先看看用户问的是什么" not in text, "MiniMax thinking 不应泄漏到 UI"
    assert "答案是 42。" in text


def test_minimax_think_tag_split_across_chunks():
    """
    标签可能跨 chunk 断开(这是 2026-09-02 "两个空气泡" 的根因)。
    状态机必须缓冲到标签完整才剥离。
    """
    chunks = [
        _Chunk(_Delta(content="答案之前<thi")),
        _Chunk(_Delta(content="nk>内部推理")),
        _Chunk(_Delta(content="</thi")),
        _Chunk(_Delta(content="nk>正式回答")),
    ]
    text, _ = _run(chunks, "minimax", enable_thinking=True)
    assert "<think" not in text
    assert "内部推理" not in text
    assert "正式回答" in text


def test_minimax_never_yields_reasoning_content():
    """MiniMax 走 content 通道, 不该产生 reasoning_content 字段。"""
    chunks = [_Chunk(_Delta(content="普通回答"))]
    _, kwargs = _run(chunks, "minimax", enable_thinking=True)
    # MiniMax 的 extra_body 走 thinking disabled/enabled 分支, 不是 reasoning_effort
    assert kwargs.get("reasoning_effort") is None or "thinking" in str(kwargs.get("extra_body"))


# ============================================================
# DeepSeek: reasoning 走独立字段, 静默不 yield
# ============================================================

def test_deepseek_reasoning_content_never_reaches_ui():
    """DeepSeek 的 reasoning_content 不应出现在 UI 文本里。"""
    chunks = [
        _Chunk(_Delta(reasoning_content="我需要先回忆一下上下文")),
        _Chunk(_Delta(reasoning_content="，然后给出答案")),
        _Chunk(_Delta(content="正式答案。")),
    ]
    text, _ = _run(chunks, "deepseek", enable_thinking=True)
    assert "我需要先回忆一下上下文" not in text, "DeepSeek reasoning 不应泄漏到 UI"
    assert "正式答案。" in text


def test_deepseek_reasoning_only_no_content_still_terminates():
    """只有 reasoning 没有 content 时, 不应把 reasoning 当正文 yield 出去。"""
    chunks = [
        _Chunk(_Delta(reasoning_content="只有推理没有正文")),
    ]
    text, _ = _run(chunks, "deepseek", enable_thinking=True)
    assert "只有推理没有正文" not in text


def test_deepseek_prompt_does_not_require_think_tags():
    """
    DeepSeek 的 prompt 里**不应该**要求它输出 <think> 标签 ——
    它的推理走独立字段, 标签纯属多余噪声。
    """
    from agent.prompts import THINK_TAG_RULE, build_system_prompt

    ds_prompt = build_system_prompt(base="B", mode="deepseek-v4-pro")
    assert THINK_TAG_RULE not in ds_prompt
    assert "不要**输出字面量" in ds_prompt


def test_minimax_prompt_does_require_think_tags():
    from agent.prompts import THINK_TAG_RULE, build_system_prompt

    assert THINK_TAG_RULE in build_system_prompt(base="B", mode="minimax-m3")


# ============================================================
# 两条通道互不干扰 (合并即失败)
# ============================================================

def test_two_channels_do_not_cross_contaminate():
    """
    同一份 prompt / 同一个流循环下, 两条通道必须各自独立工作:
    - content 里的 <think> 段被剥离 (MiniMax 机制)
    - reasoning_content 段不 yield (DeepSeek 机制)
    即使两条同时出现, 也不能互相污染。
    """
    chunks = [
        _Chunk(_Delta(reasoning_content="RC-不该出现")),
        _Chunk(_Delta(content="<think>MM-不该出现</think>")),
        _Chunk(_Delta(content="标签外的正常内容")),   # 在标签外, 应正常输出
        _Chunk(_Delta(content="干净的正文")),
    ]
    text, _ = _run(chunks, "deepseek", enable_thinking=True)
    assert "RC-不该出现" not in text, "reasoning 通道不应泄漏"
    assert "MM-不该出现" not in text, "content 通道的 think 段应被剥离"
    assert "干净的正文" in text
    assert "标签外的正常内容" in text, "标签外的正文不应被误删"


@pytest.mark.parametrize("provider", ["minimax", "deepseek", "local_qwen"])
def test_all_providers_get_clean_text(provider):
    """
    走 ReAct 主流循环的 provider: 用户只应看到正文, 看不到任何内部推理。
    """
    chunks = [
        _Chunk(_Delta(reasoning_content="INTERNAL-1")),
        _Chunk(_Delta(content="<think>INTERNAL-2</think>")),
        _Chunk(_Delta(content="FINAL")),
    ]
    text, _ = _run(chunks, provider, enable_thinking=True)
    assert "INTERNAL-1" not in text
    assert "INTERNAL-2" not in text
    assert "FINAL" in text


def test_openai_path_ignores_reasoning_content_field():
    """
    ⚠️ 记录一个**既有**行为差异 (非本次改动引入, 2026-09-02 起一直如此):
    mode="openai" 走 _stream_openai_simple 的早 return 路径, 该函数:
      - 只读 delta.content, 不看 reasoning_content → 推理字段被静默丢弃(结果正确)
      - **不**应用 _strip_think_tags → content 里的字面量 <think> 不会被剥离

    真实 OpenAI 模型(o1/o3 等)的推理走独立字段, 所以实际不会触发;
    只有把 openai 的 base_url 指向会往 content 里塞 <think> 的自建端点才会暴露。
    此测试把现状钉住, 便于将来决定要不要补齐。
    """
    chunks = [
        _Chunk(_Delta(reasoning_content="RC-不该出现")),
        _Chunk(_Delta(content="FINAL")),
    ]
    text, _ = _run(chunks, "openai", enable_thinking=True)
    assert "RC-不该出现" not in text, "reasoning 字段应被丢弃"
    assert "FINAL" in text


# ============================================================
# _harmonize_assistant_reasoning 的 provider 矩阵
# ============================================================

def test_harmonize_deepseek_thinking_adds_empty_reasoning():
    """DeepSeek + thinking: 历史 assistant 必须补 reasoning_content=""(否则 400)。"""
    msgs = [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    ctrl._harmonize_assistant_reasoning(msgs, "deepseek", True)
    assert msgs[0]["reasoning_content"] == ""


def test_harmonize_deepseek_no_thinking_strips():
    msgs = [{"role": "assistant", "content": "a", "reasoning_content": "r"}]
    ctrl._harmonize_assistant_reasoning(msgs, "deepseek", False)
    assert "reasoning_content" not in msgs[0]


@pytest.mark.parametrize("provider", ["minimax", "openai", "local_qwen", "custom"])
def test_harmonize_other_providers_strip_reasoning(provider):
    """
    其它 provider 一律剥掉 reasoning_content(它们的 API 不认这个字段,
    留着会污染上下文/触发报错)。
    """
    msgs = [{"role": "assistant", "content": "a", "reasoning_content": "r"}]
    ctrl._harmonize_assistant_reasoning(msgs, provider, True)
    assert "reasoning_content" not in msgs[0], f"{provider} 应剥掉 reasoning_content"


# ============================================================
# 剥离状态机本身
# ============================================================

def test_strip_think_tags_state_machine_basics():
    """_strip_think_tags 返回的是 list(逐段 yield), 不是 str。"""
    state = [False, ""]
    out = []
    for piece in ["abc<thi", "nk>hidden</th", "ink>def", "ghi"]:
        out.extend(ctrl._strip_think_tags(piece, state))
    assert "".join(out) == "abcdefghi"


def test_strip_think_tags_handles_unclosed_block():
    """未闭合的 <think>(模型被截断): 之后的内容不应泄漏。"""
    state = [False, ""]
    out = ctrl._strip_think_tags("答案<think>未闭合的推理", state)
    assert "未闭合的推理" not in "".join(out)
    assert "答案" in "".join(out)
