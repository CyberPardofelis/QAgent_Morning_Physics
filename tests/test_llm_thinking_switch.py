"""`_call_llm` 的 MiniMax thinking 开关 —— 10-07 真跑定位到的根因。

**为什么这个 bug 很贵**: `enable_thinking=False` 时原来什么都不传, 而
MiniMax-M3 默认 adaptive thinking。实测(经典阴影骨架 prompt, 31722 字符):

    不传 flag:      content=33715 字符, completion_tokens=10761, 105s
    type=disabled:  content= 1090 字符, completion_tokens=  499,  14s

JSON 结果两边一模一样 —— 97% 的输出和 87% 的等待全是白花的。

**怎么咬人的**: think 吃 max_tokens。骨架原来 `max_tokens=8000`, 96% 被 think
占掉, 答案 JSON 被截断 -> 第二轮退回无骨架的块级归纳, 第一轮的领域结构被丢弃。

顺带: M3 的 `thinking.type` 显式传 `"enabled"` 会 **400**
(`invalid thinking.type: "enabled"`), 实测确认过。所以 minimax +
enable_thinking=True 时必须走默认 adaptive, 不能走 controller 里 deepseek 那条。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent.tools.llm as llm_mod  # noqa: E402
from agent.tools.llm import _call_llm  # noqa: E402


class _FakeMsg:
    def __init__(self, content="ok"):
        self.content = content
        self.reasoning_content = None


class _FakeChoice:
    def __init__(self, content="ok"):
        self.message = _FakeMsg(content)
        self.finish_reason = "stop"


class _FakeResp:
    def __init__(self, content="ok"):
        self.choices = [_FakeChoice(content)]
        self.usage = None


class _FakeCompletions:
    def __init__(self, sink):
        self._sink = sink

    def create(self, **kwargs):
        self._sink.append(kwargs)
        return _FakeResp()


class _FakeClient:
    def __init__(self, sink):
        self.chat = type("chat", (), {})()
        self.chat.completions = _FakeCompletions(sink)


@pytest.fixture
def sent(monkeypatch):
    """装上假 client, 收集每次 create() 收到的 kwargs。"""
    calls: list[dict] = []
    monkeypatch.setattr(llm_mod, "get_client", lambda provider: _FakeClient(calls))
    monkeypatch.setattr(llm_mod, "cost_guard_charge",
                        lambda: (True, ""), raising=False)
    return calls


def _call(**kw):
    return _call_llm("prompt", provider="minimax", model="MiniMax-M3", **kw)


class TestMinimaxThinkingDisabled:
    def test_disabled_flag_is_sent(self, sent):
        """enable_thinking=False 必须显式发 disabled, 否则 M3 默认会思考。"""
        _call(enable_thinking=False)
        assert sent, "应该发出去了"
        eb = sent[0].get("extra_body")
        assert eb == {"thinking": {"type": "disabled"}}, (
            f"minimax + enable_thinking=False 时 extra_body={eb!r}, "
            f"期望 {{'thinking': {{'type': 'disabled'}}}} —— 不发的话 M3 用默认 "
            f"adaptive thinking, 实测 96% 的输出会变成 <think> 段")

    def test_never_sends_enabled_to_minimax(self, sent):
        """enable_thinking=True 时**不能**发 "enabled" —— M3 会 400。"""
        _call(enable_thinking=True)
        eb = sent[0].get("extra_body")
        assert not (eb or {}).get("thinking"), (
            f"minimax 不接受 thinking.type=enabled(实测报 400: "
            f"invalid thinking.type: \"enabled\"), 实际发了 {eb!r}")

    def test_no_penalties_for_minimax(self, sent):
        """minimax 对这两个参数支持有限, 发了会 400。"""
        _call(enable_thinking=False, frequency_penalty=0.3, presence_penalty=0.3)
        assert "frequency_penalty" not in sent[0]
        assert "presence_penalty" not in sent[0]


class TestOtherProvidersUnchanged:
    def test_deepseek_enabled_still_sends_enabled(self, sent):
        """deepseek 走原来的 enabled 分支, 不能被 minimax 的改动带偏。"""
        _call_llm("p", provider="deepseek", model="deepseek-v4-flash",
                  enable_thinking=True, reasoning_effort="high")
        assert sent[0].get("extra_body") == {"thinking": {"type": "enabled"}}
        assert sent[0].get("reasoning_effort") == "high"

    def test_deepseek_disabled_sends_no_extra_body(self, sent):
        _call_llm("p", provider="deepseek", model="deepseek-v4-flash",
                  enable_thinking=False)
        assert "extra_body" not in sent[0]

    def test_deepseek_disabled_keeps_penalties(self, sent):
        """非 minimax 不该被剥掉 penalty —— 这是 minmax 分支的局部改动。"""
        _call_llm("p", provider="deepseek", model="deepseek-v4-flash",
                  enable_thinking=False, frequency_penalty=0.3,
                  presence_penalty=0.3, top_p=0.5)
        assert sent[0].get("frequency_penalty") == 0.3
        assert sent[0].get("top_p") == 0.5


class TestRegressionGuard:
    def test_matches_controller_branch(self):
        """和 controller.py 的 minimax 分支保持一致(源级断言)。

        controller.py:482 早就写对了, 但**只存在于主界面对话路径**, 工具内部
        调用路径一直漏着 —— 这就是本 bug 的来历。两边必须同步改, 所以直接
        钉住 controller 那边的写法。
        """
        ctrl = Path(llm_mod.__file__).resolve().parents[1] / "controller.py"
        src = ctrl.read_text(encoding="utf-8")
        assert '"thinking": {"type": "disabled"}' in src, (
            "controller.py 里的 minimax 关闭分支被改掉了 —— "
            "_call_llm 是照着它对齐的, 两边要一起动")

    def test_timeout_still_passthrough(self, sent):
        """timeout 透传没被这次改动碰坏(9-04 修过一次, 别回退)。"""
        _call(enable_thinking=False, timeout=240)
        assert sent[0]["timeout"] == 240

    def test_max_tokens_still_passthrough(self, sent):
        _call(enable_thinking=False, max_tokens=32000)
        assert sent[0]["max_tokens"] == 32000