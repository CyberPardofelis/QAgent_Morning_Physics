"""
test_controller_state_machine — controller 核心状态机测试 (2026-09-21 加)

覆盖 2 个关键状态机:
  1. _harmonize_assistant_reasoning — DeepSeek thinking mode 必填字段修补
  2. _strip_think_tags_from_text — MiniMax thinking 标签剥离

为什么重要: 之前两次重大事故 (minimax 1-2s 空气泡 + DeepSeek 400) 都是这里修的。
回退会让 LLM 流式渲染输出污染 + 第二轮直接 400。
"""
from agent.controller import (
    _harmonize_assistant_reasoning,
    _strip_think_tags_from_text,
    _strip_think_tags,
)


class TestHarmonizeAssistantReasoning:
    """测试 _harmonize_assistant_reasoning: DeepSeek thinking 模式字段修补

    行为矩阵 (来自 docstring):
      deepseek + enable_thinking=True  → 补 reasoning_content=""
      deepseek + enable_thinking=False → 剥掉 reasoning_content
      minimax (任何 thinking)            → 剥掉 reasoning_content
      openai / compare / custom         → 剥掉 reasoning_content
    """

    def test_dense_bury_empty_string_when_thinking(self):
        """deepseek + thinking=True: 缺的 assistant 补 reasoning_content=''"""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "assistant", "content": "again"},  # 缺 reasoning_content
        ]
        _harmonize_assistant_reasoning(messages, "deepseek", True)
        assert messages[1].get("reasoning_content") == ""
        assert messages[2].get("reasoning_content") == ""

    def test_dense_keep_existing_reasoning(self):
        """deepseek + thinking=True: 已有 reasoning_content 不动"""
        messages = [
            {"role": "assistant", "content": "x", "reasoning_content": "old"},
        ]
        _harmonize_assistant_reasoning(messages, "deepseek", True)
        assert messages[0]["reasoning_content"] == "old"

    def test_dense_strip_when_thinking_off(self):
        """deepseek + thinking=False: 剥掉所有 reasoning_content"""
        messages = [
            {"role": "assistant", "content": "x", "reasoning_content": "deep"},
        ]
        _harmonize_assistant_reasoning(messages, "deepseek", False)
        assert "reasoning_content" not in messages[0]

    def test_minimax_strips_always(self):
        """minimax: 无论 thinking 设什么, 剥掉 reasoning_content"""
        for thinking in [True, False]:
            messages = [
                {"role": "assistant", "content": "x", "reasoning_content": f"m{thinking}"},
            ]
            _harmonize_assistant_reasoning(messages, "minimax", thinking)
            assert "reasoning_content" not in messages[0]

    def test_openai_strips(self):
        """openai: 剥掉 reasoning_content"""
        messages = [
            {"role": "assistant", "content": "x", "reasoning_content": "oa"},
        ]
        _harmonize_assistant_reasoning(messages, "openai", True)
        assert "reasoning_content" not in messages[0]

    def test_in_place_mutation(self):
        """函数就地修改, 不返新 list"""
        messages = [{"role": "assistant", "content": "x"}]
        original_id = id(messages)
        result = _harmonize_assistant_reasoning(messages, "deepseek", True)
        assert result is None  # 不返任何东西
        assert id(messages) == original_id  # 同一个 list

    def test_user_messages_not_modified(self):
        """只动 assistant 消息, user 消息原样"""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        _harmonize_assistant_reasoning(messages, "deepseek", True)
        assert "reasoning_content" not in messages[0]  # user 不补
        assert messages[1]["reasoning_content"] == ""  # assistant 补


class TestStripThinkTagsFromText:
    """测试 _strip_think_tags_from_text: 整段文本剥离 think 标签

    vs _strip_think_tags (流式状态机): 这个是一次性剥离,
    history 加载 / 发送 LLM 前 / UI 旧对话回放 时用。
    """

    def test_simple_think_block(self):
        """最简单: 一段文本夹一个 think 块"""
        text = "Before<think>let me think</think>after"
        assert _strip_think_tags_from_text(text) == "Beforeafter"

    def test_no_think_block(self):
        """没 think 标签 → 原样返回 (零拷贝)"""
        text = "Just plain text without any tags"
        assert _strip_think_tags_from_text(text) == text

    def test_empty_text(self):
        assert _strip_think_tags_from_text("") == ""

    def test_none_input(self):
        # 函数应该容错处理 None
        result = _strip_think_tags_from_text(None)  # type: ignore[arg-type]
        assert result is None or result == ""

    def test_multiline_think_block(self):
        """多行 think 块 (DOTALL 应能跨越行)

        实现用 regex sub(r"<think>.*?</think>", "", text).strip(),
        strip 只剥首尾空白, 中间的换行保留
        """
        text = "Intro<think>\nline 1\nline 2\n</think>\nOutro"
        result = _strip_think_tags_from_text(text)
        # 容错: 包含 Intro + Outro, 不含 think 内容
        assert "Intro" in result
        assert "Outro" in result
        assert "<think>" not in result
        assert "line 1" not in result
        assert "line 2" not in result

    def test_multiple_think_blocks(self):
        """多个 think 块都被剥"""
        text = "A<think>1</think>B<think>2</think>C"
        assert _strip_think_tags_from_text(text) == "ABC"

    def test_think_block_at_start(self):
        """think 块在开头"""
        text = "<think>first</think>rest"
        assert _strip_think_tags_from_text(text) == "rest"

    def test_think_block_at_end(self):
        """think 块在末尾"""
        text = "start<think>end</think>"
        assert _strip_think_tags_from_text(text) == "start"


class TestStripThinkTagsStream:
    """测试 _strip_think_tags: 流式状态机 (跨 chunk)"""

    def test_initial_state_normal(self):
        """初始 state [False, '']: chunk 不含 think 标签直接 yield"""
        state = [False, ""]
        out = _strip_think_tags("hello world", state)
        assert out == ["hello world"]
        assert state == [False, ""]

    def test_state_persists_across_chunks(self):
        """state 跨 chunk 持续 (累积到 state[1])"""
        state = [False, ""]
        # chunk 1: "<think>thinking"
        out1 = _strip_think_tags("<think>thinking", state)
        # chunk 2: "</think>rest"
        out2 = _strip_think_tags("</think>rest", state)

        # chunk 1: 进入 thinking 状态, 不 yield
        assert out1 == []
        # chunk 2: 退出 thinking, yield "rest"
        assert out2 == ["rest"]
        # state 应该重置为 NORMAL
        assert state == [False, ""]

    def test_chunk_split_open_tag(self):
        """chunk 1: '<th' (open 标签前缀), chunk 2: 'ink>rest'

        实现: chunk 1 累积 "<th", 不 yield (waiting for open tag prefix)
        chunk 2 完成 open 标签后立即进入 thinking 状态, 'rest' 也被 thinking 包住,
        等到 chunk 3 出现 </think> 才 yield.
        """
        state = [False, ""]
        out1 = _strip_think_tags("<th", state)
        # chunk 1: open 标签前缀, 等待, 不 yield
        assert out1 == []
        # state[1] 应该有累积的 "<th"
        assert state[1] == "<th"

        out2 = _strip_think_tags("ink>rest", state)
        # chunk 2: 完成 open 标签, 进入 thinking 状态
        # 但 "rest" 也被 thinking 包住, 等到 close 才 yield
        # out2 可能是 [] (因为整段在 thinking)
        assert out2 == []
        # state[0] 应该是 True (in_thinking)
        assert state[0] is True
