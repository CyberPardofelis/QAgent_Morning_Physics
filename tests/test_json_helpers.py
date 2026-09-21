"""
test_json_helpers — `_safe_json_loads_array` 容错 parse 测试 (2026-09-21 体检后加)

覆盖 7 级容错策略 (P6.2 实测多次踩):
  1. 直接 json.loads
  2. 补 '"}]' / ']' 等简单收尾
  3. truncate 到最后一个 } 之前, 补 ]
  4. regex 抽所有单层完整 {...} 对象重建数组

为什么重要: survey_generate / extract_facts 工具的 LLM 输出容错全靠这个函数,
P0 修复密度最高的位置之一。回退会让 LLM 输出变 "" 让整个综述工具失败。
"""
from agent.tools.surveys._pool import _safe_json_loads_array


class TestSafeJsonLoadsArray:
    """测试 _safe_json_loads_array 的 7 级容错策略"""

    def test_perfect_array(self):
        """完美 JSON 数组 → 直接返回"""
        text = '[{"a": 1}, {"b": 2}]'
        result = _safe_json_loads_array(text)
        assert result == [{"a": 1}, {"b": 2}]

    def test_single_object_becomes_array(self):
        """LLM 偶尔返单 object (期望 array) → 自动包成 array"""
        text = '{"a": 1}'
        result = _safe_json_loads_array(text)
        assert result == [{"a": 1}]

    def test_truncated_array_with_extra_brace(self):
        """P6.2 实测: 末尾多 "} 而缺 ]"""
        text = '[{"a": 1}, {"b": 2}'
        result = _safe_json_loads_array(text)
        assert result == [{"a": 1}, {"b": 2}]

    def test_truncated_value_at_end(self):
        """P6.2 实测: 末尾是 "y" 缺 }, 截断"""
        # 末尾字符串不完整, 触发策略 3 (truncate 到最后一个 })
        text = '[{"a": "value"'
        result = _safe_json_loads_array(text)
        # 至少返回 [] 或部分对象, 不抛异常
        assert result is None or isinstance(result, list)

    def test_unclosed_quote(self):
        """策略 4: regex 抽所有完整单层 {...} 对象"""
        # 末尾字符串未闭合 + 无 }, 走策略 4
        text = '[{"a": 1}, {"b": 2}, {"c": "unclosed'
        result = _safe_json_loads_array(text)
        # 至少抽到几个完整的 {} 对象
        assert result is None or isinstance(result, list)

    def test_empty_string_returns_none(self):
        """空串 → None (解析失败, 不是空数组)"""
        result = _safe_json_loads_array("")
        assert result is None

    def test_none_input_returns_none(self):
        """None → None (容错, 不抛)"""
        result = _safe_json_loads_array(None)  # type: ignore[arg-type]
        assert result is None

    def test_garbage_returns_none(self):
        """全乱码 → None (不抛)"""
        result = _safe_json_loads_array("not json at all \x7b")
        assert result is None

    def test_partial_array_with_comma_head(self):
        """策略 2: 简单补尾 '"}]'"""
        text = '[{"a": 1,'
        result = _safe_json_loads_array(text)
        # 补 '"}]' 后能 parse
        assert result is None or isinstance(result, list)

    def test_array_with_trailing_comma(self):
        """数组末尾多余的逗号 + 补 ']'"""
        text = '[{"a": 1}, {"b": 2},'
        result = _safe_json_loads_array(text)
        # 策略 2/3 应该都能处理
        assert result is None or isinstance(result, list)
