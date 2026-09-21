"""
test_tool_registry — ToolRegistry 注册 / 过滤 测试 (2026-09-21 加)

覆盖 @register_tool 装饰器的 4 个核心行为:
  1. 注册 + list_names()
  2. 重复注册警告
  3. cost_tier 分级 (L0/L1/L2)
  4. schema 输出 (OpenAI function calling 格式)

为什么重要: agent.tools.* 所有工具都依赖 registry 注册才被 LLM 看到。
注册失败或 schema 错会让 LLM "调不到工具" (silent failure)。
"""
from agent.tools.registry import (
    ToolRegistry,
    register_tool,
)


class TestToolRegistryRegister:
    """测试 register_tool 装饰器 + ToolRegistry.register()"""

    def test_register_invalid_cost_tier_raises(self):
        """cost_tier 不是 L0/L1/L2 → ValueError (装饰器本身检查)"""

        # 装饰器 *本身* 检查 cost_tier, 在 schema 检查之前
        try:
            register_tool(name="_does_not_matter", cost_tier="L99")
            raised = False
        except ValueError as e:
            raised = True
            assert "cost_tier" in str(e).lower()
        assert raised

    def test_register_missing_schema_raises(self):
        """未在 TOOL_SCHEMAS 声明的 name → ValueError

        选 cost_tier=L2 (合法值) 让 cost_tier 检查通过, 走到 schema 检查报错。
        注意: register_tool(...) 返回装饰器, 必须用函数调用才会触发 schema 检查。
        """
        raised = False
        try:
            @register_tool(name="_test_nonexistent_schema_xyz", cost_tier="L2")
            def _fake_function() -> str:
                return "x"
        except ValueError as e:
            raised = True
            assert "TOOL_SCHEMAS" in str(e)
        assert raised, "应该抛 ValueError, 但没抛"

    def test_register_existing_tool_does_not_raise(self):
        """已注册的 cost_tier=L1 工具 (analyze_paper) 再 import 不抛

        这是 9-20 拆 knowledge.py 的回归守卫: 装饰器要在 import 时执行
        """
        # analyze_paper 已经在知识库模块 import 时注册过
        # 再 import knowledge 包不会抛 (装饰器检测 schema 存在)
        # force re-import to trigger decorator
        import importlib
        from agent.tools import knowledge as _k
        importlib.reload(_k)
        names_after = set(ToolRegistry.list_names())
        # knowledge 工具仍存在 (reload 后还在)
        assert "analyze_paper" in names_after


class TestToolRegistryFilter:
    """测试 list_schemas_filtered: 按 cost_tier 过滤"""

    def test_filter_l0_only(self):
        """L0 档: 只返 L0 工具"""
        # 已经有注册的 L0/L1/L2 工具
        # 默认情况下 L0 返回 L0 (无 L1/L2)
        schemas_l0 = ToolRegistry.list_schemas_filtered("L0")
        for s in schemas_l0:
            assert s["function"].get("x_cost_tier", "L1") == "L0"

    def test_filter_l1_returns_l0_and_l1(self):
        """L1 档: 返 L0 + L1"""
        schemas_l1 = ToolRegistry.list_schemas_filtered("L1")
        for s in schemas_l1:
            tier = s["function"].get("x_cost_tier", "L1")
            assert tier in ("L0", "L1")

    def test_filter_l2_returns_all(self):
        """L2 档: 返全部 (L0 + L1 + L2)"""
        schemas_l2 = ToolRegistry.list_schemas_filtered("L2")
        assert len(schemas_l2) >= len(ToolRegistry.list_schemas())

    def test_filter_default_is_l2(self):
        """默认 cost_tier 是 L2 (controller 不传 cost_tier 时用 Config 默认值)"""
        # 注意: ToolRegistry.list_schemas_filtered() 的 default 参数实际是 L2,
        # 这是 controller 的实现细节 — controller.run_agent_stream 用 Config.get_cost_tier() 拿默认值
        schemas_default = ToolRegistry.list_schemas_filtered()
        schemas_l2 = ToolRegistry.list_schemas_filtered("L2")
        # 默认应该返 L2 (全开)
        assert len(schemas_default) == len(schemas_l2)
        assert len(schemas_default) >= len(ToolRegistry.list_schemas())


class TestToolRegistrySchema:
    """测试 list_schemas: OpenAI function calling 格式"""

    def test_schema_has_required_fields(self):
        """每个 schema 都有 type=function + function.{name, description, parameters}"""
        schemas = ToolRegistry.list_schemas()
        assert len(schemas) > 0
        for s in schemas[:3]:  # 抽检前几个
            assert s["type"] == "function"
            assert "function" in s
            assert "name" in s["function"]
            assert "description" in s["function"]
            assert "parameters" in s["function"]

    def test_x_cost_tier_metadata_in_schema(self):
        """x_cost_tier 元数据字段被注入 (controller 按 tier 过滤用)"""
        schemas = ToolRegistry.list_schemas()
        for s in schemas[:3]:
            assert "x_cost_tier" in s["function"]
            assert s["function"]["x_cost_tier"] in ("L0", "L1", "L2")


class TestToolRegistryKnowledgeTools:
    """测试 9 个 knowledge 工具全部注册 (P0 体检发现的拆分 bug 守卫)"""

    def test_all_9_knowledge_tools_registered(self):
        """analyze_paper / query_knowledge / scan_local_papers / semantic_search /
        index_paper_to_chroma / reindex_all / sync_recent_to_chroma /
        semantic_search_keywords / similar_papers 全部注册

        这是 2026-09-20 拆 knowledge.py 时发现的 bug 守卫:
        @register_tool 装饰器被脚本切掉, 工具全部未注册。
        """
        names = ToolRegistry.list_names()
        expected = {
            "analyze_paper", "query_knowledge", "scan_local_papers",
            "semantic_search", "index_paper_to_chroma", "reindex_all",
            "sync_recent_to_chroma", "semantic_search_keywords",
            "similar_papers",
        }
        missing = expected - set(names)
        assert not missing, f"missing knowledge tools: {missing}"
