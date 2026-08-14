"""
Tool Registry: 集中管理 LLM 可调用的工具（参考 WAgent 实现）。

设计目标：
- 用 @register_tool 装饰器把工具函数注册到全局 registry
- 工具的 schema（OpenAI function calling 格式）从 agent.tool_definitions.TOOL_SCHEMAS 字典按 name 查找
- controller 通过 dispatch(name, args) 调用，registry 自动判断是同步还是流式
  - 同步工具：handler 返回 str/dict/list/None → yield 一次完整结果
  - 流式工具（is_streaming=True）：handler 是生成器 → yield 每个 chunk
- 工具的"LLM 端名字"和"Python 函数名"解耦（通过装饰器 name 参数）

用法：
    @register_tool(name="my_tool", is_streaming=False)
    def my_function(arg1: str, arg2: int = 0) -> str:
        return "..."

新增工具流程：
1. 在 agent/tool_definitions.py 的 TOOL_SCHEMAS 添加新 key
2. 在对应工具文件中用 @register_tool(name="your_tool_name") 装饰函数
3. 完成。controller 无需改动。
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    is_streaming: bool = False


class ToolRegistry:
    _tools: Dict[str, ToolDef] = {}

    @classmethod
    def register(cls, tool: ToolDef) -> None:
        if tool.name in cls._tools:
            logger.warning("工具 %r 重复注册，后注册的会覆盖", tool.name)
        cls._tools[tool.name] = tool
        logger.debug("注册工具: %s (streaming=%s)", tool.name, tool.is_streaming)

    @classmethod
    def get(cls, name: str) -> Optional[ToolDef]:
        return cls._tools.get(name)

    @classmethod
    def all(cls) -> List[ToolDef]:
        return list(cls._tools.values())

    @classmethod
    def list_schemas(cls) -> List[Dict[str, Any]]:
        """导出 OpenAI function calling 格式的 schema 列表（给 LLM 看）"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            }
            for t in cls._tools.values()
        ]

    @classmethod
    def list_names(cls) -> List[str]:
        """所有已注册工具的名字列表"""
        return list(cls._tools.keys())

    @classmethod
    def dispatch(cls, name: str, args: Dict[str, Any]) -> Iterator[str]:
        """
        统一调度入口：返回字符串迭代器，controller 直接 yield from。

        - 流式工具：handler 是生成器，yield 每个 chunk
        - 同步工具：handler 返回 str/dict/list/None，yield 一次完整结果（dict/list 自动 json.dumps）
        - 异常：捕获后 yield 错误信息，**不抛出**（避免中断 LLM 对话）
        """
        tool = cls._tools.get(name)
        if tool is None:
            yield f"[工具错误] 未知工具: {name}"
            return

        try:
            result = tool.handler(**args)
        except TypeError as e:
            logger.warning("工具 %r 参数错误: %s", name, e)
            yield f"[工具错误] 参数错误 {name}: {e}"
            return
        except Exception as e:
            logger.exception("工具 %r 执行失败", name)
            yield f"[工具错误] {name} 执行失败: {e}"
            return

        if hasattr(result, "__next__"):
            # 生成器：流式
            try:
                yield from result
            except Exception as e:
                logger.exception("流式工具 %r 中断", name)
                yield f"\n[工具错误] {name} 流式中断: {e}\n"
        else:
            # 同步：一次性 yield
            yield cls._format_result(result)

    @staticmethod
    def _format_result(result: Any) -> str:
        """工具返回值统一转字符串（dict/list 自动序列化）"""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list, tuple)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)


def register_tool(
    name: Optional[str] = None,
    is_streaming: bool = False,
):
    """
    装饰器：把一个 Python 函数注册为 LLM 工具。

    schema 自动从 agent.tool_definitions.TOOL_SCHEMAS 字典按 name 查找。

    用法：
        @register_tool(name="run_python", is_streaming=False)
        def run_python(purpose: str, code: str) -> str: ...

    参数：
        name: 工具在 LLM 端的名字（默认用函数名）
        is_streaming: 是否为流式工具（handler 是生成器函数）

    约束：
        - 必须在 agent/tool_definitions.py 的 TOOL_SCHEMAS 字典中先注册 schema
        - 注册在 import 时发生：模块被 import 时装饰器立即执行
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__

        # 延迟 import 避免循环依赖
        from agent.tool_definitions import TOOL_SCHEMAS
        schema = TOOL_SCHEMAS.get(tool_name)
        if schema is None:
            raise ValueError(
                f"工具 {tool_name!r} 未在 agent/tool_definitions.py 的 TOOL_SCHEMAS 中注册。"
                f"请先在那里添加 schema 声明，再使用 @register_tool 装饰器。"
            )

        fn_block = schema["function"]
        ToolRegistry.register(ToolDef(
            name=tool_name,
            description=fn_block.get("description", ""),
            parameters=fn_block.get("parameters", {"type": "object", "properties": {}}),
            handler=func,
            is_streaming=is_streaming,
        ))
        return func
    return decorator
