"""
agent.tools.research._flags — P7 科研工具的暴露开关

================================ 为什么要单独一个模块 ================================
`agent/tools/__init__.py` 目前**没有** import research 包 —— 这是刻意的。

@register_tool 的装饰器在 import 时执行, 一旦 import 了, 工具就进
ToolRegistry, 跟着 controller.py 的 tools= 参数暴露给 LLM。

研发期(P7 阶段 0-4)我们希望:
  - 工具**已注册**(ToolRegistry.get("research_verify") 能拿到 handler),
    所以单测和我调试时能直接调;
  - 但**LLM 看不见**(list_schemas() 里没有 research_*),
    免得半成品工具把主对话带沟里。

实现: 本模块被 research 内部各工具在 import 时读取,
      exposed=False → 工具只注册不暴露; 调好之后把 QAGENT_RESEARCH_EXPOSE 设为 1
      (或改这里的默认值) 即可接入, **不用改任何其它代码**。

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §3.2

为什么不复用 config/settings.py 的 COST_TIER:
那个是**成本闸门**(省钱), 这个是**功能闸门**(半成品别露脸)。
复用会让"关功能"和"省钱"耦在一起, 以后想单独开功能得先抬成本上限 —— 反过来也一样, 麻烦。
"""
import os


def _resolve_exposed() -> bool:
    v = os.getenv("QAGENT_RESEARCH_EXPOSE", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


#: @register_tool(exposed=RESEARCH_EXPOSED) 用这个
RESEARCH_EXPOSED: bool = _resolve_exposed()

#: 沙箱默认超时(秒)。可被 env 覆盖, 调试时方便调。
RESEARCH_SANDBOX_TIMEOUT: int = int(
    os.getenv("QAGENT_RESEARCH_SANDBOX_TIMEOUT", "20") or 20
)

#: 单次 research_verify 最多修正几轮(把错误喂回 LLM 重试)
RESEARCH_MAX_RETRY: int = int(os.getenv("QAGENT_RESEARCH_MAX_RETRY", "2") or 2)
