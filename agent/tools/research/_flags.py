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
#:
#: 为什么从 2 提到 3: 重试的**性质**变了。
#: 以前一轮失败往往只是"代码跑挂了", 重试收益有限。
#: 现在每一轮的反馈都是**机器给的判决理由**, 具体到病灶:
#:   「你只查了 |p|,|q|<=200, 那不是全域证明」
#:   「你驳的是你自己构造错的那个着色, 不是原命题」
#:   「那个参考值是代码里的字面量, 不是算出来的见证值」
#: 这种反馈是这整个系统里信息密度最高的输入, 每一轮都该用上。
#:
#: 而且**多试是安全的**: 每轮都要过机器门, 门不认的结论流不下去。
#: V2 hard-007 就是靠第 3 轮才把 Ramsey 从假 REFUTED 修成 PROVED 的。
#:
#: 提到 5 的直接原因(2026-10-08 V2 hard-020「求最小的恰有 1000 个因子的 n」):
#: 5 轮尝试**全是代码 bug**, 没有一轮是结论错 ——
#:   0: 递归爆栈(Recursive backtrack 994 层)
#:   1: 忘了 import sympy —— 而且 stdout 里**已经算出了正确答案 810810000**,
#:      只是崩在 print VERDICT 之前
#:   2: 想用 sys.setrecursionlimit 抬递归上限 → sys 不在白名单, 静态门直接拒
#:   3: sympy.prime(0) → ValueError
#: 每一轮的错误都极具体, 而重试预算先见底了。这类「纯实现 bug」型失败
#: 本来就是重试最划算的场景 —— 它不需要换思路, 只需要再改一行。
RESEARCH_MAX_RETRY: int = int(os.getenv("QAGENT_RESEARCH_MAX_RETRY", "5") or 5)


# ============================================================
# 架构版本标签
# ============================================================
#: 每条证据都会被打上这个版本号, 汇总报告也会显示它。
#:
#: ★ 为什么必须打 ★
#: 2026-10-09 架构要重搭(单发 codegen -> Explorer/Worker/Verifier),
#: 新旧两套并行存在。而**分数在架构、题库、判据、verifier 里任意一样
#: 改动后都不可比** —— 看到"分数从 9/20 变成 15/20"时, 如果不知道
#: 那是哪一版架构 + 哪一版题库跑出来的, 这个对比就是无效的。
#:
#: 历史上已经踩过:`all60_process.md` / `progress_and_failures.md` 里的
#: 数字都对应当时那套代码, 几个月后没人说得清。
#:
#: 用法: 换架构时改这个默认值, 或运行时覆盖
#:     $env:QAGENT_RESEARCH_ARCH = "v2-explorer-worker"
#: 已有的旧证据没有这个字段 —— 报告里会显示为 "unstamped", 不会假装它
#: 属于哪一版。
RESEARCH_ARCH_VERSION: str = (
    os.getenv("QAGENT_RESEARCH_ARCH", "").strip() or "v1-single-shot"
)
