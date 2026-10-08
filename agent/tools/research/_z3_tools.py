"""
agent.tools.research._z3_tools — SMT 求解器接入 (P7 阶段 1)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §1.5.1 / §6.1

================================ 为什么 Z3 不走沙箱 ================================
P0 的 run_code 把代码扔进子进程, 但 research_z3 是个**确定性工具**: 输入 SMT-LIB2 文本,
不需要 LLM 参与, 也没有 Python 注入面(SMT-LIB2 不是可执行代码)。
所以直接调 z3 反而更快更安全 —— z3 自带 set(timeout=...) 控制求解时长。

⚠️ 例外说明: SMT-LIB2 里没有 `eval`, 沙箱 AST 门那套对它不适用也用不上。

=============================== 最关键的语义映射: sat/unsat → refuted/proved =========
这是本文件唯一需要动脑的地方。

**`goal="prove"` 不是"检查可满足性", 而是"证明命题成立"**, 所以调用方必须这样构造约束:
    前提 = 命题为真的断言
    结论 = 命题为假的断言 (negation)
    → 若整体 unsat, 说明"命题成立 ∧ 命题不成立"不可满足, 即命题成立 → proved
    → 若整体 sat,   求解器给出了"假设成立但命题不成立"的赋值 = **反例** → refuted

这个映射让 Z3 成为**唯一能给出全域级反例**的工具: 图枚举只覆盖 n≤7, 数值抽查只覆盖采样点,
而 Z3 的 sat 模型是一个满足所有前提的具体赋值 —— 那是货真价实的反例, 不是"没找到"。

历史:
  - 2026-10-08 P7.1 首版
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

#: 求解超时上限 (毫秒)。给 LLM 调, 但硬夹住防止它写 timeout=99999999
MAX_TIMEOUT_MS = 120_000

DEFAULT_TIMEOUT_MS = 5_000


class Z3Error(Exception):
    """约束解析失败 / z3 内部错误 —— 与"unsat"(证明成功)是完全不同的两回事"""


def _ensure_z3():
    """延迟 import z3: 没装的时候给出可执行的提示, 而不是 ImportError 栈"""
    try:
        import z3  # noqa: PLC0415
        return z3
    except ImportError as e:
        raise Z3Error(
            "z3-solver 未安装。装它: pip install z3-solver "
            "(已在 P7.1 装入项目 venv, 若报缺依赖请检查是否跑在同一个 venv)"
        ) from e


def solve(
    constraints: str,
    goal: str = "sat",
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> dict[str, Any]:
    """
    对 SMT-LIB2 约束求解。

    Args:
        constraints: SMT-LIB2 断言, 多个用换行分隔。
            例: "(declare-const x Int)\n(assert (= x (* 2 3)))\n(assert (= x 7))"
            也支持 `(push)`/`(pop)`/`(check-sat)` 混排 —— 但我们**忽略文本里的
            check-sat**, 由 goal 参数决定动作, 免得两处指令打架。
        goal:
            "sat"    → 只判可满足性, 返回 model(若 sat)
            "unsat"  → 只要不可满足性(证明"不可能同时成立")
            "prove"  → 按上文映射: unsat→proved, sat→refuted(模型即反例)
        timeout_ms: 求解超时

    Returns:
        {result: sat|unsat|unknown, model: str|None, elapsed_ms, verdict, reason, n_assertions}
    """
    z3 = _ensure_z3()

    t = _clamp_timeout(timeout_ms)
    started = time.time()

    try:
        parsed = z3.parse_smt2_string(constraints)
    except z3.Z3Exception as e:
        raise Z3Error(f"SMT-LIB2 解析失败: {e}") from e
    except Exception as e:  # z3 偶尔抛非 Z3Exception
        raise Z3Error(f"SMT-LIB2 解析异常: {type(e).__name__}: {e}") from e

    solver = z3.Solver()
    solver.set("timeout", t)
    for f in parsed:
        solver.add(f)

    n_assertions = len(parsed)
    result = None
    try:
        result = solver.check()
    except z3.Z3Exception as e:
        raise Z3Error(f"求解器异常: {e}") from e

    elapsed_ms = int((time.time() - started) * 1000)
    is_sat = result == z3.sat
    is_unsat = result == z3.unsat

    model_str = ""
    if is_sat:
        try:
            m = solver.model()
            # 逐个变量取值 —— 反例就在这里, 要完整给出来
            parts = []
            for d in m.decls():
                parts.append(f"{d.name()} = {m[d]}")
            model_str = "; ".join(parts) if parts else "(模型无可打印赋值)"
        except z3.Z3Exception:
            model_str = "(取模型失败)"

    if goal == "prove":
        if is_unsat:
            verdict = "proved"
            reason = ("约束组不可满足 —— 在给定前提下未找到使结论失败的赋值, "
                      "即命题在该前提下成立")
        elif is_sat:
            verdict = "refuted"
            reason = f"找到满足前提但使结论失败的赋值(**这是反例**): {model_str}"
        else:
            verdict = "inconclusive"
            reason = f"求解器在 {t}ms 内无法定论 (unknown), 不能作为证明"
    else:
        verdict = "untested"
        if is_sat:
            reason = f"可满足, 模型: {model_str}" if model_str else "可满足"
        elif is_unsat:
            reason = "不可满足 —— 这些断言不可能同时成立"
        else:
            reason = f"求解器返回 unknown (超时 {t}ms)"

    return {
        "result": "sat" if is_sat else ("unsat" if is_unsat else "unknown"),
        "model": model_str,
        "elapsed_ms": elapsed_ms,
        "verdict": verdict,
        "reason": reason,
        "n_assertions": n_assertions,
        "timeout_ms": t,
    }


def find_model(constraints: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> dict[str, Any] | None:
    """sat 时返回模型变量赋值 dict; 不 sat / unknown 返回 None。

    专治"找反例": 把命题为真的前提 + 命题为假的结论丢进去, sat 就说明命题不成立。
    """
    out = solve(constraints, goal="sat", timeout_ms=timeout_ms)
    if out["result"] != "sat" or not out["model"]:
        return None
    model: dict[str, str] = {}
    for part in out["model"].split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            model[k.strip()] = v.strip()
    return model


def _clamp_timeout(timeout_ms: int) -> int:
    """把 LLM 可能写的离谱 timeout 夹进 [100, MAX_TIMEOUT_MS]"""
    try:
        t = int(timeout_ms)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_MS
    return max(100, min(t, MAX_TIMEOUT_MS))


# ============================================================
# 提示模板 —— 给 LLM 写对 SMT-LIB2 的概率提升很大
# ============================================================
Z3_HINT = """【SMT-LIB2 写法】
约束要写成 SMT-LIB2 断言, 例如:

    (declare-const x Int)
    (declare-const y Int)
    (assert (>= x 0))
    (assert (= (+ x y) 10))
    (assert (> (* x y) 24))     ; 最后这个不可满足 → 证明 x+y=10 时 xy 不会超过 24

常用: declare-const / assert / let / and / or / not / = / distinct / < > <= >= /
      Int / Real / Bool / (+ - * div mod abs)
变量名: 提问用 x/y/n/k, 解可以用 a b c(别用 d, 会被读成差分)
多行断言用换行分隔即可, 不要自己写 (check-sat) —— 工具会自己判定。

**goal="prove" 的构造法**(务必读)**:
  前提 = 命题成立的断言;  结论 = 命题不成立的断言
  → 整体 unsat 表示"命题不可能不成立" → 工具报 proved
  → 整体 sat 会给出具体赋值, 那就是**反例** → 工具报 refuted
"""
