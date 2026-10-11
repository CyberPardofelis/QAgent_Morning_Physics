"""强制换方法: 咬合 + 不误杀, 双向往返。

## 这道门在拦什么

DS 诊断的「知道-行动鸿沟」在 agent 侧的典型形态:
同一道 Pell 题, 聊天模式说"用连分数", agent 模式写 `for y in range(1, 10**9)`
然后超时。**根因不是它不会, 是 agent 循环里没人逼它停下来想"这次用什么形态"。**

## 为什么不是"再给一条提示"

注入提示有效, 但那是**给答案的方向**。用户明确反对:
「依赖已知结果并且提示的, 说明其实还不是很妙」。
所以这里**不指定换什么**, 只要求换 —— 它对 Pell / BBP / TSP 一视同仁。

## 不误杀的面(比咬合的面更重要)

1. 编码错(普通异常)**不算**方法错 —— 逼它换方法只会越改越乱
2. 没声明 STRATEGY 时, 抱怨的是"你没说打算怎么算", 不是"你换了没"
3. 连续两次不改就放行 —— 免得这道门本身变成死循环
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agent.tools.research._strategy import (  # noqa: E402
    declared_strategy, method_class_failure, strategy_switch_penalty,
)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("=" * 70)
    print("强制换方法 —— 咬合 + 不误杀")
    print("=" * 70)

    # ---------- 读标签 ----------
    print("\n>>> declared_strategy")
    check("识别 solver",
          declared_strategy("import z3\n# STRATEGY: solver\n") == "solver")
    check("识别 construct",
          declared_strategy("# STRATEGY: construct\nimport itertools\n") == "construct")
    check("没有标签返回 None", declared_strategy("import z3\n") is None)
    check("非法标签不算数",
          declared_strategy("# STRATEGY: bruteforce\n") is None)
    check("大小写不敏感",
          declared_strategy("# strategy: Enumerate\n") == "enumerate")

    # ---------- 什么算方法错 ----------
    print("\n>>> method_class_failure（判据必须窄）")
    check("超时算方法错",
          method_class_failure("failed", "执行超时被终止, 没拿到判定结果") is not None)
    check("用错模块算方法错",
          method_class_failure("failed",
                               "代码被沙箱静态门拒绝: line 3: `sympy` 里没有 "
                               "`convergents` 这个名字") is not None)
    check("普通异常不算方法错",
          method_class_failure("failed",
                               "代码执行出错(AttributeError), 没有产生可复核的结果")
          is None)
    check("verdict 非 failed 一律不算",
          method_class_failure("partial", "执行超时被终止") is None)

    # ---------- 换方法门 ----------
    print("\n>>> strategy_switch_penalty（换了就放过 / 没换就挡回）")
    prev, hint = "enumerate", "规模问题: 同一套写法再跑一遍不会更快"
    check("换了方法 -> 放行",
          strategy_switch_penalty(prev, "# STRATEGY: solver\nimport z3\n", hint)
          is None)
    why = strategy_switch_penalty(prev, "# STRATEGY: enumerate\nfor y in ...\n", hint)
    check("没换方法 -> 挡回", why is not None, (why or "")[:40])
    check("判词里不指定该选哪个",
          why is not None and "我不断言" not in why and "你不告诉我" not in why)
    check("判词给出了可选标签",
          why is not None and "solver" in why and "formal" in why)
    check("这轮没声明标签 -> 不拦(交给'你没说打算怎么算'那条)",
          strategy_switch_penalty(prev, "import z3\n", hint) is None)
    check("上一轮没标签 -> 不拦",
          strategy_switch_penalty(None, "# STRATEGY: solver\n", hint) is None)

    print("\n" + "=" * 70)
    if FAILS:
        print(f"✗ {len(FAILS)} 项不通过: " + ", ".join(FAILS))
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())