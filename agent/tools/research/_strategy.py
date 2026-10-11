"""给验证代码加**策略自述**，并在重试时强制换方法。

## 要解决的缺口

DS 对"知道-行动鸿沟"的诊断: 聊天模式激活的是"这类问题的经典算法是什么",
agent 模式激活的是"下一步写什么代码"。同一道 Pell 题, 聊天里它说"用连分数",
agent 里它写 `for y in range(1, 10**9)` 然后超时。

我目前的对策是**注入提示** —— plan_method 按题型把技法直接写进 prompt。
有效(pell/BBP 都因此过了), 但那是**给答案的方向**, 用户明确不想要:
「依赖已知结果并且提示的, 那说明其实还不是很妙」。

## 这一层做什么

不告诉它该用什么方法, 只做两件事:

1. **代码必须自述策略** —— 一行 `# STRATEGY: enumerate|symbolic|solver|
   numeric|construct|formal`。这不是提示, 是**标签**: 让机器能看见它换了路。
2. **失败后必须换标签** —— 超时/规模类失败被机器判成"方法错"时, 下一轮
   若还声明同一个 STRATEGY, 直接拒绝并要求换。

★ 关键: 第2 条不指定换什么, 只要求**换** ★
所以这道门是通用的 —— 它对 Pell、BBP、100 皇后、TSP 一视同仁,
不需要知道任何一道题的正确方法。这是「补漏洞」和「一般化」的分界线。

## 为什么先只做 REFUTED 之外的失败

R5b 那次事故: 重试把模型推着升级成假的 PROVED。强制换方法看起来是同一个
风险(逼模型产出它还没把握的东西), 所以这里**不碰判定** —— 它只约束
"下一轮用什么方法", verdict 仍然完全由机器门决定。
"""
from __future__ import annotations

import re

#: 允许的策略标签。
#:
#: 刻意**粗** —— 分类越细越容易让模型"选一个标签糊弄过去"。
#: 这几个标签覆盖了数学验证里真正不同的计算形态:
#:   enumerate  有界枚举(拿不到全域结论)
#:   symbolic   符号化简/求解
#:   solver     SAT/SMT 完备判定(z3)
#:   numeric    数值/高精度
#:   construct  构造出一个对象(再验证)
#:   formal     形式化证明(Lean)
STRATEGIES = ("formal", "construct", "solver", "symbolic", "numeric",
              "enumerate")

#: `# STRATEGY: solver` —— 也接受 `STRATEGY: solver` 出现在别的注释行里
_STRATEGY = re.compile(
    r"^\s*#\s*STRATEGY\s*:\s*(?P<s>[a-z_]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def declared_strategy(code: str) -> str | None:
    """从代码里读出它自述的策略。读不到返回 None。"""
    if not code:
        return None
    m = _STRATEGY.search(code)
    if not m:
        return None
    s = m.group("s").strip().lower()
    return s if s in STRATEGIES else None


def method_class_failure(prev_verdict: str, reason: str) -> str | None:
    """
    这次失败是不是「方法选错了」而不是「代码写错了」?

    判据要**窄**, 因为误判的代价是逼模型去换方法而问题本来就在编码上 ——
    那会让它越改越乱。只认两类硬信号:
      · 执行超时 —— 规模问题, 在同一套写法上不可能变快
      · 静态门拒了「用错模块」类的东西 —— 比如写了 `from sympy import convergents`
    其余(failed 里的普通异常)一律**不算**方法错, 因为那确实可能是编码错。
    """
    if prev_verdict != "failed":
        return None
    if "执行超时" in reason:
        return "规模问题: 同一套写法再跑一遍不会更快, 必须换计算策略"
    if "里没有" in reason and "这个名字" in reason:
        return "用错了模块/函数: 正确写法在别的模块里, 换工具比改代码快"
    return None


def strategy_switch_penalty(
    prev_strategy: str | None,
    code: str,
    hint: str,
) -> str | None:
    """
    上一轮被判「方法错」, 而这一轮还声明同一个策略 -> 返回拒绝理由。

    刻意**不指定换什么**: 那就成了注入提示(用户明确不想要)。
    这里只说"你换了吗", 逼它自己回到题目上重新想。
    """
    cur = declared_strategy(code)
    if prev_strategy is None or cur is None:
        return None
    if cur != prev_strategy:
        return None
    return (
        f"上一轮的失败是**方法问题**({hint}), 你却又声明了同样的策略 "
        f"`STRATEGY: {cur}`。\n"
        f"  原样重跑不可能变好 —— 那正是上次失败的原因。\n"
        f"  可选的策略标签: {', '.join(STRATEGIES)}\n"
        f"  ★ 我不告诉你该选哪个 ★ 请回到题目本身重新判断: 这个问题的\n"
        f"    定义域是什么规模? 是连续量还是离散结构? 有没有完备的判定过程\n"
        f"    (SAT/SMT/符号化简)可以用? 要交付的是一个对象还是一个等式?\n"
        f"  先想清楚再写 `# STRATEGY:` 那一行。"
    )