"""认题必须真的改变方法目录 —— 否则它只是装饰。

★ 这条测试是补 DeepSeek 指出的一处漏洞 ★
2026-10-09 实测: `methods_for(shapes)` 无论传什么形状都返回同一份全量目录,
于是 prompt 里"机器从题面认出的形状标签: ..."这一句对模型没有任何影响 ——
**假装在做识别**。

而且第一版修复还踩了个坑: `_SHAPE_TO_METHODS` 按内部 key("combinatorial")
建表, 而 `classify()` 返回的是中文标签("组合结构"), 一条都匹配不上,
`methods_for` 又退化成全量。这类"两张表靠字符串对齐"的错必须用测试锁住。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agent.tools.research._methods import (  # noqa: E402
    _SHAPE_TO_METHODS, classify, methods_for,
)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("=" * 76)
    print("认题 → 方法目录 必须真的联动")
    print("=" * 76)

    print("\n>>> 认题本身不崩、且能认出东西")
    cases = [
        ("验证 R(3,4)=9: 证明任意 9 个顶点的红蓝完全图必有红三角或蓝 K4，"
         "并给出 8 个顶点无反例的着色。", "组合结构"),
        ("在 Lean 4 + Mathlib 中形式化并证明 sum C(n,k) = 2^n。",
         "形式化证明(Lean/Mathlib)"),
        ("求 Pell 方程 x^2 - 61 y^2 = 1 的最小正整数解。",
         "丢番图/整系数解"),
    ]
    for claim, want in cases:
        got = classify(claim)["shapes"]
        check(f"认出 {want}", want in got, f"实得 {got}")

    print("\n>>> ★ 映射表的键必须和 classify 可能产出的标签对得上 ★")
    # ★ 对着 _SHAPES 声明的标签查, 而不是拿几道题取样 ★
    # 第一版只用 3 道题收集标签, 结果凑不齐 9 个, 把"表里有 4 个键没被
    # 采到"误报成"键失效"。**取样得出的"没出现"不等于"不存在"。**
    from agent.tools.research._methods import _SHAPES
    declared = {label for _key, _rx, label in _SHAPES}
    orphan = sorted(set(_SHAPE_TO_METHODS) - declared)
    missing = sorted(declared - set(_SHAPE_TO_METHODS))
    check("映射表没有失效的键", not orphan, f"失效: {orphan}")
    check("每个形状都有对应方法", not missing, f"缺: {missing}")

    print("\n>>> ★ 不同形状给出不同的主力方法 ★")
    mains = {}
    for claim, _ in cases:
        sh = classify(claim)["shapes"]
        mains[claim[:16]] = [m["name"] for m in methods_for(sh)
                             if m["_rank"] == "主力"]
    for k, v in mains.items():
        check(f"{k:18} 主力非空", bool(v), str(v))
    distinct = {tuple(v) for v in mains.values()}
    check("不同形状的主力集合互不相同",
          len(distinct) == len(mains), f"{len(distinct)} 种不同结果")

    print("\n>>> 认不出形状时不该假装识别成功")
    ms = methods_for([])
    check("空形状 -> 全部标兜底", all(m["_rank"] == "兜底" for m in ms))

    print("\n>>> 不能把不相干的路线删光(认题可能认错)")
    for sh in (["组合结构"], ["不等式"], ["丢番图/整系数解"]):
        ms = methods_for(sh)
        check(f"{sh[0]:16} 仍返回全部 {len(_SHAPE_TO_METHODS) and len(ms)} 条",
              len(ms) == 9, f"实得 {len(ms)}")

    print("\n" + "=" * 76)
    if FAILS:
        print(f"✗ {len(FAILS)} 项不通过: " + "; ".join(FAILS))
        return 1
    print("✓ 全部通过 —— 认题真的在影响方法目录")
    return 0


if __name__ == "__main__":
    sys.exit(main())