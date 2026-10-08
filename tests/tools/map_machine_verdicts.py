"""机器侧复核: 20 道题各自能被哪种机器方法独立判真伪。

用户的要求(2026-10-08): 「结论对不对」和「结论有没有证据」最后都必须
问机器 —— LLM 的判断也得是基于符号/数值分析的结果, 而不是 LLM 自己
下判断。

现状: judge_verdict 只读代码里 print 的 VERDICT 标记, 机器一句校验都没有
(_provenance.py:199)。LLM 写 PROVED 就是 PROVED, 哪怕它算错了。

所以先摸清: 这 20 道题, 每道能用哪种机器方法**独立**判定?
按「谁来判」给每题分类:
  A 符号消元/化简  sympy: 化简到 0 就是真, 非 0 就是假 —— 机器说了算
  B 穷举 + 判定式  题目自带有限范围, 机器遍历后按判定式算 —— 穷尽即真
  C 数值收敛检查  有独立的高精度参照值可比对
  D 存在性证书    找到见证值就是真(找到=证书); 找不到只能说没找到

D 类是根本性的不对称: **找到反例可以证明假, 没找到不能证明真。**
这不是保守, 是逻辑 —— 所以任何"没找到反例所以真"的判定都必须降级。
"""
from __future__ import annotations

import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))

#: 每题的机器复核方法。这不是"参考", 是**独立于 LLM 判定**的第二条路径。
#: 两条路径不一致 = 有一条错了, 那就该以机器为准并报分歧。
MACHINE_CHECKS = {
    "num-001-euler-prime-polynomial": {
        "method": "穷举 + 素性判定",
        "verdict_is_machine": True,
        "note": "n=0..39 逐个 isprime + n=40 分解。全是机器判定, 无需 LLM。",
        "can_disprove": True,   # 找到非素数的 n 就是证书
    },
    "num-002-goldbach-small": {
        "method": "穷举 + 素数对搜索",
        "verdict_is_machine": True,
        "note": "[4,1000] 全部偶数遍历。找不到素数对 = 反例证书。",
        "can_disprove": True,
    },
    "num-003-carmichael-561": {
        "method": "模幂运算穷举",
        "verdict_is_machine": True,
        "note": "pow(a,560,561) 逐个算, 纯算术。",
        "can_disprove": True,
    },
    "num-004-100-factorial-trailing-zeros": {
        "method": "精确整数运算",
        "verdict_is_machine": True,
        "note": "factorial(100) 是精确整数, 末尾零个数唯一确定。",
        "can_disprove": True,
    },
    "num-005-inclusion-exclusion-100": {
        "method": "穷举计数",
        "verdict_is_machine": True,
        "note": "1..100 逐个判断。计数唯一。",
        "can_disprove": True,
    },
    "alg-006-fibonacci-matrix": {
        "method": "矩阵幂精确比较",
        "verdict_is_machine": True,
        "note": "sympy Matrix 幂是精确整数运算, == 无误差。",
        "can_disprove": True,
    },
    "alg-007-am-gm": {
        "method": "符号化简到平方和",
        "verdict_is_machine": True,
        "note": "sympy factor 后是 (a-b)^2/4 >= 0。化简结果机器给。",
        "can_disprove": True,
    },
    "alg-008-circle-max-linear-sum": {
        "method": "符号代换 + 约束校验",
        "verdict_is_machine": True,
        "note": "x=cos(pi/4) 代入后 sympy 化简, 机器验证 x^2+y^2==1。",
        "can_disprove": True,
    },
    "alg-009-determinant-multiplicative": {
        "method": "符号恒等式(不用随机)",
        "verdict_is_machine": True,
        "note": "sympy 符号矩阵展开 det(AB)-det(A)det(B) 化简为 0, "
                "比随机测 100 组更强(随机只是抽样)。",
        "can_disprove": True,
    },
    "comb-010-eight-queens": {
        "method": "穷举回溯计数",
        "verdict_is_machine": True,
        "note": "解数唯一, 92。",
        "can_disprove": True,
    },
    "logic-011-de-morgan": {
        "method": "真值表穷举",
        "verdict_is_machine": True,
        "note": "2 个变量 4 种赋值, 全枚举。恒真 = 机器可判。",
        "can_disprove": True,
    },
    "opt-012-small-lp": {
        "method": "顶点枚举 + 精确分数",
        "verdict_is_machine": True,
        "note": "可行域顶点有限, 用 Fraction 精确算。",
        "can_disprove": True,
    },
    "comb-013-hanoi": {
        "method": "递推 + 闭式精确比较",
        "verdict_is_machine": True,
        "note": "整数运算, 无浮点。",
        "can_disprove": True,
    },
    "graph-014-ramsey-3-3": {
        "method": "全染色枚举",
        "verdict_is_machine": True,
        "note": "K6 有 2^15=32768 种, 全枚举。K5 有 1024 种。机器可判。",
        "can_disprove": True,
    },
    "graph-015-labeled-trees-5": {
        "method": "图枚举 + 连通性判定",
        "verdict_is_machine": True,
        "note": "2^10=1024 个子图, 全枚举。",
        "can_disprove": True,
    },
    "graph-016-petersen-properties": {
        "method": "图不变量计算",
        "verdict_is_machine": True,
        "note": "度数/直径/围长都是可计算的图不变量。",
        "can_disprove": True,
    },
    "prob-017-birthday-paradox": {
        "method": "精确有理数运算",
        "verdict_is_machine": True,
        "note": "Fraction 精确算, 无浮点误差, > 1/2 是精确比较。",
        "can_disprove": True,
    },
    "prob-018-two-dice-expectation": {
        "method": "穷举均值",
        "verdict_is_machine": True,
        "note": "36 种结果全枚举。",
        "can_disprove": True,
    },
    "alg-019-pythagorean-formula": {
        "method": "符号展开",
        "verdict_is_machine": True,
        "note": "sympy expand 后恒等于 0。",
        "can_disprove": True,
    },
    "num-020-bisection-root": {
        "method": "★ 这题机器判不了真值, 只能验收敛 ★",
        "verdict_is_machine": False,
        "note": "根 2.0945514815423265 确实存在且唯一(导数单调), 但"
                "「二分法在给定迭代次数内达到 1e-6」这件事的判定, "
                "取决于代码实现而不是数学 —— 所以这题 REFUTED 永远该是"
                "INCONCLUSIVE: 机器只能说'还没收敛', 不能说'命题为假'。",
        "can_disprove": False,   # ★ 唯一一题: 找不到反例 ≠ 命题为真
    },
}


def main() -> None:
    import json

    with io.open(os.path.join(HERE, "baseproblems.txt"), encoding="utf-8") as f:
        cases = {c["id"]: c for c in json.load(f)["items"]}

    print(f"{'题目':38} {'机器能独立判?':12} {'能出反例证书?':12} 方法")
    print("-" * 110)
    n_machine = 0
    n_cert = 0
    for cid in cases:
        m = MACHINE_CHECKS.get(cid)
        if not m:
            print(f"{cid:38} {'缺登记':12}")
            continue
        a = "能" if m["verdict_is_machine"] else "不能★"
        b = "能" if m["can_disprove"] else "不能★"
        n_machine += m["verdict_is_machine"]
        n_cert += m["can_disprove"]
        print(f"{cid:38} {a:12} {b:12} {m['method']}")

    print("-" * 110)
    print(f"机器能独立判定的: {n_machine}/{len(cases)}")
    print(f"能出反例证书的:   {n_cert}/{len(cases)}")

    print("\n=== 结论 ===")
    print("19/20 道题的结论可以完全由机器下 —— LLM 的角色只是")
    print("「把题目翻译成可执行的判定式」, 而不是「判断对错」。")
    print("")
    print("num-020 是唯一例外: 根的存在性能证, 但「二分法这次收敛到")
    print("1e-6 没有」只是实现没调好, 跟命题真假无关。这类必须降级成")
    print("INCONCLUSIVE, 而且**降级判据也要由机器做** —— LLM 写")
    print("`else: REFUTED` 时, 机器看见「没有给出任何反例值」就该否决它。")
    print("")
    print("★ 这就是 judge_verdict 要补的: REFUTED 必须附具体反例 ★")


if __name__ == "__main__":
    main()