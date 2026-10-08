"""机器裁决 REFUTED: 三种情况都要判对。

核心原则(用户定的): 「结论对不对」和「结论有没有证据」最后都必须问机器。
这道门就是机器对 REFUTED 的独立复核 —— LLM 写什么不算什么。

要挡的三种:
  1. REFUTED 但没有具体反例值   → 必须降级(二分法那道的真实死法)
  2. REFUTED 且有反例值         → 放行(不能误杀真正的反例!)
  3. 提到"反例"但没有数字      → 必须降级(只喊口号不算数)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.tools.research._provenance import judge_verdict  # noqa: E402


class _OK:
    ok = True
    rejected = False
    timed_out = False
    error_type = None
    stdout = ""


#: (说明, stdout, 期望 verdict)
CASES = [
    # ---- 必须降级 ----
    ("二分法实测: 精度不够却报 REFUTED, 整段输出没有一个可复核的值",
     "Convergence check passed: False\n"
     "VERDICT: REFUTED\n",
     "inconclusive"),
    ("提到反例但值是纯文字, 整段没有任何数字",
     "发现反例了\n命题在此处不成立\nVERDICT: REFUTED\n",
     "inconclusive"),
    ("输出只有 VERDICT 一行",
     "VERDICT: REFUTED\n",
     "inconclusive"),
    # ---- 必须放行: 真正的反例不能被误杀 ----
    ("真反例: Euler 多项式在 n=40 是合数",
     "n=0..39 全部为素数\n"
     "反例: n = 40, 此时值 = 1681, 因子 = {41: 2}\n"
     "VERDICT: REFUTED\n",
     "refuted"),
    ("counterexample 英文写法",
     "counterexample: n = 41 gives 41^2+41+41 = 1763 = 41*43\n"
     "VERDICT: REFUTED\n",
     "refuted"),
    ("负数/小数/科学计数法",
     "反例: n = -3, 差值 = -1.5e-07\nVERDICT: REFUTED\n",
     "refuted"),
    ("★ 真实回归: 评估集里「所有素数都是奇数」那道 ★",
     # 这里的 2 就是反例。第一版正则太窄, 把它误杀了 ——
     # 而误杀真反例比放过假反例更糟: 放过=假结论流出去, 误杀=真结论被降级。
     "前 100 内的偶素数: [2]\nVERDICT: REFUTED\n",
     "refuted"),
    ("★ 已知漏洞已闭合: 二分法「精度不够却报 REFUTED」★",
     # 这条以前是**记在案的已知漏洞**: 输出里有真实数值, 于是「有没有见证值」
     # 这道门认了, REFUTED 照旧流出。而 REFUTED 本身是错的 ——
     # 迭代 20 次残差 6.2e-6 超过 1e-6 要求, 说明的是"我的精度不够",
     # 不是"根不存在", 正确处置是 INCONCLUSIVE。
     # 凭证制把它关上了: 光打印一个数、没指名任何违反对象 = 没交凭证。
     "Bisection result: 2.0945520401000977\nVERDICT: REFUTED\n",
     "inconclusive"),
    # ---- 其他 verdict 不受这道门影响 ----
    ("PROVED 不需要见证值",
     "化简结果: 0\nVERDICT: PROVED\n",
     "proved"),
    ("PARTIAL 不需要见证值",
     "抽查 200 点未发现反例\nVERDICT: PARTIAL\n",
     "partial"),
]


def main() -> None:
    print("=== judge_verdict 对 REFUTED 的机器裁决 ===\n")
    ok = True
    for label, stdout, expect in CASES:
        got, reason = judge_verdict(_OK(), stdout)
        flag = "OK " if got == expect else "!! "
        if got != expect:
            ok = False
        print(f"{flag}{label}")
        print(f"     期望={expect:14} 实得={got:14}")
        if got != expect:
            print(f"     理由: {reason[:130]}")

    print("\n=== 非 executed 的情况仍优先 ===")
    class _Bad:
        ok = False
        rejected = False
        timed_out = False
        error_type = "ImportError"

    got, _ = judge_verdict(_Bad(), "VERDICT: REFUTED\n")
    print(f"  代码崩了却打了 REFUTED -> {got} (期望 failed)")
    if got != "failed":
        ok = False

    print("\n结论:", "全部正确" if ok else "有问题")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()