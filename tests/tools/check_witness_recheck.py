"""第二层机器复核: 见证值在不在 ≠ 见证值对不对。

真实案例(Goldbach 2026-10-08): 代码把素数 list 存进 set, 然后
`for p in primes: if p > n: break`。set 迭代顺序对整数不保证升序
(实测 [2,3,5,7,521,11,...]), 遇到 521 就提前 break, n=98 的合法素数对
(19+79 / 31+67 / 37+61) 一个没试到, 于是得出"98 是反例"。

反例值 98 是代码**真的算出来的**, 不是编的 —— 所以第一道门
(有没有见证值)拦不住它。但它同时暴露了代码形态, 机器能自己复核。

钉三件事:
  1. Goldbach 那种形态必须被降级
  2. sorted() 过的同类代码不能被误杀
  3. 给出明确算式的输出必须放行
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.tools.research._provenance import judge_verdict  # noqa: E402


class _Run:
    """模拟一次成功的执行 —— judge_verdict 用鸭子类型读属性。"""

    def __init__(self, code: str, stdout: str = ""):
        self.ok = True
        self.rejected = False
        self.timed_out = False
        self.error_type = None
        self.stdout = stdout
        self.code = code


GOLDBACH_CODE = '''
import sympy
LO, HI = 4, 1000
primes = set(sympy.primerange(2, HI + 1))
bad = []
for n in range(LO, HI + 1, 2):
    found = None
    for p in primes:
        if p > n:
            break
        if (n - p) in primes:
            found = (p, n - p); break
    if found is None:
        bad.append(n); break
print(f"未通过个数: {len(bad)}")
if bad:
    print(f"反例: n = {bad[0]} 无法表示为两个素数之和")
    print("VERDICT: REFUTED")
'''

GOLDBACH_OUT = (
    "检查偶数个数: 48\n"
    "未通过个数: 1\n"
    "反例: n = 98 无法表示为两个素数之和\n"
    "VERDICT: REFUTED\n"
)

FIXED_CODE = '''
import sympy
primes = sorted(sympy.primerange(2, 1001))
pset = set(primes)
for n in range(4, 1001, 2):
    for p in primes:
        if p > n // 2:
            break
        if (n - p) in pset:
            break
print("VERDICT: REFUTED")
'''

EXPLICIT_OUT = (
    "反例: n = 8 无法表示为两个素数之和\n"
    "核验 8 = 4 + 4, 而 4 不是素数\n"
    "VERDICT: REFUTED\n"
)

#: (说明, code, stdout, 期望)
CASES = [
    ("★ Goldbach 真实案例: set + break 提前退出",
     GOLDBACH_CODE, GOLDBACH_OUT, "inconclusive"),
    ("对照: sorted() 过的同类代码不能误杀",
     FIXED_CODE, "反例: n = 98 无解\nVERDICT: REFUTED\n", "refuted"),
    ("给出明确算式(= 右边是算式)必须放行",
     GOLDBACH_CODE.replace("break", "pass"), EXPLICIT_OUT, "refuted"),
    ("第一层: 完全没见证值仍然降级",
     GOLDBACH_CODE, "未通过\nVERDICT: REFUTED\n", "inconclusive"),
    ("PROVED 不受这两道门影响",
     GOLDBACH_CODE, "全部成立\nVERDICT: PROVED\n", "proved"),
]


def main() -> None:
    print("=== 第二层机器复核: 见证值对不对 ===\n")
    ok = True
    for label, code, stdout, expect in CASES:
        got, reason = judge_verdict(_Run(code, stdout))
        flag = "OK " if got == expect else "!! "
        if got != expect:
            ok = False
        print(f"{flag}{label}")
        print(f"     期望={expect:14} 实得={got}")
        if got != expect:
            print(f"     {reason[:160]}")

    print("\n结论:", "全部正确" if ok else "有问题")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()