# -*- coding: utf-8 -*-
"""
PROVED 侧机器门的咬合 / 不误杀 双测试。

夹具用的是 2026-10-08 V2 实跑**真实抓到的**代码与 stdout, 不是手编的最小样例 ——
因为这道门能不能拦住, 取决于真实 LLM 输出里的措辞长什么样。

两条例子的选取是刻意的, 一正一反:
  正例 = hard-016 sqrt(2) 无理    —— 假 PROVED, 必须被降级
  反例 = hard-007 Ramsey Part(1) —— z3 真证出了 R(3,4)<=9, **不能**被降级

只看正例的话, 这道门完全可能是靠"一律降级"通过的; 必须同时证明它放过反例,
否则就是把通过率做假 —— 那比不做这道门更糟。
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, r"D:\pythonProject\QAgent")

from agent.tools.research._provenance import (  # noqa: E402
    _counterexample_credential, judge_verdict,
)


def _res(stdout: str, code: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        stdout=stdout, stderr="", ok=True, rejected=False,
        timed_out=False, error_type="", code=code,
    )


# ============================================================
# 夹具 1 (正例): V2 hard-016 「证明 sqrt(2) 是无理数」attempt 2
# ============================================================
SQRT2_CODE = '''
"""Verify (symbolically) the proposition: sqrt(2) is irrational"""
import sympy
from sympy import symbols, sqrt, Rational, simplify, expand, factor, Eq, solve, S, gcd

p, q = symbols('p q', integer=True, positive=True)
eq = Eq(p**2, 2 * q**2)
sol = solve(eq, p)
factored = factor(p**2 - 2*q**2)
N = 200
hits = []
for pp in range(-N, N + 1):
    for qq in range(-N, N + 1):
        if pp * pp - 2 * qq * qq == 0 and (pp, qq) != (0, 0):
            hits.append((pp, qq))
r = sqrt(2)
print("General solutions for p (in terms of q):", sol)
print(f"Non-trivial integer solutions with |p|,|q| <= {N}:", hits)
print("sympy classifies sqrt(2) as irrational =", getattr(r, "is_irrational", "N/A"))
if not hits:
    print("No non-trivial integer solution found in the checked range "
          f"|p|,|q| <= {N}.")
    print("General solution forces q = 0 for p,q in Z "
          "(since sqrt(2) is irrational).")
    print("VERDICT: PROVED")
else:
    print("VERDICT: REFUTED")
'''

SQRT2_STDOUT = '''Equation to solve over positive integers: p**2 - 2*q**2 = 0
General solutions for p (in terms of q): []
Factorisation of p**2 - 2*q**2 : p**2 - 2*q**2
Non-trivial integer solutions of p^2=2q^2 with |p|,|q| <= 200: []
sympy classifies sqrt(2) as: Pow   (is_Rational = False , is_irrational = True )
No non-trivial integer solution found in the checked range |p|,|q| <= 200.
General solution forces q = 0 for p,q in Z (since sqrt(2) is irrational).
VERDICT: PROVED
'''

SQRT2_CLAIM = "在 Lean 4 + Mathlib 中形式化并证明 sqrt(2) 是无理数。"


# ============================================================
# 夹具 2 (反例): V2 hard-007 R(3,4)=9 的 Part (1) —— z3 完备判定
# ============================================================
RAMSEY_CODE = '''
import z3
from itertools import combinations

def sat_for(n):
    """任意 n 顶点红蓝完全图必有红三角或蓝 K4 的否定式"""
    edges = list(combinations(range(n), 2))
    c = {e: z3.Bool(f"c_{e[0]}_{e[1]}") for e in edges}
    s = z3.Solver()
    s.add(z3.Not(has_red_tri_or_blue_k4(n, c)))
    return s.check()   # unsat = 命题对所有着色成立 = 完备证明

# Part (2): 8 顶点构造(有限枚举, 只用于给出下界见证)
n8 = 8
coloring = {}
for i in range(n8):
    for j in range(i+1, n8):
        coloring[(i,j)] = 'R' if (i ^ j).bit_count() == 1 else 'B'

r3 = has_red_K3(coloring, n8)
k4 = has_blue_K4(coloring, n8)
part1_proved = (sat_for(9) == z3.unsat)
part2_proved = (r3 is None) and (k4 is None)
print("Part (1): satisfiable(NOT formula) =", sat_for(9) == z3.sat)
print("Part (1) PROVED:", part1_proved)
if part1_proved and part2_proved:
    print("VERDICT: PROVED")
else:
    print("VERDICT: REFUTED")
'''

RAMSEY_STDOUT = '''Part (1): satisfiable(NOT formula) = False
Part (1): simplify_logic(formula) is tautology = False
Part (1) PROVED: True
Part (2): 红色三角形 in 8-vertex coloring = None
Part (2): 蓝色 K_4   in 8-vertex coloring = (0, 3, 5, 6)
Part (1) R(3,4) <= 9  : YES
VERDICT: REFUTED
'''

RAMSEY_CLAIM = (
    "验证 R(3,4)=9：证明任意 9 个顶点的红蓝完全图必有红色三角形或蓝色 K4，"
    "并给出 8 个顶点无反例的着色。"
)


# ============================================================
# 测试
# ============================================================
def test_bites_false_proved():
    """核心: 有界枚举 + 循环论证的假 PROVED 必须被降级"""
    v, reason = judge_verdict(
        _res(SQRT2_STDOUT, SQRT2_CODE), code=SQRT2_CODE, claim=SQRT2_CLAIM
    )
    assert v != "proved", f"假 PROVED 竟然放行了 (verdict={v})"
    assert v in ("partial", "inconclusive"), f"降级方向不对: {v}"
    # 理由必须指向真正的病灶, 否则 LLM 不知道该改什么
    assert ("范围内" in reason) or ("循环论证" in reason) or ("前提" in reason), (
        f"理由没说明问题: {reason}"
    )
    print(f"  [咬住] sqrt2 假 PROVED -> {v}")
    print(f"         理由首句: {reason.splitlines()[0]}")


def test_does_not_kill_complete_proof():
    """反向: z3 完备判定的真证明不能被降级"""
    v, reason = judge_verdict(
        _res(RAMSEY_STDOUT, RAMSEY_CODE), code=RAMSEY_CODE, claim=RAMSEY_CLAIM
    )
    # 这道题整体 verdict 是 REFUTED(Part2 构造错了), REFUTED 侧另有一套门,
    # 这里只关心 **PROVED 不该被 R1/R2 误伤**, 所以直接喂 PROVED 那段输出。
    proved_out = "Part (1): satisfiable(NOT formula) = False\nunsat\nVERDICT: PROVED\n"
    v2, reason2 = judge_verdict(
        _res(proved_out, RAMSEY_CODE), code=RAMSEY_CODE, claim=RAMSEY_CLAIM
    )
    assert v2 == "proved", f"z3 完备证明被误杀: {v2} / {reason2}"
    print(f"  [放过] z3 unsat 的真 PROVED -> {v2}  ({reason2})")
    _ = (v, reason)


def test_finite_loop_without_nothing_found_not_killed():
    """
    边界: 有 for-range 但结论不是「没找到」时不该触发 R1。
    防止这道门退化成「看到 range 就降级」。
    """
    code = (
        "for n in range(2, 100):\n"
        "    assert is_prime(n) or n < 2\n"
        "print('全部通过')\n"
        "print('VERDICT: PROVED')\n"
    )
    out = "全部通过\nVERDICT: PROVED\n"
    v, reason = judge_verdict(_res(out, code), code=code, claim="验证 100 以内的素数性质")
    assert v == "proved", f"无「没找到」措辞却被降级: {v} / {reason}"
    print(f"  [放过] 有界循环但结论非「没找到」 -> {v}")


def test_conclusion_word_without_premise_marker_not_killed():
    """
    边界: 提到结论词但**没有前提标记**时不该触发 R2。
    「结论词」本身不是循环论证, 同行出现前提标记才是。
    """
    code = "print('sqrt(2) 确实是无理数')\nprint('VERDICT: PROVED')\n"
    out = "sqrt(2) 确实是无理数\nVERDICT: PROVED\n"
    v, reason = judge_verdict(
        _res(out, code), code=code, claim=SQRT2_CLAIM
    )
    assert v == "proved", f"仅提到结论词就被降级: {v} / {reason}"
    print(f"  [放过] 只提到结论词、无前提标记 -> {v}")


# ============================================================
# 夹具 3 (边界): V2 hard-019 「不存在 a,b,c<1000 使 a³+b³=c³」
# ============================================================
# 这条专门锁住 R1 的一处过度杀伤。命题**自带范围** a,b,c < 1000,
# 于是枚举 1..999 恰好覆盖全称量化的整个定义域 —— PROVED 本来就是对的。
# 第一版 R1 不看命题就一律降级, 把这种该判 PROVED 的题打成了 partial,
# 通过率掉下去、掉的却是对的题。所以必须有这条反向测试。
FLT3_CODE = '''
found = []
for a in range(1, 1000):
    for b in range(1, 1000):
        for c in range(1, 1000):
            if a**3 + b**3 == c**3:
                found.append((a, b, c))
print("穷举范围: a, b, c ∈ [1, 999]")
print("范围内未找到 a^3 + b^3 = c^3 的正整数解")
if not found:
    print("VERDICT: PROVED")
else:
    print("VERDICT: REFUTED")
'''

FLT3_STDOUT = """穷举范围: a, b, c ∈ [1, 999]
范围内未找到 a^3 + b^3 = c^3 的正整数解
VERDICT: PROVED
"""

FLT3_CLAIM = "验证不存在正整数 a,b,c < 1000 满足 a^3 + b^3 = c^3。"


def test_scoped_claim_not_killed():
    """命题自带有限范围时, 有界搜索是完整证明, R1 不该触发"""
    v, reason = judge_verdict(
        _res(FLT3_STDOUT, FLT3_CODE), code=FLT3_CODE, claim=FLT3_CLAIM
    )
    assert v == "proved", f"命题自带范围的完整证明被误杀: {v} / {reason}"
    print(f"  [放过] 命题自带 a,b,c<1000 的有界证明 -> {v}")


def test_unscoped_claim_with_same_code_is_caught():
    """
    反向锁定: **同一份代码**, 换一个没有范围限定的命题, 就必须被拦。
    证明这道门是在读命题, 不是在无脑拦「有 range 的代码」。
    """
    v, reason = judge_verdict(
        _res(FLT3_STDOUT, FLT3_CODE),
        code=FLT3_CODE,
        claim="证明不存在正整数 a,b,c 满足 a^3 + b^3 = c^3。",
    )
    assert v != "proved", f"换成无界命题后竟然放行: {v}"
    print(f"  [咬住] 同一份代码 + 无界命题 -> {v}")


# ============================================================
# R3: REFUTED 的靶子绑定 (V2 hard-007 真实输出)
# ============================================================
# 这道门和前两道性质不同: REFUTED 的**见证值是真的**, 机器前两层都认。
# 错的是它驳的对象 —— 「我自己构造错的那个 8 顶点着色」, 而不是 R(3,4)=9。
RAMSEY_LLM_CODE = '''
def has_blue_K4(col, n):
    for quad in combinations(range(n), 4):
        a,b,c,d = quad
        if all(col[(min(x,y),max(x,y))]=='B' for x,y in
               [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]):
            return quad
    return None

part1 = (sat_for(9) == z3.unsat)          # 上界: 证出来了
r3 = has_red_K3(coloring, 8)
k4 = has_blue_K4(coloring, 8)
part2 = (r3 is None) and (k4 is None)     # 下界: 构造写错了
print("Part (1): satisfiable(NOT formula) = False")
print("Part (1) PROVED:", part1)
print("Part (2): 蓝色 K_4 =", k4)
if not part1:
    print("VERDICT: REFUTED")
elif not part2:
    print("VERDICT: REFUTED")             # ★ 病根在这里
else:
    print("VERDICT: PROVED")
'''

RAMSEY_LLM_STDOUT = """Part (1): satisfiable(NOT formula) = False
Part (1) PROVED: True
Part (2): 红色三角形 in 8-vertex coloring = None
Part (2): 蓝色 K_4   in 8-vertex coloring = (0, 3, 5, 6)
Part (1) R(3,4) <= 9  : YES
Part (2) R(3,4) >  8  : NO
综合 R(3,4) = 9        : NO
VERDICT: REFUTED
"""

RAMSEY_CLAIM_COMPOUND = (
    "验证 R(3,4)=9：证明任意 9 个顶点的红蓝完全图必有红色三角形或蓝色 K4，"
    "并给出 8 个顶点无反例的着色。"
)


def test_r3_bites_wrong_subject():
    """核心: 假 REFUTED(见证值为真但驳错了靶子)必须被降级"""
    v, reason = judge_verdict(
        _res(RAMSEY_LLM_STDOUT, RAMSEY_LLM_CODE),
        code=RAMSEY_LLM_CODE, claim=RAMSEY_CLAIM_COMPOUND,
    )
    assert v != "refuted", f"驳错靶子的 REFUTED 竟然放行了: {v}"
    assert v == "partial", f"降级方向不对: {v}"
    assert "自相矛盾" in reason or "合取" in reason, f"理由没指到病灶: {reason}"
    print(f"  [咬住] Ramsey 假 REFUTED -> {v}")


def test_r3_does_not_kill_simple_refuted():
    """
    反向: **单向**命题上的真 REFUTED 不能被 R3 拦。
    这是最关键的一条 —— R3 只在"复合命题 + 输出自相矛盾"时才动手。
    """
    out = "找到反例: n = 15 不满足条件\nVERDICT: REFUTED\n"
    code = "for n in range(1, 100):\n    if bad(n): print('找到反例: n =', n); break\n"
    claim = "证明对所有 n >= 1, f(n) > 0。"
    v, reason = judge_verdict(_res(out, code), code=code, claim=claim)
    assert v == "refuted", f"单向命题的真 REFUTED 被误杀: {v} / {reason}"
    print(f"  [放过] 单向命题的真 REFUTED -> {v}")


def test_r3_does_not_kill_consistent_refuted():
    """
    反向: 复合命题上**输出不矛盾**的 REFUTED(真的推翻了整体)不能被拦。
    防止 R3 退化成"看到双向题就降级"。
    """
    out = ("检查 9 顶点着色: 找到一类无红三角且无蓝 K4 的着色\n"
           "反例: 顶点集 {0..8}, 全部边染红\n"
           "VERDICT: REFUTED\n")
    claim = RAMSEY_CLAIM_COMPOUND
    v, reason = judge_verdict(_res(out, "x = 1\n"), code="x = 1\n", claim=claim)
    assert v == "refuted", f"一致的真 REFUTED 被误杀: {v} / {reason}"
    print(f"  [放过] 输出不矛盾的真 REFUTED -> {v}")


# ============================================================
# R4: REFUTED 不能来自「和硬编码的已知值比不上」
# ============================================================
CATALAN_CODE = '''from math import comb, factorial
c50 = comb(100, 50) // 51
a = c50
b = c50
c = 1
for i in range(51):
    c = c * (50 - i) // (i + 1)
d = c
agree = (a == b == d)
KNOWN = 27596641055636946559611599865755672141241041882849239780
print("C_50 via binomial formula :", a)
print("C_50 via recurrence       :", c)
print("All three methods agree?  :", agree)
print("Known reference value     :", KNOWN)
print("Computed == reference:", a == KNOWN)
if a != KNOWN:
    print("VERDICT: REFUTED")
else:
    print("VERDICT: PROVED")
'''

CATALAN_STDOUT = """C_50 via binomial formula : 1978261657756160653623774456
C_50 via recurrence       : 1978261657756160653623774456
All three methods agree?  : True
Known reference value     : 27596641055636946559611599865755672141241041882849239780
Computed == reference: False
VERDICT: REFUTED
"""


def test_r4_bites_hardcoded_reference():
    """核心: 拿编造的「已知值」证伪真命题, 必须被拦"""
    v, reason = judge_verdict(
        _res(CATALAN_STDOUT, CATALAN_CODE), code=CATALAN_CODE,
        claim="计算第 50 个 Catalan 数 C_50 的精确值。",
    )
    assert v != "refuted", f"硬编码参考值证伪竟然放行: {v}"
    assert v == "inconclusive", f"降级方向不对: {v}"
    print(f"  [咬住] Catalan 拿编造参考值证伪 -> {v}")


def test_r4_does_not_kill_real_counterexample():
    """
    反向: **算出来的**真反例不能被 R4 拦。
    判据是「那个数在源码里是不是字面量」—— 算出来的数不会以裸字面量出现。
    """
    code = ('for n in range(2, 1000):\n'
            '    if 4 % n != 0:\n'
            '        print("反例: n =", n, "此时 4 不能被整除")\n'
            '        print("VERDICT: REFUTED")\n'
            '        break\n')
    out = "反例: n = 3 此时 4 不能被整除\nVERDICT: REFUTED\n"
    v, reason = judge_verdict(_res(out, code), code=code, claim="证明 4 能被所有 2<=n<=999 整除。")
    assert v == "refuted", f"真反例被误杀: {v} / {reason}"
    print(f"  [放过] 算出来的真反例 -> {v}")


def test_r4_does_not_kill_computed_large_number():
    """
    反向: 代码里没有那个数字的字面量时不该触发 ——
    即便输出提到了"参考值"两个字。
    """
    code = ("from math import comb\n"
            "c = comb(100, 50) // 51\n"
            "ref = c * 2\n"                 # 参考值是**算出来的**
            "print('Known reference value:', ref)\n"
            "print('反例: C_50 =', c)\n"
            "print('VERDICT: REFUTED')\n")
    out = ("Known reference value: 3956523314104731\n"
           "反例: C_50 = 1978261657756160653623774456\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, code), code=code, claim="C_50 等于某个值。")
    assert v == "refuted", f"参考值是算出来的却被误杀: {v} / {reason}"
    print(f"  [放过] 参考值是算出来的 -> {v}")


ERDOS_STRAUS_CODE = '''
import sympy
from itertools import combinations
found = []
checked = 0
for n in range(2, 1001):
    ok = False
    for x, y, z in combinations(range(1, 400), 3):
        if sympy.Rational(4, n) == sympy.Rational(1, x) + sympy.Rational(1, y) + sympy.Rational(1, z):
            ok = True
            break
    checked += 1 if ok else 0
    if not ok:
        found.append(n)
print("检查范围: 2 到 1000")
print("已验证的 n 数量:", checked)
if found:
    print("反例: n =", found[0])
    print("VERDICT: REFUTED")
else:
    print("在完整枚举范围内未找到正整数解")
    print("VERDICT: REFUTED")
'''

ERDOS_STRAUS_STDOUT = """检查范围: 2 到 1000
已验证的 n 数量: 0
反例: n = 2
在完整枚举范围内未找到正整数解
VERDICT: REFUTED
"""


def test_r5_bites_refuted_while_finding_nothing():
    """核心: 「没找到」+ REFUTED 的自相矛盾输出必须被拦"""
    v, reason = judge_verdict(
        _res(ERDOS_STRAUS_STDOUT, ERDOS_STRAUS_CODE), code=ERDOS_STRAUS_CODE,
        claim="验证 Erdos-Straus 猜想对 2 <= n <= 1000 成立：4/n = 1/x + 1/y + 1/z 有正整数解。",
    )
    assert v != "refuted", f"自相矛盾的 REFUTED 竟然放行: {v}"
    print(f"  [咬住] Erdos-Straus 边说没找到边判 REFUTED -> {v}")


def test_r5_does_not_kill_partial_path():
    """
    反向: 同样说「未找到」, 但老实判 PARTIAL 的, 不该被 R5 动。
    (R5 只在 verdict 是 refuted 时触发)
    """
    out = "在完整枚举范围内未找到正整数解\nVERDICT: PARTIAL\n"
    v, reason = judge_verdict(
        _res(out, "x=1\n"), code="x=1\n",
        claim="验证 Erdos-Straus 猜想对 2 <= n <= 1000 成立。",
    )
    assert v == "partial", f"老实的 PARTIAL 被误伤: {v} / {reason}"
    print(f"  [放过] 说「未找到」但判 PARTIAL -> {v}")


def test_r5_does_not_kill_genuine_refutation():
    """反向: 真找到了反例的输出不该被拦(输出里没有「没找到」措辞)"""
    out = "找到反例: n = 7 不满足条件, 差值 = 0.25\nVERDICT: REFUTED\n"
    code = "for n in range(1,100):\n    if bad(n): print('找到反例: n =', n); break\n"
    v, reason = judge_verdict(_res(out, code), code=code, claim="证明对所有 n 有 f(n)=0。")
    assert v == "refuted", f"真反例被误杀: {v} / {reason}"
    print(f"  [放过] 真的找到反例 -> {v}")


# ---------- R5b: 嘴上说「全过了」却判 REFUTED ----------
FIB_CODE = '''
import sympy
all_passed = True
for n in range(1, 21):
    A = sympy.Matrix([[1, 1], [1, 0]]) ** n
    rhs = sympy.Matrix([[fib(n + 1), fib(n)], [fib(n), fib(n - 1)]])
    if A != rhs:
        all_passed = False
        print("反例: n =", n)
print("Checked n = 1..20: all_passed =", all_passed)
if not all_passed:
    print("VERDICT: REFUTED")
else:
    print("VERDICT: PROVED")
'''

FIB_STDOUT = """Checked n = 1..20: all_passed = True
Base case n=1 diff: [[0, 0], [0, 0]]
Recurrence holds: False
VERDICT: REFUTED
"""


def test_r5b_bites_refuted_after_all_passed():
    """核心: 自己说「全过了」却判 REFUTED, 必须被拦"""
    v, reason = judge_verdict(
        _res(FIB_STDOUT, FIB_CODE), code=FIB_CODE,
        claim="验证矩阵 [[1,1],[1,0]]^n = [[F_{n+1},F_n],[F_n,F_{n-1}]] 对 n<=20 成立。",
    )
    assert v != "refuted", f"「全过了」+ REFUTED 竟然放行: {v}"
    # ★ 这份夹具里**根本没有反例声明**, 所以凭证门先拦下它 ——
    #   这正是凭证制的一般化之处: 它不需要认出"说全过了"这种形状。
    assert "凭证" in reason, f"理由没指到病灶: {reason}"
    print(f"  [咬住] Fibonacci 全过了却判 REFUTED -> {v}")


def test_r5b_fallback_still_bites_with_credential():
    """
    兜底层仍要生效: **凭证齐全**时, 「说全过了却判 REFUTED」照样要拦。

    这是凭证制的对照实验 —— 如果凭证齐全之后 R5b 就失效了, 说明
    兜底门和凭证门覆盖的不是同一片区域, 删兜底之前必须想清楚。
    """
    out = ("Checked n = 1..20: all_passed = True\n"
           "Recurrence holds: False\n"
           "COUNTEREXAMPLE: n=17 | lhs=A17 | rhs=B17 | violated=True\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "FIB = 1\n"), code="FIB = 1\n",
                              claim="验证矩阵 [[1,1],[1,0]]^n = "
                                    "[[F_{n+1},F_n],[F_n,F_{n-1}]] 对 n<=20 成立。")
    assert v != "refuted", f"凭证齐全但自述矛盾仍被放行: {v}"
    assert "自相矛盾" in reason, f"兜底门没给出自相矛盾的判词: {reason}"
    print(f"  [咬住] 凭证齐全 + 自述矛盾 -> {v}（兜底层仍然生效）")


def test_r5b_does_not_kill_all_passed_proved():
    """反向: 说「全过了」且老实判 PROVED 的, 不该被 R5b 动"""
    out = "Checked n = 1..20: all_passed = True\nVERDICT: PROVED\n"
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n", claim="对 n<=20 成立。")
    assert v == "proved", f"正常的 PROVED 被误杀: {v} / {reason}"
    print(f"  [放过] 说「全过了」且判 PROVED -> {v}")


# ---------- R5b 补: 问号写法(真实 Pell 翻车) ----------
PELL_STDOUT = """首个满足 x^2 - D*y^2 = 1 的收敛下标 = 21
sqrt(61) 的连分数周期长度 = 11
sympy 给出的最小正解: x = 1766319049, y = 226153980
两种独立方法一致? True
方程成立? True
结构最小性(首个 +1 收敛下标 = 周期长度)? False
反例: 最小正整数解: x = 1766319049
VERDICT: REFUTED
"""


def test_r5b_bites_question_mark_wording():
    """
    「两种独立方法一致? True」「方程成立? True」这种**问号**写法也算自述成功。

    第一版只匹配冒号(`一致: True`), 漏掉了问号, 于是这道题被放行 ——
    而它实际上算对了答案(1766319049 就是真解), 只是自研的结构判据写错了
    (周期为奇数时解的下标是 2L-1 而不是 L), 就宣布命题为假。
    """
    v, reason = judge_verdict(
        _res(PELL_STDOUT, "x = 1\n"), code="x = 1\n",
        claim="求 Pell 方程 x^2 - 61 y^2 = 1 的最小正整数解 (x,y)。",
    )
    assert v != "refuted", f"自述成立却判 REFUTED 被放行: {v}"
    assert "自相矛盾" in reason, f"理由没指到病灶: {reason}"
    print(f"  [咬住] Pell 问号写法 -> {v}")


def test_r5b_does_not_kill_multi_check_with_real_counterexample():
    """
    反向锁定: 同一份输出, **如果那一项检查确实是 False 并据此证伪**, 不该被拦。

    这是 R5b 最危险的误杀面 —— 真实的反例往往也伴随"某些检查 True"。
    所以判据不能是"出现过 True 就拦", 而是"**整体自述成功**却判 REFUTED"。
    下面的输出里 True 的那项是无关的前置检查, 而反例是实打实的。
    """
    out = ("先确认测试环境正常: 环境校验? True\n"
           "反例: n = 15, 4*15 = 60 但 60 不能表示成三个互质的 x,y,z\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n",
                              claim="证明 4/n = 1/x+1/y+1/z 有解。")
    assert v == "refuted", f"真反例被误杀: {v} / {reason}"
    print(f"  [放过] 真反例(前置检查 True 不构成矛盾) -> {v}")


# ---------- R6: 「几路验算对不上」被当成了反例 ----------
PARTITION_STDOUT = """p(100) via sympy.partition     = 190569292
p(100) via pentagonal recurrence = 58527732062
p(100) via DP (parts <= k)       = 190569292
All three methods agree: False
VERDICT: REFUTED
"""

PARTITION_CODE = """import sympy
n = 100
a = sympy.partition(n)
p = [0] * (n + 1); p[0] = 1
gp = sorted(set([1, 2, 5, 7, 12, 15, 22, 26, 35, 40, 51, 57]))
signs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
for i in range(1, n + 1):
    s = 0
    for idx, gk in enumerate(gp):
        if gk > i:
            break
        s += signs[idx % len(signs)] * p[i - gk]
    p[i] = s
print("p(100) via sympy.partition     =", a)
print("p(100) via pentagonal recurrence =", p[n])
print("All three methods agree:", a == p[n])
print("VERDICT: REFUTED")
"""


def test_r6_bites_method_disagreement():
    """
    核心: REFUTED 的依据只是「几路验算不一致」→ 必须降级。

    ★ 实测来源(2026-10-08 V2 hard-002 p(100))★
    前五道门**每一道都没错, 却全都放行了**: 第一层兜底规则"任何带数字的行
    都算见证"把 `via sympy.partition = 190569292` 抓成了反例 —— 而且抓到的
    正是**正确答案**那一行; 第二层查 set/break 形态, 代码里有 sorted() 所以不成立;
    R3 靶子不对(非双向题); R4 不匹配 "debug-known constant" 的措辞;
    R5a/b/c 都不命中。没有任何一道门在问「这个 REFUTED 拿什么当依据」。
    """
    v, reason = judge_verdict(
        _res(PARTITION_STDOUT, PARTITION_CODE), code=PARTITION_CODE,
        claim="求整数 100 的整数分拆数 p(100) 的精确值。",
    )
    assert v == "inconclusive", f"方法分歧当成反例竟然放行: {v}"
    assert "几路验算" in reason or "不一致" in reason, f"理由没指到病灶: {reason}"
    print(f"  [咬住] p(100) 方法分歧 -> {v}")


def test_r6_does_not_kill_refutation_with_explicit_counterexample():
    """
    反向锁定: **真反例 + 一路比对 False**, 不该被拦。

    这是 R6 最危险的误杀面 —— 真实反例的输出里也很常见
    「某个前置检查不通过」。所以豁免条件必须是「**输出里明确声明了反例**」,
    而不是「有没有出现 False」。
    """
    out = ("两种独立实现结果一致? False\n"
           "逐项复核后确认: 反例: n = 15, 4*15 = 60 无法表示成三个互质的 x,y,z\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n",
                              claim="证明 4/n = 1/x+1/y+1/z 对 2<=n<=1000 有解。")
    assert v == "refuted", f"真反例被 R6 误杀: {v} / {reason}"
    print(f"  [放过] 真反例(有反例声明, 分歧只是佐证) -> {v}")


def test_r6_does_not_kill_selfcheck_saying_no_disagreement():
    """
    反向锁定: 输出里说「不一致的项: 无」(即**自检通过**)时, 不能被当成分歧。

    这是「不一致」三字的经典误伤面 —— 否定句里也会出现这个词。
    """
    out = ("已完成自检, 不一致的项: 无\n"
           "对称性验证: x = -7, y = 12, 差值 = 169.0 != 0\n"
           "反例: x = -7, y = 12\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n",
                              claim="对所有整数 x,y 都有 x^2 - y^2 >= 0。")
    assert v == "refuted", f"自检通过的真反例被 R6 误杀: {v} / {reason}"
    print(f"  [放过] 自检「不一致的项: 无」+ 真反例 -> {v}")


# ---------- R3 补: part 级标签写成 PROVED/REFUTED ----------
PETERSEN_STDOUT = """Petersen 图边数: 15 (应为 15)
=== Part A: 3 着色构造 ===
顶点着色: [0..9] -> [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
  冲突! 边 (0,1) 端点同色 2
=== Part B: 不存在 2 着色证明 ===
z3 判定: unsat
  sat 不可达 -> 任意 2 着色尝试都会冲突 -> chi > 2 ✓
=== 总判定 ===
Part A (存在 3 着色): REFUTED
Part B (不存在 2 着色): PROVED
VERDICT: REFUTED
"""

PETERSEN_CODE = """from z3 import Bool, Solver, Xor, sat

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
         (5, 7), (6, 8), (7, 9), (8, 5), (9, 6),
         (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
N = 10
colors = [Bool("b_%d" % v) for v in range(N)]
s = Solver()
for u, v in EDGES:
    s.add(Xor(colors[u], colors[v]))
print("z3 判定:", s.check())
print("Part A (存在 3 着色): REFUTED")
print("Part B (不存在 2 着色): PROVED")
print("VERDICT: REFUTED")
"""

PETERSEN_CLAIM = "证明 Petersen 图的色数是 3：给出 3 着色，并证明不能 2 着色。"


def test_r3_bites_proved_refuted_part_labels():
    """
    ★ 实测翻车(2026-10-08 V2 hard-011 Petersen 色数)★
    双向题, z3 已经把 Part B(不存在 2 着色)证成了, 整体却判了 REFUTED。

    R3 之前拦住的是 hard-007, 那次模型写的是 `Part (1) ... : YES`。
    这次模型写的是**更规整的机器可读标签** `Part A: REFUTED` / `Part B: PROVED`,
    而判据只认 YES/True/成立 和 NO/False/失败 —— 门在**最规整的写法上失效**。
    """
    v, reason = judge_verdict(_res(PETERSEN_STDOUT, PETERSEN_CODE),
                              code=PETERSEN_CODE, claim=PETERSEN_CLAIM)
    assert v == "partial", f"一半证成了却放行成假: {v}"
    assert "自相矛盾" in reason, f"理由没指到病灶: {reason}"
    print(f"  [咬住] Petersen part 级 PROVED/REFUTED -> {v}")


def test_r3_does_not_kill_both_parts_refuted():
    """
    反向锁定: **两部分都 REFUTED** 的双向题, 不该被 R3 降级。

    这正是 R3 最大的误杀面 —— 双向题被整体证伪是完全合法的情况。
    判据必须是"一部分成立 + 另一部分不成立", 而不是"输出里有 REFUTED"。
    """
    out = ("Part A (存在 3 着色): REFUTED\n"
           "冲突! 边 (0,1) 端点同色 —— 该构造不合法\n"
           "Part B (不存在 2 着色): REFUTED\n"
           "反例: n = 15, 60 不能表示成三个互质的 x,y,z\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n",
                              claim=PETERSEN_CLAIM)
    assert v == "refuted", f"双向题被整体证伪却误杀: {v} / {reason}"
    print(f"  [放过] 双向题两部分都 REFUTED -> {v}")


# ---------- R5d: 各部分自判 INCONCLUSIVE, 却汇总成 REFUTED ----------
RAMSEY_STDOUT = """A：9 顶点全称命题的反例可满足性 = sat
A 判定： INCONCLUSIVE
8 顶点构造的红色边 = [(0, 1), (0, 2), (0, 7), (1, 2), (1, 3), (2, 3)]
构造检查：红色三角形 = (0, 1, 2)
独立求解所得见证经子集检查有效 = False
B 判定： INCONCLUSIVE
8 顶点完整染色空间上界：2^C(8,2) = 2^28 = 268435456
VERDICT: REFUTED
"""

RAMSEY_CLAIM = ("验证 R(3,4)=9：证明任意 9 个顶点的红蓝完全图必有红色三角形或"
                "蓝色 K4，并给出 8 个顶点无反例的着色。")


def test_r5d_bites_own_inconclusive_then_refuted():
    """
    ★ 实测翻车(2026-10-08 V2 hard-007, R(3,4)=9)★
    z3 对「找一个 9 顶点反例」回了 sat, 代码**紧接着校验见证、发现不成立**,
    约束编码错了。它却汇总成了 REFUTED —— 而 R(3,4)=9 是已知成立的定理。

    为什么 R3 拦不住: R3 要求输出里同时有"一部分成立"和"一部分不成立",
    这里两部分都是 INCONCLUSIVE, `_PART_TRUE`/`_PART_FALSE` 全部落空。
    """
    v, reason = judge_verdict(_res(RAMSEY_STDOUT, "x=1\n"), code="x=1\n",
                              claim=RAMSEY_CLAIM)
    assert v == "inconclusive", f"各部分都定不了却放行成假: {v}"
    assert "INCONCLUSIVE" in reason, f"理由没指到病灶: {reason}"
    print(f"  [咬住] Ramsey 各部分 INCONCLUSIVE 却 REFUTED -> {v}")


def test_r5d_does_not_kill_real_counterexample_with_inconclusive_part():
    """
    反向锁定: 一部分 inconclusive + **另一部分真找到反例**, 不该被拦。

    这是 R5d 最危险的误杀面 —— 双向题里"这半没验出来、那半找到反例了"
    汇总成 REFUTED 是完全合法的。判据必须要求"**没有任何反例声明**"。
    """
    out = ("A 判定： INCONCLUSIVE\n"
           "B 判定： REFUTED\n"
           "反例: n = 15, 4*15 = 60 不能表示成三个互质的 x,y,z\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n",
                              claim=RAMSEY_CLAIM)
    assert v == "refuted", f"有真反例却被 R5d 误杀: {v} / {reason}"
    print(f"  [放过] 另一半 inconclusive + 有反例声明 -> {v}")


# ---------- R7: 报的计数和它自己列的清单对不上 ----------
ERDOS_STRAUS_STDOUT = """Sanity check (small n, expected: all have solutions):
  n=2: solution=(1, 1, 1)
  n=4: solution=None
  n=5: solution=None
  n=6: solution=None
Total failures in [2,1000]: 997
Failed n values: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
VERDICT: REFUTED
"""


def test_r7_bites_count_list_mismatch():
    """
    ★ 实测翻车(2026-10-08 V2 hard-012 Erdős–Straus)★
    声明 997 个失败, 只列出 20 个。而 n=4 的解平凡到 `1/2+1/3+1/6 = 1` ——
    它自己的输出还写着 `expected: all have solutions`, 却对 n=4..23 全报 None。

    ★ 为什么这道门值得单开 ★
    前六道门问的全是「结论和**命题**对不上吗」, 没有一道问「结论和
    **它自己的输出**对不上吗」。而"声明 997、列出 20"是纯机械可查的,
    不需要知道 Erdős–Straus 是什么。
    """
    v, reason = judge_verdict(_res(ERDOS_STRAUS_STDOUT, "x=1\n"), code="x=1\n",
                              claim="验证 Erdos-Straus 猜想对 2 <= n <= 1000 成立。")
    assert v == "inconclusive", f"计数与清单矛盾却放行: {v}"
    # 夹具里没有反例声明 → 凭证门先拦。这份证据的根本缺陷就是"没交出凭证"。
    assert "凭证" in reason, f"理由没指到病灶: {reason}"
    print(f"  [咬住] Erdos-Straus 声明 997 却只列 20 项 -> {v}")


def test_r7_fallback_still_bites_with_credential():
    """
    兜底层仍要生效: **凭证齐全**时, 计数与清单对不上照样要拦。

    和 R7 的对照: 同一类缺陷, 一个没凭证(凭证门拦)、一个凭证齐全(兜底门拦)。
    """
    out = ("COUNTEREXAMPLE: n=4 | lhs=1 | rhs=2 | violated=True\n"
           "Failed n values: [4, 5, 6, 7]\n"
           "Total failures: 997\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n",
                              claim="证明 4/n = 1/x+1/y+1/z 对所有 n 有解。")
    assert v == "inconclusive", f"凭证齐全但计数矛盾却放行: {v}"
    assert "997" in reason, f"兜底门没指出计数矛盾: {reason}"
    print(f"  [咬住] 凭证齐全 + 计数 997 vs 清单 4 -> {v}（兜底层仍然生效）")


def test_r7_does_not_kill_matching_count_and_list():
    """反向锁定: 计数和清单**一致**的真反例, 不该被拦。"""
    out = ("反例: n = 15, 60 不能表示成三个互质的 x,y,z\n"
           "Failed n values: [15]\n"
           "Total failures: 1\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n",
                              claim="证明 4/n = 1/x+1/y+1/z 有解。")
    assert v == "refuted", f"计数一致的真反例被误杀: {v} / {reason}"
    print(f"  [放过] 计数与清单一致(1 vs 1) -> {v}")


def test_r7_does_not_kill_honest_truncation():
    """
    反向锁定: 明写「前 N 个」的截断**是诚实输出**, 不该被拦。

    计数远大于清单长度本身不是错误 —— 只要它说清楚了。
    """
    out = ("失败项太多, 只列前 3 个作示例\n"
           "Failed n values: [4, 5, 6]\n"
           "Total failures: 997\n"
           "反例: n = 4, 1/2+1/3+1/6 = 1 但方程要求别的分解\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n",
                              claim="证明 4/n = 1/x+1/y+1/z 对所有 n 有解。")
    assert v == "refuted", f"诚实的截断被 R7 误杀: {v} / {reason}"
    print(f"  [放过] 明写「只列前 3 个」的截断 -> {v}")


# ---------- 凭证制: 不匹配谎言, 匹配凭证 ----------
def test_credential_required_for_refuted():
    """
    ★ 核心: REFUTED 必须出示凭证行 ★

    这是对六道门的**一般化替代**。六道门各自认一种「说谎的形状」, 只能拦住
    已经见过的; 凭证制只问一件事 —— **你交出具体输入和违反它的值了吗**。
    没交出就是没交出, 不管它是「没找到」「几路对不上」还是「数错了」。
    """
    out = ("p(100) via sympy.partition     = 190569292\n"
           "p(100) via pentagonal recurrence = 58527732062\n"
           "All three methods agree: False\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n",
                              claim="求整数 100 的整数分拆数 p(100) 的精确值。")
    assert v == "inconclusive", f"没有凭证却放行了 REFUTED: {v}"
    assert "COUNTEREXAMPLE" in reason, f"理由没给出凭证格式: {reason}"
    print(f"  [咬住] 无凭证的 REFUTED -> {v}")


def test_credential_accepts_wellformed_counterexample():
    """
    反向锁定: **凭证齐全**的 REFUTED 必须放行 —— 否则这道门就变成了"一律不许证伪"。
    """
    out = ("对 n=15: 4*15 = 60\n"
           "  穷举所有互质三元组 (x,y,z), 无一满足 1/x+1/y+1/z = 1/15\n"
           "COUNTEREXAMPLE: n=15 | lhs=60 | rhs=无法分解为三个互质倒数 | "
           "violated=True\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n",
                              claim="证明 4/n = 1/x+1/y+1/z 对所有 n 有解。")
    assert v == "refuted", f"凭证齐全的真反例被误杀: {v} / {reason}"
    print(f"  [放过] 凭证齐全的真反例 -> {v}")


def test_credential_rejects_self_negating_violated_false():
    """
    `violated=False` —— 它自己说这不构成反例, 却判了 REFUTED。
    这种自相矛盾不该被当成有效凭证。
    """
    out = ("COUNTEREXAMPLE: n=15 | lhs=60 | rhs=60 | violated=False\n"
           "实际两边相等, 反例不成立\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n",
                              claim="证明 4/n = 1/x+1/y+1/z 对所有 n 有解。")
    assert v == "inconclusive", f"自己说 violated=False 却被放行: {v}"
    print(f"  [咬住] 凭证自认 violated=False -> {v}")


def test_credential_needs_a_concrete_input_value():
    """凭证里一个数字都没有 → 等于没交凭证。"""
    out = ("COUNTEREXAMPLE: n=bad | lhs=none | rhs=none | violated=True\n"
           "VERDICT: REFUTED\n")
    v, reason = judge_verdict(_res(out, "x=1\n"), code="x=1\n",
                              claim="证明 4/n = 1/x+1/y+1/z 对所有 n 有解。")
    assert v == "inconclusive", f"没有具体数值的凭证被放行: {v}"
    print(f"  [咬住] 凭证无任何数值 -> {v}")


def test_credential_alone_is_not_enough():
    """
    ★ 凭证门**取代不了**兜底门 —— 这条测试是那个结论的证据 ★

    凭证门管的是「在不在」: REFUTED 有没有交出具体输入 + 违反值。
    兜底门管的是「在的东西自不自相矛盾」: 凭证在场, 但它和程序其余输出打架。

    ★ Pell-61 就是跨不过去的那条 ★
    它**确实**打了反例声明(凭证在场), 但那个值 1766319049 同时被另一句
    「方程成立? True」确认成正解 —— 凭证自相矛盾。
    实测判定: 凭证门**漏掉**, 兜底门拦住。

    为什么值得单独立一条测试:
    曾经打算「凭证机制立住了就删掉六道兜底门」。真拿 5 条历史假反例做判决
    实验(见 tests/tools/credential_vs_fallback.py)才发现凭证漏了 Pell-61。
    这个结论一旦没测住, 下一个版本很可能又有人把兜底门删了 —— 然后静悄悄地
    开一个洞: 一个「交出了凭证、但凭证自己打自己脸」的假反例会直接流出去。
    """
    v, reason = judge_verdict(_res(PELL_STDOUT, "x = 1\n"), code="x = 1\n",
                              claim="求 Pell 方程 x^2 - 61 y^2 = 1 的最小正整数解 (x,y)。")
    # 凭证层会放过它(凭证在场) —— 这正是它必须由兜底层来接的原因
    assert _counterexample_credential(PELL_STDOUT) is not None, \
        "夹具前提变了: Pell-61 现在居然没有凭证, 判决实验的结论要重测"
    # 最终仍然被拦下, 但拦住它的是**一致性**这一层
    assert v != "refuted", f"凭证自相矛盾却放行了: {v}"
    assert "自相矛盾" in reason, f"拦住它的不是一致性检查: {reason}"
    print(f"  [钉住] 凭证在场但自相矛盾 -> {v}（兜底层不可删）")


if __name__ == "__main__":
    print("机器门 —— 咬合 + 不误杀 (PROVED 侧 R1/R2, REFUTED 侧 R3/R4/R5/R6)")
    print("=" * 66)
    tests = [
        test_bites_false_proved,
        test_does_not_kill_complete_proof,
        test_finite_loop_without_nothing_found_not_killed,
        test_conclusion_word_without_premise_marker_not_killed,
        test_scoped_claim_not_killed,
        test_unscoped_claim_with_same_code_is_caught,
        test_r3_bites_wrong_subject,
        test_r3_does_not_kill_simple_refuted,
        test_r3_does_not_kill_consistent_refuted,
        test_r4_bites_hardcoded_reference,
        test_r4_does_not_kill_real_counterexample,
        test_r4_does_not_kill_computed_large_number,
        test_r5_bites_refuted_while_finding_nothing,
        test_r5_does_not_kill_partial_path,
        test_r5_does_not_kill_genuine_refutation,
        test_r5b_bites_refuted_after_all_passed,
        test_r5b_fallback_still_bites_with_credential,
        test_r5b_does_not_kill_all_passed_proved,
        test_r5b_bites_question_mark_wording,
        test_r5b_does_not_kill_multi_check_with_real_counterexample,
        test_r6_bites_method_disagreement,
        test_r6_does_not_kill_refutation_with_explicit_counterexample,
        test_r6_does_not_kill_selfcheck_saying_no_disagreement,
        test_r3_bites_proved_refuted_part_labels,
        test_r3_does_not_kill_both_parts_refuted,
        test_r5d_bites_own_inconclusive_then_refuted,
        test_r5d_does_not_kill_real_counterexample_with_inconclusive_part,
        test_r7_bites_count_list_mismatch,
        test_r7_fallback_still_bites_with_credential,
        test_r7_does_not_kill_matching_count_and_list,
        test_r7_does_not_kill_honest_truncation,
        test_credential_required_for_refuted,
        test_credential_accepts_wellformed_counterexample,
        test_credential_rejects_self_negating_violated_false,
        test_credential_needs_a_concrete_input_value,
        test_credential_alone_is_not_enough,
    ]
    failed = []
    for t in tests:
        print(f"\n>>> {t.__name__}")
        try:
            t()
        except AssertionError as e:
            print(f"  [失败] {e}")
            failed.append(t.__name__)
    print("\n" + "=" * 60)
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print(f"全部通过 ({len(tests)}/{len(tests)})")