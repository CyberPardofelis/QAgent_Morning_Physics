"""small-math-agent-benchmark 20 题的 0 LLM 验证代码 —— 单一来源。

为什么单独一个文件: 探测脚本(probe_base20.py)和 pytest 测试都要跑这些
代码。两处各写一份必然漂移, 而「验证器本身写错」是真实发生过的
(Ramsey 那版把顶点范围硬编码成 6, K5 分支恒判错, 整题误报 REFUTED)。

这些代码是**我方写的标准答案验证器**, 不是 LLM 写的。跑它们测的是
「沙箱 + sympy/networkx 能不能算对」, 与 LLM 水平无关 —— 所以这层
0 token, 适合进 CI。

判据统一: 全部 gold=proved。这 20 题要么验证某恒等式/性质, 要么
求出某确定值, 目标都是「证明题成立」, 所以任何非 PROVED 都算没答对。
(别拿这个 gold 套 P4 的 8 道陷阱题 —— 那批的 gold 是「认输」。)
"""
from __future__ import annotations

#: 每题一段独立代码, 自己 print('VERDICT: ...'), 不依赖外部 helper。
VERIFIER_CODE: dict[str, str] = {
    "num-001-euler-prime-polynomial": """
from sympy import isprime, factorint
bad = [n for n in range(40) if not isprime(n * n + n + 41)]
v40 = 40 * 40 + 40 + 41
print("non-prime in n=0..39:", bad)
print("n=40:", v40, "factorization:", factorint(v40), "41^2:", 41 ** 2)
if bad:
    print("VERDICT: REFUTED")
elif v40 == 41 ** 2:
    # n=0..39 是题目指定的范围, 验完即全域; n=40 的值是可直接算出的事实。
    print("VERDICT: PROVED")
else:
    print("VERDICT: REFUTED")
""",
    "num-002-goldbach-small": """
from sympy import isprime, primerange
primes = list(primerange(2, 1001))
ps = set(primes)
missing = [n for n in range(4, 1001, 2)
           if not any((n - p) in ps for p in primes if p <= n - p)]
print("even numbers in [4,1000] with no prime pair:", missing)
print("VERDICT: PROVED" if not missing else "VERDICT: REFUTED")
""",
    "num-003-carmichael-561": """
from math import gcd
from sympy import isprime
bad = [a for a in range(1, 561) if gcd(a, 561) == 1 and pow(a, 560, 561) != 1]
print("561 is prime?", isprime(561), "| coprime violations:", len(bad))
if not isprime(561) and not bad:
    print("VERDICT: PROVED")
else:
    print("VERDICT: REFUTED")
""",
    "num-004-100-factorial-trailing-zeros": """
import math
n = 100
digits = str(math.factorial(n))
count = len(digits) - len(digits.rstrip("0"))
formula = n // 5 + n // 25
print("trailing zeros:", count, "| formula:", formula)
print("VERDICT: PROVED" if count == 24 == formula else "VERDICT: REFUTED")
""",
    "num-005-inclusion-exclusion-100": """
cnt = sum(1 for k in range(1, 101) if k % 2 and k % 3 and k % 5)
print("count of integers in 1..100 not divisible by 2,3,5:", cnt)
print("VERDICT: PROVED" if cnt == 26 else "VERDICT: REFUTED")
""",
    "alg-006-fibonacci-matrix": """
import sympy as sp
M = sp.Matrix([[1, 1], [1, 0]])
bad = []
for n in range(1, 21):
    lhs = M ** n
    rhs = sp.Matrix([[sp.fibonacci(n + 1), sp.fibonacci(n)],
                     [sp.fibonacci(n), sp.fibonacci(n - 1)]])
    if lhs != rhs:
        bad.append(n)
print("mismatching n in 1..20:", bad)
print("VERDICT: PROVED" if not bad else "VERDICT: REFUTED")
""",
    "alg-007-am-gm": """
import sympy as sp
a, b = sp.symbols("a b", positive=True)
diff = sp.expand(((a + b) / 2) ** 2 - a * b)
factored = sp.factor(diff)
print("((a+b)/2)^2 - ab =", factored)
# 用 == / simplify 判定, 不用 `is True` —— 后者对 sympy 对象恒为 False。
ok = sp.simplify(factored - (a - b) ** 2 / 4) == 0
print("equals (a-b)^2/4:", ok)
print("VERDICT: PROVED" if ok else "VERDICT: PARTIAL")
""",
    "alg-008-circle-max-linear-sum": """
import sympy as sp
t = sp.symbols("t", real=True)
# x=cos t, y=sin t 上的 x+y 最大值在 t=pi/4 取到。
# ★ 这里 x 和 y 各是 cos/sin 的值, 不是 x+y 本身 —— 早先误把 x+y 的最大值
#   同时当成 x 和 y 代进约束, 算出 x^2+y^2=4, 于是自己判成 PARTIAL。
x, y = sp.cos(sp.pi / 4), sp.sin(sp.pi / 4)
value = sp.simplify(x + y)
on_circle = sp.simplify(x ** 2 + y ** 2)
print("at t=pi/4: x =", x, "y =", y, "| x+y =", value)
print("constraint x^2+y^2 there:", on_circle)
# 上界: (x+y)^2 <= 2(x^2+y^2) = 2, 且上面这个点确实取到, 所以就是最大值。
upper = sp.sqrt(2)
print("upper bound from (x+y)^2 <= 2(x^2+y^2):", upper)
ok = (value == upper) and (on_circle == 1)
print("VERDICT: PROVED" if ok else "VERDICT: PARTIAL")
""",
    "alg-009-determinant-multiplicative": """
import random
import sympy as sp
random.seed(20261008)
worst = 0.0
for _ in range(100):
    A = sp.Matrix(3, 3, [random.randint(-9, 9) for _ in range(9)])
    B = sp.Matrix(3, 3, [random.randint(-9, 9) for _ in range(9)])
    worst = max(worst, float(abs((A * B).det() - A.det() * B.det())))
print("worst |det(AB)-det(A)det(B)| over 100 random 3x3 pairs:", worst)
print("VERDICT: PROVED" if worst < 1e-9 else "VERDICT: REFUTED")
""",
    "comb-010-eight-queens": """
def solve(n):
    res, q = [], [-1] * n
    def bt(i):
        if i == n:
            res.append(tuple(q))
            return
        for c in range(n):
            if all(c != q[j] and abs(c - q[j]) != i - j for j in range(i)):
                q[i] = c
                bt(i + 1)
    bt(0)
    return res
sols = solve(8)
print("8-queens solutions:", len(sols))
print("VERDICT: PROVED" if len(sols) == 92 else "VERDICT: REFUTED")
""",
    "logic-011-de-morgan": """
from itertools import product
def NOT(v): return not v
def AND(a, b): return a and b
def OR(a, b): return a or b
bad = []
for A, B in product([False, True], repeat=2):
    if NOT(AND(A, B)) != OR(NOT(A), NOT(B)):
        bad.append(("and", A, B))
    if NOT(OR(A, B)) != AND(NOT(A), NOT(B)):
        bad.append(("or", A, B))
print("counterexamples:", bad)
print("VERDICT: PROVED" if not bad else "VERDICT: REFUTED")
""",
    "opt-012-small-lp": """
# 顶点枚举: 目标 x+y 在 x+2y<=4, 3x+y<=6, x,y>=0 上的最大值
from itertools import combinations
from fractions import Fraction

def feasible(x, y):
    return x + 2 * y <= 4 and 3 * x + y <= 6 and x >= 0 and y >= 0

lines = [(lambda x: 4 - 2 * x, lambda y: y, 0, 4),   # x+2y=4
         (lambda x: (6 - x) / 3, lambda y: y, 0, 6)]   # 3x+y=6
verts = []
for a, b in [(Fraction(0), Fraction(0)), (Fraction(4), Fraction(0)),
             (Fraction(0), Fraction(4))]:
    verts.append((a, b))
for f, g, ylo, yhi in lines:
    # x+2y=4 与 3x+y=6 的交点: x=8/5, y=6/5
    verts.append((Fraction(8, 5), Fraction(6, 5)))
    verts.append((Fraction(2), Fraction(0)))   # 3x+y=6 在 y=0
    verts.append((Fraction(0), Fraction(2)))   # x+2y=4 在 x=0
feas = [(x, y) for x, y in verts if feasible(x, y)]
best = max(feas, key=lambda p: p[0] + p[1])
print("feasible vertices:", sorted(set(feas)))
print("argmax:", best, "| max value:", best[0] + best[1])
print("VERDICT: PROVED" if best[0] + best[1] == Fraction(14, 5) else "VERDICT: REFUTED")
""",
    "comb-013-hanoi": """
def min_steps(n):
    return 2 ** n - 1
bad = [n for n in range(1, 11) if min_steps(n) != 2 ** n - 1]
print("n with wrong step count in 1..10:", bad)
# 也跑一遍真实递推, 确认递推本身给出同一个数
def recur(n, memo={}):
    if n <= 1:
        return 0 if n == 0 else 1
    if n not in memo:
        memo[n] = 2 * recur(n - 1, memo) + 1
    return memo[n]
bad2 = [n for n in range(1, 11) if recur(n) != 2 ** n - 1]
print("n where recurrence disagrees with 2^n-1:", bad2)
print("VERDICT: PROVED" if not bad and not bad2 else "VERDICT: REFUTED")
""",
    "graph-014-ramsey-3-3": """
from itertools import combinations

def has_mono_triangle(n, red):
    # n 必须来自参数 —— 早期版本硬编码 range(6), K5 分支因此恒判有单色
    # 三角形, cnt5 永远 0, 整题误报 REFUTED。是验证器坏了不是工具坏了。
    for tri in combinations(range(n), 3):
        es = [tuple(sorted(e)) for e in combinations(tri, 2)]
        cols = [e in red for e in es]
        if all(cols) or not any(cols):
            return True
    return False

def count_colorings(n):
    E = list(combinations(range(n), 2))
    without = 0
    for mask in range(1 << len(E)):
        red = {E[i] for i in range(len(E)) if (mask >> i) & 1}
        if not has_mono_triangle(n, red):
            without += 1
    return (1 << len(E)) - without, without

k5_with, k5_without = count_colorings(5)
k6_with, k6_without = count_colorings(6)
print("K5: with mono =", k5_with, "without mono =", k5_without)
print("K6: with mono =", k6_with, "without mono =", k6_without)
# R(3,3)=6: K6 全染色有单色三角形 且 K5 存在反例。两个子命题都是穷尽的。
print("VERDICT: PROVED" if k6_without == 0 and k5_without > 0 else "VERDICT: REFUTED")
""",
    "graph-015-labeled-trees-5": """
import itertools

def is_tree(n, edges):
    if len(edges) != n - 1:
        return False
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[ru] = rv
    return len({find(i) for i in range(n)}) == 1

E = list(itertools.combinations(range(5), 2))
cnt = sum(1 for m in range(1 << len(E))
          if is_tree(5, [E[i] for i in range(len(E)) if (m >> i) & 1]))
print("labeled trees on 5 vertices:", cnt, "(Cayley 5^3 =", 5 ** 3, ")")
print("VERDICT: PROVED" if cnt == 125 else "VERDICT: REFUTED")
""",
    "graph-016-petersen-properties": """
import networkx as nx
G = nx.petersen_graph()
degrees = set(dict(G.degree()).values())
diameter, girth = nx.diameter(G), nx.girth(G)
print("nodes:", G.number_of_nodes(), "| degrees:", degrees)
print("diameter:", diameter, "| girth:", girth)
ok = (G.number_of_nodes() == 10 and degrees == {3}
      and diameter == 2 and girth == 5)
print("VERDICT: PROVED" if ok else "VERDICT: REFUTED")
""",
    "prob-017-birthday-paradox": """
from fractions import Fraction
no_match = Fraction(1, 1)
for i in range(23):
    no_match *= Fraction(365 - i, 365)
at_least_one = 1 - no_match
print("exact:", at_least_one)
print("decimal:", float(at_least_one))
print("VERDICT: PROVED" if float(at_least_one) > 0.5 else "VERDICT: REFUTED")
""",
    "prob-018-two-dice-expectation": """
from itertools import product
outcomes = list(product(range(1, 7), repeat=2))
mean = sum(a + b for a, b in outcomes) / len(outcomes)
print("outcomes:", len(outcomes), "| mean sum:", mean)
print("VERDICT: PROVED" if abs(mean - 7) < 1e-12 else "VERDICT: REFUTED")
""",
    "alg-019-pythagorean-formula": """
import sympy as sp
m, n = sp.symbols("m n", positive=True)
a, b, c = m ** 2 - n ** 2, 2 * m * n, m ** 2 + n ** 2
resid = sp.expand(a ** 2 + b ** 2 - c ** 2)
print("a^2+b^2-c^2 simplifies to:", resid)
print("VERDICT: PROVED" if resid == 0 else "VERDICT: PARTIAL")
""",
    "num-020-bisection-root": """
def f(x):
    return x ** 3 - 2 * x - 5

lo, hi = 2.0, 3.0
if f(lo) * f(hi) > 0:
    print("no sign change on [2,3]")
    print("VERDICT: INCONCLUSIVE")
else:
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-12:
            break
    root = (lo + hi) / 2
    residual = abs(f(root))
    print("root:", root, "| residual:", residual, "| bracket width:", hi - lo)
    ok = abs(root - 2.0945514815423265) < 1e-6 and residual < 1e-6
    print("VERDICT: PROVED" if ok else "VERDICT: REFUTED")
""",
}

GOLD = "proved"