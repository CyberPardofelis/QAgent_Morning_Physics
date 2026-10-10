"""`extract_code` 必须真能抽出可执行的 Python —— 用 smoke 里的真实失败原文钉住。

## 为什么单独写这个

2026-10-10 第一次 smoke: 三个角色管线是通的, 但三轮 Worker 全挂在**抽取**上:
  · 代码块首行是中文说明, 带 `→`  -> SyntaxError
  · 代码块首行是"用 python 从 n=0 开始" -> NameError: python 未定义
  · 说明里带 `²` -> SyntaxError

三次都白烧一个 Explorer 轮次。如果不管, 我会写下"模型不会写 Python"这个结论,
而真相是我的抽取器坏了。**这跟之前 res-019 那次一模一样**:
外部看着像"模型不听话", 实际什么都没发生。所以这类抽取必须用真实样本钉死,
不能只测"完美输入"。

## 反向要求
`test_keeps_valid_code` 那一组保证: 正常的代码不能被"救活"逻辑改坏。
抽取器为了容错而乱切代码, 一样是灾难。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from agent.tools.research.roles._util import extract_code  # noqa: E402
from agent.tools.research.roles.worker import looks_like_code  # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    FAILS.append(f"{label}" if not cond else "")
    print(f"  [{'OK ' if cond else 'FAIL'}] {label}"
          + (f"   {detail}" if not cond and detail else ""))


def _parses(src: str) -> bool:
    import ast
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------- 真实失败样本

S1 = """这是思路 → 先枚举再判素数：

```python
用 python 从 n=0 开始枚举, f(n) = n² + n + 41
for n in range(200):
    v = n*n + n + 41
    if not is_prime(v):
        print(n, v)
```

这样就能找到第一个反例。"""

S2 = """```python
思路: 用 python 枚举 → 逐个试除
import sympy
for n in range(200):
    v = n**2 + n + 41
    if not sympy.isprime(v):
        print(n, v)
```"""

S3 = """```python
# 从 n=0 开始, f(n) = n² + n + 41
# 用试除法判断素数
def is_prime(m):
    if m < 2: return False
    i = 2
    while i*i <= m:
        if m % i == 0: return False
        i += 1
    return True
for n in range(200):
    v = n*n + n + 41
    if not is_prime(v):
        print(n, v)
```"""

# ---------------------------------------------------------------- 正常输入不能被改坏

GOOD = """```python
import sympy
ans = None
for n in range(200):
    v = n**2 + n + 41
    if not sympy.isprime(v):
        ans = (n, v)
        break
print("ANSWER: " + __import__("json").dumps({"n": ans[0]}))
```"""

TWO_BLOCKS = """先试个小的：

```python
x = 1
```

正式的：

```python
import sympy
print(sympy.isprime(1681))
```"""

# ★ smoke 第二轮的真实输出原文 ★
# 模型的输出是**完全正确**的: 干净围栏、对的答案、对的制品。
# 我的抽取器却把语言标记 "python" 留在了代码第一行, 于是
# `unterminated string literal`, 三轮全废。
#
# 为什么我原来的单元测试没抓到: 它只断言"抽出来能 parse"和"包含某关键字",
# 而错块的"救活"版本恰好也能满足这两条。**测试太松, 等于给真洞发通行证。**
REAL_SMOKE = """```python
import sympy

def trial_isprime(x):
    if x < 2:
        return False
    i = 3
    while i * i <= x:
        if x % i == 0:
            return False
        i += 2
    return True

n = 40
v = n * n + n + 41
print("ANSWER: " + __import__("json").dumps({"n": n, "v": v,
      "factorization": sympy.factorint(v)}))
```"""

#: 抽出来的代码首行**不允许**是语言标记或纯散文。
LANG_WORDS = ("python", "py", "json", "javascript", "bash", "sh", "text")


def main() -> int:
    print("=" * 70)
    print("extract_code: 真实失败样本能不能救回来")
    print("=" * 70)

    for name, sample, must_contain in (
            ("S1 中文说明带箭头+散文首行", S1, "is_prime"),
            ("S2 散文首行 + 正常代码", S2, "sympy"),
            ("S3 注释里有上标 ²", S3, "for n in range"),
    ):
        code = extract_code(sample)
        check(f"{name} -> 抽出的代码可解析", _parses(code),
              f"抽出的是:\n{code[:200]}")
        check(f"{name} -> 保留了真正的代码体", must_contain in code,
              f"缺 {must_contain!r}")

    print("\n" + "=" * 70)
    print("反向: 正常代码不能被容错逻辑改坏")
    print("=" * 70)
    g = extract_code(GOOD)
    check("好代码原样保留", _parses(g) and "ANSWER:" in g, f"得到:\n{g[:200]}")
    t = extract_code(TWO_BLOCKS)
    check("多块时取最有用的那块(不是第一个)",
          _parses(t) and "isprime" in t, f"得到:\n{t[:200]}")
    check("空输入不炸", extract_code("") == "")

    print("\n" + "=" * 70)
    print("★ smoke 真实输出: 语言标记不能漏进代码 ★")
    print("=" * 70)
    r = extract_code(REAL_SMOKE)
    first = (r.split("\n") or [""])[0].strip().lower()
    check("smoke 原文抽出的代码可解析", _parses(r), f"第一行={first!r}")
    check("★ 首行不是语言标记",
          first not in LANG_WORDS, f"第一行是 {first!r} —— 围栏语言名漏进来了")
    check("smoke 原文保留了真正的代码体",
          "trial_isprime" in r and "ANSWER:" in r, f"得到:\n{r[:200]}")
    for name, sample in (("S1", S1), ("S2", S2), ("S3", S3), ("GOOD", GOOD)):
        c = extract_code(sample)
        f0 = (c.split("\n") or [""])[0].strip().lower()
        check(f"{name} 首行不是语言标记", f0 not in LANG_WORDS,
              f"第一行是 {f0!r}")

    print("=" * 70)
    bad = [f for f in FAILS if f]
    if bad:
        print(f"x {len(bad)} 项未过:")
        for b in bad:
            print("  - " + b)
        return 1

    # ================================================================
    # looks_like_code: part1 实测里 Worker 真的交上来的两份"代码"
    # ================================================================
    print()
    print("=" * 70)
    print("★ looks_like_code: 挡住『直接手写答案』")
    print("=" * 70)

    # part1 实测逐字: 这个在 Python 里是**合法**的 dict 字面量,
    # 所以代码"跑通了"、stdout 为空、什么都没算 —— 比报错更阴险。
    print("\n>>> 该挡住的")
    for label, junk, why in (
        ("纯 JSON 答案(实测)",
         '{"answer": 1}', "这是数据不是程序"),
        ("纯 JSON 数组(实测)",
         "[1, 2, 3]", "数组字面量"),
        ("纯字符串答案",
         '"答案是 1"', "字符串字面量"),
        ("英文散文(实测)",
         "I'll execute the plan to compute eigenvalues.", "散文"),
        ("空", "", "空的"),
    ):
        ok, reason = looks_like_code(junk)
        check(f"{label} 被挡", not ok, f"竟然通过了: {reason!r}")
        check(f"{label} 理由可读", bool(reason), "没给理由")

    print("\n>>> 不该挡住的")
    for label, good in (
        ("带 print 的真程序",
         'import sympy\nprint(sympy.simplify("x**2+2*x+1"))'),
        ("带赋值的真程序", "x = 3\ny = x * 2\nprint(y)"),
        ("带循环的真程序",
         "for n in range(10):\n    print(n)"),
        ("只有一个 print 的短程序",
         'print("ANSWER: {}".format(1 + 1))'),
        ("定义函数", "def f(x):\n    return x + 1\nprint(f(1))"),
    ):
        ok, reason = looks_like_code(good)
        check(f"{label} 放行", ok, f"被挡了: {reason!r}")

    print("=" * 70)
    bad = [f for f in FAILS if f]
    if bad:
        print(f"x {len(bad)} 项未过:")
        for b in bad:
            print("  - " + b)
        return 1
    print("o extract_code + looks_like_code: 全部通过")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())