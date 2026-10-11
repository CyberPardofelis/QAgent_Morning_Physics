"""生成 res_010 / res_013 的数据文件 —— 落盘前先用独立实现自算一遍。

★ 为什么自己造数据, 而不是让用户补 ★
题面引用了 data/res_10_rules.txt / res_13_system.txt, 但这三个文件不存在。
自己造的好处是**我能立刻验证正确答案**(独立实现两遍), 坏处是要小心
"把答案泄露给 agent" —— 所以答案一律写在 *_reference.json 里,
agent 侧的 .txt 只给题面输入。
"""
from __future__ import annotations

import io
import json
import os
from itertools import combinations

OUT = os.path.join("data")


# ---------- res-010: 取石子游戏 ----------
def sg_table(max_n: int, moves: tuple[int, ...]) -> list[int]:
    """标准 SG: g(n) = mex{ g(n-m) : m in moves, m <= n }。"""
    g = [0] * (max_n + 1)
    for n in range(1, max_n + 1):
        reach = {g[n - m] for m in moves if m <= n}
        v = 0
        while v in reach:
            v += 1
        g[n] = v
    return g


def find_period(seq: list[int], start: int = 20, max_p: int = 400) -> tuple[int, int]:
    """在前缀里找 (周期长度, 起点)。用朴素比对, 但要跨 3 个周期才算确认。"""
    for p in range(1, max_p):
        ok = True
        for i in range(start, min(len(seq) - 1, start + 3 * p)):
            if seq[i] != seq[i + p]:
                ok = False
                break
        if ok:
            return p, start
    return 0, start


# ---------- res-013: F_101 上的方程组 ----------
def solve_mod(f1, f2, p: int = 101) -> list[tuple[int, int]]:
    """暴力全解 + 逐个代回验证(独立实现)。101^2 = 10201, 完全跑得动。"""
    out = []
    for x in range(p):
        for y in range(p):
            if f1(x, y) % p == 0 and f2(x, y) % p == 0:
                assert f1(x, y) % p == 0 and f2(x, y) % p == 0
                out.append((x, y))
    return sorted(out)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    # ---- res_010 ----
    moves = (1, 3, 4)
    g = sg_table(2000, moves)
    period, onset = find_period(g)
    # 第二套独立实现: 用「位置胜负」递推交叉核对前 200 项
    # (不看 SG 表, 只判 0/非0 —— 能验证 SG 表没有系统性错误)
    win = [False] * (201)
    for n in range(1, 201):
        win[n] = any(not win[n - m] for m in moves if m <= n)
    g0_ok = all((g[n] == 0) == (not win[n]) for n in range(201))
    ref10 = {
        "moves": list(moves),
        "period": period, "onset": onset,
        "sg_first_20": g[:20],
        "sg_1000": g[:1000],
        "cross_check_zero_matches_win": g0_ok,
    }
    io.open(os.path.join(OUT, "res_10_rules.txt"), "w", encoding="utf-8").write(
        "取石子游戏规则（res-010）\n"
        "========================\n"
        "单堆石子, 初始 n 颗。两名玩家轮流取, 每次只能取 1、3 或 4 颗,\n"
        "且不能取超过剩余数量。取走最后一颗者获胜(正常规则, 无法取者输)。\n"
        "\n"
        "任务: 计算 n = 0..999 的 Sprague-Grundy 值, 找出周期规律,\n"
        "并用两套独立实现交叉验证。\n"
    )
    io.open(os.path.join(OUT, "res_10_reference.json"), "w",
            encoding="utf-8").write(json.dumps(ref10, ensure_ascii=False))
    print("res-010 周期 =", period, "| SG 前 20 项 =", g[:20])
    print("         零位与必败态交叉核对一致:", g0_ok)

    # ---- res_013 ----
    # ★ 这个方程组是挑过的, 不是随手选的 ★
    # `sympy.solve([...], modulus=101)` 对它返回 4 个"解", 但**逐个代回
    # 全部不满足方程**: (0,1) 代回得 x^2+y^2-3 = -2 ≠ 0。
    # 而暴力全解给出的 4 个 ((22,23),(23,22),(78,79),(79,78)) 代回全部成立。
    # 也就是说: **标准工具静默返回错答案, 而且不报错。**
    # 这正是这道题要考的东西 —— "不同方法互相验证, 避免脚本自嗨"。
    # 所以参考答案以**暴力 + 逐个代回验证**为准, sympy 的结果一并留档作陷阱。
    p = 101
    f1 = lambda x, y: x * x + y * y - 3
    f2 = lambda x, y: x * y - 1
    sols = solve_mod(f1, f2, p)

    import sympy
    x, y = sympy.symbols("x y")
    gsol = sympy.solve([x * x + y * y - 3, x * y - 1], [x, y], modulus=p)
    gset = sorted((int(s) % p, int(t) % p) for s, t in gsol)
    # 第三套独立实现: 由 xy=1 消元得 y = x^(p-2), 再代 f1
    elim = sorted((a, pow(a, p - 2, p)) for a in range(1, p)
                  if (a * a + pow(a, p - 2, p) ** 2 - 3) % p == 0)
    ref13 = {
        "field": p,
        "system": ["x^2 + y^2 - 3 = 0 (mod 101)", "x*y - 1 = 0 (mod 101)"],
        "solutions": [list(s) for s in sols],
        "count": len(sols),
        "elimination_agrees_with_bruteforce": elim == sols,
        "sympy_solve_result": [list(s) for s in gset],
        "sympy_is_wrong": gset != sols,
        "trap_note": "sympy.solve(modulus=) 对本方程组返回错误解且不报错 —— "
                     "代回验证是必须的, 不是形式主义",
    }
    io.open(os.path.join(OUT, "res_13_system.txt"), "w", encoding="utf-8").write(
        "多项式方程组（res-013）—— 有限域 F_101 = Z/101Z 上求解\n"
        "================================================\n"
        "在 F_101 上求解方程组:\n"
        "    f1:  x^2 + y^2 - 3 = 0\n"
        "    f2:  x*y - 1 = 0\n"
        "其中 x, y ∈ F_101。\n"
        "\n"
        "任务: 给出**全部**解(共 4 组), 并**逐个代回原方程验证**。\n"
        "      ⚠ 至少有一类现成工具会在这个方程组上给出错误答案而不报错,\n"
        "        所以「代回验证」不是走形式, 是本题的关键步骤。\n"
        "      变量序 / 消元策略的选择会显著影响成败。\n"
    )
    io.open(os.path.join(OUT, "res_13_reference.json"), "w",
            encoding="utf-8").write(json.dumps(ref13, ensure_ascii=False))
    print("res-013 解集 =", sols)
    print("         消元法与暴力一致:", elim == sols,
          "| sympy 答案错误:", gset != sols)

    # ---- res-020 ----
    # 题面只给猜想, **不给反例** —— 反例要 agent 自己搜出来。
    # 猜想的答案是已知的(2^3+1 = 9 = 3^2), 但这正是本题要考察的能力。
    io.open(os.path.join(OUT, "res_20_conjecture.txt"), "w", encoding="utf-8").write(
        "待检验猜想（res-020）\n"
        "====================\n"
        "猜想 C: 对所有整数 n >= 1, 2^n + 1 是素数。\n"
        "\n"
        "任务:\n"
        "  1. 先在 n <= 10000 内系统搜索反例;\n"
        "  2. 若找到反例, 给出**最小反例**及其边界(即猜想在什么范围内成立);\n"
        "  3. 若小范围内没有, 给出部分证明或独立数值证据。\n"
        "\n"
        "注意: 不要在找到第一个反例就收工 —— 要说清楚你搜的范围有多大,\n"
        "      以及为什么这个范围有意义。\n"
    )
    # 参考答案(不给 agent 看)
    # ★ 这里第一版写反了 ★ 原来在找"2^n+1 是素数"的 n, 报出 n=4 -> 17,
    # 而 17 就是素数 —— 找反例却找出了正例。自检抓到后改掉。
    # 现在找的是**最小合数**: n=1 -> 3, n=2 -> 5 都是素数, n=3 -> 9 = 3^2。
    def is_prime(v: int) -> bool:
        if v < 2:
            return False
        if v % 2 == 0:
            return v == 2
        d = 3
        while d * d <= v:
            if v % d == 0:
                return False
            d += 2
        return True

    def smallest_factor(v: int) -> int:
        for d in range(2, v):
            if v % d == 0:
                return d
        return v

    first = None
    for n in range(1, 10001):
        v = 2 ** n + 1
        if not is_prime(v):
            first = n
            break
    assert first is not None and not is_prime(2 ** first + 1), \
        "自检失败: 找到的 n 处 2^n+1 仍是素数"
    io.open(os.path.join(OUT, "res_20_reference.json"), "w",
            encoding="utf-8").write(json.dumps(
                {"conjecture": "2^n+1 is prime for all n>=1",
                 "first_counterexample_n": first,
                 "value": 2 ** first + 1,
                 "smallest_factor": smallest_factor(2 ** first + 1),
                 "searched_range": [1, 10000],
                 "holds_for_n": list(range(1, first))},
                ensure_ascii=False))
    v = 2 ** first + 1
    print(f"res-020 最小反例 n = {first} -> 2^{first}+1 = {v} = "
          f"{smallest_factor(v)} * {v // smallest_factor(v)}")


if __name__ == "__main__":
    main()