# -*- coding: utf-8 -*-
"""
V2 hard-013 的人工基准验证器: 用 BBP 公式求 pi 十六进制展开第 10^6 位。

★ 核心恒等式(这份代码的全部效率来源)★
    目标数字 = int(frac(16^j · pi) · 16), 即 16^j·pi 小数部分的第一个十六进制位。
    BBP:  pi = sum_{k>=0} S_k / 16^k,  S_k = 4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6)

    16^j · pi = sum_k S_k · 16^(j-k)
    k > j 的项指数为负, 数值极小, 直接 double 求和。
    k <= j 的项要的是**小数部分**:

        frac(c · 16^(j-k) / d) = ((c · P) mod d) / d,   其中 P = 16^(j-k)

    推导: c·P/d = floor(c·P/d) + (c·P mod d)/d, 小数部分就是 (c·P mod d)/d。
    而 (c·P) mod d = (c · (P mod d)) mod d —— **只需要 P mod d**!

    所以 k<=j 的部分只需要 `pow(16, j-k, d)` 这种**小模数**模幂,
    完全不涉及 16^j 级别的大整数。

★ 为什么"笨办法"要 3 小时, 这个只要 2.4 秒 ★
    直接对 2^(4j) 取模的写法, 模数有百万位, CPython 的 Karatsuba 乘法慢两个数量级。
    **算法选错的代价是 100 倍以上** —— 这套 V2 题的核心压力就在这里。
    实测: 错误路线估算 11105 秒; 本路线 2.4 秒。差 4600 倍。

★ 两个实测踩过的坑(都留在这里当注释, 别删)★
    1. 索引差一位: int(frac(16^j·pi)·16) 给的是**第 j+1 位**(1-indexed)。
       漏掉那个 -1, 结果整体错位(got[n] = 正确的第 n-1 位)。**交叉验证抓到的。**
    2. 尾和写成 `_S(k) / 16.0**(k-j)`: 指数到 300 多时 float 直接 OverflowError。
       小规模交叉验证(n 小)根本碰不到, 只有 n=10^6 才炸 —— 典型的
       "小规模测试全过、大规模才崩"。改成逐次 `scale *= 1/16` 累加。

白名单: 只用 math / time, 全在沙箱内。
"""
from __future__ import annotations

import time

N = 10 ** 6              # 要抽的是第 10^6 位(1-indexed)

#: pi 十六进制展开的已知前缀, 用来做小规模交叉验证。
#: pi = 3.243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89
KNOWN_HEX = "243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89"


def digit_at(n: int) -> tuple[int, float]:
    """
    返回 (第 n 位十六进制数字(0-15), 累加和)。

    n 是 1-indexed: n=1 表示 16 进制展开的第 1 个小数位。
    """
    j = n - 1

    # ---- k <= j: 只取小数部分, 全靠小模数模幂 ----
    # frac(c * 16^(j-k) / d) = ((c * (16^(j-k) mod d)) mod d) / d
    s = 0.0
    for k in range(0, j + 1):
        e = j - k
        for d, c in ((8 * k + 1, 4), (8 * k + 4, -2),
                     (8 * k + 5, -1), (8 * k + 6, -1)):
            p = pow(16, e, d)
            s += ((c * p) % d) / d

    # ---- k > j: 指数为负, 逐次除以 16 累加 ----
    # 16^-200 ≈ 1e-240, 远超 double 的 1e-16 有效位, 取 200 项足够。
    # (误差论证: |S_k| <= 8/(8k) < 1/k, 尾和 < sum 16^-(k-j)/(k) < 16^-200)
    scale = 1.0
    for k in range(j + 1, j + 201):
        scale *= 1.0 / 16.0
        s += (4.0 / (8 * k + 1) - 2.0 / (8 * k + 4)
              - 1.0 / (8 * k + 5) - 1.0 / (8 * k + 6)) * scale
    frac = s - int(s)
    return int(frac * 16), s


def cross_check() -> tuple[bool, list[str]]:
    """小规模交叉验证: 抽出来的数字必须与 pi 的已知十六进制展开一致。"""
    bad: list[str] = []
    for n in range(1, len(KNOWN_HEX) + 1):
        got, _ = digit_at(n)
        want = int(KNOWN_HEX[n - 1], 16)
        if got != want:
            bad.append(f"n={n} got={got:x} want={want:x}")
    return (not bad), bad


def verify() -> dict:
    t0 = time.time()
    ok, bad = cross_check()
    print(f"[交叉验证] 前 {len(KNOWN_HEX)} 位比对已知十六进制展开: "
          f"{'全部一致' if ok else '不一致'}")
    for b in bad[:5]:
        print("   ", b)
    t1 = time.time()
    print(f"[交叉验证] 耗时 {t1 - t0:.1f}s")

    if not ok:
        # ★ 这是**我的实现错了**, 不是命题为假 ★
        # 拿自己的 bug 去 REFUTED 别人的命题 —— 这是最不该犯的错之一。
        # (子代理的第一版就是这么错的: 抽取逻辑与已知展开不符, 却输出 REFUTED。)
        print("实现与已知展开不符, 说明算法有问题, 不能据此对命题下结论")
        print("VERDICT: INCONCLUSIVE")
        return {"verdict": "inconclusive", "detail": "抽取逻辑自检失败",
                "witness": None}

    d, s = digit_at(N)
    print(f"[目标] 第 {N} 位十六进制数字 = {d:x}")
    print(f"[累加和] frac 部分 = {s - int(s):.12f}")
    print(f"[耗时] {time.time() - t1:.1f}s")
    print(f"hex digit at position {N} = {d:x}  (computed, cross-checked)")
    print("VERDICT: PROVED")
    return {"verdict": "proved", "detail": f"第 {N} 位 = {d:x}",
            "witness": {"position": N, "digit": d}}


if __name__ == "__main__":
    print(verify())