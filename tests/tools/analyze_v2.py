"""V2 中级 20 题: 逐题判定「机器能不能独立判」+「现有工具够不够」。

比 V1 更严的判据 —— 用户已经明确要求:
  「结论对不对」和「结论有没有证据」最后都必须问机器。

所以这题里每道都要回答两个问题:
  A. 机器怎么独立判它? (穷举/符号/求解器/形式化)
  B. 现有 11 个 research 工具支持吗? 不支持缺什么?

★ 特别注意 D 类: 「找不到 = 成立」永远不成立 ★
open problem(哥德巴赫/孪生素数无限性)不能被"跑了一遍没反例"判成 proved。
"""
from __future__ import annotations

import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

#: 每题: 机器判定方法 / 是否需要新工具 / 缺什么
ANALYSIS = {
    "hard-001-pell-61": {
        "machine": "连续分数展开(sympy pell / 手写 CF)",
        "tool_ok": True,
        "note": "sympy 有 pell 函数; 但'最小解'要给出 Pell 方程的 CF 论证才算证明, "
                "单纯算出一个解只能说'找到了解', 不等于证明了最小。",
        "verdict_limit": "找到解 → PARTIAL(存在性成立); 证明最小 → 需 CF 论证",
    },
    "hard-002-partition-p100": {
        "machine": "DP 递推 / Euler  pentagonal 定理",
        "tool_ok": True,
        "note": "五边形数定理: p(n) = sum_k (-1)^(k-1) [p(n-k(3k-1)/2) + p(n-k(3k+1)/2)], "
                "O(n^1.5) 精确整数。完全机器可判。",
        "verdict_limit": "无限制, p(100)=79270 是精确值",
    },
    "hard-003-sum-of-three-cubes-33": {
        "machine": "★ 暴力搜索不可行, 需要真正的数论构造 ★",
        "tool_ok": False,
        "note": "x^3+y^3+z^3=33 的解涉及 8.3e21 量级的数(|x|~10^7)。"
                "暴力枚举到 10^7 需要约 10^14 次立方 —— 单机跑不完。"
                "需要: (a) 数论构造法, 或 (b) 高度优化的 C 搜索 + 分布式。",
        "verdict_limit": "★ 这题是真正的 hard: 不是工具调用问题, 是算法问题",
    },
    "hard-004-catalan-50": {
        "machine": "C(2n,n)/(n+1) 精确整数",
        "tool_ok": True,
        "note": "纯组合公式, math.comb 精确算。秒级。",
        "verdict_limit": "无限制",
    },
    "hard-005-collatz-up-to-1e6": {
        "machine": "确定性模拟, 逐个 n 走 Collatz",
        "tool_ok": True,
        "note": "n<=10^6 最长链起点 837799, 525 步。纯整数运算, 秒级。"
                "★ 但注意: 验 10^6 只是 PARTIAL, Collatz 是 open problem, "
                "不能说'猜想成立'。",
        "verdict_limit": "★ 题目措辞陷阱: 只能判'在 1..10^6 范围内成立'(PARTIAL/PROVED-on-range)",
    },
    "hard-006-mersenne-127": {
        "machine": "★ Lucas-Lehmer 检验 —— 纯确定性整数运算 ★",
        "tool_ok": True,
        "note": "s_0=4, s_{i+1} = s_i^2 - 2, 127-2=125 轮全整数 mod 2^127。"
                "这是**充分必要条件**, 跑完即严格证明。完美机器判定。",
        "verdict_limit": "无限制 —— 这是全 20 题里机器判定最干净的一道",
    },
    "hard-007-ramsey-3-4": {
        "machine": "R(3,4)=9 的证明 + 8 顶点反例",
        "tool_ok": False,
        "note": "K8 有 2^28=2.68 亿种染色, 全枚举单机跑不完(即使 C 也要几小时)。"
                "需要: (a) 强剪枝回溯, (b) 已知反例构造(8 顶点 Ramsey 图), "
                "(c) 机器辅助的组合论证。",
        "verdict_limit": "★ 需要剪枝枚举或已知构造, 不是朴素全枚举",
    },
    "hard-008-twin-primes-1e7": {
        "machine": "筛法 + 计数",
        "tool_ok": True,
        "note": "埃氏筛到 10^7, 数 (p,p+2) 对。秒级, 精确。"
                "★ 但孪生素数**猜想**是 open problem, 这题只数到 1e7, "
                "不能说'孪生素数无穷多'。",
        "verdict_limit": "PARTIAL 边界题: 有限计数可以 PROVED, 无限性不行",
    },
    "hard-009-smallest-factorial-over-1e1000": {
        "machine": "★ 对数估计 + 精确验证 ★",
        "tool_ok": True,
        "note": "Stirling: log10(n!) ≈ n log10(n/e) + n log10(2πn)/2。"
                "先估出 n≈2475 附近, 再逐个精确验。完全机器可判。",
        "verdict_limit": "无限制",
    },
    "hard-010-squarefree-1e6": {
        "machine": "Möbius 函数求和 / 筛法",
        "tool_ok": True,
        "note": "count = sum_{d<=sqrt(n)} mu(d) * floor(n/d^2), 或筛法。"
                "sympy 有 mobius。精确。",
        "verdict_limit": "无限制",
    },
    "hard-011-petersen-chromatic-3": {
        "machine": "★ 构造 + 穷举反证 ★",
        "tool_ok": True,
        "note": "(a) 3 着色: 直接给一个显式着色方案, 机器验证相邻不同色。"
                "(b) 不能 2 着色: 2^10=1024 种分配全枚举, 机器可判。"
                "两边都机器可判 —— 这是个好例子: '给出着色'+'排除更少色'。",
        "verdict_limit": "无限制, 两半都能机器验证",
    },
    "hard-012-erdos-straus-1000": {
        "machine": "★ 每 n 的三元组搜索 ★",
        "tool_ok": True,
        "note": "4/n = 1/x+1/y+1/z 有正整数解。对每个 n 枚举 x,y,z。"
                "10^3 个 n, 每个的搜索空间有界(可剪枝), 秒级。"
                "★ Erdős–Straus 猜想是 open problem, 这题只到 1000。",
        "verdict_limit": "PARTIAL 边界题",
    },
    "hard-013-bbp-pi-hex-digit": {
        "machine": "★ BBP 公式直接给第 n 位十六进制数字 ★",
        "tool_ok": True,
        "note": "BBP 允许**直接跳到**第 n 位, O(n) 时间 O(log n) 空间, "
                "不需要算前面所有位。n=10^6 完全可行(几秒)。"
                "★ 这是本题的核心技巧: 不是大数计算问题, 是'跳位'算法。",
        "verdict_limit": "无限制, 但要机器实现 BBP 不是让 LLM 现编",
    },
    "hard-014-sum-cubes-1000": {
        "machine": "精确大整数 + 符号恒等式",
        "tool_ok": True,
        "note": "sum k^3 = (sum k)^2 是恒等式, sympy 一行化简。"
                "同时给两边的精确大整数(1000^3 量级, Python 大整数原生支持)。",
        "verdict_limit": "无限制",
    },
    "hard-015-fermat-little-lean": {
        "machine": "★ Lean 4 + Mathlib —— kernel 级证明 ★",
        "tool_ok": False,
        "note": "Mathlib 里可能已有现成定理; 否则要自己写 Lean 证明。"
                "★ kernel 验证的是**证明项**本身, 不是代码逻辑 —— "
                "这是最强的一档机器判定: kernel 不可能出错(小内核信任假设)。"
                "需要: 装 Lean 4 + elan + mathlib, 然后 LSP 或 `lake build`。",
        "verdict_limit": "Lean 报 no goals 才是 PROVED; 报错或有 sorry = 未证明",
    },
    "hard-016-sqrt2-irrational-lean": {
        "machine": "★ Lean 4 + Mathlib ★",
        "tool_ok": False,
        "note": "Mathlib 有 irrational 相关引理。经典反证法证明。"
                "同 015。",
        "verdict_limit": "同 015",
    },
    "hard-017-100-queens-sat": {
        "machine": "★ Z3 / SAT 求解器 ★",
        "tool_ok": True,
        "note": "已有 z3。100 皇后用 CP 编码成 Z3 的 Bool 约束, "
                "求解器几秒内出解(且能验证无冲突)。"
                "★ 这题正好是现有 research_z3 的用武之地。",
        "verdict_limit": "找到解 → 存在性 PROVED; '求解器说无解' 才是不存在性 PROVED",
    },
    "hard-018-zeta2-pi2-6": {
        "machine": "★ 收敛估计要控制截断误差 ★",
        "tool_ok": True,
        "note": "直接求和到 N=10^7 的浮点误差约 1e-7 量级, 勉强够 1e-6。"
                "★ 但严谨做法要用 Euler-Maclaurin 加速或尾部积分估计 —— "
                "'跑完 < 1e-6' 不等于'收敛到 pi^2/6', 差一个误差分析。",
        "verdict_limit": "★ 陷阱题: 数值接近 != 证明收敛",
    },
    "hard-019-no-flt-n3-small": {
        "machine": "穷举 + 精确整数立方",
        "tool_ok": True,
        "note": "a,b,c<1000 全枚举, 10^9 组合但可剪枝(排序去重后 ~10^8), "
                "Python 太慢, 但对称性剪枝 + C 加速后几秒。"
                "★ 实际可用 sympy 的 diophantine 或直接剪枝枚举。"
                "★ 顺带: 费马大定理 n=3 已被 Euler 证明, n>=3 无解是定理。",
        "verdict_limit": "在 a,b,c<1000 范围内 PROVED; 全域需要 Fermat 大定理",
    },
    "hard-020-min-n-with-n-divisors": {
        "machine": "★ DFS 枚举质因数指数组合 ★",
        "tool_ok": True,
        "note": "n 恰好有 1000 个因子: 1000=2^3*5^3。最小 n 是把大指数配给小质数。"
                "DFS 枚举指数分配即可。完全机器可判。"
                "★ 这题考的是搜索策略(按质数升序、指数降序剪枝)。",
        "verdict_limit": "无限制",
    },
}


def main() -> None:
    with io.open(os.path.join(HERE, "baseproblems_V2.txt"), encoding="utf-8") as f:
        items = json.load(f)["items"]

    print(f"V2 中级 {len(items)} 题 —— 机器判定能力盘点\n")
    ok = need = 0
    trap = []
    for it in items:
        a = ANALYSIS.get(it["id"])
        if not a:
            print(f"!! 缺分析: {it['id']}")
            continue
        mark = "OK " if a["tool_ok"] else "!! "
        if a["tool_ok"]:
            ok += 1
        else:
            need += 1
        if "陷阱" in a.get("verdict_limit", "") or "PARTIAL" in a.get("verdict_limit", ""):
            trap.append(it["id"])
        print(f"{mark}{it['id']:34} {a['machine'][:44]}")
        if not a["tool_ok"]:
            print(f"      ★ 需要新能力: {a['note'][:96]}")
        if "★" in a.get("verdict_limit", ""):
            print(f"      ★ 判定边界: {a['verdict_limit'][:96]}")

    print(f"\n{'='*78}")
    print(f"现有工具能顶住: {ok}/{len(items)}")
    print(f"需要新能力:     {need}/{len(items)}")
    print(f"判定有陷阱的:   {len(trap)} 道 —— {', '.join(trap)}")
    print()
    print("=== 三种新能力(按必要性) ===")
    print("1. Lean 4 + Mathlib  ← hard-015, 016 明确要求, 且是最强机器判定")
    print("2. 剪枝枚举/组合搜索  ← hard-003, 007, 019 需要(不是现成库能解决的)")
    print("3. 误差分析模板      ← hard-018 这类'数值接近≠证明'的题")
    print()
    print("=== V2 里最值得注意的: 不是工具不够, 是判定边界 ===")
    print("Collatz(1e6)、孪生素数(1e7)、Erdos-Straus(1000) 全是 open problem 的")
    print("有限片段 —— 验完了只能判'在给定范围内成立', **不能**判'猜想成立'。")
    print("这三道题正好检验的是: 闸门能不能拦住'跑了一遍没反例所以 proved'。")


if __name__ == "__main__":
    main()