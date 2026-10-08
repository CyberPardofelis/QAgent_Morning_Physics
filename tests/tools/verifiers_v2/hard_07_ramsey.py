"""V2 基准 hard-007 —— R(3,4)=9 的双向验证器。

题目: 「验证 R(3,4)=9: 证明任意 9 个顶点的红蓝完全图必有红色三角形或蓝色 K4,
        并给出 8 个顶点无反例的着色。」

======================================================================
★ 这是一道**双向题**, 而 2026-10-08 实跑里 LLM 正是在这里翻的车 ★
======================================================================
R(3,4)=9 是**双向**的, 拆成两半:

  Part(1)  上界 R(3,4) <= 9 : 9 个顶点的任意红蓝染色都必然出
                             红三角形 或 蓝 K4  -> 证 UNSAT 即可
  Part(2)  下界 R(3,4) >  8 : 存在一个 8 顶点的染色, 既无红三角形
                             又无蓝 K4          -> 给一个 witness

★ 翻车现场 ★
LLM 的 Part(1) 用 z3 证对了(unsat, 完全正确)。Part(2) 它自己构造了一个
8 顶点染色, 而它自己的检查器在该染色里抓到了一个蓝 K4 (0,3,5,6), 于是它
把整道题判成了 `VERDICT: REFUTED`。

**这是假 REFUTED。** 抓到蓝 K4 证明的是「**我构造的那个染色**是错的」,
不是「**这个定理**是错的」。R(3,4)=9 是真命题, 8 顶点的构造是存在的
(本文件用 z3 在 SAT 时现搜一个, 毫秒级)。
判据层面的教训写死在本文件的 `verify()` 里, 见下面那段合取判据注释。

======================================================================
★ 写给 LLM 的写法要点 ★
======================================================================
1. **双向题的 verdict 必须是两部分的合取, 而且 REFUTED 永远不是合法输出。**
   本文件把三态拆清楚:
     两半都成立              -> PROVED
     只成立一半              -> PARTIAL(明确说清是哪半、另一半缺什么)
     两半都没拿到            -> INCONCLUSIVE
   「我的 witness 没验过」永远落在 PARTIAL/INCONCLUSIVE, 不许上升成 REFUTED。
   要判 REFUTED, 你得给出**定理本身为假**的证据(比如 9 个顶点的一个真反例),
   而不是「我给的 8 顶点染色有蓝 K4」。

2. **witness 验不过 = 你自己写错了, 正确反应是换/搜一个 witness, 不是驳命题。**
   witness 是证明的载体, 载体坏了去修载体。Part(2) 挂了不影响 Part(1)。

3. **sat / unsat / unknown 三态要分开。** 设了 timeout 之后 check() 可能
   返回 unknown; unknown 既不是 sat 也不是 unsat, 只能判 INCONCLUSIVE。
   把 unknown 当 unsat, 就是把「没算完」谎报成「证明了不存在」。

4. **别碰 model.eval(...)**: 沙箱静态门在 AST 层拦属性名 .eval,
   正确写法是 `model[var]` + `z3.is_true(...)`。同名黑名单还有
   exec/compile/__import__/open/input/vars/globals/locals;
   import 白名单里也没有 `__future__`, 写 `from __future__ import annotations`
   会被判「模块 `__future__` 不在沙箱白名单内」。

5. **双向题必须给 n-1 跑一次 sat 当哨兵。**
   编码写歪时(比如把「无蓝 K4」写成「每 4 元组红边 >= 5」)上界那边照样
   unsat, 看起来「证完了」, 其实证的是一个更强、更假的问题。只有 n=8
   真能 sat, 才知道编码的强度和题目一致。本文件把这一步写进了主流程。
"""

from itertools import combinations

import networkx as nx
import z3

# ------------------------------------------------------------------
# Part(2) 的 8 顶点构造: Wagner 图 M8 = 8-环 + 4 条对面弦
# ------------------------------------------------------------------
# 红边取 Wagner(Moebius-Kantor 之外最标准的 3-正则 8 环带弦图):
#   环边 (0,1) (1,2) (2,3) (3,4) (4,5) (5,6) (6,7) (7,0)
#   弦边 (0,4) (1,5) (2,6) (3,7)
# 为什么它正好合格:
#   * 红图无三角形 -> 没有红三角形。
#     环上任意三个顶点不可能两两相邻(8-环上 3 点必有一个距离 >= 2);
#     弦 i-i+4 把顶点分成 {0,4}{1,5}{2,6}{3,7} 四对, 每对之间要成环还得经
#     中间点, 单条弦永远凑不出三角形。
#   * 红图独立数 alpha = 3 -> 红图没有 4 个两两不相邻的点, 即蓝图没有 K4。
# 注意: 本文件**不靠这段注释下结论**, 下面用程序把两件事逐条数一遍。
RED_8_EDGES: list[tuple[int, int]] = (
    [(i, (i + 1) % 8) for i in range(8)]
    + [(i, i + 4) for i in range(4)]
)
RED_8_EDGES = [(a, b) if a < b else (b, a) for a, b in RED_8_EDGES]

SOLVER_TIMEOUT_MS = 30_000


# ------------------------------------------------------------------
# 与求解器无关的纯检查器(验证 witness 用)
# ------------------------------------------------------------------
def check_coloring(red: set, n: int) -> dict:
    """给定红边集合(无序对的小端在前), 逐个子集数违规。

    返回 {"red_triangles": [...], "blue_k4": [...]}。
    完备性: C(n,3) 个三元组全查红三角形, C(n,4) 个四元组全查蓝 K4,
    两类子集数都是小常数, 遍历完就是穷尽的。
    """
    red_tri = []
    for tri in combinations(range(n), 3):
        if all((min(a, b), max(a, b)) in red for a, b in combinations(tri, 2)):
            red_tri.append(tri)

    blue_k4 = []
    for quad in combinations(range(n), 4):
        # 全蓝 <=> 这 6 条边一条都不在红边里
        if all((min(a, b), max(a, b)) not in red for a, b in combinations(quad, 2)):
            blue_k4.append(quad)

    return {"red_triangles": red_tri, "blue_k4": blue_k4}


def edges_from_bools(get_bool, n: int) -> set:
    """把一个 0/1 布尔矩阵读成红边集合。只用下标取值, 不碰 .eval。"""
    return {(i, j) for i in range(n) for j in range(i + 1, n) if get_bool(i, j)}


# ------------------------------------------------------------------
# z3 建模: 「无红三角 且 无蓝 K4」染色的存在性
# ------------------------------------------------------------------
def build_coloring_solver(n: int, timeout_ms: int = SOLVER_TIMEOUT_MS):
    """n 个顶点的红蓝完全图, red[i][j] 为 True <=> 边 (i,j) 是红色。

    约束两条, 正好是 R(3,4) 反例的完整定义:
      (A) 无红三角形: 每个三元组至少一条蓝边 —— 3-CNF 子句
      (B) 无蓝 K4   : 每个四元组至少一条红边 —— 6-析取子句
    返回 (solver, red)。check() == unsat 就是「n 个顶点不可能满足」,
    也就是 R(3,4) <= n; sat 就是找到一个反例染色, 即 R(3,4) > n。
    """
    red = {(i, j): z3.Bool(f"r_{i}_{j}")
           for i in range(n) for j in range(i + 1, n)}
    s = z3.Solver()
    s.set("timeout", timeout_ms)

    # (A) 无红三角形: 三条边不能同时为红
    for tri in combinations(range(n), 3):
        (a, b), (a, c), (b, c) = combinations(tri, 2)
        s.add(z3.Or(z3.Not(red[a, b]), z3.Not(red[a, c]), z3.Not(red[b, c])))

    # (B) 无蓝 K4: 蓝 K4 = 某个 4 元组的 6 条边全蓝。
    #     所以「不存在蓝 K4」= 每个 4 元组的 6 条边里**至少有 1 条红边**,
    #     编码成一个 6 析取的子句: Or(red_e1, ..., red_e6)。
    #
    #     ★ 这里极易写成「每个 4 元组红边 >= 5 条」★
    #     那是「每个 4 元组蓝边 <= 1 条」, 比「无蓝 K4」强得多(蓝图必须极稀疏),
    #     强到 n=8 都能证出 unsat —— 也就是它会伪造出一个「8 个顶点没有反例」的
    #     假结论, 顺带把真正的上界证明也变成废话(unsat 只针对那个假问题)。
    #     写完编码一定要用 n=8 跑一次看是不是 sat: 双向题里 n-1 必须 sat,
    #     这是「约束强度写歪了」最便宜的哨兵。
    for quad in combinations(range(n), 4):
        s.add(z3.Or(*[red[min(u, v), max(u, v)]
                     for u, v in combinations(quad, 2)]))

    return s, red


def solve_and_extract(n: int, timeout_ms: int = SOLVER_TIMEOUT_MS):
    """跑一次建模并把模型读成红边集合。返回 ("sat", 红边集) /
    ("unsat", None) / ("unknown", reason)。"""
    s, red = build_coloring_solver(n, timeout_ms)
    res = s.check()
    if res == z3.sat:
        m = s.model()
        # ★ model[var] 下标取值 —— 不能写 model.eval(var), 静态门拦 .eval
        edges = {(i, j) for (i, j), var in red.items() if z3.is_true(m[var])}
        return "sat", edges
    if res == z3.unsat:
        return "unsat", None
    return "unknown", s.reason_unknown()


# ------------------------------------------------------------------
# verify
# ------------------------------------------------------------------
def verify() -> dict:
    """返回 {"verdict", "detail", "witness"}。

    ★★ 合取判据(本题的核心, 别写错)★★
    本题是双向命题 R(3,4)=9, 拆成 part1 (上界 <=9) 和 part2 (下界 >8)。
    两半各自独立, 各自有各自的失败模式, 所以:

        part1_ok = z3 对 n=9 返回 unsat          -> 证明了 R(3,4) <= 9
        part2_ok = 8 顶点 witness 通过纯检查器   -> 证明了 R(3,4) >  8

        两半都成立                       -> PROVED
        只有一半成立                     -> PARTIAL(逐半说清缺什么)
        两半都没拿到                     -> INCONCLUSIVE

    ★ 本文件永远不返回 REFUTED ★
    理由: 判 REFUTED 要求「R(3,4)=9 为假」的证据, 也就是给出 9 个顶点的
    一个真反例染色。而任何「8 顶点 witness 自己没验过」的失败 —— 2026-10-08
    那次抓出蓝 K4 (0,3,5,6) 就是典型 —— 说明的是**这个 witness 写错了**,
    是载体有 bug, 不是命题被推翻。这类失败必须降级成 PARTIAL 并去修 witness,
    绝不允许上升成 REFUTED。把「我的证明材料有 bug」误报成「定理为假」,
    是这类双向题最主要的翻车方式。
    """
    checks: dict[str, bool] = {}
    witness: dict = {}

    # ---------------- Part(1): 上界 R(3,4) <= 9 ----------------
    print("[Part1] 建模 n=9 (9 顶点全红蓝染色中不存在反例), 目标 = UNSAT ...")
    p1_status, _ = solve_and_extract(9)
    print(f"[Part1] check() = {p1_status}")
    if p1_status == "unknown":
        # ★ 坑 3: unknown 不是 unsat。超时被杀只能算「没算完」。
        checks["part1_unsat_9"] = False
        witness["part1_note"] = "z3 unknown (超时), 上界未判定"
    else:
        checks["part1_unsat_9"] = (p1_status == "unsat")
    witness["part1_result"] = p1_status

    # ---------------- Part(2a): 显式构造的检查 ----------------
    print(f"[Part2] 8 顶点构造 = Wagner 图 M8, 红边 = {RED_8_EDGES}")
    rep = check_coloring(set(RED_8_EDGES), 8)
    print(f"[Part2] 红三角形 = {rep['red_triangles'] if rep['red_triangles'] else '无'}")
    print(f"[Part2] 蓝 K4    = {rep['blue_k4'] if rep['blue_k4'] else '无'}")
    checks["explicit_no_red_triangle"] = (rep["red_triangles"] == [])
    checks["explicit_no_blue_k4"] = (rep["blue_k4"] == [])
    witness["explicit_8_red_edges"] = sorted(RED_8_EDGES)

    # 交叉核对: networkx 对同一构造独立复算一遍(子集计数 + 图论指标)
    g_red = nx.Graph()
    g_red.add_nodes_from(range(8))
    g_red.add_edges_from(RED_8_EDGES)
    g_blue = nx.complement(g_red)
    tri_free = nx.triangles(g_red)                     # 每点参与的三角形数
    has_red_tri = any(tri_free.values())
    # 蓝图的最大团大小 = omega(蓝图); 蓝图无 K4 <=> omega(蓝图) <= 3
    omega_blue = max(len(c) for c in nx.find_cliques(g_blue))
    has_blue_k4 = omega_blue >= 4
    # alpha(红图) = omega(蓝图) 是同一个数的两种说法, 这里显式记下来对照
    alpha_red = max(len(c) for c in nx.find_cliques(nx.complement(g_red)))
    print(f"[交叉] 红图 3-正则 = {all(g_red.degree(v) == 3 for v in g_red.nodes())}, "
          f"nx 查到三角形 = {has_red_tri}, 蓝图 omega = {omega_blue}, 红图 alpha = {alpha_red}")
    checks["nx_no_red_triangle"] = (not has_red_tri)
    checks["nx_omega_blue_le_3"] = (omega_blue <= 3)
    checks["nx_no_blue_k4"] = (not has_blue_k4)
    checks["nx_alpha_equals_omega_blue"] = (alpha_red == omega_blue)
    # 两个独立方法必须一致; 不一致说明其中一个有 bug, 此时不许报 PROVED
    checks["two_checkers_agree"] = (
        has_red_tri == (rep["red_triangles"] != [])
        and has_blue_k4 == (rep["blue_k4"] != [])
    )
    witness["explicit_alpha_red"] = alpha_red

    # ---------------- Part(2b): z3 现搜一个 8 顶点反例 ----------------
    # 这一步的意义: 万一手抄的构造写错了, z3 能立刻再搜一个正确的;
    # 也说明「8 顶点反例存在」本身是被求解器独立确认过的。
    print("[Part2] 再用 z3 在 n=8 上搜一个反例染色 (目标 = SAT) ...")
    p2_status, p2_edges = solve_and_extract(8)
    print(f"[Part2] check() = {p2_status}")
    if p2_status == "sat":
        p2_rep = check_coloring(p2_edges, 8)
        checks["z3_sat_8"] = (p2_rep["red_triangles"] == [] and p2_rep["blue_k4"] == [])
        witness["z3_8_red_edges"] = sorted(p2_edges)
        print(f"[Part2] z3 解的红边 = {sorted(p2_edges)}")
        print(f"[Part2] 对该解复检: 红三角形={p2_rep['red_triangles'] or '无'}, "
              f"蓝K4={p2_rep['blue_k4'] or '无'}")
    else:
        checks["z3_sat_8"] = False
        witness["z3_8_note"] = f"n=8 未取得 SAT ({p2_status})"
        print(f"[Part2] n=8 未取得 SAT ({p2_status}) —— 与已知 R(3,4)>8 矛盾, 需排查")

    # ---------------- 两半的合取 ----------------
    # ★ 承上: part2 成立与否(无论显式构造还是 z3 搜到的)合起来构成下界
    part2_ok = (checks["explicit_no_red_triangle"]
                and checks["explicit_no_blue_k4"]
                and checks["two_checkers_agree"])
    part1_ok = checks["part1_unsat_9"]
    checks["checker_self_consistent"] = checks["two_checkers_agree"]

    if part1_ok and part2_ok:
        verdict = "PROVED"
        detail = "Part(1) 9 顶点 UNSAT => R(3,4) <= 9; Part(2) 8 顶点 witness 通过 => R(3,4) > 8; 合并 R(3,4) = 9"
    elif part1_ok:
        verdict = "PARTIAL"
        detail = ("Part(1) 成立(已证 R(3,4) <= 9), Part(2) 的 8 顶点 witness 未通过校验 —— "
                  "★ 这是「我的构造写错了」, 不是「定理为假」, 不得判 REFUTED; "
                  "补一个正确的 8 顶点染色即可闭合")
    elif part2_ok:
        verdict = "PARTIAL"
        detail = ("Part(2) 成立(已证 R(3,4) > 8), Part(1) 的 9 顶点 UNSAT 未取得 "
                  "(可能是 unknown/超时) —— 不得判 REFUTED, 放宽预算重跑上界")
    else:
        verdict = "INCONCLUSIVE"
        detail = "上下界两半都未拿到, 证据不足"

    all_ok = all(checks.values())
    print("[逐项] " + "; ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in checks.items()))
    print(f"[判据] part1={part1_ok} AND part2={part2_ok} -> {verdict}")
    print(f"[汇总] {detail}")

    # 全部逐项检查都通过时, PROVED; 否则按上面的三态降级。
    if verdict == "PROVED" and not all_ok:
        verdict, detail = "PARTIAL", "存在逐项 FAIL, 从 PROVED 降级: " + detail

    return {
        "verdict": verdict,
        "detail": detail,
        "witness": {
            "part1_status_9": witness.get("part1_result"),
            "part2_explicit_red_edges": sorted(RED_8_EDGES),
            "part2_explicit_alpha_red": witness.get("explicit_alpha_red"),
            "part2_z3_red_edges_8": witness.get("z3_8_red_edges"),
            "checker": "check_coloring (穷举 C(n,3) / C(n,4) 子集, 与求解器无关)",
        },
    }


if __name__ == "__main__":
    result = verify()
    print(f"RESULT: {result['verdict']} | {result['detail']}")
    print(f"VERDICT: {result['verdict']}")