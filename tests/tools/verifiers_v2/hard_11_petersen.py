"""V2 基准 hard-011-petersen-chromatic-3 —— 人工手写正确验证器。

题目: 「证明 Petersen 图的色数是 3：给出 3 着色，并证明不能 2 着色。」

★ 这道题为什么该被 PROVED 而不是 INCONCLUSIVE ★
选型对了, 它比符号推理简单得多, 全文只有两件事, 两件都是构造 + 穷举:

  (上界) chi(G) <= 3 : 给出一个显式的 3 着色, 逐边检查两端异色。
         这是一个 witness, 机器逐条边验一遍就完事, 不需要任何推理。
  (下界) chi(G) > 2 : 2 着色 = 给 10 个顶点各选一个颜色值 {0,1},
         也就是一个长度 10 的 0/1 向量, 总共 2^10 = 1024 种, 一次全枚举。
         全部枚举完没有一个合法 => 不是 2 着色 => chi >= 3。

合起来 chi = 3, 闭合成 PROVED。

★ 反面教材(2026-10-08 实跑翻车)★
LLM 生成的 58 行代码判了 INCONCLUSIVE。问题不在能力, 在选型:
它去对着染色数做符号推理 / 试图像素搜索, 而没注意到
「10 个顶点的 2-着色」搜索空间只有 1024, 全枚举是毫秒级的。
能穷举的命题就不要推理 —— 穷举是完备的, 推理是猜的。
"""

from __future__ import annotations

from itertools import product

import networkx as nx

# --------------------------------------------------------------------------
# 1. 显式构造 Petersen 图
# --------------------------------------------------------------------------
# 标准画法: 外圈五边形 + 内圈五角星 + 5 根辐条(spoke)。
#   v0..v4 : 外五边形, 边 v_i - v_{i+1 mod 5}          -> 5 条
#   u0..u4 : 内五角星, 边 u_i - u_{i+2 mod 5}          -> 5 条(星形是 5-环, i+2 走遍全部邻接)
#   辐条   : 边 v_i - u_i                              -> 5 条
# 合计 10 顶点 / 15 边 / 每点度数 3。

OUTER = ["v0", "v1", "v2", "v3", "v4"]
INNER = ["u0", "u1", "u2", "u3", "u4"]

EDGES: list[tuple[str, str]] = []
EDGES += [(OUTER[i], OUTER[(i + 1) % 5]) for i in range(5)]      # 外圈
EDGES += [(INNER[i], INNER[(i + 2) % 5]) for i in range(5)]      # 内星
EDGES += [(OUTER[i], INNER[i]) for i in range(5)]                 # 辐条

EDGES = [(a, b) if a <= b else (b, a) for a, b in EDGES]
EDGES = sorted(set(EDGES))


def build_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(OUTER + INNER)
    g.add_edges_from(EDGES)
    return g


# --------------------------------------------------------------------------
# 2. 上界: 给一个显式 3 着色
# --------------------------------------------------------------------------
# 手推出来的(不是搜出来的, 显式列在这里方便人眼核对):
#   外圈 v0..v4 = 1,2,1,2,3   —— 5-环不能 2 染, 必须出现第三色
#   内星 u0..u4 = 2,3,3,1,1
COLORING_3 = {
    "v0": 1, "v1": 2, "v2": 1, "v3": 2, "v4": 3,
    "u0": 2, "u1": 3, "u2": 3, "u3": 1, "u4": 1,
}


def check_coloring(coloring: dict[str, int]) -> list[str]:
    """返回违反着色规则的边(合法则为空)。每条边两端必须异色。"""
    return [f"{a}-{b}" for a, b in EDGES if coloring[a] == coloring[b]]


# --------------------------------------------------------------------------
# 3. 下界: 全枚举 2 着色
# --------------------------------------------------------------------------
# ★ 完备性就靠这一句 ★
# 「一个 2 着色」在数学上就是一个函数 c: V -> {0,1}。
# V 固定就是这 10 个顶点, 颜色集固定就是 {0,1},
# 所以所有 2 着色与长度 10 的 0/1 向量一一对应, 恰 2^10 = 1024 个。
# product((0,1), repeat=10) 把这 1024 个向量不重不漏地全部枚举了一遍 ——
# 因此枚举结束还没找到合法解, 是穷尽的否定, 不是「跑了 1024 次没撞上」。
def enumerate_2colorings(g: nx.Graph) -> tuple[int, int]:
    nodes = sorted(g.nodes())
    total = 0
    legal = 0
    for bits in product((0, 1), repeat=len(nodes)):
        total += 1
        color = dict(zip(nodes, bits))
        if all(color[u] != color[v] for u, v in g.edges()):
            legal += 1
    return total, legal


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
def verify() -> dict:
    g = build_graph()
    lines: list[str] = []
    checks: dict[str, bool] = {}

    # --- 构造自检 ---
    n_v, n_e = g.number_of_nodes(), g.number_of_edges()
    deg_ok = all(g.degree(x) == 3 for x in g.nodes())
    print(f"[构造] 顶点 {n_v} / 边 {n_e} / 3-正则={deg_ok}")
    print(f"[构造] 边集 = {EDGES}")
    checks["graph_shape"] = (n_v == 10 and n_e == 15 and deg_ok)

    # --- 与 networkx 现成构造交叉核对 (nx.petersen_graph 是否同构) ---
    iso = nx.is_isomorphic(g, nx.petersen_graph())
    print(f"[交叉] 与 nx.petersen_graph 同构 = {iso}")
    checks["isomorphic_to_nx"] = iso

    # --- 上界: 3 着色 witness ---
    print(f"[3着色] 方案 = {COLORING_3}")
    bad3 = check_coloring(COLORING_3)
    print(f"[3着色] 违反的边 = {bad3 if bad3 else '无 —— 合法'}")
    checks["three_coloring_legal"] = (bad3 == [])
    checks["uses_3_colors"] = (len(set(COLORING_3.values())) == 3)

    # --- 下界: 穷举 2 着色 ---
    total, legal = enumerate_2colorings(g)
    print(f"[2着色] 共枚举 {total} 种 2-赋值 (应为 2^10 = {2 ** 10})")
    print(f"[2着色] 其中合法 2-着色 {legal} 种 (应为 0)")
    checks["exhaustive_count"] = (total == 2 ** 10)
    checks["no_2coloring"] = (legal == 0)

    # --- 交叉核对: 非二分图 ---
    bip = nx.is_bipartite(g)
    print(f"[交叉] nx.is_bipartite = {bip} (False <=> 不是 2 着色)")
    checks["not_bipartite"] = (not bip)
    # 穷举与 is_bipartite 必须一致, 不一致说明其中一个有问题
    checks["two_methods_agree"] = ((legal == 0) == (not bip))

    ok = all(checks.values())
    detail = "; ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in checks.items())
    print(f"[汇总] {detail}")

    verdict = "PROVED" if ok else "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "detail": detail,
        "witness": {"edges": EDGES, "coloring_3": COLORING_3,
                   "two_colorings_total": total, "two_colorings_legal": legal},
    }


if __name__ == "__main__":
    result = verify()
    print(f"RESULT: {result['verdict']} | {result['detail']}")
    print(f"VERDICT: {result['verdict']}")