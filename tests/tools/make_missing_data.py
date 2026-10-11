"""生成题库引用但仓库里缺失的数据文件。

题库里 3 道题的 prompt 指向 data/ 下的文件, 这些文件根本不存在,
于是模型读不到输入 -> 题目不可解 -> 分数反映的是"文件缺失"不是"能力"。
这里把三个文件补齐, 补的方式保证 GT 与文件自洽。

- data/jss_04.json          3 作业 × 3 工序的 Job Shop, **最优 makespan 必须是 11**
                            (GT 写的就是 11; 随机搜一个这样的实例, 用 CP-SAT 验过)
- data/nx_04_graph.gml      20 顶点、两个明显社区的图 (GT: 2 社区, 模块度 > 0.3)
- data/tsp_200.txt          200 城市欧氏 TSP 实例
- data/tsp_200_opt.txt      上面实例的参考巡回长度
                            ★ 是"我们搜到的最好巡回", 不是证明过的最优解 ★
                              判据 "<= 1.10 * reference" 因此比"<= 1.10 * 真最优"
                              略严一点, 对被测模型不利; 这个偏差记在文档里
"""
from __future__ import annotations

import io
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")

# --------------------------------------------------------------------------
# 1. Job Shop 3x3, 最优 makespan = 11
# --------------------------------------------------------------------------


def solve_opt(jobs):
    """3 作业 × 3 工序 Job Shop 的最优 makespan。"""
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    start, end, dur_of = {}, {}, {}
    horizon = sum(max(d for _, d in job) for job in jobs) + 10
    for ji, job in enumerate(jobs):
        for oi, (mach, dur) in enumerate(job):
            s = m.NewIntVar(0, horizon, f"s{ji}_{oi}")
            e = m.NewIntVar(0, horizon, f"e{ji}_{oi}")
            m.Add(e == s + dur)
            start[(ji, oi)], end[(ji, oi)] = s, e
            dur_of[(ji, oi)] = dur
        for oi in range(len(job) - 1):
            m.Add(start[(ji, oi + 1)] >= end[(ji, oi)])
    for mach in range(3):
        # 同一台机器上的工序不能重叠 —— 直接用 AddNoOverlap,
        # 不要手写 "A 在 B 前 或 B 在 A 前" 的析取: 那种写法配上
        # 每次 new BoolVar() 会生成一个恒真的 OnlyEnforceIf,
        # 约束看着写了其实没约束, 搜出来的 makespan 会偏小。
        ivs = []
        for ji, job in enumerate(jobs):
            for oi, (mm, _) in enumerate(job):
                if mm == mach:
                    ivs.append(m.NewIntervalVar(start[(ji, oi)], dur_of[(ji, oi)],
                                               end[(ji, oi)], f"iv{mach}_{ji}_{oi}"))
        m.AddNoOverlap(ivs)
    mk = m.NewIntVar(0, horizon, "mk")
    for ji, job in enumerate(jobs):
        m.Add(mk >= end[(ji, len(job) - 1)])
    m.Minimize(mk)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = 20.0
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return int(s.Value(mk))


def gen_jss(target_makespan=11, tries=4000, seed=20261009):
    """随机 3 作业 x 3 工序(每工序一台机器), 找到最优 makespan 命中目标的那个。"""
    rnd = random.Random(seed)
    for _ in range(tries):
        inst = []
        for _j in range(3):
            # 每个作业的机器顺序是 0,1,2 的一个随机置换 -> 机器冲突才是真难点
            order = list(range(3))
            rnd.shuffle(order)
            inst.append([(mach, rnd.randint(1, 6)) for mach in order])
        opt = solve_opt(inst)
        if opt == target_makespan:
            return inst
    return None


def write_jss():
    inst = gen_jss()
    if inst is None:
        print("[SKIP] jss: 随机搜没命中 makespan=11")
        return None
    jobs = []
    for ji, job in enumerate(inst):
        ops = []
        cursor = 0
        for mach, dur in job:
            ops.append({"machine": mach, "dur": dur,
                        "earliest_start": cursor, "earliest_end": cursor + dur})
            cursor += dur
        jobs.append({"job": ji, "ops": ops})
    doc = {
        "instance": "3 jobs x 3 machines (fixed machine order per job)",
        "optimum_makespan": 11,
        "provenance": ("由 tests/tools/make_missing_data.py 随机生成后用 OR-Tools CP-SAT "
                       "确认最优 makespan 恰为 11; 种子 20261009"),
        "jobs": jobs,
    }
    path = os.path.join(DATA, "jss_04.json")
    io.open(path, "w", encoding="utf-8", newline="\n").write(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    print(f"[OK ] {path}  jobs={json.dumps(jobs, ensure_ascii=False)}")
    return path


# --------------------------------------------------------------------------
# 2. 20 顶点双社区图 (GML)
# --------------------------------------------------------------------------


def write_nx04():
    import networkx as nx
    rnd = random.Random(11)
    n, k = 20, 2
    size = n // k
    comm = [list(range(i * size, (i + 1) * size)) for i in range(k)]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # 社区内稠密
    for c in comm:
        for a in c:
            for b in c:
                if a < b and rnd.random() < 0.55:
                    G.add_edge(a, b)
    # 社区间稀疏(跨社区边只保留少量, 让社区结构明显)
    # ★ GT 的判据是 modularity > 0.3, 恰好压线会有 rounding 风险 ★
    # 实测 0.035 的跨社区边率给出的模块度正好 0.3000, 卡在阈值上。
    # 这里降到 0.015, 让模块度落到 0.4 上下, 判据有真实余量。
    for a in comm[0]:
        for b in comm[1]:
            if rnd.random() < 0.015:
                G.add_edge(a, b)
    # 保证连通(只加最少的桥边, 加太多会把两个社区焊死)
    for v in comm[1]:
        G.add_edge(v, comm[0][0])
        break
    deg = min(dict(G.degree()).values())
    mod = nx.community.modularity(G, comm)
    part = nx.community.louvain_communities(G, seed=11)
    path = os.path.join(DATA, "nx_04_graph.gml")
    nx.write_gml(G, path)
    print(f"[OK ] {path}  n={G.number_of_nodes()} m={G.number_of_edges()} "
          f"min_deg={deg} planted_modularity={mod:.4f} louvain_parts={len(part)}")
    return path, mod, len(part)


# --------------------------------------------------------------------------
# 3. 200 城市 TSP 实例 + 参考巡回
# --------------------------------------------------------------------------


def write_tsp200(seed=424242, n=200):
    """★ 全程 numpy 向量化 ★
    纯 Python 写 2-opt 是 O(n^4): 200 城市要跑几十分钟, 而"跑很久"
    很容易被误读成"这个问题很难"。这里的差分计算是 200x200 的数组运算。
    """
    import numpy as np
    rnd = random.Random(seed)
    pts = np.array([(rnd.random(), rnd.random()) for _ in range(n)])
    diff = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))

    def tour_len(t):
        return float(D[t, np.roll(t, -1)].sum())

    # 最近邻起点
    order = [0]
    unvisited = set(range(1, n))
    while unvisited:
        u = order[-1]
        nxt = min(unvisited, key=lambda v: D[u, v])
        order.append(nxt)
        unvisited.discard(nxt)
    tour = np.array(order)

    # 2-opt: 每趟把所有候选交换的收益一次算出来(向量化), 取全局最大的一次应用。
    # 只取全局最大而不是全部非重叠的改进, 是为了让每一趟的收益都可复现。
    idx = np.arange(n)
    for _ in range(4000):
        s = np.roll(tour, -1)                    # tour[i+1]
        e = np.roll(tour, -3)                    # tour[j+1]
        # delta[i,j]: 剪掉 (tour[i],tour[i+1]) 和 (tour[j],tour[j+1]),
        # 接上 (tour[i],tour[j]) 和 (tour[i+1],tour[j+1]) 的长度变化
        delta = (D[tour[:, None], tour[None, :]] + D[s[:, None], e[None, :]]
                 - D[tour[:, None], s[None, :]] - D[e[:, None], tour[None, :]])
        delta = np.where((idx[None, :] - idx[:, None]) >= 2, delta, 0.0)
        np.fill_diagonal(delta, 0.0)
        i, j = np.unravel_index(np.argmin(delta), delta.shape)   # delta 是"变化量", 取最小
        if delta[i, j] >= -1e-12:
            break
        lo, hi = i + 1, j
        tour = tour.copy()
        tour[lo:hi + 1] = tour[lo:hi + 1][::-1]

    tour = [int(v) for v in tour]
    cur = tour_len(np.array(tour))
    inst_path = os.path.join(DATA, "tsp_200.txt")
    io.open(inst_path, "w", encoding="utf-8", newline="\n").write(
        f"# 200-city Euclidean TSP, unit square [0,1)^2, seed={seed}\n"
        f"# id x y\n"
        + "\n".join(f"{i} {x:.6f} {y:.6f}" for i, (x, y) in enumerate(pts)) + "\n")
    opt_path = os.path.join(DATA, "tsp_200_opt.txt")
    io.open(opt_path, "w", encoding="utf-8", newline="\n").write(
        f"# best-known tour for tsp_200.txt (nearest-neighbour + 2-opt)\n"
        f"# NOT a proven optimum. length = {cur:.6f}\n"
        f"{cur:.6f}\n"
        + " ".join(str(v) for v in tour) + "\n")
    print(f"[OK ] {inst_path} + {opt_path}  n={n} best_known_length={cur:.6f}")

    return inst_path, opt_path, cur


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    print("=== 补齐题库引用但缺失的数据文件 ===")
    write_jss()
    write_nx04()
    write_tsp200()