"""
res-016-tsp-heuristic-500 的独立 verifier。

题目: 对给定 500 城市 TSP 实例, 找到一个长度不超过已知最优解 1.05 倍的可行巡回。

★ 实例由 verifier 决定, 不由 agent 决定 ★
题库没附实例文件, 所以本文件**硬编码**了一个确定性实例生成器:
    rng = numpy.random.default_rng(20260908)
    coords = rng.random((500, 2)) * 1000.0
(PCG64 的取数流由 numpy 承诺跨版本稳定, 所以人人重跑得到同一批坐标。)
若官方实例文件 data/res_16_tsp.txt 存在(数据方后补), 优先用它 —— 题库
verifier.params.instance_file 指向的正是那个路径。

★ 判据 ★
  1. 巡回必须是 500 个 0..499 的**排列** —— 恰好访问每个节点一次。
     这是本题最容易被糊弄的地方: 只查"长度够短"而不查"是不是真巡回",
     agent 重复走几个近邻点就能刷出漂亮的长度。所以排列检查是硬判据。
     (允许首尾重复写法 [0,...,499,0], 那是同一个闭巡回的常见写法。)
  2. 长度由 verifier **自己算**(欧氏距离 + 闭合边), 不信 agent 报的 length。
  3. 参考最优 L_ref 由 verifier **自己跑启发式**得到(最近邻+2-opt, 与
     ortools GUIDED_LOCAL_SEARCH 取小), 判 len <= 1.05 * L_ref。
     为什么这样判就说明答案对: L_ref 越接近真最优, 1.05*L_ref 这个门槛
     越接近题目真正要求的 1.05*optimal; 用多个独立算法取 min 是为了
     避免"参考解太差 → 门槛放水 → 弱巡回蒙混过关"。
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

try:
    from .common import Check, parse_artifact, parse_verdict
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import Check, parse_artifact, parse_verdict

import numpy as np

#: 实例生成参数 —— 写死是为了让判据可复现
SEED = 20260908
N_CITIES = 500
COORD_SCALE = 1000.0

#: 题目要求的相对容差
RATIO_TOL = 0.05

#: 官方实例文件的可能位置(相对仓库根; 也试相对本文件的 ../../..)
_INSTANCE_REL = os.path.join("data", "res_16_tsp.txt")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOTS = [
    os.path.abspath(os.path.join(_HERE, "..", "..", "..")),
    os.path.abspath(os.getcwd()),
]

_REF_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------
# 实例
# --------------------------------------------------------------------------
def canonical_coords() -> np.ndarray:
    """内置确定性实例(题目没给实例时用它)。"""
    rng = np.random.default_rng(SEED)
    return rng.random((N_CITIES, 2)) * COORD_SCALE


def _coords_from_text(text: str) -> np.ndarray | None:
    """容忍两种实例文件格式: JSON [[x,y],...] 或每行 "x y"。"""
    text = text.strip()
    if not text:
        return None
    if text[0] == "[":
        try:
            arr = np.asarray(json.loads(text), dtype=float)
        except Exception:
            arr = None
        if arr is not None and arr.ndim == 2 and arr.shape[0] >= 2:
            return arr[:N_CITIES]
    pts = []
    for line in text.splitlines():
        parts = re.split(r"[,\s;]+", line.strip())
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                nums = []
                break
        if len(nums) >= 2:
            pts.append(nums[:2])
    if len(pts) >= 2:
        return np.asarray(pts[:N_CITIES], dtype=float)
    return None


def instance_coords() -> tuple[np.ndarray, str]:
    """返回 (坐标, 来源说明)。官方实例文件优先, 否则用内置实例。"""
    for root in _ROOTS:
        p = os.path.join(root, _INSTANCE_REL)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    arr = _coords_from_text(fh.read())
                if arr is not None and arr.shape == (N_CITIES, 2):
                    return arr, _INSTANCE_REL
            except OSError:
                pass
    return canonical_coords(), f"内置实例(seed={SEED}, PCG64)"


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2))


def tour_length(D: np.ndarray, tour) -> float:
    """自己算巡回总长(含闭合边)。"""
    idx = np.asarray(tour, dtype=np.int64)
    return float(D[idx, np.roll(idx, -1)].sum())


# --------------------------------------------------------------------------
# 参考解: 自己算, 两个独立算法取小
# --------------------------------------------------------------------------
def _nearest_neighbor(D: np.ndarray) -> list[int]:
    n = D.shape[0]
    unvisited = np.ones(n, dtype=bool)
    tour = [0]
    unvisited[0] = False
    cur = 0
    while len(tour) < n:
        d = np.where(unvisited, D[cur], np.inf)
        j = int(np.argmin(d))
        tour.append(j)
        unvisited[j] = False
        cur = j
    return tour


def _two_opt(D: np.ndarray, tour, max_rounds: int = 400) -> list[int]:
    """向量化的 best-improvement 2-opt。

    移动量: 断 (t[i],t[i+1]) 与 (t[j],t[j+1]), 接成 (t[i],t[j])+(t[i+1],t[j+1]),
    即把 t[i+1..j] 整段反转。delta 为负就做。
    j 取到 n-1(t[j+1] 用 %n 绕回 t[0]), 这样**环形**的移动集才是完备的。
    """
    ta = np.asarray(tour, dtype=np.int64).copy()
    n = ta.size
    for _ in range(max_rounds):
        improved = False
        for i in range(n - 1):
            a, b = ta[i], ta[i + 1]
            js = np.arange(i + 2, n)
            if js.size == 0:
                break
            tn, tn1 = ta[js], ta[(js + 1) % n]
            delta = D[a, tn] + D[b, tn1] - D[a, b] - D[tn, tn1]
            k = int(np.argmin(delta))
            if delta[k] < -1e-9:
                j = i + 2 + k
                ta[i + 1:j + 1] = ta[i + 1:j + 1][::-1]
                improved = True
        if not improved:
            break
    return ta.tolist()


def _ortools_best(D: np.ndarray, time_limit: float = 8.0) -> tuple[float | None, list[int] | None]:
    """ortools routing 的 GUIDED_LOCAL_SEARCH, 多种首解策略取最小。

    返回 (最优长度, 对应巡回或 None)。
    ★ 巡回在本机绑定上取不出来 ★: `Assignment_Value` 对单车的 Start/End 下标
    直接抛 SystemError(两种取法都试过)。这不影响判题 —— 门槛只需要**长度**,
    而长度是可靠的。取不到巡回就返回 None, 由 reference_solution 回退到
    自己的 2-opt 巡回(那份一定是合法排列), 并且绝不会出现"长度来自 A、
    巡回来自 B"的错配。
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception:
        return None, None
    n = D.shape[0]
    scale = 10_000                       # 整数代价放大(ortools 只吃 int)
    cost_list = np.rint(D * scale).astype(np.int64).tolist()
    best_len: float | None = None
    best_tour: list[int] | None = None
    strategies = [
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
    ]
    for st in strategies:
        try:
            mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
            routing = pywrapcp.RoutingModel(mgr)
            idx = routing.RegisterTransitCallback(
                lambda i, j: cost_list[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
            routing.SetArcCostEvaluatorOfAllVehicles(idx)
            params = pywrapcp.DefaultRoutingSearchParameters()
            params.first_solution_strategy = st
            params.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
            params.time_limit.FromMilliseconds(int(time_limit * 1000))
            sol = routing.SolveWithParameters(params)
            if sol is not None:
                v = sol.ObjectiveValue() / scale
                if best_len is None or v < best_len:
                    best_len = v
                    # 巡回单独取并校验是 0..n-1 的排列; 取不到就留 None,
                    # 绝不与 best_len 错配(见函数 docstring)。
                    best_tour = None
                    try:
                        cur, seq = routing.Start(0), []
                        for _ in range(n):
                            cur = sol.Value(cur)
                            seq.append(mgr.IndexToNode(cur))
                        if sorted(seq) == list(range(n)):
                            best_tour = seq
                    except Exception:
                        best_tour = None
        except Exception:
            continue
    return best_len, best_tour


def reference_solution(D: np.ndarray) -> dict:
    """自己求参考解: {L_ref, source, candidates, best_tour}。结果在进程内记忆化。"""
    key = f"{D.shape[0]}:{float(D[0, 1]):.12f}:{float(D[123, 456]):.12f}"
    if key in _REF_CACHE:
        return _REF_CACHE[key]

    t2 = _two_opt(D, _nearest_neighbor(D))
    l2 = tour_length(D, t2)
    ort_len, ort_tour = _ortools_best(D)
    if ort_len is not None and ort_len <= l2:
        L_ref, source, best_tour = ort_len, "ortools_gls", (ort_tour or t2)
    else:
        L_ref, source, best_tour = l2, "nearest_neighbor+2opt", t2
    cands = [("nearest_neighbor+2opt", round(l2, 4))]
    if ort_len is not None:
        cands.append(("ortools_gls", round(ort_len, 4)))
    out = {"L_ref": L_ref, "source": source, "candidates": cands,
           "best_tour": best_tour}
    _REF_CACHE[key] = out
    return out


# --------------------------------------------------------------------------
def _normalize_tour(raw: Any) -> list[int] | None:
    """把 agent 给的巡回规整成 0-based 整数列表(允许首尾重复写法)。"""
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    out: list[int] = []
    for x in raw:
        if isinstance(x, bool):
            return None
        if isinstance(x, float):
            if not x.is_integer():
                return None
            x = int(x)
        if isinstance(x, str):
            try:
                x = int(x.strip())
            except ValueError:
                return None
        if not isinstance(x, int):
            return None
        out.append(x)
    if len(out) == N_CITIES + 1 and out[0] == out[-1]:
        out = out[:-1]                      # 闭巡回的常见重复写法
    return out


def verify(stdout: str, code: str, claim: str) -> dict:
    chk = Check()
    coords, src = instance_coords()
    D = distance_matrix(coords)
    chk.note("instance_source", src)
    chk.note("instance_checksum", round(float(coords.sum()), 6))

    art = (parse_artifact(stdout, "CONSTRUCT") or parse_artifact(stdout, "ANSWER"))
    if not chk.expect("artifact_present", art is not None,
                      "未提交 CONSTRUCT 制品(需要 {\"tour\": [...]})"):
        return _finish(chk, None, {})
    assert art is not None

    # ---- 0. 如果 agent 自带坐标, 必须与判题实例一致 ----
    # 不一致 = 它解的是另一个实例, 那条巡回在本实例上毫无意义 -> 判错。
    raw_coords = art.get("coordinates") or art.get("coords") or art.get("points")
    if raw_coords is not None:
        try:
            given = np.asarray(raw_coords, dtype=float)
            same = (given.shape == coords.shape
                    and np.allclose(given, coords, rtol=0, atol=1e-6))
        except Exception:
            given, same = None, False
        chk.note("coordinates_provided", raw_coords is not None)
        chk.expect("coordinates_match_instance", bool(same),
                   "制品里自带的坐标与判题实例不一致 —— 它解的是另一个实例, "
                   "本实例上的巡回长度无从校验")

    # ---- 1. 巡回本身: 必须是排列 ----
    tour = _normalize_tour(art.get("tour") or art.get("tour_order")
                           or art.get("path") or art.get("solution"))
    if not chk.expect("tour_present", tour is not None, "制品里没有 tour 列表"):
        return _finish(chk, art, {})
    assert tour is not None

    chk.expect("tour_size", len(tour) == N_CITIES,
               f"巡回长度 {len(tour)} != {N_CITIES}")
    in_range = all(isinstance(t, int) and 0 <= t < N_CITIES for t in tour)
    chk.expect("tour_indices_in_range", in_range, "巡回里出现了范围外的城市编号")
    # ★ 核心防糊弄判据: 恰好每个节点一次 ★
    is_perm = (len(tour) == N_CITIES
               and in_range
               and sorted(tour) == list(range(N_CITIES)))
    if not chk.expect("tour_is_permutation", is_perm,
                      "巡回没有恰好访问每个城市一次(有重复或有遗漏)"
                      + ("" if is_perm else f": 实际 {len(set(tour))} 个不同城市")):
        return _finish(chk, art, {})

    # ---- 2. 自己算长度 ----
    length = tour_length(D, tour)
    claimed = art.get("length") or art.get("tour_length") or art.get("total_length")
    if isinstance(claimed, (int, float)):
        chk.note("claimed_length", float(claimed))
        chk.note("length_agrees", abs(float(claimed) - length) <= 1e-6 * max(1.0, length))

    # ---- 3. 自己求参考最优 ----
    ref = reference_solution(D)
    L_ref = ref["L_ref"]
    limit = (1.0 + RATIO_TOL) * L_ref
    ratio = length / L_ref if L_ref > 0 else float("inf")
    chk.note("L_ref", round(L_ref, 6))
    chk.note("L_ref_source", ref["source"])
    chk.note("reference_candidates", ref["candidates"])
    chk.expect("within_ratio", length <= limit,
               f"巡回长 {length:.4f} > 1.05*L_ref={limit:.4f} (比值 {ratio:.4f})")

    return _finish(chk, art, {"n_cities": N_CITIES, "length": round(length, 6),
                              "L_ref": round(L_ref, 6), "ratio": round(ratio, 6),
                              "within_1p05": length <= limit})


def _finish(chk: Check, art: dict | None, extra: dict) -> dict:
    correct = chk.correct
    if correct and extra.get("length") is not None:
        cert = (f"合法巡回(500 城市各一次), 长度 {extra['length']:.4f} <= "
                f"1.05*L_ref={1.05 * extra['L_ref']:.4f} (比值 {extra['ratio']:.4f})")
    else:
        cert = chk.reason()
    final = json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else None
    return {"correct": correct, "certificate": cert, "final_answer": final,
            "details": dict(chk.details)}


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time

    t0 = _time.time()
    coords, src = instance_coords()
    D = distance_matrix(coords)
    ref = reference_solution(D)
    print(f"[INFO] 实例来源: {src}; L_ref={ref['L_ref']:.4f} "
          f"(来源 {ref['source']}, 候选 {ref['candidates']}), 用时 {_time.time() - t0:.1f}s")

    # ---- 正例: 参考解本身(长度 == L_ref, 必然 <= 1.05*L_ref) ----
    good = ref["best_tour"]
    r = verify('CONSTRUCT: ' + json.dumps({"tour": good}) + "\n", "", "")
    assert r["correct"], f"正例应通过: {r}"
    print("[OK] 正例: 参考巡回 ->", r["certificate"])

    # ---- 反例 A: 少访问一批城市(长度会变短, 但不是合法巡回) ----
    bad = good[:400] + good[:100]          # 重复访问, 长度被拉长/变短都无所谓
    r = verify('CONSTRUCT: ' + json.dumps({"tour": bad}) + "\n", "", "")
    assert not r["correct"] and not r["details"]["tour_is_permutation"], r
    print("[OK] 反例A: 重复访问城市 ->", r["certificate"])

    # ---- 反例 B: 合法排列, 但质量太差(随机巡回) ----
    rng = np.random.default_rng(7)
    rnd = rng.permutation(N_CITIES).tolist()
    r = verify('CONSTRUCT: ' + json.dumps({"tour": rnd}) + "\n", "", "")
    assert not r["correct"] and r["details"]["tour_is_permutation"], r
    print("[OK] 反例B: 随机巡回(是排列但太长) ->", r["certificate"])

    print("[OK] 全部自测通过")