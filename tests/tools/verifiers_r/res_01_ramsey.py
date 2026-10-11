# -*- coding: utf-8 -*-
"""res-001-ramsey-34-lower-bound —— R(3,4) > 8 的独立验证器。

题目: 「证明 R(3,4) > 8: 构造一个 8 个顶点的红蓝完全图, 既无红色三角形,
       也无蓝色 K4, 并给出机器可验证的证书。」

★ 为什么这道题「给个见证就能判对」★
R(3,4) > 8 是一个**存在性**命题, 而存在性命题的证明载体就是见证本身:
一个 8 顶点的红蓝完全图染色 C。判定它是否成立, 完全是有限检查 ——
  * 8 个顶点的红边子图里没有三角形      -> 枚举 C(8,3) = 56 个三元组
  * 8 个顶点的蓝边子图里没有 K4         -> 枚举 C(8,4) = 70 个四元组
两个枚举各自把全部可能的子集**不重不漏**地走了一遍, 所以「枚举完没抓到」
是穷尽的否定, 不是「跑了几十次没撞上」。这两句合起来就是完整证明。

★ 因此 verifier 的判据是穷举, 不是复述 agent 的自检输出 ★
本文件**不执行** agent 的 `code` 参数。执行等于把判据交还给它自己:
agent 若把 `no_red_K3` 写错成「只看连通块」, 它自己就会报 PASS。
这里只认 `CONSTRUCT:` 制品, 然后由本文件**自己**重新枚举。
"""

from __future__ import annotations

import json
import pathlib
import sys
from itertools import combinations

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import Check, parse_artifact, parse_verdict  # noqa: E402

#: 题目固定的顶点数。R(3,4) > 8 里的那个 8 就是它。
N = 8

#: 颜色归一化。之所以允许 0/1 与 "R"/"B" 多种写法, 是因为**内容**会被完整
#: 重新验证, 放宽写法不会放宽判据 —— 而卡死写法会让一个数学上正确的 agent
#: 因为换了记号被判 0 分。归一化失败(第三种颜色)的元素则一律拒绝。
_COLOR_MAP: dict[object, str] = {
    "red": "red", "r": "red", "0": "red", 0: "red",
    "blue": "blue", "b": "blue", "1": "blue", 1: "blue",
}

#: 对角线(顶点到自己)不是 K8 的边, 允许是这些「无意义」占位符。
_DIAG_JUNK = {"", "-", "x", "none", "null", "_", "?"}


def _as_color(v: object) -> str | None:
    """把制品里的一个格子归一化成 'red' / 'blue'; 不是二值颜色则返回 None。"""
    if isinstance(v, bool):          # JSON 的 true/false 不当颜色, 防止 True==1 混进 dict
        return None
    if isinstance(v, str):
        return _COLOR_MAP.get(v.strip().lower())
    if isinstance(v, int):
        return _COLOR_MAP.get(v)
    return None


def _matrix_to_edges(mat: list) -> tuple[dict[tuple[int, int], str], list[str]]:
    """从 8x8 矩阵读出边表 { (i,j) 升序 : 'red'/'blue' }。

    ★ 为什么只用上三角 ★
    K8 的边是无序对 {i,j}, i<j, 共 28 条 —— 每条边**恰好一种**颜色。
    下三角如果和上三角冲突, 说明这份制品自相矛盾; 这种情况记为警告但不
    改变判据: 上三角已经完整决定了那 28 条边的颜色, 判据本身是完备的。
    """
    problems: list[str] = []
    edges: dict[tuple[int, int], str] = {}
    for i in range(N):
        for j in range(N):
            raw = mat[i][j]
            col = _as_color(raw)
            if col is None:
                if i == j and (raw is None or (isinstance(raw, str) and raw.strip().lower() in _DIAG_JUNK)):
                    continue          # 对角线占位符: 合法
                if i != j:
                    problems.append(f"格子[{i}][{j}]={raw!r} 不是 red/blue 二值颜色")
                continue
            if i == j:
                continue              # 对角线无论填什么都不影响 K8 的边
            if j > i:
                edges[(i, j)] = col
    return edges, problems


def _edges_from_red_list(red_pairs: list) -> tuple[dict[tuple[int, int], str], list[str]]:
    """备用制品格式: 只给红边集合, 蓝边取补。这同样是 2-着色的完整刻画。"""
    problems: list[str] = []
    red = set()
    for e in red_pairs:
        if not (isinstance(e, (list, tuple)) and len(e) == 2):
            problems.append(f"红边项 {e!r} 不是 [i, j]")
            continue
        a, b = e
        if not (isinstance(a, int) and isinstance(b, int) and 0 <= a < N and 0 <= b < N) or a == b:
            problems.append(f"红边项 {e!r} 端点非法")
            continue
        red.add((min(a, b), max(a, b)))
    edges = {(i, j): ("red" if (i, j) in red else "blue")
             for i, j in combinations(range(N), 2)}
    return edges, problems


def extract_coloring(art: dict) -> tuple[dict[tuple[int, int], str] | None, list[str]]:
    """从制品里取出 28 条边的颜色表; 取不出返回 (None, 原因)。"""
    problems: list[str] = []
    declared = art.get("n", N)
    if declared != N:
        problems.append(f"制品声明 n={declared!r}, 题目要求 n={N}")

    mat = None
    for key in ("coloring", "matrix", "adjacency"):
        if key in art:
            mat = art[key]
            break
    if mat is not None:
        if not (isinstance(mat, list) and len(mat) == N
                and all(isinstance(r, list) and len(r) == N for r in mat)):
            return None, [f"'{key}' 不是 {N}x{N} 的矩阵"]
        return _matrix_to_edges(mat)

    if isinstance(art.get("red_edges"), list):
        return _edges_from_red_list(art["red_edges"])

    return None, ["制品里既没有 coloring/matrix, 也没有 red_edges"]


def verify(stdout: str, code: str, claim: str) -> dict:
    """独立判定: 制品里的 8 顶点染色是否真的既无红三角又无蓝 K4。

    `code` 参数**故意不使用** —— 见文件头: 执行 agent 代码 = 把判据交给它自己。
    """
    ck = Check()
    art = parse_artifact(stdout, "CONSTRUCT") or parse_artifact(stdout)
    if art is None:
        return {
            "correct": False,
            "certificate": None,
            "final_answer": None,
            "details": {"artifact": False,
                        "note": "未提交 CONSTRUCT: 制品 —— 按契约视为未作答"},
        }
    ck.expect("artifact", True)

    edges, problems = extract_coloring(art)
    if edges is None:
        ck.expect("coloring_wellformed", False, "; ".join(problems))
        return _finish(ck, None, art, stdout)
    ck.expect("coloring_wellformed", not problems, "; ".join(problems))
    ck.expect("edge_count_28", len(edges) == N * (N - 1) // 2,
              f"只读到 {len(edges)} 条边, 应为 28")

    # --- 判据 1: 无红色三角形。枚举全部 C(8,3)=56 个三元组 ---
    red_tri = [t for t in combinations(range(N), 3)
               if all(edges[(min(a, b), max(a, b))] == "red" for a, b in combinations(t, 2))]
    ck.expect("no_red_triangle", not red_tri,
              f"红三角: {[list(t) for t in red_tri[:5]]}")

    # --- 判据 2: 无蓝色 K4。枚举全部 C(8,4)=70 个四元组 ---
    blue_k4 = [q for q in combinations(range(N), 4)
               if all(edges[(min(a, b), max(a, b))] == "blue" for a, b in combinations(q, 2))]
    ck.expect("no_blue_K4", not blue_k4,
              f"蓝 K4: {[list(q) for q in blue_k4[:5]]}")

    # 两种颜色真的都出现了(否则上面任一判据必挂, 这里只作旁证)
    used = {c for c in edges.values()}
    ck.note("colors_used", sorted(used))

    return _finish(ck, edges, art, stdout)


def _finish(ck: Check, edges, art: dict, stdout: str) -> dict:
    if edges is None:
        return {"correct": False, "certificate": None, "final_answer": None,
                "details": {**ck.details, "verdict": parse_verdict(stdout),
                            "reason": ck.reason()}}
    correct = ck.correct
    cert = None
    if correct:
        cert = json.dumps({
            "claim": "R(3,4) > 8",
            "n": N,
            "red_edges": sorted([list(e) for e, c in edges.items() if c == "red"]),
            "enumerated_triples": 56,
            "enumerated_quadruples": 70,
            "red_triangles": 0,
            "blue_K4": 0,
        }, ensure_ascii=False)
    return {
        "correct": correct,
        "certificate": cert,
        "final_answer": {
            "statement": "R(3,4) > 8",
            "n": N,
            "red_edge_count": sum(1 for c in edges.values() if c == "red"),
            "blue_edge_count": sum(1 for c in edges.values() if c == "blue"),
        } if correct else None,
        "details": {
            **ck.details,
            "verdict": parse_verdict(stdout),
            "reason": ck.reason(),
        },
    }


# ==========================================================================
# 自测
# ==========================================================================
def _matrix_from_red(red_pairs) -> list[list[str]]:
    """把红边集合铺成 8x8 矩阵(红 R / 蓝 B / 对角线 '-')。"""
    red = {tuple(sorted(e)) for e in red_pairs}
    m = [[("-" if i == j else ("R" if (min(i, j), max(i, j)) in red else "B")) for j in range(N)]
         for i in range(N)]
    return m


def _good_coloring() -> list[list[str]]:
    """正例: Wagner 图 M8 = 8-环 + 4 条对面弦 作红边。

    为什么它合法:
      * 红图 = C8 加上 4 条直径 (i, i+4)。环上任意三点不可能两两相邻,
        直径边也接不进任何三角形 -> 红图无三角形。
      * 蓝图 = 差分 {±2, ±3} 的循环图, 无三角形 -> 更无 K4。
    这两个性质正例代码会用**和 verify 完全相同的枚举**去确认, 不靠嘴说。
    """
    red = [(i, (i + 1) % N) for i in range(N)] + [(i, i + 4) for i in range(4)]
    return _matrix_from_red(red)


def _selftest() -> bool:
    ok = True

    # 1) 正例
    good = _good_coloring()
    out = f"CONSTRUCT: {json.dumps({'kind': 'coloring', 'n': N, 'coloring': good})}\nVERDICT: PROVED\n"
    r = verify(out, "print('CONSTRUCT ...')", "证明 R(3,4) > 8")
    ok &= r["correct"] is True
    print(f"[{'OK' if r['correct'] else 'FAIL'}] 正例(Wagner M8) correct={r['correct']} reason={r['details'].get('reason')}")

    # 2) 反例 A: 仍无红三角, 但把环边 (0,1) 翻成蓝 -> 立刻冒出蓝 K4 {0,1,3,6}
    bad = [row[:] for row in good]
    bad[0][1] = bad[1][0] = "B"
    out = f"CONSTRUCT: {json.dumps({'kind': 'coloring', 'n': N, 'coloring': bad})}\nVERDICT: PROVED\n"
    r = verify(out, "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例A(翻 (0,1) 为蓝) correct={r['correct']} reason={r['details'].get('reason')}")

    # 3) 反例 B: 把 (0,2) 也染成红 -> 红三角 {0,1,2}
    bad2 = [row[:] for row in good]
    bad2[0][2] = bad2[2][0] = "R"
    out = f"CONSTRUCT: {json.dumps({'kind': 'coloring', 'n': N, 'coloring': bad2})}\nVERDICT: PROVED\n"
    r = verify(out, "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例B(加红边 (0,2)) correct={r['correct']} reason={r['details'].get('reason')}")

    # 4) 反例 C: 压根没交制品
    r = verify("VERDICT: PROVED\n", "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例C(无制品) correct={r['correct']} reason={r['details'].get('note')}")

    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)