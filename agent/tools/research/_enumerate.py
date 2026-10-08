"""
agent.tools.research._enumerate — 有限枚举反例搜索 (P7 阶段 1)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §1.5.1 判据 2 / §6.1

================================ 这一层的定位 =================================
找反例有三种手段, 强度差别很大, **不可互相替代**:

    Z3 sat 模型      → 全域级反例 (一个满足全部前提的具体赋值)  最强
    本模块的枚举     → n ≤ 7 的**全覆盖**反例                    中
    NumPy 随机抽查   → 采样点反例, 漏得多                      最弱

所以本模块的价值不是"穷举", 而是"**在可穷举的范围内保证不漏**":
graph_atlas 给出 n≤7 的全部 1253 个非同构简单图 —— 在这个范围内, 说"没找到反例"
是**穷尽性结论**, 而不是"试了几个没试出来"。

⚠️ 但即使穷尽也只到 n=7, 所以**枚举永远只能给 `partial`, 永远不能给 `proved`**
(见 §6.4 判定纪律)。想让判定升级为 proved, 只有两条路: 符号计算到 0, 或 Z3 unsat。

历史:
  - 2026-10-08 P7.1 首版
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 图性质检查器
# ============================================================
def _safe(fn: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """性质检查器不该让异常冒泡 —— 退化点在 n=0/1 的退化图上很常见"""

    def wrapper(G):
        try:
            return bool(fn(G))
        except Exception:
            return False      # 退化图上判定不了 → 视为"不满足", 宁可不漏反例

    wrapper.__name__ = fn.__name__
    return wrapper


# --- 各性质的实现: 局部 import networkx, 避免模块级硬依赖 ---
def _p_bipartite(G):
    import networkx as nx
    return nx.is_bipartite(G)


def _p_triangle_free(G):
    import networkx as nx
    # ⚠️ nx.triangles(G) 返回 {节点: 该节点上的三角形数量(int)}, 例如单点图是 {0: 0}
    #    —— 值是**计数**不是嵌套 dict(踩过两次: 先误以为值是 dict 用 len() 取长度,
    #    更早还写过 `any(True for _ in nx.triangles(G))` 那是"有没有节点",恒为真)。
    #    正确写法: 所有计数都为 0 才算无三角形。
    #    写错的后果: 单点图被误判成含三角形 → 假反例, 而且永远在 n=1 就命中。
    return not any(nx.triangles(G).values())


def _p_connected(G):
    import networkx as nx
    return G.number_of_nodes() <= 1 or nx.is_connected(G)


def _p_planar(G):
    import networkx as nx
    return nx.check_planarity(G)[0]


def _p_eulerian(G):
    return all(d % 2 == 0 for _, d in G.degree())


def _p_acyclic(G):
    import networkx as nx
    return nx.is_forest(G)


def _p_chordal(G):
    import networkx as nx
    return nx.is_chordal(G)


def _p_perfect_matching(G):
    import networkx as nx
    # 奇数节点图显然没有完美匹配, 直接判 False(而不是交给 nx 抛异常)
    if G.number_of_nodes() % 2:
        return False
    return nx.has_perfect_matching(G)


def _p_regular(G):
    degs = {d for _, d in G.degree()}
    return len(degs) <= 1


def _p_hamiltonian(G):
    import networkx as nx
    return nx.is_hamiltonian(G)


def _p_connected_bipartite(G):
    return _p_connected(G) and _p_bipartite(G)


GRAPH_PROPERTIES: dict[str, Callable[[Any], bool]] = {
    "bipartite": _safe(_p_bipartite),
    "triangle_free": _safe(_p_triangle_free),
    "connected": _safe(_p_connected),
    "planar": _safe(_p_planar),
    "eulerian": _safe(_p_eulerian),
    "acyclic": _safe(_p_acyclic),
    "chordal": _safe(_p_chordal),
    "has_perfect_matching": _safe(_p_perfect_matching),
    "regular": _safe(_p_regular),
    "hamiltonian": _safe(_p_hamiltonian),
    "connected_and_bipartite": _safe(_p_connected_bipartite),
}

GRAPH_PROPERTY_DOC: dict[str, str] = {
    "bipartite": "是二部图 (无奇环)",
    "triangle_free": "无三角形",
    "connected": "连通",
    "planar": "平面可画",
    "eulerian": "所有顶点度数为偶数 (欧拉图)",
    "acyclic": "无环 (森林)",
    "chordal": "弦图",
    "has_perfect_matching": "存在完美匹配",
    "regular": "正则图 (每个顶点度数相同)",
    "hamiltonian": "存在哈密顿回路",
    "connected_and_bipartite": "连通且二部",
}


# ============================================================
# 整数序列
# ============================================================
def _seq_partition(k: int) -> list[int]:
    """k 的整数划分(升序展开), 如 partition(5) = [1,1,1,1,1]"""
    out: list[int] = []

    def rec(rem, cur):
        if rem == 0:
            out.extend(cur)
            return
        for v in range(min(rem, len(cur) + 1), 0, -1):
            cur.append(v)
            rec(rem - v, cur)
            cur.pop()
    rec(k, [])
    return out


def _seq_partition_count(k: int) -> list[int]:
    """整数划分数 p(0..k): 1,1,2,3,5,7,11,..."""
    from sympy.functions.combinatorial.numbers import partition
    return [int(partition(i)) for i in range(0, max(10, k))]


def _seq_primes(k: int) -> list[int]:
    from sympy import primerange
    return [int(p) for p in primerange(1, max(50, k * 25))]


def _seq_fibonacci(k: int) -> list[int]:
    from sympy import fibonacci
    return [int(fibonacci(i)) for i in range(max(10, k))]


def _seq_powers_of_two(k: int) -> list[int]:
    return [2 ** i for i in range(max(10, k))]


def _seq_factorials(k: int) -> list[int]:
    from sympy import factorial
    return [int(factorial(i)) for i in range(max(6, min(12, k or 6)))]


SEQUENCE_KINDS: dict[str, Callable[[int], list[int]]] = {
    "partition_count": _seq_partition_count,
    "primes": _seq_primes,
    "fibonacci": _seq_fibonacci,
    "powers_of_two": _seq_powers_of_two,
    "factorials": _seq_factorials,
}


def _s_all_prime(s: list[int]) -> bool:
    from sympy import isprime
    return all(isprime(x) for x in s if x > 1)


def _safe_seq(fn):
    """序列性质检查器的异常护栏(与 _safe 同理, 但收列表入参)"""
    def wrapper(s):
        try:
            return bool(fn(s))
        except Exception:
            return False
    wrapper.__name__ = fn.__name__
    return wrapper


SEQUENCE_PROPERTIES: dict[str, Callable[[list[int]], bool]] = {
    "all_prime": _safe_seq(_s_all_prime),
    "all_distinct": _safe_seq(lambda s: len(set(s)) == len(s)),
    "monotonic_increasing": _safe_seq(lambda s: all(s[i] < s[i + 1] for i in range(len(s) - 1))),
    "monotonic_non_decreasing": _safe_seq(lambda s: all(s[i] <= s[i + 1] for i in range(len(s) - 1))),
    "convex": _safe_seq(lambda s: all(
        (s[i + 1] - s[i]) <= (s[i + 2] - s[i + 1]) for i in range(len(s) - 2))),
    "log_convex": _safe_seq(lambda s: all(
        (s[i] * s[i + 2]) <= (s[i + 1] ** 2) for i in range(len(s) - 2))),
}


# ============================================================
# 枚举主入口
# ============================================================
#: graph_atlas 覆盖的最大顶点数 —— 硬上限, 超过会慢到不可用
MAX_ATLAS_N = 7


def enumerate_graph_property(prop: str, n_max: int = MAX_ATLAS_N) -> dict[str, Any]:
    """
    在 n=1..n_max 的**全部非同构简单图**上检查性质, 找反例。

    n≤7 时 graph_atlas 覆盖了所有非同构图, 所以这个范围内"无反例"是穷尽结论。

    Returns:
        {ok, task, prop, checked, counterexample, verdict, reason, elapsed_ms, atlas_exhaustive}
    """
    import networkx as nx

    # ⚠️ networkx 3.x 里 graph_atlas() 需要下标参数, 全量遍历要用 graph_atlas_g()
    #    (2.x 时代叫 graph_atlas())。写错这行 → TypeError 直接崩。
    from networkx.generators.atlas import graph_atlas_g

    started = time.time()

    if prop not in GRAPH_PROPERTIES:
        raise ValueError(
            f"未知图性质 {prop!r}, 可用: {', '.join(sorted(GRAPH_PROPERTIES))}"
        )

    n_max = max(1, min(int(n_max), MAX_ATLAS_N))
    check = GRAPH_PROPERTIES[prop]

    checked = 0
    counterexample: dict[str, Any] | None = None
    per_n: dict[str, int] = {}

    for G in graph_atlas_g():
        n = G.number_of_nodes()
        if n == 0 or n > n_max:
            continue
        checked += 1
        per_n[str(n)] = per_n.get(str(n), 0) + 1
        if not check(G):
            counterexample = {
                "n": n,
                "edges": sorted(G.edges()),
                "degrees": dict(G.degree()),
                "graph6": nx.to_graph6_bytes(G, header=False).decode() if n <= 6 else "(图6编码略)",
            }
            break       # 找到第一个就够 —— 反例只需要一个

    elapsed_ms = int((time.time() - started) * 1000)
    doc = GRAPH_PROPERTY_DOC.get(prop, prop)

    if counterexample:
        verdict = "refuted"
        reason = (f"找到反例: n={counterexample['n']} 的非同构图(边 {counterexample['edges']}) "
                  f"**不满足**「{doc}」。这是穷尽性搜索下的真反例, 不是抽样没碰到。")
    else:
        verdict = "partial"
        reason = (f"在 n=1..{n_max} 的全部 {checked} 个非同构简单图上均满足「{doc}」。"
                  f"注意: 这只是**有限范围**的穷尽检查, n>{n_max} 未覆盖, "
                  f"因此不能宣称命题全域成立。")

    return {
        "ok": True,
        "task": "graph",
        "prop": prop,
        "prop_doc": doc,
        "n_max": n_max,
        "checked": checked,
        "graphs_per_n": per_n,
        "atlas_exhaustive": n_max <= MAX_ATLAS_N,
        "counterexample": counterexample,
        "verdict": verdict,
        "reason": reason,
        "elapsed_ms": elapsed_ms,
    }


def enumerate_sequence_property(
    kind: str,
    prop: str,
    n_terms: int = 20,
) -> dict[str, Any]:
    """
    对内置序列检查性质, 找最早失效的前缀。

    Returns:
        与 enumerate_graph_property 同构
    """
    started = time.time()

    if kind not in SEQUENCE_KINDS:
        raise ValueError(f"未知序列 {kind!r}, 可用: {', '.join(sorted(SEQUENCE_KINDS))}")
    if prop not in SEQUENCE_PROPERTIES:
        raise ValueError(
            f"未知序列性质 {prop!r}, 可用: {', '.join(sorted(SEQUENCE_PROPERTIES))}"
        )

    full = SEQUENCE_KINDS[kind](n_terms)
    check = SEQUENCE_PROPERTIES[prop]

    # 逐步加长前缀, 找到最早失败的 n —— 短前缀失败意味着再长也失败
    first_bad: int | None = None
    for n in range(3, len(full) + 1):
        if not check(full[:n]):
            first_bad = n
            break

    elapsed_ms = int((time.time() - started) * 1000)
    shown = full[:12]

    if first_bad is not None:
        verdict = "refuted"
        reason = (f"序列 {kind} = {shown} ... 在前 {first_bad} 项时已不满足「{prop}」"
                  f"—— 最短反例前缀: {full[:first_bad + 2]}")
    else:
        verdict = "partial"
        reason = (f"序列 {kind} = {shown} ... 在前 {len(full)} 项上均满足「{prop}」。"
                  f"有限项检查, 不构成全域证明。")

    return {
        "ok": True,
        "task": "sequence",
        "kind": kind,
        "prop": prop,
        "sequence_head": shown,
        "n_terms": len(full),
        "counterexample": full[:first_bad + 2] if first_bad else None,
        "first_fail_prefix_len": first_bad,
        "verdict": verdict,
        "reason": reason,
        "elapsed_ms": elapsed_ms,
    }


def dispatch(task: str, prop: str = "", n_max: int = MAX_ATLAS_N,
             kind: str = "", n_terms: int = 20) -> dict[str, Any]:
    """按 task 分发"""
    if task == "graph":
        return enumerate_graph_property(prop, n_max)
    if task == "sequence":
        return enumerate_sequence_property(kind or "primes", prop, n_terms)
    raise ValueError(f"未知 task={task!r}, 可用: graph / sequence")
