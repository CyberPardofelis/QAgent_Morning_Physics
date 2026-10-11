"""
res-019-group-construction-60 的独立 verifier。

题目: 构造一个 60 阶非交换群, 给出乘法表和生成元, 验证群公理、非交换性、
      以及是否为 A5。

★ 为什么"阶 60 + 非交换"不够 ★
60 阶非交换群不止一个(S3 x C10, D30, 各类直积/半直积), 题库 why_hard 点名
S3xC10。所以判据必须继续往下走, 到"是不是 A5"为止。verifier 用两条独立
理由把答案钉死:

  置换路径(推荐给 agent 的表示: 生成元 + S5 上的置换):
    生成元逐个解析成 {0..4} 上的置换, 自己算它生成的子群 <- BFS 闭包。
    阶 = 60 且所有元素都是偶置换 => 该子群 ⊆ A5, 而 |A5| = 120/2 = 60,
    阶相等 => **逐元素等于 A5**。这是集合相等, 不是"看起来像"。

  乘法表路径(60x60 整数矩阵):
    拉丁方 + 单位元 + 逆元 + 全 216000 个三元组的结合律, 全部自己算。
    再算**导出子群 [G,G]**: 若 [G,G] = G, 则 G 是完美群, 而任何正规子群都
    必含 [G,G], 于是 G 平凡或全群 => **G 单**。60 阶单群唯一(经典定理)
    就是 A5。再用"元素阶分布 = {1:1, 2:15, 3:20, 5:24}"做独立旁证。

为什么这样判就能证明答案是对的:
  结合律/单位元/逆元是把群公理**逐条实例化**检查(60^3 个三元组),
  单性由完美性严格推出, 同构结论由"阶相等 + 单"或"集合相等"得到 ——
  没有任何一步依赖 agent 的说法或它的搜索代码。
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
from collections import Counter
from typing import Any

try:
    from .common import Check, parse_artifact, parse_verdict
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import Check, parse_artifact, parse_verdict

import numpy as np

TARGET_ORDER = 60
DEGREE = 5                     # S5 / A5 的作用次数
#: A5 的元素阶分布, 用作独立旁证
A5_ORDER_PROFILE = {1: 1, 2: 15, 3: 20, 5: 24}


# ==========================================================================
# 置换工具
# ==========================================================================
def _perm_mul(a: tuple, b: tuple) -> tuple:
    """置换复合: 先 a 后 b。(a o b)(i) = b(a(i))"""
    return tuple(b[a[i]] for i in range(len(a)))


def _is_even(p: tuple) -> bool:
    """排列奇偶性 = 符号 = (置换长度 - 圈数) mod 2。"""
    n = len(p)
    seen = [False] * n
    trans = 0
    for i in range(n):
        if seen[i]:
            continue
        j, L = i, 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            L += 1
        trans += L - 1
    return trans % 2 == 0


def _perm_candidates(spec: Any, n: int = DEGREE) -> list[tuple]:
    """把一个置换规格解析成**若干候选**一行记法元组。

    为什么返回候选而不是单个: 循环记法 "(1 2 3)" 在 5 元集上有 0-based 与
    1-based 两种读法且无法从文本区分。把两种读法都跑一遍, 任一读法生成的
    子群若真是 A5, 那答案就是 A5 —— 这个"或"不放松判据, 因为两条读法
    各自都要独立过完阶/非交换/A5 三道硬判据。
    """
    cands: list[tuple] = []

    def _add(g: list[int]) -> None:
        t = tuple(g)
        if sorted(t) == list(range(n)) and t not in cands:
            cands.append(t)

    # (a) 一行记法 [2,0,1,3,4]
    if isinstance(spec, (list, tuple)) and len(spec) == n \
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in spec):
        seq = [int(x) for x in spec]
        if all(float(x).is_integer() for x in spec):
            _add(seq)

    # (b) 轮换列表 [[0,1],[1,2]] —— 左作用连乘
    if isinstance(spec, (list, tuple)) and spec and all(
            isinstance(c, (list, tuple)) and len(c) >= 2 for c in spec):
        try:
            g = list(range(n))
            for c in spec:
                cs = [int(x) for x in c]
                if any(not 0 <= x < n for x in cs):
                    g = None
                    break
                for a, b in zip(cs, cs[1:]):      # 注意: 不绕回, 绕回会退化成对换
                    g[a], g[b] = g[b], g[a]
            if g is not None:
                _add(g)
        except Exception:
            pass

    # (c) 循环记法字符串 "(1 2 3)(4 5)"
    if isinstance(spec, str):
        labels = [int(x) for x in re.findall(r"\d+", spec)]
        if labels:
            bases = [0] if 0 in labels else ([1] if n in labels else [0, 1])
            for base in bases:
                g = list(range(n))
                ok = True
                for cyc in re.findall(r"\(([^)]*)\)", spec):
                    cs = [int(x) - base for x in re.findall(r"\d+", cyc)]
                    if any(not 0 <= x < n for x in cs):
                        ok = False
                        break
                    for a, b in zip(cs, cs[1:]):
                        g[a], g[b] = g[b], g[a]
                if ok:
                    _add(g)

    return cands


def _generated_subgroup(gens: list[tuple], n: int = DEGREE) -> set[tuple]:
    """自己算生成子群: 从单位元出发做 BFS 闭包。"""
    ident = tuple(range(n))
    seen = {ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for x in frontier:
            for g in gens:
                z = _perm_mul(x, g)
                if z not in seen:
                    seen.add(z)
                    nxt.append(z)
        frontier = nxt
    return seen


# ==========================================================================
# 乘法表工具
# ==========================================================================
def _identity_index(M: np.ndarray, labels: list[int]) -> int | None:
    """找单位元下标 e: 第 e 行与第 e 列都要恰是 labels 的一个排列。"""
    n = M.shape[0]
    target = np.asarray(labels)
    for e in range(n):
        if np.array_equal(np.sort(M[e]), target) and np.array_equal(np.sort(M[:, e]), target):
            return e
    return None


def _table_inverses(M: np.ndarray, e: int, labels: list[int]) -> dict[int, int] | None:
    n = M.shape[0]
    lab2idx = {v: i for i, v in enumerate(labels)}
    e_lab = labels[e]
    inv: dict[int, int] = {}
    for a in range(n):
        hit = np.nonzero(M[a] == e_lab)[0]
        if len(hit) != 1:
            return None
        b = int(hit[0])
        if M[b, a] != e_lab:                     # 两侧都得回到单位元
            return None
        inv[a] = b
    return inv


def _derived_subgroup(M: np.ndarray, e: int, inv: dict[int, int]) -> set[int]:
    """导出子群 [G,G]: 由全部对子 [a,b]=a b a^-1 b^-1 生成的闭包。"""
    comms = set()
    for a in range(M.shape[0]):
        for b in range(M.shape[0]):
            comms.add(int(M[int(M[int(M[a, b]), inv[a]]), inv[b]]))
    seen = {e}
    frontier = [e]
    gens = sorted(comms)
    while frontier:
        nxt = []
        for x in frontier:
            row = M[x]
            for y in gens:
                z = int(row[y])
                if z not in seen:
                    seen.add(z)
                    nxt.append(z)
        frontier = nxt
    return seen


def _order_profile(M: np.ndarray, e: int) -> dict[int, int]:
    prof: Counter = Counter()
    n = M.shape[0]
    for a in range(n):
        x, k = a, 1
        while x != e:
            x = int(M[a, x])
            k += 1
        prof[k] += 1
    return dict(sorted(prof.items()))


def _associative(M: np.ndarray) -> bool:
    """自己查结合律: 全部 n^3 个三元组一次性向量化。

        left[a,b,c]  = M[M[a,b], c]      <- M[M] 就是它
        right[a,b,c] = M[a, M[b,c]]      <- 行号广播成 (n,1,1),
            列号取 M[None,:,:] (即 idx[a,b,c] = M[b,c]), 两个数组广播成 (n,n,n)
    两个都是 (n,n,n), 逐元素比即可。注意 M[a,b] 已经是"行 a 列 b 的乘积",
    所以 a*(b*c) 只用两个下标, 没有第三个。
    """
    n = M.shape[0]
    ii = np.arange(n)
    left = M[M]
    right = M[ii[:, None, None], M[None, :, :]]
    return bool(np.array_equal(left, right))


def _associative_naive(M: np.ndarray) -> bool:
    """三重循环版, 只在自测里用来校验 _associative 本身没写错。"""
    n = M.shape[0]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if M[M[a, b], c] != M[a, M[b, c]]:
                    return False
    return True


def _find_noncommuting_pair(M: np.ndarray) -> tuple[int, int] | None:
    n = M.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            if M[a, b] != M[b, a]:
                return (a, b)
    return None


# ==========================================================================
def verify(stdout: str, code: str, claim: str) -> dict:
    chk = Check()

    art = parse_artifact(stdout, "CONSTRUCT") or parse_artifact(stdout, "ANSWER")
    if not chk.expect("artifact_present", art is not None,
                      "未提交 CONSTRUCT 制品(需要 group / generators / is_A5)"):
        return _finish(chk, {})
    assert art is not None

    # ---- 题目要它回答"是不是 A5" ----
    claimed_a5 = art.get("is_A5", art.get("is_a5"))
    chk.note("claimed_is_A5", claimed_a5)
    chk.expect("claims_is_A5", bool(claimed_a5),
               "题目要求验证是否为 A5, 制品里 is_A5 应为 true")

    gens_spec = art.get("generators") or art.get("gens") or art.get("generating_set")
    table_spec = art.get("table") or art.get("multiplication_table") \
        or art.get("cayley_table")

    if gens_spec is not None and table_spec is None:
        return _verify_by_permutation(chk, gens_spec)
    if table_spec is not None:
        return _verify_by_table(chk, table_spec)
    if gens_spec is not None:
        return _verify_by_permutation(chk, gens_spec)

    chk.expect("representation_given", False,
               "既没给置换表示的生成元, 也没给乘法表")
    return _finish(chk, {})


# ---------------------------------------------------------------- 置换路径
def _verify_by_permutation(chk: Check, gens_spec: Any) -> dict:
    n = DEGREE
    if not isinstance(gens_spec, (list, tuple)) or not gens_spec:
        chk.expect("generators_present", False, "generators 不是非空列表")
        return _finish(chk, {})
    chk.expect("generators_present", True)

    # 每个生成元可能有多种读法, 组合起来逐个试
    per_gen = [_perm_candidates(g, n) for g in gens_spec]
    chk.note("generator_readings", [len(c) for c in per_gen])
    if any(not c for c in per_gen):
        chk.expect("generators_parse", False,
                   f"有生成元无法解析成 {n} 元置换: {gens_spec!r}")
        return _finish(chk, {})

    attempts: list[tuple] = list(itertools.product(*per_gen))
    order = None
    best_H: set[tuple] | None = None
    for combo in attempts:
        H = _generated_subgroup(list(combo), n)
        if order is None or len(H) < order:
            order, best_H = len(H), H
        if len(H) == TARGET_ORDER:
            order, best_H = len(H), H
            break
    assert best_H is not None and order is not None

    chk.note("subgroup_order", order)
    chk.expect("order_is_60", order == TARGET_ORDER,
               f"自己算的生成子群阶 = {order} != 60 (它不是 60 阶群)")

    nonab = None
    if len(best_H) == TARGET_ORDER:
        elts = sorted(best_H)
        for a in elts:
            for b in elts:
                if a >= b:
                    continue
                if _perm_mul(a, b) != _perm_mul(b, a):
                    nonab = (a, b)
                    break
            if nonab:
                break
    chk.expect("nonabelian", nonab is not None,
               "群里所有元素两两可交换 —— 这是交换群, 不满足非交换要求")
    chk.note("noncommuting_pair", list(nonab) if nonab else None)

    # A5 判定: 阶 60 且全偶 => 逐元素等于 A5
    is_even_all = all(_is_even(g) for g in best_H) if len(best_H) == TARGET_ORDER else False
    A5 = _all_even(n)
    chk.note("all_elements_even", is_even_all)
    chk.expect("is_A5", (len(best_H) == TARGET_ORDER and best_H == A5),
               f"生成的子群(阶 {order})与 A5(偶置换集)不相等 —— 它不是 A5")

    return _finish(chk, {"representation": "permutation", "order": order,
                         "is_A5": bool(len(best_H) == TARGET_ORDER and best_H == A5),
                         "nonabelian": nonab is not None})


def _all_even(n: int = DEGREE) -> set[tuple]:
    """S5 的偶置换集合 = A5 (120/2 = 60 个)。"""
    out: set[tuple] = set()
    for p in itertools.permutations(range(n)):
        if _is_even(p):
            out.add(p)
    return out


# ---------------------------------------------------------------- 乘法表路径
def _verify_by_table(chk: Check, table_spec: Any) -> dict:
    try:
        M = np.asarray(table_spec, dtype=np.int64)
    except Exception:
        chk.expect("table_is_matrix", False, "table 不能转成整数矩阵(可能是参差的)")
        return _finish(chk, {})
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        chk.expect("table_square", False, f"乘法表不是方阵: {M.shape}")
        return _finish(chk, {})
    chk.expect("table_square", True)

    n = M.shape[0]
    labels = sorted({int(v) for v in np.unique(M)})
    chk.expect("labels_contiguous",
               len(labels) == n and (labels == list(range(n)) or labels == list(range(1, n + 1))),
               f"表的元素标签集合 {labels[:3]}... 共 {len(labels)} 个, 与阶 {n} 不符")

    # 闭包: 每行每列都是标签的一个排列 => 拉丁方
    target = np.asarray(labels)
    rows_ok = all(np.array_equal(np.sort(M[i]), target) for i in range(n))
    cols_ok = all(np.array_equal(np.sort(M[:, i]), target) for i in range(n))
    chk.expect("latin_square_closure", rows_ok and cols_ok,
               "乘法表不是拉丁方(某行/列缺元素或有重复) => 对每个元素不封闭")

    e = _identity_index(M, labels)
    if not chk.expect("identity_exists", e is not None,
                      "找不到单位元(没有哪一行/列恰好是恒等排列)"):
        return _finish(chk, {})
    assert e is not None
    chk.note("identity_index", e)

    inv = _table_inverses(M, e, labels)
    chk.expect("inverses_exist", inv is not None,
               "存在没有逆元的元素(某行里找不到单位元)")

    # 结合律: 全 n^3 = 216000 个三元组, 向量化一次算完
    chk.expect("associative", _associative(M),
               "结合律不成立: 存在 (a,b,c) 使 (ab)c != a(bc)")
    chk.note("triples_checked", int(n) ** 3)

    chk.expect("order_is_60", n == TARGET_ORDER, f"群的阶 = {n} != 60")

    pair = _find_noncommuting_pair(M)
    chk.expect("nonabelian", pair is not None, "这是交换群, 不满足非交换要求")
    chk.note("noncommuting_pair", list(pair) if pair else None)

    is_a5 = False
    if inv is not None and n == TARGET_ORDER:
        prof = _order_profile(M, e)
        chk.note("order_profile", {str(k): v for k, v in prof.items()})
        chk.expect("order_profile_matches_A5", prof == A5_ORDER_PROFILE,
                   f"元素阶分布 {prof} != A5 的 {A5_ORDER_PROFILE}")
        D = _derived_subgroup(M, e, inv)
        perfect = len(D) == n
        chk.note("derived_subgroup_order", len(D))
        chk.expect("perfect_hence_simple", perfect,
                   f"[G,G] 只有 {len(D)} 个元素 != 60 => G 不完美 => 存在非平凡正规"
                   f"子群 => G 不单 => 不可能是 A5(S3xC10 之类就栽在这里)")
        # 60 阶 + 单 => 只能是 A5 (经典定理); 阶分布再做一次独立旁证
        is_a5 = perfect and prof == A5_ORDER_PROFILE and n == TARGET_ORDER
        chk.expect("is_A5", is_a5,
                   "不是 A5: 60 阶单群唯一, 而这里群不单或元素阶分布不符")

    return _finish(chk, {"representation": "table", "order": int(n),
                         "is_A5": bool(is_a5), "nonabelian": pair is not None})


def _finish(chk: Check, extra: dict) -> dict:
    correct = chk.correct
    if correct:
        cert = (f"60 阶非交换群验证通过, 且与 A5 同构"
                f"(表示方式: {extra.get('representation')})")
    else:
        cert = chk.reason()
    final = json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else None
    return {"correct": correct, "certificate": cert, "final_answer": final,
            "details": dict(chk.details)}


# ==========================================================================
if __name__ == "__main__":
    # ---- 判据本身的自检: 向量化结合律 vs 三重循环(防止我自己写错索引) ----
    rng = np.random.default_rng(1)
    agree = 0
    for _ in range(5):                      # 随机拉丁方(几乎必然不结合)
        Mx = np.stack([rng.permutation(5) for _ in range(5)])
        assert _associative(Mx) == _associative_naive(Mx), "向量化结合律与朴素版不一致"
        agree += int(_associative(Mx))
    C4 = np.array([[(a + b) % 4 for b in range(4)] for a in range(4)])
    assert _associative(C4) and _associative_naive(C4)
    assert agree == 0, "随机拉丁方竟全是结合的, 测试样本无效"
    print(f"[OK] 判据自检: 5 个随机拉丁方(均不结合) + C4(结合), 向量化与朴素版一致")

    # A5 = <(1 2 3), (1 2 3 4 5)> —— 一行记法, i -> g[i]; 两者皆偶, 生成 60 阶
    a = (1, 2, 0, 3, 4)          # (0 1 2)     的 0-based 一行记法
    b = (1, 2, 3, 4, 0)          # (0 1 2 3 4) 的 0-based 一行记法

    r = verify('CONSTRUCT: ' + json.dumps({
        "order": 60, "is_A5": True, "nonabelian": True,
        "representation": "S5", "generators": [list(a), list(b)]}) + "\n", "", "")
    assert r["correct"], f"正例应通过: {r}"
    print("[OK] 正例: <(0 1 2),(0 1 2 3 4)> ->", r["certificate"],
          "| 阶:", r["details"]["subgroup_order"])

    # 正例 2: 同样的 A5, 改用 60x60 乘法表提交 —— 走另一条判据分支
    els = sorted(_generated_subgroup([a, b], DEGREE))
    pos = {g: i for i, g in enumerate(els)}
    a5_table = [[pos[_perm_mul(x, y)] for y in els] for x in els]
    r = verify('CONSTRUCT: ' + json.dumps({
        "is_A5": True, "table": a5_table}) + "\n", "", "")
    assert r["correct"], f"正例(乘法表路径)应通过: {r}"
    print("[OK] 正例2: A5 的 60x60 乘法表 ->", r["certificate"],
          "| 检查了三元组数:", r["details"]["triples_checked"])

    # 反例 A: 只生成 12 阶(A4: <(0 1 2),(0 1)(2 3)>), 不是 60 阶
    a4 = [(1, 2, 0, 3, 4), (1, 0, 3, 2, 4)]
    r = verify('CONSTRUCT: ' + json.dumps({
        "is_A5": True, "generators": a4}) + "\n", "", "")
    assert not r["correct"] and not r["details"]["order_is_60"], r
    print("[OK] 反例A: 生成子群阶 =", r["details"]["subgroup_order"], "->", r["certificate"])

    # 反例 B: 交换群 C60 —— 阶 60 但不是非交换、不是 A5
    # 乘法表: 元素 i 代表 C60 的 i, 乘法 = (a+b) mod 60
    c60 = [[(a + b) % 60 for b in range(60)] for a in range(60)]
    r = verify('CONSTRUCT: ' + json.dumps({
        "is_A5": True, "table": c60}) + "\n", "", "")
    assert (not r["correct"] and not r["details"]["nonabelian"]
            and not r["details"]["perfect_hence_simple"]), r
    print("[OK] 反例B: C60(阶60但交换) ->", r["certificate"])

    # 反例 C: 乘法表不是拉丁方(某格填成 0), 闭包就破了
    bad = [[(a + b) % 60 for b in range(60)] for a in range(60)]
    bad[3][4] = bad[3][5]      # 行内出现重复 -> 拉丁方破了(闭包不成立)
    r = verify('CONSTRUCT: ' + json.dumps({"is_A5": True, "table": bad}) + "\n", "", "")
    assert not r["correct"] and not r["details"]["latin_square_closure"], r
    print("[OK] 反例C: 乘法表不是拉丁方 ->", r["certificate"])

    print("[OK] 全部自测通过")