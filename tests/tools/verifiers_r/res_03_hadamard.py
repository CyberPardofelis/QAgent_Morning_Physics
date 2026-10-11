# -*- coding: utf-8 -*-
"""res-003-hadamard-28-paley —— 28 阶 Hadamard 矩阵的独立验证器。

题目: 「构造一个 28 阶 Hadamard 矩阵, 并验证 H H^T = 28 I。」

★ 判据为什么是「一次乘法」★
n 阶 Hadamard 矩阵的定义就是: H 是 n x n、元素取 ±1、且任意两行两两正交。
「两两正交」在代数上恰好等价于 H H^T = n I —— 因为 (HH^T)_ij = 第 i 行与第 j 行的
内积, 对角线 H H^T_ii = Σ H_ik^2 = n(全是 ±1), 非对角线为 0 正是正交。
所以判据就是**本文件自己把 28x28 的矩阵从头乘一遍**, 28*28*28 = 21952 次乘法,
毫秒级, 不需要任何符号推理。

★ 为什么必须重算, 而不是看 agent 打印的 HH^T ★
agent 只要把 `H @ H.T` 写错成 `H @ H`(少转置), 或者干脆把 numpy 那个
`allclose` 的 rtol 放得很大, 就会得到一个假的 PASS。这里全部用 **Python 整数**
算, 一个字节的浮点误差都没有: 28 阶矩阵的行内积必然落在 -28..28 的整数上,
用浮点比较反而引入了本不该有的容差空间。

★ 不要求规范化(normalized) ★
首行全 +1 只是 Hadamard 矩阵的一个可选规范, 题目没要求。
一个未规范化但满足 HH^T = 28I 的矩阵是**完全正确**的答案, 本文件判它通过;
反过来说, 也不接受「先除以 sqrt(28) 归一化成浮点」这种把整数变成浮点的交付。
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import Check, parse_artifact, parse_verdict  # noqa: E402

#: 题面参数: 28 阶。
ORDER = 28


def _as_pm1(v: object) -> int | None:
    """归一化成 +1 / -1; 其它一律 None(0、2、浮点 1.0 都不接受)。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v in (-1, 1) else None
    if isinstance(v, str):
        s = v.strip().replace("\u2212", "-").replace("\u2013", "-").replace("−", "-")
        if s in ("1", "+1"):
            return 1
        if s in ("-1", "\u22121"):
            return -1
    return None


def gram(M: list[list[int]]) -> list[list[int]]:
    """自己算 M M^T, 全程整数。"""
    n = len(M)
    return [[sum(M[i][k] * M[j][k] for k in range(n)) for j in range(n)] for i in range(n)]


def verify(stdout: str, code: str, claim: str) -> dict:
    """独立判定: 制品里的矩阵是不是真的 28 阶 Hadamard 矩阵。

    `code` 参数**故意不使用** —— 见文件头。
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

    raw = None
    for key in ("matrix", "H", "hadamard"):
        if key in art:
            raw = art[key]
            break
    if not isinstance(raw, list):
        ck.expect("matrix_present", False, "制品里没有 matrix")
        return _finish(ck, None, art, stdout)
    ck.expect("matrix_present", True)

    # --- 形状 ---
    ck.expect("rows_%d" % ORDER, len(raw) == ORDER, f"行数 {len(raw)} != {ORDER}")
    if len(raw) != ORDER:
        return _finish(ck, None, art, stdout)
    bad_shape = [i for i, r in enumerate(raw) if not (isinstance(r, list) and len(r) == ORDER)]
    ck.expect("cols_%d" % ORDER, not bad_shape, f"行宽不为 {ORDER} 的行: {bad_shape[:5]}")
    if bad_shape:
        return _finish(ck, None, art, stdout)

    # --- 元素只有 ±1 ---
    M: list[list[int]] = [[0] * ORDER for _ in range(ORDER)]
    bad_entries = []
    for i in range(ORDER):
        for j in range(ORDER):
            v = _as_pm1(raw[i][j])
            if v is None:
                bad_entries.append((i, j, raw[i][j]))
            else:
                M[i][j] = v
    ck.expect("entries_pm1", not bad_entries, f"非 ±1 元素: {bad_entries[:5]}")
    if bad_entries:
        return _finish(ck, None, art, stdout)

    # --- 核心判据: H H^T == 28 I ---
    G = gram(M)
    bad_pairs = []
    for i in range(ORDER):
        for j in range(ORDER):
            want = ORDER if i == j else 0
            if G[i][j] != want:
                bad_pairs.append((i, j, G[i][j], want))
    ck.expect("HHt_equals_28I", not bad_pairs,
              f"{len(bad_pairs)} 个位置不等, 例如 (i,j,实测,应为)={bad_pairs[:5]}")
    ck.note("row_norm_sq", [G[i][i] for i in range(ORDER)] if not bad_pairs else None)
    ck.note("n_offdiag_violations", len(bad_pairs))

    return _finish(ck, M, art, stdout)


def _finish(ck: Check, M, art: dict, stdout: str) -> dict:
    if M is None:
        return {"correct": False, "certificate": None, "final_answer": None,
                "details": {**ck.details, "verdict": parse_verdict(stdout),
                            "reason": ck.reason()}}
    correct = ck.correct
    cert = None
    fa = None
    if correct:
        row_sq = [sum(v * v for v in row) for row in M]
        fa = {"order": ORDER, "entries": sorted({v for row in M for v in row}),
              "row_norm_sq": row_sq[0] if row_sq else None}
        # 证书不必塞进 784 个数; 给出可独立复核的摘要 + 判定口径
        cert = json.dumps({
            "claim": f"{ORDER} 阶 Hadamard 矩阵",
            "order": ORDER,
            "entries": [-1, 1],
            "row_norm_sq_all": row_sq,
            "gram_checked": f"{ORDER}x{ORDER} = {ORDER * ORDER} entries, all equal to "
                            f"{ORDER} on diagonal and 0 off-diagonal",
            "verdict_rule": "HH^T == 28 I (exact integer arithmetic)",
        }, ensure_ascii=False)
    return {
        "correct": correct,
        "certificate": cert,
        "final_answer": fa,
        "details": {**ck.details, "verdict": parse_verdict(stdout), "reason": ck.reason()},
    }


# ==========================================================================
# 自测
# ==========================================================================
def build_paley_hadamard_28() -> list[list[int]]:
    """自测用的正例: 由 Jacobsthal 矩阵(Border 上一行一列 1)造 28 阶。

    构造(verifier 自己算, 与 agent 无关):
        q = 13 是素数, 二次剩余 Q = {1,3,4,9,10,12}
        S[0][0]=0, S[0][i]=S[i][0]=1, S[i][j]=chi(j-i)   (chi 为 Legendre 符号)
        已知 S S^T = q I_{q+1} = 13 I_14, 且 S 对称
        H = [[ S+I,  S-I ], [ S-I, -S-I ]]  ->  H H^T = 2(q+1) I = 28 I
    这是 Paley 型 II(2(q+1) 阶)。28 = 2*14, 正好对上。
    """
    q = 13
    sq = {x * x % q for x in range(1, q)}

    def chi(a: int) -> int:
        a %= q
        if a == 0:
            return 0
        return 1 if a in sq else -1

    n = q + 1
    S = [[0] * n for _ in range(n)]
    for i in range(1, n):
        S[0][i] = 1
        S[i][0] = 1
        for j in range(1, n):
            S[i][j] = chi(j - i)

    H = [[0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            d = 1 if i == j else 0
            H[i][j] = S[i][j] + d
            H[i][j + n] = S[i][j] - d
            H[i + n][j] = S[i][j] - d
            H[i + n][j + n] = -S[i][j] - d
    return H


def _mk(matrix) -> str:
    return (f"CONSTRUCT: {json.dumps({'kind': 'hadamard', 'order': ORDER, 'matrix': matrix})}\n"
            f"VERDICT: PROVED\n")


def _selftest() -> bool:
    ok = True
    good = build_paley_hadamard_28()
    assert len(good) == ORDER and all(len(r) == ORDER for r in good)

    # 1) 正例
    r = verify(_mk(good), "", "")
    ok &= r["correct"] is True
    print(f"[{'OK' if r['correct'] else 'FAIL'}] 正例(Paley II, q=13) correct={r['correct']} reason={r['details'].get('reason')}")

    # 2) 反例 A: 元素全对, 只把 H[0][0] 取反 —— 表面几乎一样, 但该行不再正交
    bad = [row[:] for row in good]
    bad[0][0] = -bad[0][0]
    r = verify(_mk(bad), "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例A(翻转 H[0][0]) correct={r['correct']} reason={r['details'].get('reason')}")

    # 3) 反例 B: 用 0/2 之类非法元素
    bad2 = [row[:] for row in good]
    bad2[3][7] = 2
    r = verify(_mk(bad2), "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例B(元素 2) correct={r['correct']} reason={r['details'].get('reason')}")

    # 4) 反例 C: 27 阶
    r = verify(_mk([row[:] for row in good][:-1]), "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例C(27 行) correct={r['correct']} reason={r['details'].get('reason')}")

    # 5) 反例 D: 无制品
    r = verify("VERDICT: PROVED\n", "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例D(无制品) correct={r['correct']} reason={r['details'].get('note')}")

    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)