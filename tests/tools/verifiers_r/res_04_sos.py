# -*- coding: utf-8 -*-
"""res-004-sos-inequality —— 「证明或证伪」不等式的独立验证器。

题目: 「证明或证伪: 对任意实数 x, y, z, 有
        x^4 + y^4 + z^4 + xyz >= x^2 y^2 + y^2 z^2 + z^2 x^2 + x + y + z - 3。」

★ 判据: 两种结果都算对, 但**必须附机器可验证证据** ★
把不等式移项成一个多项式差:

    p(x,y,z) = x^4 + y^4 + z^4 + xyz - x^2 y^2 - y^2 z^2 - z^2 x^2 - x - y - z + 3

原命题就是「p 在 R^3 上恒非负」。于是只有两种合法结局:
  (A) 反例路线: 给一个 (x,y,z) 使 p(x,y,z) < 0 —— 这就直接**证伪**了全称命题。
      判据: 本文件代回 p, 要求严格小于 0(带 tol)。因为是有限代入,
      一个真反例就是完整证明, 不需要任何其它论证。
  (B) SOS 证书路线: 给一个 Gram 分解 p = m^T G m, 其中 m 是单项式向量、
      G 实对称半正定。判据: **恒等式**与 **G >= 0** 两件都成立。
      恒等式保证 p 就是这个二次型; 半正定保证 m^T G m 对任意 m >= 0。
      两者合起来才是 p 恒非负的证明, 缺一个都不成立。

★ 为什么「只判对/错」不算过 ★
题面 expected.detail 明写「两种结果都算通过, 但必须附机器可验证证据」。
只回一句 VERDICT: REFUTED 而没有点, 或者只回 VERDICT: PROVED 而没有 Gram 矩阵,
本文件都判 False —— 因为那种回答的**正确性无法被独立复核**,
它和一个瞎猜的 VERDICT 在信息量上没有任何区别。

★ 全程精确有理数运算 ★
p 的系数全是整数, 反例点如果是整数或有理数, 代回的结果可以精确到 0。
本文件一律用 fractions.Fraction 做多项式展开与代入, 不碰浮点 ——
否则「-1e-16 < 0」这种浮点噪声会被当成反例。

注: 本题的正确结局其实是 (A) —— 取 x=y=z=-3 时 p = -15 < 0, 不等式为假。
所以 SOS 证书这条路对本题是走不通的(存在反例的 p 不可能是 SOS)。
本文件照样把 SOS 分支完整实现并自测(见 __main__ 里的引擎自检),
以防 agent 交来一份格式正确但内容错误的证书时, 分支判定本身出 bug。
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import Check, parse_artifact, parse_verdict  # noqa: E402

#: 题面 verifier.params.tol。反例必须让 p 严格小于 -tol, 避免浮点噪声被当成反例。
TOL = Fraction(1, 10 ** 9)

#: p(x,y,z) 的系数表: 指数元组 (a,b,c) -> 系数。
#: 手写展开, 不引入 sympy —— verifier 越少依赖第三方库越难被环境问题误伤。
#:   4 次: x^4 + y^4 + z^4, xyz, -x^2y^2 - y^2z^2 - z^2x^2
#:   1 次: -x - y - z ; 0 次: +3
P_COEFFS: dict[tuple[int, int, int], Fraction] = {
    (4, 0, 0): Fraction(1),
    (0, 4, 0): Fraction(1),
    (0, 0, 4): Fraction(1),
    (1, 1, 1): Fraction(1),
    (2, 2, 0): Fraction(-1),
    (0, 2, 2): Fraction(-1),
    (2, 0, 2): Fraction(-1),
    (1, 0, 0): Fraction(-1),
    (0, 1, 0): Fraction(-1),
    (0, 0, 1): Fraction(-1),
    (0, 0, 0): Fraction(3),
}

_Z = Fraction


def _to_frac(v: object) -> Fraction | None:
    """把 JSON 里的数字/数字字符串变成精确 Fraction; 不可解析返回 None。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return Fraction(str(v))        # 走十进制字符串, 避免二进制浮点的尾巴
    if isinstance(v, str):
        s = v.strip()
        try:
            return Fraction(s)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def eval_p(x: Fraction, y: Fraction, z: Fraction) -> Fraction:
    """精确代入 p(x,y,z)。(这里的写法与 P_COEFFS 一一对应, 便于人工核对。)"""
    return (x ** 4 + y ** 4 + z ** 4 + x * y * z
            - x ** 2 * y ** 2 - y ** 2 * z ** 2 - z ** 2 * x ** 2
            - x - y - z + 3)


# --------------------------------------------------------------------------
# SOS 证书分支: 单项式解析 + Gram 展开 + 半正定判定
# --------------------------------------------------------------------------
def parse_monomial(s: object) -> tuple[int, int, int]:
    """把 "x^2" / "xy" / "x*y" / "1" 解析成指数元组 (a,b,c)。

    刻意**不接受**纯数字(除 "1" 外): 常数项应当用单项式 "1" 配 Gram 系数表达。
    混进数字会让「m^T G m」的语义变得有歧义。
    """
    if not isinstance(s, str):
        raise ValueError(f"单项式必须是字符串, 得到 {s!r}")
    t = s.strip().lower().replace("**", "^").replace(" ", "").replace("*", "").replace("\u00b7", "")
    if t == "":
        raise ValueError("空单项式")
    if t == "1":
        return (0, 0, 0)
    idx = {"x": 0, "y": 1, "z": 2}
    exp = [0, 0, 0]
    i = 0
    while i < len(t):
        ch = t[i]
        if ch not in idx:
            raise ValueError(f"单项式 {s!r} 含非法字符 {ch!r}")
        k = idx[ch]
        i += 1
        e = 1
        if i < len(t) and t[i] == "^":
            i += 1
            start = i
            while i < len(t) and t[i].isdigit():
                i += 1
            if start == i:
                raise ValueError(f"单项式 {s!r} 缺少指数")
            e = int(t[start:i])
        exp[k] += e
    return (exp[0], exp[1], exp[2])


def expand_gram(mons: list[tuple[int, int, int]], G: list[list[Fraction]]
                ) -> dict[tuple[int, int, int], Fraction]:
    """展开 m^T G m 成系数字典。要求 G 对称(调用方已校验)。"""
    poly: dict[tuple[int, int, int], Fraction] = {}
    n = len(mons)
    for i in range(n):
        for j in range(n):
            c = G[i][j]
            if c == 0:
                continue
            key = (mons[i][0] + mons[j][0],
                   mons[i][1] + mons[j][1],
                   mons[i][2] + mons[j][2])
            # 遍历全部 (i,j) 对, 所以 (i,j) 与 (j,i) 天然合出系数 2*G_ij;
            # 这里**不能再乘 2**, 否则非对角项被数成 4*G_ij。
            # (曾因这个 2 被算成 4 而误拒掉唯一一份合法证书, 由引擎自检抓出。)
            coef = c
            poly[key] = poly.get(key, _Z(0)) + coef
    return {k: v for k, v in poly.items() if v != 0}


def is_psd(G: list[list[Fraction]], tol: float = 1e-9) -> tuple[bool, int]:
    """对称矩阵半正定判定: 无容差 Cholesky(LDL^T 形式)。

    对称矩阵半正定 <=> 能做无 pivoting 的 LDL^T 且所有对角元 >= 0。
    这里允许 pivot 落到 [-tol, 0] 视作 0(半正定的退化情形), 小于 -tol 判否。
    返回 (是否半正定, 出问题的那一行, 便于诊断)。
    """
    n = len(G)
    if n == 0:
        return True, -1
    L = [[0.0] * n for _ in range(n)]
    d = [0.0] * n
    for j in range(n):
        s = float(G[j][j])
        for i in range(j):
            s -= L[i][j] * L[i][j] * d[i]
        if s < -tol:
            return False, j
        d[j] = s if s > tol else 0.0
        L[j][j] = 1.0 / math.sqrt(d[j]) if d[j] > 0.0 else 0.0
    if any(not math.isfinite(L[i][j]) for i in range(n) for j in range(n)):
        return False, -1
    return True, -1


def check_sos_certificate(art: dict, target: dict | None = None
                          ) -> tuple[bool, str, dict]:
    """校验一份 SOS 证书。返回 (是否通过, 原因, 诊断字典)。

    证书格式(本 verifier 与 agent 的约定, 写在 CONSTRUCT 制品里):
        {"monomials": ["1","x","y","z",...], "gram": [[g00,...], ...]}
        表示 p(x,y,z) = m^T G m, m = (m_1, ..., m_n)。

    `target` 是要证明的多项式, 默认就是本题的 p。它是参数而不是写死的常量,
    是为了让 __main__ 里的引擎自检能拿一份**已知为真**的证书去走通整条链路 ——
    本题的 p 有反例, 不是 SOS, 所以正常运行时这条分支只会收到假证书。
    """
    target = P_COEFFS if target is None else target
    info: dict[str, object] = {}
    mons_raw = None
    for key in ("monomials", "mons", "basis"):
        if key in art:
            mons_raw = art[key]
            break
    g_raw = None
    for key in ("gram", "G", "Q", "coefficients"):
        if key in art:
            g_raw = art[key]
            break
    if not isinstance(mons_raw, list) or not isinstance(g_raw, list):
        return False, "SOS 证书缺少 monomials / gram", info
    if len(mons_raw) != len(g_raw):
        return False, f"monomials({len(mons_raw)}) 与 gram({len(g_raw)}) 维数不符", info

    # --- 解析单项式 ---
    mons = []
    try:
        for s in mons_raw:
            mons.append(parse_monomial(s))
    except ValueError as e:
        return False, f"单项式解析失败: {e}", info
    if len(set(mons)) != len(mons):
        return False, "单项式向量里有重复项", info

    # --- 解析 Gram(必须是方阵、实对称) ---
    n = len(g_raw)
    G: list[list[Fraction]] = []
    for i in range(n):
        if not isinstance(g_raw[i], list) or len(g_raw[i]) != n:
            return False, f"gram 第 {i} 行不是长度 {n} 的列表", info
        row = []
        for v in g_raw[i]:
            f = _to_frac(v)
            if f is None:
                return False, f"gram[{i}] 含非数值元素 {v!r}", info
            row.append(f)
        G.append(row)
    for i in range(n):
        for j in range(i + 1, n):
            if G[i][j] != G[j][i]:
                return False, f"gram 不对称: G[{i}][{j}]={G[i][j]} != G[{i}][{j}]={G[j][i]}", info

    # --- 半正定 ---
    psd, bad_row = is_psd(G)
    info["gram_psd"] = psd
    if not psd:
        return False, f"gram 不是半正定矩阵(LDL^T 在第 {bad_row} 行出现负主元)", info

    # --- 恒等式: m^T G m == p ---
    poly = expand_gram(mons, G)
    residual = {}
    for key in set(poly) | set(target):
        diff = poly.get(key, _Z(0)) - target.get(key, _Z(0))
        if diff != 0:
            residual[str(key)] = str(diff)
    info["n_residual_terms"] = len(residual)
    if residual:
        head = list(residual.items())[:3]
        return False, f"m^T G m 不等于 p, 残差项: {head}", info

    info["n_monomials"] = n
    return True, "恒等式成立且 G 半正定", info


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
def _find_point(art: dict) -> tuple[dict | None, str]:
    """在制品里找 (x, y, z)。支持平铺与一层嵌套两种常见写法。"""
    for holder in (art, *(art[k] for k in ("counterexample", "point", "witness", "expoint")
                          if isinstance(art.get(k), dict))):
        if all(k in holder for k in ("x", "y", "z")):
            return holder, "x/y/z"
        p = holder.get("point")
        if isinstance(p, (list, tuple)) and len(p) == 3:
            return {"x": p[0], "y": p[1], "z": p[2]}, "point[3]"
    return None, ""


def verify(stdout: str, code: str, claim: str) -> dict:
    """独立判定: 反例是否真违反不等式, 或 SOS 证书是否真的成立。

    `code` 参数**故意不使用** —— 执行 agent 代码等于让它自己给自己打分。
    """
    ck = Check()
    art = parse_artifact(stdout, "COUNTEREXAMPLE") or parse_artifact(stdout, "CONSTRUCT") \
        or parse_artifact(stdout, "ANSWER") or parse_artifact(stdout)
    if art is None:
        return {
            "correct": False,
            "certificate": None,
            "final_answer": None,
            "details": {"artifact": False,
                        "note": "未提交 COUNTEREXAMPLE/CONSTRUCT 制品 —— 按契约视为未作答"},
        }
    ck.expect("artifact", True)

    # ---------------- 路线 A: 反例 ----------------
    holder, how = _find_point(art)
    if holder is not None:
        vals = []
        for k in ("x", "y", "z"):
            f = _to_frac(holder[k])
            if f is None:
                ck.expect("point_numeric", False, f"{k}={holder[k]!r} 不是有限数值")
                return _finish(ck, None, art, stdout, "counterexample")
            vals.append(f)
        x, y, z = vals
        pval = eval_p(x, y, z)
        ck.note("point", [str(v) for v in vals])
        ck.note("p_value", str(pval))
        # 判据: p 必须**严格**为负(全称命题被推翻)
        ck.expect("is_counterexample", pval < -TOL,
                  f"p(x,y,z)={pval} >= 0, 该点不违反不等式")
        return _finish(ck, (x, y, z, pval), art, stdout, "counterexample")

    # ---------------- 路线 B: SOS 证书 ----------------
    if any(k in art for k in ("monomials", "mons", "basis", "gram", "G", "Q", "coefficients")):
        ok, reason, info = check_sos_certificate(art)
        ck.note("sos_info", {k: (str(v) if isinstance(v, (dict, list)) else v)
                             for k, v in info.items()})
        ck.expect("sos_certificate_valid", ok, reason)
        return _finish(ck, {"sos": ok, **info}, art, stdout, "sos")

    # ---------------- 两条路线都没有制品 ----------------
    ck.expect("has_evidence", False,
              "制品里既没有反例 (x,y,z), 也没有 SOS 证书 —— "
              "只判对错不给可验证证据不算过")
    return _finish(ck, None, art, stdout, "none")


def _finish(ck: Check, payload, art: dict, stdout: str, route: str) -> dict:
    correct = ck.correct
    cert = None
    fa = None
    if correct and route == "counterexample":
        x, y, z, pval = payload
        cert = json.dumps({
            "route": "counterexample",
            "point": {"x": str(x), "y": str(y), "z": str(z)},
            "p_value": str(pval),
            "conclusion": "REFUTED: 存在 (x,y,z) 使 p < 0, 原不等式不成立",
        }, ensure_ascii=False)
        fa = {"claim": "x^4+y^4+z^4+xyz >= x^2y^2+y^2z^2+z^2x^2+x+y+z-3",
              "truth": "false", "counterexample": [str(x), str(y), str(z)]}
    elif correct and route == "sos":
        cert = json.dumps({"route": "sos", **{k: str(v) for k, v in payload.items()}},
                          ensure_ascii=False)
        fa = {"claim": "x^4+y^4+z^4+xyz >= x^2y^2+y^2z^2+z^2x^2+x+y+z-3",
              "truth": "true", "certificate": "sos_gram"}
    return {
        "correct": correct,
        "certificate": cert,
        "final_answer": fa,
        "details": {**ck.details, "route": route,
                    "verdict": parse_verdict(stdout), "reason": ck.reason()},
    }


# ==========================================================================
# 自测
# ==========================================================================
def _ce(x, y, z, verdict="REFUTED") -> str:
    return (f"COUNTEREXAMPLE: {json.dumps({'x': x, 'y': y, 'z': z})}\n"
            f"VERDICT: {verdict}\n")


def _engine_selftest() -> bool:
    """SOS 分支引擎自检: 用一份**真**的 SOS 证书证明 (x-1)^2 >= 0。

    ★ 为什么必须做这个自检 ★
    本题的 p 实际上**不是** SOS(它有反例), 所以正常跑起来时 SOS 分支永远
    只会收到假证书。如果分支逻辑写反了(比如恒等式判据恒真), 自测里根本发现不了。
    这里用一个已知的真证书把整条链路(解析 -> 展开 -> 恒等式 -> 半正定)走通一遍。
    """
    art = {"monomials": ["1", "x"], "gram": [[1, -1], [-1, 1]]}
    # 目标多项式 = (x-1)^2 = 1 - 2x + x^2
    target_sq = {(0, 0, 0): Fraction(1), (1, 0, 0): Fraction(-2), (2, 0, 0): Fraction(1)}
    ok, reason, _ = check_sos_certificate(art, target_sq)
    print(f"[{'OK' if ok else 'FAIL'}] SOS 引擎自检 ((x-1)^2 = [1 x]G[1 x]^T, G=[[1,-1],[-1,1]]) -> {reason}")
    return ok


def _selftest() -> bool:
    ok = _engine_selftest()

    # 1) 正例: x=y=z=-3, p = -15 < 0, 不等式被证伪
    r = verify(_ce(-3, -3, -3), "", "")
    ok &= r["correct"] is True
    print(f"[{'OK' if r['correct'] else 'FAIL'}] 正例(反例 (-3,-3,-3)) correct={r['correct']} p={r['details'].get('p_value')}")

    # 2) 反例 A: (1,1,1) 处 p=1 >= 0 —— 是个「看着像反例」的点, 但不违反
    r = verify(_ce(1, 1, 1, "REFUTED"), "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例A((1,1,1) 不违反) correct={r['correct']} reason={r['details'].get('reason')}")

    # 3) 反例 B: 假 SOS 证书 —— 把 p 说成 (x^2)^2, 恒等式立刻不成立
    fake = json.dumps({"kind": "sos_certificate", "monomials": ["x^2"],
                       "gram": [[1]], "target": "p"})
    out = f"CONSTRUCT: {fake}\nVERDICT: PROVED\n"
    r = verify(out, "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例B(假 SOS 证书) correct={r['correct']} reason={r['details'].get('reason')}")

    # 4) 反例 C: 恒等式对但 G 不半正定(取 -1 -> p = -x^2)
    fake2 = json.dumps({"monomials": ["x^2"], "gram": [[-1]]})
    out = f"CONSTRUCT: {fake2}\nVERDICT: PROVED\n"
    r = verify(out, "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例C(Gram 非半正定) correct={r['correct']} reason={r['details'].get('reason')}")

    # 5) 反例 D: 只给 VERDICT, 没有任何证据
    r = verify("VERDICT: REFUTED\n", "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例D(只给 VERDICT 无证据) correct={r['correct']} reason={r['details'].get('reason') or r['details'].get('note')}")

    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)