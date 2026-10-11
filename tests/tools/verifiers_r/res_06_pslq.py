"""res-006-pslq-integer-relation 的独立 verifier。

题目: 找整数 a,b,c,d 使 a*log2+b*log3+c*log5+d*log7 ≈ 0, (a,b,c,d) 不全为零,
每个 |系数| <= 10^6。题库 pass_condition 写的是 residual < 1e-50。

★ 为什么题目里的 1e-50 达不到(verifier 自己算过)★
2,3,5,7 是**乘法无关**的(互不相同的素数), 由唯一分解定理:
    2^p 3^q 5^r 7^s = 1  =>  p=q=r=s=0
所以 log2,log3,log5,log7 在 Q 上**线性无关** —— 不存在残差恰好为 0 的
非零整数四元组。PSLQ 只能找"接近 0"的组合, 越接近越难。
verifier 用 dps=200、maxcoeff=10^6 跑自己的 PSLQ, 得到的最好残差约
    [83485, -398692, 582638, -286539] -> |residual| ≈ 9.4e-19
★ 所以判据不能是 1e-50(那会把真正的最优解也判错), 也不能是 1e-30
(同样比可达的最优值严 11 个数量级)★

★ 判据 ★
  TOL = max(1e-30, 100 * best_residual_pslq)   ≈ 9.4e-17
  为什么这样判就说明答案是对的:
   - 100 倍松弛仍然比"随便一个系数组合"小 16 个数量级(随机组合的残差量级
     是 0.5~2, 即 10^0); 也就是说过的必须是**结构性的**近零关系。
   - 100 倍松弛足以容纳不同精度/不同 PSLQ 实现找到的等价最优解(它们残差
     同量级但系数可以差几倍), 不会因实现细节误杀真解。
   - best_residual 由 verifier **自己跑 PSLQ** 得到, 不信 agent 报的数。
  另加硬判据: 系数是整数、不全为零、|系数| <= 10^6。

★ verifier 不 import agent 的任何代码 ★ 只读 ANSWER 制品里的四个系数。
"""
from __future__ import annotations

import json
import os
import sys

try:
    from .common import Check, parse_artifact, parse_verdict
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import Check, parse_artifact, parse_verdict

from mpmath import mp, log, mpf, pslq  # mpmath 是独立实现, 不涉及 agent 代码

#: 题目要求的系数上界
COEFF_BOUND = 10 ** 6
#: 相对"随机系数组合残差量级(~1)"仍算近零的硬门槛
ABS_FLOOR_TOL = 1e-30
#: PSLQ 独立跑出来的最优残差允许多少倍松弛
RELAX = 100.0
#: 自己算 PSLQ 用的工作精度(题目 params 写 120, 这里给足余量)
PSLQ_DPS = 200

_REF_CACHE: dict[str, float] = {}
_REL_CACHE: dict[str, tuple[list[int] | None, float]] = {}


def _logs(dps: int = PSLQ_DPS) -> list[mpf]:
    """自己算四个对数。高精度必须给足: log 的舍入会直接污染残差。"""
    with mp.workdps(dps):
        return [mpf(log(p)) for p in (2, 3, 5, 7)]


def _lincomb(coeffs, dps: int = PSLQ_DPS) -> mpf:
    """
    在 dps 位精度下算 a*log2+b*log3+c*log5+d*log7。

    ★ 整个算式都必须在 workdps 上下文里算 ★
    这是本题最容易翻车的实现细节: 系数 x log 的量级到 1e5~1e6, 四个大数
    相消, 15 位默认精度下的**绝对**舍入误差就有 1e-10 量级 —— 足以把
    一个真实的 1e-19 残差直接抹成 0(看起来"通过", 实际是精度不够)。
    所以: 高精度下算, 再与低精度结果比对量级, 两条一起当判据。
    """
    with mp.workdps(dps):
        xs = [mpf(log(p)) for p in (2, 3, 5, 7)]
        return sum(mpf(c) * x for c, x in zip(coeffs, xs))


def residual(coeffs) -> tuple[float, float, float]:
    """
    返回 (高精度残差绝对值, 带符号残差, 低精度残差绝对值)。

    ★ 用 200 位和 120 位两个精度各算一遍 ★
    为什么: 真的近零关系在两个精度下给出的量级应当一致(都受限于系数与
    实数对数的固有接近程度); 而"靠某个精度下的浮点巧合凑出来的"假关系,
    精度一变残差就放大几个数量级。所以
        残差_低精度 <= 1e6 * 残差_高精度
    是一条独立于"残差绝对值够小"的判据, 用来堵住高精度的假零。
    """
    r1 = _lincomb(coeffs, PSLQ_DPS)
    r2 = _lincomb(coeffs, 120)
    return float(abs(r1)), float(r1), float(abs(r2))


def best_relation(maxcoeff: int = COEFF_BOUND) -> tuple[list[int] | None, float]:
    """
    verifier **自己**跑一遍 PSLQ, 返回 (系数四元组, 残差)。

    为什么必须自己跑: 容差基准若采信 agent 报的数, agent 只要报一个
    极小的 residual 就能把门槛压到自己那条烂关系之上 —— 这是"自己出题
    自己判卷"的典型漏洞。

    ★ 整段都在 workdps 里 ★ 默认 15 位精度下 PSLQ 会返回一条"看着像
    关系、其实只是舍入巧合"的四元组(实测: 默认精度给
    [80102,-4819,56793,-72785], 真残差 2.2e-12; 200 位下给
    [83485,-398692,582638,-286539], 真残差 9.4e-19)。
    """
    key = f"rel:{maxcoeff}:{PSLQ_DPS}"
    if key in _REL_CACHE:
        return _REL_CACHE[key]
    with mp.workdps(PSLQ_DPS):
        xs = [mpf(log(p)) for p in (2, 3, 5, 7)]
        rel = pslq(xs, maxcoeff=maxcoeff, maxsteps=10 ** 6, tol=mpf("1e-18"))
        if rel:
            val = float(abs(sum(mpf(c) * x for c, x in zip(rel, xs))))
            rel = [int(c) for c in rel]
        else:
            val = float("inf")
    _REL_CACHE[key] = (rel, val)
    return _REL_CACHE[key]


def best_residual_pslq(maxcoeff: int = COEFF_BOUND) -> float:
    """见 best_relation —— 只要残差那一半, 作为容差基准。"""
    key = f"{maxcoeff}:{PSLQ_DPS}"
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    val = best_relation(maxcoeff)[1]
    _REF_CACHE[key] = val
    return val


def _coerce_int_list(raw) -> list[int] | None:
    """把制品里的 relation 规整成 4 个整数(容忍 "[1, 2, 3, 4]" 这种字符串写法)。"""
    if isinstance(raw, str):
        s = raw.strip().strip("[]()")
        parts = [p for p in s.replace(";", ",").split(",") if p.strip()]
        try:
            raw = [int(float(p)) for p in parts]
        except ValueError:
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    out: list[int] = []
    for v in raw:
        if isinstance(v, bool):
            return None
        if isinstance(v, float):
            if not v.is_integer():
                return None
            v = int(v)
        if isinstance(v, str):
            try:
                f = float(v.strip())
            except ValueError:
                return None
            if not f.is_integer():
                return None
            v = int(f)
        if not isinstance(v, int):
            return None
        out.append(v)
    return out


def verify(stdout: str, code: str, claim: str) -> dict:
    chk = Check()
    chk.note("verdict", parse_verdict(stdout))

    art = parse_artifact(stdout, "ANSWER") or parse_artifact(stdout, "CONSTRUCT")
    if not chk.expect("artifact_present", art is not None,
                      "未提交 ANSWWER 制品(需要 {\"relation\": [a, b, c, d]})"):
        return _finish(chk, None, {})
    assert art is not None

    raw = None
    for key in ("relation", "coeffs", "coefficients", "relation_vector", "solution"):
        if art.get(key) is not None:
            raw = art[key]
            break
    coeffs = _coerce_int_list(raw)
    if not chk.expect("relation_parsed", coeffs is not None,
                      f"relation 解析失败/不是 4 个整数: {raw!r}"):
        return _finish(chk, art, {})
    assert coeffs is not None

    # ---- 硬判据 1: 不全为零 ----
    chk.expect("not_all_zero", any(c != 0 for c in coeffs),
               "(a,b,c,d) 全为 0 —— 这不是任何意义上的'整数关系'")

    # ---- 硬判据 2: 系数上界 ----
    chk.expect("coeff_bound", all(abs(c) <= COEFF_BOUND for c in coeffs),
               f"存在 |系数| > {COEFF_BOUND}: {coeffs}")

    # ---- 硬判据 3: 残差真的接近零 ----
    abs_res, signed, abs_res_120 = residual(coeffs)
    best = best_residual_pslq()
    tol = max(ABS_FLOOR_TOL, RELAX * best if best not in (0.0,) else ABS_FLOOR_TOL)
    chk.note("residual", abs_res)
    chk.note("residual_signed", signed)
    chk.note("residual_low_precision", abs_res_120)
    chk.note("pslq_reference_best_residual", best)
    chk.note("tolerance", tol)
    chk.note("relation", coeffs)
    # 为什么这个判据能证伪"随便一个数": 随机四元组的残差量级是 10^0,
    # 而门槛在 1e-16 附近 —— 差 16 个数量级, 只有结构性近零关系能过。
    chk.note("random_baseline_magnitude", "~1 (e.g. 1*log2+2*log3+3*log5+4*log7)")
    chk.expect("residual_small", abs_res <= tol,
               f"残差 {abs_res:.3e} > 容差 {tol:.3e} "
               f"(PSLQ 在 |系数|<=1e6 下的最好残差约 {best:.3e})")
    # 高精度 vs 低精度的量级一致性: 堵住"靠单精度浮点巧合凑出的假零"
    chk.expect("residual_precision_stable",
               abs_res_120 <= 1e6 * max(abs_res, mpf(1) * 1e-300) or abs_res == 0.0,
               f"残差随精度塌陷: 120 位给 {abs_res_120:.3e}, "
               f"200 位给 {abs_res:.3e} —— 高精度下的零是假关系")

    return _finish(chk, art, {"relation": coeffs, "residual": abs_res,
                              "tolerance": tol})


def _finish(chk: Check, art: dict | None, extra: dict) -> dict:
    correct = chk.correct
    if correct:
        cert = (f"整数关系 {extra['relation']}, verifier 高精度复算残差 "
                f"{extra['residual']:.3e} <= 容差 {extra['tolerance']:.3e}")
    else:
        cert = chk.reason()
    final = json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else None
    return {"correct": correct, "certificate": cert, "final_answer": final,
            "details": dict(chk.details)}


if __name__ == "__main__":
    # 独立算一次 PSLQ, 打印真实量级 —— 判据的容差基准就来自这里
    best = best_residual_pslq()
    print(f"[INFO] verifier 自跑 PSLQ(dps={PSLQ_DPS}, maxcoeff=1e6) "
          f"最好残差 = {best:.6e}")
    print(f"[INFO] 判定容差 = max(1e-30, 100*{best:.3e}) = "
          f"{max(ABS_FLOOR_TOL, RELAX * best):.6e}")
    print(f"[INFO] 题库写的 1e-50 不可达: log2,log3,log5,log7 在 Q 上线性无关")

    # ---- 正例: verifier 自己跑出来的最优关系 ----
    ref, ref_res = best_relation()
    assert ref, "PSLQ 应能找到关系"
    r = verify("VERDICT: PROVED\nANSWER: " +
               json.dumps({"relation": ref}) + "\n", "", "")
    assert r["correct"], f"正例应通过: {r}"
    print(f"[OK] 正例(PSLQ 最优关系 {ref}) -> {r['certificate']}")

    # ---- 反例 A: 随便一组系数(残差量级 ~10) ----
    r = verify("ANSWER: " + json.dumps({"relation": [1, 2, 3, 4]}) + "\n", "", "")
    assert not r["correct"] and not r["details"]["residual_small"], r
    print("[OK] 反例A(随便一组系数) ->", r["certificate"])

    # ---- 反例 B: 全零 ----
    r = verify("ANSWER: " + json.dumps({"relation": [0, 0, 0, 0]}) + "\n", "", "")
    assert not r["correct"] and not r["details"]["not_all_zero"], r
    print("[OK] 反例B(全零) ->", r["certificate"])

    # ---- 反例 C: 超出系数上界 ----
    r = verify("ANSWER: " +
               json.dumps({"relation": [ref[0] * 10 ** 9, 1, 0, 0]}) + "\n", "", "")
    assert not r["correct"] and not r["details"]["coeff_bound"], r
    print("[OK] 反例C(系数越界) ->", r["certificate"])

    print("[OK] 全部自测通过")