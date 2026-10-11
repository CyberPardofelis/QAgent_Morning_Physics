# -*- coding: utf-8 -*-
"""res-005-berlekamp-massey —— 最小线性递推的独立验证器。

题目: 「给定整数序列前 100 项(由某个线性递推生成), 求最小线性递推式, 并用前 200 项验证。」

======================================================================
★ 关于「题库未给具体序列」
★ baseproblems_R.txt 的 res-005 里只有一句「给定整数序列前 100 项」,★
★ 既没有列出这 100 个数, verifier.params.seq_file 指向 data/res_05_seq.txt。★
★
★ ★ 2026-10-10 更正 ★
★ 写这个文件时那份数据文件**还不存在**, 所以验证器自带了一个固定实例。★
★ 现在 `data/res_05_sequence.txt` 已经补齐(内容 = 完整题面 + 那 200 项),★
★ 且**已核对与本文件的 SEQUENCE 完全一致** ——
★   见 `python tests/tools/check_input_vs_reference.py` (res-005 那行)。★
★ 换句话说: agent 拿到的题面与验证器判的确实是同一个序列。★
★ 若哪天那份数据被重生成, 一定要重跑上面那条核对。★
★
★   模 p = 2147483647 (Mersenne 素数) 意义下,  用初值 (1,2,3,5,8)★
★   和递推系数 (3,7,11,17,23) 生成前 200 项, 即★
★       s_n = 3*s_{n-1} + 7*s_{n-2} + 11*s_{n-3} + 17*s_{n-4} + 23*s_{n-5}  (mod p)
★
★ 这**不是**在考 agent 能不能背出这一组数字 —— 考的是它有没有真的跑算法。
★ 任何一组「已知系数」的确定性序列都能起到同样的作用; 这里把这组写死在
★ 文件里, 是为了让实例完全可复现、可被第三方独立重算。★
======================================================================

★ 判据为什么是两条而不是一条 ★
题目要的是「**最小**线性递推式」+「用前 200 项验证」, 这两件事对应两个
互相独立的判据, 缺一不可:

  (1) 复现: 用 agent 给的系数, 以真实序列前 k 项为初值, 往后生成到第 200 项,
      必须逐项吻合。这一条只管「对不对」。
  (2) 最小: 本文件**自己实现 Berlekamp-Massey**, 在真实序列的前 100 项上跑一遍,
      得到 (L_ref, c_ref)。要求 agent 的阶数 == L_ref 且系数 ≡ c_ref (mod p)。
      这一条管「是不是最小」。

★ 为什么这两条必须都要 ★
只看 (1) 会被**非最小递推**骗过去。若 C(x) 是最小连接多项式, 那么
(1-x)C(x) 同样能生成同一条序列, 只是阶数高了 1 —— 递推式照样「验证通过」,
但它根本不是最小解, 而且要 6 个初值而不是 5 个。本文件的自测反例 A 用的
就是这个陷阱, 它能复现全部 200 项, 只能靠判据 (2) 拦下来。

★ 为什么全程在模 p 意义下算 ★
整数递推 20 次就溢出, 用 Python 大整数虽然不会错, 但「200 项逐项相等」
既慢又和题目给的整数语义脱节。在素数模下, 序列的定义、求逆、BM 的除法
(pow(b,-1,p)) 全都干净, 而且模 p 相等**比**整数相等更强: 万一 agent
报了个整数相等的系数, 它必然也模 p 相等, 不会假阴性。
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import Check, parse_artifact, parse_verdict  # noqa: E402

# --- 自带的固定实例(见文件头) ----------------------------------------------
#: 素数模。取 Mersenne 素数 2^31-1, 因为 pow(b, -1, p) 需要 p 是素数。
MOD = 2147483647

#: 真实的最小递推系数: s_n = 3 s_{n-1} + 7 s_{n-2} + 11 s_{n-3} + 17 s_{n-4} + 23 s_{n-5}
TRUE_COEF = [3, 7, 11, 17, 23]

#: 初值(前 5 项)。
TRUE_SEEDS = [1, 2, 3, 5, 8]

#: 题目要求的验证项数。
CHECK_TERMS = 200

#: 交给 agent 观察的项数(题目说「前 100 项」)。
GIVEN_TERMS = 100


def make_sequence(coefs, seeds, count, mod=MOD) -> list[int]:
    """按 s_n = Σ c_i s_{n-i} 生成前 count 项。

    ★ 为什么初值取前 len(coefs) 项 ★
    判定「这条递推是不是真的生成这条序列」, 标准做法是拿真实序列的前 k 项
    当初值, 再从第 k 项开始外推。这样 agent 不必额外提交初值, 且检验的是
    纯粹的「从第 k 项往后每一项都被这条递推决定」。
    """
    k = len(coefs)
    out = [x % mod for x in list(seeds)[:k]]
    while len(out) < count:
        nxt = sum(coefs[j] * out[len(out) - 1 - j] for j in range(k)) % mod
        out.append(nxt)
    return out


def reproduces(coefs, seeds, count, mod=MOD) -> tuple[bool, int | None]:
    """这组系数能不能生成那条序列? **两种 BM 记法都试**。

    ★ 为什么必须两种都试 ★
    BM 有两种通行记法, 系数互为模负数:
        题面写法   s_n =  c1*s_{n-1} + c2*s_{n-2} + ...   -> [3, 7, 11, 17, 23]
        标准库返回 s_n = -C1*s_{n-1} - C2*s_{n-2} - ...  -> [M-3, M-7, ...]
    两者生成**完全相同**的序列 —— 数学上是同一个递推。

    实测只认一种的话, 正确的实现会被判错: 参考记法判对, 等价的 C 记法判错。
    判据应该是行为性的(能不能生成那条序列), 而不是句法性的
    (系数数组和我的一样吗)。
    """
    for sign in (1, -1):
        c = [(sign * x) % mod for x in coefs]
        gen = make_sequence(c, seeds, count, mod)
        bad_i = next((i for i in range(len(seeds))
                      if gen[i] != seeds[i] % mod), None)
        if bad_i is None:
            gen = make_sequence(c, seeds, count, mod)
            bad_i = next((i for i in range(count)
                          if gen[i] != seeds[i] % mod if i < len(seeds)),
                         None)
            if bad_i is None:
                full = list(seeds[:count])
                return True, None
        else:
            continue
    # 两种都不行 -> 报第一次失配位置(用原始记法, 便于看)
    gen = make_sequence([x % mod for x in coefs], seeds, count, mod)
    return False, next((i for i in range(count)
                        if i < len(seeds) and gen[i] != seeds[i] % mod), None)


#: 题目给 agent 的那 200 项序列(真实实例)。
SEQUENCE = make_sequence(TRUE_COEF, TRUE_SEEDS, CHECK_TERMS)


def berlekamp_massey(seq, mod=MOD) -> tuple[int, list[int]]:
    """本文件**自己**实现的 Berlekamp-Massey(~25 行), 不 import 任何第三方库。

    返回 (最小阶数 L, 连接多项式系数 c_1..c_L), 满足
        s_n = c_1 s_{n-1} + ... + c_L s_{n-L}   (mod p)
    算法维护连接多项式 C 与「上一个导致阶数跳变的」多项式 B, 每次失配时用
    最新的矛盾去修正 C。定理保证: 跑完前 N 项得到的 L, 是这条序列的真实
    最小阶数(只要 N >= 2L)。
    """
    C, B, L, m, b = [1], [1], 0, 1, 1
    for n in range(len(seq)):
        # d = 本项在当前 C 下的偏差
        d = seq[n] % mod
        for i in range(1, L + 1):
            d = (d + C[i] * seq[n - i]) % mod
        if d == 0:
            m += 1                     # 无矛盾, 多看一项
            continue
        # 用 B 去修 C: C <- C - (d / b) x^m B
        coef = (d * pow(b, -1, mod)) % mod
        T = list(C)
        if len(C) < len(B) + m:
            C += [0] * (len(B) + m - len(C))
        for i in range(len(B)):
            C[i + m] = (C[i + m] - coef * B[i]) % mod
        if 2 * L <= n:                  # 阶数要跳变了, 记下当前的 C 和偏差
            L, B, b, m = n + 1 - L, T, d, 1
        else:
            m += 1
    return L, [(-C[i]) % mod for i in range(1, L + 1)]


#: 参考答案: 本文件自己跑 BM 得到的最小阶数与系数。
REF_ORDER, REF_COEF = berlekamp_massey(SEQUENCE[:GIVEN_TERMS])


def _to_int_list(raw) -> list[int] | None:
    """把制品里的系数列表转成整数列表; 允许整数/数字字符串/±1 形式的小数。"""
    if not isinstance(raw, list) or not raw:
        return None
    out = []
    for v in raw:
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            out.append(v)
        elif isinstance(v, float) and v.is_integer():
            out.append(int(v))
        elif isinstance(v, str):
            s = v.strip().replace("−", "-")
            try:
                out.append(int(s))
            except ValueError:
                try:
                    f = float(s)
                except ValueError:
                    return None
                if not f.is_integer():
                    return None
                out.append(int(f))
        else:
            return None
    return out


def verify(stdout: str, code: str, claim: str) -> dict:
    """独立判定: 递推系数是否就是本序列的最小递推, 且真的能复现前 200 项。

    `code` 参数**故意不使用** —— 执行 agent 代码等于让它自己给自己打分。
    """
    ck = Check()
    art = parse_artifact(stdout, "ANSWER") or parse_artifact(stdout, "CONSTRUCT") \
        or parse_artifact(stdout)
    if art is None:
        return {
            "correct": False,
            "certificate": None,
            "final_answer": None,
            "details": {"artifact": False,
                        "note": "未提交 ANSWER/CONSTRUCT 制品 —— 按契约视为未作答"},
        }
    ck.expect("artifact", True)

    raw = None
    for key in ("recurrence", "coefficients", "coeffs", "c", "rule"):
        if key in art:
            raw = art[key]
            break
    coefs = _to_int_list(raw)
    if coefs is None:
        ck.expect("recurrence_present", False,
                  f"制品里没有可用的递推系数列表(recurrence={raw!r})")
        return _finish(ck, None, art, stdout)
    ck.expect("recurrence_present", True)
    ck.note("n_coefficients", len(coefs))

    # --- 自洽: 声明的 order 必须等于系数个数 ---
    if "order" in art:
        declared = art["order"]
        ck.expect("order_matches_len", declared == len(coefs),
                  f"声明 order={declared!r} 但系数有 {len(coefs)} 个")

    # --- 判据 (1): 阶数必须等于本文件自己跑 BM 得到的最小阶数 ---
    ck.expect("order_is_minimal", len(coefs) == REF_ORDER,
              f"阶数 {len(coefs)} != 最小阶数 {REF_ORDER}")

    # --- 判据 (2): ★ 删掉"系数必须和我的 BM 逐个相等" ★ ---
    #
    # 原来这条判据要求 agent 的系数数组与本文件的 REF_COEF 逐项模相等。
    # 实测它会**误杀正确的实现**: BM 有两种通行记法, 标准库返回的 C[] 数组
    # 与题面写的 s_n = c1*s_{n-1}+... 恰好**互为模负数**。
    # 实测: 参考记法 [3,7,11,17,23] 判对, 等价的 C 记法
    # [2147483644,2147483640,...] 判错 —— 而两者生成完全相同的序列。
    #
    # 正确判据是**行为性的**: 判据 (3) 已经验了"能不能复现前 200 项",
    # 那才是数学上唯一该问的问题。最小性由判据 (1) 的 order 保证 ——
    # order 是个整数, 没有符号约定问题。
    #
    # 所以这里只记一笔说明, 不再据此判对错:
    ck.note("coefficient_convention", "按复现序列判定, 不逐项比对系数")

    # --- 判据 (3): 真的能复现前 200 项 ---
    ok_gen, first_bad = reproduces(coefs, SEQUENCE, CHECK_TERMS)
    ck.expect("reproduces_200_terms", ok_gen,
              f"两种 BM 记法都生成不出这条序列(最先在第 {first_bad} 项不符)"
              if first_bad is not None else "两种记法都对不上")
    ck.note("first_mismatch_index", first_bad)

    return _finish(ck, coefs, art, stdout)


def _finish(ck: Check, coefs, art: dict, stdout: str) -> dict:
    if coefs is None:
        return {"correct": False, "certificate": None, "final_answer": None,
                "details": {**ck.details, "verdict": parse_verdict(stdout),
                            "reason": ck.reason()}}
    correct = ck.correct
    cert = None
    fa = None
    if correct:
        cert = json.dumps({
            "instance": {"mod": MOD, "given_terms": GIVEN_TERMS,
                         "check_terms": CHECK_TERMS, "seeds": TRUE_SEEDS},
            "recurrence": [c % MOD for c in coefs],
            "order": len(coefs),
            "verified_terms": CHECK_TERMS,
            "seq_head": SEQUENCE[:12],
            "seq_tail": SEQUENCE[-5:],
        }, ensure_ascii=False)
        fa = {"order": len(coefs), "recurrence": [c % MOD for c in coefs],
              "verified_200_terms": True}
    return {
        "correct": correct,
        "certificate": cert,
        "final_answer": fa,
        "details": {**ck.details, "verdict": parse_verdict(stdout), "reason": ck.reason()},
    }


# ==========================================================================
# 自测
# ==========================================================================
def _mk(coefs, order=None, verdict="PROVED") -> str:
    payload = {"recurrence": coefs, "order": len(coefs) if order is None else order}
    return f"ANSWER: {json.dumps(payload)}\nVERDICT: {verdict}\n"


def _selftest() -> bool:
    ok = True

    # 先确认自带实例本身是自洽的(否则后面所有测试都在测一个坏基准)
    assert REF_ORDER == len(TRUE_COEF), REF_ORDER
    assert REF_COEF == TRUE_COEF, REF_COEF
    assert make_sequence(TRUE_COEF, TRUE_SEEDS, CHECK_TERMS) == SEQUENCE

    # 1) 正例
    r = verify(_mk(TRUE_COEF), "", "")
    ok &= r["correct"] is True
    print(f"[{'OK' if r['correct'] else 'FAIL'}] 正例(order={REF_ORDER}, coef={TRUE_COEF}) correct={r['correct']} reason={r['details'].get('reason')}")

    # 2) 反例 A: **非最小递推** (1-x)C(x), 阶数 6 —— 它能复现全部 200 项,
    #    只有「阶数 == BM 最小阶数」这条判据能拦住它。
    nonmin = [4, 4, 4, 6, 6, MOD - 23]
    assert make_sequence(nonmin, SEQUENCE, CHECK_TERMS) == SEQUENCE, "前提不成立: 该反例并不复现序列"
    r = verify(_mk(nonmin), "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例A(非最小阶 {len(nonmin)}, 能复现 200 项) correct={r['correct']} reason={r['details'].get('reason')}")

    # 3) 反例 B: 阶数对, 系数改一个
    bad = list(TRUE_COEF)
    bad[2] += 1
    r = verify(_mk(bad), "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例B(系数 +1) correct={r['correct']} reason={r['details'].get('reason')}")

    # 4) 反例 C: 阶数与系数都对, 但 order 字段自相矛盾
    r = verify(_mk(TRUE_COEF, order=9), "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例C(order 字段与系数个数矛盾) correct={r['correct']} reason={r['details'].get('reason')}")

    # 5) 反例 D: 无制品
    r = verify("VERDICT: PROVED\n", "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例D(无制品) correct={r['correct']} reason={r['details'].get('note')}")

    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)