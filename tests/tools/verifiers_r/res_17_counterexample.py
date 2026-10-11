"""
res-017-counterexample-hunt 的独立 verifier。

题目: 对猜想 "对任意 n>=2, n^2+n+1 无平方因子", 搜索 1e6 以内反例。

★ 制品格式: 题库 expected.value 声明的字段是**唯一契约** ★
    COUNTEREXAMPLE: {"verdict": "false",
                     "counterexample": "n=18, 18^2+18+1=343, 平方因子 49"}   ← 字符串
    COUNTEREXAMPLE: {"verdict": "false", "n": 18, "value": 343,
                     "square_factor": 49, "searched_range": [2, 1000000]}    ← 等价 dict
``counterexample`` 是题库声明的**字符串**形状; :func:`_shape_counterexample`
把它(或 dict)统一成 n/value/square_factor 三个数, 之后照旧由 verifier
**自己重算**逐项对账 —— 放宽的是解析, 不是判定。

★ 这个猜想是假的 ★
  n=18 -> 18^2+18+1 = 343 = 7^3, 含平方因子 49 (=7^2)。
  我在自测里独立跑过最小性: n=2..17 全部无平方因子, n=18 是最小反例。

所以 verifier 的判据很硬, 而且**全部自己算**:
  1. verdict 必须是 REFUTED —— 报 PROVED / INCONCLUSIVE 判错(猜想确实可证伪);
  2. 报的反例 n 必须在 [2, 1e6];
  3. 我自己算 v = n^2+n+1, 与 agent 报的 value 对账;
  4. 我自己找平方因子(试除法, 不用 agent 的分解, 也尽量不用 sympy 当黑箱):
     存在质数 p 使 p^2 | v。因为若 d^2|v 且 d 复合, 取 d 的任一质因子 p,
     则 p^2 | d^2 | v —— 所以"含平方因子" <=> "含某个质数的平方", 试除即完备;
  5. agent 声称的搜索范围必须**自洽**(范围里要装得下它自己给的反例)。

为什么这么判就能证明答案是对的:
  判据 3+4 是一个纯粹的整除事实, 与 agent 的搜索代码无关;
  agent 的代码即使有 bug / 只搜了 100 个数, 只要它报的 n 真的满足
  n^2+n+1 被某个 p^2 整除, 它的**结论**就是对的。verifier 验证结论, 不验证过程。
  反过来, 如果 agent 谎报"搜到 1e6 都没找到", 判据 1 直接判错 —— 因为
  本题要求的不是"它搜到没搜到", 而是"真相是什么"。
"""
from __future__ import annotations

import json
import os
import re
import sys
from math import isqrt
from typing import Any

try:                                    # 作为包被导入
    from .common import Check, parse_artifact, parse_verdict
except ImportError:                     # 直接 python res_17_counterexample.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import Check, parse_artifact, parse_verdict

#: 题目规定的搜索上界(题库 verifier.params.search_bound)
SEARCH_BOUND = 1_000_000
#: 猜想的前件: n >= 2
N_MIN = 2

#: 自测/诊断用: 最小性只在这个窗口里核验(窗口内必已含真反例 18)
MINIMALITY_SCAN = 500


# --------------------------------------------------------------------------
# 纯整数算术: 不依赖 sympy, 保证判据不依赖任何外部"黑箱"
# --------------------------------------------------------------------------
def _primes_upto(n: int) -> list[int]:
    """埃氏筛。"""
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i:n + 1:i] = b"\x00" * ((n - i * i) // i + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def square_divisor(v: int) -> int | None:
    """返回最小的 d>=2 使 d^2 | v; 没有则 None。

    只试质数的平方是完备的(见模块 docstring 的证明)。
    """
    if v < 4:
        return None
    for p in _primes_upto(isqrt(v)):
        pp = p * p
        if pp > v:
            break
        if v % pp == 0:
            return pp
    return None


def smallest_counterexample(lo: int = N_MIN, hi: int = MINIMALITY_SCAN) -> int | None:
    """扫描出最小的反例 n(独立算, 用来核对 agent 的"最小"说法)。"""
    for n in range(lo, hi + 1):
        if square_divisor(n * n + n + 1) is not None:
            return n
    return None


# --------------------------------------------------------------------------
# 制品字段的容错解析
# --------------------------------------------------------------------------
def _first_key(d: dict, *names) -> Any:
    for k in names:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _as_int(x: Any) -> int | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return int(s)
        except ValueError:
            return None
    return None


def _parse_range(x: Any) -> tuple[int, int] | None:
    """把 [2, 1e6] / 1e6 / {"from":2,"to":1000000} / {"max":...} 统一成 (lo, hi)。"""
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return (N_MIN, int(x))
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        lo, hi = _as_int(x[0]), _as_int(x[1])
        if lo is not None and hi is not None:
            return (lo, hi)
        return None
    if isinstance(x, dict):
        lo = _as_int(_first_key(x, "from", "lo", "min", "start", "lower"))
        hi = _as_int(_first_key(x, "to", "hi", "max", "end", "upper"))
        if lo is not None and hi is not None:
            return (lo, hi)
    return None


def _verdict_says_false(verdict: str | None, art: dict | None) -> bool:
    """stdout 的 VERDICT 或制品里的 verdict 字段, 任一说"假"即可。"""
    if verdict and verdict.lower() in {"refuted", "false", "disproved"}:
        return True
    if art:
        raw = _first_key(art, "verdict", "result", "status")
        if isinstance(raw, str) and raw.strip().lower() in {
                "refuted", "false", "disproved", "counterexample_found", "falsified"}:
            return True
        if raw is False:                       # JSON 里的 false
            return True
    return False


#: 题库契约字段 counterexample 是字符串("n 值 + 平方因子分解"), 所以要能从
#: 一句人话里把 n / 值 / 平方因子抠出来。**只抠, 不采信** —— 抠出来的每个数
#: 都要在 verify() 里被独立重算对账。
_N_IN_TEXT = re.compile(
    r"(?:n\s*=\s*|取\s*n\s*=\s*)(\d{1,7})", re.I)
_VALUE_IN_TEXT = re.compile(
    r"(\d{1,20})\s*=\s*(?:\d+\s*\^\s*\d\s*[+]\s*\d\s*[+]\s*1|n\^2\s*\+\s*n\s*\+\s*1"
    r"|n\s*\*\s*n\s*\+\s*n\s*\+\s*1)", re.I)
#: 写法二: 表达式在左, 值在右 —— "18^2+18+1=342"
_VALUE_IN_TEXT2 = re.compile(
    r"(?:\d{1,10}\s*\^\s*\d\s*[+]\s*\d+\s*[+]\s*1|n\^2\s*\+\s*n\s*\+\s*1)\s*=\s*"
    r"(\d{1,20})", re.I)
_SQ_IN_TEXT = re.compile(
    r"(?:平方因子|平方数因子|square\s*(?:factor|divisor)|d\s*=\s*|=\s*)(\d{1,12})"
    r"\s*(?:\^|\*\*)?\s*\|\s*(?:\d+)", re.I)


def _shape_counterexample(raw: Any) -> dict | None:
    """把契约字段 counterexample 统一成 {"n":..,"value":..,"square_factor":..}。

    dict 直接取; 字符串用正则抠。抠不到任何东西返回 None（那是"字段给了但内容
    没法用"，由 counterexample_valid 判错, 不是字段名问题）。
    """
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return {"n": raw}
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.replace("−", "-")
    out: dict[str, Any] = {}
    m = _N_IN_TEXT.search(text)
    if m:
        out["n"] = int(m.group(1))
    m = _VALUE_IN_TEXT.search(text) or _VALUE_IN_TEXT2.search(text)
    if m:
        out["value"] = int(m.group(1))
    m = _SQ_IN_TEXT.search(text)
    if m:
        out["square_factor"] = int(m.group(1))
    return out or None


# --------------------------------------------------------------------------
def verify(stdout: str, code: str, claim: str) -> dict:
    chk = Check()

    art = (parse_artifact(stdout, "COUNTEREXAMPLE")
           or parse_artifact(stdout, "ANSWER")
           or parse_artifact(stdout))
    if not chk.expect("artifact_present", art is not None,
                      "未提交 COUNTEREXAMPLE/ANSWER 制品行"):
        return _finish(chk, None, {})
    assert art is not None

    verdict = parse_verdict(stdout)
    chk.note("stdout_verdict", verdict)

    # ---- 1. 结论方向: 猜想是假的, 必须判 refuted ----
    chk.expect("verdict_is_refuted", _verdict_says_false(verdict, art),
               f"猜想可证伪(n=18 时 343=7^3 含平方因子 49), "
               f"verdict={verdict!r} 必须为 refuted/false")

    # ---- 2. 反例 n ----
    # ★ 题库契约字段 counterexample 是一个**字符串**("n 值 + 平方因子分解") ★
    # 所以两种形状都收: dict(如 {"n":18,"square_factor":49}) 或
    # 字符串(如 "n=18, 18^2+18+1=343=7^3, 平方因子 49")。
    # 字符串按 :func:`_parse_counterexample_text` 正则抠出 n/value/square_factor,
    # 抠完照样交给下面第 3、4 条**自己算**去对账 —— 放宽的是解析, 不是判定。
    raw_ce = _first_key(art, "counterexample", "counterexample_desc", "witness")
    shaped = _shape_counterexample(raw_ce)
    # 字段认到了没有（字段名层面的事, 与内容能不能用分开）:
    # 契约字段 counterexample 出现且非空, 或结构化的 n 字段出现
    has_ce_field = (isinstance(raw_ce, (str, dict, int))
                    and not (isinstance(raw_ce, str) and not raw_ce.strip()))
    chk.expect("counterexample_read",
               has_ce_field or _first_key(art, "n", "counterexample_n", "value_n") is not None,
               "制品里没有反例: 题库契约字段是 counterexample"
               "(字符串' n 值 + 平方因子分解' 或 dict), n 字段也接受")

    fields: dict[str, Any] = dict(shaped or {})
    for k, names in (("n", ("n", "counterexample_n", "value_n")),
                     ("value", ("value", "n2_plus_n_plus_1", "num")),
                     ("square_factor", ("square_factor", "square_divisor",
                                        "divisor", "d", "factor"))):
        if fields.get(k) is None:
            fields[k] = _first_key(art, *names)
    chk.note("counterexample_shape", sorted(shaped.keys()) if shaped else None)

    n = _as_int(fields.get("n"))
    if not chk.expect("counterexample_valid", n is not None,
                      f"从反例里读不出整数 n（读到 {fields.get('n')!r}）; "
                      "字符串写法示例: 'n=18, 18^2+18+1=343, 平方因子 49'"):
        return _finish(chk, art, {})

    chk.expect("n_in_domain", n >= N_MIN, f"n={n} 不在前件 n>={N_MIN} 内")
    chk.expect("n_within_search_bound", n <= SEARCH_BOUND,
               f"n={n} 超出题目规定的搜索范围 {SEARCH_BOUND}")

    # ---- 3. 我自己算 n^2+n+1 并对账 ----
    v = n * n + n + 1
    claimed_v = _as_int(fields.get("value"))
    chk.note("computed_value", v)
    if claimed_v is not None:
        chk.expect("value_matches", claimed_v == v,
                   f"报的 value={claimed_v}, 独立算出 {n}^2+{n}+1={v}")
    else:
        chk.note("value_claimed", None)

    # ---- 4. 我自己找平方因子 ----
    d = square_divisor(v)
    chk.expect("has_square_factor", d is not None,
               f"独立试除: {n}^2+{n}+1={v} 在 [2,{isqrt(v)}] 内没有任何质数 p 使 p^2|{v}"
               f" —— 即它确实无平方因子, 这个 n 不是反例")
    chk.note("square_divisor_found", d)

    claimed_d = _as_int(fields.get("square_factor"))
    if claimed_d is not None and d is not None:
        # agent 可能报"平方本身"(49)也可能报"它的根"(7), 两种都认:
        #   根形式    claimed_d^2 | v
        #   平方形式  claimed_d 是完全平方数且 claimed_d | v
        as_root = v % (claimed_d ** 2) == 0
        as_square = isqrt(claimed_d) ** 2 == claimed_d and v % claimed_d == 0
        chk.expect("square_factor_valid",
                   claimed_d >= 2 and (as_root or as_square),
                   f"报的平方因子 {claimed_d} 既不是 {v} 的平方因子, "
                   f"也不是某个 d 使 d^2|{v}")
        chk.note("square_factor_claimed", claimed_d)
        chk.note("square_factor_is_minimal", claimed_d == d)

    # ---- 5. 搜索范围自洽 ----
    rng = _parse_range(_first_key(art, "searched_range", "search_range", "range",
                                  "search_bound", "searched_up_to", "bound"))
    chk.note("claimed_search_range", list(rng) if rng else None)
    if rng is not None:
        lo, hi = rng
        chk.expect("range_contains_counterexample", lo <= n <= hi,
                   f"声称搜索了 [{lo},{hi}], 却给出 n={n} —— 范围与证据不自洽")
        chk.expect("range_upper_sane", hi >= n,
                   f"搜索上界 {hi} 小于反例 {n}, 搜索不可能找到它")
    else:
        chk.note("range_declared", False)

    # ---- 6. 诊断: 是不是最小反例(题目没要求, 只记 note) ----
    if n is not None and n <= SEARCH_BOUND:
        first = smallest_counterexample(N_MIN, max(MINIMALITY_SCAN, min(n, MINIMALITY_SCAN)))
        chk.note("true_smallest_counterexample", first)
        chk.note("is_minimal_counterexample", first == n)

    return _finish(chk, art, {"n": n, "value": v, "square_divisor": d})


def _finish(chk: Check, art: dict | None, extra: dict) -> dict:
    correct = chk.correct
    if correct:
        cert = (f"n={extra.get('n')} -> n^2+n+1={extra.get('value')}, "
                f"平方因子={extra.get('square_divisor')}(独立试除确认)")
    else:
        cert = chk.reason()
    final = None
    if correct and extra:
        final = json.dumps(extra, ensure_ascii=False, sort_keys=True)
    return {
        "correct": correct,
        "certificate": cert,
        "final_answer": final,
        "details": dict(chk.details),
    }


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- 正例: 真反例 n=18, 范围 [2, 1e6] ----
    good_stdout = (
        "VERDICT: REFUTED\n"
        'COUNTEREXAMPLE: ' + json.dumps({
            "n": 18, "value": 343, "square_factor": 49,
            "searched_range": [2, 1000000], "verdict": "false"}) + "\n"
    )
    r = verify(good_stdout, "", "")
    assert r["correct"], f"正例应通过: {r}"
    print("[OK] 正例: n=18 -> 343, 49|343, 范围自洽 ->", r["certificate"])

    # ---- 反例 A: 只判"没找到" (而 n=18 就在 1e6 内) ----
    r = verify("VERDICT: PROVED\n"
               'ANSWER: ' + json.dumps({"no_counterexample_up_to": 1000000}) + "\n",
               "", "")
    assert not r["correct"] and not r["details"]["verdict_is_refuted"], r
    print("[OK] 反例A: 误判为 PROVED ->", r["certificate"])

    # ---- 反例 B: 给了反例, 但 n^2+n+1 其实没有平方因子 ----
    r = verify("VERDICT: REFUTED\n"
               'COUNTEREXAMPLE: ' + json.dumps({
                   "n": 5, "value": 31, "square_factor": 31,
                   "searched_range": [2, 1000000]}) + "\n", "", "")
    assert not r["correct"] and not r["details"]["has_square_factor"], r
    print("[OK] 反例B: n=5 -> 31 是素数, 无平方因子 ->", r["certificate"])

    # ---- 反例 C: 平方因子对, 但声称的搜索范围装不下它(不自洽) ----
    r = verify("VERDICT: REFUTED\n"
               'COUNTEREXAMPLE: ' + json.dumps({
                   "n": 18, "value": 343, "square_factor": 49,
                   "searched_range": [5000, 1000000]}) + "\n", "", "")
    assert not r["correct"] and not r["details"]["range_contains_counterexample"], r
    print("[OK] 反例C: 范围 [5000,1e6] 与 n=18 不自洽 ->", r["certificate"])

    # ---- ★ 题库契约: counterexample 是一个字符串 ----
    for txt in ["n=18, 18^2+18+1=343, 平方因子 49",
                "n=18; 18^2+18+1=343=7^3, 49 | 343",
                "取 n=18: n^2+n+1 = 343, 含平方因子 7^2=49"]:
        a = {"verdict": "false", "counterexample": txt,
             "searched_range": [2, 1000000]}
        r = verify("VERDICT: REFUTED\nCOUNTEREXAMPLE: "
                   + json.dumps(a, ensure_ascii=False) + "\n", "", "")
        assert r["correct"], f"正例(契约字符串)应通过: {txt} -> {r}"
    print("[OK] 正例(契约字段 counterexample 为字符串, 三种写法) ->", r["certificate"])

    # 字符串通道照样逐项对账: n=5 时 31 是素数, 无平方因子 -> 判错
    a = {"verdict": "false", "counterexample": "n=5, 5^2+5+1=31, 平方因子 31",
         "searched_range": [2, 1000000]}
    r = verify("VERDICT: REFUTED\nCOUNTEREXAMPLE: "
               + json.dumps(a, ensure_ascii=False) + "\n", "", "")
    assert not r["correct"] and not r["details"]["has_square_factor"], r
    print("[OK] 反例D(字符串里的 n 其实不是反例) ->", r["certificate"])

    # 字符串里的 value 与独立算的不符 -> 判错
    a = {"verdict": "false", "counterexample": "n=18, 18^2+18+1=342, 平方因子 49",
         "searched_range": [2, 1000000]}
    r = verify("VERDICT: REFUTED\nCOUNTEREXAMPLE: "
               + json.dumps(a, ensure_ascii=False) + "\n", "", "")
    assert not r["correct"] and not r["details"]["value_matches"], r
    print("[OK] 反例E(字符串里的 value 与独立算的不符) ->", r["certificate"])

    print("[OK] 全部自测通过")