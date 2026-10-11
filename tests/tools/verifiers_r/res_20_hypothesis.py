"""
res-020-hypothesis-refine-loop 的独立 verifier。

题目(data/res_20_conjecture.txt): 猜想 C = "对所有整数 n>=1, 2^n+1 是素数",
任务: 先搜反例; 若有反例, 修改为最小修正版并给出反例边界。

★ C 是假的, 且反例小得出奇 ★
  n=1 -> 3 (素), n=2 -> 5 (素), n=3 -> 2^3+1 = 9 = 3^2 (合)。
  所以"最小反例"就是 n=3。这道题考的不是算力, 而是**敢不敢推翻自己**。

verifier 的判据(全部自己算, 不复用 agent 的搜索代码):
  1. verdict 必须是假(refuted / false / refined) —— 报"成立"直接判错;
  2. 反例必须存在且就是 n=3(题目明确要"最小反例"; n=9 也是反例但不是最小的);
     ★ 反例从**三个来源**读, 谁先到谁算数 ★
       a) 题库契约字段 ``evidence``（字符串, 如 "n=3, 2^3+1=9"）——
          由 :func:`_parse_evidence` 正则抠出 n;
       b) 独立的 ``COUNTEREXAMPLE:`` 制品行（题面要求"最小反例", agent 很可能
          直接用制品行交, 而不是塞进 evidence）;
       c) 结构化字段 n / counterexample_n。
  3. 我自己确认 n=1,2 是素数、n=3 是合数(最小因子 3) —— 这是"最小"二字的地基;
  4. 必须声明搜索范围, 且范围要装得下它自己给的反例(自洽);
  5. 修正版必须给出**可核验的边界**(refined_holds_for 之类的结构化字段),
     我逐个 n 独立验证 2^n+1 确实是素数, 并确认边界之外确实塌了。

第 5 条为什么必须机器验: "最小修正版"很容易写成一句漂亮但没边界的话
("n 需要额外条件")。没有可核验边界的修正版, 无法与"蒙对了结论"区分开。

★ 边界验证的两种手段(便宜的先用) ★
  a) 代数判据: n = 2^k * m (m 为奇数), 若 m>1 则 2^n+1 可被分解为
     (2^(2^k))^m + 1, 而 a+1 | a^m+1 (m 为奇数) => 必合。
     这给出**显式因子**, 可以直接做整除验证, 所以对任意大的区间都是精确的。
  b) 素性判据: 只对"界内确实是素数"的那部分 n 用 sympy.isprime 逐个验。
     只有 n 是 2 的幂时才需要真判素性(其余已被 (a) 判合)。
     为控制耗时, 逐个素性检查有预算 BUDGET, 超出的部分在 details 里如实标注。
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

#: 题目文件规定的搜索上界(解析失败时的兜底)
DEFAULT_SEARCH_BOUND = 10000
#: 逐个做素性检查的预算(超出只做代数判据, 并在 details 里标注)
BUDGET = 400

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOTS = [os.path.abspath(os.path.join(_HERE, "..", "..", "..")),
          os.path.abspath(os.getcwd())]
_CONJECTURE_REL = os.path.join("data", "res_20_conjecture.txt")


def is_prime(n: int) -> bool:
    try:
        import sympy
        return bool(sympy.isprime(n))
    except Exception:
        # 兜底: 试除到 sqrt(n)
        from math import isqrt
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        d = 3
        while d <= isqrt(n):
            if n % d == 0:
                return False
            d += 2
        return True


def smallest_factor(n: int) -> int:
    d = 2
    while d * d <= n and n % d:
        d += 1
    return d if d * d <= n else n


def algebraic_composite_factor(n: int) -> int | None:
    """若 2^n+1 必合, 返回一个显式因子(2^(2^k)+1); 否则 None。

    n = 2^k*m, m 奇。若 m>1, 令 a = 2^(2^k), 则 2^n+1 = a^m+1,
    且 m 奇 => a+1 是 a^m+1 的因子。
    """
    if n < 1:
        return None
    m = n
    while m % 2 == 0:
        m //= 2
    if m == 1:                      # n 是 2 的幂(Fermat 数), 代数判据无效
        return None
    return (1 << (n // m)) + 1


def search_bound() -> int:
    """从题目文件里解析 "n <= 10000"; 解析不到就用默认值。"""
    for root in _ROOTS:
        p = os.path.join(root, _CONJECTURE_REL)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    txt = fh.read()
                m = re.search(r"n\s*(?:<=|≤|<)\s*(\d+)", txt)
                if m:
                    return int(m.group(1))
            except OSError:
                pass
    return DEFAULT_SEARCH_BOUND


# --------------------------------------------------------------------------
def _as_int(x: Any) -> int | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        try:
            return int(x.strip())
        except ValueError:
            return None
    return None


def _parse_range(x: Any) -> tuple[int, int] | None:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return (1, int(x))
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        lo, hi = _as_int(x[0]), _as_int(x[1])
        if lo is not None and hi is not None:
            return (lo, hi)
    if isinstance(x, dict):
        lo = next((_as_int(x[k]) for k in ("from", "lo", "min", "start", "lower")
                   if k in x), None)
        hi = next((_as_int(x[k]) for k in ("to", "hi", "max", "end", "upper")
                   if k in x), None)
        if lo is not None and hi is not None:
            return (lo, hi)
    return None


_HOLDS_KEYS = ("refined_holds_for", "holds_for", "holds_for_n", "valid_n",
               "true_for_n", "works_for", "holds_n")
_RANGE_KEYS = ("refined_holds_range", "holds_range", "valid_range", "boundary",
               "holds_for_range", "range")
_NEST_KEYS = ("refined_conjecture", "refinement", "refined", "correction")


def _find_boundary(obj: Any, depth: int = 0) -> tuple[str, Any] | None:
    """在制品里找修正版的"成立区间"。支持嵌套一层 refined_conjecture。"""
    if depth > 2:
        return None
    if isinstance(obj, dict):
        for k in _HOLDS_KEYS:
            v = obj.get(k)
            if isinstance(v, (list, tuple)) and v:
                ns = [x for x in (_as_int(i) for i in v) if x is not None]
                if ns:
                    return ("list", ns)
        for k in _RANGE_KEYS:
            r = _parse_range(obj.get(k))
            if r:
                return ("range", r)
        for k in _NEST_KEYS:
            if k in obj:
                got = _find_boundary(obj[k], depth + 1)
                if got:
                    return got
    elif isinstance(obj, (list, tuple)):
        for it in obj:
            got = _find_boundary(it, depth + 1)
            if got:
                return got
    return None


def _says_false(verdict: str | None, art: dict | None) -> bool:
    if verdict and verdict.lower() in {"refuted", "false", "disproved", "falsified"}:
        return True
    if not art:
        return False
    raw = art.get("verdict", art.get("result", art.get("status")))
    if isinstance(raw, str) and raw.strip().lower() in {
            "refuted", "false", "disproved", "falsified", "refined"}:
        return True
    return raw is False


#: 题库契约字段 evidence 是一个字符串（"反例或证明或数值证据"）, 最小反例
#: 很可能就写在里面: "n=3 时 2^3+1=9"。只抠 n, 抠到照样自己重算对账。
_N_IN_TEXT = re.compile(r"(?:取\s*)?n\s*=\s*(\d{1,9})", re.I)


def _parse_evidence(raw: Any) -> int | None:
    """从契约字段 evidence（字符串）里抠出反例 n。抠不到返回 None。"""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, dict):
        for k in ("n", "counterexample_n", "value_n"):
            v = _as_int(raw.get(k))
            if v is not None:
                return v
        return None
    if not isinstance(raw, str):
        return None
    m = _N_IN_TEXT.search(raw.replace("−", "-"))
    return int(m.group(1)) if m else None


def _verify_boundary(bound: tuple[str, Any], chk: Check) -> None:
    """独立验证修正版的成立边界。"""
    kind, val = bound
    if kind == "list":
        ns = sorted(set(val))
        chk.note("refined_holds_for", ns)
        chk.expect("boundary_excludes_false_cases", all(n != 3 for n in ns),
                   "修正版把 n=3 也算进成立范围, 但 2^3+1=9 是合数")
        bad = [n for n in ns if not is_prime(2 ** n + 1)]
        chk.expect("every_claimed_n_is_prime", not bad,
                   f"修正版声称这些 n 成立, 但 2^n+1 合数: {bad[:5]}")
        top = max(ns)
        # 边界上界以内必须全是素数, 否则"成立范围"本身就不成立
        lo = min(ns)
        wrong = [n for n in range(lo, min(top, lo + BUDGET) + 1)
                 if not is_prime(2 ** n + 1)]
        chk.expect("interval_is_prime_free", not wrong,
                   f"修正版声称 [{lo},{top}] 内成立, 但 n={wrong[:5]} 处已反例")
        if 3 in range(lo, top + 1):
            chk.expect("boundary_is_minimal", False,
                       f"修正版声称 [{lo},{top}] 成立, 但 n=3 在其中且 2^3+1=9 合数")
        else:
            chk.note("boundary_is_minimal", True)
    else:
        lo, hi = val
        chk.note("refined_holds_range", [lo, hi])
        # 代数判据: 对区间内每个 n, 若必合则立刻判负(精确, 不受预算限制)
        composites = []
        for n in range(lo, hi + 1):
            f = algebraic_composite_factor(n)
            if f is not None and (2 ** n + 1) % f == 0:
                composites.append(n)
        chk.expect("no_composite_in_claimed_range", not composites,
                   f"修正版声称 [{lo},{hi}] 内 2^n+1 全是素数, 但 n={composites[:5]} 处已反例")
        # 素性判据: 对区间逐个验证(有预算)
        hi_check = min(hi, lo + BUDGET - 1)
        wrong = [n for n in range(lo, hi_check + 1)
                 if algebraic_composite_factor(n) is None
                 and not is_prime(2 ** n + 1)]
        chk.expect("every_claimed_n_is_prime", not wrong,
                   f"修正版声称 [{lo},{hi}] 内成立, 但 n={wrong[:5]} 处 2^n+1 是合数")
        chk.note("range_fully_scanned", hi_check >= hi)
        chk.note("scan_window", [lo, hi_check])


# --------------------------------------------------------------------------
def verify(stdout: str, code: str, claim: str) -> dict:
    chk = Check()
    bound = search_bound()
    chk.note("search_bound_from_conjecture_file", bound)

    ce_art = parse_artifact(stdout, "COUNTEREXAMPLE")
    # 主体制品优先取 ANSWER/CONSTRUCT: 修正版和搜索范围一般写在那一行,
    # 独立的 COUNTEREXAMPLE 行只负责交最小反例。两行都在时要合起来读。
    art = (parse_artifact(stdout, "ANSWER")
           or parse_artifact(stdout, "CONSTRUCT")
           or ce_art)
    if not chk.expect("artifact_present", art is not None or ce_art is not None,
                      "未提交 COUNTEREXAMPLE/ANSWER 制品"):
        return _finish(chk, {})
    assert art is not None or ce_art is not None
    if art is None:                     # 只有 COUNTEREXAMPLE 行
        art = ce_art
    # 题面要"最小反例", agent 很可能用独立的 COUNTEREXAMPLE: 行交,
    # 而不是把 n 塞进 ANSWER 里。两种制品行都要能读到最小反例。
    if ce_art is not None and ce_art is not art:
        chk.note("artifact_tags", ["COUNTEREXAMPLE", "ANSWER"])

    verdict = parse_verdict(stdout)
    chk.note("stdout_verdict", verdict)
    chk.expect("verdict_is_false",
               _says_false(verdict, art) or _says_false(verdict, ce_art),
               f"猜想 C 是假的(n=3 时 2^3+1=9), verdict={verdict!r} 不能是成立")

    # ---- 反例 ----
    # ★ 三个来源都读 ★ 独立 COUNTEREXAMPLE 行 > ANSWER 里的结构化 n >
    #   契约字段 evidence 字符串。读到的 n 一律自己重算对账。
    n = None
    src = None
    if ce_art is not None and ce_art is not art:
        n, src = _as_int(ce_art.get("n", ce_art.get("counterexample_n"))), "COUNTEREXAMPLE 行"
    if n is None:
        n = _as_int(art.get("n", art.get("counterexample_n", art.get("value_n"))))
        src = "结构化 n 字段" if n is not None else None
    if n is None:
        n, src = _parse_evidence(art.get("evidence")), "契约字段 evidence 字符串"
    chk.note("counterexample_source", src)
    # 字段认到了没有（字段名层面）: evidence / COUNTEREXAMPLE 行 / n 三者任一
    chk.expect("counterexample_read",
               src is not None
               or _parse_evidence(art.get("evidence")) is not None
               or ce_art is not None,
               "制品里没有反例: 题库契约字段是 verdict + evidence + refined_conjecture; "
               "evidence 字符串里写 'n=3, 2^3+1=9' 即可, 也接受独立的 "
               "COUNTEREXAMPLE: 制品行与 n 字段")

    if chk.expect("counterexample_minimal_n_is_3", n is not None,
                  f"读不到反例 n（来源 {src}）; 字符串写法示例: 'n=3, 2^3+1=9=3^2'"):
        assert n is not None
        chk.expect("n_is_minimal_3", n == 3,
                   f"题目要求最小反例; n={n} 不是最小的(最小是 3)")
        chk.expect("n_within_bound", n <= bound, f"n={n} 超出题目给的搜索范围 {bound}")
        v = 2 ** n + 1
        chk.note("computed_value", v)
        src_art = ce_art if (ce_art is not None and ce_art is not art) else art
        claimed_v = _as_int(src_art.get("value", src_art.get("result_value")))
        if claimed_v is None and src_art is not art:
            claimed_v = _as_int(art.get("value", art.get("result_value")))
        if claimed_v is None and src == "契约字段 evidence 字符串":
            m = re.search(r"(?:2\^n\s*\+\s*1|2\^3\s*\+\s*1)\s*=\s*(\d{1,12})",
                          str(art.get("evidence")), re.I)
            claimed_v = int(m.group(1)) if m else None
        if claimed_v is not None:
            chk.expect("value_matches", claimed_v == v,
                       f"报的 value={claimed_v}, 独立算出 2^{n}+1={v}")
        sf = smallest_factor(v)
        chk.note("smallest_factor", sf)
        claimed_sf = _as_int(src_art.get("smallest_factor", src_art.get("factor")))
        if claimed_sf is None and src_art is not art:
            claimed_sf = _as_int(art.get("smallest_factor", art.get("factor")))
        if claimed_sf is not None:
            chk.expect("smallest_factor_matches", claimed_sf == sf,
                       f"报的最小因子 {claimed_sf}, 独立试除得 {sf}")

    # ---- 最小性的地基: n=1,2 素, n=3 合 ----
    p1, p2, p3 = is_prime(3), is_prime(5), is_prime(9)
    chk.note("prime_2p1", p1)
    chk.expect("n1_n2_prime", p1 and p2, "2^1+1=3 与 2^2+1=5 都应是素数, 否则前提有误")
    chk.expect("n3_composite", not p3 and smallest_factor(9) == 3,
               "2^3+1=9 不是素数 —— 若这是素数, 题目前提就有坑")

    # ---- 搜索范围: 必须声明且自洽 ----
    rng = _parse_range(art.get("searched_range", art.get("search_range",
                      art.get("range", art.get("search_bound", art.get("bound"))))))
    chk.expect("search_range_declared", rng is not None,
               "题目明确要求'说清楚你搜的范围有多大', 制品里没有范围字段")
    if rng is not None:
        lo, hi = rng
        chk.note("claimed_search_range", [lo, hi])
        chk.expect("range_contains_counterexample", lo <= 3 <= hi,
                   f"声称搜索了 [{lo},{hi}], 却给出 n=3 的反例 —— 范围与证据不自洽")
        chk.expect("range_upper_sane", hi >= 3, f"搜索上界 {hi} < 3, 搜不到这个反例")

    # ---- 修正版 + 边界 ----
    refined = art.get("refined_conjecture", art.get("refinement"))
    has_refined = isinstance(refined, (str, dict, list)) and bool(refined)
    chk.expect("refined_conjecture_present", has_refined,
               "题目要求'修改猜想为最小修正版', 制品里没有 refined_conjecture")
    bound_desc = _find_boundary(art)
    if chk.expect("refined_boundary_declared", bound_desc is not None,
                  "修正版没有给出可核验的边界"
                  "(需要 refined_holds_for=[...] 或 refined_holds_range={from,to})"):
        assert bound_desc is not None
        _verify_boundary(bound_desc, chk)

    extra = {"verdict": "false", "counterexample_n": 3, "value": 9,
             "smallest_factor": 3}
    if rng is not None:
        extra["searched_range"] = list(rng)
    if bound_desc is not None:
        extra["boundary"] = bound_desc[1] if bound_desc[0] == "list" else list(bound_desc[1])
    return _finish(chk, extra)


def _finish(chk: Check, extra: dict) -> dict:
    correct = chk.correct
    if correct:
        cert = ("verdict=false, 最小反例 n=3 (2^3+1=9=3^2) 经独立核算确认; "
                "n=1,2 为素数保证 3 确为最小; 修正版边界已逐点验证")
    else:
        cert = chk.reason()
    final = json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else None
    return {"correct": correct, "certificate": cert, "final_answer": final,
            "details": dict(chk.details)}


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- 正例 ----
    good = {
        "verdict": "false", "n": 3, "value": 9, "smallest_factor": 3,
        "searched_range": [1, 10000],
        "refined_conjecture": "n>=1 时 2^n+1 不是都素数; 成立的只有 n=1,2",
        "refined_holds_for": [1, 2],
    }
    r = verify("VERDICT: REFUTED\nCONSTRUCT: " + json.dumps(good) + "\n", "", "")
    assert r["correct"], f"正例应通过: {r}"
    print("[OK] 正例: verdict=false + n=3 + 边界[1,2] ->", r["certificate"])

    # ---- 反例 A: 判"猜想成立"(n=3 明明反例) ----
    bad_a = dict(good, verdict="true")
    r = verify("VERDICT: PROVED\nCONSTRUCT: " + json.dumps(bad_a) + "\n", "", "")
    assert not r["correct"] and not r["details"]["verdict_is_false"], r
    print("[OK] 反例A: 误判成立 ->", r["certificate"])

    # ---- 反例 B: 给了反例但不是最小的(n=9 时 513=27*19 也是合数) ----
    bad_b = dict(good, n=9, value=513, smallest_factor=3)
    r = verify("VERDICT: REFUTED\nCOUNTEREXAMPLE: " + json.dumps(bad_b) + "\n", "", "")
    assert not r["correct"] and not r["details"]["n_is_minimal_3"], r
    print("[OK] 反例B: n=9 不是最小反例 ->", r["certificate"])

    # ---- 反例 C: 修正版的边界把 n=3 也算进去(蒙一个"更大的修正版") ----
    bad_c = dict(good, refined_holds_for=[1, 2, 3, 4, 5])
    r = verify("VERDICT: REFUTED\nCONSTRUCT: " + json.dumps(bad_c) + "\n", "", "")
    assert (not r["correct"]
            and not r["details"]["boundary_excludes_false_cases"]), r
    print("[OK] 反例C: 修正版边界含 n=3 ->", r["certificate"])

    # ---- 反例 D: 没声明搜索范围 ----
    bad_d = {k: v for k, v in good.items() if k != "searched_range"}
    r = verify("VERDICT: REFUTED\nCONSTRUCT: " + json.dumps(bad_d) + "\n", "", "")
    assert not r["correct"] and not r["details"]["search_range_declared"], r
    print("[OK] 反例D: 未声明搜索范围 ->", r["certificate"])

    # ---- 反例 E: 范围与反例不自洽 ----
    bad_e = dict(good, searched_range=[500, 10000])
    r = verify("VERDICT: REFUTED\nCONSTRUCT: " + json.dumps(bad_e) + "\n", "", "")
    assert not r["correct"] and not r["details"]["range_contains_counterexample"], r
    print("[OK] 反例E: 范围 [500,10000] 装不下 n=3 ->", r["certificate"])

    # ---- ★ 契约字段 evidence（字符串）里写最小反例 ----
    ev = {"verdict": "false",
          "evidence": "n=3 时 2^3+1=9=3^2, 是合数; n=1,2 时分别是 3 和 5(素数)",
          "refined_conjecture": "只有 n=1,2 成立", "refined_holds_for": [1, 2],
          "searched_range": [1, 10000]}
    r = verify("VERDICT: REFUTED\nANSWER: " + json.dumps(ev, ensure_ascii=False) + "\n", "", "")
    assert r["correct"], f"正例(契约 evidence 字符串)应通过: {r}"
    assert r["details"]["counterexample_source"] == "契约字段 evidence 字符串", r["details"]
    print("[OK] 正例(契约字段 evidence 字符串含 n=3) ->", r["certificate"])

    # ---- ★ 独立的 COUNTEREXAMPLE 制品行（题面要求"最小反例"）----
    two_lines = ("VERDICT: REFUTED\n"
                 'COUNTEREXAMPLE: ' + json.dumps({"n": 3, "value": 9}) + "\n"
                 'ANSWER: ' + json.dumps(
                     {"verdict": "false", "refined_conjecture": "只有 n=1,2 成立",
                      "refined_holds_for": [1, 2], "searched_range": [1, 10000]},
                     ensure_ascii=False) + "\n")
    r = verify(two_lines, "", "")
    assert r["correct"], f"正例(COUNTEREXAMPLE 行 + ANSWER 行)应通过: {r}"
    print("[OK] 正例(COUNTEREXAMPLE 行 + ANSWER 行) ->", r["certificate"])

    # COUNTEREXAMPLE 行里的 n 不是最小反例 -> 判错（两个通道都逐项对账）
    bad_ce = ("VERDICT: REFUTED\n"
              'COUNTEREXAMPLE: ' + json.dumps({"n": 9, "value": 513}) + "\n"
              'ANSWER: ' + json.dumps(
                  {"verdict": "false", "refined_conjecture": "只有 n=1,2 成立",
                   "refined_holds_for": [1, 2], "searched_range": [1, 10000]},
                  ensure_ascii=False) + "\n")
    r = verify(bad_ce, "", "")
    assert not r["correct"] and not r["details"]["n_is_minimal_3"], r
    print("[OK] 反例F(COUNTEREXAMPLE 行给的不是最小反例) ->", r["certificate"])

    # evidence 字符串里的 n=9 -> 同样判错
    ev9 = dict(ev, evidence="n=9 时 2^9+1=513 合数")
    r = verify("VERDICT: REFUTED\nANSWER: " + json.dumps(ev9, ensure_ascii=False) + "\n", "", "")
    assert not r["correct"] and not r["details"]["n_is_minimal_3"], r
    print("[OK] 反例G(evidence 字符串里的 n 不是最小反例) ->", r["certificate"])

    print("[OK] 全部自测通过")