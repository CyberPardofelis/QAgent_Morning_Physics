r"""res-008-lean-binomial 的独立 verifier。

题目: 在 Lean 4 + Mathlib 中形式化并证明 sum_{k=0}^{n} C(n,k) = 2^n, 不允许 sorry。

★ 判据 ★
  1. 必须交出 Lean 源码制品(CONSTRUCT / ANSWER), 否则 correct=False。
     题库 verifier.entrypoint 指向 `verifiers/lean/Res08Binomial.lean`,
     本仓库没有这个目录, 所以判据落在 agent 的 stdout 制品里。
  2. verifier **自己调 Lean 编译**它(复用
     `agent.tools.research._lean.check_lean` —— 这是编译后端, 不是
     "答案": 它返回的是编译器对证明项的判定, 与 agent 说什么无关)。
     要求 proved=True 且 sorry_used=False。
     ★ 为什么必须用 check_lean 而不是看退出码 ★ Lean 遇到 sorry 只报
     warning, 退出码仍是 0 —— 只看退出码会把 `by sorry` 判成证明。
     check_lean 同时查三样: 退出码、": error:"、"declaration uses 'sorry'"
     以及源码里的 `axiom` 声明(提交证明时自带 axiom = 把待证命题当前提,
     编译器不会报错, 是这条路上最隐蔽的后门)。
  3. 源码必须**真的在讲这件事**: 含 sum + choose/binomial + 2^n 的痕迹,
     且不是只有 `example : True := trivial` 这种空壳。
     为什么加这条: Lean 对**任何**正确的 Lean 语句都会说 proved;
     只判"编译通过"的话, `theorem t : 1 = 1 := rfl` 也能满分, 而它跟
     二项式定理毫无关系。关键词检查挡不住更刁钻的偷梁换柱, 但能挡住
     "交个无关的定理骗编译"这一类, 而这类是最常见的。
  4. Mathlib / Lean 不在时**如实返回 False** 并在 details 里说明"环境不具备"。
     绝不用"看着像"代替机器判定 —— 那正是这套 verifier 要消灭的行为。

★ 环境 ★ Mathlib v4.9.0 在 `D:\pythonProject\Others\mathlib4-v490`
(见 `agent.tools.research._lean.MATHLIB_DIR`)。
"""
from __future__ import annotations

import json
import os
import re
import sys

try:
    from .common import Check, parse_artifact, parse_verdict
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import Check, parse_artifact, parse_verdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.tools.research._lean import check_lean, lean_version, mathlib_available  # noqa: E402

#: 题库 verifier.params.timeout_sec
TIMEOUT = 300

#: 源码里必须出现的"这题确实在讲"的痕迹(任一组即可)
_SUM_RE = re.compile(r"\bSum\b|∑|Finset\.sum|sum_range|\.sum\b")
_CHOOSE_RE = re.compile(r"\bchoose\b|\bbinomial\b|\bNat\.choose\b", re.IGNORECASE)
_POW2_RE = re.compile(r"2\s*\^|\bpow\s+2\b|two_pow")
_DECL_RE = re.compile(r"^\s*(theorem|lemma|example)\s+\S", re.MULTILINE)


def _lean_source(art: dict) -> str:
    """从制品里取 Lean 源码(容忍几种键名)。"""
    for key in ("code", "lean", "source", "src", "text", "content", "file"):
        v = art.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _looks_like_the_topic(code: str) -> tuple[bool, str]:
    """
    检查源码是否真的在证二项式定理。
    返回 (是否像, 说明)。这是"防交无关定理"的弱检查, 不是充分条件。
    """
    has_sum = bool(_SUM_RE.search(code))
    has_choose = bool(_CHOOSE_RE.search(code))
    has_pow2 = bool(_POW2_RE.search(code))
    has_decl = bool(_DECL_RE.search(code))
    ok = has_sum and has_choose and has_pow2 and has_decl
    why = (f"sum={has_sum} choose/binomial={has_choose} 2^n={has_pow2} "
           f"theorem/lemma/example={has_decl}")
    return ok, why


def verify(stdout: str, code: str, claim: str) -> dict:
    chk = Check()
    chk.note("verdict", parse_verdict(stdout))
    chk.note("lean_version", lean_version())
    chk.note("mathlib_available", mathlib_available())

    art = parse_artifact(stdout, "CONSTRUCT") or parse_artifact(stdout, "ANSWER")
    if not chk.expect("artifact_present", art is not None,
                      "未提交 CONSTRUCT 制品(需要 {\"code\": \"<lean 源码>\"})"):
        return _finish(chk, None, {})
    assert art is not None

    lean_code = _lean_source(art)
    if not chk.expect("lean_source_present", bool(lean_code.strip()),
                      "制品里没有 Lean 源码(code/lean/source 都没有内容)"):
        return _finish(chk, art, {})
    assert lean_code.strip()

    # ---- 环境不具备: 如实判错 ----
    if not mathlib_available():
        chk.note("env_ready", False)
        chk.expect("lean_env", False,
                   "环境不具备: Lean/Mathlib 不可用, 无法机器判定 —— "
                   "如实判错, 不用'看着像'代替编译")
        return _finish(chk, art, {"lean_chars": len(lean_code)})
    chk.note("env_ready", True)

    # ---- 题目相关性(弱检查) ----
    topic_ok, why = _looks_like_the_topic(lean_code)
    chk.note("topic_signature", why)
    chk.expect("topic_relevance", topic_ok,
               f"源码看不出在证 sum C(n,k)=2^n({why}) —— "
               "防的是交一个无关定理骗编译")

    # ---- ★ 机器判定: 自己编译 ★ ----
    res = check_lean(lean_code, timeout=TIMEOUT)
    chk.note("lean_exit_code", res.get("exit_code"))
    chk.note("lean_errors", (res.get("errors") or [])[:5])
    chk.note("lean_warnings", (res.get("warnings") or [])[:5])
    chk.note("lean_axioms", (res.get("axioms") or [])[:5])
    chk.expect("lean_proved", bool(res.get("proved")),
               "Lean 没判为 proved(exit=%s, errors=%d, sorry/axiom=%d)"
               % (res.get("exit_code"), len(res.get("errors") or []),
                  len(res.get("warnings") or [])))
    chk.expect("no_sorry", not res.get("sorry_used"),
               "证明里用了 sorry / 自带 axiom —— 等于没证")

    return _finish(chk, art, {"lean_chars": len(lean_code),
                              "lean_proved": bool(res.get("proved")),
                              "sorry_used": bool(res.get("sorry_used"))})


def _finish(chk: Check, art: dict | None, extra: dict) -> dict:
    correct = chk.correct
    if correct:
        cert = (f"Lean kernel 判定 proved=True 且无 sorry/axiom"
                f"(源码 {extra['lean_chars']} 字符)")
    else:
        cert = chk.reason()
    final = json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else None
    return {"correct": correct, "certificate": cert, "final_answer": final,
            "details": dict(chk.details)}


# --------------------------------------------------------------------------
# 自测用的 Lean 源码
# --------------------------------------------------------------------------
#: 真证明: 直接用 Mathlib 的 Nat.sum_range_choose(在 Data/Nat/Choose/Sum.lean)
PROOF_OK = """import Mathlib
theorem sum_choose_two_pow (n : Nat) :
    ∑ k in Finset.range (n + 1), Nat.choose n k = 2 ^ n := by
  simpa using Nat.sum_range_choose n
"""

#: 反例A: 用 sorry —— 退出码仍是 0, 所以只看退出码会被骗
PROOF_SORRY = """import Mathlib
theorem sum_choose_two_pow (n : Nat) :
    ∑ k in Finset.range (n + 1), Nat.choose n k = 2 ^ n := by
  sorry
"""

#: 反例B: 自带 axiom 把待证命题当前提
PROOF_AXIOM = """import Mathlib
axiom binomial_sum_two_pow (n : Nat) :
    ∑ k in Finset.range (n + 1), Nat.choose n k = 2 ^ n
theorem sum_choose_two_pow (n : Nat) :
    ∑ k in Finset.range (n + 1), Nat.choose n k = 2 ^ n :=
  binomial_sum_two_pow n
"""

#: 反例C: 编译通过但与本题无关(证明 1=1)
PROOF_UNRELATED = """import Mathlib
theorem one_eq_one : (1 : Nat) = 1 := rfl
"""


if __name__ == "__main__":
    if not mathlib_available():
        print("[SKIP] Lean 不可用 —— Mathlib/lean.exe 缺失, 无法机器判定")
        raise SystemExit(0)
    print(f"[INFO] {lean_version()}; mathlib_available={mathlib_available()}")

    def run(src: str) -> dict:
        return verify("VERDICT: PROVED\nCONSTRUCT: " +
                      json.dumps({"code": src}, ensure_ascii=False) + "\n", "", "")

    r = run(PROOF_OK)
    assert r["correct"], f"正例应通过: {r}"
    print("[OK] 正例(真证明, 无 sorry) ->", r["certificate"])

    r = run(PROOF_SORRY)
    assert not r["correct"] and not r["details"]["no_sorry"], r
    print("[OK] 反例A(sorry) ->", r["certificate"])

    r = run(PROOF_AXIOM)
    assert not r["correct"], r
    print("[OK] 反例B(自带 axiom) ->", r["certificate"])

    r = run(PROOF_UNRELATED)
    assert not r["correct"] and not r["details"]["topic_relevance"], r
    print("[OK] 反例C(无关定理) ->", r["certificate"])

    r = verify("VERDICT: PROVED\n", "", "")
    assert not r["correct"] and not r["details"]["artifact_present"], r
    print("[OK] 反例D(没交源码) ->", r["certificate"])

    print("[OK] 全部自测通过")