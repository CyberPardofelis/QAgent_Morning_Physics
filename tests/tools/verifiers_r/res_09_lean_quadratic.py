r"""res-009-lean-quadratic 的独立 verifier。

题目: 在 Lean 4 中形式化并证明二次公式 —— 若 a≠0, 则 a*x^2+b*x+c=0 当且仅当
x = (-b ± sqrt(b^2-4ac)) / (2a)(在适当的域/序假设下)。

★ 判据 ★
  1. 必须交出 Lean 源码制品, 否则 correct=False。
  2. verifier **自己调 Lean 编译**(复用
     `agent.tools.research._lean.check_lean`), 要求 proved=True 且
     sorry_used=False。理由同 res-008: Lean 遇 sorry 只报 warning、
     退出码仍 0, 只看退出码会把 `by sorry` 判成证明; check_lean 还查
     源码里的 `axiom` 声明 —— 那是把待证命题当前提的后门。
  3. 源码必须真的在讲二次公式: 含 sqrt、判别式(显式 b^2-4ac 或 discrim)、
     2a 的除法、以及 a≠0 的假设。为什么加这条: Lean 对**任何**正确的
     Lean 语句都判 proved, 所以只判"编译通过"的话 `theorem t : 1 = 1 := rfl`
     就能满分, 而它跟二次公式毫无关系。
  4. Lean/Mathlib 不可用时**如实返回 False**, details 写明"环境不具备"。

★ 关于"适当的域假设"★
题面的 iff 版本在 R 上需要判别式的分支讨论(下面 PROOF_OK 只证了
"是根 -> 二选一"这个方向, 已经是本题的核心代数内容)。verifier 不去判
"agent 的定理陈述是否与题面完全同构" —— 那是形式化本身的事, 由 Lean
的 kernel 保证**陈述为真**; verifier 负责的是"陈述为真 + 与二次公式
相关 + 无 sorry/axiom"这三件可判定的事。
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
TIMEOUT = 600

_SQRT_RE = re.compile(r"sqrt|√")
#: 判别式: 显式写成 b^2-4ac, 或用 Mathlib 的 discrim
_DISCRIM_RE = re.compile(r"\bdiscrim\b|4\s*\*\s*a\s*\*\s*c|b\s*\^\s*2\s*-\s*4", re.IGNORECASE)
#: 除以 2a
_DIV_RE = re.compile(r"\(\s*2\s*\*\s*a\s*\)|/\s*\(?\s*2\s*\*\s*a")
#: a ≠ 0 的假设
_HA_RE = re.compile(r"ha\s*:\s*a\s*≠\s*0|a\s*≠\s*0")
_DECL_RE = re.compile(r"^\s*(theorem|lemma|example)\s+\S", re.MULTILINE)


def _lean_source(art: dict) -> str:
    for key in ("code", "lean", "source", "src", "text", "content", "file"):
        v = art.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _looks_like_the_topic(code: str) -> tuple[bool, str]:
    """
    二次公式的关键词痕迹。返回 (是否像, 说明)。
    这是防"交个无关定理骗编译"的弱检查, 不是充分条件。
    """
    sig = {
        "sqrt": bool(_SQRT_RE.search(code)),
        "discriminant": bool(_DISCRIM_RE.search(code)),
        "divide by 2a": bool(_DIV_RE.search(code)),
        "a≠0 hypothesis": bool(_HA_RE.search(code)),
        "theorem/lemma/example": bool(_DECL_RE.search(code)),
    }
    ok = all(sig[k] for k in ("sqrt", "discriminant", "divide by 2a",
                              "a≠0 hypothesis", "theorem/lemma/example"))
    return ok, ", ".join(f"{k}={v}" for k, v in sig.items())


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
               f"源码看不出在证二次公式({why}) —— 防的是交个无关定理骗编译")

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
# 自测用的 Lean 源码(下面这条已在 Lean 4.9.0 + Mathlib v4.9.0 上实测 proved)
# --------------------------------------------------------------------------
PROOF_OK = """import Mathlib

open Polynomial

theorem quadratic_formula_roots (a b c x : ℝ) (ha : a ≠ 0) (hx : a * x * x + b * x + c = 0) :
    x = (-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a) ∨
      x = (-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a) := by
  have hsq : discrim a b c = (2 * a * x + b) ^ 2 := discrim_eq_sq_of_quadratic_eq_zero hx
  have hpos : 0 ≤ discrim a b c := by rw [hsq]; positivity
  have hsqrt : discrim a b c = Real.sqrt (discrim a b c) * Real.sqrt (discrim a b c) := by
    rw [Real.mul_self_sqrt hpos]
  rw [quadratic_eq_zero_iff ha hsqrt x] at hx
  rwa [discrim] at hx
"""

PROOF_SORRY = """import Mathlib

open Polynomial

theorem quadratic_formula_roots (a b c x : ℝ) (ha : a ≠ 0) (hx : a * x * x + b * x + c = 0) :
    x = (-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a) ∨
      x = (-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a) := by
  sorry
"""

PROOF_AXIOM = """import Mathlib

axiom roots_are_these (a b c x : ℝ) (ha : a ≠ 0) (hx : a * x * x + b * x + c = 0) :
    x = (-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a) ∨
      x = (-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)

theorem quadratic_formula_roots (a b c x : ℝ) (ha : a ≠ 0) (hx : a * x * x + b * x + c = 0) :
    x = (-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a) ∨
      x = (-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a) :=
  roots_are_these ha hx
"""

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