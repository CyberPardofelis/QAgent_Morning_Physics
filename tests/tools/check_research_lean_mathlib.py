# -*- coding: utf-8 -*-
"""research_lean 接上 Mathlib 之后的端到端验证。

要验的不只是「能不能编译」, 更是**那道判定有没有被绕过**:
  - 真定理 + #print axioms(无 sorryAx)   -> proved
  - 真定理但用了 sorry                     -> NOT proved
  - axiom 后门(编译器一声不吭)           -> NOT proved   ★ 新增的锁
  - 语法错                                 -> NOT proved
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\pythonProject\QAgent")

from agent.tools.research._lean import (  # noqa: E402
    check_lean, lean_available, mathlib_available,
)

CASES = []


def case(name, code, expect_proved):
    CASES.append((name, code, expect_proved))


# ---------- 1. 真定理 (V2 hard-016) ----------
case(
    "sqrt2 无理 (hard-016 真定理)",
    """import Mathlib
example : Irrational (Real.sqrt 2) := irrational_sqrt_two
#print axioms irrational_sqrt_two
""",
    True,
)

# ---------- 2. 费马小定理 (V2 hard-015) ----------
case(
    "费马小定理 (hard-015 真定理)",
    """import Mathlib
example (p : Nat) (hp : p.Prime) (a : Nat) (h : ¬ (p ∣ a)) :
    a ^ (p - 1) ≡ 1 [MOD p] := by
  have h' := Nat.ModEq.pow_totient (x := a) (n := p) (hp.coprime_iff_not_dvd.mpr h).symm
  simpa [Nat.totient_prime hp] using h'
""",
    True,
)

# ---------- 3. sorry 应被降级 ----------
case(
    "sorry 占位",
    """import Mathlib
example : Irrational (Real.sqrt 2) := by sorry
""",
    False,
)

# ---------- 4. ★ axiom 后门: 编译器不报错, 但根本没证 ----------
# 这条是新增那道锁的意义。`axiom foo : P` 之后 `theorem bar : P := foo`,
# Lean 编译成功、退出码 0、没有任何 warning —— 只看退出码会判成 proved。
case(
    "axiom 后门 (必须识破)",
    """import Mathlib
axiom my_sqrt2 : Irrational (Real.sqrt 2)
example : Irrational (Real.sqrt 2) := my_sqrt2
#print axioms my_sqrt2
""",
    False,
)

# ---------- 5. 语法错 ----------
case(
    "语法错",
    """import Mathlib
example : True := by this_is_not_a_tactic
""",
    False,
)

# ---------- 6. 不依赖 Mathlib 的核心定理 ----------
case(
    "Lean 核心 (无 Mathlib)",
    """theorem t (n : Nat) : n * 2 + 1 ≥ 1 := by omega
""",
    True,
)


def main() -> int:
    print(f"lean_available = {lean_available()}")
    print(f"mathlib_available = {mathlib_available()}")
    if not mathlib_available():
        print("!! Mathlib 不在, 前三道测不了")
    print("=" * 66)

    failed = []
    for name, code, expect in CASES:
        r = check_lean(code, timeout=600)
        got = bool(r.get("proved"))
        ok = got == expect
        flag = "OK " if ok else "!! "
        print(f"{flag}{name}")
        print(f"    proved={got} (期望 {expect}) exit={r.get('exit_code')} "
              f"sorry={r.get('sorry_used')} mathlib={r.get('mathlib')}")
        if r.get("axioms"):
            print(f"    公理审计: {r['axioms']}")
        if r.get("errors"):
            print(f"    错误: {r['errors'][:2]}")
        if not ok:
            failed.append(name)
    print("=" * 66)
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print(f"全部通过 ({len(CASES)}/{len(CASES)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())