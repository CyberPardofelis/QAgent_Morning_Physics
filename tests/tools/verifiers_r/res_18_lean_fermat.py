"""
res-018-formal-fermat-small 的独立 verifier。

题目: 在 Lean 4 中形式化并证明: 对素数 p=5, 2^5 ≡ 2 (mod 5), 并对 p=7 同样验证。
      用 ZMod 表达。

判据:
  1. ★ 先自己把数学结论算一遍 ★
     2^5 mod 5 = 32 mod 5 = 2, 2^7 mod 7 = 128 mod 7 = 2, 且 5, 7 都是素数。
     这一步与 Lean 无关, 目的是**排掉"题目本身有坑"**的可能: 如果题目陈述
     本身是错的, 那么任何 Lean 代码都不可能真的证明它, 此时 verifier 必须
     判负而不是替 agent 圆场。
  2. 把 agent 交的 Lean 源码交给 `check_lean` 真编译, 要求:
        proved = True 且 sorry_used = False。
     ★ 为什么不能只看 exit code ★: Lean 遇到 `sorry` 仍会 exit 0, 只报 warning。
     而且更隐蔽的是"先 axiom 再 theorem"——编译器完全不报错, #print axioms
     也干净。check_lean 两条都查(sorryAx + 源码级 axiom 声明), 这是它
     相对于裸跑 `lake env lean` 的关键增量。
     check_lean 是本仓库的 Lean 后端(机器判定的**外部 oracle**), 不是 agent
     的解题代码 —— 用它不违反"不复用 agent 代码"这条。
  3. 内容判据: 源码里必须真的出现 ZMod 5 与 ZMod 7。
     为什么要这条: 只判"编译通过且无 sorry"的话, 交 `example : True := trivial`
     也能满分。加上"必须真的写了 ZMod 5 和 ZMod 7"之后, 才可能对应题目。

★ 环境不具备时如实判负 ★
Mathlib 缺失(见 _lean.MATHLIB_DIR)时不做任何"假装通过"的处理: 直接
correct=False 并在 details 写明"环境不具备", 自测打印 [SKIP] 后 exit 0。
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

#: 仓库根 = 本文件上溯三层(tests/tools/verifiers_r -> tests/tools -> tests -> root)。
#: 直接 `python res_18_lean_fermat.py` 运行时, sys.path[0] 是本文件所在目录,
#: 仓库根不在里面, `import agent.tools...` 会失败 -> Lean 后端形同不存在,
#: 自测会误打印 [SKIP]。所以先把根补进去。
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from agent.tools.research._lean import check_lean, lean_available, mathlib_available
    _LEAN_IMPORT_ERROR = None
except Exception as exc:                     # 后端模块本身不可用
    check_lean = None                          # type: ignore[assignment]
    lean_available = mathlib_available = None  # type: ignore[assignment]
    _LEAN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

#: 从制品里取 Lean 源码的候选键名
_SRC_KEYS = ("lean", "lean_code", "lean_source", "lean4", "code", "source", "src")
#: 编译器超时(秒)
TIMEOUT = 300


def _extract_source(art: dict) -> str | None:
    for k in _SRC_KEYS:
        v = art.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def math_selfcheck() -> dict:
    """自己把费马小定理的两个具体情形算一遍(防题目本身有坑)。"""
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        d = 2
        while d * d <= n:
            if n % d == 0:
                return False
            d += 1
        return True

    return {
        "p5_prime": is_prime(5),
        "p7_prime": is_prime(7),
        "2^5_mod_5": pow(2, 5, 5),
        "2^7_mod_7": pow(2, 7, 7),
        "holds_5": pow(2, 5, 5) == 2,
        "holds_7": pow(2, 7, 7) == 2,
    }


def verify(stdout: str, code: str, claim: str) -> dict:
    chk = Check()
    ms = math_selfcheck()
    for k, v in ms.items():
        chk.note(k, v)
    # 题目前提: 5,7 是素数且 2^5≡2 (mod 5), 2^7≡2 (mod 7) 真的成立
    chk.expect("statement_selfcheck",
               ms["p5_prime"] and ms["p7_prime"]
               and ms["holds_5"] and ms["holds_7"],
               f"题目本身不成立: {ms} —— 不可能存在真的 Lean 证明")

    chk.note("lean_backend_import_error", _LEAN_IMPORT_ERROR)

    # ---- 环境闸门 ----
    if check_lean is None or mathlib_available is None:
        chk.expect("env_lean_backend", False,
                   f"环境不具备: 无法导入 Lean 后端({_LEAN_IMPORT_ERROR})")
        return _finish(chk, {})
    if not lean_available():
        chk.expect("env_lean_installed", False,
                   "环境不具备: Lean 4 工具链未安装, 不能做机器判定")
        return _finish(chk, {})
    if not mathlib_available():
        chk.expect("env_mathlib", False,
                   "环境不具备: Mathlib 不可用(见 _lean.MATHLIB_DIR), "
                   "无法编译 import Mathlib 的代码 —— 不做任何假装通过的判断")
        return _finish(chk, {})
    chk.expect("env_mathlib", True)
    chk.note("stdout_verdict", parse_verdict(stdout))

    # ---- 制品 ----
    art = parse_artifact(stdout, "CONSTRUCT") or parse_artifact(stdout, "ANSWER")
    if not chk.expect("artifact_present", art is not None,
                      "未提交 CONSTRUCT 制品(需要 {\"lean\": \"...\"})"):
        return _finish(chk, {})
    assert art is not None

    src = _extract_source(art)
    if not chk.expect("lean_source_present", src is not None,
                      "制品里没有 Lean 源码(键名建议用 lean / lean_code)"):
        return _finish(chk, {})
    assert src is not None

    # ---- 内容判据: 必须真的写了 ZMod 5 / ZMod 7 ----
    has5 = re.search(r"ZMod\s+5\b", src) is not None
    has7 = re.search(r"ZMod\s+7\b", src) is not None
    chk.note("mentions_ZMod_5", has5)
    chk.note("mentions_ZMod_7", has7)
    chk.expect("covers_both_primes", has5 and has7,
               "源码里没有同时用到 ZMod 5 与 ZMod 7 —— 题目要求两个素数都要验证")

    # ---- 真编译 ----
    # 源码没写 import 时补上 import Mathlib: ZMod 在 Mathlib 里,
    # 少这一行会报 unknown identifier, 那是环境的锅不是 agent 的锅。
    code_to_compile = src
    if re.search(r"^\s*import\s", src, re.MULTILINE) is None:
        code_to_compile = "import Mathlib\n\n" + src

    res = check_lean(code_to_compile, timeout=TIMEOUT)
    chk.note("lean_exit_code", res.get("exit_code"))
    chk.note("lean_errors", res.get("errors")[:3])
    chk.note("lean_warnings", res.get("warnings")[:3])
    chk.note("lean_axioms", res.get("axioms")[:3])
    chk.expect("lean_proved", bool(res.proved),
               "Lean 编译未证明(有 error / 非 0 退出 / 用了 sorry / 自带 axiom): "
               + "; ".join((res.get("errors") or [])[:2]
                           + (res.get("warnings") or [])[:2]))
    chk.expect("no_sorry", not res.sorry_used,
               "证明里用了 sorry 或自行声明的 axiom —— 这不算证明")

    return _finish(chk, {"proved": bool(res.proved),
                         "sorry_used": bool(res.sorry_used),
                         "checks_p5": ms["holds_5"], "checks_p7": ms["holds_7"]})


def _finish(chk: Check, extra: dict) -> dict:
    correct = chk.correct
    if correct:
        cert = ("Lean 4 + Mathlib 编译通过且无 sorry/无自带 axiom; "
                "独立复核 2^5 mod 5 == 2、2^7 mod 7 == 2 均成立")
    else:
        cert = chk.reason()
    final = json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else None
    return {"correct": correct, "certificate": cert, "final_answer": final,
            "details": dict(chk.details)}


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time

    if check_lean is None or not mathlib_available() or not lean_available():
        print("[SKIP] 环境不具备: Mathlib/Lean 不可用, 无法做机器判定, "
              "verifier 如实返回 False。跳过自测。")
        raise SystemExit(0)

    # ---- 正例: 真证明 ----
    good_lean = (
        "import Mathlib\n\n"
        "theorem fermat_two_five : (2 : ZMod 5) ^ 5 = 2 := by decide\n\n"
        "theorem fermat_two_seven : (2 : ZMod 7) ^ 7 = 2 := by decide\n"
    )
    t0 = _time.time()
    r = verify("VERDICT: PROVED\nCONSTRUCT: " + json.dumps({"lean": good_lean}) + "\n",
               "", "")
    assert r["correct"], f"正例应通过: {r}"
    print(f"[OK] 正例: 两个 ZMod 定理 decide 通过({_time.time() - t0:.1f}s) ->", r["certificate"])

    # ---- 反例 A: 用 sorry 顶着(退出码仍是 0, 这是最容易被骗的地方) ----
    sorry_lean = (
        "import Mathlib\n\n"
        "theorem fermat_two_five : (2 : ZMod 5) ^ 5 = 2 := by sorry\n\n"
        "theorem fermat_two_seven : (2 : ZMod 7) ^ 7 = 2 := by sorry\n"
    )
    r = verify("VERDICT: PROVED\nCONSTRUCT: " + json.dumps({"lean": sorry_lean}) + "\n",
               "", "")
    assert not r["correct"] and not r["details"]["no_sorry"], r
    print("[OK] 反例A: sorry 版 ->", r["certificate"])

    # ---- 反例 B: 自带 axiom(编译干净、无 warning, 但等于把结论当前提) ----
    axiom_lean = (
        "import Mathlib\n\n"
        "axiom cheat : (2 : ZMod 5) ^ 5 = 2\n"
        "theorem fermat_two_five : (2 : ZMod 5) ^ 5 = 2 := cheat\n\n"
        "theorem fermat_two_seven : (2 : ZMod 7) ^ 7 = 2 := by decide\n"
    )
    r = verify("VERDICT: PROVED\nCONSTRUCT: " + json.dumps({"lean": axiom_lean}) + "\n",
               "", "")
    assert not r["correct"] and not r["details"]["no_sorry"], r
    print("[OK] 反例B: 自带 axiom 版 ->", r["certificate"])

    # ---- 反例 C: 只验证了 p=5, 没验证 p=7(编译能过也不够) ----
    partial_lean = (
        "import Mathlib\n\n"
        "theorem fermat_two_five : (2 : ZMod 5) ^ 5 = 2 := by decide\n"
    )
    r = verify("VERDICT: PROVED\nCONSTRUCT: " + json.dumps({"lean": partial_lean}) + "\n",
               "", "")
    assert not r["correct"] and not r["details"]["covers_both_primes"], r
    print("[OK] 反例C: 只验证 p=5 ->", r["certificate"])

    print("[OK] 全部自测通过")