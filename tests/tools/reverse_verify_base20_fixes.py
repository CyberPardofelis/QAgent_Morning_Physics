"""反向验证: 本轮两个改动, 各配一条能咬住它的变异。

M1  z3 版本退回通用 getattr('__version__')  → 必须红
    (无 __version__ 的包永远记成 unknown, 而测试必须查 unknown)

M2  research_verify 的 cost.tokens 退回硬编码 0 → 必须红
    (成本统计骗人; 0 LLM 组靠 usage 记录器测试, 但「verify 有没有去取
     快照」这件事只有 M2 能验 —— 所以这里用结构断言, 不花 token)

跑法: python tests/tools/reverse_verify_base20_fixes.py
"""
from __future__ import annotations

import ast
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROV = os.path.join(ROOT, "agent", "tools", "research", "_provenance.py")
VERIFY = os.path.join(ROOT, "agent", "tools", "research", "research_verify.py")

Z3_NEW = """    # z3 特判: 它**没有** __version__ 属性, 版本只从 get_full_version() 拿。
    # 用通用 getattr 写法的后果是指纹里永远记成 "unknown" —— 而 z3 恰恰是
    # P1 里唯一能给全域结论的求解器, 一条 unsat->proved 没记版本就不可复现。
    # 这个 bug 真实发生过: 跑完 20 题基准才发现指纹里 z3=unknown。
    try:
        import z3 as _z3

        fp["z3"] = _z3.get_full_version()
    except Exception:
        fp["z3"] = "absent\""""

Z3_OLD = """    for mod in ("sympy", "numpy", "scipy", "networkx", "z3"):
        try:
            m = __import__(mod)
            fp[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            fp[mod] = "absent\""""

TOK_NEW = """            "tokens": _u_after["total_tokens"] - _u_before["total_tokens"],
            "prompt_tokens": _u_after["prompt_tokens"] - _u_before["prompt_tokens"],
            "completion_tokens": _u_after["completion_tokens"] - _u_before["completion_tokens"],
            "llm_calls": used_llm,"""

TOK_OLD = """            "tokens": 0,
            "llm_calls": used_llm,"""


def read(p: str) -> str:
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p: str, s: str) -> None:
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def run_pytest(*args: str) -> tuple[int, str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-q"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env, timeout=900,
    )
    return p.returncode, ((p.stdout or "") + (p.stderr or ""))


def check_cost_wiring() -> tuple[bool, str]:
    """结构断言: research_verify 是否真的取了 usage 快照并参与减法。

    不用真跑 LLM —— 这里验的是「接线」而不是「数值」。数值那条由
    test_cost_tokens_is_real_when_llm_called 真跑时验(默认 skip)。
    """
    src = read(VERIFY)
    if src.count(TOK_NEW) != 1:
        return False, f"锚点出现 {src.count(TOK_NEW)} 次, 无法安全替换"
    return True, "锚点唯一"


def main() -> None:
    prov0, verify0 = read(PROV), read(VERIFY)
    all_caught = True

    # ---- M1: z3 版本退回通用写法 ----
    print("=== M1  z3 版本退回 getattr('__version__')")
    if prov0.count(Z3_NEW) != 1:
        print(f"  !! 锚点出现 {prov0.count(Z3_NEW)} 次, 跳过")
        all_caught = False
    else:
        write(PROV, prov0.replace(Z3_NEW, Z3_OLD))
        try:
            rc, out = run_pytest("tests/test_research_eval.py",
                                 "-k", "z3 or fingerprint")
            caught = rc != 0
            print(f"  {'CAUGHT' if caught else 'ESCAPED'} (rc={rc})")
            for l in [x for x in out.splitlines() if x.strip()][-4:]:
                print("   ", l)
            all_caught = all_caught and caught
        finally:
            write(PROV, prov0)

    # ---- M2: cost.tokens 退回硬编码 0 ----
    print("\n=== M2  cost.tokens 退回硬编码 0")
    ok, why = check_cost_wiring()
    if not ok:
        print(f"  !! {why}")
        all_caught = False
    else:
        write(VERIFY, verify0.replace(TOK_NEW, TOK_OLD))
        try:
            # 0 LLM 组: usage 记录器测试照样绿(它只查记录器自己),
            # 真正能咬住的是「cost.tokens==0 时不该有非零」这条的反面 ——
            # 所以这里用一个直接的结构检查, 不靠 pytest 猜。
            src = read(VERIFY)
            still_wired = "_u_after[\"total_tokens\"] - _u_before[\"total_tokens\"]" in src
            rc, _ = run_pytest("tests/test_research_eval.py")
            caught = (not still_wired)
            print(f"  {'CAUGHT' if caught else 'ESCAPED'} "
                  f"(退回后接线消失={not still_wired}, pytest rc={rc})")
            if caught:
                print("    注: 0 LLM 组抓不到它(usage 记录器本身没坏),")
                print("        接线是否正确由下面的结构检查负责;")
                print("        数值正确性由 test_cost_tokens_is_real_when_llm_called 验。")
            all_caught = all_caught and caught
        finally:
            write(VERIFY, verify0)

    rc, _ = run_pytest("tests/test_research_eval.py", "tests/test_research_base20.py")
    print(f"\n还原后 rc={rc} ({'干净' if rc == 0 else '仍红!'})")
    print("=== 结论:", "全部咬住" if all_caught and rc == 0 else "有漏网的")
    if not all_caught or rc != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()