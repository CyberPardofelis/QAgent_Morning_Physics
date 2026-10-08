"""research_lean 工具的端到端测试 —— 三种情况都要判对。

核心不是"能不能编译", 是**判定判据对不对**:
  1. 真证出来了        → proved
  2. 留了 sorry        → inconclusive(不是 proved! 退出码是 0)
  3. 编译不过          → failed

第 2 条是接 Lean 后端最容易踩的坑, 也是这个工具存在的核心意义。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import agent.tools  # noqa: F401
import agent.tools.research  # noqa: F401
from agent.tools.registry import ToolRegistry  # noqa: E402

CASES = [
    ("真定理: 反射",
     "theorem t1 : (2 + 3) * 4 = 20 := by decide",
     "proved"),
    ("真定理: 内建引理",
     "theorem t2 (a : Nat) : a + 0 = a := Nat.add_zero a",
     "proved"),
    ("多条都真证完",
     "theorem t3 : 1 + 1 = 2 := by decide\n"
     "theorem t4 (n : Nat) : n * 1 = n := Nat.mul_one n",
     "proved"),
    ("★ 留 sorry —— 退出码 0 但没证完",
     "theorem t5 (n : Nat) : n * 2 + 1 ≥ 1 := by sorry",
     "inconclusive"),
    ("★ 一条真一条 sorry —— 整体不算证完",
     "theorem t6 : 1 + 1 = 2 := by decide\n"
     "theorem t7 (n : Nat) : n * 2 + 1 ≥ 1 := by sorry",
     "inconclusive"),
    ("命题不成立: 目标本身是假的",
     "theorem t8 : (1 : Nat) = 2 := by decide",
     "failed"),
    ("语法错误",
     "theorem t9 : = = =",
     "failed"),
    ("用了不存在的引理",
     "theorem t10 : True := by exact ThisLemmaDoesNotExist",
     "failed"),
]


def main() -> None:
    tool = ToolRegistry.get("research_lean")
    if tool is None:
        raise SystemExit("research_lean 没注册")

    print("=== research_lean 端到端 ===\n")
    ok = True
    for label, code, expect in CASES:
        r = tool.handler(code=code, claim=label)
        got = r.get("verdict")
        flag = "OK " if got == expect else "!! "
        if got != expect:
            ok = False
        print(f"{flag}{label}")
        print(f"     期望={expect:14} 实得={got:14} "
              f"proved={r.get('proved')} sorry={r.get('sorry_used')}")
        if got != expect:
            print(f"     summary: {str(r.get('summary'))[:180]}")
        print()

    print("=== Lean 没装时的行为 ===")
    # 临时改 PATH 模拟没装 —— 必须如实报 failed, 不能假装
    import agent.tools.research._lean as L
    orig = L._lean_exe
    try:
        L._lean_exe = lambda: None
        r = tool.handler(code="theorem t : True := trivial", claim="x")
        print(f"  没装时 verdict={r.get('verdict')} (期望 failed)")
        if r.get("verdict") != "failed":
            ok = False
        print(f"  summary: {str(r.get('summary'))[:120]}")
    finally:
        L._lean_exe = orig

    print("\n结论:", "全部正确" if ok else "有问题")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()