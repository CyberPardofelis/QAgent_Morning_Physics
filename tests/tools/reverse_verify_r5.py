"""反向验证: 本轮两处改动, 各配能咬住它的变异。

M1  import 符号检查退回只查根模块  → 必须红
M2  体检闸门的空洞真命题检查删掉    → 必须红
    (这正是本轮真实发生过的退化: 加条件触发时差点把这条防线拆掉)

跑法: python tests/tools/reverse_verify_r5.py
"""
from __future__ import annotations

import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SANDBOX = os.path.join(ROOT, "agent", "tools", "research", "_sandbox.py")
PRECHECK = os.path.join(ROOT, "agent", "tools", "research", "_precheck.py")


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


# M1: 把符号检查调用删掉
M1_NEW = """    # ---- 符号存在性检查 (只对 from-import 且有确切模块名的情形) ----
    if not isinstance(node, ast.ImportFrom):
        return
    module = node.module or ""
    root = module.split(".")[0]
    if root not in _ALLOWED_IMPORT_ROOTS or not module:
        return          # 相对导入 / 根模块本身 —— 上面已处理或该拦
    # 相对层级导入(..x)不在沙箱模型内, 不查
    if module.startswith("."):
        return
    for alias in node.names:
        if alias.name == "*":
            continue    # 星号导入没法静态展开, 交给运行时
        _check_symbol_exists(module, alias.name, node.lineno)"""

M1_OLD = """    # ---- 符号存在性检查: 本轮变异, 整块拿掉 ----
    return  # MUTANT"""


def main() -> None:
    s0, p0 = read(SANDBOX), read(PRECHECK)
    all_caught = True

    print("=== M1  import 符号检查拿掉")
    if s0.count(M1_NEW) != 1:
        print(f"  !! 锚点出现 {s0.count(M1_NEW)} 次, 跳过")
        all_caught = False
    else:
        write(SANDBOX, s0.replace(M1_NEW, M1_OLD))
        try:
            rc, out = run_pytest("tests/test_sandbox_import_gate.py")
            caught = rc != 0
            print(f"  {'CAUGHT' if caught else 'ESCAPED'} (rc={rc})")
            for l in [x for x in out.splitlines() if x.strip()][-5:]:
                print("   ", l)
            all_caught = all_caught and caught
        finally:
            write(SANDBOX, s0)

    print("\n=== M2  空洞真命题检查拿掉 (本轮真实发生过的退化)")
    m2_new = """    scope = str(spec.get("scope") or "").strip()
    if scope and _scope_too_narrow(scope):
        return False"""
    m2_old = """    scope = str(spec.get("scope") or "").strip()
    if False and scope:  # MUTANT: 删掉空洞真命题检查
        return False"""
    if p0.count(m2_new) != 1:
        print(f"  !! 锚点出现 {p0.count(m2_new)} 次, 跳过")
        all_caught = False
    else:
        write(PRECHECK, p0.replace(m2_new, m2_old))
        try:
            rc, out = run_pytest(
                "tests/test_research_quantifier.py", "tests/test_research_p2.py")
            caught = rc != 0
            print(f"  {'CAUGHT' if caught else 'ESCAPED'} (rc={rc})")
            for l in [x for x in out.splitlines() if x.strip()][-5:]:
                print("   ", l)
            all_caught = all_caught and caught
        finally:
            write(PRECHECK, p0)

    print("\n=== M3  _looks_clean 整个改成永远放行 (防线被彻底拆掉)")
    m3_new = "    # 3b: 空洞真命题"
    m3_old = "    # 3b: MUTANT 空洞真命题\n    if True:\n        return True\n    # (mutant) 空洞真命题"
    if p0.count(m3_new) != 1:
        print(f"  !! 锚点出现 {p0.count(m3_new)} 次, 跳过")
        all_caught = False
    else:
        write(PRECHECK, p0.replace(m3_new, m3_old))
        try:
            rc, out = run_pytest(
                "tests/test_research_quantifier.py", "tests/test_research_p2.py")
            caught = rc != 0
            print(f"  {'CAUGHT' if caught else 'ESCAPED'} (rc={rc})")
            for l in [x for x in out.splitlines() if x.strip()][-4:]:
                print("   ", l)
            all_caught = all_caught and caught
        finally:
            write(PRECHECK, p0)

    rc, _ = run_pytest("tests/test_sandbox_import_gate.py",
                       "tests/test_research_quantifier.py",
                       "tests/test_research_p2.py",
                       "tests/test_research_base20.py")
    print(f"\n还原后 rc={rc} ({'干净' if rc == 0 else '仍红!'})")
    print("=== 结论:", "全部咬住" if all_caught and rc == 0 else "有漏网的")
    if not all_caught or rc != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()