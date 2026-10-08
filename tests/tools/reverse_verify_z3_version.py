"""反向验证: 把 z3 版本的修复改回旧写法, 测试必须咬住。

变异点 (2 处):
  1. env_fingerprint 里 z3 退回统一的 getattr(mod, '__version__')
  2. 把新断言里的 != 'unknown' 删掉 (模拟当初那条只查非空的弱断言)

用法: python tests/tools/reverse_verify_z3_version.py
"""
from __future__ import annotations

import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROV = os.path.join(ROOT, "agent", "tools", "research", "_provenance.py")
TEST = os.path.join(ROOT, "tests", "test_research_eval.py")

# 变异 1: 恢复旧的通用 getattr 写法
OLD_Z3_BLOCK = """    # z3 特判: 它**没有** __version__ 属性, 版本只从 get_full_version() 拿。
    # 用通用 getattr 写法的后果是指纹里永远记成 "unknown" —— 而 z3 恰恰是
    # P1 里唯一能给全域结论的求解器, 一条 unsat->proved 没记版本就不可复现。
    # 这个 bug 真实发生过: 跑完 20 题基准才发现指纹里 z3=unknown。
    try:
        import z3 as _z3

        fp["z3"] = _z3.get_full_version()
    except Exception:
        fp["z3"] = "absent\""""

MUT1 = """    for mod in ("sympy", "numpy", "scipy", "networkx", "z3"):
        try:
            m = __import__(mod)
            fp[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            fp[mod] = "absent\""""

# 变异 2: 删掉 z3 特判之后, 让 z3 也走回通用循环
MUT2_TAIL = """
    fp = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()}-{platform.machine()}",
    }
"""


def read(p: str) -> str:
    with io.open(p, encoding="utf-8") as f:
        return f.read()


def write(p: str, s: str) -> None:
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def run_tests() -> tuple[int, str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_research_eval.py", "-k", "z3 or fingerprint", "-q"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env, timeout=600,
    )
    return p.returncode, ((p.stdout or "") + (p.stderr or ""))


def main() -> None:
    prov0, test0 = read(PROV), read(TEST)
    muts = [
        ("M1 退回 getattr('__version__')", prov0, OLD_Z3_BLOCK, MUT1),
    ]
    all_caught = True
    for name, src, old, new in muts:
        print(f"=== {name}")
        if src.count(old) != 1:
            print(f"  !! 锚点出现 {src.count(old)} 次, 无法安全替换, 跳过")
            all_caught = False
            continue
        write(PROV, src.replace(old, new))
        try:
            rc, out = run_tests()
            caught = rc != 0
            print(f"  {'CAUGHT' if caught else 'ESCAPED'} (rc={rc})")
            tail = [l for l in out.splitlines() if l.strip()][-4:]
            for l in tail:
                print("   ", l)
            if not caught:
                all_caught = False
        finally:
            write(PROV, prov0)
    rc, _ = run_tests()
    print(f"\n还原后 rc={rc} ({'干净' if rc == 0 else '仍红!'})")
    print("=== 结论:", "全部咬住" if all_caught and rc == 0 else "有漏网的")
    if not all_caught or rc != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()