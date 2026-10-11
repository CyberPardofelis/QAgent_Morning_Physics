"""验「stdout 与 exit code 解耦」: 崩了但留下有效制品时不该丢。

模拟 res-004 的现场: z3 打印出反例后抛异常。
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agent.tools.research.research_run_code import _execute  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


#: 打印反例 -> 打印制品行 -> 崩。复刻 res-004 的形态。
#: ★ 这里用纯 Python 而不是真调 z3 ★ 因为要验的是**"崩了但 stdout 有制品
#: 仍被采信"** 这条路径本身; z3 能不能跑通是另一回事, 混在一起会让失败
#: 原因难分(第一版就因为 z3 那段没跑通而误判成"机制没生效")。
CRASH_AFTER_PRINT = "\n".join([
    "import json",
    "print('Counterexample model found: x = -2 | y = 2 | z = 2')",
    "print('float diff = lhs - rhs = -7.0')",
    "print('COUNTEREXAMPLE: ' + json.dumps({'x': -2, 'y': 2, 'z': 2}))",
    "print('VERDICT: REFUTED')",
    "raise RuntimeError('崩在打印之后 —— 模拟 res-004')",
])

#: 崩在打印**之前** —— stdout 什么都没有
CRASH_BEFORE_PRINT = 'raise RuntimeError("一开始就崩")\n'

#: 崩且只打印了噪声, 没有制品也没有 VERDICT
CRASH_WITH_NOISE = '''\
print("6e+47 at [ 4.58e15 -4.58e15 4.58e15]")
raise RuntimeError("算崩了")
'''


def run(code: str):
    from agent.infra.research_workspace import ResearchWorkspace
    tmp = tempfile.mkdtemp(prefix="crashdet_")
    try:
        ws = ResearchWorkspace("crashdet", question="t", root=tmp)
        return _execute(code, ws, purpose="t", timeout=60,
                        claim="对任意实数 x,y,z, x^4+y^4+z^4+xyz >= x^2y^2+y^2z^2+z^2x^2+x+y+z-3",
                        claim_id="c1")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("=" * 72)
    print("stdout 与 exit code 解耦 —— 崩了但留下有效制品时不该丢")
    print("=" * 72)

    print("\n>>> 崩在打印反例之后(res-004 现场)")
    _, ev = run(CRASH_AFTER_PRINT)
    check("不判 failed", ev["verdict"] != "failed", f"实际={ev['verdict']}")
    check("判成 refuted", ev["verdict"] == "refuted", f"实际={ev['verdict']}")
    check("标注了崩溃", "崩溃" in ev["reason"])
    check("制品本身进了 reason",
          "COUNTEREXAMPLE" in ev["reason"] or "反例" in ev["reason"])

    print("\n>>> 崩在打印之前(stdout 为空)")
    _, ev = run(CRASH_BEFORE_PRINT)
    check("仍然判 failed", ev["verdict"] == "failed", f"实际={ev['verdict']}")

    print("\n>>> 崩但只打印噪声(无制品无 VERDICT)")
    _, ev = run(CRASH_WITH_NOISE)
    check("判 failed 或 untested, 但**不**判 proved/refuted",
          ev["verdict"] not in ("proved", "refuted"), f"实际={ev['verdict']}")

    print("\n" + "=" * 72)
    if FAILS:
        print(f"✗ {len(FAILS)} 项不通过: " + ", ".join(FAILS))
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())