import re
import sys
from types import SimpleNamespace

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\pythonProject\QAgent")

from agent.tools.research._provenance import (  # noqa: E402
    _AFFIRMING_LINE, _ALL_PASSED, _MARKER_PREFIX, _NOTHING_FOUND,
    _witness_number, judge_verdict,
)

src = open(r"D:\pythonProject\QAgent\tests\tools\check_proved_machine_gate.py",
           encoding="utf-8").read()
m = re.search(r'PELL_STDOUT = """(.*?)"""', src, re.S)
s = m.group(1)

print("stdout 前 3 行:", s.splitlines()[:3])
n = _witness_number(s)
print("witness_number =", n)
print("NOTHING_FOUND =", bool(_NOTHING_FOUND.search(s)))
print("ALL_PASSED =", bool(_ALL_PASSED.search(s)))
for line in s.splitlines():
    if _MARKER_PREFIX in line.upper():
        continue
    if n and n in line and _AFFIRMING_LINE.search(line):
        print("R5c 应命中:", line[:100])

r = SimpleNamespace(stdout=s, stderr="", ok=True, rejected=False,
                    timed_out=False, error_type="", code="x = 1\n")
v, reason = judge_verdict(r, code="x = 1\n",
                          claim="求 Pell 方程 x^2 - 61 y^2 = 1 的最小正整数解 (x,y)。")
print("\n最终判定:", v)
print(reason[:220])