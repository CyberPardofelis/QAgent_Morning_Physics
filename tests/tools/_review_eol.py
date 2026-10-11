"""一次性工具: 查源码换行符(避免后续 edit 踩 CRLF 坑)。"""
import io

FILES = [
    "agent/tools/research/_flags.py",
    "agent/tools/research/_provenance.py",
    "agent/tools/research/research_verify.py",
    "agent/tools/research/research_run_code.py",
    "tests/tools/run_all60.py",
    "tests/tools/run_r_scored.py",
    "tests/tools/harness_gate.py",
    "tests/tools/problem_spec.py",
]
for f in FILES:
    try:
        b = io.open(f, "rb").read()
    except OSError as e:
        print("  MISS", f, e)
        continue
    crlf = b.count(b"\r\n")
    lf = b.count(b"\n")
    kind = "CRLF" if crlf * 2 > lf else "LF"
    print(f"  {f:48} CRLF={crlf:5}  LF={lf:5}  -> {kind}")