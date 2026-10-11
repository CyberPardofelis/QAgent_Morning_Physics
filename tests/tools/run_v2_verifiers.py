# -*- coding: utf-8 -*-
"""V2 人工基准验证器跑批 —— 工具的能力天花板。

★ 这个跑批存在的意义 ★
V1 的 `base20_verifiers.py` 人工手写 20 道验证器, 20/20 全过。
它回答了一个工具本身回答不了的问题: **题到底能不能做?**

工具跑 20/20 之前, 你分不清是"题不可做"还是"工具不行" —— 两者的表现都是
判错。有了人工基准做对照, 每次工具跑完都能立刻回答:
    人工做过 / 工具没做  →  工具的锅(选型、prompt、门太严)
    人工也没做           →  题或环境的锅, 别冤枉工具
没有这个对照, 调 prompt 就是盲调。

V2 这套题是按「裸枚举必死」设计的, 所以人工基准尤其重要:
每道 hard 题的正确解法都依赖**方法选型**而非计算量。

用法:
    python tests/tools/run_v2_verifiers.py

★ 不再需要手动设 `$env:PYTHONIOENCODING` ★
子进程 UTF-8 现在由 `run_capture` 强制注入。旧写法是把这个坑绕过去:
跑批时人肉记得设环境变量, 于是**换个终端/换个人跑就静默变乱码** ——
`_marker()` 是子串匹配, 乱码不报错, 只让它读到 None。见 _proc.py。
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time

ROOT = r"D:\pythonProject\QAgent"
VDIR = os.path.join(ROOT, "tests", "tools", "verifiers_v2")

sys.path.insert(0, ROOT)
from agent.tools.research._proc import run_capture  # noqa: E402

#: (题号, 文件名, gold)
PY_CASES = [
    ("hard-007-ramsey-3-4", "hard_07_ramsey.py", "proved"),
    ("hard-011-petersen-chromatic-3", "hard_11_petersen.py", "proved"),
    ("hard-012-erdos-straus-1000", "hard_12_erdos_straus.py", "proved"),
    ("hard-013-bbp-pi-hex-digit", "hard_13_bbp.py", "proved"),
    ("hard-017-100-queens-sat", "hard_17_queens.py", "proved"),
]

#: 两道形式化题走 Lean kernel, 不是 Python 沙箱。
LEAN_CASES = [
    ("hard-015-fermat-little-lean", "proved"),
    ("hard-016-sqrt2-irrational-lean", "proved"),
]


def _marker(out: str) -> str:
    """取输出里最后一条 VERDICT 标记。"""
    found = ""
    for line in out.splitlines():
        up = line.upper()
        if "VERDICT:" in up:
            tail = line[up.index("VERDICT:") + 8:].strip().split()
            if tail:
                found = tail[0]
    return found or "(无 VERDICT)"


def run_python_cases() -> list[dict]:
    rows = []
    for cid, fn, gold in PY_CASES:
        path = os.path.join(VDIR, fn)
        if not os.path.exists(path):
            rows.append({"id": cid, "gold": gold, "got": "MISSING",
                         "seconds": 0.0})
            print(f"  {cid:34s} MISSING ({fn})", flush=True)
            continue
        t0 = time.time()
        p = run_capture([sys.executable, path], cwd=ROOT, timeout=900)
        dt = time.time() - t0
        out = (p.stdout or "") + (p.stderr or "")
        got = _marker(out)
        rows.append({"id": cid, "gold": gold, "got": got.lower(),
                     "seconds": round(dt, 1)})
        print(f"  {cid:34s} {got.lower():10s} {dt:>6.1f}s", flush=True)
    return rows


def run_lean_cases() -> list[dict]:
    sys.path.insert(0, ROOT)
    from agent.tools.research._lean import check_lean, mathlib_available  # noqa

    lean_src = {
        "hard-015-fermat-little-lean": """import Mathlib
example (p : Nat) (hp : p.Prime) (a : Nat) (h : ¬ (p ∣ a)) :
    a ^ (p - 1) ≡ 1 [MOD p] := by
  have h' := Nat.ModEq.pow_totient (x := a) (n := p) (hp.coprime_iff_not_dvd.mpr h).symm
  simpa [Nat.totient_prime hp] using h'
""",
        "hard-016-sqrt2-irrational-lean": """import Mathlib
example : Irrational (Real.sqrt 2) := irrational_sqrt_two
""",
    }
    rows = []
    if not mathlib_available():
        for cid, gold in LEAN_CASES:
            rows.append({"id": cid, "gold": gold, "got": "no_mathlib",
                         "seconds": 0.0})
            print(f"  {cid:34s} no_mathlib", flush=True)
        return rows
    for cid, gold in LEAN_CASES:
        t0 = time.time()
        r = check_lean(lean_src[cid], timeout=900)
        dt = time.time() - t0
        got = "proved" if r.get("proved") else "not_proved"
        rows.append({"id": cid, "gold": gold, "got": got,
                     "seconds": round(dt, 1)})
        print(f"  {cid:34s} {got:10s} {dt:>6.1f}s  "
              f"sorry={r.get('sorry_used')}", flush=True)
    return rows


def main() -> int:
    print("=" * 70)
    print("V2 人工基准跑批 —— 工具的能力天花板")
    print("=" * 70)
    print("\n[Python 沙箱题]")
    py = run_python_cases()
    print("\n[Lean 形式化题]")
    ln = run_lean_cases()

    rows = py + ln
    ok = [r for r in rows if r["got"] == "proved"]
    print("\n" + "=" * 70)
    print(f"人工基准: {len(ok)}/{len(rows)} PROVED")
    for r in rows:
        if r["got"] != "proved":
            print(f"  未通过: {r['id']}  gold={r['gold']} got={r['got']}")
    print("=" * 70)

    out = os.path.join(ROOT, "data", "research", "v2_gold_baseline.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump({"total": len(rows), "proved": len(ok), "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"明细写入 {out}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())