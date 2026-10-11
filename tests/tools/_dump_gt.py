"""临时: 把 part1 40 题的 ground_truth 摊平, 判断"能机械判到什么程度"。

★ 为什么先做这个 ★ C 的判分质量完全取决于 GT 有多机器可读。
  题库里 36 条 `verifier.entrypoint` 指向 `verifiers/part1/*.py`, 而那个目录
  **不存在** —— 所以判据得看 GT 本身能不能直接对。
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "lab", "_gt_dump.txt")

b = json.load(open(os.path.join(ROOT, "data", "lab",
                                "part1_tool_capability.json"), encoding="utf-8"))
its = b["items"]

lines = []
odd = []
for it in its:
    gt = it.get("ground_truth")
    if "prompt" not in it:
        odd.append(it["id"])
    lines.append("=" * 78)
    lines.append(f"{it['id']}   [{it.get('tool_under_test')}]   {it.get('difficulty','')}")
    lines.append(f"  keys   : {sorted(it.keys())}")
    lines.append(f"  prompt : {it.get('prompt')}")
    lines.append(f"  hints  : {it.get('tool_hints')}")
    lines.append(f"  pass   : {(it.get('verifier') or {}).get('pass_condition')}")
    if isinstance(gt, dict):
        for k, v in gt.items():
            lines.append(f"  gt.{k:26} = {json.dumps(v, ensure_ascii=False)[:200]}")
    else:
        lines.append(f"  gt = {gt!r}")
    if it.get("data_ref"):
        lines.append(f"  data_ref = {it['data_ref']}")

lines.append("=" * 78)
lines.append(f"★ 缺 prompt 字段的题: {odd or '无'}")

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("写出", OUT, len(lines), "行")
print("缺 prompt 的题:", odd or "无")
