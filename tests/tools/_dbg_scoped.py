"""临时: 验证 _SCOPED_TOLERANCE 在 20 道真题上的触发情况。

双向检查: 该触发的要触发, 不该触发的(尤其全域证明题)不能触发 ——
误触发会让「证明 sqrt2 无理数」这种题被暗示可以判 PROVED, 那是在制造假 PROVED。
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agent.prompts_research import _SCOPED_TOLERANCE  # noqa: E402

cases = json.load(io.open("tests/tools/base20_v2_cases.json",
                          encoding="utf-8"))["cases"]
for c in cases:
    hit = bool(_SCOPED_TOLERANCE.search(c["prompt"]))
    print(f"{'HIT ' if hit else '    '}{c['id']:38} {c['prompt'][:70]}")
print("\n共触发", sum(1 for c in cases
                      if _SCOPED_TOLERANCE.search(c["prompt"])), "/", len(cases))