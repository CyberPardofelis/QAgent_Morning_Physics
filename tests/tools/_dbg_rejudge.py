"""临时: 把某题落盘证据按**当前**门重算一遍, 打印理由全文。

用途: 门改过之后, 看已有证据在新门下会被怎么判、理由里的数字对不对。
"""
import hashlib
import io
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agent.tools.research._provenance import judge_verdict  # noqa: E402

cid = sys.argv[1]
th = "v2_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
claim = ""
for c in json.load(io.open("tests/tools/base20_v2_cases.json",
                           encoding="utf-8"))["cases"]:
    if c["id"] == cid:
        claim = c["prompt"]

p = os.path.join("data", "research", "working", th, "_evidence.jsonl")
for i, line in enumerate(io.open(p, encoding="utf-8")):
    if not line.strip():
        continue
    rec = json.loads(line)
    if rec.get("kind") != "code":
        continue
    reason = rec.get("reason") or ""
    res = SimpleNamespace(
        stdout=rec.get("stdout") or "", stderr=rec.get("stderr") or "",
        ok=not ("超时" in reason or "静态门拒绝" in reason
                or "代码执行出错" in reason),
        rejected="静态门拒绝" in reason, timed_out="超时" in reason,
        error_type="", code=rec.get("code") or "",
    )
    v, why = judge_verdict(res, code=rec.get("code") or "", claim=claim)
    print("=" * 66)
    print(f"[第{i}条] 落盘={rec.get('verdict')}  重算={v}")
    print(why)