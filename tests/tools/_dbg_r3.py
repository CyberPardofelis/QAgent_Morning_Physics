"""临时: 验证 R3 的两道判据在 hard-011 的真实 stdout 上分别命中没有。"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agent.tools.research import _provenance as P  # noqa: E402

cid = "hard-011-petersen-chromatic-3"
import hashlib

th = "v2_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
p = os.path.join("data", "research", "working", th, "_evidence.jsonl")
recs = [json.loads(x) for x in io.open(p, encoding="utf-8") if x.strip()]
rec = [r for r in recs if r.get("kind") == "code"][-1]
out = rec.get("stdout") or ""

for c in json.load(io.open("tests/tools/base20_v2_cases.json",
                           encoding="utf-8"))["cases"]:
    if c["id"] == cid:
        claim = c["prompt"]

print("COMPOUND_CLAIM 命中:", bool(P._COMPOUND_CLAIM.search(claim)))
lines = out.splitlines()
print("_PART_TRUE  命中:", [l.strip()[:70] for l in lines
                       if P._PART_TRUE.search(l)])
print("_PART_FALSE 命中:", [l.strip()[:70] for l in lines
                       if P._PART_FALSE.search(l)])
print("R3 返回:", P._refutation_targets_wrong_subject(
    rec.get("code") or "", out, claim))