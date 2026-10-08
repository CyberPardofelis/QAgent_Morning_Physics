"""临时: 打印某题所有证据的 verdict/reason/stdout 尾部, 用来定位假判定。"""
import hashlib
import io
import json
import os
import sys

cid = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
th = "v2_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
p = os.path.join("data", "research", "working", th, "_evidence.jsonl")
for i, line in enumerate(io.open(p, encoding="utf-8")):
    if not line.strip():
        continue
    e = json.loads(line)
    print("=" * 70)
    print(f"[{i}] verdict={e['verdict']}")
    print("reason:", e["reason"][:900])
    if e.get("code"):
        print("--- code ---")
        print(e["code"][:2500])
    if e.get("stdout"):
        print("--- stdout tail ---")
        print(e["stdout"][-n:])
    if e.get("stderr"):
        print("--- stderr ---")
        print(e["stderr"][:600])