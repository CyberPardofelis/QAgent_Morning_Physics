import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
spec = json.loads(
    Path(r"D:\pythonProject\QAgent\baseproblems_V2.txt").read_text(encoding="utf-8")
)
targets = {"hard-019-no-flt-n3-small", "hard-017-100-queens-sat",
           "hard-013-bbp-pi-hex-digit", "hard-012-erdos-straus-1000"}
claims = {}
for p in spec["items"]:
    if p["id"] in targets:
        claims[p["id"]] = p.get("prompt", "")
        print(p["id"], "->", p.get("prompt"))

print("\n" + "=" * 62)
root = Path(r"D:\pythonProject\QAgent\data\research\working")
for cid, claim in claims.items():
    for d in root.glob("*"):
        st = d / "_status.json"
        if not st.is_file():
            continue
        q = json.load(open(st, encoding="utf-8")).get("question", "").strip()
        if q != claim.strip():
            continue
        print(f"\n########## {cid}  ({d.name})")
        for ln in open(d / "_evidence.jsonl", encoding="utf-8", errors="replace"):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("kind") != "code":
                continue
            print(f"--- attempt={r.get('attempt')} verdict={r.get('verdict')}")
            print("reason:", (r.get("reason") or "")[:200])
            print("stdout tail:")
            print((r.get("stdout") or "")[-1200:])
        break