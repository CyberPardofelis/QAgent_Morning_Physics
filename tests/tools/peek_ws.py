import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
d = Path(sys.argv[1])
q = json.load(open(d / "_status.json", encoding="utf-8"))
print("Q:", q["question"])
ev = d / "_evidence.jsonl"
for i, l in enumerate(open(ev, encoding="utf-8", errors="replace")):
    if not l.strip():
        continue
    r = json.loads(l)
    print("=" * 60)
    print(f"[{i}] kind={r.get('kind')} verdict={r.get('verdict')} "
          f"mathlib={r.get('mathlib')}")
    print("REASON:", (r.get("reason") or "")[:250])
    print("--- code ---")
    print((r.get("code") or "")[:800])
    print("--- stdout ---")
    print((r.get("stdout") or "")[:500])
    print("--- stderr ---")
    print((r.get("stderr") or "")[:400])
ap = d / "_attempts.jsonl"
if ap.is_file():
    print("\n### attempts")
    for l in open(ap, encoding="utf-8", errors="replace"):
        if l.strip():
            a = json.loads(l)
            print(f"  [{a.get('attempt')}] {str(a.get('error_type'))[:150]}")