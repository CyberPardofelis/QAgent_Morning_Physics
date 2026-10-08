import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
root = Path(r"D:\pythonProject\QAgent\data\research\working")
needle = sys.argv[1]
for d in sorted(root.glob("*")):
    st = d / "_status.json"
    if not st.is_file():
        continue
    q = json.load(open(st, encoding="utf-8")).get("question", "")
    if needle not in q:
        continue
    print("DIR:", d.name, "|", q[:70])
    ev = d / "_evidence.jsonl"
    for i, l in enumerate(open(ev, encoding="utf-8", errors="replace")):
        if not l.strip():
            continue
        r = json.loads(l)
        print(f"  [{i}] verdict={r.get('verdict')} | {str(r.get('reason'))[:110]}")
        so = (r.get("stdout") or "")[-260:].replace("\n", " | ")
        se = (r.get("stderr") or "")[-260:].replace("\n", " | ")
        if so.strip():
            print("      out:", so)
        if se.strip():
            print("      err:", se)
    break