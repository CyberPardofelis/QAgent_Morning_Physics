import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
try:
    d = json.load(open(r"D:\pythonProject\QAgent\data\research\v2_results.json",
                       encoding="utf-8"))
except FileNotFoundError:
    print("尚无结果")
    raise SystemExit
rs = d.get("results", d) if isinstance(d, dict) else d
print(f"完成 {len(rs)}/20")
for i, r in enumerate(rs):
    print(f"{i+1:2d} {r['id']:34s} {r['verdict']:14s} gold={r['gold']:10s} "
          f"{r['seconds']:>6}s {r['tokens']:>6}tok")