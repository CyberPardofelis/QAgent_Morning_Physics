import io
import json
import subprocess

URLS = [
    "https://api.github.com/repos/leanprover-community/mathlib4/releases?per_page=5",
    "https://api.github.com/repos/leanprover-community/mathlib4/releases/latest",
]

for u in URLS:
    print("=" * 60)
    print(u)
    p = subprocess.run(["curl.exe", "-sS", "-L", "--connect-timeout", "20",
                        "--max-time", "60", u],
                       capture_output=True)
    txt = p.stdout.decode("utf-8-sig", errors="replace")
    try:
        d = json.loads(txt)
    except Exception as e:
        print("  parse fail:", e, "|", txt[:200])
        continue
    if isinstance(d, dict):
        print("  dict:", d.get("message") or d.get("tag_name"))
        d = [d]
    for rel in d[:5]:
        print(f"  tag={rel.get('tag_name')} name={rel.get('name')}")
        for a in rel.get("assets", [])[:6]:
            print(f"      {a['name']:26} {a['size']/1e6:8.1f}MB")