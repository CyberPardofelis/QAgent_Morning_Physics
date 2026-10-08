import json
import subprocess

u = ("https://api.github.com/repos/leanprover-community/mathlib4/releases"
     "/tags/v4.34.1")
p = subprocess.run(["curl.exe", "-sS", "-L", "--connect-timeout", "20",
                    "--max-time", "60", u], capture_output=True)
d = json.loads(p.stdout.decode("utf-8-sig", errors="replace"))
print("tag:", d.get("tag_name"))
print("body(前 600 字):")
print((d.get("body") or "")[:600])
print()
print("assets:")
for a in d.get("assets", []):
    print(f"  {a['name']:28} {a['size']/1e6:8.1f}MB")
    print(f"      {a['browser_download_url']}")