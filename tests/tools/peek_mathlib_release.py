import io
import json

try:
    d = json.load(io.open("mathlib_rel.json", encoding="utf-8"))
except Exception as e:
    print("parse fail:", e)
    print(io.open("mathlib_rel.json", encoding="utf-8").read()[:400])
    raise SystemExit

print("tag:", d.get("tag_name"), "| name:", d.get("name"))
print("message:", d.get("message", "")[:200])
print("assets:")
for a in d.get("assets", []):
    print(f"  {a['name']:30} {a['size']/1e6:8.1f}MB")
    print(f"      {a['browser_download_url']}")