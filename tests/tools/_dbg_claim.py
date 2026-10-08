"""临时: 打印某题的题面文本。"""
import io
import json
import os
import sys

cid = sys.argv[1]
p = os.path.join("tests", "tools", "base20_v2_cases.json")
for c in json.load(io.open(p, encoding="utf-8"))["cases"]:
    if c["id"] == cid:
        print(c["prompt"])
        break
else:
    sys.exit("没找到 " + cid)