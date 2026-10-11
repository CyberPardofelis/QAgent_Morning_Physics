"""把 DS 给的两套新题库落进 benchmark/, 并修掉已知的 JSON 语法错。

★ 落库前先验 JSON 合法 ★ 这是最基础的一道复算闸门 ——
DS 给的 part1 文件里写了 "argmax": [8/5, 6/5], 而 JSON 没有除法表达式,
整个文件 json.loads 直接炸。题库连解析都过不去, 后面的验证器无从谈起。
"""
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = r"C:\Users\Nightingale\.minimax\v2\assets\2026\10\09"
SRC = {
    "part1_tool_capability.json":
        ASSETS + r"\15-05-07-859-asset_20261009-150507-859_"
                 r"75a846261199_d0d94788-BASEPROBLEMS_BASE.txt",
    "part2_research_capability.json":
        ASSETS + r"\15-05-07-863-asset_20261009-150507-863_"
                 r"e7a9ff794cb4_473b9809-BASEPROBLEMS_RES.txt",
}

#: 已发现的语法错: JSON 不支持算术表达式
#: z3-03-optimize 的 argmax 写成了 [8/5, 6/5] -> 改成等值的 [1.6, 1.2]
JSON_FIXES = [
    ('"argmax": [8/5, 6/5]', '"argmax": [1.6, 1.2]'),
]

dst_dir = os.path.join(ROOT, "benchmark")
os.makedirs(dst_dir, exist_ok=True)

for name, path in SRC.items():
    raw = io.open(path, encoding="utf-8").read()
    applied = []
    for old, new in JSON_FIXES:
        if old in raw:
            raw = raw.replace(old, new)
            applied.append(f"{old}  ->  {new}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[FAIL] {name}: {e}")
        continue

    out = os.path.join(dst_dir, name)
    io.open(out, "w", encoding="utf-8", newline="\n").write(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    ids = [it["id"] for it in data["items"]]
    print(f"[OK ] {name}")
    print(f"       part={data['part']} items={len(ids)} 去重后={len(set(ids))}")
    if applied:
        for a in applied:
            print(f"       修了语法错: {a}")
    tools = sorted({it.get("tool_under_test") for it in data["items"]
                    if it.get("tool_under_test")})
    print(f"       tool_under_test: {tools}")