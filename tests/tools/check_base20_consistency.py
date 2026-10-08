"""base20_cases.json (20 题元数据) 与 base20_verifiers.py (验证代码) 必须一一对应。

两份文件是同一批题的两面: JSON 存题目/gold/失败模式, py 存验证代码。
它们一旦漂移, 测试就静默失效 —— 比如 JSON 有 20 题而 py 只有 19 段代码,
parametrize 会少跑一题, 报告里看不出来。
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base20_verifiers import GOLD, VERIFIER_CODE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASES_JSON = os.path.join(ROOT, "tests", "base20_cases.json")

with io.open(CASES_JSON, encoding="utf-8") as f:
    data = json.load(f)

json_ids = [c["id"] for c in data["cases"]]
code_ids = list(VERIFIER_CODE.keys())

print(f"JSON 题数: {len(json_ids)}  |  验证代码条数: {len(code_ids)}")

assert len(json_ids) == 20, f"题数不是 20: {len(json_ids)}"
assert len(set(json_ids)) == 20, "JSON 里有重复 id"

missing_code = [i for i in json_ids if i not in VERIFIER_CODE]
missing_json = [c for c in code_ids if c not in json_ids]
assert not missing_code, f"这些题没有验证代码: {missing_code}"
assert not missing_json, f"这些验证代码不在 JSON 里: {missing_json}"

for c in data["cases"]:
    assert c["gold"] == GOLD, f"{c['id']} 的 gold 不是 {GOLD}: {c['gold']}"
    assert c.get("probes"), f"{c['id']} 缺 probes"
    for key in ("domain", "difficulty", "verifier"):
        assert c.get(key), f"{c['id']} 缺字段 {key}"

# 每段验证代码都必须真的会打 VERDICT, 否则跑完拿不到判定。
for cid, code in VERIFIER_CODE.items():
    assert "VERDICT:" in code, f"{cid} 的验证代码没有 VERDICT 标记"

print("\nOK: 20 题与 20 段验证代码完全对应")
print("OK: 每段都带 VERDICT 标记")
print("OK: gold 全为 proved")