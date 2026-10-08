# -*- coding: utf-8 -*-
"""反向验证: fact 抽取「本篇独有性」约束的护栏必须真的咬得住。

变异守则(沿用本项目已验证的那套):
  1. 基线对照 —— 变异前先跑, 基线不绿直接判 FAIL
  2. 锚点自检 —— `src.count(anchor)` 必须恰好为 1
  3. 变异生效判据 —— 新文本与原文不同 且 变异体确实出现
     (不能用"旧锚点消失": 插入型变异的新文本仍包含旧锚点, 会假 FAIL)
  4. PYTHONPYCACHEPREFIX 隔离字节码缓存
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\pythonProject\QAgent")
TEST = ROOT / "tests" / "test_fact_own_contribution.py"
SRC = ROOT / "agent" / "prompts_survey.py"

PY = str(ROOT / "venv" / "Scripts" / "python.exe")
env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT)
env["PYTHONPYCACHEPREFIX"] = str(ROOT / "output" / "_pycache_rv2")


def run_tests() -> tuple[bool, str]:
    p = subprocess.run(
        [PY, "-X", "utf8", "-m", "pytest", str(TEST), "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", timeout=600,
    )
    return p.returncode == 0, (p.stdout or "") + (p.stderr or "")


print("=" * 78)
print("步骤 0: 基线对照(变异前必须全绿)")
print("=" * 78)
ok, out = run_tests()
if not ok:
    print("基线不绿 —— 护栏本来就不存在, 直接判 FAIL")
    print(out[-3000:])
    sys.exit(1)
print("基线全绿\n")

ORIGINAL = SRC.read_text(encoding="utf-8")
(ROOT / "output" / "_rv_prompts.bak").write_text(ORIGINAL, encoding="utf-8")

MUTATIONS: list[tuple[str, str, str, str]] = [
    ("M1", "整条「本篇独有性」铁律被删",
     "6. **本篇独有性: 优先抽这篇论文自己的贡献**。",
     "6. (占位)"),

    ("M2", "去掉正反例里的反例(只留禁令)",
     '   ❌ "某某方法可以做到…" / "经典方法要求…" / "某技术是一种高效方法"(纯定义)',
     "   (无反例)"),

    ("M3", "去掉「抽不到就少抽」的数量台阶(约束会被 4-12 条下限顶掉)",
     "   ⚠️ **抽不到就少抽** —— 本条优先于上面第 5 条的 4-12 条下限。宁可交 3 条\n"
     "   干净的, 也不要交 4 条里混一条\"经典阴影是一种高效方法\"。",
     "   (不豁免数量)"),

    ("M4", "去掉可操作的判据, 只剩口号",
     "   判据: 把这句话给一个只读过摘要的读者看, 他能不能说\"这是这篇论文报告的\"?\n"
     "   能 -> 抽; 只是让他听懂了某个已有方法 -> 别抽, 换一个更有本篇特色的内容。",
     "   (无判据)"),

    ("M5", "schema 夹带: 新增归属字段",
     '   "support_quote": "<原文完整相关句>"}}',
     '   "support_quote": "<原文完整相关句>",\n'
     '   "attribution": "own | prior"}}'),

    ("M6", "动 fact_type 枚举(下游多处按值分支)",
     "  - hypothesis:  假设/猜想",
     "  - hypothesis:  假设/猜想\n  - own:         本文独有贡献"),

    ('M7', '拆掉铁律 2 的主语要求(有人会想「把归属写进 claim 里」)',
     "而且主语必须是方法/概念/现象本身, 不是\"本文/该工作/作者/我们\"。",
     "主语随意。"),
]

results: list[tuple[str, str, bool, str]] = []
try:
    for code, desc, anchor, mutated in MUTATIONS:
        n = ORIGINAL.count(anchor)
        if n != 1:
            results.append((code, desc, False, f"锚点出现 {n} 次(必须为 1), 跳过"))
            print(f"{code} 跳过: 锚点出现 {n} 次")
            continue
        ast.parse(ORIGINAL)
        new = ORIGINAL.replace(anchor, mutated, 1)
        if new == ORIGINAL or mutated not in new:
            results.append((code, desc, False, "变异没有真正生效"))
            print(f"{code} FAIL(变异未生效)")
            continue
        try:
            ast.parse(new)
        except SyntaxError as e:
            results.append((code, desc, False, f"变异后语法错误: {e}"))
            print(f"{code} FAIL(语法)")
            continue

        SRC.write_text(new, encoding="utf-8")
        try:
            ok_run, out_run = run_tests()
        finally:
            SRC.write_text(ORIGINAL, encoding="utf-8")

        if ok_run:
            results.append((code, desc, False, "测试仍全绿 —— 护栏没咬到"))
            print(f"{code} ✗ 没咬到: {desc}")
        else:
            failed = [ln for ln in out_run.splitlines() if ln.startswith("FAILED")]
            results.append((code, desc, True, f"{len(failed)} 条测试变红"))
            print(f"{code} ✔ 咬住: {desc}  ({len(failed)} 条变红)")
finally:
    SRC.write_text(ORIGINAL, encoding="utf-8")
    print("\n源文件已还原")

print()
print("=" * 78)
print(f"{'编号':<6}{'结果':<8}说明")
print("=" * 78)
for code, desc, bit, why in results:
    print(f"{code:<6}{'✔ 咬住' if bit else '✗ 没咬到':<8}{desc}")
    print(f"{'':<14}{why}")
print("-" * 78)
n_bit = sum(1 for *_, b, _ in results if b)
print(f"{n_bit}/{len(results)} 变异被护栏咬住")
sys.exit(0 if n_bit == len(results) else 1)