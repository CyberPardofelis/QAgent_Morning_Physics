# -*- coding: utf-8 -*-
"""反向验证: 支撑过滤的护栏必须真的咬得住。

方法论(沿用本项目已验证的那套):
  1. **基线对照** —— 变异前先跑一遍, 基线必须全绿。基线不绿说明这条护栏本来
     就不存在, 直接判 FAIL, 不看变异结果。
  2. **锚点自检** —— `src.count(anchor)` 必须恰好为 1。>1 说明变异只改了一半,
     报出来的是"半个变异", 不是真 bug。
  3. **回显判据是"旧锚点是否消失"** —— 不是"新代码是否出现"。新代码出现不能
     证明变异生效(可能别处也有), 旧锚点消失才能。
  4. **PYTHONPYCACHEPREFIX** —— 子进程跑 pytest 必须隔离字节码缓存, 否则
     秒级 mtime 会让它加载旧字节码, 产生假 FAIL。
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\pythonProject\QAgent")
TEST = ROOT / "tests" / "test_claim_support_filter.py"
SRC = ROOT / "agent" / "tools" / "surveys" / "_paradigm.py"

PY = str(ROOT / "venv" / "Scripts" / "python.exe")
env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT)
env["PYTHONPYCACHEPREFIX"] = str(ROOT / "output" / "_pycache_rv")


def run_tests() -> tuple[bool, str]:
    p = subprocess.run(
        [PY, "-X", "utf8", "-m", "pytest", str(TEST), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8", timeout=600,
    )
    return p.returncode == 0, (p.stdout or "") + (p.stderr or "")


# ------------------------------------------------------------------ 基线
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
backup = ROOT / "output" / "_rv_paradigm.bak"
backup.write_text(ORIGINAL, encoding="utf-8")

MUTATIONS: list[tuple[str, str, str]] = [
    # (编号, 描述, 变异代码)
    ("M1", "判定结果全部当 S (等于不过滤)",
     'keep = [p for i, p in enumerate(cand, 1) if verdicts.get(i) == "S"]',
     'keep = [p for i, p in enumerate(cand, 1) if verdicts.get(i) in ("S", "U", "X")]'),

    ("M2", "全判 U 时返回空 paper_ids (会触发成文层整句丢弃)",
     '            out.append(c)\n            continue\n\n        if len(keep) != len(pids):',
     '            c["paper_ids"] = []\n            out.append(c)\n            continue\n\n        if len(keep) != len(pids):'),

    ("M3", "LLM 失败时清空出处 (静默降级)",
     '            logger.warning("[support] 命题 %s 支撑判定失败, 原样保留 %d 篇出处: %s",\n'
     '                           c.get("claim_id"), len(pids), e)\n'
     '            out.append(c)',
     '            logger.warning("[support] 命题 %s 支撑判定失败", c.get("claim_id"))\n'
     '            c["paper_ids"] = []\n            out.append(c)'),

    ("M4", "解析不出时清空出处",
     '            logger.warning("[support] 命题 %s 支撑判定没解析出, 原样保留 %d 篇出处",\n'
     '                           c.get("claim_id"), len(pids))\n'
     '            out.append(c)',
     '            logger.warning("[support] 命题 %s 没解析出", c.get("claim_id"))\n'
     '            c["paper_ids"] = []\n'
     '            out.append(c)'),

    ("M5", "顺手删 fact_ids (会打穿 fact 覆盖率)",
     '        c["support_judged"] = len(cand)\n        c["paper_ids"] = keep',
     '        c["support_judged"] = len(cand)\n'
     '        keep_fids = {f for p in keep for f in [] }\n'
     '        c["fact_ids"] = [fid for fid in (c.get("fact_ids") or []) if fid in keep_fids]\n'
     '        c["paper_ids"] = keep'),

    ("M6", "两道成本闸门同时失效: 每条命题都调 LLM",
     '        if len(cand) < min_papers:\n            out.append(c)\n            continue',
     '        if False:\n            out.append(c)\n            continue'),

    ("M7", "要求表态全覆盖的护栏被删 (漏表态的论文被当成仅使用, 静默删出处)",
     '        if len(seen) < len(cand):',
     '        if False:'),

    ("M8", "非法判定值改回首字母截断 (SUP 被砍成 S 混进支撑)",
     '            verdict = _SUPPORT_ALIASES.get(str(it.get("v") or "").strip().upper(), "")',
     '            verdict = str(it.get("v") or "").strip().upper()[:1]'),

    ("M9", "回归: 用全库 fact 建索引 (别的命题的 fact 混进判定输入)",
     '        for fid in (c.get("fact_ids") or []):\n'
     '            f = fmap.get(fid)',
     '        for f in facts or []:\n'
     '            fid = f.get("fact_id")'),

    ('M10', 'prompt 去掉「必须覆盖全部」的铁律 (回到过度保守的老坑)',
     '必须对全部 {n} 篇都给出判断, 一篇都不能漏, 不确定时按最接近的一类判。',
     '挑出最有把握的那些就行。'),
]

results: list[tuple[str, str, bool, str]] = []
try:
    for code, desc, anchor, mutated in MUTATIONS:
        # 锚点自检
        n = ORIGINAL.count(anchor)
        if n != 1:
            results.append((code, desc, False, f"锚点出现 {n} 次(必须为 1), 跳过"))
            print(f"{code} 跳过: 锚点出现 {n} 次")
            continue
        ast.parse(ORIGINAL)          # 原文件必须本身合法
        new = ORIGINAL.replace(anchor, mutated, 1)
        # 回显判据: 变异确实落在了唯一那处。
        # 注意不能断言"旧锚点消失" —— 对**插入型**变异(如 M2 在 out.append(c)
        # 前加一行), 新文本仍然包含旧锚点本身, 那条断言会假 FAIL。
        # 锚点唯一性(上面 count == 1) + 变异后文本变化 + 新代码存在, 三者合起来
        # 才能保证"只改了这一处, 而且确实改了"。
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