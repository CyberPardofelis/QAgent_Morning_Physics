# -*- coding: utf-8 -*-
"""反向验证: 篇幅配额与占位符护栏必须真的咬得住。

同一套方法论: 基线对照 / 锚点唯一性 / 变异生效判据 / pycache 隔离。
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\pythonProject\QAgent")
TEST = ROOT / "tests" / "test_write_quota.py"
ORG = ROOT / "agent" / "tools" / "surveys" / "_organize.py"
GEN = ROOT / "agent" / "tools" / "surveys" / "survey_generate.py"
PRM = ROOT / "agent" / "prompts_survey.py"

PY = str(ROOT / "venv" / "Scripts" / "python.exe")
env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT)
env["PYTHONPYCACHEPREFIX"] = str(ROOT / "output" / "_pycache_rv4")


def run_tests() -> tuple[bool, str]:
    p = subprocess.run(
        [PY, "-X", "utf8", "-m", "pytest", str(TEST), "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", timeout=600,
    )
    return p.returncode == 0, (p.stdout or "") + (p.stderr or "")


print("=" * 78)
print("步骤 0: 基线对照")
print("=" * 78)
ok, out = run_tests()
if not ok:
    print("基线不绿 —— 判 FAIL")
    print(out[-3000:])
    sys.exit(1)
print("基线全绿\n")

FILES = {ORG: None, GEN: None, PRM: None}
for f in FILES:
    FILES[f] = f.read_text(encoding="utf-8")
    (ROOT / "output" / f"_rv4_{f.name}.bak").write_text(FILES[f], encoding="utf-8")

MUTATIONS: list[tuple[str, str, Path, str, str]] = [
    ("M1", "命题数兜底失效: 配额只看主题摊",
     ORG, "    by_claim = max(1, int(n_claims)) * CHARS_PER_CLAIM\n    return max(300, by_theme, by_claim)",
     "    return max(300, by_theme)"),

    ("M2", "每条命题保底缩到 1 字符(等于没有兜底)",
     ORG, "CHARS_PER_CLAIM = 110", "CHARS_PER_CLAIM = 1"),

    ("M3", "反推系数回到 0.32(永远撞下限)",
     ORG, "CHARS_PER_FACT = 9.0", "CHARS_PER_FACT = 0.32"),

    ("M4", "占位符正则失效(模型可以继续糊弄)",
     ORG, 'r"(后续略|以命题覆盖为准|后略|详见下|以下略|（略）|\\(略\\)|余略|此处省略|省略号代替)"',
     'r"(永远不会匹配的东西xyz)"'),

    ("M5", "剔除逻辑失效: 直接原样返回",
     ORG, '    if not text or not _placeholder_spans(text):\n        return text or "", 0',
     '    if True:\n        return text or "", 0'),

    ("M6", "剔除失效: 计数不累加(丢句但用户看不见)",
     ORG, "        body, ph = _strip_placeholder_sentences(body)\n        dropped_total += ph",
     "        body, ph = _strip_placeholder_sentences(body)"),

    ("M7", "写作链路不接剔除器(占位符仍会静默进正文)",
     ORG, "        body, ph = _strip_placeholder_sentences(body)\n        dropped_total += ph",
     "        ph = 0"),

    ("M8", "prompt 去掉「写不下怎么办」",
     PRM, "【写不下怎么办 —— 10-08 实测踩过的坑】",
     "【(无)】"),

    ("M9", "_phase_write 不用新配额函数",
     GEN, "        (t.get(\"name\") or \"\").strip(): per_theme_target_chars(",
     "        (t.get(\"name\") or \"\").strip(): (lambda *a, **k: 300)("),
]

results = []
try:
    for code, desc, path, anchor, mutated in MUTATIONS:
        original = FILES[path]
        n = original.count(anchor)
        if n != 1:
            results.append((code, desc, False, f"锚点出现 {n} 次(须为 1), 跳过"))
            print(f"{code} 跳过: 锚点 {n} 次")
            continue
        new = original.replace(anchor, mutated, 1)
        if new == original or mutated not in new:
            results.append((code, desc, False, "变异未生效"))
            print(f"{code} FAIL(未生效)")
            continue
        try:
            ast.parse(new)
        except SyntaxError as e:
            results.append((code, desc, False, f"语法错误: {e}"))
            print(f"{code} FAIL(语法)")
            continue

        # 把改动写进去, 其余文件保持原样
        path.write_text(new, encoding="utf-8")
        try:
            ok_run, out_run = run_tests()
        finally:
            path.write_text(original, encoding="utf-8")

        if ok_run:
            results.append((code, desc, False, "测试仍全绿 —— 护栏没咬到"))
            print(f"{code} ✗ 没咬到: {desc}")
        else:
            failed = [ln for ln in out_run.splitlines() if ln.startswith("FAILED")]
            results.append((code, desc, True, f"{len(failed)} 条变红"))
            print(f"{code} ✔ 咬住: {desc} ({len(failed)} 条变红)")
finally:
    for f, orig in FILES.items():
        f.write_text(orig, encoding="utf-8")
    print("\n源文件已还原")

print()
print("=" * 78)
for code, desc, bit, why in results:
    print(f"{code:<5}{'✔ 咬住' if bit else '✗ 没咬到':<9}{desc}")
    print(f"{'':<14}{why}")
print("-" * 78)
n = sum(1 for *_, b, _ in results if b)
print(f"{n}/{len(results)} 变异被护栏咬住")
sys.exit(0 if n == len(results) else 1)