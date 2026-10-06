"""反向验证: NUL 字节清洗的每条护栏都配一个"改坏它, 测试必须变红"的变异。

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_text_clean.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TC = ROOT / "agent" / "infra" / "text_clean.py"
LLM = ROOT / "agent" / "tools" / "llm.py"
CTRL = ROOT / "agent" / "controller.py"
QE = ROOT / "agent" / "tools" / "surveys" / "_pool" / "_query_expand.py"
TEST = "tests/test_llm_text_clean.py"

# (编号, 说明, 文件, 变异前, 变异后, 期望失败的测试名片段)
MUTATIONS = [
    ("T1", "_call_llm 出口不再清洗 —— 所有下游 JSON 解析点重新裸奔",
     LLM,
     "        out = sanitize_llm_text(out)",
     "        pass",
     "test_call_llm_sanitizes_at_the_boundary"),

    ("T2", "strip_nul 不删 NUL —— 整件事等于没做",
     TC,
     '    if "\\x00" not in text:\n        return text\n    return text.replace("\\x00", "")',
     "    return text",
     "test_strip_nul_rejoins_a_broken_word"),

    ("T3", "strip_nul 把 NUL 换成空格 —— 留下 `Toff oli` 这种看着像真的碎片",
     TC,
     '    return text.replace("\\x00", "")',
     '    return text.replace("\\x00", " ")',
     "test_strip_nul_rejoins_a_broken_word"),

    ("T4", "strip_control_chars 顺手删掉 \\n —— 模型爱在字符串里塞裸换行",
     TC,
     '        if ch in "\\t\\n\\r" or (ord(ch) >= 0x20 and ord(ch) != 0x7F)',
     '        if ord(ch) >= 0x20 and ord(ch) != 0x7F',
     "test_strip_control_chars_keeps_tab_newline_cr"),

    ("T5", "sanitize_llm_text 只删 NUL 不删其余控制字符 —— \\x0b \\x0c 照样炸",
     TC,
     '    if any(ord(c) < 0x20 and c not in "\\t\\n\\r" for c in text) or "\\x7f" in text:\n'
     "        text = strip_control_chars(text)",
     "    pass",
     "test_sanitize_llm_text_actually_removes_nul_and_control_chars"),

    ("T6", "loads_maybe_dirty 不再清洗 —— 读历史产物那条路重新会炸",
     TC,
     "        return json.loads(strip_control_chars(strip_nul(text)), strict=False)",
     "        return json.loads(text, strict=False)",
     "test_loads_maybe_dirty_parses_nul_json"),

    ("T7", "controller 的工具参数不再清洗 —— 工具调用会随机参数解析失败",
     CTRL,
     "                args = json.loads(\n"
     '                    sanitize_llm_text(tool_call["function"]["arguments"]))',
     '                args = json.loads(tool_call["function"]["arguments"])',
     "test_controller_sanitizes_tool_call_arguments"),

    ("T8", "_parse_gap_verdict 退回自己写 replace —— 第二道防线变成局部知识",
     QE,
     "    raw = sanitize_llm_text(raw)",
     '    if "\\x00" in raw:\n        raw = raw.replace("\\x00", "")',
     "test_query_expand_uses_the_shared_sanitizer_not_its_own"),

    ("T9", "清洗顺手把非 ASCII 也删了(典型: 只留可打印 ASCII)—— 中文综述洗成空串",
     TC,
     '        if ch in "\\t\\n\\r" or (ord(ch) >= 0x20 and ord(ch) != 0x7F)',
     "        if ch in '\\t\\n\\r' and ord(ch) < 0x7F",
     "test_sanitize_keeps_chinese_and_emoji"),

    ("T10", "非 str 输入直接抛 —— 下游传 None 进来就崩",
     TC,
     "    if not isinstance(text, str):\n        return text\n    if \"\\x00\" not in text:\n        return text",
     "    if \"\\x00\" not in text:\n        return text",
     "test_strip_nul_never_raises_on_non_str"),
]


def _run_test(frag: str) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q", "-k", frag,
         "-p", "no:cacheprovider", "--no-header", "-x"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "")[-400:]


def main() -> int:
    os.chdir(ROOT)
    results: list[tuple[str, bool, str]] = []
    for mid, desc, path, before, after, frag in MUTATIONS:
        src = path.read_text(encoding="utf-8")
        if before not in src:
            results.append((mid, False, f"变异锚点没找到, 测试无效: {desc}"))
            continue
        with tempfile.TemporaryDirectory() as td:
            bak = Path(td) / path.name
            shutil.copy2(path, bak)
            try:
                path.write_text(src.replace(before, after, 1), encoding="utf-8")
                rc, out = _run_test(frag)
                if rc == 0:
                    results.append(
                        (mid, False, f"改坏后测试仍然绿 —— 这条护栏没咬到: {desc}"))
                else:
                    results.append((mid, True, f"变红 ✓  {desc}"))
            finally:
                shutil.copy2(bak, path)

    print("=" * 74)
    ok = sum(1 for _, good, _ in results if good)
    for mid, good, msg in results:
        print(f"  {'✓' if good else '✗'} {mid:<4} {msg}")
    print("=" * 74)
    print(f"反向验证 {ok}/{len(results)} 生效")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
