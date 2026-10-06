"""反向验证: 每条护栏都配一个"把它改坏, 对应测试必须变红"的变异。

用户要求(9-28): 测试不能只是"跑绿了就算"。改坏之后不变红, 说明这条断言
根本没咬到被测逻辑。

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_macro_incremental.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG = ROOT / "agent" / "tools" / "surveys" / "_organize.py"
BG = ROOT / "scripts" / "backfill_graph.py"
TEST = "tests/test_survey_structure_layers.py"

# (编号, 说明, 文件, 变异前, 变异后, 期望失败的测试名片段)
MUTATIONS = [
    ("M1", "模板去掉 {existing_block} 占位符 —— 已有板块永远进不了 prompt",
     ORG,
     "【已经归纳出的主题】\n{themes_block}\n{existing_block}\n【怎么聚】",
     "【已经归纳出的主题】\n{themes_block}\n\n【怎么聚】",
     "test_macro_prompt_template_actually_embeds_existing_block"),

    ("M2", "无已有板块时也注入块 —— 污染全量重聚那条老路子的 prompt",
     ORG,
     '    if not ex_lines:\n        return "", max(2, min(max_macros, len(names) // 2))',
     '    if not ex_lines:\n        ex_lines = ["- 「占位板块」(已收编 0 个主题)"]',
     "test_macro_prompt_parts_empty_when_no_existing"),

    ("M3", "n_hint 不再缩小 —— 增量模式下会凭空造满 5 个同类板块",
     ORG,
     "    return block, max(1, min(max_macros, len(names) // 3))",
     "    return block, max(2, min(max_macros, len(names) // 2))",
     "test_macro_prompt_parts_lists_existing_and_shrinks_hint"),

    ("M4", "复用板块不再标 reused —— 调用方会把新板块也当复用, 传空 statement",
     ORG,
     '            "reused": name in ex_names,',
     '            "reused": True,',
     "test_cluster_marks_new_macro_not_reused"),

    ("M5", "把复用判据反过来 —— 新建板块被标成复用",
     ORG,
     '            "reused": name in ex_names,',
     '            "reused": name not in ex_names,',
     "test_cluster_marks_reused_macro"),

    ("M6", "复用板块只装 1 个主题时不再豁免空壳告警 —— 误报回来",
     ORG,
     '        if len(kept) == 1 and name not in ex_names:',
     '        if len(kept) == 1:',
     "test_reused_single_theme_does_not_warn_empty_shell"),

    ("M7", "新建板块的空壳告警被吃掉 —— 反向, 会漏报真问题",
     ORG,
     '        if len(kept) == 1 and name not in ex_names:',
     '        if len(kept) == 1 and name in ex_names:',
     "test_new_single_theme_still_warns_empty_shell"),

    ("M8", "复用提示整条删掉 —— 事后分不清哪个板块是复用的",
     ORG,
     '        elif name in ex_names:\n            warnings.append(f"宏观板块「{name}」复用已有板块, 这次新增 {len(kept)} 个主题")',
     "        elif name in ex_names:\n            pass",
     "test_reused_single_theme_does_not_warn_empty_shell"),

    ("M9", "--macro-only 路径把 --dry 丢掉 —— 又会偷偷发真实请求",
     BG,
     "        n = macro_layer(args.dry)",
     "        n = macro_layer(False)",
     "test_backfill_macro_only_dry_does_not_call_llm"),
]


def _run_test(frag: str) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q", "-k", frag,
         "-p", "no:cacheprovider", "--no-header", "-x"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "")[-600:]


def main() -> int:
    os.chdir(ROOT)
    results: list[tuple[str, bool, str]] = []
    for mid, desc, path, before, after, frag in MUTATIONS:
        src = path.read_text(encoding="utf-8")
        if before not in src:
            results.append((mid, False, f"变异锚点没找到, 测试无效: {desc}"))
            continue
        # 备份必须和原文件**不在同一个目录**: 同名 .bak 落在 scripts/ 下的话,
        # 变异期间 pytest 的收集范围没变但下一次运行会看到它, 容易把自己
        # 的残留当成仓库内容。
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

    print("=" * 72)
    ok = sum(1 for _, good, _ in results if good)
    for mid, good, msg in results:
        print(f"  {'✓' if good else '✗'} {mid}  {msg}")
    print("=" * 72)
    print(f"反向验证 {ok}/{len(results)} 生效")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
