"""反向验证: 逐个把 10-05 的被测逻辑关掉, 确认对应测试会红。

每项: 打补丁 -> 跑指定测试 -> 必须失败 -> 还原 -> 再跑一次 -> 必须通过。
任何一项"打补丁后测试仍绿"或"还原后测试变红", 都说明这条测试其实没锁住东西。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\pythonProject\QAgent")
PY = str(ROOT / "venv" / "Scripts" / "python.exe")
ORG = ROOT / "agent" / "tools" / "surveys" / "_organize.py"
SG = ROOT / "agent" / "tools" / "surveys" / "survey_generate.py"

# (名称, 文件, 关闭逻辑的 old, 还原的 new, 应当变红的测试)
MUTATIONS = [
    (
        "R1 范式层不落盘",
        SG,
        "        if paradigms:\n",
        "        if False:\n",
        "tests/test_survey_resume.py::test_paradigm_success_is_persisted",
    ),
    (
        "R2 关联又把 type 存回去",
        ORG,
        '                rels.append({\n                    "to": (r.get("to") or "").strip(),\n                    "note": note,\n                })',
        '                rels.append({\n                    "to": (r.get("to") or "").strip(),\n'
        '                    "type": (r.get("type") or "support").strip(),\n'
        '                    "note": note,\n                })',
        "tests/test_survey_structure_layers.py::test_relations_drop_type_but_keep_note_verbatim",
    ),
    (
        # 忠实复现 10-05 一开始就避开的那个顺序 bug: 指向校验放回循环内,
        # 此时 seen_theme 只含**已遍历到**的主题, 排在后面的会被误报。
        # 校验代码逐字来自实现里的那条, 只是挪了位置。
        "R3 指向校验挪回循环内(顺序 bug 会误报后置主题)",
        ORG,
        "        _pending_rel_check.append((name, [r[\"to\"] for r in rels]))\n",
        "        _pending_rel_check.append((name, [r[\"to\"] for r in rels]))\n"
        "        for _r in rels:\n"
        "            if _r[\"to\"] not in seen_theme and _r[\"to\"] != name:\n"
        "                warnings.append(f\"主题「{name}」关联指向了不存在的主题: {_r['to']}\")\n",
        "tests/test_survey_structure_layers.py::test_relation_pointing_at_later_theme_is_not_flagged",
    ),
    (
        # 配套: 整个校验循环关掉 -> "该报的要报出来"那条必须红。
        # 两条一起才排除了「not_flagged 那条其实没锁东西 / 校验根本没生效」这两种假绿。
        "R3b 指向校验整个关掉(证明校验是活的)",
        ORG,
        "    for src, tos in _pending_rel_check:\n",
        "    for src, tos in []:\n",
        "tests/test_survey_structure_layers.py::test_relation_pointing_at_nonexistent_theme_is_reported",
    ),
    (
        "R4 宏观层不抢救截断",
        ORG,
        "            salvaged = _salvage_macros(out)\n",
        "            salvaged = None\n",
        "tests/test_survey_structure_layers.py::test_macro_layer_survives_truncated_llm_output",
    ),
    (
        "R5 宏观层不写回 thematic_map",
        SG,
        "    save_thematic_map(ws, theme_map)\n",
        "    pass\n",
        "tests/test_survey_structure_layers.py::test_phase_macros_writes_back_to_thematic_map",
    ),
    (
        "R6 宏观层跳过守卫失效(续跑会重复调 LLM)",
        SG,
        '    if theme_map.get("macros"):\n',
        '    if False:\n',
        "tests/test_survey_structure_layers.py::test_phase_macros_skips_llm_when_already_present",
    ),
    (
        "R7 范式层不记 paradigm 连接键",
        ORG,
        '                "paradigm": pn,\n',
        "",
        "tests/test_survey_paradigm.py::test_paradigms_layer_is_persisted_and_keyed_by_paradigm_name",
    ),
]


def run(test):
    p = subprocess.run([PY, "-m", "pytest", test, "-q", "--no-header", "-x"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


fails = []
for name, path, old, new, test in MUTATIONS:
    src = path.read_text(encoding="utf-8")
    if old not in src:
        print(f"[SKIP] {name}: 补丁锚点没找到, 逻辑可能已改")
        fails.append(name)
        continue
    # 先确认基线是绿的
    rc0, _ = run(test)
    if rc0 != 0:
        print(f"[FAIL] {name}: 基线本来就是红的, 测��无效")
        fails.append(name)
        continue
    try:
        path.write_text(src.replace(old, new, 1), encoding="utf-8")
        rc1, out1 = run(test)
    finally:
        path.write_text(src, encoding="utf-8")
    rc2, _ = run(test)
    if rc1 == 0:
        print(f"[FAIL] {name}: 关掉逻辑后测试仍然绿 -> 这条测试没锁住东西")
        fails.append(name)
    elif rc2 != 0:
        print(f"[FAIL] {name}: 还原后测试变红 -> 还原失败, 工作区已脏")
        fails.append(name)
    else:
        print(f"[ OK ] {name}: 关掉 -> 红, 还原 -> 绿")

print()
if fails:
    print(f"RESULT: {len(fails)} 项反向验证未通过: {fails}")
    sys.exit(1)
print(f"RESULT: 全部 {len(MUTATIONS)} 项反向验证通过")
