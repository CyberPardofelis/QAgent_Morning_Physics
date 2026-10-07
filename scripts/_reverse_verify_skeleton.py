"""反向验证: 全局骨架的每条护栏都配一个「改坏它, 测试必须变红」的变异。

重点在 M1/M2 —— 它们合起来复现 10-07 真跑看到的那个形态(碎片化):
  M1 去掉对齐      -> 跨块叫法不并, 主题数回到「块块不同」
  M2 骨架不保底    -> 超上限时又变成块级视角通吃

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_skeleton.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG = ROOT / "agent" / "tools" / "surveys" / "_organize.py"
TEST = "tests/test_theme_skeleton.py"

# (编号, 说明, 文件, 变异前, 变异后, 期望变红的测试名片段)
MUTATIONS = [
    ("M1", "关闭骨架对齐 —— 块块叫法不并, 退回 10-07 的碎片化形态",
     ORG,
     "    align_map: dict[str, str] = {}\n    if skeleton:",
     "    align_map: dict[str, str] = {}\n    if False:",

     "merge_collapses_per_chunk_naming_into_skeleton"),

    ("M2", "超上限时不再给骨架保底 —— 全局结构被块级视角挤掉",
     ORG,
     "        if sk_names:\n            by_norm = {_norm_name(t[\"name\"]): t for t in themes}",
     "        if False:\n            by_norm = {_norm_name(t[\"name\"]): t for t in themes}",
     "skeleton_themes_survive_truncation"),

    ("M3", "摘要只取前 100 条 fact —— 后面的语料在骨架阶段就消失了",
     ORG,
     "    out = []\n    for i, f in enumerate(facts, 1):",
     "    out = []\n    for i, f in enumerate(facts[:100], 1):",
     "digest_keeps_every_fact"),

    ("M4", "prompt 里去掉「不要按技术分类」—— 模型自然就按技术切了",
     ORG,
     "【极要紧的一条】按**任务/问题**分类, **不要**按技术分类。",
     "【铁律】认真阅读语料。",
     "skeleton_prompt_demands_task_dimension_not_technique"),

    ("M5", "不再要求 belongs 线索 —— 阶段二没有判断归属的依据",
     ORG,
     "2. 每个主题必须给出 **task**(这个任务要解决什么) 和 **belongs**(什么样的事实属于",
     "2. 每个主题必须给出 **task**(这个任务要解决什么)。",
     "skeleton_prompt_requires_membership_clues"),

    ("M6", "骨架块只列名字, 不给归属线索 —— 阶段二看得见任务但判不了归属",
     ORG,
     "        if t.get(\"belongs\"):\n            lines.append(f\"    归属线索: {t['belongs']}\")",
     "",
     "render_skeleton_lists_all_themes_with_clues"),

    ("M7", "不再要求逐字照抄骨架名 —— 模型改个名合并层就撑破了",
     ORG,
     "name 必须与上面某一条完全一致",
     "name 尽量贴近上面某一条",
     "render_skeleton_demands_exact_name_copy"),

    ("M8", "泛化名也硬塞进骨架 —— 「经典阴影」这类会错合并两个不同任务",
     ORG,
     "if len(containers) == 1:",
     "if containers:",
     "align_refuses_ambiguous_generic_name"),

    ("M9", "同名也算改写 —— 告警虚高(说归并了其实一个都没并)",
     ORG,
     "        if target and target != name:\n            mapping[name] = target",
     "        if target:\n            mapping[name] = target",
     "align_exact_name_is_a_noop"),

    ("M10", "摘要截断失效 —— 每条 fact 把整段 claim 塞进去, prompt 撑爆",
     ORG,
     "        claim = (f.get(\"claim\") or \"\").strip().replace(\"\\n\", \" \")[:claim_chars]",
     "        claim = (f.get(\"claim\") or \"\").strip().replace(\"\\n\", \" \")",
     "digest_truncates_claim_not_the_line"),

    ("M11", "骨架失败不返回空 —— 抛异常打断整条流水线",
     ORG,
     "        safe_print(f\"[skeleton] LLM 调用异常: {type(e).__name__}: {str(e)[:80]}\", flush=True)\n        return {}",
     "        raise",
     "build_skeleton_returns_empty_on_llm_error"),

    ("M12", "认不出 API 错误串 —— 会把网络故障报成模型输出格式问题",
     ORG,
     "    api_err = _is_llm_error_response(out or \"\")\n    if api_err:",
     "    api_err = None\n    if api_err:",
     "build_skeleton_returns_empty_on_api_error_string"),

    ("M13", "骨架不落盘 —— 复现/续跑时看不出这版图是怎么来的",
     ORG,
     'return {"themes": themes, "narrative_arc": arc, "gaps": gaps,\n            "warnings": warnings, "skeleton": skeleton or {}}',
     'return {"themes": themes, "narrative_arc": arc, "gaps": gaps,\n            "warnings": warnings}',
     "skeleton_is_persisted_in_result"),

    ("M14", "块内 prompt 不注入骨架 —— 阶段二看不见全局结构",
     ORG,
     "        skeleton_block=_render_skeleton(skeleton),\n    )\n    parsed = None",
     "        skeleton_block='',\n    )\n    parsed = None",
     "skeleton_block_is_injected_into_chunk_prompt"),

    ("M15", "不告诉用户归并了多少叫法 —— 改名这件事静默发生",
     ORG,
     "f\"主题名对齐到全局骨架: {len(align_map)} 个块内叫法改写成骨架名, \"",
     "f\"\",",
     "merge_announces_renames"),
]


def _run_test(frag: str) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q", "-k", frag,
         "-p", "no:cacheprovider", "--no-header", "-x"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "")[-400:]


# **测试自己崩了不等于护栏咬到了**。TypeError/NameError/SyntaxError 都会让
# pytest 返回非 0, 于是「变异 → 返非 0 → 判 ✓」就成了一句自欺的话。
# 第一次写这个脚本时 `test_skeleton_themes_survive_truncation` 里把
# `chunk_tag` 写成了 `tag`, 报 TypeError, 却被判成「变红 ✓ 护栏生效」, 白绿一场。
#
# **不要试图从 pytest 输出里区分「断言失败」和「测试崩了」** —— 试过, 不可靠:
# `E   ` 行装的是断言表达式而不是异常类型, 摘要行里又恒有 "error" 字样,
# Windows 控制台编码还会把中文冲成乱码。于是改用**基线对照**: 变异之前先
# 跑一次同一个测试, 它本来就红着的话, 无论变异成什么样都非 0, 这条护栏
# 等于不存在。基线绿 + 变异红, 才算真的咬到了。
def _baseline_ok(frag: str) -> tuple[bool, str]:
    rc, out = _run_test(frag)
    if rc == 0:
        return True, ""
    return False, out[-300:]


def main() -> int:
    os.chdir(ROOT)
    results: list[tuple[str, bool, str]] = []
    for mid, desc, path, before, after, frag in MUTATIONS:
        src = path.read_text(encoding="utf-8")
        if before not in src:
            results.append((mid, False, f"变异锚点没找到, 测试无效: {desc}"))
            continue
        base_ok, base_out = _baseline_ok(frag)
        if not base_ok:
            results.append((mid, False,
                            f"**基线就不绿** —— 这条测试本来就红着, 变异它没有意义: {desc}"))
            continue
        with tempfile.TemporaryDirectory() as td:
            bak = Path(td) / path.name
            shutil.copy2(path, bak)
            try:
                path.write_text(src.replace(before, after, 1), encoding="utf-8")
                rc, _ = _run_test(frag)
                if rc == 0:
                    results.append(
                        (mid, False, f"改坏后测试仍然绿 —— 这条护栏没咬到: {desc}"))
                else:
                    results.append((mid, True, f"变红 ✓  {desc}"))
            finally:
                shutil.copy2(bak, path)

    print("=" * 78)
    ok = sum(1 for _, good, _ in results if good)
    for mid, good, msg in results:
        print(f"  {'✓' if good else '✗'} {mid:<5} {msg}")
    print("=" * 78)
    print(f"反向验证 {ok}/{len(results)} 生效")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())