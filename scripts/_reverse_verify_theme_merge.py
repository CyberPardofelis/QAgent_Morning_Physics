"""反向验证: 合并层截断的每条护栏都配一个"改坏它, 测试必须变红"的变异。

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_theme_merge.py
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
TEST = "tests/test_claim_layer.py"

# (编号, 说明, 文件, 变异前, 变异后, 期望失败的测试名片段)
MUTATIONS = [
    ("M1", "改回按块序砍(回到 by_name 插入序)—— 前两块通吃, 后面整块归零",
     ORG,
     "        themes = selected",
     "        themes = [t for t in by_name.values()][:max_themes]",
     "test_merge_does_not_let_early_chunks_take_all_the_slots"),

    ("M2", "轮转改成全局按 fact 数取前 N —— 只是把'先来的块通吃'换成'大块通吃'",
     ORG,
     "        selected: list[dict[str, Any]] = []\n        picked: set[int] = set()\n        depth = 0\n"
     "        while len(selected) < max_themes:",
     "        flat = sorted(themes, key=lambda t: len(t.get('fact_ids') or []),\n"
     "                      reverse=True)\n"
     "        selected = flat[:max_themes]\n"
     "        picked: set[int] = set()\n"
     "        depth = 0\n"
     "        if False:\n            pass\n"
     "        while len(selected) < max_themes:",
     "test_merge_does_not_let_early_chunks_take_all_the_slots"),

    ("M3", "截断时不报丢了多少 fact —— 有损操作变成静默",
     ORG,
     '        warnings.append(\n            f"主题数 {len(themes)} 超过上限 {max_themes}',
     '        _ = (\n            f"主题数 {len(themes)} 超过上限 {max_themes}',
     "test_merge_warns_how_many_facts_were_dropped_and_from_which_chunks"),

    ("M4", "告警不说按哪几块丢的 —— 用户看不出'某几块被整块砍了'",
     ORG,
     '        chunk_desc = " ".join(\n            f"块{c}={by_chunk_dropped[c]}条" for c in sorted(by_chunk_dropped))',
     '        chunk_desc = "若干块"',
     "test_merge_warns_how_many_facts_were_dropped_and_from_which_chunks"),

    ("M5", "悬空 relation 不再剔除 —— 被砍主题的名字会进 Phase C 的 prompt",
     ORG,
     "        keep = [r for r in (t.get(\"relations\") or []) if r.get(\"to\") not in gone]",
     "        keep = list(t.get(\"relations\") or [])",
     "test_merge_drops_dangling_relations_to_truncated_themes"),

    ("M6", "把'从未定义的主题'也当悬空删掉 —— 抹掉模型给的关系信息",
     ORG,
     "        keep = [r for r in (t.get(\"relations\") or []) if r.get(\"to\") not in gone]",
     "        keep = [r for r in (t.get(\"relations\") or [])\n"
     "                if r.get(\"to\") in alive]",
     "test_merge_keeps_relations_to_never_defined_themes"),

    ("M7", "对用户宣称的那句假话又回来 —— 切块丢 fact 被说成不丢",
     ORG,
     "              f\"逐块归纳后按主题名合并 —— 切块这步一条 fact 都不丢\"\n"
     "              f\"(主题数超上限时仍会丢, 丢了多少见下面的 warnings)\", flush=True)",
     "              f\"逐块归纳后按主题名合并 —— 一条 fact 都不丢\", flush=True)",
     "test_runtime_print_does_not_promise_losslessness"),

    ("M8", "测试重新给自己开后门 max_themes=1000 —— 截断分支再次变死代码",
     ROOT / "tests" / "test_claim_layer.py",
     "    OG.organize_facts_into_themes(facts)",
     "    OG.organize_facts_into_themes(facts, max_themes=1000)",
     "test_entry_test_does_not_pass_a_custom_max_themes"),

    ("M9", "测试的 mock 换成跨块同名 —— 又一次给合并优待(生产实测 0% 命中)",
     ROOT / "tests" / "test_claim_layer.py",
     '                {"name": f"块{c}主题{i}", "statement": "", "notes": "",\n'
     '                 "fact_ids": [f"c{c}f{i}_{j}" for j in range(PER_CHUNK // N_THEMES_PER_CHUNK)],\n'
     '                 "relations": []}',
     '                {"name": f"主题{i}", "statement": "", "notes": "",\n'
     '                 "fact_ids": [f"c{c}f{i}_{j}" for j in range(PER_CHUNK // N_THEMES_PER_CHUNK)],\n'
     '                 "relations": []}',
     "test_merge_does_not_let_early_chunks_take_all_the_slots"),
]


def _run_test(frag: str) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q", "-k", frag,
         "-p", "no:cacheprovider", "--no-header", "-x"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "")[-500:]


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

    print("=" * 78)
    ok = sum(1 for _, good, _ in results if good)
    for mid, good, msg in results:
        print(f"  {'✓' if good else '✗'} {mid:<4} {msg}")
    print("=" * 78)
    print(f"反向验证 {ok}/{len(results)} 生效")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
