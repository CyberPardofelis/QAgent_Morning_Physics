"""反向验证: 审查结论落盘的每条护栏都配一个"改坏它, 测试必须变红"的变异。

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_audit_persist.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EF = ROOT / "agent" / "tools" / "surveys" / "_actions" / "_extract_facts.py"
TEST = "tests/test_fact_audit_persistence.py"

# (编号, 说明, 文件, 变异前, 变异后, 期望失败的测试名片段)
MUTATIONS = [
    ("P1", "写回整个拿掉 —— 审查结论继续只活在内存, 续跑复活被否决的 fact",
     EF,
     "    n_rewritten = _persist_audit_marks(ws, state[\"facts\"])",
     "    n_rewritten = len(state[\"facts\"])",
     "test_persist_call_sits_after_both_audits"),

    ("P2", "clear 和 append 顺序颠倒 —— 先写再清, 文件直接空掉",
     EF,
     "        ws.clear_facts()\n        for f in facts:\n            ws.append_fact(f)",
     "        for f in facts:\n            ws.append_fact(f)\n        ws.clear_facts()",
     "test_persist_writes_audit_marks_to_disk"),

    ("P3", "重写时丢掉审查标记 —— 全部写成默认值, 续跑照样复活",
     EF,
     "        ws.clear_facts()\n        for f in facts:\n            ws.append_fact(f)",
     "        ws.clear_facts()\n        for f in facts:\n"
     "            ws.append_fact({k: v for k, v in f.items()\n"
     "                             if not k.startswith('_')})",
     "test_persist_writes_audit_marks_to_disk"),

    ("P4", "条数对不上也硬重写 —— 磁盘上多出来的 fact 被静默丢掉",
     EF,
     "    if n_ondisk != len(facts):",
     "    if False:",
     "test_persist_refuses_when_disk_count_mismatches"),

    ("P5", "写回失败静默 —— 续跑复活这件事用户看不到",
     EF,
     '        yield ("⚠ 审查结论未能写回 _facts.jsonl —— 本轮综述不受影响(用的是内存), "',
     "        _ = (\"⚠ 审查结论未能写回 _facts.jsonl —— 本轮综述不受影响(用的是内存), \"",
     "test_failure_is_yielded_not_silent"),

    ("P6", "空 fact 列表也去重写 —— clear 之后写 0 条, 磁盘直接清空",
     EF,
     "    if not facts:\n        return None",
     "    if False:\n        return None",
     "test_persist_returns_none_on_empty_facts"),

    ("P7", "读 _facts.jsonl 失败时硬来 —— 拿不到基线就重写 = 赌",
     EF,
     "    except Exception as e:                              # noqa: BLE001\n"
     '        logger.warning("[extract_facts] 读 _facts.jsonl 失败, 放弃写回: %s", e)\n'
     "        return None",
     "    except Exception:                                   # noqa: BLE001\n        return 0",
     "test_persist_returns_none_when_read_back_fails"),

    ("P8", "假注释又回来 —— 断言标记本来就在文件里",
     ROOT / "agent" / "tools" / "surveys" / "survey_generate.py",
     "    # 10-07 修: 这句原来写的是「fact 里的对齐标记在 _facts.jsonl 里(旧 run 可能",
     "    # fact 里的对齐标记在 _facts.jsonl 里(旧 run 可能",
     "test_no_comment_claims_marks_are_already_persisted"),
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
