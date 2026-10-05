"""
反向验证: 逐个把 10-05 修掉的逻辑改坏, 确认对应测试真的会变红。

不做这一步的话, "853 passed" 只说明测试和当前代码**一致**, 不说明测试
**有效**。10-05 已经有两次教训:
  - 护栏测试自己写错了期望(以为 apply_rename 会连带合并另一行), 差点放过
  - 收尾正则多写一个 `。`, 是**被自己的测试**逮到的, 不是被 review 逮到的

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_placeholder.py
每项期望"至少一个测试失败"; 变绿说明这条护栏是摆设。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"D:\pythonProject\QAgent")
PS = ROOT / "agent" / "prompts_survey.py"
SS = ROOT / "scripts" / "strip_placeholder_facts.py"
PS_BAK = ROOT / "output" / "_rv_prompts_survey.py.bak"
SS_BAK = ROOT / "output" / "_rv_strip.py.bak"

# (编号, 改哪个文件, 原片段, 坏掉的片段, 说明)
MUTATIONS = [
    ("R1 少认一种形态(裸 arxiv id)",
     PS,
     '    re.compile(r"\\[(?:\\d{4}\\.\\d{4,5}(?:v\\d+)?|\\d{7})\\]"),            # [2209.02093] / [2209.02093v2]\n',
     "",
     "把 [2209.02093] 这条正则删掉 —— 正是 10-05 漏掉的那一种"),
    ("R2 剥完不收悬空空格",
     PS,
     '    s = re.sub(r"\\s+([。,.;:!?、)\\]】」》%])", r"\\1", s)\n',
     "",
     "标注被剥掉后留下 '方法有效 。' 这种悬空空格"),
    ("R3 收尾时把句号也吃掉",
     PS,
     '    s = re.sub(r"[\\s，,;；、:：]+$", "", s)\n',
     '    s = re.sub(r"[\\s，,。;；、:：]+$", "", s)\n',
     "10-05 真实犯过的错: 多写一个 `。`, 会让每条 fact 末尾都没句号"),
    ("R4 误伤正文里的裸 arxiv id",
     PS,
     '    re.compile(r"(?<![\\w.])paper[_\\s-]?id\\s*[:：]\\s*\\d{4}\\.\\d{4,5}(?![\\w])", re.I),\n',
     '    re.compile(r"\\d{4}\\.\\d{4,5}"),\n',
     "剥得过头: 正文里正常写的 arxiv 号也被当占位符删掉"),
    ("R5 plan 不模拟最终 id 集合",
     SS,
     "    existing = {r[0] for r in conn.execute(\"SELECT id FROM fact\")}\n"
     "    will_exist = set(existing)\n",
     "    existing = {r[0] for r in conn.execute(\"SELECT id FROM fact\")}\n"
     "    will_exist = set()\n",
     "10-05 实测: 10 条 merge 变 40 条才对, 少模拟会中途 UNIQUE 失败"),
    ("R6 合并时不撞车却忘了插 theme_fact",
     SS,
     '        if exist is None:\n'
     '            conn.execute(\n'
     '                "INSERT INTO theme_fact (theme_id, fact_id, asserted_by) VALUES (?,?,?)",\n'
     '                (theme_id, kept_id, asserted_by))\n',
     '        if exist is None:\n'
     '            pass\n',
     "不撞车时的主题归属被静默丢掉 —— 此前只测了撞车分支, 漏了这条"),
    ("R7 自检不拦新增悬空(全当提示)",
     SS,
     '        errors.append(f"本次新增外键悬空 {len(new_fk)} 处: {sorted(new_fk)[:3]}")\n',
     '        notes.append(f"本次新增外键悬空 {len(new_fk)} 处")\n',
     "把阻断项降级成提示 = 脏数据直接落库"),
    ("R8 不做收尾合并",
     SS,
     '    late = conn.execute(\n'
     '        "SELECT content FROM fact GROUP BY content HAVING COUNT(*)>1").fetchall()\n',
     '    late = []\n',
     "剥完才撞成同 content 的组合会留在库里(10-05 实测真库 15 组)"),
    ("R9 合并不记审计",
     SS,
     '    conn.execute("DELETE FROM fact WHERE id=?", (dropped_id,))\n',
     '    pass\n',
     "合并不可回滚 —— 10-05 明确要求每条合并都留原文"),
]


def run_tests() -> tuple[bool, str]:
    r = subprocess.run(
        [str(ROOT / "venv" / "Scripts" / "python.exe"), "-m", "pytest",
         "tests/test_fact_placeholder_strip.py", "-q", "--no-header", "-x"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stdout or "")[-400:]


def main() -> int:
    shutil.copy2(PS, PS_BAK)
    shutil.copy2(SS, SS_BAK)
    ok_all, _ = run_tests()
    print(f"基线(未改坏): {'全绿' if ok_all else '竟然不是全绿 —— 先修好再谈反向验证'}")
    if not ok_all:
        shutil.copy2(PS_BAK, PS)
        shutil.copy2(SS_BAK, SS)
        return 2

    results = []
    try:
        for name, path, old, new, why in MUTATIONS:
            src = path.read_text(encoding="utf-8")
            if old not in src:
                results.append((name, "SKIP(片段没找到, 护栏可能已失效)", False, why))
                continue
            path.write_text(src.replace(old, new, 1), encoding="utf-8")
            try:
                still_green, tail = run_tests()
            finally:
                path.write_text(src, encoding="utf-8")
            if still_green:
                results.append((name, "仍然全绿 —— 这条护栏是摆设", False, why))
            else:
                m = re.search(r"(\d+) failed", tail)
                detail = m.group(0) if m else "有测试失败"
                results.append((name, f"如期变红({detail})", True, why))
    finally:
        shutil.copy2(PS_BAK, PS)
        shutil.copy2(SS_BAK, SS)

    print()
    for name, msg, good, why in results:
        mark = "OK  " if good else "!! "
        print(f"  {mark}{name:<38} {msg}")
        print(f"      {why}")
    bad = [r for r in results if not r[1].startswith("如期")]
    print(f"\n{len(results) - len(bad)}/{len(results)} 条护栏有效")
    if bad:
        print("失效的护栏:")
        for name, msg, _, _ in bad:
            print(f"  - {name}: {msg}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
