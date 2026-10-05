"""
反向验证: 主题地图"分块 vs 抽样"的护栏 (10-05)

10-05 把生产路径从"超 200 条就均匀抽样"改成"按字符分块 + 按主题名合并"。
理由: 抽样会让被抽掉的 fact **永远不被任何主题覆盖**, 实测 309 条静默丢 35%。

不做反向验证的话, "861 passed" 只说明测试与当前代码**一致**, 不说明测试**有效**。
本轮上一批护栏里就有 4 条是摆设(测的是另一个分支), 所以这次先验自己。

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_chunking.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
OG = ROOT / "agent" / "tools" / "surveys" / "_organize.py"
BAK = ROOT / "output" / "_rv_organize.bak"

MUTATIONS = [
    ("C1 分块时丢 fact(只取每块第一条)",
     "        cur.append(f)\n        n += L\n",
     "        cur = [f]\n        n = L\n",
     "分块丢 fact = 回到'静默抽样'的病根, 只是一层包装"),
    ("C2 合并时同名主题不并 fact_ids",
     "            cur[\"fact_ids\"] = list(dict.fromkeys(\n"
     "                cur[\"fact_ids\"] + (t.get(\"fact_ids\") or [])))\n",
     "            pass\n",
     "同名主题分到两块时 fact 会各自只留一半"),
    ("C3 合并时把块的告警丢掉",
     "        warnings.extend(r.get(\"warnings\") or [])\n",
     "        pass\n",
     "某块失败/截断就没人知道了 —— 违反'宁可吵不可静默'"),
    ("C4 截断主题时不报丢失的 fact 数",
     "            f\"主题数 {len(themes)} 超过上限 {max_themes}, 已截断, \"\n"
     "            f\"被截掉的主题涉及 {dropped} 条 fact(这些 fact 不会出现在主题地图里)\")\n",
     "            f\"主题数 {len(themes)} 超过上限 {max_themes}, 已截断\")\n",
     "砍主题 = 砍 fact, 不说清楚就成了静默有损"),
    ("C5 入口超量时改回抽样",
     "    if len(chunks) > 1:\n",
     "    if False:\n",
     "直接退回 10-05 之前的静默抽样行为"),
    ("C6 截断上游时按条数切(不按字符)",
     "        if cur and (n + L > THEME_CHUNK_CHARS or len(cur) >= THEME_CHUNK_FACTS):\n",
     "        if cur and len(cur) >= THEME_CHUNK_FACTS:\n",
     "按条数封顶就是 13194 字符那次卡死 43 分钟的根因"),
]


def run_tests() -> tuple[bool, str]:
    r = subprocess.run(
        [str(ROOT / "venv" / "Scripts" / "python.exe"), "-m", "pytest",
         "tests/test_claim_layer.py", "-q", "--no-header", "-x"],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stdout or "")[-400:]


def main() -> int:
    shutil.copy2(OG, BAK)
    ok_all, _ = run_tests()
    print(f"基线(未改坏): {'全绿' if ok_all else '不是全绿 —— 先修好再谈反向验证'}")
    if not ok_all:
        shutil.copy2(BAK, OG)
        return 2

    results = []
    try:
        for name, old, new, why in MUTATIONS:
            src = BAK.read_text(encoding="utf-8")
            if old not in src:
                results.append((name, "SKIP(片段没找到)", False, why))
                continue
            OG.write_text(src.replace(old, new, 1), encoding="utf-8")
            try:
                green, tail = run_tests()
            finally:
                shutil.copy2(BAK, OG)
            if green:
                results.append((name, "仍然全绿 —— 护栏是摆设", False, why))
            else:
                m = re.search(r"(\d+) failed", tail)
                results.append((name, f"如期变红({m.group(0) if m else '有失败'})", True, why))
    finally:
        shutil.copy2(BAK, OG)

    print()
    for name, msg, good, why in results:
        print(f"  {'OK  ' if good else '!! '}{name:<34} {msg}")
        print(f"      {why}")
    bad = [r for r in results if not r[1].startswith("如期")]
    print(f"\n{len(results) - len(bad)}/{len(results)} 条护栏有效")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
