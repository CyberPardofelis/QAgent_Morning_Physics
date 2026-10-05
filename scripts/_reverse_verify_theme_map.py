"""
反向验证: 主题地图窗口的护栏 (10-05)

"19 passed" 只说明测试与当前代码**一致**, 不说明测试**有效**。本项目上一批
护栏里就有 4 条是摆设(测的是另一个分支), 所以这次先验自己。

改坏的地方都是"不报错、只是悄悄少给/多给/盖掉"的那一类:
    跨域边被默认 limit 砍掉一半 · 截断了不说 · 叶子搜不到 · 孤立 fact 不报
    · QSS 字号盖掉全局设置 · 筛选失效 · 入口没挂上

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_theme_map.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "_rv_theme_map_bak"
DLG_REL = "ui/theme_map_dialog.py"
WIN_REL = "ui/chat_window.py"
GQ_REL = "agent/infra/graph_query.py"

# (名称, 改哪个文件, 原文, 改后, 为什么这条护栏重要)
MUTATIONS = [
    ("T1 跨域边退回默认 limit 200",
     DLG_REL, "CROSS_LIMIT = 1000", "CROSS_LIMIT = 200",
     "真库 441 条边, 默认 200 → 界面安静地少显示 241 条, 用户无从察觉"),
    ("T2 截断了也不说",
     DLG_REL, 'if len(edges) >= CROSS_LIMIT else "(未截断)")',
     'if False else "(未截断)")',
     "违反「宁可吵不可静默」: 有损显示必须说出口"),
    ("T3 对话框改成模态",
     DLG_REL, "        self.setModal(False)\n        self.setWindowModality(Qt.NonModal)\n",
     "        self.setModal(True)\n        self.setWindowModality(Qt.WindowModal)\n",
     "边看结构边写综述的工作流被锁死"),
    ("T4 自己写死 QSS 字号",
     DLG_REL, "        self.setStyleSheet(get_stylesheet())\n",
     '        self.setStyleSheet(get_stylesheet() + " QDialog { font-size: 14px; }")\n',
     "QSS 的 font-size 在绘制期覆盖 QApplication 全局字体并级联 —— "
     "用户在设置里改字号会看着这个窗口纹丝不动"),
    ("T5 核心对象搜不到(不给 search)",
     DLG_REL, '                                              "search": content})',
     '                                              })',
     "搜索框承诺能搜「核心对象」, 不给正文就只剩主题名能命中 —— 界面在说谎"),
    ("T6 孤立核心对象不报",
     DLG_REL, 'if unlinked else ""', 'if False else ""',
     "1773 条没进任何主题的事实会从视野里消失, 结构看上去比实际完整"),
    ("T7 筛选命中失败也不隐藏",
     DLG_REL,
     "        keep = False\n        for i in range(item.childCount()):\n"
     "            if self._filter_recursive(item.child(i), needle):\n"
     "                keep = True\n        item.setHidden(not keep)\n        return keep\n",
     "        return True\n",
     "搜不到的东西还全留着 = 筛选器形同虚设"),
    ("T8 入口没挂到菜单",
     WIN_REL,
     '        theme_map_action = QAction("主题地图", self)\n'
     "        theme_map_action.triggered.connect(self.open_theme_map)\n"
     "        knowledge_menu.addAction(theme_map_action)\n",
     "        pass\n",
     "窗口写得再好, 菜单里没有入口就等于不存在"),
    ("T9 facts_by_theme 忽略主题过滤",
     GQ_REL, '    if theme_name:\n        sql += " WHERE t.name = ?"',
     '    if False:\n        sql += " WHERE t.name = ?"',
     "详情面板会把全库 fact 当成当前主题的, 越查越离谱"),
]


def run_tests() -> tuple[bool, str]:
    r = subprocess.run(
        [str(ROOT / "venv" / "Scripts" / "python.exe"), "-m", "pytest",
         "tests/test_theme_map_dialog.py", "-q", "--no-header", "-x"],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stdout or "")[-500:]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [DLG_REL, WIN_REL, GQ_REL]
    for rel in targets:
        shutil.copy2(ROOT / rel, OUT / Path(rel).name)

    ok_all, tail = run_tests()
    print(f"基线(未改坏): {'全绿' if ok_all else '不是全绿 —— 先修好再谈反向验证'}")
    if not ok_all:
        print(tail)
        for rel in targets:
            shutil.copy2(OUT / Path(rel).name, ROOT / rel)
        return 2

    results = []
    try:
        for name, rel, old, new, why in MUTATIONS:
            path = ROOT / rel
            bak = OUT / Path(rel).name
            src = bak.read_text(encoding="utf-8")
            if old not in src:
                results.append((name, "SKIP(片段没找到)", False, why))
                continue
            path.write_text(src.replace(old, new, 1), encoding="utf-8")
            try:
                green, out = run_tests()
            finally:
                shutil.copy2(bak, path)
            if green:
                results.append((name, "仍然全绿 —— 护栏是摆设", False, why))
            else:
                m = re.search(r"(\d+ failed)", out)
                results.append((name, f"如期变红({m.group(1) if m else '有失败'})",
                                True, why))
    finally:
        for rel in targets:
            shutil.copy2(OUT / Path(rel).name, ROOT / rel)

    print()
    for name, msg, good, why in results:
        print(f"  {'OK  ' if good else '!! '}{name:<32} {msg}")
        print(f"      {why}")
    bad = [r for r in results if not r[1].startswith("如期")]
    print(f"\n{len(results) - len(bad)}/{len(results)} 条护栏有效")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
