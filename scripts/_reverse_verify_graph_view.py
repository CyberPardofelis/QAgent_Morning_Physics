"""
反向验证: 主题关联点线图的护栏 (10-05)

"45 passed" 只说明测试与当前代码**一致**, 不说明测试**有效**。这批护栏尤其
需要验, 因为本轮踩到的两个坑都是"看着完全正常":

  · 字段名不对齐 → 441 条边一条都没画出来, 但 182 个节点照常在场景里, 看着像成功了
  · 节点不登记进 _node_items → 图画出来了, 点选整个是死的

所以这里的 mutation 刻意都选"不报错、只是少做/多做"的那一类。

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_graph_view.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "_rv_graph_bak"
LAY_REL = "agent/infra/graph_layout.py"
VIEW_REL = "ui/theme_graph_view.py"
DLG_REL = "ui/theme_map_dialog.py"

MUTATIONS = [
    ("G1 默认就把所有边画出来",
     VIEW_REL,
     "        line.setVisible(False)          # 默认不画, 见模块头\n",
     "        line.setVisible(True)\n",
     "441 条边全画出来就是一团毛线 —— 用户明确要求'不点就不显示关联'"),
    ("G2 选中时显示全部边而不是自己的",
     VIEW_REL,
     "            on = (a == name or b == name)\n",
     "            on = True\n",
     "点开还是满屏线, 这个视图就没有存在意义了"),
    ("G3 还原'再点一次同一节点就收起'",
     VIEW_REL,
     "        name = name or \"\"\n        self._selected = name\n",
     "        name = name or \"\"\n"
     "        if name and name == self._selected:\n"
     "            name = \"\"\n"
     "        self._selected = name\n",
     "两条隐藏路径互相干扰: 程序里连着选两次同一主题会被静默收走"),
    ("G4 背景圆半径不覆盖自己的主题",
     LAY_REL,
     "            rs.append(float(np.max(np.linalg.norm(pts - c, axis=1))) + padding)\n",
     "            rs.append(float(np.max(np.linalg.norm(pts - c, axis=1))) * 0.5)\n",
     "主题掉到圈外 → 读图的人根本看不出哪块属于哪个板块"),
    ("G5 不做宏观块分离",
     LAY_REL,
     "    centers = _push_apart(centers, radii)\n",
     "    pass\n",
     "五个背景圆叠成一坨, 等于没有分组(第一版就是这样的)。两处都要破坏: "
     "定尺度前不分离 → 缩放前就叠着; 松弛后不重算半径也不分离 → 挪开的点"
     "把圈撑破"),
    ("G6 布局掺随机数",
     LAY_REL,
     "        for k, i in enumerate(idxs):\n            a = k * GOLDEN_ANGLE\n",
     "        import random as _r\n        _r.seed()\n"
     "        for k, i in enumerate(idxs):\n"
     "            a = k * GOLDEN_ANGLE + _r.random()\n",
     "位置每次都抖 → 没法把上周的图和这周的图对比"),
    ("G7 颜色按 index//k 分组取",
     LAY_REL,
     "    hue = ((index * GOLDEN_ANGLE) % 360.0) / 360.0\n"
     "    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.72)\n",
     "    hue = ((index // 20) * 0.08) % 1.0\n"
     "    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.72)\n",
     "182 个主题挤成十来种颜色, '一个主题一个颜色'直接作废"),
    ("G8 缩放阈值失效(标签永远显示)",
     VIEW_REL,
     "            return abs(float(self._view.transform().m11()))\n",
     "            return 1.0\n",
     "182 个主题挤一屏全写上名字 = 一团糊, 等于没有图"),
    ("G9 点标签时不做 parent 回溯",
     VIEW_REL,
     "            it: QGraphicsItem | None = hit\n"
     "            while it is not None:\n"
     "                name = it.data(0)\n"
     "                if isinstance(name, str) and name:\n"
     "                    self.node_clicked.emit(name)\n"
     "                    break\n"
     "                it = it.parentItem()\n",
     "            self.background_clicked.emit()\n",
     "标签是节点的子项, 不回溯 parent 就找不到名字 → 点在主题名上等于点了空白, "
     "反而把刚展开的关联关了"),
    ("G10 节点不登记(点选整个是死的)",
     VIEW_REL,
     "            self._node_items[name] = dot\n",
     "            pass\n",
     "节点照常画出来, 看着像成功了, 但 select_theme 找不到任何东西"),
    ("G11 边字段名不对齐",
     DLG_REL,
     "            {\"a\": e[\"theme_a\"], \"b\": e[\"theme_b\"], \"weight\": e[\"n_papers\"]}\n",
     "            dict(e)\n",
     "441 条边一条都取不到端点 → 全被 continue 掉, 而场景看着完全正常"),
    ("G12 不做最小间距松弛(点团在一起)",
     LAY_REL,
     "    _enforce_min_gap(pos, min_gap)\n",
     "    pass\n",
     "斥力方向在两点靠近时趋零((delta/dist)·1/d², delta≈0 时单位向量也是 0), "
     "真库 33 个无边主题会叠到最小间距 2.25px, 而节点直径最大 40.8px"),
    ("G13 min_gap 被当成缩放前单位(漏乘 scale)",
     LAY_REL,
     "    _enforce_min_gap(pos, min_gap)\n",
     "    _enforce_min_gap(pos, min_gap * scale)\n",
     "min_gap 必须是最终像素: 漏乘缩放系数就等于没设 —— 实测 min_gap=34 在真库"
     "最终只做到 12px, 而节点直径最大 40.8px, 点重新团在一起"),
    ("G15 背景圆不从命中项里排除(点圆是死区)",
     VIEW_REL,
     "        if hit is not None and hit.data(ROLE_BACKGROUND):\n"
     "            hit = None\n",
     "        pass\n",
     "点圆内部会既不选主题也不收起关联 —— 靠 acceptedMouseButtons 排除是"
     "行不通的, 实测 itemAt 照样返回它"),
]


def run_tests() -> tuple[bool, str]:
    r = subprocess.run(
        [str(ROOT / "venv" / "Scripts" / "python.exe"), "-m", "pytest",
         "tests/test_theme_map_graph.py", "-q", "--no-header", "-x"],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stdout or "")[-500:]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [LAY_REL, VIEW_REL, DLG_REL]
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
            path.write_text(src.replace(old, new), encoding="utf-8")
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
