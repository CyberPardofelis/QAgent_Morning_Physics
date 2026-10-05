"""
agent.infra.graph_layout — 主题点线图的布局计算 (10-05)

给「关联图」视图算坐标。**纯 numpy, 不依赖 Qt** —— 这样布局能被单测直接验,
不用先造一个 QApplication(offscreen 造 QApplication 很慢, 而且布局本身跟
绘制没关系)。

节点 = **主题**(不是宏观主题)。边 = 两个主题共享的论文(weight = 共享篇数)。
宏观主题不画成节点, 而是画成一圈**背景圆**, 它底下的主题聚在这圈里。

**为什么不用 networkx.spring_layout**: 它不认识"宏观主题"。硬套只能靠给同宏观
的主题加假边来聚类, 而那些假边又不能画(画出来就是假的关联)。这里直接把
"同宏观 → 往本宏观质心靠"写成一项显式的力。

**两阶段, 不用力做宏观块分离(实测差别很大)**: 只加"同宏观往质心靠"这一项时,
五个块的质心最后全塌在画布中间一小片(实测真库: 圆心全落在 400-550 × 450-590,
半径 420-465), 背景圆叠成一坨, 根本读不出板块。所以拆成:
    阶段一  力导向 —— 只决定**块内**谁挨着谁(相对结构)
    阶段二  阶段一把宏观块当圆做**分离 + 整块平移**, 保证圆不重叠
阶段二不碰块内结构, 所以"同宏观抱团"这个已经拿到的东西不会被破坏。

**为什么不用 matplotlib**: 这是 PySide 应用, 外面已经有 QApplication 和一套
QSS 主题; 为画一张静态图引一套独立渲染栈, 字体配色都对不上, 滚动缩放还得
自己再实现一遍。用 QGraphicsScene 画。

**确定性**: 初始位置由"宏观主题 + 黄金角"确定性地给出, 全程没有随机数。所以
同一份数据每次打开长得一样 —— 布局抖动会让人没法把上周的图和这周的图对比。
"""
from __future__ import annotations

import colorsys
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

# 黄金角(度)。相邻 index 也能取到明显不同的色相, 182 个主题不会挤成一片
GOLDEN_ANGLE = 137.50776405003785

# 初值阶段给每个节点叠的确定性微小偏移 —— 没有它, 孤立主题会叠死在同一点
# (见 _initial_positions 里的说明)。相对初始尺度极小, 不影响布局结果。
_JITTER_EPS = 0.01


@dataclass
class Layout:
    """算完的坐标。

    `pos`          主题名 → (x, y)
    `macro_circles` 宏观主题名 → (cx, cy, r)   **保证两两不重叠**
    `width/height` 世界尺寸, 由内容反推(不固定), 视图侧 fitInView 即可
    """

    pos: dict[str, tuple[float, float]] = field(default_factory=dict)
    macro_circles: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    width: float = 0.0
    height: float = 0.0


# ============================================================
# 布局
# ============================================================

def layout_themes(
    nodes: list[dict],
    edges: list[dict],
    *,
    iterations: int = 260,
    repulsion: float = 1.0,
    attract: float = 0.02,
    gravity: float = 0.055,
    damping: float = 0.9,
    padding: float = 70.0,
    min_gap: float = 44.0,
) -> Layout:
    """把主题摆成一团, 同宏观的抱团, 边把它们往一起拉, 最后宏观块互不重叠。

    nodes: `[{"name": "主题", "macro": "宏观" 或 None}, ...]`
    edges: `[{"a": "主题A", "b": "主题B", "weight": 共享论文数}, ...]`

    `min_gap` 是**最终像素**: 世界先等比缩放到约 2200 宽再松弛, 所以这个值
    可以直接和视图里的节点直径比 —— 节点直径 = 2×(4 + 2.1×√核心对象数),
    真库最大 40.8px。默认 34 能让绝大多数节点不重叠。
    """
    names = [n["name"] for n in nodes]
    if not names:
        return Layout()

    groups = _macro_groups(nodes)
    pos = _run_forces(nodes, edges, groups, iterations, repulsion, attract,
                      gravity, damping)
    pos, circles, w, h = _pack_by_macro(pos, groups, padding, min_gap)
    return Layout(
        pos={name: (float(pos[i, 0]), float(pos[i, 1])) for i, name in enumerate(names)},
        macro_circles=circles,
        width=w,
        height=h,
    )


# ---------- 阶段一: 力导向(只管块内相对结构) ----------

def _run_forces(nodes, edges, groups, iterations, repulsion, attract,
                gravity, damping) -> np.ndarray:
    pos = _initial_positions(nodes, groups)

    # 边 → 下标对。查不到的边直接丢(图上画的边必须两端都有节点, 否则渲染会拿到
    # None 坐标炸掉整个视图) —— 宁可少画一条, 不让视图开不起来。
    idx = {node["name"]: i for i, node in enumerate(nodes)}
    ea, eb, ew = [], [], []
    for e in edges:
        i, j = idx.get(e.get("a")), idx.get(e.get("b"))
        if i is None or j is None or i == j:
            continue
        ea.append(i)
        eb.append(j)
        ew.append(max(1.0, float(e.get("weight") or 1.0)))
    ea_a = np.asarray(ea, dtype=int)
    eb_a = np.asarray(eb, dtype=int)
    ew_a = np.asarray(ew, dtype=float)

    scale0 = float(np.max(np.abs(pos))) or 1.0
    temp = scale0 * 0.08
    for step in range(max(1, iterations)):
        temp = temp * (1.0 - step / iterations) ** 1.4 + 0.5

        # 斥力: 全对全。182 个主题 → 182² ≈ 3.3 万对, numpy 下毫秒级
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + 1e-6
        np.fill_diagonal(dist, np.inf)
        inv = 1.0 / (dist * dist)
        force = (delta * inv[:, :, None]).sum(axis=1) * repulsion

        # 边引力: 共享论文越多拉得越紧(这正是"这条边有多实"的视觉编码)
        if ea_a.size:
            d = pos[ea_a] - pos[eb_a]
            dd = np.linalg.norm(d, axis=1) + 1e-6
            f = (d / dd[:, None]) * (dd[:, None] ** 2 * attract * ew_a[:, None])
            np.add.at(force, ea_a, f)
            np.add.at(force, eb_a, -f)

        # 宏观引力: 同宏观的往本宏观质心靠 —— "背景圆"成立的根基
        for idxs in groups.values():
            c = pos[idxs].mean(axis=0)
            force[idxs] += (c - pos[idxs]) * gravity

        disp = force * damping
        norm = np.linalg.norm(disp, axis=1) + 1e-9
        pos += disp * np.minimum(1.0, temp / norm)[:, None]
    return pos


def _enforce_min_gap(pos: np.ndarray, min_gap: float, sweeps: int = 80) -> None:
    """把距离 < min_gap 的点对直接推开(位置修正, 不受力的大小限制)。

    **为什么斥力不够**: 斥力是 `(delta/dist)·1/d²`, d→0 时 delta/dist 本身趋于
    0, 恰好在"最需要推开"的时候没力气。真库有 33 个**没有任何边**的孤立主题
    (没有边引力, 只剩斥力 + 宏观引力), 它们会被吸到本组的质心这个平衡点上
    叠成一堆: 最小间距 2.25px, 而节点半径就有 20.4px。

    **必须顺序处理(Gauss-Seidel), 不能一轮全改(Jacobi)**: 一轮里把所有违例
    同时修正时, 同一个节点身上的推力会累加过冲 —— 实测违例点对从 121 降到 33
    看着变好了, 但最差的一对反而从 6.36 掉到 0.28, 也就是**越修越挤**。
    顺序处理每推一下就重新量, 不会过冲, 实测精确收敛到 min_gap。

    numpy 只用来快速圈出"这一轮有哪些违例对", 修正本身走 Python 循环: 违例对
    通常只有几十到一百多对, 比全量 O(n²) 的 Python 循环快得多。

    完全重合时方向向量为 0, 用节点下标算出**确定性**的兜底角度 —— 不能用
    随机数, 布局要可复现(见模块头)。
    """
    n = pos.shape[0]
    if n < 2 or min_gap <= 0:
        return
    for _ in range(max(1, sweeps)):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)
        a_idx, b_idx = np.triu_indices(n, 1)
        bad = dist[a_idx, b_idx] < min_gap
        if not bad.any():
            return
        for i, j in zip(a_idx[bad].tolist(), b_idx[bad].tolist(), strict=True):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = math.hypot(dx, dy)
            if d >= min_gap:      # 前面推过这一对, 可能已经够了
                continue
            if d < 1e-9:
                ang = math.radians((i * GOLDEN_ANGLE + j * 0.5) % 360.0)
                dx, dy = math.cos(ang), math.sin(ang)
                d = 1.0
            push = (min_gap - d) / 2.0
            ux, uy = dx / d, dy / d
            pos[i, 0] += ux * push
            pos[i, 1] += uy * push
            pos[j, 0] -= ux * push
            pos[j, 1] -= uy * push


def _initial_positions(nodes: list[dict], groups: dict) -> np.ndarray:
    """确定性的初值: 宏观主题摆在一个大圆上, 各自的主题在块内走黄金角螺旋。"""
    n = len(nodes)
    pos = np.zeros((n, 2), dtype=float)
    keys = list(groups)
    g = len(keys)
    big = max((len(v) for v in groups.values()), default=1) or 1
    ring = 300.0
    for gi, key in enumerate(keys):
        idxs = groups[key]
        ang = 0.0 if g == 1 else 2.0 * math.pi * gi / g
        gx, gy = ring * math.cos(ang), ring * math.sin(ang)
        span_r = 30.0 + 70.0 * math.sqrt(len(idxs) / big)
        for k, i in enumerate(idxs):
            a = k * GOLDEN_ANGLE
            r = span_r * math.sqrt((k + 0.5) / len(idxs))
            pos[i] = (gx + r * math.cos(a), gy + r * math.sin(a))

    # 给每个节点叠一个**确定性的极小偏移**。看着像多此一举, 但没有它 33 个
    # 没有任何边的孤立主题会在迭代中叠到坐标完全相同, 然后再也分不开:
    # 斥力是 `(delta/dist)·1/d²`, delta≈0 时方向向量也是 0, 1e12 乘 0 还是 0
    # —— 恰好在最需要推开的时候没力气。实测 33 个重合点用这个偏移后,
    # 一步斥力就从 0.0000 变成 0.18 并持续散开。
    # 偏移量是初始尺度的 3e-5 量级, 远小于任何布局间距, 不干扰结果。
    n_idx = np.arange(n, dtype=float)
    a = n_idx * GOLDEN_ANGLE
    r = _JITTER_EPS * np.sqrt((n_idx + 0.5) / n)
    pos += np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    return pos


# ---------- 阶段二: 宏观块当圆做分离, 整块平移 ----------

def _pack_by_macro(pos: np.ndarray, groups: dict, padding: float,
                   min_gap: float, target: float = 2200.0):
    """先定尺度并推开重叠的点, 再把宏观块当圆分离 + 整块平移。

    `macro is None` 的那一堆也参与分离(不画圆, 但不能压在某个真宏观的圆上),
    只是一不给它出现在 macro_circles 里。

    顺序有讲究, 反了就错(10-05 踩过两遍):
      1. **先等比缩放到目标尺寸, 再做最小间距松弛** —— 这样 `min_gap` 就是
         **最终像素**, 和视图里节点半径同一量纲, 可以直接对着量。反过来
         (先松弛再缩放) 参数就得乘回去, 忘了乘就等于没设: 实测 min_gap=34
         最终只有 12px, 而节点直径最大 40.8px。
      2. 松弛**必须在算半径之前**: 挪完点再算半径才装得下, 反过来就有主题
         掉到圈外(第一次写 81 个掉出去)。
      3. 半径按"最远主题距离 + 边距"算, 再用**这个**半径做分离 —— 分离完才
         补边距的话, 圈会重新叠上(实测最紧的一对重叠 48px)。
      4. 分离只做整块**刚性平移**, 不动块内相对位置, 所以前面推开的最小
         间距不会被打回原形。
    """
    keys = [k for k in groups if groups[k]]
    if not keys:
        return pos, {}, 0.0, 0.0

    def _measure():
        """每个宏观块的质心 + 半径(覆盖自己所有主题 + 边距 + 松弛余量)。"""
        cs, rs = [], []
        for key in keys:
            pts = pos[groups[key]]
            c = pts.mean(axis=0)
            cs.append(c.copy())
            rs.append(float(np.max(np.linalg.norm(pts - c, axis=1))) + padding)
        return cs, rs

    centers, radii = _measure()
    centers = _push_apart(centers, radii)
    for i, key in enumerate(keys):
        idxs = groups[key]
        pos[idxs] += centers[i] - pos[idxs].mean(axis=0)

    # 等比缩放到"圆的整体"约 target, 并挪进正坐标(视图 fitInView 不用管负坐标)
    cx = np.array([c[0] for c in centers])
    cy = np.array([c[1] for c in centers])
    r = np.array(radii)
    lo = np.array([float((cx - r).min()), float((cy - r).min())])
    hi = np.array([float((cx + r).max()), float((cy + r).max())])
    scale = target / max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    pos = (pos - lo) * scale
    centers = [(c - lo) * scale for c in centers]
    radii = [rr * scale for rr in radii]

    # 松弛(最终像素单位)。放在定尺度**之后**才能让 min_gap 就是最终像素,
    # 和视图里节点直径同一量纲; 放前面的话参数要乘缩放系数, 忘了乘等于没设
    # (实测 min_gap=34 最终只有 12px, 而节点直径最大 40.8px)。
    _enforce_min_gap(pos, min_gap)

    # 松弛把点挪开了, 重新量一次半径 —— 这次量出来的才是最终圈, 因为是
    # 在松弛**之后**量的, 天然覆盖住挪开后的位置(第一版是在松弛前量的, 结果
    # 81 个主题掉到圈外)。padding 70 已经足够富余, 不必再加 min_gap 余量。
    centers, radii = _measure()
    centers = _push_apart(centers, radii)
    for i, key in enumerate(keys):
        idxs = groups[key]
        pos[idxs] += centers[i] - pos[idxs].mean(axis=0)

    lo = np.array([float(min(c[0] - rr for c, rr in zip(centers, radii, strict=True))),
                   float(min(c[1] - rr for c, rr in zip(centers, radii, strict=True)))])
    w = float(max(c[0] + rr for c, rr in zip(centers, radii, strict=True)) - lo[0])
    h = float(max(c[1] + rr for c, rr in zip(centers, radii, strict=True)) - lo[1])
    pos = pos - lo

    circles: dict = {}
    for i, key in enumerate(keys):
        if key is None:
            continue
        circles[str(key)] = (float(centers[i][0] - lo[0]),
                             float(centers[i][1] - lo[1]), radii[i])
    return pos, circles, w, h


def _push_apart(centers: list[np.ndarray], radii: list[float],
                iters: int = 600) -> list[np.ndarray]:
    """把重叠的圆两两推开。

    位移只走**需要量的 0.4**(阻尼 Gauss-Seidel): 走满 1.0 会在圆互相挤的
    时候来回震荡, 跑满迭代数还剩几十像素的重叠 —— 真库五个大圆就是这样。
    """
    centers = [np.array(c, dtype=float) for c in centers]
    for _ in range(iters):
        moved = False
        for i, j in itertools.combinations(range(len(centers)), 2):
            d = centers[j] - centers[i]
            dist = float(np.linalg.norm(d))
            overlap = radii[i] + radii[j] - dist
            if overlap <= 0:
                continue
            u = d / dist if dist > 1e-6 else np.array([1.0, 0.0])
            push = overlap * 0.4 + 0.5
            centers[i] = centers[i] - u * push
            centers[j] = centers[j] + u * push
            moved = True
        if not moved:
            break
    return centers


# ---------- 公共小工具 ----------

def _macro_groups(nodes: list[dict]) -> dict:
    """宏观主题名 → 该宏观下所有主题的下标。macro 为 None 的单独归一堆(不画圆)。"""
    out: dict = {}
    for i, node in enumerate(nodes):
        out.setdefault(node.get("macro"), []).append(i)
    return out


# ============================================================
# 配色
# ============================================================

def theme_color(index: int) -> str:
    """第 index 个主题的颜色。

    按黄金角取色相而不是 `index // k` 分组: 分组会让 182 个主题挤成十几种颜色,
    "一个主题一个颜色"就废了。黄金角保证相邻的两个主题也差得开。
    """
    hue = ((index * GOLDEN_ANGLE) % 360.0) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.72)
    return _hex(r, g, b)


def macro_color(index: int) -> str:
    """第 index 个宏观主题的背景圆颜色。

    刻意比节点**更冷更暗**: 圆是背景, 节点是内容, 谁亮谁暗定下来, 免得背景圆
    盖过节点文字。
    """
    hue = ((index * GOLDEN_ANGLE) % 360.0) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.38, 0.55)
    return _hex(r, g, b)


def _hex(r: float, g: float, b: float) -> str:
    return f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"
