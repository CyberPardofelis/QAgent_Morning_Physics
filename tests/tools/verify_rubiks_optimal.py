"""独立核实 res-014: 那 6 步是不是**真的**最短。

★ 自己实现魔方, 不复用子代理的代码 ★
做法:
  1. 用 3D 坐标模型建状态(54 个 sticker, (位置, 法向))
  2. 用题面给的 6 步解回放 -> 必须回到 solved
  3. 从打乱态 BFS 到深度 5 -> 打乱态**不在**该 ball 里, 才说明 6 是最短

★ 为什么第 3 步不能省 ★
只验"6 步能解"不能说明"最短" —— 可能存在 4 步解而题面给了一条绕远的。
而题目明确要求「验证步数最少」。
"""
from __future__ import annotations

import io
import os
import re
import sys
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TXT = os.path.join(ROOT, "data", "res_14_rubiks.txt")
REF = os.path.join(ROOT, "data", "res_14_reference.json")

#: 面 -> (法向, 行轴, 列轴)。轴向量按题面"x 右 / y 上 / z 前"。
FACES = {
    "U": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "D": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "F": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "B": ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
    "R": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "L": ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
}
ORDER = "UDFBRL"


def slots():
    """54 个 sticker 的 (位置, 法向), 顺序 = 题面给出的读取顺序。"""
    out = []
    for f in ORDER:
        n, ra, ca = FACES[f]
        for i in range(3):
            for j in range(3):
                pos = tuple(n[k] + ra[k] * (i - 1) + ca[k] * (j - 1)
                            for k in range(3))
                out.append((pos, n))
    return out


SLOTS = slots()
IDX = {s: i for i, s in enumerate(SLOTS)}


def rot(v, axis, k):
    """绕 axis 轴转 k*90°(右手定则)。"""
    if axis == "x":
        y, z = v[1], v[2]
        for _ in range(k % 4):
            y, z = -z, y
        return (v[0], y, z)
    if axis == "y":
        x, z = v[0], v[2]
        for _ in range(k % 4):
            x, z = z, -x
        return (x, v[1], z)
    x, y = v[0], v[1]
    for _ in range(k % 4):
        x, y = -y, x
    return (x, y, v[2])


#: 面 -> 绕哪个轴转多少。从该面外侧看顺时针 = 绕法向逆时针 k=1。
AXIS_K = {"U": ("y", -1), "D": ("y", 1), "F": ("z", -1),
          "B": ("z", 1), "R": ("x", -1), "L": ("x", 1)}


def move_perm(face: str, k: int) -> tuple[int, ...]:
    """返回置换: 新状态[i] = 旧状态[perm[i]]。"""
    axis, kk = AXIS_K[face]
    kn = (k * kk) % 4
    n = FACES[face][0]
    perm = list(range(54))
    for i, (pos, nrm) in enumerate(SLOTS):
        if pos[{"x": 0, "y": 1, "z": 2}[axis]] == n[{"x": 0, "y": 1, "z": 2}[axis]]:
            npos = rot(pos, axis, kn)
            nnrm = rot(nrm, axis, kn)
            perm[i] = IDX[(npos, nnrm)]
        else:
            perm[i] = i
    return tuple(perm)


def apply(state: tuple, perm: tuple) -> tuple:
    return tuple(state[p] for p in perm)


MOVES = {}
for f in ORDER:
    for suf, k in (("", 1), ("'", 3), ("2", 2)):
        name = f + suf
        MOVES[name] = move_perm(f, k)


def parse_grid(txt: str) -> str:
    """抓题面那 6 行面。

    ★ 必须严格: 冒号后**恰好 9 个颜色字母** ★
    第一版用宽松正则, 把后面演示脚本里的 `from z3 import ...` 之类也匹配上了,
    结果解析出 303 格。宽匹配在这里不是"多认几个", 是把整个文件当盘面读。
    """
    rows = []
    # ★ 两个坑都踩过 ★
    #  1. 手写重复段数错 -> 要求 11 个字母, 行里只有 9 个, 一行都匹配不上
    #  2. 用 `(?:\s+([A-Za-z])){9}` -> Python 的重复组只保留**最后一次**捕获,
    #     于是每行只拿到 1 个字符(6 行 -> 6 格)。所以这里把 9 个字母
    #     **整体**捕获成一个组, 再 split。
    pat = re.compile(r"\s*([UDFBRL]):((?:\s+[A-Za-z]){9})\s*\Z")
    for line in txt.splitlines():
        m = pat.match(line)
        if m:
            rows.append("".join(m.group(2).split()))
    return "".join(rows)


def solved_state() -> tuple:
    # 第 f 面 9 格都是 f 的颜色; 颜色名用面字母
    st = []
    for i in range(0, 54, 9):
        st.extend([ORDER[i // 9]] * 9)
    return tuple(st)


def main() -> int:
    txt = io.open(TXT, encoding="utf-8").read()
    grid = parse_grid(txt)
    if len(grid) != 54:
        print(f"[!!] 从题面只解析到 {len(grid)} 格, 期望 54")
        return 1
    scrambled = tuple(grid)
    sol = __import__("json").load(io.open(REF, encoding="utf-8"))["solution"]
    print(f"题面打乱态解析: 54 格 | 题面给的解: {sol} ({len(sol)} 步)")

    # 1) 回放
    st = scrambled
    for mv in sol:
        st = apply(st, MOVES[mv])
    solved = solved_state()
    ok = st == solved
    print(f"回放 {len(sol)} 步后 == 还原态: {ok}")
    if not ok:
        print("   我这边没对上 —— 可能是记号约定理解不同, 不据此判数据有问题")
        return 1

    # 2) 深度 5 ball 里有没有打乱态(有 -> 6 步不是最短)
    target = scrambled
    seen = {target}
    frontier = deque([(target, 0)])
    found_depth = None
    while frontier:
        st, d = frontier.popleft()
        if d >= 5:
            continue
        for mv in MOVES.values():
            ns = apply(st, mv)
            if ns in seen:
                continue
            if ns == solved:
                found_depth = d + 1
                break
            seen.add(ns)
            frontier.append((ns, d + 1))
        if found_depth:
            break
    print(f"从打乱态搜到还原态的最短步数(<=5 的 ball): "
          f"{found_depth if found_depth else '>5 (未搜到)'}")
    print(f"★ 结论: 6 步{'确实是最短' if not found_depth else '不是最短, 存在 %d 步解' % found_depth}")
    return 0 if not found_depth else 1


if __name__ == "__main__":
    sys.exit(main())