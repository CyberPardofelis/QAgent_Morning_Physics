"""独立核实 res-011 数独盘面: 存在解 + 解唯一。

★ 只从 agent 会读的 data/res_11_sudoku.txt 解析盘面 ★
不读 reference.json —— 那样等于用答案验答案。目的是确认"题面本身是良定义的"。
"""
from __future__ import annotations

import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
P = os.path.join(ROOT, "data", "res_11_sudoku.txt")


def parse_grid(text: str) -> list[list[int]]:
    """从 txt 里抓那 9 行 '.'/'1'..'9' 的盘面。"""
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) == 9 and all(ch == "." or ch.isdigit() for ch in s):
            rows.append([0 if ch == "." else int(ch) for ch in s])
    return rows


def main() -> int:
    txt = io.open(P, encoding="utf-8").read()
    grid = parse_grid(txt)
    if len(grid) != 9:
        print(f"[!!] 从 txt 里只解析出 {len(grid)} 行盘面, 期望 9 行")
        return 1
    clues = sum(1 for r in grid for v in r if v)
    print(f"从题面解析出盘面: 9 行, 线索 {clues} 个")

    import z3
    g = [[z3.Int(f"x_{r}_{c}") for c in range(9)] for r in range(9)]
    s = z3.Solver()
    for r in range(9):
        s.add(z3.Distinct(g[r]))
        for c in range(9):
            s.add(g[r][c] >= 1, g[r][c] <= 9)
    for c in range(9):
        s.add(z3.Distinct([g[r][c] for r in range(9)]))
    for br in range(3):
        for bc in range(3):
            s.add(z3.Distinct([g[r][c]
                               for r in range(br * 3, br * 3 + 3)
                               for c in range(bc * 3, bc * 3 + 3)]))
    s.add(z3.Distinct([g[i][i] for i in range(9)]))              # 主对角
    s.add(z3.Distinct([g[i][8 - i] for i in range(9)]))          # 副对角
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                s.add(g[r][c] == grid[r][c])

    st0 = s.check()
    print(f"z3 判定存在解: {st0}")
    if str(st0) != "sat":
        return 1
    m0 = s.model()
    sol = [[m0[g[r][c]].as_long() for c in range(9)] for r in range(9)]
    for row in sol:
        print("   ", "".join(map(str, row)))

    # 唯一性: 排除这个解后再问
    s.push()
    s.add(z3.Or([g[r][c] != m0[g[r][c]].as_long()
                 for r in range(9) for c in range(9)]))
    st1 = s.check()
    s.pop()
    print(f"排除该解后再查(唯一性): {st1}  ->  "
          f"{'★ 唯一' if str(st1) == 'unsat' else '★ 不唯一!'}")

    # 顺带验证: 不加对角约束时有几个解(说明对角约束是"有效"的而非摆设)
    s2 = z3.Solver()
    for r in range(9):
        s2.add(z3.Distinct(g[r]))
        for c in range(9):
            s2.add(g[r][c] >= 1, g[r][c] <= 9)
    for c in range(9):
        s2.add(z3.Distinct([g[r][c] for r in range(9)]))
    for br in range(3):
        for bc in range(3):
            s2.add(z3.Distinct([g[r][c]
                                for r in range(br * 3, br * 3 + 3)
                                for c in range(bc * 3, bc * 3 + 3)]))
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                s2.add(g[r][c] == grid[r][c])
    s2.push()
    s2.add(z3.Or([g[r][c] != m0[g[r][c]].as_long()
                  for r in range(9) for c in range(9)]))
    st2 = s2.check()
    s2.pop()
    print(f"去掉两条对角约束后同一盘面: {st2} "
          f"-> {'对角约束确实在起作用' if str(st2) == 'sat' else '去掉也一样(约束冗余)'}")

    return 0 if str(st1) == "unsat" else 1


if __name__ == "__main__":
    sys.exit(main())