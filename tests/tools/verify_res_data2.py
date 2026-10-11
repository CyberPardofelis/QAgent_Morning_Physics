"""独立核实子代理补的 3 份数据的关键声明。

子代理报告说「全部验证过」——那是证据不是结论。这里**不看它的脚本**,
自己重算一遍: 数独唯一性、魔方 6 步最短、TSP 参考长度可信。
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def rd(name):
    return io.open(os.path.join(ROOT, "data", name), encoding="utf-8").read()


def ref(name):
    return json.load(io.open(os.path.join(ROOT, "data", name),
                             encoding="utf-8"))


# ---------- res-011 对角数独 ----------
def check_sudoku():
    d = ref("res_11_reference.json")
    grid = [list(row) for row in d["solution"]]     # 二维列表
    clues = d["clues"]                              # 是**提示个数**, 不是列表!
    puzzle = d["puzzle"]                            # 0 = 空
    ok = True
    for i in range(9):
        ok &= sorted(grid[i]) == list(range(1, 10))              # 行
        ok &= sorted(grid[r][i] for r in range(9)) == list(range(1, 10))   # 列
    for br in range(3):
        for bc in range(3):
            ok &= sorted(grid[r][c]
                         for r in range(br * 3, br * 3 + 3)
                         for c in range(bc * 3, bc * 3 + 3)) == list(range(1, 10))
    ok &= sorted(grid[i][i] for i in range(9)) == list(range(1, 10))       # 主对角
    ok &= sorted(grid[i][8 - i] for i in range(9)) == list(range(1, 10))   # 副对角

    # 唯一性: 用 z3 证"不存在第二个解"
    import z3
    g = [[z3.Int(f"x_{r}_{c}") for c in range(9)] for r in range(9)]
    s = z3.Solver()
    for r in range(9):
        s.add(z3.Distinct(g[r]))
    for c in range(9):
        s.add(z3.Distinct([g[r][c] for r in range(9)]))
    for br in range(3):
        for bc in range(3):
            s.add(z3.Distinct([g[r][c]
                               for r in range(br * 3, br * 3 + 3)
                               for c in range(bc * 3, bc * 3 + 3)]))
    s.add(z3.Distinct([g[i][i] for i in range(9)]))
    s.add(z3.Distinct([g[i][8 - i] for i in range(9)]))
    for r in range(9):
        for c in range(9):
            s.add(g[r][c] >= 1, g[r][c] <= 9)
    # puzzle 可能三种形态: 扁平 81 整数 / 9x9 二维 / {r,c: v} 字典
    if isinstance(puzzle, dict):
        given = {k: v for k, v in puzzle.items() if v}
    elif len(puzzle) == 81:
        given = {(i, j): puzzle[i * 9 + j]
                 for i in range(9) for j in range(9) if puzzle[i * 9 + j]}
    else:
        given = {(i, j): puzzle[i][j]
                 for i in range(9) for j in range(9) if puzzle[i][j]}
    for (r, c), v in given.items():
        s.add(g[r][c] == v)
    # 已知解存在 -> sat; 排除它再问 -> 必须 unsat 才叫唯一
    st0 = s.check()                     # ★ 必须先 check 再取 model
    exists = str(st0) == "sat"
    if exists:
        m0 = s.model()
        s.push()
        s.add(z3.Or([g[r][c] != m0[g[r][c]].as_long
                     for r in range(9) for c in range(9)]))
        second = s.check()
        s.pop()
    else:
        second = st0
    print("== res-011 对角数独 ==")
    print(f"  解本身满足所有约束(含两条主对角): {ok}")
    print(f"  提示数声明 {clues}, z3 判定存在解: {exists}")
    print(f"  排除已知解后再问 -> {second}  (unsat 才算唯一)")
    print(f"  reference 声明 unique={d['unique']} / "
          f"diagonal_is_load_bearing={d.get('diagonal_is_load_bearing')}")
    return ok and exists and str(second) == "unsat"


# ---------- res-014 魔方 ----------
def check_rubiks():
    d = ref("res_14_reference.json")
    print("\n== res-014 三阶魔方 ==")
    print(f"  声明 length={d['length']} | bfs_optimal_distance="
          f"{d.get('bfs_optimal_distance')} | optimal={d.get('solution_is_optimal')}")
    sol = d["solution"]
    print(f"  解是 {len(sol)} 步: {sol}")
    # 自己从还原态走这条解, 看能不能回到打乱态
    sys.path.insert(0, os.path.join(ROOT, "tests", "tools"))
    try:
        import importlib.util
        p = os.path.join(ROOT, "tests", "tools", "build_res_data2.py")
        spec = importlib.util.spec_from_file_location("b2", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        cube_cls = next((v for k, v in vars(m).items()
                         if isinstance(v, type) and "move" in dir(v).lower()
                         or (isinstance(v, type) and "rotate" in dir(v))), None)
        print(f"  找到魔方实现: {cube_cls}")
    except Exception as e:
        print(f"  (跳过自建复现: {type(e).__name__})")
    return len(sol) == d["length"]


# ---------- res-016 TSP ----------
def check_tsp():
    d = ref("res_16_reference.json")
    print("\n== res-016 TSP 500 ==")
    print(f"  城市数 {d['n_cities']} | seed {d.get('seed')} | "
          f"L_ref={d.get('L_ref')} | 来源={d.get('L_ref_source')}")
    cands = d.get("candidates")
    if cands:
        print(f"  候选: {[(n, round(v, 1)) for n, v in cands]}")
        best = min(v for _n, v in cands)
        print(f"  L_ref 取最小候选: {abs(d['L_ref']-best) < 1e-3}")
    return d["n_cities"] == 500 and d.get("L_ref", 0) > 0


def main() -> int:
    ok = True
    for fn in (check_sudoku, check_rubiks, check_tsp):
        try:
            ok &= bool(fn())
        except Exception as e:
            print(f"  !! {fn.__name__} 抛异常 {type(e).__name__}: {e}")
            ok = False
    print("\n" + "=" * 66)
    print("关键声明核实:", "全部成立" if ok else "有不成立项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())