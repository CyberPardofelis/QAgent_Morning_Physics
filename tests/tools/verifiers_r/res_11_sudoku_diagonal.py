"""res-011-sudoku-diagonal-z3 的独立 verifier。

题目：用 Z3 求解一个对角数独（两条主对角线也含 1-9），并证明解唯一。

## 为什么盘面由 verifier 自己定义
题库 params 指向 ``data/res_11_sudoku.txt``，但仓库里**没有这个文件**（已确认）。
没有唯一确定的题面，就没法做独立判定 —— 所以本 verifier 把题面冻结在这里：
``PUZZLE`` 就是 agent 面对的那道题，``SOLUTION`` 是它的唯一解。

## 盘面是怎么造出来的（造法，可完全复现）
1. 用自带的 **LCG 线性同余发生器**（固定种子 ``SEED``，不依赖 Python 的 random，
   跨版本可复现）把 81 个格子打乱，再给每格打乱一个数字候选序。
2. 回溯填满 9x9，约束同时包含：行、列、宫、**两条主对角线**各为 1-9 排列。
   搜到的第一个满盘就是 ``_full_grid()``。对角数独满盘在这个约束下很容易找到
   （随机回溯几乎立刻命中），所以不需要更重的构造。
   （试过"标准盘 + 带内换行/带间换带"的 1296x1296 置换族，那个族里**不存在**
   对角数独 —— 这是为什么换成了上面的直接回溯。）
3. 从满盘出发做贪心删线索：按行优先顺序逐格尝试删除，只要"带对角线约束的
   解数仍恰好为 1"就删掉。
4. 再跑一个收尾循环：只要"不带对角线约束时解数仍为 1"，就继续删一个**非对角线**
   的线索格。目的：让对角线约束成为**承重墙** —— 只用行/列/宫的解法一定会错。
   （否则对角线只是装饰，agent 漏掉对角线约束也能蒙对。）

`python res_11_sudoku_diagonal.py --dump` 会把上面整套流程重跑一遍并打印常量，
用来核对硬编码的 PUZZLE / SOLUTION 没写错。

## 制品格式
```
CONSTRUCT: {"solution": [[1..9] x 9],
            "unique": true,
            "uniqueness_certificate": {"method": "z3",
                                       "result": "unsat",          # 或 z3_output
                                       ...}}
```

## 判据（为什么这么判就是"真的做对了"）
* 行/列/宫/双对角线各自都是 1-9 的排列 —— 这只是**合法性**，说明 agent 交的
  确实是一个对角数独解，漏掉对角线约束的解在这一步就露馅。
* 与 PUZZLE 的线索一致 + 等于 SOLUTION —— SOLUTION 是 verifier **自己枚举**出来的
  （见 :func:`_solve_clues`），所以这一条等价于"agent 给的是这道题的唯一解"。
* 唯一性证书 —— 题目要求证明唯一性。只贴一个解不算通过。证书必须给出
  "第二解查询返回 UNSAT" 的证据；拿不到就把 unique 判失败。
  注意 verifier 自己的枚举已经**独立确认了唯一性**，证书这一项是额外要求，
  但它是"agent 是否真的做了证明"与"agent 只是猜到了答案"的区分点。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # 保证无论怎么被 import 都能拿到 common
    sys.path.insert(0, str(_HERE))

from common import Check, parse_artifact  # noqa: E402  (公共件, 不重写)

# --------------------------------------------------------------------------
# 冻结的题面与答案
# --------------------------------------------------------------------------

#: 满盘构造用的固定种子（自带 LCG，不依赖 random 模块 -> 跨版本可复现）。
SEED: int = 20260908

#: 9x9 对角数独题面, 81 个整数, 0 表示空格（19 个线索）。
#: 性质已用**独立的 z3 脚本**交叉核对过：带对角线约束恰好 1 组解, 不带则 >=12 组 ——
#: 所以对角线约束是承重墙, 只用行/列/宫的解法一定会错。
PUZZLE: tuple[int, ...] = (
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 4,
    0, 7, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 6, 0,
    0, 0, 5, 0, 0, 0, 3, 8, 0,
    1, 3, 0, 0, 5, 0, 0, 0, 8,
    8, 0, 7, 0, 0, 2, 0, 0, 3,
    0, 0, 4, 6, 0, 8, 5, 1, 0,
)

#: 该题面的唯一解（对角线约束下），9x9。
SOLUTION: tuple[int, ...] = (
    3, 4, 9, 8, 6, 5, 1, 7, 2,
    5, 8, 2, 1, 7, 4, 9, 3, 6,
    7, 6, 1, 2, 9, 3, 8, 5, 4,
    6, 7, 3, 5, 8, 1, 4, 2, 9,
    2, 1, 8, 3, 4, 9, 7, 6, 5,
    4, 9, 5, 7, 2, 6, 3, 8, 1,
    1, 3, 6, 9, 5, 7, 2, 4, 8,
    8, 5, 7, 4, 1, 2, 6, 9, 3,
    9, 2, 4, 6, 3, 8, 5, 1, 7,
)

#: 手抄常量最容易错的就是长度/个数, 导入时就钉死。
assert len(PUZZLE) == 81 and len(SOLUTION) == 81
assert sum(1 for v in PUZZLE if v) == 19

_ALL9 = tuple(range(1, 10))


# --------------------------------------------------------------------------
# 核心：自己枚举，不碰 agent 的代码
# --------------------------------------------------------------------------

def _solve_clues(clues, use_diagonal: bool = True, limit: int = 2,
                 node_budget: int | None = None) -> list[tuple[int, ...]]:
    """在 (行列宫[+对角线]) 约束下枚举解，最多找 ``limit`` 个。

    位的 mask 增量维护 + MRV（最少候选优先）选格，81 格规模下毫秒级。
    这里返回的解就是 verifier 的**独立真值**，agent 说什么都不影响它。

    ``node_budget``：只在**构造题面**时用来给搜索封顶。原因是方向不对称：
    找到一个矛盾解很快，而"证明只有 1 个解"要把整棵树走完 —— 稀疏题面下
    去掉对角线约束再枚举，可能是几百万节点。不封顶的话构造阶段会卡死。
    真正的判定（:func:`verify`）一律不封顶，不接受"预算耗尽"这种结论。
    """
    grid = list(clues)
    row = [0] * 9
    col = [0] * 9
    box = [0] * 9
    d1 = d2 = 0
    for i, v in enumerate(grid):
        if not v:
            continue
        r, c = divmod(i, 9)
        b = r // 3 * 3 + c // 3
        bit = 1 << (v - 1)
        used = row[r] | col[c] | box[b]
        if use_diagonal and r == c:
            used |= d1
        if use_diagonal and r + c == 8:
            used |= d2
        if used & bit:  # 线索本身就冲突
            return []
        row[r] |= bit
        col[c] |= bit
        box[b] |= bit
        if use_diagonal and r == c:
            d1 |= bit
        if use_diagonal and r + c == 8:
            d2 |= bit

    empty = [i for i, v in enumerate(grid) if not v]
    sols: list[tuple[int, ...]] = []
    budget = [node_budget]

    def rec(todo: list[int]) -> None:
        nonlocal d1, d2  # rec 里会 d1 |= bit, 不声明 nonlocal 会被当成局部变量而读不到
        if len(sols) >= limit:
            return
        if budget[0] is not None:
            budget[0] -= 1
            if budget[0] < 0:
                return
        if not todo:
            sols.append(tuple(grid))
            return
        best, best_cands = -1, ()
        for i in todo:
            r, c = divmod(i, 9)
            b = r // 3 * 3 + c // 3
            used = row[r] | col[c] | box[b]
            if use_diagonal and r == c:
                used |= d1
            if use_diagonal and r + c == 8:
                used |= d2
            cands = [v for v in _ALL9 if not (used >> (v - 1)) & 1]
            if not cands:  # 死路
                return
            if best < 0 or len(cands) < len(best_cands):  # 注意: -1 是真值, 不能写 not best
                best, best_cands = i, cands
                if len(cands) == 1:
                    break
        rest = [i for i in todo if i != best]
        r, c = divmod(best, 9)
        b = r // 3 * 3 + c // 3
        for v in best_cands:
            bit = 1 << (v - 1)
            grid[best] = v
            row[r] |= bit
            col[c] |= bit
            box[b] |= bit
            on_d1 = use_diagonal and r == c
            on_d2 = use_diagonal and r + c == 8
            if on_d1:
                d1 |= bit
            if on_d2:
                d2 |= bit
            rec(rest)
            grid[best] = 0
            row[r] ^= bit
            col[c] ^= bit
            box[b] ^= bit
            if on_d1:
                d1 ^= bit
            if on_d2:
                d2 ^= bit
            if len(sols) >= limit:
                return

    rec(empty)
    return sols


def _check_grid(flat) -> tuple[bool, dict[str, bool]]:
    """校验 9x9 解: 行/列/宫/双对角线各自必须是 1-9 的排列。"""
    res = {
        "shape_ok": False,
        "values_ok": False,
        "rows_ok": False,
        "cols_ok": False,
        "boxes_ok": False,
        "main_diagonal_ok": False,
        "anti_diagonal_ok": False,
    }
    if not isinstance(flat, list) or len(flat) != 9:
        return False, res
    grid: list[list[int]] = []
    for row in flat:
        if not isinstance(row, list) or len(row) != 9:
            return False, res
        out = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int):
                return False, res
            out.append(v)
        grid.append(out)
    res["shape_ok"] = True
    res["values_ok"] = all(1 <= v <= 9 for row in grid for v in row)
    res["rows_ok"] = all(sorted(r) == list(_ALL9) for r in grid)
    res["cols_ok"] = all(sorted(grid[r][c] for r in range(9)) == list(_ALL9) for c in range(9))
    res["boxes_ok"] = all(
        sorted(grid[3 * br + dr][3 * bc + dc] for dr in range(3) for dc in range(3)) == list(_ALL9)
        for br in range(3)
        for bc in range(3)
    )
    res["main_diagonal_ok"] = sorted(grid[i][i] for i in range(9)) == list(_ALL9)
    res["anti_diagonal_ok"] = sorted(grid[i][8 - i] for i in range(9)) == list(_ALL9)
    return all(res.values()), res


# --------------------------------------------------------------------------
# 唯一性证书: 只认"第二解查询返回 UNSAT"的机器证据
# --------------------------------------------------------------------------
_UNSAT_KEYS = ("result", "query_result", "status", "verdict", "outcome", "z3_output", "output")
_UNSAT_STRS = ("unsat", "un_sat", "unsatisfiable")


def _certificate_proves_unsat(cert: Any) -> tuple[bool, str]:
    """证书里必须能找到 UNSAT 证据。返回 (是否成立, 说明)。

    接受两种机器可读的写法:
      * 显式字段: ``{"result": "unsat"}`` / ``{"unsat": true}`` /
        ``{"no_second_solution": true}`` / ``{"second_solution_exists": false}``
      * z3 原始输出: 任意字段里出现 "unsat" 字样
    只说 "唯一性成立" 而不给证据的, 一律判失败 —— 那是断言, 不是证明。
    """
    if not isinstance(cert, dict):
        return False, "certificate 缺失或不是对象"
    if cert.get("unsat") is True or cert.get("no_second_solution") is True:
        return True, "unsat=true"
    if cert.get("second_solution_exists") is False:
        return True, "second_solution_exists=false"
    for k in _UNSAT_KEYS:
        v = cert.get(k)
        if isinstance(v, str) and any(s in v.lower() for s in _UNSAT_STRS):
            return True, f"{k} 含 unsat 证据"
    return False, "证书里找不到 UNSAT 证据"


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

def verify(stdout: str, code: str, claim: str) -> dict:
    """独立判定 res-011 的作答。code/claim 不参与判定（不信任 agent 的代码）。"""
    ck = Check()
    art = parse_artifact(stdout, "CONSTRUCT")
    if art is None:
        ck.expect("artifact_present", False, "未提交 CONSTRUCT 制品")
        return _result(ck, None, "artifact 缺失")

    grid = art.get("solution")
    ok, sub = _check_grid(grid)
    ck.expect("grid_valid_diagonal_sudoku", ok,
              "解不是合法的对角数独(行/列/宫/双对角线须各为 1-9 排列): "
              + json.dumps(sub, ensure_ascii=False))
    ck.details["grid_checks"] = sub

    flat = [v for row in (grid if isinstance(grid, list) else []) if isinstance(row, list) for v in row] \
        if isinstance(grid, list) else []
    if len(flat) == 81 and isinstance(grid, list) and all(isinstance(r, list) for r in grid):
        flat = [v for row in grid for v in row]

    # 与冻结题面的线索一致 —— 证明它解的是**这道**题
    if SOLUTION:
        matches = len(flat) == 81 and all(
            clue == 0 or clue == flat[i] for i, clue in enumerate(PUZZLE)
        )
        ck.expect("matches_puzzle_clues", matches, "解与冻结题面 PUZZLE 的线索不一致")
        ck.expect("equals_unique_solution", list(flat) == list(SOLUTION),
                  "解不等于该题面的唯一解 SOLUTION")

    # verifier 自己的独立枚举: 确认题面确实只有一个(带对角线的)解
    if PUZZLE:
        found = _solve_clues(PUZZLE, use_diagonal=True, limit=2)
        ck.note("independent_enumeration", [list(s) for s in found])
        ck.expect("puzzle_really_unique", len(found) == 1 and tuple(found[0]) == SOLUTION,
                  "verifier 自己的枚举没复现出唯一解, 题面常量可能写错了")
        ck.note("solutions_without_diagonal", len(_solve_clues(PUZZLE, use_diagonal=False, limit=2)))

    # agent 声称的唯一性 + UNSAT 证书
    ck.expect("claims_unique", art.get("unique") is True, "未声称 unique=true")
    cert = art.get("uniqueness_certificate")
    if cert is None:
        cert = art.get("certificate")
    cert_ok, why = _certificate_proves_unsat(cert)
    ck.expect("uniqueness_certificate", cert_ok, f"唯一性证明制品不合格: {why}")
    ck.details["certificate_reason"] = why

    return _result(ck, art, "UNSAT 证书" if cert_ok else why)


def _result(ck: Check, art: dict | None, why: str) -> dict:
    final = None
    if art is not None:
        final = json.dumps(
            {"solution": art.get("solution"), "unique": art.get("unique")},
            ensure_ascii=False,
        )
    return {
        "correct": ck.correct,
        "certificate": (why if ck.correct else f"FAILED: {ck.reason()}"),
        "final_answer": final,
        "details": ck.details,
    }


# --------------------------------------------------------------------------
# 题面构造（造法见 docstring）, --dump 时重跑一遍
# --------------------------------------------------------------------------

class _LCG:
    """固定种子的线性同余发生器。

    为什么不用 ``random``：Python 各版本的 shuffle/shuffle 实现可能变，
    那会让"造法可复现"这句话失效。这里自己实现，任何版本跑出来都一样。
    """

    def __init__(self, seed: int) -> None:
        self.s = seed & 0x7FFFFFFF

    def next(self) -> int:
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s

    def shuffle(self, seq: list) -> None:
        for i in range(len(seq) - 1, 0, -1):
            j = self.next() % (i + 1)
            seq[i], seq[j] = seq[j], seq[i]


def _full_grid(seed: int = SEED) -> list[int]:
    """随机回溯填一个**对角**数独满盘（行/列/宫/双对角线各为 1-9 排列）。

    必须用 MRV（候选最少的格先填）+ 随机打破平局。只按随机格序硬填的话，
    末尾会指数爆炸 —— 实测直接卡死，加了 MRV 后同一构造毫秒级完成。
    """
    rnd = _LCG(seed)
    grid = [0] * 81

    def candidates(i: int) -> list[int]:
        r, c = divmod(i, 9)
        used = {grid[9 * r + j] for j in range(9)}
        used |= {grid[9 * j + c] for j in range(9)}
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for dr in range(3):
            for dc in range(3):
                used.add(grid[9 * (br + dr) + bc + dc])
        # 对角线约束只对**在该对角线上的格**生效, 而且要跟该对角线上的**所有**格比较。
        # (只跟"同一行的那个对角格"比是不够的 —— 两个对角线格不同行不同列不同宫,
        #  行/列/宫检查根本抓不到它们之间的冲突。)
        if r == c:
            used |= {grid[9 * j + j] for j in range(9)}
        if r + c == 8:
            used |= {grid[9 * j + (8 - j)] for j in range(9)}
        used.discard(0)
        return [v for v in _ALL9 if v not in used]

    def fill(left: list[int]) -> bool:
        if not left:
            return True
        pick, pick_cands = -1, []
        best = 99
        pool = list(left)
        rnd.shuffle(pool)
        for i in pool:
            cs = candidates(i)
            if len(cs) < best:
                pick, pick_cands, best = i, cs, len(cs)
                if best <= 1:
                    break
        if not pick_cands:
            return False
        rnd.shuffle(pick_cands)
        rest = [i for i in left if i != pick]
        for v in pick_cands:
            grid[pick] = v
            if fill(rest):
                return True
            grid[pick] = 0
        return False

    if not fill(list(range(81))):
        raise RuntimeError("未找到对角数独满盘")
    return grid


def _build_puzzle() -> tuple[tuple[int, ...], tuple[int, ...]]:
    clues = _full_grid()

    # 贪心删线索: 只要"带对角线时仍恰好一个解"就删。
    # 删除顺序 = 对角线格优先 -> 对角线线索先被削掉, 对角线约束更容易变成承重墙。
    diag = [i for i in range(81) if i // 9 == i % 9 or i // 9 + i % 9 == 8]
    order = diag + [i for i in range(81) if i not in set(diag)]
    for i in order:
        if clues[i] == 0:
            continue
        keep = clues[i]
        clues[i] = 0
        if len(_solve_clues(clues, True, 2)) != 1:
            clues[i] = keep

    return tuple(clues), _solve_clues(clues, True, 1)[0]


def _dump() -> None:
    clues, sol = _build_puzzle()
    print("PUZZLE =", clues)
    print("SOLUTION =", sol)
    print("clues =", sum(1 for v in clues if v),
          "| with diag:", len(_solve_clues(clues, True, 5)),
          "| without diag:", len(_solve_clues(clues, False, 5)))


# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------

def _selftest() -> int:
    failures: list[str] = []

    def expect_true(name: str, cond: bool) -> None:
        print(("[OK] " if cond else "[FAIL] ") + name)
        if not cond:
            failures.append(name)

    # 题面常量自洽: 硬编码的 PUZZLE/SOLUTION 必须能被自己的枚举复现
    expect_true("题面常量自洽(自己枚举出唯一解)",
                len(_solve_clues(PUZZLE, True, 2)) == 1 and _solve_clues(PUZZLE, True, 1)[0] == SOLUTION)
    expect_true("对角线是承重墙(去掉对角线就不唯一)",
                len(_solve_clues(PUZZLE, False, 2)) >= 2)
    ok, _ = _check_grid([list(SOLUTION[i * 9:i * 9 + 9]) for i in range(9)])
    expect_true("SOLUTION 自身是合法对角数独", ok)

    def stdout_of(sol_grid, unique=True, cert=True):
        art = {"solution": sol_grid, "unique": unique}
        if cert:
            art["uniqueness_certificate"] = {"method": "z3",
                                             "query": "solution & Or(any cell differs)",
                                             "z3_output": "unsat"}
        return "VERDICT: PROVED\nCONSTRUCT: " + json.dumps(art, ensure_ascii=False)

    good = [list(SOLUTION[i * 9:i * 9 + 9]) for i in range(9)]

    # 正例
    r = verify(stdout_of(good), "", "")
    expect_true("正例(唯一解+UNSAT证书) -> correct=True", r["correct"])

    # 反例1: 交换第 0 行两个数 —— 行仍是排列, 但不满足列/对角线, 也不是本题解
    bad = [row[:] for row in good]
    bad[0][0], bad[0][1] = bad[0][1], bad[0][0]
    expect_true("反例1(破坏列的对称交换) -> correct=False", not verify(stdout_of(bad), "", "")["correct"])

    # 反例2: 给出**另一个合法对角数独解**(不是本题面的解)
    other = _solve_clues([0] * 81, True, 3)[2]
    alt = [list(other[i * 9:i * 9 + 9]) for i in range(9)]
    expect_true("反例2(合法但不是本题面的解) -> correct=False",
                not verify(stdout_of(alt), "", "")["correct"])

    # 反例3: 解对, 但没有唯一性证明制品
    expect_true("反例3(只有解没有UNSAT证书) -> correct=False",
                not verify(stdout_of(good, cert=False), "", "")["correct"])

    # 反例4: 证书只是空壳断言, 没有 UNSAT 证据
    shell = "VERDICT: PROVED\nCONSTRUCT: " + json.dumps(
        {"solution": good, "unique": True, "uniqueness_certificate": {"method": "z3", "note": "唯一"}},
        ensure_ascii=False)
    expect_true("反例4(证书无UNSAT证据) -> correct=False", not verify(shell, "", "")["correct"])

    # 反例5: 完全没有制品
    expect_true("反例5(无制品) -> correct=False", not verify("VERDICT: PROVED\n", "", "")["correct"])

    if failures:
        print("[SELFTEST FAILED] " + ", ".join(failures))
        return 1
    print("[OK] 自测全部通过")
    return 0


if __name__ == "__main__":
    if "--dump" in sys.argv:
        _dump()
    else:
        sys.exit(_selftest())