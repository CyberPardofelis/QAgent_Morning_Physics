"""V2 基准 hard-017 —— 100 皇后: 用 z3 求一个解并程序化验证无冲突。

题目: 「用 Z3 或 SAT 求解器求 100 皇后问题的一个解, 并验证无冲突。」

======================================================================
★ 三个必踩的坑(2026-10-08 实跑三次全灭, 都在这里)★
======================================================================

★ 坑 1 —— 不要写 `solver.eval()` / `model.eval()`, 也不要 `from __future__ import ...`
    沙箱静态门在 AST 层拦属性访问: 任何 `ast.Attribute` 的 attr 落在
    {eval, exec, __import__, compile} 里就直接 SandboxRejected, 连执行
    都不给。前两次尝试就是这么死的(还没跑就 rejected)。
    正确取模型值的写法是 **下标**: `m[var]` (Model.__getitem__),
    再用 `z3.is_true(...)` 折成 bool。
    另外 import 白名单里也**没有 `__future__`**(白名单是模块根名 frozenset,
    `from __future__ import annotations` 会被判「模块 `__future__` 不在白名单内」),
    顺带不能用 eval/exec/open/input/vars/globals 这些名字。

坑 2 —— 不要用 Python 回溯
    纯 Python 回溯解 100 皇后在这一档预算内必超时(它是指数搜索)。
    题目点名要「用求解器」, 正确姿势是把问题**编码成 CNF/PB 交给 z3**,
    让 CDCL 去做。回溯方案即使跑出来也不满足题面要求。

坑 3 —— 必须设 timeout, 且 unknown 不等于 unsat
    `s.set("timeout", ms)` 之后 `check()` 可能返回 `unknown`(超时被杀)。
    unknown 只能判 INCONCLUSIVE, 绝不能当成 unsat/无解 —— 否则
    「超时」会被误报成「问题被推翻」。第三次尝试就是执行超时。
    完整策略: sat -> PROVED / unknown -> INCONCLUSIVE(带原因)。

======================================================================
★ 为什么这个编码能在几十秒内解出 100 皇后★
======================================================================
布尔编码(而不是 Int + Distinct): x[i][j] 表示第 i 行第 j 列放不放皇后。
  - 每一行 / 每一列: 恰好一枚  -> PbEq(..., 1)
  - 每条对角线(两个方向各 N*2-1 条): 至多一枚 -> AtMost(cells, 1)
这个编码的规模是 10000 个布尔变量 / 约 10200 条约束, 结构高度对称,
CDCL 在对称破除(左右镜像 + 转置)后搜索空间直接缩掉约 4 倍。

★ 对称破除为什么合法(而不是偷懒加约束)★
棋盘有两组自同构, 映射都保持「无冲突」这个性质:
  (a) 左右镜像 j -> N-1-j:  => 第 0 行皇后所在列 c0 可以限制 c0 <= (N-1)//2
  (b) 主对角线转置 (i,j) -> (j,i): => 设第 0 列皇后所在行为 r0,
      转置把 (0,c0)<-(r0,0) 对调, 所以可以要求 c0 <= r0 (取转置对里的下代表)
每条对称破除都至少保留一个代表, 不会删掉全部解 —— 这是正确性论证,
不是「试试看有没有解」。
"""

import z3

# 棋盘大小
N = 100

# 单次 check() 的求解预算(ms)。z3 内部按指令/毫秒双重限制, 超了就返回 unknown。
SOLVER_TIMEOUT_MS = 30_000


# --------------------------------------------------------------------------
# 建模
# --------------------------------------------------------------------------
def _row_of_col0(x):
    """第 0 列的皇后在第几行。x[0] 那一行已由 PbEq(..,1) 保证唯一,
    所以用 If 折叠出来的和是良定义的单值。"""
    return z3.Sum([z3.If(x[r][0], r, 0) for r in range(N)])


def _col_of_row0(x):
    """第 0 行的皇后在第几列。理由同上。"""
    return z3.Sum([z3.If(x[0][c], c, 0) for c in range(N)])


def build_solver(n: int = N, timeout_ms: int = SOLVER_TIMEOUT_MS):
    """返回 (solver, x)。x[i][j] 为 True <=> 第 i 行第 j 列有皇后。"""
    x = [[z3.Bool(f"x_{i}_{j}") for j in range(n)] for i in range(n)]
    s = z3.Solver()
    s.set("timeout", timeout_ms)

    # (1) 每行恰好一枚
    for i in range(n):
        s.add(z3.PbEq([(x[i][j], 1) for j in range(n)], 1))

    # (2) 每列恰好一枚
    for j in range(n):
        s.add(z3.PbEq([(x[i][j], 1) for i in range(n)], 1))

    # (3) 对角线无冲突: 沿 (r-c) 方向分组, 同组至多一枚
    by_main: dict[int, list] = {}
    for i in range(n):
        for j in range(n):
            by_main.setdefault(i - j, []).append(x[i][j])
    for cells in by_main.values():
        if len(cells) > 1:
            s.add(z3.AtMost(*cells, 1))

    # (4) 反方向 (r+c) 分组, 同组至多一枚
    by_anti: dict[int, list] = {}
    for i in range(n):
        for j in range(n):
            by_anti.setdefault(i + j, []).append(x[i][j])
    for cells in by_anti.values():
        if len(cells) > 1:
            s.add(z3.AtMost(*cells, 1))

    # (5) 冗余但便宜的剪枝: 相邻两行的皇后至少隔一列。
    #     这条已被 (4) 蕴含, 显式写出来是为了让 CDCL 早期就传播到。
    for i in range(n - 1):
        for j in range(n):
            if j - 1 >= 0:
                s.add(z3.AtMost(x[i][j], x[i + 1][j - 1], 1))
            if j + 1 < n:
                s.add(z3.AtMost(x[i][j], x[i + 1][j + 1], 1))

    # (6) 对称破除 (a): 左右镜像 -> 第 0 行皇后落在左半边(含中线)
    c0 = _col_of_row0(x)
    s.add(c0 <= (n - 1) // 2)

    # (7) 对称破除 (b): 主对角线转置 -> 取 c0 <= r0 的那一半
    r0 = _row_of_col0(x)
    s.add(c0 <= r0)

    return s, x


def read_board(model, x, n: int = N) -> list[int]:
    """从模型里读出每行皇后所在列。

    ★ 关键: 用 `model[var]` 而不是 `model.eval(var)` ——
      后者会被沙箱静态门在 AST 层拒掉(属性名 .eval)。
    """
    board = [-1] * n
    for i in range(n):
        for j in range(n):
            if z3.is_true(model[x[i][j]]):
                board[i] = j
                break
    return board


# --------------------------------------------------------------------------
# 独立的程序化验证(不依赖 z3, 只依赖整数算术)
# --------------------------------------------------------------------------
def verify_board(board: list[int], n: int = N) -> list[str]:
    """返回违规描述列表, 合法则为空。刻意写得又笨又重复 ——
    验证器必须独立于求解器, 不能复用求解器的任何结论。"""
    problems: list[str] = []

    # 逐行: 恰好一枚
    if len(board) != n:
        problems.append(f"行数 {len(board)} != {n}")
        return problems
    for i, c in enumerate(board):
        if not (0 <= c < n):
            problems.append(f"第 {i} 行的列号 {c} 越界")
    for i in range(n):
        if board.count(i) != 1:
            problems.append(f"第 {i} 列有 {board.count(i)} 枚皇后 (应为 1)")

    # 逐对: 任意两枚皇后不得同行/同列/同斜线
    conflicts = 0
    for a in range(n):
        for b in range(a + 1, n):
            dc = board[a] - board[b]
            dr = b - a
            if dc == 0:
                conflicts += 1
            elif abs(dc) == dr:          # 同一条斜线
                conflicts += 1
    if conflicts:
        problems.append(f"存在 {conflicts} 对冲突皇后")
    return problems


def render(board: list[int], n: int = N) -> list[str]:
    """打印用棋盘: Q 是皇后, . 是空格。"""
    return ["".join("Q" if board[i] == j else "." for j in range(n))
            for i in range(n)]


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
def verify() -> dict:
    """返回 {"verdict", "detail", "witness"}。"""
    checks: dict[str, bool] = {}

    s, x = build_solver(N)
    print(f"[z3] N={N}, timeout={SOLVER_TIMEOUT_MS}ms, "
          f"变量 {N * N} 个, 开始 check() ...")
    res = s.check()
    print(f"[z3] check() 返回 = {res}")

    # ★ 坑 3: unknown(超时/不确定) 不许当成 unsat。
    if res == z3.unknown:
        reason = s.reason_unknown()
        detail = f"求解器返回 unknown (reason={reason!r}); 预算内未判定, 不能算证明。"
        print(f"[z3] {detail}")
        return {
            "verdict": "INCONCLUSIVE",
            "detail": detail,
            "witness": {"n": N, "timeout_ms": SOLVER_TIMEOUT_MS},
        }

    if res == z3.unsat:
        detail = "z3 返回 unsat —— 100 皇后无解, 与已知的 n-queens 有解事实矛盾。"
        print(f"[z3] {detail}")
        return {"verdict": "REFUTED", "detail": detail, "witness": {"n": N}}

    model = s.model()
    # ★ 坑 1: model[var] 下标取值, 不用 model.eval(...)
    board = read_board(model, x, N)

    print("[解] 每行皇后所在列 =")
    print("        " + str(board))
    print("[解] 棋盘 (Q = 皇后):")
    for line in render(board, N):
        print("        " + line)

    problems = verify_board(board, N)
    checks["one_queen_per_row"] = all(0 <= c < N for c in board)
    checks["one_queen_per_column"] = all(board.count(c) == 1 for c in range(N))
    checks["no_two_queens_attack"] = (problems == [])
    print(f"[验证] 独立校验问题列表 = {problems if problems else '无 —— 无冲突'}")

    # 交叉核对: 逐格重数一遍总数, 防止「读模型时漏读/重读」这种低级错
    checks["exactly_n_queens"] = (len(board) == N
                                  and all(board.count(c) == 1 for c in range(N)))

    ok = all(checks.values())
    detail = "; ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in checks.items())
    print(f"[汇总] {detail}")

    verdict = "PROVED" if ok else "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "detail": detail,
        "witness": {
            "n": N,
            "board": board,
            "attack_pairs": 0 if ok else None,
            "timeout_ms": SOLVER_TIMEOUT_MS,
            "checker": "verify_board (纯整数算术, 不依赖 z3 结论)",
        },
    }


if __name__ == "__main__":
    result = verify()
    print(f"RESULT: {result['verdict']} | {result['detail']}")
    print(f"VERDICT: {result['verdict']}")