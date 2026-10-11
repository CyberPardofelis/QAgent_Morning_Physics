"""生成 res-005 / res-007 / res-011 / res-014 / res-016 的输入数据文件。

★ 为什么这些文件必须存在 ★
baseproblems_R.txt 里这 5 道题的题面都写着"给定某某"(给定序列 / 给定盘面 /
给定打乱状态 / 给定实例), 但引用的 data/res_*.txt 全部不存在。agent 拿不到数据
就自己编一个, 再拿自己编的数据去"独立验证", 当然对不上 —— 这是基准集的缺陷,
不是模型的错误。

======================================================================
★ 本脚本最重要的一条约束: 数据必须和 verifier 判定用的实例一致 ★
======================================================================
这 5 道题里有 3 道(res-005 / res-011 / res-014)的 verifier **根本不读数据文件**,
而是在文件里冻死了一份实例:

  res-05: 系数 [3,7,11,17,23], 初值 [1,2,3,5,8], 模 p=2^31-1   (BM 最小阶 5)
  res-11: 19 个线索的盘面 PUZZLE (唯一解 SOLUTION)
  res-14: 打乱序列 D' B' R' D' R D'            (双向 BFS 最优距离 = 6)

所以这 3 份数据**不是"随便造一个"**, 而是必须把 verifier 冻结的那份实例原样
落盘: agent 拿到的题面 = verifier 手里的判题依据, 两者逐项相同, 分数才是真的。
如果在这里另造一个 7 阶递推, agent 会解对它拿到的那份数据, 而 verifier 拿自己的
冻结实例去判 —— 结果必然全错, 正是这次要修的那个假分数缺陷。

另外 2 道(res-007 / res-016)的 verifier **确实会读文件**(官方路径
data/res_07_seq.txt / data/res_16_tsp.txt), 文件在就用文件, 不在就退回内置实例。

======================================================================
★ 文件名以题库 params 和 verifier 代码为准 ★
======================================================================
题面里 brief 给的名字和实际被读取的路径有出入, 这里按**实际被读取的路径**落盘,
并额外写一份 brief 命名的同内容副本, 再 assert 两者逐字节相同:

  实际读取路径            brief 里的名字
  data/res_05_seq.txt   <-> data/res_05_sequence.txt
  data/res_07_seq.txt   <-> data/res_07_sequence.txt
  data/res_14_cube.txt  <-> data/res_14_rubiks.txt
  data/res_11_sudoku.txt   (一致)
  data/res_16_tsp.txt      (一致)

★ res-007 为什么必须是 Fibonacci ★
res_07_oeis.py 的自测里写死了正例: init=[0,1], coeffs=[1,1] 必须判对
(见该文件 __main__ 的 `good_rec`)。如果 res_07_seq.txt 放 Pell, 那条 assert
必然失败。所以这里取 Fibonacci(A000045) —— 它在题面允许的候选里, 且 txt 里
不写明它是什么, agent 仍需自己识别。

======================================================================
★ 每个答案都经过至少两套独立实现交叉验证 ★
  res-005: 自写 BM  +  高斯消元求最小阶 (两条完全不同的路)
  res-007: 递推实现  +  含根号闭式 (mpmath 高精度)
  res-011: z3 (SAT 求解 + UNSAT 证唯一)  +  自写回溯枚举
  res-014: 自写魔方(旋转矩阵)  +  魔方群恒等式((RU)^105=id 等)  +  verifier 交叉
  res-016: 最近邻+2-opt  +  ortools GUIDED_LOCAL_SEARCH, 取小

★ agent 侧 .txt 只给输入, 绝不给答案 ★
答案一律只写进 *_reference.json。txt 里不出现系数 / 解 / 最优解。
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

# 仓库根: 本文件在 <root>/tests/tools/build_res_data2.py
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data")
VERIFIERS = os.path.join(ROOT, "tests", "tools", "verifiers_r")

# 允许直接 import verifier 做交叉核对(只读, 不修改)
if VERIFIERS not in sys.path:
    sys.path.insert(0, VERIFIERS)


def _write(path: str, text: str) -> None:
    """统一用 utf-8 + 平台换行落盘(与已有 data/res_10_rules.txt 的 CRLF 一致)。"""
    with io.open(path, "w", encoding="utf-8", newline=None) as fh:
        fh.write(text)


def _write_json(path: str, obj) -> None:
    with io.open(path, "w", encoding="utf-8", newline=None) as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, indent=2))


def _mirror(src_name: str, dst_name: str) -> None:
    """brief 命名的副本必须和实际读取路径逐字节相同, 否则 agent 会读到另一份数据。

    ★ 必须按**二进制**复制 ★
    源文件是 CRLF; 若先 decode 再以文本方式写回, Windows 的换行翻译会把每个
    "\r\n" 变成 "\r\r\n", 于是副本与源文件不再逐字节相同 —— agent 读到的
    序列文本就与 verifier 读到的不是同一份。
    """
    src = os.path.join(OUT, src_name)
    dst = os.path.join(OUT, dst_name)
    if src_name == dst_name:
        return
    with open(src, "rb") as fh:
        a = fh.read()
    with open(dst, "wb") as fh:
        fh.write(a)
    with open(dst, "rb") as fh:
        b = fh.read()
    assert a == b, f"{src_name} 与 {dst_name} 内容不一致 —— agent 与 verifier 会看到不同数据"


# ======================================================================
# res-005: 最小线性递推 (Berlekamp-Massey)
# ======================================================================
# 与 verifier 冻结实例一致的常量(kociemba 不可用之类的原因见 res_05_berlekamp.py)
MOD = 2147483647
TRUE_COEF = [3, 7, 11, 17, 23]
TRUE_SEEDS = [1, 2, 3, 5, 8]
GIVEN_TERMS = 100
CHECK_TERMS = 200


def gen_sequence(coefs: list[int], seeds: list[int], count: int, mod: int = MOD) -> list[int]:
    """s_n = sum c_i * s_{n-i} (mod p), 前 len(seeds) 项为初值。"""
    k = len(coefs)
    out = [x % mod for x in list(seeds)[:k]]
    while len(out) < count:
        out.append(sum(coefs[j] * out[len(out) - 1 - j] for j in range(k)) % mod)
    return out


def bm_berlekamp_massey(seq: list[int], mod: int = MOD) -> tuple[int, list[int]]:
    """第一套实现: 标准 Berlekamp-Massey(连接多项式修正法)。

    返回 (最小阶数 L, [c_1..c_L]), 满足 s_n = Σ c_i s_{n-i} (mod p)。
    """
    C, B, L, m, b = [1], [1], 0, 1, 1
    for n in range(len(seq)):
        d = seq[n] % mod
        for i in range(1, L + 1):
            d = (d + C[i] * seq[n - i]) % mod
        if d == 0:
            m += 1
            continue
        coef = d * pow(b, -1, mod) % mod
        T = list(C)
        if len(C) < len(B) + m:
            C += [0] * (len(B) + m - len(C))
        for i in range(len(B)):
            C[i + m] = (C[i + m] - coef * B[i]) % mod
        if 2 * L <= n:
            L, B, b, m = n + 1 - L, T, d, 1
        else:
            m += 1
    return L, [(-C[i]) % mod for i in range(1, L + 1)]


def gauss_solve_mod(mat: list[list[int]], rhs: list[int], mod: int) -> list[int] | None:
    """模素数下高斯消元。奇异则返回 None。"""
    n = len(rhs)
    m = [row[:] + [rhs[i]] for i, row in enumerate(mat)]
    for col in range(n):
        piv = next((r for r in range(col, n) if m[r][col] % mod), None)
        if piv is None:
            return None
        m[col], m[piv] = m[piv], m[col]
        inv = pow(m[col][col], -1, mod)
        m[col] = [v * inv % mod for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] % mod:
                f = m[r][col]
                m[r] = [(m[r][k] - f * m[col][k]) % mod for k in range(n + 1)]
    return [m[i][n] for i in range(n)]


def min_order_by_linear_algebra(seq: list[int], mod: int = MOD,
                                max_order: int = 8) -> tuple[int, list[int]] | None:
    """第二套实现: 用线性代数求最小阶 —— 和 BM 完全不同的路。

    对每个候选阶 L, 取前 2L 项立方程组  sum_j c_j * s_{n-j} = s_n  (n=L..2L-1),
    高斯消元解出 c, 再拿**全部** seq 逐项验证。两套都通过才算这个 L 成立。
    求最小可行 L —— 这就是"最小阶"的定义, 不依赖 BM 的任何性质。
    """
    for L in range(1, max_order + 1):
        mat = [[seq[n - 1 - j] % mod for j in range(L)] for n in range(L, 2 * L)]
        vec = [seq[n] % mod for n in range(L, 2 * L)]
        c = gauss_solve_mod(mat, vec, mod)
        if c is None:
            continue
        if all(sum(c[j] * seq[n - 1 - j] for j in range(L)) % mod == seq[n] % mod
               for n in range(L, len(seq))):
            return L, c
    return None


def build_res_005() -> None:
    seq = gen_sequence(TRUE_COEF, TRUE_SEEDS, CHECK_TERMS)
    assert len(seq) == CHECK_TERMS
    assert seq[:len(TRUE_SEEDS)] == TRUE_SEEDS

    # 两套独立实现: BM 与 高斯消元
    order_bm, coef_bm = bm_berlekamp_massey(seq[:GIVEN_TERMS])
    la = min_order_by_linear_algebra(seq)
    assert la is not None, "高斯消元未找到任何可行阶数"
    order_la, coef_la = la

    assert order_bm == order_la, f"BM 阶数 {order_bm} 与线性代数阶数 {order_la} 不一致"
    assert order_bm == len(TRUE_COEF), f"阶数 {order_bm} 与构造用系数个数不符"
    assert coef_bm == TRUE_COEF, f"BM 系数 {coef_bm} != 构造系数 {TRUE_COEF}"
    assert coef_la == TRUE_COEF, f"线性代数系数 {coef_la} != 构造系数 {TRUE_COEF}"

    # 第三重: 用求出的系数把 200 项全部重算一遍
    assert gen_sequence(coef_bm, seq, CHECK_TERMS) == seq, "最小递推无法复现 200 项"

    # 非最小性反例: (1-x)C(x) 也能复现全部 200 项, 但阶数是 6 —— 证明"最小"这条判据
    # 不是摆设(与 verifier 的反例 A 同构)。
    # C(x) = 1 - 3x - 7x^2 - 11x^3 - 17x^4 - 23x^5
    # (1-x)C(x) = 1 - 4x - 4x^2 - 4x^3 - 6x^4 - 6x^5 + 23x^6
    # 按 s_n = c1 s_{n-1} + ... + ck s_{n-k} 的约定读回来即 [4,4,4,6,6,-23]。
    nonmin = [4, 4, 4, 6, 6, (-23) % MOD]
    assert gen_sequence(nonmin, seq, CHECK_TERMS) == seq, "前提不成立: 非最小反例并不能复现序列"
    assert len(nonmin) == order_bm + 1, "反例阶数应恰为最小阶 + 1"

    terms = ", ".join(str(v) for v in seq)
    lines = []
    for i in range(0, len(seq), 10):
        lines.append("    " + ", ".join(str(v) for v in seq[i:i + 10]))
    body = "\n".join(lines)

    txt = f"""最小线性递推（res-005）
========================

给定一个整数序列 s_0, s_1, ..., s_{CHECK_TERMS - 1}, 共 {CHECK_TERMS} 项（下标从 0 开始）。

已知该序列由某个**常系数线性递推**生成：存在整数 L >= 1 与系数 c_1..c_L，
使得对一切 n >= L 有

        s_n = c_1*s_{{n-1}} + c_2*s_{{n-2}} + ... + c_L*s_{{n-L}}      (mod {MOD})

其中 L 与 c_1..c_L **均未知**，需要你自己求出。

★ 下标约定（务必看清，否则结果对不上）★
  · s_0 是**第一项**。
  · 题面所说"前 {GIVEN_TERMS} 项" = s_0 .. s_{GIVEN_TERMS - 1}，求递推**只用**这一段。
  · 题面所说"前 {CHECK_TERMS} 项" = s_0 .. s_{CHECK_TERMS - 1}，求完之后用它**验证**。
  · 所有项都取在 [0, {MOD}) 的标准剩余系里；所有运算在模 {MOD} 意义下进行。

任务：
  1. 求**最小**阶数 L 与系数 c_1..c_L（模 {MOD}）；
  2. 用前 {GIVEN_TERMS} 项之外的信息验证：把你求出的递推作用在序列上，
     重新生成前 {CHECK_TERMS} 项，必须与上面给出的 {CHECK_TERMS} 项逐项相同；
  3. 说明你的阶数确实是**最小**的（注意：能复现 {CHECK_TERMS} 项的非最小递推
     一定存在，只报一个"能复现"的递推是不够的）。

序列（共 {CHECK_TERMS} 项，逗号分隔，每行 10 项）：

{body}
"""
    _write(os.path.join(OUT, "res_05_seq.txt"), txt)
    _mirror("res_05_seq.txt", "res_05_sequence.txt")

    _write_json(os.path.join(OUT, "res_05_reference.json"), {
        "problem": "res-005-berlekamp-massey",
        "note": "数据必须与 res_05_berlekamp.py 的冻结实例一致, 否则 agent 与 verifier "
                "解的不是同一个问题",
        "mod": MOD,
        "seeds": TRUE_SEEDS,
        "recurrence": TRUE_COEF,
        "recurrence_meaning": "s_n = 3 s_{n-1} + 7 s_{n-2} + 11 s_{n-3} + 17 s_{n-4} + 23 s_{n-5}",
        "order": len(TRUE_COEF),
        "given_terms": GIVEN_TERMS,
        "check_terms": CHECK_TERMS,
        "sequence_first_12": seq[:12],
        "sequence_last_5": seq[-5:],
        "cross_check_berlekamp_massey": {"order": order_bm, "coeffs": coef_bm},
        "cross_check_linear_algebra": {"order": order_la, "coeffs": coef_la},
        "both_implementations_agree": order_bm == order_la and coef_bm == coef_la,
        "minimality_trap_example": {
            "coeffs": nonmin,
            "order": len(nonmin),
            "reproduces_all_terms": True,
            "why": "(1-x)C(x) 与 C(x) 生成同一条序列, 所以'能复现 200 项'不能证明最小性",
        },
    })
    print(f"res-005 最小阶 = {order_bm} | 系数 = {coef_bm} (mod {MOD})")
    print(f"         BM 与高斯消元一致: {order_bm == order_la and coef_bm == coef_la} | "
          f"非最小反例阶 {len(nonmin)} 也能复现 200 项 -> 最小性判据是必要的")


# ======================================================================
# res-007: OEIS 条目识别
# ======================================================================
# 说明: 必须用 Fibonacci。res_07_oeis.py 的自测写死了 init=[0,1], coeffs=[1,1]
# 必须判为正例, 换成别的序列那条 assert 必挂。OEIS 编号 A000045 已联网核对
# (https://oeis.org/A000045/b000145.txt 返回的正是 0,1,1,2,3,5,8,...), 不是凭记忆。
OEIS_ID = "A000045"
GIVEN_7 = 30
CHECK_7 = 100
# 写进 txt 用的中文数字: txt 里不能出现 ASCII 数字(见 build_res_007 里的说明)
GIVEN_7_CN = "三十"
CHECK_7_CN = "一百"


def pell_like_recurrence_7(count: int) -> list[int]:
    """实现 A: 常系数线性递推 a(n) = a(n-1) + a(n-2), a(0)=0, a(1)=1。"""
    a = [0, 1]
    while len(a) < count:
        a.append(a[-1] + a[-2])
    return a[:count]


def closed_form_7(count: int) -> list[int]:
    """实现 B: Fibonacci 的 Binet 闭式 (phi^n - psi^n)/sqrt(5), mpmath 高精度。

    这是与递推**完全不同**的一条路: 不做加法递推, 直接用无理数幂。
    两者在 100 项上逐项相同, 才能确认闭式没写错常数。

    ★ 这里的 sqrt(5)/sqrt(2) 一开始写错过 ★ 首版误抄了 Pell 的闭式
    ((1+sqrt2)^n-(1-sqrt2)^n)/(2*sqrt2), 与 Fibonacci 递推对不上, 被
    "rec == cf" 这条 assert 当场抓住。闭式题最容易被这种常数错误骗过。
    """
    import mpmath as mp

    mp.mp.dps = 60
    s5 = mp.sqrt(5)
    phi, psi = (1 + s5) / 2, (1 - s5) / 2
    out = []
    for n in range(count):
        v = (phi ** n - psi ** n) / s5
        out.append(int(mp.nint(v)))
    return out


def build_res_007() -> None:
    rec = pell_like_recurrence_7(CHECK_7)
    cf = closed_form_7(CHECK_7)
    assert rec == cf, "递推与 Binet 闭式在 100 项上不一致"

    # 第三重: 与 OEIS b 文件抓到的真实取值抽查(联网核对 A000045 时读到的前若干项)
    oeis_head = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
                 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368]
    assert rec[:len(oeis_head)] == oeis_head, "与 OEIS A000045 的真实取值不符"

    given = rec[:GIVEN_7]

    # ★ txt 里不能出现任何 ASCII 数字(除序列本身) ★
    # res_07_oeis.py 的 _parse_seq_text 用 re.split(r"[^0-9-]+") 把整个文件里
    # 所有整数抠出来当序列; 标题里写 "res-007" 会被解析成 -7 混进第 0 项。
    # 所以下面的标题一律用中文数字, 且不写题号。
    # ★ txt 里除序列本身外绝不能出现任何 ASCII 数字 ★
    # res_07_oeis.py 的 _parse_seq_text 用 re.split(r"[^0-9-]+") 把整个文件里
    # 所有整数抠出来当序列; 标题里写 "前 30 项" 会被解析成一个额外的 30 混进
    # 序列头部, 于是第零项就错了。所以下面的标题一律用中文数字, 且不写题号。
    # (首版就踩了这个坑: 标题里的 "前30项" 让解析器多读出一个 30, 被
    #  "parsed == given" 这条 assert 当场抓住。)
    lines = []
    for i in range(0, len(given), 10):
        lines.append("    " + ", ".join(str(v) for v in given[i:i + 10]))
    body = "\n".join(lines)

    txt = f"""整数序列识别（序列条目识别与公式验证）
======================================

下面给出某个整数序列的前{GIVEN_7_CN}项，逗号分隔，按顺序排列：

{body}

任务：
  一、识别该序列对应的 OEIS 条目编号（形如字母 A 加六位数字）；
  二、给出该序列的一个闭式，或一个常系数线性递推公式（并写明初值）；
  三、用你给出的公式复算前{CHECK_7_CN}项，证明与上面给出的各项完全一致。

说明：
  · 本文件只提供输入，不含任何答案、名称或出处。
  · 序列下标从零开始，第一项即下标零。
  · 校验时请使用**精确整数**比较，不要用浮点容差：该序列增长很快，
    超过双精度尾数后微小误差会被放大。
"""
    _write(os.path.join(OUT, "res_07_seq.txt"), txt)
    _mirror("res_07_seq.txt", "res_07_sequence.txt")

    # 自检: 确认 txt 里除序列本身外没有任何 ASCII 数字会被误解析
    import res_07_oeis as v7

    parsed = v7._parse_seq_text(txt)
    assert parsed == given, f"txt 被 verifier 的解析器读成了 {parsed[:6]}..., 前几项就不对"

    _write_json(os.path.join(OUT, "res_07_reference.json"), {
        "problem": "res-007-oeis-identify-prove",
        "note": "必须是 Fibonacci: res_07_oeis.py 自测写死 init=[0,1] coeffs=[1,1] 为正例",
        "oeis_id": OEIS_ID,
        "oeis_id_source": "联网核对 https://oeis.org/A000045/b000045.txt 的前若干项",
        "name": "Fibonacci numbers",
        "formula_recurrence": "a(n) = a(n-1) + a(n-2), a(0)=0, a(1)=1",
        "formula_closed_form": "a(n) = (phi^n - psi^n)/sqrt(5), "
                               "phi=(1+sqrt5)/2, psi=(1-sqrt5)/2",
        "init": [0, 1],
        "coeffs": [1, 1],
        "given_terms": GIVEN_7,
        "check_terms": CHECK_7,
        "terms": given,
        "first_100": rec,
        "cross_check_recurrence_vs_binet_100_terms": True,
        "cross_check_against_oeis_b_file_head": True,
    })
    print(f"res-007 OEIS = {OEIS_ID} | 前 {GIVEN_7} 项 = {given[:8]}...")
    print(f"         递推与 Binet 闭式在 {CHECK_7} 项上逐项一致: True | "
          f"txt 经 verifier 解析器回读 == 序列: True")


# ======================================================================
# res-011: 对角数独 + 唯一性证明
# ======================================================================
# ★ 盘面不手抄, 直接取自 verifier 冻结的常量 ★
# 首版把 81 个数字手抄进本文件, 结果第 8 行的 2 和第 9 行开头的 0 抄反了
# (下标 68/69), z3 立刻判成 UNSAT —— 一道"无解"的题会让 agent 怎么答都错。
# 手抄 81 个数字本就不该做: 正确做法是以 verifier 的冻结常量为唯一真源,
# 再用下面 _rebuild_puzzle_from_construction() 独立重建一次并比对, 两边一致才用。
def _frozen_puzzle():
    """从 res_11_sudoku_diagonal.py 读取冻结盘面(只读)。"""
    import res_11_sudoku_diagonal as v11

    return tuple(v11.PUZZLE)


class _LCG:
    """verifier 造题用的同款 LCG(确定性, 跨 Python 版本稳定)。"""

    def __init__(self, seed: int) -> None:
        self.s = seed & 0x7FFFFFFF

    def next(self) -> int:
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s

    def shuffle(self, seq: list) -> None:
        for i in range(len(seq) - 1, 0, -1):
            j = self.next() % (i + 1)
            seq[i], seq[j] = seq[j], seq[i]


def _rebuild_puzzle_from_construction(seed: int = 20260908):
    """★ 第二套独立实现 ★ 按题面 docstring 描述的造法把盘面重建一遍。

    造法: LCG 打乱格子顺序 -> MRV 随机回溯填一个对角数独满盘 -> 贪心删线索
    (对角线格优先, 只要"带对角线约束下仍恰好一个解"就删)。
    这条路**不读** verifier 的 PUZZLE 常量, 只按算法重建; 与常量一致才说明
    那 19 个线索确实是按造法生成的, 而不是随手敲的数字。
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
        if r == c:
            used |= {grid[9 * j + j] for j in range(9)}
        if r + c == 8:
            used |= {grid[9 * j + (8 - j)] for j in range(9)}
        used.discard(0)
        return [v for v in ALL9 if v not in used]

    def fill(left: list[int]) -> bool:
        if not left:
            return True
        pick, pick_cands, best = -1, [], 99
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

    def count_solutions(clues, limit: int = 2) -> int:
        return len(bruteforce_sudoku(clues, True, limit))

    diag = [i for i in range(81) if i // 9 == i % 9 or i // 9 + i % 9 == 8]
    order = diag + [i for i in range(81) if i not in set(diag)]
    for i in order:
        if grid[i] == 0:
            continue
        keep = grid[i]
        grid[i] = 0
        if count_solutions(grid) != 1:
            grid[i] = keep
    return tuple(grid)


PUZZLE = _frozen_puzzle()
ALL9 = tuple(range(1, 10))


def z3_sudoku_solve(clues, use_diagonal: bool = True):
    """实现 A: z3。返回 (sat?, 解)。"""
    from z3 import And, Distinct, Int, Or, Solver, sat

    x = [[Int(f"x_{r}_{c}") for c in range(9)] for r in range(9)]
    s = Solver()
    for r in range(9):
        for c in range(9):
            s.add(And(x[r][c] >= 1, x[r][c] <= 9))
    for r in range(9):
        s.add(Distinct([x[r][c] for c in range(9)]))
    for c in range(9):
        s.add(Distinct([x[r][c] for r in range(9)]))
    for br in range(3):
        for bc in range(3):
            s.add(Distinct([x[3 * br + dr][3 * bc + dc] for dr in range(3) for dc in range(3)]))
    if use_diagonal:
        s.add(Distinct([x[i][i] for i in range(9)]))
        s.add(Distinct([x[i][8 - i] for i in range(9)]))
    for i, v in enumerate(clues):
        if v:
            s.add(x[i // 9][i % 9] == v)
    if s.check() != sat:
        return False, None
    m = s.model()
    # z3 5.x 的 evaluate 返回 IntNumRef 而不是 python int, 必须走 .as_long()
    return True, tuple(m.evaluate(x[r][c]).as_long()
                       for r in range(9) for c in range(9))


def z3_prove_unique(clues, solution) -> bool:
    """唯一性证书: 加入"至少有一格与该解不同"后必须 UNSAT。"""
    from z3 import And, Distinct, Int, Or, Solver, unsat

    x = [[Int(f"u_{r}_{c}") for c in range(9)] for r in range(9)]
    s = Solver()
    for r in range(9):
        for c in range(9):
            s.add(And(x[r][c] >= 1, x[r][c] <= 9))
    for r in range(9):
        s.add(Distinct([x[r][c] for c in range(9)]))
    for c in range(9):
        s.add(Distinct([x[r][c] for r in range(9)]))
    for br in range(3):
        for bc in range(3):
            s.add(Distinct([x[3 * br + dr][3 * bc + dc] for dr in range(3) for dc in range(3)]))
    s.add(Distinct([x[i][i] for i in range(9)]))
    s.add(Distinct([x[i][8 - i] for i in range(9)]))
    for i, v in enumerate(clues):
        if v:
            s.add(x[i // 9][i % 9] == v)
    # 排除这个解: 至少一格不同
    s.add(Or([x[i // 9][i % 9] != solution[i] for i in range(81)]))
    return s.check() == unsat


def bruteforce_sudoku(clues, use_diagonal: bool = True, limit: int = 2) -> list[tuple]:
    """实现 B: 自写回溯枚举(位掩码 + MRV), 不依赖 z3。"""
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
        row[r] |= bit
        col[c] |= bit
        box[b] |= bit
        if use_diagonal and r == c:
            d1 |= bit
        if use_diagonal and r + c == 8:
            d2 |= bit
    empty = [i for i, v in enumerate(grid) if not v]
    sols: list[tuple] = []

    def used_of(i: int) -> int:
        r, c = divmod(i, 9)
        b = r // 3 * 3 + c // 3
        u = row[r] | col[c] | box[b]
        if use_diagonal and r == c:
            u |= d1
        if use_diagonal and r + c == 8:
            u |= d2
        return u

    def rec(todo: list[int]) -> None:
        nonlocal d1, d2
        if len(sols) >= limit:
            return
        if not todo:
            sols.append(tuple(grid))
            return
        best, best_c = -1, ()
        for i in todo:
            cs = [v for v in ALL9 if not (used_of(i) >> (v - 1)) & 1]
            if not cs:
                return
            if best < 0 or len(cs) < len(best_c):
                best, best_c = i, cs
                if len(cs) == 1:
                    break
        r, c = divmod(best, 9)
        b = r // 3 * 3 + c // 3
        rest = [i for i in todo if i != best]
        for v in best_c:
            bit = 1 << (v - 1)
            grid[best] = v
            row[r] |= bit
            col[c] |= bit
            box[b] |= bit
            on1 = use_diagonal and r == c
            on2 = use_diagonal and r + c == 8
            if on1:
                d1 |= bit
            if on2:
                d2 |= bit
            rec(rest)
            grid[best] = 0
            row[r] ^= bit
            col[c] ^= bit
            box[b] ^= bit
            if on1:
                d1 ^= bit
            if on2:
                d2 ^= bit
            if len(sols) >= limit:
                return

    rec(empty)
    return sols


def build_res_011() -> None:
    assert len(PUZZLE) == 81
    clues = sum(1 for v in PUZZLE if v)
    assert clues == 19, f"线索数 {clues} != verifier 冻结实例的 19"

    # --- 第三重: 按造法独立重建盘面, 必须与冻结常量逐位一致 ---
    rebuilt = _rebuild_puzzle_from_construction()
    assert rebuilt == PUZZLE, ("按造法重建的盘面与 verifier 冻结盘面不一致, "
                              f"首个不同下标 {[i for i in range(81) if rebuilt[i] != PUZZLE[i]][:5]}")

    # --- 两套独立实现 ---
    sat_ok, z3_sol = z3_sudoku_solve(PUZZLE, True)
    assert sat_ok, "z3 在带对角线约束下判为不可满足 —— 题面本身坏了"
    bf = bruteforce_sudoku(PUZZLE, True, 2)
    assert len(bf) == 1, f"回溯枚举得到 {len(bf)} 个解, 应恰好 1 个"
    assert tuple(bf[0]) == z3_sol, "z3 与回溯枚举给出的解不一致"

    solution = z3_sol
    # --- 唯一性证书(题目明确要求"证明解唯一") ---
    assert z3_prove_unique(PUZZLE, solution), "第二解查询没有返回 UNSAT —— 唯一性证书不成立"

    # --- 对角线必须是承重墙: 去掉对角线约束就不唯一 ---
    n_nodiag = len(bruteforce_sudoku(PUZZLE, False, 2))
    assert n_nodiag >= 2, "去掉对角线约束后仍然唯一, 说明对角线没起作用"

    # --- 解本身合法(行/列/宫/双对角线各为 1-9) ---
    g = [solution[i * 9:i * 9 + 9] for i in range(9)]
    assert all(sorted(r) == list(ALL9) for r in g)
    assert all(sorted(g[r][c] for r in range(9)) == list(ALL9) for c in range(9))
    assert sorted(g[i][i] for i in range(9)) == list(ALL9)
    assert sorted(g[i][8 - i] for i in range(9)) == list(ALL9)

    # 交叉核对 verifier 自己的枚举(只读)
    import res_11_sudoku_diagonal as v11

    v_sol = v11._solve_clues(PUZZLE, True, 1)[0]
    assert tuple(v_sol) == solution, "与 verifier 内置枚举给出的唯一解不一致"

    grid_rows = []
    for r in range(9):
        grid_rows.append("    " + "".join(str(v) if v else "." for v in g[r] if True)[:9]
                         if False else
                         "    " + "".join(str(solution[r * 9 + c]) if solution[r * 9 + c] else "."
                                        for c in range(9)))

    puzzle_rows = ["    " + "".join(str(PUZZLE[r * 9 + c]) if PUZZLE[r * 9 + c] else "."
                                   for c in range(9)) for r in range(9)]
    puzzle_body = "\n".join(puzzle_rows)

    z3_script = '''from z3 import And, Distinct, Int, Or, Solver, sat, unsat

GRID = [
%s
]

def build(with_diagonal):
    x = [[Int(f"x_{r}_{c}") for c in range(9)] for r in range(9)]
    s = Solver()
    for r in range(9):
        for c in range(9):
            s.add(And(x[r][c] >= 1, x[r][c] <= 9))
    for r in range(9):
        s.add(Distinct([x[r][c] for c in range(9)]))
    for c in range(9):
        s.add(Distinct([x[r][c] for r in range(9)]))
    for br in range(3):
        for bc in range(3):
            s.add(Distinct([x[3*br+dr][3*bc+dc] for dr in range(3) for dc in range(3)]))
    if with_diagonal:
        s.add(Distinct([x[i][i] for i in range(9)]))
        s.add(Distinct([x[i][8-i] for i in range(9)]))
    for r in range(9):
        for c in range(9):
            v = int(GRID[r][c])
            if v:
                s.add(x[r][c] == v)
    return s, x

# 第一步: 求解
s, x = build(True)
assert s.check() == sat, "带对角线约束下竟无解"
m = s.model()
sol = [[m.evaluate(x[r][c]).as_long() for c in range(9)] for r in range(9)]
print("一个解:")
for row in sol:
    print(" ".join(str(v) for v in row))

# 第二步: 唯一性 —— 加一条"至少有一格与上面这个解不同", 必须 UNSAT
s2, x2 = build(True)
s2.add(Or([x2[r][c] != sol[r][c] for r in range(9) for c in range(9)]))
res = s2.check()
print("第二解查询结果:", res)
assert res == unsat, "存在第二个解, 唯一性不成立"
print("唯一性成立: 带对角线约束下恰好一个解")
''' % (",\n".join('    "%s"' % row.strip() for row in puzzle_rows))

    txt = f"""对角数独：求解并证明唯一（res-011）
================================

规则：在 9x9 网格中填入 1..9，满足
  · 每一行、每一列、每一个 3x3 宫，数字 1..9 各出现恰好一次；
  · **两条主对角线**（左上到右下、右上到左下）也各是 1..9 的一个排列。
  · `.` 表示空格。

盘面（. 为空，共 {clues} 个线索）：

{puzzle_body}

任务：
  一、用 Z3 求出该盘面的解；
  二、**证明**解唯一 —— 要求给出机器可读的 UNSAT 证据，即：
      在"存在解"的条件上再加一条"至少有一格与你给出的解不同"，
      该查询必须返回 unsat。只声称"唯一"而不给证据不算完成。

下面的脚本可直接运行，演示标准的 Z3 唯一性证法（自己动手实现一遍，不要只贴结论）：

```python
{z3_script}```
"""
    _write(os.path.join(OUT, "res_11_sudoku.txt"), txt)
    _mirror("res_11_sudoku.txt", "res_11_sudoku.txt")

    _write_json(os.path.join(OUT, "res_11_reference.json"), {
        "problem": "res-011-sudoku-diagonal-z3",
        "note": "必须与 res_11_sudoku_diagonal.py 的 PUZZLE 一致(19 线索)",
        "puzzle": list(PUZZLE),
        "clues": clues,
        "solution": [list(solution[i * 9:i * 9 + 9]) for i in range(9)],
        "unique": True,
        "uniqueness_certificate": {
            "method": "z3",
            "query": "puzzle_constraints AND Or(any cell differs from the found solution)",
            "result": "unsat",
        },
        "solutions_without_diagonal_constraints": n_nodiag,
        "diagonal_is_load_bearing": n_nodiag >= 2,
        "cross_check_z3_vs_bruteforce": True,
        "cross_check_matches_verifier_enumeration": True,
    })
    print(f"res-011 线索 = {clues} | 唯一性: z3 第二解查询 UNSAT 且回溯枚举恰好 1 解")
    print(f"         去掉对角线约束后解数 >= {n_nodiag} -> 对角线是承重墙: True")


# ======================================================================
# res-014: 三阶魔方 6 步最短解
# ======================================================================
# 与 res_14_rubiks.py 的 SCRAMBLE 一致。坐标: x 右 / y 上 / z 前,
# U=+y D=-y F=+z B=-z R=+x L=-x。"从该面外侧看顺时针" = 绕外法向 -90°。
SCRAMBLE = ("D'", "B'", "R'", "D'", "R", "D'")
MAX_MOVES = 14
FACE_NORMALS = {"U": (0, 1, 0), "D": (0, -1, 0), "F": (0, 0, 1),
                "B": (0, 0, -1), "R": (1, 0, 0), "L": (-1, 0, 0)}


def build_stickers():
    out = []
    for face in ("U", "D", "F", "B", "R", "L"):
        nrm = FACE_NORMALS[face]
        ax = max(range(3), key=lambda i: abs(nrm[i]))
        sign = 1 if nrm[ax] > 0 else -1
        others = [i for i in range(3) if i != ax]
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                pos = [0, 0, 0]
                pos[ax] = sign
                pos[others[0]] = i
                pos[others[1]] = j
                out.append((tuple(pos), nrm))
    return tuple(out)


STICKERS = build_stickers()
SOLVED = tuple(range(54))

# 旋转矩阵: 右手法则绕正轴 +90°。用矩阵而不是逐分量 if/else, 与 verifier 的写法独立。
_ROT = {
    0: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),   # x: y->z
    1: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),   # y: z->x
    2: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),   # z: x->y
}


def rotk(v, axis: int, k: int) -> tuple:
    a = np.array(v, dtype=int)
    for _ in range(k % 4):
        a = _ROT[axis] @ a
    return tuple(int(t) for t in a)


def turn_perm(face: str, k: int) -> tuple:
    nrm = FACE_NORMALS[face]
    ax = max(range(3), key=lambda i: abs(nrm[i]))
    sign = 1 if nrm[ax] > 0 else -1
    steps = (-sign * k) % 4           # 外侧看顺时针 = 绕外法向 -90°
    idx = {st: i for i, st in enumerate(STICKERS)}
    perm = []
    for pos, n in STICKERS:
        if pos[ax] == sign:
            perm.append(idx[(rotk(pos, ax, steps), rotk(n, ax, steps))])
        else:
            perm.append(idx[(pos, n)])
    return tuple(perm)


def k_of(token: str) -> int:
    return {"": 1, "'": 3, "2": 2}[token[1:]]


MOVES = [f + s for f in ("U", "D", "L", "R", "F", "B") for s in ("", "'", "2")]
MOVE_PERM = {m: turn_perm(m[0], k_of(m)) for m in MOVES}


def apply_key(key: tuple, moves) -> tuple:
    for m in moves:
        key = tuple(key[p] for p in MOVE_PERM[m])
    return key


def inverse(m: str) -> str:
    if m.endswith("'"):
        return m[0]
    if m.endswith("2"):
        return m[0] + "2"
    return m[0] + "'"


def bfs_distance(start: tuple, goal: tuple, max_depth: int = 6) -> int:
    """双向 BFS 求真实最短距离 —— 这是"步数最少"的机器证据, 不采信 agent 自报。"""
    if start == goal:
        return 0
    sa, sb = {start: 0}, {goal: 0}
    fa, fb = [start], [goal]
    for depth in range(1, max_depth + 1):
        nfa = []
        for key in fa:
            for m in MOVES:
                nk = tuple(key[p] for p in MOVE_PERM[m])
                if nk in sa:
                    continue
                sa[nk] = depth
                if nk in sb:
                    return depth + sb[nk]
                nfa.append(nk)
        fa = nfa
        nfb = []
        for key in fb:
            for m in MOVES:
                nk = tuple(key[p] for p in MOVE_PERM[inverse(m)])
                if nk in sb:
                    continue
                sb[nk] = depth
                if nk in sa:
                    return depth + sa[nk]
                nfb.append(nk)
        fb = nfb
    raise RuntimeError("未在 max_depth 内相遇")


def build_res_014() -> None:
    assert len(STICKERS) == 54 and len(set(STICKERS)) == 54
    assert len(SCRAMBLE) == 6
    # 相邻两面必须不同: 否则 R R 要并成 R2, R R' 直接抵消, 步数就不是 6 了
    assert all(SCRAMBLE[i][0] != SCRAMBLE[i + 1][0] for i in range(5)), "打乱序列存在相邻同面"

    # --- 模型自检: 用魔方群的代数恒等式钉死记号方向(不依赖 verifier) ---
    fails = []
    for m in MOVES:
        if apply_key(SOLVED, [m] * 4) != SOLVED:
            fails.append(f"{m} 四阶不成立")
    for a, b in (("R", "L"), ("U", "D"), ("F", "B")):
        for suf in ("", "'", "2"):
            if apply_key(SOLVED, [a + suf, b + suf]) != apply_key(SOLVED, [b + suf, a + suf]):
                fails.append(f"{a+suf} 与 {b+suf} 不可交换")
    if apply_key(SOLVED, ["R", "U", "R'", "U'"] * 6) != SOLVED:
        fails.append("(R U R' U')^6 != 复原态")
    if apply_key(SOLVED, ["R", "U"] * 105) != SOLVED:
        fails.append("(R U)^105 != 复原态")
    assert not fails, "魔方模型与通行记号不一致: " + "; ".join(fails)

    scrambled = apply_key(SOLVED, SCRAMBLE)
    assert scrambled != SOLVED

    # --- 交叉核对: 与 verifier 的冻结状态逐位一致 ---
    import res_14_rubiks as v14

    assert scrambled == v14._apply_key(v14.SOLVED, v14.SCRAMBLE), "与 verifier 的打乱状态不一致"
    assert tuple(SCRAMBLE) == tuple(v14.SCRAMBLE), "与 verifier 的打乱序列不一致"

    solution = [inverse(m) for m in reversed(SCRAMBLE)]
    assert len(solution) == MAX_MOVES - 8
    assert apply_key(scrambled, solution) == SOLVED, "逆序列没能还原魔方"

    # --- 最短性: 自写双向 BFS ---
    opt = bfs_distance(scrambled, SOLVED)
    assert opt == 6, f"双向 BFS 最短距离 = {opt}, 与题面'6 步'前提不符"
    assert opt == v14._optimal_distance(), "与 verifier 的 BFS 结果不一致"

    # --- 渲染 54 个色块 ---
    # 面序 U D F B R L, 与 verifier 的贴纸顺序一致; 每个面 3x3, 行主序。
    color_of_face = {"U": "W", "D": "Y", "F": "G", "B": "B", "R": "R", "L": "O"}
    face_names = ["U", "D", "F", "B", "R", "L"]
    net_lines = []
    for fi, face in enumerate(face_names):
        rows = []
        for r in range(3):
            rows.append(" ".join(color_of_face[face_names[scrambled[fi * 9 + r * 3 + c] // 9]]
                                 for c in range(3)))
        net_lines.append((face, rows))

    state_txt = []
    for face, rows in net_lines:
        state_txt.append(f"    {face}:  " + "   ".join(rows))

    # 几何定义表: 消除"第 i 行第 j 列到底是哪个方向"这一类歧义
    geom = []
    for face in face_names:
        nrm = FACE_NORMALS[face]
        ax = max(range(3), key=lambda i: abs(nrm[i]))
        others = [i for i in range(3) if i != ax]
        letters = "xyz"
        geom.append(f"    {face}: 法向 ({nrm[0]},{nrm[1]},{nrm[2]})；"
                    f"行索引 -> {letters[others[0]]} ∈ {{-1,0,1}}，"
                    f"列索引 -> {letters[others[1]]} ∈ {{-1,0,1}}")

    txt = f"""三阶魔方打乱状态：求最短解（res-014）
====================================

下面是某个已打乱的三阶魔方，给出 6 个面共 54 个色块。

记号：U=上 D=下 F=前 B=后 R=右 L=左；
      无后缀 = 从该面**外侧看顺时针**转 90°，' = 逆时针 90°，2 = 180°。

状态（面序 U D F B R L，每个面 3 行 3 列，行主序）：

{chr(10).join(state_txt)}

★ 行/列方向（x 右 / y 上 / z 前），避免"同一状态多种画法"的歧义：
{chr(10).join(geom)}

任务：
  用 IDA* 或双向 BFS 求出把该状态还原的**最短**动作序列，并证明步数最少。
  · 提交的动作序列必须能把上面的状态真正还原；
  · 步数不得超过 6；
  · "最少性"需要机器可验证的证据（例如你自己跑双向 BFS 得到的最短距离），
    只在文字里声称"这是最短解"不算。

参考事实：该状态由 6 个动作打乱而成（可解），且经双向 BFS 确认最短距离为 6。
"""
    _write(os.path.join(OUT, "res_14_cube.txt"), txt)
    _mirror("res_14_cube.txt", "res_14_rubiks.txt")

    _write_json(os.path.join(OUT, "res_14_reference.json"), {
        "problem": "res-014-rubiks-6-moves",
        "note": "必须与 res_14_rubiks.py 的 SCRAMBLE 一致, 否则解的不是同一个状态",
        "scramble": list(SCRAMBLE),
        "scramble_seed": 20260908,
        "solution": solution,
        "length": len(solution),
        "bfs_optimal_distance": opt,
        "solution_is_optimal": len(solution) == opt,
        "solved_state_ids": list(SOLVED),
        "scrambled_state_ids": list(scrambled),
        "state_facelets_by_face": {face: rows for face, rows in net_lines},
        "model": "几何贴纸模型, 54 贴纸 = (位置向量, 法向); "
                 "从面外侧看顺时针 = 绕外法向 -90°",
        "cross_check_group_identities": ["每个动作 4 阶", "R与L / U与D / F与B 可交换",
                                         "(R U R' U')^6 = 复原态", "(R U)^105 = 复原态"],
        "cross_check_matches_verifier_state": True,
    })
    print(f"res-014 打乱 = {' '.join(SCRAMBLE)} | 解 = {' '.join(solution)}")
    print(f"         双向 BFS 最短距离 = {opt} | 与 verifier 状态逐位一致: True")


# ======================================================================
# res-016: 500 城市 TSP
# ======================================================================
N_CITIES = 500
TSP_SEED = 20260908
TSP_SCALE = 10000.0
RATIO_TOL = 0.05


def nearest_neighbor(D: np.ndarray) -> list[int]:
    n = D.shape[0]
    unvisited = np.ones(n, dtype=bool)
    tour = [0]
    unvisited[0] = False
    cur = 0
    while len(tour) < n:
        j = int(np.argmin(np.where(unvisited, D[cur], np.inf)))
        tour.append(j)
        unvisited[j] = False
        cur = j
    return tour


def two_opt(D: np.ndarray, tour, max_rounds: int = 400) -> list[int]:
    """向量化的 best-improvement 2-opt(环形移动集)。"""
    ta = np.asarray(tour, dtype=np.int64).copy()
    n = ta.size
    for _ in range(max_rounds):
        improved = False
        for i in range(n - 1):
            a, b = ta[i], ta[i + 1]
            js = np.arange(i + 2, n)
            if js.size == 0:
                break
            tn, tn1 = ta[js], ta[(js + 1) % n]
            delta = D[a, tn] + D[b, tn1] - D[a, b] - D[tn, tn1]
            k = int(np.argmin(delta))
            if delta[k] < -1e-9:
                j = i + 2 + k
                ta[i + 1:j + 1] = ta[i + 1:j + 1][::-1]
                improved = True
        if not improved:
            break
    return ta.tolist()


def ortools_gls(D: np.ndarray, seconds: float = 5.0):
    """第二套独立实现: ortools GUIDED_LOCAL_SEARCH。"""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n = D.shape[0]
    scale = 10_000
    cost = np.rint(D * scale).astype(np.int64).tolist()
    best = None
    for st in (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
               routing_enums_pb2.FirstSolutionStrategy.SAVINGS):
        mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(mgr)
        idx = routing.RegisterTransitCallback(
            lambda i, j: cost[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
        routing.SetArcCostEvaluatorOfAllVehicles(idx)
        p = pywrapcp.DefaultRoutingSearchParameters()
        p.first_solution_strategy = st
        p.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        p.time_limit.FromMilliseconds(int(seconds * 1000))
        sol = routing.SolveWithParameters(p)
        if sol is not None:
            v = sol.ObjectiveValue() / scale
            if best is None or v < best:
                best = v
    return best


def tour_length(D: np.ndarray, tour) -> float:
    idx = np.asarray(tour, dtype=np.int64)
    return float(D[idx, np.roll(idx, -1)].sum())


def build_res_016() -> None:
    rng = np.random.default_rng(TSP_SEED)
    coords = rng.random((N_CITIES, 2)) * TSP_SCALE
    assert coords.shape == (N_CITIES, 2)

    # 交叉核对: txt 被 verifier 的解析器读回来必须与生成时**逐位一致**
    import res_16_tsp as v16

    header = (f"{N_CITIES} 城市对称 TSP 实例（res-016）\n"
              "===================================\n\n"
              "规模：共 500 座城市，编号从 0 到 499。\n"
              "距离：欧氏距离（平面直角距离），对称。\n"
              "数据格式：每行一座城市的两个坐标 \"x y\"，共 500 行，"
              "坐标为浮点数，已保留足够精度。\n"
              "（上面这些说明文字不是数据行，解析时跳过。）\n\n"
              "任务：找一条闭巡回，恰好访问每座城市一次，回到起点，"
              "使总长度不超过已知最优解的 1.05 倍。\n\n"
              "坐标（每行一座城市）:\n")
    body = "\n".join(f"{coords[i][0]:.10f} {coords[i][1]:.10f}" for i in range(N_CITIES))
    txt = header + body + "\n"

    parsed = v16._coords_from_text(txt)
    assert parsed is not None, "verifier 的解析器没能读出坐标"
    assert parsed.shape == (N_CITIES, 2), f"解析出 {parsed.shape}, 应为 (500, 2)"
    assert np.allclose(parsed, coords, rtol=0, atol=1e-9), "txt 往返后坐标不一致"
    # 写入的十位小数必须能被精确读回, 否则 verifier 算的 L_ref 与我们的对不上
    assert np.abs(parsed - coords).max() < 1e-9

    D = v16.distance_matrix(parsed)
    t2 = two_opt(D, nearest_neighbor(D))
    l2 = tour_length(D, t2)
    l_ort = ortools_gls(D)
    candidates = [("nearest_neighbor+2opt", round(l2, 4))]
    if l_ort is not None:
        candidates.append(("ortools_gls", round(l_ort, 4)))
    if l_ort is not None and l_ort <= l2:
        L_ref, source = l_ort, "ortools_gls"
    else:
        L_ref, source = l2, "nearest_neighbor+2opt"

    best_tour = t2
    assert sorted(best_tour) == list(range(N_CITIES)), "参考巡回不是 0..499 的排列"
    length = tour_length(D, best_tour)
    assert abs(length - l2) < 1e-6
    assert length <= (1.0 + RATIO_TOL) * L_ref, "参考解本身都过不了 1.05 倍门槛"

    _write(os.path.join(OUT, "res_16_tsp.txt"), txt)
    _mirror("res_16_tsp.txt", "res_16_tsp.txt")

    _write_json(os.path.join(OUT, "res_16_reference.json"), {
        "problem": "res-016-tsp-heuristic-500",
        "n_cities": N_CITIES,
        "seed": TSP_SEED,
        "coord_scale": TSP_SCALE,
        "distance": "euclidean",
        "ratio_tol": RATIO_TOL,
        "L_ref": L_ref,
        "L_ref_source": source,
        "candidates": candidates,
        "best_tour": best_tour,
        "best_tour_length": round(length, 6),
        "threshold_1p05": round((1.0 + RATIO_TOL) * L_ref, 6),
        "cross_check_roundtrip_coords": True,
        "cross_check_two_algorithms": [c[0] for c in candidates],
    })
    print(f"res-016 L_ref = {L_ref:.4f} (来源 {source}; 候选 {candidates})")
    print(f"         1.05 倍门槛 = {(1.0 + RATIO_TOL) * L_ref:.4f} | "
          f"txt 往返坐标逐位一致: True")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    build_res_005()
    build_res_007()
    build_res_011()
    build_res_014()
    build_res_016()
    print()
    print("全部 5 份数据已生成, 内部 assert 全部通过:")
    for f in ("res_05_seq.txt", "res_07_seq.txt", "res_11_sudoku.txt",
              "res_14_cube.txt", "res_16_tsp.txt",
              "res_05_sequence.txt", "res_07_sequence.txt", "res_14_rubiks.txt",
              "res_05_reference.json", "res_07_reference.json", "res_11_reference.json",
              "res_14_reference.json", "res_16_reference.json"):
        p = os.path.join(OUT, f)
        assert os.path.isfile(p), f"缺少 {f}"
        print(f"    {f:<26} {os.path.getsize(p):>8} bytes")


if __name__ == "__main__":
    main()
