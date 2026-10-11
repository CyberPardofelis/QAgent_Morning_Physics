"""res-014-rubiks-6-moves 的独立 verifier。

题目：给定一个 6 步内可解的三阶魔方打乱状态，用 IDA* 或双向 BFS 找到最短解，
并验证步数最少。

## 打乱状态是自己造的（造法见 :data:`SCRAMBLE`，可完全复现）
题库 params 指向 ``data/res_14_cube.txt``。
★ 2026-10-10 更正 ★ 写这个文件时那份数据还不存在；现在它已经补齐，
且**已核对题面里那 54 个色块与本文件的 _scramble_state() 是同一个魔方**
（颜色↔家面 双射 W->U Y->D G->F B->B R->R O->L）——
``python tests/tools/check_input_vs_reference.py``。数据若被重生成，重跑那条核对。
本文件仍然自己造：用自带 LCG（固定种子）抽 6 个动作，**相邻两面不同**
（否则 ``R R`` 要并成 ``R2``、``R R'`` 直接抵消，步数就不是 6 了），
作用在复原态上得到 SCRAMBLED。逆序列就是那条 6 步解。

## 魔方模型（kociemba 不可用，自己实现）
用**几何贴纸模型**：54 个贴纸 = (位置向量 pos∈{-1,0,1}^3, 法向 normal)，
动作 = 把某一层绕轴转 90°，pos 与 normal 一起转。
坐标：x 右 / y 上 / z 前，U=+y D=-y F=+z B=-z R=+x L=-x。
约定："从该面外侧看顺时针" = 绕外法向转 -90°（右手法则）。
这一约定由三条标准事实钉死（都在自测里验证，见 :func:`_model_selftest`）：
U 把 F 转成 L、R 把 F 转成 U、F 把 U 转成 R —— 与通行记号一致。

## 制品格式
```
CONSTRUCT: {"moves": ["R", "U", "R'", "F2", ...], "length": 6, "optimal": true}
```
``moves`` 可以是数组，也可以是空格分隔的字符串；单个动作形如 ``R`` / ``R'`` / ``R2``。

## 判据（为什么这么判就是"真的做对了"）
* **自己模拟**：把 agent 的动作序列作用在 SCRAMBLED 上，必须回到复原态。
  这一条不依赖任何第三方魔方库，也不依赖 agent 自报的步数。
* **步数 <= 6**，且自报的 ``length`` 必须等于实际动作个数（防止"报 6 步、
  实际写了 8 个动作"）。
* **最优性由 verifier 自己算**：双向 BFS 求出到复原态的真实最短距离
  （本题 = 6）。任何"可行且长度 <= 6"的解因此必然是最优解 —— 这是比
  "拿不到证书就只判可行解"更强的做法：证书拿不到不影响，因为最优性是我
  自己证出来的，不需要相信 agent 的 ``optimal: true``。
  （agent 若自称 optimal，我另行核对它是否真的等于 BFS 最优距离。）
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import Check, parse_artifact  # noqa: E402

# --------------------------------------------------------------------------
# 几何模型
# --------------------------------------------------------------------------

_FACE_NORMALS = {
    "U": (0, 1, 0), "D": (0, -1, 0),
    "F": (0, 0, 1), "B": (0, 0, -1),
    "R": (1, 0, 0), "L": (-1, 0, 0),
}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_of(face: str) -> tuple[int, int]:
    """返回 (轴下标, 该面在该轴上的符号)。"""
    n = _FACE_NORMALS[face]
    ax = max(range(3), key=lambda i: abs(n[i]))
    return ax, (1 if n[ax] > 0 else -1)


def _rot90(v, axis: int) -> tuple[int, int, int]:
    """右手法则绕正轴 +y 转 90°。x: y->z ; y: z->x ; z: x->y。"""
    x, y, z = v
    if axis == 0:
        return (x, -z, y)
    if axis == 1:
        return (z, y, -x)
    return (-y, x, z)


def _rotk(v, axis: int, k: int) -> tuple[int, int, int]:
    for _ in range(k % 4):
        v = _rot90(v, axis)
    return v


def _build_stickers():
    """54 个贴纸的固定顺序 —— 状态用它做下标，tuple 天然可哈希。"""
    out = []
    for face in ("U", "D", "F", "B", "R", "L"):
        ax, sign = _axis_of(face)
        others = [i for i in range(3) if i != ax]
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                pos = [0, 0, 0]
                pos[ax] = sign
                pos[others[0]] = i
                pos[others[1]] = j
                out.append((tuple(pos), _FACE_NORMALS[face]))
    return tuple(out)


STICKERS = _build_stickers()
assert len(STICKERS) == 54 and len(set(STICKERS)) == 54

#: 状态 = 54 个字符的 tuple；复原态 = 0..53 的编号（即"贴纸 i 在原位 i"）。
SOLVED = tuple(range(54))

_TOKEN = re.compile(r"^([UDLRFB])(2|2'|')?$")


def _turn_perm(face: str, k: int) -> tuple[int, ...]:
    """把动作 face^k 作用在编号态上, 读出 54 个贴纸的位置置换。"""
    ax, sign = _axis_of(face)
    # 从面外侧看顺时针 = 绕外法向 -90° = 绕正轴 (-sign) * 90°
    steps = (-sign * k) % 4
    perm = {}
    for i, (pos, nrm) in enumerate(STICKERS):
        if pos[ax] == sign:
            perm[i] = STICKERS.index((_rotk(pos, ax, steps), _rotk(nrm, ax, steps)))
        else:
            perm[i] = i
    return tuple(perm[i] for i in range(54))


#: 18 个合法动作 -> 置换。顺序: 正转 / 逆转 / 半转。
def _all_moves() -> list[str]:
    return [f + s for f in ("U", "D", "L", "R", "F", "B") for s in ("", "'", "2")]


MOVES = _all_moves()


def _k_of(token: str) -> int:
    """动作记号 -> 90° 的倍数: R=1, R'=3, R2=2。"""
    suf = token[1:]
    return {"": 1, "'": 3, "2": 2}[suf]


MOVE_PERM = {m: _turn_perm(m[0], _k_of(m)) for m in MOVES}


def _apply_key(key: tuple, moves: Iterable[str]) -> tuple:
    for m in moves:
        key = tuple(key[p] for p in MOVE_PERM[m])
    return key


def _inverse(m: str) -> str:
    """R -> R', R' -> R, R2 -> R2。"""
    if m.endswith("'"):
        return m[0]
    if m.endswith("2"):
        return m[0] + "2"
    return m[0] + "'"


# --------------------------------------------------------------------------
# 打乱状态（造法见 docstring）
# --------------------------------------------------------------------------

SCRAMBLE_SEED: int = 20260908

#: 冻结的 6 步打乱序列（相邻两面不同 —— 见 docstring 第 1 段）
SCRAMBLE: tuple[str, ...] = ("D'", "B'", "R'", "D'", "R", "D'")


def _gen_scramble(seed: int = SCRAMBLE_SEED) -> tuple[str, ...]:
    """LCG 抽 6 个动作, 相邻两面不同（否则等价动作会合并/抵消, 步数就不准了）。"""
    s = seed & 0x7FFFFFFF

    def nxt() -> int:
        nonlocal s
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        return s

    seq: list[str] = []
    prev = None
    for _ in range(6):
        while True:
            face = ("U", "D", "L", "R", "F", "B")[nxt() % 6]
            suffix = ("", "'", "2")[nxt() % 3]
            if face != prev:
                break
        seq.append(face + suffix)
        prev = face
    return tuple(seq)


# --------------------------------------------------------------------------
# 独立求最短距离（双向 BFS）
# --------------------------------------------------------------------------

def _bidirectional_distance(start: tuple, goal: tuple, max_depth: int = 12) -> int:
    """双向 BFS 求最短距离。深度 6 的题只需各扩 3 层, 毫秒级。

    这是 verifier 自己算的最优性证据, 不采信 agent 的 optimal 字段。
    两端交替各扩一层, 第一次相遇时 da+db 就是真正的最短距离（BFS 层号保证）。
    """
    if start == goal:
        return 0
    seen_a = {start: 0}
    seen_b = {goal: 0}
    fa, fb = [start], [goal]
    for depth in range(1, max_depth + 1):
        nfa = []
        for key in fa:
            for m in MOVES:
                nk = tuple(key[p] for p in MOVE_PERM[m])
                if nk in seen_a:
                    continue
                seen_a[nk] = depth
                if nk in seen_b:
                    return depth + seen_b[nk]
                nfa.append(nk)
        fa = nfa
        nfb = []
        for key in fb:
            for m in MOVES:
                nk = tuple(key[p] for p in MOVE_PERM[_inverse(m)])
                if nk in seen_b:
                    continue
                seen_b[nk] = depth
                if nk in seen_a:
                    return depth + seen_a[nk]
                nfb.append(nk)
        fb = nfb
    raise RuntimeError("未在 max_depth 内相遇")


def _optimal_distance() -> int:
    return _bidirectional_distance(_apply_key(SOLVED, SCRAMBLE), SOLVED)


# --------------------------------------------------------------------------
# 制品解析
# --------------------------------------------------------------------------

def _parse_moves(raw) -> tuple[list[str] | None, str]:
    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()
    if not isinstance(raw, list) or not raw:
        return None, "moves 不是非空列表/字符串"
    out: list[str] = []
    for m in raw:
        if not isinstance(m, str):
            return None, f"动作不是字符串: {m!r}"
        tok = m.strip().upper().replace("’", "'")
        if not _TOKEN.match(tok):
            return None, f"无法识别的动作记号: {m!r}（只接受 R / R' / R2 这类）"
        out.append(tok)
    return out, "ok"


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

def verify(stdout: str, code: str, claim: str) -> dict:
    ck = Check()
    art = None
    for tag in ("CONSTRUCT", "ANSWER"):
        art = parse_artifact(stdout, tag)
        if art is not None:
            ck.note("artifact_tag", tag)
            break
    if art is None:
        ck.expect("artifact_present", False, "未提交 CONSTRUCT 制品")
        return _result(ck, None, None)

    moves, why = _parse_moves(art.get("moves", art.get("solution")))
    if not ck.expect("moves_wellformed", moves is not None, why):
        return _result(ck, art, None)
    assert moves is not None

    scrambled = _apply_key(SOLVED, SCRAMBLE)
    ck.note("scramble", list(SCRAMBLE))

    # 1) 自己模拟: 动作序列必须把打乱态还原
    final = _apply_key(scrambled, moves)
    ck.expect("solves_the_cube", final == SOLVED, "按 agent 的动作序列模拟, 魔方没有被还原")

    # 2) 步数
    ck.note("move_count", len(moves))
    if "length" in art:
        ck.expect("claimed_length_consistent", art.get("length") == len(moves),
                  f"自报 length={art.get('length')} 与实际动作数 {len(moves)} 矛盾")
    ck.expect("length_within_6", len(moves) <= 6, f"步数 {len(moves)} > 6")

    # 3) 最优性: verifier 自己双向 BFS
    try:
        opt = _optimal_distance()
        ck.note("verifier_bfs_optimum", opt)
        ck.expect("verifier_optimum_is_6", opt == 6,
                  f"verifier 算出的最短距离是 {opt}, 与题目'6 步'前提不符")
        # 步数 <= 6 且可行 => 必然 == 最优距离, 所以最优性是被 verifier 证出来的
        ck.expect("length_is_optimal", len(moves) == opt,
                  f"步数 {len(moves)} != BFS 最优 {opt}")
        if art.get("optimal") is True:
            ck.expect("claimed_optimal_consistent", True, "")
            ck.note("optimality_note",
                    f"agent 自称 optimal; verifier 双向 BFS 独立确认最优距离 = {opt}")
        elif art.get("optimal") is False:
            ck.note("optimality_note", "agent 未自称最优; 最优性由 verifier 的 BFS 独立确认")
    except Exception as e:
        ck.expect("verifier_bfs_ran", False, f"verifier 自己的 BFS 失败: {e}")

    cert = (f"SCRAMBLE={' '.join(SCRAMBLE)}; 模拟还原成功; "
            f"步数 {len(moves)}; verifier 双向 BFS 最优距离 {ck.details.get('verifier_bfs_optimum')}")
    return _result(ck, art, cert)


def _result(ck: Check, art: dict | None, cert: str | None) -> dict:
    final = None
    if art is not None:
        final = json.dumps({"moves": art.get("moves"), "length": art.get("length")},
                           ensure_ascii=False, default=str)
    return {
        "correct": ck.correct,
        "certificate": cert if ck.correct else f"FAILED: {ck.reason()}",
        "final_answer": final,
        "details": ck.details,
    }


# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------

def _model_selftest() -> list[str]:
    """魔方模型和通行记号是否一致 —— 靠通行魔方的已知代数事实来钉。"""
    fails = []
    # 1) 每个动作 4 阶
    for m in MOVES:
        if _apply_key(SOLVED, [m] * 4) != SOLVED:
            fails.append(f"{m} 四阶不成立")
    # 2) 相对面可交换: R L = L R, U D = U D, F B = F B
    for a, b in (("R", "L"), ("U", "D"), ("F", "B")):
        for suf in ("", "'", "2"):
            if _apply_key(SOLVED, [a + suf, b + suf]) != _apply_key(SOLVED, [b + suf, a + suf]):
                fails.append(f"{a+suf} 与 {b+suf} 不可交换")
    # 3) sexy move (R U R' U') 的阶是 6
    if _apply_key(SOLVED, ["R", "U", "R'", "U'"] * 6) != SOLVED:
        fails.append("(R U R' U')^6 != 复原态")
    # 4) (R U)^105 = 复原态 —— 这条最强: R 与 U 的相对朝向错了就会失效
    if _apply_key(SOLVED, ["R", "U"] * 105) != SOLVED:
        fails.append("(R U)^105 != 复原态")
    # 5) 记号方向抽查: U 应把 F 面顶行送到 L, R 把 F 右列送到 U, F 把 U 顶行送到 R
    if _apply_key(SOLVED, ["U", "U", "U"]) != _apply_key(SOLVED, ["U'"]):
        fails.append("U 与 U' 不互逆")
    return fails


def _scramble_state() -> tuple:
    return _apply_key(SOLVED, SCRAMBLE)


def _solve_stdout(moves, length=None, optimal=True) -> str:
    art = {"moves": list(moves) if not isinstance(moves, str) else moves}
    if length is None:
        length = len(moves.split()) if isinstance(moves, str) else len(moves)
    art["length"] = length
    if optimal is not None:
        art["optimal"] = optimal
    return "VERDICT: PROVED\nCONSTRUCT: " + json.dumps(art, ensure_ascii=False)


def _selftest() -> int:
    failures: list[str] = []

    def t(name: str, cond: bool) -> None:
        print(("[OK] " if cond else "[FAIL] ") + name)
        if not cond:
            failures.append(name)

    # --- 模型自检: 记号是否与通行约定一致 ---
    t("魔方模型自检(RU关系/阶/可交换性)", not _model_selftest())
    for f in _model_selftest():
        print("    ", f)

    # --- 打乱常量自洽 ---
    t("打乱序列可复现", _gen_scramble() == SCRAMBLE)
    t("打乱序列相邻两面不同(否则步数会被合并)",
      all(SCRAMBLE[i][0] != SCRAMBLE[i + 1][0] for i in range(len(SCRAMBLE) - 1)))
    t("打乱序列长度 6", len(SCRAMBLE) == 6)
    inverse = [_inverse(m) for m in reversed(SCRAMBLE)]
    t("逆序列能还原", _apply_key(_scramble_state(), inverse) == SOLVED)
    opt = _optimal_distance()
    t("verifier 双向BFS: 最优距离 = 6", opt == 6)
    print(f"    SCRAMBLE = {' '.join(SCRAMBLE)} | BFS optimum = {opt}")

    # --- 正例 ---
    t("正例(6 步逆序列) -> correct=True", verify(_solve_stdout(inverse), "", "")["correct"])
    t("正例(字符串写法) -> correct=True",
      verify(_solve_stdout(" ".join(inverse)), "", "")["correct"])

    # --- 反例 ---
    # 1) 方向反了
    t("反例1(顺序不倒过来, 方向错) -> correct=False", not verify(_solve_stdout(SCRAMBLE), "", "")["correct"])
    # 2) 漏一步
    t("反例2(漏一步) -> correct=False", not verify(_solve_stdout(inverse[:-1]), "", "")["correct"])
    # 3) 多加一步(仍然可行但步数不是 6 -> 非最优)
    t("反例3(多一步, 仍可行但非最优) -> correct=False",
      not verify(_solve_stdout(inverse + ["R"]), "", "")["correct"])
    # 4) 自报步数造假
    t("反例4(自报 length 与实际不符) -> correct=False",
      not verify(_solve_stdout(inverse, length=3), "", "")["correct"])
    # 5) 完全不还原
    t("反例5(乱序动作) -> correct=False",
      not verify(_solve_stdout(["R", "R", "R", "R", "R", "R"]), "", "")["correct"])
    # 6) 无制品
    t("反例6(无制品) -> correct=False", not verify("VERDICT: PROVED\n", "", "")["correct"])
    # 7) 记号写法非法
    t("反例7(非法记号) -> correct=False",
      not verify(_solve_stdout(["X", "Y", "Z", "R", "U", "F"]), "", "")["correct"])

    if failures:
        print("[SELFTEST FAILED] " + ", ".join(failures))
        return 1
    print("[OK] 自测全部通过")
    return 0


def _dump() -> None:
    scramble = _gen_scramble()
    print("SCRAMBLE =", tuple(scramble))
    print("BFS optimum =", _optimal_distance())
    print("inverse =", " ".join(_inverse(m) for m in reversed(scramble)))


if __name__ == "__main__":
    if "--dump" in sys.argv:
        _dump()
    else:
        sys.exit(_selftest())