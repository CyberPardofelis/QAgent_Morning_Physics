"""res-013-groebner-system 的独立 verifier。

题目：在 F_101 上求解多项式方程组（见 ``data/res_13_system.txt``），给出**所有**解并验证。

## 为什么判据必须是自己枚举
本题埋了一个坑：**``sympy.solve([...], modulus=101)`` 对这个方程组返回错误解而且
不报错**（已实测，返回 4 组、其中没有一组代得回原方程；见 ``details["known_trap"]``）。
如果 verifier 采信 agent 报的"我代回验证过了"，就会把错解判成对。
所以本 verifier 的真值只有一个来源：**自己把 101x101 全枚举一遍**（10201 个点，
纯 Python 毫秒级）。Groebner 基再漂亮也是"方法"，枚举才是"事实"。

## 制品格式
```
ANSWER: {"solutions": [[22,23],[23,22],[78,79],[79,78]],
         "field": 101,
         "count": 4,
         "verified": true}
```
也接受 ``{"x":22,"y":23}`` 形式的解对象，以及负号代表元的写法（-22 即 79）。

## 判据（为什么这么判就是"真的做对了"）
* 方程组从 ``data/res_13_system.txt`` **解析**而来（不是硬编码），verifier 和
  agent 看的必须是同一组方程；解析失败直接判失败。
* 解集必须与枚举结果**完全相等**：少一个（漏解）和多一个（伪解）都算错。
  只判"每个解都代回成立"是不够的 —— 那正是本题被 sympy 骗过去的地方。
* 代回验证隐含在集合相等里：相等意味着每个元素都通过了全部方程的模 p 求值。
"""
from __future__ import annotations

import json
import re
import sys
from fractions import Fraction
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import Check, parse_artifact  # noqa: E402

#: 仓库根（data/res_13_system.txt 所在处）
_REPO = _HERE.parents[2]
SYSTEM_FILE = _REPO / "data" / "res_13_system.txt"

#: 题面里的有限域特征与变量（解析出来的结果会覆盖这里的默认值）
FIELD: int = 101
VARS: tuple[str, str] = ("x", "y")

#: 枚举真值（硬编码仅为交叉核对，主判据是运行时枚举）
TRUTH_SOLUTIONS: tuple[tuple[int, int], ...] = ((22, 23), (23, 22), (78, 79), (79, 78))

_KNOWN_TRAP = (
    "sympy.solve([x**2+y**2-3, x*y-1], [x, y], modulus=101) 对本方程组返回 4 组"
    "「解」且不报错，但它们全都代不回原方程（已实测）。因此本 verifier 的真值"
    "只来自 F_101 上的暴力枚举，不采信任何「我已经代回验证过了」的说法。"
)


# --------------------------------------------------------------------------
# 方程组解析: 自己写的小多项式解析器, 不 eval 任何 agent 提供的文本
# --------------------------------------------------------------------------

def _parse_multivariate(text: str, variables: tuple[str, ...]) -> dict[tuple[int, ...], Fraction]:
    """把 ``x^2 + y^2 - 3`` 解析成 {(2,0):1, (0,2):1, (0,0):-3}。

    只支持加减乘和整数次幂；变量名限死为 variables 里的那几个。
    解析不了就抛 ValueError —— 不猜、不 eval。
    """
    s = text.replace("−", "-").replace(" ", "").replace("*", "")
    if not s:
        raise ValueError("空的多项式")
    body = s[1:] if s[0] in "+-" else s
    out: dict[tuple[int, ...], Fraction] = {}
    for term in re.split(r"(?<!^)(?=[+-])", body):
        if not term:
            continue
        sign = Fraction(1)
        if term[0] == "-":
            sign, term = Fraction(-1), term[1:]
        elif term[0] == "+":
            term = term[1:]
        if not term:
            continue
        num = re.match(r"^\d+", term)
        coef = Fraction(sign)
        if num:
            coef *= Fraction(int(num.group(0)))
            term = term[num.end():]
        powers = [0] * len(variables)
        if term:
            for var, exp in re.findall(r"([a-zA-Z])(?:\^(\d+))?", term):
                if var not in variables:
                    raise ValueError(f"未知变量 {var!r}")
                powers[variables.index(var)] += int(exp) if exp else 1
        key = tuple(powers)
        out[key] = out.get(key, Fraction(0)) + coef
    return {k: v for k, v in out.items() if v != 0}


def load_system(path: Path = SYSTEM_FILE) -> list[dict[tuple[int, ...], Fraction]]:
    """从 res_13_system.txt 读出方程组。文件不在 -> 抛错（不静默回退）。"""
    text = path.read_text(encoding="utf-8")
    eqs: list[dict[tuple[int, ...], Fraction]] = []
    for line in text.splitlines():
        m = re.match(r"\s*(?:f\d+\s*:\s*)?(.*?=\s*\d+)\s*$", line)
        if not m:
            continue
        lhs, rhs = m.group(1).split("=")
        poly = _parse_multivariate(lhs.strip(), VARS)
        poly = _subtract_const(poly, int(rhs.strip()))
        if poly:
            eqs.append(poly)
    if not eqs:
        raise ValueError("没从 %s 解析出任何方程" % path)
    return eqs


def _subtract_const(poly: dict[tuple[int, ...], Fraction], value: int) -> dict[tuple[int, ...], Fraction]:
    key = tuple([0] * len(next(iter(poly))))
    poly = dict(poly)
    poly[key] = poly.get(key, Fraction(0)) - Fraction(value)
    return {k: v for k, v in poly.items() if v != 0}


# --------------------------------------------------------------------------
# 独立真值: 全枚举
# --------------------------------------------------------------------------

def enumerate_solutions(eqs, p: int = FIELD, variables: tuple[str, ...] = VARS):
    """穷举 F_p^n 上的全部解。n 小、p=101 -> 10201 次求值，直接 Python 跑。"""
    n = len(variables)
    sols = []
    for xi in range(p):
        for yi in range(p):
            pt = (xi, yi)[:n]
            if all(sum(int(c) * _pow(pt, e) for e, c in poly.items()) % p == 0 for poly in eqs):
                sols.append(pt if n > 1 else (pt[0],))
    return sols


def _pow(pt, exponents) -> int:
    val = 1
    for base, e in zip(pt, exponents):
        val *= base ** e
    return val


def _satisfies(pt, eqs, p: int = FIELD) -> bool:
    return all(sum(int(c) * _pow(pt, e) for e, c in poly.items()) % p == 0 for poly in eqs)


# --------------------------------------------------------------------------
# 制品解析
# --------------------------------------------------------------------------

def _norm_solutions(raw) -> tuple[list[tuple[int, int]] | None, str]:
    """把 agent 给的解列表归一成 (x, y) 的整数元组列表（按 mod p 归一）。"""
    if not isinstance(raw, list) or not raw:
        return None, "solutions 不是非空列表"
    out: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, dict):
            if not all(v in item for v in VARS):
                return None, "解对象缺 x/y 字段"
            item = [item[v] for v in VARS]
        if not isinstance(item, (list, tuple)) or len(item) != len(VARS):
            return None, f"解的格式不对: {item!r}"
        vals = []
        for v in item:
            if isinstance(v, bool):
                return None, "解里出现布尔值"
            if isinstance(v, float):
                if not v.is_integer():
                    return None, f"解不是整数: {v!r}"
                v = int(v)
            if not isinstance(v, int):
                return None, f"解不是整数: {v!r}"
            vals.append(v % FIELD)  # -22 与 79 是同一个 F_101 元素, 统一归一
        out.append(tuple(vals))  # type: ignore[arg-type]
    return out, "ok"


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

def verify(stdout: str, code: str, claim: str) -> dict:
    ck = Check()
    art = None
    for tag in ("ANSWER", "COUNTEREXAMPLE", "CONSTRUCT"):
        art = parse_artifact(stdout, tag)
        if art is not None:
            ck.note("artifact_tag", tag)
            break
    if art is None:
        ck.expect("artifact_present", False, "未提交解集制品")
        return _result(ck, None)

    # 方程组从文件解析, 解析失败就是 verifier 自己出问题了 -> 明确判失败
    try:
        eqs = load_system()
    except Exception as e:
        ck.expect("system_parsed", False, f"无法从 {SYSTEM_FILE} 解析方程组: {e}")
        return _result(ck, art)
    ck.expect("system_parsed", True)
    ck.note("system", [str(eq) for eq in eqs])
    ck.note("field", FIELD)

    truth = enumerate_solutions(eqs, FIELD)
    ck.note("enumerated_solutions", [list(s) for s in truth])
    ck.expect("verifier_truth_selfcheck", sorted(tuple(s) for s in truth) == sorted(TRUTH_SOLUTIONS),
              "verifier 枚举结果与硬编码真值不一致")

    sols, why = _norm_solutions(art.get("solutions", art.get("sol", art.get("solution"))))
    if not ck.expect("solutions_wellformed", sols is not None, why):
        return _result(ck, art)
    assert sols is not None

    ck.note("duplicate_entries", len(sols) != len(set(sols)))
    got = set(sols)
    want = set(truth)
    missing = sorted(want - got)
    extra = sorted(got - want)
    ck.details["missing_solutions"] = [list(m) for m in missing]
    ck.details["extra_solutions"] = [list(e) for e in extra]
    # 少一个 = 漏解, 多一个 = 伪解（多半就是 sympy 那种"代不回去"的解）, 都判错
    ck.expect("solution_set_exact_match", not missing and not extra,
              f"解集不等: 缺 {missing}, 多 {extra}")

    # 逐个代回（集合相等已蕴含, 这里再显式核一遍并留下证据）
    bad = [list(s) for s in sols if not _satisfies(s, eqs, FIELD)]
    ck.expect("each_solution_substitutes", not bad, f"这些解代不回原方程: {bad}")

    # 自报的解数不能和实际矛盾
    if "count" in art:
        ck.expect("claimed_count_consistent", art.get("count") == len(sols),
                  f"自报 count={art.get('count')} 与实际给出 {len(sols)} 组矛盾")

    ck.note("known_trap", _KNOWN_TRAP)
    cert = f"F_{FIELD} 全枚举 -> {len(truth)} 组解 {[list(s) for s in truth]}; 提交解集完全相等"
    return _result(ck, art, cert)


def _result(ck: Check, art: dict | None, cert: str | None = None) -> dict:
    final = None
    if art is not None:
        final = json.dumps({"solutions": art.get("solutions"), "field": FIELD},
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

def _stdout(art: dict) -> str:
    return "VERDICT: PROVED\nANSWER: " + json.dumps(art, ensure_ascii=False)


def _selftest() -> int:
    failures: list[str] = []

    def t(name: str, cond: bool) -> None:
        print(("[OK] " if cond else "[FAIL] ") + name)
        if not cond:
            failures.append(name)

    eqs = load_system()
    truth = [list(s) for s in enumerate_solutions(eqs, FIELD)]
    t("verifier 枚举 = 4 组解", truth == [list(s) for s in TRUTH_SOLUTIONS])
    print("    枚举结果:", truth)

    good = {"solutions": truth, "field": 101, "count": 4, "verified": True}
    t("正例(全部 4 组解) -> correct=True", verify(_stdout(good), "", "")["correct"])

    # 反例1: sympy.solve(modulus=101) 的错误解 —— 代不回去
    wrong = {"solutions": [[0, 1], [0, 100], [1, 0], [100, 0]], "field": 101, "count": 4}
    print("    陷阱解代回结果:", [_satisfies(tuple(s), eqs, FIELD) for s in wrong["solutions"]])
    t("反例1(sympy 式错误解) -> correct=False", not verify(_stdout(wrong), "", "")["correct"])

    # 反例2: 漏一组解（只给 3 组, 而且是"看起来很整"的前 3 组）
    t("反例2(漏 1 组解) -> correct=False",
      not verify(_stdout({"solutions": truth[:3]}), "", "")["correct"])

    # 反例3: 多一组解
    t("反例3(多 1 组伪解) -> correct=False",
      not verify(_stdout({"solutions": truth + [[1, 2]]}), "", "")["correct"])

    # 反例4: 用负号代表元写 F_101 元素（应被判为正确, 这是归一化而不是放过）
    t("正例2(负号代表元 -22/-23) -> correct=True",
      verify(_stdout({"solutions": [[-22, -23], [-23, -22], [-78, -79], [-79, -78]]}), "", "")["correct"])

    # 反例5: 没交制品
    t("反例4(无制品) -> correct=False", not verify("VERDICT: PROVED\n", "", "")["correct"])

    if failures:
        print("[SELFTEST FAILED] " + ", ".join(failures))
        return 1
    print("[OK] 自测全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())