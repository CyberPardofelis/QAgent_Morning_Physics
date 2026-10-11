"""res-012-minimal-polynomial 的独立 verifier。

题目：求 alpha = sqrt(2)+sqrt(3)+sqrt(5) 在 Q 上的极小多项式，并验证其次数和不可约性。

## 制品格式（题库 expected.value 声明的字段是**唯一契约**）
```
ANSWER: {"polynomial": "x**8 - 40*x**6 + 352*x**4 - 960*x**2 + 576",
         "degree": 8,
         "irreducible": true}
```
``coeffs`` 数组写法**同样接受**（等价，不是兜底）：
```
ANSWER: {"coeffs": [576, 0, -960, 0, 352, 0, -40, 0, 1],   # coeffs[i] 是 x^i 的系数
         "degree": 8, "irreducible": true}
```
解析器见 :func:`_parse_poly`（支持省略系数、``x^8`` 与 ``x**8`` 两种幂写法）。

★ 为什么两种都认 ★
题库 ``expected.value`` 写的是 ``polynomial``（字符串），那**就是**契约。
verifier 之前只肯要 ``coeffs`` 数组，于是 2026-10-08 那次基准里，agent 交出
一字不差的正确多项式却被判错 —— 报的是"没能拿到有理系数数组"，
根因是 verifier 自己的解析器把 ``x**8`` 吃成了 ``x8``（见 :func:`_parse_poly` 归一化）。
放宽的是**解析**, 不是判定: 下面 5 条数学判据一步都没松。

## 判据（为什么这么判就是"真的做对了"）
* **精确代回**（不是数值近似）：把 agent 给的多项式代入 alpha = sqrt2+sqrt3+sqrt5，
  用 sympy 精确化简，要求**恰好等于 0**。这一条排除"近似对但其实不是根"。
  数值验证（mpmath 50 位）同时跑一遍，作为独立交叉检查并给出容差记录。
* **与真正的极小多项式一致**：verifier 自己用 sympy 算一遍 ``minimal_polynomial``，
  再和硬编码的真值常量对账。两个来源一致才算数。
* **次数 = 8**：单靠"代入为 0"挡不住"给一个 8 次幂的倍数"（比如把极小多项式平方）。
* **在 Q 上不可约**：挡不住"次数少一点的因子"和"高次的倍数"。
  极小多项式的定义 = 单项式 + 不可约 + 以 alpha 为根，三条一起才是定义本身。
  （Eisenstein 在本题不适用 —— 本题的首项系数是 1，Eisenstein 要求 p | 所有
  非首项系数且 p² 不整除常数项；p=1 会退化。所以用 sympy 的 is_irreducible。）
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

import sympy  # noqa: E402
import mpmath  # noqa: E402

_X = sympy.Symbol("x")

#: 真值（升幂系数）：x^8 - 40x^6 + 352x^4 - 960x^2 + 576
#: 独立来源有二：sympy.minimal_polynomial，以及手工消元；两者一致。
TRUTH_COEFFS_ASC: tuple[int, ...] = (576, 0, -960, 0, 352, 0, -40, 0, 1)
TRUTH_DEGREE: int = 8


# --------------------------------------------------------------------------
# 极简多项式解析器（只支持 coeffs 数组；字符串是兜底）
# --------------------------------------------------------------------------
_TERM = re.compile(
    r"^(?P<coef>[+-]?(?:\d+\.\d+|\.\d+|\d+)?(?:/\d+)?)?"
    # ★ var 允许多字母 ★ 实测 agent 交的是 `xv**8 - 40*xv**6 + ...`,
    # 单字母 [A-Za-z] 会把 xv 拆坏。变量名叫什么不影响多项式是什么 ——
    # 真正的约束是"整串只能有一个变量名"(混用 x 和 y 才是真错), 那在
    # _parse_poly 里用 used_vars 集合把。
    r"(?:\*?\s*(?P<var>[A-Za-z][A-Za-z0-9_]*)(?:\s*(?:\^|\*\*)\s*(?P<deg>\d+))?)?$"
)


def _to_fraction(s: str | None) -> Fraction | None:
    """把系数串变成有理数，**前导 + 也要认**。

    ★ 2026-10-09 实测事故 ★
    agent 交的是 `+x**8 - 40*x**6 + ...`(带前导加号 —— 答案格式约定让它
    照着字段名写), 而 `Fraction("+")` 抛
    ValueError: Invalid literal for Fraction: '+' —— **整道题记成 verifier
    异常, 一分没得**, 而它交的是正确答案。
    这是「verifier 考表示形式而不是考数学」的第 5 次实例:
    `+x**8` 和 `x**8` 是同一个多项式, 不该有任何差别。
    """
    if s is None:
        return Fraction(1)
    t = s.strip()
    while t.startswith("+"):        # Fraction("+1") 和 Fraction("+") 都非法
        t = t[1:].strip()
    if not t:
        return Fraction(1)
    if "/" in t:
        num, den = t.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(t)


def _parse_poly(text: str) -> list[Fraction] | None:
    """解析 ``x^8-40x^6+352x^4-960x^2+576`` 这类写法。

    只支持"单项式的加减"（系数可省略、可为分数、幂可写 ``x**8`` 或 ``x^8``），
    变量必须统一叫 ``x``。解析不了就返回 None —— 不猜、不迁就。

    ★ 归一化顺序是关键 ★
    必须**先**把 ``**`` 折成 ``^`` 再删剩下的 ``*``。
    反过来（先删 ``*``）会把 ``x**8`` 直接吃成 ``x8`` —— 这正是 2026-10-08
    那次假阴性的根因: 契约里的 ``polynomial`` 字段明明给了正确答案，
    解析器却返回 None, 于是报"没能拿到有理系数数组"。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.replace("−", "-").replace("–", "-")
    s = s.replace(" ", "")
    s = s.replace("**", "^")          # Python 的幂写法 -> 幂号（必须最先）
    s = s.replace("*", "")            # 再删乘号: 40*x^6 -> 40x^6
    if not s:
        return None
    body = s[1:] if s[0] in "+-" else s
    coeffs: list[Fraction] = []
    used_vars: set[str] = set()
    for term in re.split(r"(?<!^)(?=[+-])", body):
        if not term:
            continue
        m = _TERM.match(term)
        if not m:
            return None
        var = m.group("var")
        # ★ 不要求变量名必须是 x ★
        # 实测(2026-10-09): agent 交的是 `xv**8 - 40*xv**6 + ...` —— 用 xv 当变量名,
        # 数学上和 x**8 - 40*x**6 + ... 是同一个多项式。要求特定字母名属于
        # "verifier 考表示形式而不是考数学"(第 5 次)。
        # 只要**整串只有一个变量符号**就接受; 混用两个变量(如 x 与 y)才是真错,
        # 因为那就不是一元多项式了。
        if var is not None:
            used_vars.add(var)
        # 按**次数**归档, 不能按项的先后 —— "x^8-40x^6" 和 "576-960x^2+...+x^8"
        # 两种写法都要解析成同一个升幂系数数组。
        deg = int(m.group("deg") or 0) if var is not None else 0
        coef = _to_fraction(m.group("coef"))
        if coef is None:
            return None
        while len(coeffs) <= deg:
            coeffs.append(Fraction(0))
        coeffs[deg] += coef
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    # 整串只能有一个变量符号 —— 混用 x 和 y 说明它不是一元多项式, 那才是真错
    if len(used_vars) > 1:
        return None
    return coeffs


def _coeffs_from_artifact(art: dict) -> tuple[list[Fraction] | None, str]:
    """取有理系数数组。

    契约字段是 ``polynomial``（字符串），``coeffs``（数组）是等价写法, 两者都认;
    哪个先出现用哪个。两者都给不出时才返回 None。
    """
    raw = art.get("coeffs")
    if isinstance(raw, list) and len(raw) >= 2:
        out: list[Fraction] = []
        for v in raw:
            if isinstance(v, bool):
                return None, "coeffs 里有布尔值"
            if isinstance(v, int):
                out.append(Fraction(v))
            elif isinstance(v, float) and v.is_integer():
                out.append(Fraction(int(v)))
            elif isinstance(v, str) and v.strip():
                try:
                    # 复用 _to_fraction: 它已经处理了前导 + 这类写法差异
                    got = _to_fraction(v.strip())
                    if got is None:
                        return None, f"coeffs 元素无法解析: {v!r}"
                    out.append(got)
                except Exception:
                    return None, f"coeffs 元素无法解析为有理数: {v!r}"
            else:
                return None, f"coeffs 元素类型不对: {type(v).__name__}"
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return out, "coeffs"
    for key in ("polynomial", "poly", "expr"):
        if key in art:
            parsed = _parse_poly(art.get(key))
            if parsed is not None:
                return parsed, f"{key}(题库契约字段, 字符串)"
            return None, f"{key} 字符串解析失败（要 'x**8 - 40*x**6 + ...' 这种加减式）"
    return None, "既没给 polynomial 字符串, 也没给 coeffs 数组"


# --------------------------------------------------------------------------
# 独立求值 / 求真值
# --------------------------------------------------------------------------

def _eval_at_alpha(poly: list[Fraction]) -> sympy.Expr:
    """把多项式代入 alpha = sqrt2+sqrt3+sqrt5 并精确化简。"""
    expr = sympy.Integer(0)
    for i, c in enumerate(poly):
        expr += sympy.Rational(c.numerator, c.denominator) * _X ** i
    alpha = sympy.sqrt(2) + sympy.sqrt(3) + sympy.sqrt(5)
    return sympy.simplify(sympy.expand(expr.subs(_X, alpha)))


def _numeric_residual(poly: list[Fraction]) -> float:
    """mpmath 50 位下的相对残差 —— 与上面的精确判定交叉验证。"""
    mpmath.mp.dps = 50
    a = mpmath.sqrt(2) + mpmath.sqrt(3) + mpmath.sqrt(5)
    val = mpmath.mpf(0)
    scale = mpmath.mpf(0)
    for i, c in enumerate(poly):
        term = mpmath.mpf(c.numerator) / c.denominator * a ** i
        val += term
        scale += abs(term)
    return float(abs(val) / max(mpmath.mpf(1), scale))


def _true_minimal_polynomial() -> list[Fraction]:
    """verifier 自己算的真值（sympy），不读 agent 的任何东西。"""
    alpha = sympy.sqrt(2) + sympy.sqrt(3) + sympy.sqrt(5)
    mp = sympy.Poly(sympy.expand(sympy.minimal_polynomial(alpha, _X)), _X, domain=sympy.QQ)
    desc = mp.all_coeffs()
    return [Fraction(int(c)) for c in desc[::-1]]


def _same_up_to_scalar(a: list[Fraction], b: list[Fraction]) -> bool:
    """两个有理系数多项式是否只差一个非零有理因子（极小多项式按定义只差单位元）。"""
    n = max(len(a), len(b))
    a = list(a) + [Fraction(0)] * (n - len(a))
    b = list(b) + [Fraction(0)] * (n - len(b))
    nz = [(x, y) for x, y in zip(a, b) if y != 0]
    if not nz:
        return False
    scale = nz[0][0] / nz[0][1]
    return all(x == scale * y for x, y in zip(a, b)) and scale != 0


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

def verify(stdout: str, code: str, claim: str) -> dict:
    ck = Check()
    art = parse_artifact(stdout, "ANSWER")
    if art is None:
        ck.expect("artifact_present", False, "未提交 ANSWER 制品")
        return _result(ck, None, "artifact 缺失")

    coeffs, source = _coeffs_from_artifact(art)
    if not ck.expect("coeffs_parsed", coeffs is not None and len(coeffs or []) >= 2,
                     f"没能从制品里拿到有理系数数组（{source}）"):
        return _result(ck, art, source)
    ck.note("coeff_source", source)

    # 1) 精确代回: p(alpha) == 0
    try:
        exact = _eval_at_alpha(coeffs)
    except Exception as e:  # 表达式爆了就记下来, 不吞
        exact = sympy.Symbol("parse_error")
        ck.note("exact_eval_error", repr(e)[:200])
    ck.expect("root_zero_exact", exact == 0,
              f"代入 alpha 后不精确为 0（得到 {sympy.srepr(exact)[:120] if exact == 0 else str(exact)[:120]}）")

    # 2) 数值交叉验证
    try:
        resid = _numeric_residual(coeffs)
    except Exception as e:
        resid = float("inf")
        ck.note("numeric_eval_error", repr(e)[:200])
    ck.note("numeric_relative_residual", resid)
    ck.expect("root_zero_numeric", resid < 1e-09, f"数值残差 {resid:.3e} 超过 1e-9")

    # 3) 与真正的极小多项式对账（两个独立来源）
    truth = _true_minimal_polynomial()
    ck.expect("sympy_agrees_with_truth", _same_up_to_scalar(truth, TRUTH_COEFFS_ASC),
              "verifier 自己的 sympy 结果与硬编码真值不一致")
    ck.expect("is_minimal_polynomial", _same_up_to_scalar(coeffs, truth),
              "不是 alpha 的极小多项式（与 sympy minimal_polynomial 不符）")

    # 4) 次数 = 8
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    degree = len(coeffs) - 1
    ck.note("degree_computed", degree)
    ck.expect("degree_is_8", degree == TRUTH_DEGREE, f"次数 {degree} != {TRUTH_DEGREE}")

    # 5) 在 Q 上不可约
    try:
        expr = sum(sympy.Rational(c.numerator, c.denominator) * _X ** i
                   for i, c in enumerate(coeffs))
        poly = sympy.Poly(sympy.expand(expr), _X, domain=sympy.QQ)
        irreducible = bool(poly.is_irreducible)
        factored = sympy.factor_list(poly.as_expr())[1]
    except Exception as e:
        irreducible, factored = False, []
        ck.note("irreducibility_error", repr(e)[:200])
    ck.expect("irreducible_over_QQ", irreducible, "在 Q 上可约，不是极小多项式")
    ck.note("factor_list", [sympy.srepr(f)[:60] for f in factored])

    # 6) agent 自报的字段不能和真值矛盾
    if "degree" in art:
        ck.expect("claimed_degree_consistent", art.get("degree") == degree,
                  f"自报 degree={art.get('degree')} 与实际次数 {degree} 矛盾")
    if "irreducible" in art:
        ck.expect("claimed_irreducible_consistent", bool(art.get("irreducible")) == irreducible,
                  f"自报 irreducible={art.get('irreducible')} 与实际 {irreducible} 矛盾")

    cert = f"p(alpha)==0(精确), deg={degree}, irreducible={irreducible}, coeffs={coeffs}"
    return _result(ck, art, cert)


def _result(ck: Check, art: dict | None, cert: str) -> dict:
    final = None
    if art is not None:
        final = json.dumps(
            {"coeffs": art.get("coeffs") or art.get("polynomial"),
             "degree": art.get("degree"), "irreducible": art.get("irreducible")},
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

    good = {"coeffs": list(TRUTH_COEFFS_ASC), "degree": 8, "irreducible": True}

    # 正例: 契约字段 polynomial（2026-10-08 基准里被误判的那份制品，逐字照抄）
    real = ("x**8 - 40*x**6 + 352*x**4 - 960*x**2 + 576")
    t("正例(题库契约字段 polynomial, x**N 写法) -> correct=True", verify(
        _stdout({"polynomial": real, "degree": 8, "irreducible": True}), "", "")["correct"])
    t("正例(polynomial, x^N 写法) -> correct=True", verify(
        _stdout({"polynomial": "x^8-40x^6+352x^4-960x^2+576", "degree": 8, "irreducible": True}),
        "", "")["correct"])
    t("正例(polynomial, 省略系数的 x**2-1 形状) 可解析", _parse_poly("x**2 - 1") == [Fraction(-1), Fraction(0), Fraction(1)])
    # 正例: 等价的系数数组写法
    t("正例(coeffs 数组) -> correct=True", verify(_stdout(good), "", "")["correct"])
    # 正例: 非首一的等价写法（极小多项式按定义只差单位元）
    scaled = [2 * c for c in TRUTH_COEFFS_ASC]
    t("正例(差一个有理因子 2x) -> correct=True",
      verify(_stdout({"coeffs": scaled}), "", "")["correct"])

    # 反例1: 常数项 +1 —— 几乎全对, 但不是根
    bad = list(TRUTH_COEFFS_ASC)
    bad[0] += 1
    t("反例1(常数项错) -> correct=False", not verify(_stdout({"coeffs": bad}), "", "")["correct"])
    # 反例2: 可约（少一半次数的因子）—— 代入可能仍是 0? 这里取平方, 次数 16
    sq = sympy.Poly(sympy.expand((_X ** 8 - 40 * _X ** 6 + 352 * _X ** 4 - 960 * _X ** 2 + 576) ** 2), _X)
    t("反例2(极小多项式的平方, 次数16) -> correct=False",
      not verify(_stdout({"coeffs": [int(c) for c in sq.all_coeffs()[::-1]]}), "", "")["correct"])
    # 反例3: 只报次数不报多项式 —— 放宽解析后仍然判错（数学判据一步没松）
    t("反例3(无多项式/无系数) -> correct=False", not verify(_stdout({"degree": 8}), "", "")["correct"])
    # 反例3b: polynomial 是字符串但内容错 —— 字符串通道不绕过数学判据
    t("反例3b(polynomial 字符串, 常数项错) -> correct=False",
      not verify(_stdout({"polynomial": "x**8-40*x**6+352*x**4-960*x**2+577"}), "", "")["correct"])
    t("反例3c(polynomial 字符串, 次数不够) -> correct=False",
      not verify(_stdout({"polynomial": "x**4-6*x**2+1"}), "", "")["correct"])
    # 反例4: 自报不可约但实际可约
    t("反例4(自报 irreducible 与实际矛盾) -> correct=False",
      not verify(_stdout({"coeffs": [int(c) for c in sq.all_coeffs()[::-1]], "degree": 16,
                          "irreducible": True}), "", "")["correct"])
    # 反例5: 完全没提交制品
    t("反例5(无制品) -> correct=False", not verify("VERDICT: PROVED\n", "", "")["correct"])

    if failures:
        print("[SELFTEST FAILED] " + ", ".join(failures))
        return 1
    print("[OK] 自测全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())