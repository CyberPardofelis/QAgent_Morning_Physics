"""res-015-cyclic-inequality-counterexample 的独立 verifier。

题目：检验猜想：对任意正实数 a,b,c，
      a/(b+c) + b/(c+a) + c/(a+b) >= 3/2。成立给证明，不成立给反例。

## 真值：它是**成立**的（Nesbitt 不等式），等号当且仅当 a=b=c
题目说"若成立给证明，若不成立给反例"，所以 agent 有两种合法答法。
verifier 必须能同时处理，并且**不能靠随机搜索下结论** —— 题目 notes 里
明写了"别被随机搜索骗了"。因此这里的分工是:

* **证明分支**：verifier 自己用 sympy 精确验证 Cauchy-Schwarz(Titu) 那条链
  上的每一个恒等式（这是真证明，不是采样），再用**精确有理数**采样 + 边界情形
  做独立交叉验证。采样只是抓"agent 在更窄的范围内声称成立"这种错误，
  不能替代证明 —— 所以 identity 检查失败一律判错。
* **反例分支**：把 agent 给的 (a,b,c) **精确代回**。真反例会真的违反不等式；
  代不回去的"反例"就是假证伪 -> correct=False。Nesbitt 成立，所以正常情况下
  必然落到后者。

## 制品格式（题库 expected.value 声明的字段是**唯一契约**）
题库声明的是 ``verdict`` + ``proof_method``:
```
ANSWER: {"verdict": "holds", "proof_method": "Titu/Cauchy: a/(b+c)>=a^2/(ab+ac), (a+b+c)^2>=3(ab+bc+ca)"}
```
``certificate``（可机核的 SOS 结构）**仍是更强的一档**, 给了就照旧精确验证:
```
ANSWER: {"verdict": "holds", "proof_method": "cauchy_schwarz",
         "certificate": {"kind": "sos", "terms": ["(a**3+b**3+c**3-3*a*b*c)/(2*(a+b)*(a+c)*(b+c))", ...]}}
```
反例:
```
COUNTEREXAMPLE: {"a": 1, "b": 1, "c": 1, "lhs": 1.5}
```
``certificate.kind`` 取 ``sos`` 时, ``terms`` 是若干**多项式表达式**（变量 a,b,c,
有理系数）。verifier 用自己写的表达式解析器（不 eval agent 的文本）把它们求和,
要求与 ``lhs - 3/2`` **恒等**, 并在采样点上逐项非负。

★ 为什么 proof_method 也算数 ★
verifier 之前只肯要 ``certificate`` 这个自造字段, 于是按题库契约提交的
标准答案（``verdict`` + ``proof_method``）被判 "certificate_present=False" ——
考的是格式合规, 不是数学。现在 ``proof_method`` 是契约字段, 认。
但"认"不等于"放过": :func:`_verify_proof` 会照样检查
  1. verdict 确实声称成立;
  2. 声称的适用范围不窄于"任意正实数"(给了 domain 时);
  3. proof_method 里若夹带 (a,b,c) 坐标 → **精确代回**验它是不是真反例;
  4. 命题本身由 verifier 自己的 Cauchy 恒等式链证明(与 agent 说什么无关)。
文本证明没法逐字机检, 所以 details 里会写明"验到了什么、没验什么"。
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

_A, _B, _C = sympy.symbols("a b c", real=True)
#: 统一用这一套符号。sympy.Symbol('a') 和 sympy.Symbol('a', real=True) 是**不同**的对象,
#: 混用会让所有恒等式判定静默失效（simplify 不为 0）。解析器与证明链必须共用它。
SYMBOLS = {"a": _A, "b": _B, "c": _C}
LHS = _A / (_B + _C) + _B / (_C + _A) + _C / (_A + _B)
TARGET = sympy.Rational(3, 2)


# --------------------------------------------------------------------------
# 自己的表达式解析器（给 SOS 证书用）—— 只 token 化再重建, 不 eval
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?(?:/\d+)?)|(?P<var>[abc])|"
                    r"(?P<op>\*\*|\^|[+\-*/()]))")


def _parse_expr(text: str, variables=("a", "b", "c")):
    """解析多项式表达式成 sympy Expr。解析不了返回 None。"""
    if not isinstance(text, str) or not text.strip():
        return None
    toks, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == m.start():
            return None
        toks.append(m)
        pos = m.end()
    i = 0

    def peek():
        return toks[i].group("op") if i < len(toks) else None

    def expr():
        nonlocal i
        val = term()
        while peek() in ("+", "-"):
            op = peek()
            i += 1
            rhs = term()
            val = val + rhs if op == "+" else val - rhs
        return val

    def term():
        nonlocal i
        val = power()
        while peek() in ("*", "/"):
            op = peek()
            i += 1
            rhs = power()
            val = val * rhs if op == "*" else val / rhs
        return val

    def power():
        nonlocal i
        val = unary()
        if peek() in ("^", "**"):
            i += 1
            val = val ** power()
        return val

    def unary():
        nonlocal i
        if peek() == "-":
            i += 1
            return -unary()
        if peek() == "+":
            i += 1
            return unary()
        return atom()

    def atom():
        nonlocal i
        if i >= len(toks):
            return None
        t = toks[i]
        if t.group("num"):
            i += 1
            s = t.group("num")
            return sympy.Rational(s) if "/" in s else sympy.Rational(s)
        if t.group("var"):
            if t.group("var") not in SYMBOLS:
                return None
            i += 1
            return SYMBOLS[t.group("var")]
        if t.group("op") == "(":
            i += 1
            val = expr()
            if peek() != ")":
                return None
            i += 1
            return val
        return None

    try:
        val = expr()
    except Exception:
        return None
    if val is None or i != len(toks):
        return None
    return val


# --------------------------------------------------------------------------
# 真值侧: 精确证明 + 精确采样
# --------------------------------------------------------------------------

def _cauchy_chain_identities() -> dict[str, bool]:
    """逐条精确验证 Nesbitt 的证明链（sympy 化简到 0, 不是采样）。

    1) a/(b+c) = a^2 / (a(b+c))            —— 恒等变形
    2) sum y_i = 2(ab+bc+ca)               —— Titu 分母之和
    3) (a+b+c)^2 - 3(ab+bc+ca) = 1/2 * [(a-b)^2+(b-c)^2+(c-a)^2] >= 0
    由 2、3: (a+b+c)^2/(2(ab+bc+ca)) >= 3/2; 由 1 与 Titu/Cauchy:
    sum a/(b+c) >= (a+b+c)^2/(2(ab+bc+ca))。整条链给出 >= 3/2, 等号仅当 a=b=c。
    """
    def others(v: str):
        return sum(SYMBOLS[w] for w in ("a", "b", "c") if w != v)

    lhs_as_titu = sum(SYMBOLS[v] ** 2 / (SYMBOLS[v] * others(v)) for v in ("a", "b", "c"))
    ysum = sympy.expand(sum(SYMBOLS[v] * others(v) for v in ("a", "b", "c")))
    gap = sympy.expand((_A + _B + _C) ** 2 - 3 * (_A * _B + _B * _C + _C * _A)
                       - sympy.Rational(1, 2) * ((_A - _B) ** 2 + (_B - _C) ** 2 + (_C - _A) ** 2))
    return {
        "rewrite_to_titu": sympy.simplify(LHS - lhs_as_titu) == 0,
        "y_sum_is_2ab2bc2ca": sympy.simplify(ysum - 2 * (_A * _B + _B * _C + _C * _A)) == 0,
        "sos_gap_identity": sympy.simplify(gap) == 0,
    }


def _lhs_exact(a: Fraction, b: Fraction, c: Fraction) -> Fraction:
    """精确有理数求值 —— 不引入浮点, 所以"违反"判定没有容差争议。"""
    return a / (b + c) + b / (c + a) + c / (a + b)


def _sampling_violations(n: int = 4000, seed: int = 20260908) -> list[tuple]:
    """随机正整数三元组 + 极端比值, 精确有理数找违反点。

    采样只是交叉验证: Nesbitt 是对的, 所以期望一个违反点都找不到。
    找不到 ≠ 证明成立（证明在 :func:`_cauchy_chain_identities`）,
    但找到 = 立刻判错。
    """
    s = seed & 0x7FFFFFFF

    def nxt() -> int:
        nonlocal s
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        return s

    pts = []
    for _ in range(n):
        pts.append((1 + nxt() % 1000, 1 + nxt() % 1000, 1 + nxt() % 1000))
    # 极端比值 / 边界（a 远小于 b,c 之类）
    pts += [(1, 1, 1), (1, 1, 10 ** 6), (10 ** 6, 1, 1), (1, 10 ** 6, 1),
            (1, 1, 2), (1, 2, 3), (2, 3, 5), (10 ** 6, 10 ** 6, 1)]
    bad = []
    for p in pts:
        fa, fb, fc = (Fraction(x) for x in p)
        if _lhs_exact(fa, fb, fc) < TARGET:
            bad.append(p)
    return bad


# --------------------------------------------------------------------------
# 反例分支: 精确代回
# --------------------------------------------------------------------------

def _extract_point(art: dict):
    if all(k in art for k in ("a", "b", "c")):
        return [art["a"], art["b"], art["c"]], "a/b/c"
    for key in ("values", "point", "vars", "triple"):
        if isinstance(art.get(key), (list, tuple)) and len(art[key]) == 3:
            return list(art[key]), key
    return None, "无 a/b/c"


def _to_fraction(v):
    """把 agent 给的数变成精确有理数。float 会先转成 Fraction(float)（二进制精确值），
    所以 1.5 这种输入仍然精确; 无法转成有理数的（无理数写法）返回 None。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        return Fraction(v)
    if isinstance(v, str):
        try:
            return Fraction(v.strip())
        except Exception:
            return None
    return None


def _verify_counterexample(art: dict, ck: Check) -> bool:
    """反例代回验证。返回 True 表示这个反例是真的（那才判 correct）。"""
    point, source = _extract_point(art)
    if not ck.expect("counterexample_wellformed", point is not None,
                     f"反例制品里取不到 (a,b,c)（{source}）"):
        return False
    fr = [_to_fraction(v) for v in point]
    if not ck.expect("counterexample_values_numeric", all(f is not None for f in fr),
                     f"反例的取值不是可精确表示的数: {point!r}"):
        return False
    a, b, c = fr
    ck.note("counterexample_values", [str(f) for f in (a, b, c)])
    # 题目说的是"正实数"。反例落在定义域外, 什么都没证伪 —— 判错。
    if not ck.expect("counterexample_in_domain", a > 0 and b > 0 and c > 0,
                     f"反例不在正实数定义域内: a={a}, b={b}, c={c}"):
        return False
    val = _lhs_exact(a, b, c)
    ck.note("counterexample_lhs_exact", str(val))
    ck.note("counterexample_target", str(TARGET))
    genuine = val < TARGET
    ck.note("counterexample_genuine", genuine)
    return genuine


# --------------------------------------------------------------------------
# 证明分支
# --------------------------------------------------------------------------

_NARROWER_CLASS = re.compile(r"integer|integers|natural|整数|有理数|rational|modulo", re.I)
#: 只认"数域"字样。注意不能拿 positive 当实数的标志 —— "positive integers" 里有
#: positive, 但它恰恰是最窄的那种写法。
_POSITIVE_REALS = re.compile(r"\breals?\b|实数|\bR\b|irrational", re.I)


#: proof_method 文本里若夹带 (a,b,c) 坐标, 就把它抠出来**精确代回**。
#: 抓的是"一边说成立、一边给个反例坐标"这种自相矛盾（多半是抄错了）。
_POINT_IN_TEXT = re.compile(
    r"\ba\s*=\s*([-+]?\d+(?:\.\d+)?|\d+/\d+)\s*[,\s]\s*"
    r"b\s*=\s*([-+]?\d+(?:\.\d+)?|\d+/\d+)\s*[,\s]\s*"
    r"c\s*=\s*([-+]?\d+(?:\.\d+)?|\d+/\d+)",
    re.I,
)
#: 常见的证明法关键字 —— 只作为诊断记录, 不当判据（文本证明没法逐字机检,
#: 拿关键字当判据等于考措辞）。但"完全没提任何方法"值得在 details 里说一声。
_METHOD_KEYS = ("cauchy", "titu", "andreescu", "sos", "sum of squares", "schur",
                "jensen", "am-gm", "rearrangement", "uvw", "pqr", "newton",
                "柯西", "施瓦茨", "苏", "均值", "平方和", "排序", "重排")


def _verify_proof(art: dict, ck: Check) -> None:
    verdict = str(art.get("verdict", "")).lower()
    ck.expect("claims_holds", verdict in ("holds", "hold", "true", "proved", "成立"),
              f"verdict={art.get('verdict')!r} 既没声称成立, 也没给反例")

    # 声称的适用范围必须覆盖正实数（只证整数/有理数 = 没证题面）
    domain = art.get("domain") or art.get("assumptions") or art.get("scope")
    if domain is not None:
        d = str(domain)
        narrowed = bool(_NARROWER_CLASS.search(d)) and not _POSITIVE_REALS.search(d)
        ck.expect("domain_covers_positive_reals", not narrowed,
                  f"声称的适用范围是 {d!r}, 比'任意正实数'窄, 不能算证明了原命题")

    # ---- 契约字段 proof_method（题库 expected.value 里写的就是它）----
    method = art.get("proof_method") or art.get("method") or art.get("technique")
    cert = art.get("certificate", art.get("cert"))
    # 判"证明依据字段认到了没有": certificate 或 proof_method 任一即可。
    # 之前只认 certificate, 于是按契约提交的答案被判"没给证书"。
    has_evidence = isinstance(method, (str, list, dict)) and bool(method) or isinstance(cert, dict)
    ck.expect("proof_evidence_read", bool(has_evidence),
              "题库契约字段是 verdict + proof_method; "
              "proof_method 与 certificate 至少要给一个")
    if isinstance(method, str) and method.strip():
        low = method.lower()
        hit = [k for k in _METHOD_KEYS if k in low]
        ck.note("proof_method_text_len", len(method))
        ck.note("proof_method_recognized", hit or None)
        # proof_method 里夹带坐标 -> 精确代回, 自相矛盾就判错
        m = _POINT_IN_TEXT.search(method)
        if m:
            point = [m.group(1), m.group(2), m.group(3)]
            ck.note("proof_method_contains_point", point)
            fr = [_to_fraction(v) for v in point]
            if all(f is not None for f in fr):
                a, b, c = fr
                val = _lhs_exact(a, b, c) if (b + c and c + a and a + b) else None
                ck.note("proof_method_point_lhs_exact", str(val))
                ck.expect("proof_method_point_not_refuting",
                          val is None or val >= TARGET,
                          f"proof_method 里给的 (a,b,c)=({a},{b},{c}) 代回后 LHS={val} < 3/2, "
                          "自己就是反例")
            else:
                ck.note("proof_method_point_numeric", False)

    # ---- certificate 结构（给了就精确验证, 这是比文本更硬的证据）----
    if isinstance(cert, dict):
        kind = str(cert.get("kind", "")).lower()
        ck.note("certificate_kind", kind or None)
        ck.expect("certificate_kind_known",
                  kind in ("sos", "cauchy_schwarz", "titu", "symbolic", "analytic", "other"),
                  f"未知的证书类型 {kind!r}")
        if kind == "sos":
            terms = cert.get("terms")
            ck.expect("sos_terms_present", isinstance(terms, list) and terms, "SOS 证书没有 terms")
            if isinstance(terms, list) and terms:
                parsed, bad = [], []
                for t in terms:
                    e = _parse_expr(t) if isinstance(t, str) else None
                    if e is None:
                        bad.append(t)
                    else:
                        parsed.append(e)
                ck.note("sos_unparsable_terms", bad)
                if not ck.expect("sos_terms_parsable", not bad,
                                 f"这些项解析不了: {bad}"):
                    return
                total = sympy.expand(sum(parsed))
                # cancel 前先 together: 有理式的差要先通分约分, 否则 simplify 不一定收到 0
                ident = sympy.simplify(sympy.cancel(sympy.together(total - (LHS - TARGET)))) == 0
                ck.expect("sos_identity_holds", ident,
                          "SOS 各项之和与 (左端 - 3/2) 不恒等")
                # 逐项非负: 数值采样（说明白: 采样不是非负性证明,
                # 真正的非负性由 verifier 自己的 Cauchy 恒等式链保证）
                neg = []
                s = 12345
                for t in parsed:
                    f = sympy.lambdify((_A, _B, _C), t, "mpmath")
                    for _ in range(60):
                        s = (1103515245 * s + 12345) % 2147483647
                        x = (s % 1000 + 1) / 1000.0
                        s = (1103515245 * s + 12345) % 2147483647
                        y = (s % 1000 + 1) / 1000.0
                        s = (1103515245 * s + 12345) % 2147483647
                        z = (s % 1000 + 1) / 1000.0
                        if float(f(x, y, z)) < -1e-9:
                            neg.append((str(t), x, y, z))
                            break
                ck.expect("sos_terms_nonnegative_sampled", not neg,
                          f"这些项在采样点取到负值: {neg[:2]}")


def _independent_truth(ck: Check) -> None:
    """verifier 自己的证明与采样 —— 与 agent 说什么无关。"""
    ident = _cauchy_chain_identities()
    ck.note("cauchy_chain_identities", ident)
    ck.expect("verifier_proves_inequality", all(ident.values()),
              "verifier 自己的 Cauchy 恒等式链没能精确成立")
    # 等号情形 a=b=c
    eq = _lhs_exact(Fraction(1), Fraction(1), Fraction(1))
    ck.note("equality_case", f"a=b=c=1 -> LHS={eq} (=3/2)")
    ck.expect("equality_at_a_b_c", eq == TARGET, "a=b=c 处应取等号 3/2")
    bad = _sampling_violations()
    ck.note("sampling_violations", bad)
    ck.expect("no_sampled_violation", not bad,
              f"随机/极端采样里找到了违反点: {bad[:3]}")
    # 极小量级边界（float 路径, 确认定义域端点也没问题）
    mpmath.mp.dps = 40
    worst = None
    for (x, y, z) in [(mpmath.mpf("1e-12"), 1, 1), (1, mpmath.mpf("1e-12"), 1),
                      (1, 1, mpmath.mpf("1e-12")), (1, mpmath.mpf("1e8"), mpmath.mpf("1e-8"))]:
        v = x / (y + z) + y / (z + x) + z / (x + y)
        worst = v if worst is None else min(worst, v)
    ck.note("boundary_min_lhs", float(worst))
    ck.expect("boundary_inequality_holds", worst >= mpmath.mpf(1.5) - mpmath.mpf("1e-12"),
              f"边界点 LHS={worst} < 3/2")


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

#: verdict 说"成立"的说法
_HOLDS_VERDICTS = ("holds", "hold", "true", "proved", "成立")
#: verdict 说"不成立"的说法
_FALSE_VERDICTS = ("false", "refuted", "disproved", "counterexample", "fails", "不成立")


def _point_from_method(art: dict) -> dict | None:
    """从 proof_method / method / evidence 文本里抠出 (a,b,c) 坐标。"""
    for key in ("proof_method", "method", "technique", "evidence", "counterexample"):
        raw = art.get(key)
        if isinstance(raw, str):
            m = _POINT_IN_TEXT.search(raw)
            if m:
                return {"a": m.group(1), "b": m.group(2), "c": m.group(3),
                        "_from": key}
        elif isinstance(raw, dict) and all(k in raw for k in ("a", "b", "c")):
            return dict(raw)
    return None


def verify(stdout: str, code: str, claim: str) -> dict:
    ck = Check()
    counter = None
    proof = None
    for tag in ("COUNTEREXAMPLE", "ANSWER", "CONSTRUCT"):
        art = parse_artifact(stdout, tag)
        if art is None:
            continue
        if tag == "COUNTEREXAMPLE" and counter is None:
            counter = art
        elif proof is None:
            proof = art
        ck.note("artifact_tags", ck.details.get("artifact_tags", []) + [tag])

    if counter is None and proof is None:
        ck.expect("artifact_present", False, "既没给证明制品也没给反例制品")
        return _result(ck, None, None)

    # ★ ANSWER 制品里说"不成立"且带坐标 -> 按反例走精确代回 ★
    # 题库契约字段只有 verdict + proof_method, 所以反例坐标很可能写在
    # proof_method 文本里。走同一条代回路径, agent 才能拿到"假证伪"这种
    # 有信息量的判错理由, 而不是含糊的 "claims_holds"。
    if counter is None and proof is not None:
        v = str(proof.get("verdict", "")).strip().lower()
        pt = _point_from_method(proof)
        if pt is not None and (v in _FALSE_VERDICTS or v not in _HOLDS_VERDICTS):
            ck.note("counterexample_in_proof_method", pt)
            counter = pt

    # 真值侧: 无论 agent 答什么, verifier 都先自己把命题证一遍
    _independent_truth(ck)

    if counter is not None:
        genuine = _verify_counterexample(counter, ck)
        if genuine:
            cert = "反例精确代回确实违反不等式"
            return _result(ck, counter, cert)
        # 假证伪: 反例代回去其实满足不等式
        ck.expect("counterexample_refuted", False,
                  "反例代回原式后并不违反不等式 —— 这是假证伪")
        return _result(ck, counter,
                       f"FAILED: 反例不成立 (LHS={ck.details.get('counterexample_lhs_exact')} "
                       f"vs 3/2), 命题本身成立")

    assert proof is not None
    _verify_proof(proof, ck)
    cert = ("Nesbitt 成立: verifier 用 Titu/Cauchy 恒等式链精确证明 + "
            "精确有理数采样无违反; agent 的证明依据字段"
            "(certificate 或题库契约字段 proof_method)已核")
    return _result(ck, proof, cert)


def _result(ck: Check, art: dict | None, cert: str | None) -> dict:
    final = None
    if art is not None:
        final = json.dumps({"verdict": art.get("verdict"), "a": art.get("a"),
                            "b": art.get("b"), "c": art.get("c"),
                            "proof_method": art.get("proof_method"),
                            "certificate_kind": (art.get("certificate") or {}).get("kind")
                            if isinstance(art.get("certificate"), dict) else None},
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

def _selftest() -> int:
    failures: list[str] = []

    def t(name: str, cond: bool) -> None:
        print(("[OK] " if cond else "[FAIL] ") + name)
        if not cond:
            failures.append(name)

    ident = _cauchy_chain_identities()
    t("verifier 自己的 Cauchy 恒等式链全部成立", all(ident.values()))
    print("    ", ident)
    t("精确采样无违反点", not _sampling_violations(800))

    # 真 SOS 证书:
    #   左端 - 3/2 = N / (2(a+b)(a+c)(b+c)),
    #   N = 2Σa³ - Σ_sym a²b = (Σa³ - 3abc) + Schur
    #   其中 Σa³-3abc = (a+b+c)(a²+b²+c²-ab-bc-ca) >= 0, Schur = Σ_cyc a(a-b)(a-c) >= 0。
    good_proof = {"verdict": "holds", "domain": "a,b,c > 0 reals",
                  "proof_method": "cauchy_schwarz",
                  "certificate": {"kind": "sos",
                                  "terms": ["(a**3+b**3+c**3-3*a*b*c)/(2*(a+b)*(a+c)*(b+c))",
                                            "(a*(a-b)*(a-c)+b*(b-c)*(b-a)"
                                            "+c*(c-a)*(c-b))/(2*(a+b)*(a+c)*(b+c))"]}}
    proof_out = "VERDICT: PROVED\nANSWER: " + json.dumps(good_proof, ensure_ascii=False)
    t("正例(成立 + 合法 SOS 证书) -> correct=True", verify(proof_out, "", "")["correct"])

    # ★ 题库契约字段: 只有 verdict + proof_method, 没有 certificate ★
    contract = {"verdict": "holds",
                "proof_method": ("Titu/Cauchy: a/(b+c) = a^2/(ab+ac), "
                                 "(a+b+c)^2 >= 3(ab+bc+ca), 故 sum >= 3/2, "
                                 "等号仅当 a=b=c")}
    r = verify("VERDICT: PROVED\nANSWER: " + json.dumps(contract, ensure_ascii=False), "", "")
    t("正例(题库契约 verdict+proof_method, 无 certificate) -> correct=True",
      r["correct"])
    t("  └ proof_evidence_read=True", r["details"].get("proof_evidence_read") is True)
    t("  └ proof_method_recognized 认出了 Titu/Cauchy",
      bool(r["details"].get("proof_method_recognized")))

    # proof_method 里夹带的坐标代回后真的违反 -> 说明它自相矛盾, 判错。
# ★ 注意 Nesbitt 在正实数上恒成立, 所以"违反"的坐标必然不在定义域内
#   (正的三元组代回去永远 >= 3/2) —— 这正是"声称成立却掏出违反点"的样子。
    bogus = {"verdict": "holds",
             "proof_method": "Titu/Cauchy 不等式, 反例 a=-1, b=-1, c=5 时 LHS < 3/2"}
    r = verify("VERDICT: PROVED\nANSWER: " + json.dumps(bogus, ensure_ascii=False), "", "")
    t("反例7(proof_method 里夹带违反坐标) -> correct=False", not r["correct"])
    t("  └ proof_method_point_lhs_exact 已算出", r["details"].get(
        "proof_method_point_lhs_exact") is not None)

    # proof_method 里提到等号点 a=1,b=1,c=1 -> 代回是 3/2, 不是反例, 照常判对
    eqpt = {"verdict": "holds",
            "proof_method": "Cauchy/Titu 链; 等号情形 a=1, b=1, c=1 时 LHS=3/2"}
    t("正例2(proof_method 提到等号坐标, 非反例) -> correct=True",
      verify("VERDICT: PROVED\nANSWER: " + json.dumps(eqpt, ensure_ascii=False),
             "", "")["correct"])

    # ANSWER 里说"不成立" + 坐标 -> 走反例代回（假证伪, 判错）
    fake_ref = {"verdict": "false", "proof_method": "取 a=2, b=3, c=5 即得 LHS=1.4"}
    r = verify("VERDICT: REFUTED\nANSWER: " + json.dumps(fake_ref, ensure_ascii=False),
               "", "")
    t("反例8(proof_method 里给坐标声称证伪) -> correct=False", not r["correct"])
    t("  └ 走了精确代回路径", r["details"].get("counterexample_lhs_exact") is not None)

    # 只给 verdict 不给任何证明依据 -> 判错（证明依据字段不能空）
    t("反例9(只有 verdict, 无 proof_method/certificate) -> correct=False",
      not verify("VERDICT: PROVED\nANSWER: " + json.dumps({"verdict": "holds"}),
                 "", "")["correct"])

    # 假证伪: a=b=c=1 是等号点, LHS=1.5 并不违反
    fake_ce = "VERDICT: REFUTED\nCOUNTEREXAMPLE: " + json.dumps(
        {"a": 1, "b": 1, "c": 1, "lhs": 1.5}, ensure_ascii=False)
    t("反例1(拿等号点当反例) -> correct=False", not verify(fake_ce, "", "")["correct"])

    # 假证伪: 随机大数三元组, 其实都满足不等式
    fake_ce2 = "VERDICT: REFUTED\nCOUNTEREXAMPLE: " + json.dumps(
        {"a": 2, "b": 3, "c": 5, "lhs": 1.4}, ensure_ascii=False)
    t("反例2(随便编的三元组) -> correct=False", not verify(fake_ce2, "", "")["correct"])

    # 反例落在定义域外(负数) —— 什么都没证伪
    oob = "VERDICT: REFUTED\nCOUNTEREXAMPLE: " + json.dumps(
        {"a": -1, "b": 1, "c": 1}, ensure_ascii=False)
    t("反例3(定义域外的负数) -> correct=False", not verify(oob, "", "")["correct"])

    # SOS 证书恒等式不成立(项是随便写的)
    bad_sos = {"verdict": "holds", "certificate": {"kind": "sos", "terms": ["a**2 + b**2 + c**2"]}}
    t("反例4(SOS 恒等式不成立) -> correct=False",
      not verify("VERDICT: PROVED\nANSWER: " + json.dumps(bad_sos, ensure_ascii=False), "", "")["correct"])

    # 只证了整数, 适用范围比题面窄
    narrow = {"verdict": "holds", "domain": "positive integers a,b,c",
              "certificate": {"kind": "cauchy_schwarz"}}
    t("反例5(适用范围窄于正实数) -> correct=False",
      not verify("VERDICT: PROVED\nANSWER: " + json.dumps(narrow, ensure_ascii=False), "", "")["correct"])

    # 什么都没交
    t("反例6(无制品) -> correct=False", not verify("VERDICT: INCONCLUSIVE\n", "", "")["correct"])

    if failures:
        print("[SELFTEST FAILED] " + ", ".join(failures))
        return 1
    print("[OK] 自测全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())