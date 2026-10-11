"""res-007-oeis-identify-prove 的独立 verifier。

题目: 给定整数序列前 30 项, 识别 OEIS 条目, 提取闭式或递推公式, 验证前 100 项一致。

★ 题库 verifier.params 写的是 `seq_file: data/res_07_seq.txt`(由 runner 预置)。
★ 2026-10-10: 这份文件**现在存在了**(res_07_seq.txt 与 res_07_sequence.txt 内容相同),
★ 所以走的是「读文件」那条路, 与题面内联给 agent 的数据同源。核对同 res-005:
★   python tests/tools/check_input_vs_reference.py
  ★ 行为: 若 `data/res_07_seq.txt` 存在 → 用它(这是真正发给 agent 的那 30 项),
    判据 = agent 的公式必须重现它给出的每一项;
    否则 → 用本文件硬编码的固定实例(Fibonacci)。
    选 Fibonacci 是因为它有确定的 OEIS 号(A000045)且是**常系数线性递推**,
    闭式 Binet 也可以由递推推出 —— 两种提法都能被独立复算。
    前 100 项以字面量硬编码, 定义(F(n+1)=F(n)+F(n-1), F(0)=0, F(1)=1)
    在模块里也写了一份; 两者不一致时以字面量为准并在 details 里报出来
    (自检会检查这两者一致)。

★ 判据 ★
  1. OEIS 号形状必须是 A + 6 位数字。**不联网查真伪** —— verifier 必须能
     离线复现; 联网查一个字符串只会在网络不可用时静默变成"无法判定"。
  2. agent 必须交出**可复算的公式制品**。题库 expected.value 声明的契约字段是
     ``formula``（一个字符串）, 所以两种写法都认:
       ★ 契约字段（首选）★ ``formula``: 一行公式文本, 例如
                  "F[n] = F[n-1] + F[n-2]"                -> 线性递推
                  "F[n] = 3*F[n-1] - 2*F[n-2]"             -> 线性递推
                  "a_n = a_{n-1} + a_{n-2}, a_0=0, a_1=1"  -> 递推 + 初值
                  "F[n] = n^2 + 3*n + 1"                   -> 多项式闭式
                  "F[n] = 2^n * (n+1)"                     -> 指数×多项式闭式
       结构化写法（等价保留, 不是必需）:
                  {"kind":"recurrence", "init":[...], "coeffs":[...]}
                  含义 a_n = coeffs[0]*a_{n-1} + ... + coeffs[k-1]*a_{n-k}
                  {"kind":"closed_form", "family":"poly", "coeffs":[...]}
                  含义 a_n = sum coeffs[k] * n^k
                  {"kind":"closed_form", "family":"exp_poly", "base":2,
                   "coeffs":[...]}  含义 a_n = base^n * sum coeffs[k] * n^k
     ★ 只支持这三族是刻意的 ★
     verifier 必须**自己实现**公式来复算,
     绝不能 eval agent 交来的任意表达式(那等于复用 agent 的代码,
     而且一个 eval 就给了 agent 逃逸口)。文本公式由 :func:`_parse_formula_text`
     转成这三族之一 —— 那是一个**受限的递归下降解析器**(只认 a_n / F[n] / F_n
     这一个自由符号), 不是 eval。闭式里出现这三族之外的写法,
     verifier 就判不了 —— 如实判错, 而不是"看着像就放过"。
     ★ 放宽的是解析, 不是判定 ★
     判据仍然是"用 agent 给的公式自己算出前 100 项, 逐项与固定实例对账";
     文本通道和结构化通道走的是同一套 generate_* 函数, 逐项比对一模一样。
  3. agent 报的前若干项(terms, 可省)若给了, 必须与固定实例逐项相同。
     为什么单独查: agent 只要"背下 30 项"就能声称 verified_100_terms,
     不看公式的话这题就是白送。
  4. 公式生成的 100 项必须与固定实例一致 —— 这是本题真正的判据:
     能重现前 100 项的公式, 不可能只是"看起来像"的 OEIS 检索结果
     (检索到的近似公式/错公式通常在前 20~40 项就崩)。
"""
from __future__ import annotations

import json
import os
import re
import sys

try:
    from .common import Check, parse_artifact, parse_verdict
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import Check, parse_artifact, parse_verdict

#: 要比对的项数(题目 params: check_terms = 100)
N_CHECK = 100

#: 官方序列文件(题库 params 指向的路径); 存在就用它
_SEQ_REL = os.path.join("data", "res_07_seq.txt")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOTS = [
    os.path.abspath(os.path.join(_HERE, "..", "..", "..")),
    os.getcwd(),
]

#: ★ 内置固定实例: Fibonacci 前 100 项(A000045)★
#: 硬编码字面量是刻意的 —— 判据必须钉死在一个具体的数表上,
#: 不能依赖"现场再算一遍"(现场重算的实现若有 bug, 判据就跟着错)。
FIXED_OEIS_ID = "A000045"
FIXED_SEQ: list[int] = [
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584,
    4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229,
    832040, 1346269, 2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169,
    63245986, 102334155, 165580141, 267914296, 433494437, 701408733, 1134903170, 1836311903, 2971215073, 4807526976, 7778742049, 12586269025,
    20365011074, 32951280099, 53316291173, 86267571272, 139583862445, 225851433717,
    365435296162, 591286729879, 956722026041, 1548008755920, 2504730781961,
    4052739537881, 6557470319842, 10610209857723, 17167680177565, 27777890035288,
    44945570212853, 72723460248141, 117669030460994, 190392490709135,
    308061521170129, 498454011879264, 806515533049393, 1304969544928657,
    2111485077978050, 3416454622906707, 5527939700884757, 8944394323791464,
    14472334024676221, 23416728348467685, 37889062373143906, 61305790721611591,
    99194853094755497, 160500643816367088, 259695496911122585, 420196140727489673,
    679891637638612258, 1100087778366101931, 1779979416004714189,
    2880067194370816120, 4660046610375530309, 7540113804746346429,
    12200160415121876738, 19740274219868223167, 31940434634990099905,
    51680708854858323072, 83621143489848422977, 135301852344706746049,
    218922995834555169026,
]
FIXED_LEN = 30          # 题面只给前 30 项

_OEIS_ID_RE = re.compile(r"^A\d{6}$")

_SEQ_CACHE: dict[str, object] = {}


# --------------------------------------------------------------------------
# 序列来源
# --------------------------------------------------------------------------
#: 纯数字行: 只由数字/逗号/减号/空白组成。题目文件带散文说明, 不按行过滤的话
#: "下面给出某个整数序列的前 **30** 项" 里的 30 会被当成序列的第一项。
_NUM_LINE = re.compile(r"^[0-9,\-\s]+$")


def _parse_seq_text(text: str) -> list[int] | None:
    """容忍三种文件写法: 空格/逗号分隔的整数, 或 JSON 数组。

    ★ 只取"纯数字行"里的整数 ★
    题目文件是"散文说明 + 数字列表", 直接对全文 split 会把正文里的数字
    ("前 30 项"、"100 项")一起吞进序列 —— 那样判据的第一项就错了。
    """
    text = text.strip()
    if not text:
        return None
    if text[0] == "[":
        try:
            v = json.loads(text)
        except Exception:
            v = None
        if isinstance(v, list):
            try:
                out = [int(x) for x in v]
            except (TypeError, ValueError):
                return None
            return out or None
    out: list[int] = []
    for line in text.splitlines():
        if not _NUM_LINE.match(line):
            continue
        for tok in re.split(r"[^0-9-]+", line):
            if not tok or tok == "-":
                continue
            try:
                out.append(int(tok))
            except ValueError:
                continue
    return out or None


def target_sequence() -> tuple[list[int], str]:
    """
    返回 (固定序列, 来源说明)。官方文件优先, 否则用内置 Fibonacci。
    官方文件里给出的项数可能少于 N_CHECK —— 那时只校验它给出的项数,
    并在 details 里写明实际校验了多少项(不假装验了 100 项)。
    """
    if "seq" in _SEQ_CACHE:
        seq, src = _SEQ_CACHE["seq"]            # type: ignore[misc]
        return seq, src
    for root in _ROOTS:
        p = os.path.join(root, _SEQ_REL)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    seq = _parse_seq_text(fh.read())
            except OSError:
                continue
            if seq:
                _SEQ_CACHE["seq"] = (seq, _SEQ_REL)
                return seq, _SEQ_REL
    seq = list(FIXED_SEQ)
    _SEQ_CACHE["seq"] = (seq, f"内置固定实例 Fibonacci(A000045), 前 {len(seq)} 项")
    return seq, f"内置固定实例 Fibonacci(A000045), 前 {len(seq)} 项"


# --------------------------------------------------------------------------
# verifier 自己实现公式 -> 生成前 N 项
# --------------------------------------------------------------------------
def _num(x) -> float | int | None:
    """把 JSON 值转成数值; 非数值/布尔返回 None。"""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        try:
            f = float(x.strip())
        except ValueError:
            return None
        return int(f) if f.is_integer() else f
    return None


def generate_recurrence(init, coeffs, count: int) -> list[float] | None:
    """
    自己实现常系数线性递推: a_n = coeffs[0]*a_{n-1} + ... + coeffs[k-1]*a_{n-k}。
    算不动(系数个数与初值个数不等、项数不足)返回 None。
    """
    if not isinstance(init, (list, tuple)) or not isinstance(coeffs, (list, tuple)):
        return None
    if len(init) != len(coeffs) or not init:
        return None
    a = [_num(x) for x in init]
    c = [_num(x) for x in coeffs]
    if any(v is None for v in a) or any(v is None for v in c):
        return None
    k = len(c)
    seq: list[float] = list(a)                       # type: ignore[arg-type]
    while len(seq) < count:
        seq.append(sum(c[j] * seq[-1 - j] for j in range(k)))
    return seq


def generate_poly(coeffs, count: int) -> list[float] | None:
    """自己实现多项式闭式 a_n = sum coeffs[k] * n^k(Horner 展开)。"""
    if not isinstance(coeffs, (list, tuple)) or not coeffs:
        return None
    c = [_num(x) for x in coeffs]
    if any(v is None for v in c):
        return None
    c = c[::-1]                                       # 高次在前, 便于 Horner
    return [sum(ci * (n ** j) for j, ci in enumerate(c)) for n in range(count)]


def generate_exp_poly(coeffs, base, count: int) -> list[float] | None:
    """自己实现指数×多项式闭式 a_n = base^n * sum coeffs[k] * n^k。"""
    poly = generate_poly(coeffs, count)
    b = _num(base)
    if poly is None or b is None or b == 0:
        return None
    return [poly[n] * (b ** n) for n in range(count)]


def generate_from_artifact(art: dict, count: int) -> tuple[list[float] | None, str]:
    """按制品里的公式算出前 count 项。返回 (序列, 说明)。

    顺序: 题库契约字段 ``formula``（文本）优先, 其次结构化写法。
    """
    text = art.get("formula") or art.get("expression") or art.get("eq")
    if isinstance(text, str) and text.strip():
        return _parse_formula_text(text, count, art)
    kind = str(art.get("kind") or art.get("type") or "").strip().lower()
    if kind in ("recurrence", "linear_recurrence", "linrec"):
        rec = art.get("recurrence") if isinstance(art.get("recurrence"), dict) else art
        return generate_recurrence(rec.get("init") or rec.get("initial"),
                                   rec.get("coeffs") or rec.get("coefficients"),
                                   count), "recurrence"
    if kind in ("closed_form", "closed", "formula"):
        cf = art.get("closed_form") if isinstance(art.get("closed_form"), dict) else art
        family = str(cf.get("family") or cf.get("basis") or "poly").strip().lower()
        coeffs = cf.get("coeffs") or cf.get("coefficients")
        if family in ("poly", "polynomial"):
            return generate_poly(coeffs, count), "closed_form/poly"
        if family in ("exp_poly", "exponential", "exp"):
            return generate_exp_poly(coeffs, cf.get("base"), count), "closed_form/exp_poly"
        return None, f"closed_form/{family}(verifier 不支持这一族, 无法独立复算)"
    return None, f"kind={kind!r}(verifier 无法独立复算)"


# --------------------------------------------------------------------------
# 文本公式 -> 三族之一（受限解析器, 不是 eval）
# --------------------------------------------------------------------------
#: 公式左边允许的写法: F[n] / a_n / a(n) / A_n。捕获符号名。
_LHS = re.compile(r"^\s*([A-Za-z])\s*(?:\[\s*n\s*\]|_\s*\{?\s*n\s*\}?|\(\s*n\s*\))\s*=\s*")
#: 右边的项: <符号>[<下标>]。下标只允许 n 的整数线性式 k*n+k0。
#: ★ 花括号与方括号必须**成对**匹配 ★ —— 写成 `\{?...[^{}]*\}?` 时,
#:   非贪婪的下标会在 "a_{n-1}" 的 "n" 处就收工(右花括号是可选的),
#:   于是 n-1 被读成 n, 延迟 0, 整条递推被判"引用了自己"。
_TERM = re.compile(
    r"(?P<sym>[A-Za-z])\s*(?:\[\s*(?P<idx>[^\[\]]*?)\s*\]"
    r"|_\s*\{\s*(?P<sub>[^{}]*?)\s*\}"
    r"|_\s*(?P<sub2>[A-Za-z0-9]+)"
    r"|\(\s*(?P<sub3>[^()]*?)\s*\))"
)
#: n 的线性下标: 2*n-1 / n+1 / n / 3
_LIN = re.compile(r"^\s*(?:(?P<k1>[-+]?\d+)\s*\*\s*)?n(?:\s*(?P<k0>[-+]\s*\d+))?\s*$")
#: 初值: a_0 = 0 / F[0]=0 / a(1) = 1
_INIT = re.compile(
    r"([A-Za-z])\s*(?:\[\s*(\d+)\s*\]|_\s*\{?\s*(\d+)\s*\}?|\(\s*(\d+)\s*\))\s*=\s*[-+]?\d+"
)


def _split_signed(s: str) -> list[str]:
    """按顶层 + / - 拆项。**下标里的符号不能拆** ——
    ``F[n-1]+F[n-2]`` 里有三个 '-' 两个 '+', 朴素的正则会把 ``n-1`` 拆成两项;
    ``a_{n-1}`` 的花括号同样要计入深度。"""
    out, buf, depth = [], [], 0
    for i, ch in enumerate(s):
        if ch in "[({":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if depth == 0 and ch in "+-" and i > 0:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf))
    return [p for p in out if p]


def _lin_index(text: str) -> tuple[int, int] | None:
    """``n`` → (1,0); ``2*n-1`` → (2,-1); 认不了返回 None。"""
    m = _LIN.match(text)
    if not m:
        return None
    k1 = int(m.group("k1")) if m.group("k1") else 1
    k0 = 0
    if m.group("k0"):
        k0 = int(m.group("k0").replace(" ", ""))
    return (k1, k0)


def _sum_terms(rhs: str) -> tuple[float, float, list[tuple[float, int]]] | None:
    """把 ``3*F[n-1] - 2*F[n-2]`` 拆成 (常数项, n 的系数, [(延迟, 权重)])。

    只处理**延迟对 n 的线性依赖**：延迟 d = k1*n+k0 归一化成 (k1, k0),
    因为本模块的生成器只支持 a_n = c1*a_{n-1} + ... + c_k*a_{n-k}
    （k1 必须为 1, 否则不是常数系数线性递推, 判不了）。
    """
    s = rhs.replace(" ", "").replace("−", "-").replace("–", "-")
    if not s or s.startswith("="):
        s = s.lstrip("=")
    # 按 + / - 拆成带符号的项
    parts = _split_signed(s)
    const = 0.0
    ncoef = 0.0
    lags: dict[int, float] = {}
    for p in parts:
        if not p:
            continue
        sign = -1.0 if p.startswith("-") else 1.0
        body = p[1:] if p[0] in "+-" else p
        if not body:
            continue
        if "*" in body:                       # 系数 * 项
            cf, _, term = body.partition("*")
            cf = cf.replace("(", "").replace(")", "")
            coef = _num(cf)
            if coef is None:
                return None
            coef *= sign
        else:
            term, coef = body, sign
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", term):            # 纯常数
            const += coef * float(term)
            continue
        if term in ("n",) or _LIN.fullmatch(term):            # 纯 n 的线性式
            lin = _lin_index(term)
            const += coef * lin[1]
            ncoef += coef * lin[0]
            continue
        m = _TERM.search(term)
        if m is None:
            return None
        idx = (m.group("idx") or m.group("sub")
               or m.group("sub2") or m.group("sub3") or "")
        idx = idx.replace("{", "").replace("}", "").replace(" ", "")
        lin = _lin_index(idx)
        if lin is None:
            return None
        k1, k0 = lin
        if k1 != 1:
            return None                      # 不是常系数线性递推
        d = -k0
        if d < 1:
            return None                      # 引用自己或未来项, 本生成器算不了
        lags[d] = lags.get(d, 0.0) + coef
    if not lags:
        return (const, ncoef, [])
    k = max(lags)
    # 缺失的中间延迟补 0
    return (const, ncoef, [(d, lags.get(d, 0.0)) for d in range(1, k + 1)])


def _poly_from_rhs(rhs: str) -> list[float] | None:
    """``n^2 + 3*n + 1`` / ``2*n^2+1`` -> 升幂系数 [1,3,2]（无符号乘积项）。"""
    s = rhs.replace(" ", "").replace("{", "").replace("}", "").replace("−", "-")
    coeffs: dict[int, float] = {0: 0.0}
    for p in _split_signed(s):
        if not p:
            continue
        sign = -1.0 if p.startswith("-") else 1.0
        body = p[1:] if p[0] in "+-" else p
        if not body:
            continue
        deg = 1
        coef = 1.0
        # 单项要么是 <数>*n^k / n^k, 要么是**纯常数** ——
        # "n^2 + 3*n + 1" 的末尾 "+1" 没有 n, 漏掉它整条闭式就解析不了。
        m = re.fullmatch(r"(?:(\d+(?:\.\d+)?|\.\d+)\*)?(n(?:\^(\d+))?)?", body)
        if m and m.group(0):
            if m.group(1) is not None:
                coef *= float(m.group(1))
            inner = m.group(2)
            deg = (int(m.group(3)) if m.group(3) else 1) if inner else 0
        elif re.fullmatch(r"\d+(?:\.\d+)?|\.\d+", body):
            deg, coef = 0, float(body)
        else:
            return None
        coeffs[deg] = coeffs.get(deg, 0.0) + sign * coef
    out = [coeffs.get(i, 0.0) for i in range(max(coeffs) + 1)]
    return out


def _exp_poly_from_rhs(rhs: str) -> tuple[float, list[float]] | None:
    """``2^n * (n+1)`` -> (base, 升幂系数)。只认 base^n * 多项式 这一种。"""
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*\^\s*n", rhs.replace(" ", ""))
    if not m:
        return None
    base = float(m.group(1))
    rest = rhs.replace(" ", "")[m.end():].lstrip("*").strip("()")
    coeffs = _poly_from_rhs(rest) if rest else [1.0]
    return (base, coeffs) if coeffs else None


def _parse_formula_text(text: str, count: int, art: dict | None = None
                        ) -> tuple[list[float] | None, str]:
    """题库契约字段 ``formula``（字符串）→ 自己复算前 count 项。

    支持的三种写法（与结构化通道等价）:
      * 常系数线性递推  ``F[n] = 3*F[n-1] - 2*F[n-2]``
      * 多项式闭式      ``F[n] = n^2 + 3*n + 1``
      * 指数×多项式闭式  ``F[n] = 2^n * (n+1)``
    初值可以跟在后面（``a_0=0, a_1=1``）, 也可以由制品的 init/terms 字段给出;
    都没有时, **不能**猜 —— 返回 None 如实判错（否则等于 verifier 替 agent
    选初值, 那就不是复算 agent 的公式了）。
    """
    raw = text.strip()
    if not raw:
        return None, "formula 是空串"
    body = raw.replace("−", "-").replace("–", "-").replace("\\", "")
    m = _LHS.match(body)
    if not m:
        return None, f"formula 不认得的左边写法: {raw[:60]!r}"
    sym = m.group(1)
    rhs = body[m.end():].strip()

    # 初值: 公式尾巴上的 a_0=0, a_1=1 ...
    inits: dict[int, float] = {}
    parts = re.split(r"[,;]", rhs)
    tail_terms = [p for p in parts if _INIT.fullmatch(p.strip())]
    for p in tail_terms:
        mm = _INIT.fullmatch(p.strip())
        assert mm is not None
        if mm.group(1) != sym:
            continue
        idx = mm.group(2) or mm.group(3) or mm.group(4)
        val = _num(p.strip().split("=")[-1])
        if val is not None:
            inits[int(idx)] = val
    rhs_main = ",".join(p for p in parts if p not in tail_terms).strip() or rhs
    rhs_main = rhs_main.rstrip(".;")
    if not rhs_main:
        return None, f"formula 右边是空的: {raw[:60]!r}"

    # ---- 递推 ----
    if _TERM.search(rhs_main.replace(" ", "")) or rhs_main.replace(" ", "").count(sym):
        parsed = _sum_terms(rhs_main)
        if parsed is not None:
            const, ncoef, lags = parsed
            if not lags:
                return None, "formula 右边没引用左边的序列符号, 判不了"
            if const != 0 or ncoef != 0:
                return None, ("递推右边含常数项或 n 的非常数依赖, "
                              "verifier 只复算齐次常系数线性递推")
            seq = _gen_from_lags(lags, inits, count)
            if seq is None:
                # 初值可以由制品的 init/initial/terms 字段补上（题库契约里没有
                # 初值字段, 但 agent 给了就不能当没看见）。仍没有 → 如实判错:
                # verifier 不替 agent 猜初值, 那样就不是复算它的公式了。
                extra = _inits_from_artifact(art, len(lags))
                if extra:
                    seq = _gen_from_lags(lags, extra, count)
                    inits = extra
                else:
                    return None, (f"递推初值不够(公式给了 {sorted(inits)}, "
                                  f"需要 {list(range(len(lags)))}, 制品里也没有 init/terms)")
            if seq is None:
                return None, f"递推初值不够(给了 {sorted(inits)}, 阶数 {len(lags)})"
            return seq, f"formula 文本/递推(阶数 {len(lags)}, 初值 {sorted(inits)})"

    # ---- 闭式: 指数×多项式 ----
    ep = _exp_poly_from_rhs(rhs_main)
    if ep is not None:
        base, coeffs = ep
        if base <= 0:
            return None, "指数底数必须为正"
        return generate_exp_poly(coeffs, base, count), f"formula 文本/closed_form(exp_poly, base={base:g})"

    # ---- 闭式: 多项式 ----
    coeffs = _poly_from_rhs(rhs_main)
    if coeffs is not None and any(c != 0 for c in coeffs):
        return generate_poly(coeffs, count), "formula 文本/closed_form(poly)"
    return None, f"formula 文本判不了: {raw[:60]!r}"


def _inits_from_artifact(art: dict | None, k: int) -> dict[int, float]:
    """从制品的 init / initial / terms 字段取前 k 项当初值。取不到返回 {}。"""
    if not isinstance(art, dict):
        return {}
    raw = None
    for key in ("init", "initial", "initial_values", "terms", "sequence", "first_30"):
        v = art.get(key)
        if isinstance(v, (list, tuple)) and v:
            raw = v
            break
    if raw is None:
        return {}
    out: dict[int, float] = {}
    for i in range(k):
        if i >= len(raw):
            return {}
        v = _num(raw[i])
        if v is None:
            return {}
        out[i] = float(v)
    return out


def _gen_from_lags(lags, inits, count):
    """由 [(延迟, 权重)] + 初值字典自己算前 count 项（与 generate_recurrence 同源）。"""
    k = len(lags)
    if not inits:
        return None
    need = [i for i in range(k) if i not in inits]
    if need:
        return None
    a = [_int_if_integral(inits[i]) for i in range(k)]
    c = [_int_if_integral(w) for _, w in lags]
    seq = list(a)
    while len(seq) < count:
        seq.append(sum(c[j] * seq[-1 - j] for j in range(k)))
    return seq


def _int_if_integral(v):
    """整数就保持整数 —— 否则生成的项是 float, 会掉进 1e-9 相对容差那条路,
    而整数项之间本来可以**精确相等**（第 88 项起超过 2^53, float 会丢精度,
    那是 :mod:`generate_recurrence` 早就注明过的坑）。"""
    f = float(v)
    return int(f) if f.is_integer() else f


# --------------------------------------------------------------------------
def verify(stdout: str, code: str, claim: str) -> dict:
    chk = Check()
    chk.note("verdict", parse_verdict(stdout))

    seq, src = target_sequence()
    n_check = min(N_CHECK, len(seq))
    chk.note("sequence_source", src)
    chk.note("sequence_length", len(seq))
    chk.note("terms_checked", n_check)
    chk.note("reference_first_10", seq[:10])

    art = parse_artifact(stdout, "ANSWER") or parse_artifact(stdout, "CONSTRUCT")
    if not chk.expect("artifact_present", art is not None,
                      "未提交 ANSWWER 制品(需要 oeis_id + formula)"):
        return _finish(chk, None, {})
    assert art is not None

    # ---- 1. OEIS 号形状(不联网) ----
    oid = art.get("oeis_id") or art.get("oeis") or art.get("id")
    oid = str(oid).strip().upper() if oid is not None else ""
    chk.expect("oeis_id_format", bool(_OEIS_ID_RE.match(oid)),
               f"oeis_id 形状应为 A+6 位数字, 实际 {oid!r}")
    chk.note("oeis_id", oid)
    if not src.startswith("内置"):
        # 官方文件在场时 OEIS 号的**真伪**无法离线判定, 只记在 details 里,
        # 不作为硬判据(否则判据就依赖网络)。
        chk.note("oeis_id_truth", "官方序列文件在场, 真伪未联网校验(见 docstring)")

    # ---- 2. 公式制品 ----
    # ★ 分成两问 ★
    #   formula_read       题库契约字段 formula 认不认（字段名层面的事）
    #   formula_recomputable 这个公式 verifier 自己算不算得出来（能力层面的事）
    # 之前只有一个 formula_present, 把这两件事混在一起: 字段名不对时它报
    # "没给公式", 看起来像 agent 没做数学。
    raw_formula = art.get("formula") or art.get("expression") or art.get("eq")
    structured = bool(str(art.get("kind") or art.get("type") or "").strip())
    chk.expect("formula_read",
               (isinstance(raw_formula, str) and raw_formula.strip() != "") or structured,
               "制品里没有题库契约字段 formula(公式字符串), 也没有结构化的 kind")

    gen, how = generate_from_artifact(art, n_check)
    chk.expect("formula_recomputable", gen is not None,
               f"没有可独立复算的公式制品({how}); "
               "题库契约字段是 formula(字符串), 例如 "
               '"F[n] = F[n-1] + F[n-2]" 或 "F[n] = 3*F[n-1] - 2*F[n-2]"; '
               "结构化的 kind=recurrence(init,coeffs) / kind=closed_form 也仍接受")
    if gen is None:
        return _finish(chk, art, {"oeis_id": oid})
    chk.note("formula_kind", how)

    # ---- 3. 逐项比对 ----
    # ★ 整数项必须**精确相等**, 不用容差 ★ Fibonacci 第 88 项起超过 2^53,
    #   float 表示会丢精度; 用相对容差又会让"差了 1 的大项"被放过。
    first_bad = None
    for i in range(n_check):
        g, s = gen[i], seq[i]
        if isinstance(g, int) and isinstance(s, int):
            same = g == s
        else:
            same = abs(float(g) - float(s)) <= 1e-9 * max(1.0, abs(float(s)))
        if not same:
            first_bad = (i, g, s)
            break
    chk.expect("formula_reproduces_sequence", first_bad is None,
               "" if first_bad is None else
               f"第 {first_bad[0]} 项起不符: 公式给 {first_bad[1]}, "
               f"固定实例给 {first_bad[2]}")

    # ---- 4. agent 自报的 terms(若给了)必须逐项相同 ----
    # ★ 按 agent 实际给的**前缀长度**比 ★
    #   只给 2 项初值是很常见的写法(公式 F[n]=F[n-1]+F[n-2] + init)。
    #   拿 2 项去和 30 项的固定实例整体 ==, 一律判错 —— 那是长度比较,
    #   不是数学比较。逐项比它真正给的那几项才是本判据的原意。
    terms = art.get("terms") or art.get("sequence") or art.get("first_30")
    if terms is not None:
        if not isinstance(terms, (list, tuple)):
            chk.expect("agent_terms_valid", False, "terms 不是列表")
        else:
            k = min(len(terms), n_check)
            bad = None
            for i in range(k):
                g, s = _num(terms[i]), seq[i]
                if g is None:                       # 不是数值, 判不了
                    bad = i
                    break
                same = (g == s if isinstance(g, int) and isinstance(s, int)
                        else abs(float(g) - float(s)) <= 1e-9 * max(1.0, abs(float(s))))
                if not same:
                    bad = i
                    break
            chk.note("agent_terms_length", k)
            chk.expect("agent_terms_match", bad is None,
                       "" if bad is None else
                       f"agent 自报的第 {bad} 项 {terms[bad]} 与固定实例 {seq[bad]} 不一致")

    extra = {"oeis_id": oid, "formula_kind": how, "terms_checked": n_check,
             "sequence_source": src}
    if first_bad is not None:
        extra["first_mismatch"] = {"index": first_bad[0],
                                   "generated": first_bad[1], "expected": first_bad[2]}
    return _finish(chk, art, extra)


def _finish(chk: Check, art: dict | None, extra: dict) -> dict:
    correct = chk.correct
    if correct:
        cert = (f"OEIS {extra['oeis_id']}, 公式({extra['formula_kind']})独立复算后"
                f"重现固定实例前 {extra['terms_checked']} 项"
                f"(来源 {extra['sequence_source']})")
    else:
        cert = chk.reason()
    final = json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else None
    return {"correct": correct, "certificate": cert, "final_answer": final,
            "details": dict(chk.details)}


if __name__ == "__main__":
    # 内置实例的自一致性: 硬编码字面量必须等于按定义现算的值
    # ★ 用 int 比, 不用 float 比 ★ Fibonacci 第 88 项起就超过 2^53,
    #   转 float 会丢精度 -> 一个正确的字面量表会被自己判成"错"。
    computed = generate_recurrence([0, 1], [1, 1], len(FIXED_SEQ))
    assert computed == [int(v) for v in FIXED_SEQ], "内置实例字面量与定义不一致"
    print(f"[INFO] 内置固定实例: {FIXED_OEIS_ID} Fibonacci, {len(FIXED_SEQ)} 项, "
          f"字面量与现算一致")
    print(f"[INFO] 校验目标: 前 {min(N_CHECK, len(FIXED_SEQ))} 项, "
          f"前 10 项 = {FIXED_SEQ[:10]}")

    good_rec = {"oeis_id": "A000045", "kind": "recurrence", "init": [0, 1],
                "coeffs": [1, 1]}
    r = verify("VERDICT: PROVED\nANSWER: " + json.dumps(good_rec) + "\n", "", "")
    assert r["correct"], f"正例(线性递推)应通过: {r}"
    print("[OK] 正例(结构化线性递推 F=F+F) ->", r["certificate"])

    # ★ 题库契约字段: formula 是一个字符串 ★
    for txt in ["F[n] = F[n-1] + F[n-2], F[0]=0, F[1]=1",
                "a_n = a_{n-1} + a_{n-2}, a_0=0, a_1=1",
                "F[n] = 1*F[n-1] + 1*F[n-2], F[0]=0, F[1]=1",
                "F(n) = F(n-1) + F(n-2), F(0)=0, F(1)=1"]:
        a = {"oeis_id": "A000045", "formula": txt, "verified_100_terms": True}
        r = verify("VERDICT: PROVED\nANSWER: " + json.dumps(a, ensure_ascii=False) + "\n", "", "")
        assert r["correct"], f"正例(formula 文本)应通过: {txt} -> {r}"
    print("[OK] 正例(formula 文本, Fibonacci 递推三种写法) ->", r["certificate"])

    # 初值在 terms 字段里给, 公式里不带 —— 也要能复算
    a = {"oeis_id": "A000045", "formula": "F[n] = F[n-1] + F[n-2]", "terms": [0, 1],
         "verified_100_terms": True}
    r = verify("ANSWER: " + json.dumps(a, ensure_ascii=False) + "\n", "", "")
    assert r["correct"], f"正例(公式 + terms 初值)应通过: {r}"
    print("[OK] 正例(formula 文本 + terms 初值) ->", r["certificate"])

    # 文本通道也必须逐项对账: 公式是 Pell 数列, 前 6 项就分叉
    a = {"oeis_id": "A000045", "formula": "F[n] = 2*F[n-1] + F[n-2], F[0]=0, F[1]=1"}
    r = verify("ANSWER: " + json.dumps(a, ensure_ascii=False) + "\n", "", "")
    assert not r["correct"] and not r["details"]["formula_reproduces_sequence"], r
    print("[OK] 反例E(formula 文本算得出但不是该序列) ->", r["certificate"])

    # 文本里的系数写错(3*F[n-1]-2*F[n-2] 不是 Fibonacci) —— 复算照样抓到
    a = {"oeis_id": "A000045", "formula": "F[n] = 3*F[n-1] - 2*F[n-2], F[0]=0, F[1]=1"}
    r = verify("ANSWER: " + json.dumps(a, ensure_ascii=False) + "\n", "", "")
    assert not r["correct"] and not r["details"]["formula_reproduces_sequence"], r
    print("[OK] 反例E2(formula 文本, 系数写错) ->", r["certificate"])

    # 文本公式没给初值, 制品里也没有 -> 不能替它猜初值, 如实判错
    a = {"oeis_id": "A000045", "formula": "F[n] = F[n-1] + F[n-2]"}
    r = verify("ANSWER: " + json.dumps(a, ensure_ascii=False) + "\n", "", "")
    assert not r["correct"] and not r["details"]["formula_recomputable"], r
    print("[OK] 反例F(formula 文本缺初值) ->", r["certificate"])

    # exp_poly 族本身合法, 但内容不是 Fibonacci(2^n), 必须在逐项比对处判错
    wrong_cf = {"oeis_id": "A000045", "kind": "closed_form", "family": "exp_poly",
                "base": 2, "coeffs": [1]}
    r = verify("ANSWER: " + json.dumps(wrong_cf) + "\n", "", "")
    assert not r["correct"] and not r["details"]["formula_reproduces_sequence"], r
    print("[OK] 反例A(合法闭式族但内容不是该序列) ->", r["certificate"])

    bad_rec = {"oeis_id": "A000045", "kind": "recurrence", "init": [0, 1],
               "coeffs": [1, 2]}          # Pell 数列, 前 6 项就开始分叉
    r = verify("ANSWER: " + json.dumps(bad_rec) + "\n", "", "")
    assert not r["correct"] and not r["details"]["formula_reproduces_sequence"], r
    print("[OK] 反例B(递推能算但不是该序列) ->", r["certificate"])

    fake = {"oeis_id": "12345", "kind": "recurrence", "init": [0, 1], "coeffs": [1, 1]}
    r = verify("ANSWER: " + json.dumps(fake) + "\n", "", "")
    assert not r["correct"] and not r["details"]["oeis_id_format"], r
    print("[OK] 反例C(OEIS 号形状不对) ->", r["certificate"])

    # 只背 30 项、不给公式 —— 必须判错(公式才是本题的判据)
    memo = {"oeis_id": "A000045", "kind": "recurrence"}
    r = verify("ANSWER: " + json.dumps(memo) + "\n", "", "")
    assert not r["correct"] and not r["details"]["formula_recomputable"], r
    print("[OK] 反例D(只报 OEIS 号不给公式) ->", r["certificate"])

    # 闭式文本也能自己复算（多项式 / 指数×多项式）
    assert _parse_formula_text("F[n] = n^2 + 3*n + 1", 4)[0] == [1.0, 5.0, 11.0, 19.0]
    assert _parse_formula_text("F[n] = 2^n*(n+1)", 4)[0] == [1.0, 4.0, 12.0, 32.0]
    a = {"oeis_id": "A000045", "formula": "F[n] = n^2 + 3*n + 1",
         "verified_100_terms": True}
    r = verify("ANSWER: " + json.dumps(a, ensure_ascii=False) + "\n", "", "")
    assert not r["correct"] and not r["details"]["formula_reproduces_sequence"], r
    print("[OK] 正例(闭式文本能复算, 但内容不是该序列) ->", r["certificate"])

    # 判不了的三族之外写法 -> 如实判错（不放宽成 eval）
    a = {"oeis_id": "A000045", "formula": "F[n] = (2^n - 1)/(2 - 1)"}
    r = verify("ANSWER: " + json.dumps(a, ensure_ascii=False) + "\n", "", "")
    assert not r["correct"] and not r["details"]["formula_recomputable"], r
    print("[OK] 反例H(判不了的三族之外写法) ->", r["certificate"])

    print("[OK] 全部自测通过")