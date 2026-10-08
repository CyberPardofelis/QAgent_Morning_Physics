"""
agent.tools.research._sympy_tools — SymPy 符号计算 (P7 阶段 0)

为什么不让 research_sympy 直接调 sympy:
1. **注入风险** —— expr 是 LLM 生成的字符串。直接 `sympify(expr)` 等于让它能被构造成
   `__import__('os').system(...)`。所以本模块只把 expr 用 `repr()` 嵌进**生成的代码**,
   再交给 `_sandbox.run_code` 走 AST 门 + 超时保护。
2. **会卡死** —— SymPy 的 integrate/limit 能把主进程挂几十分钟。沙箱的超时是刚需。

所以本模块的职责是"把 (expr, op, params) 编译成一段安全的 Python 代码",
执行交给沙箱。这样 research_sympy 和 research_verify 共用同一套安全/超时机制。

判据(设计稿 §1.5.1 条款 1): 要"等于 0"就得走符号 —— 浮点数恰好为 0 不代表恒等于 0。
这也是 verdict 能给 proved 而不只是 partial 的唯一路径。

历史:
  - 2026-10-08 P7.0 首版
"""
from __future__ import annotations

from typing import Any

from ._sandbox import DEFAULT_TIMEOUT, SandboxResult, run_code

#: 支持的 op → 是否需要 params(以及需要几个)
_OPS_NEEDING_VARS: dict[str, int] = {
    "integrate": 1, "diff": 1, "limit": 2, "series": 3, "sum": 3, "nt_sum": 3,
}
_OPS_NO_VARS: tuple[str, ...] = (
    "simplify", "expand", "factor", "cancel", "apart",
    "trigsimp", "solve", "eqsolve", "isprime",
)

ALL_OPS: tuple[str, ...] = _OPS_NO_VARS + tuple(_OPS_NEEDING_VARS)


# ============================================================
# 代码生成
# ============================================================

def _body_for(op: str) -> str:
    """
    按 op 生成求值代码片段。

    注意所有用户输入都靠 `_e` / `_params` 这两个已绑定的名字, **不做字符串插值进表达式**,
    只有 repr() 的字面量。op 本身来自白名单, 不是用户输入。
    """
    if op in ("simplify", "expand", "factor", "cancel", "apart", "trigsimp"):
        fn = {"simplify": "simplify", "expand": "expand", "factor": "factor",
              "cancel": "cancel", "apart": "apart", "trigsimp": "trigsimp"}[op]
        return (
            f"    r = sympy.{fn}(_e)\n"
            f"    print('结果:', r)\n"
            f"    print('LaTeX:', sympy.latex(r))\n"
            f"    print('值为 0(符号):', sympy.simplify(r) == 0)\n"
        )

    if op == "solve":
        return (
            "    r = sympy.solve(sympy.Eq(_e, 0), dict=False) if isinstance(_e, sympy.Expr) "
            "else sympy.solve(_e, dict=False)\n"
            "    print('解:', r)\n"
            "    print('LaTeX:', sympy.latex(r))\n"
        )

    if op == "eqsolve":
        return (
            "    if len(_params) < 2:\n"
            "        raise ValueError('eqsolve 需要 params=\"左式 右式\"')\n"
            "    lhs = sympy.sympify(_params[0]); rhs = sympy.sympify(_params[1])\n"
            "    r = sympy.solve(sympy.Eq(lhs, rhs))\n"
            "    print('解:', r)\n"
            "    print('LaTeX:', sympy.latex(r))\n"
        )

    if op == "isprime":
        return (
            "    r = sympy.isprime(_e)\n"
            "    print('是否素数:', r)\n"
        )

    if op == "integrate":
        return (
            "    if _params:\n"
            "        r = sympy.integrate(_e, sympy.Symbol(_params[0]))\n"
            "    else:\n"
            "        r = sympy.integrate(_e)\n"
            "    print('积分:', r)\n"
            "    print('LaTeX:', sympy.latex(r))\n"
            "    print('求导验算:', sympy.simplify(sympy.diff(r, _free_var(r)) - _e))\n"
        )

    if op == "diff":
        return (
            "    if _params:\n"
            "        v = sympy.Symbol(_params[0])\n"
            "    else:\n"
            "        fv = _e.free_symbols\n"
            "        if len(fv) != 1:\n"
            "            raise ValueError('diff 无法推断变量, 请传 params=\"x\"')\n"
            "        v = list(fv)[0]\n"
            "    r = sympy.diff(_e, v)\n"
            "    print('导数:', r)\n"
            "    print('LaTeX:', sympy.latex(r))\n"
        )

    if op == "limit":
        return (
            "    if len(_params) < 2:\n"
            "        raise ValueError('limit 需要 params=\"变量 目标点\", 如 \"x 0\" 或 \"x oo\"')\n"
            "    v = sympy.Symbol(_params[0]); pt = sympy.sympify(_params[1])\n"
            "    r = sympy.limit(_e, v, pt)\n"
            "    print('极限:', r)\n"
            "    print('LaTeX:', sympy.latex(r))\n"
        )

    if op == "series":
        return (
            "    if len(_params) < 3:\n"
            "        raise ValueError('series 需要 params=\"变量 起点 阶数\", 如 \"x 0 6\"')\n"
            "    v = sympy.Symbol(_params[0]); a = sympy.sympify(_params[1]); n = int(_params[2])\n"
            "    r = sympy.series(_e, v, a, n)\n"
            "    print('级数:', r)\n"
            "    print('LaTeX:', sympy.latex(r))\n"
        )

    if op in ("sum", "nt_sum"):
        fn = "summation" if op == "sum" else "Sum"
        return (
            "    if len(_params) < 3:\n"
            f"        raise ValueError('{op} 需要 params=\"变量 下限 上限\", 如 \"i 1 10\"')\n"
            "    v = sympy.Symbol(_params[0]); lo = sympy.sympify(_params[1]); hi = sympy.sympify(_params[2])\n"
            f"    r = sympy.{fn}(_e, (v, lo, hi))\n"
            "    if r.has(sympy.Sum):\n"
            "        r = sympy.simplify(sympy.summation(r))\n"
            "    print('求和结果:', r)\n"
            "    print('LaTeX:', sympy.latex(r))\n"
        )

    raise ValueError(f"不支持的 op: {op}")


_CODE_TEMPLATE = '''"""
由 research_sympy 自动生成 —— SymPy 符号计算
expr/params 以 repr() 字面量嵌入, 不是字符串拼接, 无法注入。
"""
import sympy


def _free_var(expr):
    fv = sympy.sympify(expr).free_symbols
    if len(fv) != 1:
        raise ValueError('无法唯一确定变量, 请在 params 里显式指定')
    return list(fv)[0]


_e = sympy.sympify({expr!r})
_params = {params!r}.split()
_r = None
if True:
{body}
'''


def build_code(expr: str, op: str, params: str = "") -> str:
    """
    把 (expr, op, params) 编译成一段可交给沙箱执行的 Python 代码。

    Raises:
        ValueError: op 不在白名单
    """
    if op not in ALL_OPS:
        raise ValueError(f"不支持的 op={op!r}, 可用: {', '.join(ALL_OPS)}")
    return _CODE_TEMPLATE.format(expr=expr, params=params or "", body=_body_for(op))


# ============================================================
# 对外入口
# ============================================================

def sympy_op(
    expr: str,
    op: str = "simplify",
    params: str = "",
    scratch_dir: str = "",
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """
    做一次符号计算, 返回结构化结果 (尚未挂 evidence_id)。

    Returns:
        {ok, result_str, latex, elapsed_ms, code, stdout, stderr, error_type}
    """
    try:
        code = build_code(expr, op, params)
    except ValueError as e:
        return {
            "ok": False, "result_str": "", "latex": "", "elapsed_ms": 0,
            "code": "", "stdout": "", "stderr": str(e), "error_type": "BadOp",
        }

    if not scratch_dir:
        return {
            "ok": False, "result_str": "", "latex": "", "elapsed_ms": 0,
            "code": code, "stdout": "",
            "stderr": "sympy_op 缺少 scratch_dir (内部调用错误)",
            "error_type": "NoScratch",
        }

    res: SandboxResult = run_code(code, scratch_dir, timeout=timeout)
    result_str, latex = _parse_output(res.stdout)
    return {
        "ok": res.ok,
        "result_str": result_str,
        "latex": latex,
        "elapsed_ms": res.elapsed_ms,
        "code": code,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "error_type": res.error_type,
        "timed_out": res.timed_out,
        "rejected": res.rejected,
    }


def _parse_output(stdout: str) -> tuple[str, str]:
    """从子进程输出里抽 结果 / LaTeX 两行"""
    result_str = ""
    latex = ""
    for line in (stdout or "").splitlines():
        s = line.strip()
        if s.startswith("结果:") or s.startswith("积分:") or s.startswith("导数:") or s.startswith("解:") or s.startswith("极限:") or s.startswith("级数:") or s.startswith("求和结果:") or s.startswith("是否素数:"):
            result_str = s.split(":", 1)[1].strip()
        elif s.startswith("LaTeX:"):
            latex = s.split(":", 1)[1].strip()
    return result_str, latex
