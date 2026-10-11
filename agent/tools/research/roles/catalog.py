"""工具目录 —— Explorer 看得见, **题面看不见**。

## 这份目录为什么必须和题面分开

旧结构里 `prompts_research.py` 做的是:

    info = classify(claim)          # 从**题面**认形状
    ms   = methods_for(shapes)      # 按形状发一份方法目录
    # → 拼进 prompt: "机器从题面认出的形状标签: ... " + 对应方法表

于是模型看到的是"机器已经判断出这是组合结构题, 你该用这几个方法"。
**方法选择这道题被提前做掉了**, 剩下的是照着做。

新结构里, 题面只描述问题; 工具是**这个环境里有什么**这件事,
是系统的事实, 不是题目的提示。Explorer 拿着目录自己判断。

## 目录内容是**探测出来的**, 不是写死的

`available` 字段是现场 `importlib.util.find_spec` 的结果。
本机装不上的工具(qutip/gap/sage 这类 Python 3.14 没 wheel 的)
不会出现在可用列表里 —— 让 Explorer 选一个不存在的工具,
然后 Worker 报 ImportError, 那是**目录的错**, 不是模型的错。

## `not_for` 为什么要写

只写"能干什么"的话, 模型会把所有工具都往问题上套。
一个科研工作者知道的从来不只是"这个工具能做什么",
还有"这个工具**不擅长**什么" —— 后者才是选型的关键。
这是工具文档, 不是解题提示: 它对**所有**问题一视同仁。
"""
from __future__ import annotations

import importlib.util
import shutil
from typing import Any

from . import _registry as _reg

__all__ = ["build_catalog", "render_catalog", "CATALOG_SPEC"]

#: 工具的静态描述。`probe` = 怎么确认它在这台机器上可用。
#: ★ 这份表里**不能**出现"哪类题该用哪个" ★ 只有工具自己的能力边界。
CATALOG_SPEC: list[dict[str, Any]] = [
    {
        "id": "python",
        "name": "Python 解释器 (沙箱内)",
        "does": "通用计算: 枚举、模拟、搜索、构造对象、代回检验",
        "not_for": "规模一大就超时; 不擅长符号化简与精确有理运算",
        "how": "直接写代码; 需要精确有理数时用 fractions.Fraction",
        # ★ 不能用 find_spec("python") 探 ★ 没有叫 python 的模块,
        #   那样探测会把解释器本身报成"装不上"。
        "probe": "always",
    },
    {
        "id": "sympy",
        "name": "SymPy",
        "does": "符号化简、精确值、微积分、代数封闭形式、多项式/矩阵运算",
        "not_for": "大规模离散枚举; 数值上下溢敏感的问题",
        "how": "import sympy as sp; 用 Rational/Integer 而不是 float",
        "probe": "spec:sympy",
    },
    {
        "id": "z3",
        "name": "Z3 定理证明器 (SMT)",
        "does": "一阶可判定逻辑上的可满足性判定: 给出模型, 或给出 unsat",
        "not_for": "定义域无界时返回 unknown 而非答案; "
                   "涉及三角函数等量词的命题表达不进去",
        "how": "import z3; Solver(); s.add(约束); s.check() == z3.sat 取 model()",
        "probe": "spec:z3",
    },
    {
        "id": "networkx",
        "name": "NetworkX",
        "does": "图论算法: 最短路、着色、连通度、社区发现、树、同构",
        "not_for": "超大规模图; 图的结构不从题面来时无从下手",
        "how": "import networkx as nx",
        "probe": "spec:networkx",
    },
    {
        "id": "scipy",
        "name": "SciPy",
        "does": "数值计算: 求根、常微分方程、插值、优化、特殊函数",
        "not_for": "需要**证明**成立 —— 数值证据最多支持 PARTIAL",
        "how": "import scipy.optimize / scipy.integrate / scipy.linalg",
        "probe": "spec:scipy",
    },
    {
        "id": "mpmath",
        "name": "mpmath (高精度)",
        "does": "任意精度数值: 位抽取、特殊常数、无闭式量的高精度逼近",
        "not_for": "同样不能用来**证明**; 精度设小了会在末位出错, 而末位常是答案",
        "how": "import mpmath as mp; mp.mp.dps = 50",
        "probe": "spec:mpmath",
    },
    {
        "id": "cvxpy",
        "name": "CVXPY (凸优化)",
        "does": "线性/二次/半定规划与凸优化建模, 可求最优值与对偶证书",
        "not_for": "非凸问题会给出无意义的松弛解",
        "how": "import cvxpy as cp; Problem(Maximize/ Minimize, constraints); .solve()",
        "probe": "spec:cvxpy",
    },
    {
        "id": "ortools",
        "name": "OR-Tools (CP-SAT)",
        "does": "整数规划与约束满足: 排程、JSSP、Golomb 尺这类离散最优化",
        "not_for": "连续变量问题; 规模很大时需要给时间上限",
        "how": "from ortools.sat.python import cp_model",
        "probe": "spec:ortools",
    },
    {
        "id": "lean4",
        "name": "Lean 4 + Mathlib (形式化证明)",
        "does": "把命题写成内核可检查的形式, 得到机器裁决的证明",
        "not_for": "只适合能被精确定义并陈述的数学对象; "
                   "未装 Lean 时会静默回落 Python —— 那不构成形式化证明",
        "how": "lean_check 工具; 禁用 sorry / axiom 走后门",
        "probe": "exe:lean",
    },
    {
        "id": "oeis",
        "name": "OEIS 整数序列检索",
        "does": "给一段整数序列, 找出候选条目与递推/闭式",
        "not_for": "只对**整数**序列有效; 序列太短(<6 项)匹配一大片",
        "how": "oeis_lookup 工具",
        "probe": "always",
    },
    {
        "id": "pysat",
        "name": "PySAT (SAT 求解)",
        "does": "命题可满足性, 百万级变量; 配合基数约束求解组合问题",
        "not_for": "需要连续数值的地方; CNF 编码写错时结果照样 sat",
        "how": "from pysat.solvers import Solver; from pysat.card import CardEnc",
        "probe": "spec:pysat",
    },
]


def _spec_ok(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:  # noqa: BLE001 - find_spec 对坏包会抛
        return False


def _lean_ok() -> bool:
    """Lean 不在 PATH 上也可能装了 —— 两处都找。"""
    try:
        from agent.tools.research._lean import lean_available
        return bool(lean_available())
    except Exception:  # noqa: BLE001
        return shutil.which("lean") is not None


def _available(spec: dict) -> bool:
    p = spec.get("probe", "always")
    if p == "always":
        return True
    if p == "exe:lean":
        return _lean_ok()
    return _spec_ok(p.split(":", 1)[1])


def build_catalog(include_unavailable: bool = False) -> list[dict]:
    """返回**真工具表**(来自 MAGI-Tools 的注册表), 不是一份手写的库名清单。

    ## 为什么换成真表 (2026-10-10)

    旧版这里硬编码 11 个**库名**(`z3` / `sympy` / `cvxpy` ...), 而工具层实际提供
    17 个**工具**(`z3_optimize` / `sympy_simplify` / `numeric_code` ...)。
    两者对不上, 于是 Explorer 选的是"库名", Worker 只能写一整个 Python 程序进沙箱
    自己去 import —— 那一整套"必须交程序不交答案"的规则、以及随之而来的
    `NameError` / `SandboxRejected` / `NotCode` 四类失败, 都是**这个错配的伴生物**。
    C 路线实测: 工具层直跑 36 题判对 33 道, 每题 2~3 次 LLM 调用。

    目录从"这台机器有什么库"变成"**这台机器上有哪些工具, 各自要什么参数、能做什么、
    不擅长什么**"—— 后者才是选型真正需要的信息。

    ## `id` / `libraries` 为什么保留库名

    `tests/tools/build_lab_bank.py` 用 `t["id"]` 去核对题库里的
    `tool_under_test`(那是**库名**)。但 `numeric_code` 一个条目同时覆盖
    numpy / scipy / mpmath / cvxpy / ortools / qutip_quantum 六个库 ——
    只把 `id` 设成 `libs[0]` 的话, 那 5 个库在对拍里会被误报成
    "目录里没有这个库"。所以**一个库名展开成一条目录条目**(同一个工具出现多次),
    这样 `ids = {t["id"] for t in build_catalog()}` 拿到的就是完整的库名集合。

    映射写在 `_registry.TOOL_LIBRARIES`, 一处改, 两边都对。
    """
    out = []
    for t in _reg.tools(include_excluded=include_unavailable):
        libs = list(t.get("libraries") or [])
        if not libs:
            out.append({**t, "id": t["name"], "libraries": [], "available": True})
        for lib in libs:
            out.append({**t, "id": lib, "libraries": libs, "available": True})
    return out


def render_catalog(tools: list[dict] | None = None) -> str:
    """渲染成给 Explorer 看的那段文本(环境实况 + 真工具表)。"""
    if tools is None:
        return _reg.render_catalog()
    lines = [f"【这台机器上实际有什么】(开工前探测一次, 整轮不变)\n"
             f"{_reg.render_env()}"]
    rows = []
    for t in tools:
        rows.append(f"### {t['name']}")
        rows.append(t["description"])
        schema = t.get("inputSchema") or {}
        sprops = schema.get("properties") or {}
        props = sorted(sprops)
        req = set(schema.get("required") or [])
        # ★ 必须打印**类型** ★ (2026-10-11 实测 40 题)
        #   原来只打参数名: `参数: code*, data_files, timeout` —— 模型于是
        #   只能猜 `data_files` 是数组还是对象, 实测 **42 次**传成 list 被
        #   registry 拦下(`参数类型不对: data_files 应该是 object, 收到 list`),
        #   其中 **32 次题面里根本没有数据文件**, 是模型凭空发明这个参数。
        #   每一次都要赔一整个 LLM 回合(实测均值 18.7s)才算得出这个类型。
        #   ⇒ 类型就在 inputSchema 里, 最后一步渲染扔掉它没有任何好处。
        def _fmt(p: str) -> str:
            ty = (sprops.get(p) or {}).get("type") or "?"
            return f"{p}*: {ty}" if p in req else f"{p}: {ty}"

        rows.append("参数: " + (", ".join(_fmt(p) for p in props) or "无"))
        # ★ 属性 description 必须渲染出来 ★ (2026-10-11 实测 40 题)
        #   那句最关键的信息就**只**写在属性上, 工具级 description 里没有:
        #     「沙箱里读不到文件系统, 所以题目提到文件时**必须**用 data_files
        #       把内容交进来。代码里用**别名本身**取全文(不是路径)」
        #   模型看不到它, 于是去 `import os` / `open()` / `pathlib` 找文件 ——
        #   实测 15 次 `SandboxRejected`, 全部集中在 networkx-04 / ortools-04
        #   这两道题面**真的带数据文件**的题上。
        #
        #   代价很小: 5 个代码类工具的属性描述**是同一段文字**(共 1605 字符),
        #   只占工具描述总量(10222 字符)的 16%, 换来的是沙箱数据访问契约可见。
        for p in props:
            d = str((sprops.get(p) or {}).get("description") or "").strip()
            if d:
                rows.append(f"  · {p}: {d}")
        rows.append("")
    lines.append("【可用工具】(名字必须**逐字**用这里的, 不要发明)\n"
                 + "\n".join(rows).rstrip())
    return "\n\n".join(lines)


if __name__ == "__main__":
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    tools = build_catalog(include_unavailable=True)
    print("=" * 68)
    for t in tools:
        mark = "可用" if t["available"] else "**装不上**"
        print(f"  [{mark:8}] {t['id']:12} {t['name']}")
    print("=" * 68)
    print(f"可用 {sum(1 for t in tools if t['available'])} / 共 {len(tools)}")