"""roles ↔ MAGI-Tools 工具注册表之间的**唯一**一座桥。

## 为什么需要它, 而不是直接 import

工具住在 `MAGI-Tools` 那个项目里(它是 MCP server, 也提供 17 个可直调的函数)。
QAgent 这边要用, 就得先把它的根目录放进 `sys.path` —— 而三个入口的 cwd 各不相同:

    MAGI-Tools/score_roles_vs_calibration.py    HERE 已经是 MAGI-Tools
    QAgent/tests/tools/smoke_roles.py           只加了 QAgent
    QAgent/tests/* 单元测试                      只加了 QAgent

写死一个路径必然在其中一个入口炸。所以这里按固定顺序找, 找不到就**明确报错**:

    RegistryUnavailable: 试过这些路径 [...] —— 这是**回路缺口**, 不是模型的失败

★ 为什么必须报错而不是回退到"不接工具" ★
2026-10-10 的实测教训: 少一个后端/少一根线, 症状会伪装成其它东西 ——
`lean4` 没接线时表现为 Worker 写调用 `lean_check()` 的 Python 然后 NameError,
看上去像"模型不会写代码"。同一族错误那天出现了 4 次。
所以基础设施性缺失**必须是一个独立状态**, 混进能力分就是污染测量。

## 环境信息为什么前置

`magi_tools.registry.TOOLS` 是**静态表** —— 它不告诉你这台机器装没装 sympy。
只有 `env_probe()` 才知道。而实测 2026-10-10 的原生工具形态实验里:
**5 道题, 5 道的第一步都是调 env_probe**(占全部 15 次调用里的 5 次, 33%),
拿到的还只是一个在整轮里不会变的事实。

⇒ 所以: 探测**一次**, 写进提示开头; 并把 `env_probe` 从目录里撤掉。
   它仍在(模型硬要调就给缓存结果, 不报错), 只是不再占一个轮次。
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any

__all__ = ["RegistryUnavailable", "tools", "call", "env_report", "render_env",
           "render_catalog", "tool_libraries", "libraries", "CATALOG_EXCLUDED"]

#: 放工具仓库根目录的环境变量(优先于自动探测)
HOME_ENV = "MAGI_TOOLS_HOME"

#: 2006-10-10: 从目录里撤掉的工具 —— 它的答案已经写在提示开头。
CATALOG_EXCLUDED = {
    "env_probe": "环境实况已探测一次并前置到提示开头; 让它占一个轮次是纯浪费",
}

#: 每个工具能"练到"哪些库。**判据不是"这个工具属于哪个库"** ——
#: 而是"用这个工具能让哪个库跑起来", 因为题库 part1 是按**库**组织的
#: (`tool_under_test: "z3"`), 而目录是按**工具**给的。
#: 少了这层映射, "有没有真用上被指定的工具"这条判据就断了。
TOOL_LIBRARIES: dict[str, list[str]] = {
    "z3_smt2": ["z3"],
    "z3_solve_code": ["z3"],
    "z3_optimize": ["z3"],
    "sympy_code": ["sympy"],
    "sympy_simplify": ["sympy"],
    "sympy_solve": ["sympy"],
    "lean_check": ["lean4"],
    "gap_eval": ["gap"],
    "gap_small_groups": ["gap"],
    "gap_invariants": ["gap"],
    "graph_code": ["networkx"],
    "graph_shortest_path": ["networkx"],
    "graph_coloring": ["networkx"],
    "logic_code": ["pysat"],
    "oeis_lookup": ["oeis"],
    # ★ numeric_code 是"给一段 Python 就跑"的通用口子 —— 它不专属任何库,
    #   而是 scipy / mpmath / cvxpy / ortools / qutip 这些**没有专属工具**的库
    #   的唯一入口。写成单值会直接把 5 个库的题判成"没用上指定工具"。
    #   ★ 名字用**题库那一侧的行话** ★ 这张表就是用来把 "trace 里的工具名"
    #     映回 "题库里的库名" 的, 而题库写的是 `qutip_quantum` 不是 `qutip` ——
    #     照抄题库才对得上。(2026-10-10 库覆盖校验抓到缺这一个。)
    "numeric_code": ["numpy", "scipy", "mpmath", "cvxpy", "ortools", "qutip_quantum"],
}

#: ★ 必须是**可重入**锁, 不能是 threading.Lock ★
#   实测 2026-10-10 (C 路线第一次冒烟, 卡了十分钟):
#
#       env_report()  ->  with _lock:  ...  ->  _load()  ->  with _lock:  ✗ 死锁
#
#   `env_report()` 要先抢锁才能防重复探测, 而它内部又要调 `_load()`,
#   而 `_load()` 自己也抢同一把锁。Lock 不可重入 -> 第二次 acquire 永久阻塞。
#
#   ★ 为什么之前的校验没抓到 ★ 那些调用都**先调了 `tools()`**,
#     于是 `_mod` 已经就绪, `_load()` 在锁外面就短路返回了 ——
#     **调用顺序把 bug 盖住了**。一个只在"某个入口先被调用"时才炸的
#     死锁, 靠"跑一遍看看"是抓不到的; 必须专门有一条从冷启动入口进来的回归。
_lock = threading.RLock()
_mod: Any = None
_env_cache: dict | None = None


class RegistryUnavailable(RuntimeError):
    """MAGI-Tools 找不到或导不进来。**这是回路缺口, 不是模型的失败。**"""


def _qagent_root() -> str:
    """roles/_registry.py -> roles -> research -> tools -> agent -> QAgent

    ★ dirname 的次数要数对 ★ 第一版写 range(5)(加上初始赋值等于剥了 6 层),
    于是"QAgent 的兄弟目录"算成了 `D:\\MAGI-Tools`, 报错信息里还煞有介事地
    列出了一个根本不存在的路径 —— 定位对了但层级错了, 症状是"找不到 MAGI-Tools"。
    """
    d = os.path.dirname(os.path.abspath(__file__))   # 已在 roles/ 这一层
    for _ in range(4):
        d = os.path.dirname(d)
    return d


def _candidates() -> list[str]:
    out: list[str] = []
    env = os.environ.get(HOME_ENV, "").strip()
    if env:
        out.append(env)
    # QAgent 的兄弟目录 —— 两个项目都在 D:/pythonProject 下
    out.append(os.path.join(os.path.dirname(_qagent_root()), "MAGI-Tools"))
    # 已经可导入就不用找了
    out.append("")
    return out


def _load() -> Any:
    """定位并导入 `magi_tools.registry`。只做一次。"""
    global _mod
    if _mod is not None:
        return _mod
    with _lock:
        if _mod is not None:
            return _mod
        tried: list[str] = []
        for cand in _candidates():
            if cand:
                tried.append(cand)
                if not os.path.isdir(cand):
                    continue
                if cand not in sys.path:
                    sys.path.insert(0, cand)
            else:
                tried.append("(已在 sys.path 上)")
            try:
                from magi_tools import registry  # noqa: PLC0415
            except Exception as e:  # noqa: BLE001
                tried[-1] += f" -> {type(e).__name__}: {e}"
                continue
            _mod = registry
            return _mod
        raise RegistryUnavailable(
            "接不上 MAGI-Tools 的工具注册表 —— 这是**回路缺口, 不是模型的失败**。\n"
            f"试过: {tried}\n"
            f"修法: 设环境变量 {HOME_ENV} 指向 MAGI-Tools 根目录, "
            f"或把 MAGI-Tools 放到 QAgent 的兄弟目录。")
    return _mod


def tools(*, include_excluded: bool = False) -> list[dict[str, Any]]:
    """真工具表。每条额外带 `library` / `libraries`。

    `include_excluded=True` 会带上 `CATALOG_EXCLUDED` 里的工具(体检用),
    **不给模型看** —— 让它看见一个不打算让它调的工具, 它就会去调。
    """
    reg = _load()
    out = []
    for t in reg.TOOLS:
        name = t["name"]
        if name in CATALOG_EXCLUDED and not include_excluded:
            continue
        libs = TOOL_LIBRARIES.get(name, [])
        out.append({**t, "libraries": libs,
                    "library": libs[0] if libs else None})
    return out


def tool_libraries(name: str) -> list[str]:
    """这个工具能练到哪些库。目录里没有它 -> 空列表(不是瞎猜)。"""
    return list(TOOL_LIBRARIES.get(name, []))


def libraries(names: list[str] | None = None) -> set[str]:
    """一组工具覆盖到的库。用来把 trace 里的工具名映回题库的库名。"""
    src = names or [t["name"] for t in tools(include_excluded=True)]
    out: set[str] = set()
    for n in src:
        out.update(tool_libraries(n))
    return out


def call(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """跑一个工具。**失败一律变成返回值**, 见 `registry.call_tool` 的说明。"""
    if name in CATALOG_EXCLUDED:
        # ★ 不报错, 给缓存结果 + 说清楚它不该占一个轮次 ★
        #   报错会让模型重试; 而它要的信息其实早就有了, 只是它没看开头。
        if name == "env_probe":
            return {**env_report(),
                    "_note": "环境实况已经写在你的提示开头了, 不必再探测一次。"
                             "这一次算你重复调用, 不扣结果。"}
    reg = _load()
    return reg.call_tool(name, args or {})


def env_report(refresh: bool = False) -> dict[str, Any]:
    """探测一次这台机器有什么。**进程内缓存** —— 一批题只付一次。

    网络那一项实测要 0.6~6.5 秒, 而它在整轮里不会变。
    """
    global _env_cache
    if _env_cache is not None and not refresh:
        return _env_cache
    with _lock:
        if _env_cache is not None and not refresh:
            return _env_cache
        reg = _load()
        try:
            _env_cache = dict(reg.env_probe())
        except Exception as e:  # noqa: BLE001
            # ★ 探测本身挂掉不能静默成"什么都没装" ★
            #   那会让模型以为工具不可用, 从而避开本来能用的路。
            _env_cache = {"_probe_failed": f"{type(e).__name__}: {e}",
                          "_note": "环境探测本身失败了 —— 工具的可用性**未知**, "
                                   "不是**不可用**。拿不准就先试一下。"}
    return _env_cache


def _flag3(v: Any, ok_keys: tuple[str, ...]) -> str:
    """把一个"可用性字段"判成 在 / 不在 / 未知。

    ★ 为什么必须区分「未知」和「不在」★
    2026-10-10 我自己在这里踩了一次: `env_probe` 的 Lean 字段是个 dict
    (`{"lean": <exe 路径>, "lean_ok": True, "mathlib_ok": False}`), 我按
    `lean.get("available")` 去读 —— 键名不存在, `.get` 返回 None,
    于是一个**装好的** Lean 被渲染成「不在」。

    ★ 而且这个 bug 会伪装成别的问题 ★ 提示开头写着「Lean 不在」, 模型于是
    绕开 lean_check —— 4 道 Lean 题会全部表现为"模型不会证",
    而真正的错因在我这行渲染代码里。这正是那天栽了 4 次的同一族:
    **基础设施性缺失伪装成能力不足**。

    ⇒ 规则: 认不出来的字段一律报「未知」, 绝不许悄悄退化成「不在」。
    看到「未知」会去试, 看到「不在」会绕开 —— 前者可恢复。
    """
    if isinstance(v, bool):
        return "在" if v else "不在"
    if isinstance(v, str):
        return "在" if v.strip() else "不在"
    if isinstance(v, dict):
        for k in ok_keys:
            if k in v:
                return "在" if v[k] else "不在"
        return "未知(字段认不出来: " + ", ".join(sorted(v)) + ")"
    return "未知"


def render_env() -> str:
    """把环境实况渲染成一小段文字。"""
    e = env_report()
    if e.get("_probe_failed"):
        return (f"环境探测失败: {e['_probe_failed']}\n"
                f"⇒ 各库装没装**未知**。拿不准就直接调那个工具试 —— "
                f"它跑不起来会如实报错, 而「报错」和「这台机器没有」是两回事。")
    libs = e.get("python_libraries") or {}
    have = sorted(k for k, v in libs.items() if v)
    miss = sorted(k for k, v in libs.items() if not v)
    lean = e.get("lean")
    lean_s = _flag3(lean, ("lean_ok", "ok", "available"))
    mlib_s = ""
    if isinstance(lean, dict) and "mathlib_ok" in lean:
        # ★ Lean 本体和 Mathlib 是两件事 ★ 没有 Mathlib 也能证核心定理,
        #   但 `import Mathlib` 会失败。分开报, 免得模型以为整个 Lean 不能用。
        mlib_s = (" (Mathlib " + ("在" if lean["mathlib_ok"]
                                  else "**不在** —— 不能用 import Mathlib") + ")")
    gap_s = _flag3(e.get("gap"), ("gap_ok", "ok", "available"))
    net = e.get("network")
    net_s = "通" if net else ("不通" if net is False else "未知")
    return ("这台机器上**已经装好**的: " + (", ".join(have) or "(一个都没有)") + "\n"
            "**装不上**的(别选, 选了会执行时炸): " + (", ".join(miss) or "(无)") + "\n"
            f"Lean 4: {lean_s}{mlib_s}   GAP: {gap_s}   OEIS 网络: {net_s}\n"
            "★ 这份实况是开工前探测的, 整轮里不会变 —— **不要再去探一次**。★")


def render_catalog(*, with_env: bool = True,
                   include_excluded: bool = False) -> str:
    """给 Explorer / Worker 看的目录。环境实况在前, 工具表在后。"""
    parts: list[str] = []
    if with_env:
        parts.append("【这台机器上实际有什么】(开工前探测一次, 整轮不变)\n"
                     + render_env())
    rows = []
    for t in tools(include_excluded=include_excluded):
        rows.append(f"### {t['name']}")
        rows.append(t["description"])
        rows.append("参数(Schema): " + _schema_line(t.get("inputSchema") or {}))
        rows.append("")
    parts.append("【可用工具】(名字必须**逐字**用这里的, 不要发明)\n"
                 + "\n".join(rows).rstrip())
    return "\n\n".join(parts)


def _schema_line(schema: dict) -> str:
    props = schema.get("properties") or {}
    req = set(schema.get("required") or [])
    if not props:
        return "{}"
    bits = []
    for k, v in props.items():
        mark = "*" if k in req else ""
        bits.append(f"{k}{mark}")
    return "{" + ", ".join(bits) + "}  其中带 * 的必填"
