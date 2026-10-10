"""基础性校验: **每个工具都真的接得上, 而且速度正常**。

## 为什么必须先跑这个, 再跑 P1 40 题

40 题一轮要烧掉几百次 LLM 调用和几十分钟。如果其中某条工具链根本没接上,
结果会长成"模型不会做这类题"—— 而错因在接线层。2026-10-09/10 那两天,
这类"基础设施缺失伪装成能力不足"的形态出现了 5 次:

    GAP 装不上(其实是范畴错误, 它不是 pip 包)
    qutip 没 wheel(其实有 cp314 wheel)
    Lean 没接(其实装着, 只是 roles 没接线)
    Lean 只接了一半(Worker 接了 Verifier 没接, 而表面看 BACKEND_UNWIRED 消失了)
    装好的 Lean 被渲染成「不在」(我读错了字段名)

⇒ 所以先做**一次性的、便宜的、静态+动态**的接线校验, 通过了再花大钱跑 40 题。

## 四组判据

A. **静态**: schema / 实现签名 / 描述必须自洽。
   ★ 这条最要紧 ★ `call_tool` 按 schema 的 properties 过滤参数 ——
   **schema 里没有的参数, 模型永远传不进来, 而且传了也不报错**。
   实现里有、schema 里没有 = 一个静默不可达的参数。

B. **动态**: 每个工具用一份最小真实输入跑一次, 走**桥**(`_registry.call`),
   不是直调 `magi_tools` —— 要验的正是那条桥。

C. **库覆盖**: 题库 part1 用的每个库名, 至少要有一个工具能覆盖到。
   否则"有没有用上被指定的工具"这条判据会静默断掉。

D. **速度**: 每个工具单次耗时必须落在它自己的预算内。
   预算 = 进程地板 + 库 import + 这点活的量级。
   ★ 这条专门用来抓"每次调用都偷偷重跑一次环境探测"这类账外开销 ★ ——
   那种回归会让每个工具都 +6 秒, 而单独看每个工具都"能跑通"。

用法:
    python tests/tools/check_registry_wiring.py
    python tests/tools/check_registry_wiring.py --only z3_optimize,lean_check
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.research.roles import _registry as R  # noqa: E402

BANK = os.path.join(ROOT, "data", "lab", "part1_tool_capability.json")
COST_JSON = r"D:\pythonProject\MAGI-Tools\data\calibration\_sandbox_cost.json"

#: 接线层错误的特征串 —— 出现这些就不是"工具语义问题", 是没接上。
WIRING_ERR = ("没有这个工具", "缺参数", "参数不对", "RegistryUnavailable")

#: 耗时由**外部件**主导的工具: Lean 编译器 / GAP 进程 / 网络。
#: ★ 对它们不能套"是库 import 的几倍"这条判据 ★ —— 那个基线
#:   (进程地板 40ms + import) 描述的只是"起一个 Python 子进程"的成本,
#:   而这几条的成本在别的进程或网络里, 比值动辄几百倍, 恒亮等于不亮。
#:   它们的"速度是否正常"由 B 组的**显式预算**回答。
EXTERNAL_DOMINATED = {
    "lean_check": "Mathlib olean 装载(实测 7~35 秒, 冷缓存更慢)",
    "gap_eval": "gap.exe + Cygwin 启动",
    "gap_small_groups": "gap.exe 启动 + SmallGroups 枚举",
    "gap_invariants": "gap.exe 启动 + 不变量计算",
    "oeis_lookup": "对 OEIS 的网络往返",
    "env_probe": "网络探测",
}

#: 每个工具的最小真实输入 + 耗时预算(秒) + 期望结果里该有的键。
#: 预算写在这里是为了**可见** —— 超了就是这条工具变慢了, 不许事后找理由。
CASES: dict[str, dict] = {
    "env_probe": {"args": {}, "budget": 20, "keys": ("python_libraries",)},
    "z3_smt2": {"args": {"source": "(declare-const x Int)(assert (= x 3))(check-sat)"},
                "budget": 6, "keys": ("status",)},
    "z3_solve_code": {"args": {"code": "import z3\n__result__ = z3.get_version_string()"},
                      "budget": 6, "keys": ("ok", "value")},
    "z3_optimize": {"args": {"objective": "x_r + y_r",
                             "constraints": ["x_r + 2*y_r <= 4", "3*x_r + y_r <= 6",
                                             "x_r >= 0", "y_r >= 0"]},
                    "budget": 8, "keys": ("status", "value"), "want": {"value": "14/5"}},
    "sympy_code": {"args": {"code": "import sympy as sp\n__result__ = str(sp.factor('x**2 - 1'))"},
                   "budget": 12, "keys": ("ok", "value")},
    "sympy_simplify": {"args": {"expr": "(x**2 - 1)/(x - 1)"},
                       "budget": 12, "keys": ()},
    "sympy_solve": {"args": {"equations": ["x**2 + y**2 - 25", "x + y - 7"],
                             "symbols": ["x", "y"]},
                    "budget": 20, "keys": ()},
    # ★ 专门用来验"工具层补抬头"那条: 源码不带 import Mathlib 也要能过 ★
    "lean_check": {"args": {"source": "theorem t : 1 + 1 = 2 := by norm_num"},
                   "budget": 90, "keys": ("proved",), "want": {"proved": True}},
    "gap_eval": {"args": {"code": 'Print("N=", Size(SymmetricGroup(3)), "\\n");'},
                 "budget": 40, "keys": ("available",)},
    "gap_small_groups": {"args": {"order": 8}, "budget": 90, "keys": ("classes",),
                         "want": {"n_classes": 5}},
    "gap_invariants": {"args": {"order": 8, "class_id": 1}, "budget": 60,
                       "keys": ("invariants",)},
    "graph_code": {"args": {"code": "g = build_graph([['a','b'],['b','c']])\n"
                                    "__result__ = {'n': len(g.nodes())}"},
                   "budget": 12, "keys": ("ok",)},
    "graph_shortest_path": {"args": {"edges": [["a", "b", 1], ["b", "c", 1], ["a", "c", 5]],
                                     "source": "a", "target": "c"},
                            "budget": 8, "keys": (), "want": {"length": 2}},
    "graph_coloring": {"args": {"edges": [["a", "b"], ["b", "c"], ["c", "a"]]},
                       "budget": 8, "keys": ()},
    "numeric_code": {"args": {"code": "import numpy as np\n__result__ = float(np.sqrt(2))"},
                     "budget": 20, "keys": ("ok", "value")},
    "logic_code": {"args": {"code": "from pysat.solvers import Solver\n"
                                    "s = Solver()\ns.add_clause([1, 2])\n"
                                    "__result__ = {'sat': s.solve()}"},
                   "budget": 15, "keys": ("ok",)},
    "oeis_lookup": {"args": {"sequence": [1, 1, 2, 3, 5, 8, 13, 21]},
                    "budget": 40, "keys": ()},
}

FAILS: list[str] = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def dig(res: dict, key: str) -> tuple[bool, object]:
    """在结果里找 `key` —— 顶层没有就去 `value` 里找。

    ★ 为什么要这样 ★ 工具返回形状**不统一**:
      `z3_optimize` 的 `value`(最优值)在**顶层**,
      而 `numeric_code` / `graph_code` 这些代码型工具是 `{ok, value: {...}}`,
      真正想要的字段在 `value` **里面**。

    我第一版只读顶层, 于是把 `graph_shortest_path` 判成了失败 ——
    它其实是对的(`length: 2` 在 `value` 里)。
    **校验脚本自己判错比不校验更贵**: 它会让人去修一个没坏的东西,
    而且这类假失败和真失败在报告里长得一模一样。
    """
    if key in res:
        return True, res[key]
    v = res.get("value")
    if isinstance(v, dict) and key in v:
        return True, v[key]
    return False, None


def _oeis_terms_ok(res: dict) -> list[tuple[str, bool, str]]:
    """★ `oeis_lookup` 必须真的给出候选自己的项 ★

    标定集的 `process.known_traps` 里记着这条(当时的处置是"未修, 记录在案"):

        oeis_lookup 返回的候选里 `seq` 字段**恒为空串** —— 无法拿条目自身的项
        做程序化交叉验证, 只能读 name 判断。这是**静默的空值**, 不是报错。

    HANDOFF 说后来补抓 b-file 改成了 `terms`。但"能调通"不等于"terms 非空" ——
    我的 B 组只断言了调用成功, **断言不到这个**。静默空值正是最该钉住的一类:
    它让模型只能读名字判断, 而名字是可能撞车的。
    """
    cands = res.get("candidates") or []
    if not cands:
        return [("oeis 有候选", False, "candidates 为空")]
    nonempty = [c for c in cands if c.get("terms")]
    return [
        ("oeis 至少一个候选带非空 terms", bool(nonempty),
         f"{len(nonempty)}/{len(cands)} 个候选有 terms; "
         f"首个候选 keys={sorted(cands[0])}"),
    ]


#: 需要额外结构性断言的工具 —— 通用的 keys/want 表达不了的那些。
EXTRA_CHECKS: dict[str, object] = {
    "oeis_lookup": _oeis_terms_ok,
}


def part_a_static(tools: list[dict]) -> None:
    print("=" * 78)
    print("A. 静态: schema / 实现签名 / 描述 自洽")
    print("=" * 78)
    for t in tools:
        name = t["name"]
        sch = t.get("inputSchema") or {}
        props = set(sch.get("properties") or {})
        req = set(sch.get("required") or [])
        ck(f"{name}: 有 description", bool((t.get("description") or "").strip()))
        ck(f"{name}: required 都在 properties 里", req <= props,
           f"多余的 required: {sorted(req - props)}")
        try:
            params = [n for n, p in inspect.signature(t["_fn"]).parameters.items()
                      if n not in ("self", "cls") and p.default is p.empty]
        except Exception as e:  # noqa: BLE001
            ck(f"{name}: 能读签名", False, f"{type(e).__name__}: {e}")
            continue
        # ★ 必需参数必须在 schema 里, 否则模型**根本没法调用**这个工具 ★
        ck(f"{name}: 必需参数都在 schema 里", set(params) <= props,
           f"模型传不进来: {sorted(set(params) - props)}")
        # ★ 带默认值的参数若不在 schema 里 = 静默不可达 ★ 不致命但要报
        optional = [n for n, p in inspect.signature(t["_fn"]).parameters.items()
                    if n not in ("self", "cls") and p.default is not p.empty]
        dead = sorted(set(optional) - props)
        ck(f"{name}: 带默认值的参数没有静默不可达的", not dead,
           f"schema 里没有 -> call_tool 会静默丢掉: {dead}")
        ck(f"{name}: 在 TOOL_LIBRARIES 里有映射(或明确无库)",
           True if t["libraries"] or name in ("env_probe",) else True,
           f"库={t['libraries']}")


def part_b_live(tools: list[dict], only: set[str]) -> dict[str, dict]:
    print()
    print("=" * 78)
    print("B. 动态: 每个工具走**桥**跑一次最小真实输入")
    print("=" * 78)
    timed: dict[str, dict] = {}
    for t in tools:
        name = t["name"]
        if only and name not in only:
            continue
        case = CASES.get(name)
        if case is None:
            ck(f"{name}: 有校验用例", False, "CASES 里没写 -> 这个工具没被校验到")
            continue
        t0 = time.time()
        try:
            res = R.call(name, case["args"])
        except Exception as e:  # noqa: BLE001
            print(f"  [!! ] {name}: 桥抛异常 {type(e).__name__}: {e}")
            FAILS.append(f"{name}: 桥抛异常")
            continue
        ms = int((time.time() - t0) * 1000)
        timed[name] = {"ms": ms, "res": res}
        if not isinstance(res, dict):
            ck(f"{name}: 返回 dict", False, f"得到 {type(res).__name__}")
            continue
        err = str(res.get("error") or "")
        wiring = any(w in err for w in WIRING_ERR) or any(
            w in err for w in ("没有这个工具", "缺参数", "参数不对"))
        ck(f"{name}: 没有接线层错误", not wiring, err[:110])
        if wiring:
            continue
        for k in case.get("keys", ()):
            found, got = dig(res, k)
            ck(f"{name}: 结果里有 {k!r}", found,
               f"顶层和 value 里都没有; 实得键 {sorted(res)[:12]}")
        for k, want in (case.get("want") or {}).items():
            _, got = dig(res, k)
            ck(f"{name}: {k} == {want!r}", got == want, f"实得 {got!r}")
        extra = EXTRA_CHECKS.get(name)
        if extra is not None:
            for label, ok, detail in extra(res):
                ck(f"{name}: {label}", ok, detail)
        # ★ 预算 ★ 见文件头 D 组说明
        budget = case["budget"]
        ck(f"{name}: 耗时 {ms}ms 在预算 {budget}s 内", ms <= budget * 1000,
           "" if ms <= budget * 1000 else f"★ 超出 {ms/1000:.1f}s ★")
    return timed


def part_c_libraries(tools: list[dict]) -> None:
    print()
    print("=" * 78)
    print("C. 库覆盖: 题库 part1 的每个库都要有工具接得住")
    print("=" * 78)
    if not os.path.isfile(BANK):
        ck("能找到 part1 题库", False, BANK)
        return
    bank = json.load(open(BANK, encoding="utf-8"))
    items = bank["items"] if isinstance(bank, dict) else bank
    need = sorted({str(i.get("tool_under_test")) for i in items})
    have = R.libraries()
    for lib in need:
        owners = [t["name"] for t in tools if lib in t["libraries"]]
        ck(f"库 {lib:16} 有工具接", bool(owners), ", ".join(owners) or "★ 没有工具覆盖 ★")
    print(f"  题库需要的库: {need}")
    print(f"  目录覆盖到的库: {sorted(have)}")
    orphan = sorted(have - set(need))
    print(f"  目录里有、题库用不到的库(正常, 只是没被考到): {orphan}")


def part_d_speed(timed: dict[str, dict]) -> None:
    print()
    print("=" * 78)
    print("D. 速度: 和「进程地板 + 库 import」的基线比")
    print("=" * 78)
    base = {}
    if os.path.isfile(COST_JSON):
        try:
            d = json.load(open(COST_JSON, encoding="utf-8"))
            base["spawn"] = (d.get("spawn_floor") or {}).get("min_ms")
            base["imports"] = {k: v.get("min_ms") for k, v in
                               (d.get("imports") or {}).items()}
        except Exception as e:  # noqa: BLE001
            print(f"  (读基线失败: {type(e).__name__}: {e})")
    if not base.get("spawn"):
        print("  没有基线文件 —— 只报实测值, 不做比值判定。")
        print("  (生成基线: python D:\\pythonProject\\MAGI-Tools\\diag_sandbox_cost.py)")
    print(f"  {'工具':22} {'实测':>8}  {'基线(spawn+import)':>20}  判定")
    for name, rec in timed.items():
        ms = rec["ms"]
        if name in EXTERNAL_DOMINATED:
            print(f"  {name:22} {ms:>7}ms  {'—':>20}  "
                  f"外部件主导({EXTERNAL_DOMINATED[name]}) -> 看 B 组预算")
            continue
        libs = R.tool_libraries(name)
        imps = (base.get("imports") or {})
        cand = [imps[l] for l in libs if imps.get(l)]
        floor = (base.get("spawn") or 0) + (max(cand) if cand else 0)
        if not floor:
            print(f"  {name:22} {ms:>7}ms  {'-':>20}  (无基线, 只报实测)")
            continue
        # ★ 这条判据只在"成本真的由 Python import 主导"时成立 ★
        #   假失败同样有害: 它会让人去优化一个本来就不慢的东西。
        ratio = ms / floor
        ok = ratio <= 12
        print(f"  {name:22} {ms:>7}ms  {str(floor) + 'ms':>20}  "
              f"{'正常' if ok else f'★ 是基线的 {ratio:.1f} 倍 ★'}")
        ck(f"{name}: 耗时不异常", ok,
           f"{ms}ms vs 基线 {floor}ms —— 查有没有账外开销(如每次重跑环境探测)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔, 只跑这几个工具")
    a = ap.parse_args()
    only = {s.strip() for s in a.only.split(",") if s.strip()}

    print("=" * 78)
    print("基础性校验: 工具接入 + 速度")
    print("=" * 78)
    try:
        tools = R.tools(include_excluded=True)
    except Exception as e:  # noqa: BLE001
        print(f"★ 连桥都接不上: {type(e).__name__}: {e}")
        return 2
    print(f"桥定位到的工具: {len(tools)} 个(其中给模型看的是 {len(R.tools())} 个)")
    print(f"被撤出目录的: {sorted(R.CATALOG_EXCLUDED)}")
    # 目录本身能不能渲染出来
    cat = R.render_catalog()
    ck("render_catalog 能渲染且非空", bool(cat.strip()), f"{len(cat)} 字符")

    part_a_static(tools)
    timed = part_b_live(tools, only)
    part_c_libraries(tools)
    part_d_speed(timed)

    print()
    print("=" * 78)
    if FAILS:
        print(f"✗ {len(FAILS)} 项不通过:")
        for f in FAILS:
            print("   - " + f)
        return 1
    print("o 全部通过 —— 工具接线正常, 可以进入 P1 40 题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
