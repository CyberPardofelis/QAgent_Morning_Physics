"""验证 worker._call_ok —— 用**真实工具的真实返回**, 不靠构造的假 obs。

★ 为什么必须真调一次 ★
2026-10-11 这一天, 同一个「工具成功没有统一信号」的坑栽了两次:
  1. `_digest_one` 只认 `ok`, Lean 题 findings 里 ok=None
  2. `tool_calls` 那条连 `proved` 都没带, 于是 `brief()` 对
     「12 次 lean_check, proved=True」报「0 次成功」
而单元测试全绿 —— 因为测试里喂的是**自己造的 dict**, 形状恰好自洽。
真调一次工具, 才能看到注册表实际吐出来的形状。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.research.roles import _registry  # noqa: E402
from agent.tools.research.roles.worker import (  # noqa: E402
    _call_ok, _digest_one, brief)

# ★ 走 roles._registry 这座桥, 不直接 `from magi_tools import registry` ★
#   MAGI-Tools 不是 QAgent 的包, 它的定位逻辑(兄弟目录 / 环境变量)本来就
#   集中在 _registry 里; 在测试里另写一份 import 会和真链路漂移。
call_tool = _registry.call

CASES = [
    ("z3_smt2", {"source": "(declare-const x Int)(assert (= x 3))",
                 "timeout": 20}),
    # ★ 参数名是 `source` 不是 `code` —— 我第一版写成 code, lean_check
    #   返回 {'error':'缺参数: source', 'expected':[...], 'ok':False},
    #   测试于是"证成/证伪"两条都挂在**参数错**上, 而代码是对的。
    #   又一次"前提错了就怪实现"的教训。
    ("lean_check", {"source": "theorem t : 1 = 1 := rfl", "timeout": 120}),
    ("lean_check", {"source": "theorem t : 1 = 2 := rfl", "timeout": 120}),
    ("sympy_simplify", {"expr": "x**2 + 1"}),
    ("__no_such_tool__", {}),
]

FAILS: list[str] = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(("  [ok]   " if cond else "  [FAIL] ") + name
          + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("=" * 74)
    print("worker._call_ok: 拿真工具的真返回验")
    print("=" * 74)
    got = {}
    for name, args in CASES:
        obs = call_tool(name, args)
        got.setdefault(name, []).append((args, obs))
        prov = obs.get("proved") if isinstance(obs, dict) else "-"
        stat = obs.get("status") if isinstance(obs, dict) else "-"
        print(f"\n  {name}")
        print(f"    返回键 = {sorted(obs.keys()) if isinstance(obs, dict) else obs}")
        print(f"    _call_ok = {_call_ok(obs)}   proved={prov}   status={stat}")
        print(f"    _digest_one().ok = {_digest_one(name, args, obs).get('ok')}")

    # ① Lean 证成: 成功
    a, o = got["lean_check"][0]
    ck("lean_check 证成 -> 成功", _call_ok(o) is True and o.get("proved") is True)
    # ② Lean 证伪: 工具跑通了, 但**它给出了"这个证明不成立"的裁决**
    #    ⇒ 判为失败。这里 Lean 与 z3 的处理**故意不对称**, 不是遗漏:
    #      · Lean 家族有**明确的裁决字段** `proved`(True=证成/False=证伪),
    #        尊重它 —— findings 要回到模型下一轮, 若把证伪报成 ok,
    #        模型会以为这条路走通了。
    #      · z3 家族**没有**这种"我期望什么"的字段(status=unsat 只是
    #        "无解", 对"证明这就是完整解集"来说恰恰是**想要的**结果),
    #        所以那里只问"跑没跑通"。
    #    把这两族强行统一成一个语义, 才会真的出错。
    a2, o2 = got["lean_check"][1]
    ck("lean_check 证伪 -> 判失败(它有明确裁决字段 proved=False, 要尊重)",
       _call_ok(o2) is False and o2.get("proved") is False,
       f"proved={o2.get('proved')}")
    # ③ z3 无 ok 键
    a3, o3 = got["z3_smt2"][0]
    ck("★ z3_smt2 成功返回里没有 ok 键(踩坑前提)", "ok" not in o3)
    ck("★ z3_smt2 仍被判为成功", _call_ok(o3) is True)
    ck("★ z3_smt2 的 digest.ok 也是 True", _digest_one("z3_smt2", a3, o3)["ok"] is True)
    # ④ 有 ok 的族
    ck("sympy_simplify 有 ok 键且判成功",
       "ok" in got["sympy_simplify"][0][1]
       and _call_ok(got["sympy_simplify"][0][1]) is True)
    # ⑤ 失败形状
    ck("不存在的工具 -> 失败", _call_ok(got["__no_such_tool__"][0][1]) is False)

    # ⑥ brief 的「N 次成功」不再把 Lean 全数成 0
    txt = brief({"tool_calls": [
        {"tool": "lean_check", "args": {}, "ms": 1, "ok": True},
        {"tool": "z3_smt2", "args": {}, "ms": 1, "ok": True},
        {"tool": "lean_check", "args": {}, "ms": 1, "ok": False}]})
    print(f"\n  brief 摘要: {txt}")
    ck("brief 不再报「0 次成功」", "0 次成功" not in txt, txt)

    print("\n" + "=" * 74)
    if FAILS:
        print(f"FAIL: {len(FAILS)} 条")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("PASS: 工具成败判定在三个族上都成立")
    return 0


if __name__ == "__main__":
    sys.exit(main())
