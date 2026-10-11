"""验证 stall 的"按错误类别计数"改动 —— 用 lean-04 的**真实失败序列**。

## 为什么必须用真数据

这个改动的全部依据是: lean-04 的 17 次调用里 11 次是"猜不存在的
Mathlib 定理名", 而**每次名字都不同**。原来的 stall 只数工具名
(全是 lean_check), 所以只看得出"同工具 7 次", 看不出"同根因 7 次"。

我要验的不是"新代码能跑", 而是:
  ① 喂进真实的 error 文本, `_err_class` 能不能把 11 次判成同一类
  ② 原来的计数在这串数据上会不会触发(对照)
  ③ `_failed_ids` 能不能把模型试过的标识符**真的**抠出来
  ④ `only_tool=True` 时, 提示语里**不再出现**"你还没试过别的工具"
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.research.roles.worker import (  # noqa: E402
    STALL_N, _call_ok, _digest_one, _err_class, _failed_ids, _loop_note)

# ★ lean-04 t1 lane0 的 7 次真实诊断, 按出现顺序摘自
#   part1_d1_d1run_a.jsonl 的 trace[].findings.calls[].error
REAL_ERRORS = [
    "unknown constant 'Nat.sum_choose_eq'",
    "unknown constant 'Nat.sum_choose_eq' / 'Nat.sum_choose' / 'Nat.sum_binomial'",
    "unknown identifier 'binomial_theorem' | invalid use of field notation with '@' modifier",
    "unknown identifier 'sum_range_choose'",
    "unsolved goals",
    "unknown constant 'Nat.sum_choose_eq'",      # 与第 1 次逐字重复
    "unknown constant 'Nat.sum_choose'",
]

FAILS: list[str] = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(("  [ok]   " if cond else "  [FAIL] ") + name
          + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def _old_stall_count(errors: list[str]) -> int:
    """旧逻辑的等价物: 只数"连续同工具"。这里全部是 lean_check。"""
    n = 0
    for _ in errors:
        n += 1          # 全是同一个工具 -> 一直增长
    return n


def test_run_loop_actually_executes_stall_path():
    """★ 必须让 `_run_loop` 真跑一遍, 而不是只测三个纯函数 ★

    ## 这条测试存在的唯一理由

    2026-10-11: 我给 `_err_class` / `_failed_ids` / `_loop_note` 都写了
    单测, **全绿**, 然后就去改 `_run_loop` 里调用它们的代码。
    结果第一版写成了:

        for k in reversed(findings):     # ← 覆盖了外层的 k
            ...
        left = max_steps - k             # ← k 现在是 str

    `check_convergence.py` 和 `check_react_loop.py` 当场崩:
        TypeError: unsupported operand type(s) for -: 'int' and 'str'

    ★ 那三个纯函数的单测**一条都没抓到这个 bug** ——
      因为它们根本不经过那个循环。★
      "测过了"和"这条路径被执行过"是两件不同的事。

    ⇒ 这条测试直接把真实的 worker 主循环跑起来, 喂一串会触发 stall 的
      失败, 断言它**不抛异常**并且 stall 真的被记录了。
    """
    from agent.tools.research.roles import worker as W

    class _Boom:
        @staticmethod
        def ask(prompt, **kw):
            # 连续 4 步都调同一个工具, 且每次都是"名字不存在"这一类错误
            n = getattr(_Boom, "n", 0)
            _Boom.n = n + 1
            if n < 4:
                # ★ 必须是**证不出来**的证明 ★
                #   我第一版写 `theorem t{n} : 1=1 := rfl` —— 那是**对的**,
                #   lean_check 返回 proved=True, 于是没有任何 error,
                #   `_err_class("")` 判成 "none", 循环第一轮就 break,
                #   `n_same_err=0`。测试自己"通过"了, 但根本没走到
                #   我要验的那条路径 —— 又一次"测过了 ≠ 执行过"。
                return json.dumps({"tool": "lean_check",
                                   "args": {"source":
                                            f"theorem t{n} : 1 = 2 := rfl"}})
            return json.dumps({"final": "答案", "reason": "编不下去了"})

    _Boom.n = 0
    orig = W.ask
    try:
        W.ask = _Boom.ask
        res = W._run_loop("题面", {"method": "m"}, "", probe=False,
                          scratch_dir=".", timeout=30, prior=[])
    finally:
        W.ask = orig

    sw = res.get("switches") or []
    ck("★ _run_loop 跑完没抛异常(那个 TypeError 就是从这儿来的)",
       True)
    ck("★ stall 真的触发了并落进 switches",
       any(x.get("trigger") == "stall" for x in sw),
       json.dumps(sw, ensure_ascii=False)[:200])
    ck("★ stall 记录里带了错误类别(新字段)",
       any(x.get("err_class") for x in sw),
       json.dumps(sw, ensure_ascii=False)[:200])
    ck("★ 最终交了答案而不是崩掉", res.get("has_artifact") is True
       or res.get("error_type") is None,
       f"error_type={res.get('error_type')}")


def main() -> int:
    print("=" * 76)
    print("stall 按错误类别计数 —— 用 lean-04 的真实失败序列验")
    print("=" * 76)

    classes = [_err_class(e) for e in REAL_ERRORS]
    print(f"\n  7 次真实诊断的归类:")
    for e, c in zip(REAL_ERRORS, classes):
        print(f"    {c:<20} <- {e[:62]}")

    # ① 归类是否稳定
    #
    # ★ 这里曾经断言"至少 4 次被判成 unknown-* 类" ★
    #   我第一版把 `unknown constant` 和 `unknown identifier` 分成两类,
    #   结果从后往前数连续同类只够 2 次, 够不着 STALL_N=3 ——
    #   **改动等于没做, 而测试当时是绿的**(我断言的是"至少 4 次同类",
    #   恰好也被满足, 看不出问题)。真实数据一跑才暴露。
    #   ⇒ 归类必须按**根因**("这个名字不存在")而不是按错误措辞。
    guessy = [c for c in classes if c == "name-or-shape"]
    ck("★ 6 次猜名字/思路不对的错误落在**同一类**", len(guessy) >= 6,
       f"实得 {len(guessy)} 次: {classes}")
    ck("★ 7 次全部同类 -> 从后往前连续计数够得着阈值",
       max(classes.count(c) for c in set(classes)) == 7,
       f"最大类 {max(classes.count(c) for c in set(classes))}")

    # ★ 两层设计: 判 stall 用**粗类**, 回灌黑名单用**窄正则** ★
    #   `_err_class` 把 unknown-constant / unknown-identifier / unsolved
    #   goals / type mismatch 合成一类, 因为对"是不是卡住了"这个问题,
    #   它们的后果一样: 微调参数没用, 必须换思路。
    #   而 `_failed_ids` **不**依赖 `_err_class`, 它自己用窄正则 ——
    #   因为只有"名字不存在"才有标识符可回灌, `unsolved goals` 没有。
    #   ⚠ 这两层是**故意**分开的, 别有人看到"重复的分类逻辑"就顺手统一。
    ck("★ 判 stall 的粗类把 unsolved goals 也算进去(它同样需要换思路)",
       _err_class("unsolved goals") == "name-or-shape")
    ck("★ 但回灌黑名单**不**把 unsolved goals 当成有名字可列",
       _failed_ids([{"error": "unsolved goals"}]) == [])

    # ② 新逻辑会不会触发
    n_cls = 0
    cur = ""
    for c in reversed(classes):
        if c in ("none", "other"):
            break
        if not cur:
            cur = c
        elif c != cur:
            break
        n_cls += 1
    print(f"\n  从后往前数连续同类错误 = {n_cls}  (STALL_N={STALL_N})")
    ck("★ 会在第 3 步就触发 stall(旧逻辑也要触发, 但给不出正确建议)",
       n_cls >= STALL_N, f"n_cls={n_cls}")

    # ③ _failed_ids 能不能真的抠出标识符
    findings = [{"tool": "lean_check", "ok": False, "error": e}
                for e in REAL_ERRORS]
    ids = _failed_ids(findings)
    print(f"\n  抠出的已试标识符: {ids}")
    ck("★ 抠出了 Nat.sum_choose_eq", "Nat.sum_choose_eq" in ids, str(ids))
    ck("★ 抠出了 binomial_theorem", "binomial_theorem" in ids, str(ids))
    ck("★ 抠出了 sum_range_choose", "sum_range_choose" in ids, str(ids))
    ck("★ 没把 'unknown'/'constant' 这类词误当标识符",
       "unknown" not in ids and "constant" not in ids and "identifier" not in ids,
       str(ids))

    # ④ only_tool 的提示语
    tried = ["lean_check"] * 7
    bad = _loop_note("lean_check", 7, tried, findings, only_tool=False)
    good = _loop_note("lean_check", 7, tried, findings, only_tool=True)
    ck("★ 旧提示会说'你还没试过别的工具'(这在只有 1 个工具时是有害的)",
       "你还没试过别的工具" in bad)
    ck("★ 新提示不再说那句话", "你还没试过别的工具" not in good)
    ck("★ 新提示明说'只有一个工具, 换工具不是出路'", "只有这一个工具" in good)
    ck("★ 新提示带上了已试过的标识符黑名单",
       "Nat.sum_choose_eq" in good, good[:200])
    ck("★ 有别的工具时, 仍照旧点名它们(没把原功能改坏)",
       "你还没试过别的工具" not in _loop_note("lean_check", 3,
                                          ["lean_check", "z3_smt2"],
                                          findings, only_tool=False))

    # ★ 必须真的把 worker 主循环跑一遍, 见该函数的 docstring
    test_run_loop_actually_executes_stall_path()

    print("\n" + "=" * 76)
    if FAILS:
        print(f"FAIL: {len(FAILS)} 条")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("PASS: 错误类别计数与标识符回灌在真实数据上成立")
    return 0


if __name__ == "__main__":
    sys.exit(main())