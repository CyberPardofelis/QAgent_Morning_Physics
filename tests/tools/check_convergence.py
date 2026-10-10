"""收敛机制的行为测试 —— 不打真 LLM, 用假回复脚本驱动。

## 这道测试在防什么

2026-10-10 实测(part2 零提示 20 题): **13 道"没交出答案"的题里, 13 道
都在重复调同一个工具**(`numeric_code` 最多 x16、`gap_eval` x10、
`lean_check` x8), 而"唯一工具数"几乎全是 1。

那不是探索, 是**卡在循环里**。而"换路"这件事光写在提示里没用 ——
**模型看不到自己调了几次**, 所以它并不知道"我已经调了 8 次了"。
⇒ 必须由 harness 计数并把事实告诉它。

## 判据

1. 连续调同一个工具到 `STALL_N` 次 -> 触发 `stall` 提示
2. 剩余步数 <= `FORCE_SWITCH_AT` -> 触发 `budget_low` 提示
3. 提示内容要**点名已试过的其它工具**(让"换路"有具体对象)
4. 不该触发时**不触发**(否则每步都在喊"你卡住了", 变成噪音)
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.research.roles import worker as WK  # noqa: E402

FAILS: list[str] = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


class FakeAsk:
    """按脚本喂回复 —— 不打真 LLM。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.n = 0
        self.seen: list[str] = []

    def __call__(self, prompt, **_kw):
        self.seen.append(prompt)
        r = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1
        return r


def run(replies):
    orig = WK.ask
    fake = FakeAsk(replies)
    try:
        WK.ask = fake
        res = WK._run_loop("题面", {"method": "m"}, "目录",
                           probe=False, scratch_dir=".", timeout=30)
    finally:
        WK.ask = orig
    return res, fake


def T(name, args):
    return '{"tool": "%s", "args": %s}' % (name, args)


def main() -> int:
    print("=" * 76)
    print("① 连续调同一个工具 -> 必须触发 stall 提示")
    print("=" * 76)
    code = '{"code": "x=1"}'
    res, fake = run([T("numeric_code", code)] * 8)
    sw = res.get("switches") or []
    stalls = [x for x in sw if x.get("trigger") == "stall"]
    ck("连续 8 次 numeric_code -> 有 stall 触发", bool(stalls),
       f"switches={[(x.get('trigger'), x.get('times')) for x in sw]}")
    ck("stall 记下了是哪个工具", all(x.get("tool") == "numeric_code"
                                   for x in stalls))
    ck("stall 次数 >= STALL_N", stalls and stalls[0]["times"] >= WK.STALL_N,
       f"首次触发 times={stalls[0]['times'] if stalls else '-'}")
    ck("提示里点名了已试过的工具",
       any("numeric_code" in p and "连续" in p for p in fake.seen))
    ck("提示里说了'不要再原样调它'",
       any("不要再原样调它" in p for p in fake.seen))
    ck("提示里催它交 final(烧光预算什么都交不出更糟)",
       any("现在就交 final" in p for p in fake.seen))

    print()
    print("=" * 76)
    print("② 换工具就不该被喊'卡住了'(否则提示变噪音)")
    print("=" * 76)
    mixed = [T("numeric_code", code), T("sympy_code", code),
             T("logic_code", code), T("numeric_code", code),
             T("sympy_code", code), T("logic_code", code),
             T("numeric_code", code), T("sympy_code", code)]
    res2, fake2 = run(mixed)
    calls = [c["tool"] for c in res2["tool_calls"]]
    stalls2 = [x for x in (res2.get("switches") or [])
               if x.get("trigger") == "stall"]
    ck("交替调三个工具 -> 没有 stall 触发", not stalls2,
       f"实际触发 {[(x.get('tool'), x.get('times')) for x in stalls2]}")
    ck("交替时确实调了多种工具", len(set(calls)) == 3, f"{calls}")

    print()
    print("=" * 76)
    print("③ 预算将尽 -> 必须触发 force 提示")
    print("=" * 76)
    res3, fake3 = run([T("graph_code", code)] * 8)
    lows = [x for x in (res3.get("switches") or [])
            if x.get("trigger") == "budget_low"]
    ck("有 budget_low 触发", bool(lows),
       f"switches={[(x.get('trigger'), x.get('left')) for x in (res3.get('switches') or [])]}")
    ck("budget_low 提示里说了'只剩 N 步'",
       any("只剩" in p and "步" in p for p in fake3.seen))
    ck("budget_low 提示里给了三条出路(final/换路/give_up)",
       any("give_up" in p for p in fake3.seen))

    print()
    print("=" * 76)
    print("④ 正常收尾不该被收敛机制干扰")
    print("=" * 76)
    res4, _ = run([T("numeric_code", code),
                   '{"final": "答案是 42", "reason": "算出来的"}'])
    ck("调一次就 final -> 有制品", res4.get("has_artifact") is True)
    ck("调一次就 final -> 不该有 stall",
       not [x for x in (res4.get("switches") or [])
            if x.get("trigger") == "stall"])
    ck("答案被保留", res4.get("final_answer") == "答案是 42")

    print()
    print("=" * 76)
    print("⑤ 所有键都必须初始化过(读后写的键)")
    print("=" * 76)
    for key in ("error_type", "has_artifact", "tool_calls", "lean",
                "final_answer", "last_tool_result", "switches"):
        ck(f"键 {key!r} 在结果里存在", key in res4)

    print(f"\n失败 {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())