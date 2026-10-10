"""ReAct 闭环: Worker 的调查结果**真的交回** Explorer, 下一轮 Worker 真的**看得到**。

## 这道测试在防什么

2026-10-10 实测(part2 零提示 20 题): 13 道"没交出答案"的题里, **13 道
都在重复调同一个工具**(`numeric_code` 最多 x16、`gap_eval` x10)。
表面上像"收敛差", 但根因不是收敛机制 —— 是**信息没交回去**:

  · 旧版 Worker **完全看不到 history**。它每轮只拿到 problem + method,
    于是每次调用都像第一次调这个工具。
  · 旧版 Worker 收尾只交一个 `brief()` —— 而 brief 是"**最后一步的尾部截断**",
    信息量最低的那一段。Explorer 拿到它也无从判断"这条路走到哪了"。

⇒ 修法不是加提醒, 是**把每一步拿到了什么结构化攒起来, 交回 Explorer,
  并让下一轮 Worker 看得到**。

## 判据

1. Worker 的结果里有 `findings`, 含**逐步**的 calls(不是只有最后一步)
2. `findings` 里有 ok/value/error —— 即"拿到了什么", 不只是"调过什么"
3. 没交出答案时, `findings.stopped_because` 说明**停在哪**
4. Explorer 渲染历史时**真的用上了** findings(文本里能看到)
5. Worker 的 prompt 在有 prior 时**真的包含**前几轮的结果
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.research.roles import explorer as EX  # noqa: E402
from agent.tools.research.roles import worker as WK     # noqa: E402

FAILS: list[str] = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


class FakeAsk:
    def __init__(self, replies):
        self.replies = list(replies)
        self.n = 0
        self.seen = []

    def __call__(self, prompt, **_kw):
        self.seen.append(prompt)
        r = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1
        return r


def run(replies, prior=None):
    orig = WK.ask
    fake = FakeAsk(replies)
    try:
        WK.ask = fake
        res = WK._run_loop("题面", {"method": "m"}, "目录", probe=False,
                           scratch_dir=".", timeout=30, prior=prior)
    finally:
        WK.ask = orig
    return res, fake


def T(tool, args='{"code":"x=1"}'):
    return '{"tool": "%s", "args": %s}' % (tool, args)


def main() -> int:
    print("=" * 76)
    print("① Worker 交回 findings: 逐步记录, 不只是最后一步")
    print("=" * 76)
    res, _ = run([T("numeric_code"), T("numeric_code"),
                  T("sympy_code"),
                  '{"final": "答案是 14/5", "reason": "算出来的"}'])
    f = res.get("findings") or {}
    print(f"    findings 键: {sorted(f)}")
    calls = f.get("calls") or []
    ck("findings 存在", bool(f))
    ck("findings 记录了**每一步**的调用", len(calls) == 3, f"共 {len(calls)} 条")
    ck("每条都记了工具名", all(c.get("tool") for c in calls),
       str([c.get("tool") for c in calls]))
    ck("每条都记了 ok(拿到了没有)",
       all("ok" in c for c in calls))
    ck("★ findings 里有 value/error(拿到了什么) ★",
       any(("value" in c and c["value"]) or ("error" in c) for c in calls),
       str([{k: str(v)[:40] for k, v in c.items() if k != "tool"}
            for c in calls])[:200])
    ck("findings 有 summary 一句话总结",
       "summary" in f and str(f["summary"])[:20], str(f.get("summary"))[:80])

    print()
    print("=" * 76)
    print("② 没交出答案时, findings 说明停在哪")
    print("=" * 76)
    res2, _ = run([T("numeric_code")] * 8)
    f2 = res2.get("findings") or {}
    ck("★ 步数用完 -> stopped_because 说明了原因 ★",
       bool(f2.get("stopped_because")), str(f2.get("stopped_because")))
    ck("★ 说明是步数用完, 而不是笼统的失败 ★",
       "步数" in str(f2.get("stopped_because"))
       or "RanOutOfSteps" in str(f2.get("stopped_because")),
       str(f2.get("stopped_because")))
    ck("★ 即使没交答案, 8 步的 calls 也都记下来了 ★",
       len(f2.get("calls") or []) == 8, f"共 {len(f2.get('calls') or [])} 条")
    ck("findings 里能看到它换过路",
       bool(f2.get("switched")) or bool(res2.get("switches")),
       str(f2.get("switched"))[:90])

    print()
    print("=" * 76)
    print("③ Explorer 渲染历史时真的用上了 findings")
    print("=" * 76)
    hist = [{
        "turn": 1, "action": "execute", "tools": ["numeric_code"],
        "method": "先算个大概规模",
        "findings": {"calls": [{"tool": "numeric_code", "ok": True,
                                "value": "n=200 cities"}],
                     "summary": "共调 1 次工具"},
        "outcome": "brief 的最后一步", "lesson": "",
    }]
    txt = EX.plan_prompt("题面", hist, None)
    ck("历史文本里有 '调查摘要' 的内容",
       "numeric_code" in txt and "n=200 cities" in txt)
    ck("★ 提示 Explorer 据此判断 ★",
       "据此判断" in txt or "别重复已经做过的事" in txt)
    ck("旧的 brief 摘要也还留着",
       "brief 的最后一步" in txt)

    print()
    print("=" * 76)
    print("④ Worker 的 prompt 在有 prior 时包含前几轮结果")
    print("=" * 76)
    res3, fake3 = run([T("sympy_code")], prior=hist)
    first_prompt = fake3.seen[0] if fake3.seen else ""
    ck("★ prompt 里出现 '前几轮已经查到的' ★",
       "前几轮已经查到的" in first_prompt)
    ck("前几轮的结果真的带进去了",
       "n=200 cities" in first_prompt,
       first_prompt[:0])
    ck("提示别重复已做的事", "别重复已经做过的事" in first_prompt)

    print()
    print("=" * 76)
    print("⑤ 没有 prior 时不硬塞(别给第 1 轮加噪音)")
    print("=" * 76)
    _, fake5 = run([T("numeric_code"), '{"final":"x","reason":"y"}'])
    ck("第 1 轮 prompt 里没有 '前几轮' 段",
       "前几轮已经查到的" not in (fake5.seen[0] if fake5.seen else ""))

    print(f"\n失败 {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())