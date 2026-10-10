"""答案被"占位符/省略号"掏空 —— 与数学能力无关的一类失败。

## 现象

2026-10-10 晚 part2 实测反复出现: 模型的 `answer` 字段是

    "你的答案"      <- 把提示词里的字段名当成了内容
    "..."           <- 直接省略
    ""              <- 完全空

而这些题 `verdict` 是 `WORKER_ONLY`(它自称交出了制品)。
res-18 烧了 720 秒 / 13 次工具调用, `agree=1/2`;res-14 烧 904 秒 / 25 次
工具 / 9 轮 lane, answer 是空字符串。

## ★ 为什么这必须单独查, 不能混进 WRONG 里 ★

判据说它 WRONG 是对的, 但**原因不是"算错"**:
  · 模型可能**已经把正确结论算出来了**(中间结果在 findings 里),
    只是在组织 final 时塌成了占位符;
  · 也可能**真的没算出来**, 就是空转。

这两种在分数上一样(都记 WRONG), 但**修法完全相反**:
  · 前者 -> 收尾/格式问题, 加个检查或重试就能救;
  · 后者 -> 收敛问题, 要换方法或加预算。

⇒ 不分开, 就只能"看到 WRONG 就以为模型不会做"。

## 判据

1. 答案长度 < PLACEHOLDER_MAX 且命中占位特征 -> 归为 PLACEHOLDER
2. PLACEHOLDER **不算 WRONG 也不算 CORRECT**, 单列
3. 它必须能区分出"算出来了但写不出来" vs "压根没算出来":
   看 `findings.calls` 里有没有非平凡的数值结果。
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.research.routes.d import _nums_in  # noqa: E402

FAILS: list[str] = []

#: 答案短于这个长度且全是占位内容 -> 判为 PLACEHOLDER
PLACEHOLDER_MAX = 40

#: 命中即认为"没交出内容"。★ 不是黑名单, 是**症状** ★
PLACEHOLDER_PAT = re.compile(
    r"^\s*(?:\.{2,}|…+|-+)?\s*(?:你的?答案|答案|your\s+answer|answer|"
    r"结果|result|待定|todo|n/?a|null|none)\s*(?:\.{2,}|…+)?\s*$",
    re.IGNORECASE)


def classify_answer(answer: str) -> str:
    """把答案分成 OK / PLACEHOLDER / EMPTY。"""
    s = (answer or "").strip()
    if not s:
        return "EMPTY"
    if len(s) <= PLACEHOLDER_MAX and PLACEHOLDER_PAT.match(s):
        return "PLACEHOLDER"
    # 只有省略号也算
    if len(s) <= PLACEHOLDER_MAX and re.fullmatch(r"[.。…\s\-]+", s):
        return "PLACEHOLDER"
    return "OK"


def _has_real_work(trace: list[dict]) -> bool:
    """trace 里有没有**非平凡的**数值结果 —— 区分"算出来了写不出"与"没算"。

    ★ 这是关键区分 ★
      `findings.calls[*].value` 里出现 >=2 个互不相同的非零数,
      就说明 worker **确实算出了东西**, 只是 final 塌成了占位符。
    """
    for e in trace or []:
        f = e.get("findings") or {}
        vals: set[float] = set()
        for c in (f.get("calls") or []):
            for v in _nums_in(str(c.get("value") or "")):
                if abs(v) > 1e-12:
                    vals.add(round(v, 9))
        if len(vals) >= 2:
            return True
    return False


def main() -> int:
    print("=" * 78)
    print("占位符答案的分类")
    print("=" * 78)
    cases = [
        ("", "EMPTY"),
        ("...", "PLACEHOLDER"),
        ("……", "PLACEHOLDER"),
        ("你的答案", "PLACEHOLDER"),
        ("answer", "PLACEHOLDER"),
        ("n/a", "PLACEHOLDER"),
        ("-", "PLACEHOLDER"),
        ("  ", "EMPTY"),
        ("1.41421356", "OK"),
        ("3-√3、3、3+√3", "OK"),
        ("...", "PLACEHOLDER"),
    ]
    for text, expect in cases:
        got = classify_answer(text)
        mark = "OK " if got == expect else "!! "
        print(f"  [{mark}] {text!r:14} -> {got}")
        if got != expect:
            FAILS.append(f"分类错: {text!r} -> {got} (期望 {expect})")

    print()
    print("=" * 78)
    print("能否区分「算出来了但写不出来」vs「压根没算出来」")
    print("=" * 78)
    worked = [{"role": "worker", "findings": {"calls": [
        {"tool": "numeric_code", "value": "0.59185 0.592746 0.5945"}]}}]
    idle = [{"role": "worker", "findings": {"calls": [
        {"tool": "numeric_code", "value": "Traceback: error"}]}}]
    ck1 = _has_real_work(worked)
    ck2 = _has_real_work(idle)
    print(f"  [{'OK ' if ck1 else '!! '}] 有非平凡数值 -> 判为'算出来了' {ck1}")
    print(f"  [{'OK ' if not ck2 else '!! '}] 只有报错 -> 判为'没算出来' {not ck2}")
    if not ck1:
        FAILS.append("_has_real_work 漏判了算出来的情形")
    if ck2:
        FAILS.append("_has_real_work 把报错当成了真结果")

    print()
    print("=" * 78)
    print(f"失败 {len(FAILS)}")
    print("=" * 78)
    for f in FAILS:
        print(f"  ★ {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())