"""把当前机器门(含 R6)回放到 V2 已落盘的全部证据上, 逐条重算判定。

为什么回放而不是重跑:
  重跑要重新调 LLM, 结果不可比(模型非确定)而且贵。证据链 `_evidence.jsonl`
  里 code / stdout / claim 原样存着, 门是纯函数 —— 离线重算**完全等价**。

为什么逐条而不是只取每题最后一条:
  只看最后一条会漏掉「某轮判对、下一轮崩了」的情况, 而这类题恰恰最需要
  确认新门不会把它们翻掉。

输出的两种变化都要分开看:
  · downgraded  refuted -> inconclusive  ← **这是修复**(假反例被拦住)
  · upgraded    inconclusive -> refuted ← **这是风险**(新门在放行)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agent.tools.research._provenance import judge_verdict  # noqa: E402

#: 证据强度序。**按强度而不是按 verdict 名分类** ——
#: 早先写死成「new == inconclusive 才算降级」, 于是 refuted -> partial
#: 被归到"风险"里, 报告读起来像新门放行了假结论, 实际它同样是修复。
_RANK = {"failed": 0, "untested": 1, "inconclusive": 2,
         "partial": 3, "refuted": 4, "proved": 5}

ROOT = os.path.join("data", "research", "working")
CASES = os.path.join("tests", "tools", "base20_v2_cases.json")


def res_of(rec):
    """★ 超时/被拒必须如实还原 ★
    一律 ok=True 会把「执行超时被终止」的题在回放里重判成 untested,
    于是报告谎称门改了 N 题 —— 实际只改了几题。
    """
    reason = rec.get("reason") or ""
    timed_out = "超时" in reason
    rejected = "静态门拒绝" in reason
    # ★ 必须把「执行报错」也还原成 not ok ★
    # judge_verdict 的第一优先级是 `not ok -> failed`。只还原超时和静态门拒绝的话,
    # 一条 AttributeError 的证据在回放里会变成 ok=True, 然后落到"没有 VERDICT
    # 标记"分支变成 untested —— 回放报告就会凭空多出一批"门改了 N 题",
    # 而实际那 N 题门根本没碰过。
    crashed = "代码执行出错" in reason or "没有产生可复核的结果" in reason
    return SimpleNamespace(
        stdout=rec.get("stdout") or "",
        stderr=rec.get("stderr") or "",
        ok=not (timed_out or rejected or crashed),
        rejected=rejected,
        timed_out=timed_out,
        error_type="回放重建" if crashed else "",
        code=rec.get("code") or "",
    )


def main() -> None:
    with io.open(CASES, encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    downgraded, upgraded, n = [], [], 0
    for c in cases:
        cid, claim = c["id"], c["prompt"]
        th = "v2_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
        p = os.path.join(ROOT, th, "_evidence.jsonl")
        if not os.path.isfile(p):
            print(f"  (无证据, 跳过) {cid}")
            continue
        for i, line in enumerate(io.open(p, encoding="utf-8")):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") != "code":
                continue
            n += 1
            old = rec.get("verdict")
            new, reason = judge_verdict(res_of(rec),
                                        code=rec.get("code") or "", claim=claim)
            if new == old:
                continue
            if _RANK.get(new, -1) < _RANK.get(old, -1):
                downgraded.append((cid, i, old, new))
                tag = "->"
            else:
                upgraded.append((cid, i, old, new))
                tag = "!!"
            print(f"{tag} {cid} [第{i}条] {old} -> {new}")
            print(f"     {reason.splitlines()[0][:150]}")

    print("\n" + "=" * 62)
    print(f"回放 {n} 条证据")
    print(f"  降级(假结论被拦下) {len(downgraded)}")
    print(f"  升级(新门在放行)   {len(upgraded)}")
    for cid, i, o, nv in downgraded:
        print(f"    修复: {cid} [第{i}条] {o} -> {nv}")
    for cid, i, o, nv in upgraded:
        print(f"    风险: {cid} [第{i}条] {o} -> {nv}")


if __name__ == "__main__":
    main()