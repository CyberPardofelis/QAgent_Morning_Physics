# -*- coding: utf-8 -*-
"""把新的 PROVED 侧机器门回放到 V2 已落盘的真实证据上, 看它到底改了多少判。

为什么回放而不是重跑:
  重跑会重新调 LLM, 结果不可比(LLM 非确定), 而且贵。
  证据链(_evidence.jsonl)里 code / stdout / claim 都原样存着, 门是纯函数,
  所以离线重算判定是**完全等价**的 —— 这也是把判定逻辑做成纯函数的价值。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, r"D:\pythonProject\QAgent")

from agent.tools.research._provenance import judge_verdict  # noqa: E402

ROOT = Path(r"D:\pythonProject\QAgent\data\research\working")


def res_of(rec):
    # ★ 超时/被拒必须如实还原 ★
    # 上一版这里一律 ok=True, 把一道"执行超时被终止"的题在回放里重判成了
    # untested, 于是回放报告谎称门改了 3 题 —— 实际只改了 1 题。
    # 这跟 cost_guard「阻断时返回错误字符串」是同一类坑: 外部信号看着正常。
    reason = rec.get("reason") or ""
    timed_out = "超时" in reason
    rejected = "静态门拒绝" in reason
    return SimpleNamespace(
        stdout=rec.get("stdout") or "",
        stderr=rec.get("stderr") or "",
        ok=not (timed_out or rejected),
        rejected=rejected,
        timed_out=timed_out,
        error_type="",
        code=rec.get("code") or "",
    )


def main():
    results = json.loads(
        (ROOT.parent / "v2_results.json").read_text(encoding="utf-8")
    )
    if isinstance(results, dict):
        results = results.get("results", [])
    # 题面
    spec = json.loads(
        Path(r"D:\pythonProject\QAgent\baseproblems_V2.txt").read_text(
            encoding="utf-8"
        )
    )
    claims = {p["id"]: p.get("prompt", "") for p in spec["items"]}

    changed = []
    total = 0
    for r in results:
        cid = r["id"]
        old = r["verdict"]
        # 找该题的工作目录
        claim = claims.get(cid, "")
        best = None
        for d in ROOT.glob("*"):
            st = d / "_status.json"
            if not st.is_file():
                continue
            try:
                q = json.load(open(st, encoding="utf-8")).get("question", "")
            except Exception:
                continue
            if q.strip() != claim.strip():
                continue
            ev = d / "_evidence.jsonl"
            if not ev.is_file():
                continue
            recs = [
                json.loads(x)
                for x in ev.read_text(encoding="utf-8", errors="replace").splitlines()
                if x.strip()
            ]
            recs = [x for x in recs if x.get("kind") == "code"]
            if recs:
                best = recs[-1]
                break
        if best is None:
            print(f"  (无证据, 跳过) {cid}")
            continue
        total += 1
        new, reason = judge_verdict(res_of(best), code=best.get("code") or "", claim=claim)
        if new != old:
            changed.append((cid, old, new, reason))
            print(f"!! {cid}: {old} -> {new}")
            print(f"   {reason.splitlines()[0]}")

    print("\n" + "=" * 62)
    print(f"回放 {total} 题, 判定被改 {len(changed)} 题")
    for cid, o, n, r in changed:
        print(f"  {cid}: {o} -> {n}")


if __name__ == "__main__":
    main()