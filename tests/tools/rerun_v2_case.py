"""只重跑 V2 里的指定题, 用于验证单个改动 —— 不必每次都跑满 20 题。

用法:
    set QAGENT_COST_GUARD=off
    python tests/tools/rerun_v2_case.py hard-020-min-n-with-n-divisors
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import agent.tools                      # noqa: E402,F401
import agent.tools.research             # noqa: E402,F401
from agent.tools.llm import reset_usage, usage_snapshot  # noqa: E402
from agent.tools.registry import ToolRegistry  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v2_eval import _judge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "base20_v2_cases.json")
PER_CASE_TIMEOUT = 150


def main() -> int:
    wanted = set(sys.argv[1:])
    if not wanted:
        print("用法: python tests/tools/rerun_v2_case.py <题号…>")
        return 2
    with io.open(CASES, encoding="utf-8") as f:
        cases = [c for c in json.load(f)["cases"] if c["id"] in wanted]
    if not cases:
        print("没匹配到题目:", wanted)
        return 2

    verify = ToolRegistry.get("research_verify")
    n_ok = 0
    for c in cases:
        cid, prompt, gold = c["id"], c["prompt"], c["gold"]
        print(f"[{cid}] gold={gold}", flush=True)
        reset_usage()
        t0 = time.time()
        th = "v2_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
        ws = os.path.join("data", "research", "working", th)
        # 清掉旧工作区, 保证是干净重跑(否则会读到上一轮的历史证据)
        if os.path.isdir(ws):
            import shutil
            shutil.rmtree(ws)
        try:
            r = verify.handler(claim=prompt, task_hash=th,
                               timeout=PER_CASE_TIMEOUT)
            err = None
        except Exception as e:
            r, err = {}, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        u = usage_snapshot()
        v = r.get("verdict") or "NONE"
        # ★ 必须用 run_v2_eval 的同一套口径 ★
        # 原来这里写的是 `v == gold`, 于是 gold=partial_ok 的题
        # (Collatz 1e6) 即使判对了也会被算成失败 —— 单题复验说"没过",
        # 整轮跑测说"过了", 两份报告自相矛盾, 复验结论没法用。
        ok = _judge({"gold": gold, "verdict": v, "blocked": bool(
            r.get("blocked"))}) == "ok"
        n_ok += 1 if ok else 0
        print(f"  -> {v} (期望 {gold}) {dt:.0f}s {u['total_tokens']}tok "
              f"{u['calls']}次调用  {'OK' if ok else '!!'}")
        print(f"     {str(r.get('summary', ''))[:180]}")
        if err:
            print("     err:", err)
    print(f"\n{n_ok}/{len(cases)} 通过")
    return 0 if n_ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())