"""在任意题库套件上跑当前 research_verify, 落证据 + 打分。

## 为什么要分档

- `--suite R --pilot`  : 只跑 5 道代表题, 用来验证测量通路 + 摸清失败点
- `--suite R`          : 跑满 20 题
- `--suite all`        : 60 题

题库里的 recommended_tools **只给打分器做参照, 不进 prompt** —— 否则
就成了"给答案的提示"。用户明确要求: 依赖已知结果+提示的考法不够好。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent.tools  # noqa: E402,F401
import agent.tools.research  # noqa: E402,F401
from agent.tools.llm import reset_usage, usage_snapshot  # noqa: E402
from agent.tools.registry import ToolRegistry  # noqa: E402

import score_baseline as SB  # noqa: E402

#: 试飞用的 5 道 —— 覆盖不同领域 + 不同难点类型
PILOT = [
    "res-001-ramsey-34-lower-bound",     # SAT + 证书
    "res-003-hadamard-28-paley",         # 构造 + 交叉验证
    "res-012-minimal-polynomial",        # 符号代数
    "res-017-counterexample-hunt",       # 反例搜索
    "res-020-hypothesis-refine-loop",    # 完整研究循环
]


def task_hash(cid: str, suite: str) -> str:
    tag = {"R": "r", "V2": "v2", "base": "b"}.get(suite, "x")
    return f"{tag}_{hashlib.sha1(cid.encode()).hexdigest()[:10]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="R", choices=["base", "V2", "R", "all"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--only", default="", help="逗号分隔的题号")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args()

    suites = list(SB.SUITES) if args.suite == "all" else [args.suite]
    items: list[tuple[str, dict]] = []
    for s in suites:
        for it in SB.load_suite(s):
            if args.pilot and it["id"] not in PILOT:
                continue
            if args.only and it["id"] not in args.only.split(","):
                continue
            items.append((s, it))

    verify = ToolRegistry.get("research_verify")
    t_all = time.time()
    for n, (s, it) in enumerate(items, 1):
        cid, prompt = it["id"], it["prompt"]
        th = task_hash(cid, s)
        ws = os.path.join("data", "research", "working", th)
        if os.path.isdir(ws):
            shutil.rmtree(ws)        # 干净重跑, 否则读到上一轮
        print(f"[{n}/{len(items)}] {cid} ...", flush=True)
        reset_usage()
        t0 = time.time()
        err = None
        try:
            r = verify.handler(claim=prompt, task_hash=th, timeout=args.timeout)
        except Exception as e:            # 异常原文必须回传, 不吞
            r, err = {}, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        u = usage_snapshot()
        v = r.get("verdict") or "NONE"
        print(f"    {v:14} {dt:>5.0f}s {u['total_tokens']:>6}tok "
              f"{u['calls']}次调用", flush=True)
        if not args.no_score:
            spec = SB.to_task_spec(it)
            tr = SB.trace_from_evidence(it, spec)
            if tr is not None:
                tr.final_correct = None
                rep = SB.RS.score_trace(tr, spec)
                print(f"    过程分 {rep.total:5.1f} | 短板 {rep.weak_dims}",
                      flush=True)
        if err:
            print("    err:", err, flush=True)

    print(f"\n总耗时 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()