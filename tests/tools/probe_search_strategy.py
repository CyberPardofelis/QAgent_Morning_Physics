"""验证一个判断: LLM 拿到「暴力搜索不可行」的题, 能不能自己想出剪枝算法?

用户的质疑(2026-10-08): 「剪枝算法之类的问题, 有 docker 啊, 算不出来让它想算法,
你都知道用剪枝了, 调用工具想不出来吗?」

我这个判断要验证, 不能假设。具体跑 V2 里那两道"暴力搜索不可行"的题:
  hard-003 三立方和 33:  解在 |x|~10^7 量级, 暴力枚举要 10^14 次
  hard-007 R(3,4)=9:     K8 有 2^28 种染色

看 LLM 拿到题目 + 现有 prompt, 会不会自己:
  1. 先估一下暴力要多少次
  2. 想到剪枝 / 分解 / 对称性去重
  3. 或者: 用 z3 / 现成库绕过去

跑法 (会花 token):
    set QAGENT_COST_GUARD=off
    set QAGENT_RESEARCH_EVAL_LLM=1
    python tests/tools/probe_search_strategy.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import agent.tools  # noqa: F401,E402
import agent.tools.research  # noqa: F401,E402
from agent.tools.llm import reset_usage, usage_snapshot  # noqa: E402
from agent.tools.registry import ToolRegistry  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

#: 这两道题, 关键不在"工具够不够", 在"LLM 会不会自己想到更好的手法"
PROBES = [
    ("hard-003", "求整数 x,y,z 使 x^3 + y^3 + z^3 = 33。"),
    ("hard-007", "验证 R(3,4)=9：证明任意 9 个顶点的红蓝完全图必有红色三角形"
                 "或蓝色 K4，并给出 8 个顶点无反例的着色。"),
]

#: 什么算"想到了搜索策略" —— 用来自动判读代码
STRATEGY_MARKERS = {
    "剪枝": ("prune", "pruning", "backtrack", "回溯", "剪枝"),
    "对称性去重": ("symmetr", "set(", "canonical", "去重"),
    "模运算筛选": ("%", "mod ", "modulo"),
    "分解问题": ("decompos", "factor", "split"),
    "用求解器": ("z3", "Solver", "sat"),
    "先估规模": ("10**", "10 **", "estimate", "complexity", "规模"),
}


def main() -> None:
    with io.open(os.path.join(HERE, "baseproblems_V2.txt"), encoding="utf-8") as f:
        items = {c["id"]: c for c in json.load(f)["items"]}

    verify = ToolRegistry.get("research_verify")
    for tag, prompt in PROBES:
        cid = [k for k in items if k.startswith(tag)][0]
        print("=" * 74)
        print(f"{cid}: {prompt}")
        print("=" * 74)
        reset_usage()
        t0 = time.time()
        th = "strategy_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
        try:
            r = verify.handler(claim=prompt, task_hash=th, timeout=180)
            verdict = r.get("verdict")
            summary = r.get("summary", "")
            cost = r.get("cost") or {}
        except Exception as e:
            verdict, summary, cost = "EXC", f"{type(e).__name__}: {e}", {}
        dt = time.time() - t0
        u = usage_snapshot()
        print(f"verdict: {verdict}  ({dt:.0f}s, {u['total_tokens']}tok)")
        print(f"summary: {summary[:300]}")

        # 读它实际写的代码, 看有没有搜索策略
        ws = os.path.join("data", "research", "working", th)
        evf = os.path.join(ws, "_evidence.jsonl")
        code = ""
        if os.path.exists(evf):
            with io.open(evf, encoding="utf-8") as f:
                lines = [l for l in f.read().splitlines() if l.strip()]
            if lines:
                code = json.loads(lines[-1]).get("code", "")
        if not code:
            print("\n(没有产出代码 —— 可能被 blocked 或三轮都失败)")
            continue

        print(f"\n--- 它写的代码({len(code.splitlines())} 行) ---")
        for line in code.splitlines()[:40]:
            print("   ", line)
        if len(code.splitlines()) > 40:
            print(f"    ... (共 {len(code.splitlines())} 行)")

        print("\n--- 有没有自己想到搜索策略 ---")
        low = code.lower()
        found = {k: [m for m in ms if m.lower() in low]
                 for k, ms in STRATEGY_MARKERS.items()}
        found = {k: v for k, v in found.items() if v}
        for k, v in found.items():
            print(f"  ✓ {k}: {v}")
        if not found:
            print("  ✗ 没有任何策略标记 —— 直接暴力或根本没跑")
        print()


if __name__ == "__main__":
    main()