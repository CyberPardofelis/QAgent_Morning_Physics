"""量一次「验证一个简单命题」的真实成本 —— 判断多轮 ReAct 撑不撑得住的硬数据。

用户担心: 「回头要走多轮 ReAct 的, 小问题都复杂化, 复杂问题就解决不了。」
所以要先把「小问题现在到底多重」量出来。

对每道题记录: 总耗时 / 总 token / LLM 调用次数 / 涉及的工具调用数。
注意 0 LLM 层(run_code 直接给代码)也要量, 那是成本的下界。

跑法 (会花 token):
    set QAGENT_RESEARCH_EVAL_LLM=1
    python tests/tools/measure_cost_profile.py
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
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "data", "research", "cost_profile.json"))

# 三档难度, 各挑一题, 看看成本是不是随复杂度线性涨
PROBE = [
    ("简单·求个数", "求 1 到 100 中不能被 2、3、5 整除的数的个数。"),
    ("中·有限枚举", "验证 8x8 棋盘上 8 皇后问题的解数是 92。"),
    ("复杂·全域论证", "验证对所有 n, n^2 + n + 41 是素数。"),
]


def main() -> None:
    import agent.tools.research.research_verify as rv_mod

    verify = ToolRegistry.get("research_verify")
    rows = []

    for label, prompt in PROBE:
        th = "cost_" + hashlib.sha1(prompt.encode()).hexdigest()[:10]
        reset_usage()
        t0 = time.time()
        try:
            r = verify.handler(claim=prompt, task_hash=th, timeout=90)
            err = None
        except Exception as e:
            r, err = {}, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        u = usage_snapshot()
        ws = verify.handler.__module__  # 仅记录, 不做别的
        # 统计这一次到底调了几次 LLM: workspace 里的 attempts 数就是证据
        wspath = os.path.join("data", "research", "working", th, "_attempts.json")
        n_attempts = 0
        if os.path.exists(wspath):
            try:
                with io.open(wspath, encoding="utf-8") as f:
                    n_attempts = len(json.load(f))
            except Exception:
                n_attempts = -1
        row = {
            "label": label,
            "prompt": prompt,
            "verdict": r.get("verdict") if isinstance(r, dict) else "ERR",
            "seconds": round(dt, 1),
            "tokens": u["total_tokens"],
            "llm_http_calls": u["calls"],
            "code_attempts": n_attempts,
            "error": err,
        }
        rows.append(row)
        print(f"{label:14} {row['verdict']:12} {row['seconds']:>6}s "
              f"{row['tokens']:>7}tok  llm={row['llm_http_calls']} 代码尝试={n_attempts}", flush=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with io.open(OUT, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n=== 单次验证成本剖面 ===")
    for r in rows:
        print(f"{r['label']:14} {r['seconds']:>6}s {r['tokens']:>7}tok")
    if rows:
        avg_t = sum(r["seconds"] for r in rows) / len(rows)
        avg_k = sum(r["tokens"] for r in rows) / len(rows)
        print(f"\n平均: {avg_t:.1f}s / {avg_k:.0f} token")
        print(f"\n若一个研究任务走 N 轮 ReAct, 每轮调一次 research_verify:")
        for n in (3, 5, 10):
            print(f"  N={n:2}  ->  约 {n*avg_t:.0f}s, 约 {n*avg_k/1000:.0f}k token")
    print(f"\n明细写入 {OUT}")


if __name__ == "__main__":
    main()