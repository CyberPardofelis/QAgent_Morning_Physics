"""用同样的题重跑, 验证 prompt 判据修复是否真的有效。

关键: 用**同样的题**重跑, 同样的 LLM, 比较前后。不是换题验证。
如果 6 道从非 proved 变成 proved, 说明是判据缺失; 如果没变,
说明是模型能力问题, 该老实承认。

跑法 (会花 token):
    set QAGENT_RESEARCH_EVAL_LLM=1
    python tests/tools/rerun_base20_failed.py
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
from agent.tools.registry import ToolRegistry  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "data", "research", "base20_rerun.json"))
GOLD = "proved"

# 第一轮没答对的 6 道
FAILED6 = [
    "num-001-euler-prime-polynomial",
    "num-002-goldbach-small",
    "alg-006-fibonacci-matrix",
    "alg-007-am-gm",
    "logic-011-de-morgan",
    "graph-014-ramsey-3-3",
]


def main() -> None:
    if os.environ.get("QAGENT_RESEARCH_EVAL_LLM") != "1":
        raise SystemExit("这轮会真花 LLM token, 请先 set QAGENT_RESEARCH_EVAL_LLM=1")
    with io.open(os.path.join(HERE, "baseproblems.txt"), encoding="utf-8") as f:
        cases = {c["id"]: c for c in json.load(f)["items"]}

    verify = ToolRegistry.get("research_verify")
    results = []
    for i, cid in enumerate(FAILED6, 1):
        c = cases[cid]
        th = "rerun_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
        print(f"[{i}/6] {cid} ...", flush=True)
        t0 = time.time()
        try:
            out = verify.handler(claim=c["prompt"], task_hash=th, timeout=90)
            err = None if isinstance(out, dict) else f"non-dict: {type(out).__name__}"
        except Exception as e:
            out, err = {}, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        verdict = (out.get("verdict") if isinstance(out, dict) else None) or "NONE"
        cost = (out.get("cost") or {}) if isinstance(out, dict) else {}
        row = {
            "id": cid, "gold": GOLD, "verdict": verdict, "match": verdict == GOLD,
            "seconds": round(dt, 1), "tokens": cost.get("tokens", 0),
            "llm_calls": cost.get("llm_calls", 0), "error": err,
            "summary": str(out.get("summary", ""))[:300] if isinstance(out, dict) else "",
        }
        results.append(row)
        print(f"{'OK ' if row['match'] else '!! '}{cid}: {verdict} ({dt:.0f}s, {row['tokens']}tok)", flush=True)
        # 增量落盘 —— 长跑脚本只在结尾写等于没写
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with io.open(OUT, "w", encoding="utf-8") as f:
            json.dump({"n_ok": sum(1 for r in results if r["match"]),
                       "total": len(results), "results": results}, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for r in results if r["match"])
    print(f"\n=== 修复后: {n_ok}/{len(results)} (修复前 0/{len(FAILED6)}) ===")
    for r in results:
        if not r["match"]:
            print(f"  仍未对 {r['id']}: {r['verdict']} — {(r['error'] or r['summary'])[:110]}")
    print(f"\n明细写入 {OUT}")


if __name__ == "__main__":
    main()