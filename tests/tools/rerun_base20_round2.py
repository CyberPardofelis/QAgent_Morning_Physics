"""重跑第二轮仍失败的 2 道题 (LP / 二分法), 验证 REFUTED 判据修复。

跑法 (会花 token):
    set QAGENT_RESEARCH_EVAL_LLM=1
    python tests/tools/rerun_base20_round2.py
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
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "data", "research", "base20_rerun2.json"))
GOLD = "proved"
FAILED2 = ["opt-012-small-lp", "num-020-bisection-root"]


def main() -> None:
    if os.environ.get("QAGENT_RESEARCH_EVAL_LLM") != "1":
        raise SystemExit("这轮会真花 LLM token, 请先 set QAGENT_RESEARCH_EVAL_LLM=1")
    with io.open(os.path.join(HERE, "baseproblems.txt"), encoding="utf-8") as f:
        cases = {c["id"]: c for c in json.load(f)["items"]}

    verify = ToolRegistry.get("research_verify")
    results = []
    for i, cid in enumerate(FAILED2, 1):
        c = cases[cid]
        th = "r2_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
        print(f"[{i}/2] {cid} ...", flush=True)
        t0 = time.time()
        try:
            out = verify.handler(claim=c["prompt"], task_hash=th, timeout=120)
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
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with io.open(OUT, "w", encoding="utf-8") as f:
            json.dump({"n_ok": sum(1 for r in results if r["match"]),
                       "total": len(results), "results": results}, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for r in results if r["match"])
    print(f"\n=== 第二轮修复后: {n_ok}/{len(results)} (上一轮这 2 道是 0/2) ===")
    for r in results:
        if not r["match"]:
            print(f"  仍未对 {r['id']}: {r['verdict']} — {(r['error'] or r['summary'])[:120]}")
    print(f"\n明细写入 {OUT}")


if __name__ == "__main__":
    main()