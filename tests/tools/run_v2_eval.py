"""V2 中级 20 题 —— 真跑一遍, 看现在到底能做对几道。

★ 这轮不美化数字 ★
用户问「这 40 题都能做好了吗」, 答案必须是实跑出来的, 不是推断的。
所以:
  · 跑不了的(需要 Mathlib)照样计入, 但单独标出 —— 不能藏
  · 被 cost_guard 阻断的和判错的分开统计
  · 每道题记录实际写了什么代码、判了什么, 便于逐题复盘

跑法:
    set QAGENT_COST_GUARD=off
    python tests/tools/run_v2_eval.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import agent.tools  # noqa: F401,E402
import agent.tools.research  # noqa: F401,E402  ← 纯为注册工具, 无直接引用
from agent.tools.llm import reset_usage, usage_snapshot  # noqa: E402
from agent.tools.registry import ToolRegistry  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_JSON = os.path.join(HERE, "base20_v2_cases.json")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "data", "research", "v2_results.json"))
#: 每题硬超时。有些题(如 Collatz 1e6、R(3,4)=9) 本身就要跑很久。
PER_CASE_TIMEOUT = 150


def main() -> None:
    if os.environ.get("QAGENT_COST_GUARD", "").lower() != "off":
        print("★ 建议先 set QAGENT_COST_GUARD=off —— 20 题要几万 token, "
              "撞限额会让一半题被静默阻断(看着像判错, 其实是没跑)。\n")

    with io.open(CASES_JSON, encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    verify = ToolRegistry.get("research_verify")
    results = []
    t_all = time.time()

    for i, c in enumerate(cases, 1):
        cid, prompt, gold = c["id"], c["prompt"], c["gold"]
        print(f"[{i}/20] {cid} (gold={gold}) ...", flush=True)
        reset_usage()
        t0 = time.time()
        th = "v2_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
        # ★ 清掉旧工作区 ★
        # 不清会读到上一轮的 _evidence.jsonl —— `_better_evidence` 是拿本轮
        # best_ev 对上文件里已有的证据, 于是「上一轮做对过」会混进这一轮的分数里。
        # 验收跑测要的是「现在这套代码从零跑能对几道」, 不是「历史上最好成绩」。
        ws = os.path.join("data", "research", "working", th)
        if os.path.isdir(ws):
            shutil.rmtree(ws)
        try:
            r = verify.handler(claim=prompt, task_hash=th, timeout=PER_CASE_TIMEOUT)
            err = None
            if not isinstance(r, dict):
                r, err = {}, f"non-dict: {type(r).__name__}"
        except Exception as e:
            # 异常原文必须回传, 不吞
            r, err = {}, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        u = usage_snapshot()

        verdict = r.get("verdict") or "NONE"
        blocked = bool(r.get("blocked")) or (dt < 1 and u["total_tokens"] == 0)

        # 读它实际写的代码, 看是真做了还是没做
        ws = os.path.join("data", "research", "working", th)
        code = ""
        evf = os.path.join(ws, "_evidence.jsonl")
        if os.path.exists(evf):
            with io.open(evf, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                code = json.loads(lines[-1]).get("code", "")

        row = {
            "id": cid, "gold": gold, "verdict": verdict,
            "seconds": round(dt, 1), "tokens": u["total_tokens"],
            "llm_calls": u["calls"], "blocked": blocked, "error": err,
            "summary": str(r.get("summary", ""))[:250],
            "code_lines": len(code.splitlines()),
            "wrote_code": bool(code.strip()),
        }
        results.append(row)
        print(f"    {verdict:14} {dt:>5.0f}s {u['total_tokens']:>6}tok "
              f"code={row['code_lines']}行", flush=True)

        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with io.open(OUT, "w", encoding="utf-8") as f:
            json.dump({"total": len(results), "results": results},
                      f, ensure_ascii=False, indent=2)

    report(results, time.time() - t_all)


def _judge(row: dict) -> str:
    """按各自的 gold 口径判对错, 不能一律当 proved 比。"""
    gold, v, blocked = row["gold"], row["verdict"], row["blocked"]
    if blocked:
        return "blocked"
    if gold == "lean_needed":
        # 需要 Mathlib, 当前环境跑不了 —— 单独归类, 不算 LLM 的错
        return "need_mathlib"
    if gold == "partial_ok":
        # 陷阱题: 给该范围内的结论或 partial 都算对; 判全域 proved 算错
        if v in ("proved", "partial", "inconclusive"):
            return "ok"
        return "wrong"
    # gold == proved
    if v == "proved":
        return "ok"
    if v == "partial":
        return "half"        # 有限范围验完了但没敢判 proved
    return "wrong"


def report(results: list[dict], elapsed: float) -> None:
    cats = Counter(_judge(r) for r in results)
    n = len(results)
    print(f"\n{'='*72}")
    print(f"V2 中级 20 题 —— 实跑结果 ({elapsed:.0f}s)")
    print(f"{'='*72}")
    print(f"  做对           {cats['ok']}/{n}")
    print(f"  部分(判了partial) {cats['half']}")
    print(f"  判错           {cats['wrong']}")
    print(f"  被阻断         {cats['blocked']}  ← 压根没跑, 不算判错")
    print(f"  需要 Mathlib   {cats['need_mathlib']}  ← 环境不具备, 不算 LLM 的错")

    real = n - cats["blocked"] - cats["need_mathlib"]
    if real:
        print(f"\n  去掉跑不了和缺环境的, 实际能做好的: "
              f"{cats['ok']}/{real} = {cats['ok']/real*100:.0f}%")

    print(f"\n{'='*72}")
    print("逐题")
    print(f"{'='*72}")
    for r in results:
        cat = _judge(r)
        flag = {"ok": "OK ", "half": "~  ", "wrong": "!! ",
                "blocked": "?? ", "need_mathlib": "MA "}[cat]
        print(f"{flag}{r['id']:36} gold={r['gold']:12} got={r['verdict']:14} "
              f"{r['seconds']:>5.0f}s {r['tokens']:>6}tok")

    print(f"\n{'='*72}")
    print("判错的题 —— 逐个看摘要")
    print(f"{'='*72}")
    for r in results:
        if _judge(r) in ("wrong", "half"):
            print(f"\n  {r['id']} -> {r['verdict']}")
            print(f"     {r['summary'][:200]}")

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"total": n, "results": results,
                   "categories": dict(cats)}, f, ensure_ascii=False, indent=2)
    print(f"\n明细写入 {OUT}")


if __name__ == "__main__":
    main()