"""让 LLM 自己写验证器跑这 20 题 —— 真考验。

probe_base20.py 是手写代码, 20/20 只证明「沙箱 + sympy/networkx
能算对」, 不证明「LLM 能不能把题目翻译成可靠验证器」。后者才是
research_verify 的职责, 也是这个基准真正的用处。

每题用独立 task_hash 隔离工作区, 避免上一题的状态串到下一题。

跑法 (会真花 token):
    set QAGENT_RESEARCH_EVAL_LLM=1
    python tests/tools/run_base20_llm.py

判分: 这 20 题 gold 全是 proved —— 它们要么「验证某恒等式/性质」
要么「求出某确定值」, 目标都是证明题成立。所以任何 refuted /
inconclusive / failed 都算没答对, 不管它是不是"诚实的"。
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
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "data", "research", "base20_llm.json"))
GOLD = "proved"


def load_cases() -> list[dict]:
    with io.open(os.path.join(HERE, "baseproblems.txt"), encoding="utf-8") as f:
        return json.load(f)["items"]


def _dump(results: list[dict], n_ok: int) -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"n_ok": n_ok, "total": len(results),
                   "tokens": sum(r["tokens"] for r in results),
                   "gold": GOLD, "partial": True, "results": results},
                  f, ensure_ascii=False, indent=2)


def main() -> None:
    if os.environ.get("QAGENT_RESEARCH_EVAL_LLM") != "1":
        raise SystemExit("这轮会真花 LLM token, 请先 set QAGENT_RESEARCH_EVAL_LLM=1")
    cases = load_cases()
    verify = ToolRegistry.get("research_verify")
    results = []
    t_all = time.time()
    for i, c in enumerate(cases, 1):
        cid, prompt = c["id"], c["prompt"]
        th = "base20_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
        print(f"[{i}/20] {cid} ...", flush=True)
        t0 = time.time()
        try:
            out = verify.handler(claim=prompt, task_hash=th, timeout=60)
            err = None
            if not isinstance(out, dict):
                out, err = {}, f"non-dict return: {type(out).__name__}"
        except Exception as e:
            # 异常原文必须回传, 不吞
            out, err = {}, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        verdict = out.get("verdict") or "NONE"
        cost = out.get("cost") or {}
        row = {
            "id": cid,
            "gold": GOLD,
            "verdict": verdict,
            "match": verdict == GOLD,
            # ★ blocked: 上游把请求挡住了(配额/限流), 压根没跑 ★
            # 必须单独统计 —— 混进"判错"会让准确率虚高或虚低。
            # 2026-10-08 实踩: 12 道记成判错, 读 _attempts.jsonl 才发现
            # 因果链是 cost_guard 阻断 -> extract_code 返回空 -> 空转三轮。
            "blocked": bool(out.get("blocked", False)),
            "seconds": round(dt, 1),
            "tokens": cost.get("tokens", 0),
            "llm_calls": cost.get("llm_calls", 0),
            "error": err,
            "summary": str(out.get("summary", ""))[:400],
            "evidence_id": (out.get("all_evidence_ids") or [None])[-1],
        }
        results.append(row)
        flag = "OK " if row["match"] else "!! "
        print(f"{flag}{cid}: got={verdict} ({dt:.0f}s, {row['tokens']} tok)" + (f"  ERR={err}" if err else ""), flush=True)
        if not row["match"] and not err:
            print(f"      {row['summary'][:160]}", flush=True)
        # 每题立刻落盘: 上一次跑挂掉时因为只在结尾写, 进度全丢了。
        # 长跑任务必须增量写, 否则失败一次等于没跑。
        _dump(results, n_ok=sum(1 for r in results if r["match"]))

    # ★ 统计口径: 「没跑」和「判错」必须分开 ★
    # 2026-10-08 的教训: 12 道题被 cost_guard 阻断(0秒0token), 我按
    # "判错" 统计, 于是报出 8/20 —— 那个数字既不是真实能力也不是真实
    # 通过率, 是把两种东西混在一起的产物。混起来报还有个更坏的后果:
    # 数字看起来低, 会让人以为要继续加防线, 而实际上代码一个字都不用改。
    ran = [r for r in results if not r.get("blocked")]
    blocked = [r for r in results if r.get("blocked")]
    n_ok = sum(1 for r in ran if r["match"])
    total_tok = sum(r["tokens"] for r in results)
    print(f"\n{'='*70}")
    print(f"总题数        {len(results)}")
    print(f"  被上游阻断   {len(blocked)}  ← 压根没跑, 不算判错也不算对")
    print(f"  实际跑了     {len(ran)}")
    print(f"  其中答对     {n_ok}  → 准确率 {n_ok}/{len(ran)}"
          + (f" = {n_ok/len(ran)*100:.0f}%" if ran else ""))
    print(f"总耗时 {time.time()-t_all:.0f}s, 共 {total_tok} tokens")
    print(f"{'='*70}")
    if blocked:
        print(f"\n被阻断的 {len(blocked)} 道: {', '.join(r['id'] for r in blocked[:5])}"
              + (" ..." if len(blocked) > 5 else ""))
        print("  原因通常是 cost_guard 配额 —— 长跑评估前先设 QAGENT_COST_GUARD=off")
    for r in ran:
        if not r["match"]:
            print(f"  MISS {r['id']}: got={r['verdict']} tok={r['tokens']} "
                  f"{(r['error'] or r['summary'])[:120]}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"n_ok": n_ok, "total": len(results),
                   "n_blocked": len(blocked), "n_ran": len(ran),
                   "accuracy_over_ran": (n_ok / len(ran)) if ran else None,
                   "tokens": total_tok,
                   "gold": GOLD, "partial": False, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n明细写入 {OUT}")


if __name__ == "__main__":
    main()