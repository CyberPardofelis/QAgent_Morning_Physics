"""R 套题端到端: 跑 -> 独立验证 -> 打分 -> 汇总。

★ 与上一版的区别 ★
上一版只把 `prompt` 送进去, 于是 agent 交了制品但字段名与 verifier 对不上,
20 题里 14 道判"错"、其中 11 道其实是假阴性。现在 claim 里带上了
题库的 `expected`(输出格式约定)和数据文件位置。

★ 统计口径(严格分开) ★
  ok / wrong   —— 独立验证的结论
  unverified   —— 压根没跑到有判定的证据, **不是判错**
  missing      —— verifier 缺失或抛异常, **环境问题, 不是判错**
  ★ Lean 路径不落 code 类证据, 所以形式化题会落进 unverified ——
    那也是"没验到"而不是"判错", 后面单独补 Lean 证据的取用。
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

import problem_spec as PS  # noqa: E402
import score_baseline as SB  # noqa: E402
import verifier_map as VM  # noqa: E402
from agent.tools.research._flags import RESEARCH_ARCH_VERSION  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "research", "r_scored.json")
RANK = {"failed": 0, "untested": 1, "inconclusive": 2,
        "partial": 3, "refuted": 4, "proved": 5}


def th_of(cid: str) -> str:
    return "r_" + hashlib.sha1(cid.encode()).hexdigest()[:10]


def cert_for_trace(v: dict) -> str | None:
    """**只有判对才有证书**。失败诊断串不算证书, 返回 None。

    ★ 为什么需要这个函数(而不是就地写一行) ★
    打分器 `score_certification` 的规则是「certificate 非空 -> 1.0」, 它**不看**
    `final_correct`。而实测 2026-09: 20 个 verifier 里有 **15 个**在
    `correct=False` 时也返回一段非空的失败诊断串
    ("FAILED: 未提交制品" / "artifact_present: ...")。
    于是 **失败诊断拿了满分证书**, 方向完全反了。

    ★ 为什么修在适配器而不是改 15 个 verifier ★
    「证书只在判对时存在」是**评分契约**的性质, 不是每个验证器各自的义务。
    逐个改的话, 下一个新写的 verifier 一定会忘 —— 这是"逐个打补丁"的典型后果。
    失败诊断不丢: 它仍然写进报告的 `certificate` 字段供人看, 只是不进评分。
    """
    return (v.get("certificate") or None) if v.get("correct") else None


def best_evidence(cid: str) -> dict | None:
    """取这道题历史**最好**的一条证据。**所有 kind 都要看**。

    ★ 为什么不能再按 kind=="code" 过滤 ★
    实测事故(2026-10-09): Lean 走 `_verify_via_lean`, 落的是 kind=="lean",
    code/stdout 字段都齐全。旧代码只认 kind=="code" -> res-008/009/018
    三道**从头到尾没进过 verifier**, 直接落进 unverified。
    它们的证据就在磁盘上好好的 —— 是我筛掉了。

    "没跑到"和"判错"在统计上长得一样, 所以证据类型筛错就是静默漏网。
    """
    p = os.path.join(ROOT, "data", "research", "working", th_of(cid),
                     "_evidence.jsonl")
    if not os.path.isfile(p):
        return None
    best = None
    for line in io.open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("stdout"):
            continue
        if best is None or RANK.get(r.get("verdict"), -1) > RANK.get(
                best.get("verdict"), -1):
            best = r
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--only", default="")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    items = SB.load_suite("R")
    if args.only:
        want = set(args.only.split(","))
        items = [it for it in items if it["id"] in want]

    verify = ToolRegistry.get("research_verify")
    rows = []
    t_all = time.time()

    for n, it in enumerate(items, 1):
        cid, prompt = it["id"], it["prompt"]
        th = th_of(cid)
        claim = PS.build_claim(prompt, it.get("expected"),
                               PS.data_hint_for(cid),
                               problem_id=cid)
        if not args.score_only:
            ws = os.path.join(ROOT, "data", "research", "working", th)
            if os.path.isdir(ws):
                shutil.rmtree(ws)
            reset_usage()
            t0 = time.time()
            err = None
            try:
                exp = it.get("expected") or {}
                val = exp.get("value") if isinstance(exp, dict) else None
                fields = list(val.keys()) if isinstance(val, dict) else []
                r = verify.handler(claim=claim, task_hash=th,
                                   timeout=args.timeout,
                                   answer_fields=fields,
                                   answer_spec=(PS.expected_to_spec(
                                       exp, it["prompt"])
                                       if fields else ""))
            except Exception as e:
                r, err = {}, f"{type(e).__name__}: {e}"
            dt = time.time() - t0
            u = usage_snapshot()
            print(f"[{n}/{len(items)}] {cid}\n    agent自判="
                  f"{r.get('verdict')} {dt:.0f}s {u['total_tokens']}tok",
                  flush=True)
            if err:
                print("    err:", err, flush=True)
        else:
            print(f"[{n}/{len(items)}] {cid} (只打分)", flush=True)

        ev = best_evidence(cid)
        row = {"id": cid, "agent_verdict": (ev or {}).get("verdict"),
               # ★ 架构版本 ★ 不带它, 重构后新旧分数会被直接对比而无效
               "arch_version": (ev or {}).get("arch_version") or "unstamped",
               "category": "unverified", "score": None,
               "verifier_called": False, "evidence_kind": (ev or {}).get("kind")}
        if ev is None:
            # ★ 明确区分「压根没跑到证据」与「有证据但没验」★
            row["why"] = "NO_EVIDENCE"
            rows.append(row)
            io.open(OUT, "w", encoding="utf-8").write(json.dumps(
                {"arch_version": RESEARCH_ARCH_VERSION, "arch_version_note": "逐题 evidence 另带 arch_version; unstamped = 打戳前产生的旧证据", "rows": rows}, ensure_ascii=False, indent=2))
            continue

        try:
            mod = VM.load_verifier(cid)
            v = mod.verify(ev.get("stdout") or "", ev.get("code") or "", prompt)
            row["verifier_called"] = True
        except Exception as e:
            row["category"] = "missing"
            row["reason"] = f"{type(e).__name__}: {e}"
            row["why"] = "VERIFIER_ERROR"
            rows.append(row)
            print(f"    verifier 异常 -> {row['reason'][:70]}", flush=True)
            continue

        row["category"] = "ok" if v.get("correct") else "wrong"
        row["certificate"] = (v.get("certificate") or "")[:220]
        row["details"] = {k: str(x)[:80] for k, x in (v.get("details") or {}).items()}

        spec = SB.to_task_spec(it)
        tr = SB.trace_from_evidence(it, spec, task_hash=th)
        if tr is not None:
            tr.final_correct = bool(v.get("correct"))
            tr.certificate = cert_for_trace(v)      # 见 cert_for_trace 的说明
            fa = v.get("final_answer")
            tr.final_answer = (json.dumps(fa, ensure_ascii=False)[:300]
                               if fa is not None else None)
            rep = SB.RS.score_trace(tr, spec)
            row["score"] = rep.total
            row["breakdown"] = rep.breakdown
            row["weak_dims"] = rep.weak_dims
            print(f"    独立验证={row['category']:9} 过程分={rep.total:5.1f} "
                  f"短板={rep.weak_dims}", flush=True)
        rows.append(row)
        io.open(OUT, "w", encoding="utf-8").write(json.dumps(
            {"arch_version": RESEARCH_ARCH_VERSION, "arch_version_note": "逐题 evidence 另带 arch_version; unstamped = 打戳前产生的旧证据", "rows": rows}, ensure_ascii=False, indent=2))

    ok = sum(1 for r in rows if r["category"] == "ok")
    wrong = sum(1 for r in rows if r["category"] == "wrong")
    unver = sum(1 for r in rows if r["category"] == "unverified")
    miss = sum(1 for r in rows if r["category"] == "missing")
    scored = [r for r in rows if r.get("score") is not None]
    print("\n" + "=" * 74)
    print(f"R 套题独立验证结果 ({time.time()-t_all:.0f}s)")
    # ★ 架构版本打头 —— 不然这张表和别的架构那张表会被直接对比 ★
    stamps: dict[str, int] = {}
    for r in rows:
        k = r.get("arch_version") or "unstamped"
        stamps[k] = stamps.get(k, 0) + 1
    print(f"架构版本: {RESEARCH_ARCH_VERSION}   证据打戳: {stamps}")
    print("=" * 74)
    print(f"  独立验证通过 {ok}/{len(rows)}")
    print(f"  做错         {wrong}   ← agent 真的错了")
    print(f"  没跑到证据   {unver}   ← 不是判错")
    print(f"  verifier 异常 {miss}")
    never = [r for r in rows if not r.get("verifier_called")]
    if never:
        print(f"  ★ **验证器从未被调用的题** {len(never)}  "
              f"← 这些题分数无意义, 不是模型的问题:")
        for r in never:
            print(f"      {r['id']:38} {r.get('why','NO_EVIDENCE')} "
                  f"kind={r.get('evidence_kind')}")
    # 假阴性分离: 失败项里带 present/parsed/artifact 的 = 格式没对上
    fmt = [r for r in rows if r["category"] == "wrong"
           and any(k for k in (r.get("details") or {})
                   if any(t in k for t in ("present", "parsed", "artifact")))]
    print(f"  ★ 其中疑似**格式没对上**(非真错) {len(fmt)}")
    if scored:
        print(f"\n  平均过程分   "
              f"{sum(r['score'] for r in scored)/len(scored):.1f}")
        dims: dict[str, list[float]] = {}
        for r in scored:
            for k, v in (r.get("breakdown") or {}).items():
                dims.setdefault(k, []).append(v)
        for k, vs in sorted(dims.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            a = sum(vs) / len(vs)
            print(f"    {k:20} {a:5.2f}  {'#' * int(a * 24)}")
    print("\n  逐题:")
    for r in rows:
        mark = {"ok": "OK ", "wrong": "!! ", "unverified": "?? ",
                "missing": "MA "}[r["category"]]
        sc = f"{r['score']:5.1f}" if r.get("score") is not None else "  -  "
        print(f"  {mark}{r['id']:38} 自判={str(r['agent_verdict']):12}"
              f" 验证={r['category']:9} 分={sc} "f"arch={r.get('arch_version', '?')}")
    print("=" * 74)
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"arch_version": RESEARCH_ARCH_VERSION, "arch_version_note": "逐题 evidence 另带 arch_version; unstamped = 打戳前产生的旧证据", "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()