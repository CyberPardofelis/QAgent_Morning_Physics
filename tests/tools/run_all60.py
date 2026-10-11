"""★ 60 题全量跑测: base 20 + V2 20 + R 20 ★

## 为什么必须三个套一起跑

用户原话: 「别后面这些整得热火朝天 前面狗熊掰棒子 拿一个丢一个」。

真的发生了。今天删掉了 Pell 的技法提示(它给的是 V2 hard-001 的答案),
把 `io` 移出沙箱白名单, 改了 COMPOUND_CLAIM —— 这些全都影响 base/V2,
而我只反复跑 R 套。只跑 R 套的话, 我会漏掉"删提示反而让 V2 掉了几题"。

## 判据的分级 —— 必须写在报告里, 否则会误读

| 套 | 判据来源 | 报告里的标记 |
|---|---|---|
| R  | **独立验证器**(自己重新算, 不读 agent 代码与自由文本) | `独立` |
| base/V2 | 只有 A 线机器判定(正则门)+ agent 自述 | `自判` |

★ base/V2 没有独立验证器, 所以它们的 correct 是**自报**的 ★
这不是它们的问题, 是"还没写验证器"。但混在一起报一个通过率就是骗人 ——
所以每一行都带判据等级, 汇总也分开算。
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
from run_r_scored import cert_for_trace  # noqa: E402
from agent.tools.research._flags import RESEARCH_ARCH_VERSION  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "research", "all60_scored.json")
SUITE_TAG = {"base": "b", "V2": "v2", "R": "r"}
RANK = {"failed": 0, "untested": 1, "inconclusive": 2,
        "partial": 3, "refuted": 4, "proved": 5}


def th_of(suite: str, cid: str) -> str:
    return SUITE_TAG[suite] + "_" + hashlib.sha1(cid.encode()).hexdigest()[:10]


def best_evidence(suite: str, cid: str) -> dict | None:
    p = os.path.join("data", "research", "working", th_of(suite, cid),
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
    ap.add_argument("--suites", default="base,V2,R")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    suites = [s for s in args.suites.split(",") if s]
    want = set(args.only.split(",")) if args.only else None

    verify = ToolRegistry.get("research_verify")
    rows = []
    t0_all = time.time()

    for suite in suites:
        for it in SB.load_suite(suite):
            cid = it["id"]
            if want and cid not in want:
                continue
            th = th_of(suite, cid)
            claim = PS.build_claim(it["prompt"], it.get("expected"),
                               PS.data_hint_for(cid),
                               problem_id=cid)
            ws = os.path.join("data", "research", "working", th)
            if os.path.isdir(ws):
                shutil.rmtree(ws)
            reset_usage()
            t0 = time.time()
            err = None
            # ★ 必须传 answer_fields ★
            # 不传的话: prompt 里要求交制品(claim 里写了格式约定),
            # 但机器侧那道"你没交, 格式是这样"的门**根本不运行** ——
            # 模型收不到任何反馈, 只能盲重试或放弃, 最后验证器看到
            # "没制品" 自然判 0。
            # 实测 2026-10-09: 这一轮 R 套 15 道"错"里有 7 道是"没交制品",
            # 其中 res-017 / res-020 的 refuted 其实**是正确答案**(n=18 / n=3),
            # 只是没按规定格式交, 验证器无从核对。根因就是这里少传了一个参数。
            _exp = it.get("expected") or {}
            _val = _exp.get("value") if isinstance(_exp, dict) else None
            _fields = list(_val.keys()) if isinstance(_val, dict) else []
            # answer_spec 传的是**脱敏后**的格式约定(题面里已有的信息不加进去)
            _spec = (PS.expected_to_spec(_exp, it["prompt"])
                     if _fields else "")
            try:
                r = verify.handler(claim=claim, task_hash=th,
                                   timeout=args.timeout,
                                   answer_fields=_fields,
                                   answer_spec=_spec)
            except Exception as e:
                r, err = {}, f"{type(e).__name__}: {e}"
            dt = time.time() - t0
            u = usage_snapshot()
            print(f"[{suite}] {cid}  agent={r.get('verdict')} "
                  f"{dt:.0f}s {u['total_tokens']}tok", flush=True)
            if err:
                print("    err:", err, flush=True)

            ev = best_evidence(suite, cid)
            row = {"suite": suite, "id": cid,
                   "agent_verdict": (ev or {}).get("verdict"),
                   "evidence_kind": (ev or {}).get("kind"),
                   # ★ 架构版本 ★ 这行不带的话, 架构重构后新旧分数会被
                   # 放在一起比, 而题库/判据/verifier 一动那个对比就无效。
                   # 旧证据没有这个字段 -> 记 "unstamped", 不假装它属于哪一版。
                   "arch_version": (ev or {}).get("arch_version") or "unstamped",
                   "seconds": round(dt, 1), "tokens": u["total_tokens"],
                   "judged_by": "none", "category": "unverified",
                   "score": None}
            if ev is None:
                row["why"] = "NO_EVIDENCE"
                rows.append(row)
                _dump(rows)
                continue

            # ---- 判据分级 ----
            v = None
            if cid in VM.PROBLEM_TO_VERIFIER:
                try:
                    v = VM.load_verifier(cid).verify(
                        ev.get("stdout") or "", ev.get("code") or "",
                        it["prompt"])
                    row["judged_by"] = "独立"
                except Exception as e:
                    row["judged_by"] = "独立-异常"
                    row["why"] = f"{type(e).__name__}: {e}"
            else:
                row["judged_by"] = "自判"

            spec = SB.to_task_spec(it)
            tr = SB.trace_from_evidence(it, spec, task_hash=th)
            if tr is not None:
                if v is not None:
                    tr.final_correct = bool(v.get("correct"))
                    tr.certificate = cert_for_trace(v)
                    row["category"] = "ok" if v.get("correct") else "wrong"
                    row["cert"] = (v.get("certificate") or "")[:180]
                else:
                    # 只有自判 —— 明确标注, 不与"独立验证"混算
                    tr.final_correct = None
                    tr.certificate = None
                    row["category"] = "self_reported"
                    row["self_verdict"] = ev.get("verdict")
                fa = v.get("final_answer") if v else None
                if fa is not None:
                    tr.final_answer = json.dumps(fa, ensure_ascii=False)[:300]
                rep = SB.RS.score_trace(tr, spec)
                row["score"] = rep.total
                row["breakdown"] = rep.breakdown
                row["weak_dims"] = rep.weak_dims
            rows.append(row)
            _dump(rows)

    report(rows, time.time() - t0_all)


def _dump(rows: list) -> None:
    # ★ 顶层也写一份 arch_version ★ 以后拿到这份 json 时, 第一眼就知道
    # 这些数字是哪一版架构跑的, 不用去猜。
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"arch_version": RESEARCH_ARCH_VERSION,
         "arch_version_note": "每行 evidence 自带 arch_version; "
                              "为 unstamped 说明该证据产生于打戳之前",
         "rows": rows}, ensure_ascii=False, indent=2))


def report(rows: list, elapsed: float) -> None:
    print("\n" + "=" * 78)
    print(f"60 题全量结果 ({elapsed:.0f}s)")
    print("=" * 78)
    # ★ 架构版本打头 —— 不然这张表和下一张表的数字会被直接对比 ★
    stamps: dict[str, int] = {}
    for r in rows:
        stamps[r.get("arch_version") or "unstamped"] = \
            stamps.get(r.get("arch_version") or "unstamped", 0) + 1
    print(f"架构版本: {RESEARCH_ARCH_VERSION}  "
          f"(证据打戳情况 {stamps})")
    print("=" * 78)
    for suite in ("base", "V2", "R"):
        rs = [r for r in rows if r["suite"] == suite]
        if not rs:
            continue
        ok = sum(1 for r in rs if r["category"] == "ok")
        wrong = sum(1 for r in rs if r["category"] == "wrong")
        selfr = sum(1 for r in rs if r["category"] == "self_reported")
        unver = sum(1 for r in rs if r["category"] == "unverified")
        tag = "独立验证" if suite == "R" else "仅自判(无验证器)"
        print(f"\n[{suite}] {tag}")
        print(f"  独立验证通过 {ok} | 独立验证不通过 {wrong} | "
              f"只有自判 {selfr} | 没跑到证据 {unver}")
        if suite != "R":
            proved = sum(1 for r in rs
                         if r.get("self_verdict") in ("proved", "refuted"))
            print(f"  ← 这套**没有独立验证器**, 只能报\"agent 自称 "
                  f"proved/refuted 的有 {proved} 题\", 不能当正确率读")

    scored = [r for r in rows if r.get("score") is not None]
    if scored:
        dims: dict[str, list[float]] = {}
        for r in scored:
            for k, v in (r.get("breakdown") or {}).items():
                dims.setdefault(k, []).append(v)
        print("\n" + "-" * 78)
        print("维度均分(新适配器, 真实翻译)")
        for k, vs in sorted(dims.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            a = sum(vs) / len(vs)
            print(f"  {k:20} {a:5.2f}  {'#' * int(a * 24)}")

    print("\n" + "-" * 78)
    print("逐题  (arch=证据自带的架构版本)")
    for r in rows:
        mark = {"ok": "OK ", "wrong": "!! ", "self_reported": "自 ",
                "unverified": "?? "}.get(r["category"], "?  ")
        jb = r["judged_by"]
        sc = f"{r['score']:5.1f}" if r.get("score") is not None else "  -  "
        print(f"  {mark}{r['suite']:5}{r['id']:38} 自判="
              f"{str(r['agent_verdict']):12} 判据={jb:8} 分={sc} "
              f"arch={r.get('arch_version', '?')}")
    print("=" * 78)
    _dump(rows)


if __name__ == "__main__":
    main()