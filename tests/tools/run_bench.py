"""两套题一起跑: part1 工具能力(36 题) + part2 科研能力(20 题)。

## 预算为什么分档
  part1  题面**明示工具**, 测的是"这个工具通不通"。通了就该 1 轮出结果,
         给 2-3 轮足够。给多了反而测不出"这个工具是不是一次就对"。
  part2  题面零提示, 测的是"会不会自己找路"。要多几轮换方法。

## 断点续跑
每跑完一题就往 OUT 追加一行并落盘。中途挂掉/被打断, 重跑会跳过已完成的。
56 道题按现在的速度要跑很久, 不断电比跑得快更重要。

## ★ part1 的一个额外判据 ★
part1 题面指定了 `tool_under_test`。**它有没有真的用上那个工具**, 是
工具能力测试的核心 —— 因为"用 Python 硬算也能交出答案"的题, 考的就不是
那个工具。这条比 correct 更有诊断价值, 所以单独统计。

## 统计口径
  PROVED / REFUTED   —— 独立复核给出了确定结论
  PARTIAL            —— 只验证了一部分
  INCONCLUSIVE       —— 预算耗尽仍无法判定（**这不是失败, 是诚实的收场**）
  轮次/用时/工具组合 —— 过程量
三样东西分列, 不合并成一个数字。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from agent.tools.research.roles import Budget, build_catalog, solve  # noqa: E402
from agent.tools.llm import reset_usage, usage_snapshot  # noqa: E402

LAB = os.path.join(ROOT, "data", "lab")
OUT = os.path.join(LAB, "bench_results.jsonl")

PARTS = {
    1: {"file": "part1_tool_capability.json", "turns": 3},
    2: {"file": "part2_research_capability.json", "turns": 5},
}


def load_items(part: int) -> list[dict]:
    return json.load(open(os.path.join(LAB, PARTS[part]["file"]),
                          encoding="utf-8"))["items"]


def done_ids() -> set[str]:
    if not os.path.exists(OUT):
        return set()
    out = set()
    for line in open(OUT, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.add(json.loads(line)["id"])
        except Exception:  # noqa: BLE001 - 断电会留下半行
            continue
    return out


def prompt_of(item: dict, part: int) -> str:
    """拼真正发给 agent 的题面。

    ★ part1 把 tool_hints 拼进去, 因为它测的就是"给了工具能不能用对";
      part2 **绝对不许**拼 —— 提示只能来自工具目录, 不来自题面。
    """
    p = item["prompt"]
    if part == 1:
        hint = item.get("tool_hints") or ""
        tools = item.get("ground_truth", {})
        extra = []
        if isinstance(tools, dict):
            for k, v in tools.items():
                if isinstance(v, str) and len(v) < 200:
                    extra.append(f"  {k}: {v}")
        tail = ""
        if hint or extra:
            tail = ("\n\n【本题指定使用】" + (f" {hint}" if hint else "")
                    + ("\n【已知事实】\n" + "\n".join(extra) if extra else ""))
        return p + tail
    # part2: 确认题面里没有方法提示, 有就直接拒绝跑这道题
    return p


def check_no_leak(item: dict) -> str | None:
    """part2 的题面若含方法提示, 跑出来的分数就不可信 —— 当场拦下。"""
    bad = [w for w in ("z3", "lean", "sympy", "cvxpy", "networkx", "scipy",
                       "ortools", "groebner", "ida*", "双向 bfs")
           if w in item["prompt"].lower()]
    return (f"题面里出现方法词 {bad}") if bad else None


def run_item(item: dict, part: int, catalog: list[dict], turns: int) -> dict:
    reset_usage()
    t0 = time.time()
    leak = check_no_leak(item) if part == 2 else None
    if leak:
        return {"id": item["id"], "part": part, "verdict": "LEAKED",
                "note": f"跳过: {leak}", "turns": 0, "seconds": 0.0}

    try:
        res = solve(prompt_of(item, part), budget=Budget(max_turns=turns),
                    tools=catalog, verbose=True)
    except Exception as e:  # noqa: BLE001
        return {"id": item["id"], "part": part, "verdict": "CRASH",
                "note": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-1200:], "turns": 0,
                "seconds": round(time.time() - t0, 1)}

    trace = res.get("trace") or []
    # ★ 统计代码自己崩了也不能拖垮整批 ★
    #   56 道题要跑很久, 前面 55 道的结果已经落盘了, 让最后一道的
    #   一个 len(generator) 把整批掀掉, 代价完全不成比例。
    #   所以这里逐项 try: 算不出来就记 None, 不中断。
    def _cnt(pred) -> int | None:
        try:
            return sum(1 for t in trace if pred(t))
        except Exception:  # noqa: BLE001
            return None

    explorers = [t for t in trace if t.get("role") == "explorer"]
    workers = [t for t in trace if t.get("role") == "worker"]
    probes = _cnt(lambda t: t.get("action") == "probe") or 0
    try:
        tool_combos = {tuple(sorted((t.get("tools") or []))) for t in explorers}
        used_all = {x for t in explorers for x in (t.get("tools") or [])}
    except Exception:  # noqa: BLE001
        tool_combos, used_all = set(), set()
    wanted = item.get("tool_under_test")

    return {
        "id": item["id"], "part": part,
        "category": item.get("category"),
        "verdict": res.get("verdict"),
        "artifact": res.get("artifact"),
        "evidence": (res.get("evidence") or "")[:1200],
        "turns": res.get("turns"), "seconds": res.get("seconds"),
        "budget_exhausted": res.get("budget_exhausted"),
        "note": res.get("note"),
        # 过程量
        "n_probe": probes,
        "n_execute": _cnt(lambda t: t.get("action") == "execute"),
        "method_combos": len(tool_combos),
        "tools_used": sorted(used_all),
        "n_worker_errors": _cnt(lambda t: bool(t.get("error_type"))),
        "stripped_self_claims": [s for t in workers
                                 for s in (t.get("stripped_self_claims") or [])],
        # part1 专属: 有没有真的用上被指定的工具
        "tool_under_test": wanted,
        "used_pinned_tool": (wanted in used_all) if wanted else None,
        "usage": usage_snapshot(),
        "trace": trace,
    }


def append(rec: dict) -> None:
    os.makedirs(LAB, exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        fh.flush()


def summarize(part: int) -> str:
    if not os.path.exists(OUT):
        return "(还没有结果)"
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("part") == part]
    if not rows:
        return "(还没有结果)"
    lines = [f"===== part{part} 汇总 ({len(rows)} 题) ====="]
    cnt: dict[str, int] = {}
    for r in rows:
        cnt[r["verdict"]] = cnt.get(r["verdict"], 0) + 1
    for k in ("PROVED", "REFUTED", "PARTIAL", "INCONCLUSIVE", "LEAKED",
              "CRASH"):
        if cnt.get(k):
            lines.append(f"  {k:14} {cnt[k]}")
    decided = sum(cnt.get(k, 0) for k in ("PROVED", "REFUTED"))
    lines.append(f"  得到确定裁决  {decided}/{len(rows)}")
    # ★ 分母要说清楚 ★ BACKEND_UNWIRED / LEAKED 是**回路的缺口或题库的错**,
    #   既不是模型不会, 也不是题不可做。它们必须从正确率分母里拿掉,
    #   留在分母里就是污染测量 —— 这跟"环境不具备"是同一个道理。
    excluded = cnt.get("BACKEND_UNWIRED", 0) + cnt.get("LEAKED", 0) + cnt.get("CRASH", 0)
    real = len(rows) - excluded
    if real:
        lines.append(f"  ★ 可计分母 {real} 题(已剔除 {excluded} "
                     f"道回路缺口/题库错误)")
        if cnt.get("BACKEND_UNWIRED"):
            ids = [r["id"] for r in rows if r["verdict"] == "BACKEND_UNWIRED"]
            lines.append(f"    后端未接入: {', '.join(ids[:6])}")
    lines.append(f"  用过 probe 的题数 {sum(1 for r in rows if r.get('n_probe'))}")
    lines.append(f"  平均轮数 {sum(r.get('turns') or 0 for r in rows)/len(rows):.1f}")
    lines.append(f"  平均用时 {sum(r.get('seconds') or 0 for r in rows)/len(rows):.0f}s")
    if part == 1:
        hit = sum(1 for r in rows if r.get("used_pinned_tool"))
        have = sum(1 for r in rows if r.get("tool_under_test"))
        lines.append(f"  ★ 真的用上了指定工具 {hit}/{have}  "
                     f"(没用上 = 这道题考的不是那个工具)")
    else:
        combos = [r.get("method_combos") for r in rows if r.get("method_combos")]
        if combos:
            lines.append(f"  平均方法组合数 {sum(combos)/len(combos):.1f}")
        st = [s for r in rows for s in (r.get("stripped_self_claims") or [])]
        lines.append(f"  ★ 被剥掉的自评字段 {len(st)} 处"
                     + (f"  例: {st[:3]}" if st else ""))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0, help="1 / 2 / 0=全部")
    ap.add_argument("--turns1", type=int, default=PARTS[1]["turns"])
    ap.add_argument("--turns2", type=int, default=PARTS[2]["turns"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    if args.fresh and os.path.exists(OUT):
        os.remove(OUT)

    catalog = build_catalog()
    parts = [args.part] if args.part in (1, 2) else [1, 2]
    seen = done_ids()

    for part in parts:
        items = load_items(part)
        if args.limit:
            items = items[: args.limit]
        turns = args.turns1 if part == 1 else args.turns2
        print("=" * 74)
        print(f"part{part}: {len(items)} 题, 每题最多 {turns} 轮")
        print("=" * 74, flush=True)
        for i, it in enumerate(items, 1):
            if it["id"] in seen:
                print(f"[{i}/{len(items)}] {it['id']}  (已跑过, 跳过)", flush=True)
                continue
            print(f"\n[{i}/{len(items)}] {it['id']}", flush=True)
            try:
                rec = run_item(it, part, catalog, turns)
            except Exception as e:  # noqa: BLE001
                # 单题崩了就记一笔继续, 绝不让整批停下来
                rec = {"id": it["id"], "part": part, "verdict": "CRASH",
                       "note": f"{type(e).__name__}: {e}",
                       "tb": traceback.format_exc()[-800:],
                       "turns": 0, "seconds": 0.0, "trace": []}
            append(rec)
            print(f"    -> {rec['verdict']}  轮数={rec.get('turns')} "
                  f"用时={rec.get('seconds')}s", flush=True)
        print()
        print(summarize(part), flush=True)

    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())