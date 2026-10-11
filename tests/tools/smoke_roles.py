"""端到端 smoke: 拿 part2 里最便宜的一道题, 真跑一遍 Explorer/Worker/Verifier 循环。

## 为什么第一件事是 smoke 而不是跑 20 道
编排器一旦没跑过, 它就只是代码。跑 20 道才发现 Explorer 的 JSON 解析不出来,
等于白烧二十分钟 token 和一堆等待。

先拿一道**答案明确、且不依赖任何库**的题(res-06: n^2+n+41 是不是素数)
把三个角色各打一次, 看:
  · Explorer 能不能给出可解析的计划
  · Worker 能不能跑出代码、并且**只交答案不交自评**
  · Verifier 能不能自己重算并给出带证据的结论

预算给 3 轮 —— 够看清管线通不通, 又不会在管线有 bug 时烧太多。

## 用法
    python tests/tools/smoke_roles.py            # 跑 1 道
    python tests/tools/smoke_roles.py --all      # 跑完 part2 全部
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from agent.tools.research.roles import Budget, build_catalog, solve  # noqa: E402

BANK = os.path.join(ROOT, "data", "lab", "part2_research_capability.json")
OUT = os.path.join(ROOT, "data", "lab", "smoke_result.json")

#: smoke 用的题: 答案唯一、不需要外部数据、几秒内能算完
SMOKE_ID = "res-06-refute-fermat-poly"


def load_items() -> list[dict]:
    return json.load(open(BANK, encoding="utf-8"))["items"]


def run_one(item: dict, turns: int = 3) -> dict:
    print("=" * 74)
    print(f"{item['id']}  ({item.get('category')})")
    print(f"题面: {item['prompt'][:200]}")
    print(f"★ 工具提示字段 tool_hints = {item.get('tool_hints')!r}  "
          f"(必须是 None —— 提示只能来自目录, 不能来自题面)")
    print("=" * 74)
    res = solve(item["prompt"], budget=Budget(max_turns=turns),
                tools=build_catalog(), verbose=True)
    print("-" * 74)
    print(f"最终裁决: {res['verdict']}   轮数={res['turns']}  "
          f"用时={res['seconds']}s  预算耗尽={res['budget_exhausted']}")
    if res.get("artifact"):
        print(f"制品字段: {sorted(res['artifact'].keys())}")
    print("角色分工回顾:")
    for t in res["trace"]:
        print(f"  [{t['turn']}] {t['role']:9} {t.get('action') or '':8} "
              f"{str(t.get('verdict') or t.get('error_type') or '')[:40]}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--id", default=SMOKE_ID)
    args = ap.parse_args()

    items = load_items()
    if args.all:
        picked = items
    else:
        picked = [it for it in items if it["id"] == args.id]
        if not picked:
            print(f"题库里没有 {args.id}")
            return 1

    results = []
    for it in picked:
        results.append(run_one(it, args.turns))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"results": results}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)

    ok = sum(1 for r in results if r["verdict"] in ("PROVED", "REFUTED"))
    print("=" * 74)
    print(f"smoke: {len(results)} 道跑完, {ok} 道得到确定裁决, "
          f"{len(results) - ok} 道 INCONCLUSIVE/PARTIAL")
    print(f"结果落在 {OUT}")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())