"""收尾路径不许因为**数据形状意外**而抛异常。

## 这道测试在防什么

2026-10-10 实测(part2 零提示 20 题): B 路线的 res-05 和 res-17 两道
都记成了 `CRASH`, note 是 `TypeError: 'NoneType' object is not
subscriptable`, 而 traceback 指向

    orchestrator.py:294
      f"试过的方法: {[t.get('method', '')[:40] for t in trace ...]}"

## 根因: `dict.get(k, default)` 的 default **只在 key 不存在时**生效

Explorer 交上来的 JSON 里常是 `{"method": null}` —— **key 在, 值是 None**。
于是 `t.get('method', '')` 返回 `None`(不是 `''`), `None[:40]` 直接炸。

★ 为什么这个 bug 特别阴 ★
  它在**预算耗尽的收尾分支**上 —— 只有"模型没交出答案"时才会走到。
  于是本该记 `INCONCLUSIVE`(没交出答案, 一个**结论**) 的题,
  被记成 `CRASH`(程序炸了, 一个**基础设施故障**)。
  两道题各烧了 366s / 1038s, 最后 0 次工具调用就崩了 ——
  "模型不行"被伪装成了"程序出错"。

⇒ 两条判据:
  1. `_tried_methods` 遇到 method 为 None/缺 key 都**不许抛异常**;
  2. 更一般地: 用**假 LLM** 跑完一整道题, 拿到预算耗尽那条分支,
     断言它返回 INCONCLUSIVE 而不是抛异常。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.research.roles import orchestrator as OR  # noqa: E402

FAILS: list[str] = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_tried_methods_never_raises() -> None:
    print()
    print("=" * 78)
    print("① _tried_methods 遇到各种 method 形状都不许抛")
    print("=" * 78)
    cases = [
        ("key 不存在", {"role": "explorer"}),
        ("值是 None ★本题元凶★", {"role": "explorer", "method": None}),
        ("值是空串", {"role": "explorer", "method": ""}),
        ("正常字符串", {"role": "explorer", "method": "用 z3 求解"}),
        ("值是数字", {"role": "explorer", "method": 42}),
        ("值是列表", {"role": "explorer", "method": ["a", "b"]}),
    ]
    for label, entry in cases:
        try:
            got = OR._tried_methods([entry])
            ck(f"{label:22} 不抛异常", True, f"-> {got}")
        except Exception as e:  # noqa: BLE001
            ck(f"{label:22} 不抛异常", False, f"★ 抛了 {type(e).__name__}: {e}")

    # 非 explorer 的条目要被过滤掉
    got = OR._tried_methods([{"role": "worker", "method": None},
                             {"role": "explorer", "method": "x"}])
    ck("只取 explorer 条目", len(got) == 1 and got[0] == "x", str(got))


def test_budget_exhausted_returns_inconclusive() -> None:
    """用假 LLM 跑到预算耗尽, 断言拿到 INCONCLUSIVE 而不是异常。"""
    print()
    print("=" * 78)
    print("② 预算耗尽 → 返回 INCONCLUSIVE, 不抛异常 (端到端)")
    print("=" * 78)

    from agent.tools.research.roles import _util as U, worker as WK

    # ★ Explorer 每轮都交一个 method=null 的计划 ★
    #   这正是实测里让 res-05 / res-17 崩掉的那种形状。
    plan = ('{"read_first":"x","method":null,"tools":["numeric_code"],'
            '"rationale":null,"next_action":"execute","probe_spec":null,'
            '"done":false,"lanes":{"n":1}}')

    orig_ask, orig_call = U.ask, U._call_llm
    orig_exec, orig_probe = WK.run_execute, WK.run_probe

    # worker 交一个**没有** final 的结果 -> 每轮都"没交出答案" -> 预算耗尽
    def fake_run(*_a, **_k):
        return {"ran": True, "tool_calls": [], "lang": "tool",
                "error_type": None, "stderr": "", "has_artifact": False,
                "artifact": None, "shape_problems": [], "switches": [],
                "findings": {}, "final_answer": "",
                "stripped_self_claims": []}

    try:
        U.ask = lambda p, **_k: plan
        U._call_llm = lambda p, **_k: '{"tool":"numeric_code","args":{"code":"1+1"}}'
        WK.run_execute = fake_run
        WK.run_probe = fake_run

        res = OR.solve("题面", budget=U.Budget(max_turns=2),
                       verbose=False, verify=False,
                       provider="minimax", model="MiniMax-M3.1-Flash-Preview")
    except Exception as e:  # noqa: BLE001
        res = {"verdict": f"★抛异常★ {type(e).__name__}: {e}"}
    finally:
        U.ask, U._call_llm = orig_ask, orig_call
        WK.run_execute, WK.run_probe = orig_exec, orig_probe

    v = res.get("verdict")
    ck("没有抛异常", not str(v).startswith("★抛异常★"), str(v))
    ck("收尾分支返回 INCONCLUSIVE", v == "INCONCLUSIVE", f"实际 {v}")
    ck("note 里写出了试过的方法",
       "试过的方法" in str(res.get("note") or ""),
       str(res.get("note"))[:100])
    ck("note 里对 null 方法有兜底文案",
       "没写方法" in str(res.get("note") or ""),
       "应该显示 '(没写方法)' 而不是崩掉")


def main() -> int:
    print("=" * 78)
    print("收尾路径的健壮性 (method=null 必崩的回归)")
    print("=" * 78)
    test_tried_methods_never_raises()
    test_budget_exhausted_returns_inconclusive()
    print()
    print("=" * 78)
    print(f"失败 {len(FAILS)}")
    print("=" * 78)
    for f in FAILS:
        print(f"  ★ {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())