"""B 路线的 probe 陷阱: 连续 probe 必须被强制切回 execute。

## 这道测试在防什么

2026-10-10 实测(part2 零提示 20 题, B 路线) 的统计:

    Explorer 动作分布 = execute 20 / probe 8 / None 4
    Worker 错误类型   = UnparsedOutput 12 / GaveUp 1 / RanOutOfSteps 2
    全程只 probe、一次 execute 都没有的题: res-09-tsp-200、res-19-bifurcation-rossler

★ 这不是"收敛机制弱", 是 probe 本身是个陷阱 ★
  probe 模式下 Worker **不强制 final** —— 那是设计意图(小规模试验本来
  就不该交答案)。于是形成死循环:

      probe -> 没交出制品 -> orchestrator continue
           -> Explorer 又说"我还不确定, 先 probe"
           -> 预算在 probe<->probe 之间烧光, 一件制品都没有

  预算只有 max_turns=3, 两轮 probe 就烧掉 2/3。

⇒ 修法: **连续 probe 超过 1 次就强制 execute**。
  这是 harness 计数, 不是靠 prompt 劝 —— 实测模型能**连着 6 次**无视
  Worker 那条"你已经连续 N 次调同一个工具, 不要再原样调它"的提示。
  软提醒不管用, 机制必须硬。

## 三条判据
  1. 第一次 probe **要放行**(probe 是有价值的, 不该禁)
  2. 第二次 probe **必须被改成 execute**(且 trace 里记下来)
  3. 改完之后 Worker 收到的是 **execute 语义**(run_execute, 强制 final)
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


def _mk_plan(action: str) -> str:
    return ('{"read_first":"x","method":"枚举","tools":["numeric_code"],'
            '"rationale":"r","next_action":"%s","probe_spec":"小规模试",'
            '"done":false}' % action)


def _run(action_seq: list[str], max_turns: int = 4):
    """按给定序列喂 Explorer 计划, 记录 Worker 实际被派了 probe 还是 execute。"""
    from agent.tools.research.roles import _util as U, worker as WK

    orig_ask, orig_call = U.ask, U._call_llm
    orig_exec, orig_probe = WK.run_execute, WK.run_probe
    got: list[str] = []
    seq = list(action_seq)

    def fake_ask(prompt, **_kw):
        return _mk_plan(seq.pop(0) if seq else "execute")

    def mk(kind):
        def _run_x(*_a, **_k):
            got.append(kind)
            # probe 模式**不交制品**(那是真实里 probe 的行为),
            # 于是 orchestrator 会 continue -> 下一轮还是 probe
            return {"ran": True, "tool_calls": [{"tool": "numeric_code"}],
                    "lang": "tool", "error_type": None, "stderr": "",
                    "has_artifact": False, "artifact": None,
                    "shape_problems": [], "switches": [], "findings": {},
                    "final_answer": "", "stripped_self_claims": []}
        return _run_x

    try:
        U.ask = fake_ask
        U._call_llm = lambda p, **_k: '{"tool":"numeric_code","args":{"code":"1"}}'
        WK.run_probe = mk("probe")
        WK.run_execute = mk("execute")
        res = OR.solve("题面", budget=U.Budget(max_turns=max_turns),
                       verbose=False, verify=False,
                       provider="minimax", model="MiniMax-M3.1-Flash-Preview")
    except Exception as e:  # noqa: BLE001
        res = {"verdict": f"抛异常 {type(e).__name__}: {e}", "trace": []}
    finally:
        U.ask, U._call_llm = orig_ask, orig_call
        WK.run_execute, WK.run_probe = orig_exec, orig_probe
    return got, res


def main() -> int:
    print("=" * 78)
    print("① 第一次 probe 放行")
    print("=" * 78)
    got, _ = _run(["probe", "probe"], max_turns=1)
    ck("第 1 轮确实是 probe", got[:1] == ["probe"], str(got))

    print()
    print("=" * 78)
    print("② 连续第二次 probe 被强制切 execute")
    print("=" * 78)
    got, res = _run(["probe", "probe"], max_turns=3)
    ck("第 2 轮派的是 execute 而不是 probe",
       len(got) >= 2 and got[1] == "execute",
       f"实际序列 {got}")
    ck("第二次确实发生了", len(got) >= 2, str(got))

    tr = res.get("trace") or []
    forced = [t for t in tr
              if "强制 execute" in str(t.get("outcome") or "")]
    ck("trace 里记下了'强制 execute'", len(forced) >= 1,
       f"找到 {len(forced)} 条")

    print()
    print("=" * 78)
    print("③ 连续三次 probe: 也不会连着 probe 两次以上")
    print("=" * 78)
    got3, _ = _run(["probe", "probe", "probe"], max_turns=4)
    ck("任何两次连续里最多只有一次 probe",
       all(not (got3[i] == got3[i + 1] == "probe")
           for i in range(len(got3) - 1)),
       f"实际序列 {got3}")

    print()
    print("=" * 78)
    print(f"失败 {len(FAILS)}")
    print("=" * 78)
    for f in FAILS:
        print(f"  ★ {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())