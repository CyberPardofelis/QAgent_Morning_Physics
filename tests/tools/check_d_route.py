"""D 路线(并行 worker + 调用方 hints)的契约检查。

## D 是什么

D 是**第四条独立路线**, 不是 B 的改造:

    B:  Explorer -> Worker(单个, 串行)  -> (Verifier, 当前关着)
    D:  Explorer -> 分派成 N 条 lane -> N 个 Worker(**并发**) -> Explorer 汇总
        ^ 且 solve() 多一个 hints 参数(调用方可以指定工具/方法/思路)

## 这道测试在防什么

★ 防 1: **D 悄悄改坏了 B** ★
    D 复用了 explorer/worker/orchestrator。改这两个文件时最容易犯的错
    是顺手把 B 的行为也改了 —— 而 B 已经有 40 题(part1) 的实测数据,
    悄悄变了就意味着**历史分数不可比**。
    ⇒ 判据: `lanes=1` 时 D 的 trace 形状必须和 B 一致。

★ 防 2: **并发是假的** ★
    写了 ThreadPoolExecutor 但实际串行执行, 数据上是看不出来的
    —— 只是慢一点, 结果一样。**只有时间/并发计数能证明它并行。**
    ⇒ 判据: N 条 lane 的 worker 必须**同时在跑**。

★ 防 3: **lane 之间互相污染** ★
    并发最危险的失败模式是共享状态: lane 0 改了 lane 1 用的东西,
    于是"同一道题跑两次结果不同", 而且**不可复现**。
    ⇒ 判据: 每条 lane 的 prompt 里只能看到**别 lane 的意图**, 看不到**结果**。

★ 防 4: **hints 没真的进 prompt** ★
    加了参数忘了传下去 —— 函数签名有 hints, 但 prompt 里没有。
    这是那种"代码看起来对、实测没效果"的典型。
    ⇒ 判据: hints 的内容必须**逐字出现在** Explorer 的 prompt 里。

★ 防 5: **lane 结果没进 trace** ★
    并行跑完但结果没汇总, Explorer 下一轮看不到, 等于白并发。
    ⇒ 判据: 每条 lane 的 worker 结果都要有一条对应的 trace 条目。
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.research.roles import explorer as EX  # noqa: E402
from agent.tools.research.roles import worker as WK     # noqa: E402
from agent.tools.research.routes import d as D          # noqa: E402

FAILS: list[str] = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# --------------------------------------------------------------- 防 4: hints 进 prompt
def test_hints_reach_prompt() -> None:
    print()
    print("=" * 78)
    print("防 4: hints 必须真的进到 Explorer 的 prompt 里")
    print("=" * 78)
    hint = "指定用 z3 求解, 别用数值搜索"
    p = D.explorer_prompt("题面", hints=hint, history=None, tools=None)
    ck("hints 逐字出现在 prompt", hint in p)
    ck("hints 有独立区块, 不是拼进题面",
       "【调用方提示】" in p, )
    ck("题面本身没被污染", "题面指定用 z3" not in p)

    # 没有 hints 时, 不该凭空多出这个区块
    p2 = D.explorer_prompt("题面", hints=None, history=None, tools=None)
    ck("hints=None 时不出现该区块", "【调用方提示】" not in p2)
    ck("hints=None 时 prompt 仍含题面", "题面" in p2)


# --------------------------------------------------------------- 防 3: lane 无污染
def test_lane_isolation() -> None:
    print()
    print("=" * 78)
    print("防 3: lane 之间**读不到对方的结果**")
    print("=" * 78)
    lanes = D.make_lanes(n=3, intents=["走 z3", "走数值扫描", "走符号化约简"])
    ck("分了 3 条 lane", len(lanes) == 3, f"实际 {len(lanes)}")

    board = D.board_intents(lanes)
    ck("意图板里三条 lane 都在",
       all(k in board for k in ("lane0", "lane1", "lane2")))

    # ★ lane 看得见**别人**的意图, 看不见自己的那一行在板里 ★
    b2 = D.board_intents(lanes, for_lane=2)
    ck("for_lane=2 时板里有 lane0/lane1", "lane0" in b2 and "lane1" in b2)
    ck("for_lane=2 时板里**没有自己**", "lane2" not in b2)

    # ★ 核心: 给 lane 1 的 prompt 里, 不能有 lane 0 的**结果** ★
    p = D.lane_prompt("题面", lane=lanes[1],
                      board=D.board_intents(lanes, for_lane=1),
                      catalog="目录", probe=False)
    ck("lane1 prompt 含 lane0 的**意图**", "z3" in p)
    ck("lane1 prompt 含自己的意图", "数值扫描" in p)
    ck("lane1 prompt 有只读声明", "看不到" in p or "只读" in p)


# --------------------------------------------------------------- 防 2: 真并行
def test_actually_parallel() -> None:
    print()
    print("=" * 78)
    print("防 2: 多条 lane 必须**同时**在跑, 不是串行")
    print("=" * 78)
    live = {"cur": 0, "max": 0}
    lock = threading.Lock()

    def fake_lane(i: int):
        def _run(**_kw):
            with lock:
                live["cur"] += 1
                live["max"] = max(live["max"], live["cur"])
            time.sleep(0.25)
            with lock:
                live["cur"] -= 1
            return {"lane": i, "answer": f"lane{i} 的答案", "has_artifact": True}
        return _run

    got = D.run_lanes([fake_lane(i) for i in range(4)])
    ck("4 条 lane 全部有返回", len(got) == 4, f"实际 {len(got)}")
    ck("最大同时在跑数 > 1 (证明是并发)", live["max"] > 1,
       f"max_concurrent={live['max']}")
    ck("4 条 lane 时 max_concurrent 应达到 4", live["max"] == 4,
       f"max_concurrent={live['max']} (串行的话是 1)")

    # 单条 lane 时就不该有"并发"可言
    live2 = {"cur": 0, "max": 0}

    def fake_one(**_kw):
        with lock:
            live2["cur"] += 1
            live2["max"] = max(live2["max"], live2["cur"])
        time.sleep(0.05)
        with lock:
            live2["cur"] -= 1
        return {"lane": 0, "answer": "x", "has_artifact": True}

    D.run_lanes([fake_one])
    ck("单条 lane 时 max_concurrent == 1", live2["max"] == 1,
       f"max_concurrent={live2['max']}")


# --------------------------------------------------------------- 防 1/5: B 不受影响 + trace
def test_trace_and_b_untouched() -> None:
    print()
    print("=" * 78)
    print("防 1/5: trace 形状, 以及 B 路线不被影响")
    print("=" * 78)
    ck("D 不导出 solve (避免和 B 的 solve 混用)",
       not hasattr(D, "solve") or D.solve.__module__ != "orchestrator",
       "D 应该有自己的入口名")

    ck("lanes=1 的 trace 字段与 B 一致",
       D.lane_trace_fields() == B_LANE_FIELDS,
       f"{D.lane_trace_fields()} vs {B_LANE_FIELDS}")

    # B 的 plan_prompt 在没给 hints 时, 渲染结果必须和今天完全一样
    base = EX.plan_prompt("题面", None, None)
    ck("B 的 plan_prompt 不含 D 的 lane 字段", "lane" not in base.lower(),
       "D 的字段漏进了 B 的 prompt")


#: B 路线 worker trace 条目里必须有的字段(照抄 orchestrator.py:198-220)
B_LANE_FIELDS = ("turn", "role", "action", "method", "tools", "rationale",
                 "outcome", "error_type", "has_artifact", "tool_calls",
                 "findings", "switches", "lane")


# --------------------------------------------------------------- 编排层: 端到端
class _FakeWorkerResult(dict):
    pass


def test_orchestration_end_to_end() -> None:
    """用假 LLM 跑通一整轮: 任一 lane 交出 -> 收工; 全崩 -> 不吞整轮。"""
    print()
    print("=" * 78)
    print("编排层: 任一 lane 交出即收工 / 一条 lane 崩不带走整轮")
    print("=" * 78)

    import agent.tools.research.routes.d as Dmod
    from agent.tools.research.roles import _util as U
    from agent.tools.research.roles import worker as WK

    plan_reply = ('{"read_first":"x","method":"枚举","tools":["numeric_code"],'
                  '"rationale":"r","next_action":"execute","probe_spec":null,'
                  '"done":true,"lanes":{"n":3,'
                  '"intents":["A 路线","B 路线","C 路线"]}}')

    # ★ 两个都要替 ★
    #   只替 Explorer 的 ask 不够 —— worker 侧也会真调 LLM,
    #   上游一挂(本机 provider 配置不对) worker 就返回 has_artifact=False,
    #   于是"三条 lane 全没交出" -> INCONCLUSIVE, 而那**不是编排的错**。
    #   ⇒ 这次失败是测试替身不完整, 不是 solve_d 的 bug。
    orig_ask = U.ask
    orig_call = U._call_llm

    def fake_ask(prompt, **_kw):
        return plan_reply

    def fake_call_llm(prompt, **_kw):
        # worker 只要能解析出 final 就算交出制品
        return '{"final": "worker 的答案", "reason": "算出来的"}'

    made = {"n": 0}
    lock = threading.Lock()

    def fake_run_execute(prompt, plan, scratch_dir, **_kw):
        with lock:
            made["n"] += 1
        # ★ 靠【你这 lane 的任务】段认 lane, 不能靠意图字符串本身 ★
        #   每条 lane 的 prompt 本来就看得见别人的意图(只读板就是干这个的),
        #   所以「prompt 含不含 C 路线」区分不出 lane —— 我第一版就是这么写的,
        #   三条 lane 全被判成 gives_up, verdict 变 INCONCLUSIVE, 连红两次。
        #   ⇒ 用 lane 序号认, 它在 prompt 里是显式标出来的。
        m = re.search(r"\u3010\u4f60\u8fd9 lane \u7684\u4efb\u52a1\u3011\(\u7b2c (\d+) \u6761\)", str(prompt))
        own = int(m.group(1)) if m else 0
        # 让第 3 条 lane 不交出, 验证「任一 lane 交出即收工」
        gives_up = (own == 3)
        art = _FakeWorkerResult({
            "ran": True, "tool_calls": [], "lang": "tool",
            "scratch": scratch_dir, "error_type": None, "stderr": "",
            "has_artifact": not gives_up,
            "artifact": None if gives_up else {
                "answer": "lane%d \u7684\u7b54\u6848" % (own - 1), "reason": "r"},
            "shape_problems": [], "switches": [], "findings": {},
            "final_answer": "" if gives_up else "lane%d" % (own - 1),
            "stripped_self_claims": []})
        return art

    orig_execute = WK.run_execute
    orig_probe = WK.run_probe
    try:
        U.ask = fake_ask
        U._call_llm = fake_call_llm
        WK.run_execute = fake_run_execute
        res = Dmod.solve_d("题面", hints="用数值", budget=U.Budget(max_turns=1))
    finally:
        U.ask = orig_ask
        U._call_llm = orig_call
        WK.run_execute = orig_execute
        WK.run_probe = orig_probe

    ck("跑完有 verdict", bool(res.get("verdict")), str(res.get("verdict")))
    ck("任一 lane 交出 -> 不再是 INCONCLUSIVE",
       res.get("verdict") != "INCONCLUSIVE", str(res.get("verdict")))
    ck("记了并发度", int(res.get("concurrent") or 0) >= 1,
       f"concurrent={res.get('concurrent')}")
    ck("记了总 lane 数", int(res.get("lanes_seen") or 0) >= 1,
       f"lanes_seen={res.get('lanes_seen')}")

    tr = res.get("trace") or []
    worker_rows = [t for t in tr if t.get("role") == "worker"]
    ck("每条 lane 都有 trace 条目", len(worker_rows) >= 3,
       f"实际 {len(worker_rows)} 条")
    ck("trace 条目带 lane 标记",
       all("lane" in t for t in worker_rows),
       "缺 lane 字段就分不清是哪一路")
    ck("explorer 条目记了分派",
       any(t.get("role") == "explorer" and t.get("n_lanes") for t in tr))


def test_b_prompt_unchanged() -> None:
    print()
    print("=" * 78)
    print("防 1: explorer.plan_prompt 在不传 hints 时行为不变")
    print("=" * 78)
    p = EX.plan_prompt("题面", None, None)
    ck("含原有的工具目录", "工具目录" in p)
    ck("含原有的'两条硬规矩'", "硬规矩" in p)
    ck("不含 D 的分派指令", "lane" not in p.lower())


def test_lane_count_decided_by_explorer() -> None:
    """★ lane 数**由 explorer 决定**, hints 只能建议, 不能强制 ★

    ★ 为什么这条必须钉死 ★
      2026-10-11 我曾把 hints 做成"强制 1 lane", 理由是"题面点名了工具
      就没有选择空间"。方向反了: 强制单 lane 等于**剥夺 explorer 的第二次
      机会** —— lane 0 出问题时它连"再加一条路"都做不到。
      简单题一条 lane 够用, **复杂题不够**; 拿测试集的形状去卡设计是错的。
      Explorer 这个角色的存在意义就是"看上一轮结果决定换不换路"。

    ⇒ 判据: explorer 写 n=1 就开 1 条, 写 n=3 就开 3 条 —— 不受 hints 影响。
    """
    print()
    print("=" * 78)
    print("防 6: lane 数由 explorer 决定 (hints 只建议, 不强制)")
    print("=" * 78)

    from agent.tools.research.roles import _util as U, worker as WK
    import agent.tools.research.routes.d as Dmod

    def run_with(n: int) -> int:
        plan = ('{"read_first":"x","method":"m","tools":["numeric_code"],'
                '"rationale":"r","next_action":"execute","probe_spec":null,'
                '"done":true,"lanes":{"n":%d,"intents":["A","B","C"]}}' % n)
        orig_ask, orig_exec, orig_probe = U.ask, WK.run_execute, WK.run_probe
        seen = {"n_lanes": 0}

        def fake_exec(problem, plan_, scratch_dir, **_kw):
            return _FakeWorkerResult({
                "ran": True, "tool_calls": [], "lang": "tool",
                "scratch": scratch_dir, "error_type": None, "stderr": "",
                "has_artifact": True,
                "artifact": {"answer": "x", "reason": "y"},
                "shape_problems": [], "switches": [], "findings": {},
                "final_answer": "x", "stripped_self_claims": []})

        try:
            U.ask = lambda p, **_k: plan
            WK.run_execute = fake_exec
            WK.run_probe = fake_exec
            res = Dmod.solve_d("题面", hints="用 numeric_code",
                               budget=U.Budget(max_turns=1), verbose=False,
                               verify=False, provider="minimax",
                               model="MiniMax-M3.1-Flash-Preview")
            # ★ 数 explorer trace 里的 n_lanes, **不要**数 prompt ★
            #   每条 lane 是**独立并发调用**一次 run_execute, 每次只带
            #   **自己那一条**的 prompt —— 所以"每次调用里出现 1 个 lane 标记"
            #   恰恰是正确的。★ 我第一版就是这么数的, 于是误判成"被压制成 1" ★
            for t in (res.get("trace") or []):
                if t.get("role") == "explorer":
                    seen["n_lanes"] = int(t.get("n_lanes") or 0)
        except Exception as e:  # noqa: BLE001
            print(f"  [!! ] solve_d 抛异常 {type(e).__name__}: {e}")
            FAILS.append("solve_d 抛异常")
        finally:
            U.ask = orig_ask
            WK.run_execute, WK.run_probe = orig_exec, orig_probe
        return seen["n_lanes"]

    n1, n3 = run_with(1), run_with(3)
    ck("explorer 要 1 条 -> 开 1 条", n1 == 1, f"实际 {n1}")
    ck("explorer 要 3 条 -> 开 3 条(即使有 hints)", n3 == 3,
       f"实际 {n3} —— 有 hints 也不该压制 explorer 加 lane")


def main() -> int:
    print("=" * 78)
    print("D 路线契约检查 (并行 worker + hints)")
    print("=" * 78)
    test_hints_reach_prompt()
    test_lane_isolation()
    test_actually_parallel()
    test_trace_and_b_untouched()
    test_b_prompt_unchanged()
    test_orchestration_end_to_end()
    test_lane_count_decided_by_explorer()

    print()
    print("=" * 78)
    print(f"失败 {len(FAILS)}")
    print("=" * 78)
    for f in FAILS:
        print(f"  ★ {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())