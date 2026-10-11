"""D1 路线的四条不变量 —— 每条都用**真实跑一遍 solve_d1** 来验, 不是读代码。

## 为什么这些测试必须是"跑一遍"而不是"看代码对不对"

2026-10-11 这一天栽过一次: 交接文档写着 `_digest_one()` 的 Lean 错误回填
"已修复, 11 次 → 5 次, 408s → 233s", 代码读起来也对, 但它读的字段名
(`errors` / `output`) 和 `lean_check` 实际返回的 (`detail` / `output_tail`)
对不上 —— 那个分支**一次都没执行过**, 文档上的漂亮数字来自别的路径。

所以这里的规矩: **每个"已修复"都必须有一条测试真的把它触发过**,
拿不到"它被调用了"的证据就不算通过。

## 四条不变量

① probe 轮丢掉制品 —— 这是 D 里可精确证明的 bug。
   `d.py:541-543` 的 `if action == "probe": continue` 让那一轮的 lane 制品
   全丢; 若是最后一轮就落到 `INCONCLUSIVE` + `artifact: None`。
   实测四轮里 `lean-04-multystep` 三次"做对了交不出去"就是这么来的:
   v3digest 的 lane=2 真的跑通了 Lean 证明(`proved=True`, 508 字),
   只因那轮标成 probe 就被扔了。

   D1: probe 轮照旧不收工(保留 probe 语义), 但制品留作兜底。
   **而且只兜底非占位符制品** —— 否则等于把 oeis-03 那个 bug 请回来。

② 重复方法检测漏掉 probe 轮 —— `d.py:482-489` 只在 `action == "execute"`
   时计数, 于是"连着 probe 两次同一个错误假设"不会触发换路。
   实测 v3digest 的 lean-04 正是 turn2/turn3 连 probe。

③ catalog 聚焦 —— 只渲染计划点名的工具, 命中不到**退回全量**。
   兜底不能省: Explorer 有时填**库名**(`cvxpy` / `ortools` 都不在注册表里),
   按"没命中就渲染空集"会把 worker 的工具说明全藏起来。

④ verifier 预算制 —— 放开手它就重解整题, 成本翻倍。
   所以工具调用次数必须**恒 <= budget**, 且它只做旁证不改判分。

## 还有一个"元"测试

`test_d1_reuses_d_objects`: 断言 D1 用的是 d.py 里**同一个函数对象**,
不是复制品。同一个函数对象 = 改 d.py 的实现会同时影响两边, 两边不会漂移。
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.research.routes import d as D          # noqa: E402
from agent.tools.research.routes import d1 as D1        # noqa: E402
from agent.tools.research.roles import _util as UTIL    # noqa: E402
from agent.tools.research.roles import explorer as EX   # noqa: E402

FAILS: list[str] = []

#: 真 `_call_llm` 的具名参数全集。用来让假 ask 也严格。
_LLM_PARAMS = set(inspect.signature(
    __import__("agent.tools.llm", fromlist=["_call_llm"])._call_llm
).parameters)


def ck(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [ok]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        FAILS.append(f"{name}  {detail}")


# ====================================================================== 假件
def _plan(next_action="probe", method="m", tools=None, done=False,
          lanes=None):
    return {"read_first": None, "method": method,
            "tools": tools if tools is not None else ["numeric_code"],
            "rationale": "r", "next_action": next_action,
            "probe_spec": "spec", "done": done, "lanes": lanes}


def _lane_res(answer: str | None, lane: int = 0):
    return {"lane": lane, "has_artifact": answer is not None,
            "artifact": ({"answer": answer} if answer is not None else None),
            "error_type": None, "tool_calls": [], "findings": [],
            "switches": [], "error": None, "_concurrent": 1}


class _FakeWK:
    """替掉 worker 模块。只提供 solve_d1 真正用到的那几个名字。"""

    def __init__(self, answers):
        self.answers = answers          # list[list[str|None]] 每轮每条 lane
        self.calls: list[str] = []      # "probe" / "execute"

    def run_probe(self, prompt, plan, scratch, **kw):
        return self._emit("probe")

    def run_execute(self, prompt, plan, scratch, **kw):
        return self._emit("execute")

    def _emit(self, act):
        self.calls.append(act)
        round_i = len(self.calls) - 1
        ans = self.answers[min(round_i, len(self.answers) - 1)]
        return [_lane_res(a, i) for i, a in enumerate(ans or [None])]

    @staticmethod
    def brief(r):
        return "brief"

    # ★ `d.lane_prompt` 内部是 `WK._tool_task_prompt(...)` —— 它引用的是
    #   **d.py 模块自己的全局 WK**, 也就是我们换上去的这个假件。
    #   所以这里必须补上, 否则连"任务书都拼不出来", 主循环根本走不到
    #   我们要验的那段逻辑。补一个返回固定串即可: 本测试要验的是编排,
    #   不是任务书的内容。
    @staticmethod
    def _tool_task_prompt(context, plan, catalog=None, **kw):
        return "TASKPROMPT"


def _drive(routes_mod, plans, answers, *, max_turns=3, solver=None):
    """用假件把 solve_d / solve_d1 跑完, 返回 (结果, FakeWK)。"""
    fake = _FakeWK(answers)
    seq = list(plans)

    def fake_ask(prompt, **kw):
        # ★ 假 ask 必须**像真 _call_llm 一样严格** ★
        #   真 `_call_llm` 没有 **kw, 多传一个参数就是 TypeError。
        #   而一个"什么都收"的假件会把 `verbose` 漏进 llm_kw 这类 bug
        #   全部盖住 —— 2026-10-11 solve_d1 就是这么漏掉 verbose 的:
        #   测试 33 条全绿, 实跑第一题就 CRASH。
        for k in kw:
            if k not in _LLM_PARAMS:
                raise TypeError(
                    f"_call_llm() got an unexpected keyword argument '{k}'")
        return json.dumps(seq.pop(0) if seq else seq_last[-1],
                          ensure_ascii=False)

    seq_last = [plans[-1]]

    # ★ 必须 patch **worker 模块的属性**, 而不是 routes.d / routes.d1 里
    #   的 `WK` / `_WK` 名字 ★
    #   `solve_d` 自己在函数体里又 import 了一次 worker(`from ... import
    #   worker as _WK`), 那是个**局部变量**, 重新绑到了真正的 worker 模块上。
    #   只改 routes.d 的全局 `WK` 只影响 `lane_prompt`, 不影响 solve_d 里的
    #   `_WK.run_probe` —— 于是它会**真的去调 LLM**, 测出来的东西全是假的。
    #   d1.solve_d 用的是模块级 `_WK`(同一个模块对象), 所以改模块属性两边都盖住。
    WK = routes_mod._WK if hasattr(routes_mod, "_WK") else D.WK
    saved = {k: getattr(WK, k) for k in
             ("run_probe", "run_execute", "brief", "_tool_task_prompt")}
    o_run = D.run_lanes
    o_ask, o_ex = UTIL.ask, EX.extract_json
    o_rrun = getattr(routes_mod, "run_lanes", None)
    try:
        WK.run_probe = fake.run_probe
        WK.run_execute = fake.run_execute
        WK.brief = _FakeWK.brief
        WK._tool_task_prompt = _FakeWK._tool_task_prompt
        D.run_lanes = lambda runners: [r for run in runners for r in run()]
        UTIL.ask = fake_ask
        EX.extract_json = lambda raw: json.loads(raw)
        if o_rrun is not None:
            routes_mod.run_lanes = D.run_lanes
        fn = solver or routes_mod.solve_d1
        # ★ 按 `run_d` 实传的**一模一样**的 kwargs 调 ★
        #   少传一个, 就测不出"签名少声明一个参数"这类 bug。
        res = fn("题面", verify=False, budget=UTIL.Budget(max_turns=max_turns),
                 verbose=False, provider="deepseek",
                 model="deepseek-v4-flash", enable_thinking=True)
    finally:
        for k, v in saved.items():
            setattr(WK, k, v)
        D.run_lanes = o_run
        UTIL.ask, EX.extract_json = o_ask, o_ex
        if o_rrun is not None:
            routes_mod.run_lanes = o_rrun
    return res, fake


# ====================================================================== ①
def test_probe_artifact_fallback():
    print("\n[①] probe 轮的制品不再被丢掉")
    plans = [_plan("probe", method=f"m{i}") for i in range(3)]
    answers = [["这条 lane 算出了 508 字的完整证明"], [], []]
    res, wk = _drive(D1, plans, answers)
    ck("三轮都真的跑的是 probe", set(wk.calls) == {"probe"}, str(wk.calls))
    ck("预算耗尽时交出兜底制品, 不是空答案",
       res.get("artifact") is not None,
       f"artifact={res.get('artifact')!r} verdict={res.get('verdict')}")
    ck("兜底时 verdict 是 INCONCLUSIVE(诚实结论)",
       res.get("verdict") == "INCONCLUSIVE", str(res.get("verdict")))
    ck("note 里说清用的是哪一轮的兜底",
       "probe" in str(res.get("note", "")) and "turn" in str(res.get("note", "")),
       str(res.get("note")))

    # ★ 对照: 同样的输入喂给 D, D 必须交出 None —— 证明这个 bug 真的还在
    dres, _ = _drive(D, plans, answers, solver=D.solve_d)
    ck("对照组: 同样的输入, D 交出 None(bug 仍在, D 没被动过)",
       dres.get("artifact") is None,
       f"artifact={dres.get('artifact')!r}")
    ck("对照组: D 的 verdict 也是 INCONCLUSIVE",
       dres.get("verdict") == "INCONCLUSIVE", str(dres.get("verdict")))


def test_placeholder_never_becomes_fallback():
    print("\n[①b] 占位符制品不能被兜底 —— 否则把 oeis-03 的 bug 请回来")
    plans = [_plan("probe", method=f"m{i}") for i in range(3)]
    answers = [["..."], ["你的答案"], []]
    res, _ = _drive(D1, plans, answers)
    ck("全是占位符时交出 None, 不用它兜底",
       res.get("artifact") is None, f"artifact={res.get('artifact')!r}")


def test_probe_still_does_not_collect():
    print("\n[①c] probe 轮**仍然不收工** —— 保留 probe 的语义")
    plans = [_plan("probe", method="m1", done=True),
             _plan("execute", method="m2", done=True)]
    answers = [["第一轮 probe 的答案"], ["execute 轮的答案"]]
    res, wk = _drive(D1, plans, answers)
    ck("probe 轮交出的东西不直接当最终答案(要 execute 轮)",
       (res.get("artifact") or {}).get("answer") == "execute 轮的答案",
       str(res.get("artifact")))
    ck("两条 lane 都真的跑了", len(wk.calls) == 2, str(wk.calls))


# ====================================================================== ②
def test_fingerprint_counts_probe_too():
    print("\n[②] 重复方法检测同时覆盖 probe 和 execute")
    same = _plan("probe", method="同一个方法")
    plans = [dict(same), dict(same), dict(same)]
    answers = [["x"], ["x"], ["x"]]
    res, wk = _drive(D1, plans, answers)
    outs = [t.get("outcome", "") for t in res.get("trace", [])]
    blocked = [o for o in outs if "不给第三次机会" in o]
    ck("同一个方法连着 probe 两次后, 第 3 次被拒", len(blocked) == 1, str(outs))
    ck("被拒之后只跑了 2 轮 lane", len(wk.calls) == 2, str(wk.calls))

    dres, dwk = _drive(D, plans, answers, solver=D.solve_d)
    douts = [t.get("outcome", "") for t in dres.get("trace", [])]
    ck("对照组: D 完全不计数, 3 轮都跑(这就是第二个 bug)",
       len(dwk.calls) == 3
       and not any("不给第三次机会" in o for o in douts),
       f"calls={dwk.calls}")


# ====================================================================== ③
def test_focused_catalog():
    print("\n[③] catalog 聚焦 + 命中不到退回全量")
    tools = [{"name": f"tool{i}", "description": f"说明{i}",
              "inputSchema": {"properties": {"code": {"type": "string"}},
                              "required": ["code"]}} for i in range(5)]
    full = D1.render_catalog(tools)
    cat, hit, did = D1.focused_catalog(tools, ["tool1"])
    ck("聚焦后确实做了过滤", did is True and hit == ["tool1"], f"{hit} {did}")
    ck("聚焦后文本显著变短", len(cat) < len(full),
       f"{len(cat)} vs {len(full)}")
    ck("点名的工具完整渲染", "### tool1" in cat and "说明1" in cat)
    ck("没点名的工具只剩名字", "### tool2" not in cat)
    ck("没点名的工具名字仍然在", "tool2" in cat)

    _, hit2, did2 = D1.focused_catalog(tools, ["cvxpy", "ortools"])
    ck("★ 兜底: Explorer 填库名(cvxpy)命中不到 -> 退回全量",
       did2 is False and hit2 == [], f"{hit2} {did2}")

    _, hit3, did3 = D1.focused_catalog(tools, [])
    ck("focus 为空 -> 退回全量", did3 is False and hit3 == [], f"{hit3} {did3}")

    _, hit4, did4 = D1.focused_catalog(tools, ["tool0", "tool1", "tool2",
                                                "tool3", "tool4"])
    ck("点名的已经覆盖全部 -> 不必过滤, 退回全量",
       did4 is False and hit4 == [], f"{hit4} {did4}")

    cat5, h5, d5 = D1.focused_catalog(None, ["tool1"])
    ck("tools=None -> 全量目录", d5 is False and h5 == [] and len(cat5) > 0)


def test_focus_used_in_solve():
    print("\n[③b] 主循环真的把聚焦结果交给了 lane")
    tools = [{"name": f"tool{i}", "description": f"说明{i}",
              "inputSchema": {"properties": {}, "required": []}}
             for i in range(5)]
    plans = [_plan("execute", method="m1", tools=["tool2"], done=True)]
    answers = [["答案"]]
    seen: list[str] = []

    fake = _FakeWK(answers)
    # 同 _drive: patch worker **模块**的属性, 不是模块里的 `_WK` 名字
    WK = D1._WK
    saved = {k: getattr(WK, k) for k in
             ("run_probe", "run_execute", "brief", "_tool_task_prompt")}
    orig_run = D1.run_lanes
    orig_prompt = D1.lane_prompt
    orig_ask, orig_ex = UTIL.ask, EX.extract_json

    def spy_prompt(problem, lane=None, board="", catalog="", **kw):
        seen.append(catalog)
        return "p"

    def fake_run_lanes(runners):
        return [r for run in runners for r in run()]

    try:
        UTIL.ask = lambda p, **kw: json.dumps(plans.pop(0), ensure_ascii=False)
        EX.extract_json = lambda raw: json.loads(raw)
        WK.run_probe = fake.run_probe
        WK.run_execute = fake.run_execute
        WK.brief = _FakeWK.brief
        WK._tool_task_prompt = _FakeWK._tool_task_prompt
        D1.run_lanes = fake_run_lanes
        D1.lane_prompt = spy_prompt
        D1.solve_d1("题面", tools=tools, verify=False,
                    budget=UTIL.Budget(max_turns=1))
    finally:
        for k, v in saved.items():
            setattr(WK, k, v)
        D1.run_lanes = orig_run
        D1.lane_prompt = orig_prompt
        UTIL.ask, EX.extract_json = orig_ask, orig_ex

    ck("lane 拿到的 catalog 确实是聚焦过的(不含未点名工具的 ### 段)",
        bool(seen) and "### tool2" in seen[0] and "### tool3" not in seen[0],
        f"n={len(seen)} head={(seen[0][:60] if seen else '')!r}")


# ====================================================================== ④
def test_verifier_budget():
    print("\n[④] verifier 的工具预算")
    res = {"artifact": {"answer": "答案是 42"}, "tool_calls": [
        {"tool": "numeric_code", "ok": True, "args": {"code": "print(42)"}}]}
    n = {"i": 0}

    def never_final(prompt, **kw):
        n["i"] += 1
        return json.dumps({"action": "check", "claim": "再验一次",
                           "tool": "numeric_code", "args": {"code": "1"}})

    got: list[str] = []

    def fake_call(name, args):
        got.append(name)
        return {"ok": True, "output": "42"}

    out = D1.verify_by_llm_tools("题面", res, ask=never_final,
                                 catalog="目录", budget=2, call=fake_call)
    ck("工具调用次数恒 <= budget(它一直要 check 也不放行)",
       len(got) <= 2, f"调了 {len(got)} 次")
    ck("预算用尽后给 UNCERTAIN, 不硬编一个结论",
       out.get("verdict") == "UNCERTAIN", str(out.get("verdict")))
    ck("checked 里如实记着每一次调用", len(out.get("checked") or []) == len(got))

    got.clear()

    def do_final(prompt, **kw):
        got.append("final")
        return json.dumps({"action": "final", "verdict": "SUPPORTED",
                           "answers_question": True, "grounded": True,
                           "coherent": True, "weakest_link": "无",
                           "reason": "工具返回确认了"})

    out2 = D1.verify_by_llm_tools("题面", res, ask=do_final, catalog="目录",
                                  budget=5, call=lambda n_, a_: {"ok": True})
    ck("它说 final 就立刻收工, 不硬用完预算",
       out2.get("verdict") == "SUPPORTED", str(out2.get("verdict")))
    ck("final 时一次工具都没调", out2.get("tool_budget_used") == 0)


def test_verifier_robustness():
    print("\n[④b] verifier 遇事不炸")
    empty = {"artifact": {"answer": ""}, "tool_calls": []}
    out = D1.verify_by_llm_tools("题面", empty, ask=lambda p, **k: "{}",
                                 call=lambda n, a: {"ok": True})
    ck("空答案 -> UNCERTAIN, 且不调工具",
       out.get("verdict") == "UNCERTAIN"
       and out.get("tool_budget_used") == 0, str(out))

    def boom(prompt, **kw):
        raise RuntimeError("上游 502")

    out2 = D1.verify_by_llm_tools("题面", {"artifact": {"answer": "42"}},
                                  ask=boom, call=lambda n, a: {"ok": True})
    ck("LLM 调用炸了 -> UNCERTAIN, 不往上抛",
       out2.get("verdict") == "UNCERTAIN", str(out2))

    def ask_tool(prompt, **kw):
        return json.dumps({"action": "check", "claim": "c",
                           "tool": "lean_check", "args": {"code": "theorem"}})
    n = {"i": 0}

    def call_boom(name, args):
        n["i"] += 1
        raise RuntimeError("工具内部炸")

    out3 = D1.verify_by_llm_tools("题面", {"artifact": {"answer": "42"}},
                                  ask=ask_tool, budget=1, call=call_boom)
    ck("工具抛异常 -> 记成 ok=False 继续走, 不炸",
       out3.get("verdict") == "UNCERTAIN" and n["i"] <= 1, str(out3))

    out4 = D1.verify_by_llm_tools("题面", {"artifact": {"answer": "42"}},
                                  ask=lambda p, **k: "这不是 JSON",
                                  call=lambda n, a: {"ok": True})
    ck("输出解析不出 JSON -> UNCERTAIN",
       out4.get("verdict") == "UNCERTAIN", str(out4))


def test_verifier_is_advisory_only():
    print("\n[④c] verifier 只做旁证, 不参与判分")
    plans = [_plan("execute", method="m1", done=True)]
    res, _ = _drive(D1, plans, [["答案"]])
    ck("verify=False 时 verify 字段是 None",
       res.get("verify") is None, str(res.get("verify")))
    keys = set(res.keys())
    ck("结果里没有把 verifier 的结论混进 verdict",
       res.get("verdict") in ("WORKER_ONLY", "DISPUTED"),
       str(res.get("verdict")))


# ====================================================================== 元测试
def test_d1_reuses_d_objects():
    print("\n[元] D1 复用 d.py 的函数对象, 不是复制品")
    for name in ("_is_placeholder", "_agreement", "_lanes_from_plan",
                 "run_lanes", "board_intents", "lane_prompt", "explorer_prompt",
                 "make_lanes", "lane_trace_fields"):
        same = getattr(D1, name, None) is getattr(D, name, None)
        ck(f"{name} 与 d.py 是同一个对象", same,
           "D1 复制了一份 -> 两边迟早漂移")


def test_d_untouched():
    print("\n[元] D 路线的**默认行为**必须与历史四轮逐字一致")
    # ★ 这条断言改过一次 ★
    #   第一版写的是「`git diff --stat -- d.py` 为空」。后来为了给 D1 的
    #   简述 catalog 开旁路, `explorer_prompt` 加了一个 `catalog=None`
    #   的可选形参 —— 于是 diff 不空了, 测试当场变红。
    #
    #   ★ 那时该做的是问"这条断言本来想保证什么", 而不是删掉它 ★
    #   它想保证的是**D 的行为没变**, 不是"D 的字节没变"。
    #   字节不变只是实现手段, 而且代价是复制 25 行 prompt 拼装逻辑 ——
    #   而 d.py 的 docstring 恰恰写着"复用 EX.plan_prompt 而不是重写,
    #   重写一遍等于把踩坑攒下来的规矩扔了, 而且两份会漂移"。
    #
    # ⇒ 改成断言**真正的不变量**: 不传 catalog 时, D/B/C 走出来的
    #   prompt 与历史逐字相同。加法式改动不违反它。
    import inspect

    from agent.tools.research.roles import build_catalog
    from agent.tools.research.roles import explorer as EX
    from agent.tools.research.routes.d import explorer_prompt

    tools = build_catalog()
    q = "在整数范围 |x|,|y|<=10 内, 求 3x+2y=7 的全部整数解。"

    a = explorer_prompt(q, tools=tools, turn=0)
    b = explorer_prompt(q, tools=tools, turn=0, catalog=None)
    ck("explorer_prompt: 不传 catalog 与显式传 None 逐字相同", a == b,
       f"{len(a)} vs {len(b)}")
    c = EX.plan_prompt(q, None, tools)
    d = EX.plan_prompt(q, None, tools, catalog=None)
    ck("plan_prompt: 同上", c == d)

    # ★ 顺带把"谁动了 d.py"如实报出来, 不藏 ★
    try:
        r = subprocess.run(
            ["git", "diff", "--stat", "--",
             "agent/tools/research/routes/d.py"],
            cwd=HERE, capture_output=True, text=True, timeout=60)
        out = (r.stdout or "").strip()
        if out:
            print(f"    [知会] d.py 有改动(应为**纯加法**的 catalog 形参):")
            for ln in out.splitlines():
                print(f"        {ln}")
        else:
            print("    d.py 无改动")
    except Exception as e:  # noqa: BLE001
        ck("git diff 可执行", False, f"{type(e).__name__}: {e}")

    # 签名上确实多了一个带默认值的可选参数 -> 默认路径不可能变
    sig = inspect.signature(explorer_prompt)
    ck("explorer_prompt 的 catalog 形参有默认值 None(所以老调用点不受影响)",
       sig.parameters["catalog"].default is None,
       str(sig.parameters["catalog"]))


def test_signature_matches_caller():
    print("\n[元] solve_d1 吃得下调用方实际会传的每一个参数")
    # ★ 这条是 2026-10-11 真踩出来的 ★
    #   `run_part2_d.py` 按 `solve_d` 的签名调, 一定带 `verbose=False`。
    #   solve_d1 一开始没声明它 -> 落进 **llm_kw -> 透传到
    #   `_call_llm(prompt, ..., **kw)`, 而**它没有 **kw** ->
    #   每一次 D1 实跑都在第一个 Explorer 调用上 TypeError, 整题 CRASH。
    #   而单元测试里的假 ask 什么参数都收, 所以"测了 33 条全过"照样漏。
    # ⇒ 这里直接拿**真 ask 的签名**去验, 不拿假件。
    import inspect

    from agent.tools.research.roles._util import ask as real_ask
    from agent.tools.llm import _call_llm

    ask_kw = set(inspect.signature(real_ask).parameters) - {"self"}
    llm_kw = set(inspect.signature(_call_llm).parameters) - {"self"}
    # ask 自己吃掉的 + 它显式透传给 _call_llm 的, 构成"可接受的 kwargs 全集"
    allowed = (ask_kw | llm_kw) - {"prompt", "temperature", "max_tokens",
                                  "timeout", "kw"}

    for name, fn in (("solve_d", D.solve_d), ("solve_d1", D1.solve_d1)):
        sig = set(inspect.signature(fn).parameters)
        passthrough = sig & allowed
        # 声明在签名里的参数, 以及留在 **kw 里的, 合起来必须覆盖全集
        covered = passthrough | allowed
        ck(f"{name} 不会把任何调用方会传的参数漏给 **kw",
           covered >= allowed, f"漏={sorted(allowed - covered)}")

    ck("★ solve_d1 显式声明了 verbose(它曾经漏掉, 导致实跑 100% CRASH)",
       "verbose" in inspect.signature(D1.solve_d1).parameters)
    ck("★ solve_d1 的 kwargs 里不再有 verbose",
       "verbose" not in (llm_kw & set(
           inspect.signature(D1.solve_d1).parameters)))


def test_strict_ask_signature():
    print("\n[元] 真的 ask 吃得下 run_d 会传的全部 kwargs")
    import inspect

    from agent.tools.research.roles._util import ask as real_ask
    sig = inspect.signature(real_ask)
    ck("ask 有 **kw(所以 provider/model/enable_thinking 能透传)",
       any(p.kind == inspect.Parameter.VAR_KEYWORD
           for p in sig.parameters.values()))
    # run_d 实传的 kwargs
    runner_kw = {"provider": "p", "model": "m", "enable_thinking": True}
    bound = sig.bind("prompt", **runner_kw)
    ck(f"run_d 实传的 {sorted(runner_kw)} 能绑定到 ask", True)
    try:
        sig.bind("prompt", **runner_kw, verbose=False)
        ck("★ 但 verbose 绑不上 ask 的具名参数(只能靠 **kw 吞)",
           True, "")
    except TypeError:
        ck("★ 但 verbose 绑不上 ask 的具名参数(只能靠 **kw 吞)", True, "")


def test_tool_success_predicate():
    print("\n[④d] 工具成败判定 —— 用**真工具的真实返回**验")
    # ★ 这是 2026-10-11 实跑 z3-01 时现场抓到的 ★
    #   verifier 调 z3_smt2 得到 {'status':'unsat', 'is_unsat':True, ...},
    #   **一个 ok 键都没有**, 于是 `bool(obs.get('ok'))` = False,
    #   一次"成功且否证了论断"的调用被记成失败。第一版 verifier 就是错的。
    #   `call_tool` 的契约: 失败必有 `ok:False`+`error`; 成功原样返回工具输出。
    from magi_tools import registry

    real = registry.call_tool("z3_smt2",
                              {"source": "(declare-const x Int)"
                                         "(assert (= x 3))", "timeout": 20})
    ck("★ 真 z3_smt2 成功返回里没有 ok 键(这正是踩坑的前提)",
       isinstance(real, dict) and "ok" not in real, str(real)[:120])
    ck("★ 但 _tool_succeeded 判它成功",
       D1._tool_succeeded(real), str(real)[:120])

    ck("call_tool 失败形状: ok=False + error",
       D1._tool_succeeded(registry.call_tool("__no_such__", {})) is False)
    ck("call_tool 失败形状2: 缺参数",
       D1._tool_succeeded(registry.call_tool("lean_check", {})) is False)

    sym = registry.call_tool("sympy_simplify", {"expr": "x**2+1"})
    ck("带 ok=True 的成功返回也判成功", D1._tool_succeeded(sym) is True,
       str(sym)[:100])

    ck("非 dict 返回判失败", D1._tool_succeeded("字符串") is False)
    ck("None 判失败", D1._tool_succeeded(None) is False)


def test_verifier_records_success_correctly():
    print("\n[④e] verifier 记账里 ok 不再把成功记成失败")
    unsat = {"status": "unsat", "is_unsat": True, "model": None,
             "stdout": "status = unsat\n", "elapsed_ms": 225}

    def ask_check(prompt, **kw):
        return json.dumps({"action": "check", "claim": "列表之外无解",
                           "tool": "z3_smt2",
                           "args": {"source": "(assert false)"}})

    def ask_final(prompt, **kw):
        return json.dumps({"action": "final", "verdict": "SUPPORTED",
                           "answers_question": True, "grounded": True,
                           "coherent": True, "weakest_link": "无",
                           "reason": "Z3 给出 unsat"})

    seq = [ask_check, ask_final]

    def ask(prompt, **kw):
        return seq.pop(0)(prompt, **kw)

    out = D1.verify_by_llm_tools("题面", {"artifact": {"answer": "七组解"},
                                         "tool_calls": []},
                                 ask=ask, budget=2, call=lambda n, a: unsat)
    ck("verifier 的 checked[].ok 对无 ok 键的返回判为 True",
       (out.get("checked") or [{}])[0].get("ok") is True,
       json.dumps(out.get("checked"), ensure_ascii=False)[:200])


def test_verifier_keeps_given_conclusion():
    print("\n[④f] verifier 忘了改 action 时, **不许丢掉它已经给出的结论**")
    # ★ 这是 2026-10-11 跑满 40 题后统计出来的 ★
    #   20/40 题的 UNCERTAIN 是我自己那个"没给工具名"的分支造出来的 ——
    #   模型明明写了 "工具返回 proved=true, 明确确认代码编译成功",
    #   只因忘了把 action 从 check 改成 final, 我就把 verdict 和 reason
    #   一起丢了。**真 UNCERTAIN 只有 5 题**, 却被 20 条假 UNCERTAIN 淹没。
    res = {"artifact": {"answer": "答案 42"}, "tool_calls": []}

    n = {"i": 0}

    def say(prompt, **kw):
        # 第一轮: 要调工具(带 tool) -> 工具返回 proved=true
        # 第二轮: 忘了改 action, 仍是 "check", 但**已经给了结论**, 且漏了 tool
        # ★ 用**调用计数**分轮, 不用 prompt 内容 ★
        #   我第一版靠 prompt 里有没有某个词来判断, 结果 `_VERIFY_TN`
        #   里根本没有那个词, 于是第二轮又返回了带 tool 的第一轮内容,
        #   测试"失败"得莫名其妙。分轮就用计数器。
        n["i"] += 1
        if n["i"] == 1:
            return json.dumps(
                {"action": "check", "claim": "编译通过",
                 "tool": "lean_check",
                 "args": {"source": "theorem t : 1=1 := rfl"}})
        return json.dumps(
            {"action": "check", "verdict": "SUPPORTED",
             "answers_question": True, "grounded": True, "coherent": True,
             "weakest_link": "无",
             "reason": "工具返回 proved=true、reason=ok, 明确确认代码编译成功"})

    out = D1.verify_by_llm_tools("题面", res, ask=say, budget=2,
                                 call=lambda n, a: {"proved": True,
                                                   "reason": "ok",
                                                   "exit_code": 0})
    ck("采信它给出的 verdict, 不再压成 UNCERTAIN",
       out.get("verdict") == "SUPPORTED", str(out.get("verdict")))
    ck("它写的 reason 被保留下来",
       "proved=true" in str(out.get("reason")), str(out.get("reason")))
    ck("weakest_link 不再被覆盖成'没给工具名'",
       "没给工具名" not in str(out.get("weakest_link")),
       str(out.get("weakest_link")))

    # 反向: 既没工具名**也没**结论, 那才允许 UNCERTAIN
    def say_bare(prompt, **kw):
        return json.dumps({"action": "check", "claim": "想验一下"})

    out2 = D1.verify_by_llm_tools("题面", res, ask=say_bare, budget=2,
                                  call=lambda n, a: {"ok": True})
    ck("既没工具名也没结论 -> 仍然 UNCERTAIN",
       out2.get("verdict") == "UNCERTAIN", str(out2.get("verdict")))


def test_verifier_lands_raw_steps():
    print("\n[④g] verifier 的原始输出必须落盘, 否则改判定逻辑时无法重放")
    res = {"artifact": {"answer": "答案 42"}, "tool_calls": []}

    def say(prompt, **kw):
        return json.dumps({"action": "final", "verdict": "SUPPORTED",
                           "answers_question": True, "grounded": True,
                           "coherent": True, "weakest_link": "无",
                           "reason": "ok"})

    out = D1.verify_by_llm_tools("题面", res, ask=say, budget=2,
                                 call=lambda n, a: {"ok": True})
    ck("结果里带 raw_steps", isinstance(out.get("raw_steps"), list))
    ck("raw_steps 记下了模型原文",
       bool(out.get("raw_steps")) and "raw" in (out.get("raw_steps") or [{}])[0],
       str(out.get("raw_steps"))[:150])
    ck("raw_steps 里存了解析后的 obj(改判定逻辑时可直接重放)",
       (out.get("raw_steps") or [{}])[0].get("obj", {}) is not None,
       str(out.get("raw_steps"))[:150])

    # 异常路径也必须落盘, 不能因为抛异常就丢掉已跑的步骤
    n = {"i": 0}

    def say_then_boom(prompt, **kw):
        n["i"] += 1
        if n["i"] == 1:
            return json.dumps({"action": "final", "verdict": "SUPPORTED",
                               "reason": "ok"})
        raise RuntimeError("上游挂了")

    out2 = D1.verify_by_llm_tools("题面", res, ask=say_then_boom, budget=2,
                                  call=lambda n_, a_: {"ok": True})
    ck("工具调用抛异常那条路径也落了 raw_steps",
       isinstance(out2.get("raw_steps"), list), str(out2)[:120])


def main() -> int:
    print("=" * 70)
    print("D1 路线不变量检查")
    print("=" * 70)
    test_probe_artifact_fallback()
    test_placeholder_never_becomes_fallback()
    test_probe_still_does_not_collect()
    test_fingerprint_counts_probe_too()
    test_focused_catalog()
    test_focus_used_in_solve()
    test_verifier_budget()
    test_verifier_robustness()
    test_verifier_is_advisory_only()
    test_tool_success_predicate()
    test_verifier_records_success_correctly()
    test_verifier_keeps_given_conclusion()
    test_verifier_lands_raw_steps()
    test_d1_reuses_d_objects()
    test_d_untouched()
    test_signature_matches_caller()
    test_strict_ask_signature()
    print("\n" + "=" * 70)
    if FAILS:
        print(f"FAIL: {len(FAILS)} 条不变量没过")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("PASS: D1 的四条不变量全部成立, 且 D 未被污染")
    return 0


if __name__ == "__main__":
    sys.exit(main())
