"""回归: Worker 的**特定分支**不许藏着只在某条路上才执行的错。

## 背景 (2026-10-10, B 路线接真工具表的第一跑)

**① `res["lean"] = ctx["lean"]` -> `KeyError: 'lean'`**

`ctx` 里从来没有 `"lean"` 这个键(只有 `lean_proved` / `lean_sorry` /
`lean_reason`)。而这行**只在 Worker 调了 `lean_check` 时才执行** ——
于是 B 全量第一题(`lean-01-refl`)直接 CRASH, 后面 35 道连带全废。

★ 这一类和"只有某入口先被调用才炸的死锁"是同一族 ★
  不跑到那条分支就永远不会知道它错了。**分支覆盖率不是可选项。**

**② Verifier 认不出新形态的 Lean 制品**

Worker 换成原生工具调用后制品形状变成
`{"answer":..., "tool_result": {"proved":true, "source":"...", ...}}`,
而 `_is_lean_artifact` 只认**顶层**的 `source` + `lean_proved`。
⇒ Verifier 会拿一份 Lean 制品去走 **Python 沙箱**复核, 两个检查器全被拒,
   报成 `INCONCLUSIVE 0/2` —— 又一次**把接线问题伪装成数学问题**。

所以判据是"认得���两种形状", 而不是"认得某一种"。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.research.roles.verifier import _is_lean_artifact  # noqa: E402
from agent.tools.research.roles.worker import (  # noqa: E402
    WorkerResult, shape_problem, strip_self_claims,
)

FAILS: list[str] = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("=" * 76)
    print("① Lean 制品: verifier 两种形状都要认")
    print("=" * 76)
    old = {"answer": "已证", "source": "theorem t : 1+1=2 := by rfl",
           "lean_proved": True, "sorry_used": False}
    new = {"answer": "已证", "reason": "lean_check 通过",
           "tool_result": {"ok": True, "proved": True, "sorry_used": False,
                           "source": "theorem t : 1+1=2 := by rfl",
                           "reason": "ok", "prelude_added": ["import Mathlib"]}}
    ck("旧形态(顶层 source+lean_proved)认得出", _is_lean_artifact(old))
    ck("★ 新形态(下一层 tool_result)也认得出 ★", _is_lean_artifact(new),
       "★ 这是 2026-10-10 那个 bug: 只认顶层 -> Lean 制品被拿 Python 沙箱查 ★")
    ck("非 Lean 制品不误判",
       not _is_lean_artifact({"answer": "max=14/5",
                              "tool_result": {"ok": True, "value": "14/5"}}))
    ck("None / 空都不炸", not _is_lean_artifact(None)
       and not _is_lean_artifact({}))

    print()
    print("=" * 76)
    print("①b 判据本身: Lean 的判决在**哪一层**都要认")
    print("=" * 76)
    # ★ 同一族错误的第三次 ★ ① 是 verifier 只认顶层, 这里是**判据**只认顶层。
    #   表现完全一样(4 道 Lean 全判 WRONG), 而实测 proved=True —— 假阴性。
    sys.path.insert(0, r"D:\pythonProject\MAGI-Tools")
    try:
        from run_part1_tools import JUDGE
    except Exception as e:  # noqa: BLE001
        print(f"  (拿不到 run_part1_tools: {e}) —— 跳过")
        JUDGE = None
    if JUDGE:
        f = JUDGE["lean-01-refl"]["fn"]
        ck("C 形状(顶层 lean_proved)判对", f("", {"lean_proved": True})[0])
        ck("★ B 形状(下一层 tool_result.proved)也判对 ★",
           f("", {"tool_result": {"proved": True, "reason": "ok"}})[0],
           "★ 不认它 = B 的 Lean 题全成假阴性 ★")
        ck("真没判过时判错", f("", {"lean_proved": False,
                                  "lean_reason": "compile_error"})[0] is False)
        ck("用了 sorry 判错", f("", {"lean_proved": False,
                                    "lean_sorry": True})[0] is False)
        ck("什么都没有 -> 判错而不是崩", f("", {})[0] is False)

    print()
    print("=" * 76)
    print("② Worker 的 lean 字段: 从 ctx 逐项取, 不许写 ctx['lean']")
    print("=" * 76)
    # 直接复现 ① 的写法: ctx 的键名到底有哪些
    ctx = {"lean_proved": True, "lean_sorry": False, "lean_reason": "ok"}
    ck("ctx 里确实没有 'lean' 键(所以那行会 KeyError)", "lean" not in ctx)
    try:
        _ = ctx["lean"]
        ck("访问 ctx['lean'] 会抛", False, "居然没抛, 判据失效")
    except KeyError as e:
        ck("访问 ctx['lean'] 确实抛 KeyError", True, f"{e}")
    # 正确写法
    got = {k: ctx[k] for k in ("lean_proved", "lean_sorry", "lean_reason")}
    ck("逐项取能拿到完整三项", set(got) == {"lean_proved", "lean_sorry",
                                          "lean_reason"}, str(got))

    print()
    print("=" * 76)
    print("②b WorkerResult 初始化契约: 读后写的键必须**一开始就在**")
    print("=" * 76)
    # ★ 同一族的第三次 ★ ① ctx['lean']、② brief 读 proved、这里是
    #   `res["error_type"] = res["error_type"] or ...` —— 而 error_type
    #   初始化时没设, 于是 lean-02 CRASH: KeyError: 'error_type'。
    #   ★ 共同点: 全部只在**出错分支**上执行, 正常路径永远不读它们。
    #   ⇒ 判据不该是"跑一遍看看", 而是"这些键必须存在"。
    from agent.tools.research.roles import worker as WK
    import inspect as _inspect
    src = _inspect.getsource(WK._run_loop)
    for key in ("error_type", "stderr", "has_artifact", "artifact",
                "stripped_self_claims", "final_answer", "final_reason",
                "lean", "last_tool_result", "shape_problems",
                "tool_calls", "ran"):
        ck(f"WorkerResult 初始化里显式给了 {key!r}",
           f'"{key}"' in src.split("last_obs: dict | None = None")[0],
           "读后写的键必须在初始化时就有")

    # ★ 真正跑一遍三种终局, 确认都不炸 ★
    class _FakeAsk:
        """按脚本喂回复, 不打真 LLM。"""
        def __init__(self, replies):
            self.replies = list(replies)
            self.n = 0

        def __call__(self, prompt, **kw):
            r = self.replies[min(self.n, len(self.replies) - 1)]
            self.n += 1
            return r

    for label, replies, want in (
        ("final 终局",
         ['{"final": "x = 14/5", "reason": "顶点枚举"}'], "has_artifact"),
        ("give_up 终局", ['{"give_up": "装不上"}'], "error_type"),
        ("形状坏到底", ["不是 JSON"], "error_type"),
        ("工具调完没 final",
         ['{"tool": "sympy_simplify", "args": {"expr": "x"}}'], "error_type"),
    ):
        orig_ask = WK.ask
        try:
            WK.ask = _FakeAsk(replies)
            r = WK._run_loop("题面", {"method": "m"}, "目录",
                             probe=False, scratch_dir=".", timeout=30)
            got = bool(r.get(want))
            ck(f"{label}: 返回且 {want}={got}", got, str(r)[:120])
            ck(f"{label}: brief() 不炸", isinstance(WK.brief(r), str))
            for k in ("error_type", "has_artifact", "tool_calls", "lean",
                      "final_answer", "last_tool_result"):
                ck(f"{label}: 键 {k!r} 存在(不是靠 try 猜)", k in r)
        except Exception as e:  # noqa: BLE001
            ck(f"{label}: 不抛异常", False, f"{type(e).__name__}: {e}")
        finally:
            WK.ask = orig_ask

    print()
    print("=" * 76)
    print("③ WorkerResult 的形状: brief() 依赖的键都要在")
    print("=" * 76)
    r = WorkerResult({"ran": True, "tool_calls": [
        {"tool": "lean_check", "args": {}, "ms": 120, "ok": True}],
        "lean": {"lean_proved": True, "lean_sorry": False, "lean_reason": "ok"}})
    from agent.tools.research.roles.worker import brief
    txt = brief(r)
    ck("brief 能处理带 lean 的结果", "lean_check" in txt, txt[:120])
    ck("brief 报出了 proved", "proved=True" in txt, txt[:200])

    r2 = WorkerResult({"ran": True, "tool_calls": [],
                       "error_type": "RanOutOfSteps"})
    ck("brief 能处理空 tool_calls", "一次工具都没调" in brief(r2))
    r3 = WorkerResult({"ran": True, "tool_calls": [], "has_artifact": True,
                       "artifact": {"answer": "x"}})
    ck("brief 能处理有制品的结果", "交出了制品" in brief(r3))

    print()
    print("=" * 76)
    print("④ 形状判据与自评剥离(Worker 新形态的两个纯函数)")
    print("=" * 76)
    ck("合法 tool 放行",
       shape_problem({"tool": "z3_smt2", "args": {}}, "") is None)
    ck("合法 final 放行", shape_problem({"final": "x"}, "") is None)
    ck("合法 give_up 放行", shape_problem({"give_up": "y"}, "") is None)
    ck("散文被拦", shape_problem(None, "We need to think...") is not None)
    ck("空 tool 名被拦", shape_problem({"tool": ""}, "") is not None)
    ck("args 类型错被拦",
       shape_problem({"tool": "x", "args": "s"}, "") is not None)
    art, st = strip_self_claims({"answer": "x", "proved": True, "n": 3})
    ck("自评字段被剥", st == ["proved=True"] and "proved" not in art, str(st))
    ck("真数据没被剥", art.get("n") == 3 and art.get("answer") == "x")

    print(f"\n失败 {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
