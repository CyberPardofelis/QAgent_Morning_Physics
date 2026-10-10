"""编排器 —— 把 Explorer / Worker / Verifier 串成循环。

## ★ 这个文件里**不能**有任何"题型 → 方法"的分支 ★
这是整件事的成败点。旧结构的核心毛病就是
`classify(claim) → methods_for(shapes) → 注入 prompt`:
程序替 Explorer 把选型做完了, 模型只管照做。

所以本文件只做三件事:
  1. 循环(还有预算)
  2. 把上一轮的结果作为**历史**交给 Explorer
  3. 按 Explorer 自己说的 `next_action` 路由到 probe 或 execute

它**不知道**什么是组合结构题, 不知道该用 z3 还是不知道。它只认两个字段:
`next_action` 和 `method`。
`check_no_hardcoded_routing.py` 会静态扫这个文件, 出现题型词就红。

## 换方法是 Explorer 干的, 不是程序逼的
旧结构有个 `_strategy.py`: 上一轮方法类失败就挡回重试, 并要求"这次换个方法"。
那是补丁 —— 它在结果出来之后才反应, 而且是程序替模型决定该换。
现在换方法是 **Explorer 在看见 probe/verify 结果之后自己决定的**,
它手上有完整的"我试过什么、学到了什么"。

我们只做一件约束性的事: **同一个方法不许原样连试三次**。
这不是替它选方法, 这是防止它在同一个坑里空转把预算烧光。

## INCONCLUSIVE 是正经结果
预算耗尽 -> 输出 INCONCLUSIVE + 全部尝试记录。
**穷尽方法后停下来是诚实, 不是失败。** 硬猜一个答案才是失败。
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Any

from . import explorer as EX
from . import verifier as VF
from . import worker as WK
from ._util import Budget, upstream_error

__all__ = ["solve", "Trace"]


class Trace(list):
    """整道题的完整轨迹。打分器读它, 出错时人读它。"""


def solve(problem: str, *, budget: Budget | None = None,
          tools: list[dict] | None = None, scratch_dir: str | None = None,
          verbose: bool = True, verify: bool = True, **llm_kw) -> dict:
    """跑完一道题。返回 {verdict, artifact, trace, ...}。

    ## `verify=False` 是干什么的

    跳过 Verifier, 直接把 Worker 的产出当结论返回, 判决写成 `WORKER_ONLY`。
    **它不是"关掉校验就能算对"** —— 它是为了把"Worker 这条路做不做得出来"
    和"Verifier 复核做不做得出来"两件事**分开量**。

    为什么必须分开: 2026-10-10 实测, Verifier 仍是旧沙箱路线(自己写代码进沙箱
    重算), 两道题里它的两个检查器**都**失败 -> INCONCLUSIVE 0/2。
    于是 40 道题跑完, 分数会被 Verifier 的失败率主导, 而它跟"工具能不能被
    正确调用"无关 —— 那正是我们要先回答的问题。

    ★ 这个状态**不许**并进正确率 ★ `WORKER_ONLY` 与 `PROVED/REFUTED` 在
    统计里必须分列, 理由同 `INCONCLUSIVE`: "没验"和"验了没过"是两件事。
    """
    budget = budget or Budget()
    trace = Trace()
    own_dir = scratch_dir is None
    scratch = scratch_dir or tempfile.mkdtemp(prefix="lab_")
    used_methods: dict[str, int] = {}
    started = time.time()

    def log(msg: str) -> None:
        if verbose:
            print(f"    {msg}", flush=True)

    try:
        while not budget.exhausted():
            budget.spend()
            turn = budget.turns

            # ---- 1. Explorer 拿历史出计划 ----
            pl, raw = EX.plan(problem, history=list(trace), tools=tools,
                              turn=turn, **llm_kw)
            if pl is None:
                err = upstream_error(raw)
                if err is not None:
                    # ★ 上游挂了, 就**立刻停**, 别把预算烧在同一个 API 400 上 ★
                    #   而且状态必须和 INCONCLUSIVE 分开: INCONCLUSIVE 声称的是
                    #   "试过了, 没能判定", 而这里一次成功的调用都没有过。
                    #   混在一起会污染统计 —— "它不会做这题" 和 "它压根没调到模型"
                    #   是两回事, 后者根本不能算进能力分。
                    trace.append({
                        "turn": turn, "role": "explorer", "action": None,
                        "outcome": f"上游调用失败: {err[:400]}",
                        "lesson": "不是 Explorer 的输出问题, 是 LLM 调用本身失败了",
                        "lesson_given": True,
                    })
                    log(f"[{turn}] ★ 上游调用失败, 中止(不烧剩余预算)")
                    return {
                        "verdict": "INFRA_ERROR",
                        "artifact": None,
                        "evidence": None,
                        "trace": list(trace),
                        "turns": budget.turns,
                        "seconds": round(time.time() - started, 1),
                        "budget_exhausted": False,
                        "note": f"LLM 调用失败: {err[:300]}",
                    }
                trace.append({
                    "turn": turn, "role": "explorer", "action": None,
                    "outcome": "Explorer 的输出里没解析出计划 JSON",
                    "lesson": "下一轮把要求说得更直白一点", "lesson_given": True,
                })
                log(f"[{turn}] Explorer 没给出可解析的计划, 继续")
                continue

            fp = pl.fingerprint
            action = pl.action

            # ★ 每个成功解析出来的计划都要落进 trace ★
            #   我第一版只在「计划解析失败」和「方法重复」时写 explorer 条目,
            #   于是**正常路径的计划从来没被记录**。后果不是不好看, 是错的:
            #   tools_used 全空 -> part1 最关键的判据
            #   「有没有真用上被指定的工具」全部误判成 False,
            #   而日志里明明看见它选了 z3。
            #   ⇒ 凡是**会改变行为的输入**就必须留痕, 别只记异常分支。
            trace.append({
                "turn": turn, "role": "explorer", "action": action,
                "method": pl.get("method"), "tools": pl.tools,
                "rationale": pl.get("rationale"),
                "read_first": pl.get("read_first"),
                "probe_spec": pl.get("probe_spec"),
                "done": pl.get("done"),
                "outcome": f"计划: {action}, 工具={pl.tools}",
            })

            # ★ 后端未接入的诚实标记 ★
            # 选 lean4 意味着要往 Lean 编译器提交源码, 而本回路目前只有
            # Python 沙箱这一个后端。与其让 Worker 写一段调用
            # `lean_check()` 的 Python 然后撞 NameError 白烧三轮,
            # 不如当场把这件事记成一种**独立状态**。
            #
            # ★ 为什么不能混进"做错" ★ 它既不是模型不会, 也不是题不可做,
            #   而是**这套回路少了一个后端**。混进分母就是污染测量。
            #   这一类必须能被单独筛出来 —— 和"环境不具备"是同一个道理。
            if _needs_unwired_backend(pl):
                trace[-1].update({
                    "outcome": "Explorer 选了 lean4, 但本回路只有 Python 沙箱"
                               "一个后端, Lean 后端尚未接入",
                    "lesson": None, "lesson_given": False,
                    "backend_unwired": True,
                })
                log(f"[{turn}] 后端未接入 (lean4) —— 不跑, 也不计成失败")
                return {
                    "verdict": "BACKEND_UNWIRED",
                    "artifact": None, "evidence": None,
                    "trace": list(trace), "turns": budget.turns,
                    "seconds": round(time.time() - started, 1),
                    "budget_exhausted": False,
                    "note": "选中的工具需要 Lean 后端, 本轮未接入。"
                            "这是回路的缺口, 不是模型的失败。",
                }

            log(f"[{turn}] {action.upper()}  工具={pl.tools}  "
                f"方法={(pl.get('method') or '')[:48]}")

            # ---- 2. 同一个方法不许原样连试三次 ----
            if action == "execute" and used_methods.get(fp, 0) >= 2:
                trace[-1].update({
                    "outcome": "这个方法已经原样试过 2 次, 不给第三次机会",
                    "lesson": f"这条路已经连续失败两次: "
                              f"{(pl.get('method') or '')[:120]}。换一条根本不同的路。",
                    "lesson_given": True,
                })
                log(f"[{turn}] 同一方法已试 2 次, 挡回")
                continue
            used_methods[fp] = used_methods.get(fp, 0) + 1

            # ---- 3. Worker 去调工具 ----
            # ★ Worker 自己渲染目录 ★ 不再由编排器往里塞一个"工具列表"对象 ——
            #   那个列表以前是 `catalog.build_catalog()` 的产物(11 个库名),
            #   而 Worker 要的是**能调的工具**。两边各渲染一次, 目录定义
            #   只有 `_registry` 一个出处。
            if action == "probe":
                r = WK.run_probe(problem, pl, scratch, timeout=180,
                                 prior=trace, **llm_kw)
            else:
                r = WK.run_execute(problem, pl, scratch, timeout=420,
                                   prior=trace, **llm_kw)

            # ★ WorkerResult 是 dict 子类, 用 .get 取而不是下标 ★
            #   它只保证 run_execute 返回的对象存在, 不保证每个键都在 ——
            #   下标一个未设的键就是 KeyError, 而这会让整题 CRASH。
            trace.append({
                "turn": turn, "role": "worker", "action": action,
                "method": pl.get("method"), "tools": pl.tools,
                "rationale": pl.get("rationale"),
                "outcome": WK.brief(r),
                "error_type": r.get("error_type"),
                "timed_out": False,
                "has_artifact": r.get("has_artifact"),
                "artifact": r.get("artifact"),
                # ★ 工具调用序列要落进 trace ★ 它是"到底用没用上工具"的
                #   唯一证据, 也是 part1 那条核心判据("有没有真用上被指定的工具")
                #   的数据来源。以前只有一行 outcome 摘要。
                "tool_calls": r.get("tool_calls"),
                "shape_problems": r.get("shape_problems"),
                # ★ 收敛轨迹要进 trace ★ 它回答"为什么会换路/为什么烧光预算",
                #   没有它就只能看到"没交出答案"这个结果, 看不到过程。
                "switches": r.get("switches"),
                # ★★ ReAct 闭环: Worker 这一轮**真的做了什么** ★★
                #   Explorer 下一轮就是靠它判断"继续/换路/收卷"的。
                "findings": r.get("findings"),
                "stripped_self_claims": r.get("stripped_self_claims"),
                "_raw": str(r.get("final_answer") or "")[:2500],
            })
            if action == "probe":
                log(f"[{turn}] probe 结束, 交给 Explorer 判断")
                continue

            # ---- 4. execute 但没交制品 -> 回 Explorer, 不算终局 ----
            if not r.get("has_artifact"):
                trace[-1]["lesson"] = (
                    "这一轮没能交出答案。请明确输出 "
                    '{"final": "你的答案", "reason": "怎么得到的"}, '
                    "或者继续调工具把结论做出来 —— 不要只描述你看到了什么。")
                trace[-1]["lesson_given"] = True
                log(f"[{turn}] 没交出答案, 回 Explorer")
                continue

            # ---- 5. Verifier 独立重算 ----
            if not verify:
                return {
                    "verdict": "WORKER_ONLY",
                    "artifact": r.get("artifact"),
                    "evidence": None,
                    "trace": list(trace),
                    "turns": budget.turns,
                    "seconds": round(time.time() - started, 1),
                    "budget_exhausted": False,
                    "note": "本次跳过了 Verifier —— 只量 Worker/工具这一段。",
                }
            v = VF.check(problem, r.get("artifact"), scratch,
                         n_checkers=2, **llm_kw)
            trace.append({
                "turn": turn, "role": "verifier", "action": "verify",
                "verdict": v.get("verdict"), "agree": v.get("agree"),
                # ★ 置信度要进 trace ★ 1/2 验过 和 2/2 验过, 结论都是对的,
                #   但强度不同。混在一起统计, 就看不出"有多少结论是只验了一半的"。
                "outcome_kind": v.get("outcome_kind"),
                "n_confirmed": v.get("n_confirmed"), "n_total": v.get("n_total"),
                "confident": v.get("confident"),
                "outcome": v.get("note") or (v.get("evidence") or "")[-400:],
                "hint": v.get("hint"),
            })
            conf = (f"{v.get('n_confirmed')}/{v.get('n_total')} 验过"
                    if v.get("n_total") else "")
            log(f"[{turn}] 独立复核 -> {v.get('verdict')} "
                f"(agree={v.get('agree')}{' ' + conf if conf else ''})")

            if v.get("agree") and v.get("verdict") in ("PROVED", "REFUTED"):
                return {
                    "verdict": v["verdict"],
                    "artifact": r.get("artifact"),
                    "evidence": v.get("evidence"),
                    "trace": list(trace),
                    "turns": budget.turns,
                    "seconds": round(time.time() - started, 1),
                    "budget_exhausted": False,
                }

            # ---- 6. 没通过 -> 带诊断回 Explorer ----
            trace[-1]["lesson"] = (
                f"复核结论是 {v.get('verdict')}"
                + (f"。诊断: {(v.get('hint') or '')[:300]}"
                   if v.get("hint") else "")
                + "。下一轮不要重复交同一个东西。")
            trace[-1]["lesson_given"] = True

        # ---- 预算耗尽 ----
        return {
            "verdict": "INCONCLUSIVE",
            "artifact": _last_artifact(trace),
            "evidence": None,
            "trace": list(trace),
            "turns": budget.turns,
            "seconds": round(time.time() - started, 1),
            "budget_exhausted": True,
            "note": f"用完 {budget.max_turns} 轮预算, 没能判定。"
                    f"试过的方法: {_tried_methods(trace)}",
        }
    finally:
        if own_dir:
            shutil.rmtree(scratch, ignore_errors=True)


def _tried_methods(trace: list[dict], limit: int = 40) -> list[str]:
    """这题试过哪些方法 —— **给 note 用**, 所以不许抛异常。

    ★ 这里踩过一个坑 ★
      原来写的是 `[t.get('method', '')[:limit] for t in trace ...]`。
      `dict.get(k, default)` 的 default **只在 key 不存在时**生效 ——
      而 Explorer 交上来的 JSON 里常是 `{"method": null}`,
      **key 在、值是 None**, 于是 get 返回 None, `None[:40]` 直接 TypeError。

    ★ 为什么这条路径值得单独修 ★
      它在**预算耗尽的收尾分支**上 —— 也就是"模型没交出答案"时才会走到。
      崩在这里会让**本该记 INCONCLUSIVE(没交出答案) 的题变成 CRASH**,
      于是"能力不足"被记成"程序炸了", 真实原因被掩盖。
      实测 2026-10-10: part2 的 res-05 与 res-17 两道都栽在这一行,
      两道各烧了 366s / 1038s, 结果 0 次工具调用就崩了。

    ⇒ 收尾路径上的任何一行都**不允许**因为数据形状意外而抛异常:
      这里宁可显示 "(没写方法)" 也不能让整题崩掉。
    """
    out: list[str] = []
    for t in trace:
        if t.get("role") != "explorer":
            continue
        m = t.get("method")
        out.append(str(m)[:limit] if m else "(没写方法)")
    return out


def _needs_unwired_backend(plan: dict) -> bool:
    """这份计划是不是需要一个**本回路还没有执行后端**的工具?

    ★ Lean 4 已经接上了 ★
    底座是 `agent.tools.research._lean.check_lean` —— 编译判定 = exit 0 且无 error
    且**无 sorry**(sorry 会让退出码仍为 0, 只看退出码会把"没证"当成"证好了")。
    Worker 现在认 `plan["tools"]` 里的 `lean4`, 走编译而不是 Python 沙箱。

    2026-10-10 之前这里把 `lean4` 当"后端未接入"直接返回 BACKEND_UNWIRED,
    于是 part1 里 4 道 Lean 题**一道都没跑过**就出局了。
    而 Lean 明明装好了、QAgent 侧也早有 `research_lean` ——
    **缺的是这根线, 不是那个工具。**

    判据看的是 **Explorer 自己选的 tools**, 不是题面。
    —— 它选 lean4 就是它要去 Lean 编译器, 程序只负责"执行它选的",
    不负责替它重选。这与"按题面派发方法"是两回事。
    """
    return False


def _last_artifact(trace: list) -> Any:
    for t in reversed(trace):
        if t.get("role") == "worker" and t.get("has_artifact"):
            return t.get("artifact")
    return None