"""Explorer —— 拿到**无提示**的问题, 自己决定怎么查。

## 这个角色存在的理由

旧结构里没有 Explorer: `research_verify(claim, how="symbolic")` 一次 LLM 调用
同时干两件事 —— 决定用什么方法、写代码。两件事塞在一个上下文里, 于是:

  · 模型没有"先想清楚再动手"的位置;
  · 更要命的是, 旁边还挂着一张**由题面算出来**的方法表
    (`classify(claim)` → `methods_for(shapes)`), 等于把选型提前做完了。

现在选型是一个**独立的、只有 Explorer 能做的决定**, 而且它拿到的工具信息
只有一份: 本机有什么工具(目录)。题面里没有任何"该用什么"的暗示。

## ★ 这个 prompt 里**不能**有什么 ★
不能出现任何 "如果是组合结构题 → 用 SAT / 枚举" 这样的映射。
一旦写进去, 整个重构就退化回旧结构 —— 只是把硬编码从 Python 挪到了 prompt。
`check_no_hardcoded_routing.py` 会静态扫这件事。

## 记忆
Explorer 是唯一持有历史记忆的角色。它看到"上一轮试了什么、结果如何",
于是**换一个方法**是它自己的判断, 不是被程序逼的 —— 旧结构里那个
`_strategy.py`(失败后挡回重试并要求换路)可以整个删掉。

## INCONCLUSIVE 是合法输出
预算耗尽时, "我没能判定"是一个**诚实的结论**, 不是失败。
把它写进 schema, 是为了让 Explorer 在穷尽方法后敢于停下来。
"""
from __future__ import annotations

from typing import Any

from ._util import ask, extract_json, upstream_error
from .catalog import render_catalog

__all__ = ["PLAN_FIELDS", "plan_prompt", "plan", "Plan"]

#: 计划里必须有、且**只**有这些字段。加 verdict 之类就破坏了分工。
PLAN_FIELDS = ("read_first", "method", "tools", "rationale",
               "next_action", "probe_spec", "done")

PLAN_INSTRUCT = """\
【你现在的任务】
读题、想清楚打算怎么做, 然后给出**一份计划**。你不用写代码, 也不用给答案。

【输出格式】
只输出一个 JSON 对象, 不要有任何别的文字:

{
  "read_first": "我对题面的理解: 这题到底在问什么, 交出什么才算做完",
  "method":     "我打算怎么做(自然语言描述做法, 不要只写工具名)",
  "tools":      ["目录里的工具 id", ...],
  "rationale":  "为什么这条路可行; 相比我考虑过的其它路, 为什么选这条",
  "next_action":"probe" 或 "execute",
  "probe_spec": "next_action=probe 时写清楚: 在多小的规模上试、试什么、看什么信号; execute 时填 null",
  "done":       "我认为这个计划已经足以给出可复核的制品吗? true/false"
}

【next_action 怎么选】
  probe    —— 我还不确定这条路行不行, 想先在小规模上试一下拿证据。
              小规模实验**失败是正常的**, 那是给我判断用的信息, 不是终局。
  execute  —— 我认为这条路能直接产出可复核的制品。

【两条硬规矩】
1. tools 里的每个 id 必须**逐字**来自上面的工具目录。不要发明新工具。
2. 你**看不到**出题人的解题思路, 也**没人**告诉你该用什么。这是有意的 ——
   判断方法本身就是这份工作里最该由你决定的部分。

【关于不确定性】
拿不准就 probe, 或者在 rationale 里写清楚你缺什么信息。
预算耗尽时允许你以"未能判定"收场 —— 那是诚实的结论, 不是失败。
不要为了给出一个结论而在计划里假装有把握。
"""


class Plan(dict):
    """一份计划。字段缺失时是 None, 由 orchestrator 兜底, 不在这里猜。"""

    @property
    def action(self) -> str:
        a = str(self.get("next_action") or "").lower()
        return a if a in ("probe", "execute") else "probe"

    @property
    def tools(self) -> list[str]:
        t = self.get("tools")
        return [str(x) for x in t] if isinstance(t, list) else []

    @property
    def fingerprint(self) -> str:
        """同一个方法的指纹 —— 用来判"你是不是又原样来一遍"。"""
        m = str(self.get("method") or "").strip()
        return f"{sorted(self.tools)}|{m[:120]}"


def plan_prompt(problem: str, history: list[dict] | None = None,
                tools: list[dict] | None = None,
                catalog: str | None = None) -> str:
    # ★ `catalog` 形参是**给 D1 用的旁路**, 默认 None 走原路径 ★
    #   实测 2026-10-11: Explorer 的 prompt 中位 **21518 字符**, 其中
    #   catalog(全量) 占 **93.6%** —— 而 Explorer **只选路、不调工具**,
    #   参数 schema 是给 Worker 用的, 对它是纯噪声。
    #
    #   但这里**不能**默认改成简述: B/C/D 三条路线都走这个函数,
    #   改了它们的选路质量就跟着变, 而它们是对照组。
    #   ⇒ 做成可选旁路, 只有 D1 显式传值时才生效。
    cat = catalog if catalog is not None else render_catalog(tools)
    hist = ""
    if history:
        rows = []
        for h in history:
            # ★ Worker 交回的结构化调查摘要 ★
            #   旧版这里只有 `发生了什么`(brief 的前 400 字) + `lesson`,
            #   而 Worker 的 brief 是"最后一步的尾部" ——
            #   于是 Explorer 看到的是"这一轮末尾长什么样",
            #   **看不到这一轮到底做了什么、算出了什么**。
            #
            #   ★ 后果 (实测 2026-10-10, part2 零提示 20 题) ★
            #   13 道"没交出答案"的题里 13 道都在重复调同一个工具
            #   (`numeric_code` 最多 x16) —— 因为 Worker 每轮都**从零开始**,
            #   上一轮的中间结果/报错/半成品全部烂在上一轮上下文里,
            #   Explorer 又拿不到, 于是它也无从判断"这条路已经走到哪"。
            #   ★ 这不是收敛机制的问题, 是**信息没交回去** ★
            rows.append(_history_entry(h))
        hist = ("\n【你已经试过的】\n" + "\n".join(rows) +
                "\n\n★ 不要重复已经证明走不通的路; 但也不要因为一次失败就否定整个方向 ★\n"
                "\n★ 下面每一轮的 `调查摘要` 是 Worker **真的做了什么**; "
                "据此判断该继续这条路、换路、还是证据已经够了可以收卷 ★\n")
    return (f"【工具目录】(这是这台机器上真实可用的东西, 就这些)\n{cat}\n"
            f"{hist}\n【问题】\n{problem}\n\n" + PLAN_INSTRUCT)


def _history_entry(h: dict) -> str:
    """把一条 worker 记录渲染成 Explorer 能据以决策的形态。

    ★ 刻意**先给调查摘要, 再给 outcome** ★
      旧版只有 outcome(= brief 前 400 字), 而 brief 是"最后一步 + 尾部截断",
      信息量最低。现在 `findings` 是 Worker 逐步攒出来的结构化事实:
      调了哪些工具、每次拿到什么值、有没有中途的异常。
    """
    out = [f"- 第 {h.get('turn')} 轮 | 动作={h.get('action')} | "
           f"工具={h.get('tools')}"]
    m = str(h.get("method") or "")
    if m:
        out.append(f"    方法: {m[:180]}")
    f = h.get("findings")
    if isinstance(f, dict):
        if f.get("calls"):
            for c in f["calls"][:8]:
                v = str(c.get("value") or c.get("error") or "")[:110]
                out.append(f"      · {c.get('tool')}: ok={c.get('ok')} {v}")
        if f.get("artifact_preview"):
            out.append(f"      制品: {str(f['artifact_preview'])[:140]}")
        if f.get("stopped_because"):
            out.append(f"      停在哪: {f['stopped_because']}")
    if h.get("outcome"):
        out.append(f"    发生了什么: {str(h['outcome'])[:400]}")
    if h.get("lesson"):
        out.append(f"    我从中学到的: {str(h['lesson'])[:300]}")
    return "\n".join(out)


def plan(problem: str, history: list[dict] | None = None,
         tools: list[dict] | None = None, *, turn: int = 0,
         **kw) -> tuple[Plan | None, str]:
    """问 Explorer 要一份计划。返回 (计划, 原始输出)。

    返回 None 表示**没解析出计划** —— 调用方要把它当成"这一轮没进展",
    而不是当成一个空计划继续跑。两者混起来会让预算被无声地烧掉。
    """
    p = plan_prompt(problem, history, tools)
    if turn:
        p = p.replace("【你现在的任务】",
                      f"【你现在的任务】(第 {turn + 1} 轮)")
    raw = ask(p, temperature=0.6, **kw)
    # ★ 先问"这是不是上游报错" ★ 顺序不能反: 先判报错再解析,
    #   否则一段报错文本会走"解析失败"那条路, 然后被当成模型的输出问题去重试。
    err = upstream_error(raw)
    if err is not None:
        return None, err
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return None, raw
    out = Plan(obj)
    out["_turn"] = turn
    out["_raw"] = raw
    return out, raw