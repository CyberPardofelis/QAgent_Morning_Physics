"""把单发式证据链如实翻译成 AgentTrace。

★ 为什么重写(2026-10-09)★
审计实测: 旧适配器把 tool 恒填 "python"、method 恒填 "brute_force"(而且
`last_method` 初值之后再也不更新), 所以:
  · types 只可能是 ['code','run','reflect'] —— **结构上产不出**
    experiment / search / cross_check
  · 于是 exploration 恒 0, 与"模型有没有做实验"无关
  · cross_validation 恒 0.4 或 0.0(取自"有没有一条 success 的 run"),
    测的是"机器门响过没有", 与"有没有独立路径"无关
反事实检验: 凭空贴两个 experiment 标签, 总分 42.2 -> 59.7, 而行为零变化。

★ 改完之后 0 分的含义变了 ★
改之前: exploration=0 只说明"适配器产不出这种步"。
改之后: exploration=0 说明**它真的没做实验** —— 这才是可以据以改代码的信号。

★ 只翻译真实发生过的事, 一律不补 ★
凭空造 experiment/cross_check 步来提分 = 自己给自己打分作弊。
所以 cross_check 只在"**确实存在两条不同方法、且其中一条成功**"时才记。
"""
from __future__ import annotations

import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import reward_scorer as RS  # noqa: E402

#: 代码里自述的策略标签(我们自己要求它写的)。
_STRATEGY = re.compile(r"^\s*#\s*STRATEGY\s*:\s*([a-z_]+)\s*$",
                        re.IGNORECASE | re.MULTILINE)

#: 从代码的 import 里认出实际动用了哪些库 —— 这是真实的 tool 信号,
#: 不是"它理论上可以用什么"。
_IMPORT = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)",
                     re.MULTILINE)

#: import 名 -> 打分器认识的 tool 名。
_TOOL_NAME = {
    "sympy": "sympy", "z3": "z3", "numpy": "numpy", "scipy": "scipy",
    "networkx": "networkx", "ortools": "ortools", "cvxpy": "cvxpy",
    "mpmath": "mpmath", "math": "python", "fractions": "python",
    "decimal": "python", "itertools": "python", "json": "python",
}


def derive_method(rec: dict) -> str:
    """这一轮实际用的计算形态。

    优先用代码里自述的 `# STRATEGY:`(那是我们要求它写的, 是真实信号);
    没有就退回 evidence 的 `how`; 都没有才叫 unknown —— **不用默认
    "brute_force"**, 那个默认值会让所有轨迹的 method_diversity 变成常数。
    """
    m = _STRATEGY.search(rec.get("code") or "")
    if m:
        return m.group(1).strip().lower()
    how = (rec.get("how") or "").strip().lower()
    return how or "unknown"


def derive_tools(rec: dict) -> list[str]:
    """这一轮实际 import 了哪些库。空列表表示"只用纯 Python 内建"。"""
    seen = []
    for mod in _IMPORT.findall(rec.get("code") or ""):
        root = mod.split(".")[0]
        name = _TOOL_NAME.get(root)
        if name and name not in seen:
            seen.append(name)
    if not seen:
        seen = ["python"]
    return seen


def primary_tool(tools: list[str]) -> str:
    """这一轮**靠什么驱动** —— 非 python 的那个。

    为什么要挑而不是全列: TraceStep 只有一个 tool 字段, 而打分器的
    method_diversity / tool_quality / cross_validation 都是数**不同 tool
    的个数**。如果每轮都填 tools[0], 而 tools[0] 永远是 json→python,
    那么"用了 z3"这个真实信息就被丢掉了 —— 实测 res-001 第 2 轮明明
    import 了 z3, 记出来的 tool 却只有 python。
    """
    non_py = [t for t in tools if t != "python"]
    return non_py[0] if non_py else "python"


def trace_from_evidence(item: dict, spec: RS.TaskSpec,
                        task_hash: str | None = None) -> RS.AgentTrace | None:
    """把 research_verify 的证据链翻译成一条**如实**的 AgentTrace。"""
    if task_hash is None:
        task_hash = "r_" + hashlib.sha1(item["id"].encode()).hexdigest()[:10]
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "research", "working",
        task_hash, "_evidence.jsonl")
    if not os.path.isfile(p):
        return None
    recs = [json.loads(x) for x in io.open(p, encoding="utf-8") if x.strip()]
    recs = [r for r in recs if r.get("kind") in ("code", "lean")]
    if not recs:
        return None

    trace = RS.AgentTrace(task_id=item["id"])
    for r in recs:
        ok = r.get("verdict") in ("proved", "refuted")
        method = derive_method(r)
        tools = derive_tools(r)
        primary = primary_tool(tools)
        trace.steps.append(RS.TraceStep(
            type="code", tool=primary, method=method,
            content=f"生成并执行验证器({r.get('verdict')})", success=ok))
        trace.steps.append(RS.TraceStep(
            type="run", tool=primary, method=method,
            content=str(r.get("reason", ""))[:200], success=ok))
        if r.get("verdict") in ("failed", "inconclusive"):
            trace.steps.append(RS.TraceStep(
                type="reflect", method="retry",
                content="机器判定未通过, 换写法重试", success=False))

    # ★ cross_check 只在**真的发生过**时才记 ★
    # 条件: 存在两条 method 不同的轮次, 且至少有一条成功 ——
    # 那才是"用另一条路又算了一遍"(哪怕是靠重试实现的)。
    # 不满足就一条都不加: 凭空贴标签提分 = 自己骗自己。
    oks = [(derive_method(r), set(derive_tools(r)))
           for r in recs if r.get("verdict") in ("proved", "refuted")]
    methods = {m for m, _ in oks}
    if len(methods) >= 2:
        m2, t2 = oks[-1]
        trace.steps.append(RS.TraceStep(
            type="cross_check", tool=primary_tool(sorted(t2)),
            method=m2,
            content=f"用另一种方法({m2})独立重算并得到一致结论",
            success=True))

    trace.final_answer = (recs[-1].get("reason") or "")[:300]
    return trace


import json  # noqa: E402  (放在后面只为让上面的 __import__ 更显眼)
import hashlib  # noqa: E402