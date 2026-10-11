"""★ 判定从「agent 声明」改成「从制品推导」 ★

## DS 的诊断(2026-10-09)

> Executor 的 schema 里不该有 verdict。verdict 必须由独立阶段派生。

今晚 res-006/013/014 三道题的死法都是同一个: agent 宣称 PROVED, 而制品
根本不支持(res-006 交了 a=b=c=d=0 的平凡关系; res-013 交了 89 组全错的解;
res-014 交的动作序列连魔方都解不开)。**"宣布结论"这个动作的成本是零** ——
打一行 VERDICT 不需要做任何事, 而交出制品要费力气, 所以模型总选前者。

## 为什么这里不是简单地"删掉 VERDICT 行"

删掉会连带丢掉两样有用的东西:
  1. **纪律**: 强制模型对结果表态, "跑通了"不等于"证明了"
  2. **矛盾检测的触发条件**: 好几道机器门("说全过了却判假"、"各部分都定不了
     却汇总成假")都是拿声明去对 stdout 找冲突

所以这里改成: **VERDICT 是 agent 的"声明", 不是判定。判定另由推导得出。**

    agent_declared  = 从 stdout 的 VERDICT 行读的(模型的自我表态)
    derived         = 从**制品**推出的(凭证是否存在 + 是否与输出自洽)

两者都记进证据。**不一致本身就是一个新的诊断量** —— 它直接回答
"模型对自己的结论有多不靠谱", 而这是我此前完全看不到的。

## 推导规则(全部来自制品, 不看声明)

  有反例凭证(COUNTEREXAMPLE / 反例自然写法)  -> refuted
  没有任何制品                                -> inconclusive(没拿出东西就没法判定)
  有制品但程序崩了                            -> 取已打印的部分, 标注崩溃
  有制品且程序正常结束                        -> 由 A 线机器门复核后给出
  其余                                        -> partial
"""
from __future__ import annotations

import json
from typing import Any

#: 判定强度序, 用于取"最好的一条"
_RANK = {"failed": 0, "untested": 1, "inconclusive": 2,
         "partial": 3, "refuted": 4, "proved": 5}


def declared_verdict(evidence: dict) -> str | None:
    """模型自己声明的判定。它是**声明**, 不是判定。"""
    out = evidence.get("stdout") or ""
    for line in out.splitlines():
        if "VERDICT" in line.upper() and ":" in line:
            tok = line.upper().split("VERDICT", 1)[1].lstrip(" :\t")
            tok = tok.split()[0].lower() if tok.split() else ""
            if tok in _RANK:
                return tok
    return None


def has_artifact(evidence: dict) -> bool:
    """输出里有没有一份可复核的制品(不论标签)。"""
    out = evidence.get("stdout") or ""
    for tag in ("CONSTRUCT", "ANSWER", "COUNTEREXAMPLE"):
        if f"{tag}:" in out:
            return True
    return False


def has_counterexample(evidence: dict) -> bool:
    """有没有一份**反例**制品。"""
    out = evidence.get("stdout") or ""
    return "COUNTEREXAMPLE:" in out


def derive(evidence: dict, judged: str, judged_reason: str,
           require_artifact: bool = True) -> dict[str, Any]:
    """
    从制品推导判定。`judged` 是 A 线机器门已经给出的判定。

    ★ `require_artifact` 是什么 ★
    只有**题目要求交出一个对象**时, 制品才是判定的前提。
    真假判定题(如 sin^2+cos^2=1)不需要制品 —— **那个计算本身就是证明**。
    ★ 第一版没加这个开关, 结果把 9 道确定性题全杀了 ★
    全量回归实测抓到: `prove_05_trig_identity` 期望 proved 实得 inconclusive。
    这就是"拿一个丢一个"的典型 —— 加一条规则时必须问"它会误伤哪一类"。

    Returns: {"verdict":..., "reason":..., "declared":..., "mismatch": bool}
    """
    declared = declared_verdict(evidence)
    art = has_artifact(evidence)
    counter = has_counterexample(evidence)

    if counter:
        # 有一份反例制品 —— 判假的依据就在输出里, 不用问模型
        verdict = "refuted"
        basis = "输出里有一份反例制品"
    elif require_artifact and not art and judged in ("proved", "refuted"):
        # ★ 实测事故形态 ★ 宣称成立/证伪, 却什么都没交 ——
        # 没有制品的判定是不可复核的, 一律降级。
        verdict = "inconclusive"
        basis = "宣称了结论但没有交出任何制品 —— 不可复核"
    else:
        verdict = judged
        basis = "A 线机器门复核"

    mismatch = (declared is not None and declared != verdict)
    return {
        "verdict": verdict,
        "reason": f"[制品推导:{basis}] {judged_reason}",
        "declared": declared,
        "mismatch": mismatch,
        "basis": basis,
        "has_artifact": art,
        "has_counterexample": counter,
    }


def better(a: dict | None, b: dict | None) -> dict | None:
    """两条推导结果里挑更好的。"""
    if a is None:
        return b
    if b is None:
        return a
    return a if _RANK.get(a["verdict"], 0) >= _RANK.get(b["verdict"], 0) else b


def summarize(rows: list[dict]) -> str:
    """给报告用: 声明与推导不一致的分布 —— 这是新拿到的一个诊断量。"""
    n = len(rows)
    mis = [r for r in rows if r.get("mismatch")]
    if not n:
        return "无数据"
    lines = [f"参与推导 {n} 条; 声明与推导不一致 {len(mis)} 条 ({len(mis)/n*100:.0f}%)"]
    pairs: dict[str, int] = {}
    for r in mis:
        pairs[f"声明{r.get('declared')} → 推导{r.get('verdict')}"] = \
            pairs.get(f"声明{r.get('declared')} → 推导{r.get('verdict')}", 0) + 1
    for k, v in sorted(pairs.items(), key=lambda kv: -kv[1]):
        lines.append(f"   {k}: {v}")
    return "\n".join(lines)


_ = json  # 保持与证据结构的对应关系, 便于将来直接塞结构化制品