"""P4 LLM 层基线 —— **离线重判**。

## 为什么要有这个脚本

跑一遍 LLM 层要 ~10 分钟、~60 次 LLM 调用（8 道题）。但判定逻辑（哪个 verdict
算对、哪道题硬吹了、哪道题结论矛盾）是**纯函数** —— 只依赖 research.db 里已经落好的
claim 判定，跟 LLM 无关。

所以：改完判定规则后**不要重跑闭环**，直接读库重算。省钱省时间，且保证
"重判"和"实跑"用的是同一份证据。

## 用法

    python scripts/_research_eval_baseline.py

有 `QAGENT_RESEARCH_EVAL_LLM=1` 时先跑真闭环，没有则直接读库重判。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent.infra.research_workspace import hash_question  # noqa: E402
from agent.tools.research._db import list_claims  # noqa: E402

CASES_PATH = ROOT / "tests" / "research_cases.json"

OVERCONFIDENT = frozenset({"proved", "refuted"})
STATE_TO_VERDICT = {"undecidable": "inconclusive", "pending": "untested"}
RUN_LLM = os.environ.get("QAGENT_RESEARCH_EVAL_LLM") == "1"


def normalize(v: str | None) -> str | None:
    if not v:
        return None
    return STATE_TO_VERDICT.get(v, v)


def claims_of(task_hash: str, since: str = "") -> list[str]:
    """取本轮的 claim 判定。

    `since` 必须有: task_hash 按 question 生成, 重跑同一问题会落进同一个
    task_hash, 而 claim 主键是内容哈希 —— 上次拆出的命题如果和这次不同,
    **旧记录会残留**。实测踩过: 审美题残留 13:49 的 undecidable + 14:14 的 proved,
    混在一起被误判成"结论矛盾", 基线数据不可信。
    """
    rows = list_claims(task_hash=task_hash)
    if since:
        rows = [r for r in rows if (r.get("updated_at") or "") >= since]
    return [normalize(r.get("verdict")) for r in rows]


def fold(verdicts: list[str]) -> str | None:
    vs = [v for v in verdicts if v]
    if not vs:
        return None
    for v in vs:
        if v in OVERCONFIDENT:
            return v
    return vs[-1]


def contradiction(verdicts: list[str]) -> bool:
    c = [v for v in verdicts if v in OVERCONFIDENT]
    return "proved" in c and "refuted" in c


def run_loop(question: str) -> None:
    from agent.tools.research.research_investigate import research_investigate

    for _ in research_investigate(question=question):
        pass


def main() -> int:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    llm = [c for c in cases if c["tier"] == "llm"]
    since = datetime.now().isoformat(timespec="seconds")

    if RUN_LLM:
        print("模式: 跑真实 LLM 闭环 ...\n")
        for c in llm:
            print(f"  → {c['id']}")
            run_loop(c["question"])

    print("=== LLM 层评估基线 ===\n")
    print(f"{'题目':38} {'类别':10} {'代表判定':14} 全部")
    print("-" * 100)

    n_wrong = n_trap_bad = n_contra = 0
    wrong, blew, contra = [], [], []

    for c in llm:
        th = hash_question(c["question"])
        vs = claims_of(th, since=since)
        actual = fold(vs)

        exp = c["expect_verdict"]
        exp_set = {exp} if isinstance(exp, str) else set(exp)
        ok = actual in exp_set

        print(f"{c['id']:38} {c['category']:10} {str(actual):14} {vs}")

        if not ok:
            n_wrong += 1
            wrong.append(f"{c['id']}: 期望 {sorted(exp_set)}, 实得 {actual!r}  (全部 {vs})")

        if c["category"] == "trap":
            bad = sorted({v for v in vs if v in OVERCONFIDENT})
            if bad:
                n_trap_bad += 1
                blew.append(f"{c['id']}: 出现 {bad} —— 对不可判定问题硬下了结论")

        if contradiction(vs):
            n_contra += 1
            contra.append(f"{c['id']}: 同一任务内 proved 与 refuted 并存: {vs}")

    n = len(llm)
    n_trap = len([c for c in llm if c["category"] == "trap"])
    print("-" * 100)
    print(f"判定错误 {n_wrong}/{n} | 陷阱硬吹 {n_trap_bad}/{n_trap} | 结论矛盾 {n_contra}/{n}")

    if wrong:
        print("\n判定错误明细:")
        for x in wrong:
            print("  " + x)
    if blew:
        print("\n陷阱硬吹明细:")
        for x in blew:
            print("  " + x)
    if contra:
        print("\n结论矛盾明细:")
        for x in contra:
            print("  " + x)

    return 0


if __name__ == "__main__":
    sys.exit(main())
