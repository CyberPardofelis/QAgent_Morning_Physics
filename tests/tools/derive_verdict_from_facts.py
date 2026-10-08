"""★ 日心说能不能站住: 判定真的能从输出推出来吗? ★

## 要验的

「不让模型写 VERDICT, 判定由机器从事实推出来」这套设计, 成立的前提是:
**事实里真的包含足够的信息**。如果推不出来, 那只是把说谎从模型挪到了
正则里 —— 那还是本轮, 不是日心说。

所以拿 V2 实跑落盘的真实 stdout 试: 遮住 VERDICT 那一行, 只看剩下的事实,
机器能不能推出和当时一样的判定。

## 怎么算「推对」

`baseline`: 当初带 VERDICT 那轮机器的判定
`derived` : 遮住 VERDICT 行后, 只按「证据是什么 -> 判定就是什么」推出来的
两者一致 = 事实里信息够; 不一致 = 事实里缺东西, 提案不成立。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import agent.tools.research._provenance as P  # noqa: E402

CASES = "tests/tools/base20_v2_cases.json"
ROOT = os.path.join("data", "research", "working")


def strip_verdict(stdout: str) -> str:
    """把 VERDICT 标记行遮掉 —— 模拟"模型根本不写判定"。"""
    return "\n".join(l for l in stdout.splitlines()
                     if "VERDICT" not in l.upper())


def derive(fact_only_stdout: str, claim: str, code: str) -> tuple[str, str]:
    """日心说式的判定: 只看事实, 不问模型说了什么。"""
    # 1. 跑成了吗
    if code and "SyntaxError" in code:
        return "failed", "没跑成(语法)"
    # 2. 交出凭证了吗 —— 有就是 refuted, 不需要模型宣布
    cred = P._counterexample_credential(fact_only_stdout)
    if cred:
        return "refuted", f"交出了凭证: {list(cred)[:4]}"
    # 3. 真的做过检查吗? 没有任何计算痕迹 -> 什么都没证明
    if not re_numbers(fact_only_stdout):
        return "failed", "输出里没有任何计算结果"
    # 4. 覆盖了命题自述的范围吗 -> proved; 否则 partial
    scoped = P._claim_declares_finite_scope(claim)
    said_all_passed = bool(P._ALL_PASSED.search(fact_only_stdout))
    if said_all_passed:
        return ("proved" if scoped else "partial"), "自述全过"
    # 5. 只报了"没找到" -> 什么都没证明
    if P._NOTHING_FOUND.search(fact_only_stdout):
        return "partial", "自述没找到"
    return "partial", "有计算但不足以支撑 proved"


def re_numbers(s: str) -> bool:
    import re
    return bool(re.search(r"\d", s))


def main() -> None:
    cases = {c["id"]: c for c in json.load(io.open(CASES, encoding="utf-8"))["cases"]}
    rows = []
    for cid, c in cases.items():
        th = "v2_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
        p = os.path.join(ROOT, th, "_evidence.jsonl")
        if not os.path.isfile(p):
            continue
        recs = [json.loads(x) for x in io.open(p, encoding="utf-8")
                if x.strip()]
        recs = [r for r in recs if r.get("kind") == "code" and r.get("stdout")]
        if not recs:
            continue
        for r in recs:
            out = r.get("stdout") or ""
            base, _ = P.extract_verdict_marker(out)
            if base is None:
                continue
            facts = strip_verdict(out)
            got, why = derive(facts, c["prompt"], r.get("code") or "")
            rows.append((cid, base, got, why))

    same = [r for r in rows if r[1] == r[2]]
    diff = [r for r in rows if r[1] != r[2]]
    print("=" * 74)
    print(f"遮住 VERDICT 行后, 机器能不能推出同样的判定? (共 {len(rows)} 条证据)")
    print("=" * 74)
    print(f"  推对   {len(same)}")
    print(f"  推错   {len(diff)}")
    for cid, base, got, why in diff:
        print(f"    {cid:36} 模型说={base:12} 推出={got:12} ({why})")
    print("=" * 74)
    if not diff:
        print("★ 事实里包含的信息足够 —— 「模型不写 VERDICT, 机器从事实推」成立")
    else:
        print(f"★ 有 {len(diff)} 条推不出来 —— 事实里缺信息, 提案**不成立**")
        print("  (典型: 模型心里知道该判 PROVED 但没在输出里留下痕迹)")


if __name__ == "__main__":
    main()