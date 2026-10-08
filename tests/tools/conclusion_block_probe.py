"""★ 结论块(conclusion block)能不能替代那些歧义启发式? ★

## 上一轮实验否证了什么

「不让模型写 VERDICT, 判定全由机器从事实推出来」—— 0/23 全错。
致命的一条: hard-007(Ramsey) 的探索过程里留下一行
`A：9 顶点全称命题的反例可满足性 = sat`, 而最终结论是 PROVED。
没有边界, 机器读不出"这行是探索还是结论", 于是把 proved 推成 refuted。

⇒ 最终判定行是**承重的**, 不是装饰。

## 那真正缺的是什么

缺的不是"更多检查", 缺的是**输出结构**: 机器需要知道哪些行是**声明**。
现在所有机器门都只能扫描整份 stdout, 于是不得不为每种歧义加启发式
(复合题识别、各种措辞猜测) —— 那就是本轮。

## 这个实验验的是

如果 stdout 里有一个**显式结论块**, 机器只对块内的行裁决:
  · 探索期的 `反例:` 行落在块外 → 不算凭证 → 不会误判 refuted ✓
  · 结论块里的自相矛盾 → 照样拦 ✓

那么现有的歧义启发式(R3 复合题识别等)就不需要了 —— 不是因为它们错了,
而是因为歧义**从源头消失了**。

本实验: 拿真实证据模拟"结论块 = VERDICT 行前最后 N 行", 看两类判据的
判定是否与原始一致。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import agent.tools.research._provenance as P  # noqa: E402

CASES = "tests/tools/base20_v2_cases.json"
ROOT = os.path.join("data", "research", "working")


def conclusion_block(stdout: str, window: int = 6) -> str:
    """结论块 = VERDICT 行及其前若干行。早于此的探索痕迹一律不算声明。"""
    lines = stdout.splitlines()
    idx = max((i for i, l in enumerate(lines) if "VERDICT" in l.upper()),
              default=len(lines) - 1)
    return "\n".join(lines[max(0, idx - window + 1):idx + 1])


def judge(code: str, full: str, block: str) -> tuple[str, str]:
    """同一个裁判, 喂整份输出 vs 只喂结论块。"""
    def r(out: str) -> SimpleNamespace:
        return SimpleNamespace(stdout=out, stderr="", ok=True, rejected=False,
                               timed_out=False, error_type="", code=code)
    return P.judge_verdict(r(full), code=code, claim="") or \
        P.judge_verdict(r(block), code=code, claim="")


def main() -> None:
    cases = {c["id"]: c for c in json.load(io.open(CASES, encoding="utf-8"))["cases"]}
    rows = []
    for cid, c in cases.items():
        th = "v2_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
        p = os.path.join(ROOT, th, "_evidence.jsonl")
        if not os.path.isfile(p):
            continue
        for x in io.open(p, encoding="utf-8"):
            if not x.strip():
                continue
            r = json.loads(x)
            if r.get("kind") != "code" or not r.get("stdout"):
                continue
            out = r["stdout"]
            base, _ = P.extract_verdict_marker(out)
            if base is None:
                continue
            blk = conclusion_block(out)
            v_full, _ = judge(r.get("code") or "", out, blk)
            v_blk, _ = judge(r.get("code") or "", out, blk)
            rows.append((cid, base, v_full, v_blk))

    # 只看结论块版本有没有把 proved/inconclusive 变成 refuted —— 那是危险的漂移
    risky = [r for r in rows if r[1] != "refuted" and r[3] == "refuted"]
    changed = [r for r in rows if r[2] != r[3]]
    print("=" * 74)
    print(f"结论块 vs 整份输出 (共 {len(rows)} 条证据)")
    print("=" * 74)
    print(f"  判定发生变化      {len(changed)}")
    print(f"  **危险漂移**      {len(risky)}  ← 非 refuted 被推成 refuted")
    for cid, base, a, b in changed:
        flag = "  <-- 危险" if (cid, base, a, b) in risky else ""
        print(f"    {cid:36} {base:12} 整份={a:12} 结论块={b:12}{flag}")
    print("=" * 74)
    if not risky:
        print("★ 结论块没有把任何非 refuted 推成 refuted —— 探索期歧义被隔开了")
    else:
        print(f"★ 仍有 {len(risky)} 条危险漂移 —— 结论块窗口不够或判据需调整")


if __name__ == "__main__":
    main()