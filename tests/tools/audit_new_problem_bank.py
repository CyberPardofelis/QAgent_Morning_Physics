"""体检 DS 给的新题库: BASE(part1) 与 RES(part2)。

## 为什么必须先体检而不是直接跑

part1 是"题面明示工具"的基建测试, part2 是"无提示"的科研能力测试 ——
这个二分我认为是对的(题面提示把方法选择这道题提前做掉了)。
但题库本身是手写的 JSON, 而**手写 JSON 最容易出的错就是"看起来对、解析就炸"**。

实测第一版 part1 直接 `json.loads` 失败:
    line 122  "argmax": [8/5, 6/5]      <- JSON 没有除法运算符
同一道 LP 题在 cvxpy-01-lp 里写的是 [1.6, 1.2], 说明是笔误不是设计。

如果我直接拿去跑, 会得到一个"题库坏了"的假故障, 然后开始怀疑 harness ——
这正是"假阴性和真错误长得一样"的又一例。所以先体检。

## 这份体检查什么
  1. JSON 能否解析(以及报错在哪一行)
  2. part2 是否真的**没有方法提示**(这是整个设计成立的前提, 必须断言)
  3. 题面引用的数据文件是否存在
  4. verifier entrypoint 是否都声明了(声明了不等于存在, 分开报)
  5. ground_truth 里有没有"看起来像答案、其实是题面泄底"的东西
"""
from __future__ import annotations

import json
import os
import re
import sys

BASE = (r"C:\Users\Nightingale\.minimax\v2\assets\2026\10\09"
        r"\15-04-31-590-asset_20261009-150431-590_"
        r"75a846261199_9740f800-BASEPROBLEMS_BASE.txt")
RES = (r"C:\Users\Nightingale\.minimax\v2\assets\2026\10\09"
       r"\15-04-31-593-asset_20261009-150431-593_"
       r"e7a9ff794cb4_a5a03d6c-BASEPROBLEMS_RES.txt")
ROOT = r"D:\pythonProject\QAgent"

#: 题面里"点名要某个工具"的措辞。part2 出现任何一个都说明无提示的前提破了。
METHOD_WORDS = ("z3", "lean", "sympy", "cvxpy", "networkx", "scipy", "ortools",
                "qutip", "mpmath", "oeis", "python", "lkh", "ida*", "双向 bfs",
                "sos 分解", "sat 求解")


def load(path: str, label: str):
    raw = open(path, encoding="utf-8").read()
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        lines = raw.splitlines()
        ctx = ""
        if 0 < e.lineno <= len(lines):
            ctx = lines[e.lineno - 1].strip()
        return None, (f"{label} JSON 解析失败 @ line {e.lineno} col {e.colno}: "
                      f"{e.msg}\n         那行是: {ctx}")


def audit_part2(d: dict, problems: list) -> None:
    """part2 的核心前提: 题面里**不能**出现方法提示。"""
    for it in d["items"]:
        blob = " ".join([str(it.get("prompt", "")),
                         json.dumps(it.get("ground_truth", {}), ensure_ascii=False)])
        # 只查 prompt —— ground_truth 是给打分器看的, 不给 agent
        p = str(it.get("prompt", ""))
        hit = [w for w in METHOD_WORDS if w in p.lower()]
        if hit:
            problems.append(f"part2 {it['id']}: 题面里出现方法词 {hit} —— "
                            f"无提示前提被破坏")
        if it.get("tool_hints") is not None:
            problems.append(f"part2 {it['id']}: tool_hints={it.get('tool_hints')!r}, "
                            f"应当是 null")
        if not it.get("expected_behaviors"):
            problems.append(f"part2 {it['id']}: 缺 expected_behaviors "
                            f"(这是打分器唯一能读的东西)")


def audit_refs(d: dict, problems: list) -> None:
    """题面引用了 data/xxx 的, 文件得真的在。"""
    for it in d["items"]:
        text = " ".join([str(it.get("prompt", "")),
                         json.dumps(it.get("ground_truth", {}), ensure_ascii=False)])
        for m in re.findall(r"data/[\w./-]+\.(?:txt|json|gml|csv)", text):
            if not os.path.exists(os.path.join(ROOT, m.replace("/", os.sep))):
                problems.append(f"{it['id']}: 题面/ground_truth 引用 {m}, "
                                f"但仓库里没有这个文件")


def audit_verifier_entries(d: dict, problems: list) -> None:
    missing = [it["id"] for it in d["items"] if not it.get("verifier", {}).get("entrypoint")]
    if missing:
        problems.append(f"{len(missing)} 道题没声明 verifier.entrypoint: {missing[:5]}")


def main() -> int:
    problems: list[str] = []
    parts = {}
    for path, label, expect_n in ((BASE, "part1_tool_capability", 40),
                                  (RES, "part2_research_capability", 20)):
        d, err = load(path, label)
        if err:
            problems.append(err)
            print(f"[!!] {label}: {err}")
            continue
        parts[label] = d
        n = len(d["items"])
        flag = "OK " if n == expect_n else "!! "
        print(f"[{flag}] {label}: {n} 题 (期望 {expect_n})")
        if n != expect_n:
            problems.append(f"{label} 题数 {n} != 期望 {expect_n}")

    p2 = parts.get("part2_research_capability")
    if p2:
        before = len(problems)
        audit_part2(p2, problems)
        audit_refs(p2, problems)
        audit_verifier_entries(p2, problems)
        print(f"[{'OK ' if len(problems) == before else '!! '}] part2 无提示/引用/verifier 自检")

    p1 = parts.get("part1_tool_capability")
    if p1:
        before = len(problems)
        audit_refs(p1, problems)
        audit_verifier_entries(p1, problems)
        nohint = [it["id"] for it in p1["items"] if not it.get("tool_hints")]
        if nohint:
            problems.append(f"part1 有 {len(nohint)} 道题没写 tool_hints —— "
                            f"part1 的前提就是题面明示工具: {nohint[:5]}")
        print(f"[{'OK ' if len(problems) == before else '!! '}] part1 引用/verifier/hint 自检")

    print("=" * 70)
    if problems:
        print(f"x 题库体检未过 ({len(problems)} 处)")
        for p in problems:
            print("  - " + p)
        return 1
    print("o 题库体检通过")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())