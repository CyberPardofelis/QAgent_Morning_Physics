"""验证「判定从制品推导」的三条核心路径。

★ 判据 ★
  1. 有反例制品 → 推导 refuted, **与模型声明无关**
  2. 宣称结论却无制品 → 降级 inconclusive(这是今晚 res-006/013/014 的形态)
  3. 模型声明与推导不一致时, mismatch=True —— 这个诊断量是新的
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agent.tools.research import _verdict as V  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("=" * 74)
    print("判定从制品推导 —— 不是从 agent 声明")
    print("=" * 74)

    print("\n>>> 路径 1: 有反例制品 -> 判假, 不看它宣称什么")
    ev = {"stdout": "反例: n=18, n^2+n+1=343 含平方因子 49\n"
                    "COUNTEREXAMPLE: {\"n\": 18}\n"
                    "VERDICT: PROVED\n"}          # ← 它竟然宣称 proved
    d = V.derive(ev, "refuted", "A线: 有反例凭证")
    check("有反例制品 -> refuted", d["verdict"] == "refuted", f"实得 {d['verdict']}")
    check("认出它宣称的是 proved", d["declared"] == "proved")
    check("标记为不一致", d["mismatch"] is True,
          f"声明={d['declared']} 推导={d['verdict']}")

    print("\n>>> 路径 2: 宣称结论却无制品 -> 降级(今晚三道题的形态)")
    for claimed, name in (("proved", "宣称 proved"), ("refuted", "宣称 refuted")):
        ev = {"stdout": f"我算出来了\nVERDICT: {claimed.upper()}\n"}
        d = V.derive(ev, "proved", "A线: 自称成立")
        check(f"{name} 但无制品 -> inconclusive",
              d["verdict"] == "inconclusive", f"实得 {d['verdict']}")
        check("  记为不一致", d["mismatch"] is True)

    print("\n>>> 路径 3: 什么都没说 -> untested, 不是无中生有的结论")
    ev = {"stdout": "一些运行日志, 没有 VERDICT 也没有制品\n"}
    d = V.derive(ev, "untested", "A线: 没有标记")
    check("无声明无制品 -> untested", d["verdict"] == "untested", f"实得 {d['verdict']}")
    check("不算不一致(它本来就没声明)", d["mismatch"] is False)

    print("\n>>> 路径 4: 有制品且程序正常 -> 采信 A 线")
    ev = {"stdout": "CONSTRUCT: {\"x\": 1}\nVERDICT: PROVED\n"}
    d = V.derive(ev, "proved", "A线: 全部检查通过")
    check("制品+正常结束 -> proved", d["verdict"] == "proved", f"实得 {d['verdict']}")
    check("声明与推导一致", d["mismatch"] is False)

    print("\n>>> 汇总输出(新诊断量)")
    rows = [
        {"declared": "proved", "verdict": "refuted", "mismatch": True},
        {"declared": "proved", "verdict": "inconclusive", "mismatch": True},
        {"declared": "proved", "verdict": "proved", "mismatch": False},
        {"declared": None, "verdict": "untested", "mismatch": False},
    ]
    print(V.summarize(rows))

    print("\n" + "=" * 74)
    if FAILS:
        print(f"✗ {len(FAILS)} 项不通过: " + "; ".join(FAILS))
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())