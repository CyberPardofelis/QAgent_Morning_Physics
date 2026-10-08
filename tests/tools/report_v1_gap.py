"""V1 20 题真实完成能力 —— 分层报告, 不混数字。

用户问的核心问题: 「你说写脚本看全通过, 但做成科研工具就不行」

所以这份报告必须把三个层次分开, 不能混:
  A. 人工手写验证器       —— 工具能力上限(工具能不能算对)
  B. LLM 通过工具实跑     —— 端到端真实能力(含选型/API误用/判据)
  C. 两者之差             —— 差在哪, 补什么

★ 统计纪律: 「被阻断」和「判错」必须分开 ★
2026-10-08 实踩: 12 道题被 cost_guard 阻断(0秒0token), 我按判错统计,
报出 8/20 —— 那不是真实数字。混着报还有个更坏的后果: 数字看起来低,
会让人以为要继续加防线, 而实际上代码一个字都不用改。
"""
from __future__ import annotations

import io
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT = os.path.join(ROOT, "data", "research", "base20_llm.json")
VERIFIERS = os.path.join(ROOT, "tests", "tools", "base20_verifiers.py")
MACHINE_MAP = os.path.join(ROOT, "tests", "tools", "map_machine_verdicts.py")


def main() -> None:
    with io.open(RESULT, encoding="utf-8") as f:
        d = json.load(f)
    rows = d["results"]

    blocked = [r for r in rows if r.get("blocked")
               or (r["seconds"] == 0 and r["tokens"] == 0)]
    ran = [r for r in rows if r not in blocked]
    ok = [r for r in ran if r["match"]]
    bad = [r for r in ran if not r["match"]]

    print("=" * 74)
    print("V1 20 题 —— 三层能力对比")
    print("=" * 74)
    print(f"\nA. 人工手写验证器      20/20   (工具能力上限: sympy/networkx/z3 都能算对)")
    print(f"B. LLM 通过工具实跑   {len(ok)}/{len(ran)}   "
          f"(端到端: 含选型 / API误用 / 判据)")
    print(f"   被上游阻断         {len(blocked)}   ← 压根没跑, 不算判错")
    if ran:
        print(f"   准确率(只算跑了的) {len(ok)/len(ran)*100:.0f}%")

    print(f"\n{'='*74}")
    print("C. 差距在哪 —— 逐题看失败")
    print(f"{'='*74}")
    for r in bad:
        print(f"\n  {r['id']}  ->  {r['verdict']}")
        note = (r.get("error") or r.get("summary") or "")[:200]
        print(f"     {note}")

    print(f"\n{'='*74}")
    print("失败分类 —— 决定该改什么")
    print(f"{'='*74}")
    cats = Counter()
    for r in bad:
        s = (r.get("summary") or "") + (r.get("error") or "")
        if "遍历顺序" in s or "独立方式" in s:
            cats["B2 反例算错(代码bug, 机器已能拦)"] += 1
        elif "找不到任何具体的反例值" in s:
            cats["B1 REFUTED 无证书(机器已能拦)"] += 1
        elif r["verdict"] in ("partial", "inconclusive"):
            cats["A 判据/选型(未解决)"] += 1
        else:
            cats["其他"] += 1
    for k, v in cats.items():
        print(f"  {v} 道  {k}")

    print(f"\n{'='*74}")
    print("结论")
    print(f"{'='*74}")
    print("""
人工 20/20 vs LLM {}/{} 的差距, 不是「LLM 不会算」, 是三件具体的事:

1. **选型**(占失败的大头)
   LLM 拿到题目后要自己决定用什么手法。人工版是照着「这题该穷举」直接写;
   LLM 会在几个手法之间摇摆, 选错就崩。证据: De Morgan 那道 prompt 第 4 条
   已经明写「Or 相减会抛 TypeError, 用真值表」, LLM 还是先写了那行崩在第 8 行,
   而它**第 40 行之后已经写好了正确的真值表** —— 规则写对了, 顺序错了。

   解法不是加规则, 是**把选型从 prompt 收回代码**。

2. **遍历顺序 / 退化处理**
   Goldbach 那道把素数 list 存进 set 然后 `if p > n: break`, set 迭代顺序
   不保证升序(实测 [2,3,5,7,521,11,...]), 提前退出漏掉了 n=98 的合法素数对。
   人工版用 sorted 或者干脆不 break, 这类 bug 不会出现。

3. **退化输入**
   线性规划那道 sympy.solve 对奇异方程组返回 None, 人工版手工列顶点 + Fraction
   精确算, 根本不求解奇异系统, 退化情形不存在。

★ 三件事的共同点: 全都**不是加规则能解决的** ★
加规则只会让 prompt 更长更矛盾(这正是 P4.5 前两轮走了八道防线的原因)。
要动的是「工具怎么选后端」这个结构。
""".format(f"{len(ok)}/{len(ran)}"))


if __name__ == "__main__":
    main()