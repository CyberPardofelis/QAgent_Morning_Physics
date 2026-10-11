"""把用户给的特殊题加入 part2 题库, 并附带已验证的参考答案与证明路线。

## 为什么单独一个脚本
这道题是用户后加的, 不是题库原有的。它有两个特别之处, 值得在题库里写清楚:

1. **题面有歧义**。图片里的式子 `e/(e-1) - sum (k/n)^n < 1/(n+1)`,
   分式整体带 n 次方。按组合数 `C(k,n)` 读, 求和恒等于 1, 命题退化成
   `1/(e-1) < 1/(n+1)`, 对 n>=2 **恒假**。
   ⇒ 题库里必须记死正确读法, 否则以后有人按错读法判分会得到"命题为假"的结论。

2. **它是渐近紧的**。相对余量稳定在 ~0.386%, 所以
   "随便加点误差"或"用浮点数蒙"都可能翻车。
   ⇒ 参考答案里必须记: 校验要用**精确有理数 + e 的有理区间**, 不能靠浮点。

用法:
    python tests/tools/add_special_problem.py --check
    python tests/tools/add_special_problem.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = r"D:\pythonProject\QAgent"
BANK = os.path.join(ROOT, "data", "lab", "part2_research_capability.json")

SPECIAL = {
    "id": "res-special-e-sum-power",
    "part": 2,
    "category": "proof",
    "group": "special",
    "prompt": (
        "证明：当正整数 n >= 2，\n"
        "    e/(e-1)  -  sum_{k=1}^{n} (k/n)^n  <  1/(n+1)\n"
        "其中 sum 对 k 从 1 求和到 n。"
    ),
    "tool_hints": None,
    "ground_truth": {
        "verdict": "true",
        "reading_critical": (
            "★ 求和项是 (k/n)^n —— 分式整体取 n 次方, **不是组合数 C(k,n)** ★\n"
            "按 C(k,n) 读: k<n 时该项为 0, 只有 k=n 贡献 1, 求和恒为 1,\n"
            "命题变成 1/(e-1) < 1/(n+1), 对 n>=2 恒假 —— 那是**读错题**, 不是命题假。\n"
            "(核验: n=2 时 (1/2)^2+(2/2)^2 = 1.25; e/(e-1)-1.25 = 0.33198 < 1/3 ✓)"
        ),
        "verified_range": "n = 2..400",
        "verification_method": (
            "★ 严格证法: S_n 用精确有理数; e 用有理区间 "
            "[2.71828182845904523535, 2.71828182845904523537]。\n"
            "f(e)=e/(e-1) 单调递减, 故 f(E_HI) 是 f(e) 的上界; "
            "若 上界 - S_n 仍 < 1/(n+1), 则不等式无条件成立。\n"
            "★ 不可用浮点: 相对余量只有 ~0.386%, 浮点尾数会掩盖真实余量。★"
        ),
        "asymptotic_constant": {
            "C": "e(e+1)/(2(e-1)^3) = 0.9961677...",
            "meaning": "diff = e/(e-1) - S_n 渐近于 C/n, 而 RHS = 1/(n+1) 渐近于 1/n",
            "relative_margin": "n>=50 起稳定在 ~0.386%",
            "observed_diff_times_n": {"n=100": 0.986264, "n=400": 0.993658},
        },
        "proof_plan": {
            "step1_reduction": (
                "令 j = n-k, 则 S_n = 1 + sum_{j=1}^{n-1}(1-j/n)^n; "
                "又 e/(e-1) - 1 = 1/(e-1)。故只需证\n"
                "    D_n := 1/(e-1) - sum_{j=1}^{n-1}(1-j/n)^n  <  1/(n+1)"
            ),
            "step2_key_inequality": (
                "对 u=j/n:  -n·ln(1-u) - nu = n·sum_{m>=2} u^m/m >= n·u^2/2 = j^2/(2n)\n"
                "故 (1-j/n)^n = e^{-j}·e^{-theta}, 其中 theta >= j^2/(2n)"
            ),
            "step3_bound_each_term": (
                "e^{-j} - (1-j/n)^n = e^{-j}(1 - e^{-theta}) "
                "<= e^{-j}(1 - e^{-j^2/(2n)}) <= e^{-j}·j^2/(2n)   [用 1-e^{-x} <= x]"
            ),
            "step4_sum": (
                "D_n <= (1/2n)·sum_{j>=1} j^2 e^{-j} + sum_{j>=n} e^{-j}\n"
                "    = e(e+1)/(2n(e-1)^3) + (e-1)^{-1}·e^{1-n}\n"
                "用到 sum_{j>=1} j^2 x^j = x(1+x)/(1-x)^3 在 x=1/e 处。"
            ),
            "step5_conclusion": (
                "记 C := e(e+1)/(2(e-1)^3) = 0.9961677... < 1。\n"
                "要 C/n < 1/(n+1) 需 C < n/(n+1), 即 n > C/(1-C) ≈ 260。\n"
                "=> **n >= 261 由该界直接证出**; n = 2..260 用精确有理数 + e 的"
                "有理区间逐个核验。"
            ),
            "caveat": (
                "★ 有限段的核验是穷举, 不是证法。真正的证明依赖 step2 那条对**所有** n "
                "成立的界。拿 n=2..400 的核验当作『证完了』是错的 —— 它只覆盖到 400。★"
            ),
        },
        "scoring_note": (
            "判对需要给出 step2 那条关键不等式(把 (1-u)^n 的二阶项抓出来)。"
            "只报『数值成立到 400』应判 PARTIAL。"
        ),
        "expected_behaviors": [
            "把求和变量代换 j=n-k, 化归到 D_n 的上界",
            "发现 -n·ln(1-u) 的二阶项 n·u^2/2 是关键 —— 它正好给出 j^2/(2n)",
            "用 sum j^2 e^{-j} 的闭式把差估成 C/n, C<1",
            "识别出 C/n < 1/(n+1) 只对 n 大于约 260 成立, 其余需有限核验并说明这是穷举",
            "不要只给数值证据就宣布证毕",
        ],
        "_verified": "2026-10-10",
    },
    "verifier": {
        "entrypoint": "manual",
        "pass_condition": (
            "给出上述归约与关键不等式(不得只给数值); "
            "并正确说明 C/n < 1/(n+1) 的适用范围 n>=261 与有限段的穷举性质"
        ),
    },
    "expected_behaviors": [
        "归约到 D_n",
        "用二阶 Taylor 项 -n·ln(1-u) >= nu + nu^2/2",
        "求和闭式 sum j^2 e^{-j} = e(e+1)/(e-1)^3",
        "区分『界对所有 n 成立』与『有限段穷举』",
    ],
    "difficulty": "hard",
    "notes": "用户 2026-10-10 追加的特殊考核题。图片给出, 求和范围 k=1..n 由用户补充说明。",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    bank = json.load(open(BANK, encoding="utf-8"))
    ids = [it["id"] for it in bank["items"]]
    if SPECIAL["id"] in ids:
        print(f"已存在 {SPECIAL['id']}, 先移除再重加")
        bank["items"] = [it for it in bank["items"] if it["id"] != SPECIAL["id"]]

    # 自检: 题面里的读法必须是 (k/n)^n
    assert "(k/n)^n" in SPECIAL["prompt"], "题面读法必须写死"
    assert "C(k,n)" not in SPECIAL["prompt"], "题面里不能出现错的读法"
    gt = SPECIAL["ground_truth"]
    assert gt["verdict"] == "true"
    assert "证明路线" not in gt     # 只是防手滑, 真字段名见 proof_plan

    bank["items"].append(SPECIAL)
    bank["_special_added"] = {
        "date": "2026-10-10",
        "ids": [SPECIAL["id"]],
        "why": "用户追加; 题面读法有歧义(易与 C(k,n) 混), 且不等式渐近紧(相对余量 0.386%), "
               "必须在题库里记死正确读法与严格校验方式。",
    }

    print(f"题库现有 {len(bank['items'])} 题 (原 20 + 特殊 1)")
    if not args.apply:
        print("(只校验, 未写回。加 --apply 才落盘)")
        return 0

    with open(BANK, "w", encoding="utf-8") as fh:
        json.dump(bank, fh, ensure_ascii=False, indent=2)
    back = json.load(open(BANK, encoding="utf-8"))          # 复读
    got = [it["id"] for it in back["items"] if it["id"] == SPECIAL["id"]]
    print(f"o 已写回并复读通过: {got} | 总题数 {len(back['items'])}")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())