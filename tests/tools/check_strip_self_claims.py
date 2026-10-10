"""`strip_self_claims` 必须在**数据层**拦住自评字段 —— 用 smoke 的真实制品钉住。

## 背景
2026-10-10 smoke 实测: Worker prompt 里明写「凡是你对答案的评价一律不要写」,
它照样交上来 `{"proposition_is_false": true, "witness": {...}}`。
**prompt 层的禁令拦不住** —— 宣布结论不要钱, 这条纪律必须落到代码上。

旧结构里那个 `SELF_EVAL_FIELDS` 是**已知字段名**的白名单, 结构上抓不到新名字
(`proposition_is_false` 就不在里面)。所以这里按**形状**判: 键名像断言 **且** 值像判定。

## 两条必须同时成立
1. 剥得掉: 自评字段真的进不了裁决
2. 不误伤: 真的把布尔当答案的题, 以及答案对象里恰好叫 verified 的数据, 不能被剥

第 2 条比第 1 条更重要 —— 剥错会把正确答案扔掉, 那比放过一个自评字段恶劣得多。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from agent.tools.research.roles.worker import strip_self_claims  # noqa: E402

FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    if not cond:
        FAILS.append(label)
    print(f"  [{'OK ' if cond else 'FAIL'}] {label}"
          + (f"\n         {detail}" if not cond and detail else ""))


# ★ smoke 里 Worker 真实交上来的制品, 逐字 ★
REAL = {"proposition_is_false": True,
        "witness": {"n": 40, "value": 1681, "factor_a": 41, "factor_b": 41}}


def main() -> int:
    print("=" * 70)
    print("制品层的自评剥离")
    print("=" * 70)

    print("\n>>> smoke 真实制品")
    out, st = strip_self_claims(REAL)
    check("自评字段被剥掉", "proposition_is_false" not in out,
          f"剩下: {out}")
    check("剥掉的是命题判定", st == ["proposition_is_false=True"], f"实得 {st}")
    check("★ 真正的答案(witness)必须完整保留",
          out.get("witness") == REAL["witness"], f"剩下: {out}")

    print("\n>>> 不误伤: 这些必须原样留下")
    cases = {
        "布尔本身就是答案(纠缠判定)": {"entangled": True},
        "答案对象里叫 verified 的数据": {"verified": [1, 2, 3]},
        "名字像断言但值是数据": {"exists_list": [1, 2, 3]},
        "普通数值": {"n": 40, "value": 1681},
        "嵌套里的数据": {"witness": {"n": 40, "value": 1681}},
        "名字像断言但值是长文本": {"verdict": "n=40 时 1681=41^2 故命题不成立"},
    }
    for label, art in cases.items():
        out, st = strip_self_claims(art)
        check(f"{label} 原样保留", out == art and not st,
              f"被改成 {out}, 剥了 {st}")

    print("\n>>> 该剥的都剥掉")
    more = {
        "最优性声明": {"optimal": True},
        "已验证声明": {"verified": True},
        "唯一性声明": {"unique": True},
        "字符串判定": {"verdict": "proved"},
        "成立声明": {"holds": True},
    }
    for label, art in more.items():
        out, st = strip_self_claims(art)
        check(f"{label} 被剥", out == {} and len(st) == 1,
              f"剩下 {out}, 剥了 {st}")

    print("\n>>> 边界")
    out, st = strip_self_claims(None)
    check("None 安全", out is None)
    out, st = strip_self_claims({})
    check("空字典安全", out == {} and not st)

    print("=" * 70)
    if FAILS:
        print(f"x {len(FAILS)} 项未过:")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("o 制品层自评剥离: 全部通过")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())