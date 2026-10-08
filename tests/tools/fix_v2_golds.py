# -*- coding: utf-8 -*-
"""更新 base20_v2_cases.json 的 gold 分类。

改三处, 每处都有理由 —— 分类标准只有一个: **命题自己有没有限定范围**。

1. hard-015 / hard-016: `lean_needed` -> `proved`
   理由: Mathlib v4.9.0 已装好并实测编译通过(费马小定理、sqrt2 无理),
   "环境不具备"已经不成立。它们现在是**正常的待证明题**, 不是免责项。
   留着 lean_needed 等于给这两道题永久豁免, 而豁免会掩盖真实能力。

2. hard-012 (Erdos-Straus): `partial_ok` -> `proved`
   理由: 命题原文写死了「对 2 <= n <= 1000 成立」—— 这是**有界命题**,
   穷举 2..1000 就是完整证明, 判 PROVED 完全正确。
   之前把它当陷阱题("判全域 proved 算错")是我判错了:
   "全域"是针对**无界**命题的词, 对有界命题用它是范畴错误。
   这条修正来自 tests/tools/verifiers_v2/hard_12_erdos_straus.py 的实跑(0.10s PROVED)。
"""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "base20_v2_cases.json")

CHANGES = {
    "hard-015-fermat-little-lean": "proved",
    "hard-016-sqrt2-irrational-lean": "proved",
    "hard-012-erdos-straus-1000": "proved",
}


def main() -> int:
    with io.open(P, encoding="utf-8") as f:
        data = json.load(f)
    n = 0
    for c in data["cases"]:
        new = CHANGES.get(c["id"])
        if new and c.get("gold") != new:
            print(f"  {c['id']}: {c['gold']} -> {new}")
            print(f"     理由: {c.get('gold_note', '(见脚本 docstring)')}")
            c["gold"] = new
            c.pop("gold_note", None)
            n += 1
    data["gold_revision"] = (
        "2026-10-08 修: 015/016 因 Mathlib v4.9.0 装好而转正为普通题; "
        "012 的命题自带 2<=n<=1000 范围, 判 PROVED 是对的, 不再当陷阱题。"
    )
    with io.open(P, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n改了 {n} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())