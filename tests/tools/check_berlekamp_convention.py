"""验 res-005 的 verifier 现在接受**两种 BM 记法**, 且仍拦得住非最小阶。

背景: 原来它要求 agent 的系数数组与参考逐项模相等, 而 BM 有两种通行记法
(题面的 c_i 记法 与 标准库返回的 C[] 记法) 互为模负数 —— 于是正确的实现
被判错。这是"verifier 考表示形式而不是考数学"的第三例。
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import verifier_map as VM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
P = os.path.join(ROOT, "data", "res_05_sequence.txt")


def load_seq():
    txt = io.open(P, encoding="utf-8").read()
    blocks = [b for b in re.findall(r"(?m)^[\s\d,\-]+$", txt)
              if len([x for x in re.split(r"[,\s]+", b) if x]) >= 20]
    return [int(t) for t in re.split(r"[,\s]+", blocks[0]) if t]


def main() -> int:
    seq = load_seq()
    mod = VM.load_verifier("res-005-berlekamp-massey")
    seeds = seq[:5]
    M = 2147483647
    cases = [
        ("参考记法 c_i", {"init_vals": seeds, "coeffs": [3, 7, 11, 17, 23],
                          "order": 5}, True),
        ("BM 的 C 记法(模负)",
         {"init_vals": seeds,
          "coeffs": [(M - x) % M for x in (3, 7, 11, 17, 23)], "order": 5},
         True),
        ("非最小(阶6, 但能复现)",
         {"init_vals": seeds, "coeffs": [3, 7, 11, 17, 23, 3],
          "order": 6}, False),
        ("系数改一个", {"init_vals": seeds, "coeffs": [4, 7, 11, 17, 23],
                       "order": 5}, False),
        ("没给制品", None, False),
    ]
    bad = []
    for name, art, want in cases:
        if art is None:
            stdout = "VERDICT: PROVED\n"
        else:
            stdout = ("ANSWER: " + json.dumps(art, ensure_ascii=False)
                      + "\nVERDICT: PROVED\n")
        got = bool(mod.verify(stdout, "", "")["correct"])
        flag = "OK " if got == want else "!! "
        if got != want:
            bad.append(name)
        print(f"  [{flag}] {name:24} 期望={want} 实际={got}")
    print("=" * 66)
    print("全部符合预期" if not bad else f"不符: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())