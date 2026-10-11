# -*- coding: utf-8 -*-
"""res-002-golomb-ruler-15 —— 15 阶 Golomb 尺的独立验证器。

题目: 「构造一个 15 阶 Golomb 尺, 长度不超过 200, 并给出所有两两距离互不相同的证书。」

★ 判据为什么「短」却「够」★
Golomb 尺的定义就是: 15 个刻度 m_0 < m_1 < ... < m_14, 且**任意两刻度之差两两不同**。
C(15,2) = 105 个差值必须恰好占满 105 个互不相同的正整数。于是:

    105 个互不相同的正整数 <= m_14 - m_0 <= 200

也就是说这把尺的判据是一个**纯有限检查**, 不需要任何搜索、任何最优性论证:
  1. 结构: 15 个整数、m_0 = 0、严格递增;
  2. 长度: m_14 <= 200;
  3. 距离: 105 个差值全部互异(本文件自己重算, 不看 agent 的自检输出)。

★ 为什么长度上界只当「题面要求」而不是「最优性」★
15 阶最优 Golomb 尺长度是 151, 但题目只要求 <= 200。verifier 判**题目**,
不判「是否最优」—— 越界地要求最优会把一个满足题面的正确答案判 0 分。
最优性属于 `final_answer` 之外的加分项, 这里刻意不设卡。

★ 顺序敏感性 ★
本文件**不排序** agent 给的 marks 再验。若 agent 交出乱序或含重复的刻度,
那把尺按定义就非法(刻度是尺上的位置, 必须互异且有序), 必须判 0 分;
默默替它排序等于替它修答案。
"""

from __future__ import annotations

import json
import pathlib
import sys
from itertools import combinations

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import Check, parse_artifact, parse_verdict  # noqa: E402

#: 题面参数(见 baseproblems_R.txt 的 verifier.params)。
ORDER = 15
MAX_LENGTH = 200


def _all_distances(marks: list[int]) -> list[int]:
    """自己重算全部 C(15,2) 个两两距离。

    ★ 为什么必须绝对值 ★
    严格递增的 marks 已经保证 m_j - m_i > 0 (i<j), 所以这里不用 abs;
    但为了让函数在乱序输入下也给出**稳定**的语义, 仍统一取正差。
    """
    ds = []
    for i, j in combinations(range(len(marks)), 2):
        d = marks[j] - marks[i]
        ds.append(d if d >= 0 else -d)
    return ds


def _find_duplicates(ds: list[int]) -> dict[int, int]:
    """找出重复的距离值 -> 出现次数(诊断用)。"""
    counts: dict[int, int] = {}
    for d in ds:
        counts[d] = counts.get(d, 0) + 1
    return {d: c for d, c in counts.items() if c > 1}


def verify(stdout: str, code: str, claim: str) -> dict:
    """独立判定: 制品里的 15 个刻度是否构成长度 <= 200 的 Golomb 尺。

    `code` 参数**故意不使用** —— 执行 agent 代码等于让它自己给自己打分。
    """
    ck = Check()
    art = parse_artifact(stdout, "CONSTRUCT") or parse_artifact(stdout)
    if art is None:
        return {
            "correct": False,
            "certificate": None,
            "final_answer": None,
            "details": {"artifact": False,
                        "note": "未提交 CONSTRUCT: 制品 —— 按契约视为未作答"},
        }
    ck.expect("artifact", True)

    marks_raw = None
    for key in ("marks", "ruler", "positions"):
        if key in art:
            marks_raw = art[key]
            break
    if not isinstance(marks_raw, list):
        ck.expect("marks_present", False, "制品里没有 marks 列表")
        return _finish(ck, None, art, stdout)
    ck.expect("marks_present", True)

    # --- 结构: 必须恰好 ORDER 个整数 ---
    ints = [m for m in marks_raw if isinstance(m, int) and not isinstance(m, bool)]
    ck.expect("len_marks_%d" % ORDER, len(marks_raw) == ORDER,
              f"刻度数 {len(marks_raw)} != {ORDER}")
    ck.expect("all_int", len(ints) == len(marks_raw),
              f"非整数刻度: {[m for m in marks_raw if m not in ints]}")
    if len(marks_raw) != ORDER or len(ints) != len(marks_raw):
        return _finish(ck, None, art, stdout)

    marks = ints

    # --- m_0 = 0 且严格递增 ---
    ck.expect("marks0_is_zero", marks[0] == 0, f"marks[0]={marks[0]} != 0")
    ck.expect("strictly_increasing", all(marks[i] < marks[i + 1] for i in range(ORDER - 1)),
              "刻度未严格递增(或存在重复刻度)")
    ck.expect("non_negative", all(m >= 0 for m in marks), "存在负刻度")

    # --- 长度 ---
    length = marks[-1] - marks[0]
    ck.expect("length_le_%d" % MAX_LENGTH, length <= MAX_LENGTH,
              f"尺长 {length} > {MAX_LENGTH}")
    # agent 自报的 length 只作旁证; 判据永远用自己算的 length
    if "length" in art:
        ck.note("reported_length", art["length"])
        ck.note("reported_length_agrees", art["length"] == length)

    # --- 核心判据: 105 个两两距离两两不同 ---
    ds = _all_distances(marks)
    ck.expect("distance_count_105", len(ds) == ORDER * (ORDER - 1) // 2,
              f"距离数 {len(ds)} != 105")
    dups = _find_duplicates(ds)
    ck.expect("all_distinct", not dups,
              f"重复距离: {dict(list(dups.items())[:5])}")

    ck.note("length", length)
    ck.note("n_distinct_distances", len(set(ds)))
    return _finish(ck, marks, art, stdout)


def _finish(ck: Check, marks, art: dict, stdout: str) -> dict:
    if marks is None:
        # 早退路径也要带 reason, 否则打分器看到 details 缺字段没法归因
        return {"correct": False, "certificate": None, "final_answer": None,
                "details": {**ck.details, "verdict": parse_verdict(stdout),
                            "reason": ck.reason()}}
    correct = ck.correct
    ds = sorted(_all_distances(marks))
    cert = None
    if correct:
        cert = json.dumps({
            "marks": marks,
            "length": marks[-1] - marks[0],
            "n_distances": len(ds),
            "sorted_distances": ds,
            "distinct": len(set(ds)) == len(ds),
        }, ensure_ascii=False)
    return {
        "correct": correct,
        "certificate": cert,
        "final_answer": {
            "order": len(marks),
            "length": marks[-1] - marks[0],
            "n_distinct_distances": len(set(ds)),
        } if correct else None,
        "details": {**ck.details, "verdict": parse_verdict(stdout), "reason": ck.reason()},
    }


# ==========================================================================
# 自测
# ==========================================================================
#: 正例: 一把 15 阶 Golomb 尺(长度 170 <= 200), 105 个距离互不相同。
#: 由随机化贪心搜索得到; 本文件在 _selftest 里会用与 verify 相同的判据复核它。
_GOOD_MARKS = [0, 2, 11, 21, 25, 28, 43, 55, 63, 79, 92, 125, 164, 165, 170]


def _mk(marks, verdict="PROVED") -> str:
    return (f"CONSTRUCT: {json.dumps({'kind': 'golomb_ruler', 'marks': marks})}\n"
            f"VERDICT: {verdict}\n")


def _selftest() -> bool:
    ok = True

    # 1) 正例
    r = verify(_mk(_GOOD_MARKS), "", "")
    ok &= r["correct"] is True
    print(f"[{'OK' if r['correct'] else 'FAIL'}] 正例(长度 {170}) correct={r['correct']} reason={r['details'].get('reason')}")

    # 2) 反例 A: 仍是 15 个递增整数、长度不变, 只把第 2 个刻度 2->3。
    #    这种「几乎对」的答案最容易被 agent 的自检漏掉, 因此必须测。
    bad = list(_GOOD_MARKS)
    bad[1] = 3
    r = verify(_mk(bad), "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例A(marks[1] 2->3) correct={r['correct']} reason={r['details'].get('reason')}")

    # 3) 反例 B: 14 个刻度(数量不够)
    r = verify(_mk(_GOOD_MARKS[:14]), "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例B(只有 14 个刻度) correct={r['correct']} reason={r['details'].get('reason')}")

    # 4) 反例 C: 等差数列 0..14 —— 结构合法但 105 个距离里全是重复
    r = verify(_mk(list(range(15))), "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例C(等差 0..14) correct={r['correct']} reason={r['details'].get('reason')}")

    # 5) 反例 D: 无制品
    r = verify("VERDICT: PROVED\n", "", "")
    ok &= r["correct"] is False
    print(f"[{'OK' if r['correct'] is False else 'FAIL'}] 反例D(无制品) correct={r['correct']} reason={r['details'].get('note')}")

    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)