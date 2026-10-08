"""★ 覆盖率矩阵: 门必须由真实样本支撑, 不许凭空增加 ★

## 这条测试存在的理由

2026-10-08 用户当面批评「你是不是又在把问题越搞越复杂了」。复盘发现:
六道 REFUTED 侧的门**里���有 6 个是同一个问题**的不同措辞, 而每次翻车
就加一道, 门数只增不减。根因不是判断错了, 是**没有任何机制阻止下一次
"因为上一道门不够用"就再加一道**。

所以把那条纪律写成可执行的检查:

| 要求 | 拦住的坏行为 |
|---|---|
| 每个矛盾形态有 ≥1 条**咬住**测试 | 加了个门却没真实样本证明它咬得住 |
| 每个矛盾形态有 ≥1 条**不误杀**测试 | 只测"能拦住", 不测"不该拦时没拦" |
| 每道门至少被一条测试**实际调用** | 加了门忘了写测试 —— 那是死代码 |
| 判决实验覆盖全部**真实**假反例 | 用编的样例代替真实翻车现场 |

★ 判据本身也必须是真实样本 ★
样本统一放在 `REAL_SAMPLES`, 每条注明来源(哪道题、哪一天)。
这样"新增一条样本"和"新增一道门"是两件不同的事 ——
前者是采样, 后者需要判决实验背书。

## 怎么用

跑这个文件: `python tests/tools/check_gate_coverage.py`
- 全绿 = 门的存在都有数据支撑
- 红了 = 要么补样本, 要么把没用的门删掉(不要去改测试迁就实现)
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent.tools.research._provenance as P  # noqa: E402
import check_proved_machine_gate as G  # noqa: E402


#: ★ 真实翻车现场样本库 ★
#:
#: 每条都注明出处。**编的样例不进这里** —— 编的样例只能证明"代码按我写的
#: 思路运行", 不能证明"真实翻车长这样"。这是本轮踩过的坑。
REAL_SAMPLES = {
    "found_nothing": [
        # 出处: 2026-10-08 V2 hard-012 Erdős–Straus
        ("""检查范围: 2 到 1000
已验证的 n 数量: 0
在完整枚举范围内未找到正整数解
VERDICT: REFUTED
""", "说出没找到却判假"),
    ],
    "all_passed": [
        # 出处: 2026-10-08 V1 alg-006 Fibonacci 矩阵
        (G.FIB_STDOUT, "逐个验证全对却判假"),
    ],
    "affirmed_witness": [
        # 出处: 2026-10-08 V2 hard-001 Pell-61
        (G.PELL_STDOUT, "把正解标成反例, 另一句又说它成立"),
    ],
    "own_inconclusive": [
        # 出处: 2026-10-08 V2 hard-007 R(3,4)=9
        (G.RAMSEY_STDOUT, "各部分都定不了却汇总成假"),
    ],
    "disagreement": [
        # 出处: 2026-10-08 V2 hard-002 p(100)
        (G.PARTITION_STDOUT, "几路验算对不上就宣布命题为假"),
    ],
    "count_list": [
        # 出处: 2026-10-08 V2 hard-012 Erdős–Straus 另一轮
        (G.ERDOS_STRAUS_STDOUT, "声明 997 个失败只列了 20 个"),
    ],
}

#: 必须有"不误杀"反证的形态。
#:
#: `own_inconclusive` / `count_list` 没有单独的反向样本 —— 它们的豁免条件
#: 与别的形态共用(「输出里没有任何反例声明」), 由
#: `test_r5d_does_not_kill_real_counterexample_with_inconclusive_part` 等
#: 统一覆盖。所以这里列的是**该形态被豁免时由谁兜底**, 而不是重复造样本。
SHARED_EXEMPTION = {
    "own_inconclusive": "test_r5d_does_not_kill_real_counterexample_with_inconclusive_part",
    "count_list": "test_r7_does_not_kill_honest_truncation",
    "disagreement": "test_r6_does_not_kill_refutation_with_explicit_counterexample",
}


def _res(stdout: str, code: str = "x=1\n") -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr="", ok=True, rejected=False,
                           timed_out=False, error_type="", code=code)


#: PROVED 侧的真实样本。**连 code 和 claim 一起带** —— R1 是"读命题"的门,
#: 只给 stdout 会测到另一条路径去。
#:
#: FLT3 那个特别值得注意: **同一份代码 + 自带范围的命题 → R1 不触发(正确),
#: 换成无界命题 → R1 触发**。所以样本必须带 claim, 否则测不出来。
#: 「假 PROVED 比假 REFUTED 更严重」—— 错结论流出去当事实。
PROVED_SAMPLES = {
    "R2 把待证命题当前提用": [
        (G.SQRT2_STDOUT, G.SQRT2_CODE, G.SQRT2_CLAIM,
         "循环论证: 拿 sqrt(2) 是无理数当前提去证同一句"),
    ],
    "R1 有限枚举冒充全域": [
        (G.FLT3_STDOUT, G.FLT3_CODE,
         "证明不存在正整数 a,b,c 满足 a^3 + b^3 = c^3。",  # ★无界*命题
         "有界枚举 1..999 却宣布无界命题成立"),
    ],
}


def _all_gates() -> list[tuple[str, object, list]]:
    """(门名, 探针, 真实样本) 三元组。

    探针统一签名 `probe(stdout) -> 非 None 表示开火`。
    PROVED 侧的探针要把 code/claim **绑进闭包**, 因为 R1 是"读命题"的门 ——
    用默认参数绑定, 不用循环变量延迟求值(那会绑错值)。
    """
    out: list[tuple[str, object, list]] = []
    for kind, samples in REAL_SAMPLES.items():
        out.append((
            f"一致性门:{kind}",
            lambda s, k=kind: P._contradiction_msg(k, s),
            samples,
        ))
    out.append((
        "凭证门",
        P._refuted_without_credential,
        [(s, w) for samples in REAL_SAMPLES.values() for s, w in samples],
    ))
    for label, samples in PROVED_SAMPLES.items():
        first = samples[0]
        out.append((
            f"PROVED 侧:{label}",
            lambda s, code=first[1], claim=first[2]:
                P._audit_proved(code, s, claim),
            [(s, w) for s, _c, _cl, w in samples],
        ))
    return out


_ALL_GATES = _all_gates()


def check_every_kind_has_real_sample() -> list[str]:
    """每个矛盾形态都必须有真实样本。"""
    problems = []
    kinds = {k for k, _ in P._CONTRADICTIONS}
    for kind in sorted(kinds):
        if kind not in REAL_SAMPLES:
            problems.append(f"形态 `{kind}` 没有任何真实样本 —— "
                            f"加门之前先补一条真实翻车现场")
    for kind in sorted(set(REAL_SAMPLES) - kinds):
        problems.append(f"样本库里的 `{kind}` 已经不在门表里了 —— "
                        f"要么门被删了(删样本), 要么门改名了(改样本)")
    return problems


def check_samples_actually_bite() -> list[str]:
    """每条真实样本必须真的被对应形态咬住。"""
    problems = []
    for kind, samples in REAL_SAMPLES.items():
        for stdout, why in samples:
            if P._contradiction_msg(kind, stdout) is None:
                problems.append(f"形态 `{kind}` 咬不住它的真实样本({why}) —— "
                                f"这条门要么改坏了, 要么该样本不该放在这里")
    return problems


def check_exemption_still_works() -> list[str]:
    """加了凭证豁免之后, 这些形态必须还能放过真反例。"""
    problems = []
    for kind, test_name in SHARED_EXEMPTION.items():
        fn = getattr(G, test_name, None)
        if fn is None:
            problems.append(f"形态 `{kind}` 声称由 `{test_name}` 提供反向覆盖, "
                            f"但那个测试不存在")
            continue
        try:
            fn()
        except AssertionError as e:
            problems.append(f"形态 `{kind}` 的反向测试挂了: {e}")
    return problems


def check_every_gate_actually_fires() -> list[str]:
    """★ 每道门必须在真实样本上实际开火 ★

    第一版写成"在测试文件里 grep 到函数名", 那是**假检查** —— 所有门都是被
    `judge_verdict` 间接调用的, 测试里根本不会出现内部函数名, 于是全线误报。
    真正要问的是: 这道门**有没有在真实样本上拦下过东西**。

    抓的是"加了门忘了写测试"和"加了门但它其实什么都不拦"。
    """
    problems = []
    for label, probe, samples in _ALL_GATES:
        fired = [why for stdout, why in samples if probe(stdout) is not None]
        if not fired:
            problems.append(f"门 `{label}` 在 {len(samples)} 条真实样本上一条都没"
                            f"开火 —— 它要么是死代码, 要么样本选错了")
    return problems


def main() -> int:
    print("=" * 72)
    print("覆盖率矩阵 —— 门的存在必须有数据支撑")
    print("=" * 72)
    print(f"  门表里的形态      {len(P._CONTRADICTIONS)} 个: "
          f"{', '.join(k for k, _ in P._CONTRADICTIONS)}")
    print(f"  真实样本库        {len(REAL_SAMPLES)} 类 / "
          f"{sum(len(v) for v in REAL_SAMPLES.values())} 条\n")

    checks = [
        ("每个形态都有真实样本", check_every_kind_has_real_sample),
        ("真实样本真的被咬住", check_samples_actually_bite),
        ("豁免路径不误杀", check_exemption_still_works),
        ("每道门都在真实样本上开火", check_every_gate_actually_fires),
    ]
    bad = 0
    for label, fn in checks:
        problems = fn()
        if problems:
            bad += 1
            print(f"  [!!] {label}")
            for p in problems:
                print(f"        - {p}")
        else:
            print(f"  [OK] {label}")

    print("\n" + "=" * 72)
    if bad:
        print(f"✗ {bad}/{len(checks)} 项不通过 —— "
              f"**补真实样本, 或把没数据支撑的门删掉**")
        print("  ★ 不要改测试去迁就实现 —— 那就退回到'因为上一道门不够用"
              "就再加一道'了")
        return 1
    print(f"✓ {len(checks)}/{len(checks)} 项通过 —— 门的存在都有真实数据支撑")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())