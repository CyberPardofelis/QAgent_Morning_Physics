"""★ verifier 与题库契约的**字段名一致性审计** ★

## 为什么必须有这道检查

真实翻车(2026-10-08 R 套 res-012):
    agent 交  ANSWER: {"polynomial": "x**8 - 40*x**6 + ...", "degree": 8, ...}
    verifier 报 coeffs_parsed: 没能拿到有理系数数组(未给 coeffs 数组)
而我独立用 sympy 验过: agent 给的多项式**一字不差就是答案**, 代回得 0。

也就是说 verifier 判错了一道**完全正确的题**。根因不是模型, 是
verifier 认的字段名和题库 `expected` 里声明的字段名对不上 ——
于是它退化成"考格式合规", 而不是"考数学"。

## 这道检查怎么抓

对每道题: 按题库 `expected.value` 的**字段名**造一份结构正确的制品,
喂给 verifier, 看它的解析环节过不过。

    全过 = verifier 认题库声明的字段名
    有不过 = 该 verifier 与契约不一致, 它给出的 wrong 是假阴性

★ 它抓不到的 ★
如果某个 verifier 需要某个字段**有意义的值**(比如 6x6 魔方要能解、
A5 要真同构), 占位值必然过不了。这时它报的是"假阳性不一致",
需要人看 details 区分。所以本脚本输出后, 我会逐条核对字段名而不是只看成败。
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import problem_spec as PS  # noqa: E402
import score_baseline as SB  # noqa: E402
import verifier_map as VM  # noqa: E402

#: 按字段名猜一个"形状正确"的值, 让 verifier 的解析环节能过。
#: 目的是**只检验字段名认不认**, 不是检验数学。
PLACEHOLDER = {
    "polynomial": "x**2 - 1",
    "coefficients": [1, 0, -1],
    "coeffs": [1, 0, -1],
    "degree": 2,
    "order": 2,
    "irreducible": True,
    "unique": True,
    "optimal": True,
    "verified": True,
    "verified_200_terms": True,
    "verified_100_terms": True,
    "verified_by_second_impl": True,
    "distinct_distances": True,
    "is_nonabelian": True,
    "is_A5": True,
    "verdict": "false",
    "relation": [1, 0, 0, 0],
    "quadruple": [1, 0, 0, 0],
    "period": 7,
    "sg_values": [0, 1],
    "marks": [0, 1],
    "matrix": [[1]],
    "coloring": [[0]],
    "solution": [[1]],
    "solutions": [[0, 0]],
    "tour": [0, 1],
    "group": "x",
    "generators": [1],
    "counterexample": "x",
    "certificate": "ok",
    "proof_method": "x",
    "refined_conjecture": "x",
    "evidence": "x",
    "formula": "F=F+F",
    "oeis_id": "A000045",
    "init_vals": [1],
    "recurrence": [1],
    "length": 1,
    "kind": "coloring",
    "value": 1,
    "n": 1,
    "system": [],
    "residual": 0.0,
    "sg_1000": [],
}


def placeholder_for(key: str) -> object:
    return PLACEHOLDER.get(key, 1)


#: 差分探针要试的「题面没让交」的候选键。
#: ★ 为什么要带证书类 ★ SELF_EVAL_FIELDS 里没有 `certificate` 系列, 而
#: res-011 那个 verifier 硬性要求 `uniqueness_certificate`(读不到时退回读
#: `certificate`)。只按 SELF_EVAL_FIELDS 试, 这条矛盾就漏过去了。
_EXTRA_CERTS = ("certificate", "uniqueness_certificate", "proof_certificate",
                "unsat_certificate")
_CANDIDATE_EXTRA = ("unique", "claims_unique", "optimal", "is_optimal",
                    "verified", "verified_200_terms", "verified_100_terms",
                    "verified_terms", "verified_by_second_impl", "checked",
                    "proof", "proof_method", "evidence") + _EXTRA_CERTS


def positive_for(key: str) -> object:
    """给候选键一个**最强的正值** —— 只探"这个键被不被读", 不探值对不对。

    ★ 这个探针的能力边界(必须写清楚, 否则会拿沉默当结论) ★
    它只能抓**猜得到正值**的闸门。猜不到就不报 —— 那不等于"没有闸门",
    只等于"本探针测不到"。res-011 的 `claims_unique` 是靠 `unique=True`
    撞出来的; `uniqueness_certificate` 要靠 `"unsat": true` 这种断言式正值。
    所以证书类键一律用「断言目标结果」当正值, 这是证书的通用最强写法。
    """
    if key in _EXTRA_CERTS:
        return {"unsat": True, "result": "unsat", "z3_output": "unsat"}
    return PLACEHOLDER.get(key, True)


def _failing(det: dict) -> set:
    """details 里判 False 的项。注意 details 的值可能是 dict/list。"""
    out = set()
    for k, v in (det or {}).items():
        if v is False:
            out.add(k)
    return out


def audit_self_cert_gating(items) -> list[str]:
    """★ 差分探针: 判据会不会因为「题面没让交的字段」而改变? ★

    ## 为什么需要它(第一版审计抓不到 res-011 的矛盾)

    第一版只把「字段名没认上」当失败, 剩下的失败一律归为「占位值导致, 属正常」。
    而 res-011 的两条硬性要求 —— `claims_unique` 和 `uniqueness_certificate` ——
    **恰好都落在这个"正常"桶里**, 于是被挥手放过。

    ## 它怎么抓

    同��份契约字段造两份制品:
      A = 题面实际让 agent 交的字段(`expected.value` 去掉自评字段)
      B = A + 一个候选多余字段
    若某条判据在 A 下不过、在 B 下过了, 说明**这条判据被一个题面禁交的字段把持了**。

    这是**差分**出来的, 不是猜的: 占位值本身再假, 也不会因为多了一个不相干的
    字段就突然变对。

    ## 它会把哪一类正常东西判成异常(必须先问)

    「B 比 A 多一个字段」如果让判据变过, 只可能是**判据读了那个字段**。
    所以命中的几乎必然是真矛盾; 仍有可能把 `certificate` 这种**通用名**
    在个别题里当合法字段用, 所以报告里会带上题号与具体键名, 供人复核。
    """
    from agent.tools.research.research_verify import SELF_EVAL_FIELDS

    bad: list[str] = []
    for it in items:
        exp = it.get("expected")
        if not isinstance(exp, dict) or not isinstance(exp.get("value"), dict):
            continue                                   # Lean/证明题没有制品契约
        cid = it["id"]
        # ★ 题面**实际**让 agent 交的字段 = 题库声明的字段去掉自评字段 ★
        # 这正是 problem_spec 干的事; 不这么算, 就把 `unique` 当成题面要求的了,
        # 矛盾就永远看不见。
        declared = [k for k in exp["value"] if k not in SELF_EVAL_FIELDS]
        if not declared:
            continue
        tag = _tag_for(declared)
        try:
            mod = VM.load_verifier(cid)
        except Exception:
            continue
        art_a = {k: placeholder_for(k) for k in declared}
        base = _failing((mod.verify(_stdout(tag, art_a), "", it["prompt"])
                         .get("details") or {}))
        if not base:
            continue                                   # 占位值居然全过, 无基线可比
        for extra in _CANDIDATE_EXTRA:
            if extra in art_a:
                continue
            art_b = dict(art_a)
            art_b[extra] = positive_for(extra)
            flipped = base - _failing((mod.verify(_stdout(tag, art_b), "",
                                                  it["prompt"]).get("details") or {}))
            for name in sorted(flipped):
                msg = (f"{cid}: 判据 `{name}` 只要多交 `{extra}` 就变过, "
                       f"而题面只让交 {declared} —— verifier 在奖励题面禁交的字段")
                if msg not in bad:
                    bad.append(msg)
    return bad


def _stdout(tag: str, art: dict) -> str:
    return (f"{tag}: " + json.dumps(art, ensure_ascii=False)
            + "\nVERDICT: PROVED\n")


def _tag_for(fields: list) -> str:
    return "COUNTEREXAMPLE" if "counterexample" in fields else (
        "ANSWER" if any(k in fields for k in ("degree", "verdict", "answer",
                                              "relation", "sg_values"))
        else "CONSTRUCT")


def main() -> int:
    items = SB.load_suite("R")
    bad: list[str] = []
    unknown_fields: list[str] = []
    print("=" * 78)
    print("verifier × 题库契约 字段名一致性审计")
    print("=" * 78)
    for it in items:
        exp = it.get("expected")
        if not isinstance(exp, dict) or not isinstance(exp.get("value"), dict):
            continue
        cid = it["id"]
        fields = list(exp["value"].keys())
        for k in fields:
            if k not in PLACEHOLDER:
                unknown_fields.append(f"{cid}.{k}")

        art = {k: placeholder_for(k) for k in fields}
        tag = _tag_for(fields)
        stdout = _stdout(tag, art)
        try:
            mod = VM.load_verifier(cid)
            r = mod.verify(stdout, "", it["prompt"])
        except Exception as e:
            bad.append(f"{cid}: verifier 抛异常 {type(e).__name__}: {e}")
            print(f"  [!!] {cid:38} 异常 {type(e).__name__}")
            continue
        det = r.get("details") or {}
        # 关注「没拿到/没解析出制品」这一类失败 —— 那说明字段名没认
        miss = [k for k, v in det.items()
                if v is False and any(t in k for t in
                                      ("parsed", "present", "artifact",
                                       "extract", "found"))]
        # 值不合法导致的失败是正常的(占位值本来就是假的)
        val_bad = [k for k, v in det.items() if v is False and k not in miss]
        if miss:
            bad.append(f"{cid}: 字段名没认上 {miss}  (制品字段={fields})")
            print(f"  [!!] {cid:38} 字段名没认: {miss}")
            print(f"       制品字段: {fields}")
        else:
            print(f"  [OK ] {cid:38} 字段名认得上"
                  + (f"  (占位值导致 {len(val_bad)} 项校验失败, 属正常)"
                     if val_bad else ""))

    print("\n" + "=" * 78)
    print("差分探针: 判据会不会被『题面禁交的字段』把持")
    print("=" * 78)
    gate_bad = audit_self_cert_gating(items)
    if gate_bad:
        for b in gate_bad:
            print("  [!!] " + b)
        print(f"\n✗ {len(gate_bad)} 处: 题面禁交的字段能改变判据 —— "
              f"完全遵守题面的 agent 必然判错(**假阴性**)")
    else:
        print("  [OK ] 没有判据依赖题面禁交的字段")

    print("\n" + "=" * 78)
    if bad:
        print(f"✗ {len(bad)} 道题的 verifier 与题库契约字段名对不上 —— "
              f"它给出的 wrong 是**假阴性**")
        for b in bad:
            print("  -", b)
    else:
        print("✓ 全部 verifier 都认题库声明的字段名")
    if unknown_fields:
        print(f"\n(占位表里还没有的字段 {len(unknown_fields)} 个, "
              f"这些题的结果要人工核对: {', '.join(unknown_fields[:8])})")
    print("=" * 78)
    return 1 if (bad or gate_bad) else 0


if __name__ == "__main__":
    sys.exit(main())