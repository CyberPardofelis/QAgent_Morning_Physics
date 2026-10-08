"""验证体检闸门的条件触发: 干净的命题不该调 LLM, 可疑的才调。

这个测试的核心是**别让优化悄悄把防线拆了** ——
「跳过 LLM 复读」听起来像省钱, 但如果可疑的也没问, 闸门就形同虚设。
所以两边都要钉: 干净的跳过 / 可疑的一个都不许漏。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.tools.research._precheck import (  # noqa: E402
    _looks_clean,
    run_precheck,
)

Q = "对所有正整数 n，是否存在 n 使 n^2 + n + 41 是素数？"

#: (说明, spec, 期望是否干净)
CASES = [
    ("正常带量词的命题",
     {"claim_text": "对所有正整数 n 满足 n^2 + n + 41 是素数",
      "quantifier": "forall"}, True),
    ("主观词 —— 必须去问",
     {"claim_text": "哥德巴赫的证明思路是优美的", "quantifier": "forall"}, False),
    ("没填量词 —— 必须去问",
     {"claim_text": "对所有正整数 n 满足 n^2 + n + 41 是素数", "quantifier": ""}, False),
    ("量词冲突 —— 必须去问",
     {"claim_text": "存在正整数 n 使 n^2 + n + 41 是素数", "quantifier": "exists"}, False),
    ("空命题 —— 必须去问",
     {"claim_text": "", "quantifier": "forall"}, False),
    ("太短 —— 必须去问",
     {"claim_text": "它是素数", "quantifier": "forall"}, False),
]


def main() -> None:
    print("=== _looks_clean 单条判定 ===")
    ok = True
    for label, spec, expect_clean in CASES:
        got = _looks_clean(Q, spec)
        flag = "OK " if got == expect_clean else "!! "
        if got != expect_clean:
            ok = False
        print(f"  {flag}{label:24} clean={got} (期望 {expect_clean})")

    print("\n=== 只问可疑的: LLM 被喂了什么 ===")
    seen: list[str] = []

    def fake_llm(prompt, **kw):
        seen.append(prompt)
        # prompt 里每条命题以 "N. claim_text:" 开头, 数它就知道送进去几条
        n = sum(1 for line in prompt.splitlines()
                if line.strip()[:1].isdigit() and "claim_text:" in line)
        return ('{"results": [' + ",".join(
            f'{{"index": {i}, "same_question": true, "decidable": true, "why": ""}}'
            for i in range(1, n + 1)) + "]}")

    specs = [s for _, s, _ in CASES]
    out = run_precheck(Q, specs, fake_llm, only_suspicious=True)
    print(f"  6 条里送进 LLM 的是: {len(seen)} 次调用")
    if len(seen) != 1:
        print("  !! 期望只调 1 次(把所有可疑的合并成一次问)")
        ok = False
    n_in_prompt = sum(1 for line in (seen[0] if seen else "").splitlines()
                      if line.strip()[:1].isdigit() and "claim_text:" in line)
    print(f"  送进去的命题条数: {n_in_prompt} (期望 5 = 6 条里唯一干净的那条被跳过)")
    if n_in_prompt != 5:
        print("  !! 可疑的没全送进去 —— 防线被削弱了")
        ok = False

    print("\n=== index 映射回原始下标 ===")
    # 可疑的是第 2,3,4,5,6 条(1-based), 干净的是第 1 条
    # LLM 返回的 index 是子集内序号 1..5, 必须映射回 2..6
    expect_keys = {2, 3, 4, 5, 6}
    got_keys = set(out.keys())
    print(f"  返回的原始下标: {sorted(got_keys)}  期望: {sorted(expect_keys)}")
    if got_keys != expect_keys:
        print("  !! 映射错了 —— 会把体检结果写到别的命题上")
        ok = False

    print("\n=== 全都干净时: 一次都不该调 LLM ===")
    seen.clear()
    out2 = run_precheck(Q, [{"claim_text": "对所有偶数 n 满足 4<=n<=1000 存在素数 p,q 使 p+q=n",
                              "quantifier": "forall"}], fake_llm, only_suspicious=True)
    print(f"  LLM 调用次数: {len(seen)} (期望 0)")
    if seen:
        print("  !! 干净的也去问了, 优化没生效")
        ok = False
    print(f"  返回: {out2} (期望空 dict)")

    print("\n=== 老路径 still works (only_suspicious=False) ===")
    seen.clear()
    out3 = run_precheck(Q, specs, fake_llm, only_suspicious=False)
    n_sent = sum(1 for line in (seen[0] if seen else "").splitlines()
                 if line.strip()[:1].isdigit() and "claim_text:" in line)
    print(f"  LLM 调用次数: {len(seen)} (期望 1), 送进去 {n_sent} 条 (期望 6), "
          f"返回条数: {len(out3)} (期望 6)")
    if n_sent != 6 or len(out3) != 6:
        ok = False

    print("\n结论:", "全部正确" if ok else "有问题")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()