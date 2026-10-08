"""统计: 凭证门在整轮跑测里到底开过几次火, 以及六道兜底门各自开过几次。

为什么要单独数: 对比两轮分数时, 如果机械门没开过火, 那两轮分数的差异
就**在定义上**与机械门无关 —— 那是模型随机性, 不是回归。只看分数会把这
两种情况混为一谈, 而这正是上一轮犯过的错。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import agent.tools.research._provenance as P  # noqa: E402

CASES = "tests/tools/base20_v2_cases.json"
ROOT = os.path.join("data", "research", "working")

#: 逐一探针: 每道门单独跑一遍这份 stdout, 看它会不会返回非 None。
GATES = {
    "凭证门(新)": P._refuted_without_credential,
    "兜底门: 没找到": lambda s: ("x",) if P._NOTHING_FOUND.search(s) else None,
    "兜底门: 全过了": lambda s: ("x",) if P._ALL_PASSED.search(s) else None,
    "兜底门: 验算分歧": lambda s: ("x",) if P._DISAGREE_FAILED.search(s) else None,
    "兜底门: 各部分定不了": lambda s: ("x",) if P._PART_INCONCLUSIVE.search(s) else None,
}


def main() -> None:
    cases = json.load(io.open(CASES, encoding="utf-8"))["cases"]
    fired: dict[str, list[str]] = {k: [] for k in GATES}
    n_declared_refuted = 0
    total = 0

    for c in cases:
        cid = c["id"]
        th = "v2_" + hashlib.sha1(cid.encode()).hexdigest()[:10]
        p = os.path.join(ROOT, th, "_evidence.jsonl")
        if not os.path.isfile(p):
            continue
        for line in io.open(p, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") != "code":
                continue
            total += 1
            out = rec.get("stdout") or ""
            # 只看「代码自称 REFUTED」的证据 —— 其余判定与 REFUTED 侧无关
            declared, _ = P.extract_verdict_marker(out)
            if declared != "refuted":
                continue
            n_declared_refuted += 1
            for name, probe in GATES.items():
                if probe(out) is not None:
                    fired[name].append(f"{cid}[{rec.get('verdict')}]")

    print(f"回放 {total} 条证据, 其中代码自称 REFUTED 的有 {n_declared_refuted} 条\n")
    for name, hits in fired.items():
        mark = "  <-- 本轮开火" if hits else ""
        print(f"  {name:22} 开火 {len(hits):>2} 次{mark}")
        for h in hits:
            print(f"      {h}")
    if n_declared_refuted == 0:
        print("\n★ 本轮没有任何一条证据自称 REFUTED —— "
              "REFUTED 侧的所有机械门本轮都没机会参与。")


if __name__ == "__main__":
    main()