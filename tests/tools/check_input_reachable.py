"""验「输入数据真的进了题面」—— 这一条是 2026-10-09 那 8 道不可解题的修复验收。

★ 为什么必须是自动检查 ★
上一轮的错就是: 我以为"沙箱禁读文件, 你把数据抄进代码里"就够了,
而模型**根本没有数据可抄**。这类"提示给了但东西没给"的错, 人眼很难发现,
因为题面上确实有一句话提到了数据文件。必须用脚本断言"数据真的在 claim 里"。
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import problem_data  # noqa: E402
import problem_spec as PS  # noqa: E402
import score_baseline as SB  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("=" * 76)
    print("输入可达性 —— 题目输入必须真的在题面里")
    print("=" * 76)

    items = {it["id"]: it for suite in ("base", "V2", "R")
             for it in SB.load_suite(suite)}

    print("\n>>> 8 道引用外部数据的题: 数据是否内联进了 claim")
    for cid, fn in problem_data.DATA_FILES.items():
        data = problem_data.read_data(cid)
        check(f"{cid:38} 数据可读", data is not None, fn)
        if data is None:
            continue
        claim = PS.build_claim(items[cid]["prompt"], items[cid].get("expected"),
                               PS.data_hint_for(cid), problem_id=cid)
        # 取数据里最有辨识度的一小段, 断言它逐字出现在 claim 里
        toks = [t for t in re.findall(r"\d+", data) if len(t) >= 2]
        probe = None
        for t in toks[:40]:
            if t in claim:
                probe = t
                break
        check(f"{'':38} 数据片段出现在 claim", probe is not None,
              f"探针={probe}" if probe else "**题面里找不到任何数据!**")

    print("\n>>> 红线: 参考答案绝不能出现在 claim 里")
    leaked = []
    for cid in items:
        claim = PS.build_claim(items[cid]["prompt"], items[cid].get("expected"),
                               PS.data_hint_for(cid), problem_id=cid)
        # ★ 判据要点: 输入与参考**天然重叠** ★
        # res_05_reference.json 里记了序列的末 5 项, 而输入文件里本来就有全部
        # 200 项 —— 那是输入的一部分, 不是泄露。
        # 只有「出现在 reference、但**不**出现在内联输入里」的数字才算泄露。
        inlined = problem_data.read_data(cid) or ""
        for f in os.listdir("data"):
            if not f.endswith("_reference.json"):
                continue
            body = io.open(os.path.join("data", f), encoding="utf-8").read()
            for n in sorted(set(re.findall(r"\d{6,}", body)), key=len,
                           reverse=True)[:8]:
                if n in claim and n not in inlined:
                    leaked.append(f"{cid} 含 {f} 独有指纹 {n[:20]}")
                    break
    check("没有任何参考答案**独有**内容进 claim", not leaked, str(leaked[:3]))

    print("\n>>> 宣告式: claim 里不再出现「去读文件」这种拿不到的东西")
    bad = []
    for cid in problem_data.DATA_FILES:
        claim = PS.build_claim(items[cid]["prompt"], items[cid].get("expected"),
                               PS.data_hint_for(cid), problem_id=cid)
        if "沙箱禁读文件" in claim and "本题的输入数据" not in claim:
            bad.append(cid)
    check("不再出现无法执行的「抄进代码」指示", not bad, str(bad))

    print("\n" + "=" * 76)
    if FAILS:
        print(f"✗ {len(FAILS)} 项不通过: " + "; ".join(FAILS))
        return 1
    print("✓ 全部通过 —— 题目输入都在题面里, 参考答案都不在")
    return 0


if __name__ == "__main__":
    sys.exit(main())