"""看每个工具的 description 长什么样 —— 用来定"一行简述"怎么截。

★ 为什么要先看再写 ★
"截第一句"听起来合理, 但工具描述的写法不统一: 有的第一句就是核心
能力, 有的第一句是"这个工具用来...", 有的第一句半句没说完。
不先看样本就定规则, 截出来的东西可能是半句话。
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.research.roles import build_catalog  # noqa: E402


def first_sentence(d: str, limit: int = 56) -> str:
    """取第一句, 再截到 limit。**按句号/换行/分号**切, 不是按字数硬切。"""
    d = (d or "").strip()
    if not d:
        return ""
    for sep in ("。", "\n", "; "):
        if sep in d:
            d = d.split(sep)[0]
            break
    d = d.strip()
    return d[:limit] + ("…" if len(d) > limit else "")


def main() -> int:
    ts = build_catalog()
    seen: set[str] = set()
    rows = []
    for t in ts:
        n = t["name"]
        if n in seen:
            continue
        seen.add(n)
        d = str(t.get("description") or "")
        rows.append((n, len(d), first_sentence(d)))
    print(f"{'工具':<20}{'desc 全长':>10}  一行简述(拟)")
    print("-" * 96)
    tot_full = 0
    tot_short = 0
    for n, ln, sh in rows:
        print(f"{n:<20}{ln:>10}  {sh}")
        tot_full += ln
        tot_short += len(sh)
    print("-" * 96)
    print(f"合计: 完整 description {tot_full} 字符 -> 一行简述 {tot_short} 字符"
          f"  (省 {(1-tot_short/max(1,tot_full))*100:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())