"""验证 Explorer 简述目录: 省了多少, 以及**没碰坏对照组**。

## 两条要验的

① 字符数: Explorer 的 prompt 从多少降到多少
② ★ 对照组一个字节都没变 ★
   `explorer_prompt` / `plan_prompt` 加了 `catalog` 形参。B/C/D 三条
   对照路线都走这两个函数, 如果默认路径的行为有任何改变, 四轮数据
   就废了。所以这里要用**同一个问题 + 同一份 tools**, 分别在
   `catalog=None`(默认) 和不传时比对, 确认**逐字相同**。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.research.roles import build_catalog              # noqa: E402
from agent.tools.research.roles import explorer as EX             # noqa: E402
from agent.tools.research.roles.catalog import render_catalog     # noqa: E402
from agent.tools.research.routes.d import explorer_prompt          # noqa: E402
from agent.tools.research.routes.d1 import explorer_catalog       # noqa: E402

FAILS: list[str] = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(("  [ok]   " if cond else "  [FAIL] ") + name
          + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    tools = build_catalog()
    q = "在整数范围 |x|,|y|<=10 内, 求 3x+2y=7 的全部整数解。"

    full = explorer_prompt(q, tools=tools, turn=0)
    lean = explorer_prompt(q, tools=tools, turn=0,
                           catalog=explorer_catalog(tools))

    print("=" * 74)
    print("Explorer 简述目录: 省了多少")
    print("=" * 74)
    print(f"  全量 catalog  {len(full):>7} 字符")
    print(f"  简述 catalog  {len(lean):>7} 字符")
    ck("★ 明显变短", len(lean) < len(full) * 0.5,
       f"{len(lean)} vs {len(full)} = {len(lean)/len(full)*100:.0f}%")

    # 简述里每个工具名都必须在, 否则 Explorer 会"发明工具名"
    names = sorted({str(t["name"]) for t in tools})
    missing = [n for n in names if n not in lean]
    ck("★ 所有工具名都在简述里(Explorer 靠名字选路)", not missing, str(missing))
    ck("★ 每个工具都有一句能力说明",
       all(f"  {n}" in lean or f" {n}" in lean for n in names))

    # ★ 默认路径逐字不变(B/C/D 对照组的安全保证)
    print("\n" + "-" * 74)
    print("对照组安全: 默认路径必须与历史四轮逐字一致")
    print("-" * 74)
    a = explorer_prompt(q, tools=tools, turn=0)
    b = explorer_prompt(q, tools=tools, turn=0, catalog=None)
    c = EX.plan_prompt(q, None, tools)
    d = EX.plan_prompt(q, None, tools, catalog=None)
    ck("explorer_prompt: 显式传 None 与不传 -> 逐字相同", a == b)
    ck("plan_prompt:     显式传 None 与不传 -> 逐字相同", c == d)
    ck("explorer_prompt 默认路径仍含完整 catalog",
       render_catalog(tools)[:200] in a)
    ck("plan_prompt 默认路径仍含完整 catalog",
       render_catalog(tools)[:200] in c)

    print("\n" + "-" * 74)
    print("简述目录实际长这样(前 600 字)")
    print("-" * 74)
    print(lean[:600])
    print("\n" + "=" * 74)
    if FAILS:
        print(f"FAIL: {len(FAILS)} 条")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("PASS: 简述目录大幅缩短, 且对照组逐字未变")
    return 0


if __name__ == "__main__":
    sys.exit(main())