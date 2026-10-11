"""一次性改写脚本: 给 build_evidence 的产出加 arch_version 字段。

_provenance.py 是 CRLF, 编辑工具的精确匹配会失败, 所以用
io.open(..., newline='') 原样读写。
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "agent", "tools", "research", "_provenance.py")

src = io.open(P, encoding="utf-8", newline="").read()
orig = src

# 1) 加 import —— 紧跟 typing 那一行之后(本模块不 import 其他 research 子模块)
# ★ 本文件是 CRLF, 所以匹配时不能带 \n, 要用行边界正则取整行。
if "RESEARCH_ARCH_VERSION" not in src:
    m = re.search(r"^from typing import Any\r?\n", src, re.M)
    if not m:
        sys.exit("找不到 typing import 锚点, 手工处理")
    nl = "\r\n" if "\r\n" in m.group(0) else "\n"
    src = (src[:m.end()]
           + nl + "from ._flags import RESEARCH_ARCH_VERSION" + nl
           + src[m.end():])

# 2) 在证据字典里加字段(同样避开 \n 匹配)
old_field = '"elapsed_ms": elapsed_ms,'
new_field = (
    '"elapsed_ms": elapsed_ms,'
    '{NL}        # ★ 架构版本: 这条证据是哪一版回路产出的 ★'
    '{NL}        # 不打戳的话, 架构重构后新旧分数会被放在一起比, 而那个对比无效。'
    '{NL}        "arch_version": RESEARCH_ARCH_VERSION,'
)
if '"arch_version": RESEARCH_ARCH_VERSION' not in src:
    if old_field not in src:
        sys.exit("找不到 build_evidence 的字段锚点, 手工处理")
    nl = "\r\n" if "\r\n" in src else "\n"
    src = src.replace(old_field, new_field.replace("{NL}", nl), 1)

if src == orig:
    print("已经打过戳了, 无需改动")
else:
    io.open(P, "w", encoding="utf-8", newline="").write(src)
    print(f"已改写 {os.path.relpath(P, ROOT)}")

# 自检
check = io.open(P, encoding="utf-8", newline="").read()
print("  import 存在:", "RESEARCH_ARCH_VERSION" in check.split("def ")[0])
print("  字段存在  :", '"arch_version": RESEARCH_ARCH_VERSION' in check)
crlf = check.count("\r\n")
lf = check.count("\n")
print(f"  换行保持  : CRLF={crlf} LF={lf} -> {'CRLF 未破坏' if crlf * 2 > lf else '!! 换行被改了'}")