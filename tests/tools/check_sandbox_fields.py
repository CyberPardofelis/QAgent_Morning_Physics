"""静态核对: roles/ 里对 dataclass 字段的访问, 字段真的存在吗。

## 为什么要有这道检查

2026-10-10 smoke 实测, 同一类错犯了**两次**:
  · `worker.py`    写 `sb.print_out` -> AttributeError
  · `verifier.py`  写 `sb.print_out` -> AttributeError

我第一次修完 worker 就以为完事了, 结果下一个角色同样栽在同一个字段名上 ——
**我修的是实例, 不是这一类**。

而这类错的代价特别大: 它在 smoke 的**最后一环**炸, 前面 Explorer/Worker 烧掉的
token 和时间全白费, 而且报出来的是 AttributeError, 很容易被误当成"沙箱坏了"。

`SandboxResult` 是 dataclass, 字段是写死在 `_sandbox.py` 里的常量,
所以这件事完全可以静态查 —— 犯不着等运行时才发现。

## 为什么不放进常规 pytest
它要 import `_sandbox`(会拖进 agent 包)。放这里可以独立跑,
也便于在改完 sandbox 后手动重跑一次。
"""
from __future__ import annotations

import ast
import dataclasses
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from agent.tools.research import _sandbox  # noqa: E402

ROLES = os.path.join(ROOT, "agent", "tools", "research", "roles")

#: roles/ 里跑在沙箱里的局部变量名 -> 返回 SandboxResult 的那个调用
SB_VARS = ("sb", "sandbox", "sbr", "result_sb")


def real_fields() -> set[str]:
    return {f.name for f in dataclasses.fields(_sandbox.SandboxResult)}


def main() -> int:
    fields = real_fields()
    print("=" * 70)
    print(f"SandboxResult 字段: {sorted(fields)}")
    print("=" * 70)

    bad: list[str] = []
    for fn in sorted(f for f in os.listdir(ROLES) if f.endswith(".py")):
        path = os.path.join(ROLES, fn)
        src = open(path, encoding="utf-8").read()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            bad.append(f"{fn}: 语法错误 line {e.lineno}: {e.msg}")
            continue
        # 只看**代码**, 注释里的反面记录不算
        for node in ast.walk(tree):
            seg = (ast.get_source_segment(src, node) or "")
            if seg.lstrip().startswith("#"):
                continue
            if not isinstance(node, ast.Attribute):
                continue
            if not isinstance(node.value, ast.Name) or node.value.id not in SB_VARS:
                continue
            if node.attr not in fields:
                line = node.lineno
                bad.append(f"{fn}:{line}  {node.value.id}.{node.attr} "
                           f"—— SandboxResult 没有这个字段")

    print()
    if bad:
        print(f"x 发现 {len(bad)} 处不存在的字段访问 —— "
              f"会在运行时炸 AttributeError, 而且往往炸在最后一环")
        for b in bad:
            print("  - " + b)
        return 1
    print("o roles/ 里对 SandboxResult 的字段访问全部合法")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())