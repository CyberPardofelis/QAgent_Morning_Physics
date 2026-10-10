"""回归: 桥的**冷启动入口**不许死锁。

## 被钉住的这一次 (2026-10-10, C 路线第一次冒烟: 卡了十分钟, 一次 LLM 都没调到)

    env_report()  ->  with _lock: ... -> _load() -> with _lock:   ✗ 永久阻塞

`env_report()` 要先抢锁才能防重复探测, 而它内部又要调 `_load()`, 而 `_load()`
自己也抢同一把锁。`threading.Lock` **不可重入** —— 第二次 acquire 直接卡死。
修法: 换成 `threading.RLock()`。

## ★ 为什么之前那一堆校验全都放过了它 ★

它们都**先调了 `tools()`**: 那时 `_mod` 已就绪, `_load()` 在拿锁之前就
`return _mod` 了, 于是死锁路径一次都没走到。
**调用顺序把 bug 盖住了** —— 而这正是最危险的一种"已验证":
校验通过了, 但通过的是另一条路径。

## 所以本文件的判据是"入口顺序", 不是"功能对不对"

每种入口都从**全新进程**、**第一个调用**的位置进来。只有这样才能保证
真正在测"冷启动路径", 而不是复用上一个入口热好的模块状态。

★ 超时必须有 ★ 死锁的表现是**挂住不返回**, 不设超时的测试会跟着一起挂,
那比没有测试更糟 —— 它会卡住整条 CI。
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: 每个入口都**只**调那一个 —— 顺序就是被测对象
ENTRIES = [
    ("render_catalog 作为第一个调用", "R.render_catalog()"),
    ("render_env 作为第一个调用", "R.render_env()"),
    ("env_report 作为第一个调用", "R.env_report()"),
    ("tools 作为第一个调用", "R.tools()"),
    ("libraries 作为第一个调用", "R.libraries()"),
    ("tool_libraries 作为第一个调用", "R.tool_libraries('z3_smt2')"),
    ("call 作为第一个调用", "R.call('z3_smt2', {'source': '(check-sat)'})"),
]

#: 冷启动后**紧接着**再调另一个入口 —— 覆盖"第一次拿锁 + 第二次拿锁"的嵌套
CHAIN = [
    ("render_catalog 之后再 render_env",
     "R.render_catalog(); R.render_env()"),
    ("render_env 之后再 render_catalog",
     "R.render_env(); R.render_catalog()"),
    ("env_report 之后再 tools 再 call",
     "R.env_report(); R.tools(); R.call('sympy_simplify', {'expr': 'x+x'})"),
]

FAILS: list[str] = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def run_fresh(expr: str, timeout: int = 90) -> tuple[bool, str]:
    """在**全新进程**里把这个表达式跑一遍。超时 = 死锁。"""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "from agent.tools.research.roles import _registry as R\n"
        f"{expr}\n"
        "print('OK')\n"
    )
    try:
        p = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"★ 超过 {timeout}s 没返回 —— 死锁"
    if p.returncode != 0:
        return False, f"退出码 {p.returncode}: {(p.stderr or '')[-300:]}"
    return ("OK" in (p.stdout or "")), (p.stdout or "").strip()[-200:]


def main() -> int:
    print("=" * 76)
    print("桥的冷启动入口 —— 每个都从全新进程里当**第一个调用**")
    print("=" * 76)
    for label, expr in ENTRIES:
        ok, detail = run_fresh(expr)
        ck(label, ok, detail)

    print()
    print("=" * 76)
    print("连续调用 —— 覆盖嵌套抢锁")
    print("=" * 76)
    for label, expr in CHAIN:
        ok, detail = run_fresh(expr)
        ck(label, ok, detail)

    print(f"\n失败 {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
