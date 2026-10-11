"""spawn 一个 Python 子进程, 并**无损**读回它的文本输出。

## 为什么这个模块存在 (2026-10-10)

Windows 上, stdout/stderr 是管道时, Python 子进程默认用 **locale 编码 (cp936/GBK)**
写字节; 而父进程如果按 utf-8 解码, 拿到的是乱码。

★ 关键: **乱码不会抛异常, 只会让子串判断静默失效。** ★

真实事故 —— `harness_gate` 的 L2 门要求每个验证器的自测输出里同时出现
「正例」「反例」两个词。20 个验证器**自测全部 returncode=0 通过**,
但因为编码错配, `"正例" in out` 恒为 False, 于是门一次报了 **20 条失败**。

如果我信了这条输出, 下一步就是去"修" 20 个根本没坏的验证器 ——
正是我一直警告的「假阴性和真错误在统计上长得一模一样」。

## 规则

**凡是我 spawn 的 Python 子进程, 一律走这里。**

    from agent.tools.research._proc import run_capture
    p = run_capture([sys.executable, "x.py"], timeout=1200)
    out = p.stdout + p.stderr          # 一定是可读中文, 不是乱码

自己手写 `subprocess.run(..., encoding="utf-8")` **等于没加 env** ——
那正是事故的成因。`check_subprocess_utf8.py` 会静态扫出这类漏网点。

## 例外: 生产沙箱不走这里

`_sandbox.py` 的 `_child_env()` 是**故意**全量过滤密钥的(防沙箱里的代码
调 LLM API 烧钱), 且 runner 用 `os.write(2, ....encode("utf-8"))` 显式编码,
那条路本来就对。它不能复用本模块的全量 env。
"""
from __future__ import annotations

import os
import subprocess
import sys

__all__ = ["utf8_child_env", "run_capture"]


def utf8_child_env(base: dict | None = None) -> dict[str, str]:
    """在给定环境(默认 `os.environ`)上强制子进程用 UTF-8。

    `PYTHONIOENCODING` 管 stdio; `PYTHONUTF8=1` 额外把 `locale.getpreferredencoding`
    也掰成 UTF-8, 覆盖 `open()` 的默认编码 —— 两者都设才不留缝。
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_capture(argv: list[str], *, cwd=None, timeout: float | None = None,
                env: dict | None = None, **kwargs) -> subprocess.CompletedProcess:
    """跑一个子进程并按 UTF-8 读回文本输出。

    与 `subprocess.run` 的区别只有一处, 但那一处就是全部:
    **env 里被强制塞了 `PYTHONIOENCODING=utf-8`**。

    额外把 `python` 换成 `sys.executable` —— `["python", ...]` 在只装了
    py launcher 或多版本共存的机器上可能拉到另一个解释器, 那是另一类
    "以为在跑 A 其实在跑 B" 的静默错误。
    """
    if argv and isinstance(argv[0], str) and argv[0].split("/")[-1].split("\\")[-1] in (
            "python", "python.exe", "python3", "python3.exe"):
        argv = [sys.executable] + list(argv[1:])
    return subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env=utf8_child_env(env), timeout=timeout, **kwargs,
    )