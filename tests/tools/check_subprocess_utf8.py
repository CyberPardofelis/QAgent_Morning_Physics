"""自检: 我 spawn 的 Python 子进程, 必须能用 UTF-8 跟我说话。

## 事故 (2026-10-10)

`harness_gate` 的 L2 门跑 20 个 verifier 自测, 一次报了 **20 条失败**。
实际上 20 个的 `returncode` **全是 0**, 自测全过。

根因: `subprocess.run(..., encoding="utf-8")` 只规定了**父进程怎么解码**,
没规定**子进程用什么编码写**。Windows 上 stdout 是管道时, Python 用 locale
编码 (cp936/GBK) 写字节, 于是父进程拿到乱码。而 L2 的判据是

    if "正例" not in out or "反例" not in out:   # <-- 静默恒真

**乱码不抛异常, 只让子串判断永远不成立。** 如果我信了那条输出,
下一步就是去"修" 20 个根本没坏的验证器。

## 为什么这不只是 harness_gate 的事

"我 spawn 的 Python 子进程"是个**类**, 不是一处 bug。修 4 个调用点是补漏洞;
真正该固化的是**能力**: 一个统一的 spawn 入口 + 一条会红的回归。

## 怎么用
    python tests/tools/check_subprocess_utf8.py     # 返回码非 0 = 不过
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from agent.tools.research._proc import run_capture, utf8_child_env  # noqa: E402

#: 被审的目录 —— 只看"会 spawn 解释器"的工具/评测代码
SCAN_DIRS = ("agent/tools", "tests/tools")

#: 允许整目录豁免的 marker: 这些地方自己处理了编码
EXEMPT_MARKER = ("PYTHONIOENCODING", "run_capture", "utf8_child_env")

_MARKER = "编码探针-Ω-✓"


def _write_child_printing_marker(tmpdir: str) -> str:
    """一个只做一件事的子进程: 往 stdout 印中文标记。"""
    p = os.path.join(tmpdir, "child_cn.py")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(f"print({_MARKER!r})\n")
    return p


def check_helper_roundtrip() -> list[str]:
    """能力本体: run_capture 拿回来的中文必须可读。"""
    bad = []
    with tempfile.TemporaryDirectory() as td:
        child = _write_child_printing_marker(td)
        p = run_capture([sys.executable, child], timeout=120)
        if _MARKER not in (p.stdout or ""):
            bad.append(f"run_capture 没能读回中文标记 (stdout={p.stdout!r})")
    return bad


def check_helper_has_teeth() -> list[str]:
    """★ 反向测试: 证明上面那条不是永远绿 ★

    一个恒真的检查等于没有检查。这里**故意**用坏写法(只给 encoding 不给 env),
    断言它**必须**读不到中文 —— 如果哪天 Windows 默认编码变了、或者父进程环境里
    已经带了 PYTHONIOENCODING, 这条会红, 提醒我它已经不再能证伪了。
    """
    bad = []
    scrubbed = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    with tempfile.TemporaryDirectory() as td:
        child = _write_child_printing_marker(td)
        try:
            p = subprocess.run([sys.executable, child], capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               env=scrubbed, timeout=120)
        except Exception as e:  # noqa: BLE001
            return [f"反向测试跑不起来: {type(e).__name__}: {e}"]
        if _MARKER in (p.stdout or ""):
            bad.append(
                "反向测试失效: 不设 env 也能读到中文 —— 说明这台机器的默认编码"
                "已经是 UTF-8, 那么 check_helper_roundtrip 就恒真, 没有鉴别力。"
                "(不影响正确性, 但这条自检得改成别的探针)")
    return bad


def check_env_helper() -> list[str]:
    bad = []
    env = utf8_child_env()
    if env.get("PYTHONIOENCODING") != "utf-8":
        bad.append("utf8_child_env 没设 PYTHONIOENCODING=utf-8")
    if env.get("PYTHONUTF8") != "1":
        bad.append("utf8_child_env 没设 PYTHONUTF8=1 (open() 默认编码会漏)")
    if "DEEPSEEK_API_KEY" in env and env["DEEPSEEK_API_KEY"]:
        pass                       # 全量继承是预期行为, 仅确认没有在这里被剥掉
    base = utf8_child_env({"FOO": "bar"})
    if base.get("FOO") != "bar":
        bad.append("utf8_child_env(base) 没保留调用方给的变量")
    return bad


# ============================================================
# 静态审计: 有没有**新写的**调用点忘了走 helper
# ============================================================
_PY_ARGV = re.compile(r"""(sys\.executable|["']python(3(\.\d+)?)?(\.exe)?["'])""")


def _spawns_python(tree: ast.AST, src: str) -> bool:
    """这个文件里有没有 spawn Python 解释器的调用。"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute) or f.attr not in (
                "run", "Popen", "check_output", "call"):
            continue
        seg = ast.get_source_segment(src, node) or ""
        if _PY_ARGV.search(seg):
            return True
    return False


def check_no_bare_python_spawn() -> list[str]:
    """扫 agent/tools 与 tests/tools: spawn 解释器却没管编码的文件。"""
    bad: list[str] = []
    for rel in SCAN_DIRS:
        base = os.path.join(ROOT, rel)
        for dirpath, _dirs, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                # ★ utf-8-sig ★ 不是 utf-8。
                # PowerShell 5.1 的 `Set-Content -Encoding UTF8` 会写 BOM,
                # 用 utf-8 读进来首字符是 \ufeff -> ast.parse 抛 SyntaxError。
                # 我第一版在这里 `continue` 掉了 —— 结果这个审计**恒绿**,
                # 拿一个坏文件试都抓不到。一个"假装在工作"的检查比没有检查更坏。
                try:
                    src = open(path, encoding="utf-8-sig").read()
                except Exception as e:                # noqa: BLE001
                    bad.append(f"{os.path.relpath(path, ROOT)}: 读不出来 {type(e).__name__}")
                    continue
                if not re.search(r"subprocess\.(run|Popen|check_output|call)", src):
                    continue
                if any(m in src for m in EXEMPT_MARKER):
                    continue
                try:
                    tree = ast.parse(src)
                except SyntaxError as e:
                    # ★ 解析不了就报出来 ★ 静默跳过 = 这个文件没被审到
                    bad.append(f"{os.path.relpath(path, ROOT)}: 解析失败, 审计未覆盖 "
                               f"(line {e.lineno}: {e.msg})")
                    continue
                if _spawns_python(tree, src):
                    bad.append(f"{os.path.relpath(path, ROOT)}: spawn 解释器但没管编码 "
                               f"(应走 agent.tools.research._proc.run_capture, "
                               f"或至少 env 里设 PYTHONIOENCODING)")
    return bad


def main() -> int:
    checks = (
        ("能力: run_capture 无损读回中文", check_helper_roundtrip),
        ("★ 反向测试: 坏写法必须读不到", check_helper_has_teeth),
        ("helper: env 变量齐全", check_env_helper),
        ("静态审计: spawn 解释器须管编码", check_no_bare_python_spawn),
    )
    failures: list[str] = []
    print("=" * 70)
    print("子进程 UTF-8 能力自检")
    print("=" * 70)
    for label, fn in checks:
        problems = fn()
        if problems:
            failures.extend(problems)
            print(f"  [!!] {label}")
            for p in problems[:15]:
                print(f"        - {p}")
            if len(problems) > 15:
                print(f"        ... 另有 {len(problems)-15} 处")
        else:
            print(f"  [OK ] {label}")
    print("=" * 70)
    if failures:
        print(f"x 子进程编码自检未过 ({len(failures)} 处)")
        return 1
    print("o 子进程编码自检通过")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())