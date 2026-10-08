"""Lean 4 形式化后端 (P7.3) —— kernel 级机器判定。

## 为什么需要它

用户的要求(2026-10-08): 「结论对不对」和「结论有没有证据」最后都
必须问机器, LLM 的判断也得基于**符号或数值分析、Lean 之类的结果**。

Lean kernel 是这几种后端里最强的一档: 它验证的是**证明项本身**
(proof term), 不是代码逻辑。kernel 只有一万多行, 信任假设极小 ——
这是整个形式化验证领域的根基。

V2 里 hard-015(费马小定理)、hard-016(sqrt2 无理) 明确要求 Lean。

## ★ 最关键的一条: 退出码 0 不等于证明了 ★

Lean 遇到 `sorry` 会报 warning 但**编译通过**, 退出码仍是 0。
实测: `theorem t (n : Nat) : n*2+1 ≥ 1 := by sorry` → exit=0 + warning。

所以本模块的判据是三条一起看:
    exit_code == 0  且  没有 `: error:`  且  **没有 `declaration uses 'sorry'`**

任何一条不满足 → 不是 proved。这是接 Lean 后端最容易踩的坑,
也是为什么这里要单独检 sorry 而不是只看返回值。

## 依赖

elan + Lean 4 工具链, 默认装在 %USERPROFILE%\\.elan\\。
不装也能导入本模块 —— 只是 `_lean_exe()` 返回 None,
调用方应据此回退到别的后端(而不是假装证明了)。

## 现在能做什么 / 不能做什么

能: 不依赖 Mathlib 的核心定理(归纳、代数基础)。
不能: 需要 Mathlib 的定理(费马小定理、sqrt2 无理都在 Mathlib 里)。
     Mathlib 的缓存约 3GB 且走 GitHub release, 国内网络下不了 ——
     要用得先解决镜像, 见 docs/RESEARCH_CLOSED_LOOP_PLAN.md §23。
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile

__all__ = ["lean_available", "lean_version", "check_lean", "LeanResult"]

#: Lean 4.9.0 是 2026-10-08 装的版本。写死是为了让证据可复现 ——
#: 不同 Lean 版本的 tactic 库和 elaboration 行为可能不同。
TOOLCHAIN_VERSION = "v4.9.0"

_ELAN_BIN = os.path.join(os.environ.get("USERPROFILE", ""), ".elan", "bin")
_LEAN_BIN = os.path.join(
    os.environ.get("USERPROFILE", ""), ".elan", "toolchains",
    f"leanprover--lean4---{TOOLCHAIN_VERSION}", "bin")

#: `sorry` 的告警原文。Lean 对未完成证明的声明给出这个 warning。
_SORRY_RE = re.compile(r"declaration uses 'sorry'")
#: 真正的编译错误
_ERROR_RE = re.compile(r": error:")


def lean_available() -> bool:
    """Lean 有没有装好。**没有就是没有**, 别假装能证明。"""
    return _lean_exe() is not None


def _lean_exe() -> str | None:
    """找 lean.exe —— elan/bin 下没有, 在 toolchains/<ver>/bin。"""
    for d in (_ELAN_BIN, _LEAN_BIN):
        p = os.path.join(d, "lean.exe")
        if os.path.exists(p):
            return p
    return None


def _env() -> dict[str, str]:
    e = dict(os.environ)
    e["ELAN_HOME"] = os.path.join(os.environ.get("USERPROFILE", ""), ".elan")
    e["PATH"] = _ELAN_BIN + os.pathsep + _LEAN_BIN + os.pathsep + e.get("PATH", "")
    return e


def lean_version() -> str:
    exe = _lean_exe()
    if exe is None:
        return "absent"
    try:
        p = subprocess.run([exe, "--version"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           env=_env(), timeout=60)
        return (p.stdout or p.stderr or "").strip()
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


class LeanResult(dict):
    """Lean 一次检查的结果。就是个 dict, 带几个便捷属性。"""

    @property
    def proved(self) -> bool:
        return bool(self.get("proved"))

    @property
    def sorry_used(self) -> bool:
        return bool(self.get("sorry_used"))


def check_lean(code: str, *, timeout: int = 300) -> LeanResult:
    """
    编译一段 Lean 代码, 判断它**是否真的证明了所有声明**。

    ★ 判定 = exit_code==0 且无 error 且**无 sorry** ★
    见模块 docstring: `sorry` 会让退出码仍为 0, 只看退出码会误判。

    Args:
        code: Lean 源码。可以是 theorem 声明, 也可以是完整文件。
        timeout: 硬超时秒数。大型 Mathlib 项目编译很慢。

    Returns:
        LeanResult:
            proved       bool  —— 全部声明都证完了
            sorry_used   bool  —— 源码里有 sorry (= 没证完)
            exit_code    int
            errors       list  —— 编译器报的行
            warnings     list
            output       str   —— 完整输出, 供回传给 LLM 修
            available    bool  —— Lean 装了没
    """
    exe = _lean_exe()
    if exe is None:
        return LeanResult(
            proved=False, sorry_used=False, exit_code=-1, errors=[],
            warnings=[], available=False,
            output=("Lean 未安装 —— 不能用它做机器判定。"
                    "装法见 docs/RESEARCH_CLOSED_LOOP_PLAN.md §23"),
        )

    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "Check.lean")
        try:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(code if code.endswith("\n") else code + "\n")
        except Exception as e:
            return LeanResult(proved=False, sorry_used=False, exit_code=-1,
                              errors=[f"写临时文件失败: {e}"], warnings=[],
                              available=True, output="")

        try:
            p = subprocess.run([exe, "Check.lean"], cwd=td, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               env=_env(), timeout=timeout)
            out = (p.stdout or "") + (p.stderr or "")
            code_out = p.returncode
        except subprocess.TimeoutExpired:
            return LeanResult(
                proved=False, sorry_used=False, exit_code=-2, errors=[],
                warnings=[f"编译超时(>{timeout}s)"], available=True,
                output="Lean 编译超时 —— 证明项太大或卡住了",
            )
        except Exception as e:
            return LeanResult(proved=False, sorry_used=False, exit_code=-1,
                              errors=[f"{type(e).__name__}: {e}"], warnings=[],
                              available=True, output="")

    errors = [l for l in out.splitlines() if _ERROR_RE.search(l)]
    warnings = [l for l in out.splitlines() if _SORRY_RE.search(l)]
    sorry_used = bool(warnings)

    # ★ 三条一起看, 缺一不可 ★
    proved = (code_out == 0) and not errors and not sorry_used

    return LeanResult(
        proved=proved,
        sorry_used=sorry_used,
        exit_code=code_out,
        errors=errors[:20],
        warnings=warnings[:20],
        available=True,
        output=out.strip()[-4000:],
    )