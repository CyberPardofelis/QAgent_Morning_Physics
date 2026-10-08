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
能: **依赖 Mathlib 的定理** —— 2026-10-08 已装好 Lean 4.9.0 对应的
     Mathlib **v4.9.0**(tag 与 toolchain 一一对应, 见
     `tests/tools/find_mathlib_for_lean49.py`)。费马小定理、sqrt2 无理
     都已实测编译通过, 公理审计无 `sorryAx`。
     Mathlib 本体在 `D:\\pythonProject\\Others\\mathlib4-v490`(大件不进项目目录)。

### Mathlib 装机踩的三个坑(都跟"网络"无关, 别再往网络方向查)

1. **版本必须配对**: Mathlib 最新 v4.34.1 要求 Lean 4.19+, 与装的 4.9.0 不兼容。
   用 tag `v4.9.0`, 其 `lean-toolchain` 正是 `leanprover/lean4:v4.9.0`。
2. **`lake update` 会毁掉配对**: 它把依赖(batteries/Qq/aesop...)全解析到
   **默认分支最新**, 那些版本是 lake5+ 的 lakefile 格式(要求 `source` 键),
   与 lake4.x 冲突 → `missing required key: source`。
   正解: 恢复仓库自带的 `lake-manifest.json`, 按里面钉死的 rev 逐个 checkout,
   然后**只**跑 `lake exe cache get`, 不跑 `lake update`。
3. **`lake` 下 leantar 走 curl 直连 GitHub Releases, git 代理对它无效**:
   `GIT_CONFIG_*` 的 insteadOf 只作用于 git, lake 调的是 curl。
   解法: 用代理把 `leantar-<ver>.exe.zip` 下到 `%USERPROFILE%\\.cache\\mathlib\\`,
   再解压出 `leantar-<ver>.exe`(lake 先查 exe 存在与否, 有就跳过下载)。
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile

__all__ = ["lean_available", "lean_version", "check_lean", "mathlib_available",
           "LeanResult"]

#: Lean 4.9.0 是 2026-10-08 装的版本。写死是为了让证据可复现 ——
#: 不同 Lean 版本的 tactic 库和 elaboration 行为可能不同。
TOOLCHAIN_VERSION = "v4.9.0"

#: Mathlib v4.9.0 本体。放在 Others 下是因为它连缓存有 3GB+,
#: 项目目录放不下也不该放。大件路径写死 = 证据可复现。
MATHLIB_DIR = r"D:\pythonProject\Others\mathlib4-v490"
MATHLIB_VERSION = "v4.9.0"

_ELAN_BIN = os.path.join(os.environ.get("USERPROFILE", ""), ".elan", "bin")
_LEAN_BIN = os.path.join(
    os.environ.get("USERPROFILE", ""), ".elan", "toolchains",
    f"leanprover--lean4---{TOOLCHAIN_VERSION}", "bin")
_LAKE_EXE = os.path.join(_LEAN_BIN, "lake.exe")

#: `sorry` 的告警原文。Lean 对未完成证明的声明给出这个 warning。
_SORRY_RE = re.compile(r"declaration uses 'sorry'")
#: 真正的编译错误
_ERROR_RE = re.compile(r": error:")
#: ★ 更强的一道: 依赖 sorryAx 的公理 = 证明里藏着洞 ★
#: `#print axioms` 会列出用到的公理, 含 sorryAx 就说明这个"证明"是靠 sorry 撑的。
_SORRY_AX_RE = re.compile(r"sorryAx")

#: ★★ 更隐蔽的一道: 自己声明 axiom ★★
#:
#: 实测(2026-10-08): 下面这段 Lean **编译通过、退出码 0、没有任何 warning**,
#: `#print axioms` 也只说 "does not depend on any axioms" —— 因为它确实没依赖
#: 任何公理, 它**自己就是**公理:
#:
#:     axiom my_sqrt2 : Irrational (Real.sqrt 2)
#:     example : Irrational (Real.sqrt 2) := my_sqrt2
#:
#: 只查 sorryAx 完全看不出问题: sorryAx 查的是"借来的洞", 这里是自己凿的洞。
#: 所以必须回到**源码**, 看有没有 axiom 声明 —— 提交证明却附带 axiom,
#: 等于把待证命题直接当前提, 和循环论证是同一类错误。
_AXIOM_DECL_RE = re.compile(
    r"^[ \t]*(?:(?:private|protected|unsafe|noncomputable)[ \t]+)*axiom[ \t]+",
    re.MULTILINE,
)


def _strip_lean_comments(code: str) -> str:
    """
    去掉 Lean 注释, 免得注释里写了个 axiom 就被当成声明。

    Lean 的注释: `--` 行注释、`/- ... -/` 块注释(可嵌套)。
    先扫掉块注释, 再按行去掉行注释 —— 顺序反了会被块注释里的 `--` 打断。
    """
    out = []
    i, n = 0, len(code or "")
    depth = 0
    while i < n:
        two = code[i:i + 2]
        if depth == 0 and two == "/-":
            depth, i = 1, i + 2
            continue
        if depth > 0:
            if two == "/-":
                depth += 1
                i += 2
                continue
            if two == "-/":
                depth -= 1
                i += 2
                continue
            i += 1
            continue
        if two == "--":
            j = code.find("\n", i)
            i = n if j < 0 else j
            continue
        out.append(code[i])
        i += 1
    return "".join(out)


def lean_available() -> bool:
    """Lean 有没有装好。**没有就是没有**, 别假装能证明。"""
    return _lean_exe() is not None


def mathlib_available() -> bool:
    """Mathlib 在不在(决定能不能 import Mathlib)。"""
    return (
        os.path.isfile(os.path.join(_LAKE_EXE))
        and os.path.isfile(os.path.join(MATHLIB_DIR, "lakefile.lean"))
    )


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


#: 源码里到底有没有 import Mathlib。
#: 只有真需要时才走 `lake env` —— 那条路慢得多(要加载 Mathlib 的环境)。
_IMPORT_MATHLIB_RE = re.compile(
    r"^\s*import\s+Mathlib\b", re.MULTILINE)


def _imports_mathlib(code: str) -> bool:
    return _IMPORT_MATHLIB_RE.search(code or "") is not None


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
            warnings=[], available=False, mathlib=False,
            output=("Lean 未安装 —— 不能用它做机器判定。"
                    "装法见 docs/RESEARCH_CLOSED_LOOP_PLAN.md §23"),
        )

    # 源码要 import Mathlib 时, 必须经由 `lake env lean` 跑 ——
    # 直接调 lean.exe 找不到 Mathlib 的 oleans, 会报 unknown package。
    use_mathlib = mathlib_available() and _imports_mathlib(code)
    if use_mathlib:
        cmd = [_LAKE_EXE, "env", "lean", None]
        cwd = MATHLIB_DIR
    else:
        cmd = [exe, None]
        cwd = None

    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "Check.lean")
        try:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(code if code.endswith("\n") else code + "\n")
        except Exception as e:
            return LeanResult(proved=False, sorry_used=False, exit_code=-1,
                              errors=[f"写临时文件失败: {e}"], warnings=[],
                              available=True, mathlib=use_mathlib, output="")

        argv = [c if c is not None else f for c in cmd]
        try:
            p = subprocess.run(argv, cwd=cwd, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               env=_env(), timeout=timeout)
            out = (p.stdout or "") + (p.stderr or "")
            code_out = p.returncode
        except subprocess.TimeoutExpired:
            return LeanResult(
                proved=False, sorry_used=False, exit_code=-2, errors=[],
                warnings=[f"编译超时(>{timeout}s)"], available=True,
                mathlib=use_mathlib,
                output="Lean 编译超时 —— 证明项太大或卡住了",
            )
        except Exception as e:
            return LeanResult(proved=False, sorry_used=False, exit_code=-1,
                              errors=[f"{type(e).__name__}: {e}"], warnings=[],
                              available=True, mathlib=use_mathlib, output="")

    errors = [l for l in out.splitlines() if _ERROR_RE.search(l)]
    warnings = [l for l in out.splitlines() if _SORRY_RE.search(l)]
    # ★ 公理审计: 用了 sorryAx 也算没证完 ★
    # 这不是多余的 —— `#print axioms` 是 LLM 常用的自证手段, 而它能暴露
    # "先 axiom 再 theorem" 这种编译器完全不会报错的后门。
    axioms = [l for l in out.splitlines() if _SORRY_AX_RE.search(l)]

    # ★ 源码级: 自己声明了 axiom = 把待证命题当前提, 同样不算证明 ★
    # (必须在算 sorry_used **之前**发现它 —— 上一版把警告追加到 warnings
    #  之后才重算 proved, 结果 self_axioms 永远不影响判定, 后门照放。
    #  这跟"阻断时返回错误字符串"是同一类坑: 看着加了防线, 其实没接线。)
    self_axioms = _AXIOM_DECL_RE.findall(_strip_lean_comments(code or ""))
    if self_axioms:
        warnings = warnings + [
            f"源码里声明了 axiom({len(self_axioms)} 处) —— "
            f"提交证明时附带 axiom, 等于把待证命题当前提, 不算证明"
        ]

    sorry_used = bool(warnings) or bool(axioms)

    # ★ 四条一起看, 缺一不可 ★
    proved = (code_out == 0) and not errors and not sorry_used

    return LeanResult(
        proved=proved,
        sorry_used=sorry_used,
        exit_code=code_out,
        errors=errors[:20],
        warnings=warnings[:20],
        axioms=axioms[:20],
        available=True,
        mathlib=use_mathlib,
        output=out.strip()[-4000:],
    )