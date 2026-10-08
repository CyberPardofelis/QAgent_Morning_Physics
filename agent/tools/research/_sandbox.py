"""
agent.tools.research._sandbox — S0 受限 Python 执行器 (P7 阶段 0)

一句话:把一段 Python 丢进**独立子进程**跑,带超时、AST 静态门、输出截断,
把 stdout / 异常原文 / 耗时拿回来当证据。

================================ 威胁模型 (务必读) ================================
⚠️ **S0 不是安全边界, 它是防手滑的护栏。**

挡得住的:
  - LLM 生成 `import os` / `open(...)` 这类越界代码 —— AST 门直接拒, 不执行
  - 代码写错 / 依赖缺失 —— 异常原文原样回传, 喂回 LLM 修
  - 死循环 / 跑飞 —— 硬超时 + 杀整棵进程树

挡不住的 (S0 的已知缺口, 别当成不存在):
  - 任何**有意**的沙箱逃逸: sympy / numpy 自身能起子进程、能调 ctypes
    (`numpy.ctypeslib`、`sympy.external.gmpy` 等), AST 门看不见运行时行为
  - 内存炸弹: 分配 10GB 内存的代码, S0 不限内存 (Windows 无 rlimit)
  - 网络: S0 没有禁网。要禁网必须走 S1 (Docker `--network=none`, 见文档 §5.4)

因此 S0 的使用前提写死在 `_ALLOWED_IMPORT_ROOTS` 与下面的
`assert_trusted_source` 注释里: **只跑本机自己(LLM 生成且过了 AST 门)的代码,
不接受外部来源的代码。** 要跑别人给的代码, 先上 Docker。

真正的隔离 = S1 = Docker/E2B。P0 阶段 S0 是默认, S1 是 P3 的升级项。

============================================================================

实现要点 (踩过的坑都在注释里):
1. 代码**写文件**再让子进程跑, 不用 `python -c`:
   - Windows 命令行有 ~32KB 上限, SymPy 验证代码容易超
   - 有文件名 → 异常 traceback 的路径可读, `-c` 只显示 `<string>`
2. 结果走**哨兵行 + os.write(2)** 回传, 不用 stdout:
   - 用户代码的 print 会被 redirect 吞进 buffer, 不会污染协议
   - os.write 是最低层, 不受用户代码里 sys.stderr = ... 的篡改影响
3. 超时后杀**整棵进程树** (`taskkill /T /F`), 只杀父进程不管用:
   - 用户代码 spawn 出来的孙进程会继续持有 stdout 管道
   - 只 kill 父进程的话 communicate() 会一直阻塞等管道关闭 → 卡死
4. Windows 没有 rlimit, 所以 S0 的"资源限制"只有超时一项 (见 §5.2)

历史:
  - 2026-10-08 P7.0 首版
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================
# 常量
# ============================================================

#: 子进程回传结果的哨兵前缀 (走 fd 2, 见模块 docstring 第 2 点)
_SENTINEL = "__QAGENT_SANDBOX_RESULT__ "

#: stdout / stderr 各自的截断上限 —— 超了标记 truncated, 不让 LLM 被日志淹掉
MAX_OUTPUT_CHARS = 8000

#: 默认硬超时 (秒)。SymPy 积分可能慢, 但 20s 是"卡住了"的合理判据
DEFAULT_TIMEOUT = 20

#: 超时后留给 communicate() 的收尾时间; 再不返回就强行放弃管道
_DRAIN_TIMEOUT = 3

#: 允许 import 的模块根名。**白名单, 不是黑名单** —— 新东西要显式加进来。
#: 故意不含: os sys subprocess socket shutil ctypes importlib requests urllib
#:             multiprocessing threading tempfile pathlib pickle
_ALLOWED_IMPORT_ROOTS = frozenset({
    # 标准库: 纯计算/容器
    "math", "cmath", "itertools", "functools", "fractions", "decimal", "statistics",
    "random", "json", "re", "string", "operator", "copy", "heapq", "bisect",
    "collections", "typing", "enum", "dataclasses", "abc", "numbers", "array",
    # 标准库: 时间/IO (受限用途, 靠超时兜住)
    "time", "datetime", "io", "csv", "textwrap",
    # 科学计算 (已在 requirements.txt 锁定版本)
    "sympy", "numpy", "scipy", "networkx",
    # P7.1 加: z3 的 Python API —— 让"用代码调 Z3"这条路和 research_z3 工具
    # 走同一套沙箱/超时/证据机制(而不是两套)。
    # ⚠️ 这让 S0 沙箱多了一块原生扩展(z3 的 .so/.dll), 逃逸面比纯 Python 大;
    #    接受它是因为 research_verify 的代码路径需要它。S1(Docker)才谈真正的隔离。
    "z3",
})

#: 禁止出现的**名字** —— 这些能力跟沙箱逃逸直接相关, 无论从哪来
_BANNED_NAMES = frozenset({
    "__import__", "eval", "exec", "compile", "open", "input",
    "globals", "locals", "vars", "breakpoint", "exit", "quit", "help",
    "memoryview",  # 可拿 buffer 原始内存
    "__builtins__",  # 拿到就能翻出上面那些
})

#: 禁止的 ast 节点类型 —— 与 names 互补(有些危险用法不需要可疑名字)
_BANNED_NODES: tuple[type[ast.AST], ...] = (
    ast.Import,       # 走 _check_import 逐个根名校验
    ast.ImportFrom,   # 同上
)


class SandboxRejected(Exception):
    """代码没过 AST 静态门 —— 抛这个, 不执行"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class SandboxResult:
    """一次执行的完整结果。工具层负责转成 §6.4 的 5 字段 dict"""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    elapsed_ms: int = 0
    timed_out: bool = False
    rejected: bool = False
    reject_reason: str = ""
    truncated: bool = False
    returncode: int | None = None
    #: 异常类型名, 如 "TypeError"。给 LLM 看比原始 traceback 更省 token
    error_type: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """工具返回给 LLM 的形态(尚未挂 evidence_id, 那是 _provenance 的活)"""
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed_ms": self.elapsed_ms,
            "timed_out": self.timed_out,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
            "truncated": self.truncated,
            "returncode": self.returncode,
            "error_type": self.error_type,
        }


# ============================================================
# AST 静态门
# ============================================================

def _check_import(node: ast.Import | ast.ImportFrom) -> None:
    """校验 import 的每个根模块在白名单里"""
    if isinstance(node, ast.Import):
        roots = [(a.name or "").split(".")[0] for a in node.names]
    else:
        # `from . import x` → node.module 为 None, 是相对导入, 沙箱里无意义
        roots = [(node.module or "").split(".")[0] or "<relative>"]

    for root in roots:
        if root not in _ALLOWED_IMPORT_ROOTS:
            raise SandboxRejected(
                f"line {node.lineno}: 模块 `{root}` 不在沙箱白名单内。"
                f"可用模块: {', '.join(sorted(_ALLOWED_IMPORT_ROOTS))}"
            )


def check_code(code: str) -> None:
    """
    静态检查代码 —— 过了才执行。不通过抛 SandboxRejected。

    这是 S0 唯一的"门"。它检查的是**源码文本**, 所以:
      - 动态构造的 import (getattr 拼出来) / 运行时逃逸, 它看不见 (§ 模块 docstring)
      - 它的目标只是"让 LLM 生成的代码别顺手 import os / open 读你的文件"
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxRejected(
            f"语法错误 line {e.lineno}: {e.msg} — 代码根本没被解析, 修正后重试"
        ) from e

    for node in ast.walk(tree):
        # import 类节点 → 走白名单校验
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node)
            continue
        # 其他被禁节点类型(留了扩展位, 当前无额外类型)
        if isinstance(node, _BANNED_NODES):
            continue
        # 名字引用 → 黑名单
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise SandboxRejected(
                f"line {node.lineno}: 禁止使用 `{node.id}` —— "
                f"沙箱里 eval/exec/open/__import__/globals 一律不给。"
                f"需要计算就用正常表达式, 需要 IO 就让工具层来。"
            )
        if isinstance(node, ast.Attribute) and node.attr in _BANNED_NAMES:
            # 拦 `.eval` / `.exec` / `.__import__` 这类属性访问
            if node.attr in ("eval", "exec", "__import__", "compile"):
                raise SandboxRejected(
                    f"line {node.lineno}: 禁止访问属性 `.{node.attr}`"
                )


# ============================================================
# 子进程 runner (这段源码被写进临时文件, 在子进程里跑)
# ============================================================

_RUNNER_SRC = r'''
import sys, os, io, json, contextlib, traceback

SENTINEL = "__QAGENT_SANDBOX_RESULT__ "
SRC_PATH = sys.argv[1]

buf = io.StringIO()
res = {"ok": False, "print_out": "", "traceback": "", "error_type": ""}

try:
    with open(SRC_PATH, "r", encoding="utf-8") as fh:
        src = fh.read()
except Exception:
    src = ""

g = {"__name__": "__main__"}
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        exec(compile(src, "<sandbox>", "exec"), g)
    res["ok"] = True
except BaseException as e:            # noqa: BLE001 - 子进程里什么都可能抛
    res["traceback"] = traceback.format_exc()
    res["error_type"] = type(e).__name__

res["print_out"] = buf.getvalue()

# os.write 直写 fd 2: 不经过 sys.stderr, 用户代码改不动它。
# 一行一个 JSON —— 用 \n 转义保证单行, 父进程按行找哨兵。
os.write(2, (SENTINEL + json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
'''


# ============================================================
# 杀进程树
# ============================================================

def kill_tree(proc: subprocess.Popen) -> None:
    """
    杀掉子进程**整棵树**, 不只是父进程。

    Windows: `taskkill /T /F` (T = tree, F = force)
    POSIX:   杀掉进程组 (killpg)

    为什么必须杀树 —— 用户代码里的 `sympy`/`numpy` 可能 spawn 孙进程,
    孙进程会一直持有 stdout/stderr 管道; 只 kill 父进程的话 communicate()
    会一直等管道关闭, 整个调用挂死 (这是最容易被忽略的一个坑)。
    """
    if proc.poll() is not None:
        return  # 已退出, 没什么好杀的

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=_DRAIN_TIMEOUT, check=False,
            )
        except Exception:
            pass  # 落到下面的 proc.kill() 兜底
    else:
        try:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass

    try:
        proc.kill()
    except Exception:
        pass


# ============================================================
# 主入口
# ============================================================

def run_code(
    code: str,
    scratch_dir: str | Path,
    timeout: int = DEFAULT_TIMEOUT,
    *,
    check: bool = True,
) -> SandboxResult:
    """
    在受限子进程里执行 code, 返回 SandboxResult。

    Args:
        code: Python 源码
        scratch_dir: 工作目录(唯一可写处), 会自动创建; runner 也在里面
        timeout: 硬超时秒数
        check: True = 过不了 AST 门就抛 SandboxRejected(不返回 rejected 结果);
               False = 把拒绝信息放进 result.rejected / reject_reason

    无论成功失败, **异常原文都会回传** —— 这是"喂回 LLM 修"的燃料, 不能吞。
    """
    # ★ 必须 resolve 成绝对路径 (2026-10-08 实测踩坑):
    #   Popen(cwd=相对路径) 会先把进程 chdir 到那个目录, **再** 用命令行里的相对路径
    #   去定位脚本 —— 于是 `python data/.../scratch/_runner.py` 被解析成
    #   `data/.../scratch/data/.../scratch/_runner.py`, 脚本不存在, 子进程秒退,
    #   表现为 SandboxBootstrapError。单测用 tmp_path(绝对)全绿, 一接 workspace
    #   (相对路径 data/research/working/...) 就 100% 挂 —— 典型的只在集成路径炸的坑。
    scratch = Path(scratch_dir).resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    # 静态门两种处理: check=True → 直接抛; check=False → 转成 rejected 结果返回。
    # 两种都**必须真的跑一遍 check_code** —— 否则 check=False 等于门形同虚设,
    # 代码会被照常执行, rejected 永远是 False(2026-10-08 单测时才发现)。
    reject_reason = ""
    try:
        check_code(code)
    except SandboxRejected as e:
        if check:
            raise
        reject_reason = e.reason

    if reject_reason:
        return SandboxResult(
            ok=False,
            elapsed_ms=0,
            rejected=True,
            reject_reason=reject_reason,
            error_type="SandboxRejected",
        )

    src_path = scratch / "main.py"
    src_path.write_text(code, encoding="utf-8")
    runner_path = scratch / "_runner.py"
    runner_path.write_text(_RUNNER_SRC, encoding="utf-8")

    started = time.time()
    timed_out = False
    proc = subprocess.Popen(
        [sys.executable, str(runner_path), str(src_path)],
        cwd=str(scratch),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env=_child_env(),
    )

    try:
        _, err_bytes = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_tree(proc)
        try:
            _, err_bytes = proc.communicate(timeout=_DRAIN_TIMEOUT)
        except Exception:
            # 管道被孙进程占着不放 —— 放弃读取, 不能卡在这里
            err_bytes = b""

    elapsed_ms = int((time.time() - started) * 1000)

    payload, stderr_raw = _parse_sentinel(err_bytes or b"")

    result = SandboxResult(
        ok=False,
        elapsed_ms=elapsed_ms,
        timed_out=timed_out,
        returncode=proc.returncode,
    )

    if timed_out:
        result.stderr = (
            f"[超时] 执行超过 {timeout}s 已被强制终止(整棵进程树已杀)。"
            f"常见原因: 死循环 / 递归深度爆了 / 计算量太大。"
            f"请缩小规模(如 n 从 1000 降到 20)、改用符号方法, 或直接调大 timeout。"
        )
        result.error_type = "Timeout"
        return result

    if payload is None:
        # 哨兵都没收到 —— 子进程在 import/启动阶段就死了(例如 numpy 装坏了)
        text = _decode(stderr_raw).strip()
        result.stderr = (
            f"[沙箱启动失败] 子进程没跑起来就没了 (returncode={proc.returncode})。\n"
            f"原始输出:\n{text or '(空)'}"
        )
        result.error_type = "SandboxBootstrapError"
        return result

    stdout = _truncate(payload.get("print_out", ""))
    result.truncated = len(payload.get("print_out", "")) > MAX_OUTPUT_CHARS

    result.ok = bool(payload.get("ok"))
    result.stdout = stdout
    result.error_type = payload.get("error_type", "")
    if not result.ok:
        result.stderr = _truncate(payload.get("traceback", ""))

    return result


# ============================================================
# 内部工具
# ============================================================

def _child_env() -> dict:
    """
    子进程环境变量 —— 刻意**不继承任何密钥**。

    父进程的 os.environ 里有 DEEPSEEK_API_KEY / 各种 token, 全量传给子进程
    等于"沙箱里的代码能直接调 LLM API 烧钱"。这里只留跑 Python 必需的。
    """
    keep = {"PATH", "PYTHONPATH", "PYTHONIOENCODING", "PYTHONHOME",
            "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
            "LOCALAPPDATA", "APPDATA", "NUMBER_OF_PROCESSORS", "LANG"}
    env = {k: v for k, v in os.environ.items() if k.upper() in {x.upper() for x in keep}}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"   # 别在 scratch 里撒 .pyc
    return env


def _parse_sentinel(data: bytes) -> tuple[dict | None, bytes]:
    """从子进程 stderr 里抠出哨兵行; 找不到返回 (None, 原始 stderr)"""
    text = _decode(data)
    for line in reversed(text.splitlines()):
        idx = line.find(_SENTINEL)
        if idx >= 0:
            try:
                return json.loads(line[idx + len(_SENTINEL):]), data
            except json.JSONDecodeError:
                return None, data
    return None, data


def _decode(data: bytes) -> str:
    return (data or b"").decode("utf-8", errors="replace")


def _truncate(text: str) -> str:
    """
    截断过长输出 —— **保留头尾**。

    只留头会丢掉最关键的末尾 traceback, 只留尾会丢掉 LLM 自己 print 的上下文。
    中间用标记掐掉。
    """
    if not text or len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    omitted = len(text) - MAX_OUTPUT_CHARS
    return (
        text[:half]
        + f"\n... [中间 {omitted} 字符因超长被截断, 共 {len(text)} 字符] ...\n"
        + text[-half:]
    )
