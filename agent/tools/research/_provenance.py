"""
agent.tools.research._provenance — 证据链: 编号 / 环境指纹 / verdict 判定 (P7 阶段 0)

设计文档: docs/RESEARCH_CLOSED_LOOP_PLAN.md §6.4 / §7

================================ 这一层为什么单独存在 ================================
沙箱只会告诉你"代码跑通了"。但**代码跑通 ≠ 命题成立** —— 写错断言的代码照样跑通,
写个 `print("VERDICT: PROVED")` 更是一秒证明一切。

所以本模块负责把"一次执行"变成一条**可复核的证据**, 守住三条:

1. **verdict 由代码里的显式标记决定, 不是由"跑通了"推断。**
   LLM 生成的验证代码必须在末尾 print 一行 `VERDICT: PROVED|REFUTED|PARTIAL`。
   没有标记 → 一律 `untested`, 哪怕 stdout 里全是 True。
   这条是整个功能的核心纪律: 让 LLM 自己声明判定依据, 且这行声明必须和代码逻辑对应。

2. **每条证据带环境指纹** (sympy/numpy 版本 + Python 版本 + 平台)。
   同一份代码在 sympy 1.13 和 1.14 上结果可能不同。没指纹的证据半年后无法复现,
   等于没有证据。

3. **证据编号连续且稳定** (ev_0001 / ev_0002...), 报告里靠它做引用锚点。

历史:
  - 2026-10-08 P7.0 首版
"""
from __future__ import annotations

import hashlib
import platform
import sys
from typing import Any

# ============================================================
# verdict 硬枚举
# ============================================================

#: 6 档判定, research_audit 会拒收表外值 (文档 §6.4)
VERDICTS: tuple[str, ...] = (
    "proved",        # 有可复现代码 + 符号断言为真
    "refuted",       # 找到了反例
    "partial",       # 有限范围检查/数值抽查通过, 不是全域证明
    "failed",        # 代码跑挂/超时/被门拒 —— 没能验
    "inconclusive",  # 验了但定不了 (如多次尝试仍不确定)
    "untested",      # 没有任何可复核的判定依据 (最常见: 代码跑通但没写 VERDICT)
)

#: 代码里允许出现的 VERDICT 标记值 → 内部枚举
_VERDICT_MARKERS = {
    "PROVED": "proved",
    "REFUTED": "refuted",
    "PARTIAL": "partial",
    "INCONCLUSIVE": "inconclusive",
}

_MARKER_PREFIX = "VERDICT:"


# ============================================================
# 编号
# ============================================================

def make_evidence_id(index: int) -> str:
    """ev_0001 形式。4 位够一个任务用(单任务证据上百条已是极端)"""
    return f"ev_{index:04d}"


def claim_id_for(claim_text: str, scope: str = "") -> str:
    """
    命题内容哈希 —— 对标 agent/tools/facts.py:105 `fact_id_for`。

    为什么照抄 fact 的做法: 同一个命题被验多次, 内容哈希天然合并成一条,
    多次验证变成多条证据挂在它下面。证据**追加不覆盖**, 于是
    "这个命题验过 3 次, 第 2 次结论不一样" 这种翻车能直接查出来。
    """
    raw = f"{(claim_text or '').strip()}|{(scope or '').strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


# ============================================================
# 环境指纹
# ============================================================

_FINGERPRINT_CACHE: dict[str, str] | None = None


def env_fingerprint() -> dict[str, str]:
    """
    记下这条证据是在什么环境里产生的 —— 缺了它, 证据不可复现。

    科学计算库版本是决定性因素: sympy 1.13 和 1.14 的化简结果可能不同,
    numpy 的浮点精度差异也会让边界用例翻转。软链一下拿不到版本号,
    所以各自 try。

    z3 同样在列 —— 它是 P1 引入的决定性依赖: 不同 z3-solver 版本对
    某个 unsat 的判定、超时后的 unknown 行为、smt 算子支持都可能不一样。
    一条 `unsat -> proved` 的结论如果没记 z3 版本, 半年后无法判断是
    真证还是换了求解器。
    """
    global _FINGERPRINT_CACHE
    if _FINGERPRINT_CACHE is not None:
        return dict(_FINGERPRINT_CACHE)

    fp = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()}-{platform.machine()}",
    }
    for mod in ("sympy", "numpy", "scipy", "networkx"):
        try:
            m = __import__(mod)
            fp[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            fp[mod] = "absent"

    # z3 特判: 它**没有** __version__ 属性, 版本只从 get_full_version() 拿。
    # 用通用 getattr 写法的后果是指纹里永远记成 "unknown" —— 而 z3 恰恰是
    # P1 里唯一能给全域结论的求解器, 一条 unsat->proved 没记版本就不可复现。
    # 这个 bug 真实发生过: 跑完 20 题基准才发现指纹里 z3=unknown。
    try:
        import z3 as _z3

        fp["z3"] = _z3.get_full_version()
    except Exception:
        fp["z3"] = "absent"

    _FINGERPRINT_CACHE = fp
    return dict(fp)


def fingerprint_line() -> str:
    """给报告用的一行人类可读摘要"""
    fp = env_fingerprint()
    return " | ".join(f"{k}={v}" for k, v in fp.items())


# ============================================================
# verdict 判定
# ============================================================

def extract_verdict_marker(stdout: str) -> tuple[str | None, str]:
    """
    从 stdout 里抠出最后一条 `VERDICT: XXX` 标记。

    取**最后一条**: LLM 可能先探索再定论, 最后一条才是它最终的判断。
    返回 (verdict | None, 原始标记文本)。
    """
    if not stdout:
        return None, ""
    found: str | None = None
    raw = ""
    for line in stdout.splitlines():
        idx = line.upper().find(_MARKER_PREFIX)
        if idx >= 0:
            token = line[idx + len(_MARKER_PREFIX):].strip().split()[0] if line[idx + len(_MARKER_PREFIX):].strip() else ""
            raw = line[idx:].strip()
            if token in _VERDICT_MARKERS:
                found = _VERDICT_MARKERS[token]
    return found, raw


def judge_verdict(sandbox_result: Any, stdout: str | None = None) -> tuple[str, str]:
    """
    由一次执行结果 + 代码里的 VERDICT 标记, 决定这条证据的判定。

    优先级 (从硬到软):
      1. 没跑成 (rejected / timed_out / 异常)      → failed    ← 无条件压过一切
      2. 代码没写 VERDICT 标记                      → untested ← 纪律核心
      3. 代码写了标记且值合法                       → 该值

    第 1 条优先级最高: 代码崩了却 print 了 VERDICT(比如把标记写在崩溃前),
    那是无效证据, 不能采信。

    Args:
        sandbox_result: _sandbox.SandboxResult (用鸭子类型, 避免循环依赖)
        stdout: 实际 stdout, 默认取 result.stdout

    Returns:
        (verdict, reason) —— reason 是给 LLM 看的一句话解释
    """
    out = stdout if stdout is not None else getattr(sandbox_result, "stdout", "")

    # 1. 没跑成
    if getattr(sandbox_result, "rejected", False):
        return "failed", f"代码被沙箱静态门拒绝: {getattr(sandbox_result, 'reject_reason', '')}"
    if getattr(sandbox_result, "timed_out", False):
        return "failed", "执行超时被终止, 没拿到判定结果"
    if not getattr(sandbox_result, "ok", False):
        return "failed", (
            f"代码执行出错({getattr(sandbox_result, 'error_type', '?')}), "
            f"没有产生可复核的结果"
        )

    # 2/3. 看标记
    verdict, raw = extract_verdict_marker(out)
    if verdict is None:
        return "untested", (
            "代码跑通了, 但输出里没有 `VERDICT: PROVED/REFUTED/PARTIAL` 标记 —— "
            "**跑通不等于证明成立**。请在验证代码末尾显式 print 出判定依据。"
        )
    return verdict, f"代码显式标记 {raw}"


# ============================================================
# 证据组装
# ============================================================

def build_evidence(
    index: int,
    *,
    kind: str,
    verdict: str,
    reason: str,
    code: str = "",
    stdout: str = "",
    stderr: str = "",
    elapsed_ms: int = 0,
    claim_id: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    组装一条证据记录 (对应 _evidence.jsonl 的一行)。

    字段与文档 §7 的 evidence 表一一对应, 外加 env 指纹与 reason:
      id       ev_0001 引用锚点, 报告里靠它
      kind     code | solver | lean | citation (文档 §6.4)
      verdict  6 档硬枚举
      reason   为什么是这个 verdict —— 给人和 LLM 复核用
      env      环境指纹, 没它证据不可复现
    """
    ev: dict[str, Any] = {
        "id": make_evidence_id(index),
        "kind": kind,
        "verdict": verdict if verdict in VERDICTS else "untested",
        "reason": reason,
        "claim_id": claim_id,
        "code": code,
        "stdout": stdout,
        "stderr": stderr,
        "elapsed_ms": elapsed_ms,
        "env": env_fingerprint(),
    }
    if extra:
        ev.update(extra)
    return ev


def record(ws, *, kind: str, verdict: str, reason: str, **fields) -> dict[str, Any]:
    """
    把一条证据落进 workspace —— 所有工具共用的收口。

    编号从 ev_0001 起(内部把 count+1), 落盘 _evidence.jsonl。
    有了它, 各工具就不必各自重复"算 index → build → append"这三行,
    也就不会出现"有的工具从 0000 起、有的从 0001 起"这类漂移。
    """
    index = ws.evidence_count() + 1
    ev = build_evidence(
        index, kind=kind,
        verdict=verdict if verdict in VERDICTS else "untested",
        reason=reason, **fields,
    )
    ws.append_evidence(ev)
    return ev


def claim_block(verdict: str, ev_ids: list[str]) -> str:
    """
    结论区渲染: 判定 + 证据编号。

    报告里每条结论都必须长得像这样 —— 带 ev_ 编号, 没有编号的句子不许进结论区
    (文档 §8 条款 1)。这个函数就是那条规则的代码形态。
    """
    if not ev_ids:
        return f"**{verdict}** — ⚠️ 无证据支撑(不可作为结论引用)"
    refs = ", ".join(ev_ids)
    return f"**{verdict}** — 证据: {refs}"
