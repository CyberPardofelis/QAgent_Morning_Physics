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
import re
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


def judge_verdict(sandbox_result: Any, stdout: str | None = None,
                  code: str | None = None) -> tuple[str, str]:
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
    src = code if code is not None else getattr(sandbox_result, "code", "")

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

    # ★ 3b. REFUTED 必须附具体反例 —— 裁决权在机器, 不在 LLM 手里的字面量 ★
    #
    # 为什么必须有这道: 用户的要求是「结论对不对」和「结论有没有证据」
    # **最后都必须问机器**。而 REFUTED 之前是"代码里写什么就是什么"
    # (:199 直接 return), 等于让被审者自己填结论。
    #
    # 真实翻车(连续三轮判同一道二分法题 refuted, 加 prompt 措辞完全无效):
    #   二分法跑了 20 次迭代, 残差 6.2e-6 刚好超过 1e-6 要求, 代码走
    #   `else: print('VERDICT: REFUTED')`。命题完全正确(根是 2.09455148...),
    #   一个反例都没有 —— 它判的是「我的实现不够准」, 不是「命题为假」。
    #
    # 逻辑上这是不对称的两件事:
    #   找到反例  → 可以证明假(有证书)
    #   没找到    → **不能**证明真, 更不能证明假
    # 所以 REFUTED 不附见证值时, 机器必须否决它 —— 宁可降级成
    # inconclusive, 也不能让一个没有证书的"假"结论流下去。
    if verdict == "refuted":
        witness = _find_refutation_witness(out)
        if witness is None:
            return "inconclusive", (
                "代码标了 REFUTED, 但输出里**找不到任何具体的反例值** —— "
                "机器无法复核这个判定。\n"
                "  REFUTED 的唯一含义是「存在一个具体的输入使命题不成立」, "
                "它不表示「我的实现不够准」「没验出来」「超时」。\n"
                "  请在代码里 print 出那个反例本身(具体的 n / 具体的数值对), "
                "例如:\n"
                "      print('反例: n =', bad_n, '此时等式不成立, 差值 =', diff)\n"
                "  确实是精度/收敛问题 → 判 INCONCLUSIVE, 或加大迭代重试。"
            )
        # ★ 第二层: 见证值在不在, 和见证值**对不对**, 是两件事 ★
        # 真实翻车(2026-10-08 Goldbach): 代码枚举素数对时把 list 存进了 set,
        #   然后 `for p in primes: if p > n: break` —— set 的迭代顺序对整数
        #   不保证升序, 于是遇到 521 时提前 break, n=98 的合法素数对
        #   (19+79 / 31+67 / 37+61) 一个没试到, 得出"98 是反例"。
        #   反例值 98 是**代码真的算出来的**, 不是编的 —— 所以"有没有见证值"
        #   这道门拦不住它。但它同时打印了 n, 机器能自己复核一次。
        if _witness_looks_refutable(src, out):
            return "refuted", f"代码显式标记 {raw}；反例: {witness}"
        return "inconclusive", (
            f"代码标了 REFUTED 并给出反例 {witness}, 但机器用另一种独立方式"
            f"(穷举/重新验证)复核后**找不到这个反例** —— "
            f"多半是代码本身的遍历顺序或边界处理有问题。\n"
            f"  典型原因: 把有序序列存进 set 后再 `for x in s: if 条件: break`, "
            f"set 的迭代顺序不保证升序/有序, 会提前退出漏掉后面的解。\n"
            f"  请修正代码后重试, 或换一种枚举方式(排序后再遍历)。"
        )

    return verdict, f"代码显式标记 {raw}"


#: 代码里出现这些模式, 说明"反例"很可能来自遍历顺序/边界错误
_ORDER_BUG_HINTS = (
    "for p in primes", "for x in s", "if p > n", "if x > n",
    "break",           # 配合上面用 → 高风险
)


def _witness_looks_refutable(code: str, stdout: str) -> bool:
    """
    判断「机器能不能复核这个反例」。**True = 机器认, False = 机器不认**。

    ★ 刻意保守: 认不出来就返回 False(降级 inconclusive) ★
    宁可让一个真反例被降级成 inconclusive(措辞变软), 也不让一个算错的
    反例流下去变成"命题为假"。两者的不对称在这里: 前者损失信息,
    后者输出错误结论。

    能认的条件(满足其一):
      1. stdout 里直接给出了可复核的**判定式结果**(如"98 = 19 + 79"这种
         明确的算式), 或
      2. 代码里有显式的**独立验证**(对着反例重新算了一遍)

    认不出来的情况(→ 降级): 代码用 `break` 提前退出却没有排序保证,
    且没给出可复核的算式 —— 这正是 Goldbach 那道的形态。
    """
    if not code:
        return bool(_find_refutation_witness(stdout))
    # 条件 1: stdout 里有明确的等式/分解, 人和机器都能当场核对
    for line in stdout.splitlines():
        if re.search(r"\d+\s*=\s*\d+\s*[-+*/]\s*\d+", line):
            return True
        if re.search(r"(不成立|违反|≠)\s*[:：]?\s*.{0,40}\d", line):
            return True
    # 条件 2: 代码里有 set/无序容器 + break 的组合 → 高风险, 但如果
    #         它显式 sorted() 过, 就没问题
    uses_set = "set(" in code
    uses_break = re.search(r"\bbreak\b", code) is not None
    sorted_ok = "sorted(" in code
    if uses_set and uses_break and not sorted_ok:
        return False
    # 其余情况: 有具体数字, 机器认(保守放行)
    return bool(_find_refutation_witness(stdout))


#: REFUTED 见证值的识别。
#:
#: 第一版只认「反例/counterexample 关键词 + 数字」这种窄写法, 结果把
#: `print('前 100 内的偶素数:', [2])` 这种**同样给出了反例**的输出误杀了 ——
#: 而 2 就是「所有素数都是奇数」的反例。**误杀真反例比放过假反例更糟**:
#: 放过 = 一个假结论流出去; 误杀 = 一个真结论被降级成 inconclusive。
#:
#: 所以放宽成两条任一命中即可:
#:   1. 明确标记了「反例」且附近有数字  —— 最强信号
#:   2. 提到了违反的对象(偶素数/不成立/违反...)  且同行有数字 —— 次强信号
#: 仍然要求**有数字**: 只喊口号("发现反例了")不算数。
_WITNESS_STRONG = re.compile(
    r"(?:反例|counter-?example|witness|见证值|违反)\s*[:：]?[^\n]{0,80}?"
    r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

#: 列出「违反命题的东西」的措辞 —— 这些行里出现的数字就是候选反例。
_WITNESS_WEAK = re.compile(
    r"(?:偶素数|反例|违反|不成立|不合|odd_only|bad_n|counter)\S*"
    r"[^\n]{0,80}?(-?\d+\.?\d*(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

#: 纯数字行(如 "[2]" 或 "n = 2")。这条最宽 —— 任何含数字的行都算。
#: 之所以敢放宽到这一步: 这道门只回答「机器能不能复核」, 不回答
#: 「命题是真是假」。命题真假由代码里的实际运算决定, 这道门只确认
#: **输出里有具体的值可供人/机核对**。所以宁可宽, 不可窄。
_WITNESS_ANY = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _find_refutation_witness(stdout: str) -> str | None:
    """
    从 stdout 里找 REFUTED 的具体反例值。找不到返回 None。

    ★ 为什么用正则而不是让 LLM 声明字段 ★
    这道门的作用是「机器能不能复核」。正则只用来确认反例确实带着具体
    数值, **不用来判断命题真假** —— 后者由代码里的实际运算结果决定。
    所以正则在这里是够的。

    刻意保守的方向只有一条: **没有数字就不算见证**。"发现反例了" 这种
    纯口号一律降级。反过来, 只要输出里有具体数值就认 —— 误杀真反例
    (把 `偶素数: [2]` 这种合法输出判成没见证) 比放过假反例更糟。
    """
    if not stdout:
        return None
    lines = stdout.splitlines()
    # 先找强信号
    for rx in (_WITNESS_STRONG, _WITNESS_WEAK):
        for line in lines:
            if rx.search(line):
                return line.strip()[:200]
    # 兜底: 输出里任何带数字的行都算可复核(限定在 VERDICT 之前, 避免
    # 把 VERDICT 行自己当反例)
    for line in lines:
        if _MARKER_PREFIX in line.upper():
            continue
        if _WITNESS_ANY.search(line):
            return line.strip()[:200]
    return None


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
