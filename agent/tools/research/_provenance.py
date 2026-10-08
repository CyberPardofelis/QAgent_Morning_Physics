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


#: 复合命题(双向题)的机器判据。命中才启用 R3 —— 单向命题上这套会误伤。
_COMPOUND_CLAIM = re.compile(
    r"(?:并(?:给出|证明|验证|构造|展示|输出)|"
    r"并且|同时(?:给出|证明|验证|构造)|"
    r"(?:给出|证明|验证).{0,40}?的?(?:着色|构造|解|见证)"
    r".{0,40}?(?:并|且|再))"
)

#: 「这一部分成立了」的措辞
#:
#: ★ 为什么收 `PROVED` 这个词, 而不是只收 YES/True/成立 ★
#: 实测翻车(2026-10-08 V2 hard-011 Petersen 色数):
#:     Part A (存在 3 着色): REFUTED
#:     Part B (不存在 2 着色): PROVED
#:     VERDICT: REFUTED
#: 双向题, 一部分证成了另一部分没证成, 正确处置是 PARTIAL —— 但 R3 没拦住,
#: 因为模型写的恰恰是**最规整的机器可读标签** PROVED/REFUTED, 而这两条正则
#: 只认 YES/True/成立 和 NO/False/失败。
#:
#: 规律: 判据必须认模型**会主动打出来**的机器可读写法, 而不只是人话。
#: 收紧到人话上, 门就只在人话出现时才生效。
#:
#: 刻意**不**收 unsat/sat: `z3 判定: unsat` 在「证明不能 2 着色」里恰恰是
#: **该部分成立**, 而在「证明是 2 可着色」里是不成立 —— 词义随命题翻转,
#: 拿它当"失败"是反的, 只能靠人读上下文, 机器读不了。
_PART_TRUE = re.compile(
    r"(?:[:：]\s*)?(?:YES|True|OK|PASS|成立|通过|正确|已证|✓|✔)\b"
    r"|(?<!NOT )(?<!NOT_)\bPROVED\b",
    re.IGNORECASE)
#: 「这一部分失败了」的措辞
_PART_FALSE = re.compile(
    r"(?:[:：]\s*)?(?:NO|False|FAIL|失败|不成立|不满足|抓出|找到|✗|✘|✖)\b"
    r"|(?<!NOT )\bREFUTED\b",
    re.IGNORECASE)


#: 「已知/参考值」的措辞
_REFERENCE_CONST = re.compile(
    r"(?:known\s+(?:reference|expected)|reference\s+value|expected\s+value"
    r"|known\s+value|标准值|参考值|已知值|文献值|公认值)",
    re.IGNORECASE,
)
#: 源码里的裸整数字面量
_LITERAL_NUM = re.compile(r"(?<![A-Za-z_0-9.])(\d{4,})(?![A-Za-z_0-9.])")


def _refuted_by_hardcoded_reference(
    code: str, stdout: str,
) -> tuple[str, str] | None:
    """
    R4: REFUTED 是不是「和源码里硬编码的『已知值』比不上」得来的？
    **返回非 None = 靶子不成立**。

    ★ 真实翻车(2026-10-08 V2 hard-004 Catalan-50)★
    代码用三种独立方法(二项式/阶乘/递推)都算出了同一个值, 并且互相印证
    `All three methods agree? : True`; 然后它又去和一个硬编码的
    "Known reference value" 比, 不等, 输出 `VERDICT: REFUTED`。
    **那个参考值是模型自己编的** —— C_50 算出来的值是对的, 它证伪了一条真命题。

    为什么 prompt 提醒不够: 这类输出**完全合法**(代码跑通了、打印了数值、
    反例值具体), 前两道 REFUTED 门都认。必须从源码上看出来:
    那个"参考值"是**字面量**而不是算出来的 —— 算出来的数字不会以裸整数字面量
    的形式出现在源码里。

    判据: 输出里出现"参考值"措辞, 且它给出的那个大数字**在源码里是字面量**。
    """
    if not stdout or not _REFERENCE_CONST.search(stdout):
        return None
    if not code:
        return None
    for m in _REFERENCE_CONST.finditer(stdout):
        line = stdout[m.start():].splitlines()[0]
        nums = _LITERAL_NUM.findall(line)
        for num in nums:
            # 在源码里作为**独立字面量**出现 = 硬编码, 不是算出来的
            if _LITERAL_NUM.search(code) and num in code:
                return (
                    "inconclusive",
                    "REFUTED 来自「和源码里硬编码的『已知值』比不上」—— "
                    f"但那个参考值是**代码里的字面量**, 不是算出来的见证值。\n"
                    f"  参考值出现在源码中: {num[:32]}{'...' if len(num) > 32 else ''}\n"
                    "  你记忆里的『已知答案』会错, 而且错得很自信 —— 2026-10-08 "
                    "Catalan-50 就是这样证伪了一条真命题。\n"
                    "  REFUTED 的正确含义是: 找到一个**具体的输入**, 让命题在它上面不成立。\n"
                    "  而不是「我的结果和一个我没算出来的常量对不上」。\n"
                    "  如果几种独立方法互相印证过了, 那个结果就比参考值可信。"
                )
    return None


#: 「确认某事成立」的措辞。用于 R5b: 见证值出现在这些句子里 = 它其实被当成正解。
_AFFIRMING_LINE = re.compile(
    r"(?:成立|一致|正确|正解|通过|找到|吻合|一致了|verified|holds)", re.IGNORECASE)

#: 「全部通过 / 全部成立」这类**整体**自述成功的措辞。
#:
#: ★ 这里刻意**不**收「任意中文词 + ? + True」★
#: 试过, 被自己的反向测试打回: 真实反例的输出里也常有前置自检
#: 「环境校验? True」—— 那是**对环境的肯定**, 不是对命题的肯定,
#: 按宽规则会把真反例误杀。误杀真反例 = 一个真结论被降级, 比放过假结论更糟。
#: 所以整体自述只认下面这些**明确指向命题本身**的写法。
_ALL_PASSED = re.compile(
    r"(?:all_passed\s*[=:]\s*True|all_good\s*[=:]\s*True"
    r"|holds\s*[=:]\s*True|passed\s*[=:]\s*True"
    r"|全部通过|全部一致|全部成立|都成立|验证通过)",
    re.IGNORECASE,
)


#: 「见证值」那一行里最大的那个整数 —— 用来判断它是否在别处被当成正解。
#: 取**最大**值是对的: 见证行常同时印好几个数(n=7, 差值=0.25), 而
#: "被当成正解"的那个通常是主结果。
def _witness_number(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if _MARKER_PREFIX in line.upper():
            continue
        if not re.search(r"(?:反例|counter-?example|witness|见证值)", line,
                         re.IGNORECASE):
            continue
        nums = re.findall(r"\d{3,}", line)
        if nums:
            return max(nums, key=len)
    return None


#: 「几路验算互相不一致」的措辞 —— R6 的抓手。
#:
#: 两条分支强度不同:
#:   1. 明确把比对结果写成 False ——「All three methods agree: False」
#:      「两种算法一致? False」, 这类措辞指向"某项比对没过", 是强信号。
#:   2. 裸的「不一致/mismatch/disagree」—— 弱信号, 因为否定句里也会出现,
#:      所以必须排除「不一致的项: 无」这种**自检说没有不一致**的写法。
_DISAGREE_FAILED = re.compile(
    r"(?:一致|吻合|相同|相等|符合|agree\w*|match\w*|consistent|verify\w*)"
    r"[^\n]{0,24}?(?:[:：=]\s*|is\s+)\s*False"
    r"|(?:不一致|不符|不相等|mismatch\w*|disagree\w*)"
    r"(?![^\n]{0,12}?(?:[:：=]\s*)?(?:无|没有|None|空|none))",
    re.IGNORECASE,
)


# ============================================================
# REFUTED 侧: 唯一的一道门 —— 「输出和它自己的判定对不对得上」
# ============================================================
#
# ★ 这里曾经有四道独立的门, 每次翻一次车加一道 ★
# 结果是门数只增不减, 而它们问的其实是**同一个问题**。收成一张表:
# **加新形态 = 加一行, 不是加一个函数**。
# (这条是被当面指出来的: "你是不是又在把问题越搞越复杂了"。)
#
# 四道门各自的真实翻车现场 —— 形态不同, 病根是同一个:
#   「说没找到却判假」  Erdős–Straus: 「已验证 0 个 / 未找到正整数解 / 反例: n=2」
#   「说全过了却判假」  Fibonacci 矩阵: n=1..20 all_passed=True, 却输出 REFUTED
#   「反例被当成正解」  Pell-61: 正解 1766319049 被标成「反例」, 同一份输出里
#                     另一句写着「方程成立? True」
#   「各部分都定不了」  R(3,4)=9: z3 回 sat 但见证自检 False, A/B 都判 INCONCLUSIVE,
#                     却汇总成 REFUTED
#   「几路验算对不上」  p(100): sympy 和独立 DP 都算出 190569292(对), 第三路算出
#                     58527732062, 于是宣布命题为假
#   「计数和清单不符」  Erdős–Straus 另一轮: 声明 997 个失败, 只列了 20 个
#
# 共同的病根: **REFUTED 的依据不是"找到了一个反例"** —— 是找不到、是实现有 bug、
# 是编码错了、是自己数错了。这些统统都不该输出 REFUTED。
#
# ★ 为什么前几道门全都放行 ★
# 它们问的全是「结论和**命题**对不对得上」, 没有一道问「结论和**它自己的输出**
# 对不对得上」。所以这道门不是又一道门, 是补上了一个从来没被问过的维度。

#: 「明确声明了一个反例」—— 所有自相矛盾检查的**统一豁免条件**。
#:
#: 真反例的输出里也常出现「某项检查 False」「不一致」这类无关字样
#: (前置环境自检就会打「环境校验? True」), 所以豁免必须是
#: 「**它指出了一个具体的反例**」, 而不是「有没有出现过 False」。
def _declares_counterexample(stdout: str) -> bool:
    return bool(stdout) and bool(_WITNESS_STRONG.search(stdout))


#: 「几路验算互相不一致」。
#: 两条分支强度不同: 明确写成 `agree: False` 是强信号; 裸的「不一致」是弱信号,
#: 因为否定句里也会出现, 所以要排除「不一致的项: 无」这种自检行。
_DISAGREE_FAILED = re.compile(
    r"(?:一致|吻合|相同|相等|符合|agree\w*|match\w*|consistent|verify\w*)"
    r"[^\n]{0,24}?(?:[:：=]\s*|is\s+)\s*False"
    r"|(?:不一致|不符|不相等|mismatch\w*|disagree\w*)"
    r"(?![^\n]{0,12}?(?:[:：=]\s*)?(?:无|没有|None|空|none))",
    re.IGNORECASE,
)

#: 各部分自报的「定不了」。只看**非总结论行** —— 否则分不清哪个才是最终判定。
_PART_INCONCLUSIVE = re.compile(r"\binconclusive\b", re.IGNORECASE)

#: 「一共 N 个」这类计数声明。
_COUNT_CLAIM = re.compile(
    r"(?:total|count|number|共|总共|合计|计数)"
    r"[^\n]{0,40}?(?:[:：=]\s*)(\d{1,9})\s*$"
    r"|[^\n]{0,30}?(?:失败|反例|counterexample|failure|bad|solution\s*=\s*None)"
    r"[^\n]{0,20}?(?:[:：=]\s*)(\d{1,9})\s*$",
    re.IGNORECASE | re.MULTILINE,
)
#: 清单行 `Failed n values: [...]`。
#:
#: ★ 方括号前**必须紧跟冒号或等号** ★
#: 第一版只要求"关键词后面有个方括号", 结果把 `Total failures in [2,1000]: 997`
#: 里的**定义域范围** `[2,1000]` 当成了反例清单, 判词里写出的数字完全是编的。
#: 冒号/等号在括号**前面** 才说明这个列表是某个字段的值; 在后面说明是范围。
_LIST_LINE = re.compile(
    r"(?:反例|counter-?example|fail\w*|bad|found|n\s*values?|invalid|"
    r"失败|异常|列表|list)\S*[^\n]{0,24}?[:：=]\s*\[([^\]\n]{3,})\]",
    re.IGNORECASE,
)
#: 「只列前 N 个」这类显式截断说明。有它就不能拿长度去对计数 —— 那是诚实输出。
_TRUNCATED = re.compile(
    r"(?:前\s*\d+\s*(?:个|条|项)|省略|截断|truncat|elided|omitted|\.\.\.|…)",
    re.IGNORECASE,
)


def _contradiction_msg(kind: str, stdout: str) -> str | None:
    """
    判断 stdout 有没有命中某一种自相矛盾。命中返回判词, 否则 None。

    这几个检查里只有三个需要真的解析(而不是一条正则), 所以只有它们是函数:
      · affirmed_witness —— 要在**同一个数字**上跨行比对
      · own_inconclusive  —— 要逐行看, 并排除总结论行
      · count_list       —— 要数清单项数
    其余是纯正则, 直接写在表里。
    """
    if kind == "found_nothing":
        if not _NOTHING_FOUND.search(stdout):
            return None
        return (
            "你的输出**自相矛盾**: 一边说「范围内未找到 / 没找到反例」, "
            "一边又判了 REFUTED。\n"
            "  REFUTED 的唯一含义是「**找到了**一个具体输入使命题不成立」。"
            "一个说自己什么都没找到的程序, 不可能同时找到了反例。\n"
            "  实测教训(2026-10-08 Erdős–Straus): 输出里同时出现"
            "「已验证的 n 数量: 0」「未找到正整数解」「反例: n = 2」—— "
            "那个 n=2 是编的, 而 n=2 其实有解。\n"
            "  请二选一:\n"
            "    · 真的跑完了并且没找到反例 → 那是 PARTIAL, 不是 REFUTED\n"
            "    · 真的找到了反例 → 把它算清楚再打印出来, 并确认那个值确实违反命题"
        )
    if kind == "all_passed":
        if not _ALL_PASSED.search(stdout):
            return None
        return (
            "你的输出**自相矛盾**: 一边说「全部通过 / all_passed = True」, "
            "一边又判了 REFUTED。\n"
            "  你在定义域内**逐个验证了命题成立**, 这和「命题为假」不可能同时为真。\n"
            "  实测教训(2026-10-08 V1 Fibonacci 矩阵): 逐个验证 n=1..20 全对, "
            "然后尝试补一个归纳证明、归纳步公式写错了, 于是输出 REFUTED —— "
            "被驳的其实是「我那个写错的归纳步」, 不是命题。\n"
            "  正确的处置:\n"
            "    · 你要的结论已经拿到了(范围内全过) → 按范围给 PROVED 或 PARTIAL\n"
            "    · 想给全域证明 → 把归纳步改对。归纳步写错只说明"
            "**证明没写出来**, 不说明命题是假的"
        )
    if kind == "disagreement":
        # 没有反例声明才拦 —— 见 _declares_counterexample
        if _declares_counterexample(stdout) or not _DISAGREE_FAILED.search(stdout):
            return None
        return (
            "代码标了 REFUTED, 但它给出的唯一依据是「**我的几路验算互相不一致**」—— "
            "输出里没有任何一个具体的反例。\n"
            "  REFUTED 的含义是「存在一个具体的输入使命题不成立」。"
            "几路验算对不上, 说明**其中某一路实现有 bug**, "
            "不说明命题为假 —— 尤其当多数路都给出同一个值时。\n"
            "  实测教训(2026-10-08 p(100)): sympy 的 partition 和独立 DP 两路都算出 "
            "190569292(正确值), 第三路广义五边形数递推算出一个物理上不可能的巨大值, "
            "就宣布 REFUTED。错的是那一路, 不是命题。\n"
            "  正确处置: ①先确认哪几路是独立的 —— 共享同一个错误假设的实现不算独立"
            "校验(比如三种写法都调 sympy); ②把不一致的那一路单独复核(逐行对��规模输入"
            "验算递推式); ③若你要的答案已经拿到了(多路一致), 直接按拿到的结论判 "
            "PROVED/PARTIAL。\n"
            "  确实存在反例 → 把它 print 出来(`反例: n = ..., 差值 = ...`)再判 REFUTED。"
        )
    if kind == "own_inconclusive":
        if _declares_counterexample(stdout):
            return None                  # 另一半真找到反例是合法的
        for line in stdout.splitlines():
            if _MARKER_PREFIX in line.upper():
                continue                 # 总结论行不算"某一部分的判定"
            if _PART_INCONCLUSIVE.search(line):
                return (
                    "你的输出里各部分自己都判了 **INCONCLUSIVE**"
                    f"（「{line.strip()[:120]}」）, 却汇总成了 REFUTED。\n"
                    "  判定只能往下合取, 不能往上抬: 一部分说'定不了', "
                    "推不出整个命题为假。\n"
                    "  实测教训(2026-10-08 R(3,4)=9): z3 对'找一个 9 顶点反例'回了 sat, "
                    "但代码**紧接着校验那个见证、发现它不成立**"
                    "(「经子集检查有效 = False」)—— 是约束编码错了, 不是命题被证伪。"
                    "R(3,4)=9 是已知成立的定理。\n"
                    "  正确处置: 求解器说 sat 但见证自检不过 = **编码有问题**。"
                    "回去改约束(尤其是'不存在红三角'这种否定条件要用 "
                    "`Or([Not(x), Not(y), Not(z)])` 而不是写错成合取), "
                    "重跑后再判。\n"
                    "  确实找到了反例 → 把反例本身 print 出来再判 REFUTED。"
                )
        return None
    if kind == "count_list":
        got = _count_vs_list(stdout)
        if got is None:
            return None
        claimed, listed = got
        return (
            f"你的输出**自相矛盾**: 声明有 **{claimed}** 个反例/失败, "
            f"但实际列出来的清单只有 **{listed}** 项。\n"
            "  REFUTED 建立在「我找到了 N 个反例」这个计数上, 而计数和清单对不上, "
            "这个结论就没有依据了。\n"
            "  实测教训(2026-10-08 Erdős–Straus): 声明 997 个失败却只列了 20 个; "
            "而且它对 n=4..23 全报 `solution=None`, 但 n=4 的解平凡到 "
            "`1/2 + 1/3 + 1/6 = 1` —— **搜索函数本身是坏的**, 不是猜想被证伪。\n"
            "  正确处置: ①计数和清单必须一致(只列前 N 个就明写「前 N 个」); "
            "②先在**小 n 上自检**你的求解器 —— 已知有解的那些, 你必须能找出来; "
            "③求解器找不到解 ≠ 不存在解。\n"
            "  确实有反例 → 把反例本身逐个 print 出来, 且让计数和清单一致。"
        )
    if kind == "affirmed_witness":
        num = _witness_number(stdout)
        if not num:
            return None
        for line in stdout.splitlines():
            if _MARKER_PREFIX in line.upper():
                continue
            # 判据很窄: **被当作反例的那个数字, 在别处被一句"确认它成立"的话提到**。
            # 比"输出里出现过 True 就拦"精确得多 —— 「环境校验? True」是对
            # **环境**的肯定, 跟命题无关, 不能拿来当矛盾证据。
            if num in line and _AFFIRMING_LINE.search(line):
                return (
                    f"你的输出**自相矛盾**: 你把 **x = {num}** 当成反例, "
                    f"但同一份输出里另有一句确认它成立:\n"
                    f"  「{line.strip()[:160]}」\n"
                    "  这两句话不可能同时为真。真反例不会被你自己确认成正解。\n"
                    "  实测教训(2026-10-08 Pell-61): 两种独立方法都算出了正解 "
                    "1766319049, 但一个**自研的结构判据**没通过 —— "
                    "周期为奇数时解的下标是 2L-1 而不是 L —— "
                    "于是它把这个**正确答案**标成了反例。\n"
                    "  处置: 你要的答案已经拿到了。请确认它满足方程、"
                    "并把「最小性」用遍历证明(穷举所有更小的候选), "
                    "然后按实际拿到的范围给 PROVED 或 PARTIAL。"
                )
        return None
    raise ValueError(f"未知的自相矛盾检查: {kind}")


def _count_vs_list(stdout: str) -> tuple[int, int] | None:
    """声明的计数 vs 实际列出的清单项数。对不上返回 (声明数, 清单数)。"""
    counts = [int(a or b) for a, b in
              ((m.group(1), m.group(2)) for m in _COUNT_CLAIM.finditer(stdout))
              if (a or b)]
    if not counts:
        return None
    listed: list[str] | None = None
    for line in stdout.splitlines():
        m = _LIST_LINE.search(line)
        if not m:
            continue
        items = [x for x in re.split(r"[,\s]+", m.group(1)) if x]
        # 必须**只含数字项**, 否则那不是一份数据清单(可能是字符串列表)
        if items and all(re.fullmatch(r"-?\d+(?:\.\d+)?", x) for x in items):
            listed = items
            break
    if listed is None or _TRUNCATED.search(stdout):
        return None
    claimed = max(counts)
    return None if len(listed) == claimed else (claimed, len(listed))


#: 六种自相矛盾的**顺序表**。顺序有意义: 从"措辞最明确"排到"需要解析"。
#: 加新形态往这里加一行, 不要新开一个门。
_CONTRADICTIONS = (
    ("found_nothing", "说没找到却判假"),
    ("all_passed", "说全过了却判假"),
    ("affirmed_witness", "反例被自己当成正解"),
    ("own_inconclusive", "各部分都定不了却判假"),
    ("disagreement", "几路验算对不上"),
    ("count_list", "计数与清单不符"),
)


#: 凭证行: `COUNTEREXAMPLE: n=15 | lhs=60 | rhs=无解 | violated=True`
#:
#: ★ 为什么用**一条格式**而不是让机器去猜「这句像不像谎言」 ★
#: 历史上 REFUTED 侧堆了六道门(说没找到/说全过了/反例被自己当成正解/
#: 各部分都定不了/几路验算对不上/计数与清单不符), 它们问的其实是同一个问题 ——
#: 「这个 REFUTED 拿什么当依据」。六道门只能拦住**已经见过的**那种谎。
#:
#: 凭证制把这道门反过来了: **不匹配谎言, 匹配凭证**。
#: 一个 REFUTED 成立, 当且仅当它交出了一个具体输入 + 在该输入上违反命题的值。
#: 交不出来就是交不出来 —— 不管它是「没找到」「几路对不上」还是「数错了」。
#: 这样一条规则覆盖六种形态, 而且能拦住我没见过的。
_COUNTEREXAMPLE = re.compile(
    r"^\s*COUNTEREXAMPLE\s*:\s*(?P<fields>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FIELD = re.compile(r"(\w+)\s*=\s*([^|]+?)\s*(?=\||$)")


def _counterexample_credential(stdout: str) -> dict[str, str] | None:
    """
    判断 REFUTED 有没有交出**具体反例**。两种合格形态:

      1. 标准凭证行 `COUNTEREXAMPLE: n=15 | lhs=60 | ... | violated=True`
      2. 自然写法 `反例: n = 15, 差值 = 0.25` —— 标记词 + 同行有具体数值

    ★ 「凭证」的边界是量出来的, 不是拍出来的 ★
    两个极端都试过, 都错:
      · 只认 `COUNTEREXAMPLE:` 标准格式 → **误杀真反例**。
        实测回归: `前 100 内的偶素数: [2]` —— 2 就是「所有素数都是奇数」的反例,
        但它没打标准格式, 于是被降级成 inconclusive。这是"误杀真反例",
        代价大于放过假反例, 不能接受。
      · 放宽到"任何带数字的行都算凭证" → **等于没做**。
        兜底规则会随手抓到**正确**的那一行 —— p(100) 那次
        `via sympy.partition = 190569292` 就是被抓成了"反例", 然后整条漏过去。
    真正的分界线是中间那条: **一行里指名了违反的东西, 并给出了它的值**。
    也就是 `_WITNESS_STRONG`(反例/counterexample/见证值/违反 + 数值) ∪
    `_WITNESS_WEAK`(偶素数/不成立/odd_only/bad_n + 数值), 但**不含**最宽松的
    `_WITNESS_ANY`(任何带数字的行)—— 那一档正是 p(100) 漏网的原因。

    实测两边都拦得住/都放过:
      拦下: p(100)「几路对不上」、R6、R7 的「计数 vs 清单」(都没有指名违反对象)
      放过: 「前 100 内的偶素数: [2]」(指名了"偶素数"这个违反对象 + 值 2)
    """
    if not stdout:
        return None
    for m in _COUNTEREXAMPLE.finditer(stdout):
        fields = dict(_FIELD.findall(m.group("fields")))
        if not fields:
            continue
        if fields.get("violated", "").strip().lower() in ("false", "no", "0"):
            return None
        # 必须带一个**具体数值** —— 全是 none/bad 这种等于没交
        if not any(re.search(r"\d", x) for x in fields.values()):
            continue
        return fields
    # 自然写法: 指名了违反的对象, 且同行给出了值
    for line in stdout.splitlines():
        if (_WITNESS_STRONG.search(line) or _WITNESS_WEAK.search(line)) \
                and re.search(r"\d", line):
            return {"natural": line.strip()[:160]}
    return None


def _refuted_without_credential(stdout: str) -> tuple[str, str] | None:
    """凭证缺失 → REFUTED 不成立。返回非 None = 不成立。"""
    if _counterexample_credential(stdout) is not None:
        return None
    return (
        "inconclusive",
        "你判了 REFUTED, 但没有出示**凭证行**, 机器无法复核这个判定。\n"
        "  必须打这一行(字段名和值可以改, 但要有具体的输入和值):\n"
        '      print(f"COUNTEREXAMPLE: n={bad_n} | lhs={l} | rhs={r} '
        '| violated=True")\n'
        "  为什么这么严: 「找到了一个反例」和「我这几路验算对不上」是两件\n"
        "  完全不同的事 —— 后者只说明**某一路实现有 bug**, 不说明命题为假。\n"
        "  实测教训(2026-10-08 p(100)): sympy 和独立 DP 两路都算出了正确的\n"
        "  190569292, 第三路算出一个物理上不可能的巨大值, 就宣布 REFUTED。\n"
        "  **几路验算对不上, 说明其中某一路有 bug, 不说明命题为假。**\n"
        "  交不出凭证 → 那是 INCONCLUSIVE。想 REFUTED, 就把那一个具体的、\n"
        "  确实违反命题的输入和值算出来打出来。"
    )


def _refuted_contradicts_its_own_output(stdout: str) -> tuple[str, str] | None:
    """REFUTED 侧的兜底门。凭证机制已就位后这六行应当退休。

★ 这是"每翻一次车加一道门"的旧账, 现在合并成一张表 ★
六道门问的是同一个问题(输出和它自己的判定对不对得上), 差别只在措辞。
凭证机制上线后它们应该变成**冗余** —— 留着只为这一轮验证: 如果凭证机制
立住了, 就把这张表和 `_contradiction_msg` 一起删掉。
"""
    if not stdout:
        return None
    for kind, _label in _CONTRADICTIONS:
        msg = _contradiction_msg(kind, stdout)
        if msg is not None:
            return "inconclusive", msg
    return None


def _refutation_targets_wrong_subject(
    code: str, stdout: str, claim: str,
) -> tuple[str, str] | None:
    """
    R3: REFUTED 驳的是不是原命题？**返回非 None = 靶子错了**。

    ★ 真实翻车(2026-10-08 V2 hard-007, R(3,4)=9)★
    命题是双向的: 「证明任意 9 顶点红蓝完全图必有红三角或蓝 K4,
    **并给出** 8 顶点无反例的着色」。
    LLM 用 z3 **正确证出了**上半(R(3,4) ≤ 9, 输出 `Part (1) ... : YES`),
    但下半的 8 顶点着色构造写错了, 它自己的代码抓到构造里有蓝 K4, 于是输出
    `VERDICT: REFUTED` —— 驳的是「我构造错的那个着色」, 不是 R(3,4)=9。

    机器怎么认出它? **输出自相矛盾**: 一行说成立, 一行说不成立。
    真的反例不会长这样 —— 真的反例会让命题的所有部分一起倒。
    自相矛盾说明: 有一部分是成立的, 那么整体就不可能是"被证伪",
    只能是"没证完"。

    触发条件收得很窄:
      1. 命题本身是复合/双向的(否则单向题上会误伤)
      2. 输出里同时有"某部分成立"和"某部分失败"
      3. 判定是 REFUTED
    """
    if not stdout or not _COMPOUND_CLAIM.search(claim or ""):
        return None
    lines = stdout.splitlines()
    has_true = any(_PART_TRUE.search(l) for l in lines)
    has_false = any(_PART_FALSE.search(l) for l in lines)
    if not (has_true and has_false):
        return None
    return (
        "partial",
        "输出**自相矛盾**: 有的行说某一部分成立, 有的行说另一部分不成立, "
        "而你判了 REFUTED。\n"
        "  这道题是**双向/复合命题**, 整体判定必须是各部分的**合取**。"
        "一部分成立说明它没被证伪 —— 驳掉的只是「你自己那个没验过的构造」, "
        "不是原命题。\n"
        "  正确处置: 构造本身验不过 → 换个构造(z3 直接搜一个通常几毫秒) "
        "或把下界改成显式已知构造, 然后重新判定。\n"
        "  两部分都成立 → PROVED; 只成立一部分 → PARTIAL; "
        "真的被证伪要求给出**推翻整个命题**的反例。"
    )


def judge_verdict(sandbox_result: Any, stdout: str | None = None,
                  code: str | None = None, claim: str = "") -> tuple[str, str]:
    """
    由一次执行结果 + 代码里的 VERDICT 标记, 决定这条证据的判定。

    优先级 (从硬到软):
      1. 没跑成 (rejected / timed_out / 异常)      → failed    ← 无条件压过一切
      2. 代码没写 VERDICT 标记                      → untested ← 纪律核心
      3. 代码写了标记且值合法                       → 该值, 但要过机器复核
         3a. REFUTED 要过两道门 (有见证值 / 见证值对不对)
      3b. PROVED  也要过机器复核 (见 `_audit_proved`)

    第 1 条优先级最高: 代码崩了却 print 了 VERDICT(比如把标记写在崩溃前),
    那是无效证据, 不能采信。

    Args:
        sandbox_result: _sandbox.SandboxResult (用鸭子类型, 避免循环依赖)
        stdout: 实际 stdout, 默认取 result.stdout
        code: 验证代码原文 —— 判定要复核代码形态, 拿不到就只做弱检查
        claim: 命题原文 —— 只有它能让机器认出「代码把待证命题当前提用」

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
        if not _witness_looks_refutable(src, out):
            return "inconclusive", (
                f"代码标了 REFUTED 并给出反例 {witness}, 但机器用另一种独立方式"
                f"(穷举/重新验证)复核后**找不到这个反例** —— "
                f"多半是代码本身的遍历顺序或边界处理有问题。\n"
                f"  典型原因: 把有序序列存进 set 后再 `for x in s: if 条件: break`, "
                f"set 的迭代顺序不保证升序/有序, 会提前退出漏掉后面的解。\n"
                f"  请修正代码后重试, 或换一种枚举方式(排序后再遍历)。"
            )
        # 第三层: 反例在不在 / 对不对 / 驳的是不是原命题
        wrong = _refutation_targets_wrong_subject(src, out, claim)
        if wrong is not None:
            return wrong
        # 第四层: 反例是不是「和硬编码的已知值比不上」
        hardcoded = _refuted_by_hardcoded_reference(src, out)
        if hardcoded is not None:
            return hardcoded
        # 第六层: 凭证制 —— REFUTED 必须出示"具体输入 + 违反命题的值"
        no_cred = _refuted_without_credential(out)
        if no_cred is not None:
            return no_cred
        # 第七层(兜底): 输出和它自己的判定对不上
        contradiction = _refuted_contradicts_its_own_output(out)
        if contradiction is not None:
            return contradiction
        return "refuted", f"代码显式标记 {raw}；反例: {witness}"

    # ★ 3c. PROVED 同样要过机器复核 ★
    #
    # 为什么这道门以前不存在: 前两轮我一直盯着 REFUTED, 因为翻车都出在
    # "没有反例却判成假"。但 2026-10-08 V2 实跑 20 题时出了**镜像**的翻车,
    # 而且更严重 —— 一个**根本没有证明**的东西被判成了 PROVED:
    #
    #   题: 「证明 sqrt(2) 是无理数」
    #   attempt 1: 老老实实判 PARTIAL, 输出里明写
    #              "Domain is unbounded; finite enumeration is insufficient"
    #   attempt 2: 加了一句 sympy.is_irrational 就升级成 PROVED, 并打印
    #              "General solution forces q = 0 ... (since sqrt(2) is irrational)"
    #
    # 那是**把待证命题当前提用**(循环论证), 外加只枚举了 |p|,|q| <= 200。
    # 全程代码还 print 了 "VERDICT: PROVED" —— 而 PROVED 此前零机器复核,
    # 于是它一路畅通地进了 v2_results.json, 被当成真结论。
    #
    # 逻辑上的不对称和 REFUTED 那侧是**同向**的:
    #   找到反例 → 能证假;  没找到 → 不能证真, 更不能证假
    #   证不出来 → 只是没证成;  假证成 → 一个错结论流出去当事实
    # 所以宁可降级成 partial/inconclusive, 也不能让没有证明的"真"流下去。
    if verdict == "proved":
        audit = _audit_proved(src, out, claim)
        if audit is not None:
            return audit
        return verdict, f"代码显式标记 {raw}；机器复核通过"

    return verdict, f"代码显式标记 {raw}"


# ============================================================
# PROVED 侧的机器复核 (0 LLM, 0 措辞)
# ============================================================
#
# 设计纪律与 REFUTED 侧一致: **宁可降级, 不可放过**。
#   误杀真证明 → 结论变软成 partial/inconclusive, 损失信息, 可恢复
#   放过假证明 → 一个错误结论流出去当事实, 且下游再没有第二道门能拦
#
# 两条规则都刻意做窄。宽一点就能拦住更多, 但会把"有限范围内的完整证明"
# 一起打死, 那样通过率会好看、结论全是废话 —— 那不是我们要的。

#: 「查了但没查到」的措辞
_NOTHING_FOUND = re.compile(
    r"(?:没有?找到|未找到|没找到|不存在反例|无反例"
    r"|no\s+\w{0,12}\s+found|not\s+found|none\s+found"
    r"|no\s+counter-?example|found\s+no|empty\s+set)",
    re.IGNORECASE,
)

#: 代码/输出自己承认了「只查了有限范围」
_FINITE_BOUND = re.compile(
    r"(?:checked\s+range|search(?:ed)?\s+range|within\s+range|up\s+to"
    r"|范围|区间内|以内|以下|之前|穷举到|枚举到)"
    r"[^\n]{0,30}?\d+"
    r"|\|\s*\w+\s*\|\s*<=\s*\d+"
    r"|range\s*\(\s*\d+\s*\)",
    re.IGNORECASE,
)

#: 有界搜索的代码形态
_FINITE_LOOP = re.compile(r"for\s+\w+\s+in\s+range\s*\(", re.IGNORECASE)

#: ★ 完备判定过程的特征 ★
#: 出现这些, 说明不是"查了没查到", 而是"求解器/内核**证明了不存在**"。
#: 这是 R1 的豁免名单 —— 少了它, 用 z3 正确证出 R(3,4)<=9 的那道会被误杀。
#:   (V2 hard-007 实测: z3 输出 "satisfiable(NOT formula) = False" 是**真证明**,
#:    全域完备, 不能因为同文件里还有个有界枚举就被降级)
#
#: 刻意**不**包含 simplify / solve / factor —— 它们只做符号化简, 不完备。
#: sqrt(2) 那道恰恰用了 solve()+simplify(), 所以它不在豁免名单里, 会被拦。
_COMPLETE_METHOD = (
    "unsat",              # z3: 证明不存在反例
    "Solver",             # z3
    "check()",
    "sat_all",            # pysat
    "decide",             # Lean 内核判定
    "is_true",            # sympy 全称判定
    "qelim",              # sympy 量词消去
    "reduce_inequalities",
    "solveset",
    "nthroot",
)

#: 前提标记 —— 出现在同一行, 且这行含结论词, 就是在拿结论当前提
_PREMISE_MARKER = re.compile(
    r"(?:因为|由于|既然|已知|假定|假设"
    r"|\bsince\b|\bbecause\b|\bgiven\s+that\b|\bas\s+we\s+know\b)",
    re.IGNORECASE,
)


def _conclusion_terms(claim: str) -> set[str]:
    """
    从命题原文里抽出「结论词」—— 也就是待证的那个东西。

    例: 「在 Lean 4 + Mathlib 中形式化并证明 sqrt(2) 是无理数。」
        → 去掉证明动词之前的所有铺垫, 只看**最后一个证明动词之后**的部分
        → 得到 sqrt / 无理数, 不含 "Lean" "Mathlib"(那是工具, 不是结论)。

    ★ 收紧到只取结论子句, 是被真实回放打回来的 ★
    第一版把整句都扫了一遍, 结果单字「数」和数字「2」都被当成了结论词,
    于是一句正常的 Collatz 输出("(因为数...)" 直接被误判成循环论证。
    误杀真证明 = 把通过率做假, 那比不做这道门更糟。

    因此:
      - 只取**最后一个**证明动词之后的后缀(结论子句)
      - 只取长度 >= 3 的 CJK 词 / 长度 >= 4 的英文数字词
      - 纯数字一律不取("2" 是常量, 不是结论)
    """
    if not claim:
        return set()
    # 只看最后一个证明动词之后的结论子句
    parts = re.split(
        r"(?:形式化并证明|形式化证明|证明|验证|verify|prove|show|demonstrate)",
        claim,
        flags=re.IGNORECASE,
    )
    tail = parts[-1] if len(parts) > 1 else claim
    if len(parts) > 1 and not tail.strip():
        tail = claim  # 命题就是"证明X"这种形态时退回全句

    terms: set[str] = set()
    for run in re.findall(r"[一-鿿]{3,}", tail):
        terms.add(run)
    for tok in re.findall(r"[A-Za-z_][A-Za-z_0-9]{3,}", tail):
        terms.add(tok.lower())
    return terms


#: 命题里**自带**有限范围。对这种命题, 有界搜索就是完整证明, R1 必须放行。
#:
#: 真实回归(2026-10-08 V2 hard-019): 命题是「验证不存在正整数 a,b,c < 1000
#: 满足 a^3+b^3=c^3」—— 枚举 1..999 恰好覆盖全称量化的整个定义域, PROVED 是对的。
#: 第一版 R1 不看命题, 一律降级, 就把这种**本来就该判 PROVED** 的题打下去了。
#: 这正是"误杀真证明比放过假证明更糟"的具体形态: 通过率掉下去, 但掉的是对的题。
_CLAIM_SCOPED = re.compile(
    r"(?:<|<=|≤)\s*\d"                 # a,b,c < 1000
    r"|\d+\s*(?:以内|以下|之前|个以内|余)"
    r"|前\s*\d+\s*(?:个|项|条|个素数)"
    r"|对\s*\d+\s*(?:<|<=|≤)"
    r"|up\s+to\s+\d+|first\s+\d+",
    re.IGNORECASE,
)


def _claim_declares_finite_scope(claim: str) -> bool:
    """命题自己就把定义域限死在有限范围内吗? 限死了 → 有界搜索是完整证明。"""
    return bool(claim) and _CLAIM_SCOPED.search(claim) is not None


def _audit_proved(code: str, stdout: str, claim: str) -> tuple[str, str] | None:
    """
    机器复核一条自称 PROVED 的证据。**返回 None = 放行**。

    R1 「有限范围内没查到」被当成全域证明  → partial
       (命题自带有限范围时**不触发**, 那种有界搜索就是完整证明)
    R2 把待证命题当前提用(循环论证)        → inconclusive
    """
    if not stdout:
        return None
    text = f"{code or ''}\n{stdout}"

    # ---- 豁免: 用了完备判定过程 ----
    # 先查 stdout 再查 code: 有界枚举的代码可以和一个完备的 z3 证法共处,
    # 只要它**真的**用了完备判定, 就不能因为有界枚举被误杀。
    if any(h in stdout for h in _COMPLETE_METHOD):
        return None
    if any(h in (code or "") for h in _COMPLETE_METHOD):
        return None

    # ---- R1: 有限枚举冒充全域证明 ----
    # 先问「命题自己限没限定范围」: 限定了的话, 有界搜索本就是完整证明。
    if (not _claim_declares_finite_scope(claim)
            and _NOTHING_FOUND.search(stdout)
            and _FINITE_BOUND.search(text)
            and _FINITE_LOOP.search(code or "")):
        return "partial", (
            "代码自己承认只查了有限范围(\"范围内/checked range ... N\")并以"
            "\"没找到反例\"为依据, 却判了 PROVED —— "
            "**在有限范围内没找到反例, 不能推出全域成立**, "
            "这在 verdict 枚举里就叫 `partial`。\n"
            "  想拿到 PROVED, 只有两条路:\n"
            "    (a) 用完备判定过程, 让机器自己证明不存在性: "
            "z3 判 `unsat` / `Solver().check() == unsat`、"
            "sympy 的 `solveset` / `qelim` / `is_true`、或 Lean 内核的 `decide`。\n"
            "    (b) 如果命题本身就是有范围的, 把范围写进命题本身再判 PROVED, "
            "别用一个全域命题的措辞配一个有界的验证。"
        )

    # ---- R2: 循环论证 ----
    terms = _conclusion_terms(claim)
    if terms:
        for line in stdout.splitlines():
            if not _PREMISE_MARKER.search(line):
                continue
            low = line.lower()
            hit = next(
                (t for t in terms
                 if (t.lower() in low if t.isascii() else t in line)),
                None,
            )
            if hit:
                return "inconclusive", (
                    f"输出里出现「{hit}」被**当前提**使用"
                    f"(同行有前提标记), 而它正是本题要证的结论 —— "
                    f"这是循环论证, 不是证明。\n"
                    f"  原文: {line.strip()[:160]}\n"
                    f"  循环论证的特征: 把待证命题写进推理的前提里。"
                    f"要么证明它(那就得给出不依赖它的推导), "
                    f"要么换成机器可判的完备方法(z3 unsat / decide / solveset)。"
                )
    return None


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
