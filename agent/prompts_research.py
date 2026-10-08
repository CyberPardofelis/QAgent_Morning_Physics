"""
agent.prompts_research — 科研验证闭环的 prompt 模板 (P7 阶段 0)

对标 agent/prompts_survey.py: `XXX_PROMPT` 常量 + `build_xxx_prompt()` 函数。
规则写成**模块级常量**再注入 prompt, 不在各个 prompt 里各写一遍 ——
重复的代价是改一处忘另一处, 两边规则悄悄漂移。

================================ 这一层的核心纪律 =================================
科研闭环与综述最大的不同: **综述的失败是编引用, 科研的失败是"假装算过"。**

LLM 写验证代码时最容易干的三件事, 全部在这里堵:
  1. 写个 `print("验证通过")` 就当证明成立      → VERDICT_CONTRACT 条款 1
  2. 拿一个小 n 的数值抽查说成全域成立           → NUMERICS_ARE_NOT_PROOF 整块
  3. 代码里 import 不存在的东西 / 用错 API       → 明确白名单 + 异常原文会回传

所以 `_provenance.judge_verdict` 的规则是: **代码跑通但没 print VERDICT 标记 → untested**。
prompt 的任务就是让 LLM 稳定地写出那行标记, 并且标记值要和代码里的实际计算对应。

历史:
  - 2026-10-08 P7.0 首版 (建模 + 生成验证代码)
"""
from __future__ import annotations

from typing import Any

# ============================================================
# 共享规则块 —— 单一真相来源, 多个 prompt 注入
# ============================================================

#: VERDICT 标记契约。_provenance.extract_verdict_marker 按 "VERDICT:" 前缀解析,
#: 所以这里的写法是硬约定, 改了要同步改那个函数。
VERDICT_CONTRACT = """【VERDICT 契约 — 必须遵守, 否则结论作废】
你的代码必须在**最后**打印一行判定标记, 机器会解析它决定这条证据的判定:

    VERDICT: PROVED      命题在给定范围内被证明成立(符号计算/精确判定)
    VERDICT: REFUTED     找到了反例
    VERDICT: PARTIAL     有限范围检查通过, 但不构成全域证明(见下面第 0 条)
    VERDICT: INCONCLUSIVE 验了但定不了

0. ★ 先判断这个范围是不是题目自己写的 ★
   PARTIAL 的前提是「你只查了一部分, 但命题要求全域」。
   **如果题目本身就把范围写死了** —— 例如「n=0 到 39」「4 到 1000 的所有偶数」
   「对 n<=20 成立」「n 层汉诺塔, n<=10」—— 那么**穷举完这个范围就是 PROVED**,
   不是 PARTIAL。此时 PARTIAL 的前提根本不成立。

   判据一句话: **题目要求的范围 == 你检查的范围, 且该范围是有限可枚举的 → PROVED**;
   题目要求的是「对所有 n」(没给上界) 而你只查了有限个 → PARTIAL。

   反例(实测真实失败, 别踩):
     题目: 「验证 n^2+n+41 在 n=0 到 39 时全为素数, 并检查 n=40 时的值」
     错误写法: 因为"有限范围"就给 PARTIAL
     正确写法: n=0..39 是题目指定的范围, 逐个验完全对 → 那个子命题 PROVED;
               n=40 的值 1681=41^2 是可直接算出的事实 → 也是 PROVED。
               整个合取命题成立 → PROVED
   同样地: 「对 4..1000 的所有偶数找素数和」验完就是 PROVED, 不是 PARTIAL。

   ★ 合取命题(A 且 B)的判定: 分别判 A 和 B, 两个都 PROVED 合起来才是 PROVED。
     不要因为"A 和 B 都只验了有限范围"就把整体降级 —— 各自范围合法就不是降级理由。

1. **没有 VERDICT 标记 = 判定为 untested**, 哪怕 stdout 里全是 True。
   "代码跑通"不等于"命题成立", 不要用 print 掩盖判定。
2. VERDICT 的值必须由代码里**真实的计算结果**推出, 不许直接写死。
   正例:
       expr = sympy.simplify(原式 - 右边)
       print("化简结果:", expr)
       print("VERDICT: PROVED" if expr == 0 else "VERDICT: REFUTED")
   反例(禁止):
       print("VERDICT: PROVED")     # 这叫编造, 不叫证明
3. 多种标记同时出现时, **以最后一行**为准 —— 探索过程可以先打标记, 但最后要有定论。
4. ★ 别用 `is` 比较 sympy 的对象 ★
   `simplify_logic()` 返回的是 sympy 的 BooleanTrue, **不是** Python 的 True,
   所以 `simp is True` 恒为 False; 同理 `sympy.Or(...) - sympy.Or(...)` 会抛
   TypeError(布尔对象不支持减法)。
   正解: 用 `==` / `bool()` 转换, 或直接用真值表枚举、satisfiable()。
     sat = satisfiable(Not(Equivalent(lhs, rhs)))   # False = 无反例, None = 有反例
   跨类型比较失败是纯 API 误用, 与命题真假无关 —— 但它会让本该 PROVED 的题
   变成 INCONCLUSIVE/REFUTED, 所以写完先检查这一类错误。
5. ★ REFUTED 只有一个含义: 找到了**具体的反例** ★
   REFUTED = 「存在一个具体的 n/输入使命题不成立」。
   它**不**表示「我的实现不够准」或「没验出来」。两者必须分开:

     错误: 二分法迭代 19 次后残差 6e-6 > 1e-6 精度要求 → REFUTED
           (这是"我的迭代不够多", 不是"根不存在"。多迭代几次就到了。)
     正确: 这种情况判 INCONCLUSIVE, 或者**加大迭代次数直到达标**再判 PROVED。

   判据: 你能打印出一个具体的、明确违反命题的值吗?
     能 → REFUTED(把这个值 print 出来)
     不能, 只是没达到精度/没跑完/超时 → INCONCLUSIVE

   同理: 代码报错(TypeError/ImportError)→ 那是 failed 不是 REFUTED。
   一行 import 写错会让整份验证器崩掉, 那是你的 bug 不是命题的结论。
   多 import 一个不存在的东西(如 `from sympy import maximize` —— SymPy 没有
   这个名字)会让本来正确的顶点枚举整体崩掉, 少写就够。
"""

#: 数值 vs 证明 —— 借鉴 prompts_survey.POOL_SAMPLING_RULE 的思路(限定范围话术)
NUMERICS_ARE_NOT_PROOF = """【数值检查 ≠ 证明】
- 随机/浮点/枚举得到的结果**只能**支持 PARTIAL, 不能支持 PROVED。
  例外: 浮点数恰好为 0 **不**代表符号上恒等于 0, 必须转符号计算。
- 报告结论时必须带上范围限定: "在 n=1..1000 范围内未发现反例" ≠ "命题成立"。
  前者是 PARTIAL, 后者需要全域论证。
- 确需全域结论时, 用 sympy 的精确化简 / 求解, 或交给求解器判定(本阶段未接入)。
"""

#: 沙箱约束 —— 必须告诉 LLM 它在什么环境里跑, 否则它会写 import os / 读文件
SANDBOX_RULES = """【运行环境限制】
- 你的代码在**独立子进程的受限沙箱**里跑, 默认超时 20 秒, 超时会整棵进程树被杀。
- **可用模块(白名单)**: math cmath itertools functools fractions decimal statistics
  random json re string operator copy heapq bisect collections typing enum dataclasses
  abc numbers array time datetime io csv textwrap sympy numpy scipy networkx
- **禁止**: import os / sys / subprocess / socket / shutil / ctypes / importlib /
  requests / multiprocessing; 禁止 eval / exec / compile / open / input /
  globals / __builtins__
- 写文件、读文件、联网一律不行。数据直接在代码里构造。
- 异常会原样回传给你, 请根据 traceback 修正后重试 —— 不要用 try/except 吞掉错误。
- 算不动就**缩小规模**(n 从 1000 降到 20)或改用符号方法, 不要死循环硬算。
"""

#: 禁止编造 —— 与综述侧禁止编 paper_id 同一路数
NO_FABRICATION = """【禁止编造】
- 只输出 JSON / Python 代码, 不要解释性前言和"希望对你有帮助"之类的话尾。
- 不要声称引用了不存在的定理、文献或 API。SymPy 的函数名你不确定就别用。
- 不要输出你没算出来的数字。数字只能来自代码实际执行结果。
"""


# ============================================================
# 1. RESEARCH_MODELING_PROMPT — 问题 → 可判定命题 (P2 才用, 先写好备着)
# ============================================================
RESEARCH_MODELING_PROMPT = """你是数学建模助手。任务:把一个研究问题拆成**机器可判定**的原子命题。

【研究问题】
{question}

【什么算"可判定"】
好命题: 有明确的变量、明确的变量域、明确的真假判据, 能写成代码断言。
  - "对所有 n≥1, sum(k=1..n) k^2 = n(n+1)(2n+1)/6"
  - "在 n=1..500 的所有简单连通图中, 平均最短路长 ≤ 4.2"

坏命题(不可判定, 必须标 undecidable):
  - "这个猜想是否有趣" / "最优算法是什么" / "这段证明是优美的"
  - 需要很长创造性构造、无法写成断言的结论
  - 连"多大范围算够"都说不清的开放问题

【第一条纪律: 不得偷换命题 ★★】
claim_text 必须与研究问题**断言同一件事**。不允许替换成"相邻的、已知的、
好验证的"那个命题 —— 哪怕那个命题更漂亮、更容易给出结论。

【★★ 第二条纪律: 不得偷换范围(scope) ★★】
量词对不代表命题对。如果研究问题限定了范围("n>100000" / "在 n=1..500 ��"),
spec 的 scope 必须**覆盖**问题要求的那个范围, 不许缩小到其中一小段。

实测抓到的翻车(第一版没这条防线时):
  问: 存在 n>100000 使得 n!+1 是素数吗
  错: scope 写成 "200 < n <= 500"  → 在这段里确实能找到素数 → 判 proved
  为什么错: 用户问的是 n>100000 那段。n=200~500 有例子, 与 n>100000 是不是
          同一件事完全是两回事。

以下是另一类翻车:
  问: 这段证明是优美的
  错: 拆成 "归纳步 k→k+1 成立"并判为可证明
  为什么错: "优美"没有判定标准, 整条都该进 undecidable。

判断方法: 把 claim_text 和研究问题并排放, 如果一个懂这个领域的人会说
"这不是他问的那个", 那就是偷换。**scope 同样要这么比一遍。**

【★★ 第三条纪律: 量词必须显式化 ★★】
每条命题都要填 quantifier 字段, 取值只能是下面三个之一:
  - "forall"  对所有 n 满足……   (对应"对每个 n")
  - "exists"  存在 n 使得……      (对应"有没有 n")
  - "none"    不存在 n 使得……    (对应"不存在 n")

量词错一个, 结论就整个反过来: "存在 n 使 n²<2n" 是**真**的,
"对所有 n 有 n²<2n" 是**假的**。所以不要靠 claim_text 的中文措辞隐含量词,
必须单独填这个字段。

【★★ 第四条纪律: 不许给原问题加限定 ★★】
除了换范围, 还可以换**数量阈值** —— 那是更隐蔽的一种偷换。

实测抓到的翻车: 「哥德巴赫猜想成立吗」被拆成
  "对所有偶数 n 满足 4<=n<=200000, **存在至少 1000 组**素数对"
→ 代码发现 n=4 只凑得出 1 组 → 判 REFUTED

逻辑上一点没错(n=4 确实凑不出 1000 组), 但哥德巴赫问的是"能写成两素数之和",
也就是**至少 1 组**。"至少 1 组"被换成"至少 1000 组", 拿加强版去反驳原问题,
再宣布"哥德巴赫被驳倒了" —— 这不是验证, 这是换个问法自己玩。

检查一遍: 命题里的每个限定(范围、阈值、附加条件), 问题里有吗?
问题没要求就不要加。

【★★ 第五条纪律: 别造空洞真命题 ★★】
把约束收得极弱, 让命题因为"几乎没有反例"而为真, 这不叫证明。

实测抓到的翻车: 「最小的不存在的素数是多少」被拆成
"不存在 m>=1 使得 n=2^m-1 是素数且 n<2" 并判 proved ——
这条为真只因为 n<2 压根没几个数, 和用户问的"素数"也不是一回事。

要点:
- 命题里如果加了问题**没有要求**的限定条件(取子集、换对象), 就是偷换
- 拿不准命题是否空洞, 就放进 undecidable, 别硬凑一个可判定命题

【拆解规则】
1. 每个命题必须含: claim_text + scope + how + quantifier(四个都不能少)。
2. 优先级: 先拆能被完全证伪的小命题(容易找反例), 再拆需要长论证的。
3. 数量控制在 1-4 条 —— 拆成一堆验不完等于没拆。
4. 不可判定的部分放进 undecidable 数组, **不要硬凑成可判定命题**。

{SANDBOX_RULES}
{NO_FABRICATION}

只输出 JSON:
{{"specs": [{{"claim_text": "...", "scope": "...", "how": "enumerate|symbolic|numeric|solver", "quantifier": "forall|exists|none", "priority": 1}}],
  "undecidable": ["说明为什么这条判不了"]}}"""



#: 量词 → 给代码生成器的说明。**必须讲清 verdict 方向**, 因为
#: "存在" 和 "对所有" 在同一个循环下 verdict 正好相反 —— 这是 P4 实跑
#: 「对所有 n, n²<2n」被判 proved 的直接原因(代码对着存在版在跑)。
_QUANTIFIER_HINTS: dict[str, str] = {
    "forall": "`forall` —— 对**所有** n 成立。找到一个违反的 n → REFUTED; "
               "范围内全过 → 判据取决于范围是题目给的还是你自选的: "
               "题目写死了上界(如「对 n<=20」「n=0..39」)→ 验完就是 PROVED; "
               "题目要求全域但你只查了有限个 → PARTIAL。",
    "exists": "`exists` —— **存在** n 使其成立。找到一个见证值 → PROVED; "
               "没找到 → INCONCLUSIVE(绝不能 REFUTED: 没搜到不等于不存在)。",
    "none": "`none` —— **不存在** n 使其成立。找到见证值 → REFUTED; "
            "整个范围跑完没找到 → PROVED(此时范围就是全域结论)。",
}


def _quantifier_hint(quantifier: str) -> str:
    return _QUANTIFIER_HINTS.get(
        (quantifier or "").strip().lower(),
        "(未指定 —— 请自己从命题文本判断量词, 并据此决定 verdict 方向)",
    )


def build_modeling_prompt(question: str) -> str:
    """构造建模 prompt"""
    return RESEARCH_MODELING_PROMPT.format(
        question=question or "(未指定问题)",
        SANDBOX_RULES=SANDBOX_RULES,
        NO_FABRICATION=NO_FABRICATION,
    )


# ============================================================
# 2. RESEARCH_CODE_PROMPT — 命题 → 验证代码 (P0 主 prompt)
# ============================================================
RESEARCH_CODE_PROMPT = """你是数学验证代码生成器。任务:为一条命题写一段 Python 代码, **用计算证明或反驳它**。

【待验命题】
{claim_text}

【变量域与范围】
{scope}

【建议的验证方法】
{how_hint}

【这条命题的量词(P4 加 —— 不许自己改)】
{quantifier_hint}

★ 如果上面的量词是 `forall`(对所有), 你的代码必须**穷举或符号证明整个范围** ——
  找到一个反例就说明命题是假的, 判 REFUTED; 范围内没找到也只能判 PARTIAL。
★ 如果是 `exists`(存在), 你的代码只需要**找到那一个见证值**就足以判 PROVED ——
  找不到不能说明不存在, 判 INCONCLUSIVE 而不是 REFUTED。
★ 如果是 `none`(不存在), 必须把整个范围跑完确认没有见证值才能判 PROVED。

**量词决定 verdict 的方向**: "存在 n 使 P" 没找到 → inconclusive;
"对所有 n 有 P" 没找到反例 → partial。同一个循环, verdict 正好相反。

【怎么写才算合格的验证代码】
1. 必须**真的调用计算**: sympy 做符号化简/因式/求和, numpy 做数值, networkx 做图枚举。
   空的循环体、只 print 结论的代码, 一律不合格。
2. 命题若含自由变量, 必须在代码里把它们符号化或遍历, 并把范围写进代码。
3. 找到反例时, **把反例本身打印出来**(n 的取值、违反的具体数值), 这是最有价值的证据。
4. 计算量要克制: 默认 20 秒超时。图枚举 n≤8, 枚举循环 n≤10000。

{VERDICT_CONTRACT}
{NUMERICS_ARE_NOT_PROOF}
{SANDBOX_RULES}
{NO_FABRICATION}

只输出 Python 代码, 用 ```python 代码块 ``` 包起来, 不要在代码块外写任何解释。
如果命题本身无法写成可判定断言, 只输出一行注释说明为什么无法判定。"""


def build_code_prompt(
    claim_text: str,
    scope: str = "",
    how_hint: str = "symbolic",
    extra_context: str = "",
    quantifier: str = "",
) -> str:
    """
    构造验证代码生成 prompt。

    Args:
        claim_text: 待验命题(必填)
        scope: 变量域与范围 —— 会直接写进 prompt, 逼 LLM 按这个范围算,
               而不是自己拍一个范围(它最爱擅自扩大范围)
        how_hint: 建议方法 (enumerate / symbolic / numeric / solver)
        extra_context: 上一轮的失败信息(_attempts.jsonl 里的错误原文)。
                       **修正轮必须带上**, 否则 LLM 会重复犯同一个错。
        quantifier: forall / exists / none (P4 加)。决定 verdict 的方向, 见 prompt 正文。
    """
    ctx = f"\n【上一轮失败, 请避免重犯】\n{extra_context}\n" if extra_context else ""
    return RESEARCH_CODE_PROMPT.format(
        claim_text=claim_text or "(未指定命题)",
        scope=scope or "(未指定 —— 请在代码里显式声明你遍历的范围)",
        how_hint=how_hint or "symbolic",
        quantifier_hint=_quantifier_hint(quantifier),
        VERDICT_CONTRACT=VERDICT_CONTRACT,
        NUMERICS_ARE_NOT_PROOF=NUMERICS_ARE_NOT_PROOF,
        SANDBOX_RULES=SANDBOX_RULES,
        NO_FABRICATION=NO_FABRICATION,
    ) + ctx


# ============================================================
# 3. 断言模板库 —— 给 LLM 抄的最小骨架, 显著提高 VERDICT 标记命中率
# ============================================================

#: 这些不是"答案", 是**写法示范**。空 prompt 里 LLM 经常忘了 VERDICT 那行,
#: 给两个具体例子比讲十句规则管用 (2026-10-08 实测同类思路, 对标
#: prompts_survey 里 format_pool 给 LLM 看真实数据而不是抽象描述)。
ASSERTION_TEMPLATES = {
    "symbolic": '''import sympy

x = sympy.symbols('x')
# 把命题左右两边都写成符号式
lhs = <左边>
rhs = <右边>
diff = sympy.simplify(lhs - rhs)
print("化简(lhs - rhs) =", diff)
print("VERDICT: PROVED" if diff == 0 else "VERDICT: REFUTED")
''',
    "enumerate": '''# 小范围枚举找反例
# ★ 先问一句: 题目自己写死了范围吗(如「n<=20」「n=0..39」「4..1000 的偶数」)?
#   写死了 → 下面的循环范围就是全域, else 分支要打 PROVED 而不是 PARTIAL。
#   没写死 → 你自选的 1..200 只是抽查, PARTIAL 才是对的。
found = None
for n in range(1, 201):
    if <该 n 下命题不成立>:
        found = (n, <具体违反的数值>)
        break
if found:
    print("找到反例:", found)
    print("VERDICT: REFUTED")
else:
    print("n=1..200 范围内未发现反例")
    # ★ 范围是题目给的 → PROVED; 范围是你自选的抽查 → PARTIAL
    print("VERDICT: PARTIAL")
''',
    "numeric": '''# 数值抽查: 只能支持 PARTIAL, 不能支持 PROVED
import numpy as np
mismatch = []
for t in np.linspace(0.1, 10.0, 200):
    if abs(<左边> - <右边>) > 1e-9:
        mismatch.append((t, <左边>, <右边>))
if mismatch:
    print("数值失配(可能是反例, 也可能是浮点误差):", mismatch[:3])
    print("VERDICT: REFUTED")
else:
    print("200 个采样点均一致 —— 仅数值抽查")
    print("VERDICT: PARTIAL")
''',
    "graph": '''import networkx as nx

bad = []
for n in range(2, 7):
    for G in nx.nonisomorphic_trees(n):          # 逐个非同构图
        if not <该图上命题成立>:
            bad.append((n, G.number_of_nodes()))
            break
    if bad:
        break
if bad:
    print("找到反例图:", bad[0])
    print("VERDICT: REFUTED")
else:
    print("小 n 范围内未发现反例 —— 有限检查")
    print("VERDICT: PARTIAL")
''',
}


def template_for(how_hint: str) -> str:
    """按验证方法取写法示范; 没有对应模板返回空串"""
    return ASSERTION_TEMPLATES.get((how_hint or "").strip().lower(), "")


def format_attempts(attempts: list[dict[str, Any]], limit: int = 3) -> str:
    """
    把 _attempts.jsonl 里的失败记录格式化成喂回 LLM 的文本。

    只取最近 limit 条 —— 塞太多错误原文会让 LLM 顾此失彼,
    而且反复犯同一个错时, 近期的比早期的更有指导意义。
    """
    if not attempts:
        return ""
    recent = attempts[-limit:]
    lines = []
    for i, a in enumerate(recent, 1):
        err = (a.get("error_type") or "?")
        tb = (a.get("stderr") or a.get("reject_reason") or "").strip()
        # traceback 太长就掐头留尾 (traceback 的最后一行最关键)
        if len(tb) > 1200:
            tb = tb[:400] + "\n... [掐断] ...\n" + tb[-800:]
        lines.append(f"第 {i} 次尝试: 失败类型={err}\n{tb}")
    return "\n\n".join(lines)
