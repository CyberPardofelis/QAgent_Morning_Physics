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

import re
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
5. ★ REFUTED 只有一个含义: 找到了**具体的反例**, 而且必须把它打印出来 ★
   REFUTED = 「存在一个具体的 n/输入使命题不成立」。
   它**不**表示「我的实现不够准」或「没验出来」。两者必须分开:

     错误: 求根迭代若干次后残差仍大于题目要求的精度 → 直接判 REFUTED
           (这是"我的迭代不够多", 不是"根不存在"。多迭代几次就到了。)
     正确: 这种情况判 INCONCLUSIVE, 或者**加大迭代次数直到达标**再判 PROVED。

   ★ REFUTED 必须出示**凭证行**, 没有它判定会被否决 ★
   机器不看你的措辞, 只看你有没有交出一个**具体的输入 + 在它上面违反命题的值**。
   「找到了一个反例」和「我这几路验算对不上」是两件完全不同的事 ——
   后者只说明某一路实现有 bug, 不说明命题为假。
   格式(照抄, 改字段名和值即可):

     print(f"COUNTEREXAMPLE: n={bad_n} | lhs={l} | rhs={r} | violated=True")
     print("VERDICT: REFUTED")

5. ★ 构造/答案必须用**制品行**交出来, 只有 VERDICT 不算数 ★
   机器会用**独立于你的第二套实现**去复核这个制品(它不读你的代码, 也不读
   你的自由文本), 所以你必须把"要交付的那个东西本身"用 JSON 打印出来:

     # 构造类(着色/矩阵/序列/巡回/群…):
     print("CONSTRUCT: " + json.dumps({"kind": "...", ...}))
     # 数值/闭式类(多项式/系数/序列/关系向量…):
     print("ANSWER: " + json.dumps({...}))
     # 反例类:
     print("COUNTEREXAMPLE: " + json.dumps({"n": 18, "value": 343}))

   为什么必须是 JSON 而不是散文: 独立复核器只解析这一行。它**不看**你的
   stdout 措辞, 所以「我构造了一个满足条件的 28 阶矩阵」这种话它读不出来,
   只会判你"没交制品"。

   要点:
   - 用 `import json` 后 `json.dumps(...)`, 不要手拼字符串(引号/逗号极易出错)
   - 值必须是**能被别人重新验证的东西**: 完整的矩阵/序列, 不是"见上面输出"
   - 数据量大也要直接打在 stdout 里 —— 沙箱禁写文件

   光打印"发现反例"而没有具体数值不行 —— 机器只认带数字的那行。
   判据一句话: 你能打印出一个**具体的、明确违反命题的值**吗?
     能 → 打印凭证行, 然后 REFUTED
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
- 确需全域结论时, 用 sympy 的精确化简 / 求解, 或**交给求解器判定(z3 已接入, 在白名单里)**。
  `s.check() == unsat` 就是"不存在反例"的**全域**证明, 可以支撑 PROVED;
  而 Python 暴力枚举在无界定义域上永远只能支撑 PARTIAL。这两者的区别不是风格问题。
- ★ 判断"该用什么方法"的一条硬规则 ★
  先看**定义域**:
    - 定义域**有限且小**(≤ 10^5 量级) → 直接穷举, 但要确认题目是否限定了范围,
      限定了就 PROVED, 没限定就 PARTIAL。
    - 定义域是**结构化的**(图、着色、整数方程、SAT) → 上 z3。它是完备的。
    - 命题**无界**且要证"不存在" → 只能靠符号恒等式或构造性下降论证(Lean/sympy),
      枚举做不到。
"""

#: 复合命题 —— 2026-10-08 V2 hard-007 实测翻车后加
#:
#: ★ 这里原来直接写着「验证 R(3,4)=9: ...」和「R(3,4)=9 是真的」 ★
#: 那是 V2 hard-007 的**原题和答案**。这段文本被**无条件注入每一条 prompt**,
#: 于是 R 套的 res-001(问的正是 R(3,4))**连题目带答案一起发给了模型**。
#: 一个靠"模型不知道答案"成立的基准, 这样就作废了 —— 已实测确认
#: `build_code_prompt` 对任意题面都会把这两句带出来。
#:
#: 教训和记忆里那条一致: **提示里不能出现具体题目的答案**。
#: 讲"这类题该怎么判"是合法的教学; 讲"那道题的答案是 X"是泄题。
COMPOUND_CLAIM = """【复合命题(双向题)的判定】
- 有些命题天然是**合取**: 形如「证明方向 A, **并**给出方向 B 的见证/构造/解」。
  这类命题必须拆成 A、B 两部分**分别**验。
- 顶层 `VERDICT` 必须是各部分的**合取**, 不能拿其中一部分的失败去驳整道题:
    A、B 都成立            → PROVED
    某一方向被证伪         → REFUTED (打印**反例本身**, 不是"我构造错了")
    一部分成功一部分没验出来 → PARTIAL
- ★ 最容易犯的错 ★ 一个方向证出来了, 另一个方向的**构造写错了**,
  代码抓到自己的构造里有不该有的子结构, 于是输出 `VERDICT: REFUTED`。
  这是**假 REFUTED** —— 被驳的是"我构造错的那个东西", 不是原命题。
  抓到自己的构造有问题, 正确的处置是**回头把构造改对**(或者换个构造再试),
  而不是宣布命题为假。
- 自检: 打印每个部分的独立判定, 并让顶层 verdict 由它们**算出来**, 别手写。
"""

#: 沙箱约束 —— 必须告诉 LLM 它在什么环境里跑, 否则它会写 import os / 读文件
#:
#: ★ 白名单部分**运行时从沙箱代码里取**, 不再手抄 ★
#: 2026-10-08 实测事故: 这段文本是 P0 时期手写的, 后来 P7.1 往白名单里加了
#: `z3`(P7.1 加: z3 的 Python API), **这里没跟着改**。于是:
#:   - LLM 以为 z3 不可用, 有完备判定能力时改用 Python 硬算 → 超时/判不出
#:   - LLM 想用 mpmath 做高精度(BBP 那道题), 白名单里也没有 → attempt 1
#:     被沙箱静态门拒绝, 白白烧掉一轮
#: 一份和真实配置漂移的白名单说明, 比没有说明更糟: 它让模型**基于错误前提
#: 做选型**, 而选型失败是这个系统最大的失败模式。
#: 硬编码在 prompt 里的白名单迟早还会漂第二次, 所以改成从源头取。
def _sandbox_rules() -> str:
    try:
        # 延迟导入: prompts_research 被 research_verify 导入,
        # 模块级再 import research 会成环。
        from agent.tools.research._sandbox import _ALLOWED_IMPORT_ROOTS

        mods = " ".join(sorted(_ALLOWED_IMPORT_ROOTS))
    except Exception:                                   # pragma: no cover
        mods = "(沙箱模块不可用 —— 请假设只有标准库和 sympy/numpy/scipy/networkx/z3)"
    return SANDBOX_RULES_TEMPLATE.format(allowed_modules=mods)


SANDBOX_RULES_TEMPLATE = """【运行环境限制】
- 你的代码在**独立子进程的受限沙箱**里跑, 默认超时 20 秒, 超时会整棵进程树被杀。
- **可用模块(白名单, 以下就是全部)**:
  {allowed_modules}
- **禁止**: import os / sys / subprocess / socket / shutil / ctypes / importlib /
  requests / multiprocessing; 禁止 eval / exec / compile / open / input /
  globals / __builtins__
- ★ **`.eval` 属性也被禁**(任意对象的), 例如 z3 的 `Solver().eval(...)` 一律过不了
  静态门。要让求解器给模型取值, 用 `s.add(...)` 声明约束后 `s.check()`, 再从
  `s.model()` 取 —— 不要调 `.eval`。
- ★ **没有 mpmath**。需要任意精度十进制运算时用标准库 `decimal`(设好 context 精度),
  它完全够用且在白名单里。
- ★ **z3 在白名单里, 是完备判定器**, 但**别无脑往上套** ★
  用对的地方: 图着色 / SAT / 组合结构 / 整数方程的「存在性」与「全域否定」。
    - 存在性/构造: `Solver()` → `s.add(约束)` → `s.check()` → `s.model()` 取解
    - 全域否定: `s.add(Not(命题))` 后 `check() == unsat` 就是「不存在反例」的全域证明
  **用错的地方(实测踩过)**: 大整数/级数/数值计算题。
    Catalan 那种题我用 z3 求解反而绕远, 直接 `math.comb` / 递推 /
    `math.factorial` 又快又准。**先判断定义域是什么类型, 再选工具** ——
    符号代数问 sympy, 组合搜索问 z3, 大整数问 math/decimal。
- ⚠️ **`check()` 是三态**: `sat` / `unsat` / **`unknown`**。
  设了 `timeout` 就会有 `unknown`。**把 unknown 当 unsat 是错的** ——
  unknown 的意思是"我算不动", 结论必须降级成 `VERDICT: INCONCLUSIVE`。
- ★ **绝不拿"我记得的已知答案"当反例或判据 ★
  真实翻车(2026-10-08): 代码用三种方法都算出了同一个值(三种方法互相验证, True),
  然后又去和一个硬编码的"已知参考值"比较, 不等, 于是输出 `VERDICT: REFUTED`。
  **那个参考值是模型自己编的**, 而算出来的值是对的 —— 结果它证伪了一条真命题。
  规矩:
    - 反例必须是**代码算出来的具体输入**, 不是你记忆里的常量。
    - 你记忆里的"已知答案"**只配当调试提示**, 绝不能进判定分支。
      你的记忆会错, 而且错得很自信 —— 编程验证器也会崩。
    - 多种独立方法**互相印证**的结果, 比任何"参考值"都可信。
- ★ 代码必须在 import 之后声明**策略标签** ★

      # STRATEGY: solver

  可选标签: formal / construct / solver / symbolic / numeric / enumerate
    formal    形式化证明(Lean)
    construct 构造出一个对象, 再验证它
    solver    SAT/SMT 完备判定(z3)
    symbolic  符号化简 / 精确求解
    numeric   数值 / 高精度
    enumerate 有界枚举

  为什么要有这一行: **它不是提示, 是标签。** 上一轮若是规模问题(超时),
  机器会检查你有没有换标签 —— 换个写法但实质还是同一条路, 照样被挡回来。
  它逼你在动手前先想"我这次用的是什么计算形态", 而不是写完再发现又超时。

  ★ 我不会告诉你该选哪个 ★ 一律先想清楚再写这一行。

- 写文件、读文件、联网一律不行。数据直接在代码里构造。
- ★ **递归爆栈了不要用 `sys`**。`sys` 不在白名单, 静态门直接拒
  (实测踩过: 写了个递归回溯, 994 层爆栈, 然后想 `sys.setrecursionlimit(...)` 补救,
  结果整份代码被门拒掉, 白白烧掉一轮)。
  补救办法按优先级:
    1. **改成迭代**(显式栈 / 生成器), 通常本来就更清晰;
    2. 记忆化: `functools.lru_cache`(在白名单里)—— 递归深度通常是被重复子问题撑爆的;
    3. 剪枝: 提前排除不可能的分支, 深度自然降下来。
- ★ **`from` 导入的每个名字都要真的 import 过 ★
  实测踩过: `sympy` 用来着但忘了 `import sympy`, 崩在最后一行,
  而**正确答案早就算出来了**, 就差最后那个 print VERDICT。
  结论白丢。写完先扫一遍: 每个用到的模块名都有对应的 import 吗?
- ★ `sympy.prime` / `prime(n)` 的参数从 **1** 开始 ★
  `prime(0)` 会抛 `ValueError: nth must be a positive integer`。
  这行会被静态门**在执行前**拒掉 —— 实测踩过, 白白烧掉一轮。
  Python 3.14 默认就是 postponed annotation, 你根本不需要这行。
- ★ **z3 取值不要用 `model.eval(...)`**(`.eval` 是全局禁用的属性名)。
  正确写法: `m = s.model()` 然后 `v = m[var]`, 配合 `z3.is_true(v)` 判断真假。
- 异常会原样回传给你, 请根据 traceback 修正后重试 —— 不要用 try/except 吞掉错误。
- 算不动就**缩小规模**(n 从 1000 降到 20)或换用符号/求解器方法, 不要死循环硬算。
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
        SANDBOX_RULES=_sandbox_rules(),
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
5. ★ **算出来是空集时, 先怀疑自己的代码 ★
   「没找到解」「所有解: []」这类输出, 九成是实现错了, 不是命题为空。
   实测两次踩坑: 一次 Pell 方程算出「所有正整数解: []」, 一次 Erdős–Straus
   对显然有解的小 n 判成「无解」。两次都是实现问题, 不是命题问题。
   判 INCONCLUSIVE 之前, 至少做一次**独立 sanity check**:
     · 拿一个你知道答案的小规模实例试你的代码, 看它能不能跑出来;
     · 换个算法路线再算一遍;
     · 检查边界/取模/精度这类最常见的实现错。
   你是会犯错的 —— 编程验证器也会崩。空结果不等于命题为空。

{VERDICT_CONTRACT}
{NUMERICS_ARE_NOT_PROOF}
{COMPOUND_CLAIM}
{SANDBOX_RULES}
{NO_FABRICATION}

只输出 Python 代码, 用 ```python 代码块 ``` 包起来, 不要在代码块外写任何解释。
如果命题本身无法写成可判定断言, 只输出一行注释说明为什么无法判定。"""


def answer_spec_block(spec: str) -> str:
    """把「本题要求交出的东西的结构」变成 prompt 里的一段。

    ★ 为什么这不是"给提示" ★
    提示 = 告诉它**答案是什么**(用连分数 / 该找 n=18 / 构造是 Paley q=13)。
    这里给的是**输出格式**: 「交一个 8x8 的 0/1 矩阵, 外加一份无红三角无蓝K4
    的验证输出」。知道要交矩阵, 不等于知道矩阵长什么样 —— 前者是接口约定,
    后者才是解题。真实研究里, 交什么格式也是题目的一部分。

    ★ 不给会怎样(实测 2026-10-08 R 套 20 题)★
    14 道判"错"里有 11 道是 artifact_present: False —— 不是它做错,
    是它交了 CONSTRUCT: {"quadruple": [...]} 而独立验证器在找 relation,
    **格式对不上, 两边都以为对方有问题**。这种假阴性比假阳性更难查:
    它长得一模一样地像"模型做错了"。
    """
    if not spec or not spec.strip():
        return ""
    return (
        "\n【本题要求你交出的东西 — 输出格式】\n"
        + spec.strip()
        + "\n  ★ 这是**输出格式约定**, 不是解题提示 —— 它不告诉你答案是什么,"
          "只告诉你答案要装进什么样的字段里。\n"
        "  制品行的 JSON 必须**包含这些字段**(字段名照抄):\n"
        "      print(\"ANSWER: \" + json.dumps({...包含上述字段...}))\n"
        "  构造类用 CONSTRUCT, 反例类用 COUNTEREXAMPLE。"
    )


def build_code_prompt(
    claim_text: str,
    scope: str = "",
    how_hint: str = "symbolic",
    extra_context: str = "",
    quantifier: str = "",
    answer_spec: str = "",
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
        answer_spec: 本题要求交出的东西的结构(JSON 字段约定)。**不是解题提示** ——
                    它只说答案要装进哪些字段, 不说答案是什么。缺了它会出现
                    极端难查的假阴性: agent 交了制品但字段名与独立验证器对不上。
    """
    ctx = f"\n【上一轮失败, 请避免重犯】\n{extra_context}\n" if extra_context else ""
    return RESEARCH_CODE_PROMPT.format(
        claim_text=claim_text or "(未指定命题)",
        scope=scope or "(未指定 —— 请在代码里显式声明你遍历的范围)",
        how_hint=how_hint or "symbolic",
        quantifier_hint=_quantifier_hint(quantifier),
        VERDICT_CONTRACT=VERDICT_CONTRACT,
        NUMERICS_ARE_NOT_PROOF=NUMERICS_ARE_NOT_PROOF,
        COMPOUND_CLAIM=COMPOUND_CLAIM,
        SANDBOX_RULES=_sandbox_rules(),
        NO_FABRICATION=NO_FABRICATION,
    ) + ctx + answer_spec_block(answer_spec)


# ============================================================
# 3b. LEAN_CODE_PROMPT — 形式化命题 → Lean 证明 (P7.3)
# ============================================================
# 为什么要**硬路由**而不是让 LLM 自己选:
#   2026-10-08 V2 实测, hard-015/016 明确写着「在 Lean 4 + Mathlib 中形式化并证明」,
#   但整条链路只会生成 Python —— LLM 于是用 sympy 硬凑, 结果
#   hard-016 输出了一个「有界枚举 + 循环论证」的假 PROVED。
#   根因不是模型不会 Lean, 是**它没有 Lean 这条路可选**。
#   这跟 research_sympy / research_z3 / research_enumerate 从不被调用是同一个病:
#   能力装上了不等于用得上。缺的是路由。
LEAN_CODE_PROMPT = """你是 Lean 4 形式化证明生成器。任务:写一段 **Lean 4 代码**, 形式化并证明下面这条命题。

【待证命题】
{claim_text}

【范围/前提】
{scope}

【环境】
- Lean 4.9.0 + Mathlib **v4.9.0**(两者版本严格配对, 换版本会编不过)。
- 要用 Mathlib 就写 `import Mathlib`。
- 下面是**在这个环境里实测编译通过**的两段, 照这个风格写:

```lean
import Mathlib
-- sqrt 2 是无理数
example : Irrational (Real.sqrt 2) := irrational_sqrt_two
#print axioms irrational_sqrt_two
```

```lean
import Mathlib
-- 费马小定理: p 为素数且 p ∤ a  =>  a^(p-1) ≡ 1 [MOD p]
example (p : Nat) (hp : p.Prime) (a : Nat) (h : ¬ (p ∣ a)) :
    a ^ (p - 1) ≡ 1 [MOD p] := by
  have h' := Nat.ModEq.pow_totient (x := a) (n := p) (hp.coprime_iff_not_dvd.mpr h).symm
  simpa [Nat.totient_prime hp] using h'
```

  第二段的要点: v4.9.0 里**没有** `Nat.Submod.pow_right`(新版才有),
  得走费马-欧拉 `Nat.ModEq.pow_totient` 再特化 φ p = p - 1。
  定理名不确定**不要猜**: 先写个错的名字, 编译器报的 `unknown constant`
  会告诉你该去哪个文件找。这比凭空猜一个名字然后自说自话可靠得多。

【判定规则 —— 机器会独立复核, 这不是形式】
1. **禁止 `sorry`**。Lean 遇 `sorry` 只报 warning、**退出码仍然是 0**,
   但那等于没证。机器会扫 `declaration uses 'sorry'` 并判不通过。
2. **禁止 `axiom` 声明**。`axiom foo : P` + `theorem bar : P := foo` 编译通过、
   退出码 0、零 warning, `#print axioms` 甚至说「不依赖任何公理」—— 因为它自己
   就是公理。这等于把待证命题当前提, 机器会在**源码**里扫 axiom 声明判不通过。
3. 末尾加 `#print axioms <你的定理名>`。机器会看输出里有没有 `sorryAx`;
   只有 `propext` / `Classical.choice` / `Quot.sound` 是正常的。
4. 只输出能编译的代码, 代码块外不要写任何解释。
5. ★ 证不出来就**明说**: 写一行注释说明卡在哪。
   **不要用 sorry 或 axiom 假装证完了** —— 机器会拦, 而且假成功比诚实失败糟得多。

只输出 ```lean 代码块```。
如果命题无法形式化, 只输出一行注释说明原因。"""

#: 判一条命题是不是「必须用 Lean」。命中就走形式化后端, 不生成 Python。
_LEAN_TRIGGER = re.compile(
    r"(?:lean\s*4|lean4|mathlib|形式化|形式证明|formaliz)", re.IGNORECASE)


def needs_lean(claim_text: str) -> bool:
    """
    这条命题是不是必须走 Lean。

    触发词只取**明确无歧义**的: 判错的方向是「该走 Lean 却走了 Python」
    (白白浪费一轮, 就是 hard-016 那个假 PROVED 的由来)。
    「用 z3 验证」这种指向 Python 求解器的措辞不会命中。
    """
    return _LEAN_TRIGGER.search(claim_text or "") is not None


def build_lean_prompt(claim_text: str, scope: str = "",
                      extra_context: str = "") -> str:
    """构造 Lean 证明生成 prompt。"""
    ctx = f"\n【上一轮失败, 请避免重犯】\n{extra_context}\n" if extra_context else ""
    return LEAN_CODE_PROMPT.format(
        claim_text=claim_text or "(未指定命题)",
        scope=scope or "(无额外范围限制)",
    ) + ctx


# ============================================================
# 3c. 方法选型规划 —— 让「选型」变成机器算出来的东西
# ============================================================
# 为什么要单独做这一步:
#
# 这套 V2 题是**按「裸枚举必死」设计的** —— 规模 1e6 / 1e7 / K9 / 100 皇后,
# 精度上大整数 / Pell / Catalan / 三种立方和(浮点必翻车),
# 证明上 Lean 形式化 / R(3,4) 双向 / 最小性验证(不能只给答案)。
#
# 实测里 LLM 栽的跟头**几乎全是选型, 不是算力**:
#   - 100 皇后   → 写 Python 回溯, 裸枚举在 20s 里必死 → failed
#   - Petersen 色数 → 没意识到 2^10=1024 种 2-着色可以直接穷举 → inconclusive
#   - Erdős–Straus → x/y/z 三重枚举, 顺序不对直接超时(正确的做法是 x、y 有界, z 解出)
#   - sqrt2       → 以为 Python 有界枚举能证「不存在」
#
# 所以关键不是「提醒它别硬算」(提醒过, 会忘), 而是**把规模算出来给它看**,
# 让它自己判断裸枚举来不来得及。而且这个判断必须由机器做 ——
# 人对着「1e7」能直觉感到大, 对着「100 顶点图的所有 2-着色」反而容易低估。
def _extract_scales(claim: str) -> list[str]:
    """从命题里抽出所有规模标记, 原样回给 LLM 看。"""
    if not claim:
        return []
    out: list[str] = []
    out += re.findall(r"\d+\s*[eE]\s*\d+", claim)          # 1e6
    out += re.findall(r"\d+\s*\^\s*\d+", claim)              # 10^1000
    out += re.findall(r"\d+\s*\*\*\s*\d+", claim)            # 10**6
    out += re.findall(r"\bK_?\d+", claim)                     # K9 / K_4
    out += re.findall(r"\d[\d,]*\s*(?:个顶点|皇后|顶点|以内|以下|万元|次)",
                      claim)
    out += re.findall(r"\b\d{6,}\b", claim)                   # 裸的 1000000
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:12]


#: 组合/结构类问题 —— 裸枚举代价最容易被低估的一类
_COMBINATORIAL_RE = re.compile(
    r"(?:皇后|Ramsey|ramsey|R\(|着色|色数|图论|连通|SAT|求解器|汉诺塔|魔方"
    r"|排列|组合|子集|匹配|独立集)",
    re.IGNORECASE,
)
#: 存在性 —— 只需一个见证值, 但见证可能极难找
_WITNESS_RE = re.compile(
    r"(?:存在|求一个|找出一个|构造|一个解|一个见证|解\s*一个)"
)
#: 全域/证明要求
_PROOF_RE = re.compile(r"(?:证明|证|最小|双向|最小性|判定)")


#: 求某个数的第 N 位数字 —— 大数运算的高发翻车区
_DIGIT_EXTRACT = re.compile(
    r"(?:第\s*[\d^e+*\-]+\s*位|位数字|第\s*\d+\s*位数字|"
    r"\d+\s*位(?:的)?(?:数字|十六进制|二进制)|"
    r"hex\s+dig|base-?\d+\s+dig)",
    re.IGNORECASE,
)


#: 丢番图方程 / 求最小解 —— 同样存在一条固定的高效路线
_DIOPHANTINE = re.compile(
    r"(?:Pell| pell|丢番图|Diophantine|diophantine"
    r"|最小正整数解|最小解|x\^?\s*2\s*-\s*\d+\s*\*?\s*y\^?\s*2"
    r"|x\s*2\s*-\s*\d+\s*y\s*2\s*=)",
    re.IGNORECASE,
)


#: 命题**自带范围 + 容差** —— 判据就是它自己写出来的那两个数。
#:
#: 这类命题和"全域命题"是两件完全不同的事, 必须分开处理:
#:   全域:  "证明 sqrt2 是无理数"      —— 有界验证再多次也只是 PARTIAL
#:   自带:  "取 N=10^7, 误差小于 1e-6" —— 跑完这个 N 就是 PROVED
#: 混起来的后果是**双向的**: 全域题被硬吹成 PROVED(假 PROVED, 更严重),
#: 自带范围的题被无谓降级成 PARTIAL(做对了却记成没做对)。
_TOLERANCE = r"(?:误差|精度|绝对误差|相对误差|tolerance|精度要求)"
_SCOPED_TOLERANCE = re.compile(
    r"(?:取\s*N\s*=|N\s*=\s*[\d^e+]+|n\s*<=\s*\d|n\s*=\s*\d+)"
    r"[^\n]{0,40}?" + _TOLERANCE
    + r"|" + _TOLERANCE + r"[^\n]{0,40}?(?:N\s*=\s*[\d^e+]+|1e[-+]?\d+|10\^?[-{]?\d*)",
    re.IGNORECASE,
)


def method_catalog_block(claim: str) -> str:
    """注入**可查的方法目录**, 而不是按题型预设的技法提示。

    ★ 为什么不再用按题型的技法提示 ★
    2026-10-09 实测: 那类提示有两个硬伤 ——
      1. **泄题**: Pell 那条提示里写着 D=61 的答案 (1766319049, 226153980),
         而 V2 hard-001 问的就是 Pell-61 —— 那道题的答案就写在它自己的
         prompt 里, 而且**无条件注入每一条**(已实测确认)。已删除。
      2. **覆盖不了**: 它是按我见过的题型硬编码的, 遇到没预设过的题型零帮助;
         而且每加一个题型提示, 门数就 +1 —— 那正是本轮的教训。

    ★ 目录和提示的关键区别 ★
      · 提示说"这类题该用 X"; 目录说"有哪些方法、各自什么时候**不该**用"
      · 排除权交给模型自己 —— 那才是"认题"; 直接给路线是替它认题
      · 目录是**通用方法学**, 不需要为每个新题型增补

    ★ 不塞具体数字 ★
    规模数字进 prompt 等于泄题(它本身就是答案的一部分)。这里只用它提醒
    "自己算一遍来不来得及", 不给数字。
    """
    from agent.tools.research._methods import classify, methods_for

    try:
        info = classify(claim)
    except Exception:
        return ""                 # 认题失败就当没认出来, 绝不让它变成噪声
    shapes = info.get("shapes") or []
    ms = methods_for(shapes) if shapes else []
    if not ms:
        return ""
    lines = [
        "\n【可用方法目录 —— 工具侧提供的通用方法学, 不是本题的解法】",
        f"  机器从题面认出的形状标签: {', '.join(shapes)}",
        f"  {info['note']}",
        "  ★ 下面是**目录**, 不是推荐。用之前先读每条的\"不适用\"那一栏,",
        "    把不适用的**排除掉**。选一条你排除不掉的路, 并在代码里用",
        "    `# STRATEGY: <名字>` 声明它。选错也没关系: 方法类失败会被拦下来,",
        "    那时你必须换一条。",
    ]
    for m in ms:
        lines.append(f"\n  · {m['name']}  (工具: {m['tool']})")
        lines.append(f"      适用: {m['for']}")
        lines.append(f"      不适用: {m['not_for']}")
        if m.get("how"):
            lines.append(f"      怎么做: {m['how']}")
        if m.get("watch"):
            lines.append(f"      ★ 注意: {m['watch']}")
    return "\n".join(lines)


def plan_method(claim: str, how: str = "symbolic") -> dict[str, Any]:
    """
    机器做一次方法选型预判, 返回要注入 prompt 的说明与建议的 how。

    Returns:
        {"note": str, "how": str, "scales": [...], "must_use_complete_method": bool}
    """
    scales = _extract_scales(claim)
    combinatorial = bool(_COMBINATORIAL_RE.search(claim or ""))
    witness = bool(_WITNESS_RE.search(claim or ""))
    proof = bool(_PROOF_RE.search(claim or ""))
    must_complete = (combinatorial and (witness or proof)) or (
        bool(scales) and combinatorial)

    parts: list[str] = []
    if scales:
        parts.append(
            f"【机器已从命题里读到的规模标记】{'、'.join(scales)}\n"
            f"  → 先**自己算一遍**搜索空间有多大: 定义域里每个元素都试一遍的话,"
            f"这个规模能在 20 秒超时内跑完吗? 算不出来就换方法, 别硬写循环。\n"
            f"  ★ 如果最后确实只能有界枚举, 你必须在注释里**写出枚举的确切上界**,"
            f"并说明它为什么覆盖了命题的全部定义域。没写清楚这一条, 机器会判你 PARTIAL。"
        )
    if combinatorial:
        note = (
            "【机器已判定: 这是结构化 / 组合类问题】\n"
            "  裸枚举是最容易低估代价的写法。这类问题**默认走 z3**(完备判定, 在白名单里):\n"
            "    - 存在性/构造题: `Solver()` → `s.add(约束)` → `s.check()` → `s.model()` 取解\n"
            "    - 全域否定题: `s.add(Not(命题))` 后 `check() == unsat` "
            "**就是**「不存在反例」的全域证明, 可以支撑 PROVED\n"
            "  只有当搜索空间小到你能**说得出确切数字**(比如 10 顶点 2-着色 = 2^10 = 1024 种)"
            "时, 才用穷举, 并在注释里写出这个数字来证明穷举是完备的。\n"
            "  ⚠️ 禁止调用 `.eval` 属性 —— 沙箱静态门直接拒。要取值走 `s.model()`。"
        )
        if witness and must_complete:
            note += (
                "\n  本题是**存在性**的: 你需要的是一个见证值。"
                "纯 Python 暴力搜在上面的规模下大概率超时, 求解器才是这条路。"
            )
        if proof:
            note += (
                "\n  本题还要求**证明**, 不只是给答案: 把结论写成求解器能判定的形式, "
                "别只跑出一个解就收工。"
            )
        parts.append(note)
    if _DIGIT_EXTRACT.search(claim or ""):
        parts.append(
            "【机器已识别: 这是「求第 N 位数字」类问题】\n"
            "  ★ 这里有个巨大的算法分岔, 选错的代价是**几千倍** ★\n"
            "  笨办法: 算出 16^(N+4) 这么大的整数再取十六进制位。\n"
            "    实测 N=10^6 时这条路线估算 **11105 秒**(≈3 小时), 直接超预算。\n"
            "  巧办法: **只要模运算, 不要大整数**。设要抽的是 frac(16^j·x) 的首位,\n"
            "    对形如 c/d 的项, 有\n"
            "        frac(c·16^(j-k)/d) = ((c·(16^(j-k) mod d)) mod d) / d\n"
            "    于是整条式子只需要 `pow(16, j-k, d)` 这种**小模数**模幂,\n"
            "    完全不碰 16^j 级别的整数。实测同题 **2.4 秒**。\n"
            "  实现要点:\n"
            "    · 用 `pow(16, e, d)`, **不要**写 `16 ** e % d`(那是大整数再取模)\n"
            "    · 指数为负的尾和**逐次除以 16 累加**, 不要写 `16.0 ** (k-j)`\n"
            "      (指数到 300 多时 float 直接 OverflowError —— 小规模测试根本碰不到)\n"
            "    · **先在小规模上自检**: 抽前 30~60 位, 和已知的那个数的展开逐位比对。\n"
            "      这一步能抓住索引差一位、系数写错这类错误, 而且几乎不花时间。\n"
            "      (实测两次翻车都是自检抓到的: 一次索引差一位, 一次浮点溢出。)"
        )
    if _DIOPHANTINE.search(claim or ""):
        # ★ 这里原来塞了一整段可运行的 pell_min() 代码, 还带着 D=61 的答案 ★
        # (1766319049, 226153980) —— 而 V2 hard-001 问的正是 Pell-61。
        # 也就是说**那道题的答案就写在它自己的 prompt 里**, 这条提示让它白送一分。
        #
        # 用户的要求: 「尽量不要给太多提示, 真实的问题往往是没有提示的」。
        # 而现在已经有**通用机制**兜底了(见 _strategy.py): 方法类失败(超时)之后,
        # 强制它换 `# STRATEGY:` 标签, 判词里明写「我不告诉你该选哪个」。
        # 所以删掉这段提示, 正好是一次真实检验: 通用机制能不能替代题型提示?
        parts.append(
            "【机器已识别: 这是丢番图方程 / 求最小解类问题】\n"
            "  ★ 先**自己判断这个问题的规模**: 逐个变量从小到大暴力试,\n"
            "    最坏情况要试到一个和答案同量级的界 —— 那必然超时。\n"
            "  ★ 所以动手前先想: 这个方程有没有**结构**? 定义域是连续的还是离散的?\n"
            "    有没有「遍历一遍就能穷尽」的完备判定过程(SAT/符号化简)?\n"
            "  ★ 只交了「解本身」不算「最小解」。最小性必须**程序化验证**:\n"
            "    枚举所有更小的候选, 确认都不满足 —— 跳过这步就只能判 PARTIAL。\n"
            "  ★ 如果第一轮超时了, 那说明**方法选错了**, 不是代码写错了:\n"
            "    换一个计算形态再试(枚举 → 符号 → 完备判定 → 构造), 别原样重跑。"
        )
    if _SCOPED_TOLERANCE.search(claim or ""):
        parts.append(
            "【机器已判定: 本题命题**自带范围和容差**, 不是全域命题】\n"
            "  这类命题的「判据」就是它自己写出来的那个范围 + 误差上界 ——\n"
            "  **把那个范围完整跑完、并给出误差估计, 就算 PROVED**。\n"
            "  ★ 不要因为「这只是数值验证, 不是符号证明」就自降 PARTIAL ★\n"
            "  实测翻车(2026-10-08 Basel): 命题是「取 N=10^7, 误差小于 1e-6」,\n"
            "  代码算出的误差是 1.0e-07(合规), 却因为「不是欧拉恒等式的符号证明」\n"
            "  自己判了 PARTIAL —— 命题从来没要求符号证明。**判据要看命题, 不看你\n"
            "  觉得命题本该长什么样。**\n"
            "  必须做的三件事:\n"
            "    1. 明确 print 出你用的 N / 范围, 以及它就是命题里的那个范围;\n"
            "    2. 给出**误差上界**并算出实际误差(尾部误差不能靠「大概够小」);\n"
            "    3. 打 `VERDICT: PROVED`。命题要求的范围全部覆盖到了, 就别降级。"
        )
    if not parts:
        return {"note": "", "how": how, "scales": [],
                "must_use_complete_method": False}
    return {"note": "\n\n".join(parts), "how": "solver" if must_complete else how,
            "scales": scales, "must_use_complete_method": must_complete}


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
