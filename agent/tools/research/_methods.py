"""★ 题型识别 + 方法目录 —— 让工具自己认出"这是什么题、该用什么" ★

## 为什么是这两个东西，而不是再加一段提示

2026-10-09 删掉 Pell 的技法提示时我意识到一件事: 我一直靠在 prompt 里塞
"这类题该用什么方法", 而那有两个致命问题:
  1. **泄题** —— 提示里写了 D=61 的答案(实测确认, V2 hard-001 就在题库里)
  2. **覆盖不了** —— 提示是按题型硬编码的, 遇到没预设过的题型就零帮助;
     而且每加一个题型提示, 门数就 +1(这是本轮的教训)。

把方法改成**工具可查的目录**, 两个问题一起消失:
  · 不泄题 —— 目录只说"有什么方法、各自适用什么条件", 不说"本题答案是 X"
  · 不需要预设题型 —— 目录是**通用的方法学**, 覆盖组合/代数/数论/几何/证明

★ 目录必须写"什么时候**不该**用它" ★
只列适用条件的方法表会诱导模型盲目套用。每条都要写失败模式, 让模型
自己排除 —— 这才是"认题", 而不是"查表"。

## 三个动作的分工(对着用户那三句话)

  认题     classify(claim) -> 形状 + 规模 + 结构标签  [确定性, 不调 LLM]
  找方法   methods_for(shape) -> 候选 + 适用条件 + **失败模式**
  验证     由 _strategy.py 的 STRATEGY 标签 + 制品契约 + 独立验证器承担

前两个都是纯函数, 不花 token, 也不泄题。
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------
# 1. 认题 —— 从题面里抽"形状 / 规模 / 结构"标签
# ---------------------------------------------------------------

#: 规模标记。题面里出现的数字直接决定"裸枚举来不来得及"。
#: ★ 数字必须有左边界 ★ 不加的话 "1.05" 会被抠出 "05" 当成一个规模标记 ——
#: 那是个纯粹的噪声, 会让"先算一遍来不来得及"这句话建立在错数据上。
_SCALE = re.compile(
    r"(?<![\w.])(?:1e[+-]?\d+|\d+\s*\^?\s*\d+|10\s*的\s*\d+\s*次方|"
    r"\d{3,}\s*(?:个|万|亿|项|位|个城市)|前\s*\d{2,})",
    re.IGNORECASE)

#: 结构标签 -> (判定正则, 人类可读名)
#:
#: ★ 每个正则都要排除**更常见的那个意思** ★
#: 第一版把 "证明" 单独当形式化信号, 结果每道题都被判成"形式化证明"(几乎
#: 所有题面都有"证明"二字); 又把 "反例" 当反例搜索信号, 结果
#: 「并给出 8 个顶点**无反例**的着色」被误判。**假阳性的分类比没有分类更糟** ——
#: 它会主动把模型引向错误的路线。所以宁可漏判, 不可乱判。
_SHAPES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("combinatorial",
     re.compile(r"着色|染色|顶点|红蓝完全图|独立集|完美匹配|子集和|"
                r"coloring|graph|vertex|clique|knight|Ramsey|色数"),
     "组合结构"),
    ("construction",
     re.compile(r"构造|给出一个?\s*\d*\s*[阶个]|生成元|乘法表|"
                r"construct|build a"),
     "构造题(交一个对象 + 验证它)"),
    ("counterexample",
     # ★ 必须排除「无反例」「非反例」★ 那些是在说"不存在反例", 不是要找反例
     re.compile(r"(?<!无)(?<!非)反例|证伪|推翻|counterexample|refute|disprove"),
     "反例搜索"),
    ("exact_computation",
     re.compile(r"精确|精确值|十六进制|最小正整数解|最小解|极小|下界|"
                r"恰好第|第\s*\d+\s*位|integer\s+relation"),
     "精确计算(浮点不可用)"),
    ("formal",
     # ★ 必须点名 Lean/Mathlib/形式化, 不能只靠"证明" ★
     re.compile(r"Lean|Mathlib|形式化|formal|theorem|定理机器"),
     "形式化证明(Lean/Mathlib)"),
    ("inequality",
     re.compile(r"不等式|SOS|证明.*>=|证明.*<=|inequality"),
     "不等式"),
    ("optimization",
     re.compile(r"最优解|最小化|最短解|TSP|巡回|可行解|heuristic|optimal|"
                r"minimize"),
     "优化(要构造一个够好的可行解)"),
    ("diophantine",
     re.compile(r"丢番图|Pell\s*方程|整系数解|Diophantine|"
                r"x\^?\s*2\s*-\s*\d+\s*\*?\s*y\^?\s*2"),
     "丢番图/整系数解"),
    ("proof_request",
     re.compile(r"证明|prove|论证"),
     "要求给证明"),
)


def classify(claim: str) -> dict[str, Any]:
    """认题: 题面 -> 形状标签 + 规模线索。

    ★ 刻意保持确定性(纯正则), 不调 LLM ★
    理由: 让模型自己"认题"是靠不住的(它可以随便写一个标签);
    而"题面里有哪些结构性词汇"是机器能确定回答的问题。
    这里给出**线索**, 选哪条路仍然由模型负责 —— 工具不替它做决定。
    """
    text = claim or ""
    shapes = [name for _, rx, name in _SHAPES if rx.search(text)]
    scales = _SCALE.findall(text)
    return {
        "shapes": shapes,
        "scale_hints": [s.strip() for s in dict.fromkeys(scales)][:6],
        # 有规模线索就提醒它先算一遍来不来得及 —— 这是它最容易忽略的一步
        "note": ("题面含规模标记, 动手前先自己算一遍'裸枚举来不来得及'"
                 if scales else "题面没有明显规模标记, 但仍需自己判断规模"),
    }


# ---------------------------------------------------------------
# 2. 方法目录 —— 可查的通用方法学, 不是按题预设的提示
# ---------------------------------------------------------------

#: 每条 = (方法名, 适用形状, 什么时候**不该**用它)
#:
#: ★ 第三项是这条表的核心 ★
#: 只写适用条件的表会诱导模型盲目套用最上面那条。写了失败模式, 模型才能
#: 真正"排除"掉不合适的路 —— 那才是认题, 不是查表。
_METHODS: tuple[dict[str, str], ...] = (
    {"name": "solver_sat", "tool": "z3",
     "for": "组合结构 / 存在性 / 有限定义域上的全域否定",
     "not_for": ("定义域无界(求解器会返回 unknown 而不是答案); "
                 "表述不是一阶可判定的(比如涉及三角函数等量词)"),
     "how": "s.add(约束); check()==sat 取 model; "
            "再加 Not(命题) 后 unsat 就是'不存在反例'的全域证明",
     "watch": "check() 是三态: sat/unsat/**unknown**。把 unknown 当 unsat 是错的"},
    {"name": "symbolic", "tool": "sympy",
     "for": "符号恒等式、精确值、化简、微积分/代数封闭形式",
     "not_for": "大规模离散枚举; 数值上溢/下溢敏感的问题",
     "how": "符号化简到 0 或精确有理式; 必要时用 Rational/Integer 避免浮点",
     "watch": "浮点数恰好为 0 **不代表**符号上恒等于 0"},
    {"name": "numeric_highprecision", "tool": "mpmath/decimal",
     "for": "无闭式但可高精度逼近的量(位抽取、特殊常数、超大整数)",
     "not_for": "需要**证明**成立(数值只能支持 PARTIAL)",
     "how": "设 mp.dps 足够高; 位抽取用小模数模幂而不是构造大整数",
     "watch": "精度设小了会在最后几位出错, 而那恰好是答案所在"},
    {"name": "bounded_enumeration", "tool": "python",
     "for": "**题目自己划定了有限范围**时的完整证明",
     "not_for": "范围没给的题(那只是抽查, 最多支持 PARTIAL)",
     "how": "先在注释里写清枚举的确切上界, 并说明它覆盖了命题的全部定义域",
     "watch": "★ 先算一遍这个范围跑不跑得完; 跑不完就不是证明而是超时"},
    {"name": "continued_fraction", "tool": "sympy",
     "for": "二次无理数的收敛分数类问题(Pell 型、周期结构)",
     "not_for": "一般丢番图方程; 答案不在收敛分数上时",
     "how": "continued_fraction_periodic 拿到循环节 -> 展开成收敛分数 -> 逐个验方程",
     "watch": "★ 周期为**奇数**时解的下标是 2L-1 而不是 L —— 这是高频错点"},
    {"name": "construct_and_check", "tool": "python",
     "for": "题目要求**交出一个对象**并证明它满足性质",
     "not_for": "性质是「不存在」类的题(要构造就得同时证明没有别的)",
     "how": "构造后**必须**自己写一个独立的检查器把性质逐条验一遍",
     "watch": "★ 只交对象不交检查 = 无法被复核, 评分按没交处理"},
    {"name": "formal_proof", "tool": "lean4+mathlib",
     "for": "题目点名 Lean/Mathlib/形式化",
     "not_for": "Lean 没装的机器(会静默回落 Python, 那不构成形式化证明)",
     "how": "先 #check 关键引理是否存在, 再写证明; 失败时读编译器的具体报错",
     "watch": "★ 绝不允许 sorry; 也不允许先 axiom 再 theorem(那是后门)"},
    {"name": "search_then_verify", "tool": "python",
     "for": "先怀疑猜想、再决定方向的题",
     "not_for": "有标准解法且规模很小的题(直接做更快)",
     "how": "小规模实验找趋势/反例 -> 据此缩小范围 -> 再上重武器",
     "watch": "★ 先在小规模上自检求解器本身, 否则你不知道自己找的是不是真的"},
    {"name": "optimise_heuristic", "tool": "ortools/numpy",
     "for": "给定实例求一个**够好**的可行解(不要求最优)",
     "not_for": "要求最优并需要证明最优性的题",
     "how": "最近邻/贪心起步 -> 局部搜索(2-opt/换边)改进; "
            "**先确认解是合法排列**(每个点恰好一次)",
     "watch": "★ 非法解的长度没有意义; 合法性要先查再比长度"},
)

_METHOD_BY_NAME = {m["name"]: m for m in _METHODS}


#: 方法 → 它主要服务哪些**形状标签**。
#:
#: ★ 键必须和 classify() 返回的**中文标签**一致 ★
#: classify 返回的是 `_SHAPES` 里每一项的第三个元素("组合结构" 这种),
#: 我第一版按内部 key("combinatorial")建表, 结果一条都匹配不上,
#: methods_for 退化成全量 —— 认题又是装饰品。
#: 这类"两张表靠字符串对齐"的错, 应该在 classify 里直接返回结构体,
#: 但先保证键对齐, 并加测试锁住(见 check_methods_catalog.py)。
_SHAPE_TO_METHODS = {
    "组合结构": ("solver_sat", "bounded_enumeration", "construct_and_check"),
    "构造题(交一个对象 + 验证它)": ("construct_and_check", "search_then_verify"),
    "反例搜索": ("search_then_verify", "bounded_enumeration", "solver_sat"),
    "精确计算(浮点不可用)": ("symbolic", "numeric_highprecision"),
    "形式化证明(Lean/Mathlib)": ("formal_proof",),
    "不等式": ("symbolic", "solver_sat"),
    "优化(要构造一个够好的可行解)": ("optimise_heuristic", "search_then_verify"),
    "丢番图/整系数解": ("continued_fraction", "symbolic"),
    "要求给证明": ("symbolic", "solver_sat"),
}


def methods_for(shapes: list[str]) -> list[dict[str, str]]:
    """按认出的形状给候选方法。

    ★ 返回**顺序有意义**: 排在前面的是该形状下的主力路线 ★
    但**不删掉其余的** —— 认题可能认错, 全删掉会让模型无路可走。
    做法: 相关方法在前, 其余的标成"兜底"跟在后面。

    没认出任何形状时返回全量(此时不该假装识别成功)。
    """
    if not shapes:
        return [dict(m, _rank="兜底") for m in _METHODS]

    wanted: list[str] = []
    for sh in shapes:
        for name in _SHAPE_TO_METHODS.get(sh, ()):
            if name not in wanted:
                wanted.append(name)

    head = [dict(m, _rank="主力") for m in _METHODS if m["name"] in wanted]
    tail = [dict(m, _rank="兜底") for m in _METHODS if m["name"] not in wanted]
    return head + tail


def method_doc(name: str) -> dict[str, str] | None:
    return _METHOD_BY_NAME.get(name)