"""把题库的 expected(答案结构)转成 answer_spec, 并传给 research_verify。

## 为什么必须传

实测(2026-10-08 R 套 20 题): 14 道判"错"里有 11 道是 `artifact_present: False`。
agent 确实交了制品 —— res-006 交的是 `CONSTRUCT: {"quadruple": [-3,13,-10,2], ...}`,
res-005 交的是 `ANSWER: {"true_coeffs": [1,2,3], ...}` —— 而独立验证器在找
`relation` / `coeffs`。**格式对不上, 两边都以为对方有问题。**

答案结构本来就写在题库的 `expected` 字段里, 只是从没传给 agent。
不传的话, 我以为在测「模型能不能解题」, 实际测的是「模型能不能猜中我自己
定的字段名」—— 后者与研究能力毫无关系。

## 为什么这不是给提示

expected 说的是"交一个 8x8 的 0/1 矩阵 + 一份验证输出", 不是"用 Paley q=13
构造"。前者是接口约定, 后者才是解题。
★ 唯一不给的是 `recommended_tools` —— 那是**解题方向**, 属于提示, 不传 ★
"""
from __future__ import annotations

import re
import json
from typing import Any


#: ★ 对答案的**评价** —— 这些字段**一律不进契约** ★
#:
#: 2026-10-09 实测事故: 我原来的脱敏只把 expected.value 里的**值**换成占位符,
#: 却把**字段名**原样写进 prompt 说"必须包含这些字段"。而这些字段本身就是
#: 对答案的评价 —— 于是我在**主动要求 agent 填自评**, 它就编:
#:     res-005  verified_200_terms: true  (递推式根本生成不出那条序列)
#:     res-014  optimal: true             (动作序列连魔方都解不开)
#:     res-013  verified                  (89 组解全错)
#:
#: 这些量本来就该由**独立验证器自己算**: 拿 recurrence 生成 200 项自己比对 /
#: 用 BFS 判最优性 / 自己验证唯一性与不可约性。让 agent 自报等于让它自己给自己
#: 打分 —— 那正是打���器里 certification 维度的存在理由的反面。
SELF_EVAL_FIELDS = frozenset({
    "verified", "verified_200_terms", "verified_100_terms",
    "verified_by_second_impl", "verified_terms", "checked",
    "optimal", "is_optimal", "unique", "claims_unique",
    "irreducible", "distinct_distances", "all_distinct",
    "holds", "holds_5", "holds_7", "is_prime", "solved", "closed",
})


#: 描述性短语 —— 这些是"要交什么", 不是答案, 可以原样给 agent
_DESCRIPTIVE_HINTS = ("的", "表示", "输出", "列表", "序列", "分解", "判定",
                      "验证", "检查", "确认", "格式", "证明")


#: 「交付形态」词表 —— 说的是"交什么", 不是"是什么 / 怎么做"。
_SHAPE_WORDS = (
    "矩阵", "列表", "序列", "数组", "系数", "元组", "字符串", "对象", "网格",
    "生成元", "顶点", "解", "排列", "巡回", "刻度", "着色", "四元组", "整数",
    "布尔", "数值", "三元组", "二元组", "序列值", "距离", "周期",
)

#: ★ 解法 / 答案词表 —— 出现在 expected 里就是泄露, 一律脱敏 ★
#: 注意 "A5" 不在这里: 它本来就在 res-019 的题面里("是否为 A5"), 不是泄露。
_ANSWER_WORDS = (
    "Cauchy", "Titu", "Schwarz", "SOS", "Schoenberg", "Paley", "Williamson",
    "Fibonacci", "A000045", "Fermat", "Eisenstein", "对称群", "最优",
    "唯一解", "不可约",
)


def _redact(v: Any, known: str = "") -> str:
    """把 expected 的值改成**类型说明**, 绝不把答案本身给出去。

    ★ 为什么必须脱敏 ★
    题库的 expected 字段里, 有些值是**输出格式**(如 '8x8 对称矩阵, 取值
    red/blue'), 有些值**就是答案**(res-012 的 degree=8、irreducible=True;
    res-017 的 verdict='false')。照抄进 prompt 等于泄题 ——
    用户明确说过「依赖已知结果并且提示的, 说明其实还不是很妙」。

    规则: 数字、布尔、以及形如 'A000045'/'false' 这类短值一律换成类型说明;
    带描述性措辞的长串(说明"要交什么样的东西")才保留。
    """
    if isinstance(v, bool):
        return "<True/False, 表示该性质是否成立>"
    if isinstance(v, (int, float)):
        return f"<{'整数' if isinstance(v, int) else '数值'}>"
    if isinstance(v, (list, tuple)):
        return f"<长度 {len(v)} 的列表>"
    if isinstance(v, dict):
        return "<对象>"
    s = str(v)

    # ★ 白名单式, 不再是"含中文就放行" ★
    #
    # 原来的启发式(2026-09-08/09 写的)是: "含中文 → 是描述, 放行; 短 ASCII →
    # 是答案, 脱敏"。审计实测它**两边都漏**:
    #   · res-015 的 proof_method = "Cauchy-Schwarz / Titu 引理 或 SOS"
    #     含中文 → 原样进入 prompt = **直接给了这道题的解法方向**
    #   · len(s) <= 16 的阈值形同虚设: 17 字符的 "<= 1.05 * optimal"、
    #     19 字符的 "A000045 = Fibonacci" 全部原样通过
    #
    # 改成白名单: 只有明确属于"交付形态"的词才放行, 其余一律脱敏。
    # 判断"交付形态"看的是**这个词在说"交什么", 还是在说"是什么/怎么做"**。
    # ★ 这些词是**解法/答案**, 一旦出现在 expected 里就是泄露 ★
    if any(w in s for w in _ANSWER_WORDS):
        return "<自行计算>"
    # ★ 数量也是答案的一部分 ★
    # res-019 的 expected 写着 generators = "两个生成元", 而题面只说"给出生成元",
    # 没说几个 —— "两个"等于告诉了它目标群需要两个生成元。
    if any(c.isdigit() for c in s) or any(
            c in "一二三四五六七八九十两百千万" for c in s):
        # ★ 题面里已有的数字不是泄露 ★ 本来就知道, 说出来不构成提示。
        # 不加这个判断会**过度脱敏**: res-001 的
        # "8x8 对称矩阵, 取值 red/blue" 会被抹成 "<矩阵, 数量自行确定>",
        # 连"取值是 red/blue 两种颜色"这个格式要求也一起丢 ——
        # 而题面本来就写的是"8 个顶点的红蓝完全图"。
        if not (known and _all_numbers_in(s, known)):
            return f"<{_shape_hint(s)}>"
    if any(w in s for w in _SHAPE_WORDS):
        return s
    # 说不清形态的一律脱敏 —— 少给一句类型提示, 换来不泄露答案
    return "<自行计算>"


def _all_numbers_in(s: str, known: str) -> bool:
    """s 里出现的所有数字(ASCII 与中文数字都算), 是否在 known(题面)里都能找到。

    ★ 中文数字必须一起查 ★
    第一版只 `re.findall(r"\\d+")`, 于是 s = "两个生成元" 时得到空列表,
    `all([])` 恒为 True -> 被当成"题面已知"而放行 —— 而 res-019 的题面
    根本没提"两个", 那正是我们要拦的泄露。
    """
    ascii_nums = re.findall(r"\d+", s)
    cn_nums = [c for c in s if c in "一二三四五六七八九十两百千万"]
    return (all(n in known for n in ascii_nums)
            and all(c in known for c in cn_nums))


def _shape_hint(s: str) -> str:
    """带数量的描述, 只留"交什么"那部分, 丢掉数量。"""
    for w in _SHAPE_WORDS:
        if w in s:
            return f"{w}, 数量自行确定"
    return "自行计算"


def expected_to_spec(expected: Any, known: str = "") -> str:
    """题库 expected -> 一段人能读、模型能照抄的输出格式说明(已脱敏)。"""
    if not expected:
        return ""
    if isinstance(expected, str):
        return expected.strip()
    if not isinstance(expected, dict):
        return str(expected).strip()

    lines: list[str] = []
    etype = expected.get("type")
    kind_map = {
        "object": "构造/答案对象", "boolean": "真假判定",
        "condition": "条件性结论(可能是反例也可能是证明)",
    }
    if etype:
        lines.append(f"- 结论类型: {kind_map.get(etype, etype)}")
    val = expected.get("value")
    if isinstance(val, dict):
        # ★ 剥掉自评字段 ★ 见 SELF_EVAL_FIELDS 的说明
        answer_only = {k: v for k, v in val.items()
                       if k not in SELF_EVAL_FIELDS}
        dropped = sorted(set(val) - set(answer_only))
        if answer_only:
            lines.append("- 需要提交的字段(只交**答案本身**):")
            for k, v in answer_only.items():
                lines.append(f"    * `{k}` —— {_redact(v, known)}")
        if dropped:
            # ★ 原来这里会打印「★ 不要报告这些: irreducible / optimal / unique」★
            # ★ 而那行本身**就是在泄题** ★
            # 它等于宣布"你这份答案就是**不可约**的 / **唯一**的 / **最优**的" ——
            # 对 res-011(unique)、res-012(irreducible)、res-014(optimal) 三题,
            # 这行字直接说出了结论。
            #
            # 正确做法: 只说"有这类字段你不该填", **不点名是哪些**。
            # 模型不必知道字段名, 知道才危险; 而"只交答案本身"这一句
            # 已经足够让它不去自评。
            lines.append(
                "  ★ 只交**答案本身**。凡是你对答案的**评价**(是否可约 / 是否唯一 /"
                " 是否最优 / 是否已验证过), 一律不要写 —— 它们由独立复核器自己算, "
                "你自己写的版本不作数(实测会被编造)。")
        if not answer_only:
            lines.append("- 本题不需要交具体对象, 只交判定即可。")
    elif val is not None:
        lines.append("- 需要提交的值: " + _redact(val, known))
    if expected.get("detail"):
        lines.append(f"- 补充要求: {expected['detail']}")
    lines.append("  (上面只给字段和类型, **不给答案值** —— 值必须你自己算)")
    return "\n".join(lines)


def spec_example(expected: Any) -> str:
    """给一个字段名齐备的空壳示例, 降低字段名猜错的概率(值全是占位符)。"""
    if not isinstance(expected, dict):
        return ""
    val = expected.get("value")
    if not isinstance(val, dict):
        return ""
    shell = {k: _redact(v) for k, v in val.items()
             if k not in SELF_EVAL_FIELDS}
    return json.dumps(shell, ensure_ascii=False) if shell else ""


def build_claim(prompt: str, expected: Any, data_hint: str = "",
               problem_id: str = "") -> str:
    """把题面 + 输出格式 + 数据文件位置拼成最终送进工具的 claim。"""
    parts = [prompt]
    spec = expected_to_spec(expected, prompt)
    if spec:
        parts.append(
            "\n\n【本题要求你交出的东西 — 输出格式】\n" + spec
            + "\n  ★ 这是**输出格式约定**, 不是解题提示 —— 它不告诉你答案是什么,\n"
              "    只告诉你答案要装进什么样的字段里。")
        shell = spec_example(expected)
        if shell:
            parts.append(f"\n  参考壳子(字段名照抄, 值自己算): {shell}")
    if data_hint:
        # ★ 内联数据, 而不是告诉模型去读文件 ★
        # 2026-10-09 实测事故: 原来只说"数据在 data/res_13_system.txt,
        # 沙箱禁读文件, 按题面描述写进代码" —— 而题面里一个数字都没有。
        # 模型拿不到数据却被告诫"把数据抄进代码", 于是自己发明了题面
        # (res-013 那个"发明第三个变量"就是这么来的), 那 8 道题不可解。
        try:
            import problem_data
            blk = problem_data.data_block(problem_id or "")
        except Exception:
            blk = ""
        if blk:
            parts.append(blk)
        else:
            parts.append("\n\n【本题用到的数据文件】\n" + data_hint)
    return "".join(parts)


#: 题号 -> 需要告诉 agent 去哪读数据(题面引用了但文件曾缺失的那些)
DATA_HINTS = {
    "res-005-berlekamp-massey":
        "序列在 `data/res_05_sequence.txt`(已生成时)。"
        "注意: 沙箱禁读文件, 所以你必须**把数据抄进代码里**或用代码生成。",
    "res-007-oeis-identify-prove":
        "序列前 30 项在 `data/res_07_sequence.txt`(已生成时)。"
        "沙箱禁读文件, 把数字抄进代码里。",
    "res-010-sprague-grundy":
        "规则在 `data/res_10_rules.txt`。沙箱禁读文件, 按题面描述实现即可。",
    "res-011-sudoku-diagonal-z3":
        "盘面在 `data/res_11_sudoku.txt`(已生成时)。沙箱禁读文件, 抄进代码。",
    "res-013-groebner-system":
        "方程组在 `data/res_13_system.txt`。沙箱禁读文件, 按题面描述写进代码。",
    "res-014-rubiks-6-moves":
        "打乱状态与解在 `data/res_14_rubiks.txt`(已生成时)。沙箱禁读文件。",
    "res-016-tsp-heuristic-500":
        "实例在 `data/res_16_tsp.txt`(已生成时)。沙箱禁读文件。",
    "res-020-hypothesis-refine-loop":
        "猜想在 `data/res_20_conjecture.txt`。沙箱禁读文件。",
}


def data_hint_for(problem_id: str) -> str:
    return DATA_HINTS.get(problem_id, "")