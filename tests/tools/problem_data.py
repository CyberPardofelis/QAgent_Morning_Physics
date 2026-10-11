"""把题目的**输入数据**内联进 claim。

## 为什么要内联

★ 2026-10-09 实测事故(DeepSeek 指出的, 已核实成立)★
8 道题的题面引用了 `data/res_*.txt` 里的输入数据, 而我给模型的 data_hint 是:
    「方程组在 data/res_13_system.txt。沙箱禁读文件, 按题面描述写进代码。」
    -> claim 里含数据 = False
**这等于让它把一份它拿不到的数据抄进代码。** res-013「发明第三个变量
x,y,z」就是这么来的 —— 题面没说有几个方程几个变量, 它就自己编了一个。
那 8 道题从头到尾**不可解**, 却算进了我的基线。

## 为什么内联不是泄题

- 题面本来就**应该**带输入。这批数据是题目的一部分(方程组/盘面/序列/
  TSP 实例), 不是答案。
- 真正不能给的是 `*_reference.json`(答案), 这里一行都不碰。
- 而且内联之后**沙箱的读文件禁令可以变严**: 模型不需要读任何东西,
  输入在题面上, 参考答案读不到。

## 红线

`_reference.json` 绝不内联。这里只读 `res_*` 里**不含 reference** 的输入文件。
"""
from __future__ import annotations

import io
import os
from typing import Any

#: 题号 -> 输入数据文件(必须是题目输入, 绝不是答案)
DATA_FILES = {
    "res-005-berlekamp-massey": "res_05_sequence.txt",
    "res-007-oeis-identify-prove": "res_07_sequence.txt",
    "res-010-sprague-grundy": "res_10_rules.txt",
    "res-011-sudoku-diagonal-z3": "res_11_sudoku.txt",
    "res-013-groebner-system": "res_13_system.txt",
    "res-014-rubiks-6-moves": "res_14_rubiks.txt",
    "res-016-tsp-heuristic-500": "res_16_tsp.txt",
    "res-020-hypothesis-refine-loop": "res_20_conjecture.txt",
}

#: 超过这个长度就只给前若干行, 并明确告诉模型"数据很长, 这是它的开头"
#: —— 静默截断会让模型以为那就是全部, 然后推出错误结论。
MAX_INLINE = 12000


def read_data(problem_id: str) -> str | None:
    """读题目的输入数据。**不含答案**。"""
    fn = DATA_FILES.get(problem_id)
    if not fn:
        return None
    if "reference" in fn:            # 双保险: 万一映射写错
        raise AssertionError(f"{problem_id} 的数据文件指向了参考答案: {fn}")
    p = os.path.join("data", fn)
    if not os.path.isfile(p):
        return None
    text = io.open(p, encoding="utf-8").read().strip()
    if len(text) > MAX_INLINE:
        head = text[:MAX_INLINE]
        return (f"{head}\n\n"
                f"[数据很长, 以上是前 {MAX_INLINE} 字符。剩下的部分对你这题"
                f"未必需要; 如果确实需要, 说明这题的表述本身有问题。]")
    return text


def data_block(problem_id: str) -> str:
    """拼成题面里的一段数据。"""
    text = read_data(problem_id)
    if not text:
        return ""
    return (
        "\n\n【本题的输入数据 —— 直接用下面的内容, 不要去读文件】\n"
        "沙箱里读不到任何文件, 但这题需要的数据**已经完整贴在下面**。\n"
        "----------------\n"
        f"{text}\n"
        "----------------\n"
    )


def input_reachable(problem_id: str) -> bool:
    """这道题的输入是不是真的可达(用于评测前置门)。"""
    if problem_id not in DATA_FILES:
        return True                    # 本来就没有外部数据
    return read_data(problem_id) is not None


_ = Any  # 保持签名可扩展