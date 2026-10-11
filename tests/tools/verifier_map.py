"""题号 -> verifier 模块的显式映射。

## 为什么不用「从题号推导文件名」

题库 id 和 verifier 文件名对不上:
    res-001-ramsey-34-lower-bound -> res_01_ramsey.py
    res-002-golomb-ruler-15       -> res_02_golomb.py
    res-013-groebner-system        -> res_13_groebner.py
数字能推, 单词不能。所以显式列出来 —— **映射错了会静默地让某题跳过
独立验证然后按"没验"计分**, 那是比分错更糟的错。

## 为什么必须有这个表

DS 打分器里 correctness 权重 0.40, 且明确要求
「correctness 一定得来自外部 verifier」。没有这张表, correctness 恒为 0,
总分封顶 60, 而且我无法回答"到底有没有全对"。
"""
from __future__ import annotations

PROBLEM_TO_VERIFIER = {
    "res-001-ramsey-34-lower-bound": "res_01_ramsey",
    "res-002-golomb-ruler-15": "res_02_golomb",
    "res-003-hadamard-28-paley": "res_03_hadamard",
    "res-004-sos-inequality": "res_04_sos",
    "res-005-berlekamp-massey": "res_05_berlekamp",
    "res-006-pslq-integer-relation": "res_06_pslq",
    "res-007-oeis-identify-prove": "res_07_oeis",
    "res-008-lean-binomial": "res_08_lean_binomial",
    "res-009-lean-quadratic": "res_09_lean_quadratic",
    "res-010-sprague-grundy": "res_10_sprague_grundy",
    "res-011-sudoku-diagonal-z3": "res_11_sudoku_diagonal",
    "res-012-minimal-polynomial": "res_12_minimal_polynomial",
    "res-013-groebner-system": "res_13_groebner",
    "res-014-rubiks-6-moves": "res_14_rubiks",
    "res-015-cyclic-inequality-counterexample": "res_15_cyclic_inequality",
    "res-016-tsp-heuristic-500": "res_16_tsp",
    "res-017-counterexample-hunt": "res_17_counterexample",
    "res-018-formal-fermat-small": "res_18_lean_fermat",
    "res-019-group-construction-60": "res_19_group60",
    "res-020-hypothesis-refine-loop": "res_20_hypothesis",
}


def load_verifier(problem_id: str):
    """按题号取 verifier; 映射缺失要**显式报错**, 不能静默返回 None。

    静默 None 会让这道题按"没有独立验证"处理, 而统计里它和"验证了但错了"
    长得一样 —— 正是本项目反复踩的"没跑"和"判错"必须分开那个坑。
    """
    import importlib.util
    import os
    import sys

    mod_name = PROBLEM_TO_VERIFIER.get(problem_id)
    if mod_name is None:
        raise KeyError(
            f"题号 {problem_id} 没有对应的 verifier —— 覆盖率会出缺口, "
            f"correctness 无法计算。请先补 verifier 再跑。")
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    for p in (here, root):
        if p not in sys.path:
            sys.path.insert(0, p)
    path = os.path.join(here, "verifiers_r", mod_name + ".py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"verifier 文件不存在: {path}")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "verify"):
        raise AttributeError(f"{path} 里没有 verify()")
    return mod