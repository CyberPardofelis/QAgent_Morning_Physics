"""按独立复算的结果修正题库参考答案, 并留审计痕迹。

★ 为什么改题库而不是只报告 ★
GT 错 -> verifier 稳定判错 -> 分数看起来正常 -> 从分数里读不出来。
所以这不是"记一笔备注"的事, 必须把库改对, 否则闸门天天拦。

★ 改了什么必须可追溯 ★
每条修正写进题库顶层 `_corrections`, 记: 原值 / 新值 / 依据哪个复算路径 /
为什么原值错。下一个人(尤其 DS)可以直接对着复算方法核, 而不是信我。

三类改动:
  A. 数值订正   —— 复算值客观唯一, 直接改 (7 道)
  B. 判据改写   —— 原 GT 不是良定义标量, 改成行为性判据 (3 道)
  C. 自洽性修正 —— GT 自身不自洽, 改成可验证的事实陈述 (1 道)
"""
from __future__ import annotations

import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(ROOT, "benchmark")


# --------------------------------------------------------------------------
# A. 数值订正
# --------------------------------------------------------------------------
CORRECTIONS = {
    "z3-01-lin-int": {
        "why": "GT 漏掉了 |y|<=10 这个约束: x=-5 对应 y=11, 不在允许范围内",
        "gt": {"solutions": "x ∈ {-3, -1, 1, 3, 5, 7, 9}, 对应 y = (7-3x)/2",
               "expected_count": 7},
        "gt_before": {"solutions": "x ∈ {-5, -3, -1, 1, 3, 5, 7, 9}, 对应 y = (7-3x)/2",
                      "expected_count": 8},
        "method": "tests/tools/reference_recompute.py::_z3_01 —— 纯 Python 双层 for 穷举",
    },
    "ortools-01-int-prog": {
        "why": "(3,4) 可行(2*3+4=10<=10, 3+3*4=15<=15)且目标 17 > 16; GT 给的 16 不是最优",
        "gt": {"max": 17, "argmax": [3, 4]},
        "gt_before": {"max": 16, "argmax": [4, 2]},
        "method": "tests/tools/reference_recompute.py::_ortools_01 —— 纯 Python 网格穷举",
        "verifier": {"pass_condition": "obj == 17 and constraints satisfied"},
    },
    "cvxpy-02-qp": {
        "why": "配方 f = (x-1)^2 + (y-2)^2, 在 (1,2) 取 0; 该点满足 x+y=3<=3 可行",
        "gt": {"min": 0.0, "argmin": [1, 2]},
        "gt_before": {"min": 1.0, "argmin": [1, 2]},
        "method": "tests/tools/reference_recompute.py::_cvxpy_02 —— 配方 + 可行性检查",
        "verifier": {"pass_condition": "abs(optimal - 0.0) < 1e-6"},
    },
    "cvxpy-04-sos": {
        "why": ("多项式 x^4+y^4+1-x*y-x-y 不是非负的: f(0.5,0.5) = -0.125, "
                "网格+坐标下降的全局最小 -0.4303593 @ (0.760690, 0.760690)"),
        "gt": {"verdict": "false",
               "counterexample_point": [0.5, 0.5],
               "value_at_counterexample": -0.125,
               "global_min": -0.4303593067,
               "argmin": [0.760690, 0.760690]},
        "gt_before": {"verdict": "true", "method": "SOS 分解或对偶 SDP"},
        "method": "tests/tools/reference_recompute.py::_cvxpy_04 —— 801x801 网格 + 坐标下降精化",
        # 题面原本预设了假命题("验证 ... 成立"), 预设假命题的题面本身是缺陷
        "prompt": ("用 CVXPY 判定多项式 x^4 + y^4 + 1 - x*y - x - y 是否对所有实数 x, y "
                   "都非负。若成立给出 SOS 分解或对偶证书; 若不成立给出反例点。"),
        "verifier": {"pass_condition":
                     "verdict is false AND a counterexample (x,y) with "
                     "f(x,y) < -1e-9 is given"},
    },
    "scipy-04-oscillatory-integral": {
        "why": ("解析闭式 (1/2)sqrt(pi/a)exp(-b^2/4a) 在 a=0.1,b=1 下 = 0.2300428481, "
                "与 T=60 的 60 万段 Simpson 互验一致; GT 的 2.4053 像是串到了别的数"),
        "gt": {"value": 0.2300428481424944, "tolerance": 1e-6},
        "gt_before": {"value": 2.4053, "tolerance": 0.0001},
        "method": "tests/tools/reference_recompute.py::_scipy_04 —— 闭式 + 复合 Simpson 双路",
        "verifier": {"pass_condition": "abs(value - 0.2300428481424944) < 1e-6"},
    },
    "res-03-sylvester-matrix": {
        "why": ("消元 AX+XB=I: 8d=1 -> d=1/8; 7c+d=0 -> c=-1/56; 6b+2d=0 -> b=-1/24; "
                "5a+b+2c=1 -> a=15/70。GT 的四个分量全错(代回 AX+XB 不等于 I)"),
        "gt": {"verdict": "存在唯一解",
               "X": [[0.2154761904761905, -0.041666666666666664],
                     [-0.017857142857142856, 0.125]],
               # ★ a = (1 - b - 2c)/5 = (1 + 1/24 + 1/28)/5 = (181/168)/5 = 181/840 ★
               # 这里我一开始写成 3/14 —— 3/14 代回残差是 -1/168, 不是 0。
               # 这个字段没有任何检查读它, 所以错了也没人知道, 直到外部审查发现。
               "X_exact": [["181/840", "-1/24"], ["-1/56", "1/8"]],
               "residual": 0.0},
        "gt_before": {"verdict": "存在唯一解",
                      "X": [[0.20512820512820512, -0.017094017094017096],
                            [-0.017094017094017096, 0.09401709401709402]]},
        "method": "tests/tools/reference_recompute.py::_res_03 —— Fraction 精确消元 + 残差验算",
        "verifier": {"pass_condition":
                     "residual ||AX+XB-I||_max < 1e-10 AND spectral condition checked"},
    },
    "res-20-cross-method-matrix": {
        "why": ("真特征值 1.2679492, 3.0, 4.7320508。GT 那组的迹 9.0 恰好对, "
                "但乘积 22.7006 != det(M)=18 —— 只核迹的 verifier 会漏掉它"),
        "gt": {"eigenvalues": "约 1.2679491924, 3.0, 4.7320508076",
               "checks": ["特征多项式", "数值对角化", "幂迭代或 QR"],
               "invariant_checks": {"trace": 9.0, "det": 18.0}},
        "gt_before": {"eigenvalues": "约 4.4605, 2.5198, 2.0197",
                      "checks": ["特征多项式", "数值对角化", "幂迭代或 QR"]},
        "method": "tests/tools/reference_recompute.py::_res_20 —— Faddeev–LeVerrier 手算系数 + 多项式求根 + numpy 交叉验",
        "verifier": {"pass_condition":
                     "eigenvalues match {1.2679492, 3.0, 4.7320508} within 1e-6 "
                     "AND at least 3 independent methods reported"},
    },
}

# --------------------------------------------------------------------------
# B. 判据改写: 原 GT 不是良定义标量 -> 改成行为性判据
# --------------------------------------------------------------------------
# 为什么不是把参数钉死: 钉死需要我替 DS 定一套物理协议(步长/瞬态/种子),
# 那是重设计题。改成"判据"则是承认这类题的可复算方式是"看曲线/看标度",
# 不是"比一个数"——这也正是 Part 2 该有的判据形态。
CRITERION_REWRITES = {
    "res-13-chaos-logistic-scan": {
        "why": ("原 GT 给的是'混沌区 r 集合', 但混沌与否取决于观察时长与判据"
                "(轨道发散/周期窗口), 题面没定死 -> 不是良定义标量"),
        "gt": {"criterion_type": "behavioral",
               "must_report": [
                   "lambda_max(r) 曲线(横轴 r, 纵轴最大 Lyapunov 指数)",
                   "至少标出周期倍增的 Feigenbaum 累积点 r_inf, 判据 |r_inf - 3.5699| < 5e-3",
                   "显式声明瞬态丢弃长度与后继迭代长度",
                   "显式声明判混沌用的判据(例如 lambda_max > 0 或最大周期)"],
               "feasible_checks": ["curve_present", "r_inf_in_[3.565,3.575]",
                                   "protocol_declared"],
               "not_checkable_as_scalar": "chaotic r 的精确集合"},
        "gt_before": {"method": "最大 Lyapunov 指数 > 0 判混沌",
                      "critical_region": "r > 3.5699 (Feigenbaum 点) 进入混沌带",
                      "method_fingerprint": "λ_max(r) 曲线、分岔图、周期窗口"},
    },
    "res-18-scaling-percolation": {
        "why": "原 GT 给 p_c 的文献值 0.592746, 但 MC 估计没定种子/样本数 -> 不可复算",
        "gt": {"criterion_type": "behavioral",
               "must_report": [
                   "多个格点尺寸(如 L=16,32,64,128)各自的 p_c(L) 与每尺寸的采样次数",
                   "外推到 L→∞ 的 p_c 估计与误差棒",
                   "随机种子(必须写出来, 否则结果不可复现)"],
               "feasible_checks": ["p_c_estimate_in_[0.578,0.607]",
                                   "finite_size_study_with_ge_3_sizes",
                                   "error_bar_reported",
                                   "seed_declared"],
               "not_checkable_as_scalar": "p_c 的文献精确值"},
        "gt_before": {"p_c_infinite": 0.592746, "tolerance": 0.015,
                      "finite_size_sizes": "[32, 64, 128]"},
    },
    "res-19-bifurcation-rossler": {
        "why": "原 GT 给的 c≈4.2/4.6/5.0 依赖参数化与采样密度, 题面没定死 -> 不可复算",
        "gt": {"criterion_type": "behavioral",
               "must_report": [
                   "分岔图(Poincaré 截面交点横坐标 vs 参数 c)",
                   "进入混沌的转变点估计及所用的判据",
                   "至少标出 2 个主要周期窗口及其周期数",
                   "积分步长与瞬态丢弃长度"],
               "feasible_checks": ["bifurcation_diagram_present",
                                   "transition_reported_with_criterion",
                                   "period_windows_ge_2",
                                   "protocol_declared"],
               "not_checkable_as_scalar": "周期窗口的精确 c 位置"},
        "gt_before": {"main_transition": "c≈4.2 附近进入混沌",
                      "period_windows": "c≈4.6, 5.0 附近有周期窗口",
                      "structure": "period-doubling route"},
    },
}

# --------------------------------------------------------------------------
# C. 自洽性修正
# --------------------------------------------------------------------------
CONSISTENCY = {
    "res-11-classify-group-60": {
        "why": ("60 阶群有 13 个同构类, 原 GT 只列 3 个还带 '等/其他' 占位 -> "
                "GT 自身不自洽, 而且没法判对错。改成可验证的事实陈述 + 显式承认"
                "完整分类需要 GAP/Sage"),
        "gt": {"criterion_type": "hybrid",
               "verifiable_facts": [
                   "存在唯一一个 60 阶非交换单群, 且同构于 A5",
                   "给出的任何不变量(如换位子群/元数分布/Sylow 结构)必须能把所列各类分开",
                   "若声称完整分类, 必须说明用了 GAP SmallGroup 或 Sage 的哪条命令"],
               "not_verifiable_here": ("完整同构类清单(13 个)需要 GAP/Sage, "
                                       "本环境无此工具, 复算闸门不做"),
               "explicitly_do_not_claim": "不得把 3 个类说成'全部'"},
        "gt_before": {"classes": ["A5", "C3 ⋊ (C2×C2) 等", "其他"],
                      "primary_one": "A5", "reference": "GAP 的 SmallGroup(60, k) 分类"},
        "verifier": {"pass_condition":
                     "unique non-abelian simple group of order 60 is A5, AND "
                     "any full classification claim names the GAP/Sage command used"},
    },
}


# --------------------------------------------------------------------------
# D. 补齐可交叉核对的字段
# --------------------------------------------------------------------------
# res-09 的 GT 里只有 ratio_tolerance, 没有任何"参考长度", 于是:
#   - 复算层无从核对判据阈值落点
#   - 负控扰动 ratio_tolerance 时判定纹丝不动(闸门自检抓到过)
# 修法: 把生成出来的参考长度写进 GT, 并**显式声明它不是已证明的最优解**。
DATA_BACKED = {
    "res-09-tsp-200": {
        "why": ("GT 里没有参考长度, 导致判据阈值无处落脚, 负控也抓不到。"
                "补上 reference_length(由 make_missing_data.py 生成)并声明它"
                "不是已证明的最优解 —— 于是 '<=1.10 x reference' 比 "
                "'<=1.10 x 真最优' 略严, 对被测模型不利, 这点要写明"),
        "gt": {"criterion_type": "quantitative",
               "ratio_tolerance": 0.1,
               "instance": "data/tsp_200.txt",
               "reference": "data/tsp_200_opt.txt",
               "reference_length": 15.940704,
               "reference_is_proven_optimal": False,
               "reference_method": "nearest-neighbour + 2-opt",
               "length_threshold": 17.534774},
        "method": "tests/tools/reference_recompute.py::_res_09 —— 按坐标重算巡回长度并与文件互验",
    },
}


def apply_to_bank(fname):
    path = os.path.join(BENCH, fname)
    bank = json.loads(io.open(path, encoding="utf-8").read())
    by_id = {it["id"]: it for it in bank["items"]}
    log = []

    for src in (CORRECTIONS, CRITERION_REWRITES, CONSISTENCY, DATA_BACKED):
        for cid, spec in src.items():
            it = by_id.get(cid)
            if it is None:
                log.append((cid, "SKIP", "题库里没有这一题"))
                continue
            entry = {
                "id": cid,
                "kind": ("数值订正" if src is CORRECTIONS else
                         "判据改写" if src is CRITERION_REWRITES else
                         "自洽性修正" if src is CONSISTENCY else "补齐可核对字段"),
                "why": spec["why"],
                "gt_before": spec.get("gt_before"),
                "gt_after": spec.get("gt"),
                "recompute_method": spec.get("method"),
                "by": "mavis 复算闸门 tests/tools/check_reference_truth.py",
            }
            if "gt" in spec:
                it["ground_truth"] = spec["gt"]
            if "prompt" in spec:
                entry["prompt_before"] = it["prompt"]
                it["prompt"] = spec["prompt"]
                entry["prompt_after"] = spec["prompt"]
            if "verifier" in spec:
                entry["verifier_before"] = it.get("verifier")
                it["verifier"].update(spec["verifier"])
            log.append((cid, "OK", entry))      # ★ 追加 entry 本身,
            # 不是 entry["kind"] —— 之前误写 kind, 结果 _corrections 里
            # 存的全是 "数值订正" 这种字符串, 而 GT/prompt/verifier 的改动
            # 照常生效, 于是"改动生效但审计是垃圾"且不报错。

    # 同一 id 重跑时覆盖旧记录, 保证脚本可重入(否则 _corrections 会越积越长)。
    # 用 isinstance 过滤掉历史遗留的非 dict 条目。
    old = {e["id"]: e for e in bank.get("_corrections", [])
           if isinstance(e, dict) and "id" in e}
    for _cid, _st, e in log:
        if _st == "OK" and isinstance(e, dict):
            old[e["id"]] = e
    bank["_corrections"] = list(old.values())
    bank["_corrections_note"] = (
        "本文件含 minimax 复算闸门做的参考答案订正。每条都在 _corrections 里留了"
        "原值、新值、复算路径与出错原因, 可直接对着 tests/tools/reference_recompute.py 核。")

    io.open(path, "w", encoding="utf-8", newline="\n").write(
        json.dumps(bank, ensure_ascii=False, indent=2) + "\n")
    return fname, log


if __name__ == "__main__":
    for fname in ("part1_tool_capability.json", "part2_research_capability.json"):
        name, log = apply_to_bank(fname)
        print(f"=== {name} ===")
        for cid, st, info in log:
            extra = "" if st == "OK" else f" ({info})"
            print(f"  [{st}] {cid}{extra}")
        ok = sum(1 for _, s, _ in log if s == "OK")
        print(f"  合计 {ok}/{len(log)} 条已改\n")