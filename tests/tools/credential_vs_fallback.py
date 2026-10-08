"""★ 凭证机制能否取代六道兜底门? —— 拿真实假反例做判决实验 ★

## 为什么上一轮跑测回答不了这个问题

V2 验证轮跑完 39 条证据里, 代码自称 REFUTED 的有 **0 条** —— 模型在那些题上
根本没走到 REFUTED。所以"两轮分数 18/20 vs 19/20"只证明了凭证机制**没弄坏**东西,
完全没证明它**够用**。差的那一分是 hard-017(100皇后)的模型随机性,
六道门和凭证门都没开过火。

## 这个实验怎么做

用**历史上真实产生过的假 REFUTED**(都已逐字转录成夹具), 对比两种配置:
  A. 只开凭证门     —— 凭证机制能不能独立拦住它们?
  B. 只开六道兜底门   —— 它们各自本来是被谁拦住的?

判据很直接: A 拦住 = 凭证可取代兜底; A 漏掉 = 凭证不够, 六道门还得留着。
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import agent.tools.research._provenance as P  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_proved_machine_gate as FIX  # noqa: E402

#: 每条都是**真实翻车现场**逐字转录(stdout 原文), 不是编的样例。
#: 夹具放在门测试模块里, 就是因为它们同时是"咬合测试"的输入 —— 两边共用一份,
#: 免得判决实验用的样例和门测试用的样例悄悄漂移。
REAL_FALSE_REFUTATIONS = [
    ("p(100) 分拆数", FIX.PARTITION_STDOUT, FIX.PARTITION_CODE,
     "两路权威方法算出正确值, 第三路写坏 -> 宣布 REFUTED"),
    ("R(3,4)=9", FIX.RAMSEY_STDOUT, "x=1\n",
     "各部分都判 INCONCLUSIVE -> 汇总成 REFUTED"),
    ("Erdos-Straus 计数", FIX.ERDOS_STRAUS_STDOUT, "x=1\n",
     "声明 997 个失败只列 20 个 -> 宣布 REFUTED"),
    ("Pell-61", FIX.PELL_STDOUT, "x = 1\n",
     "把正解 1766319049 标成反例, 同一份输出又说'方程成立? True'"),
    ("Fibonacci 矩阵", FIX.FIB_STDOUT, FIX.FIB_CODE,
     "n=1..20 all_passed=True 却判 REFUTED"),
]


def _res(stdout: str) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr="", ok=True, rejected=False,
                           timed_out=False, error_type="", code="")


def credential_only(stdout: str) -> bool:
    """配置 A: 只有凭证门。"""
    return P._refuted_without_credential(stdout) is not None


def fallback_only(stdout: str) -> str | None:
    """配置 B: 只有六道兜底门, 返回命中的是哪一道。"""
    for kind, label in P._CONTRADICTIONS:
        if P._contradiction_msg(kind, stdout) is not None:
            return label
    return None


def main() -> None:
    print("=" * 78)
    print("凭证机制 vs 六道兜底门 —— 拿真实假反例做判决")
    print("=" * 78)
    miss_a = miss_b = 0
    for name, stdout, code, why in REAL_FALSE_REFUTATIONS:
        a = credential_only(stdout)
        b = fallback_only(stdout)
        if not a:
            miss_a += 1
        if b is None:
            miss_b += 1
        print(f"\n{name}  ——  {why}")
        print(f"   凭证门(新)      : {'拦住' if a else '★ 漏掉'}")
        print(f"   六道兜底门(旧)   : {'拦住 —— ' + b if b else '★ 漏掉'}")
        _ = _res(stdout), code      # 保留原始形态, 便于人工核对

    n = len(REAL_FALSE_REFUTATIONS)
    print("\n" + "=" * 78)
    print(f"真实假反例 {n} 条")
    print(f"  凭证门漏掉   {miss_a}")
    print(f"  兜底门漏掉   {miss_b}")
    print("=" * 78)
    if miss_a == 0 and miss_b == 0:
        print("★ 两层都覆盖了全部真实假反例 → 冗余但互不替代, "
              "建议**两层都留**(凭证层更通用, 兜底层给的判词更具体)")
    elif miss_a == 0:
        print("★ 凭证门覆盖了全部真实假反例 → 可以取代六道兜底门")
    else:
        print(f"★ 凭证漏掉 {miss_a} 条 → **六道兜底门必须留着**")
        print("  漏掉的这些形态凭证制接不住, 因为它们确实'指名了违反对象并给了值',")
        print("  只是那个对象和值是自相矛盾的 —— 那是**一致性**问题, 不是**存在性**问题。")


if __name__ == "__main__":
    main()