"""★ 评分契约的两条硬规矩, 钉成回归测试 ★

这两条都不是"某个 verifier 写错了", 而是**评分契约本身**的性质,
所以要在这里测, 而不是分散到 20 个 verifier 里各测一遍。

## 规矩 1: 只有判对才有证书

打分器 `score_certification` 的实现是: `certificate` 非空 -> 1.0。
它**不看** `final_correct`。所以只要失败诊断是一段非空字符串, 就会拿满分。

实测(2026-10-09): 20 个 verifier 里有 **15 个**在 `correct=False` 时
仍然返回非空的失败诊断串("FAILED: 未提交制品" / "artifact_present: ...")。
于是 certification 这一维**方向完全反了** —— 答错的题拿满分证书。

修法在适配器一处: `tr.certificate = v["certificate"] if v["correct"] else None`。
失败诊断不丢, 留在报告的 `certificate` 字段里供人看, 但不进评分。

## 规矩 2: certificate 不能从失败路径混进来

除了适配器, 这里还直接验一遍: 打分器拿到一个"判错 + 非空诊断串"的轨迹时,
必须给 0 分。这样即使将来有人改了适配器, 这里也会拦下。
"""
from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import reward_scorer as RS  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    task = RS.TaskSpec(id="t", prompt="p", requires_certificate=True)

    print("=" * 72)
    print("评分契约: 失败诊断不能当证书")
    print("=" * 72)

    print("\n>>> 规矩 1: 判错 + 失败诊断串 -> 适配器必须把 certificate 置空")
    import run_r_scored as RR

    for diag in [
        "FAILED: solutions 不是非空列表",
        "artifact_present: 未提交 ANSWER 制品",
        "FAILED: 魔方没有被还原; 步数 4 != BFS 最优 6",
    ]:
        v = {"correct": False, "certificate": diag}
        got = RR.cert_for_trace(v)
        check(f"判错 -> certificate=None  ({diag[:32]})",
              got is None, f"实得 {got!r}")
        # 走完整链路: 把适配器的结果喂给打分器
        tr = RS.AgentTrace(task_id="t", final_correct=False, certificate=got)
        s = RS.score_certification(tr, task)
        check(f"  → certification 实得 {s} (应为 0)", s == 0.0)

    print("\n>>> 规矩 2: 判对 + 真证书 -> certification 才是 1")
    v = {"correct": True, "certificate": "Lean 编译通过且无 sorry; 2^5 mod 5 == 2"}
    got = RR.cert_for_trace(v)
    check("判对时证书原样保留", got == "Lean 编译通过且无 sorry; 2^5 mod 5 == 2")
    tr = RS.AgentTrace(task_id="t", final_correct=True, certificate=got)
    check("  → certification=1", RS.score_certification(tr, task) == 1.0)

    print("\n>>> 边界: 判对但证书为空 -> 也不能凭空给 1")
    tr = RS.AgentTrace(task_id="t", final_correct=True, certificate=None)
    check("certification < 1", RS.score_certification(tr, task) < 1.0,
          f"实得 {RS.score_certification(tr, task)}")

    print("\n>>> 反向检查: 规则本身没错, 是用法错了")
    tr = RS.AgentTrace(task_id="t", final_correct=False, certificate="诊断")
    check("打分器对非空串确实给 1.0(所以必须由适配器挡住)",
          RS.score_certification(tr, task) == 1.0,
          "—— 这条不是 bug, 它就是规则; 责任在调用方")

    print("\n" + "=" * 72)
    if FAILS:
        print(f"✗ {len(FAILS)} 项不通过: " + "; ".join(FAILS))
        return 1
    print("✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())