"""research 下的**路线层**: 同一个底座, 不同的编排策略。

## 为什么单独开一个包

路线是**可对照的实验变量**(part1/part2 就是这么测的), 而
`roles/` 里放的是**被所有路线共用的能力**(explorer / worker / verifier)。

把它们分开是为了让"改了哪一层"一眼可见:

  · 改 `roles/`   = 改了**所有路线**的底座 —— 历史分数全部不可比
  · 改 `routes/`  = 只改了**这一条**路线 —— 别的路线不受影响

## 现有路线

    B  = roles/orchestrator.solve()      Explorer -> 单 Worker(串行)
    C  = run_part2_tools.run_c()          单函数 ReAct 循环(不走 roles)
    D  = routes/d.py                      Explorer -> N 条 lane -> N Worker(**并发**)
                                          且接受 hints(调用方可指定方法)

★ B 的代码在 `routes/` 里一行都没有 ★ D 复用 `roles/` 的能力,
  但**不修改** `roles/` 的任何文件 —— 这样 B 的 40 题实测数据仍然可比。
"""
from __future__ import annotations

__all__ = ["d"]