"""agent.tools.research.roles —— Explorer / Worker / Verifier 三种子工作类型。

## 为什么是"子工作类型"而不是"多 Agent"
它们是**同一个模型、三份互不可见的上下文**。真正拉开距离的是上下文隔离,
不是模型不同 —— 同模型在同上下文里会自欺(见 verifier.py 的诚实边界)。

## 为什么必须有 Explorer
旧结构里 `research_verify(claim, how="symbolic")` 一次调用同时"选方法 + 写代码",
旁边还挂着一张由题面算出来的方法表。**选型被提前做掉了**, 剩下照做。

## 这里的每个模块都不含"题型 → 方法"的映射
- catalog.py  只有工具本身的能力边界, 对所有问题一视同仁
- explorer.py 只把题面 + 目录 + 历史交给模型, 让它自己判断
- worker.py   只执行, **schema 里没有 verdict**
- verifier.py 只看题面 + 制品, 自己重算
- orchestrator.py 只做循环/预算/路由, 路由依据是 Explorer 自己说的 next_action

`check_no_hardcoded_routing.py` 静态守着这条线。
"""
from .catalog import build_catalog, render_catalog          # noqa: F401
from .orchestrator import solve                             # noqa: F401
from ._util import Budget                                   # noqa: F401

__all__ = ["solve", "Budget", "build_catalog", "render_catalog"]