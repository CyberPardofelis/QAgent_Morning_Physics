"""
agent.tools.surveys._parallel — 综述流水线的并发执行基元 (10-02)

为什么要有这个模块
------------------
一轮 42 篇的完整综述要发 200+ 次 LLM 请求。实测 (v60d, 50 篇) 的阶段拆解:

    fact 抽取   49 次 / 1675 秒   每篇论文独立
    命题合并    16 次 / 1022 秒   每个 subject 聚簇独立
    命题成文    26 次 / 1720 秒   每批 claim 独立
    ------------------------------------------------
    小计        91 次 / 4417 秒   占全流程 LLM 等待时间的 59%

这三段原来是三个普通的 `for` 循环, 严格串行, 而批次之间互不依赖。
并发后这 59% 可以压到约 1/4。

并发 vs 合批 —— 这是最关键的区别, 别混为一谈
------------------------------------------
合批把 N 个输入拼成 1 个 prompt, **改了单次调用的输入**。所以实测出了
"多篇合抽丢 2/3 论文"(模型按"每次调用"分配输出, 10 篇合抽只活下来 3 篇)
这类**质量**回归 —— 合批是拿质量换次数。

并发**一个字节都不改输入**: 每篇论文照样单独发一次请求、单独解析、单独返回。
所以质量风险只剩一个 —— **顺序**, 而顺序可以精确还原。

因此本模块的硬约束 (改代码前先读这段)
------------------------------------
**并发只负责"拿到结果", 状态变更一律回主线程按原序做。**

    ✗ 禁止在 worker 里 state["facts"].append(...)
    ✗ 禁止在 worker 里 ws.append_fact(...)
    ✗ 禁止用 as_completed() 收结果后直接 append

原因不是洁癖, 是这几个顺序都是**对外可见的**:
    - _organize.py 的 `seq` 是全局递增的 claim_id 计数器 -> claim_id 变了,
      正文引用编号整体漂移;
    - 成文批次的段落顺序决定最终排版;
    - `_facts.jsonl` 的行序是 fact_id 的分配依据。

所以正确用法是: worker 是个**纯函数**(只发请求、只解析、只返回数据),
主线程拿到 `run_ordered()` 的结果后, 按原始索引逐条做状态变更。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class TaskResult:
    """单个任务的执行结果。**失败不抛**, 由调用方决定怎么处理。

    语义:
        ok=True  -> worker 正常返回, value 是它的返回值
                    (注意: LLM 层把异常转成**错误字符串**返回而不抛,
                     所以 ok=True 不代表内容有效, 那是调用方的事)
        ok=False -> worker 抛了异常, exc 里是原异常
    """

    ok: bool
    value: Any = None
    exc: BaseException | None = None

    def unwrap(self, default: Any = None) -> Any:
        return self.value if self.ok else default


# LLM 层 classify_llm_error 对 429 的返回文案 (见 agent/llm_clients.py):
#   "❌ MiniMax 触发限流, 建议稍后重试。"
# 加上直接抛 RateLimitError 的情况。
_RATE_LIMIT_MARKERS = ("触发限流", "RateLimit", "rate_limit", "429", "Too Many Requests")


def is_rate_limited(value: Any) -> bool:
    """判断一次调用是不是撞了限流。

    `_call_llm` 把异常**转成字符串**返回而不是抛 (agent/tools/llm.py),
    所以光看 `ok` 认不出 429, 必须查内容。
    """
    if isinstance(value, BaseException):
        return any(m in type(value).__name__ for m in _RATE_LIMIT_MARKERS) or "429" in str(value)
    if not isinstance(value, str):
        return False
    return any(m in value for m in _RATE_LIMIT_MARKERS)


def _resolve_workers(max_workers: int | None, n_tasks: int) -> int:
    if max_workers is None:
        from agent.tools.config import get_survey_max_workers
        max_workers = get_survey_max_workers()
    # 任务比 worker 还少时别空转
    return max(1, min(int(max_workers), n_tasks))


def run_ordered(
    tasks: Sequence[T] | Iterable[T],
    worker: Callable[[T], R],
    *,
    max_workers: int | None = None,
    desc: str = "",
    max_retries: int = 2,
    backoff_base: float = 3.0,
) -> list[TaskResult]:
    """并发跑 `worker`, 但结果**严格按 tasks 的原始顺序**返回。

    这是本模块存在的唯一理由: 顺序必须和串行跑的时候一模一样。
    `ThreadPoolExecutor.map` 满足这个性质; `as_completed` 不满足, 别用。

    Args:
        tasks:      待处理项
        worker:     **纯函数**, 只发请求 / 解析 / 返回数据, 不要改任何共享状态
        max_workers: None = 走 config 的 QAGENT_SURVEY_MAX_WORKERS (默认 4);
                     1 = 退回串行(测试 / 排障用)
        desc:       日志前缀, 便于定位是哪一段并发
        max_retries: 撞限流时的重试次数 (指数退避)
        backoff_base: 首次退避秒数, 之后 ×2

    Returns:
        与 tasks 等长、同序的 TaskResult 列表。
    """
    items = list(tasks)
    if not items:
        return []
    n_workers = 1 if len(items) == 1 else _resolve_workers(max_workers, len(items))

    def _one(item: T) -> TaskResult:
        delay = backoff_base
        for attempt in range(max_retries + 1):
            last = attempt >= max_retries
            try:
                val = worker(item)
            except BaseException as e:      # noqa: BLE001 — 单项失败不拖垮整批
                if last or not is_rate_limited(e):
                    return TaskResult(ok=False, exc=e)
                logger.warning("[parallel:%s] 限流, %.1fs 后重试 (%d/%d)",
                               desc, delay, attempt + 1, max_retries)
            else:
                if last or not is_rate_limited(val):
                    return TaskResult(ok=True, value=val)
                logger.warning("[parallel:%s] 限流, %.1fs 后重试 (%d/%d)",
                               desc, delay, attempt + 1, max_retries)
            time.sleep(delay)
            delay *= 2
        # 到不了这里: 循环最后一次一定 return
        raise AssertionError("unreachable")

    if n_workers == 1:
        # 完全退回串行 —— 除了省掉线程池开销, 行为与并发路径逐位一致
        return [_one(it) for it in items]

    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix=f"survey-{desc or 'task'}") as pool:
        # map 保序, 这是本函数全部意义所在
        return list(pool.map(_one, items))


__all__ = ["TaskResult", "is_rate_limited", "run_ordered"]
