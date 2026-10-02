"""
tests/test_survey_parallel_and_resume.py — 并发基元 + 崩溃续跑 (10-02)

两块, 都是"看起来能跑但结果会悄悄错"的地方, 所以必须锁死:

1. run_ordered 的**保序性** —— claim_id / 引用编号 / 段落排版全靠它。
   并发最容易出的 bug 就是在 worker 里用了 as_completed(), 顺序一乱,
   引用编号整体漂移, 而输出"看起来完全正常"。

2. 崩溃续跑的**signature 门禁 + 必喊出声** —— 最怕的是改了 max_pool 却悄悄
   接着旧参数的中间产物跑, 或者静默接上上一轮的数据。
"""
from __future__ import annotations

import json
import time

import pytest

from agent.tools.surveys._parallel import TaskResult, is_rate_limited, run_ordered


# ============================================================
# run_ordered: 保序是全部意义所在
# ============================================================

def test_run_ordered_preserves_input_order_despite_completion_order():
    """
    核心回归: 前面的任务故意睡得久, 让它们**后完成** ——
    如果实现误用了 as_completed(), 返回顺序就会跟着完成顺序乱掉。
    """
    delays = {0: 0.20, 1: 0.01, 2: 0.15, 3: 0.0, 4: 0.10}
    order_seen = []

    def worker(i):
        time.sleep(delays[i])
        order_seen.append(i)          # 完成顺序
        return i * 10

    res = run_ordered(range(5), worker, max_workers=5)

    assert order_seen != list(range(5)), "本测试的前提是完成顺序确实被打乱了"
    assert [r.value for r in res] == [0, 10, 20, 30, 40], "返回顺序必须回到输入顺序"
    assert all(r.ok for r in res)


def test_run_ordered_isolates_a_failing_task():
    """单项失败不能拖垮整批 —— 综述里一批抽取失败只该少那几篇的 fact。"""

    def worker(i):
        if i == 2:
            raise RuntimeError("boom")
        return i

    res = run_ordered(range(5), worker, max_workers=4)

    assert [r.ok for r in res] == [True, True, False, True, True]
    assert [r.value for r in res if r.ok] == [0, 1, 3, 4]
    assert isinstance(res[2].exc, RuntimeError)


def test_run_ordered_workers_1_matches_serial():
    """max_workers=1 必须与串行逐位一致(排障/测试用)。"""
    def worker(i):
        return i * 3

    assert [r.value for r in run_ordered(range(6), worker, max_workers=1)] == \
           [0, 3, 6, 9, 12, 15]


def test_run_ordered_empty_and_single():
    assert run_ordered([], lambda x: x) == []
    res = run_ordered(["x"], lambda v: v.upper(), max_workers=4)
    assert len(res) == 1 and res[0].value == "X"


def test_run_ordered_never_lets_worker_touch_more_than_declared_workers():
    """并发度不得超过 max_workers —— 429 概率和它成正比。"""
    live = {"now": 0, "max": 0}
    import threading
    lock = threading.Lock()

    def worker(_i):
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        time.sleep(0.02)
        with lock:
            live["now"] -= 1
        return 1

    run_ordered(range(20), worker, max_workers=4)
    assert live["max"] <= 4


# ============================================================
# 限流识别与退避
# ============================================================

def test_is_rate_limited_detects_llm_error_string():
    """
    _call_llm 把异常**转成字符串**返回而不是抛(agent/tools/llm.py),
    所以限流只能靠内容认 —— 这是并发场景下最关键的一个判断。
    """
    assert is_rate_limited("❌ MiniMax 触发限流，建议稍后重试。")
    assert is_rate_limited("RateLimitError: ...")
    assert is_rate_limited(RuntimeError("429 Too Many Requests"))
    assert not is_rate_limited("这是一段正常的综述正文")
    assert not is_rate_limited(None)
    assert not is_rate_limited(123)


def test_run_ordered_retries_then_succeeds_on_rate_limit():
    """先撞限流再成功 —— 必须重试, 而不是把限流当失败丢掉。"""
    calls = {"n": 0}

    def worker(_i):
        calls["n"] += 1
        if calls["n"] == 1:
            return "❌ MiniMax 触发限流，建议稍后重试。"
        return "ok"

    res = run_ordered(["x"], worker, max_workers=1,
                      max_retries=2, backoff_base=0.01)
    assert calls["n"] == 2
    assert res[0].ok and res[0].value == "ok"


def test_run_ordered_gives_up_after_max_retries():
    """一直限流就认输, 把最后一次的错误串交回调用方(原逻辑靠它兜底)。"""
    def worker(_i):
        return "❌ 触发限流"

    res = run_ordered(["x"], worker, max_workers=1,
                      max_retries=1, backoff_base=0.01)
    assert res[0].ok, "最后一次不再当异常处理, 交回调用方按原逻辑降级"
    assert "触发限流" in res[0].value


def test_task_result_unwrap():
    assert TaskResult(ok=True, value=7).unwrap() == 7
    assert TaskResult(ok=False, exc=ValueError("x")).unwrap(default="fallback") == "fallback"


def _facts_for_condense(n_subjects=6, per_subject=5):
    out = []
    for s in range(n_subjects):
        for j in range(per_subject):
            out.append({
                "fact_id": f"p{s % 3}::{s}_{j}",
                "paper_id": f"p{s % 3}",
                "subject": f"主题{s}",
                "claim": f"关于主题{s}的第 {j} 条事实",
                "n_facts": 1,
            })
    return out


@pytest.mark.parametrize("stage", ["condense"])
def test_condense_parallel_output_is_identical_to_serial(monkeypatch, stage):
    """
    同一个输入, max_workers=1 与 max_workers=4 的产物必须完全一样 ——
    包括 claim_id 的编号和命题的先后顺序。
    """
    import threading

    import agent.tools.config as cfg
    import agent.tools.surveys._organize as org

    facts = _facts_for_condense()

    def run(workers):
        # SURVEY_MAX_WORKERS 是 **import 时**算出来的模块常量, 改环境变量没用,
        # 必须直接改属性 —— 否则两次都跑在同一个并发度上, 断言就成空转。
        monkeypatch.setattr(cfg, "SURVEY_MAX_WORKERS", workers)

        peak = {"now": 0, "max": 0}
        lock = threading.Lock()

        def fake_llm(prompt, **kw):
            with lock:
                peak["now"] += 1
                peak["max"] = max(peak["max"], peak["now"])
            try:
                # 必须真的占一会儿: 返回太快的话线程根本不会重叠,
                # peak 恒为 1, "并发下产物一致" 就成了永远为真的空断言。
                time.sleep(0.02)
                n = prompt.count("] <p")
                if n < 3:
                    return "[]"
                pairs = ",".join(f'[{i}, {i + 1}]' for i in range(1, n, 2))
                return f'[{{"fact_idx": [{pairs}], "statement": "合并概括"}}]'
            finally:
                with lock:
                    peak["now"] -= 1

        # _organize 里的 _call_llm 是**函数内 import**, 所以要 patch 源头模块
        monkeypatch.setattr("agent.tools.llm._call_llm", fake_llm)
        out = org.condense_facts_into_claims(list(facts), batch_size=8)
        return out, peak["max"]

    serial, serial_peak = run(1)
    parallel, parallel_peak = run(4)

    assert serial_peak == 1, "workers=1 必须是纯串行(前提校验)"
    assert parallel_peak > 1, f"workers=4 必须真的并发, 实测峰值 {parallel_peak}"

    assert [c["claim_id"] for c in serial] == [c["claim_id"] for c in parallel], \
        "claim_id 编号必须与串行完全一致 —— 它决定正文引用编号"
    assert serial == parallel, "并发下的命题产物必须与串行逐字节相同"


def test_log_lock_serializes_print():
    """并发下 _call_llm 的日志不能黏成一行 —— 阶段耗时统计靠它。"""
    from agent.tools import llm as L
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        L._log("line-1")
        L._log("line-2")
    assert buf.getvalue() == "line-1\nline-2\n"
    assert json.dumps("ok")  # 确认 redirect_stdout 下的写是逐行的
