"""
tests/test_user_settings_isolation.py — 锁住"跑测试不能改用户真实设置"(10-06 加)

**为什么这个文件存在**
实测过: 跑一次 `pytest tests/test_cost_guard.py`, 用户真实的 LLM 日计数器从
**4242 变成 1**。测试全绿, 但用户已经花掉的额度显示归零 —— 这是最难发现的一类
污染, 因为它不报错、不失败, 只是把数字改没了。

根因比"忘了还原"更难看一点: 那个 fixture 里有一套看起来很专业的隔离 ——
`QSettings.setDefaultFormat(IniFormat)` + `setPath(...)`。**它是空操作**。
Qt 的 `QSettings(org, app)` 双参构造写死用 NativeFormat, 不看 `setDefaultFormat`
(后者只对无参 `QSettings()` 生效), 所以 `guard.reset()` 一直直接写 Windows
注册表。**假隔离比没隔离更坏**: 它让人以为已经隔离了。

修法在 conftest 的 `_protect_user_settings`(快照/还原 cost_guard 两个键)。

**为什么必须是子进程测试**
要验的是"**跑完另一个 pytest 进程**之后设置还在" —— 同一个进程里写完再读,
读到的永远是当前值, 验不出任何东西。所以这里真的起一个 pytest 子进程。
(这也是这个文件唯一的测试就跑 1.5 秒的原因, 换来的是一条真的回归锁。)
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)

from agent.cost_guard import CostGuard  # noqa: E402
from config.settings import get_settings  # noqa: E402

# 一个一眼就不像真实用量的哨兵值: 真跑一轮综述也就几百次
_SENTINEL = 987654


def _read_count() -> int:
    v = get_settings().value(CostGuard.KEY_COUNT, 0)
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _write_count(n: int) -> None:
    s = get_settings()
    s.setValue(CostGuard.KEY_DATE, date.today().isoformat())
    s.setValue(CostGuard.KEY_COUNT, n)
    s.sync()


def test_running_tests_does_not_touch_real_daily_counter():
    """起一个 pytest 子进程跑 test_cost_guard.py, 计数器必须原样不动。

    ★ 为什么不用 987654 当哨兵了 ★
    第一版用的是 987654, 靠 `finally` 还原。实测 2026-10-08 出了事:
    跑测试时同时在跑 LLM 评估, 进程被打断, `finally` 没执行完,
    哨兵值就**留在了用户真实的日计数器里** —— 于是真实用量显示成
    "987654/2000, 100% 已阻断", 而实际当天只用了 259 次。
    一眼就能看出是假的, 但它会真的把人挡在门外, 且极难联想到是测试。

    现在的做法: **哨兵值改成「真实值 + 一个不可能自然发生的偏移」**,
    并在写入前先备份、写入后立刻校验。这样即使中途被打断, 下次跑这个
    测试时也能从「before 读出来的不正常」立刻发现污染, 而不是默默生效。

    更根本的一条: 真实值不该被测试碰。conftest 的 _protect_user_settings
    才是主力防线, 这条测试是验那条防线在的。
    """
    before = _read_count()
    sentinel = before + 987654     # 偏移而非绝对值, 撞上真实用量的概率极低
    _write_count(sentinel)
    try:
        # 写入后立刻确认一次: 这一步失败说明连写都没写进去, 别往下走
        assert _read_count() == sentinel, (
            f"哨兵值没写进去, 设置存储可能坏了: "
            f"期望 {sentinel}, 实际 {_read_count()}"
        )
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_cost_guard.py",
             "-q", "-p", "no:cacheprovider", "--no-header"],
            cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300)
        assert proc.returncode == 0, (
            f"子进程测试没过, 没法验隔离: {proc.stdout[-800:]}")
        after = _read_count()
        assert after == sentinel, (
            f"跑测试把用户真实日计数器改了: {sentinel} → {after}\n"
            f"(这是静默污染: 测试全绿, 但用户额度显示被清掉了)\n"
            f"注意 before={before}, 如果你看到 {sentinel} 出现在设置里, "
            f"手动改回 {before} 即可。"
        )
    finally:
        # 显式还原 —— 即便这层断言炸了, 也不给用户留一个假数字。
        # (conftest 的 _protect_user_settings 退出时还会再兜一次)
        _write_count(before)


def test_protect_fixture_covers_both_guard_keys():
    """保护 fixture 必须同时管住 date 和 count —— 只管 count 会写出错日期。"""
    conftest_src = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "_protect_user_settings" in conftest_src, \
        "conftest 里没有 _protect_user_settings —— 保护没了"
    body = conftest_src.split("def _protect_user_settings", 1)[1].split("\ndef ", 1)[0]
    # 认**符号名**而不是字面量 —— conftest 里写的是 CostGuard.KEY_DATE,
    # 不是 "cost_guard/date"; 认字面量的话这条测试会因为写法不同而假绿。
    assert "CostGuard.KEY_DATE" in body, f"没保护 {CostGuard.KEY_DATE}"
    assert "CostGuard.KEY_COUNT" in body, f"没保护 {CostGuard.KEY_COUNT}"


def test_the_fake_qsettings_isolation_is_gone():
    """锁住"假隔离不许回来"。

    `setDefaultFormat(IniFormat)` + `setPath(...)` 那套看着专业、实际是空操作 ——
    只要它还在, 下一个人就会以为测试已经隔离了。解释它为什么无效的 docstring
    可以留(那是反面教材), 但**代码里不许再出现这两个调用**。

    用 ast 而不是 grep: docstring 是 ast.Constant, 天然进不来 —— 之前按行
    剥 `#` 注释那套会把整段 docstring 当代码, 结果解释文字里出现
    `setDefaultFormat` 就判红, 自己把自己判失败了。
    """
    import ast

    tree = ast.parse((ROOT / "tests" / "test_cost_guard.py").read_text(
        encoding="utf-8"))
    used = {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)}
    for banned in ("setDefaultFormat", "setPath"):
        assert banned not in used, \
            f"又加回了那个对 QSettings(org, app) 无效的 {banned} 隔离"
