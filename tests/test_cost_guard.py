"""
test_cost_guard — CostGuard 单元测试 (2026-09-21 加)

覆盖:
1. 基础行为: check_and_increment 计数 + 80% / 100% 阈值
3. 跨天重置
4. reset()
5. thread-safety (锁)
6. 集成: controller 的 LLM 调用计数问题 (这是用户报告"感觉不到意义"的根因)

为什么重要: 用户反馈"要么感觉不到意义 要么工作方式有问题"
- cost_guard 默认 ceiling=50 (单 run_agent_stream 算 1)
- 但 controller while 循环每次 round 调 LLM 一次
- 实际 LLM 调用次数 >> cost_guard 计数次数
- 100% 阻断永远不发生
- 这是设计层面的 bug
"""
import os
import sys
import threading

import pytest

# 设置 offscreen Qt 平台 (跟 conftest.py 一样)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

# 确保 QApplication 已创建 (QSettings 在 Windows 用注册表需要)
_app = QApplication.instance() or QApplication(sys.argv)

from agent.cost_guard import CostGuard  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_cost_guard():
    """每个测试用独立 QSettings Organization/AppName, 避免污染全局"""
    # 用临时 organization 隔离
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        QStandardPaths_writableLocation(),  # noqa: F821
    )
    # 用 pytest tmp_path 风格的 org/app name
    guard = CostGuard()
    guard.reset()  # 每次测试前重置
    yield guard


def QStandardPaths_writableLocation():  # helper, 避免 lint 报错
    """返回临时目录路径用于 QSettings"""
    import tempfile
    return tempfile.mkdtemp(prefix="qagent_cost_guard_test_")


@pytest.fixture(autouse=True)
def _isolated_config():
    """每个测试前后**快照并还原** cost_ceiling_daily。

    ⚠️ 9-30 修: 原来 cleanup 是硬编码 `set_cost_ceiling_daily(50)`。
    那是把"默认值"当成了"用户当前值" —— 但 cost_ceiling_daily 存在 QSettings 里,
    是**用户设置**, 跑一次测试就把用户的值改掉了。实测用户被改成 50
    (代码默认值其实是 100), 害得完整综述验证撞上成本守卫硬阻断。

    正确做法: 进入测试前存下当前值, 退出时原样写回。测试改的应该是测试自己的
    副本, 不是用户设置。
    """
    from config._config import Config
    original = Config.get_cost_ceiling_daily()
    yield
    # cleanup: 原样还原用户当时的值, 而不是某个"默认值"
    Config.set_cost_ceiling_daily(original)


class TestCostGuardBasic:
    """基础行为: check_and_increment + 计数"""

    def test_first_call_under_limit(self, _isolated_cost_guard):
        """ceiling=10, 第一次调用返回 (True, '')"""
        guard = _isolated_cost_guard
        is_ok, msg = guard.check_and_increment(ceiling=10)
        assert is_ok is True
        assert msg == ""
        assert guard.current_count() == 1

    def test_subsequent_calls_increment(self, _isolated_cost_guard):
        """连续 5 次调用, count 累加到 5"""
        guard = _isolated_cost_guard
        for i in range(5):
            is_ok, msg = guard.check_and_increment(ceiling=10)
            assert is_ok is True
            assert msg == ""
        assert guard.current_count() == 5

    def test_80_percent_warns(self, _isolated_cost_guard):
        """ceiling=10, 第 8 次 (80%) 警告但仍 ok=True"""
        guard = _isolated_cost_guard
        # 跑 7 次, 全部正常
        for i in range(7):
            is_ok, msg = guard.check_and_increment(ceiling=10)
            assert is_ok is True
            assert msg == ""
        # 第 8 次 (8/10 = 80%) 警告
        is_ok, msg = guard.check_and_increment(ceiling=10)
        assert is_ok is True  # 仍放行
        assert "80%" in msg or "80" in msg
        # count 应该是 8
        assert guard.current_count() == 8

    def test_100_percent_blocks_and_increments_to_ceiling(self, _isolated_cost_guard):
        """ceiling=10, 第 10 次到 ceiling 时返回 (False, 100% msg)"""
        guard = _isolated_cost_guard
        # 跑 9 次, 全部正常
        for i in range(9):
            is_ok, msg = guard.check_and_increment(ceiling=10)
            assert is_ok is True
        # 第 10 次: 到 ceiling, block
        is_ok, msg = guard.check_and_increment(ceiling=10)
        assert is_ok is False
        assert "100%" in msg or "\u4e0a\u9650" in msg or "ceiling" in msg.lower()
        # count 应该是 10
        assert guard.current_count() == 10

    def test_over_ceiling_blocks_without_increment(self, _isolated_cost_guard):
        """ceiling=10 触达后, 再调用仍然 block 且不递增"""
        guard = _isolated_cost_guard
        # 跑到 10 (block)
        for i in range(10):
            guard.check_and_increment(ceiling=10)
        assert guard.current_count() == 10
        # 再调 3 次, 都应该 block, count 仍 10
        for i in range(3):
            is_ok, msg = guard.check_and_increment(ceiling=10)
            assert is_ok is False
        assert guard.current_count() == 10  # 没递增


class TestCostGuardReset:
    """reset() + 跨天重置"""

    def test_reset_clears_count(self, _isolated_cost_guard):
        """reset() 后 count 归 0"""
        guard = _isolated_cost_guard
        for i in range(5):
            guard.check_and_increment(ceiling=10)
        assert guard.current_count() == 5
        guard.reset()
        assert guard.current_count() == 0

    def test_cross_day_reset(self, _isolated_cost_guard):
        """模拟跨天: date 改到昨天, 下次调用应该 reset + increment 1"""
        from config.settings import get_settings
        guard = _isolated_cost_guard
        for i in range(5):
            guard.check_and_increment(ceiling=10)
        assert guard.current_count() == 5
        # 模拟"昨天用了 5 次"
        settings = get_settings()
        import datetime
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        settings.setValue(CostGuard.KEY_DATE, yesterday)
        settings.setValue(CostGuard.KEY_COUNT, 5)
        # 下次调用: 应该 reset 后 +1
        is_ok, msg = guard.check_and_increment(ceiling=10)
        assert is_ok is True
        assert msg == ""  # 1/10 < 80%
        assert guard.current_count() == 1


class TestCostGuardThreadSafety:
    """thread-safety: 多线程并发 check_and_increment"""

    def test_concurrent_increments_no_lost_updates(self, _isolated_cost_guard):
        """10 个线程各调 10 次, count 应该累加到 100"""
        guard = _isolated_cost_guard

        def worker():
            for _ in range(10):
                guard.check_and_increment(ceiling=200)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 不丢更新 = 100
        assert guard.current_count() == 100


class TestCostGuardConfigIntegration:
    """Config.get_cost_ceiling_daily() 集成"""

    def test_default_ceiling_when_unset(self):
        """**代码默认值**是 100 —— 键不存在时该返回它。

        ⚠️ 9-30 修: 原来这个测试叫 test_default_ceiling_is_50 并断言 == 50, 但它
        读的是 QSettings 里**用户持久化的值**, 不是代码默认值。两个后果:
          1. 断言的数是错的 —— 代码默认 9-27 就从 50 上调到 100 了
          2. 它一直"通过"只是因为另一个测试的 cleanup 把用户设置写死成 50
        所以想验默认值必须**先把键删掉**, 否则验的是用户当前设置。
        还原交给 autouse 的 _isolated_config (它存的是进入测试前的用户真实值)。
        """
        from config._config import Config
        s = Config._settings()
        s.remove("cost_ceiling_daily")   # 清掉持久化值, 逼它走代码默认值分支
        assert Config.get_cost_ceiling_daily() == 400

    def test_set_ceiling_then_get(self, _isolated_cost_guard):
        """set_cost_ceiling_daily(100) → get_cost_ceiling_daily() == 100"""
        from config._config import Config
        Config.set_cost_ceiling_daily(100)
        assert Config.get_cost_ceiling_daily() == 100
        # 还原交给 autouse 的 _isolated_config (它存的是进入测试前的用户真实值)

    def test_set_invalid_ceiling_clamps_to_default(self):
        """
        set_cost_ceiling_daily("invalid") -> 回落到默认值。

        9-27: 这里原本硬编码 assert == 50, 默认值从 50 上调到 100 后直接失败。
        说明这类"把默认值写进断言"的测试每调一次默认值就要改一次 ——
        改成断言新的默认值, 并在注释里记下为什么改。
        """
        from config._config import Config
        Config.set_cost_ceiling_daily("not_a_number")  # type: ignore
        assert Config.get_cost_ceiling_daily() == 400


class TestCostGuardRealUsageIssue:
    """⚠️ 9-21 体检发现的设计 bug: 计数单位不对

    2026-09-21 P0 fix: cost_guard.check_and_increment() 已在 4 处落地 —
    run_agent_stream 头部 (算 session start) + while 循环每轮 (按 LLM 调用计) +
    _stream_openai_simple + call_model (覆盖 OpenAI 模式 fall-back 路径)。

    一次完整对话 cost = 1 (启动) + N (round 数) = 实际 LLM 调用数。
    ceiling=50 现在能正常触达, 用户"感觉不到意义"问题修复。

    本组 test 验证 controller 代码静态层面已包含 check_and_increment ——
    防止未来重构时退回"只算 1 次"的旧设计。
    """

    def test_run_agent_stream_has_check_in_while_loop(self):
        """✅ controller.run_agent_stream while 循环每轮都调 check_and_increment

        用 AST 静态分析验证 — cost_guard 计数应反映实际 LLM 调用次数,
        不是"按对话次数"算。
        """
        import ast
        import inspect
        from agent.controller import run_agent_stream

        source = inspect.getsource(run_agent_stream)
        tree = ast.parse(source)

        # 找 while 循环 (True:) 里的 check_and_increment 调用
        while_has_check = False
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                # 检查 while body 里有没有 check_and_increment
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Attribute) and func.attr == "check_and_increment":
                            while_has_check = True
                            break

        assert while_has_check, (
            "controller.run_agent_stream while 循环必须调 check_and_increment, "
            "否则 cost_guard 还是按 run_agent_stream 计数 (设计 bug 已修, 防止退回)"
        )

    def test_call_model_invokes_cost_guard(self):
        """✅ controller.call_model() 现在走 cost_guard

        之前完全不走 cost_guard, 用户"感觉不到意义"的根因之一。
        修后: call_model 头部 check_and_increment, 阻断时返错误信息 string。
        """
        import inspect
        from agent import controller
        source = inspect.getsource(controller.call_model)
        assert "check_and_increment" in source, (
            "controller.call_model 必须调 check_and_increment, 否则非流式调用不计费"
        )
        assert "cost_guard" in source, "controller.call_model 应引入 cost_guard"

    def test_stream_openai_simple_invokes_cost_guard(self):
        """✅ controller._stream_openai_simple() 现在走 cost_guard

        OpenAI 模式 fall-back 路径, 修后: 头部 check_and_increment, 阻断时 yield NEW_BUBBLE。
        """
        import inspect
        from agent import controller
        source = inspect.getsource(controller._stream_openai_simple)
        assert "check_and_increment" in source, (
            "controller._stream_openai_simple 必须调 check_and_increment, "
            "否则 OpenAI 模式调用不计费"
        )
        assert "cost_guard" in source, "controller._stream_openai_simple 应引入 cost_guard"
