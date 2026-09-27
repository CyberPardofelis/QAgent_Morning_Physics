"""
agent.cost_guard — LLM 每日调用计数 + ceiling 监控（P0.7 新增 2026-09-02）

职责:
- 记录当日 LLM 调用次数（按 0 点本地时区重置）
- 80% 时给 warning（通过 yield 给 LLM 看到）
- 100% 时 block（controller 直接 return，不让调 LLM）

设计:
- 状态存 QSettings（轻量,跨进程也能看到,UI 可以读出来显示）
- 不存数据库（次数轻量,不需事务）
- thread-safe：lock 保护 increment（controller 多线程跑 tool 循环时会有并发）

为什么不存 sqlite: 每天最多几千次调用,纯 count 用 QSettings 够;
sqlite 写一次事务太重,反而拖慢 controller。
"""
import logging
import threading
from datetime import date

logger = logging.getLogger(__name__)


class CostGuard:
    """每日 LLM 调用 ceiling 监控

    用法:
        from config import Config
        from agent.cost_guard import get_cost_guard

        guard = get_cost_guard()
        ceiling = Config.get_cost_ceiling_daily()  # 默认 50
        # 每次 controller 进入 LLM 调用前:
        is_ok, msg = guard.check_and_increment()
        if not is_ok:
            yield msg  # 告诉用户已用满
            return
        if msg:  # 80% 警告
            yield msg
    """

    # QSettings 键
    KEY_DATE = "cost_guard/date"  # ISO date string, 用于检测跨天重置
    KEY_COUNT = "cost_guard/count"  # int, 今日已用次数

    def __init__(self):
        self._lock = threading.Lock()
        # 不在 __init__ 读 QSettings（PySide6 没启动时不能调），
        # 第一次 check_and_increment() 时 lazy 读

    def _load(self) -> tuple[str, int]:
        """从 QSettings 读 (date_str, count)。第一次读就缓存。"""
        from config.settings import get_settings
        s = get_settings()
        d = s.value(self.KEY_DATE, "") or ""
        c = s.value(self.KEY_COUNT, 0)
        try:
            c = int(c)
        except (TypeError, ValueError):
            c = 0
        return d, c

    def _save(self, date_str: str, count: int) -> None:
        from config.settings import get_settings
        s = get_settings()
        s.setValue(self.KEY_DATE, date_str)
        s.setValue(self.KEY_COUNT, int(count))

    def _reset_if_new_day(self, current_date: str) -> int:
        """如果跨天了, 重置 count 为 0 并返回新 count。"""
        saved_date, count = self._load()
        if saved_date != current_date:
            count = 0
            self._save(current_date, 0)
        return count

    def current_count(self) -> int:
        """今日已用次数（不会重置,只读）"""
        with self._lock:
            today = date.today().isoformat()
            self._reset_if_new_day(today)
            _, count = self._load()
            return count

    def check_and_increment(self, ceiling: int | None = None) -> tuple[bool, str]:
        """检查 + 递增 + 返回 (is_ok, message)

        Returns:
            (True, "")          — 正常,已 increment
            (True, "⚠ 今日已用 X/Y (80%)") — 80% 警告,但仍放行
            (False, "❌ 今日 LLM 调用已用满 X/Y,已阻断") — 100% 阻断(第 ceiling+1 次起)

        Args:
            ceiling: 每日上限, 留空用 Config.get_cost_ceiling_daily() 默认 50
        """
        with self._lock:
            from config import Config
            if ceiling is None:
                ceiling = Config.get_cost_ceiling_daily()
            today = date.today().isoformat()
            count = self._reset_if_new_day(today)

            # 100% 阻断(用完 ceiling 次后,下一步起不再放行)
            # 2026-09-02 v2 修:之前先 block 再 increment 顺序写反了,5/5=1.0 走了 80% 分支
            # 现在先 increment 到 ceiling 再 block,语义清晰
            if count >= ceiling:
                msg = (
                    f"\n\n[CostGuard] ❌ 今日 LLM 调用已达上限 "
                    f"({count}/{ceiling}, 100%), 已阻断。\n"
                    f"明天 0 点自动重置, 或调高 'cost_ceiling_daily'。"
                )
                logger.warning("[CostGuard] blocked at %d/%d", count, ceiling)
                return False, msg

            # 放行 + 递增
            count += 1
            self._save(today, count)

            # 100% 阻断(已用满,从此步起不再放行)
            if count >= ceiling:
                msg = (
                    f"\n\n[CostGuard] ❌ 今日 LLM 调用已达上限 "
                    f"({count}/{ceiling}, 100%), 明天 0 点自动重置。"
                )
                logger.warning("[CostGuard] reached ceiling at %d/%d", count, ceiling)
                return False, msg

            # 80% 警告
            if count / ceiling >= 0.8:
                pct = int(count / ceiling * 100)
                msg = (
                    f"\n\n[CostGuard] ⚠ 今日 LLM 已用 {count}/{ceiling} ({pct}%)"
                )
                logger.info("[CostGuard] warn at %d/%d (%d%%)", count, ceiling, pct)
                return True, msg

            return True, ""

    def reset(self) -> None:
        """手动重置今日计数(给 UI 调试用)"""
        with self._lock:
            today = date.today().isoformat()
            self._save(today, 0)
            logger.info("[CostGuard] 手动重置今日计数")


# ============================================================
# 全局单例
# ============================================================
_guard_instance: CostGuard | None = None
_guard_lock = threading.Lock()


def get_cost_guard() -> CostGuard:
    """获取 CostGuard 全局单例（lazy）"""
    global _guard_instance
    if _guard_instance is None:
        with _guard_lock:
            if _guard_instance is None:
                _guard_instance = CostGuard()
    return _guard_instance


# ============================================================
# 计费收口 (9-27 P0)
# ============================================================
# 全项目**每一个真实 LLM 请求**都必须先过一次这里。
#
# 之前的漏洞:
#   - 只有 controller 的 4 个 call site 计费, agent/tools/llm.py 的 _call_llm
#     (所有工具内部的 LLM 调用) 和 memory/extractor.py 的抽取调用完全不经过计费
#   - controller.py 还在 run_agent_stream 开头预扣了一次, 第一轮被重复计费
#
# 实际影响: survey_generate 一次运行内部发 100+ 次 LLM, 守卫只看见约 50 次,
# ceiling 完全约束不住真实开销。


def charge(ceiling: int | None = None) -> tuple[bool, str]:
    """记一次 LLM 调用。返回 (is_ok, message)。

    - (True, "")  → 放行
    - (True, "⚠…") → 放行 + 80% 警告(调用方决定怎么呈现)
    - (False, "❌…") → 已达上限, 调用方**不应**再发请求

    每个发起 LLM 请求的地方都应该在请求前调它, 阻断时直接返回错误信息。
    """
    return get_cost_guard().check_and_increment(ceiling)
