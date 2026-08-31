"""
BrowserPool — Playwright 浏览器实例池（懒启动、复用、page 用完即关）

解决问题：
- 之前每次工具调用都重新启动 firefox/chromium，启动 10-30s 慢；
- 多次调用累积进程，内存爆。
- sync_playwright 绑定启动线程，跨线程调用会抛错。

设计：
- 【v2 2026-08-31】按线程懒创建 BrowserPool,每个 QThread worker 在自己的线程里持有独立 pool。
  之前是全局单例 + owner_thread 校验,但 QAgent 的工具在 QThread worker 子线程里跑,
  单例 + 校验导致跨线程被 playwright 拒绝。改成 threading.local 后,每个 thread 第一次
  调 get_browser_pool() 时创建自己的 pool(同线程,playwright 校验通过),代价是多 worker
  时每个 thread 一份 firefox 进程(内存翻倍),但 QAgent 现在只有 1 个 worker thread,
  实际只多一份。
- 每次调用开独立 context + page，with 退出时 page/context 立即关掉，browser 保留。
- 退出时（atexit 或显式 shutdown）关 browser + playwright，释放内存。
  worker thread 的 pool 在 thread GC 时由 weakref.finalize 兜底关 playwright。
"""
import atexit
import logging
import threading
import weakref
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


# 默认 user agent（沿用 aps.py 原值）
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class BrowserPool:
    """单实例：每个进程一个 BrowserPool，所有工具共用。"""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._owner_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        # GC 兜底:worker thread 退出时,如果 BrowserPool 没显式 shutdown,
        # 至少把 playwright stop 掉,避免 firefox 进程残留。
        # 注意:finalize callback 必须在 self 创建时的线程里 stop(playwright 硬约束),
        # 但 weakref.finalize 会在 GC 触发线程里调,通常还是原线程(线程结束后 GC 该 thread
        # 局部变量时,通常仍在该线程 context 里,安全);跨线程 stop 可能抛错,已被 _finalize 内部 try 吞掉。
        weakref.finalize(self, BrowserPool._finalize, self)

    @staticmethod
    def _finalize(pool: "BrowserPool") -> None:
        """GC 兜底:清理 playwright 资源(可能跨线程,失败就吞)。"""
        try:
            if pool._playwright is not None:
                pool._playwright.stop()
        except Exception as e:
            logger.debug("[BrowserPool] GC 兜底 stop playwright 失败(忽略): %s", e)

    # ---------- 内部 ----------
    def _ensure_started(self, browser_type: str) -> None:
        """懒启动：第一次调用时初始化 playwright + browser。"""
        if self._browser is not None:
            return
        with self._lock:
            if self._browser is not None:
                return
            # sync_playwright() 返回的实例必须和后续 launch 在同一线程用
            # 我们把所有 browser 操作都锁在 owner 线程里
            from playwright.sync_api import sync_playwright
            self._owner_thread = threading.current_thread()
            self._playwright = sync_playwright().start()
            launcher = getattr(self._playwright, browser_type, None)
            if launcher is None:
                raise RuntimeError(
                    f"Playwright 未安装 {browser_type} 浏览器，"
                    f"请运行: playwright install {browser_type}"
                )
            kwargs = {"headless": True}
            if browser_type == "chromium":
                # 防风控标记（沿用 aps 原 _download_with_playwright 的参数）
                kwargs["args"] = ["--disable-blink-features=AutomationControlled"]
            self._browser = launcher.launch(**kwargs)
            logger.info(
                "[BrowserPool] 浏览器已启动 (%s, thread=%s, 后续复用此实例)",
                browser_type, self._owner_thread.name,
            )

    def _check_thread(self) -> None:
        """sync_playwright API 必须和启动线程一致；不一致直接 raise。"""
        if self._owner_thread is None:
            return
        if threading.current_thread() is not self._owner_thread:
            raise RuntimeError(
                "BrowserPool 跨线程使用：sync_playwright 必须与启动线程一致。"
                f"owner={self._owner_thread.name!r}, current={threading.current_thread().name!r}。"
                "请确保所有浏览器调用都在同一个 QThread worker 内执行。"
            )

    # ---------- 公共 ----------
    @contextmanager
    def acquire_page(
        self,
        browser_type: str = "chromium",
        user_agent: str = DEFAULT_USER_AGENT,
        accept_downloads: bool = True,
        viewport: Optional[dict] = None,
    ):
        """
        上下文管理器：返回一个新 page。退出时自动关 page + context，browser 保留。

        多次调用安全；page 之间的 state 完全隔离（各自 context）。
        """
        self._ensure_started(browser_type)
        self._check_thread()

        context_kwargs = {"user_agent": user_agent}
        if accept_downloads:
            context_kwargs["accept_downloads"] = True
        if viewport:
            context_kwargs["viewport"] = viewport

        context = self._browser.new_context(**context_kwargs)
        page = context.new_page()
        try:
            yield page
        finally:
            for close_fn in (page.close, context.close):
                try:
                    close_fn()
                except Exception as e:
                    logger.debug("[BrowserPool] 关闭 page/context 失败（忽略）: %s", e)

    def shutdown(self) -> None:
        """显式关闭（atexit 或测试时调）。可重复调用，安全。"""
        with self._lock:
            if self._browser is not None:
                try:
                    self._browser.close()
                except Exception as e:
                    logger.debug("[BrowserPool] 关闭 browser 失败（忽略）: %s", e)
                self._browser = None
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception as e:
                    logger.debug("[BrowserPool] 停止 playwright 失败（忽略）: %s", e)
                self._playwright = None
            self._owner_thread = None
            logger.info("[BrowserPool] 浏览器实例已关闭")

    @property
    def is_running(self) -> bool:
        return self._browser is not None


# ============================================================
# 按线程懒创建（v2:threading.local,支持 QThread worker）
# ============================================================
_local = threading.local()
_init_lock = threading.Lock()
_main_pool: Optional[BrowserPool] = None  # 主线程的 pool 引用,atexit 时关


def get_browser_pool() -> BrowserPool:
    """按线程懒创建 BrowserPool。

    - 主线程第一次调:创建主线程 pool(走 atexit 关闭)。
    - QThread worker 子线程第一次调:创建该 thread 自己的 pool(GC 兜底)。
    - 同线程后续调用:直接返回旧 pool(复用 firefox,避免 10-30s 启动开销)。
    """
    pool = getattr(_local, "pool", None)
    if pool is not None:
        return pool
    with _init_lock:
        # 双重检查:可能在锁等待期间已被其他协程创建
        pool = getattr(_local, "pool", None)
        if pool is not None:
            return pool
        pool = BrowserPool()
        _local.pool = pool
        # 主线程的 pool 单独记录,atexit 显式 shutdown
        global _main_pool
        if threading.current_thread() is threading.main_thread():
            _main_pool = pool
        logger.info(
            "[BrowserPool] 新建 pool for thread=%s  (后续同线程复用)",
            threading.current_thread().name,
        )
        return pool


def shutdown_browser_pool() -> None:
    """进程退出时调；幂等。仅关闭主线程 pool(worker thread 的 pool 由 GC finalize 兜底)。"""
    global _main_pool
    if _main_pool is not None:
        try:
            _main_pool.shutdown()
        except Exception as e:
            logger.debug("[BrowserPool] shutdown 失败(忽略): %s", e)
        _main_pool = None


atexit.register(shutdown_browser_pool)
