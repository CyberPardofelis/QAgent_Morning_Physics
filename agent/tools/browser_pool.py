"""
BrowserPool — Playwright 浏览器实例池（懒启动、复用、page 用完即关）

解决问题：
- 之前每次工具调用都重新启动 firefox/chromium，启动 10-30s 慢；
- 多次调用累积进程，内存爆。
- sync_playwright 绑定启动线程，跨线程调用会抛错。

设计：
- 全局单例；第一次 acquire_page() 才启动 playwright + browser。
- 每次调用开独立 context + page，with 退出时 page/context 立即关掉，browser 保留。
- 同一线程内复用：threading.Lock 保护懒初始化，owner thread 校验防止跨线程用。
- 退出时（atexit 或显式 shutdown）关 browser + playwright，释放内存。
"""
import logging
import threading
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
# 全局单例（懒创建）
# ============================================================
_pool: Optional[BrowserPool] = None
_pool_lock = threading.Lock()


def get_browser_pool() -> BrowserPool:
    """获取全局 BrowserPool 单例。"""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            _pool = BrowserPool()
        return _pool


def shutdown_browser_pool() -> None:
    """进程退出时调；幂等。"""
    global _pool
    if _pool is None:
        return
    _pool.shutdown()
    _pool = None
