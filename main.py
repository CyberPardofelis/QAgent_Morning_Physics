"""
main.py — 应用入口（重构版）

变化：
- 用 config.Config + agent.infra.api_keys 统一配置来源
- 配置加载顺序：QSettings 显式值 > .api_keys.json（自动迁移自 QSettings 旧 key）> .env 兜底
- 全局异常钩子保留（写 error.log）
- 退出时 atexit 关闭 BrowserPool（释放 playwright + browser 进程）
"""
import atexit
import signal
import subprocess
import sys
import os
import ctypes
import traceback
from datetime import datetime

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QIcon
    from ui import ChatWindow
    from config import Config, AppInfo
    # 提前 import agent.tools，触发所有 @register_tool 装饰器
    import agent.tools  # noqa: F401
    # 触发 API key 从 QSettings 一次性迁移到 .api_keys.json
    from agent.infra import api_keys  # noqa: F401
    # 注册退出 hook：关闭共享的 Playwright 浏览器实例（懒创建的，shutdown 是幂等的）
    from agent.tools.browser_pool import shutdown_browser_pool
    atexit.register(shutdown_browser_pool)
except Exception:
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, 'error.log')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n--- {datetime.now()} ---\n")
        traceback.print_exc(file=f)
    raise


def excepthook(exc_type, exc_value, exc_traceback):
    """全局异常钩子：写 error.log，便于事后排查"""
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, 'error.log')

    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- {datetime.now()} ---\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    except OSError:
        pass
    # 同时打印到 stderr（PyCharm/终端可见）
    traceback.print_exception(exc_type, exc_value, exc_traceback)


sys.excepthook = excepthook


def _apply_user_settings(app: QApplication):
    """从 QSettings 读用户设置（字体等）并应用到 app"""
    font_size = Config.get_font_size()
    font = app.font()
    font.setPointSize(font_size)
    app.setFont(font)


def _load_qsettings_into_config():
    """
    旧版 main.py 里的逻辑：把 QSettings 里的值同步到 config 模块全局变量。
    现在 Config 类每次从 QSettings 读最新值，不再需要这样手动同步。
    但为了兼容还在 `import config; config.X` 的代码，这里仍然把
    QSettings 里的旧 key 写到 .api_keys.json（如果还没迁移过）。
    """
    # agent.infra.api_keys 第一次调用 get_api_key 时会自动迁移
    # 这里只是触发一下
    try:
        # 读一下 deepseek key，触发迁移
        _ = api_keys.get_api_key("deepseek")
    except Exception:
        pass


def _cleanup_on_quit():
    """
    进程退出 hook(注册到 app.aboutToQuit):
    显式清理 QWebEngineView(FormulaRenderer) + Worker QThread,避免 Chromium 子进程残留。

    根因:QWebEngineView 启动 Chromium zygote + GPU + renderer + utility 多子进程,
        主 Python 进程退出时,Qt 不保证所有 Chromium 子进程被 terminate(Windows 上尤其),
        结果:PyCharm run tab 显示"已停止"但任务管理器还有残留 python.exe / Chromium 进程。
    修法:在 app.aboutToQuit 显式:
        1. QWebEngineView.deleteLater() + page().deleteLater() + processEvents() 让清理事件跑
        2. 跑一次 Chromium 关闭命令(QWebEnginePage.WebAction.Stop)
        3. setQuitOnLastWindowClosed 已默认 True,主窗口关后 app.exec() 返回
    """
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        return

    # 1. 清理 FormulaRenderer 的 QWebEngineView(主要残留源)
    try:
        from ui.formula_renderer import get_formula_renderer
        renderer = get_formula_renderer()
        if renderer is not None and renderer._view is not None:
            try:
                # 停止 Chromium 加载
                renderer._view.stop()
                # 释放 page
                page = renderer._view.page()
                if page is not None:
                    page.deleteLater()
            except Exception:
                pass
            try:
                renderer._view.close()
                renderer._view.deleteLater()
            except Exception:
                pass
            renderer._view = None
    except Exception:
        pass

    # 2. 找所有 QWebEngineView 子 widget(可能有其他创建)
    try:
        for w in app.allWidgets():
            try:
                if w.__class__.__name__ == 'QWebEngineView':
                    w.stop()
                    w.close()
                    w.deleteLater()
            except Exception:
                pass
    except Exception:
        pass

    # 3. 让 Qt 处理 deleteLater 队列
    try:
        for _ in range(3):
            app.processEvents()
    except Exception:
        pass

    # 4. 兜底杀残余 Chromium 子进程(以防 deleteLater 异步队列没跑完)
    _kill_chromium_residual()


def _kill_chromium_residual():
    """
    兜底清理:扫描所有以当前进程为祖先的 Chromium / python.exe 子进程并强杀。
    解决 PyCharm run tab 关掉时(主进程被 SIGKILL,aboutToQuit 来不及跑)
    Chromium 子进程残留的问题。

    用 wmic 查所有 ProcessId + ParentProcessId,递归找所有 descendant,
    然后 taskkill /F /T 强杀。
    """
    my_pid = os.getpid()

    def _find_children(parent_pid):
        """返回所有 parent_pid 的子进程 pid 列表"""
        children = []
        try:
            result = subprocess.run(
                ['wmic', 'process', 'where', f'ParentProcessId={parent_pid}',
                 'get', 'ProcessId'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    children.append(int(line))
        except Exception:
            pass
        if not children:
            # fallback:PowerShell + CIM(Win11 22H2+ 兼容)
            try:
                ps_cmd = (
                    f'Get-CimInstance Win32_Process '
                    f'| Where-Object {{ $_.ParentProcessId -eq {parent_pid} }} '
                    f'| Select-Object -ExpandProperty ProcessId'
                )
                result = subprocess.run(
                    ['powershell', '-NoProfile', '-Command', ps_cmd],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line.isdigit():
                        children.append(int(line))
            except Exception:
                pass
        return children

    # 递归找所有 descendant
    all_pids = set()
    queue = [my_pid]
    while queue:
        parent = queue.pop(0)
        for child in _find_children(parent):
            if child not in all_pids:
                all_pids.add(child)
                queue.append(child)

    for pid in all_pids:
        try:
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(pid)],
                capture_output=True, timeout=3
            )
        except Exception:
            pass


# ============================================================
# Windows Job Object:内核级进程树清理(主防线)
# ============================================================
# 根因:PyCharm run tab 的 "Stop" 按钮 = TerminateProcess 直接强杀主进程,
#       Python 层面的 atexit / signal handler / Qt aboutToQuit 全部来不及跑。
#       即使跑了,_kill_chromium_residual 也只是兜底,且 PyCharm 强杀时不一定来得及。
#
# 方案:用 Windows Job Object + KILL_ON_JOB_CLOSE — Windows 内核对象,只要主进程
#       handle 关闭(无论怎么死:SIGKILL / TerminateProcess / 异常崩溃),
#       内核自动 kill Job 内所有进程。这是 100% 可靠的进程树清理机制。
#
# 关键点:
#   1. 主进程必须显式 AssignProcessToJobObject(GetCurrentProcess())
#   2. QWebEngineView 启动的 Chromium 子进程通过 CreateProcess 继承父 Job
#   3. 部分 Qt 内部进程可能 CREATE_BREAKAWAY_FROM_JOB,所以加 QTimer 周期性收编
#   4. 跨平台兼容:Windows 专属,非 Windows 平台所有函数 no-op
# ============================================================

_IS_WINDOWS = sys.platform == 'win32'
_job_handle = None  # type: ignore[var-annotated]

# ctypes 定义
if _IS_WINDOWS:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9
    PROCESS_ALL_ACCESS = 0x1F0FFF
    ERROR_NOT_ENOUGH_QUOTA = 145  # 进程已属于其他 Job,无法加入
    # Windows ctypes 没有 c_uintptr_t,用 c_size_t 替代(64-bit Windows 都是 8 字节)
    _SIZE_T = ctypes.c_size_t

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ('ReadOperationCount', ctypes.c_uint64),
            ('WriteOperationCount', ctypes.c_uint64),
            ('OtherOperationCount', ctypes.c_uint64),
            ('ReadTransferCount', ctypes.c_uint64),
            ('WriteTransferCount', ctypes.c_uint64),
            ('OtherTransferCount', ctypes.c_uint64),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', ctypes.c_int64),
            ('PerJobUserTimeLimit', ctypes.c_int64),
            ('LimitFlags', ctypes.c_uint32),
            ('MinimumWorkingSetSize', _SIZE_T),
            ('MaximumWorkingSetSize', _SIZE_T),
            ('ActiveProcessLimit', ctypes.c_uint32),
            ('Affinity', _SIZE_T),
            ('PriorityClass', ctypes.c_uint32),
            ('SchedulingClass', ctypes.c_uint32),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ('IoInfo', IO_COUNTERS),
            ('ProcessMemoryLimit', _SIZE_T),
            ('JobMemoryLimit', _SIZE_T),
            ('PeakProcessMemoryUsed', _SIZE_T),
            ('PeakJobMemoryUsed', _SIZE_T),
        ]

    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                  ctypes.c_void_p, ctypes.c_uint32]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]


def _setup_windows_job() -> bool:
    """
    创建 Windows Job Object,设 KILL_ON_JOB_CLOSE,主进程加入 Job。
    一旦 Job handle 关闭(主进程退出),Windows 内核自动 kill 所有 Job 成员。
    这是 100% 可靠的进程树清理,不依赖任何 Python hook。

    Returns: True = 成功,False = 失败(非 Windows / 平台不支持 / PyCharm 已加入其他 Job 等)
    """
    global _job_handle
    if not _IS_WINDOWS:
        return False
    try:
        hJob = kernel32.CreateJobObjectW(None, None)
        if not hJob:
            print(f"[job-object] CreateJobObjectW 失败: error={ctypes.get_last_error()}")
            return False
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            hJob, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info)
        )
        if not ok:
            err = ctypes.get_last_error()
            print(f"[job-object] SetInformationJobObject 失败: error={err}")
            kernel32.CloseHandle(hJob)
            return False
        ok = kernel32.AssignProcessToJobObject(hJob, kernel32.GetCurrentProcess())
        if not ok:
            err = ctypes.get_last_error()
            if err == ERROR_NOT_ENOUGH_QUOTA:
                # 进程已属于其他 Job(PyCharm 启动时可能加进了自己的 Job)
                # 此时 KILL_ON_JOB_CLOSE 由外层 Job 控制,无法由我们保证
                print(f"[job-object] 主进程已属于其他 Job(可能是 PyCharm),KILL_ON_JOB_CLOSE 无法生效")
            else:
                print(f"[job-object] AssignProcessToJobObject 失败: error={err}")
            kernel32.CloseHandle(hJob)
            return False
        _job_handle = hJob
        return True
    except Exception as e:
        print(f"[job-object] 异常: {e}")
        return False


def _add_pid_to_job(pid: int) -> bool:
    """尝试把指定 pid 加入 Job。失败(已属于其他 Job / 进程已死)返回 False。"""
    if not _IS_WINDOWS or _job_handle is None:
        return False
    try:
        hProcess = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        if not hProcess:
            return False
        ok = kernel32.AssignProcessToJobObject(_job_handle, hProcess)
        kernel32.CloseHandle(hProcess)
        return bool(ok)
    except Exception:
        return False


def _scan_and_add_children_to_job():
    """扫描所有以当前进程为祖先的子进程,尝试加入 Job。
    用于处理 CREATE_BREAKAWAY_FROM_JOB 启动的子进程(Qt 内部可能用)。
    """
    if not _IS_WINDOWS or _job_handle is None:
        return
    try:
        my_pid = os.getpid()
        # 复用 _kill_chromium_residual 的扫描,但这里只 add,不 kill
        all_pids = set()
        queue = [my_pid]
        while queue:
            parent = queue.pop(0)
            # 用 wmic / PowerShell 找子进程(简化版,避免重复 import subprocess 函数)
            children = []
            try:
                result = subprocess.run(
                    ['wmic', 'process', 'where', f'ParentProcessId={parent}',
                     'get', 'ProcessId'],
                    capture_output=True, text=True, timeout=3
                )
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line.isdigit():
                        children.append(int(line))
            except Exception:
                pass
            if not children:
                try:
                    ps_cmd = (
                        f'Get-CimInstance Win32_Process '
                        f'| Where-Object {{ $_.ParentProcessId -eq {parent} }} '
                        f'| Select-Object -ExpandProperty ProcessId'
                    )
                    result = subprocess.run(
                        ['powershell', '-NoProfile', '-Command', ps_cmd],
                        capture_output=True, text=True, timeout=3
                    )
                    for line in result.stdout.splitlines():
                        line = line.strip()
                        if line.isdigit():
                            children.append(int(line))
                except Exception:
                    pass
            for child in children:
                if child not in all_pids:
                    all_pids.add(child)
                    queue.append(child)
        for pid in all_pids:
            _add_pid_to_job(pid)
    except Exception:
        pass


def _start_job_watchdog(app: QApplication):
    """启动 QTimer 周期性扫描子进程并加入 Job(处理 breakaway 启动的子进程)。"""
    if not _IS_WINDOWS or _job_handle is None:
        return
    try:
        from PySide6.QtCore import QTimer
        timer = QTimer()
        timer.setInterval(2000)  # 每 2 秒
        timer.timeout.connect(_scan_and_add_children_to_job)
        timer.start()
    except Exception:
        pass


if __name__ == "__main__":
    # === 必须最早:QApplication 创建之前就 setup Job Object ===
    # 这样 QApplication 启动后所有子进程(QWebEngineView Chromium 链)都自动 inherit Job,
    # 主进程一旦被强杀(PyCharm Stop),Windows 内核自动 kill 整个进程树。
    _setup_windows_job()

    app = QApplication(sys.argv)

    # 触发 API key 一次性迁移
    _load_qsettings_into_config()

    # 应用图标
    if getattr(sys, 'frozen', False):
        _base_dir = os.path.dirname(sys.executable)
    else:
        _base_dir = os.path.dirname(os.path.abspath(__file__))
    _icon_path = os.path.join(_base_dir, "resources", "app_icon.ico")
    if os.path.exists(_icon_path):
        app.setWindowIcon(QIcon(_icon_path))

    # 全局字体
    _apply_user_settings(app)

    # 注册退出 hook(比 atexit 更可靠 — atexit 在 sys.exit() 时跑,但 PyQt 可能不走 atexit)
    app.aboutToQuit.connect(_cleanup_on_quit)
    # atexit 兜底:即使 PyCharm 用 SIGTERM 强杀(aboutToQuit 来不及跑),atexit 也会执行
    atexit.register(_kill_chromium_residual)
    # Job watchdog:每 2s 扫子进程加入 Job(处理 CREATE_BREAKAWAY_FROM_JOB 启动的子进程)
    _start_job_watchdog(app)

    # 信号 handler:PyCharm 关 run tab 时给 Python 发 SIGTERM,默认被 Python 解释器
    # 抛 SystemExit,但 Qt event loop 不响应 → Python 进程 hang。改成触发 app.quit()
    # 让 Qt event loop 正常退出,aboutToQuit 才会跑,清理才会生效。
    def _on_sigterm_or_sigint(sig, frame):
        try:
            app.quit()
        except Exception:
            pass
    try:
        signal.signal(signal.SIGTERM, _on_sigterm_or_sigint)
        signal.signal(signal.SIGINT, _on_sigterm_or_sigint)
    except (ValueError, OSError):
        # Windows / 子线程可能不支持
        pass

    # 主窗口
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())
