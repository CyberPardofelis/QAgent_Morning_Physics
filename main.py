"""
main.py — QAgent 应用入口

公式渲染走 QWebEngineView (Chromium 嵌入式),所以退出时偶尔会留 QtWebEngineProcess.exe 进程。
清理策略跟 WAgent/SAgent 一样简洁:
  1. 正常退出:Qt aboutToQuit 钩子让 QWebEngineView 主动 close + processEvents,Qt 自己通知 Chromium 关闭
  2. 异常退出(PyCharm Stop / 断电):PID file 记录上次启的 QtWebEngineProcess PID,
     下次启动时杀死的(死的才杀,不影响 chrome.exe,不影响还在跑的另一个 QAgent 实例)
"""
import atexit
import os
import subprocess
import sys
import traceback
from datetime import datetime

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QIcon
    from ui import ChatWindow
    from config import Config, AppInfo
    # 提前 import agent.tools,触发所有 @register_tool 装饰器
    import agent.tools  # noqa: F401
    # 触发 API key 从 QSettings 一次性迁移到 .api_keys.json
    from agent.infra import api_keys  # noqa: F401
    # 注册退出 hook:关闭共享的 Playwright 浏览器实例(懒创建的,shutdown 是幂等的)
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
    """全局异常钩子:写 error.log,便于事后排查"""
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
    # 同时打印到 stderr(PyCharm/终端可见)
    traceback.print_exception(exc_type, exc_value, exc_traceback)


sys.excepthook = excepthook


def _apply_user_settings(app: QApplication):
    """从 QSettings 读用户设置(字体等)并应用到 app"""
    font_size = Config.get_font_size()
    font = app.font()
    font.setPointSize(font_size)
    app.setFont(font)


def _load_qsettings_into_config():
    """
    旧版 main.py 里的逻辑:把 QSettings 里的值同步到 config 模块全局变量。
    现在 Config 类每次从 QSettings 读最新值,不再需要这样手动同步。
    但为了兼容还在 `import config; config.X` 的代码,这里仍然把
    QSettings 里的旧 key 写到 .api_keys.json(如果还没迁移过)。
    """
    # agent.infra.api_keys 第一次调用 get_api_key 时会自动迁移
    # 这里只是触发一下
    try:
        # 读一下 deepseek key,触发迁移
        _ = api_keys.get_api_key("deepseek")
    except Exception:
        pass


# ============================================================
# QWebEngineProcess.exe 残留清理(2 行级实现)
# ============================================================
# 思路:
#   - 正常退出:aboutToQuit 时 _record_qte_pids() 把当前所有 QtWebEngineProcess PID 写进 .qagent_qte_pids
#   - 下次启动:_cleanup_qwe_residual_at_startup() 读 .qagent_qte_pids,杀死的(孤儿),活的 skip
#   - 异常退出(PyCharm Stop):file 留着上次记录的 PID,下次启动杀死的
#   - 安全:QtWebEngineProcess.exe ≠ chrome.exe,不会误杀用户 Chrome 浏览器
# ============================================================

_QTE_PIDS_FILE = os.path.join(
    os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else os.path.abspath(__file__)),
    '.qagent_qte_pids'
)


def _read_qte_pids():
    try:
        with open(_QTE_PIDS_FILE, 'r', encoding='utf-8') as f:
            return [int(x) for x in f.read().split() if x.strip().isdigit()]
    except Exception:
        return []


def _write_qte_pids(pids):
    try:
        with open(_QTE_PIDS_FILE, 'w', encoding='utf-8') as f:
            f.write(' '.join(str(p) for p in pids))
    except Exception:
        pass


def _is_pid_alive(pid):
    try:
        r = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
            capture_output=True, text=True, timeout=3
        )
        return str(pid) in r.stdout
    except Exception:
        return False


def _kill_pids(pids):
    for pid in pids:
        try:
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(pid)],
                capture_output=True, timeout=3
            )
        except Exception:
            pass


def _cleanup_qwe_residual_at_startup():
    """启动时:杀 .qagent_qte_pids 里的死 PID(上次 QAgent 异常退出后的 QtWebEngineProcess 残留)"""
    last_pids = _read_qte_pids()
    if not last_pids:
        return
    killed = [pid for pid in last_pids if not _is_pid_alive(pid)]
    if killed:
        print(f"[main] 启动清理: 杀 {len(killed)} 个 QtWebEngineProcess 残留", flush=True)
        _kill_pids(killed)
    # 启动时清空 file,本次 QAgent 自己的 PID 在 aboutToQuit 时由 _record_qte_pids 写入
    _write_qte_pids([])


def _record_qte_pids():
    """aboutToQuit 钩子:QAgent 退出前记录当前所有 QtWebEngineProcess.exe PID 到 .qagent_qte_pids"""
    try:
        r = subprocess.run(
            ['wmic', 'process', 'where', 'Name="QtWebEngineProcess.exe"', 'get', 'ProcessId'],
            capture_output=True, text=True, timeout=5
        )
        pids = [int(line.strip()) for line in r.stdout.splitlines() if line.strip().isdigit()]
        _write_qte_pids(pids)
    except Exception:
        pass


def _cleanup_on_quit():
    """
    Qt aboutToQuit 标准退出钩子:
      1. 让所有 QWebEngineView 主动 close,Qt 通知 Chromium 关闭
      2. processEvents 跑完 deleteLater 队列
      3. 记录本次启的 QtWebEngineProcess PID(下次启动清理用)
    """
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return
        for w in app.allWidgets():
            try:
                if w.__class__.__name__ == 'QWebEngineView':
                    w.stop()
                    w.close()
                    w.deleteLater()
            except Exception:
                pass
        for _ in range(3):
            app.processEvents()
    except Exception:
        pass
    _record_qte_pids()


if __name__ == "__main__":
    # === 第一件事:清理上次异常退出后的 QtWebEngineProcess 残留 ===
    _cleanup_qwe_residual_at_startup()

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

    # 2026-09-02 加:WAgent 主题系统 — 启动时加载用户自定义主题 + 应用上次保存的主题
    # (必须在 ChatWindow 之前,否则 ChatWindow 会用默认 light 覆盖)
    try:
        from ui.styles import load_user_themes, apply_current_theme
        load_user_themes()
        apply_current_theme(app)
    except Exception as e:
        print(f"[main] 主题系统初始化失败(非致命,已忽略): {e}", flush=True)

    # === Qt 标准退出钩子 ===
    app.aboutToQuit.connect(_cleanup_on_quit)

    # 主窗口
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())
