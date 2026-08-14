"""
main.py — 应用入口（重构版）

变化：
- 用 config.Config + agent.infra.api_keys 统一配置来源
- 配置加载顺序：QSettings 显式值 > .api_keys.json（自动迁移自 QSettings 旧 key）> .env 兜底
- 全局异常钩子保留（写 error.log）
"""
import sys
import os
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


if __name__ == "__main__":
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

    # 主窗口
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())
