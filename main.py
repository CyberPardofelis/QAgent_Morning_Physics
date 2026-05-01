import sys
import traceback
import os
from datetime import datetime

try:
    # 原始导入和代码
    import sys
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from ui import ChatWindow
    import config

except Exception:
    # 确定日志路径
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, 'error.log')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n--- {datetime.now()} ---\n")
        traceback.print_exc(file=f)
    # 重新抛出，让系统处理（可选）
    raise

def excepthook(exc_type, exc_value, exc_traceback):
    # 确定日志文件存放位置（与 exe 同目录或当前目录）
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, 'error.log')

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n--- {datetime.now()} ---\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    # 也可以打印到控制台（如果有）
    traceback.print_exception(exc_type, exc_value, exc_traceback)


sys.excepthook = excepthook


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 读取字体大小设置
    settings = QSettings("CyberPardofelis", "LLM科研助手")
    font_size_str = settings.value("font_size", "10px")  # 默认10px
    try:
        # 提取数字部分，例如 "14px" -> 14
        font_size = int(font_size_str.replace('px', ''))
    except:
        font_size = 10

    # 设置全局字体
    font = app.font()
    font.setPointSize(font_size)
    app.setFont(font)

    settings = QSettings("CyberPardofelis", "LLM科研助手")
    if settings.contains("OPENAI_API_KEY"):
        config.OPENAI_API_KEY = settings.value("OPENAI_API_KEY")
    if settings.contains("OPENAI_MODEL"):
        config.OPENAI_MODEL = settings.value("OPENAI_MODEL")
    if settings.contains("DEEPSEEK_API_KEY"):
        config.DEEPSEEK_API_KEY = settings.value("DEEPSEEK_API_KEY")
    if settings.contains("DEEPSEEK_MODEL"):
        config.DEEPSEEK_MODEL = settings.value("DEEPSEEK_MODEL")
    if settings.contains("DEEPSEEK_MODEL_CHAT"):
        config.DEEPSEEK_MODEL_CHAT = settings.value("DEEPSEEK_MODEL_CHAT")
    if settings.contains("EMBEDDING_ENABLED"):
        val = settings.value("EMBEDDING_ENABLED")
        config.EMBEDDING_ENABLED = val in [True, "true", "1"]
    if settings.contains("EMBEDDER_TYPE"):
        config.EMBEDDER_TYPE = settings.value("EMBEDDER_TYPE")
    if settings.contains("OPENAI_EMBEDDING_MODEL"):
        config.OPENAI_EMBEDDING_MODEL = settings.value("OPENAI_EMBEDDING_MODEL")
    if settings.contains("LOCAL_EMBEDDER_NAME"):
        config.LOCAL_EMBEDDER_NAME = settings.value("LOCAL_EMBEDDER_NAME")
    if settings.contains("MODEL_PATH"):
        config.MODEL_PATH = settings.value("MODEL_PATH")
    if settings.contains("DATA_DIR"):
        config.DATA_DIR = settings.value("DATA_DIR")
    if settings.contains("VECTOR_DIR"):
        config.VECTOR_DIR = settings.value("VECTOR_DIR")

    window = ChatWindow()
    window.show()
    sys.exit(app.exec())