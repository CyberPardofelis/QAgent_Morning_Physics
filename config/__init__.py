"""
config — 统一配置层

这是项目里唯一的 `config` 入口。原来的根 `config.py` 已删除（避免与本目录冲突）。

参考 WAgent 设计：
- 优先 QSettings（用户运行时改的）
- 兜底 .env（首次启动 / 自动化场景）
- LLM API key 推荐走 .api_keys.json（agent.infra.api_keys），更安全

两种用法：
    from config import Config, get_settings, AppInfo, SettingsKeys
    from config import OPENAI_API_KEY, DEEPSEEK_MODEL, EMBEDDING_ENABLED  # 模块级常量（首次读 .env）

模块级常量（如 OPENAI_API_KEY, DEEPSEEK_MODEL 等）只用于兼容旧代码。
新代码推荐用 `Config.get_xxx()` 或 `SettingsKeys.xxx`。
"""
import os
from dotenv import load_dotenv

# 触发 .env 加载
load_dotenv()


# ============================================================
# 模块级常量（兼容旧 import: `from config import OPENAI_API_KEY` 等）
# 优先从 .env 读；实际用户改的值在 QSettings 里（用 Config.get_xxx() 读）
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# 兜底：用户也可以在环境变量 DEEPSEEK_API_KEY_FALLBACK 里再放一份
if not DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY_FALLBACK")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_MODEL_CHAT = os.getenv("DEEPSEEK_MODEL_CHAT", "deepseek-chat")

EMBEDDING_ENABLED = os.getenv("EMBEDDING_ENABLED", "false").lower() in ("true", "1", "yes")
EMBEDDER_TYPE = os.getenv("EMBEDDER_TYPE", "local")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
MODEL_PATH = os.getenv("MODEL_PATH", r"D:\pythonProject\BGE_Large_en_v1.5")
LOCAL_EMBEDDER_NAME = os.getenv("LOCAL_EMBEDDER_NAME", "BAAI/bge-base-zh")

DATA_DIR = os.getenv("QAGENT_DATA_DIR", "data/papers")
VECTOR_DIR = os.getenv("QAGENT_VECTOR_DIR", "data/vector_store")

ARXIV_SEC = os.getenv("QAGENT_ARXIV_SEC", "quant-ph")


# ============================================================
# 延迟从 config.settings 导入（避免 PySide6 在 import 时崩溃）
# ============================================================
def __getattr__(name):
    """延迟导入：仅在被访问时才 import config.settings（需要 PySide6）"""
    if name in ("get_settings", "AppInfo", "SettingsKeys", "Config"):
        from config.settings import get_settings, AppInfo, SettingsKeys
        # 本地定义 Config 类（避免循环 import settings <-> __init__）
        from config._config import Config
        # 缓存到模块全局
        import sys
        mod = sys.modules[__name__]
        mod.get_settings = get_settings
        mod.AppInfo = AppInfo
        mod.SettingsKeys = SettingsKeys
        mod.Config = Config
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
