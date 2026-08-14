"""
统一的 QSettings 访问入口（参考 WAgent 实现）。

解决的问题：
- 散落各文件的 `QSettings("CyberPardofelis", "LLM科研助手")` 硬编码
- 拼写不一致风险
- 改组织名/应用名要 grep 替换全部

设计：
- get_settings() 提供统一 QSettings 实例
- AppInfo 集中组织名/应用名
- SettingsKeys 集中键名（部分常用键，未来逐步把 key 字符串也替换为常量）

注意：SettingsKeys 不强制替代所有 settings.value("xxx") 里的字面量。
那些是数据值，不是配置——大量替换会引入回归风险。
"""
from PySide6.QtCore import QSettings


# ============================================================
# 应用信息（org/app name）
# ============================================================
class AppInfo:
    ORG_NAME = "CyberPardofelis"
    APP_NAME = "LLM科研助手"


# ============================================================
# 统一访问入口
# ============================================================
def get_settings() -> QSettings:
    """
    获取 QSettings 单例。
    所有读写配置都走这里，避免硬编码 org/app name。
    """
    return QSettings(AppInfo.ORG_NAME, AppInfo.APP_NAME)


# ============================================================
# 常用键名常量（部分键，未来按需扩展）
# ============================================================
class SettingsKeys:
    # LLM 配置（API key 推荐走 .api_keys.json，旧键作为迁移来源）
    DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"  # 旧 QSettings 键名（迁移来源，勿删）
    DEEPSEEK_MODEL = "DEEPSEEK_MODEL"
    DEEPSEEK_MODEL_CHAT = "DEEPSEEK_MODEL_CHAT"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    OPENAI_MODEL = "OPENAI_MODEL"

    # 字体
    FONT_SIZE = "font_size"

    # 聊天模式
    CHAT_MODEL_PARAMS = "chat_model_params"
    DAILY_SUMMARY_JOURNALS = "daily_summary_journals"

    # 系统提示词
    DEFAULT_SYSTEM_PROMPT = "default_system_prompt"

    # 对比模式
    COMPARISON_CONFIG_A = "comparison/config_a"
    COMPARISON_CONFIG_B = "comparison/config_b"

    # Embedding
    EMBEDDING_ENABLED = "EMBEDDING_ENABLED"
    EMBEDDER_TYPE = "EMBEDDER_TYPE"
    OPENAI_EMBEDDING_MODEL = "OPENAI_EMBEDDING_MODEL"
    LOCAL_EMBEDDER_NAME = "LOCAL_EMBEDDER_NAME"
    MODEL_PATH = "MODEL_PATH"
    DATA_DIR = "DATA_DIR"
    VECTOR_DIR = "VECTOR_DIR"

    # 分组设置（带 group_id 后缀）
    @staticmethod
    def group_system_prompt(group_id: str) -> str:
        return f"group_system_prompt/{group_id}"
