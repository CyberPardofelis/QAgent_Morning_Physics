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
    MINIMAX_API_KEY = "MINIMAX_API_KEY"  # 旧 QSettings 键名（迁移来源，勿删）
    MINIMAX_MODEL = "MINIMAX_MODEL"
    MINIMAX_BASE_URL = "MINIMAX_BASE_URL"

    # 字体
    FONT_SIZE = "font_size"

    # UI 主题(WAgent 主题系统: light / dark / paper / user:xxx)
    UI_THEME = "ui_theme"

    # 聊天模式
    CHAT_MODEL_PARAMS = "chat_model_params"
    DAILY_SUMMARY_JOURNALS = "daily_summary_journals"
    # 2026-09-02 加:主界面"模式"下拉 + "思考模式"勾选 的持久化
    # (用户报"选择模型没有退出保存,每次打开要重新选")
    CHAT_MODE = "chat_mode"               # mode_combo.currentText() (e.g. "deepseek-v4-flash")
    CHAT_THINK_MODE = "chat_think_mode"   # think_checkbox.isChecked() (bool)

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

    # chroma 向量库（P0，2026-09-02 起）
    CHROMA_PATH = "chroma_path"
    CHROMA_COLLECTION = "chroma_collection"
    CHROMA_EMBEDDING_MODEL_PATH = "chroma_embedding_model_path"

    # 成本控制 P0.7（2026-09-02 起）
    # 默认 L2 开, 给个人用方便; 给课题组用时调成 L1 或 L0
    COST_TIER = "cost_tier"  # "L0" | "L1" | "L2"
    COST_CEILING_DAILY = "cost_ceiling_daily"  # int, 默认 50

    # 分组设置（带 group_id 后缀）
    @staticmethod
    def group_system_prompt(group_id: str) -> str:
        return f"group_system_prompt/{group_id}"
