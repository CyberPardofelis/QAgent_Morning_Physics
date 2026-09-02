"""
config._config — Config 类实现

独立成文件以避免 config/__init__.py 的循环 import。
（Config 类要 from .settings import get_settings，但 settings 要避免触发 __init__ 的全部执行）
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import get_settings, AppInfo, SettingsKeys


class Config:
    """统一配置管理，每次调用都从 QSettings 读取最新值（避免缓存问题）"""

    @staticmethod
    def _settings():
        from config.settings import get_settings
        return get_settings()

    # ---------- 字体 ----------
    @staticmethod
    def get_font_size() -> int:
        size_str = Config._settings().value("font_size", "10px")
        try:
            return int(size_str.replace('px', ''))
        except Exception:
            return 10

    @staticmethod
    def set_font_size(value_px) -> None:
        Config._settings().setValue("font_size", value_px)

    # ---------- LLM 基础配置（API key 推荐走 .api_keys.json） ----------
    @staticmethod
    def get_deepseek_model() -> str:
        # 优先 QSettings，兜底用环境变量
        v = Config._settings().value("DEEPSEEK_MODEL", "")
        return v if v else "deepseek-v4-flash"

    @staticmethod
    def set_deepseek_model(value: str) -> None:
        Config._settings().setValue("DEEPSEEK_MODEL", value)

    @staticmethod
    def get_openai_model() -> str:
        v = Config._settings().value("OPENAI_MODEL", "")
        return v if v else "gpt-4o-mini"

    @staticmethod
    def set_openai_model(value: str) -> None:
        Config._settings().setValue("OPENAI_MODEL", value)

    # ---------- 系统提示词 ----------
    @staticmethod
    def get_default_system_prompt(default: str = "") -> str:
        return Config._settings().value("default_system_prompt", default)

    @staticmethod
    def set_default_system_prompt(value: str) -> None:
        Config._settings().setValue("default_system_prompt", value)

    # ---------- Embedding ----------
    @staticmethod
    def get_embedding_enabled() -> bool:
        v = Config._settings().value("EMBEDDING_ENABLED", False)
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    @staticmethod
    def set_embedding_enabled(value: bool) -> None:
        Config._settings().setValue("EMBEDDING_ENABLED", bool(value))

    # ---------- chroma 向量库（P0，2026-09-02 起） ----------
    @staticmethod
    def get_chroma_path() -> str:
        """chroma 持久化目录。默认 data/chroma_db"""
        v = Config._settings().value("chroma_path", "")
        return v if v else "data/chroma_db"

    @staticmethod
    def set_chroma_path(value: str) -> None:
        Config._settings().setValue("chroma_path", value)

    @staticmethod
    def get_chroma_collection() -> str:
        """默认 papers collection。后续 P2.5/P6 会加 paper_keywords / survey_sections"""
        v = Config._settings().value("chroma_collection", "")
        return v if v else "papers"

    @staticmethod
    def set_chroma_collection(value: str) -> None:
        Config._settings().setValue("chroma_collection", value)

    @staticmethod
    def get_chroma_embedding_model_path() -> str:
        """chroma 用的 embedding function 本地路径。默认 D:\\pythonProject\\BGE_Large_en_v1.5
        （已下好的 bge-large-en-v1.5，避免 chroma 默认 80MB onnx 下载）
        改这个值会触发重新建索引（dim 不一样），需要清空 data/chroma_db。"""
        v = Config._settings().value("chroma_embedding_model_path", "")
        return v if v else r"D:\pythonProject\BGE_Large_en_v1.5"

    @staticmethod
    def set_chroma_embedding_model_path(value: str) -> None:
        Config._settings().setValue("chroma_embedding_model_path", value)
