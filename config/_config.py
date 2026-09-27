"""
config._config — Config 类实现

独立成文件以避免 config/__init__.py 的循环 import。
（Config 类要 from .settings import get_settings，但 settings 要避免触发 __init__ 的全部执行）
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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

    # ---------- 成本控制（P0.7，2026-09-02 起） ----------
    @staticmethod
    def get_cost_tier() -> str:
        """当前允许的最高档 L0/L1/L2, 默认 L2 (个人用方便, 课题组用时调低)
        - L0: 只本地检索/SQL/下载, 0 次 LLM 调用
        - L1: + 单次 LLM 工具 (analyze_paper, save_memory 等)
        - L2: + 重型多轮 (ReAct 调研 / 综述生成, 等 P4 / P6 实现)
        """
        v = Config._settings().value("cost_tier", "")
        if v in ("L0", "L1", "L2"):
            return v
        return "L2"  # 默认 L2 开, 给个人用方便

    @staticmethod
    def set_cost_tier(value: str) -> None:
        if value not in ("L0", "L1", "L2"):
            raise ValueError(f"cost_tier 必须是 L0/L1/L2, 收到 {value!r}")
        Config._settings().setValue("cost_tier", value)

    @staticmethod
    def get_cost_ceiling_daily() -> int:
        """每日 LLM 调用上限, 默认 50 (P0.7 ROADMAP_NEXT §2 给课题组用时建议 50)"""
        v = Config._settings().value("cost_ceiling_daily", 50)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 50

    @staticmethod
    def set_cost_ceiling_daily(value: int) -> None:
        try:
            v = int(value)
        except (TypeError, ValueError):
            v = 50
        if v < 1:
            v = 1
        Config._settings().setValue("cost_ceiling_daily", v)

    # ---------- turn 级 deadline (9-27 加) ----------
    @staticmethod
    def get_turn_deadline_seconds() -> int:
        """
        单次用户对话(一个 turn)的总时间上限, 默认 1800s (30 分钟)。

        为什么需要 (9-27 P0):
        之前只有**单次** LLM 调用的 timeout(controller 默认 600s),
        工具执行则完全没有超时。于是最坏情况的墙上时间没有上界 ——
        综述工具 max_pool=200 时, extract_facts 会串行发 200 次 LLM,
        每次 timeout=180s, 全部发生在**一个** controller 轮次里,
        合计可达数小时, 期间 UI 只能干等, 也无法中断。

        语义: 这是"预算", 不是"每步超时"。controller 每轮开始前检查
        是否超预算, 超了就优雅收尾; 传给 LLM 的 timeout 也会被压到
        剩余预算以内, 保证单次调用不会把整个预算吃完。
        """
        v = Config._settings().value("turn_deadline_seconds", 1800)
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 1800
        # 至少给 60s, 否则单次长 prompt 都不够
        return v if v >= 60 else 60

    @staticmethod
    def set_turn_deadline_seconds(value: int) -> None:
        try:
            v = int(value)
        except (TypeError, ValueError):
            v = 1800
        if v < 60:
            v = 60
        Config._settings().setValue("turn_deadline_seconds", v)
