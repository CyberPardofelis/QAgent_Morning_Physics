"""
LLM 客户端管理（重构版，参考 WAgent 实现）。

设计目标：
- 单例缓存：每个 provider 只创建一次 client，避免每次调用都重建
- 超时：默认 60 秒
- 自定义异常：LLMClientError 带 provider 信息，便于上层统一处理
- Provider 注册表：声明式管理各 provider 配置
- 不再用 Config.get_xxx_api_key() 旧接口，统一从 agent.infra.api_keys 读

当前支持的 provider：
- deepseek: DeepSeek 官方 API（OpenAI 兼容）
- openai:   OpenAI 官方 API
"""
import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from openai import OpenAI

from agent.infra.api_keys import get_api_key

logger = logging.getLogger(__name__)


# ============================================================
# 常量
# ============================================================
DEFAULT_TIMEOUT = 60.0  # 秒


# ============================================================
# Provider 注册表
# ============================================================
@dataclass(frozen=True)
class ProviderConfig:
    name: str                # 内部 id
    display_name: str        # UI 显示名
    default_base_url: str    # 默认 base_url
    default_model: str       # 默认 model name


PROVIDERS: Dict[str, ProviderConfig] = {
    "deepseek": ProviderConfig(
        name="deepseek",
        display_name="DeepSeek",
        default_base_url="https://api.deepseek.com",
        default_model="deepseek-v4-flash",
    ),
    "openai": ProviderConfig(
        name="openai",
        display_name="OpenAI",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
    ),
}


# ============================================================
# 自定义异常
# ============================================================
class LLMClientError(RuntimeError):
    """LLM 客户端相关错误（API key 缺失、网络错误、配置错误等）"""

    def __init__(self, provider: str, message: str, original: Optional[Exception] = None):
        self.provider = provider
        self.original = original
        super().__init__(f"[{provider}] {message}")


# ============================================================
# 单例缓存
# 缓存 (client, base_url, api_key_fingerprint)，三者任一变化就重建。
# 这样用户在 SettingsDialog 改了 base_url / API key 后，无须手动调
# reset_client_cache，下一次 get_client 调用会自动检测到变化并重建。
# api_key 存 hash 而非明文，避免泄露到日志/异常。
# ============================================================
_clients: Dict[str, tuple] = {}  # provider_name -> (client, base_url, api_key_hash)
_lock = threading.Lock()


def _fingerprint(value: str) -> str:
    """短 hash 指纹，用于检测 api_key 是否变化。不用于安全场景。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _get_user_base_url(provider_name: str) -> str:
    """
    从 QSettings 读用户配置的 base_url（覆盖默认值）。
    找不到返回空字符串，由 caller 决定回退到默认值。
    """
    try:
        from config.settings import get_settings
        settings = get_settings()
        raw = (settings.value(f"{provider_name}_base_url", "") or "").strip()
        return raw
    except Exception:
        # QSettings 不可用（无 Qt 环境）时直接返回空
        return ""


def get_provider_config(provider_name: str) -> ProviderConfig:
    """获取 provider 配置；未知 provider 抛 LLMClientError"""
    if provider_name not in PROVIDERS:
        raise LLMClientError(
            "unknown",
            f"未知 provider: {provider_name!r}。支持: {list(PROVIDERS.keys())}",
        )
    return PROVIDERS[provider_name]


def get_client(provider_name: str, force_new: bool = False) -> OpenAI:
    """
    获取 OpenAI 兼容 client（按 provider 单例缓存）。

    缓存策略：缓存 (client, base_url, api_key_fingerprint)，
    三者任一变化就自动重建，**无须调用方手动 reset**——
    改了 SettingsDialog 里的 base_url / API key 后，下次调用自动用新值。

    参数:
        provider_name: provider id（"deepseek" / "openai"）
        force_new: 强制新建（一般不需要，缓存会自动检测变化）

    异常:
        LLMClientError: provider 不存在、API key 缺失、client 创建失败
    """
    cfg = get_provider_config(provider_name)

    api_key = get_api_key(provider_name)
    if not api_key:
        raise LLMClientError(
            provider_name,
            f"未设置 {cfg.display_name} API Key，请在设置中配置。",
        )

    # base_url：用户配置优先，否则用 provider 默认值
    user_base_url = _get_user_base_url(provider_name).strip()
    base_url = user_base_url or cfg.default_base_url

    key_fp = _fingerprint(api_key)

    with _lock:
        if force_new:
            _clients.pop(provider_name, None)
        elif provider_name in _clients:
            cached_client, cached_base_url, cached_key_fp = _clients[provider_name]
            if cached_base_url == base_url and cached_key_fp == key_fp:
                return cached_client
            _clients.pop(provider_name, None)

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=DEFAULT_TIMEOUT,
            )
            _clients[provider_name] = (client, base_url, key_fp)
            logger.info(
                "[LLM] 创建 %s client (base_url=%s, timeout=%.0fs)",
                cfg.display_name, base_url, DEFAULT_TIMEOUT,
            )
            return client
        except Exception as e:
            raise LLMClientError(
                provider_name, f"创建 client 失败: {e}", original=e
            ) from e


def reset_client_cache(provider_name: Optional[str] = None) -> None:
    """
    清空 client 缓存。
    - 不传参数：清空所有
    - 传 provider_name：只清该 provider
    场景：用户在 SettingsDialog 改了 API key 或 base_url 后调用
    """
    with _lock:
        if provider_name is None:
            _clients.clear()
        else:
            _clients.pop(provider_name, None)


# ============================================================
# 兼容旧接口（保留调用方不破坏）
# ============================================================
def get_deepseek_client() -> OpenAI:
    """兼容旧调用：等价于 get_client("deepseek")"""
    return get_client("deepseek")


def get_openai_client() -> OpenAI:
    """兼容旧调用：等价于 get_client("openai")"""
    return get_client("openai")
