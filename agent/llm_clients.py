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
- minimax:  MiniMax（MiniMax）官方 API（OpenAI 兼容，国内稳定，支持工具调用 + 思考模式）
"""
import hashlib
import logging
import threading
from dataclasses import dataclass

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


PROVIDERS: dict[str, ProviderConfig] = {
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
    "minimax": ProviderConfig(
        name="minimax",
        display_name="MiniMax",
        # 2026-09-02 修(抄 SAgent):默认走国内端点 api.minimaxi.com/v1。
        # 用户是国内账号(同 SAgent 2026-08-18 排查:同 key 配 minimaxi.com/v1 → 200,
        # 配 minimax.io/v1 → 401 2049)。官方原话: "Match Global keys to
        # minimax.io and Mainland China keys to minimimaxi.com."
        # 跨 region 走 MiniMax 服务端会判 key 失效,返回 2049 invalid api key。
        # 如果用户未来切到国际账号,在 SettingsDialog → LLM API → MiniMax →
        # base_url 改 https://api.minimax.io/v1 即可。
        # 注意:必须保留 /v1 路径前缀,OpenAI SDK 会自动拼 /chat/completions。
        default_base_url="https://api.minimaxi.com/v1",
        default_model="MiniMax-M3",
    ),
}


# ============================================================
# MiniMax 真实支持的 model 列表(2026-09-02 抄 SAgent)
# 用于校验/防御 user 在 QSettings / SettingsDialog 里配 model。
# 注意:没有"Reasoning"命名的 model — thinking/reasoning 是 extra_body.thinking 启用。
# 未来 minimax 加 model 时,这里和 settings_dialog 的 dropdown 都要同步更新。
# ============================================================
MINIMAX_VALID_MODELS = (
    "MiniMax-M2",
    "MiniMax-M2.1",
    "MiniMax-M2.1-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "MiniMax-M3",
)


# 已知失效的 base_url 历史值(用于自动清理 QSettings 残留)
# ⚠️ 重要:这里只放"全球范围都失效"的 URL。区域端点不能加进来!
#   - 国内端点 api.minimaxi.com/v1 对国内账号是有效的
#   - 国际端点 api.minimax.io/v1 对国际账号是有效的
#   - 跨 region 走 MiniMax 服务端会判 key 失效(2049),不是 URL 失效
# 添加新条目时,附上注释说明何时/为何失效
_DEPRECATED_BASE_URLS = {
    # 2026-09-02: QAgent 历史错误值(全 region 不可用,会回 401)
    # 用户最初配的 "https://api.MiniMax.com/v1" 是无效的占位域名
    # → 自动回退到 minimaxi.com/v1
    "https://api.MiniMax.com/v1",
    # 2026-08: 旧 anthropic 端点(全 region 不可用,404)
    # https://api.minimaxi.com/anthropic 在 MiniMax 域名迁移后已全局失效
    "https://api.minimaxi.com/anthropic",
    # 故意不放 minimaxi.com/v1 — 它对国内账号是有效 URL
    # 误加这条会导致国内用户被强制回退到国际端点,key 跨 region 报 2049
}


# ============================================================
# 自定义异常
# ============================================================
class LLMClientError(RuntimeError):
    """LLM 客户端相关错误（API key 缺失、网络错误、配置错误等）"""

    def __init__(self, provider: str, message: str, original: Exception | None = None):
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
_clients: dict[str, tuple] = {}  # provider_name -> (client, base_url, api_key_hash)
_lock = threading.Lock()


def _fingerprint(value: str) -> str:
    """短 hash 指纹，用于检测 api_key 是否变化。不用于安全场景。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _get_user_base_url(provider_name: str) -> str:
    """
    从 QSettings 读用户配置的 base_url（覆盖默认值）。
    找不到返回空字符串，由 caller 决定回退到默认值。

    2026-09-02 加(抄 SAgent):如果用户配置的值在 _DEPRECATED_BASE_URLS 里
    (旧版失效的 URL 残留),自动忽略,等同于没配置(返回空串 → caller 回退到新默认值)。
    这样用户不需要手动去 SettingsDialog 清掉旧值。
    """
    try:
        from config.settings import get_settings
        settings = get_settings()
        raw = (settings.value(f"{provider_name}_base_url", "") or "").strip()
        if raw and raw in _DEPRECATED_BASE_URLS:
            logger.warning(
                "[LLM] 忽略 %s provider 的失效 base_url=%s,自动使用新默认值",
                provider_name, raw,
            )
            return ""
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


def reset_client_cache(provider_name: str | None = None) -> None:
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
# 旧 model 名兼容（统一来源，controller 和 tools/llm 都从这 import）
# ============================================================
LEGACY_MODEL_MAP: dict[str, tuple] = {
    # 旧 model 名 -> (新 model, enable_thinking, reasoning_effort)
    "deepseek-reasoner": ("deepseek-v4-pro", True, "max"),
    "deepseek-chat":     ("deepseek-v4-flash", False, "high"),
}


def resolve_legacy_model(model: str, enable_thinking: bool) -> tuple:
    """
    旧 model 名 → 新 model 的兼容映射。
    未命中时直接返回 (model, enable_thinking, "high")。
    """
    if model in LEGACY_MODEL_MAP:
        new_model, want_thinking, effort = LEGACY_MODEL_MAP[model]
        logger.info(
            "[llm_clients] 旧 model %r → %r (thinking=%s, effort=%s)",
            model, new_model, want_thinking, effort,
        )
        return new_model, want_thinking, effort
    return model, enable_thinking, "high"


# ============================================================
# LLM 错误分类（401/Timeout/RateLimit/其他）→ 用户友好提示
# ============================================================
def classify_llm_error(provider_display: str, exc: Exception) -> str:
    """
    把 OpenAI 兼容 SDK 抛出的异常分类成用户友好的错误提示。
    provider_display: 用于错误头部的可读 provider 名（如 "DeepSeek" / "OpenAI"）
    返回 markdown 错误消息。
    """
    err_name = type(exc).__name__
    err_str = str(exc)[:500]  # 2026-09-02 加长:让 402/403 错误体的 JSON 能完整显示

    # 2026-09-02 加:MiniMax 402 insufficient_balance_error (1008)
    # 用户经常以为是"key 错了",其实是账户余额不足 → 给明确指引
    if "402" in err_str or "insufficient_balance" in err_str.lower() or "1008" in err_str:
        return (
            f"💰 {provider_display} 账户余额不足（{err_name}）\n\n"
            f"**这不是 key 错误,是账户没钱了。**\n\n"
            f"处理方法：\n"
            f"  1. 登录 {provider_display} 官方充值页面（国内：platform.minimaxi.com / 国际：platform.minimax.io）\n"
            f"  2. 充值或购买套餐后,几分钟内自动恢复\n"
            f"  3. 充值后**直接重试**即可,不用重置 key\n\n"
            f"原始错误：{err_str}"
        )

    if (
        "Authentication" in err_name
        or "401" in err_str
        or "Incorrect API key" in err_str
        or "governor" in err_str.lower()
        or "2049" in err_str  # 2026-09-02 加:MiniMax region mismatch (跨 region 鉴权失败)
    ):
        return (
            f"❌ {provider_display} API 鉴权失败（{err_name}）\n\n"
            f"可能原因：\n"
            f"  1. API key 无效或已过期 — 在「首选项 → LLM API」或 .env 中检查\n"
            f"  2. {provider_display} 账户余额不足\n"
            f"  3. key 被风控系统拦截（governor），需要重新生成 key\n"
            f"  4. 跨 region 鉴权失败(2049):key 是国内账号但 base_url 走了国际,或反之\n\n"
            f"原始错误：{err_str}"
        )
    if "Timeout" in err_name:
        return f"❌ {provider_display} 请求超时（{err_name}）：{err_str}"
    if "RateLimit" in err_name or "429" in err_str:
        return f"❌ {provider_display} 触发限流，建议稍后重试。\n\n原始：{err_str}"
    return f"❌ {provider_display} 调用失败（{err_name}）：{err_str}"


# ============================================================
# 兼容旧接口（保留调用方不破坏）
# ============================================================
def get_deepseek_client() -> OpenAI:
    """兼容旧调用：等价于 get_client("deepseek")"""
    return get_client("deepseek")


def get_openai_client() -> OpenAI:
    """兼容旧调用：等价于 get_client("openai")"""
    return get_client("openai")


def get_minimax_client() -> OpenAI:
    """兼容旧调用：等价于 get_client("minimax")"""
    return get_client("minimax")
