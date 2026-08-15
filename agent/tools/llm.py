"""
工具内部调 LLM 的统一入口（agent.tools.llm._call_llm）。

之前版本直接用 agent.tools.config 模块级常量（DEEPSEEK_API_KEY），
但 config 只在 import 时读一次 .env，且没用 agent.infra.api_keys 的 3 层兜底，
所以改 key 后不生效，QSettings 里的 key 也读不到。

新版修复：
- 改用 agent.infra.api_keys.get_api_key(provider) 动态读（保证永远最新）
- 旧 model 名兼容 + 错误分类统一在 agent.llm_clients
- 支持 provider 参数（"deepseek" / "minimax" / ...），默认 deepseek 保持向后兼容
"""
import logging
from openai import OpenAI

from agent.llm_clients import (
    PROVIDERS,
    get_client,
    resolve_legacy_model,
    classify_llm_error,
)

logger = logging.getLogger(__name__)


def _get_provider_key(provider: str) -> str:
    """动态从 agent.infra.api_keys 读 key（3 层兜底：.api_keys.json > QSettings > .env）"""
    try:
        from agent.infra.api_keys import get_api_key
        return get_api_key(provider)
    except Exception as e:
        logger.warning("[tools.llm] 读 %s api_key 失败: %s", provider, e)
        # 最后兜底：读模块级常量（仅 deepseek / openai 有）
        try:
            from agent.tools import config as _cfg
            return getattr(_cfg, f"{provider.upper()}_API_KEY", "") or ""
        except Exception:
            return ""


def _call_llm(
        prompt: str,
        model: str = "deepseek-v4-flash",
        enable_thinking: bool = False,
        reasoning_effort: str = "high",  # "high" 或 "max"
        temperature: float = 0.3,
        top_p: float = 0.8,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 40000,
        timeout: int = 180,
        provider: str = "deepseek",
) -> str:
    """
    工具内部用的 LLM 调用（同步、非流式）。
    用于 generate_daily_summary_arxiv / generate_aps_daily_summary / analyze_paper 等工具内的总结生成。

    参数:
        provider: LLM provider id（"deepseek" / "minimax" / "openai"），默认 "deepseek" 保持向后兼容。
                  客户端、API key、base_url 都从 agent.llm_clients 拿，保持 3 层兜底。

    返回：成功时为 LLM 输出文本；失败时为带错误提示的字符串（不再 print 错误就吞掉）。
    """
    # 校验 provider
    if provider not in PROVIDERS:
        return f"❌ 未知 LLM provider: {provider!r}。支持: {list(PROVIDERS.keys())}"

    cfg = PROVIDERS[provider]
    display_name = cfg.display_name

    # 1. 拿 client（自动处理 3 层兜底 + 缓存）
    try:
        client = get_client(provider)
    except Exception as e:
        # get_client 内部已经封装错误，e 是 LLMClientError
        return classify_llm_error(display_name, e)

    # 2. 旧 model 名兼容（仅 deepseek 有 legacy 映射，其他 provider 透传）
    model, enable_thinking, reasoning_effort = resolve_legacy_model(model, enable_thinking)

    # 3. 构造请求参数
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    if enable_thinking:
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        })

    # MiniMax 接口对 frequency_penalty / presence_penalty 支持有限，统一剥离避免 400。
    if provider == "minimax":
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)

    try:
        return client.chat.completions.create(**kwargs).choices[0].message.content
    except Exception as e:
        return classify_llm_error(display_name, e)
