"""
工具内部调 LLM 的统一入口（agent.tools.llm._call_llm）。

之前版本直接用 agent.tools.config 模块级常量（DEEPSEEK_API_KEY），
但 config 只在 import 时读一次 .env，且没用 agent.infra.api_keys 的 3 层兜底，
所以改 key 后不生效，QSettings 里的 key 也读不到。

新版修复：
- 改用 agent.infra.api_keys.get_api_key("deepseek") 动态读（保证永远最新）
- 加旧 model 名兼容映射（deepseek-reasoner / deepseek-chat → 新模型）
- 加友好错误处理（不再只 print 然后 return ""）
"""
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


def _get_deepseek_key() -> str:
    """动态从 agent.infra.api_keys 读 key（3 层兜底：.api_keys.json > QSettings > .env）"""
    try:
        from agent.infra.api_keys import get_api_key
        return get_api_key("deepseek")
    except Exception as e:
        logger.warning("[tools.llm] 读 api_key 失败，回退到 .env: %s", e)
        from agent.tools.config import DEEPSEEK_API_KEY
        return DEEPSEEK_API_KEY


def _resolve_legacy_model(model: str, enable_thinking: bool) -> tuple:
    """
    兼容旧 model 名（与 agent.controller.LEGACY_MODEL_MAP 保持一致）
    - deepseek-reasoner → deepseek-v4-pro（开启思考，effort='max'）
    - deepseek-chat     → deepseek-v4-flash（关闭思考）
    """
    LEGACY = {
        "deepseek-reasoner": ("deepseek-v4-pro", True, "max"),
        "deepseek-chat":     ("deepseek-v4-flash", False, "high"),
    }
    if model in LEGACY:
        new_model, want_thinking, effort = LEGACY[model]
        logger.info(
            "[tools.llm] 检测到旧 model 名 %r，自动映射到 %r (thinking=%s, effort=%s)",
            model, new_model, want_thinking, effort,
        )
        return new_model, want_thinking, effort
    return model, enable_thinking, "high"


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
) -> str:
    """
    工具内部用的 LLM 调用（同步、非流式）。
    用于 generate_daily_summary_arxiv / generate_aps_daily_summary / analyze_paper 等工具内的总结生成。

    返回：成功时为 LLM 输出文本；失败时为带错误提示的字符串（不再 print 错误就吞掉）。
    """
    api_key = _get_deepseek_key()
    if not api_key:
        return (
            "❌ DeepSeek API Key 未配置。请在「首选项 → LLM API」中填写，"
            "或在 .env 文件中设置 DEEPSEEK_API_KEY。"
        )

    # 旧 model 名兼容
    model, enable_thinking, reasoning_effort = _resolve_legacy_model(model, enable_thinking)

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=timeout)
    except Exception as e:
        return f"❌ 创建 DeepSeek 客户端失败: {e}"

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

    try:
        return client.chat.completions.create(**kwargs).choices[0].message.content
    except Exception as e:
        err_name = type(e).__name__
        err_str = str(e)[:300]
        # 401 / AuthenticationError（最常见，key 无效/过期/被 governor 拒）
        if "Authentication" in err_name or "401" in err_str or "Incorrect API key" in err_str or "governor" in err_str.lower():
            return (
                f"❌ DeepSeek API 鉴权失败（{err_name}）\n\n"
                f"可能原因：\n"
                f"  1. API key 无效或已过期 — 在「首选项 → LLM API」或 .env 中检查\n"
                f"  2. DeepSeek 账户余额不足\n"
                f"  3. 你的 key 之前在 OpenAI 风控被拦截（governor），需要重新生成 key\n\n"
                f"原始错误：{err_str}"
            )
        # 超时
        if "Timeout" in err_name:
            return f"❌ DeepSeek 请求超时（{err_name}）：{err_str}"
        # 限流
        if "RateLimit" in err_name or "429" in err_str:
            return f"❌ DeepSeek 触发限流，建议稍后重试。\n\n原始：{err_str}"
        # 其它
        return f"❌ DeepSeek 调用失败（{err_name}）：{err_str}"
