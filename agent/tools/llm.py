"""
工具内部调 LLM 的统一入口（agent.tools.llm._call_llm）。

之前版本直接用 agent.tools.config 模块级常量（DEEPSEEK_API_KEY），
但 config 只在 import 时读一次 .env，且没用 agent.infra.api_keys 的 3 层兜底，
所以改 key 后不生效，QSettings 里的 key 也读不到。

新版修复：
- 改用 agent.infra.api_keys.get_api_key("deepseek") 动态读（保证永远最新）
- 旧 model 名兼容 + 错误分类统一在 agent.llm_clients
"""
import logging
from openai import OpenAI

from agent.llm_clients import resolve_legacy_model, classify_llm_error

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
    model, enable_thinking, reasoning_effort = resolve_legacy_model(model, enable_thinking)

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
        return classify_llm_error("DeepSeek", e)
