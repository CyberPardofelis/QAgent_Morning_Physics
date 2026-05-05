from openai import OpenAI
from .config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL


# ----------------------------
# 内部函数：调用LLM提取结构化信息
# ----------------------------
''' # 旧版本
def _call_llm(prompt: str) -> str:
    """使用DeepSeek模型（非流式）生成分析结果"""
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        timeout=150
    )
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=8000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM 调用错误: {e}")
        return ""
# '''

def _call_llm(
        prompt: str,
        model: str = "deepseek-v4-flash",
        enable_thinking: bool = False,
        reasoning_effort: str = "high",  # "high" 或 "max"
        temperature: float = 0.3,
        top_p: float = 0.8,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 40000,  # V4 最大384k
        timeout: int = 180,
) -> str:

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com", timeout=timeout)

    # 自动处理旧模型名称
    if model == "deepseek-chat":
        model = "deepseek-v4-flash"
        enable_thinking = False
    elif model == "deepseek-reasoner":
        model = "deepseek-v4-pro"
        enable_thinking = True
        reasoning_effort = "max"

    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}

    if enable_thinking:
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["extra_body"] = {"enable_thinking": True}
        # 思考模式下温度等参数不生效，但传了也无害；为清晰就不传了
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
        print(f"LLM 调用错误: {e}")
        return ""