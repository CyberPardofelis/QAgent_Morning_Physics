from openai import OpenAI
from config import (
    OPENAI_API_KEY,
    DEEPSEEK_API_KEY
)

def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

def get_deepseek_client():
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )