import os
from pathlib import Path
from dotenv import load_dotenv

# 关键：模块加载时触发 .env 读取，否则模块级常量拿不到 .env 里的 key
# （注意：这只在第一次 import 时执行；后续 .env 修改需要重启进程才能生效）
load_dotenv()

# 路径配置
DATA_DIR = Path("data")
PAPER_DIR = DATA_DIR / "papers"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
INDEX_FILE = KNOWLEDGE_DIR / "index.json"
VECTOR_DIR = "vector_data"  # 向量索引存放路径
DB_PATH = KNOWLEDGE_DIR / "papers.db"

# 确保目录存在
os.makedirs(PAPER_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# 注意：
# - DEEPSEEK_API_KEY 在这里用 .env 兜底
# - 工具内部实际调 LLM 时（_call_llm），优先用 agent.infra.api_keys.get_api_key("deepseek")
#   （走 3 层兜底：.api_keys.json > QSettings > .env），保证改 key 后立即生效
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
EMBEDDING_ENABLED = os.getenv("EMBEDDING_ENABLED", "false").lower() in ("true", "1", "yes")