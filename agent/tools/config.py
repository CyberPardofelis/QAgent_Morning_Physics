import os
from pathlib import Path

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

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"
EMBEDDING_ENABLED = False