import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

OPENAI_MODEL = "gpt-4o-mini"
DEEPSEEK_MODEL = "deepseek-reasoner"
DEEPSEEK_MODEL_CHAT = "deepseek-chat"

# 是否启用需要 embedding 的功能，如本地知识库查询等（本地 GPU/在线 API）
EMBEDDING_ENABLED = False   # 默认为 False，用户可按需开启
# 使用本地或在线embedder：
EMBEDDER_TYPE = "local"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"  # 或其他模型
MODEL_PATH = r"D:\pythonProject\BGE_Large_en_v1.5"
LOCAL_EMBEDDER_NAME = "BAAI/bge-base-zh" # 或其他模型


DATA_DIR = "data/papers"
VECTOR_DIR = "data/vector_store"

PRL_SEC = "quantum-information-science"
PRA_SEC = "quantum-information-science-and-technology"