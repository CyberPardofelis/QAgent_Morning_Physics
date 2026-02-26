import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

OPENAI_MODEL = "gpt-4o-mini"
DEEPSEEK_MODEL = "deepseek-reasoner"
DEEPSEEK_MODEL_CHAT = "deepseek-chat"

DATA_DIR = "data/papers"
VECTOR_DIR = "data/vector_store"

INTERESTED_TOPICS = [
    "quantum information",
    "quantum computing",
    "quantum error correction",
    "quantum machine learning"
]