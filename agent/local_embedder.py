import importlib
import threading
import numpy as np
from config import EMBEDDER_TYPE, OPENAI_EMBEDDING_MODEL, MODEL_PATH, OPENAI_API_KEY, LOCAL_EMBEDDER_NAME
import os
import openai

# 2026-09-05 P0 fix: 改读 Config.get_embedding_enabled() (QSettings) 而非 env EMBEDDING_ENABLED
# 之前只读 .env, 跟 UI 设置的 QSettings 不同步, 偶发 BGE 在用户禁用后仍被加载
def _is_embedding_enabled() -> bool:
    """统一入口: QSettings 优先, env 兜底, 默认 False"""
    try:
        from config import Config
        return Config.get_embedding_enabled()
    except Exception:
        pass
    # env 兜底
    return os.getenv("EMBEDDING_ENABLED", "false").lower() in ("true", "1", "yes")


_EMBEDDING_ENABLED = _is_embedding_enabled()
torch = None
SentenceTransformer = None
if _EMBEDDING_ENABLED:
    try:
        torch = importlib.import_module('torch')
        sentence_transformers = importlib.import_module('sentence_transformers')
        SentenceTransformer = sentence_transformers.SentenceTransformer
    except ImportError:
        pass

class Embedder:
    _model = None  # 本地模型缓存
    _openai_client = None  # OpenAI 客户端缓存
    # 2026-09-02 P0 fix: 双检查锁防 bge-large-en-v1.5 (~1.3GB) 被多线程并发加载
    # 之前裸 `if cls._model is None: cls._model = SentenceTransformer(...)`,
    # N 个线程同时 first call 会拉起 N 份模型 → N * 1.3GB 内存 → 蓝屏
    _init_lock = threading.Lock()

    @classmethod
    def get_model(cls):
        # 2026-09-05 P0 fix: 运行时再检查, 防止 SentenceTransformer 已被 import 但用户禁用了
        if not _is_embedding_enabled():
            raise RuntimeError(
                "Embedding 已被禁用 (EMBEDDING_ENABLED=False). 改用 OpenAI 嵌入或启用本地模型."
            )
        if EMBEDDER_TYPE == "local":
            if cls._model is None:           # 快路径: 已加载直接返
                with cls._init_lock:         # 慢路径: 抢锁防并发
                    if cls._model is None:   # 双重检查: 持锁后再确认一次
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        # 自动判断：是本地路径还是在线ID
                        model_source = MODEL_PATH
                        if os.path.isdir(model_source):
                            print(f"[Embedding] Loading model from local path: {model_source}")
                        else:
                            print(f"[Embedding] Local path not found, will try to download from Hugging Face: BAAI/bge-large-en-v1.5")
                        print(f"[Embedding] Loading model on {device} ...")
                        cls._model = SentenceTransformer(
                            LOCAL_EMBEDDER_NAME,
                            device=device
                        )
            return cls._model
        elif EMBEDDER_TYPE == "openai":
            if cls._openai_client is None:
                if not OPENAI_API_KEY:
                    raise ValueError("OpenAI API key not configured.")
                openai.api_key = OPENAI_API_KEY
                cls._openai_client = openai.Embedding  # 直接使用 openai 模块作为客户端
            return cls._openai_client  # 返回模块本身，方便后续调用

        else:
            raise ValueError(f"Unknown EMBEDDER_MODE: {EMBEDDER_TYPE}")

    @classmethod
    def embed(cls, texts):
        # 运行时再检查 (防止 import 时检查过后用户改了设置)
        if not _is_embedding_enabled():
            raise RuntimeError("Embedding 已被禁用 (EMBEDDING_ENABLED=False)")
        model = cls.get_model()
        if EMBEDDER_TYPE == "local":
            # 本地模型：直接调用 encode
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings

        elif EMBEDDER_TYPE == "openai":
            # OpenAI API：调用 embedding 接口
            # 注意：text-embedding-ada-002 支持一次输入多个文本，但需注意长度限制
            # 此处简化处理，未做分块和错误重试
            response = model.create(
                input=texts,
                model=OPENAI_EMBEDDING_MODEL
            )
            # 按输入顺序提取 embedding
            embeddings = [item['embedding'] for item in response['data']]
            return np.array(embeddings)