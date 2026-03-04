import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDER_TYPE, OPENAI_EMBEDDING_MODEL, MODEL_PATH, OPENAI_API_KEY, LOCAL_EMBEDDER_NAME
import os
import openai


class Embedder:
    _model = None  # 本地模型缓存
    _openai_client = None  # OpenAI 客户端缓存

    @classmethod
    def get_model(cls):
        if EMBEDDER_TYPE == "local":
            if cls._model is None:
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