import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from config import MODEL_PATH
import os

class LocalEmbedder:
    _model = None

    @classmethod
    def get_model(cls):
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
                "BAAI/bge-large-en-v1.5",
                device=device
            )
        return cls._model

    @classmethod
    def embed(cls, texts):
        model = cls.get_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings