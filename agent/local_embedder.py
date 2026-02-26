import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class LocalEmbedder:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
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