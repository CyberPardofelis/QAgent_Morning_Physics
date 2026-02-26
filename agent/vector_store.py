import faiss
import os
import numpy as np
import pickle

INDEX_PATH = "index/faiss.index"
META_PATH = "index/meta.pkl"

class VectorStore:

    def __init__(self, dim=1024):
        self.dim = dim
        self.index = None
        self.meta = []

        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.meta = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(dim)

    def add(self, embeddings, metadata):
        self.index.add(embeddings)
        self.meta.extend(metadata)
        self.save()

    def search(self, query_embedding, top_k=5):
        D, I = self.index.search(query_embedding, top_k)
        results = [self.meta[i] for i in I[0]]
        return results

    def save(self):
        os.makedirs("index", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.meta, f)