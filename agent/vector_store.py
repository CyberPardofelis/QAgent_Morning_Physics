import faiss
import os
import pickle
import warnings

# ============================================================
# ⚠️ DEPRECATED — 2026-09-02 P0 chroma 切换后保留作孤儿代码
# 实际项目已切到 agent.vector_db.VectorDB（chroma + bge-large-en-v1.5）
# 老 faiss 索引位于 data/vector_store/，P0.5 全量重建后考虑删除。
# 任何新代码请勿 import 本模块。
# ============================================================
warnings.warn(
    "agent.vector_store (faiss) is deprecated since 2026-09-02 P0."
    " Use agent.vector_db.VectorDB instead.",
    DeprecationWarning,
    stacklevel=2,
)

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