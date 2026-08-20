"""
agent.memory.embedder — 包装 QAgent 现有 Embedder（CPU 优先）

设计：
- 复用 agent.local_embedder.Embedder，不引入新模型
- 单例 + lazy load（首次调 encode 时才真正加载）
- 阻塞推理，但每次只 encode 1-2 条 memory 文本（短），CPU 可接受
- 线程安全：依赖 Embedder._model 自身的线程安全假设（PyTorch 推理在主线程安全）
"""
import logging
import threading
from typing import List, Optional, Union

import numpy as np

from .schemas import DEFAULT_EMBEDDING_DIM

logger = logging.getLogger(__name__)


class MemoryEmbedder:
    """
    记忆库专用 embedder（薄包装）。

    为什么不直接用 Embedder.embed()？
    - 加了 dim 缓存
    - 加了 is_ready() / model info 状态查询
    - 加了 lazy load 守卫（避免每次都查 EMBEDDING_ENABLED）
    """

    _instance: Optional["MemoryEmbedder"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._backend = None  # 'local' / 'openai' / None (未配置)
        self._dim: Optional[int] = None

    def is_ready(self) -> bool:
        """检查 embedding 是否可用。"""
        try:
            from agent.local_embedder import Embedder, EMBEDDER_TYPE, EMBEDDING_ENABLED
        except ImportError:
            return False
        if not EMBEDDING_ENABLED:
            return False
        if EMBEDDER_TYPE not in ("local", "openai"):
            return False
        # local 模式：检查 torch + sentence-transformers 是否导入成功
        if EMBEDDER_TYPE == "local":
            try:
                from agent.local_embedder import SentenceTransformer  # type: ignore
                return SentenceTransformer is not None
            except ImportError:
                return False
        # openai 模式：检查 API key
        if EMBEDDER_TYPE == "openai":
            from config import OPENAI_API_KEY
            return bool(OPENAI_API_KEY)
        return False

    def get_dim(self) -> int:
        """获取 embedding 维度。优先从配置读，否则返回默认值。"""
        if self._dim is not None:
            return self._dim
        try:
            from config import LOCAL_EMBEDDER_NAME, OPENAI_EMBEDDING_MODEL
            # 启发式：根据 model name 推断维度
            name = (LOCAL_EMBEDDER_NAME or "").lower()
            if "bge-large" in name or "bge_en_large" in name:
                self._dim = 1024
            elif "bge-base" in name or "bge_en_base" in name:
                self._dim = 768
            elif "bge-small" in name:
                self._dim = 512
            else:
                self._dim = DEFAULT_EMBEDDING_DIM
        except ImportError:
            self._dim = DEFAULT_EMBEDDING_DIM
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        """
        单条文本编码。返回 1D numpy array (dim,)。
        阻塞调用（CPU 30-80ms，BGE-large-en-v1.5）。
        """
        if not self.is_ready():
            raise RuntimeError(
                "Embedding 不可用。请检查 .env 中 EMBEDDING_ENABLED=true 且 EMBEDDER_TYPE=local/openai 已正确配置。"
            )
        from agent.local_embedder import Embedder
        vecs = Embedder.embed([text])
        if vecs is None or len(vecs) == 0:
            raise RuntimeError("Embedder.embed 返回空结果")
        v = np.asarray(vecs[0], dtype=np.float32)
        # 归一化以走余弦相似度（Embedder.embed 本地模式已 normalize_embeddings=True）
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v = v / norm
        return v

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量编码。返回 2D numpy array (n, dim)。
        """
        if not texts:
            return np.zeros((0, self.get_dim()), dtype=np.float32)
        if not self.is_ready():
            raise RuntimeError("Embedding 不可用")
        from agent.local_embedder import Embedder
        vecs = Embedder.embed(texts)
        if vecs is None or len(vecs) == 0:
            raise RuntimeError("Embedder.embed 返回空结果")
        arr = np.asarray(vecs, dtype=np.float32)
        # 归一化每一行
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        return arr / norms

    def get_backend_info(self) -> dict:
        """用于 debug / UI 显示。"""
        try:
            from config import EMBEDDER_TYPE, LOCAL_EMBEDDER_NAME, OPENAI_EMBEDDING_MODEL
            return {
                "enabled": self.is_ready(),
                "backend": EMBEDDER_TYPE,
                "local_model": LOCAL_EMBEDDER_NAME,
                "openai_model": OPENAI_EMBEDDING_MODEL,
                "dim": self.get_dim(),
            }
        except Exception as e:
            return {"enabled": False, "error": str(e)}


# ============================================================
# 全局单例 + 便捷函数
# ============================================================
def get_embedder() -> MemoryEmbedder:
    """获取全局 MemoryEmbedder 单例。"""
    return MemoryEmbedder()


def encode_text(text: str) -> np.ndarray:
    """便捷函数：encode 单条文本。"""
    return get_embedder().encode(text)


def encode_texts(texts: List[str]) -> np.ndarray:
    """便捷函数：encode 批量文本。"""
    return get_embedder().encode_batch(texts)
