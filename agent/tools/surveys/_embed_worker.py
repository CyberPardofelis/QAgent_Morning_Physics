"""
_embed_worker — 编码子进程本体(只被 _embed.py 拉起, 不要直接跑)。

单独成文件的原因: 它会在一个**没有 QWebEngineView 的进程**里加载
sentence-transformers, 从而把 `.env` 记的 0xC0000005(QWebEngineView + BGE
共享 native) 隔离出去 —— 崩了也只是这次编码失败, 主进程不受影响。

用法: python _embed_worker.py <in.json> <out.npy>
    in.json  = {"model": "...", "texts": ["...", ...]}
    out.npy  = float32 向量矩阵 (n, dim), 已 L2 归一化
"""
from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    src, dst = argv[1], argv[2]
    with open(src, encoding="utf-8") as f:
        payload = json.load(f)
    from sentence_transformers import SentenceTransformer

    device = "cuda" if _cuda_ok() else "cpu"
    m = SentenceTransformer(payload["model"], device=device)
    v = m.encode(payload["texts"], convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=False)
    import numpy as np
    np.save(dst, v.astype("float32"))
    return 0


def _cuda_ok() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main(sys.argv))
