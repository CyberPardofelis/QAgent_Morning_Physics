import PyPDF2
from config import EMBEDDING_ENABLED
from agent.vector_store import VectorStore

import importlib
import logging
logger = logging.getLogger(__name__)

Embedder = None
if EMBEDDING_ENABLED:
    try:
        embedder_module = importlib.import_module('agent.local_embedder')
        Embedder = embedder_module.Embedder
    except ImportError as e:
        # 2026-09-02 P0 fix: 裸 `except:` 会吞掉 KeyboardInterrupt / SystemExit / MemoryError,
        # 用户按 Ctrl+C 都无法中断进程。改成只捕获 ImportError。
        logger.warning("[paper_indexer] 加载 agent.local_embedder 失败(非致命): %s", e)

def extract_text(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def index_paper(pdf_path):
    print(f"[Indexer] Processing {pdf_path}")
    text = extract_text(pdf_path)

    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]

    embeddings = Embedder.embed(chunks)

    store = VectorStore()
    metadata = [{"source": pdf_path, "chunk": chunk} for chunk in chunks]

    store.add(embeddings, metadata)
    print("[Indexer] Done.")