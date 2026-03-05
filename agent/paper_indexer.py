import PyPDF2
from agent.local_embedder import Embedder
from agent.vector_store import VectorStore

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