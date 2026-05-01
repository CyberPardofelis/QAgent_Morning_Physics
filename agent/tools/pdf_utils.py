import fitz  # PyMuPDF
from PyPDF2 import PdfReader

# ----------------------------
# 工具函数：提取PDF文本（支持两种库）
# ----------------------------
def extract_text_from_pdf(filepath, method="pypdf2"):
    """提取PDF文本，method可选 'fitz' 或 'pypdf2'"""
    if method == "fitz":
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    else:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text

# ----------------------------
# 工具函数：文本分块
# ----------------------------
def chunk_text(text, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks