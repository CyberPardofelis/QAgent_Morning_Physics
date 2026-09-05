import fitz  # PyMuPDF
from PyPDF2 import PdfReader

# 2026-09-04 改:默认走 fitz(PyMuPDF)而不是 PyPDF2
# 原因:
# - PyPDF2 对扫描/公式/LaTeX 排版的论文 PDF 经常吐截断或乱码
#   (用户报"analyze_paper 撞 pdftotext 截断" 根因)
# - PyMuPDF 排版还原更好,数字/公式/表格文字完整
# - LLM 拿全文分析(JSON parse 不再因截断失败)
# 兼容:调用方仍可显式传 method="pypdf2" 走老路径(fitz 装坏/不兼容时)
DEFAULT_METHOD = "fitz"


def extract_text_from_pdf(filepath, method=DEFAULT_METHOD, max_chars=None):
    """
    提取PDF文本(2026-09-04 改:默认 fitz + 可选 max_chars)

    Args:
        filepath: PDF 路径
        method: 'fitz' (默认, PyMuPDF, 完整文本) 或 'pypdf2' (旧路径, 易截断)
        max_chars: 截断上限(默认 None = 不截断,让 LLM 拿全文)
                   设大数可避免 32K+ token 提示(典型 50000 ~= 12K token)

    Returns:
        提取的文本(str),失败抛异常让调用方处理

    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 提取过程失败(库报错/格式损坏)
    """
    import os
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF 不存在: {filepath}")

    text = ""
    if method == "fitz":
        try:
            doc = fitz.open(filepath)
            for page in doc:
                # 2026-09-04 加:每页 try,单页失败不连累整文
                try:
                    text += page.get_text()
                except Exception as e:
                    # 单页失败不致命,继续下页;末尾追加警告便于排查
                    text += f"\n[页提取失败(已跳过): {e}]\n"
            doc.close()
        except Exception as e:
            # fitz 整体失败(库问题/文件损坏) → 降级 pypdf2
            try:
                reader = PdfReader(filepath)
                for page in reader.pages:
                    try:
                        content = page.extract_text()
                        if content:
                            text += content + "\n"
                    except Exception:
                        continue
            except Exception as e2:
                raise RuntimeError(
                    f"PDF 提取失败(fitz 报错 {e};pypdf2 降级也失败 {e2})"
                ) from e2
    else:
        # 显式走 pypdf2 旧路径
        reader = PdfReader(filepath)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return text


def chunk_text(text, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
