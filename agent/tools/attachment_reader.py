"""
附件文本提取统一接口（移植自 WAgent）。

设计目标：
- InputPanel 调用一个入口 extract_text(path)，不再关心后端
- 后端可插拔（PDF / 纯文本 / Word），失败时降级到下一个
- 暴露 SUPPORTED_EXTS 给文件对话框过滤用

支持类型：
- .pdf  → pdf_utils.extract_text_from_pdf（fitz 优先，pypdf2 兜底）
- .txt  → 纯文本（utf-8/utf-8-sig/gbk 兜底）
- .md   → 纯文本
- .docx → 尝试 win32com（需要 Office / WPS）

注意：
- 调用方要捕获异常并向用户提示（这里只抛，不弹 UI）
- compose_llm_user_content 用于把附件内容拼到 LLM 消息
"""
from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from pathlib import Path

# 支持的扩展名（小写，含点）
SUPPORTED_EXTS: list[str] = [".docx", ".pdf", ".txt", ".md"]

# 文件对话框用的过滤串
FILE_DIALOG_FILTER: str = (
    "可解析文档 (*.docx *.pdf *.txt *.md);;"
    "Word 文档 (*.docx);;"
    "PDF (*.pdf);;"
    "文本文件 (*.txt *.md);;"
    "所有文件 (*.*)"
)

# 单文件最大提取字符数（防止一次性塞爆 prompt）
MAX_CHARS: int = 50_000


# ============================================================
# 各种文件的提取器
# ============================================================
def _read_text_file(path: str) -> str:
    """纯文本类：txt / md / 其它文本后缀的兜底"""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_docx_via_com(path: str) -> str:
    """用 Word COM 把 .docx 解成纯文本（保留段落分隔）"""
    import win32com.client  # 仅在用到时导入，失败也好定位

    word = win32com.client.DispatchEx("Word.Application")
    try:
        word.Visible = False
        word.DisplayAlerts = False
        doc = word.Documents.Open(
            os.path.abspath(path),
            ReadOnly=True,
            ConfirmConversions=False,
        )
        try:
            return doc.Content.Text
        finally:
            doc.Close(SaveChanges=False)
    finally:
        word.Quit()


def _read_pdf(path: str) -> str:
    """复用已有的 pdf_utils（按文件路径懒加载，绕开 agent.tools 包的 __init__）"""
    _pdf_utils_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pdf_utils.py"
    )
    _spec = importlib.util.spec_from_file_location("_pdf_utils", _pdf_utils_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    extract_text_from_pdf = _mod.extract_text_from_pdf

    try:
        return extract_text_from_pdf(path, method="fitz")
    except Exception:
        return extract_text_from_pdf(path, method="pypdf2")


# 扩展名 -> 提取器（按优先级排列；同后缀多个时前面的失败才用后面的）
_EXTRACTORS: dict[str, list[Callable[[str], str]]] = {
    ".docx": [_read_docx_via_com],
    ".pdf": [_read_pdf],
    ".txt": [_read_text_file],
    ".md": [_read_text_file],
}


def is_supported(path: str) -> bool:
    return Path(path).suffix.lower() in _EXTRACTORS


def extract_text(path: str) -> str:
    """
    提取文件文本。失败抛异常，由调用方决定怎么提示用户。

    返回的文本会被截断到 MAX_CHARS 以内。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"附件不存在: {path}")

    ext = Path(path).suffix.lower()
    extractors = _EXTRACTORS.get(ext)
    if not extractors:
        raise ValueError(f"暂不支持的附件类型: {ext}")

    last_err: Exception | None = None
    for fn in extractors:
        try:
            text = fn(path)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "\n\n[…内容过长，已截断…]"
        return text

    raise RuntimeError(
        f"无法读取 {os.path.basename(path)} ({ext})：{last_err}"
    ) from last_err


# ============================================================
# 拼给 LLM 的 user 消息（带附件内容展开）
# ============================================================
def compose_llm_user_content(text: str, attachments: list) -> str:
    """
    把附件文本拼到 user 消息后面（只用于本轮发给 LLM，不会写库）

    attachments: [{name, size, path, content}, ...]
    - content: 已由 extract_text 提取出的文本
    """
    if not attachments:
        return text or ""

    blocks = []
    for a in attachments:
        content = a.get("content") or ""
        size_str = a.get("size", "")
        chars = len(content)
        chars_hint = f"{chars} 字" if chars < 10000 else f"{chars // 1000}k 字"
        blocks.append(
            f'<附件 file="{a.get("name", "")}" size="{size_str}" '
            f'chars="{chars_hint}">\n{content}\n</附件>'
        )
    attachment_section = "\n\n".join(blocks)
    if text:
        return f"{text}\n\n{attachment_section}"
    return attachment_section
