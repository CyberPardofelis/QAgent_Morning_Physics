# agent/stream_utils.py
import re

def split_markdown_bubbles(content: str) -> list[str]:
    """
    按 Markdown 三级及以上标题（###、#### 等）分割内容。
    每个标题及其后的内容作为一个独立的气泡。
    如果没有任何三级标题，则返回整个内容作为单个气泡。
    """
    if not content or not content.strip():
        return []

    # 确保第一行也被匹配（如果本身就是标题）
    content = '\n' + content

    # 分割点：行首的 ### 或更多 # 号（忽略前导空格）
    parts = re.split(r'(?=^#{2,}\s+)', content, flags=re.MULTILINE)

    # 去除可能因前置换行产生的空字符串
    bubbles = [part.strip() for part in parts if part.strip()]
    return bubbles