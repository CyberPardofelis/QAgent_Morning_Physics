"""
agent.pdf_chunker — P0.5 section-aware 论文切分（2026-09-02）

为什么需要:
- 原来 500 token 无脑切分对学术论文是灾难(一段 method 跨 2 块,检索不准)
- section 级检索是科研核心场景("找 X 论文的 method 部分" / "对比 2 篇的 experiment")
- 走 ROADMAP_NEXT §4:决定 L0(章节级) / L1(段落级) / L2(表格/公式级)
  → 当前实现 L1(段落级,200-500 token) + 标 has_equation/has_table

设计:
- 用 pymupdf 提 PDF outline (bookmark),优先用(精确,无启发式误差)
- 没有 outline 时(PRL/PRA/部分 arxiv),用启发式:
  1. `^[IVX]+\\.\\s+[A-Z]` 罗马数字 (I. INTRODUCTION)
  2. `^\\d+(\\.\\d+)*\\s+[A-Z]` 数字编号 (1. Introduction)
  3. 全大写行(< 80 字符)
  4. 排除: Abstract / References / Acknowledgments / Appendix / Figure / Table / Bibliography
- 段落级切:双换行 / pyMuDP blocks;短段落(< 50 词)合并,长段落按句子切
- metadata: section / section_idx / chunk_idx / has_equation / has_table / page_start / page_end

撤回条件(ROADMAP §ADR-N1):
- 章节解析准确率 < 70% 时回到 500 token 无脑切
- 2026-09-02 现状:3 篇不同期刊 smoke 验证 (要 4 个 section 全部识别)
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================
@dataclass
class Chunk:
    """一个段落级 chunk"""
    text: str
    section: str           # e.g. "Introduction" / "Methods" / "References"
    section_idx: int       # 0-based, 在文档中是第几个 section
    chunk_idx: int         # 在该 section 内是第几段
    page_start: int
    page_end: int
    has_equation: bool = False
    has_table: bool = False
    # paper 级 metadata(pdf_chunker 不知道, 由调用方填)
    paper_id: str = ""
    title: str = ""
    year: Optional[int] = None
    journal: str = ""
    source: str = "arxiv"

    def to_metadata(self) -> Dict[str, Any]:
        """转成 chroma metadata(扁平,str/int/bool)"""
        return {
            "paper_id": self.paper_id,
            "section": self.section,
            "section_idx": int(self.section_idx),
            "chunk_idx": int(self.chunk_idx),
            "page_start": int(self.page_start),
            "page_end": int(self.page_end),
            "has_equation": bool(self.has_equation),
            "has_table": bool(self.has_table),
            "title": self.title,
            "year": int(self.year) if self.year else 0,
            "journal": self.journal,
            "source": self.source,
        }


# ============================================================
# 启发式 section 标题识别(无 outline 时的 fallback)
# ============================================================
# 优先级从高到低
_SECTION_PATTERNS = [
    re.compile(r'^([IVX]+)\.?\s+([A-Z][A-Z\s\-:,\(\)]{3,})$'),  # I. INTRODUCTION
    re.compile(r'^(\d+)\.\s+([A-Z][A-Za-z\s\-:,\(\)]{3,})$'),  # 1. Introduction
    re.compile(r'^(\d+)\.(\d+)\s+(.+)$'),  # 2.1 Sub-section (sub)
]

# 排除列表:这些行看似标题但不是章节
_EXCLUDE_TITLES = {
    "abstract", "references", "bibliography", "acknowledgments", "acknowledgements",
    "appendix", "appendices", "figure captions", "table of contents",
    "supplementary material", "supplementary information",
    "data availability", "author contributions", "competing interests",
    "funding", "conflict of interest",
}

# 启发式排除模式(PRL 短论文启发式常误识的)
# 2026-09-02 修:启发式会把 "PRL 116, 150502 (2016)" / "FIG. 1." / "[62] D." 误识为 section 标题
_EXCLUDE_TITLE_PATTERNS = [
    re.compile(r'^(PRL|PRA|PRB|PRX|PRM|PRPER|RMP|PR|PRAPPLIED)\s+\d+'),  # 期刊标号
    re.compile(r'^\d+\s+\d+\s*\(\d{4}\)'),   # "91, 052115 (2015)" doi 片段
    re.compile(r'^FIG\.?\s*\d+', re.IGNORECASE),  # FIG. 1
    re.compile(r'^TABLE\s+[IVX]+\.?', re.IGNORECASE),  # TABLE I
    re.compile(r'^\(\d+\)'),   # (62) 引用编号
    re.compile(r'^\[\d+\]'),   # [62] 引用编号
    re.compile(r'^\d+\.\d+\.\s*\d+'),   # DOI 10.1103
    re.compile(r'^arXiv:\s*\d+', re.IGNORECASE),  # arXiv:2401.12345
    re.compile(r'^[A-Z]\s*[<≤=≥]\s*1'),  # 公式 W ≤1:
    re.compile(r'^\$\s*\\?'),  # $ 公式开头
]

# 段落切分阈值
_MIN_PARAGRAPH_WORDS = 30   # 短于这个合并到前一段
_MAX_PARAGRAPH_WORDS = 500  # 长于这个按句子切
_TARGET_CHUNK_WORDS = 250   # 目标


def _is_section_title(line: str) -> Tuple[bool, str, str]:
    """判断一行是否像 section 标题

    Returns:
        (is_title, prefix, clean_title)
        - is_title: 是否像
        - prefix: "I" / "1" / "2.1" 等编号(空字符串如果没)
        - clean_title: 清理后的标题("Introduction" / "METHODS" 等)
    """
    line = line.strip()
    if not line or len(line) > 120:  # 太长不像标题
        return False, "", ""

    # 2026-09-02 加:启发式排除模式(PRL 短论文常误识)
    for pat in _EXCLUDE_TITLE_PATTERNS:
        if pat.match(line):
            return False, "", ""

    # 罗马数字
    m = _SECTION_PATTERNS[0].match(line)
    if m:
        prefix, title = m.group(1), m.group(2).strip()
        if title.lower() not in _EXCLUDE_TITLES:
            return True, prefix, title
    # 数字编号
    m = _SECTION_PATTERNS[1].match(line)
    if m:
        prefix, title = m.group(1), m.group(2).strip()
        if title.lower() not in _EXCLUDE_TITLES:
            return True, prefix, title
    # 短全大写行(可能没编号)
    if 4 < len(line) < 80 and line == line.upper() and any(c.isalpha() for c in line):
        if line.lower() not in _EXCLUDE_TITLES:
            return True, "", line
    return False, "", ""


# ============================================================
# 段落切分
# ============================================================
def _split_into_paragraphs(text: str) -> List[str]:
    """把一节文字切成段落

    策略:
    - 双换行作为段落边界
    - 单换行但前后是完整句子,合并
    - 短段落(< 30 词)合并到前一段
    - 长段落(> 500 词)按句子切
    """
    # 1. 按双换行初分
    raw_paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = []
    for p in raw_paragraphs:
        p = p.strip()
        if not p:
            continue
        # 段落内多换行 → 空格
        p = re.sub(r'\s*\n\s*', ' ', p)
        paragraphs.append(p)

    # 2. 合并短段落到前一段
    merged = []
    for p in paragraphs:
        words = p.split()
        if merged and len(words) < _MIN_PARAGRAPH_WORDS:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)

    # 3. 切长段落(按句子,简单启发式:. ! ? 后接大写)
    final = []
    for p in merged:
        words = p.split()
        if len(words) <= _MAX_PARAGRAPH_WORDS:
            final.append(p)
            continue
        # 按句子切:. 后面接大写
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', p)
        current = ""
        for sent in sentences:
            test = (current + " " + sent).strip() if current else sent
            if len(test.split()) > _MAX_PARAGRAPH_WORDS and current:
                final.append(current.strip())
                current = sent
            else:
                current = test
        if current:
            final.append(current.strip())

    return [p for p in final if p.strip()]


def _detect_equation_or_table(text: str) -> Tuple[bool, bool]:
    """检测段落里有没有公式/表格的迹象

    启发式:
    - 公式:多 = 号, 或 常见符号 (∑ ∫ ∂ × ÷ →)
    - 表格:多 | 字符, 或 Table/Fig. 编号
    """
    has_equation = (
        text.count("=") >= 2
        or any(s in text for s in ["∑", "∫", "∂", "→", "×", "⊗", "⊕"])
    )
    has_table = (
        text.count("|") >= 3
        or bool(re.search(r'\bTable\s+\d+', text, re.IGNORECASE))
        or bool(re.search(r'\bFig(?:ure)?\.?\s+\d+', text, re.IGNORECASE))
    )
    return has_equation, has_table


# ============================================================
# 章节切分(主函数)
# ============================================================
def _split_by_outline(doc: fitz.Document) -> List[Tuple[str, int, int]]:
    """用 PDF outline 切 section, 返回 [(title, page_start, page_end), ...]

    outline 为空时返空 list(调用方应 fallback 到启发式)
    """
    toc = doc.get_toc()
    if not toc:
        return []
    # 只取 L1(top-level section)
    sections = []
    for entry in toc:
        level, title, page = entry
        if level != 1:
            continue
        # 清理标题
        title = re.sub(r'^\d+\.?\s+', '', title).strip()
        title = re.sub(r'^[IVX]+\.?\s+', '', title).strip()
        sections.append((title, page))
    # 补 page_end:下一节开始 / 文档末尾
    total_pages = len(doc)
    result = []
    for i, (title, page_start) in enumerate(sections):
        page_end = sections[i + 1][1] - 1 if i + 1 < len(sections) else total_pages
        result.append((title, page_start, page_end))
    return result


def _split_by_heuristic(doc: fitz.Document) -> List[Tuple[str, int, int]]:
    """启发式:逐页找 section 标题,记录 page number

    简单:每页第一段扫一次,找匹配的行作为 section 起点
    """
    total_pages = len(doc)
    section_starts = []  # [(page_idx, title)]

    for page_idx in range(total_pages):
        page = doc[page_idx]
        text = page.get_text("text")
        lines = text.split("\n")
        for line in lines:
            is_title, prefix, title = _is_section_title(line)
            if is_title:
                section_starts.append((page_idx + 1, title))  # page_idx 是 0-based, 返 1-based
                break  # 一页只算一次

    # 去掉相邻重复(同一 section 跨页)
    deduped = []
    for page, title in section_starts:
        if deduped and deduped[-1][1].lower() == title.lower():
            continue
        deduped.append((page, title))

    # 2026-09-02 加:启发式质量检查 — 如果找到的 section 没有 intro/conclusion 等
    # 真实学术标题,大概率是误识(PRL 短论文启发式常误识期刊标号/figure/ref)
    # 这种 fallback 到 "Body" 全文
    if deduped:
        real_titles = {"introduction", "methods", "method", "results", "result",
                       "discussion", "conclusion", "summary", "background",
                       "experiments", "experiment", "evaluation", "related work",
                       "preliminaries", "approach", "framework", "model",
                       "analysis", "applications"}
        if not any(t[1].lower() in real_titles for t in deduped):
            logger.info(
                "[pdf_chunker] 启发式未识别任何真实 section 标题, fallback 到 Body 全文"
            )
            return [("Body", 1, total_pages)]

    # 如果启发式找不到任何 section,把全文作为 1 个 "Body" section
    if not deduped:
        return [("Body", 1, total_pages)]

    # 补 page_end
    result = []
    for i, (page_start, title) in enumerate(deduped):
        page_end = deduped[i + 1][0] - 1 if i + 1 < len(deduped) else total_pages
        result.append((title, page_start, page_end))
    return result


def _extract_section_text(doc: fitz.Document, page_start: int, page_end: int) -> str:
    """提取 [page_start, page_end] (1-based, 含) 范围内文字"""
    parts = []
    for p in range(page_start, page_end + 1):
        if 1 <= p <= len(doc):
            parts.append(doc[p - 1].get_text("text"))
    return "\n\n".join(parts)


# ============================================================
# 公共入口
# ============================================================
def chunk_by_section(
    pdf_path: str,
    paper_id: str = "",
    title: str = "",
    year: Optional[int] = None,
    journal: str = "",
    source: str = "arxiv",
    min_paragraph_words: int = _MIN_PARAGRAPH_WORDS,
) -> List[Chunk]:
    """主入口:section-aware 切分

    Args:
        pdf_path: PDF 文件路径
        paper_id, title, year, journal, source: paper 级 metadata
        min_paragraph_words: 短段落合并阈值(默认 30 词)

    Returns:
        List[Chunk]:每个 chunk 一个段落级(200-500 词),带 section metadata
    """
    pdf_path = str(pdf_path)
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        # 1. outline 优先,失败 fallback 启发式
        section_bounds = _split_by_outline(doc)
        method = "outline"
        if not section_bounds:
            section_bounds = _split_by_heuristic(doc)
            method = "heuristic"
        logger.info(
            "[pdf_chunker] %s: %d sections (%s method)",
            paper_id or Path(pdf_path).name, len(section_bounds), method,
        )

        # 2. 逐 section 切段落
        chunks: List[Chunk] = []
        for section_idx, (sec_title, page_start, page_end) in enumerate(section_bounds):
            sec_text = _extract_section_text(doc, page_start, page_end)
            if not sec_text.strip():
                continue
            paragraphs = _split_into_paragraphs(sec_text)
            for chunk_idx, para in enumerate(paragraphs):
                # 太短的(只合并了 1 个)继续合
                if (chunk_idx + 1 < len(paragraphs)
                        and len(para.split()) < min_paragraph_words
                        and len(chunks) > 0
                        and chunks[-1].section == sec_title
                        and len(chunks[-1].text.split()) < _TARGET_CHUNK_WORDS):
                    # 合到前一段
                    chunks[-1].text = chunks[-1].text + " " + para
                    chunks[-1].page_end = page_end
                    continue
                has_eq, has_tbl = _detect_equation_or_table(para)
                chunks.append(Chunk(
                    text=para,
                    section=sec_title,
                    section_idx=section_idx,
                    chunk_idx=chunk_idx,
                    page_start=page_start,
                    page_end=page_end,
                    has_equation=has_eq,
                    has_table=has_tbl,
                    paper_id=paper_id,
                    title=title,
                    year=year,
                    journal=journal,
                    source=source,
                ))

        logger.info(
            "[pdf_chunker] %s: 切出 %d chunks",
            paper_id or Path(pdf_path).name, len(chunks),
        )
        return chunks
    finally:
        doc.close()


# ============================================================
# 健康检查
# ============================================================
def health_check() -> Dict[str, Any]:
    return {
        "pymupdf_version": fitz.__version__,
        "patterns_count": len(_SECTION_PATTERNS),
        "exclude_count": len(_EXCLUDE_TITLES),
    }
