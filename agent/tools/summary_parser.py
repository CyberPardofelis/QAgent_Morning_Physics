"""
agent.tools.summary_parser — 解析 LLM 输出的"每日总结" markdown,提取结构化字段

用途:
- generate_aps_daily_summary / generate_daily_summary_arxiv 拿 LLM 文本
- 主流程同步调 parse_summary_markdown() → list[PaperSummary]
- 每条 PaperSummary 直接 save_summary_item() 落库(summary_items.db)

设计原则(2026-09-03):
- **零后台线程**:主流程同步解析+同步落库,不走 _async,防 P0.6 蓝屏复刻
- **鲁棒**:LLM 输出格式不稳定,parser 容错多种变体
  - `#### 1. 标题` / `#### 1、标题` / `### 1. 标题` 都接受
  - 字段 marker: `**核心贡献**:` / `**核心贡献**:` / `**核心贡献**: ` 都接受
  - 缺字段:该字段为空字符串(不抛)
  - 找不到 paper_id:跳过该 block(warn log)
- **职责单一**:只解析,不落库、不调 LLM、不读文件
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================
@dataclass
class PaperSummary:
    """从 LLM 输出里解析出的一篇论文结构化总结"""
    paper_id: str                    # 必填,失败解析时为空字符串
    title: str                       # 必填,从 header 拿
    authors: str = ""                # 可选,从 - **作者** 拿
    core_contrib: str = ""           # 核心贡献
    core_method: str = ""            # 方法
    outlook: str = ""                # 意义与展望
    keywords: str = ""               # 关键词(逗号分隔字符串)

    def is_valid(self) -> bool:
        """是否可入库(至少要有 paper_id 和 title)"""
        return bool(self.paper_id) and bool(self.title)


# ============================================================
# 解析器
# ============================================================
# 匹配 `#### N. 标题` 或 `### N. 标题` 或 `#### N、标题`(中英文逗号)
# 标题里可能含 markdown 链接 [text](url) — 保留原样
_PAPER_HEADER_RE = re.compile(
    r'^(#{3,4})\s+\d+[.,、:：]\s*(.+?)\s*$',
    re.MULTILINE,
)

# 匹配 marker 行: `- **核心贡献**: xxx` / `**核心贡献**: xxx` / `**核心贡献** xxx`
# 兼容:冒号后有无空格、bullet 有无、全半角冒号
_FIELD_MARKER_RE = re.compile(
    r'^\s*(?:-\s*)?\*\*\s*([^*]+?)\s*\*\*\s*[:：]?\s*(.*)$'
)

# 已知 marker 名字(归一化)→ 字段名
_KNOWN_FIELDS = {
    "作者": "authors",
    "doi": "doi",
    "arxiv id": "arxiv_id",
    "arxivid": "arxiv_id",
    "发表日期": "published_date",
    "发布日期": "published_date",
    "日期": "published_date",
    "核心贡献": "core_contrib",
    "方法": "core_method",
    "意义与展望": "outlook",
    "意义和展望": "outlook",
    "关键词": "keywords",
    "keyword": "keywords",
    "keywords": "keywords",
}


def _normalize_marker(marker: str) -> Optional[str]:
    """把 marker 文本归一化到字段名;未知返 None"""
    m = marker.strip().lower()
    # 去括号注释: "核心贡献(2-3句话)" → "核心贡献"
    m = re.sub(r'[（(].*?[)）]', '', m).strip()
    return _KNOWN_FIELDS.get(m)


def _extract_field_value(text: str, field_name: str) -> str:
    """从一段 markdown 文本里找 `- **field_name**: value`,返 value"""
    # 按行扫(单 marker 不跨行,LLM prompt 限制了)
    lines = text.split('\n')
    out_lines: List[str] = []
    in_field = False
    for line in lines:
        m = _FIELD_MARKER_RE.match(line)
        if m:
            marker, value = m.group(1), m.group(2)
            mapped = _normalize_marker(marker)
            if mapped == field_name:
                in_field = True
                if value.strip():
                    out_lines.append(value.strip())
                continue
            else:
                # 别的 marker 出现 → 当前 field 结束
                if in_field:
                    in_field = False
                    # 不 break,继续扫(防 LLM 字段顺序乱)
        if in_field and line.strip() and not _FIELD_MARKER_RE.match(line):
            # 续行(LLM 可能换行,markdown 列表里少见但 prompt 长输出时可能)
            out_lines.append(line.strip())
    return ' '.join(out_lines).strip()


def _extract_paper_id_from_header(header_text: str) -> str:
    """
    从 `#### 1. 标题` 的 header_text 里尝试提 paper_id。
    优先级:
      1. markdown 链接里的 URL 末尾(aps 用 /abstract/10.1103/xxx,arxiv 用 /abs/2501.xxxxx)
      2. 内联文本里 `(10.1103/xxx)` 或 `(2501.xxxxx)` 形式
    """
    # 1) [title](https://.../abs/2501.xxxxx) 或 (https://.../abstract/10.1103/xxx)
    link_match = re.search(r'\[.+?\]\((https?://[^\s)]+)\)', header_text)
    if link_match:
        url = link_match.group(1)
        # arxiv: 末尾是 NNNN.NNNNN
        arxiv_m = re.search(r'/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)', url)
        if arxiv_m:
            return arxiv_m.group(1)
        # APS / DOI: 末尾是 10.1103/xxx
        doi_m = re.search(r'(10\.\d{4,9}/[^\s/?#]+)', url)
        if doi_m:
            return doi_m.group(1)
    # 2) 内联 (10.1103/xxx) 或 (2501.xxxxx)
    m = re.search(r'\((10\.\d{4,9}/[^\s)]+)\)', header_text)
    if m:
        return m.group(1)
    m = re.search(r'\((\d{4}\.\d{4,5}(?:v\d+)?)\)', header_text)
    if m:
        return m.group(1)
    return ""


def _clean_title(raw: str) -> str:
    """清理 header 标题:去 markdown 链接包裹、去前后空白、去尾部标点"""
    t = raw.strip()
    # 去 [text](url) → text
    t = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', t)
    # 去尾部 (2501.xxxxx) / (10.1103/xxx)
    t = re.sub(r'\s*\((?:\d{4}\.\d{4,5}(?:v\d+)?|10\.\d{4,9}/[^\s)]+)\)\s*$', '', t)
    # 去尾部中英文句号
    t = t.rstrip('.,;:。、')
    return t.strip()


def parse_summary_markdown(md_text: str, fetched_papers: List[dict] = None) -> List[PaperSummary]:
    """
    主入口:解析 LLM 输出的整篇 markdown 总结。

    支持的 block 起始:
      - `#### N. 标题...`  (aps 和 arxiv 主流)
      - `### N. 标题...`   (arxiv 在研究方向组下)
    不解析:
      - `## 整体概览` / `### 研究方向` 这种非 paper header
        (它们要么没有数字开头,要么不以 `N.` 数字开头)

    2026-09-04 加 `fetched_papers` 参数:title-based 回退匹配
    - 之前:LLM 输出里漏掉 DOI/URL 时,paper_id 空 → 整个 block 跳过(用户报"20 条 APS 缺字段")
    - 现在:传已抓取的 paper 列表进来(每条有 paper_id + title),parser 在 paper_id 空时
      拿 header 里的 title 跟 fetched_papers 比对,找到最匹配的填上 paper_id
    - 匹配规则:normalize 后子串匹配(去掉 markdown/标点/大小写)
    - 极小开销:只在 paper_id 缺失时启用,正常 LLM 输出不受影响
    """
    if not md_text or not md_text.strip():
        return []

    # 找所有 paper header
    blocks: List[tuple[int, str, str]] = []  # (start_pos, level_marker, header_text)
    for m in _PAPER_HEADER_RE.finditer(md_text):
        blocks.append((m.start(), m.group(1), m.group(2)))

    if not blocks:
        logger.warning("[summary_parser] 没找到任何 paper header(`#### N.` 或 `### N.`)")
        return []

    results: List[PaperSummary] = []
    for i, (start, marker, header_text) in enumerate(blocks):
        # block 内容范围:[start, 下一个 header start 或文末)
        end = blocks[i + 1][0] if i + 1 < len(blocks) else len(md_text)
        block_text = md_text[start:end]

        # 解析 paper_id(优先从 header 链接拿;拿不到再从 - **DOI/ID** 拿)
        paper_id = _extract_paper_id_from_header(header_text)
        if not paper_id:
            # 退到 - **DOI** 或 - **arXiv ID** 字段
            doi_field = _extract_field_value(block_text, "doi")
            if doi_field:
                # 可能是 [DOI 链接](https://doi.org/10.1103/xxx) 或纯 10.1103/xxx
                doi_match = re.search(r'10\.\d{4,9}/[^\s)\]]+', doi_field)
                if doi_match:
                    paper_id = doi_match.group(0)
            if not paper_id:
                arxiv_field = _extract_field_value(block_text, "arxiv_id")
                if arxiv_field:
                    arxiv_match = re.search(r'\d{4}\.\d{4,5}(?:v\d+)?', arxiv_field)
                    if arxiv_match:
                        paper_id = arxiv_match.group(0)

        # 2026-09-04 加:title-based 回退匹配
        # LLM 偶尔漏掉 DOI/URL(实测),paper_id 空 → block 跳过 → UPDATE 落空
        # 用已抓取的 paper 列表,按 title 匹配回 paper_id
        if not paper_id and fetched_papers:
            title_for_match = _clean_title(header_text)
            paper_id = _match_paper_by_title(title_for_match, fetched_papers)
            if paper_id:
                logger.debug(
                    "[summary_parser] title 匹配: header='%s' → paper_id='%s'",
                    header_text[:60], paper_id,
                )

        # 解析 title
        title = _clean_title(header_text)

        # 解析其它字段
        authors = _extract_field_value(block_text, "authors")
        core_contrib = _extract_field_value(block_text, "core_contrib")
        core_method = _extract_field_value(block_text, "core_method")
        outlook = _extract_field_value(block_text, "outlook")
        keywords_raw = _extract_field_value(block_text, "keywords")
        # 关键词归一化:去方括号、引号、连续空白
        keywords = re.sub(r'[\[\]"""\'\'「」]', '', keywords_raw)
        keywords = re.sub(r'\s+', ' ', keywords).strip().strip(',').strip()

        ps = PaperSummary(
            paper_id=paper_id,
            title=title,
            authors=authors,
            core_contrib=core_contrib,
            core_method=core_method,
            outlook=outlook,
            keywords=keywords,
        )
        if ps.is_valid():
            results.append(ps)
        else:
            logger.warning(
                "[summary_parser] 跳过 block(缺 paper_id 或 title): "
                "header='%s' paper_id='%s'",
                header_text[:80], paper_id,
            )
    return results


def _normalize_title_for_match(title: str) -> str:
    """title 归一化:去 markdown/标点/空格/大小写,方便模糊匹配

    - '**Paper A**' → 'papera'
    - '[Paper A](https://...)' → 'papera'  (link 部分去掉了)
    - 'Paper A: B' → 'papera b'
    """
    t = title.lower()
    # 去 markdown 强调
    t = re.sub(r'\*+', '', t)
    # 去 markdown 链接(只保留文本部分)
    t = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', t)
    # 去标点
    t = re.sub(r'[\[\]()（）【】""''「」、,。:;!?]', ' ', t)
    # 多空白 → 单空格
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _match_paper_by_title(header_title: str, fetched_papers: List[dict]) -> str:
    """根据 header 里的 title,在已抓取的 papers 里找最匹配的,返回 paper_id

    匹配规则(从严到松):
    1. 完全相等(normalize 后)
    2. 子串匹配(header 是 fetched title 的子串,或反过来)
    3. 共同 token 累计长度 >= 20(防误匹配)

    用途:LLM 偶尔漏掉 DOI/URL 字段,header 里没 paper_id,fallback 用 title 找回
    """
    if not fetched_papers or not header_title:
        return ""

    h_norm = _normalize_title_for_match(header_title)
    if not h_norm or len(h_norm) < 5:
        return ""

    # 预 normalize 候选 titles
    candidates = []
    for p in fetched_papers:
        pid = p.get("id", "")
        ptitle = p.get("title", "")
        if not pid or not ptitle:
            continue
        p_norm = _normalize_title_for_match(ptitle)
        if not p_norm:
            continue
        candidates.append((pid, ptitle, p_norm))

    # 1) 完全相等
    for pid, ptitle, p_norm in candidates:
        if h_norm == p_norm:
            return pid

    # 2) 子串(双向)
    for pid, ptitle, p_norm in candidates:
        if h_norm in p_norm or p_norm in h_norm:
            shorter = min(len(h_norm), len(p_norm))
            if shorter >= 20:
                return pid

    # 3) 共同 token 累计长度(启发式,适应 LLM 简化标题的情况)
    best_pid = ""
    best_overlap = 0
    h_tokens = set(h_norm.split())
    for pid, ptitle, p_norm in candidates:
        p_tokens = set(p_norm.split())
        common = h_tokens & p_tokens
        overlap = sum(len(t) for t in common)
        if overlap > best_overlap and overlap >= 20:
            best_overlap = overlap
            best_pid = pid
    return best_pid
