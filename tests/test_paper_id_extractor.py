"""
test_paper_id_extractor — `_extract_paper_ids` + 3 种 regex 测试 (2026-09-21 加)

覆盖 3 种 paper_id 引用格式:
  1. 简洁:  [2609.10003] / [0607734]
  2. arXiv:  arXiv:2609.10003
  3. 链接:   papers/2609.10003.md

为什么重要: survey_generate 章节内容渲染 + claim-source pairing 全靠这个函数。
回退会让综述出现 "missing paper_id" 假引用, 审计失败。
"""
from agent.tools.surveys._pool import (
    _extract_paper_ids,
    _PAPER_ID_SIMPLE_RE,
    _PAPER_ID_ARXIV_RE,
    _PAPER_ID_PAPERS_MD_RE,
)


class TestPaperIdSimpleRegex:
    """测试 _PAPER_ID_SIMPLE_RE: 简洁 [xxxx.xxxxx] 格式"""

    def test_new_format_4digit_5digit(self):
        """新版 4 位 .5 位 arXiv ID (2007 后)"""
        m = _PAPER_ID_SIMPLE_RE.findall("[2609.10003]")
        assert m == ["2609.10003"]

    def test_old_format_7_digits(self):
        """旧版 7 位纯数字 arXiv ID (2007 前)"""
        m = _PAPER_ID_SIMPLE_RE.findall("[0607734]")
        assert m == ["0607734"]

    def test_multiple_in_text(self):
        """同一文本中多个 paper_id"""
        m = _PAPER_ID_SIMPLE_RE.findall(
            "Section 1: [2609.10003] and [2501.12345] reference [0607734]"
        )
        assert m == ["2609.10003", "2501.12345", "0607734"]

    def test_no_match(self):
        """无 paper_id → 空列表"""
        m = _PAPER_ID_SIMPLE_RE.findall("No paper IDs here")
        assert m == []


class TestPaperIdArxivRegex:
    """测试 _PAPER_ID_ARXIV_RE: arXiv:xxxx.xxxxx 格式"""

    def test_new_format(self):
        m = _PAPER_ID_ARXIV_RE.findall("arXiv:2609.10003")
        assert m == ["2609.10003"]

    def test_old_format(self):
        m = _PAPER_ID_ARXIV_RE.findall("arXiv 0607734")  # 也支持空格分隔
        assert m == ["0607734"]


class TestPaperIdPapersMdRegex:
    """测试 _PAPER_ID_PAPERS_MD_RE: papers/xxxx.md 链接格式"""

    def test_link_format(self):
        m = _PAPER_ID_PAPERS_MD_RE.findall("papers/2609.10003.md")
        assert m == ["2609.10003"]

    def test_old_format(self):
        m = _PAPER_ID_PAPERS_MD_RE.findall("papers/0607734.md")
        assert m == ["0607734"]


class TestExtractPaperIds:
    """测试 _extract_paper_ids: 综合 3 种格式 + 去重保序"""

    def test_empty_text(self):
        assert _extract_paper_ids("") == []

    def test_none_text(self):
        assert _extract_paper_ids(None) == []  # type: ignore[arg-type]

    def test_simple_format_only(self):
        result = _extract_paper_ids("Reference [2609.10003] discusses...")
        assert result == ["2609.10003"]

    def test_arxiv_format_only(self):
        result = _extract_paper_ids("See arXiv:2501.12345 for details")
        assert result == ["2501.12345"]

    def test_link_format_only(self):
        result = _extract_paper_ids("Read papers/2609.10003.md for the full analysis")
        assert result == ["2609.10003"]

    def test_mixed_formats_dedup_order(self):
        """3 种格式混用 + 去重保序"""
        text = (
            "[2609.10003] is cited first.\n"
            "Also see arXiv:2609.10003 (same paper, arxiv format).\n"
            "Full text at papers/2609.10003.md.\n"
            "Different paper: [2501.12345]"
        )
        result = _extract_paper_ids(text)
        # 期望: 2609.10003 出现 3 次, 去重只保留第一次; 2501.12345 保留
        assert result == ["2609.10003", "2501.12345"]

    def test_old_format_compatibility(self):
        """旧 7 位格式 (2007 前) 仍能识别"""
        result = _extract_paper_ids("Historical paper [0607734] from 2006")
        assert result == ["0607734"]

    def test_url_embedded_in_link(self):
        """完整 URL 路径里的 paper_id"""
        text = "Full URL: https://arxiv.org/abs/2609.10003 [2609.10003]"
        # URL 不被 papers/ regex 匹配 (路径前缀不对), 但 [2609.10003] 被 simple 匹配
        result = _extract_paper_ids(text)
        assert "2609.10003" in result
