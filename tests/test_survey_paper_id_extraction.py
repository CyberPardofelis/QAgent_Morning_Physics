"""
tests/test_survey_paper_id_extraction.py — paper_id 抽取的单元测试

9-27 实测发现的 P0 修复。

问题:
    _pool/_utils.py 原来只有 3 条 regex, **全都只认 arXiv 格式**:
        [2401.12345] / [1234567] / arXiv:2401.12345 / papers/2401.12345.md
    一条都匹配不到 APS/IOP 的 DOI 形式 `10.1103/2401.12345`。
    而 APS 是 QAgent 的主要数据源之一(source: aps_daily / openalex APS 召回)。

后果(实测数据):
    - 一段 10892 字符的章节, 正文里有 [10.1103/revmodphys.90.035006]、
      [10.1103/zsb1-gx7f] 等**真实且在候选池内**的引用, 审计却只认出 1 个
      2111.08018, 并把它判为池外
    - out_of_pool > 0 触发重写循环(最多 2 次), 模型被告知"你引用了池外 id"
      后大幅收敛, 章节从 10892 字符被砍到 3075 字符(72% 内容丢失)
    - in_pool 恒为 0 → 覆盖度审计失真; APS 幻觉引用抓不到 → 审计形同虚设
"""
from __future__ import annotations


from agent.tools.surveys._pool._utils import _audit_section_paper_ids, _extract_paper_ids


# ============================================================
# arXiv 格式(回归: 原来能识别的不能坏)
# ============================================================

def test_arxiv_short_form():
    assert _extract_paper_ids("对比 [2111.08018] 和 [2401.12345]") == [
        "2111.08018", "2401.12345",
    ]


def test_arxiv_7digit():
    assert _extract_paper_ids("旧格式 [0607734]") == ["0607734"]


def test_arxiv_annotated():
    assert _extract_paper_ids("见 [Bombin 2006, arXiv:0607734](x)") == ["0607734"]


def test_arxiv_link():
    assert _extract_paper_ids("见 [Bombin 2006](papers/0607734.md)") == ["0607734"]


# ============================================================
# DOI 格式 (9-27 新增)
# ============================================================

def test_aps_doi_bracket_form():
    """实测正文里就是这个形式 —— 原来完全识别不到。"""
    assert _extract_paper_ids("综述指出 [10.1103/revmodphys.90.035006] 的结论") == [
        "10.1103/revmodphys.90.035006",
    ]


def test_aps_new_style_doi():
    assert _extract_paper_ids("见 [10.1103/zsb1-gx7f]") == ["10.1103/zsb1-gx7f"]


def test_multiple_aps_dois_deduped_in_order():
    got = _extract_paper_ids("见 [10.1103/zsb1-gx7f] 与 [10.1103/2cvx-6hqz]")
    assert got == ["10.1103/zsb1-gx7f", "10.1103/2cvx-6hqz"]


def test_doi_link_strips_md_suffix():
    """链接形式不能把 .md 吞进 id —— 否则跟论文实际 paper_id 对不上。"""
    got = _extract_paper_ids("链接 [Author 2020](papers/10.1103/24zv-dwcj.md)")
    assert "10.1103/24zv-dwcj" in got
    assert "10.1103/24zv-dwcj.md" not in got


def test_bare_doi_with_slash_in_suffix():
    """部分 DOI 后缀含斜杠(10.1088/1367-2630/aad1ea), 不能被截断。"""
    assert _extract_paper_ids("裸 DOI 10.1088/1367-2630/aad1ea 出现") == [
        "10.1088/1367-2630/aad1ea",
    ]


def test_bare_doi_trailing_punctuation_not_swallowed():
    got = _extract_paper_ids("如 10.1103/RevModPhys.90.035006。")
    assert got == ["10.1103/RevModPhys.90.035006"]


def test_mixed_formats_in_one_text():
    got = _extract_paper_ids(
        "A[2111.08018] B[10.1103/zsb1-gx7f] C[10.1088/1367-2630/aad1ea] D[2401.12345]"
    )
    assert got == [
        "2111.08018", "10.1103/zsb1-gx7f",
        "10.1088/1367-2630/aad1ea", "2401.12345",
    ]


def test_empty_text():
    assert _extract_paper_ids("") == []
    assert _extract_paper_ids(None) == []


def test_no_false_positive_on_urls():
    """普通 URL 不应被当成 paper_id。"""
    got = _extract_paper_ids("详见 https://example.com/paper/12345 页面")
    assert got == []


# ============================================================
# 审计: APS 引用现在能被正确分类
# ============================================================

def test_aps_in_pool_audit():
    """
    核心回归: 池内的 APS 引用必须算 in_pool, 不能再被判幻觉。
    """
    pool = {"10.1103/revmodphys.90.035006", "10.1103/zsb1-gx7f", "2111.08018"}
    content = (
        "A 指出 [10.1103/revmodphys.90.035006]。\n\n"
        "B 指出 [10.1103/zsb1-gx7f]。\n\n"
        "C 引用 [2111.08018]。\n\n"
        "D 编造了 [10.9999/fake.0001]。"
    )
    b = _audit_section_paper_ids(content, pool)
    assert sorted(b["in_pool"]) == [
        "10.1103/revmodphys.90.035006", "10.1103/zsb1-gx7f", "2111.08018",
    ]
    assert b["out_of_pool"] == ["10.9999/fake.0001"]


def test_aps_only_section_is_not_flagged():
    """
    实测场景: 章节里全是池内 APS 引用 → out_of_pool 必须为空,
    否则会误触发重写循环把内容砍掉 72%。
    """
    pool = {"10.1103/revmodphys.90.035006", "10.1103/zsb1-gx7f", "10.1103/2cvx-6hqz"}
    content = (
        "A [10.1103/revmodphys.90.035006] 与 B [10.1103/zsb1-gx7f] 提出不同路径, "
        "C [10.1103/2cvx-6hqz] 进一步验证。"
    )
    b = _audit_section_paper_ids(content, pool)
    assert b["out_of_pool"] == [], "池内 APS 引用不该被判为幻觉"
    assert len(b["in_pool"]) == 3


def test_audit_still_catches_real_hallucination():
    """修复不能把幻觉放过去 —— 池外的 APS id 仍要被抓出来。"""
    pool = {"10.1103/zsb1-gx7f"}
    b = _audit_section_paper_ids("引用 [10.1103/not-in-pool-xyz]。", pool)
    assert b["out_of_pool"] == ["10.1103/not-in-pool-xyz"]
    assert b["in_pool"] == []
