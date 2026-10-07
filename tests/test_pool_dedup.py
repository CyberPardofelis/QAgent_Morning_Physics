"""候选池跨来源去重 (10-07)。

背景: 实跑(经典阴影 v3)发现池子标称 123 篇, **唯一 paper_id 只有 87 个** ——
28% 是纯重复, 同一个 `2107.04817` 出现了 8 次, 而且每一次都重新调了一次
summarize LLM。两条独立成因:

  1. `action_search_more` 的 `seen_in_search` 每次调用都重置, 完全不看
     `state["pool"]` 里已经有什么 —— gap fill 连打 5 次检索, 每次把同样那批
     论文再塞一遍。
  2. 跨来源认不出同一篇: arXiv API 召回预印本给 arXiv id (`2110.03629`),
     OpenAlex(APS/IOP) 召回期刊版给 DOI (`10.1103/physreva.107.042403`),
     `dedup_key` 折不出来 —— 参考文献里于是出现重复条目。

user 2026-10-07 拍板: 重复时**以 OpenAlex 的为准**, arxiv 的可能是发表前挂的。
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.tools.surveys._paper_id import dedup_key, looks_like_doi  # noqa: E402
from agent.tools.surveys._pool._utils import (  # noqa: E402
    _entry_rank,
    _identity_key,
    dedup_across_sources,
    dedup_existing,
)


def _ax(pid: str, doi: str = "", title: str = "T", abstract: str = "A"):
    return {"paper_id": pid, "doi": doi, "title": title, "abstract": abstract,
            "authors": "X", "journal": "arxiv", "year": "2021"}


def _oa(pid: str, title: str = "T", abstract: str = "A"):
    return {"paper_id": pid, "doi": pid if looks_like_doi(pid) else "",
            "title": title, "abstract": abstract, "authors": "X",
            "journal": "Phys. Rev. X", "year": "2023"}


# ============================================================
# 身份键
# ============================================================

class TestIdentityKey:
    def test_doi_beats_arxiv_id(self):
        """带 doi 字段的 arXiv 条目 -> 身份键走 DOI。"""
        e = _ax("2110.03629", doi="10.1103/physreva.107.042403")
        assert _identity_key(e) == "doi:10.1103/physreva.107.042403"

    def test_paper_id_doi_without_doi_field(self):
        """OpenAlex 条目没有 doi 字段但 paper_id 就是 DOI -> 同样走 DOI。"""
        e = {"paper_id": "10.1103/physrevlett.127.200501"}
        assert _identity_key(e) == "doi:10.1103/physrevlett.127.200501"

    def test_doi_case_insensitive(self):
        """DOI 大小写不敏感, 身份键必须折。"""
        a = _identity_key({"paper_id": "10.1103/PhysRevLett.127.200501"})
        b = _identity_key({"paper_id": "10.1103/physrevlett.127.200501"})
        assert a == b

    def test_preprint_without_doi_stays_arxiv(self):
        """没 DOI 的纯预印本 -> 退回 arXiv 身份键(不猜)。"""
        assert _identity_key({"paper_id": "2110.03629"}).startswith("ax:")

    def test_arxiv_version_suffix_folded(self):
        a = _identity_key({"paper_id": "2209.02093v1"})
        b = _identity_key({"paper_id": "2209.02093"})
        assert a == b, "版本号必须折掉, 否则同一篇还是两个身份"


# ============================================================
# 优先级: OpenAlex 赢
# ============================================================

class TestPriority:
    def test_openalex_doi_form_wins_over_arxiv(self):
        """user 拍板: 以 OpenAlex 为准。paper_id 是 DOI = 来自 OpenAlex。"""
        assert _entry_rank(_oa("10.1103/physreva.107.042403")) > \
               _entry_rank(_ax("2110.03629", doi="10.1103/physreva.107.042403"))

    def test_richer_metadata_wins_within_same_form(self):
        thin = {"paper_id": "10.1103/x", "title": "T"}
        fat = {"paper_id": "10.1103/x", "title": "T", "abstract": "A",
               "authors": "X", "journal": "PRX", "year": "2023"}
        assert _entry_rank(fat) > _entry_rank(thin)

    def test_merged_keeps_openalex_paper_id(self):
        out, st = dedup_across_sources([
            _ax("2110.03629", doi="10.1103/physreva.107.042403", abstract="预印本摘要"),
            _oa("10.1103/physreva.107.042403"),
        ])
        assert len(out) == 1
        assert out[0]["paper_id"] == "10.1103/physreva.107.042403", \
            "应保留 OpenAlex 的 DOI 版"
        assert st["n_arxiv_to_doi"] == 1


# ============================================================
# 折叠本身
# ============================================================

class TestDedupAcrossSources:
    def test_preprint_and_journal_version_collapse(self):
        """实测数据里的真实一对: 同一篇的 arXiv 版和 PRA 正式版。"""
        out, st = dedup_across_sources([
            _ax("2110.03629", doi="10.1103/physreva.107.042403"),
            _oa("10.1103/physreva.107.042403"),
        ])
        assert st["n_in"] == 2 and st["n_out"] == 1
        assert out[0]["paper_id"] == "10.1103/physreva.107.042403"
        assert "2110.03629" in (out[0].get("aliases") or []), \
            "落败的标识要记进 aliases, 不能凭空消失"

    def test_identical_paper_id_many_times_collapses_to_one(self):
        """实测: 2107.04817 出现 8 次。8 次 summarize LLM 是纯浪费。"""
        out, st = dedup_across_sources([_ax("2107.04817") for _ in range(8)])
        assert len(out) == 1, f"8 条同 id 应折成 1 条, 实际 {len(out)}"
        assert st["n_dedup_same"] == 7

    def test_distinct_papers_all_survive(self):
        """最要紧的护栏: 不能误杀。"""
        papers = [_ax("2107.04817"), _ax("2110.02965"), _oa("10.1103/x-abc123"),
                  _oa("10.1088/1367-2630/acfcd3"), {"paper_id": "W12345"}]
        out, st = dedup_across_sources(papers)
        assert len(out) == len(papers), "不同论文不能被折在一起"
        assert st["n_in"] == st["n_out"]

    def test_no_doi_preprint_not_mismerged(self):
        """两个都没有 DOI 的预印本, 只能按 arXiv id 分, 不许乱折。"""
        out, _ = dedup_across_sources([_ax("2107.04817"), _ax("2110.02965")])
        assert len(out) == 2

    def test_order_preserved(self):
        entries = [_oa("10.1103/aaa"), _ax("2107.04817"), _oa("10.1103/bbb")]
        out, _ = dedup_across_sources(entries)
        assert [e["paper_id"] for e in out] == \
               ["10.1103/aaa", "2107.04817", "10.1103/bbb"]

    def test_loser_metadata_fills_gaps(self):
        """落败方的非空字段要补进胜出者, 别把摘要/作者弄丢。"""
        out, _ = dedup_across_sources([
            {"paper_id": "10.1103/physreva.107.042403", "title": "T",
             "abstract": "", "journal": "PRA"},
            {"paper_id": "2110.03629", "doi": "10.1103/physreva.107.042403",
             "title": "T", "abstract": "预印本那份更全的摘要", "authors": "H K P"},
        ])
        assert out[0]["abstract"] == "预印本那份更全的摘要"
        assert out[0]["authors"] == "H K P"

    def test_entry_without_id_kept_not_crashed(self):
        out, _ = dedup_across_sources([{"paper_id": "", "title": "x"}, _ax("2107.04817")])
        assert len(out) == 2, "没有 id 的条目要原样放行, 不能崩也不能丢"

    def test_dedup_existing_returns_count(self):
        out, n = dedup_existing([_ax("2107.04817")] * 3 + [_ax("2110.02965")])
        assert n == 2 and len(out) == 2


# ============================================================
# 用真实池子数据回归
# ============================================================

REAL_POOL = ROOT / "data" / "surveys" / "working" / "89309897a97c" / "_pool.json"


@pytest.mark.skipif(not REAL_POOL.exists(), reason="需要真实池子数据")
class TestAgainstRealPool:
    def test_pool_is_28pct_duplicate(self):
        """守住这个测试的前提: 真实池子确实存在大量重复。

        如果哪天池子干净了, 这个测试会 fail —— 那不是 bug, 是前提变了,
        到时候把这条删掉就行(它是个"事实记录", 不是护栏)。
        """
        data = json.loads(REAL_POOL.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else (data.get("papers") or [])
        assert items, "池子不该是空的"
        uniq = len({p.get("paper_id") for p in items})
        assert uniq < len(items), "预期现在有重复(前提), 若已修复请更新/删除本测试"

    def test_dedup_key_folds_nothing_extra(self):
        """dedup_key 折不掉的那些才是真重复 —— 证明问题不在归一层。"""
        data = json.loads(REAL_POOL.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else (data.get("papers") or [])
        by_key = {}
        for p in items:
            by_key.setdefault(dedup_key(p.get("paper_id") or ""), []).append(p)
        dup = {k: v for k, v in by_key.items() if len(v) > 1}
        assert dup, "预期存在同一 dedup_key 重复的条目"
        # 最大的那组应该就是实测里的 2107.04817 x8
        assert max(len(v) for v in dup.values()) >= 5


# ============================================================
# 接线: arXiv / OpenAlex 必须带 doi 字段
# ============================================================

class TestRecallFields:
    def test_arxiv_module_reads_doi(self):
        src = (ROOT / "agent/tools/surveys/_pool/_arxiv.py").read_text(encoding="utf-8")
        # 必须两处都有: _arxiv_search 和 _arxiv_search_batch 是两个独立入口,
        # gap fill 走的是后者。只断言 >=1 的话, 漏掉一处照样绿(第一版就这样
        # 写成了 >=1, 反向验证立刻报"锚点出现 2 次, 变异只改第一处")。
        n = src.count('"doi": (getattr(r, "doi", "") or "").strip()')
        assert n == 2, (
            f"arXiv 召回的两个入口都该读 <arxiv:doi>, 实际 {n} 处 —— "
            f"少一处跨来源折叠就有一条路径失效")

    def test_openalex_module_sets_doi(self):
        src = (ROOT / "agent/tools/surveys/_pool/_openalex.py").read_text(encoding="utf-8")
        assert '"doi": normalize_doi(work.get("doi") or "")' in src

    def test_search_more_dedupes_before_summarize(self):
        """去重必须在 summarize **之前** —— 否则重复论文照样各烧一次 LLM。"""
        src = (ROOT / "agent/tools/surveys/_actions/_search_more.py").read_text(encoding="utf-8")
        assert "dedup_across_sources(raw_papers)" in src
        i_dedup = src.index("dedup_across_sources(raw_papers)")
        i_sum = src.index("build_summarize_prompt(")
        assert i_dedup < i_sum, "去重点在 summarize 之后 —— 重复论文还是各烧一次 LLM"

    def test_search_more_checks_existing_pool(self):
        """以前的 seen_in_search 每次调用都重置, 完全不看 state['pool']。"""
        src = (ROOT / "agent/tools/surveys/_actions/_search_more.py").read_text(encoding="utf-8")
        assert 'state.get("pool")' in src, "去重没跟已有池子比"
        assert "pool_keys" in src

    def test_search_more_carries_doi_into_pool(self):
        """doi 必须带进新条目, 否则下次追加时又认不出跨来源重复。"""
        src = (ROOT / "agent/tools/surveys/_actions/_search_more.py").read_text(encoding="utf-8")
        assert '"doi": p.get("doi")' in src