"""
tests/test_paper_identity.py — 论文身份归一层 (10-04)

背景: 10-04 体检统计全部 949 个真实 paper_id, 发现同一个池子里并存 5 套词汇表,
其中一类是**确定的 bug** —— 旧代码三处逐字相同的

    r.entry_id.split("/")[-1].split("v")[0]

对 `http://arxiv.org/abs/quant-ph/0003004v1` 会先按 "/" 切掉 `quant-ph/`,
只剩 `0003004`。实测池里躺着 5 条这样的裸号(Shor-Preskill 的 BB84 那篇等)。

本文件锁三件事:
  1. 旧格式 arXiv 的 archive 前缀**必须保留**(单射性)
  2. 推导侧产出的形态, 抽取侧必须认得(往返不变式)—— 以前这两套各写各的,
     于是 9-27 的"APS DOI 砍掉 72% 内容"在旧格式 arXiv 上原样重演
  3. 归一/去重键的边界: 归一不折大小写(避免存量数据对不上), 去重键折
"""
from __future__ import annotations

import importlib
import json

import pytest

PID = importlib.import_module("agent.tools.surveys._paper_id")
POOL_UTILS = importlib.import_module("agent.tools.surveys._pool._utils")
ORG = importlib.import_module("agent.tools.surveys._organize")


# ============================================================
# 1. arXiv 归一: 核心是旧格式前缀不能丢
# ============================================================

@pytest.mark.parametrize("raw,expect", [
    # 新格式
    ("2501.10012", "2501.10012"),
    ("2501.10012v3", "2501.10012"),
    ("arXiv:2501.10012", "2501.10012"),
    ("http://arxiv.org/abs/2501.10012v1", "2501.10012"),
    ("https://arxiv.org/pdf/2209.02093v1.pdf", "2209.02093"),
    # 旧格式 —— 前缀必须活下来
    ("http://arxiv.org/abs/quant-ph/0003004v1", "quant-ph/0003004"),
    ("quant-ph/0003004", "quant-ph/0003004"),
    ("hep-th/9901001v2", "hep-th/9901001"),
    ("math.GT/0309136", "math.GT/0309136"),
    ("arXiv:cond-mat/9901001", "cond-mat/9901001"),
    # 垃圾输入不许炸
    ("", ""),
    (None, ""),
    ("   ", ""),
])
def test_normalize_arxiv_id(raw, expect):
    assert PID.normalize_arxiv_id(raw) == expect


def test_old_style_arxiv_ids_stay_distinct():
    """**这条就是 10-03 那两起生产事故的根因所在。**

    旧写法把 archive 前缀切掉, 于是下面 4 个不同的论文塌缩成同一个 "0003001"。
    去重会误杀真论文; 更糟的是 fact_id = `{paper_id}::{idx}`, 塌缩之后两篇论文的
    fact 会共用前缀, 抽取侧一按 paper_id 查表就会把别人的 fact 挂过来
    (9-29 实测"51 篇抽出 0 条 / 挂错")。
    """
    ids = [
        PID.normalize_arxiv_id("http://arxiv.org/abs/quant-ph/0003001v1"),
        PID.normalize_arxiv_id("http://arxiv.org/abs/hep-ph/0003001v1"),
        PID.normalize_arxiv_id("http://arxiv.org/abs/cond-mat/0003001v2"),
        PID.normalize_arxiv_id("http://arxiv.org/abs/hep-th/0003001v1"),
    ]
    assert len(set(ids)) == 4, f"旧格式 id 塌缩了: {ids}"
    # 旧写法的行为, 留作对照 —— 明确它是不对的
    legacy = [x.split("/")[-1].split("v")[0]
              for x in ("http://arxiv.org/abs/quant-ph/0003001v1",
                        "http://arxiv.org/abs/hep-ph/0003001v1")]
    assert legacy[0] == legacy[1] == "0003001", "对照: 旧写法确实塌缩"


def test_version_suffix_never_eats_into_old_style_number():
    """`\\d{7}` 里不可能出现 v, 所以剥版本号不会误伤 —— 但要锁住。"""
    assert PID.normalize_arxiv_id("quant-ph/9711201v1") == "quant-ph/9711201"
    assert PID.normalize_arxiv_id("solv-int/9711201") == "solv-int/9711201"


# ============================================================
# 2. 往返不变式: 推导侧产出什么形态, 抽取侧就必须认得
# ============================================================

@pytest.mark.parametrize("pid", [
    "2501.10012", "quant-ph/0003004", "hep-th/9901001", "math.GT/0309136",
    "0003004",                       # 存量裸号, 抽取侧要继续认
    "10.1103/PhysRevLett.123.100501",
    "10.1103/2frt-tdg9",
    "10.1007/JHEP11(2013)158",       # 10-04 加: JHEP DOI 带圆括号
    "10.1007/jhep11(2013)158",
    "W3104022488",                   # 10-04 加: 无 DOI 的 OpenAlex 文献
    "OA-3f2a1b9c8d7e",               # 10-04 加: 标题 hash 兜底
])
def test_derived_id_is_extractable_from_body(pid):
    """正文里写 `[<推导出的 id>]` 必须抽得回来。

    这是 9-27 那个 bug 的形状: 推导侧产出一种形态、抽取侧认另一种, 于是
    in_pool 恒为 0 -> 误判幻觉 -> 触发重写 -> 砍内容(实测 10892 -> 3075)。

    参数表里的每一行都是**真实存量里出现过的形态** —— 10-04 拿 258 个去重后的
    真实 id 跑这条不变式, 头一次跑出 6 条抽不回来, 其中 3 条就是下面三种
    (JHEP 圆括号 DOI 2 条 + OpenAlex W id 2 条), 已一并修掉。
    """
    got = POOL_UTILS._extract_paper_ids(f"该方法已被多处验证 [{pid}]。")
    assert pid in got, f"推导出的 {pid!r} 抽不回来, 抽到的只有 {got}"


def test_jhep_doi_with_parens_is_not_truncated():
    """`10.1007/JHEP11(2013)158` 必须整个抽出来, 不能只剩 `10.1007/JHEP11`。

    10-04 之前 DOI 的方括号正则把右括号排除在外, 于是 JHEP 的 DOI 全被截断,
    跟池子里的 id 对不上 -> 判成幻觉 -> 触发重写。JHEP 是高能主要刊物。
    """
    for pid in ("10.1007/JHEP11(2013)158", "10.1007/JHEP02(2024)021"):
        got = POOL_UTILS._extract_paper_ids(f"见 [{pid}] 及其后续工作")
        assert got == [pid], f"{pid} 抽成了 {got}"


def test_doi_regex_does_not_swallow_trailing_parenthetical():
    """DOI 后面跟括号说明时不能把括号一起吞掉。"""
    got = POOL_UTILS._extract_paper_ids("见 10.1103/PRXQuantum.3.040310 (2023 年的工作)")
    assert got == ["10.1103/PRXQuantum.3.040310"], f"吞多了: {got}"


def test_old_style_arxiv_survives_the_full_pool_roundtrip():
    """一条旧格式论文走完"推导 -> 抽取 -> 与池子求交"全程都必须算 in_pool。"""
    entry_id = "http://arxiv.org/abs/quant-ph/0003004v1"
    pid = PID.normalize_arxiv_id(entry_id)
    body = f"Shor 与 Preskill 的安全性证明给出了精确界 [{pid}]。"
    bucket = POOL_UTILS._audit_section_paper_ids(body, {pid})
    assert bucket["in_pool"] == [pid], f"旧格式论文被判成池外引用: {bucket}"
    assert bucket["out_of_pool"] == []


def test_audit_normalizes_both_sides():
    """池子里存规范形态, 正文里模型可能带 arXiv: 前缀或改 DOI 大小写, 都要算命中。"""
    pid = PID.normalize_arxiv_id("http://arxiv.org/abs/quant-ph/0003004v1")
    for written in (pid, f"arXiv:{pid}"):
        b = POOL_UTILS._audit_section_paper_ids(f"见 [{written}]", {pid})
        assert b["in_pool"], f"正文写 {written!r} 应命中池里的 {pid!r}, 实得 {b}"
    # DOI 大小写
    b = POOL_UTILS._audit_section_paper_ids(
        "见 [10.1103/physrevlett.123.100501]", {"10.1103/PhysRevLett.123.100501"})
    assert b["in_pool"], "DOI 大小写不同不该被判成幻觉"


# ============================================================
# 3. DOI 归一 + 去重键的边界
# ============================================================

@pytest.mark.parametrize("raw,expect", [
    ("10.1103/PhysRevLett.123.100501", "10.1103/PhysRevLett.123.100501"),
    ("https://doi.org/10.1103/PhysRevLett.123.100501", "10.1103/PhysRevLett.123.100501"),
    ("http://dx.doi.org/10.1103/PhysRevLett.123.100501", "10.1103/PhysRevLett.123.100501"),
    ("doi:10.1103/PhysRevLett.123.100501", "10.1103/PhysRevLett.123.100501"),
    ("  10.1103/PhysRevLett.123.100501  ", "10.1103/PhysRevLett.123.100501"),
    ("", ""),
])
def test_normalize_doi(raw, expect):
    assert PID.normalize_doi(raw) == expect


def test_normalize_preserves_doi_case_but_dedup_key_folds_it():
    """归一不折大小写(存量数据是混合大小写, 折了会新旧对不上), 去重键折。

    这条边界是刻意的: 身份归一和去重是两件事, 混在一起做会逼出数据迁移。
    """
    doi = "10.1103/PhysRevLett.123.100501"
    assert PID.normalize_doi(doi) == doi, "归一必须保大小写"
    assert (PID.dedup_key(doi)
            == PID.dedup_key("10.1103/physrevlett.123.100501")), "去重键必须折"
    assert PID.dedup_key(doi) == PID.dedup_key("https://doi.org/" + doi)


def test_dedup_key_does_not_fold_arxiv():
    """arXiv 侧本来就没有大小写问题, 也不该被折成同一个键。"""
    assert PID.dedup_key("2501.10012") == "2501.10012"
    assert PID.dedup_key("arXiv:2501.10012v2") == "2501.10012"


# ============================================================
# 4. OpenAlex 侧的 id 与 arxiv 侧必须收敛到同一个值
# ============================================================

def test_openalex_pdf_url_and_arxiv_api_agree():
    """同一篇论文走两条路必须得到同一个 paper_id。

    10-04 之前 OpenAlex 分支是 `.split("/")[-1].replace(".pdf","")`, **保留了
    版本号** -> `2209.02093v1`, 而 arxiv API 分支给 `2209.02093`。两个 id 双双
    通过去重, 于是同一篇重复烧一次相关性审查 + 一次 fact 抽取, 参考文献表出现
    重复条目。
    """
    work = {
        "doi": "",
        "primary_location": {
            "source": {"id": "S4306400194", "display_name": "arXiv (Cornell University)"},
            "pdf_url": "https://arxiv.org/pdf/2209.02093v1.pdf",
        },
    }
    assert POOL_UTILS._safe_paper_id(work) == "2209.02093" == \
        PID.normalize_arxiv_id("http://arxiv.org/abs/2209.02093v2")


def test_safe_paper_id_prefers_doi_then_arxiv_then_openalex():
    assert POOL_UTILS._safe_paper_id(
        {"doi": "https://doi.org/10.1103/PRXQuantum.3.040310"}) \
        == "10.1103/PRXQuantum.3.040310"
    assert POOL_UTILS._safe_paper_id(
        {"doi": "", "id": "https://openalex.org/W2741809807"}) == "W2741809807"
    assert POOL_UTILS._safe_paper_id({"doi": "", "id": ""}).startswith("OA-")


# ============================================================
# 5. _IDISH_RE: 无 DOI / 旧格式的 id 不能逃出参考文献表
# ============================================================

@pytest.mark.parametrize("pid", [
    "10.1103/revmodphys.93.045003",
    "arXiv:2401.01234",
    "2301.01234",
    "quant-ph/0003004",      # 10-04 补
    "math.GT/0309136",       # 10-04 补
    "W2741809807",           # 10-04 补 (无 DOI 的 OpenAlex 文献)
    "OA-3f2a1b9c8d7e",       # 10-04 补 (标题 hash 兜底)
])
def test_idish_accepts_every_id_shape_the_pool_can_produce(pid):
    assert ORG._looks_like_paper_id(pid), f"{pid!r} 没被认成 paper_id"


@pytest.mark.parametrize("not_pid", ["1", "42", "见[1]", "", "  "])
def test_idish_still_rejects_plain_numbers(not_pid):
    """不能因为放宽就把正常的引用编号 [1] 也当成 paper_id。"""
    assert not ORG._looks_like_paper_id(not_pid)


def test_old_style_id_gets_numbered_and_a_reference_entry():
    """旧格式论文必须既被编号、也进文献表。

    10-04 之前 _IDISH_RE 不认它 -> renumber 原样保留 [quant-ph/0003004] 且不进
    pid2num -> build_reference_list 不给它出条目 -> 正文挂着引用、文末查无此文,
    覆盖度审计还报它 0 引用, create_section 存进去的 paper_ids 是空的。
    """
    text = ("精确界由 Shor-Preskill 给出 [quant-ph/0003004], "
            "并被后续工作推广 [2501.10012]。")
    out, pid2num = ORG.renumber_citations(text)
    assert "quant-ph/0003004" not in out, "旧格式 id 应被编号替换"
    assert "[" in out and "]" in out
    assert len(pid2num) == 2, f"两篇都该编号, 实得 {pid2num}"


# ============================================================
# 6. 真实数据表征测试 (10-04)
# ============================================================
# 这一段是整个文件里最"真"的部分: 不造数据, 直接拿 data/surveys/working/*/
# _pool.json 里 **10-03 实际跑出来的** 258 个去重 paper_id 跑往返不变式。
# 为什么要有这个:
#   - 单测里的 id 是我挑的, 天然覆盖得住; 真实数据里有我没预料到的形态
#     (JHEP 带括号 DOI、OpenAlex W id、丢了前缀的裸号) 全是这一段先发现的
#   - 以后任何新增/改动造成新形态抽不回来, 这条会立刻红
#
# KNOWN_UNRECOVERABLE 是**故意写死**的例外清单, 不是"容错"。归一化救不回一个
# 从来没被记录过的前缀, 这 3 条只能靠一次性数据映射或直接丢弃。写在这里是
# 为了让它们有名字, 免得以后被当成"已知噪声"默默滑过去。

_KNOWN_UNRECOVERABLE = {
    "wfqb-rfl9": "APS 新式 DOI, 丢了 10.1103/ 前缀; 来自全文导入路径, 非综述召回",
    "gmpy-2sgh": "同上",
    "Elben2022": "BibTeX 式引用键, 不是任何标准标识符形态",
}


def _real_pool_ids():
    """扫真实 workspace 的 _pool.json, 返回 (id -> title)。没有数据就跳过。"""
    import pathlib
    root = pathlib.Path("data/surveys/working")
    if not root.is_dir():
        return {}
    out: dict[str, str] = {}
    for f in root.rglob("_pool.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        arr = j if isinstance(j, list) else (j.get("pool") or j.get("papers") or [])
        for e in arr:
            pid = (e or {}).get("paper_id")
            if pid:
                out.setdefault(pid, (e or {}).get("title") or "")
    return out


def test_real_pool_ids_roundtrip_through_body_extraction():
    """真实 258 个 id: 抽不回来的必须且只能是 KNOWN_UNRECOVERABLE 里那几个。"""
    ids = _real_pool_ids()
    if not ids:
        pytest.skip("没有 data/surveys/working/*/_pool.json, 跳过真实数据表征测试")

    failures = []
    for pid in ids:
        norm = PID.normalize_paper_id(pid)
        got = POOL_UTILS._extract_paper_ids(f"参见 [{norm}]。")
        if norm not in got:
            failures.append((pid, norm, got))

    unexpected = {p for p, _, _ in failures} - set(_KNOWN_UNRECOVERABLE)
    assert not unexpected, (
        f"这 {len(unexpected)} 个真实 id 抽不回来了(新增的回归): {sorted(unexpected)}\n"
        f"全部失败项: {failures}"
    )
    # 例外清单里的必须真的还失败 —— 哪天数据修好了, 这里会提醒你把清单改小
    still_bad = {p for p, _, _ in failures}
    assert still_bad == set(_KNOWN_UNRECOVERABLE), (
        f"例外清单跟实际不符: 实际={sorted(still_bad)} "
        f"清单={sorted(_KNOWN_UNRECOVERABLE)}"
    )


def test_real_pool_ids_are_all_dedupe_stable():
    """真实 id 归一之后必须幂等: normalize(normalize(x)) == normalize(x)。"""
    ids = _real_pool_ids()
    if not ids:
        pytest.skip("同上")
    for pid in ids:
        once = PID.normalize_paper_id(pid)
        twice = PID.normalize_paper_id(once)
        assert once == twice, f"{pid!r} 归一不幂等: {once!r} -> {twice!r}"
        assert PID.dedup_key(pid) == PID.dedup_key(once), f"{pid!r} 去重键不幂等"


def test_real_paper_id_distribution_is_documented():
    """把真实库里"认得出 / 认不出"的形态分布记下来, 认不出的会变多就会看到。

    这条不是断言质量, 是**变化探测器**: 某天归一化被改坏、或者新增了一条不归一
    的摄取路径, 这个分布会漂。
    """
    ids = _real_pool_ids()
    if not ids:
        pytest.skip("同上")
    known = unknown = 0
    unknown_shapes = set()
    for pid in ids:
        n = PID.normalize_paper_id(pid)
        if (PID.looks_like_arxiv_id(n) or PID.looks_like_doi(n)
                or PID.looks_like_openalex_id(n)):
            known += 1
        else:
            unknown += 1
            unknown_shapes.add(n)
    total = known + unknown
    ratio = unknown / total
    assert ratio <= 0.05, (
        f"认不出的 id 占比 {ratio:.1%} ({unknown}/{total}) 超过 5% 上限, "
        f"说明有新的不归一来源混进来了: {sorted(unknown_shapes)[:10]}"
    )

