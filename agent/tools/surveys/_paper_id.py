"""
agent.tools.surveys._paper_id — 论文身份的**统一归一层** (10-04)

为什么要有这一层
----------------
10-04 体检统计了 data/surveys/working/*/_pool.json 里全部 949 个真实 paper_id,
发现同一个候选池里并存 **5 套互不兼容的词汇表**, 而产生它们的地方没有任何归一化:

    558  2501.10012                 arXiv 新格式 (2007-04 之后)
    367  10.1103/physrevlett.123…   DOI
      8  0003004                    ← 旧格式 arXiv, archive 前缀被整段剥掉
      6  wfqb-rfl9                  ← APS 新式 DOI, "10.1103/" 前缀被剥掉
      3  Elben2022                  BibTeX 式引用键
      2  W2741809807                OpenAlex work id

**第 3 类是真 bug**, 来自三处逐字相同的
`r.entry_id.split("/")[-1].split("v")[0]`(`_pool/_arxiv.py:26,72`,
`_pool/_build_pool.py:269`)。对

    http://arxiv.org/abs/quant-ph/0003004v1

先按 "/" 切 —— `quant-ph/` 没了, 只剩 `0003004v1`; 再按 "v" 切 —— 只剩
`0003004`。后果有两层:

  1. **单射性被破坏**: `quant-ph/0003004` / `hep-ph/0003004` / `cond-mat/0003004`
     塌缩成同一个 id, 于是去重会误杀真论文, 或者让两条不同的论文共用一个 fact 前缀
     (fact_id = `{paper_id}::{idx}`, 见 prompts_survey)。
  2. **推导侧和抽取侧对不上** —— 这一层更隐蔽。`_extract_paper_ids` 的正则全都
     只认"裸 7 位数"和新格式, 认不出带 archive 前缀的形态。所以即使池子里的 key
     是对的, 正文写 `[quant-ph/0003004]` 也抽不出 id -> `in_pool=0` ->
     触发"幻觉引用"重写 -> 砍内容。9-27 已经因为 APS DOI 踩过一次同一形状的坑
     (一段 10892 字符被误判触发 2 次重写, 砍到 3075), 这次是换了个来源。

本模块只做一件事: **把各种来源的原始字符串收敛成唯一规范形态**, 并让
"正文抽取"和"ID 推导"**共用同一套形态定义** —— 这两套以前各写各的, 必然漂移。

刻意不做的事 (留给后续, 别在这里扩大范围)
----------------------------------------
  - 不做 DOI <-> arXiv 的实体解析(同一篇的预印本与期刊版合并成一条)。那需要
    OpenAlex 批量查询, 属于 _pool 层的活; `agent/tools/openalex.py::openalex_batch_by_doi`
    已经写好了(1 credit 拿 200 条)但综述路径还没调用。
  - 不做历史数据迁移。现有 _pool.json / summary_items.db 里的旧 id 原样保留,
    重跑该主题时自然按新规则生成。旧数据靠 dedup_key 的兼容分支继续可用。
  - 不改 DOI 的大小写(见 normalize_doi 的说明)。
"""
from __future__ import annotations

import re

# ============================================================
# 形态定义 —— 推导侧和抽取侧共用这一份, 不要再各写一套
# ============================================================

# arXiv 新格式: 2501.10012 (YYMM.NNNNN), 2007-04 之后启用
ARXIV_NEW_RE = re.compile(r"^\d{4}\.\d{4,5}$")

# arXiv 旧格式: <archive>/<YYMMNNN>, 2007-03 之前。archive 可带学科后缀
# (math.GT/0309136) 也可不带 (hep-th/9901001)。
# **前缀必须保留** —— 它是身份的一部分, 剥掉之后不同 archive 的同号论文会塌缩。
ARXIV_OLD_RE = re.compile(r"^[a-z][a-z0-9-]*(?:\.[A-Za-z]{2})?/\d{7}$", re.I)

# DOI: 10.NNNN/suffix (APS 是 10.1103, IOP 是 10.1088)
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.I)

# OpenAlex work id (无 DOI 的文献才会落到这里)
OPENALEX_RE = re.compile(r"^W\d+$", re.I)

# _safe_paper_id 的标题 hash 兜底
OA_HASH_RE = re.compile(r"^OA-[0-9a-f]{8,}$", re.I)

# 正文抽取用的 arXiv 备选串。注意**保留了裸 7 位数**:
# 现存 workspace 里 8 条 paper_id 就是被剥掉前缀的裸号, 保留它是为了让旧
# workspace 的审计/重写逻辑继续可用, 而不是顺手把旧数据判成幻觉。
# 假阳性(比如 "[Appendix B/1234567]")只会落到 out_of_pool 桶里报一次
# "疑似幻觉", 不会污染数据 —— `_audit_section_paper_ids` 会拿它跟池子求交。
ARXIV_ID_ALT = r"\d{4}\.\d{4,5}|[a-z][a-z0-9-]*(?:\.[A-Za-z]{2})?/\d{7}|\d{7}"


# ============================================================
# 前缀 / 后缀清洗
# ============================================================

_ARXIV_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf|html|format)/", re.I)
_ARXIV_SCHEME_RE = re.compile(r"^arxiv[:\s/]+", re.I)
_ARXIV_PDF_SUFFIX_RE = re.compile(r"\.pdf$", re.I)
# 版本号只在末尾。旧格式的 7 位数字里不可能出现 'v', 所以不会误伤
# (quant-ph/9901001 里没有 v; math.GT/0309136 里也没有)
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")

_DOI_URL_RE = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/", re.I)
_DOI_SCHEME_RE = re.compile(r"^doi[:\s]+", re.I)


# ============================================================
# 归一函数
# ============================================================

def normalize_arxiv_id(raw: str) -> str:
    """把任意形态的 arXiv 标识收敛成规范形态。旧格式**保留 archive 前缀**。

        http://arxiv.org/abs/quant-ph/0003004v1  ->  quant-ph/0003004
        https://arxiv.org/pdf/2209.02093v1.pdf  ->  2209.02093
        arXiv:2501.10012                         ->  2501.10012
        2501.10012v3                             ->  2501.10012

    与 10-03 之前的 `entry_id.split("/")[-1].split("v")[0]` 的区别就是第一行:
    旧写法把 `quant-ph/` 一起丢掉了, 于是 quant-ph/0003004 和 hep-ph/0003004
    会变成同一个 paper_id。
    """
    s = (raw or "").strip()
    if not s:
        return ""
    s = _ARXIV_URL_RE.sub("", s)
    s = _ARXIV_SCHEME_RE.sub("", s)
    s = _ARXIV_PDF_SUFFIX_RE.sub("", s)
    s = _VERSION_SUFFIX_RE.sub("", s)
    return s.strip("/").strip()


def normalize_doi(raw: str) -> str:
    """剥掉 URL / doi: 前缀。**保留大小写**。

    DOI 按规范是大小写不敏感的, 但现存 summary_items.db / _pool.json 里存的是
    混合大小写(`10.1103/PhysRevLett.123.100501` 和全小写写法都有)。如果这里
    直接折成小写, 新旧两批数据在去重时会对不上, 同一篇论文变成两条 —— 那是拿
    一个新 bug 换旧 bug。

    大小写折叠交给 dedup_key() 做: **身份归一**和**去重键**是两件事, 分开做
    就都不需要数据迁移。什么时候想把存量也折小写, 那是另一个显式的迁移动作。
    """
    s = (raw or "").strip()
    if not s:
        return ""
    s = _DOI_URL_RE.sub("", s)
    s = _DOI_SCHEME_RE.sub("", s)
    return s.strip().rstrip(".").strip()


def normalize_paper_id(raw: str) -> str:
    """认出是哪种标识就按哪种归一; 认不出就原样返回(去空白 + 剥 http 前缀)。"""
    s = (raw or "").strip()
    if not s:
        return ""
    if looks_like_arxiv_id(s):
        return normalize_arxiv_id(s)
    if looks_like_doi(s):
        return normalize_doi(s)
    # OpenAlex / OA-hash: 本来就是规范形态, 但 URL 形态要剥
    s = re.sub(r"^https?://openalex\.org/", "", s, flags=re.I).strip()
    return s


def dedup_key(pid: str) -> str:
    """候选池去重键。

    与 normalize_paper_id 的区别: 这里**额外**折 DOI 大小写, 所以
    `10.1103/PhysRevLett.123.100501` 和 `10.1103/physrevlett.123.100501`
    会被认成同一篇, 而存进 pool_dict 的 key 仍是各自原样(不引入迁移)。
    arXiv 侧不含大小写敏感内容, 直接归一即可。
    """
    s = normalize_paper_id(pid)
    if not s:
        return ""
    if looks_like_doi(s):
        return s.lower()
    return s


# ============================================================
# 形态判定
# ============================================================

def looks_like_arxiv_id(s: str) -> bool:
    """新格式 / 旧格式 / 带 arXiv: 前缀 / 带版本号 都算。

    注意必须**先剥版本号再判形态** —— 否则 `arXiv:2501.10012v2` 判不出是 arXiv,
    normalize_paper_id 就会把它当未知串原样返回, 去重键也跟着带上版本号,
    同一篇论文在两条路上又变成两个 id (10-04 写这个模块时被自己的测试抓到)。
    """
    t = (s or "").strip()
    t = _ARXIV_URL_RE.sub("", t)
    t = _ARXIV_SCHEME_RE.sub("", t)
    t = _ARXIV_PDF_SUFFIX_RE.sub("", t)
    t = _VERSION_SUFFIX_RE.sub("", t)
    return bool(ARXIV_NEW_RE.match(t) or ARXIV_OLD_RE.match(t))


def looks_like_doi(s: str) -> bool:
    t = (s or "").strip()
    t = _DOI_URL_RE.sub("", t)
    t = _DOI_SCHEME_RE.sub("", t)
    return bool(DOI_RE.match(t))


def looks_like_openalex_id(s: str) -> bool:
    t = (s or "").strip()
    t = re.sub(r"^https?://openalex\.org/", "", t, flags=re.I).strip()
    return bool(OPENALEX_RE.match(t) or OA_HASH_RE.match(t))


def is_old_style_arxiv_id(s: str) -> bool:
    """只认旧格式(带 archive 前缀)。迁移/诊断脚本用。"""
    t = normalize_arxiv_id(s)
    return bool(ARXIV_OLD_RE.match(t))


__all__ = [
    "ARXIV_NEW_RE", "ARXIV_OLD_RE", "DOI_RE", "OPENALEX_RE", "OA_HASH_RE",
    "ARXIV_ID_ALT",
    "normalize_arxiv_id", "normalize_doi", "normalize_paper_id", "dedup_key",
    "looks_like_arxiv_id", "looks_like_doi", "looks_like_openalex_id",
    "is_old_style_arxiv_id",
]
