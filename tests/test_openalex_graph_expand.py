"""
test_openalex_graph_expand.py — 引文图原语 + 扩展层 (9-30)

两块:
  A. openalex 原语 —— 重点是**方向语义**(OpenAlex 命名反直觉, 两个 key 与字面相反)
  B. _pool._expand —— 深度 / 广度 / 剪枝 三道约束

全程不联网: OpenAlex 侧用假 session, 扩展层用注入的 fetch。
"""
from __future__ import annotations

import importlib
import sys

import pytest


# ============================================================
# 模块解析 (与 test_openalex_rate_limit.py 同因:
# test_no_circular_imports.py 会重新导入模块, sys.modules 里换成另一个对象,
# 顶层绑死的 oa 会失效 —— 桩子打上去, 被测函数在别的对象上找属性, 静默不验)
# ============================================================
def _mod(name: str):
    if name not in sys.modules:
        importlib.import_module(name)
    return sys.modules[name]


OA = _mod("agent.tools.openalex")
EX = _mod("agent.tools.surveys._pool._expand")
OX = _mod("agent.tools.surveys._pool._openalex")


# ============================================================
# 假 OpenAlex
# ============================================================
class _Resp:
    def __init__(self, status, headers=None, payload=None, text=""):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload if payload is not None else {"results": []}
        self.text = text or str(payload)[:200]

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _CaptureSession:
    """记录发出去的 params, 并按预设脚本返回响应。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def get(self, url, params=None, timeout=None):
        self.sent.append(dict(params or {}))
        return self.responses.pop(0) if self.responses else _Resp(200, payload={"results": []})


def _patch(monkeypatch, responses):
    sess = _CaptureSession(responses)
    monkeypatch.setattr(OA, "_build_session", lambda: sess)
    monkeypatch.setattr(OA.time, "sleep", lambda s: None)
    return sess


def _work(wid, title="Title", cites=0, wtype="article", abstract="quantum error correction"):
    """造一条 OpenAlex work (abstract_inverted_index 格式)。"""
    inv = None
    if abstract:
        toks = abstract.split()
        inv = {w: [i] for i, w in enumerate(toks)}
    return {
        "id": f"https://openalex.org/{wid}",
        "doi": f"https://doi.org/10.1/{wid.lower()}",
        "title": title,
        "display_name": title,
        "publication_year": 2020,
        "publication_date": "2020-01-01",
        "authorships": [],
        "primary_location": {"source": {"display_name": "Test Journal"}},
        "best_oa_location": None,
        "abstract_inverted_index": inv,
        "cited_by_count": cites,
        "open_access": {},
        "type": wtype,
    }


# ============================================================
# A. 原语: 方向语义 (最重要)
# ============================================================

def test_openalex_cites_uses_cites_key(monkeypatch):
    """
    前向 = 谁引用了它 -> 必须发 filter=cites:...

    命名坑: referenced_works 听着像后向, 实际也是前向(cites 的别名)。
    """
    sess = _patch(monkeypatch, [_Resp(200, payload={"results": [_work("W1")]})])
    OA.openalex_cites(["W1"], per_page=10, caller="T")
    f = sess.sent[0]["filter"]
    assert f.startswith("cites:W1"), f"前向应发 cites:, 实发 {f!r}"
    assert "referenced_works" not in f


def test_openalex_references_uses_cited_by_key(monkeypatch):
    """
    后向 = 它引用了谁 -> 必须发 filter=cited_by:...

    ★ 这是最容易搞反的一处: "cited_by" 名字听着像前向, 实际是后向。
      实测: cited_by:W123 返回的每一条, 自己的 referenced_works 里都有 W123。
    """
    sess = _patch(monkeypatch, [_Resp(200, payload={"results": [_work("W1")]})])
    OA.openalex_references(["W1"], per_page=10, caller="T")
    f = sess.sent[0]["filter"]
    assert f.startswith("cited_by:W1"), f"后向应发 cited_by:, 实发 {f!r}"
    assert not f.startswith("cites:"), "后向绝不能发 cites:(那是前向)"


def test_two_directions_are_distinct(monkeypatch):
    """前向和后向必须发不同的 filter key —— 搞反了整个扩展层就废了。"""
    s1 = _patch(monkeypatch, [_Resp(200, payload={"results": []})])
    OA.openalex_cites(["W1"], caller="T")
    fwd = s1.sent[0]["filter"]
    s2 = _patch(monkeypatch, [_Resp(200, payload={"results": []})])
    OA.openalex_references(["W1"], caller="T")
    bwd = s2.sent[0]["filter"]
    assert fwd != bwd, f"前向/后向撞了: {fwd}"


def test_seeds_are_or_joined(monkeypatch):
    """多 seed 用 | 连接 = OR 语义(实测非 AND)。"""
    sess = _patch(monkeypatch, [_Resp(200, payload={"results": []})])
    OA.openalex_cites(["W1", "W2", "W3"], caller="T")
    assert sess.sent[0]["filter"] == "cites:W1|W2|W3"


def test_reviews_only_adds_type_review(monkeypatch):
    """综述通道: 用户"不一定需要完整文章, 也可以只找综述"的需求。"""
    sess = _patch(monkeypatch, [_Resp(200, payload={"results": []})])
    OA.openalex_cites(["W1"], reviews_only=True, caller="T")
    assert "type:review" in sess.sent[0]["filter"]


def test_seeds_normalized_and_deduped(monkeypatch):
    """完整 URL 和裸 id 混着传也要归一 + 去重。"""
    sess = _patch(monkeypatch, [_Resp(200, payload={"results": []})])
    OA.openalex_cites(
        ["https://openalex.org/W1", "W1", "W2", "", None], caller="T")
    assert sess.sent[0]["filter"] == "cites:W1|W2"


def test_seeds_truncated_to_50(monkeypatch):
    """单次上限 50 seed —— 静默截断会让人以为"扩了全部"。"""
    sess = _patch(monkeypatch, [_Resp(200, payload={"results": []})])
    OA.openalex_cites([f"W{i}" for i in range(80)], caller="T")
    n = len(sess.sent[0]["filter"].split(":")[1].split("|"))
    assert n == 50, f"应截断到 50, 实得 {n}"


def test_empty_seeds_short_circuits(monkeypatch):
    """没 seed 就不该发请求(省 credit)。"""
    sess = _patch(monkeypatch, [])
    assert OA.openalex_cites([], caller="T") == []
    assert sess.sent == []


def test_per_page_capped_at_200(monkeypatch):
    sess = _patch(monkeypatch, [_Resp(200, payload={"results": []})])
    OA.openalex_cites(["W1"], per_page=99999, caller="T")
    assert sess.sent[0]["per_page"] == 200


def test_graph_http_error_raises_with_status(monkeypatch):
    """引文图失败必须报出来 —— 扩展层塌了就只剩另一路来源了。"""
    _patch(monkeypatch, [_Resp(400, text="bad filter")])
    with pytest.raises(RuntimeError, match="400"):
        OA.openalex_cites(["W1"], caller="T")


def test_batch_by_doi_normalizes_and_keys(monkeypatch):
    sess = _patch(monkeypatch, [
        _Resp(200, payload={"results": [
            _work("W1"), _work("W2"),
        ]}),
    ])
    out = OA.openalex_batch_by_doi(
        ["https://doi.org/10.1/W1", "10.1/w2", "", None], caller="T")
    f = sess.sent[0]["filter"]
    assert f.startswith("doi:") and "10.1/w1" in f and "10.1/w2" in f
    assert set(out) == {"10.1/w1", "10.1/w2"}, f"返回键应是小写裸 DOI, 实得 {set(out)}"


def test_batch_by_doi_empty_no_request(monkeypatch):
    sess = _patch(monkeypatch, [])
    assert OA.openalex_batch_by_doi([], caller="T") == {}
    assert sess.sent == []


# ============================================================
# B. 扩展层: 深度 / 广度 / 剪枝
# ============================================================

def _entry(wid, title="T", abstract="quantum error correction", cited=0):
    return {
        "paper_id": wid.lower(), "title": title, "abstract": abstract,
        "cited_by_count": cited, "type": "article",
        "_openalex_id": wid, "_source": "test",
    }


class _FakeFetch:
    """按 seed 生成下一层, 记录调用。"""

    def __init__(self, fanout=2, n_per_seed=1, start=1):
        self.calls = []
        self.fanout = fanout
        self.n_per_seed = n_per_seed
        self._n = start

    def __call__(self, direction, seed_ids, per_page, reviews_only):
        self.calls.append((direction, tuple(seed_ids), per_page, reviews_only))
        out = []
        for _ in range(self.fanout):
            for s in seed_ids:
                for _ in range(self.n_per_seed):
                    out.append(_work(f"W{self._n}", abstract="quantum error correction"))
                    self._n += 1
        return out


def test_expand_respects_depth_limit():
    """深度限制: max_rounds=1 就只跑一轮。"""
    f = _FakeFetch()
    r = EX.expand_from_seeds(
        [_entry("W1")],
        config=EX.ExpandConfig(max_rounds=1, forward_breadth=2, backward_breadth=2),
        fetch=f,
    )
    assert r.stats["rounds"] == 1
    assert "max_rounds" in r.stats["stopped_reason"]


def test_expand_runs_exactly_max_rounds_when_work_remains():
    f = _FakeFetch()
    r = EX.expand_from_seeds(
        [_entry("W1")],
        config=EX.ExpandConfig(max_rounds=3, forward_breadth=2, backward_breadth=2),
        fetch=f,
    )
    assert r.stats["rounds"] == 3


def test_expand_respects_breadth_limit():
    """广度限制: 每轮前向/后向各只取 breadth 篇去扩。"""
    f = _FakeFetch()
    EX.expand_from_seeds(
        [_entry(f"W{i}") for i in range(1, 30)],
        config=EX.ExpandConfig(max_rounds=1, forward_breadth=3, backward_breadth=5),
        fetch=f,
    )
    for direction, seeds, _, _ in f.calls:
        want = 3 if direction == "forward" else 5
        assert len(seeds) == want, f"{direction} 用了 {len(seeds)} 篇, 应为 {want}"


def test_expand_caps_requests():
    """credit 兜底: 请求数不能超过 max_requests。"""
    f = _FakeFetch()
    r = EX.expand_from_seeds(
        [_entry(f"W{i}") for i in range(1, 40)],
        config=EX.ExpandConfig(
            max_rounds=5, forward_breadth=20, backward_breadth=20, max_requests=3),
        fetch=f,
    )
    assert r.stats["requests"] <= 3
    assert "max_requests" in r.stats["stopped_reason"]


def test_expand_caps_candidates():
    f = _FakeFetch(fanout=5, n_per_seed=3)
    r = EX.expand_from_seeds(
        [_entry("W1")],
        config=EX.ExpandConfig(
            max_rounds=3, forward_breadth=5, backward_breadth=5, max_candidates=7),
        fetch=f,
    )
    assert len(r.entries) <= 7
    assert "max_candidates" in r.stats["stopped_reason"]


def test_expand_dedups_across_rounds():
    """同一篇会被多个 seed / 多轮反复召回, 必须只留一份。"""
    def same_every_time(direction, seed_ids, per_page, reviews_only):
        return [_work("W999", abstract="quantum error correction")]
    r = EX.expand_from_seeds(
        [_entry("W1"), _entry("W2")],
        config=EX.ExpandConfig(max_rounds=2, forward_breadth=5, backward_breadth=5),
        fetch=same_every_time,
    )
    ids = [e["_openalex_id"] for e in r.entries]
    assert len(ids) == len(set(ids)), f"有重复: {ids}"


def test_expand_survives_fetch_failure(capsys):
    """
    扩展层塌了一路, 不能整轮崩掉, 而且必须**打印出来**。

    9-30 实测教训: OpenAlex 429 时旧代码隐式返回 None, 被 except Exception
    吞掉只进 logger, 控制台一片正常, 并集静默退化成只剩 arXiv 一路 ——
    在材料类学科那等于丢了 77/78。
    """
    def boom(direction, seed_ids, per_page, reviews_only):
        raise RuntimeError("[http 429] 限流")
    r = EX.expand_from_seeds(
        [_entry("W1")],
        config=EX.ExpandConfig(max_rounds=1, forward_breadth=2, backward_breadth=2),
        fetch=boom,
    )
    out = capsys.readouterr().out
    assert "429" in out, "扩展失败必须打到控制台, 不能只进 logger"
    # 一轮里前向 1 次 + 后向 1 次; 全失败后 frontier 空 -> 提前收敛, 所以是 2 不是 4
    assert r.stats["requests"] == 2, "失败不该中断整轮, 其余请求应继续"


# ============================================================
# C. 剪枝
# ============================================================

def test_prune_drops_seen_and_counts():
    seen = {"W1"}
    cands = [_entry("W1"), _entry("W2")]
    kept, drops = EX.prune(cands, seen)
    assert [e["_openalex_id"] for e in kept] == ["W2"]
    assert drops["seen"] == 1


def test_prune_drops_no_title():
    kept, drops = EX.prune([_entry("W1", title="   ")], set())
    assert kept == []
    assert drops["no_title"] == 1


def test_prune_off_topic():
    cands = [_entry("W1", title="Kirigami stent for heart surgery",
                    abstract="medical device for cardiovascular")]
    kept, drops = EX.prune(cands, set(), topic="tensor network contraction",
                           min_topic_overlap=0.5)
    assert kept == []
    assert drops["off_topic"] == 1


def test_prune_keeps_on_topic():
    cands = [_entry("W1", title="Tensor network contraction algorithms",
                    abstract="we study tensor network contraction cost")]
    kept, drops = EX.prune(cands, set(), topic="tensor network contraction",
                           min_topic_overlap=0.3)
    assert len(kept) == 1 and drops["off_topic"] == 0


def test_prune_low_cited():
    kept, drops = EX.prune([_entry("W1", cited=2)], set(), min_cited_by=10)
    assert kept == [] and drops["low_cited"] == 1


def test_prune_drops_entries_without_oa_id():
    kept, drops = EX.prune([{"title": "x", "abstract": "y"}], set())
    assert kept == [] and drops["no_id"] == 1


def test_prune_never_seeds_topic_filter_when_disabled():
    """min_topic_overlap=0 时不该剪(默认关掉, 免得误杀)。"""
    cands = [_entry("W1", title="Anything", abstract="unrelated")]
    kept, drops = EX.prune(cands, set(), topic="tensor network", min_topic_overlap=0.0)
    assert len(kept) == 1


# ============================================================
# D. 广度选择的排序
# ============================================================

def test_select_frontier_prefers_has_abstract_then_citations():
    es = [
        _entry("W1", abstract="", cited=999),        # 没摘要但高被引
        _entry("W2", abstract="quantum", cited=1),    # 有摘要但低被引
        _entry("W3", abstract="quantum", cited=50),
    ]
    got = [e["_openalex_id"] for e in EX.select_frontier(es, 2)]
    assert got == ["W3", "W2"], f"有摘要的应优先, 再按被引排序; 实得 {got}"


def test_select_frontier_topic_first_over_citations():
    """
    9-30 实测修正: 给了 topic 就按主题匹配度排, 被引量只做次序。

    否则被引量最高的会是邻域里"只是顺带引用"的高引经典论文 ——
    实测扩出 727 篇取前 40, 词法门控毙掉 39, 只采纳 1 篇。
    """
    es = [
        _entry("W1", title="High cited but adjacent",
               abstract="quantum error correction codes for surface lattices", cited=9999),
        _entry("W2", title="On topic tensor network work",
               abstract="tensor network states for quantum simulation of lattice models", cited=1),
    ]
    got = EX.select_frontier(es, 1, topic="tensor network states quantum simulation")
    assert [e["_openalex_id"] for e in got] == ["W2"], \
        "主题切题的应压过被引量高但只是邻域的"


def test_select_frontier_without_topic_keeps_old_order():
    """不给 topic 时保持原排序(有摘要优先, 再按被引量), 行为不变。"""
    es = [
        _entry("W1", abstract="", cited=999),
        _entry("W2", abstract="quantum", cited=1),
        _entry("W3", abstract="quantum", cited=50),
    ]
    got = [e["_openalex_id"] for e in EX.select_frontier(es, 2)]
    assert got == ["W3", "W2"]


def test_select_frontier_respects_breadth():
    es = [_entry(f"W{i}") for i in range(50)]
    assert len(EX.select_frontier(es, 7)) == 7
    assert EX.select_frontier(es, 0) == []


# ============================================================
# E. 词表/重合度
# ============================================================

def test_topic_overlap_full_and_zero():
    assert EX.topic_overlap("tensor network", "Tensor Network contraction") == 1.0
    assert EX.topic_overlap("tensor network", "kirigami medical stent") == 0.0


def test_topic_overlap_empty_topic_is_permissive():
    assert EX.topic_overlap("", "anything") == 1.0


def test_topic_overlap_empty_text_is_zero():
    assert EX.topic_overlap("tensor network", "") == 0.0


def test_topic_overlap_handles_chinese():
    assert EX.topic_overlap("张量网络", "张量网络的收缩算法") > 0


# ============================================================
# F. pool entry 携带扩展层需要的字段
# ============================================================

def test_work_to_entry_carries_graph_fields():
    """
    9-30: 扩展层要靠 _openalex_id 去重和当 seed, 靠 cited_by_count 排序,
    靠 type 走综述通道。这三个字段原来没有。
    """
    e = OX._openalex_work_to_pool_entry(
        _work("W123", title="A", cites=42, wtype="review"), "src")
    assert e["_openalex_id"] == "W123"
    assert e["cited_by_count"] == 42
    assert e["type"] == "review"


def test_work_to_entry_keeps_legacy_keys():
    """纯增量: 已有 key 不能被改坏。"""
    e = OX._openalex_work_to_pool_entry(_work("W1"), "src")
    for k in ("paper_id", "title", "year", "journal", "abstract",
              "_match_score", "_source"):
        assert k in e, f"丢了旧 key: {k}"


def test_oa_id_accepts_url_form():
    assert EX._entry_oa_id({"id": "https://openalex.org/W9"}) == "W9"
    assert EX._entry_oa_id({"_openalex_id": "https://openalex.org/W8"}) == "W8"
    assert EX._entry_oa_id({"id": "not-an-oa-id"}) == ""


# ============================================================
# G. 配置自洽
# ============================================================

def test_requests_for_is_an_upper_bound():
    """
    requests_for() 是**上界**不是精确值 —— 剪枝会让 frontier 变小、提前收敛会让轮数变少,
    实际用量只会更少。所以校验口径是 <= 而不是 ==。
    """
    cfg = EX.ExpandConfig(max_rounds=2, forward_breadth=40, backward_breadth=40)
    f = _FakeFetch(fanout=1, n_per_seed=1)
    r = EX.expand_from_seeds([_entry("W1")], config=cfg, fetch=f)
    assert r.stats["requests"] <= cfg.requests_for()


def test_requests_for_batch_counted_per_direction():
    """
    forward / backward 是两个独立方向, 批次数要分别算。

    踩过的坑: 曾把两边的 seed 加起来再除以 50
    (ceil((40+40)/50)*2 = 4), 实际是 ceil(40/50)+ceil(40/50) = 2 —— 预算虚高一倍。
    """
    cfg = EX.ExpandConfig(max_rounds=1, forward_breadth=40, backward_breadth=40)
    assert cfg.requests_for() == 2

    # 塞到 50 以上才会真的分成多批
    cfg2 = EX.ExpandConfig(max_rounds=1, forward_breadth=120, backward_breadth=10)
    assert cfg2.requests_for() == 3 + 1


def test_requests_for_bound_respected_with_full_breadth():
    """frontier 塞满时确实用到上界 —— 说明估算没有低估。"""
    seeds = [_entry(f"W{i}") for i in range(1, 81)]
    cfg = EX.ExpandConfig(max_rounds=1, forward_breadth=40, backward_breadth=40)
    f = _FakeFetch(n_per_seed=0)
    r = EX.expand_from_seeds(seeds, config=cfg, fetch=f)
    assert r.stats["requests"] == cfg.requests_for() == 2
    assert r.stats["requests"] <= cfg.max_requests
