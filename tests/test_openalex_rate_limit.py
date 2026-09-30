"""
tests/test_openalex_rate_limit.py — OpenAlex 限流处理

9-30。

问题:
    1. User-Agent 里写死 `mailto:qagent@example.com` —— **假邮箱**。
       OpenAlex 的 polite pool 要求真实联系方式, 用假邮箱等于没申请,
       拿到的就是较紧的配额。实测撞到 429:
         "Anonymous search is temporarily rate-limited while the search cluster
          is under elevated load. Please retry in 36s, or use a free API key"
    2. 只对 5xx 重试, 429 直接当 4xx 放弃 —— 撞上限流这一轮召回就废了,
       Phase A 因此卡住(实测一次 run 卡在 Phase A 十几分钟)。

修:
    1. User-Agent 的邮箱改成读 OPENALEX_MAILTO 环境变量, 没配就用占位符
       并在 429 日志里提示怎么配。
    2. 429 按响应里的 Retry-After 退避重试(与 5xx 分开计数)。
"""
from __future__ import annotations

import importlib
import sys

import pytest


def _mod():
    """取 sys.modules 里**当前注册**的 openalex 模块对象。

    不在 import 时把顶层 `oa` 绑死: tests/test_no_circular_imports.py 会把
    agent.tools.openalex 重新导入一遍, sys.modules 里注册的随即换成另一个对象。
    此时顶层 oa 指向的是旧对象, monkeypatch 打上去, 被测函数却在别的对象上找属性,
    桩子静默失效 —— 测试"绿"但什么都没验到。这里统一走 sys.modules 取。
    """
    name = "agent.tools.openalex"
    if name not in sys.modules:
        importlib.import_module(name)
    return sys.modules[name]


def _reload(monkeypatch, mailto):
    """重载模块, 让 OPENALEX_MAILTO 生效。mailto=None 表示未设置该环境变量。"""
    if mailto is None:
        monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
    else:
        monkeypatch.setenv("OPENALEX_MAILTO", mailto)
    # importlib.reload() 校验的是 `sys.modules.get(name) is not module` —— 身份,
    # 不是名字, 所以必须重载 sys.modules 里当前那个。
    return importlib.reload(_mod())


# ============================================================
# User-Agent / polite pool
# ============================================================

def test_user_agent_uses_env_email(monkeypatch):
    m = _reload(monkeypatch, "xuefeng@realmail.com")
    assert "xuefeng@realmail.com" in m._USER_AGENT
    assert m._openalex_identity_note() is None, "配了真实邮箱就不该再提示"


def test_falls_back_to_placeholder(monkeypatch):
    m = _reload(monkeypatch, None)
    assert "qagent@example.com" in m._USER_AGENT
    note = m._openalex_identity_note()
    assert note and "OPENALEX_MAILTO" in note, "没配邮箱时应提示怎么配"


def test_blank_env_treated_as_unset(monkeypatch):
    """空串 / 纯空格都该当没配 —— 否则 UA 里会出现 'mailto:   ' 这种废值。"""
    for blank in ("", "   ", "\t"):
        m = _reload(monkeypatch, blank)
        assert "qagent@example.com" in m._USER_AGENT, f"空白值 {blank!r} 应回落到占位邮箱"
        assert m._openalex_identity_note(), f"空白值 {blank!r} 应仍提示去配邮箱"


def test_no_fake_mailto_in_ua_prefix(monkeypatch):
    """UA 格式要符合 OpenAlex 约定 (mailto:xxx)。"""
    m = _reload(monkeypatch, "me@example.org")
    assert m._USER_AGENT.startswith("QAgent/")
    assert "mailto:" in m._USER_AGENT


# ============================================================
# 429 退避
# ============================================================

class _FakeResp:
    def __init__(self, status, headers=None, text="", payload=None):
        self.status_code = status
        self.headers = headers or {}
        self.text = text
        self._payload = payload or {"results": [], "meta": {"count": 0}}

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


@pytest.fixture
def fast_sleep(monkeypatch):
    """把退避 sleep 掐掉, 测试不用真等。"""
    slept = []
    monkeypatch.setattr(_mod().time, "sleep", lambda s: slept.append(s))
    return slept


def test_429_retries_then_succeeds(monkeypatch, fast_sleep):
    """429 之后应该退避重试并最终成功 —— 之前是直接放弃。"""
    sess = _FakeSession([
        _FakeResp(429, headers={"Retry-After": "6"}, text="rate limited"),
        _FakeResp(429, headers={"Retry-After": "4"}, text="rate limited"),
        _FakeResp(200, payload={"results": [{"id": "W1"}], "meta": {"count": 7}}),
    ])
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    out = _mod().openalex_search("q", source_ids=["S1"], max_results=5, caller="T")
    assert len(out) == 1 and out[0]["id"] == "W1"
    assert sess.calls == 3
    assert fast_sleep, "429 之后应真的退避"
    assert fast_sleep[0] >= 4, "应按 Retry-After 退避(不是固定值)"


def test_429_exhausted_raises(monkeypatch, fast_sleep):
    sess = _FakeSession([_FakeResp(429, headers={"Retry-After": "1"}, text="rl")] * 8)
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    with pytest.raises(RuntimeError, match="429"):
        _mod().openalex_search("q", source_ids=["S1"], max_results=5, caller="T")
    assert sess.calls == _mod()._RETRY_ON_429 + 1


def test_429_wait_capped(monkeypatch, fast_sleep):
    """Retry-After 特别大时不能傻等。"""
    sess = _FakeSession([
        _FakeResp(429, headers={"Retry-After": "9999"}, text="rl"),
        _FakeResp(200, payload={"results": []}),
    ])
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    _mod().openalex_search("q", source_ids=["S1"], max_results=5, caller="T")
    assert fast_sleep[0] <= _mod()._RETRY_ON_429_CAP


def test_429_without_retry_after_header(monkeypatch, fast_sleep):
    """没有 Retry-After 头时用递增退避, 不能崩。"""
    sess = _FakeSession([
        _FakeResp(429, text="rl"),
        _FakeResp(200, payload={"results": []}),
    ])
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    _mod().openalex_search("q", source_ids=["S1"], max_results=5, caller="T")
    assert sess.calls == 2 and fast_sleep


def test_4xx_other_than_429_still_fails_fast(monkeypatch, fast_sleep):
    """
    400(query 写错)不该重试 —— 重试也没用, 只是浪费时间。
    429 才是"等一会就好"的。
    """
    sess = _FakeSession([_FakeResp(400, text="bad field")])
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    with pytest.raises(RuntimeError, match="4xx"):
        _mod().openalex_search("q", source_ids=["S1"], max_results=5, caller="T")
    assert sess.calls == 1, "非 429 的 4xx 不应重试"
    assert not fast_sleep


def test_5xx_still_retries(monkeypatch, fast_sleep):
    """5xx 的原有重试不能被改坏。"""
    sess = _FakeSession([
        _FakeResp(503, text="unavailable"),
        _FakeResp(200, payload={"results": [{"id": "W"}]}),
    ])
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    out = _mod().openalex_search("q", source_ids=["S1"], max_results=5, caller="T")
    assert sess.calls == 2 and out


# ============================================================
# sort 必须走顶层参数, 不能拼进 filter
# ============================================================

class _CapturingSession(_FakeSession):
    """记录真正发出去的 params。"""

    def __init__(self, responses):
        super().__init__(responses)
        self.sent = []

    def get(self, url, params=None, timeout=None):
        self.sent.append(dict(params or {}))
        return super().get(url, params=params, timeout=timeout)


def test_sort_is_top_level_param_not_filter(monkeypatch, fast_sleep):
    """
    回归: sort 被错拼进 filter 串 -> OpenAlex 回 400 "sort is not a valid field"。

    OpenAlex 收到 filter=sort:cited_by_count:desc 时, 会把 "sort" 当成一个
    filter 字段名去解析, 找不到就 400。所以只要不传默认的 relevance 排序
    (比如"按被引量挑高质量论文"), 召回这一步必失败。
    """
    sess = _CapturingSession([_FakeResp(200, payload={"results": []})])
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    _mod().openalex_search(
        "q", source_ids=["S1"], max_results=5,
        sort="cited_by_count:desc", caller="T",
    )
    p = sess.sent[0]
    assert p.get("sort") == "cited_by_count:desc", "sort 应作为顶层查询参数发出"
    assert "sort" not in p.get("filter", ""), "sort 绝不能出现在 filter 串里"


def test_default_sort_omitted(monkeypatch, fast_sleep):
    """默认 relevance 不必显式发 sort (OpenAlex 默认就是它)。"""
    sess = _CapturingSession([_FakeResp(200, payload={"results": []})])
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    _mod().openalex_search("q", source_ids=["S1"], max_results=5, caller="T")
    assert "sort" not in sess.sent[0]


def test_filter_still_carries_source_and_year(monkeypatch, fast_sleep):
    """修 sort 不能把 source_ids / from_year 的过滤弄丢。"""
    sess = _CapturingSession([_FakeResp(200, payload={"results": []})])
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    _mod().openalex_search(
        "q", source_ids=["S1", "S2"], from_year=2019,
        sort="publication_date:desc", caller="T",
    )
    f = sess.sent[0].get("filter", "")
    assert "primary_location.source.id:S1|S2" in f
    assert "from_publication_date:2019-01-01" in f
    assert "sort" not in f


def test_from_year_uses_supported_filter_syntax(monkeypatch, fast_sleep):
    """
    回归: from_year 原来拼成 publication_year:>=2019, OpenAlex 回 400。

    "Value for param publication_year must be a number" —— 它把 ">=2019"
    整个当年份值解析了。publication_year 只支持单年(=2019), "某年及以后"
    必须走日期型下界 from_publication_date。
    """
    sess = _CapturingSession([_FakeResp(200, payload={"results": []})])
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    _mod().openalex_search("q", from_year=2019, max_results=5, caller="T")
    f = sess.sent[0].get("filter", "")
    assert ">=" not in f, f"OpenAlex filter 不支持 >= 比较符, 实发: {f!r}"
    assert "publication_year:>=" not in f


def test_from_year_zero_adds_no_filter(monkeypatch, fast_sleep):
    """from_year=0 (默认) 表示不过滤, 不该往 filter 里塞东西。"""
    sess = _CapturingSession([_FakeResp(200, payload={"results": []})])
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    _mod().openalex_search("q", from_year=0, max_results=5, caller="T")
    assert "publication_date" not in sess.sent[0].get("filter", "")


def test_unsupported_sort_still_rejected(monkeypatch, fast_sleep):
    """白名单仍要拦, 别让拼错的 sort 静悄悄发出去。"""
    with pytest.raises(ValueError, match="sort"):
        _mod().openalex_search("q", max_results=5, sort="bogus_field:desc", caller="T")


def test_429_never_returns_none(monkeypatch, fast_sleep):
    """
    回归: 429 重试耗尽时**绝不能**隐式返回 None。

    之前 429 分支写在 openalex_search 的共用重试循环里, 循环上界是
    _RETRY_ON_5XX+1=3, 而它想按 _RETRY_ON_429=3 重试(共 4 次)。第三次
    走的是 `continue`, 循环已到尾 -> 直接掉出函数, Python 隐式 return None。
    调用方 `for w in openalex_search(...)` 拿 None 去迭代, 表现就是
    Phase A 召回阶段"卡住不动、也不报错", 排查时完全看不出根因。
    """
    sess = _FakeSession([_FakeResp(429, headers={"Retry-After": "1"}, text="rl")] * 8)
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    try:
        out = _mod().openalex_search("q", source_ids=["S1"], max_results=5, caller="T")
    except RuntimeError:
        return  # 抛错才是正确行为
    pytest.fail(f"429 耗尽后返回了 {out!r}, 应该是 RuntimeError 而不是 None")


def test_429_budget_is_independent_of_5xx(monkeypatch, fast_sleep):
    """
    429 的重试预算不该被 5xx 的循环上界截断 —— 限流等一会就好, 和服务端抽风是两码事。
    """
    sess = _FakeSession([_FakeResp(429, headers={"Retry-After": "1"}, text="rl")] * 8)
    monkeypatch.setattr(_mod(), "_build_session", lambda: sess)
    with pytest.raises(RuntimeError, match="429"):
        _mod().openalex_search("q", source_ids=["S1"], max_results=5, caller="T")
    assert sess.calls == _mod()._RETRY_ON_429 + 1 == 4, (
        f"429 应独立重试 4 次, 实际 {sess.calls} 次"
    )

