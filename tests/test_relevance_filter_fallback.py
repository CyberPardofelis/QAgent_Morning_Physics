"""
test_relevance_filter_fallback.py — 相关性审查的「LLM 整体不可用」保护网 (10-07)

对应 bug:
  `_relevance_filter_pool` 里那个「所有 paper 都是 [fallback]/[batch_failed]
  → LLM 整体失败 → 不应用任何过滤」的保护网, 原来写在 `if strict:` 分支**后面**。
  而唯一调用方 `_pool_audit.py:156` 传的恰恰是 `strict=True` —— 于是整段是死代码,
  在生产里从来没执行过。

  后果 (比静默更贵):
    MiniMax 挂掉 → 每批都失败 → 每篇都打 fallback 标记 → 全被踢光 → strict 返空 →
    调用方重试三轮(每轮同样失败) → 最终 pool=[] → 界面报「换个主题词(优先英文)」。
    归因完全错了: 真因是相关性审查 API 不可用。用户会照着错误建议去改选题。

  注意这跟「主题太偏」是两件不同的事:
    - 全部 fallback/batch_failed → LLM 没给出任何判断 → 不该据此下结论
    - 拿到了判断且全不相关      → 这才是真的主题不匹配, strict 该返空

所以本文件钉两件事:
  1. 保护网提到 strict 之前 (行为断言)
  2. 它没被挪回去 (源码结构断言 —— 行为断言测不出「顺序」这件事)
"""
from __future__ import annotations

import ast
import importlib
import json
import pathlib
import re
import sys

import pytest


def _mod(name: str):
    if name not in sys.modules:
        importlib.import_module(name)
    return sys.modules[name]


RF_MOD = _mod("agent.tools.surveys._pool._relevance_filter")
PA_MOD = _mod("agent.tools.surveys._pool._pool_audit")

_relevance_filter_pool = RF_MOD._relevance_filter_pool


def _rf_src() -> str:
    return pathlib.Path(RF_MOD.__file__).read_text(encoding="utf-8")


def _papers(n: int = 5) -> dict[str, dict]:
    """造 n 篇候选论文 (字段对齐 build_relevance_score_prompt 的最低要求)。"""
    return {
        f"p{i}": {
            "paper_id": f"p{i}",
            "title": f"Paper {i} on quantum error correction",
            "year": "2024",
            "journal": "Nature",
            "core_method": "surface code + decoder",
            "core_contrib": "improves logical error rate by 3x",
            "abstract": "We study quantum error correction.",
        }
        for i in range(1, n + 1)
    }


def _patch_llm(monkeypatch, handler):
    """替换 agent.tools.llm._call_llm。

    注意 `_relevance_filter_pool` 是在**函数体内**做
    `from agent.tools.llm import _call_llm`, 所以 patch 模块属性就够了。
    """
    import agent.tools.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_call_llm", handler)


def _boom(*args, **kwargs):
    raise RuntimeError("API 挂了 (模拟 MiniMax 不可用)")


def _unrated(*args, **kwargs):
    """LLM 返了东西但解析不出来 → 每篇打 [fallback] 标记。"""
    return "这不是 JSON, 模型胡言乱语"


def _prompt_of(args) -> str:
    """_call_llm 的第一个位置参数就是 prompt, 从里面捞回 paper_id 列表。"""
    return args[0] if args else ""


def _verdict(score: int, about: bool):
    """把每一批里的所有 paper 判成给定分数。"""
    def _fn(*args, **kwargs):
        pids = re.findall(r"paper_id=(\S+)", _prompt_of(args))
        rows = [
            {
                "paper_id": pid,
                "score": score,
                "is_about_topic": about,
                "is_adjacent_field": not about,
                "reason": "test verdict",
            }
            for pid in pids
        ]
        return json.dumps(rows, ensure_ascii=False)
    return _fn


# ============================================================
# 保护网: strict 路径下也必须生效
# ============================================================

def test_strict_all_batch_failed_returns_full_pool(monkeypatch):
    """strict=True + 每批都炸 → 原样保留全部候选, 不返空。

    改之前这条返 {} (保护网在 strict 后面, 走不到)。
    """
    _patch_llm(monkeypatch, _boom)
    pool = _papers(5)
    kept = _relevance_filter_pool("quantum error correction", pool, threshold=8, strict=True)

    assert len(kept) == 5, f"LLM 整体挂掉时不该把候选池清空, 实际返 {len(kept)} 篇"


def test_strict_all_unrated_returns_full_pool(monkeypatch):
    """LLM 有返回但全是无法解析的垃圾 → 同样视为「审查不可用」, 放行。"""
    _patch_llm(monkeypatch, _unrated)
    pool = _papers(5)
    kept = _relevance_filter_pool("quantum error correction", pool, threshold=8, strict=True)

    assert len(kept) == 5


def test_non_strict_all_batch_failed_does_not_raise(monkeypatch):
    """non-strict 路径原本会抛 ValueError 让调用方 fallback。

    现在保护网先命中, 直接返全量 —— 语义上对调用方等价(都保住池子),
    但不再走异常路径。钉住「不再抛」这个事实。
    """
    _patch_llm(monkeypatch, _boom)
    pool = _papers(5)
    kept = _relevance_filter_pool("quantum error correction", pool, threshold=8, strict=False)

    assert len(kept) == 5


def test_kept_papers_carry_scores_even_when_net_fires(monkeypatch):
    """放行不等于不打分 —— 下游审计/排序还读 _relevance_score。"""
    _patch_llm(monkeypatch, _boom)
    pool = _papers(3)
    kept = _relevance_filter_pool("quantum error correction", pool, threshold=8, strict=True)

    for pid, p in kept.items():
        assert "_relevance_score" in p, f"{pid} 缺 _relevance_score"
        assert p["_relevance_reason"].startswith("[batch_failed]")


# ============================================================
# 保护网不能误伤: 真的判过就照常过滤
# ============================================================

def test_strict_all_irrelevant_still_returns_empty(monkeypatch):
    """全部拿到真实判断且全不相关 → strict 该返空, 保护网不许插手。

    这是本次改动的核心语义边界: 「没判」≠「判为不相关」。
    """
    _patch_llm(monkeypatch, _verdict(score=3, about=False))
    pool = _papers(5)
    kept = _relevance_filter_pool("quantum error correction", pool, threshold=8, strict=True)

    assert kept == {}, "拿到真实判断且全不相关时必须返空 (保护网不该放行)"


def test_strict_all_relevant_still_filters_by_threshold(monkeypatch):
    _patch_llm(monkeypatch, _verdict(score=9, about=True))
    pool = _papers(5)
    kept = _relevance_filter_pool("quantum error correction", pool, threshold=8, strict=True)

    assert len(kept) == 5


def test_mixed_real_and_fallback_does_not_fire_net(monkeypatch):
    """一篇真判相关 + 四篇 LLM 没评 → 保护网不许把四篇也放进来。

    `all(...)` 的语义就是这个: 有一篇真判过就不算「整体不可用」。
    """
    def _partial(*args, **kwargs):
        return json.dumps([{
            "paper_id": "p1",
            "score": 9,
            "is_about_topic": True,
            "is_adjacent_field": False,
            "reason": "真判: 核心就是 QEC",
        }], ensure_ascii=False)

    _patch_llm(monkeypatch, _partial)
    pool = _papers(5)
    kept = _relevance_filter_pool("quantum error correction", pool, threshold=8, strict=True)

    assert set(kept) == {"p1"}, f"只该保留真判相关的那篇, 实际 {sorted(kept)}"


def test_all_relevant_but_one_batch_failed_keeps_net_off(monkeypatch):
    """4 篇真判相关 + 1 篇 batch_failed → 不触发保护网, 只保留那 4 篇。

    反向验证「保护网条件写得太松」这种错法(比如写成 any 而不是 all)。
    """
    def _mixed(*args, **kwargs):
        pids = re.findall(r"paper_id=(\S+)", _prompt_of(args))
        rows = [
            {"paper_id": pid, "score": 9, "is_about_topic": True,
             "is_adjacent_field": False, "reason": "真判相关"}
            for pid in pids if pid != "p5"
        ]
        if not rows:  # p5 单独一批时炸掉
            raise RuntimeError("这一批挂了")
        return json.dumps(rows, ensure_ascii=False)

    _patch_llm(monkeypatch, _mixed)
    pool = _papers(5)
    kept = _relevance_filter_pool("quantum error correction", pool, threshold=8, strict=True)

    assert set(kept) == {"p1", "p2", "p3", "p4"}, f"实际 {sorted(kept)}"


# ============================================================
# 调用方链路: strict 返空让兜底永不触发, 那条 except 是死的
# ============================================================

def test_audit_with_retry_keeps_pool_when_llm_down(monkeypatch):
    """端到端: 相关性审查 API 挂掉时, `_audit_pool_with_retry` 不该交出空池。

    min_pool=3 < 5 篇, 所以 round 0 就 meets_target, 不会走 retry 分支
    (retry 会真去调 OpenAlex/APS 网络)。
    """
    _patch_llm(monkeypatch, _boom)
    pool = _papers(5)
    res = PA_MOD._audit_pool_with_retry(
        "quantum error correction",
        pool,
        min_pool=3,
        max_pool=30,
        relevance_threshold=8,
    )

    assert len(res["kept_dict"]) == 5, f"API 挂掉不该让候选池变空, 实际 {len(res['kept_dict'])} 篇"
    assert res["summary"]["pool_meets_target"] is True


def test_audit_with_retry_returns_empty_when_genuinely_irrelevant(monkeypatch):
    """对照组: 真判不相关时, audit 仍然如实交出空池 + pool_meets_target=False。

    钉住「保护网不是无脑放行」。
    """
    _patch_llm(monkeypatch, _verdict(score=2, about=False))
    pool = _papers(5)
    res = PA_MOD._audit_pool_with_retry(
        "quantum error correction",
        pool,
        min_pool=3,
        max_pool=30,
        relevance_threshold=8,
        max_retries=0,   # 别去走 retry(会发网络请求)
    )

    assert res["kept_dict"] == {}
    assert res["summary"]["pool_meets_target"] is False


# ============================================================
# 源码结构断言: 光看行为测不出「顺序被挪回去」
# ============================================================

def _func_node() -> ast.FunctionDef:
    tree = ast.parse(_rf_src())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_relevance_filter_pool":
            return node
    raise AssertionError("_relevance_filter_pool 没找到")


def test_fallback_net_sits_before_strict_branch():
    """保护网必须在 `if strict:` 之前, 而且**不能被 strict 门控**。

    这正是本次 bug 的形状。行为断言测不出这件事: 保护网挪回 strict 后面时,
    strict=True 的全部-fallback 场景恰好仍然返 {}(和保护网不存在时一样),
    测试会继续绿。
    """
    fn = _func_node()
    strict_lineno = None
    net_lineno = None
    net_test = None

    for node in ast.walk(fn):
        # `if strict:` 这一层
        if (isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "strict"):
            strict_lineno = node.lineno
        # 保护网: `if pool_dict and all(...)` —— 条件里带 startswith 的元组
        if isinstance(node, ast.If) and ast.dump(node.test).find("startswith") != -1:
            net_lineno = node.lineno
            net_test = node.test

    assert strict_lineno is not None, "没找到 `if strict:` 分支"
    assert net_lineno is not None, "没找到 fallback/batch_failed 保护网"
    assert net_lineno < strict_lineno, (
        f"保护网在第 {net_lineno} 行, strict 分支在第 {strict_lineno} 行 —— "
        "保护网被挪到 strict 后面了, 唯一调用方传 strict=True, 整段又会变死代码"
    )

    # 位置对了还不够: 条件里不能出现 `not strict` 这种门控。
    # 生产路径**就是** strict, 保护网一旦被 strict 挡掉, 等于挪回死代码。
    gated = [n.id for n in ast.walk(net_test)
             if isinstance(n, ast.Name) and n.id == "strict"]
    assert not gated, (
        f"保护网条件里出现了 {gated} —— 唯一调用方传 strict=True, "
        "带这个门控等于把整段保护网变回死代码"
    )


def test_comment_documents_why_net_is_hoisted():
    """注释得说清楚为什么在这个位置 —— 不然下一个人会「顺手挪回去」。

    这条断言有点脆, 但正是它在防「重构时对齐代码风格」把 bug 改回来。
    """
    src = _rf_src()
    assert "死代码" in src, "缺少「这段曾经是死代码」的说明"
    assert "_pool_audit" in src, "缺少对调用方传 strict=True 的引用"


# ============================================================
# 告警得说对归因方向 (「宁可吵, 不可静默」的反面: 不能用错话替代安静)
# ============================================================

def test_warning_points_at_api_not_at_topic(monkeypatch, capsys):
    """告警必须明确告诉用户「这是 API 故障, 不是主题不匹配」。

    原来的 bug 不只是静默 —— 它让界面报「换个主题词(优先英文)」, 用户会
    照着改选题。放行候选池只解决一半, 剩下那一半是别让人往错的方向查。
    """
    _patch_llm(monkeypatch, _boom)
    _relevance_filter_pool("quantum error correction", _papers(5), threshold=8, strict=True)

    out = capsys.readouterr().out
    assert "相关性审查整体不可用" in out, "告警没说清是审查本身不可用"
    assert "不是主题不匹配" in out, "告警没把归因跟「主题不匹配」区分开"
    assert "API 连通性" in out, "告警没给排查方向(API/额度), 用户会跑去换选题"


def test_caller_actually_uses_strict_true():
    """调用方必须传 strict=True —— 保护网是专为 strict 路径准备的。

    有人把这里改成 strict=False 时, 本文件所有行为断言照样全绿
    (non-strict 本来就走 rescue 路径), 但生产路径整个变了。
    """
    src = pathlib.Path(PA_MOD.__file__).read_text(encoding="utf-8")
    calls = re.findall(r"_relevance_filter_pool\([^)]*strict\s*=\s*(\w+)", src, flags=re.DOTALL)
    assert calls, "没在 _pool_audit.py 里找到 _relevance_filter_pool 的调用"
    assert all(c == "True" for c in calls), (
        f"调用方的 strict 参数变成了 {calls} —— 保护网是专为 strict 路径准备的, "
        "改成 strict=False 会让整段重新变成死代码"
    )
