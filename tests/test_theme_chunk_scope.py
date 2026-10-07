"""
test_theme_chunk_scope.py — 分块归纳时告诉模型「第 k/N 块」+ 全库真实条数 (10-07)

问题:
`organize_facts_into_themes` 在 fact 超 `THEME_CHUNK_CHARS` 时会切块逐块归纳
(10-05 改, 替掉了「按 subject 均匀抽样 35% 静默丢失」)。但 prompt 的措辞从头
到尾都是「下面是从文献中抽取的原子事实笔记」, 对单块和全量**一模一样** ——
模型于是合理地(但错误地)认为这就是全部语料。

后果:
  1. 它把一块的 subject 分布当全领域分布。分布表是模型归纳时的锚, 锚错位
     整套结构就跟着错位。
  2. 它拿 1/N 的视野对**整个文献**下判断。实测中性原子那次回补 422 条 fact 切
     8 块, 块块主题名复用率 0%, 合并层 32 个名字抢 8 个名额, 第 3~8 块 0% 命中。

修法: prompt 开场加一个【本次范围】块。取景写在语料分布**之前** —— 锚之前
得先说清锚的覆盖范围。

关键约束: 折半重试会真的砍 fact, 所以范围块里的条数必须跟着变, 写死块大小
就是在对模型说假话。
"""
from __future__ import annotations

import importlib
import json
import sys

import pytest


def _mod(name: str):
    if name not in sys.modules:
        importlib.import_module(name)
    return sys.modules[name]


OG = _mod("agent.tools.surveys._organize")


def _facts(n: int, claim_len: int = 20) -> list[dict]:
    return [
        {"fact_id": f"f{i}", "subject": f"对象{i % 5}", "claim": "长" * claim_len,
         "paper_id": "p1", "fact_type": "finding"}
        for i in range(n)
    ]


def _theme_json(facts: list[dict]) -> str:
    return json.dumps({
        "themes": [{
            "name": "主题一", "statement": "一句领域判断",
            "fact_ids": [f["fact_id"] for f in facts], "notes": "线索", "relations": [],
        }],
        "narrative_arc": ["主题一"],
        "gaps": [],
    }, ensure_ascii=False)


def _skeleton_json() -> str:
    return json.dumps({
        "themes": [{
            "name": "骨架任务一", "task": "解什么问题",
            "belongs": "线索A / 线索B", "not_belongs": "别混这个",
        }],
        "structure_note": "骨架关系说明",
    }, ensure_ascii=False)


def _is_skeleton_prompt(p: str) -> bool:
    """阶段一(骨架)和阶段二(块内)的 prompt 长得不一样, 按内容区分。

    10-07 晚加了骨架阶段, 于是分块时**多一次** LLM 调用 —— 之前这组测试
    假设 seen[N] 就是第 N 块的 prompt, 现在第 0 次是骨架。靠内容认比靠
    序号认稳(顺序将来再变一次, 这些测试就会集体误判)。
    """
    return "结构骨架" in p or "极简摘要" in p


def _capture_llm(monkeypatch):
    """把发给 LLM 的 prompt 全存下来, 返回列表。"""
    import agent.tools.llm as llm_mod

    seen: list[str] = []

    def _fake(prompt, *args, **kwargs):
        seen.append(prompt)
        return _skeleton_json() if _is_skeleton_prompt(prompt) else _theme_json(_facts(1))

    monkeypatch.setattr(llm_mod, "_call_llm", _fake)
    return seen


def _chunk_prompts(seen: list[str]) -> list[str]:
    """从所有调用里挑出块内归纳的那几次(排除骨架)。"""
    return [p for p in seen if not _is_skeleton_prompt(p)]


# ============================================================
# _chunk_scope_line 本身
# ============================================================

def test_scope_line_states_chunk_index_and_global_total():
    s = OG._chunk_scope_line(facts_n=60, total_facts=422, chunk_idx=3, n_chunks=8)
    assert "第 3/8 块" in s
    assert "422" in s, "必须给全库真实条数, 模型得知道它只看到了一小部分"
    assert "60" in s, "必须给本块实际条数"


def test_scope_line_says_histogram_is_chunk_local():
    """分布表是锚, 必须说清它只统计了这一块。"""
    s = OG._chunk_scope_line(facts_n=60, total_facts=422, chunk_idx=3, n_chunks=8)
    assert "只统计了这一块" in s


def test_single_chunk_scope_line_says_this_is_everything():
    """单块时反过来要**明确**说这是全部 —— 不说的话模型默认「我看到的就是全部」,
    而这次要靠默认值成立, 下次就悄悄变了。"""
    s = OG._chunk_scope_line(facts_n=120, total_facts=120, chunk_idx=1, n_chunks=1)
    assert "全部语料" in s
    assert "第 1/1 块" not in s, "单块不需要块序号, 写了反而像在掩盖分块"


def test_scope_line_warns_against_chunk_local_gaps():
    """「这块没提到 X」被写成「文献没覆盖 X」是最典型的块级幻觉。"""
    s = OG._chunk_scope_line(facts_n=60, total_facts=422, chunk_idx=3, n_chunks=8)
    assert "不要推测块外还有什么" in s


def test_scope_line_tells_model_not_to_number_theme_names():
    s = OG._chunk_scope_line(facts_n=60, total_facts=422, chunk_idx=3, n_chunks=8)
    assert "序号痕迹" in s


# ============================================================
# 真的进了 prompt (不是只写了个没人调用的函数)
# ============================================================

def test_scope_line_appears_in_actual_prompt(monkeypatch):
    seen = _capture_llm(monkeypatch)
    OG.organize_facts_into_themes(_facts(50))

    assert len(_chunk_prompts(seen)) == 1, "50 条应该走单块"
    assert "【本次范围" in seen[0], "范围块没进真实 prompt"


def test_scope_line_precedes_subject_histogram(monkeypatch):
    """取景必须前置: 分布表是锚, 得先知道锚只覆盖 1/N 再看它。"""
    seen = _capture_llm(monkeypatch)
    OG.organize_facts_into_themes(_facts(50))

    p = seen[0]
    assert p.index("【本次范围") < p.index("【语料分布"), \
        "范围块必须排在语料分布之前 —— 否则模型先把一块的分布当成了全领域分布"


def test_every_chunk_gets_its_own_index(monkeypatch):
    """8 块就得到 8 份不同的范围块, 且每份都写全库总数。"""
    monkeypatch.setattr(OG, "THEME_CHUNK_FACTS", 40)   # 强制切多块
    seen = _capture_llm(monkeypatch)
    facts = _facts(200)
    OG.organize_facts_into_themes(facts)

    prompts = _chunk_prompts(seen)
    n_chunks = len(OG.split_fact_chunks(facts))
    assert len(prompts) == n_chunks > 1, \
        f"应当逐块调用, 实际 {len(prompts)} 次 / {n_chunks} 块"

    for i, prompt in enumerate(prompts, 1):
        assert f"第 {i}/{n_chunks} 块" in prompt, f"第 {i} 块的 prompt 没写自己的序号"
        assert "200" in prompt, f"第 {i} 块的 prompt 没说全库总数"


def test_first_and_last_chunk_report_different_indexes(monkeypatch):
    """钉住「序号不是常量」—— 全都写「第 1 块」等于没做。"""
    monkeypatch.setattr(OG, "THEME_CHUNK_FACTS", 40)
    seen = _capture_llm(monkeypatch)
    OG.organize_facts_into_themes(_facts(200))

    prompts = _chunk_prompts(seen)
    assert "第 1/" in prompts[0]
    assert f"第 {len(prompts)}/" in prompts[-1]
    assert (prompts[0].split("【本次范围")[1][:12]
            != prompts[-1].split("【本次范围")[1][:12])


# ============================================================
# 折半重试: 条数必须跟着变, 写死就是撒谎
# ============================================================

def test_halved_single_chunk_never_claims_it_is_everything():
    """折半重试 + 单块: 说「这是全部语料」是撒谎 —— 模型会以为没看到的那些不存在。

    这条是被 `test_scope_line_follows_actual_count_after_halving` 逼出来的:
    单块分支原本无条件走「这是全部」, 折半后模型只拿到一半, 却被告知全都看到了。
    """
    s = OG._chunk_scope_line(facts_n=50, total_facts=100, chunk_idx=1, n_chunks=1)
    assert "这是全部" not in s, "折半后还声称'这是全部' = 对模型撒谎"
    assert "全部语料共 100 条" in s
    assert "前 50 条" in s
    assert "没有别的批次" not in s


def test_single_chunk_full_count_claims_everything():
    """对照组: 没被折半(facts_n == total)时, 该说还得说。"""
    s = OG._chunk_scope_line(facts_n=100, total_facts=100, chunk_idx=1, n_chunks=1)
    assert "这是全部" in s


def test_scope_line_follows_actual_count_after_halving(monkeypatch):
    """截断重试会 `facts = facts[:n_now]`。范围块里如果还写块大小, 就是对模型
    说假话 —— 模型会以为自己看全了。"""
    monkeypatch.setattr(OG, "THEME_CHUNK_FACTS", 1000)   # 强制单块
    monkeypatch.setattr(OG, "THEME_CHUNK_CHARS", 100000)
    calls: list[str] = []

    def _fake(prompt, *args, **kwargs):
        calls.append(prompt)
        chunk_calls = [p for p in calls if not _is_skeleton_prompt(p)]
        if len(chunk_calls) == 1:
            return "这不是 JSON"          # 触发折半
        return _theme_json(_facts(1))

    import agent.tools.llm as llm_mod
    monkeypatch.setattr(llm_mod, "_call_llm", _fake)

    OG.organize_facts_into_themes(_facts(100))

    chunk_prompts = _chunk_prompts(calls)
    assert len(chunk_prompts) == 2, f"应当折半重试一次, 实际 {len(chunk_prompts)} 次"
    scope2 = chunk_prompts[1].split("【本次范围")[1].split("】")[0]
    assert "50" in scope2, f"折半后范围块应写 50 条(实际看到的条数), 原文: {scope2}"
    assert "这是全部" not in scope2, f"折半后仍声称是全部语料, 原文: {scope2}"


# ============================================================
# 源码结构: 别让范围块被挪到分布之后 / 被摘掉
# ============================================================

def test_prompt_template_has_scope_line_placeholder():
    assert "{scope_line}" in OG.THEME_MAP_PROMPT, \
        "THEME_MAP_PROMPT 里少了 {scope_line} 占位符"


def test_every_format_call_site_passes_scope_line():
    """三处 format 都得传 —— 漏一处就是某条路径上模型看不到范围。"""
    import inspect
    src = inspect.getsource(OG)
    n_format = src.count("THEME_MAP_PROMPT.format(")
    n_scope = src.count("scope_line=_chunk_scope_line(")
    n_skel = src.count("skeleton_block=_render_skeleton(")
    assert n_format == 3, f"预期 3 处 format 调用, 实际 {n_format}"
    assert n_scope == n_format, \
        f"有 format 调用没传 scope_line ({n_scope}/{n_format}) —— 某条路径上模型看不到范围"
    assert n_skel == n_format, \
        f"有 format 调用没传 skeleton_block ({n_skel}/{n_format}) —— " \
        "那条路径上模型看不到全局骨架, 会退回块级命名"
