"""
tests/test_survey_resume.py — 崩溃续跑 (10-02)

v60e 跑了 76 分钟 / 151 次调用死在主题归纳那一步, 全轮作废。
这里锁的是续跑的两个硬要求: **signature 对不上就不许续**(否则旧参数下的中间
产物接到新参数上, 输出看着正常但结论是错的), 以及**续跑必须喊出声**(宁可吵,
不可静默接上旧数据)。
"""
from __future__ import annotations

import importlib
import re

import pytest

# `from agent.tools.surveys import survey_generate` 拿到的是**函数**
# (包 __init__ 把同名函数 re-export 出来盖住了模块), 必须用 importlib。
SG_MOD = importlib.import_module("agent.tools.surveys.survey_generate")


def _org():
    """**调用时**取当前的 _organize 模块。

    10-03 踩过同一个坑(见 test_survey_paradigm.py 的同名 helper): 同套件里
    test_openalex_rate_limit / test_tool_registry 会 reload 模块, 于是本文件
    顶层 import 到的模块对象变成了 reload 前的旧对象 —— 而 `_phase_claims`
    内部是**函数级** `from ._organize import ...`, 取的是 reload 后的新对象。
    补丁打在旧对象上完全无效, 于是测试真调了一次 LLM。
    所以补丁一律现取 sys.modules。

    用 import_module 而不是直接 sys.modules[]: 单跑本文件时 _organize 可能还
    没被加载(它只被 _phase_claims 懒加载), 直接下标会 KeyError。
    """
    import sys
    name = "agent.tools.surveys._organize"
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)


# ============================================================
# workspace checkpoint / claims / narratives
# ============================================================

@pytest.fixture
def ws(tmp_path):
    from agent.infra.survey_workspace import SurveyWorkspace
    return SurveyWorkspace("testhash", topic="t", root=tmp_path)


def test_checkpoint_roundtrip(ws):
    assert ws.load_checkpoint() is None
    ws.save_checkpoint("facts", signature="abc", n_facts=42)
    ck = ws.load_checkpoint()
    assert ck["stage"] == "facts"
    assert ck["signature"] == "abc"
    assert ck["n_facts"] == 42
    assert ck["updated_at"]
    ws.clear_checkpoint()
    assert ws.load_checkpoint() is None


def test_checkpoint_rejects_corrupt_file(ws):
    ws.path.mkdir(parents=True, exist_ok=True)
    (ws.path / "_checkpoint.json").write_text("{不是 json", encoding="utf-8")
    assert ws.load_checkpoint() is None, "坏文件不该让流水线崩掉"


def test_claims_and_narratives_roundtrip(ws):
    # 关键区分: **没有文件**返回 None, **有文件但空**也返回 None
    # —— 续跑据此判断"能不能接着走", 分不清就会拿空数据接着跑。
    assert ws.load_claims() is None
    assert ws.load_narratives() is None

    ws.save_claims([{"claim_id": "c1", "statement": "s"}])
    assert [c["claim_id"] for c in ws.load_claims()] == ["c1"]

    ws.save_narratives({"主题A": {"text": "正文", "used_fact_ids": ["f1"]}})
    nar = ws.load_narratives()
    assert nar["主题A"]["text"] == "正文"
    assert nar["主题A"]["used_fact_ids"] == ["f1"]


# ============================================================
# _resolve_resume: signature 门禁
# ============================================================

_STAGES = ("pool", "facts", "themes", "claims", "narratives", "done")


def _sig_for(ws, topic_hash="H", max_pool=50, target_chars=0):
    """算出该参数组合的真实签名(走一次空跑)。"""
    _st, info = SG_MOD._resolve_resume(ws, topic_hash, max_pool, target_chars,
                                       False, _STAGES)
    return info["signature"]


def test_resume_from_checkpoint_returns_that_stage(ws):
    ws.save_checkpoint("themes", signature=_sig_for(ws))
    stage, info = SG_MOD._resolve_resume(ws, "H", 50, 0, True, _STAGES)
    assert stage == "themes"
    assert info["signature"] == _sig_for(ws)
    assert any("♻️" in ln for ln in info["lines"]), "续跑必须喊出声, 绝不静默"


def test_resume_rejects_mismatched_signature(ws):
    """
    改了 max_pool / target_chars 就不能续 —— 否则旧参数下的中间产物接到新参数上,
    那种错最难查: 输出看着完全正常, 结论却是错的。
    """
    ws.save_checkpoint("claims", signature=_sig_for(ws, max_pool=50))
    stage, info = SG_MOD._resolve_resume(ws, "H", 999, 0, True, _STAGES)
    assert stage == "pool", "signature 对不上必须从头跑"
    assert ws.load_checkpoint() is None, "失效的 checkpoint 要清掉"
    assert any("参数对不上" in ln for ln in info["lines"])


def test_resume_disabled_ignores_checkpoint(ws):
    ws.save_checkpoint("claims", signature=_sig_for(ws))
    stage, _info = SG_MOD._resolve_resume(ws, "H", 50, 0, False, _STAGES)
    assert stage == "pool", "resume=False 必须能强制从头跑"


def test_resume_rejects_unknown_stage(ws):
    ws.save_checkpoint("garbage-stage", signature=_sig_for(ws))
    stage, _ = SG_MOD._resolve_resume(ws, "H", 50, 0, True, _STAGES)
    assert stage == "pool"
    assert ws.load_checkpoint() is None


def test_resume_signature_depends_on_max_pool_and_target_chars():
    """签名必须真的把这两个参数算进去, 否则"改参数自动失效"是假的。"""
    from agent.infra.survey_workspace import SurveyWorkspace
    import tempfile
    from pathlib import Path

    sigs = set()
    with tempfile.TemporaryDirectory() as td:
        for mp, tc in ((50, 0), (60, 0), (50, 6000)):
            w = SurveyWorkspace("H", topic="t", root=Path(td))
            _st, info = SG_MOD._resolve_resume(w, "H", mp, tc, True, _STAGES)
            sigs.add(info["signature"])
    assert len(sigs) == 3, "max_pool / target_chars 任一变化都必须改签名"


# ============================================================
# 续跑接线的回归护栏
# ============================================================

def test_organize_phase_wires_resume_throughout():
    import inspect as _i
    src = (_i.getsource(SG_MOD._run_organize_phase)
           + _i.getsource(SG_MOD._phase_facts)
           + _i.getsource(SG_MOD._phase_claims)
           + _i.getsource(SG_MOD._phase_write))
    for must in ("_resolve_resume", "load_pool", "load_facts",
                 "load_thematic_map", "load_claims"):
        assert must in src, f"续跑必须接上 {must}"
    # 跑完要清 checkpoint, 否则下次同参数重跑会接着上一轮的 narratives 走
    assert "clear_checkpoint" in src


# ============================================================
# 阶段梯自洽性: 每一级都真的会被写, 也真的会被读
# ============================================================
# 10-04 补。背景: 10-03 上线时 _STAGES 有 6 级, 但
#   - "done"  从来没被 save_checkpoint 写过  -> 读端是死代码
#   - "narratives" 只在写端出现, 三处读端一处都没接 -> 续跑会重跑最贵的主题地图
#   - 命题层的 save 嵌在 fallback 分支里      -> 主路线从不落盘
# 而上面那条测试三条都抓不到: 它只 assert 字符串在源码里。
# 下面这几条是**行为断言**, 会在任何一个回归发生时立刻红。

def test_every_stage_in_ladder_is_readable_and_writable():
    """每一级阶段都必须有 save_checkpoint 写端, 也必须有 load_* 读端。"""
    import inspect as _i
    written = set(re.findall(r'save_checkpoint\(\s*"([a-z_]+)"',
                             _i.getsource(SG_MOD)))
    # done 写在收尾处, 其余各阶段写在对应 _phase_* 里
    for fn in (SG_MOD._phase_facts, SG_MOD._phase_claims, SG_MOD._phase_write):
        written |= set(re.findall(r'save_checkpoint\(\s*"([a-z_]+)"', _i.getsource(fn)))
    written.add("pool")          # _run_organize_phase 里 pool 阶段那一处
    missing = set(_STAGES) - written
    assert not missing, f"阶段梯里这些级别从来没有被写入过: {sorted(missing)}"


@pytest.mark.parametrize("stage", ["claims", "narratives", "done"])
def test_resume_point_actually_reuses_downstream_artifacts(stage):
    """from_stage 落在 claims/narratives/done 时, 下游产物必须真的被读回来。"""
    import inspect as _i
    claims_src = _i.getsource(SG_MOD._phase_claims)
    write_src = _i.getsource(SG_MOD._phase_write)
    org_src = _i.getsource(SG_MOD._run_organize_phase)

    # 命题层: claims / narratives / done 都要触发 load_claims
    assert 'from_stage in ("claims", "narratives", "done")' in claims_src, \
        "命题层的复用条件必须包含 narratives, 否则成文阶段续跑会重跑命题合并"
    # 成文层: narratives / done 都要触发 load_narratives
    assert 'from_stage in ("narratives", "done")' in write_src, \
        "成文层的复用条件必须包含 narratives, 否则逐主题落盘的收益拿不到"
    # 主题地图层: 也必须包含 narratives
    assert 'from_stage in ("themes", "claims", "narratives", "done")' in org_src, \
        "主题地图的复用条件必须包含 narratives, 否则续跑会重跑最贵的一步"


def test_prompt_version_is_part_of_resume_signature(monkeypatch):
    """改 prompt/逻辑版本必须让旧 checkpoint 自动作废。

    10-04 补。10-03 把命题层从 subject 合并换成范式层时签名没变, 于是旧产物
    会被当成"同参数的合法产物"接上 —— 输出看着完全正常, 结论却是旧逻辑的。
    """
    import tempfile
    from pathlib import Path
    from agent.infra.survey_workspace import SurveyWorkspace

    def _sig_with(version):
        monkeypatch.setattr(SG_MOD, "PROMPT_VERSION", version)
        with tempfile.TemporaryDirectory() as td:
            w = SurveyWorkspace("H", topic="t", root=Path(td))
            _st, info = SG_MOD._resolve_resume(w, "H", 50, 0, True, _STAGES)
            return info["signature"]

    assert _sig_with("2026-10-04") != _sig_with("2026-10-05"), \
        "PROMPT_VERSION 变了签名必须跟着变"


# ============================================================
# _phase_claims 行为测试 (不是源码 grep)
# ============================================================

def _drain(gen):
    """跑完一个生成器, 收集它 yield 的所有文本 + return 值。"""
    chunks = []
    while True:
        try:
            chunks.append(next(gen))
        except StopIteration as e:
            return "".join(chunks), e.value


def test_claims_resume_does_not_call_llm_at_all(monkeypatch):
    """续跑命题层时**一次 LLM 都不许调**。

    10-04 补, 直接对着 10-03 的两个 bug:
      (a) try 块缩进在 if/else 之外 -> 续跑也重跑范式层(白花钱)
      (b) import 写在 else 里 -> 续跑时 UnboundLocalError -> 静默退到劣质旧路线
    当时的写法下, 这个测试会得到 "1 次调用 + 0 条命题"。
    """
    from agent.infra.survey_workspace import SurveyWorkspace
    import tempfile
    from pathlib import Path

    calls = []

    def _boom_paradigm(*a, **kw):
        calls.append("paradigm")
        return [{"claim_id": "q1", "statement": "x", "fact_ids": ["f1"],
                 "paper_ids": ["p1"], "theme_name": "T", "n_facts": 1}]

    def _boom_theme(*a, **kw):
        calls.append("theme")
        return []

    monkeypatch.setattr(_org(), "condense_facts_by_paradigm",
                        _boom_paradigm, raising=False)
    monkeypatch.setattr(_org(), "condense_facts_by_theme",
                        _boom_theme, raising=False)

    with tempfile.TemporaryDirectory() as td:
        w = SurveyWorkspace("H", topic="t", root=Path(td))
        saved = [{"claim_id": "q1", "statement": "缓存的命题", "fact_ids": ["f1"],
                  "paper_ids": ["p1"], "theme_name": "T", "n_facts": 1}]
        w.save_claims(saved)

        out, ret = _drain(SG_MOD._phase_claims(w, [{"fact_id": "f1"}],
                                               {"themes": []}, "claims", "SIG"))

    assert calls == [], f"续跑不该调 LLM, 实际调了: {calls}"
    assert ret[0] == saved, "续跑必须原样返回 checkpoint 里的命题"
    assert "♻️" in out and "沿用 checkpoint" in out, "续跑必须喊出声"


def test_paradigm_success_is_persisted(monkeypatch):
    """范式层成功(主路线)也必须落盘 + 打 checkpoint。

    10-04 补。原来 save_claims 嵌在 `if not claims:` 里, 等于只有范式层失败
    走兜底时才写。实测: 10-03 15:09 走范式的 workspace 没有 _claims.json。
    """
    from agent.infra.survey_workspace import SurveyWorkspace
    import tempfile
    from pathlib import Path

    made = [{"claim_id": "q1", "statement": "范式命题", "fact_ids": ["f1"],
             "paper_ids": ["p1"], "theme_name": "T", "n_facts": 1}]
    monkeypatch.setattr(_org(), "condense_facts_by_paradigm",
                        lambda *a, **kw: made, raising=False)
    monkeypatch.setattr(_org(), "condense_facts_by_theme",
                        lambda *a, **kw: [], raising=False)

    with tempfile.TemporaryDirectory() as td:
        w = SurveyWorkspace("H", topic="t", root=Path(td))
        out, ret = _drain(SG_MOD._phase_claims(w, [{"fact_id": "f1"}],
                                               {"themes": []}, "pool", "SIG"))
        claims, _stage = ret
        reloaded = w.load_claims()
        ck = w.load_checkpoint()

    assert claims == made
    assert reloaded == made, "范式层成功也必须写 _claims.json"
    assert ck and ck.get("stage") == "claims", "范式层成功也必须打 claims checkpoint"
    assert ck.get("signature") == "SIG"


def test_paradigm_failure_falls_back_to_theme_merge(monkeypatch):
    """范式层抛异常时退主题合并, 且两条路都失败时给出警告而不是静默。"""
    from agent.infra.survey_workspace import SurveyWorkspace
    import tempfile
    from pathlib import Path

    def _raise(*a, **kw):
        raise RuntimeError("模型没吐出 JSON")

    theme_made = [{"claim_id": "q1", "statement": "兜底命题", "fact_ids": ["f1"],
                   "paper_ids": ["p1"], "theme_name": "T", "n_facts": 1}]
    monkeypatch.setattr(_org(), "condense_facts_by_paradigm",
                        _raise, raising=False)
    monkeypatch.setattr(_org(), "condense_facts_by_theme",
                        lambda *a, **kw: theme_made, raising=False)

    with tempfile.TemporaryDirectory() as td:
        w = SurveyWorkspace("H", topic="t", root=Path(td))
        out, ret = _drain(SG_MOD._phase_claims(w, [{"fact_id": "f1"}],
                                               {"themes": []}, "pool", "SIG"))
        claims, _stage = ret
        ck = w.load_checkpoint()

    assert claims == theme_made, "范式层失败必须退主题合并而不是返回空"
    assert ck and ck.get("stage") == "claims", "兜底成功也要落盘"

    # 两条路都断 -> 明确警告 + 不写 checkpoint
    monkeypatch.setattr(_org(), "condense_facts_by_paradigm",
                        _raise, raising=False)
    monkeypatch.setattr(_org(), "condense_facts_by_theme",
                        lambda *a, **kw: [], raising=False)
    with tempfile.TemporaryDirectory() as td:
        w = SurveyWorkspace("H", topic="t", root=Path(td))
        out2, ret2 = _drain(SG_MOD._phase_claims(w, [{"fact_id": "f1"}],
                                                {"themes": []}, "pool", "SIG"))
        claims2, _ = ret2
        ck2 = w.load_checkpoint()
    assert claims2 == []
    assert "退回逐 fact 直写" in out2, "两条路都断时必须在输出里说明"
    assert ck2 is None, "没有命题就不该打 claims checkpoint"


# ============================================================
# 真实链路测试: 真的跑 condense_facts_by_paradigm, 只把 LLM 换成回放
# ============================================================
# 10-04 补。上面那些测试把整个 condense_facts_by_paradigm monkeypatch 掉了, 于是
# _parse / _salvage_paradigms / _seed / assign 这些**生产里真会炸的函数**一次都
# 没被跑到。而 10-03 实测: 7 个主题里 2 个的范式调用是"结构写对了但在 max_tokens
# 处被截断" —— 这条路径必须有测试, 否则下次改 prompt 把它改坏了没人知道。
#
# 下面这几条走的是**真实的** condense_facts_by_paradigm -> build_paradigms ->
# _parse -> _salvage_paradigms -> assign 全链路, 只替换两个边界:
#   - agent.tools.llm._call_llm  -> 回放固定文本(生产里就是拿这种文本喂进去的)
#   - _paradigm.encode           -> 换成无依赖的确定性向量(真实实现要 spawn
#                                   bge-m3 子进程, 测试里不能真跑)
# fact / theme_map 的字段形状照抄 data/surveys/working/*/_facts.jsonl 与
# _thematic_map.json 的真实产物。

def _sysmod(name: str):
    """调用时取模块(同 _org(), 10-03 踩过 reload 的坑)。

    import_module 而不是直接 sys.modules[]: 有些模块(如 _paradigm)是**懒加载**的
    —— 只在 condense_facts_by_paradigm 内部第一次被用到时才进 sys.modules。
    这里主动 import 一次, 保证拿到的是**同一个对象**(函数内 import 也会命中
    sys.modules 缓存, 不会另造一份)。
    """
    import sys
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)


def _fake_encode(texts):
    """确定性的 bag-of-chars 向量, 代替 bge-m3。

    方向大致对就行: 共享字符多的文本相似度高, 足以让"同一技术路线的 fact 聚到
    一起"这种结构性行为被测到, 又不引入子进程和模型下载。
    """
    import math

    import numpy as np
    out = []
    for t in texts:
        v = [0.0] * 64
        for ch in (t or "")[:400]:
            v[ord(ch) % 64] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return np.array(out)


def _realistic_facts():
    """字段形状照抄真实 _facts.jsonl(含 fact_id = {paper_id}::{idx})。"""
    def f(paper_id, i, subject, claim):
        return {
            "subject": subject,
            "fact_type": "result",
            "claim": claim,
            "support_quote": claim[:60],
            "paper_id": paper_id,
            "fact_id": f"{paper_id}::{i}",
            "local_fact_id": f"c{i}",
            "_grounded": True,
        }
    return [
        f("2501.10012", 0, "矩阵乘积态",
          "矩阵乘积态(MPS)沿一维链用 rank-3 张量表示多体波函数, 正则形式保证在面积律下标度良好"),
        f("quant-ph/0003004", 0, "正则形式",
          "正则形式(canonical form)消除了 MPS 分解的冗余度, 给出唯一的规范表示"),
        f("2407.17706", 0, "投影纠缠对",
          "投影纠缠对表示(PEPS)把矩阵乘积态推广到二维格点, 但精确收缩代价随格距指数增长"),
        f("2212.02933", 0, "MERA",
          "MERA 用树状层次结构显式编码多尺度纠缠, 变分优化可收缩到低维"),
    ]


def _realistic_theme_map():
    """字段形状照抄真实 _thematic_map.json。"""
    return {
        "themes": [{
            "name": "张量网络的表示与纠缠结构",
            "statement": "有限纠缠态可由低维张量网络有效参数化。",
            "fact_ids": ["2501.10012::0", "quant-ph/0003004::0",
                         "2407.17706::0", "2212.02933::0"],
            "relations": [],
        }],
        "narrative_arc": ["张量网络的表示与纠缠结构"],
        "gaps": [],
        "warnings": [],
    }


# 生产实测: 第一个对象写完了, 第二个在 statement 中间被 max_tokens 切断
_TRUNCATED_PARADIGM_JSON = (
    '{"paradigms": ['
    '{"name": "矩阵乘积态表示", "statement": "矩阵乘积态(MPS)沿一维链用一组 rank-3 '
    '张量表示多体波函数, 在面积律束缚下以多项式标度精确表示一维系统的纠缠结构。", '
    '"keywords": ["矩阵乘积态", "MPS", "正则形式", "一维链"]}, '
    '{"name": "投影纠缠对表示", "statement": "投影纠缠对表示(PEPS)把矩阵乘积态推广到二维格'
)

_FULL_PARADIGM_JSON = (
    '{"paradigms": ['
    '{"name": "矩阵乘积态表示", "statement": "MPS 用 rank-3 张量沿一维链表示多体波函数。", '
    '"keywords": ["矩阵乘积态", "MPS", "正则形式", "一维链"]}, '
    '{"name": "投影纠缠对表示", "statement": "PEPS 是 MPS 在二维格点上的推广。", '
    '"keywords": ["投影纠缠对", "PEPS", "二维格点"]}, '
    '{"name": "多尺度纠缠重正化", "statement": "MERA 用树状层次编码多尺度纠缠。", '
    '"keywords": ["MERA", "多尺度", "树状"]}'
    ']}'
)


def _run_real_paradigm(monkeypatch, tmp_path, response):
    """跑真实的 _phase_claims(内部走真实范式层), LLM 回放 `response`。"""
    from agent.infra.survey_workspace import SurveyWorkspace

    def _fake_call_llm(prompt, **kw):
        return response

    monkeypatch.setattr(_sysmod("agent.tools.llm"), "_call_llm",
                        _fake_call_llm, raising=False)
    monkeypatch.setattr(_sysmod("agent.tools.surveys._paradigm"), "encode",
                        _fake_encode, raising=False)

    w = SurveyWorkspace("H", topic="张量网络态", root=tmp_path)
    out, ret = _drain(SG_MOD._phase_claims(w, _realistic_facts(),
                                           _realistic_theme_map(), "pool", "SIG"))
    return w, out, ret


def test_real_paradigm_layer_survives_truncated_llm_output(monkeypatch, tmp_path):
    """**生产里最常见的失败模式**: LLM 把范式 JSON 写对了但在 max_tokens 处被截断。

    10-03 实测 7 个主题里 2 个这样挂掉, 整块丢掉等于该主题白跑(而且会退回旧
    路线, 暴露出"同一篇论文连引 8 次"的老毛病)。这里用**真实的** _salvage_paradigms
    验证抢救能生效。

    ⚠ 这条断言的写法很要紧。第一版写成 `assert any("矩阵乘积态" in statement)`,
    结果是**假绿**: 范式层全灭时主题合并兜底照样能产出命题, 而 fact 原文里就有
    "矩阵乘积态"这几个字, 于是断言照样通过 —— 等于什么都没锁。
    现在改成锁**范式层的签名**: condense_facts_by_paradigm 造的命题 statement 是
    f"{范式名}: {范式 statement[:120]}"(见 _organize.py), 以范式名开头; 主题
    合并兜底造的命题 statement 来自 fact 原文, 不会长这样。
    """
    w, out, ret = _run_real_paradigm(monkeypatch, tmp_path, _TRUNCATED_PARADIGM_JSON)
    claims, _stage = ret

    assert claims, "截断输出应该被抢救出至少一个范式并生成命题, 实得 0 条"
    # 范式层的签名: statement 以"范式名: "开头
    paradigm_claims = [c for c in claims if c["statement"].startswith("矩阵乘积态表示:")]
    assert paradigm_claims, (
        "抢救没生效 —— 这些命题不是范式层产的:\n"
        + "\n".join(f"    {c['statement'][:60]}" for c in claims)
    )
    # 走了范式层就不该出现兜底告警(兜底了说明抢救失败)
    assert "范式层没产出" not in out, f"不该退化, 实得输出: {out!r}"
    ck = w.load_checkpoint()
    assert ck and ck.get("stage") == "claims", "抢救成功也要落盘, 否则续跑接不上"
    for c in paradigm_claims:
        assert c["claim_id"].startswith("q"), f"claim_id 形态不对: {c['claim_id']!r}"
        assert c["fact_ids"] and c["paper_ids"], f"命题必须带 fact 和 paper: {c}"


def test_real_paradigm_layer_with_full_output(monkeypatch, tmp_path):
    """完整输出时, 真实链路应该把 fact 归到多个范式, 且每个命题都带范式签名。"""
    w, out, ret = _run_real_paradigm(monkeypatch, tmp_path, _FULL_PARADIGM_JSON)
    claims, _stage = ret

    assert claims, "完整输出必须产出命题"
    # 范式层的签名(statement 以"范式名: "开头) —— 用来区分真的走了范式层
    paradigm_claims = [c for c in claims if ": " in c["statement"]
                       and c["statement"].split(":")[0] in
                       ("矩阵乘积态表示", "投影纠缠对表示", "多尺度纠缠重正化")]
    assert paradigm_claims, (
        "这些命题不是范式层产的:\n"
        + "\n".join(f"    {c['statement'][:60]}" for c in claims))
    covered = {fid for c in claims for fid in c["fact_ids"]}
    assert covered, "命题必须带上 fact 引用"
    n_cited = {p for c in claims for p in c["paper_ids"]}
    assert len(n_cited) >= 2, \
        f"范式层的作用就是让一句话挂多篇, 实得只挂了 {sorted(n_cited)}"
    assert "压缩比" in out, "覆盖率/压缩比的汇报是用户判断质量的主要依据, 不能丢"


def test_real_paradigm_layer_degrades_loudly_on_garbage(monkeypatch, tmp_path):
    """LLM 只吐 think 一个字都没有时, 必须**明说**走了兜底。

    10-04 补。原先范式层失败只有一行 logger.warning, 用户的流式输出里看不到 ——
    而这一层直接决定引文密度(实测 12.0/句 vs 旧路线 0.94/句), 静默退回等于
    "产物看着完全正常但质量掉了一档"。这正是模块开头"宁可吵, 不可静默"要防的。

    兜底本身是**设计内**的行为(范式失败 -> 主题合并), 不是错误; 这里断言的是
    "兜底发生了"这件事必须被说出来, 而且压缩比 1.00 会把退化摆在明面上。
    """
    w, out, ret = _run_real_paradigm(
        monkeypatch, tmp_path, "<think>让我想想这个问题</think> 嗯, 稍等我再想想。")
    claims, _stage = ret

    assert "范式层没产出" in out, f"范式层失败必须在输出里明说, 实得: {out!r}"
    assert "退回主题合并" in out, f"必须说清走了哪条兜底, 实得: {out!r}"
    # 兜底成功后压缩比会摊在明面上: 1.00 = 一条没合并 = 退化可见
    assert "压缩比" in out, "覆盖率/压缩比的汇报是用户判断质量的主要依据, 不能丢"
    assert "1.00:1" in out, f"退化时压缩比应显示 1.00 让用户看出来, 实得: {out!r}"
    assert w.load_checkpoint() and w.load_checkpoint()["stage"] == "claims", \
        "兜底成功也要落盘"



def test_workspace_has_all_resume_pieces():
    from agent.infra.survey_workspace import SurveyWorkspace
    for m in ("save_checkpoint", "load_checkpoint", "clear_checkpoint",
              "save_claims", "load_claims", "save_narratives", "load_narratives"):
        assert hasattr(SurveyWorkspace, m), f"缺 {m}"


def test_run_actually_uses_run_ordered_in_the_three_hot_loops():
    """
    护栏: 三段最贵的循环必须走 run_ordered。
    这三段占了实测全流程 LLM 等待时间的 59%, 忘了并行就白改。
    """
    import inspect as _i
    from agent.tools.surveys._actions import _extract_facts as ef
    from agent.tools.surveys import _organize as org

    assert "run_ordered" in _i.getsource(ef.action_extract_facts)
    assert "run_ordered" in _i.getsource(org.condense_facts_into_claims)
    assert "run_ordered" in _i.getsource(org.write_theme_narrative)
    assert "run_ordered" in _i.getsource(org.write_theme_narrative_from_claims)


def test_workers_never_touch_shared_state_in_extract():
    """
    护栏: _extract_one 必须是纯函数 —— 它只发请求和解析,
    state/workspace 的写入一律留在主线程的归并循环里。

    切的是 _extract_one 的**函数体**(从它的定义到归并循环起点),
    不是"文件里 _extract_one 之后的所有内容"。
    """
    import inspect as _i
    from agent.tools.surveys._actions import _extract_facts as ef
    src = _i.getsource(ef.action_extract_facts)
    body = src[src.index("def _extract_one"):src.index("from agent.prompts_survey import clean_fact")]
    for forbidden in ("state[", "ws.append_fact", "_friendly_error"):
        assert forbidden not in body, f"worker 里不能碰 {forbidden}"


# ============================================================
# 端到端: 并发跑与串行跑的产物必须**逐字节相同**
# ============================================================
# 这是整个并发改动唯一真正重要的验证。前面的测试都只证明 run_ordered
# 自己保序; 这里证明接上真实流水线后, claim_id 编号 / 命题顺序 / 段落顺序
# 在并发下与串行完全一致 —— 因为 claim_id 决定正文引用编号,
# 差一位就是整篇综述的引用全部漂移。

def _facts_for_condense(n_subjects=6, per_subject=5):
    out = []
    for s in range(n_subjects):
        for j in range(per_subject):
            out.append({
                "fact_id": f"p{s % 3}::{s}_{j}",
                "paper_id": f"p{s % 3}",
                "subject": f"主题{s}",
                "claim": f"关于主题{s}的第 {j} 条事实",
                "n_facts": 1,
            })
    return out


def test_config_max_workers_default():
    from agent.tools.config import _resolve_survey_max_workers
    import os
    old = os.environ.pop("QAGENT_SURVEY_MAX_WORKERS", None)
    try:
        assert _resolve_survey_max_workers() == 4
        os.environ["QAGENT_SURVEY_MAX_WORKERS"] = "1"
        assert _resolve_survey_max_workers() == 1
        os.environ["QAGENT_SURVEY_MAX_WORKERS"] = "999"
        assert _resolve_survey_max_workers() == 4, "越界回落到默认"
        os.environ["QAGENT_SURVEY_MAX_WORKERS"] = "abc"
        assert _resolve_survey_max_workers() == 4, "非法值回落到默认"
    finally:
        os.environ.pop("QAGENT_SURVEY_MAX_WORKERS", None)
        if old is not None:
            os.environ["QAGENT_SURVEY_MAX_WORKERS"] = old
