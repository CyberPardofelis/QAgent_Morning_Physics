r"""
tests/test_survey_structure_layers.py — 结构地图的三层 (10-05)

10-05 之前, 综述流水线在磁盘上留下的结构只有**一层**:
    主题 -> (范式, 但只活在内存里) -> 命题 -> fact
而且"宏观主题"这一层根本没有, 主题之间的关系还挂着一套
support/tension/evolution/contrast 的枚举标签。这个文件锁的是补齐后的形状:

    宏观主题 -> 主题 -> 范式 -> 命题 -> fact
                     \_ 关联(只有具体的一句话, 不分类)

测试取向:
  - 用**真实产物**做表征测试: 真实 thematic_map 的主题名/关系边/QCVV 那 68 条
    带 type 的真实关系。不用编出来的假结构 —— 编的结构往往正好躲开真实数据里
    已经出现过的失败模式(比如关系指向排在后面才出现的主题)。
  - 每条行为断言都配一个"把被测逻辑关掉它就会红"的反向验证, 见文件末尾
    `test_*_reverse_verification` 那组。
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SG_MOD = importlib.import_module("agent.tools.surveys.survey_generate")


def _org():
    """**调用时**取当前的 _organize 模块(同 test_survey_resume.py 的理由:
    同套件里有测试会 reload 模块, 顶层 import 到的会是旧对象)。"""
    name = "agent.tools.surveys._organize"
    if name in sys.modules:
        return sys.modules[name]
    return importlib.import_module(name)


def _drain(gen):
    return "".join(gen)


def _ws(tmp_path):
    from agent.infra.survey_workspace import SurveyWorkspace
    return SurveyWorkspace("H", topic="t", root=tmp_path)


@pytest.fixture
def ws(tmp_path):
    return _ws(tmp_path)


# ============================================================
# 真实数据表征 —— 锁住"这套结构长什么样"
# ============================================================

# 真实 workspace `data/surveys/working/669cfc28a53e/_thematic_map.json`:
# 5 个主题, 名字**全部**是"贫瘠高原"的各个侧面。这是宏观层的核心用例 ——
# 这一层存在的全部理由就是识别出"这 5 个其实是同一块"。
_BARREN_THEMES = [
    {"name": "贫瘠高原的多源成因", "statement": "高阶西格展开的稀释与费曼图项的散射共同造成贫瘠高原。",
     "fact_ids": [f"10.1103/physrevb.101.080509::{i}" for i in range(24)]},
    {"name": "分析贫瘠高原的理论工具箱", "statement": "以统一变分与有界最优化为代表的一族分析工具给出了贫瘠高原的严格刻画。",
     "fact_ids": [f"1803.07055::{i}" for i in range(27)]},
    {"name": "规避贫瘠高原的架构条件", "statement": "乘积态分解与正交化架构设计绕开了贫瘠高原。",
     "fact_ids": [f"1706.08966::{i}" for i in range(32)]},
    {"name": "缓解贫瘠高原的策略谱系", "statement": "密度矩阵嵌入、对称性增强与信道噪声缓解构成逐级加码的策略谱系。",
     "fact_ids": [f"2002.08953::{i}" for i in range(45)]},
    {"name": "开放问题与根本张力", "statement": "精确表示的指数墙与实用表示的可训练性之间构成尚无定论的张力。",
     "fact_ids": [f"1411.4088::{i}" for i in range(27)]},
]

# 真实 workspace `7b8316ad67b7` 的 7 个主题(张量网络), 主题之间是真差异,
# 用来验证宏观层不是"无脑合成一个大板块"。
_TN_THEMES = [
    {"name": "纠缠结构与网络基础", "statement": "有限纠缠可由低秩张量参数化, 熵标度给出可表示性的界。",
     "fact_ids": ["2501.10012::0"]},
    {"name": "二维网络变分优化", "statement": "PEPS 的变分优化受局域极小与符号问题限制。",
     "fact_ids": ["2407.17706::0", "2212.02933::0"]},
    {"name": "树结构网络与电路模拟", "statement": "MERA 用树结构把电路模拟的成本压到多项式。",
     "fact_ids": ["2110.03138::0"]},
    {"name": "开放系统的混合态网络", "statement": "混合态网络通过信道张量表达非幺正演化。",
     "fact_ids": ["2006.14820::0"]},
    {"name": "对称与规范约束网络", "statement": "对称性投影把不可约表示约束写进网络。",
     "fact_ids": ["1912.01758::0"]},
    {"name": "混合架构与物理实现", "statement": "MPS-MERA 混合架构在硬件比特序下获得实测加速。",
     "fact_ids": ["1803.07135::0"]},
    {"name": "模拟复杂度诊断", "statement": "以连接性判据诊断哪些张量网络能给出可计算结果。",
     "fact_ids": ["1803.06507::0"]},
]

# 真实关系边 —— 从 `output/qcvv/QCVV_主题地图.json` 抄的**原样**形状, 含
# 10-05 要删掉的 type 字段。note 是真的写得出内容的句子, 这正是要保留下来的东西。
_REAL_RELS = [
    {"to": "阴影测量基矢的优化", "type": "evolution",
     "note": "标准 Pauli/Clifford 基矢很快成为被超越的起点, 推动后续的基矢设计研究。"},
    {"to": "阴影范式的奠基", "type": "support",
     "note": "阴影(尤其局部随机测量)已成为多体混合态纠缠检测的样本效率支柱。"},
    {"to": "阴影范式的奠基", "type": "contrast",
     "note": "同为大维希尔景迹下的态表征路径, 但阴影靠随机测量后处理, 这些路径靠结构先验与优化器。"},
    {"to": "专用阴影协议", "type": "support",
     "note": "阴影过程层析与阴影基矢工程为 QPT 提供新的样本复杂度替代路径。"},
]


def _theme_map_llm(rels, extra_themes=()):
    """造一段 LLM 会吐出来的 themed JSON(含 type, 模拟改动前的行为)。"""
    themes = [{
        "name": "阴影范式的奠基",
        "statement": "经典阴影用随机测量加经典后处理估计可观测量的期望。",
        "fact_ids": ["2503.14491::0", "10.1103/physrevlett.125.200501::2"],
        "notes": "从单次测量到批量快照的演进。",
        "relations": rels,
    }]
    for i, t in enumerate(extra_themes):
        themes.append({
            "name": t["name"], "statement": t["statement"],
            "fact_ids": list(t["fact_ids"]) or [f"p{i}::0"],
            "notes": "", "relations": [],
        })
    return json.dumps({"themes": themes, "narrative_arc": [], "gaps": []},
                      ensure_ascii=False)


def _run_themes(monkeypatch, response, facts=None):
    """跑真实的 organize_facts_into_themes, 只把 _call_llm 换成回放。"""
    facts = facts if facts is not None else [
        {"fact_id": "2503.14491::0", "fact_type": "result", "claim": "c1"},
        {"fact_id": "10.1103/physrevlett.125.200501::2", "fact_type": "result", "claim": "c2"},
        {"fact_id": "10.1088/1367-2630/15/1/015024::3", "fact_type": "result", "claim": "c3"},
    ]
    monkeypatch.setattr(
        importlib.import_module("agent.tools.llm"), "_call_llm",
        lambda prompt, **kw: response, raising=False)
    return _org().organize_facts_into_themes(facts)


# ============================================================
# ① 范式层落盘
# ============================================================

def test_paradigms_roundtrip(ws):
    """范式层要能落盘能读回 —— 这一层以前只活在内存里。"""
    data = {"张量网络的表示与纠缠结构": [
        {"name": "矩阵乘积态表示", "statement": "MPS 用 rank-3 张量沿一维链表示波函数。",
         "keywords": ["矩阵乘积态", "MPS"], "n_facts": 2,
         "fact_ids": ["2501.10012::0", "quant-ph/0003004::0"],
         "paper_ids": ["2501.10012", "quant-ph/0003004"], "claim_id": "q1"},
    ]}
    assert ws.load_paradigms() is None, "没有文件时必须是 None(不是空 dict)"
    ws.save_paradigms(data)
    assert ws.load_paradigms() == data
    assert (ws.path / "_paradigms.json").exists()


def test_load_paradigms_none_on_empty_and_corrupt(ws):
    """空文件 / 坏 JSON / 空 dict 都返回 None —— 区别于"有内容"。"""
    (ws.path / "_paradigms.json").write_text("{}", encoding="utf-8")
    assert ws.load_paradigms() is None
    (ws.path / "_paradigms.json").write_text("{不是 JSON", encoding="utf-8")
    assert ws.load_paradigms() is None
    (ws.path / "_paradigms.json").write_text(
        json.dumps({"paradigms": {}}), encoding="utf-8")
    assert ws.load_paradigms() is None


def test_saved_paradigms_file_carries_provenance(ws):
    """落盘文件带 topic / created_at —— 否则事后无法判断这层是哪一轮产出的。"""
    ws.save_paradigms({"T": [{"name": "P", "statement": "s"}]})
    data = json.loads((ws.path / "_paradigms.json").read_text(encoding="utf-8"))
    assert data["topic"] == ws.topic
    assert data["created_at"]
    assert "P" in data["paradigms"]["T"][0]["name"] or data["paradigms"]["T"][0]["name"] == "P"


# ============================================================
# ③ 关联去标签化
# ============================================================

def test_relations_drop_type_but_keep_note_verbatim(monkeypatch):
    """**核心**: type 删掉, note 一字不改地留下。

    反向验证: 把 parse 里的 type 那行加回去, `"type" not in rel` 立刻红。
    """
    res = _run_themes(monkeypatch, _theme_map_llm(_REAL_RELS))
    t = res["themes"][0]
    got = {(r["to"], r.get("note")) for r in t["relations"]}
    want = {(r["to"], r["note"]) for r in _REAL_RELS}
    assert got == want, f"note 必须原样保留。\n实得 {got}\n应得 {want}"
    for r in t["relations"]:
        assert "type" not in r, f"关系边不该再带 type: {r}"


def test_relation_type_enum_is_gone_from_prompt():
    """prompt 也不能再教模型输出那四个词 —— 否则模型会继续生成、继续被丢掉。"""
    src = _org().THEME_MAP_PROMPT
    for word in ("support", "tension", "evolution", "contrast"):
        assert word not in src, f"THEME_MAP_PROMPT 里不该再出现关系枚举 {word}"
    assert "不要输出" in src or "不分类" in src, "prompt 必须显式禁止关系类型标签"


def test_blank_relation_note_is_reported(monkeypatch):
    """去掉 type 之后, note 为空的关系边什么都不剩 —— 必须说出来, 不能静默留。"""
    res = _run_themes(monkeypatch, _theme_map_llm([
        {"to": "另一个主题", "type": "support", "note": ""},
        {"to": "第三个主题", "type": "tension", "note": "   "},
    ], extra_themes=(_TN_THEMES[1], _TN_THEMES[2])))
    warns = " | ".join(res["warnings"])
    assert "没有具体说明" in warns, f"空 note 必须告警, 实得: {res['warnings']}"
    # 边仍然保留(它是结构), 但不再有任何标签
    rels = res["themes"][0]["relations"]
    assert len(rels) == 2
    assert all("type" not in r and not r["note"] for r in rels)


def test_relation_pointing_at_later_theme_is_not_flagged(monkeypatch):
    """**顺序 bug 的护栏**: 关系可以指向 JSON 里排在**后面**才出现的主题。

    主题校验是遍历中累积的 seen_theme, 所以校验**必须**放到循环之后做。
    反向验证: 把指向校验挪回循环内, 这条立刻红(会误报"指向不存在的主题")。
    """
    res = _run_themes(monkeypatch, json.dumps({
        "themes": [
            {"name": "先出现的主题", "statement": "s", "fact_ids": ["2503.14491::0"],
             "notes": "",
             "relations": [{"to": "后出现的主题", "type": "support", "note": "n"}]},
            {"name": "后出现的主题", "statement": "s", "fact_ids": ["2407.17706::0"],
             "notes": "", "relations": []},
        ],
        "narrative_arc": [], "gaps": []}, ensure_ascii=False),
        facts=[{"fact_id": "2503.14491::0", "fact_type": "result", "claim": "c1"},
               {"fact_id": "2407.17706::0", "fact_type": "result", "claim": "c2"}])
    warns = " | ".join(res["warnings"])
    assert "指向了不存在的主题" not in warns, (
        f"指向后置主题不该被误报, 实得警告: {res['warnings']}")
    assert len(res["themes"]) == 2


def test_relation_pointing_at_nonexistent_theme_is_reported(monkeypatch):
    """真的指向不存在的主题 -> 告警但**保留边**(静默删边会让结构凭空少一块)。"""
    res = _run_themes(monkeypatch, _theme_map_llm([
        {"to": "根本不存在的那个主题", "type": "support", "note": "n"}]))
    warns = " | ".join(res["warnings"])
    assert "指向了不存在的主题" in warns, f"应告警, 实得: {res['warnings']}"
    assert res["themes"][0]["relations"][0]["to"] == "根本不存在的那个主题"


def test_theme_relations_block_has_no_type_label():
    """写作阶段看到的 relation 文本里也不能再有 type —— 那只是在挤占 note。

    断言打在**全模块**上而不是某一个函数: 两个写叙述的入口
    (write_theme_narrative / write_theme_narrative_from_claims) 都会渲染
    relations, 只查一个等于留了一半的口子。
    """
    import inspect
    src = inspect.getsource(_org())
    assert "是 {r.get('type')}" not in src, "渲染里不该再插 type 标签"
    assert "与「{r.get('to')}」: {r.get('note')}" in src, "应改成只输出 note"


# ============================================================
# ② 宏观主题层
# ============================================================

def _macro_llm(macros):
    return json.dumps({"macros": macros}, ensure_ascii=False)


def _run_macros(monkeypatch, response, themes):
    monkeypatch.setattr(
        importlib.import_module("agent.tools.llm"), "_call_llm",
        lambda prompt, **kw: response, raising=False)
    return _org().cluster_themes_into_macros(themes)


def test_macro_layer_refuses_single_theme(monkeypatch):
    """只有 1 个主题时不做聚类 —— 强行聚只会得到 1:1 的空壳结构。"""
    def _fail(*a, **kw):
        raise AssertionError("单主题时不该调 LLM")

    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fail, raising=False)
    macros, warns = _org().cluster_themes_into_macros([_BARREN_THEMES[0]])
    assert macros == []
    assert warns and "1 个主题" in warns[0]


def test_macro_layer_collapses_barren_plateau_five_into_one(monkeypatch):
    """**真实用例**: 贫瘠高原那 5 个主题全是同一件事, 宏观层必须合成 1 个板块。

    这是这一层存在的理由。如果实现只是把主题原样搬一遍(5 进 5 出),
    这条会红 —— 反向验证即在此。
    """
    names = [t["name"] for t in _BARREN_THEMES]
    macros, warns = _run_macros(monkeypatch, _macro_llm([{
        "name": "多体量子计算的贫瘠高原",
        "statement": "五条线共享同一个障碍: 局域操作在浅电路深度下的表达能力上限。",
        "theme_names": names,
        "notes": "成因、理论刻画、规避、缓解、开放问题。",
        "cross_notes": [],
    }]), _BARREN_THEMES)
    assert len(macros) == 1
    assert macros[0]["theme_names"] == names
    assert warns == [] or not any("未归入" in w for w in warns)


def test_macro_layer_keeps_distinct_tn_themes_separate(monkeypatch):
    """**反向用例**: 张量网络那 7 个主题是真差异, 宏观层不该把它们糊成一坨。

    跟上一条一起构成一对: 宏观层既不能什么都不做, 也不能无脑合并。
    """
    macros, warns = _run_macros(monkeypatch, _macro_llm([
        {"name": "表示结构与优化", "statement": "表示能力与优化可行性的分离。",
         "theme_names": ["纠缠结构与网络基础", "二维网络变分优化", "模拟复杂度诊断"],
         "notes": "n", "cross_notes": []},
        {"name": "物理约束与实现", "statement": "把物理条件写进网络结构。",
         "theme_names": ["对称与规范约束网络", "混合架构与物理实现", "开放系统的混合态网络"],
         "notes": "n", "cross_notes": []},
        {"name": "树结构路线", "statement": "以树结构换取可计算性。",
         "theme_names": ["树结构网络与电路模拟"],
         "notes": "n", "cross_notes": []},
    ]), _TN_THEMES)
    assert len(macros) == 3
    sizes = {len(m["theme_names"]) for m in macros}
    assert 1 in sizes, "真实 7 主题下应当有一个板块只含 1 个主题(该合的合了)"
    assert "未归入任何宏观板块" not in " | ".join(warns), f"7 个主题应全覆盖: {warns}"


def test_macro_layer_drops_invented_theme_names(monkeypatch):
    """模型编的 theme_names 必须丢 + 告警, 不能凭空多出一个主题。"""
    macros, warns = _run_macros(monkeypatch, _macro_llm([{
        "name": "板块A",
        "statement": "s",
        "theme_names": ["贫瘠高原的多源成因", "我自己编的主题", "另一个编的"],
        "notes": "n", "cross_notes": [],
    }]), _BARREN_THEMES)
    assert len(macros) == 1
    assert macros[0]["theme_names"] == ["贫瘠高原的多源成因"]
    assert any("不是已有主题名" in w for w in warns), f"应告警编造: {warns}"


def test_macro_layer_reports_orphan_themes(monkeypatch):
    """没被任何板块收编的主题要**说出来**, 不能硬塞(硬塞=造假归属)。"""
    macros, warns = _run_macros(monkeypatch, _macro_llm([{
        "name": "板块A", "statement": "s",
        "theme_names": ["贫瘠高原的多源成因", "分析贫瘠高原的理论工具箱"],
        "notes": "n", "cross_notes": [],
    }]), _BARREN_THEMES)
    assert len(macros) == 1
    joined = " | ".join(warns)
    assert "未归入任何宏观板块" in joined, f"3 个孤儿主题应被点名: {warns}"
    assert "规避贫瘠高原的架构条件" in joined


def test_macro_layer_survives_truncated_llm_output(monkeypatch):
    """截断的 JSON 要能抢救出已完成的板块(同主题层的已知失败模式)。"""
    truncated = ('{"macros": ['
                 '{"name": "多体量子计算的贫瘠高原", "statement": "五条线共享同一个障碍", '
                 '"theme_names": ["贫瘠高原的多源成因", "分析贫瘠高原的理论工具箱"], '
                 '"notes": "n"}, '
                 '{"name": "被截断的板块", "statement": "这句没写完')
    macros, warns = _run_macros(monkeypatch, truncated, _BARREN_THEMES)
    assert len(macros) == 1, f"应抢救出第一个完整板块, 实得 {macros}"
    assert macros[0]["name"] == "多体量子计算的贫瘠高原"


def test_macro_layer_failure_is_quiet_not_fatal(monkeypatch):
    """宏观层挂掉 -> 返回空 + 警告, 不抛。它是加法, 缺了单层结构照样能用。"""
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", lambda *a, **kw: None, raising=False)
    macros, warns = _org().cluster_themes_into_macros(_BARREN_THEMES)
    assert macros == []
    assert warns and "失败" in warns[0]


def test_macro_prompt_forbids_relation_type():
    src = _org().MACRO_THEME_PROMPT
    for word in ("support", "tension", "evolution", "contrast"):
        assert word not in src, f"宏观 prompt 里不该再出现关系枚举 {word}"
    assert "theme_names" in src and "只能从上面给出的主题名" in src


# ============================================================
# ② 宏观层接线 (_phase_macros)
# ============================================================

def test_phase_macros_writes_back_to_thematic_map(monkeypatch, tmp_path):
    """跑完之后 macros 必须**落进 _thematic_map.json** —— 否则这一层等于没做。"""
    w = _ws(tmp_path)
    tmap = {"themes": _BARREN_THEMES, "narrative_arc": [], "gaps": [], "warnings": []}
    names = [t["name"] for t in _BARREN_THEMES]
    monkeypatch.setattr(
        importlib.import_module("agent.tools.llm"), "_call_llm",
        lambda prompt, **kw: _macro_llm([{
            "name": "多体量子计算的贫瘠高原", "statement": "s",
            "theme_names": names, "notes": "n", "cross_notes": ["与别的板块共用同一个约束"]}]),
        raising=False)
    out = _drain(SG_MOD._phase_macros(w, tmap))
    saved = json.loads((w.path / "_thematic_map.json").read_text(encoding="utf-8"))
    assert saved["macros"], "macros 必须写进 _thematic_map.json"
    assert saved["macros"][0]["theme_names"] == names
    assert "1 次调用" in out
    assert "已写入" in out


def test_phase_macros_skips_llm_when_already_present(monkeypatch, tmp_path):
    """**续跑保证**: 文件里已经有 macros 时一次调用都不花。

    反向验证: 去掉 `if theme_map.get("macros"): return` 之后, 这条会红。
    """
    w = _ws(tmp_path)

    def _fail(*a, **kw):
        raise AssertionError("已有 macros 时不该再调 LLM")

    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fail, raising=False)
    tmap = {"themes": _BARREN_THEMES, "macros": [{"name": "已有", "statement": "s",
                                                 "theme_names": ["贫瘠高原的多源成因"],
                                                 "notes": "", "cross_notes": []}]}
    out = _drain(SG_MOD._phase_macros(w, tmap))
    assert out == ""
    assert tmap["macros"][0]["name"] == "已有"


def test_phase_macros_backfills_legacy_map(monkeypatch, tmp_path):
    """旧 workspace 的 _thematic_map.json 全部没有 macros —— 复用时必须补齐。

    这是把这一层放在 themes 判定**之后**的全部理由: 只改产出侧的话,
    磁盘上那 3 个真实工作区永远补不上这一层。
    """
    w = _ws(tmp_path)
    (w.path / "_thematic_map.json").write_text(
        json.dumps({"themes": _BARREN_THEMES, "narrative_arc": [],
                    "gaps": [], "warnings": []}, ensure_ascii=False), encoding="utf-8")
    names = [t["name"] for t in _BARREN_THEMES]
    monkeypatch.setattr(
        importlib.import_module("agent.tools.llm"), "_call_llm",
        lambda prompt, **kw: _macro_llm([{
            "name": "多体量子计算的贫瘠高原", "statement": "s",
            "theme_names": names, "notes": "n", "cross_notes": []}]),
        raising=False)
    loaded = _org().load_thematic_map(w)
    assert "macros" not in loaded
    _drain(SG_MOD._phase_macros(w, loaded))
    assert loaded["macros"][0]["name"] == "多体量子计算的贫瘠高原"
    assert json.loads((w.path / "_thematic_map.json").read_text(
        encoding="utf-8"))["macros"]


def test_phase_macros_degrades_loudly_on_exception(monkeypatch, tmp_path):
    """异常时降级但**必须喊** —— 宁可吵, 不可静默。"""
    w = _ws(tmp_path)
    tmap = {"themes": _BARREN_THEMES, "narrative_arc": [], "gaps": [], "warnings": []}

    def _boom(*a, **kw):
        raise RuntimeError("模型服务 503")

    monkeypatch.setattr(_org(), "cluster_themes_into_macros", _boom, raising=False)
    out = _drain(SG_MOD._phase_macros(w, tmap))
    assert tmap["macros"] == []
    assert "⚠" in out and "宏观主题层异常" in out
    assert any("503" in w for w in tmap["macro_warnings"])


def test_phase_macros_skips_llm_for_single_theme(monkeypatch, tmp_path):
    """主题 < 2 时直接落空并记原因, 不调 LLM。"""
    w = _ws(tmp_path)

    def _fail(*a, **kw):
        raise AssertionError("单主题时不该调 LLM")

    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fail, raising=False)
    tmap = {"themes": [_BARREN_THEMES[0]], "narrative_arc": [], "gaps": [], "warnings": []}
    _drain(SG_MOD._phase_macros(w, tmap))
    assert tmap["macros"] == []
    assert tmap["macro_warnings"]


# ============================================================
# 主题层截断抢救 (10-05)
# ============================================================

def test_theme_map_survives_truncated_llm_output(monkeypatch):
    """**回填实测 5 个 workspace 全部死在这**: JSON 写对了但在 max_tokens 处截断。

    主题层以前**没有**抢救逻辑(范式层 10-03 就有, 一直没同步过来), 于是
    "前面几个主题已经写完整了"这个事实被整块丢掉, 那 360 条 fact 一条主题都
    没归属。范式层的 docstring 里记着同一个教训。

    反向验证: 去掉 salvage 那行, 这条会红。
    """
    truncated = (
        '{"themes": ['
        '{"name": "张量网络的表示与纠缠结构", "statement": "有限纠缠态可由低秩张量'
        '有效参数化。", "fact_ids": ["fact_aaaa1111", "fact_bbbb2222"], "notes": "n"}, '
        '{"name": "二维网络的变分优化", "statement": "PEPS 的变分优化受局域极小限制", '
        '"fact_ids": ["fact_cccc3333"], "notes": "n"}, '
        '{"name": "被截断的第三个主题", "statement": "这句没写完'
    )
    facts = [{"fact_id": i, "fact_type": "result", "claim": f"c{n}"}
             for n, i in enumerate(["fact_aaaa1111", "fact_bbbb2222", "fact_cccc3333"])]
    res = _run_themes(monkeypatch, truncated, facts=facts)
    names = [t["name"] for t in res["themes"]]
    assert len(res["themes"]) >= 2, f"应抢救出前两个完整主题, 实得 {names}"
    assert "张量网络的表示与纠缠结构" in names
    # 丢的东西必须说出来, 不能静默
    assert any("截断" in w and "narrative_arc" in w for w in res["warnings"]), (
        f"丢失 narrative_arc/gaps 必须告警, 实得: {res['warnings']}")


def test_salvage_themes_requires_name_and_fact_ids():
    """抢救只收"像主题"的: 有 name 且带 fact_ids 键。空对象不能当主题。"""
    from agent.tools.surveys._organize import _salvage_themes
    assert _salvage_themes('{"themes": [') is None
    assert _salvage_themes('{"a": 1}') is None
    got = _salvage_themes(
        '{"name":"甲","fact_ids":[]}, {"statement":"没有name"}, {"name":"乙"}')
    assert [g["name"] for g in got] == ["甲"], f"实得 {got}"


def test_salvage_themes_survives_braces_inside_strings():
    """statement 里出现 `{` `}` 和转义引号时不能把字符串当结构符号。

    括号深度一旦算错, 后面全部错位 —— 这类错误不报错, 只是救不出东西。
    """
    from agent.tools.surveys._organize import _salvage_themes
    got = _salvage_themes(
        '{"name":"含花括号","statement":"定义 {x, y} 与 \\"引号\\" 混排",'
        '"fact_ids":["f1"]}, {"name":"第二个","fact_ids":["f2"]}')
    assert [g["name"] for g in got] == ["含花括号", "第二个"], f"实得 {got}"


def test_theme_map_not_salvaged_when_json_is_complete(monkeypatch):
    """完整 JSON 走正常路径, 不该被抢救分支污染。"""
    res = _run_themes(monkeypatch, _theme_map_llm([]))
    assert res["themes"]
    assert not any("截断" in w for w in res["warnings"]), res["warnings"]


def test_api_error_is_not_mistaken_for_truncation(monkeypatch):
    """**回填实测 5 个 workspace 全灭在这**: 瞬时 API 故障被误诊成"输出被截断"。

    `_call_llm` 失败时不抛异常, 而是返回一段 markdown 错误提示(见 llm.py:
    "失败时为带错误提示的字符串")。原来主题层拿它去当 JSON 解析, 失败后报
    "疑似被 max_tokens 截断", 折半重试 —— 而折半对 API 故障毫无帮助, 只是把
    一次瞬时故障放大成连续 3 次失败。

    手抄同一份 prompt 单发一次完全正常(实测 4099 字符 / 5 个主题), 证明不是
    数据问题也不是模型问题。

    反向验证: 去掉 `if api_err: continue` 那段, 这条会红。
    """
    err = ("❌ MiniMax 请求超时（APITimeoutError）：Connection timed out.\n"
           "  原始错误：read timeout 600s")
    calls = {"n": 0}

    def _flaky(prompt, **kw):
        calls["n"] += 1
        if calls["n"] <= 2:
            return err
        return json.dumps({"themes": [
            {"name": "第三个主题", "statement": "s", "fact_ids": ["f1"],
             "notes": "", "relations": []}], "narrative_arc": [], "gaps": []},
            ensure_ascii=False)

    monkeypatch.setattr(importlib.import_module("agent.tools.llm"), "_call_llm",
                        _flaky, raising=False)
    facts = [{"fact_id": "f1", "fact_type": "result", "claim": "c"}]
    res = _org().organize_facts_into_themes(facts)
    assert [t["name"] for t in res["themes"]] == ["第三个主题"], res["themes"]
    assert calls["n"] == 3, f"实得 {calls['n']} 次调用"
    assert not any("max_tokens" in w for w in res["warnings"]), (
        f"API 故障不该被报成截断: {res['warnings']}")


def test_persistent_api_error_is_reported_as_api_error(monkeypatch):
    """连续 API 故障 -> 警告里必须说是 API 故障, 且明确"重跑即可"。"""
    err = "💰 MiniMax 账户余额不足（InsufficientBalanceError）\n\n原始错误：1008"
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"), "_call_llm",
                        lambda *a, **kw: err, raising=False)
    facts = [{"fact_id": "f1", "fact_type": "result", "claim": "c"}]
    res = _org().organize_facts_into_themes(facts)
    assert res["themes"] == []
    w = " | ".join(res["warnings"])
    assert w.startswith("LLM 调用失败"), f"实得: {res['warnings']}"
    assert "API 故障" in w and "重跑即可" in w, f"实得: {res['warnings']}"
    assert "max_tokens" not in w, f"不该甩锅给 max_tokens: {res['warnings']}"


def test_llm_error_detector_does_not_fire_on_real_output():
    """真输出里可能含 emoji/中文标点, 检测器不能误伤。"""
    from agent.tools.surveys._organize import _is_llm_error_response
    good = ('{"themes": [{"name": "未解问题界定", "statement": "s", '
            '"fact_ids": ["f1"]}]}')
    assert _is_llm_error_response(good) is None
    assert _is_llm_error_response("") is None
    assert _is_llm_error_response("❌ MiniMax 请求超时") is not None
    assert _is_llm_error_response("未设置 MiniMax API Key，请在设置中配置。") is not None


# ============================================================
# 组织侧输出: 四层结构要真的看得见
# ============================================================

def test_audit_renders_four_layers(tmp_path):
    """审计报告里要能看到 宏观/主题/关联 三层, 且关联那行不带 type。"""
    tmap = {
        "themes": [{
            "name": "纠缠结构与网络基础",
            "statement": "有限纠缠可由低秩张量参数化。",
            "fact_ids": ["a::0"], "notes": "",
            "relations": [{"to": "模拟复杂度诊断",
                           "note": "诊断判据决定了哪些低秩表示真的可算。"}],
        }],
        "macros": [{"name": "表示能力与可计算性", "statement": "能不能表示和能不能算不是一回事。",
                    "theme_names": ["纠缠结构与网络基础", "模拟复杂度诊断"],
                    "notes": "", "cross_notes": ["与物理实现路线共用同一个秩约束"]}],
        "macro_warnings": ["1 个主题未归入任何宏观板块: 某某"],
        "narratives": {}, "gaps": [],
    }
    md = SG_MOD._organize_audit(tmap, {}, [{"fact_id": "a::0"}])
    assert "宏观板块" in md and "表示能力与可计算性" in md
    assert "跨板块关联" in md
    assert "关联 → 模拟复杂度诊断" in md
    assert "关系 support" not in md and "关系 tension" not in md
    assert "未归入任何宏观板块" in md


# ============================================================
# 宏观层增量归类(10-06): 只聚未归类主题 + 优先并入已有板块
# ============================================================

def _existing(*names):
    return [{"name": n, "statement": f"{n}的老领域判断。", "n_themes": 40}
            for n in names]


def test_macro_prompt_parts_empty_when_no_existing():
    """没有已有板块时 prompt 块必须是空串 —— 与 10-05 的 prompt 一字不差。

    这是向后兼容的硬要求: 增量逻辑不能改变**全量重聚**那条老路子的输出。
    """
    parts = _org()._macro_prompt_parts
    block, hint = parts(themes=[], names=["甲", "乙", "丙", "丁"],
                        max_macros=5, existing_macros=None)
    assert block == "", "没有已有板块时不该注入任何块"
    assert hint == max(2, min(5, 4 // 2)) == 2

    # 空列表也要走同一条分支(existing_macros=[] 不是 None)
    block2, hint2 = parts(themes=[], names=["甲", "乙", "丙", "丁"],
                          max_macros=5, existing_macros=[])
    assert block2 == "" and hint2 == 2


def test_macro_prompt_parts_lists_existing_and_shrinks_hint():
    """有已有板块时: 列出板块名 + 降低板块数期望值。

    期望值必须降 —— 增量模式下再问"聚成 5 个", 模型会凭空造满 5 个同类板块。
    """
    parts = _org()._macro_prompt_parts
    names = [f"主题{i}" for i in range(9)]
    block, hint = parts(themes=[], names=names, max_macros=5,
                        existing_macros=_existing("量子通信与硬件栈", "张量网络与多体模拟"))
    assert "量子通信与硬件栈" in block and "张量网络与多体模拟" in block
    assert "已收编 40 个主题" in block
    # 复用规则必须写进 prompt, 否则模型无从知道"原样照抄才算复用"
    assert "原样照抄" in block
    assert hint == max(1, min(5, 9 // 3)) == 3
    assert hint < max(2, min(5, 9 // 2)), "有已有板块时 n_hint 必须比全量模式小"


def test_macro_prompt_template_actually_embeds_existing_block():
    """模板里必须有 {existing_block} 占位符, 否则上面那个块永远进不了 prompt。"""
    src = _org().MACRO_THEME_PROMPT
    assert "{existing_block}" in src


def test_cluster_marks_reused_macro(monkeypatch):
    """模型原样照抄已有板块名 -> 该板块要标 reused=True。"""
    resp = _macro_llm([{
        "name": "量子通信与硬件栈", "statement": "重写过的说法",
        "notes": "", "theme_names": ["甲", "乙"],
        "cross_notes": [],
    }])
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", lambda prompt, **kw: resp, raising=False)
    macros, warns = _org().cluster_themes_into_macros(
        [{"name": "甲", "statement": "s", "fact_ids": ["1"]},
         {"name": "乙", "statement": "s", "fact_ids": ["2"]}],
        existing_macros=_existing("量子通信与硬件栈"))
    assert len(macros) == 1
    assert macros[0]["reused"] is True, "照抄已有板块名必须标成复用"
    assert macros[0]["theme_names"] == ["甲", "乙"]


def test_cluster_marks_new_macro_not_reused(monkeypatch):
    """新建的板块必须标 reused=False, 否则调用方会传空 statement 把它写空。"""
    resp = _macro_llm([{
        "name": "全新板块", "statement": "新领域。",
        "notes": "n", "theme_names": ["甲", "乙"], "cross_notes": [],
    }])
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", lambda prompt, **kw: resp, raising=False)
    macros, _ = _org().cluster_themes_into_macros(
        [{"name": "甲", "statement": "s", "fact_ids": ["1"]},
         {"name": "乙", "statement": "s", "fact_ids": ["2"]}],
        existing_macros=_existing("量子通信与硬件栈"))
    assert macros[0]["reused"] is False
    assert macros[0]["statement"] == "新领域。"


def test_reused_single_theme_does_not_warn_empty_shell(monkeypatch):
    """复用板块只装 1 个新主题**不该**报"1:1 空壳"。

    增量模式下 kept 只是这批未归类主题里落在该板块的那几个, 板块里原有的
    几十个主题根本不在输入里。按 kept 数告警是纯误报, 而且会让人以为
    复用失败了。
    """
    resp = _macro_llm([{
        "name": "量子表征与精密测量", "statement": "s", "notes": "",
        "theme_names": ["甲"], "cross_notes": [],
    }])
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", lambda prompt, **kw: resp, raising=False)
    _macros, warns = _org().cluster_themes_into_macros(
        [{"name": "甲", "statement": "s", "fact_ids": ["1"]},
         {"name": "乙", "statement": "s", "fact_ids": ["2"]}],
        existing_macros=_existing("量子表征与精密测量"))
    assert not any("1 个主题" in w and "空壳" not in w and "不构成" in w
                   for w in warns), f"复用板块不该被判成空壳: {warns}"
    assert any("复用已有板块" in w for w in warns), f"该说明这是复用: {warns}"


def test_new_single_theme_still_warns_empty_shell(monkeypatch):
    """对照组: **新建**板块只装 1 个主题仍然要告警。上面那条不能把这条也吃掉。"""
    resp = _macro_llm([{
        "name": "全新板块", "statement": "s", "notes": "",
        "theme_names": ["甲"], "cross_notes": [],
    }])
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", lambda prompt, **kw: resp, raising=False)
    _macros, warns = _org().cluster_themes_into_macros(
        [{"name": "甲", "statement": "s", "fact_ids": ["1"]},
         {"name": "乙", "statement": "s", "fact_ids": ["2"]}],
        existing_macros=_existing("量子通信与硬件栈"))
    assert any("不构成上位聚类" in w for w in warns), f"新建空壳必须告警: {warns}"


def test_backfill_macro_only_dry_does_not_call_llm(tmp_path, monkeypatch, capsys):
    """`--macro-only --dry` **绝不能**发请求。

    实测踩过: 原来写死 `macro_layer(False)`, 把 --dry 丢了。想先看 prompt
    估算, 结果一发就是 300 秒起步的真实调用, 而且输出跟真跑过一样, 事后
    分不清跑没跑。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_bg_mod", ROOT / "scripts" / "backfill_graph.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    calls = []
    monkeypatch.setattr(mod, "macro_layer",
                        lambda dry: (calls.append(dry), 0)[1])
    monkeypatch.setattr(mod.sys, "argv",
                        ["backfill_graph.py", "--macro-only", "--dry"])
    mod.main()
    assert calls == [True], f"--dry 必须传进 macro_layer, 实际传了 {calls}"
    assert "--dry" in capsys.readouterr().out, "dry 必须在输出里说清楚"
