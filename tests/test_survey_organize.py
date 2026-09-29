"""
tests/test_survey_organize.py — fact 优先的综述组织 (Phase B + Phase C)

9-29 新增。

核心不变量(按重要性):
    1. 模型**只能输出 fact_id**, 永远拿不到 paper_id;
       编造的 fact_id 被丢弃, 不会被换算成假引用。
    2. 引用由 fact_id → paper_id 机械映射生成。
    3. 无 fact 标注的句子被剔除(无出处的断言不发布)。
    4. Phase B 的 fact_id 覆盖情况要如实报告, 不静默。
    5. 允许分析(关系/对照), 禁止凭空补充经验内容 —— 体现在 prompt 上。
"""
from __future__ import annotations

import json

import pytest

import importlib  # noqa: E402

from agent.tools.surveys import _organize as O

# surveys/__init__.py 把 survey_generate 这个**函数**重导出到包命名空间, 会遮蔽同名
# **模块**属性 —— 拿函数去 inspect._run_organize_phase 会 AttributeError。
SG_MOD = importlib.import_module("agent.tools.surveys.survey_generate")


def _fact(i, pid="pA", ftype="result", claim=None):
    return {
        "fact_id": f"{pid}::{i}",
        "paper_id": pid,
        "fact_type": ftype,
        "claim": claim or f"结论 {i}",
        "support_quote": f"quote for {i}",
    }


FACTS = [
    _fact(0, "pA", "method", "提出快照算子把状态重构化为线性反演"),
    _fact(1, "pA", "result", "快照样本复杂度随被估计量规模对数增长"),
    _fact(2, "pB", "method", "最大似然层析在中等规模多体系统给出满意结果"),
    _fact(3, "pB", "limitation", "最大似然在噪声下解不唯一, 需物理正则"),
    _fact(4, "pC", "result", "置换不变表示下 20 比特态可在数分钟内重构"),
    _fact(5, "pC", "conclusion", "结构化先验是突破维度瓶颈的主要途径"),
    _fact(6, "pD", "result", "局域测量即可完成态验证, 拷贝数远少于层析"),
    _fact(7, "pD", "method", "设备无关验证仅需极少量测量即可判定保真度"),
]


# ============================================================
# Phase B: 主题地图
# ============================================================

class _FakeLLM:
    """可编排的 LLM 替身: 按调用次序返回预设响应。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt, **_kw):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


@pytest.fixture
def patched_llm(monkeypatch):
    def _set(responses):
        fake = _FakeLLM(responses)
        import agent.tools.llm as lm
        monkeypatch.setattr(lm, "_call_llm", fake)
        return fake
    return _set


THEME_JSON = json.dumps({
    "themes": [
        {
            "name": "随机测量与快照",
            "statement": "本领域已从完整重建转向只需部分信息的任务导向表征。",
            "fact_ids": ["pA::0", "pA::1"],
            "notes": "快照把重构化为线性反演, 样本复杂度对数增长。",
            "relations": [
                {"to": "数值重构方法", "type": "contrast", "note": "前者先验解析, 后者先验数值"},
            ],
        },
        {
            "name": "数值重构方法",
            "statement": "在带噪条件下最大似然成为标配, 但依赖物理正则。",
            "fact_ids": ["pB::2", "pB::3", "pC::4", "pC::5"],
            "notes": "结构化先验显著压缩问题维度。",
            "relations": [
                {"to": "随机测量与快照", "type": "support", "note": "两者共同降低采样成本"},
            ],
        },
    ],
    "narrative_arc": ["数值重构方法", "随机测量与快照"],
    "gaps": ["态验证的实验验证案例偏少"],
})


def test_organize_produces_themes(patched_llm):
    patched_llm([f"```json\n{THEME_JSON}\n```"])
    m = O.organize_facts_into_themes(FACTS)
    assert len(m["themes"]) == 2
    assert m["themes"][0]["name"] == "随机测量与快照"
    assert m["narrative_arc"] == ["数值重构方法", "随机测量与快照"]
    assert m["gaps"] == ["态验证的实验验证案例偏少"]


def test_organize_keeps_model_thinking_stripped(patched_llm):
    patched_llm(["<think>模型自述</think>" + THEME_JSON])
    m = O.organize_facts_into_themes(FACTS)
    assert len(m["themes"]) == 2, "think 标签应被剥掉再 parse"


def test_organize_drops_invented_fact_ids(patched_llm):
    """
    核心: 模型编造的 fact_id 必须被丢弃, 且如实报告。
    这里 'pZ::9' 和 'made_up' 都不在 fact 集合里。
    """
    bad = json.dumps({
        "themes": [{
            "name": "T", "statement": "s",
            "fact_ids": ["pA::0", "pZ::9", "made_up", "pA::1"],
            "notes": "", "relations": [],
        }],
        "narrative_arc": ["T"], "gaps": [],
    })
    patched_llm([bad])
    m = O.organize_facts_into_themes(FACTS)
    assert m["themes"][0]["fact_ids"] == ["pA::0", "pA::1"]
    assert any("无效 fact_id" in w for w in m["warnings"])


def test_organize_reports_uncovered_facts(patched_llm):
    """没被任何主题覆盖的 fact 要如实报告 —— 不静默丢。"""
    patched_llm([THEME_JSON])
    m = O.organize_facts_into_themes(FACTS)
    assert any("未被任何主题覆盖" in w for w in m["warnings"])


def test_organize_skips_theme_without_valid_facts(patched_llm):
    d = json.dumps({
        "themes": [{"name": "空主题", "statement": "s", "fact_ids": ["nope"],
                    "notes": "", "relations": []}],
        "narrative_arc": [], "gaps": [],
    })
    patched_llm([d])
    m = O.organize_facts_into_themes(FACTS)
    assert m["themes"] == []


def test_organize_dedups_theme_names(patched_llm):
    d = json.dumps({
        "themes": [
            {"name": "T", "statement": "a", "fact_ids": ["pA::0"], "notes": "", "relations": []},
            {"name": "T", "statement": "b", "fact_ids": ["pA::1"], "notes": "", "relations": []},
        ],
        "narrative_arc": [], "gaps": [],
    })
    patched_llm([d])
    m = O.organize_facts_into_themes(FACTS)
    assert len(m["themes"]) == 1
    assert any("重复" in w for w in m["warnings"])


def test_organize_respects_max_themes(patched_llm):
    themes = [{"name": f"T{i}", "statement": "s", "fact_ids": ["pA::0"],
               "notes": "", "relations": []} for i in range(12)]
    patched_llm([json.dumps({"themes": themes, "narrative_arc": [], "gaps": []})])
    m = O.organize_facts_into_themes(FACTS, max_themes=5)
    assert len(m["themes"]) == 5
    assert any("上限" in w for w in m["warnings"])


def test_organize_handles_empty_facts():
    m = O.organize_facts_into_themes([])
    assert m["themes"] == [] and m["warnings"]


def test_organize_reports_json_failure(patched_llm):
    patched_llm(["完全不是 JSON"])
    m = O.organize_facts_into_themes(FACTS)
    assert m["themes"] == []
    assert any("解析失败" in w for w in m["warnings"])


def test_organize_prompt_has_no_paper_reference_list(patched_llm):
    """
    关键: Phase B 的 prompt 只给 fact, **不给** 参考文献表/论文标题/可引用的 id 清单。

    注意 fact_id 形如 `{paper_id}::{序号}`, 所以 paper_id 字符串必然出现在 fact_id 里
    (这是可调试性的代价, 可接受)。真正要保证的是: 没有一份独立的"可以引用哪些论文"
    的清单交给模型 —— 那样它就能自由挑引用。paper_titles 传了也不应被渲染进 prompt。
    """
    fake = patched_llm([THEME_JSON])
    O.organize_facts_into_themes(FACTS, paper_titles={"pA": "SOME PAPER TITLE"})
    p = fake.prompts[0]
    assert "SOME PAPER TITLE" not in p, "论文标题不应进 Phase B 的 prompt"
    assert "pA::0" in p, "但 fact_id 必须在"
    for marker in ("paper_id", "paper_ids", "可引用", "参考文献"):
        assert marker not in p, f"prompt 不应含 {marker!r} 这类可自由选引用的线索"


# ============================================================
# Phase C: 引用映射
# ============================================================

def test_render_citations_maps_fact_id_to_paper_id():
    text = "快照把重构化为线性反演 [[pA::0]]，样本复杂度对数增长 [[pA::1]]。"
    out = O._render_citations(text, {"pA::0": "2401.12345", "pA::1": "2401.12345"})
    assert "[[" not in out
    assert out.count("[2401.12345]") == 2


def test_render_citations_drops_invented_fact_id():
    text = "真实断言 [[pA::0]]，编造的 [[fake::1]]。"
    out = O._render_citations(text, {"pA::0": "2401.12345"})
    assert "[2401.12345]" in out
    assert "fake" not in out
    assert "[]" not in out


def test_render_citations_handles_combined():
    out = O._render_citations("一条断言 [[pA::0, pB::1]]。", {"pA::0": "2401.1", "pB::1": "2401.2"})
    assert "[2401.1]" in out and "[2401.2]" in out


def test_render_citations_merges_adjacent_same_paper():
    """相邻的同一篇引用要合并(连着标三次 [X] 很丑)。非相邻的保留。"""
    out = O._render_citations("同一篇 [[pA::0]] [[pA::1]]。", {"pA::0": "X", "pA::1": "X"})
    assert out.count("[X]") == 1, "相邻重复应合并"

    out2 = O._render_citations("同一篇 [[pA::0]] 中间隔一句 [[pA::1]]。", {"pA::0": "X", "pA::1": "X"})
    assert out2.count("[X]") == 2, "非相邻的引用应保留(每处都是独立佐证)"


def test_render_citations_never_emits_empty_brackets():
    out = O._render_citations("无标注断言。 [[unknown]]", {})
    assert "[]" not in out and "[[" not in out


# ============================================================
# Phase C: 句子级过滤
# ============================================================

def test_drop_unattributed_keeps_tagged_only():
    text = "第一句有标注 [[pA::0]]。\n无标注的断言。\n第三句也有 [[pA::1]]。"
    kept, dropped = O._drop_unattributed(text)
    assert dropped == 1
    assert "无标注的断言" not in kept
    assert "第一句" in kept and "第三句" in kept


def test_drop_unattributed_keeps_paragraph_structure():
    text = "甲 [[pA::0]]。\n\n乙无标注。\n\n丙 [[pA::1]]。"
    kept, dropped = O._drop_unattributed(text)
    assert dropped == 1
    assert "乙" not in kept
    assert len(kept.split("\n\n")) == 2


# ============================================================
# Phase C: 写主题叙述
# ============================================================

def test_write_theme_returns_cited_prose(patched_llm):
    patched_llm([
        "随机测量把状态重构化为线性反演问题 [[pA::0]]。"
        "其样本复杂度随被估计量规模对数增长 [[pA::1]]。"
    ])
    theme = {"name": "T", "statement": "s", "notes": "", "fact_ids": ["pA::0", "pA::1"],
             "relations": []}
    r = O.write_theme_narrative(theme, FACTS)
    assert "[[" not in r["text"]
    assert r["text"].count("[pA]") == 2
    assert set(r["used_fact_ids"]) == {"pA::0", "pA::1"}


def test_write_theme_drops_fact_ids_not_in_theme(patched_llm):
    """
    模型给了一个**不属于本主题**的 fact_id(pB::3) ——
    它不在 fact2pid 里, 标注会被丢弃(而不是把 pB 的 paper 挂进来)。
    """
    patched_llm(["本主题的断言 [[pA::0]]，以及一条越界的 [[pB::3]]。"])
    theme = {"name": "T", "statement": "s", "notes": "", "fact_ids": ["pA::0"], "relations": []}
    r = O.write_theme_narrative(theme, FACTS)
    assert "[pA]" in r["text"]
    assert "[pB]" not in r["text"], "越界的 fact_id 不应带出别的论文"
    assert "pB::3" not in r["used_fact_ids"]


def test_write_theme_retries_when_unattributed(patched_llm):
    patched_llm([
        "完全没有标注的一大段流畅文字,没有任何出处。",     # 第一次: 无标注
        "这次有标注了 [[pA::0]]。",                          # 重试
    ])
    theme = {"name": "T", "statement": "s", "notes": "", "fact_ids": ["pA::0"], "relations": []}
    r = O.write_theme_narrative(theme, FACTS)
    assert r["text"], "重试后应产出内容"
    assert "没有任何出处" not in r["text"]


def test_write_theme_gives_up_after_max_retry(patched_llm):
    patched_llm(["无标注 A", "无标注 B", "无标注 C"])
    theme = {"name": "T", "statement": "s", "notes": "", "fact_ids": ["pA::0"], "relations": []}
    r = O.write_theme_narrative(theme, FACTS)
    assert r["text"] == "" or "无标注" not in r["text"]
    assert any("未能建立 fact 对应" in w for w in r["warnings"])


def test_write_theme_handles_missing_facts(patched_llm):
    r = O.write_theme_narrative(
        {"name": "T", "fact_ids": ["pZ::9"]}, FACTS,
    )
    assert r["text"] == ""
    assert r["warnings"]


# ============================================================
# prompt 纪律
# ============================================================

def test_write_prompt_forbids_fabricated_empirical_content():
    p = O.THEME_WRITE_PROMPT.format(
        theme_name="T", theme_statement="s", theme_notes="n",
        theme_relations="r", facts_block="[pA::0] c", n_para=3,
    )
    assert "禁止" in p and "经验内容" in p
    assert "装置平台" in p or "平台" in p


def test_write_prompt_requires_analysis():
    """综述区别于论文摘要的地方 —— 必须要求做分析, 不能一起禁掉。"""
    p = O.THEME_WRITE_PROMPT.format(
        theme_name="T", theme_statement="s", theme_notes="n",
        theme_relations="r", facts_block="[pA::0] c", n_para=3,
    )
    assert "必须做" in p
    assert "支持" in p and "相反" in p or "条件" in p
    assert "摘要" in p, "应点明综述 vs 摘要的区别"


def test_write_prompt_contains_no_paper_id():
    p = O.THEME_WRITE_PROMPT.format(
        theme_name="T", theme_statement="s", theme_notes="n",
        theme_relations="r", facts_block="[pA::0] c", n_para=3,
    )
    assert "[pA]" not in p
    assert "[pA::0]" in p


def test_theme_map_prompt_forbids_paper_listing():
    p = O.THEME_MAP_PROMPT.format(facts_block="[pA::0] c", n_subject_histogram="(无)")
    assert "论文列举" in p
    assert "领域判断" in p


# ============================================================
# 组装
# ============================================================

def test_render_survey_orders_by_arc():
    tm = {
        "themes": [
            {"name": "A", "statement": "sa", "fact_ids": ["pA::0"], "notes": "", "relations": []},
            {"name": "B", "statement": "sb", "fact_ids": ["pA::1"], "notes": "", "relations": []},
        ],
        "narrative_arc": ["B", "A"],
        "gaps": ["缺实验验证"],
    }
    nar = {"A": {"text": "AA [[pA::0]]。"}, "B": {"text": "BB [[pA::1]]。"}}
    md = O.render_survey(tm, nar, title="综述")
    assert md.index("## B") < md.index("## A"), "应按 narrative_arc 排序"
    assert "# 综述" in md
    assert "## 文献覆盖的不足" in md


def test_render_survey_skips_empty_narratives():
    tm = {"themes": [{"name": "A", "statement": "", "fact_ids": [], "notes": "", "relations": []}],
          "narrative_arc": [], "gaps": []}
    md = O.render_survey(tm, {"A": {"text": ""}})
    assert "## A" not in md


# ============================================================
# 落盘
# ============================================================

def test_thematic_map_roundtrip(tmp_path):
    from agent.infra.survey_workspace import SurveyWorkspace
    ws = SurveyWorkspace("deadbeef1234", root=tmp_path, topic="t")
    tm = {"themes": [{"name": "A", "statement": "s", "fact_ids": ["pA::0"],
                      "notes": "", "relations": []}], "narrative_arc": ["A"], "gaps": []}
    O.save_thematic_map(ws, tm)
    back = O.load_thematic_map(ws)
    assert back["themes"][0]["name"] == "A"
    assert "_saved_at" in back
    assert (ws.path / "_thematic_map.json").exists()


def test_load_thematic_map_missing(tmp_path):
    from agent.infra.survey_workspace import SurveyWorkspace
    ws = SurveyWorkspace("feedface9999", root=tmp_path, topic="t")
    assert O.load_thematic_map(ws) is None


# ============================================================
# JSON 容错提取 (9-29 实测: MiniMax 把答案吃进 think 块 / 被截断)
# ============================================================

_MAP = {"themes": [{"name": "T", "statement": "s", "fact_ids": ["pA::0"],
                    "notes": "", "relations": []}],
        "narrative_arc": ["T"], "gaps": []}


def test_parse_plain_json():
    assert O._parse_theme_json(json.dumps(_MAP))["themes"][0]["name"] == "T"


def test_parse_fenced_json():
    assert O._parse_theme_json("```json\n" + json.dumps(_MAP) + "\n```") is not None


def test_parse_strips_think():
    got = O._parse_theme_json("<think>先想一下</think>" + json.dumps(_MAP))
    assert got and got["themes"][0]["name"] == "T"


def test_parse_falls_back_into_raw_when_think_eats_everything():
    """
    实测场景: 思考吃光 max_tokens, 剥掉 think 后是空串,
    但 JSON 写在 think 块内部 —— 必须回退。
    """
    got = O._parse_theme_json("<think>" + json.dumps(_MAP))
    assert got and got["themes"][0]["name"] == "T"


def test_parse_recovers_truncated_json():
    """答案被 max_tokens 截断, 右括号缺失 —— 补齐后应能解析。"""
    truncated = json.dumps(_MAP)[:-1]
    got = O._parse_theme_json(truncated)
    assert got and got["themes"][0]["name"] == "T"


def test_parse_rejects_non_map():
    assert O._parse_theme_json('{"foo": 1}') is None
    assert O._parse_theme_json('{"themes": []}') is None   # 空主题不算成功
    assert O._parse_theme_json("完全不是 JSON") is None
    assert O._parse_theme_json("") is None


# ============================================================
# 接入 survey_generate (9-29)
# ============================================================

def test_survey_generate_exposes_organize():
    """旧路线必须保留 —— organize 是新增的一档, 不是替换。"""
    from agent.tool_definitions import TOOL_SCHEMAS
    import inspect as _i
    props = TOOL_SCHEMAS["survey_generate"]["function"]["parameters"]["properties"]
    assert "organize" in props, "schema 要暴露 organize 开关"
    assert props["organize"]["default"] is False, "默认必须仍是旧路线"
    assert props["phase"]["enum"] == ["all", "plan", "write", "organize"]
    sig = _i.signature(SG_MOD.survey_generate)
    assert sig.parameters["organize"].default is False
    assert sig.parameters["phase"].default == "all"


def test_old_route_still_default():
    """phase 默认 all + organize 默认 False = 行为完全不变。"""
    from agent.tool_definitions import TOOL_SCHEMAS
    props = TOOL_SCHEMAS["survey_generate"]["function"]["parameters"]["properties"]
    assert props["phase"]["default"] == "all"


def test_organize_phase_routes_to_organize_phase():
    import inspect as _i
    src = _i.getsource(SG_MOD.survey_generate)
    assert "_run_organize_phase" in src
    assert 'phase == "organize" or organize' in src


def test_organize_phase_shares_pool_and_facts_with_old_route():
    """
    主题驱动只替换 3/4(主题地图+主题叙述), Phase A 候选池和 fact 抽取必须复用 ——
    否则两条路线不可对照, 而且重复造轮子。
    """
    import inspect as _i
    src = _i.getsource(SG_MOD._run_organize_phase)
    assert "_build_candidate_pool" in src, "必须复用 Phase A 候选池"
    assert "action_extract_facts" in src, "必须复用 fact 抽取"
    assert "organize_facts_into_themes" in src, "Phase B 主题地图"
    assert "write_theme_narrative" in src, "Phase C 主题叙述"
    assert "create_section" in src, "章节落 surveys.db"


def test_organize_phase_filters_ungrounded_facts():
    """未通过落地审查的 fact 不能进主题地图 —— 否则审查形同虚设。"""
    import inspect as _i
    src = _i.getsource(SG_MOD._run_organize_phase)
    assert "usable" in src
    assert "_fact_is_usable" in src, "必须先过滤不可用 fact"
    # 传给 Phase B 的必须是 usable 而不是 facts
    assert "organize_facts_into_themes(usable" in src


def test_organize_writes_thematic_map_to_workspace():
    import inspect as _i
    src = _i.getsource(SG_MOD._run_organize_phase)
    assert "save_thematic_map" in src, "主题地图要落盘(可编辑复用)"


def test_organize_audit_covers_fact_usage():
    """审计要报 fact 使用率 —— 哪些 fact 没被用上, 是文献覆盖的信号。"""
    _organize_audit = SG_MOD._organize_audit
    tm = {"themes": [{"name": "T", "statement": "s", "fact_ids": ["a::0", "a::1"],
                      "notes": "", "relations": []}],
          "gaps": ["缺 X"]}
    nar = {"T": {"text": "正文", "used_fact_ids": ["a::0"], "dropped_sentences": 2,
                 "warnings": []}}
    facts = [{"fact_id": "a::0"}, {"fact_id": "a::1"}]
    md = _organize_audit(tm, nar, facts)
    assert "fact 使用率" in md
    assert "未被引用: 1 条" in md
    assert "缺 X" in md
    assert "2 句因无出处剔除" in md


# ============================================================
# subject 聚类作为 Phase B 的归纳锚 (9-29)
# ============================================================

def _sf(fid, subject, pid="pA"):
    """模拟 clean_fact 产出的 fact(带 subject 校验标记)。"""
    from agent.prompts_survey import _check_subject
    f = {"fact_id": fid, "paper_id": pid, "subject": subject, "claim": "c",
         "fact_type": "method", "support_quote": "q"}
    f.update(_check_subject(subject))
    return f


def test_cluster_by_subject_groups_and_sorts():
    facts = [
        _sf("a::0", "随机测量快照"), _sf("a::1", "随机测量快照"), _sf("a::2", "随机测量快照"),
        _sf("b::0", "SIC-POVM 态层析"), _sf("b::1", "SIC-POVM 态层析"),
        _sf("c::0", "张量网络"),
    ]
    got = O.cluster_by_subject(facts)
    names = [s for s, _ in got]
    assert names[0] == "随机测量快照", "最大的簇应排最前"
    assert names[1] == "SIC-POVM 态层析"
    assert "其他/未归类" in names, "只出现 1 次的 subject 应并入其他"


def test_cluster_by_subject_all_without_subject():
    """全都没有 subject 时应归成一个桶, 不是返回空。"""
    got = O.cluster_by_subject([{"fact_id": "a::0", "claim": "c"}])
    assert len(got) == 1 and got[0][0] == "其他/未归类"


def test_cluster_by_subject_empty_input():
    assert O.cluster_by_subject([]) == []


def test_cluster_totals_match_input():
    """
    护栏: histogram 的总数必须等于 fact 总数。
    否则 LLM 会看到"你说 39 条, 我这有 65 条"这种对不上的数字。
    (9-29 实测踩到: 没 subject 的 fact 被静默丢掉)
    """
    facts = [
        _sf("a::0", "随机测量快照"), _sf("a::1", "随机测量快照"),
        _sf("b::0", "SIC-POVM"), _sf("b::1", "SIC-POVM"),
        _sf("c::0", ""),            # 空 subject
        {"fact_id": "d::0", "claim": "c"},   # 根本没有 subject 键
    ]
    cl = O.cluster_by_subject(facts)
    assert sum(len(fs) for _s, fs in cl) == len(facts)
    assert any(s == "其他/未归类" for s, _ in cl)
    h = O._subject_histogram(cl)
    assert f"共 {len(facts)} 条 fact" in h


def test_subject_histogram_renders_counts():
    facts = [
        _sf("a::0", "随机测量快照"), _sf("a::1", "随机测量快照", pid="pB"),
        _sf("b::0", "SIC-POVM 态层析"), _sf("b::1", "SIC-POVM 态层析", pid="pB"),
    ]
    h = O._subject_histogram(O.cluster_by_subject(facts))
    assert "随机测量快照: 2 条 fact, 来自 2 篇不同论文" in h
    assert "SIC-POVM 态层析: 2 条 fact, 来自 2 篇不同论文" in h
    assert "2 个对象簇" in h


def test_subject_histogram_handles_missing():
    assert "无法按对象聚类" in O._subject_histogram([])


def test_theme_map_prompt_includes_histogram_slot():
    """护栏: Phase B 的 prompt 必须带语料分布槽位, 并说明它是提示不是硬约束。"""
    assert "{n_subject_histogram}" in O.THEME_MAP_PROMPT
    assert "这是提示, 不是硬约束" in O.THEME_MAP_PROMPT
    assert "客观数据" in O.THEME_MAP_PROMPT


def test_organize_passes_histogram_to_llm(patched_llm):
    fake = patched_llm([THEME_JSON])
    facts = [
        _sf("a::0", "随机测量快照"), _sf("a::1", "随机测量快照"),
        _sf("b::0", "SIC-POVM 态层析"), _sf("b::1", "SIC-POVM 态层析"),
    ]
    O.organize_facts_into_themes(facts)
    p = fake.prompts[0]
    assert "随机测量快照: 2 条 fact" in p, "LLM 应看到 subject 分布"
    assert "SIC-POVM 态层析: 2 条 fact" in p


def test_write_prompt_shows_subject_per_fact(monkeypatch):
    """写作时也要看到每条 fact 的核心对象。"""
    fake = _FakeLLM(["断言 [[a::0]]。"])
    import agent.tools.llm as lm
    monkeypatch.setattr(lm, "_call_llm", fake)
    theme = {"name": "T", "statement": "s", "notes": "",
             "fact_ids": ["a::0"], "relations": []}
    O.write_theme_narrative(theme, [_sf("a::0", "随机测量快照")])
    assert "<对象: 随机测量快照>" in fake.prompts[0]


def test_fact_prompt_asks_for_subject():
    """fact 抽取必须问 subject, 否则库里这列永远是空的。"""
    from agent.prompts_survey import SURVEY_FACT_PROMPT
    assert '"subject"' in SURVEY_FACT_PROMPT
    assert "可跨论文对齐" in SURVEY_FACT_PROMPT
    assert "同一个对象" in SURVEY_FACT_PROMPT, "应说明同一对象必须用同一字符串"


def test_clean_fact_keeps_subject():
    from agent.prompts_survey import clean_fact
    f = clean_fact({"subject": "  随机测量快照  ", "claim": "c",
                    "support_quote": "q"}, "pA", 0)
    assert f["subject"] == "随机测量快照"
    g = clean_fact({"claim": "c", "support_quote": "q"}, "pA", 1)
    assert g["subject"] == ""


# ============================================================
# subject 校验 + 变体检测 (9-27)
# ============================================================

def test_subject_check_flags_self_reference():
    from agent.prompts_survey import _check_subject
    r = _check_subject("本文方法")
    assert r["_subject_ok"] is False
    assert any("自指" in x for x in r["_subject_issues"])
    r2 = _check_subject("our method")
    assert r2["_subject_ok"] is False


def test_subject_check_flags_empty_and_length():
    from agent.prompts_survey import _check_subject
    assert _check_subject("")["_subject_ok"] is False
    assert _check_subject("X")["_subject_ok"] is False            # 过短
    long_s = "a" * 60
    assert _check_subject(long_s)["_subject_ok"] is False         # 过长
    assert _check_subject("随机测量快照")["_subject_ok"] is True


def test_subject_check_flags_trailing_punct():
    from agent.prompts_survey import _check_subject
    assert _check_subject("随机测量快照。")["_subject_ok"] is False


def test_clean_fact_attaches_subject_check():
    from agent.prompts_survey import clean_fact
    f = clean_fact({"subject": "本文方法", "claim": "c", "support_quote": "q"}, "pA", 0)
    assert f["_subject_ok"] is False
    assert f["_subject_issues"]
    g = clean_fact({"subject": "随机测量快照", "claim": "c", "support_quote": "q"}, "pA", 1)
    assert g["_subject_ok"] is True


def test_cluster_excludes_bad_subjects():
    """subject 不合格的不能当聚类键, 否则污染分布。"""
    good = [_sf("a::0", "随机测量快照"), _sf("a::1", "随机测量快照"),
            _sf("a::2", "随机测量快照")]
    bad = [_sf("b::0", "本文方法"), _sf("b::1", "本文方法")]
    cl = O.cluster_by_subject(good + bad)
    assert sum(len(fs) for _s, fs in cl) == 5, "总数必须守恒"
    other = next(fs for s, fs in cl if s == "其他/未归类")
    assert len(other) == 2


def test_subject_variant_detection():
    """同一对象的多种叫法要能被标记出来(但不合并)。"""
    # "经典阴影层析" 是 "经典阴影" 的子串 -> 应被判为变体
    # "张量网络态层析" 与它们 token 不重叠 -> 不该被拉进来
    facts = [
        _sf("a::0", "经典阴影"), _sf("a::1", "经典阴影"),
        _sf("b::0", "经典阴影层析"), _sf("b::1", "经典阴影层析"),
        _sf("c::0", "张量网络态层析"), _sf("c::1", "张量网络态层析"),
    ]
    groups = O.flag_subject_variants(facts)
    assert groups, "应发现变体组"
    members = {m["subject"] for m in groups[0]["members"]}
    assert "经典阴影" in members and "经典阴影层析" in members, (
        "子串关系的两个 subject 应在同一组"
    )
    # 打了标记但没真合并
    assert len({f["subject"] for f in facts}) == 3
    # 打了标记但没合并
    assert len({f["subject"] for f in facts}) == 3, "不该真的合并"


def test_subject_variant_substring_case():
    """子串关系("经典阴影" vs "经典阴影层析")应判为可能同一对象。"""
    facts = [_sf("a::0", "经典阴影"), _sf("b::0", "经典阴影层析")]
    groups = O.flag_subject_variants(facts)
    assert groups and "经典阴影" in {m["subject"] for m in groups[0]["members"]}


def test_short_subject_excluded_from_variant_detection():
    """
    过短的 subject(如"阴影")本身不可靠, 不该参与变体判定 ——
    否则一个没意义的短串会把两个不相干的对象"合并"成一组。
    """
    facts = [_sf("a::0", "阴影"), _sf("b::0", "随机测量")]
    assert O.flag_subject_variants(facts) == []


def test_subject_variant_unrelated_not_flagged():
    facts = [_sf("a::0", "随机测量"), _sf("b::0", "张量网络态层析")]
    assert O.flag_subject_variants(facts) == []


def test_subject_variant_skips_bad_subjects():
    facts = [_sf("a::0", "本文方法"), _sf("b::0", "该方法")]
    assert O.flag_subject_variants(facts) == [], "自指 subject 不参与变体判定"


def test_subject_variant_empty_input():
    assert O.flag_subject_variants([]) == []
    assert O.flag_subject_variants([_sf("a::0", "快照")]) == []
