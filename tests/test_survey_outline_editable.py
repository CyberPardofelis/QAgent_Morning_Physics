"""
tests/test_survey_outline_editable.py — 大纲可编辑 + phase 拆分 的单元测试

9-27 新增。

背景:
    大纲原本由 survey_generate 的 ReAct 循环内部生成, 生成后立刻锁死
    (_write_outline 的 `if state.get("outline_sections"): return`), 然后同一次运行
    里把 5 章全写完。**用户看到不对只能整体重跑** —— 这是"综述工具很别扭"的主因之一。

    工作区其实早就把大纲持久化到 _outline.json 且可回读, 只是没有编辑入口。

本测试覆盖:
    - normalize_outline_sections: 共享校验(LLM 生成和用户编辑同一套)
    - survey_outline view / update / 校验 / 规范化警告
    - survey_generate phase="plan" / "write" 的状态装配(不调 LLM)
    - 单章重写的索引裁剪
    - 5 段模板是"保底"而非"白名单" —— custom 类型要能通过
"""
from __future__ import annotations

import importlib
import json

import pytest

from agent.infra.survey_workspace import SurveyWorkspace, hash_topic
from agent.tools.surveys._outline import (
    normalize_outline_sections,
    normalize_sec_type,
    outline_to_json,
    render_outline_md,
)

# surveys/__init__.py 把 survey_outline / survey_generate 这两个**函数**重导出到包
# 命名空间, 会遮蔽同名**模块**属性。monkeypatch 必须拿真实模块对象,
# 直接 setattr("pkg.survey_outline.X") 会打到函数上去。
OUTLINE_MOD = importlib.import_module("agent.tools.surveys.survey_outline")
GENERATE_MOD = importlib.import_module("agent.tools.surveys.survey_generate")

TOPIC = "test-topic-editable-outline"


def _patch_ws(monkeypatch, mod, ws):
    """把模块里的 SurveyWorkspace 换成隔离实例(避免污染真实 data/)。"""
    monkeypatch.setattr(mod, "SurveyWorkspace", lambda h, topic="": ws)


@pytest.fixture
def ws(tmp_path):
    """隔离的 workspace。"""
    return SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC)


@pytest.fixture
def with_outline(ws):
    ws.save_pool([{"paper_id": "p1", "title": "t"}])
    secs, _ = normalize_outline_sections([
        {"title": "引言", "section_type": "intro", "sub_points": ["a", "b"]},
        {"title": "进展", "section_type": "progress", "sub_points": ["c"]},
    ], fill_to_five=False)
    ws.save_outline({"sections": secs})
    return ws


# ============================================================
# normalize_sec_type
# ============================================================

@pytest.mark.parametrize("raw,expected", [
    ("intro", "intro"),
    ("Method-Compare", "method_compare"),
    ("GAP", "gap"),
    ("out look", "outlook"),        # 带空格 → 归一化后要能命中
    ("background", "intro"),
    ("timeline", "progress"),
    ("comparison", "method_compare"),
    ("challenge", "gap"),
    ("future", "outlook"),
])
def test_normalize_sec_type(raw, expected):
    assert normalize_sec_type(raw) == expected


def test_normalize_sec_type_empty_is_none():
    """空串不能误命中 —— 之前 `"" in "outlook"` 会返回 True。"""
    assert normalize_sec_type("") is None
    assert normalize_sec_type(None) is None
    assert normalize_sec_type("   ") is None


def test_unknown_type_becomes_custom_not_dropped():
    """
    5 段模板是**保底**, 不是白名单。
    用户自定义章节(如"产业落地")必须能通过, 而不是被静默丢弃。
    """
    assert normalize_sec_type("应用") == "custom"
    assert normalize_sec_type("zzz_unknown_xyz") == "custom"


# ============================================================
# normalize_outline_sections
# ============================================================

def test_normalize_keeps_custom_sections():
    out, _ = normalize_outline_sections([
        {"title": "引言", "section_type": "intro", "sub_points": ["a"]},
        {"title": "产业落地", "section_type": "应用", "sub_points": ["b"]},
    ], fill_to_five=False)
    assert [s["section_type"] for s in out] == ["intro", "custom"]
    assert out[1]["title"] == "产业落地"


def test_normalize_dedups_duplicate_types():
    out, warns = normalize_outline_sections([
        {"title": "A", "section_type": "intro", "sub_points": ["x"]},
        {"title": "B", "section_type": "intro", "sub_points": ["y"]},
    ], fill_to_five=False)
    assert len(out) == 1
    assert any("重复" in w for w in warns)


def test_normalize_skips_missing_title_with_warning():
    out, warns = normalize_outline_sections([
        {"section_type": "intro", "sub_points": ["x"]},
    ], fill_to_five=False)
    assert out == []
    assert any("title" in w for w in warns)


def test_normalize_accepts_json_string_element():
    out, _ = normalize_outline_sections([
        json.dumps({"title": "A", "section_type": "intro", "sub_points": ["x"]}),
    ], fill_to_five=False)
    assert len(out) == 1


def test_normalize_fills_five_when_requested():
    out, _ = normalize_outline_sections([
        {"title": "引言", "section_type": "intro", "sub_points": ["x"]},
    ], fill_to_five=True)
    assert len(out) == 5
    assert {s["section_type"] for s in out} == {
        "intro", "progress", "method_compare", "gap", "outlook",
    }


def test_normalize_does_not_fill_when_disabled():
    """用户编辑路径不补齐 —— 删到 1 章就是想要 1 章。"""
    out, _ = normalize_outline_sections([
        {"title": "引言", "section_type": "intro", "sub_points": ["x"]},
    ], fill_to_five=False)
    assert len(out) == 1


def test_normalize_sub_points_string_becomes_list():
    out, _ = normalize_outline_sections([
        {"title": "A", "section_type": "intro", "sub_points": "只有一个"},
    ], fill_to_five=False)
    assert out[0]["sub_points"] == ["只有一个"]


def test_normalize_empty_sub_points_placeholder():
    out, warns = normalize_outline_sections([
        {"title": "A", "section_type": "intro", "sub_points": []},
    ], fill_to_five=False)
    assert out[0]["sub_points"] == ["(待定)"]
    assert any("sub_points" in w for w in warns)


# ============================================================
# 渲染
# ============================================================

def test_render_outline_md_uses_index_labels():
    """带 [i] 标号 —— 用户要能说"改第 3 章"。"""
    md = render_outline_md([
        {"title": "引言", "section_type": "intro", "sub_points": ["a"], "is_transition": "yes"},
        {"title": "进展", "section_type": "progress", "sub_points": ["b"], "is_transition": "no"},
    ])
    assert "**[0] `intro` 引言**" in md
    assert "**[1] `progress` 进展**" in md
    assert "过渡章" in md  # intro 标了 is_transition=yes


def test_render_outline_md_empty():
    assert render_outline_md([]) == "(大纲为空)"


def test_outline_to_json_roundtrip():
    secs, _ = normalize_outline_sections([
        {"title": "A", "section_type": "intro", "sub_points": ["x"]},
    ], fill_to_five=False)
    assert json.loads(outline_to_json(secs))["sections"][0]["title"] == "A"


# ============================================================
# survey_outline 工具
# ============================================================

def test_view_without_outline_tells_user_what_to_do(monkeypatch, tmp_path):
    _patch_ws(monkeypatch, OUTLINE_MOD,
              SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC))
    out = OUTLINE_MOD.survey_outline(topic=TOPIC, action="view")
    assert "还没有大纲" in out
    assert 'phase="plan"' in out, "应告诉用户下一步怎么生成大纲"


def test_view_renders_outline(monkeypatch, with_outline):
    _patch_ws(monkeypatch, OUTLINE_MOD, with_outline)
    out = OUTLINE_MOD.survey_outline(topic=TOPIC, action="view")
    assert "引言" in out and "进展" in out
    assert 'action="update"' in out


def test_update_writes_normalized_sections(monkeypatch, with_outline):
    _patch_ws(monkeypatch, OUTLINE_MOD, with_outline)
    payload = json.dumps({"sections": [
        {"title": "改过的引言", "section_type": "intro", "sub_points": ["新要点1", "新要点2"]},
        {"title": "产业落地", "section_type": "应用", "sub_points": ["商业化"]},
    ]}, ensure_ascii=False)
    out = OUTLINE_MOD.survey_outline(topic=TOPIC, action="update", sections=payload)
    assert "已更新" in out
    assert "改过的引言" in out

    saved = with_outline.load_outline()
    assert saved["edited"] is True
    assert [s["title"] for s in saved["sections"]] == ["改过的引言", "产业落地"]
    assert saved["sections"][1]["section_type"] == "custom"


def test_update_accepts_bare_array(monkeypatch, with_outline):
    _patch_ws(monkeypatch, OUTLINE_MOD, with_outline)
    out = OUTLINE_MOD.survey_outline(
        topic=TOPIC, action="update",
        sections=json.dumps([{"title": "只有一章", "section_type": "gap", "sub_points": ["x"]}]),
    )
    assert "已更新" in out
    assert len(with_outline.load_outline()["sections"]) == 1, "update 不应补齐到 5 段"


def test_update_invalid_json_gives_clear_error(monkeypatch, with_outline):
    _patch_ws(monkeypatch, OUTLINE_MOD, with_outline)
    out = OUTLINE_MOD.survey_outline(topic=TOPIC, action="update", sections="{not json")
    assert "不是合法 JSON" in out


def test_update_requires_sections(monkeypatch, with_outline):
    _patch_ws(monkeypatch, OUTLINE_MOD, with_outline)
    out = OUTLINE_MOD.survey_outline(topic=TOPIC, action="update", sections="")
    assert "必须传 sections" in out


def test_update_all_invalid_reports_reason(monkeypatch, with_outline):
    _patch_ws(monkeypatch, OUTLINE_MOD, with_outline)
    out = OUTLINE_MOD.survey_outline(
        topic=TOPIC, action="update", sections=json.dumps([{"nope": 1}]),
    )
    assert "没有有效章节" in out
    assert "title" in out


def test_empty_topic_rejected():
    assert "topic 不能为空" in OUTLINE_MOD.survey_outline(topic="", action="view")


def test_bad_action_rejected():
    assert "action 只能是" in OUTLINE_MOD.survey_outline(topic=TOPIC, action="delete")


def test_export_returns_editable_json(monkeypatch, with_outline):
    _patch_ws(monkeypatch, OUTLINE_MOD, with_outline)
    out = OUTLINE_MOD.survey_outline_export(topic=TOPIC)
    assert "```json" in out
    payload = out.split("```json")[1].split("```")[0]
    assert json.loads(payload)["sections"][0]["title"] == "引言"


# ============================================================
# survey_generate phase 拆分
# ============================================================

def test_generate_rejects_bad_phase():
    out = "".join(GENERATE_MOD.survey_generate(topic="x", phase="bogus"))
    assert "phase 只能是" in out


def test_generate_rejects_empty_topic():
    out = "".join(GENERATE_MOD.survey_generate(topic="  "))
    assert "topic 不能为空" in out


def test_write_phase_without_pool_explains_plan_first(monkeypatch, tmp_path):
    _patch_ws(monkeypatch, GENERATE_MOD,
              SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC))
    out = "".join(GENERATE_MOD.survey_generate(topic=TOPIC, phase="write"))
    assert "没有候选池" in out
    assert 'phase="plan"' in out


def test_write_phase_without_outline_explains_plan_first(monkeypatch, tmp_path):
    w = SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC)
    w.save_pool([{"paper_id": "p1", "title": "t"}])
    _patch_ws(monkeypatch, GENERATE_MOD, w)
    out = "".join(GENERATE_MOD.survey_generate(topic=TOPIC, phase="write"))
    assert "还没有大纲" in out


def test_write_phase_uses_edited_outline(monkeypatch, tmp_path):
    """
    核心: phase="write" 必须使用 workspace 里(用户改过的)大纲, 而不是重新生成。
    """
    w = SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC)
    w.save_pool([{"paper_id": "p1", "title": "t", "core_method": "m", "core_contrib": "c"}])
    w.append_fact({"paper_id": "p1", "claim": "cl", "fact_type": "result", "support_quote": "q"})
    w.save_outline({
        "sections": [{"title": "用户改过的章", "section_type": "custom",
                      "sub_points": ["用户自己加的要点"], "is_transition": "no"}],
        "edited": True,
    })
    _patch_ws(monkeypatch, GENERATE_MOD, w)
    monkeypatch.setattr(GENERATE_MOD, "_find_or_create_theme", lambda topic, edited: 1)

    captured = {}

    def fake_decide(state):
        captured.update(
            outline=list(state["outline_sections"]),
            pool=state["pool"], facts=state["facts"],
            edited=state.get("outline_edited"),
        )
        state["finish_requested"] = True
        return {"thought": "stop", "action": "finish", "args": {}}

    monkeypatch.setattr(GENERATE_MOD, "_decide_action", fake_decide)
    monkeypatch.setattr(GENERATE_MOD, "_finalize_survey", lambda s, w_: "final")

    out = "".join(GENERATE_MOD.survey_generate(topic=TOPIC, phase="write"))
    assert captured["outline"][0]["title"] == "用户改过的章"
    assert captured["outline"][0]["section_type"] == "custom"
    assert captured["edited"] is True
    assert len(captured["pool"]) == 1
    assert len(captured["facts"]) == 1
    assert "用户已编辑" in out


def test_section_idx_trims_outline_to_single(monkeypatch, tmp_path):
    """单章重写: section_idx 把 state 里的 outline 裁到只剩那一章。"""
    w = SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC)
    w.save_pool([{"paper_id": "p1", "title": "t"}])
    w.save_outline({"sections": [
        {"title": "第0章", "section_type": "intro", "sub_points": ["a"], "is_transition": "no"},
        {"title": "第1章", "section_type": "progress", "sub_points": ["b"], "is_transition": "no"},
        {"title": "第2章", "section_type": "gap", "sub_points": ["c"], "is_transition": "no"},
    ]})
    _patch_ws(monkeypatch, GENERATE_MOD, w)
    monkeypatch.setattr(GENERATE_MOD, "_find_or_create_theme", lambda topic, edited: 1)
    monkeypatch.setattr(GENERATE_MOD, "_drop_existing_section", lambda tid, sec: 0)

    captured = {}

    def fake_decide(state):
        captured["outline"] = list(state["outline_sections"])
        captured["max_steps"] = state["max_steps"]
        state["finish_requested"] = True
        return {"thought": "stop", "action": "finish", "args": {}}

    monkeypatch.setattr(GENERATE_MOD, "_decide_action", fake_decide)

    out = "".join(GENERATE_MOD.survey_generate(topic=TOPIC, phase="write", section_idx=1))
    assert len(captured["outline"]) == 1
    assert captured["outline"][0]["title"] == "第1章"
    assert captured["max_steps"] == 1, "单章重写只跑一轮"
    assert "单章模式" in out


def test_section_idx_out_of_range_writes_all(monkeypatch, tmp_path):
    """越界的 section_idx 退化成写全部, 而不是崩。"""
    w = SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC)
    w.save_pool([{"paper_id": "p1", "title": "t"}])
    w.save_outline({"sections": [
        {"title": "唯一章", "section_type": "intro", "sub_points": ["a"], "is_transition": "no"},
    ]})
    _patch_ws(monkeypatch, GENERATE_MOD, w)
    monkeypatch.setattr(GENERATE_MOD, "_find_or_create_theme", lambda topic, edited: 1)
    monkeypatch.setattr(GENERATE_MOD, "_finalize_survey", lambda s, w_: "final")

    captured = {}

    def fake_decide(state):
        captured["n"] = len(state["outline_sections"])
        state["finish_requested"] = True
        return {"thought": "s", "action": "finish", "args": {}}

    monkeypatch.setattr(GENERATE_MOD, "_decide_action", fake_decide)
    "".join(GENERATE_MOD.survey_generate(topic=TOPIC, phase="write", section_idx=99))
    assert captured["n"] == 1


# ============================================================
# schema 一致性
# ============================================================

def test_survey_generate_schema_exposes_phase_and_section_idx():
    from agent.tool_definitions import TOOL_SCHEMAS
    props = TOOL_SCHEMAS["survey_generate"]["function"]["parameters"]["properties"]
    assert "phase" in props
    assert props["phase"]["enum"] == ["all", "plan", "write"]
    assert "section_idx" in props


def test_new_tools_registered():
    from agent.tool_definitions import TOOL_SCHEMAS
    assert "survey_outline" in TOOL_SCHEMAS
    assert "survey_outline_export" in TOOL_SCHEMAS


def test_new_tools_are_l0():
    """大纲查看/编辑是纯本地操作, 不能标成 L1/L2 骗 token 预算。"""
    from agent.tools.registry import ToolRegistry
    import agent.tools.surveys  # noqa: F401 - 触发注册
    for name in ("survey_outline", "survey_outline_export"):
        assert ToolRegistry.get(name).cost_tier == "L0", f"{name} 应是 L0"
