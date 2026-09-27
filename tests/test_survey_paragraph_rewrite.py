"""
tests/test_survey_paragraph_rewrite.py — 单段重写 的单元测试

9-27 新增。

核心风险在 _paragraphs: 切分和拼接必须**严格互逆**, 否则重写一次就把
章节格式搞乱了(丢段、重复段、列表被切碎)。所以往返一致性是首要测试对象。

覆盖:
    - split/join 往返(普通段落 / 列表 / 标题 / 代码块 / 多空行)
    - 段类型识别
    - 上下文抽取
    - 工具层: dry_run 零 LLM、参数校验、越界、清洗 LLM 脏输出
    - prompt: 素材/上下文/目标段都进去
"""
from __future__ import annotations

import importlib

import pytest

from agent.tools.surveys._paragraphs import (
    extract_context,
    join_paragraphs,
    render_outline_for_rewrite,
    split_paragraphs,
)

RW_MOD = importlib.import_module("agent.tools.surveys.survey_rewrite_paragraphs")
TOPIC = "test-paragraph-rewrite"


# ============================================================
# 往返一致性 —— 最重要
# ============================================================

SIMPLE = """第一段内容。第二句也在第一段。

第二段内容。

第三段内容。"""

LISTS = """开头一段。

- 列表项 A
- 列表项 B
- 列表项 C

结尾一段。"""

MIXED = """## 小节标题

小节正文第一段。

- 要点 1
- 要点 2

小节正文第二段。

1. 编号项 1
2. 编号项 2

> 引用块

最后一段。"""

FENCED = """前言。

```python
def f():
    return 1
```

后语。"""

MANY_BLANKS = """第一段。


第二段。




第三段。"""


@pytest.mark.parametrize("md", [SIMPLE, LISTS, MIXED, FENCED])
def test_split_join_roundtrip(md):
    """切完再拼必须完全还原 —— 否则重写一次就破坏章节格式。"""
    assert join_paragraphs(split_paragraphs(md)) == md.strip()


def test_many_blank_lines_normalize_to_canonical_form():
    """
    连续多空行会被规范化成单个空行分隔 —— 这是**期望**行为(规范化),
    要求的是"再切一次不再变"(幂等), 而不是逐字节保留畸形空行。
    """
    once = join_paragraphs(split_paragraphs(MANY_BLANKS))
    assert once == "第一段。\n\n第二段。\n\n第三段。"
    assert join_paragraphs(split_paragraphs(once)) == once


@pytest.mark.parametrize("md", [SIMPLE, LISTS, MIXED, FENCED])
def test_roundtrip_is_stable_twice(md):
    """再切一次结果不变(幂等)。"""
    once = join_paragraphs(split_paragraphs(md))
    twice = join_paragraphs(split_paragraphs(once))
    assert once == twice


def test_split_empty():
    assert split_paragraphs("") == []
    assert split_paragraphs("   \n\n  ") == []


# ============================================================
# 段类型识别
# ============================================================

def test_kinds_detected():
    paras = split_paragraphs(MIXED)
    kinds = {p["idx"]: p["kind"] for p in paras}
    assert kinds[0] == "heading"
    assert "list" in kinds.values()
    assert "body" in kinds.values()


def test_list_items_stay_together():
    """
    关键: 列表项之间的空行**不能**切段 ——
    否则"只重写第 2 段"会把一个列表切成两半, 语义就碎了。
    """
    paras = split_paragraphs("- A\n\n- B\n\n- C")
    assert len(paras) == 1, "空行分隔的列表项应视为同一段"
    assert "A" in paras[0]["text"] and "C" in paras[0]["text"]


def test_heading_is_own_paragraph():
    paras = split_paragraphs("## 标题\n\n正文。")
    assert paras[0]["kind"] == "heading"
    assert paras[0]["text"].strip() == "## 标题"


def test_fenced_code_stays_intact():
    """
    代码块内部(含空行)不能被切碎。
    代码块自己是一个独立段 —— 这比"跟前一段合并"更合理。
    """
    paras = split_paragraphs("前。\n\n```\na\n\nb\n```\n\n后。")
    assert len(paras) == 3
    code = paras[1]["text"]
    assert code.startswith("```") and code.endswith("```")
    assert "a" in code and "b" in code
    assert paras[0]["text"] == "前。"
    assert paras[2]["text"] == "后。"


def test_preview_is_short_single_line():
    paras = split_paragraphs("很长的一句话" * 50)
    assert len(paras[0]["preview"]) <= 60
    assert "\n" not in paras[0]["preview"]


def test_crlf_normalized():
    assert join_paragraphs(split_paragraphs("A。\r\n\r\nB。")) == "A。\n\nB。"


# ============================================================
# 上下文抽取
# ============================================================

def test_extract_context_neighbors():
    paras = split_paragraphs("\n\n".join(f"第{i}段。" for i in range(5)))
    ctx = extract_context(paras, [2], before=1, after=1)
    assert "第1段" in ctx and "第3段" in ctx
    assert "第2段" not in ctx, "目标段本身不该出现在上下文里"
    assert "第0段" not in ctx, "窗口外的段不该出现"


def test_extract_context_window_respects_bounds():
    paras = split_paragraphs("A。\n\nB。\n\nC。")
    ctx = extract_context(paras, [0], before=2, after=2)
    assert "A。" not in ctx          # 目标段
    assert "B。" in ctx and "C。" in ctx


def test_extract_context_none_when_single_paragraph():
    paras = split_paragraphs("只有一段。")
    assert extract_context(paras, [0]) == ""


def test_extract_context_empty_targets():
    assert extract_context(split_paragraphs(SIMPLE), []) == ""


# ============================================================
# 渲染给 LLM 看的结构
# ============================================================

def test_render_marks_targets():
    paras = split_paragraphs("A。\n\nB。\n\nC。")
    out = render_outline_for_rewrite(paras, [1])
    assert "[第 1 段 — 要重写]" in out
    assert "[第 0 段 — 保持不变]" in out
    assert "[第 2 段 — 保持不变]" in out


# ============================================================
# 工具层参数校验 (零 LLM)
# ============================================================

def test_empty_topic():
    assert "topic 不能为空" in RW_MOD.survey_rewrite_paragraphs(topic="", section_idx=0)


def test_bad_section_idx_type():
    out = RW_MOD.survey_rewrite_paragraphs(topic=TOPIC, section_idx="abc")
    assert "必须是整数" in out


def test_no_outline(monkeypatch, tmp_path):
    from agent.infra.survey_workspace import SurveyWorkspace, hash_topic
    w = SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC)
    monkeypatch.setattr(RW_MOD, "SurveyWorkspace", lambda h, topic="": w)
    out = RW_MOD.survey_rewrite_paragraphs(topic=TOPIC, section_idx=0)
    assert "还没有大纲" in out


def test_section_idx_out_of_range(monkeypatch, tmp_path):
    from agent.infra.survey_workspace import SurveyWorkspace, hash_topic
    w = SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC)
    w.save_outline({"sections": [{"title": "A", "section_type": "intro", "sub_points": ["x"], "is_transition": "no"}]})
    monkeypatch.setattr(RW_MOD, "SurveyWorkspace", lambda h, topic="": w)
    out = RW_MOD.survey_rewrite_paragraphs(topic=TOPIC, section_idx=5)
    assert "越界" in out
    assert 'action="view"' in out


def test_bad_para_indices(monkeypatch, tmp_path):
    from agent.infra.survey_workspace import SurveyWorkspace, hash_topic
    w = SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC)
    w.save_outline({"sections": [{"title": "A", "section_type": "intro", "sub_points": ["x"], "is_transition": "no"}]})
    monkeypatch.setattr(RW_MOD, "SurveyWorkspace", lambda h, topic="": w)
    monkeypatch.setattr(RW_MOD, "_find_section_content", lambda *a, **k: "P0。\n\nP1。")
    out = RW_MOD.survey_rewrite_paragraphs(topic=TOPIC, section_idx=0, para_indices="a,b")
    assert "逗号分隔的整数" in out


def test_para_idx_out_of_range(monkeypatch, tmp_path):
    from agent.infra.survey_workspace import SurveyWorkspace, hash_topic
    w = SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC)
    w.save_outline({"sections": [{"title": "A", "section_type": "intro", "sub_points": ["x"], "is_transition": "no"}]})
    monkeypatch.setattr(RW_MOD, "SurveyWorkspace", lambda h, topic="": w)
    monkeypatch.setattr(RW_MOD, "_find_section_content", lambda *a, **k: "P0。\n\nP1。")
    out = RW_MOD.survey_rewrite_paragraphs(topic=TOPIC, section_idx=0, para_indices="9")
    assert "越界" in out
    assert "dry_run" in out


def test_no_content_yet(monkeypatch, tmp_path):
    from agent.infra.survey_workspace import SurveyWorkspace, hash_topic
    w = SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC)
    w.save_outline({"sections": [{"title": "A", "section_type": "intro", "sub_points": ["x"], "is_transition": "no"}]})
    monkeypatch.setattr(RW_MOD, "SurveyWorkspace", lambda h, topic="": w)
    monkeypatch.setattr(RW_MOD, "_find_section_content", lambda *a, **k: "")
    out = RW_MOD.survey_rewrite_paragraphs(topic=TOPIC, section_idx=0)
    assert "还没有正文" in out
    assert 'phase="write"' in out


def test_dry_run_makes_no_llm_call(monkeypatch, tmp_path):
    """dry_run 必须零 LLM —— 用户要靠它试参数。"""
    from agent.infra.survey_workspace import SurveyWorkspace, hash_topic
    w = SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC)
    w.save_outline({"sections": [{"title": "第一章", "section_type": "intro", "sub_points": ["x"], "is_transition": "no"}]})
    monkeypatch.setattr(RW_MOD, "SurveyWorkspace", lambda h, topic="": w)
    monkeypatch.setattr(RW_MOD, "_find_section_content", lambda *a, **k: SIMPLE)

    def boom(*a, **k):
        raise AssertionError("dry_run 不应调 LLM")

    monkeypatch.setattr("agent.tools.llm._call_llm", boom)

    out = RW_MOD.survey_rewrite_paragraphs(topic=TOPIC, section_idx=0, para_indices="1", dry_run=True)
    assert "dry_run" in out
    assert "将重写第 [1] 段" in out
    assert "未调 LLM" in out


def test_dry_run_default_targets_all_body_paragraphs(monkeypatch, tmp_path):
    """不传 para_indices 时, 目标 = 所有非标题段。"""
    from agent.infra.survey_workspace import SurveyWorkspace, hash_topic
    w = SurveyWorkspace(hash_topic(TOPIC), root=tmp_path, topic=TOPIC)
    w.save_outline({"sections": [{"title": "第一章", "section_type": "intro", "sub_points": ["x"], "is_transition": "no"}]})
    monkeypatch.setattr(RW_MOD, "SurveyWorkspace", lambda h, topic="": w)
    monkeypatch.setattr(RW_MOD, "_find_section_content", lambda *a, **k: MIXED)

    out = RW_MOD.survey_rewrite_paragraphs(topic=TOPIC, section_idx=0, dry_run=True)
    # MIXED 的第 0 段是 heading, 不该进目标
    assert "[第 0 段 — 要重写]" not in out
    assert "[第 1 段 — 要重写]" in out


# ============================================================
# 清洗 LLM 脏输出
# ============================================================

def test_clean_strips_markers_headings_fences():
    raw = "```\n[第 1 段]\n## 不该有的标题\n新正文 [2401.12345]\n```"
    out = RW_MOD._clean_rewritten(raw)
    assert "第 1 段" not in out
    assert "不该有的标题" not in out
    assert "```" not in out
    assert "新正文" in out


def test_split_blocks_exact_match():
    assert RW_MOD._split_output_blocks("A。\n\nB。", 2) == ["A。", "B。"]


def test_split_blocks_by_marker():
    raw = "[第 0 段]\nA。\n\n[第 2 段]\nB。"
    blocks = RW_MOD._split_output_blocks(raw, 2)
    assert len(blocks) == 2
    assert "A。" in blocks[0] and "B。" in blocks[1]


def test_split_blocks_mismatch_returns_what_it_got():
    """段数对不上时不猜, 返回实际块数让上层警告。"""
    blocks = RW_MOD._split_output_blocks("A。\n\nB。\n\nC。", 2)
    assert len(blocks) == 3


# ============================================================
# prompt 组装
# ============================================================

def test_rewrite_prompt_contains_everything():
    from agent.prompts_survey import build_paragraph_rewrite_prompt
    paras = split_paragraphs("A。\n\nB。\n\nC。")
    p = build_paragraph_rewrite_prompt(
        topic="T", section_type="progress", section_title="进展",
        outline_section="- 要点1",
        current_structure=render_outline_for_rewrite(paras, [1]),
        context=extract_context(paras, [1]),
        target_idxs=[1],
        papers=[{"paper_id": "p1", "title": "t", "core_method": "m", "core_contrib": "c"}],
        facts=[{"paper_id": "p1", "fact_type": "result", "claim": "10x 改进", "support_quote": "q"}],
        user_instruction="补个反例",
    )
    assert "10x 改进" in p, "fact 素材要进去"
    assert "补个反例" in p, "用户要求要进去"
    assert "第 1 段" in p, "目标段号要进去"
    assert "第 0 段" in p, "上下文段要进去"
    assert "{pool_text}" not in p and "{context}" not in p
