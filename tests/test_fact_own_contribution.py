# -*- coding: utf-8 -*-
"""fact 抽取层的「本篇独有性」约束 (10-08)

背景
----
命题层实测出现过重归属: q1 陈述是"由 Huang/Kueng/Preskill 提出的**原始**经典
阴影协议", 实测挂了 20 篇论文 —— 其中 16 篇的 fact 是在**介绍经典阴影是什么**
(定义型/背景型), 不是在建立这个协议。

根因在抽取层: 现状 prompt 只要求 fact "自足、去语境化", 并明令主语不能是
"本文/该工作/作者"(铁律 2)。于是**归属信息被主动删掉了** —— 剩下的 fact
既看不出是本文的贡献, 也看不出是转述他人, 只剩一段中立的技术描述。
实测 625 条 fact 里只有 13 条残留"本文/该工作"类主语, 铁律 2 执行得很干净。

但被删掉的归属信息是可以**用另一种方式**留下的: 不写进 claim 文本(那会
破坏自足性), 而是作为**抽取时的取舍标准** —— 优先抽这篇论文自己的贡献。

A/B 实测(真实论文, 各 8 篇 × 2 组, 24 次调用):
              A 现状 prompt      B 加本篇独有性铁律
    S 建立者    21 条 fact       22 条(零损失)
    U 仅使用    55 条 fact       43 条(-22%)

减掉的正是那批定义型/背景型 fact。S 组零误伤是关键的 —— 如果建立者被误伤,
说明这条约束会削掉真正该留的内容。

本测试锁定:
  1. prompt 里必须有这条约束, 且写明正反例
  2. 不得改动 JSON schema(改了就和历史 checkpoint 不兼容, 且属于夹带)
  3. 不得改动铁律 2 的主语要求(那是自足性的地基, 不能为了保留归属而破坏它)
"""
from __future__ import annotations

import inspect

import pytest

from agent import prompts_survey as PS


PROMPT = PS.SURVEY_FACT_PROMPT


# ---------------------------------------------------------------- 约束存在

def test_prompt_has_own_contribution_constraint():
    """必须有这条约束。"""
    assert "本篇独有性" in PROMPT, \
        "fact 抽取 prompt 缺少「本篇独有性」约束 —— 背景/定义型 fact 会被当成支撑出处"


def test_prompt_constraint_states_positive_criterion():
    """要给出**可操作的**正向判据, 不能只是口号。"""
    assert ('判据: 把这句话给一个只读过摘要的读者看, 他能不能说"这是这篇论文报告的"?'
            in PROMPT), "判据必须逐字保留(反向验证 M4 换成 '(无判据)' 时靠这条接住)"


def test_prompt_constraint_has_negative_examples():
    """必须有反例 —— 实测纯抽象表述(只说不要转述)模型会无视。"""
    assert '❌ "某某方法可以做到…" / "经典方法要求…" / "某技术是一种高效方法"(纯定义)' \
        in PROMPT, "反例必须逐字保留(反向验证 M2 把它换成 '(无反例)' 时靠这条接住)"


def test_prompt_constraint_allows_fewer_facts():
    """必须显式允许抽得比 4 条少。

    这是防误伤的关键 —— 现状 prompt 写着"数量: 4-12 条", 越界就等于质量差。
    不给这条台阶, 模型只会硬凑到 4 条, 转述型 fact 照样被塞回来。
    """
    assert "⚠️ **抽不到就少抽** —— 本条优先于上面第 5 条的 4-12 条下限" in PROMPT, \
        "数量豁免必须逐字保留(反向验证 M3 删掉它时靠这条接住)"


# ------------------------------------------------------- schema 不能被改

def test_json_schema_unchanged():
    """输出字段必须仍是 subject/fact_type/claim/support_quote。

    改 schema = 和历史 checkpoint 不兼容 = 属于夹带, 不在这次授权范围里。
    """
    for field in ('"subject"', '"fact_type"', '"claim"', '"support_quote"'):
        assert field in PROMPT, f"输出字段 {field} 不能动"


def test_fact_types_unchanged():
    """fact_type 枚举不能变 —— 下游多处按这些值分支。

    正反两面都要查: 光查"老值还在"不够, 有人**新增**一个值(比如 own)同样会
    打穿下游 —— 反向验证 M6 就是新增一个值, 只查老值会假绿。
    """
    import re
    block = PROMPT[PROMPT.find("【fact 类型】"):PROMPT.find("【subject 字段")]
    declared = set(re.findall(r"^\s*-\s*([a-z_]+)\s*:", block, re.M))
    expected = {"method", "result", "conclusion", "limitation", "hypothesis"}
    assert declared == expected, \
        f"fact_type 枚举必须恰好是 {sorted(expected)}, 实际 {sorted(declared)}"


def test_no_new_field_added():
    """不允许新增字段(比如 attribution)—— 用户明确否决了记归属字段这条路。"""
    for forbidden in ('"attribution"', '"is_own"', '"ownership"', '"contribution_type"'):
        assert forbidden not in PROMPT, \
            f"不能新增归属字段 {forbidden} —— 走 prompt 约束, 不动数据结构"


# ------------------------------------------------------ 地基不能被拆

def test_self_contained_rule_preserved():
    """铁律 2(主语不能是本文/该工作/作者)必须保留。

    这是 fact 自足性的地基 —— 拿去正文里"本文"没有指代对象。
    有人会想"那把归属写进 claim 里不就行了", 那会直接破坏它, 不允许。
    """
    assert "主语必须是方法/概念/现象本身" in PROMPT, "铁律 2 被破坏了"
    assert "不是" in PROMPT and "本文" in PROMPT


def test_build_fact_prompt_still_renders():
    """模板占位符没被破坏(加了一段带花括号的话最容易踩这个坑)。"""
    out = PS.build_fact_prompt(
        paper_id="10.1103/x", title="T", year="2024", journal="PRL",
        core_method="m", core_contrib="c", abstract="a", outlook="o")
    assert "10.1103/x" in out and "T" in out
    # 不能残留未替换的占位符
    assert "{facts_block}" not in out
    for ph in ("{paper_id}", "{title}", "{abstract}", "{core_method}"):
        assert ph not in out, f"占位符 {ph} 没被替换 —— 模板里多了一层花括号"


def test_constraint_appears_in_rendered_prompt():
    """约束必须在真正发出去的那段 prompt 里, 不能只存在于某个没被 format 的常量。"""
    out = PS.build_fact_prompt(
        paper_id="p", title="t", year="2024", journal="j",
        core_method="m", core_contrib="c", abstract="a", outlook="o")
    assert "本篇独有性" in out, \
        "约束没进 build_fact_prompt 的输出 —— 说明它被加在了不会被 format 的地方"


def test_extract_facts_uses_build_fact_prompt():
    """接线护栏: 抽取走的就是 build_fact_prompt, 别绕过它。"""
    from agent.tools.surveys._actions import _extract_facts as EF
    assert "build_fact_prompt" in inspect.getsource(EF.action_extract_facts)


def test_survey_fact_prompt_version_bump():
    """改 prompt 必须升 PROMPT_VERSION —— 否则复跑和续跑会用同一个版本号
    跑出不同的 fact, checkpoint 就失去意义了。"""
    # `from agent.tools.surveys import survey_generate` 拿到的是模块里**同名的函数**
    # (被 @register_tool 装饰过), 不是模块 —— 常量取不到, 会假红。
    import importlib
    SG = importlib.import_module("agent.tools.surveys.survey_generate")
    assert SG.PROMPT_VERSION >= "2026-10-08", \
        "PROMPT_VERSION 没升 —— 抽 fact 的 prompt 变了却还用旧版本号"