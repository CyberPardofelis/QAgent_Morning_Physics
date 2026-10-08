# -*- coding: utf-8 -*-
"""写作层篇幅配额 (10-08)

真跑实测出的问题(workspace 22cb70452a69, 量子纠错编码理论)
-------------------------------------------------------
「解码算法设计」章正文末尾出现:

    把复杂度从 O(d²R) 降到 O(k), 在统一线…(后续略,以命题覆盖为准) [2605.17156]

这不是幻觉, 也不是被截断 —— `dropped_sentences=0`, 那句带引用, 完整过了
校验。**是模型自己发明的占位符。**

根因是两条要求打架:
  - prompt 铁律:「N 条命题必须全部写到, 一条都不能丢」
  - 配额:「本节目标约 300 字符」
实测该章 7 条命题(MWPM 代数化 / 纠偏率解码 / BP-OSD / 量身定制表面码 /
模拟单次解码 / 时域并行化 / Mamba 解码器), 每条都是一个独立技术点, 但只
给 300 字符。模型发现写不下, 又不能丢命题, 于是自造占位符糊过去。

**更糟的是它静默通过** —— 正文里完全看不出这里省略了内容, 读者只会以为
那篇论文只讲了个开头。

而配额小得离谱是算出来的:
    target_chars = max(1200, 263 * 0.32) = max(1200, 84) = 1200   <- 取下限
    per_theme    = max(300, 1200 // 7)       = max(300, 171) = 300
两处下限各吃掉了 14 倍和 1.8 倍。而该章实际写了 1334 字符 —— 配额根本不是
硬约束, 它只是给模型递了个「可以省略」的台阶。

本测试锁定:
  1. 配额要**按命题数兜底**(每条命题保底若干字符), 不能只按主题数摊
  2. 反推系数不能小到永远撞下限
  3. 占位符必须被当成"引不出处"处理 —— 宁可丢句, 不能糊弄
"""
from __future__ import annotations

import inspect

from agent.tools.surveys import survey_generate as _sg_mod
import importlib

SG = importlib.import_module("agent.tools.surveys.survey_generate")


def test_per_theme_chars_has_floor_per_claim():
    """每主题配额必须按**命题数**兜底, 不能只有 300 的绝对下限。

    7 条命题配 300 字符 = 每条 43 字符, 写不下 -> 模型自造占位符。
    """
    from agent.tools.surveys._organize import per_theme_target_chars

    # 7 条命题的章, 配额必须明显高于 300
    got = per_theme_target_chars(target_chars=1200, n_themes=1, n_claims=7)
    assert got >= 7 * 100, f"7 条命题只给 {got} 字符, 每条不足 100 —— 又会触发省略"

    # 命题越多, 配额必须跟着涨(至少线性)
    a = per_theme_target_chars(target_chars=1200, n_themes=1, n_claims=3)
    b = per_theme_target_chars(target_chars=1200, n_themes=1, n_claims=12)
    assert b > a, "命题数翻 4 倍, 配额不能纹丝不动"


def test_per_theme_chars_never_below_absolute_floor():
    from agent.tools.surveys._organize import per_theme_target_chars
    got = per_theme_target_chars(target_chars=0, n_themes=10, n_claims=1)
    assert got >= 300, "绝对下限 300 不能丢"


def test_total_target_chars_coefficient_not_too_small():
    """反推系数 0.32 实测永远撞 1200 下限(263 fact * 0.32 = 84)。

    系数必须调到"多数真跑的 fact 规模下不撞下限"。
    """
    from agent.tools.surveys._organize import derive_target_chars

    # 实测规模: 263 条可用 fact -> 旧公式给出 84(撞 1200 下限)
    got = derive_target_chars(n_usable_facts=263)
    assert got > 1200, f"263 条 fact 反推出 {got}, 仍然撞下限 -> 配额形同虚设"
    # 必须真的随规模增长, 而不是被 clamp 卡住不动
    a = derive_target_chars(n_usable_facts=263)
    b = derive_target_chars(n_usable_facts=500)
    assert b > a, f"fact 从 263 涨到 500, 配额没涨(仍被 clamp): {a} -> {b}"

    # 大规模也不能太夸张(别反过来把正文撑爆)
    big = derive_target_chars(n_usable_facts=2000)
    assert big < 30000, f"2000 条 fact 反推出 {big}, 太夸张了"


def test_write_prompt_forbids_placeholders():
    """prompt 必须明说: 写不下就写短句, 不许自造省略标记。

    逐字锁两处 —— 之前只查关键词, 反向验证 M8 整段删掉都没变红。
    """
    from agent.prompts_survey import THEME_WRITE_FROM_CLAIMS_PROMPT as P
    assert "【写不下怎么办 —— 10-08 实测踩过的坑】" in P, \
        "这段标题必须逐字保留 —— 它标明了这是实测踩过的坑, 不是随口加的规则"
    assert "不要写「后续略」「其余类似」「详见下节」这类话" in P, \
        "必须逐字点名要禁掉的那几个说法, 泛泛说'不要省略'模型会无视"
    assert "宁可少写细节" in P, "必须给出正面出路, 只给禁令模型会钻空子"


def test_placeholder_markers_are_rejected():
    """占位符必须被识别出来 —— 这是静默糊弄, 比丢句更糟。"""
    from agent.tools.surveys._organize import _placeholder_spans
    for bad in ("在统一线…(后续略,以命题覆盖为准)",
                "……(后续略)",
                "(以命题覆盖为准)",
                "其余略"):
        assert _placeholder_spans(bad), f"没识别出占位符: {bad}"
    # 正常句子不能被误伤
    ok = "把复杂度从 O(d²R) 降到 O(k), 在统一线性代数框架下实现"
    assert not _placeholder_spans(ok), f"正常句子被误判成占位符: {ok}"


def test_phase_write_uses_new_quota_helper():
    """接线: _phase_write 必须走新的配额算法。"""
    src = inspect.getsource(SG._phase_write)
    assert "per_theme_target_chars(" in src, \
        "_phase_write 还在用旧的 max(300, target//n)"
    # 光有名字不够 —— 反向验证 M9 把调用换成 `(lambda *a, **k: 300)(...)`,
    # 函数名照样在源码里, 测试却全绿。所以要断言**真调用**的形态。
    assert ": per_theme_target_chars(" in src, \
        "per_theme_target_chars 只是被 import 了, 没有真的调用"
    # 配额必须**逐主题**传下去, 而不是取个全局单值 —— 各主题命题数不同,
    # 用单值会让命题多的章再次写不下(实测「解码算法设计」7 条命题按 1 条配额写)。
    assert "quota.get(name" in src, \
        "配额得按主题名取, 用全局单值等于把命题多的章按最少命题的配额写"
    assert "target_chars=per_theme_chars" not in src, \
        "还在用全局单值配额(那是这次要修的 bug 本身)"


def test_write_loop_wires_placeholder_stripper():
    """链路护栏: 剔除器必须接在 `write_theme_narrative_from_claims` 里面。

    前两轮反向验证(M6/M7)都证明: 只测 `_strip_placeholder_sentences` 这个
    纯函数是**不够**的 —— 把它从写作链路里摘掉, 测试照样全绿, 而占位符
    又会静默回到正文里。所以这里直接查真实函数的源码。
    """
    from agent.tools.surveys._organize import write_theme_narrative_from_claims
    src = inspect.getsource(write_theme_narrative_from_claims)
    assert "_strip_placeholder_sentences(body)" in src, \
        "剔除器没接进写作链路 —— 纯函数存在但没人调用"
    # 计数必须累加进 dropped_total, 否则丢句了用户也看不见
    assert "dropped_total += ph" in src, \
        "剔除的句子数没计入 dropped_total —— 等于静默删内容"


def test_narrative_text_strips_placeholder_sentences():
    """端到端: 带占位符的句子要被剔除, 并计入 dropped_sentences。

    参照既有的 `_render_claim_citations_or_drop` 规则 —— 引不出处的句子
    直接丢。这里是同一类问题的另一半。
    """
    from agent.tools.surveys._organize import _strip_placeholder_sentences
    text = ("第一句正常内容 [p1][p2]。\n"
            "第二句带占位符 在统一线…(后续略,以命题覆盖为准) [p3]。\n"
            "第三句也正常 [p4]。")
    kept, dropped = _strip_placeholder_sentences(text)
    assert "后续略" not in kept, "占位符没有被剔除"
    assert "第一句正常内容" in kept and "第三句也正常" in kept, "正常句子被误删"
    assert dropped == 1, f"dropped 应为 1, 实际 {dropped}"