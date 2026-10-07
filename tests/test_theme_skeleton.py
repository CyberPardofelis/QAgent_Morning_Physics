"""
test_theme_skeleton.py — 全局骨架 + 块内细分 (10-07 晚)

**这个 commit 在修什么**
10-07 真跑(经典阴影, 89309897a97c)暴露了一个比所有已修 bug 都严重的问题:

  10-03: fact 一次 LLM 看全部 -> LLM 站在全局视角出 5-6 个主题
         进主题地图的论文 **69%**, 每主题平均覆盖 **5.1 篇论文**
  10-07: 307 条 fact 切 6 块, 每块各自出 4-5 个主题 -> 27 个抢 8 个名额
         进主题地图的论文 **37%**, 每主题平均覆盖 **2.8 篇论文**

碎片化翻倍 = 综述从"领域结构"降级成"技术罗列"。用户原话:「没有对于领域的
整体理解, 只是罗列一些技术」。

根因是 `5958f70`(10-06)引入的分块: 它让每块只看到 1/N, 每块于是各自按
**技术**维度切主题, 而"按任务分类"(估计/层析/validation)必须要有全局视角
才做得到。

**两阶段的分工**
  阶段一 骨架: 全部 fact 的极简摘要(每条一行, 一条不丢) -> 5-8 个**任务**,
          每个带 belongs(可判定的归属线索)/ not_belongs(边界)
  阶段二 细分: 每块带着骨架归纳, 主题名必须逐字照抄骨架
  合并    : 块主题名先归一到骨架名再合并; 超上限时骨架主题保底

骨架失败不抛异常, 回退到今天之前的块级行为 —— 不比现状更差。
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

SKELETON = {
    "themes": [
        {"name": "经典阴影估计", "task": "从少量测量预测大量可观测量期望值",
         "belongs": "shadow norm / 样本复杂度 / Pauli 权重 / 预测性质",
         "not_belongs": "完整密度矩阵重建"},
        {"name": "经典阴影层析", "task": "把态重建出来",
         "belongs": "密度矩阵重构 / MPS / 二体约化",
         "not_belongs": "只算期望值不做重构"},
        {"name": "噪声下的经典阴影", "task": "在含噪器件上把偏差压下去",
         "belongs": "误差缓解 / 去偏置 / 鲁棒协议",
         "not_belongs": "理想态下的效率分析"},
    ],
    "structure_note": "前者是后者的特例",
}


def _facts(n: int, claim_len: int = 20) -> list[dict]:
    return [{"fact_id": f"f{i}", "subject": f"对象{i % 5}", "claim": "长" * claim_len,
             "paper_id": "p1", "fact_type": "finding"} for i in range(n)]


def _th(names: list[str], per: int = 3, chunk_tag: str = "") -> list[dict]:
    out = []
    for i, nm in enumerate(names):
        out.append({"name": nm, "statement": "判断", "notes": "",
                    "fact_ids": [f"{chunk_tag}c{i}_{j}" for j in range(per)],
                    "relations": []})
    return out


# ============================================================
# 阶段一: 摘要与骨架 prompt
# ============================================================

def test_digest_keeps_every_fact():
    """摘要丢的是 claim 的后半截, **不是 fact 的存在** —— 这是骨架能成立的前提。

    307 条里哪怕漏掉 30 条, 骨架就少了 10% 的语料视野, 而那 10% 恰恰可能是
    某个小众任务唯一的证据, 于是整个任务从骨架里消失。
    """
    fs = _facts(307)
    digest = OG._fact_digest(fs)
    lines = [ln for ln in digest.splitlines() if ln.strip()]
    assert len(lines) == len(fs), f"摘要丢了 fact: {len(fs)} -> {len(lines)}"
    for i in range(1, len(fs) + 1):
        assert f"[{i}]" in digest, f"第 {i} 条 fact 不在摘要里"


def test_digest_truncates_claim_not_the_line():
    fs = [{"fact_id": "f1", "subject": "快照",
           "claim": "A" * 500, "paper_id": "p"}]
    line = OG._fact_digest(fs).splitlines()[0]
    assert len(line) < 120, f"claim 没被截断, 整份摘要会撑爆 prompt: {len(line)}"
    assert "A" in line


def test_digest_growth_is_linear_in_fact_count():
    """摘要必须是 **O(fact 数) 且每条增量有界** —— 否则 fact 一多就把 prompt 撑爆。

    这比"比全文小几倍"更贴契约: 全文的大小取决于 support_quote 有多长(那是
    论文原文, 波动很大), 摘要的大小只取决于 fact 条数, 可预测。
    """
    fs = _facts(200)
    d200 = len(OG._fact_digest(fs))
    d201 = len(OG._fact_digest(fs + [_facts(1)[0] | {"fact_id": "f999"}]))
    assert d201 > d200, "加了 fact 摘要没变 —— 那条 fact 不在里面"
    assert (d201 - d200) < 120, f"单条 fact 的增量 {d201 - d200} 太大, 摘要没在截断"
    assert len(OG._fact_digest(fs)) < len(OG._facts_block(fs, {})), \
        "摘要比全文还大 —— 骨架阶段不如直接喂全文"


def test_skeleton_prompt_demands_task_dimension_not_technique():
    """prompt 必须明确禁止按技术分类 —— 这是整个修复的核心指令。

    不写这条, 模型会自然地按它看得见的技术切(浅电路/纠缠基/Pauli 不变酉),
    那正是 10-07 那版的产出。**这条断言防的是 prompt 被"简化"回去。**
    """
    p = OG.THEME_SKELETON_PROMPT
    assert "按**任务/问题**分类" in p
    assert "不要**按技术分类" in p or "不要**按技术分类" in p.replace("**", "**")
    assert "反例(按技术切)" in p, "缺少'按技术切'的反例 —— 只说正例模型仍会按技术切"
    assert "正例(按任务切)" in p


def test_skeleton_prompt_requires_membership_clues():
    """belongs 是阶段二判断归属的唯一依据, 缺了它骨架就只是几个空标题。

    断言用带星号的 `**belongs**` 而不是裸 `belongs` —— 裸串会被铁律 3 里的
    `**not_belongs**` 满足, 于是「删掉 belongs 要求」这个变异照样绿。
    踩过一次。
    """
    p = OG.THEME_SKELETON_PROMPT
    assert "**belongs**" in p, "没要求 belongs(带粗体标记的完整字段名)"
    assert "**not_belongs**" in p
    assert "可判定" in p, "没要求 belongs 可判定, 模型会写'相关的事实'"


# ============================================================
# 阶段二: 骨架注入
# ============================================================

def test_render_skeleton_lists_all_themes_with_clues():
    s = OG._render_skeleton(SKELETON)
    for t in SKELETON["themes"]:
        assert t["name"] in s
        assert t["belongs"] in s
        assert t["not_belongs"] in s
    assert "结构说明" not in s
    assert SKELETON["structure_note"] in s


def test_render_skeleton_demands_exact_name_copy():
    """必须写死「逐字照抄」—— 否则模型改个名, 合并层就把骨架撑破了。"""
    s = OG._render_skeleton(SKELETON)
    assert "必须与上面某一条完全一致" in s
    assert "逐字照抄" in s


def test_render_skeleton_empty_when_no_skeleton():
    assert OG._render_skeleton(None) == ""
    assert OG._render_skeleton({"themes": []}) == ""
    assert OG._render_skeleton({}) == ""


def test_skeleton_block_is_injected_into_chunk_prompt(monkeypatch):
    """真进 prompt, 不是只写了个没人调用的函数。"""
    prompts: list[str] = []

    def _fake(prompt, *a, **kw):
        prompts.append(prompt)
        if "结构骨架" in prompt or "极简摘要" in prompt:
            return json.dumps(SKELETON, ensure_ascii=False)
        return json.dumps({"themes": [{"name": "经典阴影估计", "statement": "",
                                       "fact_ids": ["f1"], "notes": "", "relations": []}],
                           "narrative_arc": [], "gaps": []}, ensure_ascii=False)

    import agent.tools.llm as llm_mod
    monkeypatch.setattr(llm_mod, "_call_llm", _fake)
    monkeypatch.setattr(OG, "THEME_CHUNK_FACTS", 40)
    OG.organize_facts_into_themes(_facts(200))

    chunk_prompts = [p for p in prompts if "结构骨架" not in p and "极简摘要" not in p]
    assert chunk_prompts, "没有块内 prompt"
    for p in chunk_prompts:
        assert "【全局骨架" in p, "块内 prompt 里没有骨架"
        assert "经典阴影估计" in p


# ============================================================
# 名字归一: 合并层的接缝
# ============================================================

def test_align_exact_name_is_a_noop():
    """本来就叫骨架名 = 没发生改名, **不能**算进映射表。

    算进去的话告警会说「3 个叫法已归并」而其实一个都没并 —— 虚高比不说更糟。
    """
    out, m = OG._align_to_skeleton(_th(["经典阴影估计"]), SKELETON)
    assert m == {}, f"同名的也算改写了: {m}"
    assert out[0]["name"] == "经典阴影估计"


def test_align_strips_parenthetical_qualifier():
    out, m = OG._align_to_skeleton(_th(["经典阴影估计（通用）"]), SKELETON)
    assert m == {"经典阴影估计（通用）": "经典阴影估计"}
    assert out[0]["name"] == "经典阴影估计"


def test_align_handles_abbreviation():
    """模型把「经典阴影层析」缩写成「层析」是常事, 得能接住。"""
    out, _ = OG._align_to_skeleton(_th(["层析"]), SKELETON)
    assert out[0]["name"] == "经典阴影层析"


def test_align_refuses_ambiguous_generic_name():
    """「经典阴影」同时是三个骨架名的子串 —— 拿哪个都是猜, **不许硬塞**。

    错误合并两个不同的任务, 比多留一个块级主题糟得多。
    """
    out, m = OG._align_to_skeleton(_th(["经典阴影"]), SKELETON)
    assert out[0]["name"] == "经典阴影", f"泛化名被硬塞进了某个骨架: {m}"
    assert m == {}


def test_align_uses_clue_words_as_last_resort():
    """中文 belongs 是完整短语, 一个 4 字短语命中就够 —— 见 _align_to_skeleton 里
    的分级说明。这条防的是把「>=2 个命中」写回去(对中文几乎永远不成立)。"""
    th = _th(["误差缓解路线"])
    th[0]["statement"] = "通过 PEC 与零噪声外推处理误差缓解"
    out, m = OG._align_to_skeleton(th, SKELETON)
    assert out[0]["name"] == "噪声下的经典阴影", f"线索词没兜住: {m}"


def test_align_ignores_short_clue_words():
    """3 字以内的词太泛, 命中也不归类 —— 只用 >= 4 字的特异词。"""
    th = _th(["某某东西"]); th[0]["statement"] = "只提到去偏置"
    out, m = OG._align_to_skeleton(th, SKELETON)
    assert m == {}, f"3 字线索词就归类了: {m}"


def test_align_preserves_fact_ids():
    """归一只许改名字。fact_ids 被动过 = 丢数据。"""
    th = _th(["层析"], per=4)
    ids = list(th[0]["fact_ids"])
    out, _ = OG._align_to_skeleton(th, SKELETON)
    assert out[0]["fact_ids"] == ids


# ============================================================
# 合并: 跨块同名真的并起来
# ============================================================

def test_merge_collapses_per_chunk_naming_into_skeleton():
    """**这是这次修复的核心断言**。

    10-07 那版的实测形态: 6 块各出 4-5 个块级主题, 跨块主题名复用率 0%,
    合并出 27 个主题抢 8 个名额。有骨架之后, 块块叫法不同也必须并成骨架主题。
    """
    results = [
        {"themes": _th(["经典阴影估计（通用）"]), "narrative_arc": [], "gaps": [], "warnings": []},
        {"themes": _th(["经典阴影估计"]), "narrative_arc": [], "gaps": [], "warnings": []},
        {"themes": _th(["层析"]), "narrative_arc": [], "gaps": [], "warnings": []},
        {"themes": _th(["噪声缓解"]), "narrative_arc": [], "gaps": [], "warnings": []},
    ]
    m = OG._merge_chunk_results(results, 8, skeleton=SKELETON)
    names = [t["name"] for t in m["themes"]]
    assert "经典阴影估计" in names
    assert "经典阴影层析" in names
    assert len(names) == len(set(names)), "合并后还有重名"
    assert len(m["themes"]) <= 3, f"4 块应塌成 <=3 个骨架主题, 实际 {names}"


def test_merge_unions_fact_ids_across_chunks():
    a = _th(["经典阴影估计"], per=3, chunk_tag="a")
    b = _th(["经典阴影估计"], per=2, chunk_tag="b")
    results = [{"themes": a, "narrative_arc": [], "gaps": [], "warnings": []},
               {"themes": b, "narrative_arc": [], "gaps": [], "warnings": []}]
    m = OG._merge_chunk_results(results, 8, skeleton=SKELETON)
    est = next(t for t in m["themes"] if t["name"] == "经典阴影估计")
    assert len(est["fact_ids"]) == 5, f"并集丢了 fact: {est['fact_ids']}"
    assert len(set(est["fact_ids"])) == 5, "并集里有重复"


def test_merge_announces_renames():
    """归并了多少个叫法必须说出来 —— 用户要能看出模型不听话的程度。"""
    results = [{"themes": _th(["经典阴影估计（通用）"]),
                "narrative_arc": [], "gaps": [], "warnings": []}]
    m = OG._merge_chunk_results(results, 8, skeleton=SKELETON)
    assert any("对齐到全局骨架" in w for w in m["warnings"]), m["warnings"]


def test_merge_without_skeleton_keeps_old_chunk_rotation():
    """**退路**: 骨架失败时不能比现状更差, 得退回块轮转。"""
    results = [{"themes": _th([f"块{i}主题{j}" for j in range(4)], chunk_tag=f"{i}"),
                "narrative_arc": [], "gaps": [], "warnings": []} for i in range(1, 4)]
    m = OG._merge_chunk_results(results, 4)
    assert len(m["themes"]) == 4
    assert {t["name"] for t in m["themes"]} >= {"块1主题0", "块2主题0", "块3主题0"}


# ============================================================
# 超上限时骨架主题保底
# ============================================================

def test_skeleton_themes_survive_truncation():
    """砍主题时**骨架任务必须先活**。

    轮转的语义是"每块保一个", 而块是技术视角的切片 —— 保证每块活一个,
    等于保证图里塞满块级视角, 全局结构反而被挤掉。骨架才是全局结构本身。

    ⚠️ 测试数据要满足两个条件, 否则这条会**因为没走到截断分支**而绿:
      1. 骨架主题数 > max_themes —— 否则 8 个 <= 上限, 压根不截断
      2. 每块的**块独有主题 fact 数远多于骨架主题** —— 否则块轮转(按 fact 数
         降序取最大)会挑中骨架主题, 两种策略答案一样, 保底与否都绿。
      这两点都踩过 —— 第一版还让骨架名在每块都排第一, 结果两种策略输出相同。
    """
    sk5 = {"themes": [
        {"name": f"任务{i}", "task": "解什么", "belongs": f"线索{i}甲乙丙", "not_belongs": "别混"}
        for i in range(1, 6)]}
    results = []
    for i in range(1, 4):
        # 每块只带 2 个骨架任务(2 条 fact), 外加一个 12 条 fact 的块独有主题
        block = _th([f"任务{i}", f"任务{i + 1}"], per=2, chunk_tag=f"{i}")
        block += _th([f"块{i}独有主题"], per=12, chunk_tag=f"{i}")
        results.append({"themes": block, "narrative_arc": [], "gaps": [], "warnings": []})

    m = OG._merge_chunk_results(results, 3, skeleton=sk5)
    names = [t["name"] for t in m["themes"]]
    assert len(names) == 3, names
    assert all(n.startswith("任务") for n in names), \
        f"名额被块级主题占了, 全局结构被挤掉: {names}"


def test_truncation_warning_says_skeleton_was_protected():
    results = [{"themes": _th([f"块{i}主题{j}" for j in range(4)], chunk_tag=f"{i}"),
                "narrative_arc": [], "gaps": [], "warnings": []} for i in range(1, 5)]
    m = OG._merge_chunk_results(results, 4, skeleton=SKELETON)
    w = [x for x in m["warnings"] if "超过上限" in x]
    assert w, m["warnings"]
    assert "骨架" in w[0], f"截断告警没说骨架保底: {w[0]}"


# ============================================================
# 骨架落盘
# ============================================================

def test_skeleton_is_persisted_in_result():
    """骨架必须落进 theme_map, 否则复现/续跑时看不出这版图是怎么来的。"""
    results = [{"themes": _th(["经典阴影估计"]), "narrative_arc": [],
                "gaps": [], "warnings": []}]
    m = OG._merge_chunk_results(results, 8, skeleton=SKELETON)
    assert m["skeleton"]["themes"], "骨架没落进返回值"
    assert len(m["skeleton"]["themes"]) == 3


def test_merge_result_has_skeleton_key_even_without_skeleton():
    """字段不能有时有时无 —— 下游读 `theme_map['skeleton']` 时不能 KeyError。"""
    results = [{"themes": _th(["A"]), "narrative_arc": [], "gaps": [], "warnings": []}]
    m = OG._merge_chunk_results(results, 8)
    assert "skeleton" in m
    assert not m["skeleton"]


# ============================================================
# 骨架失败必须回退而不是崩
# ============================================================

def test_build_skeleton_returns_empty_on_llm_error(monkeypatch):
    import agent.tools.llm as llm_mod

    def _boom(*a, **kw):
        raise RuntimeError("API 挂了")
    monkeypatch.setattr(llm_mod, "_call_llm", _boom)
    assert OG._build_skeleton(_facts(10), []) == {}


def test_build_skeleton_returns_empty_on_api_error_string(monkeypatch, capsys):
    """`_call_llm` 失败时**不抛异常**, 返回一段错误提示串 —— 得认出来。

    断言的是**告警文案**而不是返回值: 错误串当然也解析不出 JSON, 所以返回值
    两种路径都是 {}。`_is_llm_error_response` 的真正价值是**归因说对** ——
    不认的话会报成"骨架 JSON 解析失败", 那是把 API 故障说成模型输出的问题,
    排查方向会跑偏(这跟 10-05 那条成文路径的病一模一样)。
    """
    import agent.tools.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_call_llm",
                        lambda *a, **kw: "❌ MiniMax 请求超时（APITimeoutError）：xx")
    assert OG._build_skeleton(_facts(10), []) == {}
    out = capsys.readouterr().out
    assert "请求超时" in out, f"没把 API 故障的原因说出来: {out!r}"
    assert "JSON 解析失败" not in out, \
        "把 API 故障报成了 JSON 解析失败 —— 排查方向会指向 prompt/模型而不是网络"


def test_build_skeleton_returns_empty_on_bad_json(monkeypatch):
    import agent.tools.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_call_llm", lambda *a, **kw: "这不是 JSON")
    assert OG._build_skeleton(_facts(10), []) == {}


def test_build_skeleton_parses_real_output(monkeypatch):
    import agent.tools.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_call_llm",
                        lambda *a, **kw: json.dumps(SKELETON, ensure_ascii=False))
    sk = OG._build_skeleton(_facts(10), [])
    assert len(sk["themes"]) == 3
    assert sk["themes"][0]["name"] == "经典阴影估计"
    assert sk["themes"][0]["belongs"]


def test_build_skeleton_dedupes_repeated_names(monkeypatch):
    import agent.tools.llm as llm_mod

    dup = {"themes": [{"name": "A", "task": "t", "belongs": "b", "not_belongs": "n"},
                      {"name": "A", "task": "t2", "belongs": "b2", "not_belongs": "n2"}]}
    monkeypatch.setattr(llm_mod, "_call_llm",
                        lambda *a, **kw: json.dumps(dup, ensure_ascii=False))
    assert len(OG._build_skeleton(_facts(10), [])["themes"]) == 1


def test_build_skeleton_on_empty_facts():
    assert OG._build_skeleton([], []) == {}