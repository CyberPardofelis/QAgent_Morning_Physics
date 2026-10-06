"""
tests/test_survey_gap_fill.py — 覆盖空白回补(10-06)

**这个文件为什么存在**
`organize` 路线是直线管线, **没有 ReAct 循环** —— `survey_generate` 开头
`phase == "organize"` 那个 early return 直接把下面 900 行的循环跳过了。而老路线
那套 `section_gaps -> action_search_more` 只认 `write_section` 里的
`<!-- QAgent: missing_queries -->` 标记, 管不到主题地图这条路。

结果就是: Phase B 归纳出的 31 条空白全部只能进渲染文件, 一条都没转化成
"再检索一轮"。跑完还是那 39 篇。

这里锁的是"别把回补改回死路", 以及三个真实踩过的坑:
  - `enable_thinking=False` 挡不住 think, 那部分照样吃 max_tokens
  - 模型输出会混进 **NUL 字节**(`Toffoli` -> `Toff\\x00oli`), strict=False 治不了
  - 模型会**改写**空白原文, 必须靠 idx 对回原始文本

测试取向: 用**真实跑出来的输出**做表征测试(见 _REAL_*), 不用编出来的理想结构。
"""
from __future__ import annotations

import importlib
import json
import pathlib

import pytest

QE = importlib.import_module("agent.tools.surveys._pool._query_expand")
SG = importlib.import_module("agent.tools.surveys.survey_generate")
CFG = importlib.import_module("agent.tools.config")

# 真实跑出来的那次 verdict 调用**原始输出**(中性原子综述, 8538 字符)。
# 保留它是因为它是 NUL 字节污染的**唯一物证**: 里面有 think 前缀 + 大量 \x00,
# 手工重构不出这种形态 —— 这正是本轮前两次猜错方向的原因。
_RAW_PATH = pathlib.Path(__file__).resolve().parents[1] / "output" / "_gap_fill_raw.txt"
_REAL_RAW = _RAW_PATH.read_text(encoding="utf-8") if _RAW_PATH.exists() else None


def _parse_real() -> dict:
    """**必须在用例里现算**, 不能在 import 时算成模块级常量。

    踩过的坑: 模块级 `_REAL_PARSED = _parse_gap_verdict(raw)` 看着一样, 但源码里
    那个 `replace("\\x00", "")` 被改坏时, pytest 早就 import 完了、值已缓存, 测试
    照样绿 —— 反向验证 G1 就是这么"改坏了没咬到"的。表征测试必须每次现算。
    """
    return QE._parse_gap_verdict(_REAL_RAW)


def _fake_llm(payload, box=None):
    import json as _json

    def _fn(prompt, **kw):
        if box is not None:
            box["prompt"] = prompt
            box["kw"] = kw
        return payload if isinstance(payload, str) else _json.dumps(payload, ensure_ascii=False)
    return _fn


# ============================================================
# 解析层: NUL / 截断 / 控制字符
# ============================================================

def test_parse_strips_nul_bytes():
    """NUL 字节必须被洗掉。

    10-06 实测 6 次调用 6 次中招, 长相是:
        "Toff\\x00oli\\x00\\x00 GHZ global addressing"
    `json.loads(strict=False)` **治不了**: strict=False 只放行 \\n \\t \\r,
    NUL 照样报 "Expecting value"。踩了两轮才定位到。
    """
    raw = ('<think>t</think>{"gaps": [{"idx": 1, "score": 9, '
           '"query": "Toff\x00oli GHZ global addressing neutral atom"}]}')
    got = QE._parse_gap_verdict(raw)
    assert got is not None, "NUL 字节让整个解析失败"
    assert got["gaps"][0]["query"] == "Toffoli GHZ global addressing neutral atom"


def test_parse_tolerates_raw_control_chars():
    """字符串里的裸换行/制表用 strict=False 放行。"""
    raw = '<think>t</think>{"gaps": [{"idx": 1, "score": 5, "query": "a b\n c\td"}]}'
    got = QE._parse_gap_verdict(raw)
    assert got is not None and got["gaps"][0]["query"].startswith("a b")


def test_parse_flags_unclosed_think():
    """think 没闭合 = 输出被截断在思考里, 后面根本没有 JSON。

    不单独识别的话, 贪婪的 `\\{.*\\}` 会从 think 散文里某个 `{` 一路吃到结尾,
    上层报成"找不到 JSON" —— 那是误导, 真实原因是截断。
    """
    got = QE._parse_gap_verdict('<think> 让我想想 {"a": 1')
    assert got is not None and got.get("__truncated_think__") is True


def test_parse_returns_none_on_garbage():
    assert QE._parse_gap_verdict("") is None
    assert QE._parse_gap_verdict("完全没有 JSON 的一段话") is None


def test_gap_fill_budget_big_enough_for_think():
    """max_tokens 必须够 think 吃完还有余。

    6000 时实测返回 23635 字符里**一个 JSON 都没有**(think 全吃光),
    静默变成"判定失败"。
    """
    import pathlib as _pl
    src = _pl.Path(QE.__file__).read_text(encoding="utf-8")
    assert "max_tokens=20000" in src, \
        "max_tokens 回到 20000 以下会被 think 吃光(实测 6000 必失败)"


# ============================================================
# 打分选择: 靠 idx 对齐, 靠分数排序
# ============================================================

def test_idx_aligns_to_original_gap_text(monkeypatch):
    """模型改写了 gap 文本, 但 idx 正确 -> 必须对回**原始**那条。

    10-06 实测: 不按 idx 对齐时, 改写版进 ranked、原文又被当成"模型漏回"
    补进 unfillable, 31 条空白一度数出 59 条"不回补"。

    **低分排在前面是故意的**: 模型输出是按 idx 顺序的, 所以高分那条必然不在
    第一位 —— 只有这样才能验出"按分数排序"这条护栏(否则去掉排序照样绿)。
    """
    real = ["第一条空白", "第二条空白", "第三条空白"]
    payload = {"gaps": [
        {"idx": 1, "score": 3, "query": "query for first gap neutral atom"},
        {"idx": 2, "score": 9, "query": "query for second gap neutral atom"},
        {"idx": 3, "score": 5, "query": "query for third gap neutral atom"},
    ]}
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fake_llm(payload), raising=False)
    r = QE.gaps_to_queries("neutral atom quantum computing", real, max_queries=1)
    assert r["ok"] is True
    assert len(r["fillable"]) == 1, "max_queries=1 必须只放行 1 条(成本闸门)"
    assert r["fillable"][0]["gap"] == "第二条空白", \
        "idx=2 必须对回原始第二条, 且分数最高的那条胜出"
    assert r["fillable"][0]["score"] == 9


def test_gaps_model_omitted_still_land_in_unfillable(monkeypatch):
    """模型漏回的空白必须落进 unfillable —— 空白不能凭空消失。

    判定失败 / 漏回是常事(31 条里漏几条很常见)。漏掉的那些如果直接丢, 用户
    在清单里就再也看不到它们, 等于静默说谎: 看着像"没有空白", 其实是"没人管"。
    """
    real = ["甲空白", "乙空白", "丙空白", "丁空白"]
    # 模型只回了第 1、3 条, 漏了乙和丁
    payload = {"gaps": [
        {"idx": 1, "score": 9, "query": "first neutral atom query alpha"},
        {"idx": 3, "score": 9, "query": "third neutral atom query gamma"},
    ]}
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fake_llm(payload), raising=False)
    r = QE.gaps_to_queries("neutral atom quantum computing", real, max_queries=4)
    assert set(r["unfillable"]) == {"乙空白", "丁空白"}, \
        f"漏回的两条必须留在不回补里, 实得 {r['unfillable']}"


def test_duplicate_input_gaps_do_not_multiply_output(monkeypatch):
    """theme_map 里出现重复空白文本时, 输出不能跟着重复。

    Phase B 的 gaps 是模型自由文本, 同一句被写两遍完全可能。不去重的话
    输出清单会凭空膨胀 —— 这正是 10-06 数出"31 条空白 / 59 条不回补"的机制。
    """
    real = ["甲空白", "甲空白"]        # 模型把同一句写了两遍
    payload = {"gaps": []}             # 判定里也没提到任何一条
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fake_llm(payload), raising=False)
    r = QE.gaps_to_queries("neutral atom quantum computing", real, max_queries=3)
    assert r["unfillable"].count("甲空白") == 1, \
        f"同一条空白只能出现一次, 实得 {r['unfillable']}"


def test_falls_back_to_text_match_without_idx(monkeypatch):
    """没有 idx 字段时退回按文本匹配(旧格式/其他模型也能用)。"""
    real = ["甲空白", "乙空白"]
    payload = {"gaps": [{"gap": "乙空白", "score": 9,
                         "query": "Rydberg blockade gate neutral atom"}]}
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fake_llm(payload), raising=False)
    r = QE.gaps_to_queries("neutral atom quantum computing", real, max_queries=1)
    assert r["fillable"][0]["gap"] == "乙空白"


def test_unfillable_plus_fillable_equals_input(monkeypatch):
    """回补 + 不回补 **必须等于**输入条数, 一条不多一条不少。

    重复计数会让空白清单凭空膨胀; 漏计会让空白凭空消失 —— 两种都是静默说谎。
    """
    real = [f"空白{i}" for i in range(1, 11)]
    payload = {"gaps": [{"idx": i, "score": (9 if i <= 2 else 1),
                         "query": f"neutral atom query number {i}"}
                        for i in range(1, 11)]}
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fake_llm(payload), raising=False)
    r = QE.gaps_to_queries("neutral atom quantum computing", real, max_queries=2)
    assert len(r["fillable"]) == 2
    total = r["fillable"] + [{"gap": g} for g in r["unfillable"]]
    assert len(total) == len(real)
    assert len({x["gap"] for x in total}) == len(real), "同一条空白被重复计数"
    assert {x["gap"] for x in total} == set(real), "有空白凭空消失或出现"


def test_dup_gap_keeps_highest_score(monkeypatch):
    """模型重复输出同一条时保留最高分, 不能出现两条。"""
    real = ["甲"]
    payload = {"gaps": [
        {"idx": 1, "score": 3, "query": "first neutral atom query alpha"},
        {"idx": 1, "score": 9, "query": "second neutral atom query beta"},
    ]}
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fake_llm(payload), raising=False)
    r = QE.gaps_to_queries("neutral atom quantum computing", real, max_queries=3)
    assert len(r["fillable"]) == 1
    assert r["fillable"][0]["score"] == 9


def test_chinese_query_is_dropped(monkeypatch):
    """检索式必须是英文 —— 说了英文却给中文, 等于拿中文去搜 arxiv。"""
    payload = {"gaps": [{"idx": 1, "score": 10, "query": "中性原子 量子计算"}]}
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _fake_llm(payload), raising=False)
    r = QE.gaps_to_queries("neutral atom quantum computing", ["甲"], max_queries=3)
    assert r["fillable"] == [], "中文检索式必须被挡掉"
    assert r["unfillable"] == ["甲"], "挡掉之后这条仍要留在不回补里"


def test_no_why_field_leaks_to_user():
    """输出里不能有 why 字段。

    10-06 实测: 模型的"为什么补不了"会写出"这方向的文献存在"这种跟标签
    自相矛盾的句子, 出现在最终综述里比没有理由更糟。
    """
    import pathlib as _pl
    src = _pl.Path(QE.__file__).read_text(encoding="utf-8")
    body = src.split("def gaps_to_queries", 1)[1]
    assert '"why"' not in body, "gaps_to_queries 不该再产出 why 字段"
    assert "fillable" in body and "score" in body, "应该产出 score 而不是 why"


# ============================================================
# 失败路径: 必须响亮, 不能静默吞掉
# ============================================================

def test_fails_loudly_when_llm_raises(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("socket closed")
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _boom, raising=False)
    r = QE.gaps_to_queries("t", ["甲", "乙"], max_queries=2)
    assert r["ok"] is False
    assert r["fillable"] == [] and r["unfillable"] == []
    assert "socket closed" in r["error"]


def test_retries_once_on_unparseable(monkeypatch):
    """第一次吐半截、第二次正常 -> 必须成功(实测同一 prompt 输出格式不稳定)。"""
    calls = []

    def _flaky(prompt, **kw):
        calls.append(1)
        if len(calls) == 1:
            return "<think> 没说完就被截断了"
        return '{"gaps": [{"idx": 1, "score": 9, "query": "Rydberg gate neutral atom"}]}'

    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _flaky, raising=False)
    r = QE.gaps_to_queries("neutral atom quantum computing", ["甲"], max_queries=2)
    assert r["ok"] is True, "重试一次还没成功才算失败"
    assert len(calls) == 2


def test_gives_up_after_two_attempts(monkeypatch):
    n = []

    def _always_bad(prompt, **kw):
        n.append(1)
        return "<think> 只有思考"
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _always_bad, raising=False)
    r = QE.gaps_to_queries("t", ["甲"], max_queries=1)
    assert r["ok"] is False
    assert len(n) == 2, "最多重试 1 次(共 2 次调用), 别无限重试烧额度"
    # 断言**具体原因**, 不能只断言 error 非空 —— `{**empty}` 自带 "error": "未调用",
    # 只判非空的话把 last_err 抹掉照样绿(反向验证 G13 就是这么溜过去的)
    assert "思考" in r["error"], f"失败原因没带出来: {r['error']!r}"


def test_empty_gaps_short_circuits(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("没有空白时不该调 LLM")
    monkeypatch.setattr(importlib.import_module("agent.tools.llm"),
                        "_call_llm", _boom, raising=False)
    assert QE.gaps_to_queries("t", [], max_queries=2)["ok"] is False


# ============================================================
# 标注: 试过的要标出来
# ============================================================

def test_annotate_marks_filled_and_unfillable():
    """回补跑完之后, 空白清单必须能看出哪些试过、哪些判了补不上。

    不标的话清单跟没跑过一样, 用户分不清"试过补不上"和"没人管"。
    """
    out = SG._annotate_gaps(["A", "B", "C"], {"A"})
    assert out[0] == "（已检索回补）A"
    assert out[1] == "（判定补不上）B"
    assert out[2] == "（判定补不上）C"


def test_annotate_never_shows_a_reason():
    """标注里不能夹带模型给的理由 —— 那个字段已被证明会自相矛盾。"""
    out = SG._annotate_gaps(["A"], set())
    assert "：" not in out[0] and ":" not in out[0], \
        "标注只能有固定前缀, 不能回显模型理由"


def test_annotate_empty_is_empty():
    assert SG._annotate_gaps([], set()) == []


# ============================================================
# 接线与开关
# ============================================================

def test_gap_fill_wired_after_theme_map_before_macros():
    """回补必须在**主题地图之后、宏观层之前**。

    顺序反了就是错的: 新论文会改变主题地图, 宏观层必须在**最终**地图上算,
    否则宏观板块的归属是基于一张马上就过期的地图。
    """
    src = pathlib.Path(SG.__file__).read_text(encoding="utf-8")
    # 注意必须找**调用点**("yield from _phase_gap_fill("), 不能找 "def _phase_gap_fill(" ——
    # 定义在 582 行, 调用在 856 行, 找 def 会把顺序判反
    i_b = src.find("归纳主题结构 (从 fact 内容, 而非预设模板)")
    i_fill = src.find("yield from _phase_gap_fill(")
    i_macro = src.find("_phase_macros(ws, theme_map)")
    assert i_b != -1 and i_fill != -1 and i_macro != -1, \
        "三个阶段标记没找全(顺序标记可能改了写法)"
    assert i_b < i_fill < i_macro, \
        f"顺序错了: theme_map({i_b}) < gap_fill({i_fill}) < macros({i_macro})"
    # 空白标注也必须真的挂在回补里 —— 直接调 _annotate_gaps 的单测盖不住"忘了接线"。
    # 注意它写在 _phase_gap_fill **函数体内**(704 行), 而不是那个 yield from(856 行),
    # 所以只能按"落在函数体内"来判, 不能按全局偏移比大小。
    i_def = src.find("def _phase_gap_fill")
    i_next_def = src.find("def _annotate_gaps")
    i_anno = src.find('tmap["gaps"] = _annotate_gaps(')
    assert i_anno != -1, "回补后没给空白打标记, 清单跟没跑过一样"
    assert i_def < i_anno < i_next_def, \
        f"标注调用({i_anno})必须落在 _phase_gap_fill({i_def})函数体内"


def test_gap_fill_prompt_demands_english_query():
    """prompt 必须要求英文检索式。

    代码侧的 `_LATIN` 筛是最后一道闸, 但闸门前模型就给中文的话, 一轮检索式
    全废 —— 那一轮的成本照付、产出为零。所以 prompt 这层也要锁。
    """
    src = pathlib.Path(
        importlib.import_module("agent.prompts_survey").__file__
    ).read_text(encoding="utf-8")
    body = src.split("SURVEY_GAP_FILL_PROMPT", 1)[1].split('"""')[1]
    # **必须盯住"指令"那一条**, 不能只断言全文出现过 —— "英文检索式" 在这份
    # prompt 里出现两次: 一次在指令(L20), 一次在输出 schema 的示例里(L27)。
    # 只查全文的话, 把指令整句删掉测试照样绿(反向验证 G21 就是这么溜过去的)。
    instr = [ln for ln in body.splitlines() if "不管分数高低" in ln]
    assert instr, "prompt 里找不到'每条都要给检索式'那条指令"
    assert "英文检索式" in instr[0], f"指令里没要求英文: {instr[0]}"
    assert "arxiv" in body and "英文 API" in body, \
        "prompt 应该说明为什么必须英文(arxiv/OpenAlex 是英文 API)"


def test_gap_fill_switch_off(monkeypatch):
    """QAGENT_SURVEY_GAP_FILL_ROUNDS=0 时整段回补被跳过。"""
    monkeypatch.setattr(CFG, "_SURVEY_GAP_FILL_ROUNDS", 0)
    assert CFG.get_gap_fill_rounds() == 0
    import pathlib as _pl
    src = _pl.Path(SG.__file__).read_text(encoding="utf-8")
    assert "if get_gap_fill_rounds() > 0:" in src, \
        "回补必须能被开关整段关掉, 而不是只靠轮数循环自然结束"


def test_gap_fill_bounds():
    """三个闸门都要有合理上限 —— 每多一条检索式都是真金白银。"""
    import pathlib as _pl
    src = _pl.Path(CFG.__file__).read_text(encoding="utf-8")
    assert '"QAGENT_SURVEY_GAP_FILL_ROUNDS", 1, 0, 5)' in src, "轮数合法区间 (0,5)"
    assert '"QAGENT_SURVEY_GAP_FILL_QUERIES", 5, 1, 12)' in src, "检索式上限"
    assert '"QAGENT_SURVEY_GAP_FILL_PAPERS", 20, 1, 200)' in src, "新论文上限"


def test_gap_fill_only_calls_workspace_methods_that_exist():
    """回补里调的每个 `ws.<方法>()` 都必须真的存在于 SurveyWorkspace 上。

    **这条是被一个真 P0 逼出来的**: 10-06 真跑验证前, 29 个用例全绿, 但
    `ws.save_facts(facts)` 这行 —— SurveyWorkspace **根本没有 save_facts**
    (全仓库只有那一处提到它)。它会在"回补真的抽到了 fact"那一刻
    AttributeError 崩掉, 也就是回补唯一有价值的时刻。

    为什么前面的用例没抓到: 它们要么是纯函数测试(`gaps_to_queries`), 要么是
    源码字符串断言 —— **没有一条真的拿一个 workspace 跑过 `_phase_gap_fill`
    的函数体**。纯函数测不出来"调了不存在的方法"这类接线错误。
    """
    import ast

    from agent.infra.survey_workspace import SurveyWorkspace

    src = pathlib.Path(SG.__file__).read_text(encoding="utf-8")
    # 用 **ast** 而不是正则扫原文: 正则会把解释这次 P0 的那段注释里的
    # `ws.save_facts(facts)` 也算成真调用(踩过一次), 而 ast 走语法树, 注释
    # 根本进不来, 也不用去凑 tokenize 那套片段定界的坑。
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_phase_gap_fill"), None)
    assert fn is not None, "找不到 _phase_gap_fill 函数体"
    called = {n.attr for n in ast.walk(fn)
              if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
              and n.value.id == "ws"}
    assert called, "没扫到任何 ws 调用 —— 扫描方式本身坏了, 这条测试是空壳"
    missing = sorted(m for m in called if not hasattr(SurveyWorkspace, m))
    assert not missing, \
        f"回补调了 SurveyWorkspace 上不存在的方法: {missing}"


def test_gap_fill_reuses_existing_actions():
    """必须复用 action_search_more / action_extract_facts, 不能自己重写一套。

    那两个 action 已经处理了多源召回、逐篇 summarize、写 summary_items.db、
    单篇失败不阻塞 batch。重写一遍就是把这些边界条件全丢。
    """
    src = pathlib.Path(SG.__file__).read_text(encoding="utf-8")
    body = src.split("def _phase_gap_fill", 1)[1].split("def _annotate_gaps", 1)[0]
    assert "action_search_more(" in body
    assert "action_extract_facts(" in body


def test_gap_fill_caps_papers(monkeypatch):
    """新论文超过上限要砍, 不然后面 summarize + 抽 fact 会失控。"""
    import pathlib as _pl
    src = _pl.Path(SG.__file__).read_text(encoding="utf-8")
    body = src.split("def _phase_gap_fill", 1)[1].split("def _annotate_gaps", 1)[0]
    assert "get_gap_fill_max_papers()" in body
    assert "cap" in body and "[:cap]" in body


@pytest.mark.skipif(_REAL_RAW is None,
                    reason="需要 output/_gap_fill_raw.txt(真实 verdict 原始输出)")
def test_real_raw_actually_contains_nul():
    """表征数据自身的前置断言: 这份原始输出里真的有 NUL。

    没有这一条, 下面那条就可能因为"哪天模型不吐 NUL 了"而变成一个
    永远不会失败、也永远不会证明任何事的空壳。
    """
    assert "\x00" in _REAL_RAW, \
        "表征数据失真: 真实输出里已经没有 NUL 了, 该物证作废"
    assert "<think>" in _REAL_RAW, "还应该带 think 前缀"


@pytest.mark.skipif(_REAL_RAW is None,
                    reason="需要 output/_gap_fill_raw.txt(真实 verdict 原始输出)")
def test_real_raw_parses_to_all_31_gaps():
    """把带 NUL 的真实字节喂进解析器, 31 条一条不能少。

    这是 NUL 修复的**回归锁**: 把 `raw.replace("\\x00", "")` 去掉, 本条立刻红
    (所以必须现算, 见 _parse_real 的注释)。
    """
    parsed = _parse_real()
    assert isinstance(parsed, dict), "真实输出必须解析成功"
    rows = parsed["gaps"]
    assert len(rows) == 31, f"真实 31 条空白, 解析出 {len(rows)} 条"
    assert all(r.get("query") for r in rows), "有条空白没拿到检索式"
    leaked = [r for r in rows if "\x00" in str(r)]
    assert not leaked, f"NUL 没洗干净, 残留 {len(leaked)} 条"


@pytest.mark.skipif(_REAL_RAW is None,
                    reason="需要 output/_gap_fill_raw.txt(真实 verdict 原始输出)")
def test_real_raw_queries_are_all_english():
    """真实输出里的检索式必须全是英文 —— 这是它能喂进 arxiv 的前提。"""
    for r in _parse_real()["gaps"]:
        assert QE._LATIN.findall(r["query"]), f"检索式没有英文: {r['query']!r}"
