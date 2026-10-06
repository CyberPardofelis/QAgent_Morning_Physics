"""
tests/test_llm_text_clean.py — LLM 输出文本的字节级清洗 (10-06 加)

**这个文件为什么存在**
10-06 跑覆盖空白回补, MiniMax-M3 **连着 6 次输出 6 次混进 NUL 字节**。
长相是 `"Toff\\x00oli\\x00\\x00 GHZ global addressing"` —— `Toffoli` 变成
`Toff\\x00oli`。这是输出流在**字节层面**被破坏, 不是模型打了隐藏字符。

之前两次都猜错了方向(以为是截断、以为是格式不稳定), 直到把原始输出抓下来
才看清。代价是白白浪费两轮。

**为什么修在 `_call_llm` 出口而不是二十多个解析点**
NUL 是传输层损坏。代码库里解析 LLM 输出的地方有二十多处, 每处打一个补丁
意味着漏一处就漏一处, 而漏掉的症状是"那个工具偶尔静默失败", 极难归因。
出口洗一次, 下面全部受益 —— 这条由 `test_call_llm_sanitizes_at_the_boundary` 锁住。

**表征数据用真的**
`_gap_fill_raw.txt` 是那次被污染的真实输出(8538 字符, 含 think 前缀 + 大量
`\\x00`)。手工重构不出这种形态, 而"造一个带 NUL 的字符串"测不出真实分布。
"""
from __future__ import annotations

import importlib
import pathlib

import pytest

TC = importlib.import_module("agent.infra.text_clean")
QE = importlib.import_module("agent.tools.surveys._pool._query_expand")
LLM = importlib.import_module("agent.tools.llm")

_RAW_PATH = pathlib.Path(__file__).resolve().parents[1] / "output" / "_gap_fill_raw.txt"
_REAL_RAW = _RAW_PATH.read_text(encoding="utf-8") if _RAW_PATH.exists() else None


# ============================================================
# 清洗函数本身
# ============================================================

def test_strip_nul_rejoins_a_broken_word():
    """删 NUL 而不是替换成空格。

    替换成空格会留下 `Toff oli` 这种"看着像真的"的碎片, 比缺字更难发现;
    删掉至少还原成可读的 `Toffoli`。
    """
    got = TC.strip_nul("Toff\x00oli\x00\x00 GHZ")
    assert got == "Toffoli GHZ"


def test_strip_nul_leaves_clean_text_untouched():
    src = " Rydberg dressing analog quantum simulation"
    assert TC.strip_nul(src) is src, "干净文本不该被重新分配"


def test_strip_nul_never_raises_on_non_str():
    """下游有可能传 None / dict 进来 —— 不能因为清洗而崩。"""
    for bad in (None, 123, ["a"], {"k": 1}):
        assert TC.strip_nul(bad) is bad


def test_strip_control_chars_keeps_tab_newline_cr():
    """`\\t \\n \\r` 要留 —— strict=False 放行它们, 删了会改变正常文本。"""
    assert TC.strip_control_chars("a\tb\nc\rd") == "a\tb\nc\rd"


def test_strip_control_chars_kills_vertical_tab_and_del():
    """\\x0b \\x0c \\x07 \\x7f 在 JSON 字符串里同样是非法的。"""
    assert TC.strip_control_chars("a\x0bb\x0cc\x07d\x7fe") == "abcde"


def test_sanitize_llm_text_actually_removes_nul_and_control_chars():
    """**必须真删**。这条是 T5/T9 两条变异能溜过去的直接原因。

    之前的 `test_sanitize_llm_text_is_idempotent` 里写的是 `\\x0b` —— 那是
    **字面的反斜杠 + x0b 四个字符**, 不是真的垂直制表符, 于是:
      - "删掉 NUL 之外的控制字符"这个行为压根没被测到
      - "删 NUL"这个行为也没被直接断言(幂等性对"什么都没删"是成立的)
    两层都白测了。踩过一次: 反向验证 T5/T9 直接报"改坏后仍然绿"。
    """
    dirty = '<think>x</think>{"q": "Toff\x00oli\x0b test\x7fdone"}'
    clean = TC.sanitize_llm_text(dirty)
    assert "\x00" not in clean, "NUL 没删掉"
    assert "\x0b" not in clean, r"\x0b(垂直制表符) 没删掉"
    assert "\x7f" not in clean, "DEL 没删掉"
    assert "Toffoli" in clean and "testdone" in clean, \
        f"该留的字被误删了: {clean!r}"
    # think 包裹不在本函数职责内, 但也不能被顺手改掉
    assert clean.startswith("<think>x</think>")


def test_sanitize_keeps_chinese_and_emoji():
    """清洗**绝不能**误伤中文 / emoji —— 这是"删控制字符"最典型的翻车方式。

    一个看起来很合理的实现是 `re.sub(r"[^\x20-\x7e]", "", text)`("只留可打印
    ASCII"), 那样会把整篇中文综述洗成空字符串。这里拿真实综述里会出现的
    字符钉住: 中文标点、破折号、中文正文、以及 emoji。
    """
    src = '覆盖空白：纠错协议（表面码）— 中性原子 · Rydberg 🔎 门操作'
    clean = TC.sanitize_llm_text(src)
    assert clean == src, f"清洗把正常内容改坏了:\n  期望 {src!r}\n  实得 {clean!r}"
    # **必须直接打 strip_control_chars**: 上面那句输入里没有控制字符,
    # sanitize_llm_text 根本不会调它 —— 那样这条测试就是空壳, 把
    # strip_control_chars 换成"只留可打印 ASCII"的实现照样绿(反向验证 T9 踩过)。
    assert TC.strip_control_chars(src + "\x0b") == src, \
        "strip_control_chars 把非 ASCII 一起删了 —— 中文综述会洗成空串"


def test_sanitize_llm_text_is_idempotent():
    """洗两次等于洗一次 —— 出口清洗 + 解析点兜底会叠着跑。"""
    raw = '<think>x</think>{"q": "Toff\x00oli\x0b test"}'
    once = TC.sanitize_llm_text(raw)
    assert TC.sanitize_llm_text(once) == once


def test_loads_maybe_dirty_parses_nul_json():
    """给拿不到 `_call_llm` 出口的路径用(读历史产物 / 复现现场)。"""
    got = TC.loads_maybe_dirty('{"a": "Toff\x00oli"}')
    assert got == {"a": "Toffoli"}


def test_loads_maybe_dirty_returns_none_on_garbage():
    assert TC.loads_maybe_dirty("完全没有 JSON") is None
    assert TC.loads_maybe_dirty("") is None


# ============================================================
# 边界: _call_llm 出口必须洗
# ============================================================

def test_call_llm_sanitizes_at_the_boundary(monkeypatch):
    """**这是本次修复的核心断言**: 清洗发生在 `_call_llm` 的返回上。

    改坏它 → 二十多个 JSON 解析点重新开始随机失败, 而症状会散落在完全
    不相干的工具里。
    """
    dirty = "<think>先想一下</think>{\"themes\": [{\"name\": \"Toff\x00oli 门\"}]}"

    class _Msg:
        content = dirty
        reasoning_content = None

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kw):
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(LLM, "get_client", lambda provider: _Client())
    monkeypatch.setattr(LLM, "cost_guard_charge", lambda: (True, ""))

    out = LLM._call_llm("prompt", provider="minimax", model="MiniMax-M3",
                        enable_thinking=False)
    assert "\x00" not in out, \
        "NUL 字节穿过了 _call_llm 的出口 —— 所有下游解析点都在裸奔"
    assert "Toffoli" in out


def test_call_llm_source_calls_the_sanitizer():
    """源码层锁一道: 万一有人重构把那一行删了, 上面那条测试不好定位。"""
    import ast

    tree = ast.parse(pathlib.Path(LLM.__file__).read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_call_llm"), None)
    assert fn is not None
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "sanitize_llm_text" in names, \
        "_call_llm 里没有 sanitize_llm_text —— 出口清洗被删了?"


def test_controller_sanitizes_tool_call_arguments():
    """controller 的工具参数也来自模型输出流, 同样要过清洗。

    这条是反向验证 T7 逼出来的: 之前只测了 `_call_llm`, 漏了 controller 这条
    平行路径 —— 它不走 `_call_llm`, 所以出口清洗**覆盖不到**它。
    """
    import ast

    import agent.controller as CTRL

    tree = ast.parse(pathlib.Path(CTRL.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "sanitize_llm_text" in names, \
        "controller 解析工具调用参数时没走清洗 —— 工具调用会随机参数解析失败"


# ============================================================
# 表征: 真实被污染的那份输出
# ============================================================

@pytest.mark.skipif(_REAL_RAW is None,
                    reason="需要 output/_gap_fill_raw.txt(真实被污染的原始输出)")
def test_real_polluted_output_has_nul():
    """表征数据自身的前置断言。

    没有这条, 下面两条就可能因为"哪天模型不吐 NUL 了"变成永远不失败的空壳。
    """
    assert "\x00" in _REAL_RAW, "这份输出里已经没有 NUL 了, 物证作废"
    assert _REAL_RAW.count("\x00") > 50, "NUL 数量异常少, 可能不是同一份数据"


@pytest.mark.skipif(_REAL_RAW is None,
                    reason="需要 output/_gap_fill_raw.txt(真实被污染的原始输出)")
def test_real_output_parses_after_boundary_sanitize():
    """**这份真实输出在清洗前 `json.loads` 是必然失败的。**

    这是整个 NUL 问题的证据链: 不是我们 prompt 写得不好, 不是格式不稳定,
    是字节被破坏了。先证明"洗之前失败", 再证明"洗之后成功"。
    """
    import json as _json

    with pytest.raises(Exception):
        _json.loads(_REAL_RAW[_REAL_RAW.find("{"):], strict=False)

    clean = TC.sanitize_llm_text(_REAL_RAW)
    assert "\x00" not in clean
    payload = _json.loads(clean[clean.find("{"):], strict=False)
    assert len(payload["gaps"]) == 31


@pytest.mark.skipif(_REAL_RAW is None,
                    reason="需要 output/_gap_fill_raw.txt(真实被污染的原始输出)")
def test_parse_gap_verdict_keeps_working_on_real_output():
    """解析层在真实脏数据上仍然拿满 31 条(它自己也洗一遍, 是第二道防线)。"""
    got = QE._parse_gap_verdict(_REAL_RAW)
    assert got is not None and len(got["gaps"]) == 31
    assert not any("\x00" in str(r) for r in got["gaps"])


# ============================================================
# 下游解析点: 出口洗过之后, 它们本来就该是干净的
# ============================================================

def test_no_parse_site_needs_its_own_nul_patch():
    """提醒性的断言: **不要**给每个解析点各打一个 NUL 补丁。

    修法收敛在出口一处。这里列出的这些点就是当初盘出来的全部 LLM 输出解析点,
    它们都该从 `_call_llm` 拿到已经洗干净的文本。
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    sites = [
        "agent/tools/surveys/_organize.py",
        "agent/tools/surveys/_pool/_utils.py",
        "agent/tools/surveys/_fact_audit.py",
        "agent/tools/surveys/_actions/_decide.py",
        "agent/tools/surveys/_actions/_search_more.py",
        "agent/tools/surveys/_actions/_write_outline.py",
        "agent/memory/extractor.py",
    ]
    import ast

    offenders = []
    for rel in sites:
        p = root / rel
        if not p.exists():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "replace"
                    and n.args and isinstance(n.args[0], ast.Constant)
                    and n.args[0].value == "\x00"):
                offenders.append(rel)
    # 连 `_query_expand` 那处也应该是 0 —— 它现在调的是共用的 sanitize_llm_text,
    # 不是自己 replace。任何一处手写 replace 都意味着"这一处知道有 NUL, 但别人不知道",
    # 正是要避免的那种局部知识。
    assert not offenders, \
        f"这些地方自己 replace 了 NUL, 局部知识会盖住出口清洗失效: {offenders}"


def test_query_expand_uses_the_shared_sanitizer_not_its_own():
    """`_parse_gap_verdict` 是第二道防线, 但要用**共用函数**而不是自己 replace。"""
    import ast

    tree = ast.parse(pathlib.Path(QE.__file__).read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and n.name == "_parse_gap_verdict"), None)
    assert fn is not None
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "sanitize_llm_text" in names, \
        "_parse_gap_verdict 该调共用的 sanitize_llm_text, 而不是自己写 replace"
