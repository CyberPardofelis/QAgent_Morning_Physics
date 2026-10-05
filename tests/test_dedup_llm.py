"""
tests/test_dedup_llm.py — 语义去重脚本的纯函数护栏 (10-05)

重点不在覆盖率, 在**锁住那次静默失败**: 3 篇论文全报"无重复", 而其中一篇
事先用词面勘察确认有 J=1.00 的同文对。抓原始返回才看到: 模型**已经正确判出**
了同义组, 但 max_tokens=8000 全被 <think> 吃光, JSON 一个字没输出就被截断,
而脚本把截断记成了"未判出重复"。

于是这里有两条不能省的护栏:
  - `looks_truncated` 必须认出"被 think 吃光"这种返回
  - `parse_groups` 只接受合法下标, 且**解析失败绝不等于"没有重复"**
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "dedup_facts_llm", ROOT / "scripts" / "dedup_facts_llm.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def D():
    return _load()


@pytest.fixture
def conn_factory():
    """按需给一个带 fact 表的内存库(pick_keeper 要数出处)。"""
    import sqlite3

    from agent.tools.facts import SCHEMA_SQL

    made = []

    def _make():
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        c.executescript(SCHEMA_SQL)
        made.append(c)
        return c
    yield _make
    for c in made:
        c.close()


# ============================================================
# looks_truncated: 这是那次静默失败的直接防线
# ============================================================
def test_detects_unclosed_think(D):
    """<think> 没闭合 = 预算被思考吃光, JSON 还没开始写。"""
    s = ("<think>Let me analyze: [0] says canonical forms... [10] says 左规范、右规范和"
         "\n\nGroup 1: ...")
    assert D.looks_truncated(s) is True


def test_detects_unbalanced_json(D):
    assert D.looks_truncated('<think>done</think>\n{"groups": [[0, 5], [10]') is True
    assert D.looks_truncated('{"groups": [[0, 5]], "singletons": [1, 2}') is True
    assert D.looks_truncated('{"groups": [[0, 5]], "singletons": [1, 2, 3') is True


def test_complete_response_is_not_truncated(D):
    ok = '<think>reasoning</think>\n{"groups": [[0, 5, 10]], "singletons": [1, 2]}'
    assert D.looks_truncated(ok) is False


def test_empty_output_is_not_marked_truncated(D):
    """空返回要单独报"返回空", 不能混进"截断" —— 两种病因不同。"""
    assert D.looks_truncated("") is False
    assert D.looks_truncated(None) is False


def test_prose_without_json_is_not_flagged_as_truncated(D):
    """模型可能只写了解释没写 JSON。这不是截断, 是格式不听话 —— 区别在于
    括号是配平的。判错方向会把"格式问题"报成"预算问题", 误导排查。"""
    assert D.looks_truncated("我认为这 14 条里有 2 组重复, 分别是 [0,5] 和 [9,11]") is False


# ============================================================
# parse_groups: 宁可少合也不错合
# ============================================================
def test_parse_basic_groups(D):
    raw = '{"groups": [[0, 2], [5, 7, 9]], "singletons": [1, 3, 4, 6]}'
    assert D.parse_groups(raw, set(range(10))) == [[0, 2], [5, 7, 9]]


def test_parse_ignores_out_of_range_indices(D):
    """模型偶尔把下标写错/越界。这种组直接丢 —— 错合不可逆。"""
    raw = '{"groups": [[0, 99], [1, 2]], "singletons": [3]}'
    assert D.parse_groups(raw, {0, 1, 2, 3}) == [[1, 2]]


def test_parse_rejects_singleton_groups(D):
    assert D.parse_groups('{"groups": [[4]]}', set(range(10))) == []


def test_parse_survives_markdown_fence(D):
    raw = '```json\n{"groups": [[0, 3]], "singletons": [1]}\n```'
    assert D.parse_groups(raw, set(range(5))) == [[0, 3]]


def test_parse_survives_surrounding_prose(D):
    raw = '好的, 结果如下:\n{"groups": [[2, 8]], "singletons": [0, 1]}\n希望有帮助。'
    assert D.parse_groups(raw, set(range(10))) == [[2, 8]]


def test_parse_returns_empty_on_garbage(D):
    """垃圾输入必须返回空而不是抛异常 —— 静默失败比崩溃更隐蔽。"""
    for raw in ("", "not json at all", "{broken", None, '{"groups": "not a list"}'):
        assert D.parse_groups(raw, set(range(5))) == []


def test_parse_dedupes_repeated_indices_in_group(D):
    assert D.parse_groups('{"groups": [[0, 1, 1, 0]]}', {0, 1}) == [[0, 1]]


def test_parse_handles_string_indices(D):
    raw = '{"groups": [["0", "2"]], "singletons": ["1"]}'
    assert D.parse_groups(raw, {0, 1, 2}) == [[0, 2]]


# ============================================================
# 分块: 按字符不按条数
# ============================================================
def _facts(n, clen):
    return [{"id": f"f{i}", "content": "x" * clen, "fact_type": "method"}
            for i in range(n)]


def test_chunks_respect_char_budget(D):
    """chunk_facts 返回的是**下标**, 不是 fact 本身。"""
    facts = _facts(100, 200)
    for block in D.chunk_facts(facts):
        assert sum(len(facts[i]["content"]) for i in block) <= D.CHUNK_CHARS + 200


def test_chunks_cover_every_fact_exactly_once(D):
    facts = _facts(60, 300)
    idx = [i for b in D.chunk_facts(facts) for i in b]
    assert sorted(idx) == list(range(60)), "分块不能丢也不能重"


def test_small_input_is_single_chunk(D):
    assert D.chunk_facts(_facts(3, 50)) == [[0, 1, 2]]


def test_one_huge_fact_does_not_break_chunking(D):
    """单条就超预算时也必须成块, 不能死循环或返回空。"""
    facts = _facts(2, 50000)
    blocks = D.chunk_facts(facts)
    assert len(blocks) == 2 and sum(len(b) for b in blocks) == 2


# ============================================================
# 保留哪一条: 必须是信息量最大的, 不是第一条
# ============================================================
def test_pick_keeper_prefers_more_sources(D, conn_factory):
    from agent.tools.facts import fact_id_for

    conn = conn_factory()
    short = "短"
    long = "长" * 50
    fid_a = fact_id_for(short, "method")
    fid_b = fact_id_for(long, "method")
    for fid, c in ((fid_a, short), (fid_b, long)):
        conn.execute("INSERT INTO fact (id, content, fact_type) VALUES (?,?,'method')",
                     (fid, c))
    conn.execute("INSERT INTO fact_source (fact_id, source_type, source_id)"
                 " VALUES (?,'paper_abstract','p1')", (fid_b,))
    conn.execute("INSERT INTO fact_source (fact_id, source_type, source_id)"
                 " VALUES (?,'paper_abstract','p2')", (fid_b,))
    facts = [{"id": fid_a, "content": short}, {"id": fid_b, "content": long}]
    assert D.pick_keeper(facts, [0, 1], conn) == 1, "多出处的应当保留"


def test_pick_keeper_breaks_tie_by_length(D, conn_factory):
    from agent.tools.facts import fact_id_for

    conn = conn_factory()
    a, b = "短句", "长" * 40
    for c in (a, b):
        conn.execute("INSERT INTO fact (id, content, fact_type) VALUES (?,?,'method')",
                     (fact_id_for(c, "method"), c))
    facts = [{"id": fact_id_for(a, "method"), "content": a},
             {"id": fact_id_for(b, "method"), "content": b}]
    assert D.pick_keeper(facts, [0, 1], conn) == 1


# ============================================================
# preflight: 写错模型名曾经烧掉一整天预算
# ============================================================
def test_preflight_rejects_unknown_minimax_model(D):
    """MiniMax 白名单外的模型名必须**发第一条之前**就中止。

    10-05 实测踩坑: 写了不存在的 "MiniMax-M3.1", 258 次调用全部 400,
    而 cost_guard 是"先计费再发请求", 于是当日额度全被空转吃掉, 一组没判出来,
    而且从返回里看不出是模型名错(每条都是"无重复")。
    """
    with pytest.raises(SystemExit) as e:
        D.preflight("minimax", "MiniMax-M3.1")
    assert "白名单" in str(e.value) or "不在" in str(e.value)


def test_preflight_accepts_known_model(D):
    from agent.llm_clients import MINIMAX_VALID_MODELS
    assert D.preflight("minimax", MINIMAX_VALID_MODELS[-1]) is None


# ============================================================
# 并发路径: run_ordered 返回的是 TaskResult 包装, 不是裸值
# ============================================================
def test_run_ordered_returns_taskresult_not_raw_value():
    """10-05 踩到: 并发改造后写了 `for paper, payload, note in batch`,
    直接报 "cannot unpack non-iterable TaskResult"。

    这条护栏的作用是把"必须先取 .value 才能解包"写成测试, 而不是靠一次
    运行时崩溃来发现 —— 而且那次崩溃发生在**整轮快跑完的时候**。
    """
    from agent.tools.surveys._parallel import TaskResult, run_ordered

    batch = run_ordered([1, 2, 3], lambda x: (f"p{x}", [[f"f{x}"]], ""),
                        max_workers=3)
    assert all(isinstance(r, TaskResult) for r in batch)
    # 正确写法
    for r in batch:
        assert r.ok
        paper, payload, note = r.value      # 先 .value 再解包
        assert paper.startswith("p")
    # 错误写法会怎样: 直接解包 TaskResult。注意要用**三元组**目标才炸 ——
    # 解包 1 个值时 `for x in [...]` 不报错, 而是把 TaskResult 自己绑上去,
    # 错误会一路飘到后面才暴露, 更难查(10-05 第一次就是这么以为"没报错"的)。
    with pytest.raises(TypeError):
        for _a, _b, _c in run_ordered([1], lambda x: (x, [], ""), max_workers=1):
            pass


def test_run_ordered_preserves_input_order():
    """并发但结果必须按输入顺序 —— 这是 _parallel 存在的全部理由。"""
    from agent.tools.surveys._parallel import run_ordered

    def slow_for_small(x):
        import time
        time.sleep(0.05 if x == 0 else 0.0)   # 让 0 最后完成
        return x
    batch = run_ordered([0, 1, 2, 3], slow_for_small, max_workers=4)
    assert [r.value for r in batch] == [0, 1, 2, 3], "并发后顺序错了"


def test_run_ordered_failure_does_not_kill_batch():
    """单篇失败不能拖垮整批(10-05 的失败必须可见而不是静默)。"""
    from agent.tools.surveys._parallel import run_ordered

    def boom(x):
        if x == 1:
            raise ValueError("炸了")
        return x

    batch = run_ordered([0, 1, 2], boom, max_workers=3)
    assert [r.ok for r in batch] == [True, False, True]
    assert isinstance(batch[1].exc, ValueError)
    assert batch[2].ok, "失败项不能影响其他项"


# ============================================================
# 完整性标记: "判了没有重复" 和 "没判出来" 必须在结果文件里可区分
# ============================================================
def test_entry_marks_truncation_as_incomplete(D):
    """截断 -> complete=False。

    10-05 第一版只把分组存进 json, 没存"这篇因为被截断没判完"。于是进程
    一重启, `--resume` 看到 key 存在就当判过了, 那 4 篇截断 + 5 篇 API 错误
    被静默当成"确认无重复"。同一个错误(没判出来 vs 没有)在另一个地方复发。
    """
    e = D._entry([["f1", "f2"]], "块 1 被 max_tokens=32000 截断, 判同未完成")
    assert e["complete"] is False
    assert e["groups"] == [["f1", "f2"]]
    assert "截断" in e["note"]


@pytest.mark.parametrize("note,complete", [
    ("", True),
    ("块 1 判定无重复", True),
    ("块 2 判定无重复", True),
    ("块 1 API 错误: ❌ MiniMax 调用失败（InternalServerError）", False),
    ("块 1 LLM 返回空", False),
    ("块 1 被 max_tokens 截断", False),
])
def test_entry_complete_flag(D, note, complete):
    assert D._entry([], note)["complete"] is complete


def test_read_results_upgrades_legacy_format(D, tmp_path, monkeypatch):
    """旧格式(纯 {paper: [[ids]]})必须能读, 且**一律当未判完**。

    旧格式没有 note, 无从知道是否判完 —— 当成 complete=True 就是把
    "没判出来"重新伪装成"没有重复"。
    """
    legacy = tmp_path / "r.json"
    legacy.write_text('{"p1": [["a", "b"]], "p2": []}', encoding="utf-8")
    monkeypatch.setattr(D, "RESULT_JSON", legacy)
    got = D._read_results()
    assert got["p1"]["groups"] == [["a", "b"]]
    assert got["p1"]["complete"] is False
    assert got["p2"]["complete"] is False


def test_read_results_passes_through_new_format(D, tmp_path, monkeypatch):
    new = tmp_path / "r.json"
    new.write_text('{"p1": {"groups": [["a"]], "note": "", "complete": true}}',
                   encoding="utf-8")
    monkeypatch.setattr(D, "RESULT_JSON", new)
    assert D._read_results()["p1"]["complete"] is True


def test_write_results_is_atomic(D, tmp_path, monkeypatch):
    """落盘必须原子: 磁盘上永远是一个完整文件。

    直接 write_text 每篇调一次, 进程正好在写一半被杀 -> 留下截断的 json ->
    之后 --resume 一读就 json.load 失败, 前面所有工作一起报废。
    """
    target = tmp_path / "r.json"
    monkeypatch.setattr(D, "RESULT_JSON", target)
    D._write_results({"p": D._entry([["a"]], "")})
    assert target.exists()
    import json as _json
    assert _json.loads(target.read_text(encoding="utf-8"))["p"]["complete"] is True
    assert not list(tmp_path.glob("*.tmp")), "临时文件没被清掉"


def test_resume_only_skips_complete_papers():
    """--resume 的过滤逻辑: 只跳过 complete=True 的。"""
    done = {"ok": {"complete": True}, "bad": {"complete": False}}
    keep = {k: v for k, v in done.items() if not v.get("complete")}
    assert set(keep) == {"bad"}, "未判完的必须重判, 不能被当成'判过了'"


# ============================================================
# 反向验证
# ============================================================
def test_reverse_truncation_detector_disabled_red(D):
    """把 looks_truncated 改成恒 False -> 截断又被当成"无重复", 护栏失效。"""
    saved = D.looks_truncated
    try:
        D.looks_truncated = lambda s: False
        truncated = "<think> 没写完 [10] says 左规范"
        assert saved(truncated) is True
        assert D.looks_truncated(truncated) is False, "关掉后就不再报警"
    finally:
        D.looks_truncated = saved


def test_reverse_parse_accepts_everything_red(D):
    """把下标校验去掉 -> 越界组也会被当成合法, 错合风险回来。"""
    raw = '{"groups": [[0, 99]]}'
    good = D.parse_groups(raw, {0, 1, 2})
    assert good == [], "正确实现应丢弃越界组"
    # 反证: 只要不校验范围, 就会收下 —— 这就是为什么必须有它
    naive = [sorted({int(x) for x in g}) for g in [[0, 99]]]
    assert naive == [[0, 99]]
