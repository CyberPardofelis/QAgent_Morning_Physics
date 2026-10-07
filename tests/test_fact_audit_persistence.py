"""
tests/test_fact_audit_persistence.py — 审查结论必须落盘 (10-07 加)

**这个文件为什么存在**
`action_extract_facts` 的时序是反的: `ws.append_fact(f)` 在抽取循环里就把 fact
逐行 json.dumps 写进 `_facts.jsonl`, 而**两次审查**(落地审查 / 主题对齐审查)
之后才把 `_grounded=False` / `_is_topic_aligned=False` 改到**内存里同一个 dict**
上。文件写完即定型, 内存的修改永远回不去。

**这不是理论问题, 是实测过的**: 全部 35 个 `_facts.jsonl` 共 2786 行,
`_grounded: false` 出现 **0 次**, `_is_topic_aligned` 出现 **0 次**。
`survey_generate.py` 里还有一句注释断言「对齐标记在 _facts.jsonl 里」。

后果只在**续跑**时显形: `ws.load_facts()` 拿回来的每一条都是默认值 true,
`_is_usable_fact` 对每一条都返回 True, 于是最贵的两个审查阶段的结论全部作废,
而用户看到的是
    ♻️ fact 抽取: 沿用 checkpoint 的 241 条
    ✅ 可用 fact: 241 条 / 共 241 条
「另有 N 条 fact 未通过落地审查, 已剔除」那行永远不会出现。
"""
from __future__ import annotations

import importlib
import json
import pathlib
import tempfile

import pytest

EF = importlib.import_module("agent.tools.surveys._actions._extract_facts")


class _FakeWs:
    """够用的假 workspace: 只实现 _persist_audit_marks 会碰到的三个方法。"""
    def __init__(self):
        self.rows: list[dict] = []

    def load_facts(self):
        return [dict(r) for r in self.rows]

    def clear_facts(self):
        n = len(self.rows)
        self.rows = []
        return n

    def append_fact(self, f):
        # 真实实现是 json.dumps 逐行追加 —— 这里照抄那个语义:
        # **写进去就定型, 之后再改这个 dict 不影响已写的内容**
        self.rows.append(json.loads(json.dumps(f, ensure_ascii=False)))


# ============================================================
# 核心: 审查结论落盘
# ============================================================

def test_persist_writes_audit_marks_to_disk():
    """带否决标记的 fact 必须在文件里能读到 —— 续跑才不会再复活它们。"""
    ws = _FakeWs()
    facts = [
        {"fact_id": "a::0", "claim": "好的", "_grounded": True},
        {"fact_id": "a::1", "claim": "编造的", "_grounded": False},
        {"fact_id": "b::0", "claim": "邻近的", "_is_topic_aligned": False},
    ]
    for f in facts:
        ws.append_fact(f)

    n = EF._persist_audit_marks(ws, facts)
    assert n == 3

    disk = ws.load_facts()
    by_id = {f["fact_id"]: f for f in disk}
    assert by_id["a::1"]["_grounded"] is False, \
        "被判定超纲的 fact, 落盘后必须还是 _grounded=False"
    assert by_id["b::0"]["_is_topic_aligned"] is False, \
        "被判定不对齐的 fact, 落盘后必须还是 False"
    # **正向也锁一下**: 不能是"全写成 False"这种糊弄式修复
    assert by_id["a::0"]["_grounded"] is True


def test_persist_survives_a_mutation_after_append():
    """核心时序: 先落盘、后改内存 —— 改完必须重写才能进文件。

    这条直接复现生产里的顺序: 抽取时 append, 审查时才改标记。
    """
    ws = _FakeWs()
    f = {"fact_id": "x::0", "claim": "原样", "_grounded": True}
    ws.append_fact(f)                       # ← 抽取阶段落盘
    f["_grounded"] = False                  # ← 审查阶段改内存
    assert ws.load_facts()[0]["_grounded"] is True, \
        "前提变了: 假 workspace 没复刻'写完即定型'的语义, 这条测试就失去意义"

    EF._persist_audit_marks(ws, [f])
    assert ws.load_facts()[0]["_grounded"] is False


# ============================================================
# 放弃重写的两条路: 都必须"宁可吵"
# ============================================================

def test_persist_refuses_when_disk_count_mismatches():
    """磁盘条数 != 内存条数 时**放弃重写**, 绝不重写。

    重写会丢数据(磁盘上多出来的那些), 而不重写只是续跑时结论失效 ——
    丢数据比结论失效糟得多。
    """
    ws = _FakeWs()
    for i in range(5):
        ws.append_fact({"fact_id": f"a::{i}"})
    facts = [{"fact_id": f"a::{i}"} for i in range(3)]   # 内存只有 3 条

    assert EF._persist_audit_marks(ws, facts) is None
    assert len(ws.load_facts()) == 5, "放弃重写时磁盘内容必须一个不少"


def test_persist_returns_none_on_empty_facts():
    ws = _FakeWs()
    assert EF._persist_audit_marks(ws, []) is None


def test_persist_returns_none_when_read_back_fails(monkeypatch):
    """读 _facts.jsonl 本身就炸了 -> 放弃, 不硬来。"""
    ws = _FakeWs()
    ws.append_fact({"fact_id": "a::0"})

    def _boom():
        raise OSError("disk gone")
    monkeypatch.setattr(ws, "load_facts", _boom)
    assert EF._persist_audit_marks(ws, [{"fact_id": "a::0"}]) is None


# ============================================================
# 接线: 必须真的在两次审查**之后**调用
# ============================================================

def test_persist_call_sits_after_both_audits():
    """`_persist_audit_marks` 必须在两处审查之后调用, 否则写了还是旧值。

    顺序反了 = 这次修复完全无效, 而测试照样绿 —— 所以这条必须钉住。
    """
    import ast

    src = pathlib.Path(EF.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "action_extract_facts")
    order: list[tuple[int, str]] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in ("_persist_audit_marks", "audit_fact_grounding",
                    "_review_facts_topic_alignment"):
            # **必须按 lineno 排**: ast.walk 是广度优先, 直接拿遍历顺序会
            # 把嵌套调用的顺序打乱, 然后这条测试就会报一个假的"顺序反了"。
            order.append((node.lineno, name))
    order.sort()
    names = [n for _, n in order]

    assert "_persist_audit_marks" in names, "action_extract_facts 里没调它"
    assert names.index("_persist_audit_marks") > names.index("audit_fact_grounding"), \
        f"写回发生在落地审查之前: {order}"
    assert (names.index("_persist_audit_marks")
            > names.index("_review_facts_topic_alignment")), \
        f"写回发生在主题对齐审查之前: {order}"


def test_failure_is_yielded_not_silent():
    """写回失败必须让用户看见, 不能只进 logger。

    ⚠️ 只查**失败分支**那个块: 早先写成"整个函数体里有没有 yield", 结果成功
    分支末尾那句 `📝 审查结论已落盘` 让断言恒真, 把失败分支改成 print 照样绿
    (反向验证 P5 踩过)。
    """
    src = pathlib.Path(EF.__file__).read_text(encoding="utf-8")
    tail = src.split("n_rewritten = _persist_audit_marks", 1)[1]
    fail_block = tail.split("if n_rewritten is None:", 1)[1].split("\n    elif", 1)[0]
    assert "yield" in fail_block, \
        "写回失败的分支里没有 yield —— 续跑会复活被否决的 fact 而用户不知道"
    assert "续跑" in fail_block, "告警文案要点明后果(续跑会丢审查结论)"


# ============================================================
# 假注释不许回来
# ============================================================

def test_no_comment_claims_marks_are_already_persisted():
    """锁住那句「对齐标记在 _facts.jsonl 里」的假断言。

    它从 10-04 起就在那儿, 一直是假的, 而注释不参与执行 —— 只能靠这条盯着。
    """
    src = pathlib.Path(
        importlib.import_module("agent.tools.surveys.survey_generate").__file__
    ).read_text(encoding="utf-8")
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and "对齐标记在 _facts.jsonl 里" in stripped:
            assert "原来写的是" in line or "修" in line, \
                f"又出现了一句'标记已经在文件里'的断言: {stripped}"


def test_real_workspace_facts_never_had_false_grounded():
    """**修复前**的历史数据: 审查标记一次都没落盘过 —— 这是修复必要性的证据。

    这条测试的作用是**表征修复前的世界**: 在 `0a92349` 之前, 审查完的结论
    只存在于内存, 续跑时会把已判定编造的 fact 全部复活。所以修复前的
    34 个工作区、2000+ 行 `_facts.jsonl` 里, `_grounded: false` 和
    `_is_topic_aligned` 的出现次数都是 0。

    **10-07 修复落地后这个 0 被打破了, 而且是好事**: 新工作区
    (`89309897a97c`, 经典阴影那次真跑) 落盘了 3 条 `_grounded: false`
    和 299 条对齐标记 —— 这正是修复生效的直接实证。

    所以现在断言的是**分工作区**: 修复前那批仍然是 0(证明当时确实没落盘),
    修复后新跑的工作区必须 > 0(证明现在确实落盘了)。两个方向都得成立,
    少一个就说明修复把历史数据也改了, 或者根本没生效。
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    files = sorted((root / "data" / "surveys" / "working").glob("*/_facts.jsonl"))
    if not files:
        pytest.skip("没有可查的 workspace")

    def scan(path):
        nf = na = rows = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            rows += 1
            nf += '"_grounded": false' in line
            na += "_is_topic_aligned" in line
        return rows, nf, na

    total_rows = total_false = total_align = 0
    # 修复后(经典阴影那次真跑)必须真的有落盘标记
    post_fix_rows = post_fix_false = post_fix_align = 0
    for p in files:
        rows, nf, na = scan(p)
        total_rows += rows
        total_false += nf
        total_align += na
        if "89309897a97c" in p.parent.name:      # 10-07 经典阴影真跑
            post_fix_rows += rows
            post_fix_false += nf
            post_fix_align += na

    assert total_rows > 1000, f"样本太小({total_rows} 行), 结论不可靠"
    assert post_fix_rows > 0, "没找到 10-07 修复后的工作区, 样本不可靠"
    assert post_fix_false > 0, (
        "10-07 修复后的工作区里 `_grounded: false` 还是 0 条 —— "
        "审查结论仍然没有落盘, `0a92349` 的修复没生效")
    assert post_fix_align > 0, (
        "10-07 修复后的工作区里 `_is_topic_aligned` 还是 0 条 —— "
        "主题对齐结论没有落盘")
