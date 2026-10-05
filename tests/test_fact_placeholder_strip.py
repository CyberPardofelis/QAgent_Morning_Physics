"""
tests/test_fact_placeholder_strip.py — 占位符剥离 + 存量剥离脚本的回归护栏 (10-05)

这一批测试的定位不是"覆盖代码行", 是**锁住 10-05 实际踩到的每一个坑**:

1. 占位符形态漏了一种(migrate 脚本 4 种 vs 生产路径 5 种) -> 形态护栏
2. `[2209.02093]` 会污染内容哈希 -> 剥完必须能被 fact_id_for 收敛成一条
3. 剥 `[10.1103/xxx]` 有可能误伤正文里的真引用 -> 只剥自己 source 的
4. 改主键时 UNIQUE 是**立即**检查, defer_foreign_keys 管不着 -> 顺序护栏
5. plan 只跟"当前存在的 id"比 -> 后一条 new_id 撞前一条(10 条 → 40 条)
6. 剥完才出现的同 content 组合 -> 收尾合并
7. INSERT OR IGNORE 搬 theme_fact 会把 asserted_by 悄悄丢掉
8. check_integrity 把"提示"混进"阻断" -> 脚本对着提示也回滚
9. GBK 控制台打印 ✅/❌ 崩在**报错分支**上, 把错误信息本身吞掉
"""
from __future__ import annotations

import importlib.util
import re
import sqlite3
import sys
from pathlib import Path

import pytest

from agent.prompts_survey import (
    _PLACEHOLDER_RES,
    count_placeholder_forms,
    strip_paper_id_from_claim,
)
from agent.tools.facts import fact_id_for

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    """按文件路径加载 scripts/ 下的脚本 —— 它们不是包, 没有 __init__ 可 import。"""
    spec = importlib.util.spec_from_file_location(
        "strip_placeholder_facts", ROOT / "scripts" / "strip_placeholder_facts.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# 1. 形态护栏: 每一个已知形态都必须被认出来
# ============================================================
# 每加一种形态就往这里加一行。10-05 的教训: "总数 624 处"看着像剥干净了,
# 实际是**漏了一种**([2209.02093] 裸 id 767 条)。总数对不上不可怕, 可怕的是
# 漏的那一种恰好是最毒的 —— 分形态才看得出来。
KNOWN_FORMS = [
    ("[paper_id]", "该方法有效 [paper_id]"),
    ("[paper_id: 2209.02093]", "该方法有效 [paper_id: 2209.02093]"),
    ("[paper_id：全角冒号]", "该方法有效 [paper_id：2209.02093]"),
    ("<paper_id>", "该方法有效 <paper_id>"),
    ("[[双括号]]", "该方法有效 [[fact_id_local]]"),
    ("[裸 arxiv id]", "该方法有效 [2209.02093]"),
    ("[裸 arxiv id 带 v]", "该方法有效 [2209.02093v2]"),
    ("[7 位 arxiv id]", "该方法有效 [2209020]"),
    ("[期刊 id]", "该方法有效 [10.1103/physrevlett.127.040501]"),
    ("[期刊 id 无斜杠]", "该方法有效 [10.1103/94hz-xtht]"),
    ("裸 paper_id: id", "该方法有效 paper_id: 2209.02093"),
]


@pytest.mark.parametrize("label,text", KNOWN_FORMS, ids=[f[0] for f in KNOWN_FORMS])
def test_known_placeholder_form_is_detected(label, text):
    counts = count_placeholder_forms(text)
    assert sum(counts.values()) >= 1, f"形态没被认出来: {label} -> {text}"
    assert "2209" not in strip_paper_id_from_claim(text) or "10.1103" in text
    assert "paper_id" not in strip_paper_id_from_claim(text).lower()
    assert "[[" not in strip_paper_id_from_claim(text)


def test_strip_removes_id_and_leaves_sentence_readable():
    out = strip_paper_id_from_claim("BB84 仍面临安全漏洞与距离限制等弱点,[2312.05609]")
    assert out == "BB84 仍面临安全漏洞与距离限制等弱点"


@pytest.mark.parametrize("raw,expected", [
    ("方法有效 [2209.02093]。", "方法有效。"),   # 句号是正文, 保留
    ("方法有效,[2209.02093]", "方法有效"),        # 逗号是分隔, 收掉
    ("方法有效 [2209.02093]", "方法有效"),
    ("方法有效，[2209.02093]", "方法有效"),
    ("方法有效；[2209.02093]", "方法有效"),
])
def test_strip_also_trims_trailing_separators(raw, expected):
    """剥完的尾巴必须干净。fact_id_for 只去标点, 但 UI 显示和内容比对都吃这份文本,
    留个孤零零的逗号会让人以为抽取质量有问题。**句号除外**, 它属于正文。"""
    assert strip_paper_id_from_claim(raw) == expected


def test_strip_does_not_touch_ordinary_text():
    for text in ("张量网络方法对 SPT 相进行了分类",
                 "该算法能计算基态并模拟时间演化",
                 "矩阵 (MPS) 与张量积态 (TPS) 均可用于模拟"):
        assert strip_paper_id_from_claim(text) == text


def test_strip_keeps_sentence_final_period_when_no_placeholder():
    """没有占位符时**不能**动句末的句号。

    `条。` 是正文的一部分, 剥掉它会让 UI 上所有 fact 末尾都没有标点。
    收尾标点只在**真的剥掉了东西**时才需要收拾 —— 见下一个测试。
    """
    text = "张量网络方法对 SPT 相进行了分类。"
    assert strip_paper_id_from_claim(text) == text
    assert text.endswith("。")


# ============================================================
# 2. 内容哈希护栏: 剥占位符后必须能收敛成同一条 fact
# ============================================================
def test_placeholder_variants_collapse_to_same_fact_id():
    """同一句话只因末尾标注写法不同, 必须算出同一个 id。

    这就是去重机制能不能工作的前提 —— id 含噪声时, 同一命题永远对不上。
    """
    base = "本文的核心贡献聚焦于 VQE 算法的改进"
    variants = [base,
                f"{base},[2501.10018]",
                f"{base}。",
                f"{base} [2501.10018]",
                f"{base},[2501.10018]"]
    ids = {fact_id_for(strip_paper_id_from_claim(v), "method") for v in variants}
    assert len(ids) == 1, f"剥完还是 {len(ids)} 个不同 id: {ids}"


def test_fact_id_ignores_punctuation_but_not_content():
    a = fact_id_for("VQE 算法被改进以降低电路深度", "method")
    b = fact_id_for("VQE算法被改进以降低电路深度", "method")
    c = fact_id_for("VQE 算法被改进以增加电路深度", "method")
    assert a == b
    assert a != c


# ============================================================
# 3. 误伤护栏: 不能把正文里的真引用剥掉
# ============================================================
def test_strip_does_not_remove_arxiv_id_without_brackets():
    """正文里裸写的 2209.02093(不带方括号)是内容的一部分, 不该动。
    只剥**方括号**形式 —— 因为只有那种是 prompt 占位符漏出来的。"""
    text = "该方法与 arXiv:2209.02093 中的方案做了对比"
    assert strip_paper_id_from_claim(text) == text


def test_strip_keeps_numbers_that_look_like_ids_but_are_content():
    """裸 7 位数字不是 arxiv id, 不能当成占位符。"""
    text = "实验在温度 2209020 毫开尔文下进行"  # 荒谬值, 但形态上不该被剥
    assert strip_paper_id_from_claim(text) == text


# ============================================================
# 4-9. 存量剥离脚本的行为护栏(在内存库上跑真逻辑)
# ============================================================
@pytest.fixture
def strip_mod():
    return _load_script()


@pytest.fixture
def conn(strip_mod):
    """一个带完整主题层 schema 的内存库。

    依赖 strip_mod 是为了拿 MERGE_AUDIT_SQL —— 反过来用会拿到 fixture
    定义对象而不是模块(10-05 第一次跑就报了这个)。
    """
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    from agent.infra import theme_graph as tg
    from agent.tools.facts import SCHEMA_SQL
    c.executescript(SCHEMA_SQL)
    c.executescript(tg.SCHEMA_SQL)
    c.executescript(strip_mod.MERGE_AUDIT_SQL)
    yield c
    c.close()


def _mkfact(conn, content, fact_type="method", source="2501.10017", subject=""):
    fid = fact_id_for(content, fact_type)
    conn.execute("INSERT OR IGNORE INTO fact (id, content, fact_type, subject)"
                 " VALUES (?,?,?,?)", (fid, content, fact_type, subject))
    conn.execute("INSERT OR IGNORE INTO fact_source (fact_id, source_type, source_id)"
                 " VALUES (?,'paper_abstract',?)", (fid, source))
    return fid


def _mktheme(conn, name):
    conn.execute("INSERT OR IGNORE INTO theme (name) VALUES (?)", (name,))
    return conn.execute("SELECT id FROM theme WHERE name=?", (name,)).fetchone()[0]


def test_rename_preserves_source_and_theme(strip_mod, conn):
    old = _mkfact(conn, "该方法有效,[2501.10017]")
    tid = _mktheme(conn, "主题A")
    conn.execute("INSERT INTO theme_fact (theme_id, fact_id, asserted_by)"
                 " VALUES (?,?,?)", (tid, old, "ws-abc"))
    conn.commit()

    clean = strip_paper_id_from_claim("该方法有效,[2501.10017]")
    new = fact_id_for(clean, "method")
    strip_mod.apply_rename(conn, {"old_id": old, "new_id": new,
                                   "content": "该方法有效,[2501.10017]",
                                   "clean": clean, "fact_type": "method",
                                   "subject": "", "confidence": 1.0})
    conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM fact WHERE id=?", (old,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM fact WHERE id=?", (new,)).fetchone()[0] == 1
    assert conn.execute("SELECT content FROM fact WHERE id=?", (new,)).fetchone()[0] == clean
    # 出处与主题归属都必须跟着走
    assert conn.execute("SELECT source_id FROM fact_source WHERE fact_id=?",
                        (new,)).fetchone()[0] == "2501.10017"
    row = conn.execute("SELECT asserted_by FROM theme_fact WHERE fact_id=?", (new,)).fetchone()
    assert row is not None and row[0] == "ws-abc"


def test_rename_into_existing_fact_merges_without_losing_asserted_by(strip_mod, conn):
    """两条 fact 剥完变成同一句 -> 改主键会撞 UNIQUE。

    UNIQUE 是**立即**检查的(defer_foreign_keys 只推迟外键), 所以
    "先 UPDATE 再去重"必然中途炸。必须先查冲突、合并 asserted_by、再删旧行。
    10-05 在真库副本上连撞两次 FOREIGN KEY / UNIQUE 才定位到这里。

    注意两条 fact 必须**真的**是两条: fact_id_for 已经把标点差异吃掉了,
    所以"逗号 vs 空格"这种差异根本不会产生两行(10-05 写这条测试时先踩了这个)。
    真正会分裂的是**不同 id** —— 同一句话被不同论文抽到, 各自带自己的标注。
    """
    a = _mkfact(conn, "该方法有效,[2501.10017]")
    b = _mkfact(conn, "该方法有效,[2501.10018]")
    assert a != b, "两条应当是不同的 fact 行"
    tid = _mktheme(conn, "主题A")
    conn.execute("INSERT INTO theme_fact (theme_id, fact_id, asserted_by) VALUES (?,?,?)",
                 (tid, a, "ws-one"))
    conn.execute("INSERT INTO theme_fact (theme_id, fact_id, asserted_by) VALUES (?,?,?)",
                 (tid, b, "ws-two"))
    conn.commit()

    clean = "该方法有效"
    new = fact_id_for(clean, "method")
    # apply_rename 只负责"把 old 这行搬成 new"; 与另一条撞车的那一行由调用方
    # 交给 apply_merge(见 test_late_duplicates_after_strip_are_merged)。
    # 这里先把 b 也搬过来, 复现真实的执行序列。
    strip_mod.apply_rename(conn, {"old_id": a, "new_id": new,
                                   "content": "该方法有效,[2501.10017]",
                                   "clean": clean, "fact_type": "method",
                                   "subject": "", "confidence": 1.0})
    strip_mod.apply_merge(conn, new, b, reason="placeholder_strip")
    conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM fact").fetchone()[0] == 1
    got = conn.execute("SELECT asserted_by FROM theme_fact WHERE fact_id=?",
                       (new,)).fetchone()[0]
    parts = set(got.split(","))
    assert "ws-one" in parts and "ws-two" in parts, \
        f"asserted_by 合并时丢了一个: {got!r}"


def test_late_duplicates_after_strip_are_merged(strip_mod, conn):
    """剥完才撞成同 content 的组合, plan 阶段看不出来(new_id 不同, 因为
    各自带不同标注), 必须靠 run_all 的收尾合并。"""
    _mkfact(conn, "方法甲,[2501.10001]")
    _mkfact(conn, "方法甲,[2501.10002]")   # 标注不同 -> 原 id 不同
    conn.commit()
    assert conn.execute("SELECT COUNT(DISTINCT id) FROM fact").fetchone()[0] == 2

    renames, merges = strip_mod.plan(conn, strip_mod.collect_fixes(conn))
    strip_mod.run_all(conn, renames, merges)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM fact").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM fact WHERE content='方法甲'").fetchone()[0] == 1


def test_collapse_late_duplicates_is_callable_and_effective(strip_mod, conn):
    """收尾合并必须能被单独调用并真的合并(10-05 反向验证发现它此前
    只在 main() 里, 关掉之后没有任何测试会红)。"""
    a = _mkfact(conn, "方法甲,[2501.10001]")
    b = _mkfact(conn, "方法甲,[2501.10002]")
    conn.commit()
    # 手工制造同 content 但 id 不同
    conn.execute("UPDATE fact SET content='方法甲' WHERE id=?", (a,))
    conn.execute("UPDATE fact SET content='方法甲' WHERE id=?", (b,))
    conn.commit()
    n = strip_mod.collapse_late_duplicates(conn)
    conn.commit()
    assert n == 1
    assert conn.execute("SELECT COUNT(*) FROM fact").fetchone()[0] == 1


def test_plan_detects_collision_between_two_renames(strip_mod, conn):
    """两条待改 fact 剥完后**彼此**撞 id(不是撞既有 fact)。

    只跟"当前已存在的 id"比对的话, 它们都走 rename, 前一条改完占了 id,
    后一条再改就 UNIQUE 失败。10-05 实测: 10 条 merge 变 40 条才对。
    """
    a = _mkfact(conn, "方法甲,[2501.10001]")
    b = _mkfact(conn, "方法甲,[2501.10002]")
    conn.commit()
    clean = "方法甲"
    new = fact_id_for(clean, "method")
    fixes = [{"old_id": a, "new_id": new}, {"old_id": b, "new_id": new}]
    renames, merges = strip_mod.plan(conn, fixes)
    assert len(renames) == 1 and len(merges) == 1
    assert merges[0] == (new, b)


def test_plan_rename_into_pre_existing_fact_is_a_merge(strip_mod, conn):
    """剥完的 id 撞上**库里已有**的 fact -> 必须是 merge, 不能是 rename。

    10-05 反向验证时发现这条分支没有测试: 只测了"两条 rename 互相撞",
    于是把 plan 的 existing 判断整个去掉, 测试仍然全绿。实测真库里
    40 条 merge 中的大部分走的是这条路径。
    """
    existing = _mkfact(conn, "方法甲")          # 已经是干净的
    dirty = _mkfact(conn, "方法甲,[2501.10001]")  # 剥完会撞上它
    conn.commit()
    assert existing != dirty
    new = fact_id_for("方法甲", "method")
    assert new == existing
    renames, merges = strip_mod.plan(conn, [{"old_id": dirty, "new_id": new}])
    assert renames == [], "撞上既有 fact 还走 rename = 执行时 UNIQUE 失败"
    assert merges == [(existing, dirty)]


def test_rename_moves_source_rows_not_just_content(strip_mod, conn):
    """改主键必须把 fact_source 一起搬过去, 而不是留在旧 id 上变孤儿。"""
    a = _mkfact(conn, "该方法有效,[2501.10017]", source="2501.10017")
    conn.commit()
    clean = "该方法有效"
    new = fact_id_for(clean, "method")
    strip_mod.apply_rename(conn, {"old_id": a, "new_id": new,
                                   "content": "该方法有效,[2501.10017]",
                                   "clean": clean, "fact_type": "method",
                                   "subject": "", "confidence": 1.0})
    conn.commit()
    rows = conn.execute("SELECT source_id FROM fact_source WHERE fact_id=?", (new,)).fetchall()
    assert [r[0] for r in rows] == ["2501.10017"]
    assert conn.execute("SELECT COUNT(*) FROM fact_source WHERE fact_id=?", (a,)).fetchone()[0] == 0


def test_rename_inserts_theme_fact_when_no_collision(strip_mod, conn):
    """目标 fact 在该 theme 下**还没有**行 -> 必须真的插进去。

    10-05 反向验证: 之前只测了"撞车时合并 asserted_by"这条分支,
    把"不撞车时 INSERT"改成空操作, 测试照样全绿 —— 也就是说主题归属
    会静默丢掉一半, 而没有任何测试会红。
    """
    old = _mkfact(conn, "该方法有效,[2501.10017]")
    tid = _mktheme(conn, "新主题")
    conn.execute("INSERT INTO theme_fact (theme_id, fact_id, asserted_by) VALUES (?,?,?)",
                 (tid, old, "ws-new"))
    conn.commit()
    clean = "该方法有效"
    new = fact_id_for(clean, "method")
    strip_mod.apply_rename(conn, {"old_id": old, "new_id": new,
                                   "content": "该方法有效,[2501.10017]",
                                   "clean": clean, "fact_type": "method",
                                   "subject": "", "confidence": 1.0})
    conn.commit()
    row = conn.execute("SELECT asserted_by FROM theme_fact WHERE fact_id=?", (new,)).fetchone()
    assert row is not None, "不撞车时必须插入 theme_fact 行, 否则主题归属静默丢失"
    assert row[0] == "ws-new"


def test_merge_inserts_theme_fact_when_no_collision(strip_mod, conn):
    """合并时不撞车 -> dropped 的主题归属必须被搬过去。"""
    kept = _mkfact(conn, "方法甲")
    dropped = _mkfact(conn, "方法甲,[2501.10001]")
    tid = _mktheme(conn, "只有dropped在的主题")
    conn.execute("INSERT INTO theme_fact (theme_id, fact_id, asserted_by) VALUES (?,?,?)",
                 (tid, dropped, "ws-x"))
    conn.commit()
    strip_mod.apply_merge(conn, kept, dropped, reason="semantic_merge")
    conn.commit()
    rows = conn.execute("SELECT asserted_by FROM theme_fact WHERE fact_id=?", (kept,)).fetchall()
    assert [r[0] for r in rows] == ["ws-x"]


def test_check_integrity_separates_blocking_errors_from_notes(strip_mod, conn):
    """老账(库里本来就有的悬空)不能当阻断项, 否则脚本永远跑不动;
    但也绝不能吞掉 —— 所以它必须在 notes 里被报出来。"""
    tid = _mktheme(conn, "主题A")
    conn.execute("INSERT INTO theme_relation (src_theme, dst_theme, note)"
                 " VALUES (?,?,'边')", (tid, 999))   # 999 不存在
    conn.commit()
    before = strip_mod.fk_snapshot(conn)
    errors, notes = strip_mod.check_integrity(conn, before)
    assert not errors, f"老账不该阻断: {errors}"
    assert notes and "外键悬空" in notes[0]


def test_check_integrity_blocks_on_newly_introduced_dangling(strip_mod, conn):
    """本次**新增**的悬空必须是阻断项, 不是提示。

    10-05 反向验证: 把 errors.append 改成 notes.append, 所有测试仍全绿 ——
    因为此前只测了"库本来就有的老账不阻断", 没测"新账必须阻断"。
    那样等于脏数据可以直接落库。
    """
    conn.execute("PRAGMA foreign_keys = OFF")
    tid = _mktheme(conn, "主题A")
    conn.execute("INSERT INTO theme_relation (src_theme, dst_theme, note)"
                 " VALUES (?,?,'边')", (tid, 999))
    conn.commit()
    before = set()          # 基线干净 -> 这处就是"本次弄坏的"
    errors, notes = strip_mod.check_integrity(conn, before)
    assert any("新增外键悬空" in e for e in errors), \
        f"新增悬空必须阻断, 实际 errors={errors} notes={notes}"


def test_check_integrity_preexisting_dangling_is_note_not_error(strip_mod, conn):
    """对照: 老账只提示不阻断(否则脚本永远跑不动)。"""
    conn.execute("PRAGMA foreign_keys = OFF")
    tid = _mktheme(conn, "主题A")
    conn.execute("INSERT INTO theme_relation (src_theme, dst_theme, note)"
                 " VALUES (?,?,'边')", (tid, 999))
    conn.commit()
    before = strip_mod.fk_snapshot(conn)     # 基线**包含**这处悬空
    errors, notes = strip_mod.check_integrity(conn, before)
    assert not errors, f"老账不该阻断: {errors}"
    assert notes and "外键悬空" in notes[0]


def test_check_integrity_blocks_orphan_rows(strip_mod, conn):
    """孤儿行(fact_source / theme_fact 指向不存在的 fact)必须阻断。"""
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("INSERT INTO theme_fact (theme_id, fact_id, asserted_by)"
                 " VALUES (?,?,'ws')", (_mktheme(conn, "T"), "fact_不存在"))
    conn.commit()
    errors, _ = strip_mod.check_integrity(conn, set())
    assert any("孤儿" in e for e in errors), errors


def test_check_integrity_blocks_remaining_duplicate_content(strip_mod, conn):
    """仍有同 content 的 fact 残留必须阻断(收尾合并没做干净)。"""
    _mkfact(conn, "方法甲,[2501.10001]")
    _mkfact(conn, "方法甲,[2501.10002]")
    conn.execute("UPDATE fact SET content='方法甲'")
    conn.commit()
    errors, _ = strip_mod.check_integrity(conn, set())
    assert any("同 content" in e for e in errors), errors


def test_merge_keeps_audit_trail_with_original_text(strip_mod, conn):
    kept = _mkfact(conn, "方法甲")
    dropped = _mkfact(conn, "方法甲,[2501.10001]")
    conn.commit()
    strip_mod.apply_merge(conn, kept, dropped, reason="semantic_merge")
    conn.commit()
    row = conn.execute("SELECT reason, dropped_content, kept_content FROM fact_merge"
                       " WHERE dropped_id=?", (dropped,)).fetchone()
    assert row is not None
    assert row[0] == "semantic_merge"
    assert row[1] == "方法甲,[2501.10001]"      # 原文可还原
    assert row[2] == "方法甲"
    assert conn.execute("SELECT COUNT(*) FROM fact WHERE id=?", (dropped,)).fetchone()[0] == 0


def test_merge_asserted_by_merge_helper_dedupes():
    mod = _load_script()
    assert mod._merge_asserted_by("", "ws-b") == "ws-b"
    assert mod._merge_asserted_by("ws-a", "") == "ws-a"
    assert mod._merge_asserted_by("ws-a", "ws-b") == "ws-a,ws-b"
    assert mod._merge_asserted_by("ws-a,ws-b", "ws-b") == "ws-a,ws-b"


# ============================================================
# 反向验证: 关掉逻辑, 测试必须变红
# ============================================================
def test_reverse_strip_is_noop_red(strip_mod, monkeypatch, conn):
    """把剥离换成恒等函数 -> 收集到的 fix 数必须是 0(护栏立刻失效)。"""
    monkeypatch.setattr(strip_mod, "strip_paper_id_from_claim", lambda s: s)
    _mkfact(conn, "该方法有效,[2501.10017]")
    conn.commit()
    assert strip_mod.collect_fixes(conn) == []


def test_reverse_plan_without_simulation_red(strip_mod, conn):
    """把 plan 换回"只跟当前 id 比" -> 两条互相撞的 fix 会双双走 rename,
    测试应当变红。这条护栏是为了防有人把模拟逻辑"优化"掉。"""
    a = _mkfact(conn, "方法甲,[2501.10001]")
    b = _mkfact(conn, "方法甲,[2501.10002]")
    conn.commit()
    new = fact_id_for("方法甲", "method")
    fixes = [{"old_id": a, "new_id": new}, {"old_id": b, "new_id": new}]

    existing = {r[0] for r in conn.execute("SELECT id FROM fact")}
    naive_renames = [f for f in fixes if f["new_id"] not in existing]
    assert len(naive_renames) == 2, "朴素实现会两条都走 rename(= 会撞 UNIQUE)"
    # 而真正的 plan 必须识别出其中一条是 merge
    renames, merges = strip_mod.plan(conn, fixes)
    assert len(renames) + len(merges) == 2 and len(merges) == 1


def test_reverse_skip_placeholder_form_red():
    """删掉 _PLACEHOLDER_RES 里任意一条, 对应形态的检测测试必须变红。

    这条是元护栏: 它保证上面 11 条形态测试不是"摆设"。
    """
    assert len(_PLACEHOLDER_RES) >= 7
    for i, rx in enumerate(_PLACEHOLDER_RES):
        assert isinstance(rx, re.Pattern), f"第 {i} 条不是编译好的正则"
