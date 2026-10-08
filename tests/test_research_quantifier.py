"""P4.5 第三轮: 命题体检闸门 —— 以及"为什么删掉三条防线"。

## 这一轮的转向

前两轮的做法是**加防线**: 偷换量词就查量词, 缩小范围就查范围, 加阈值就查阈值。
攒到八条, 其中三条是拿正则匹配自然语言:

    UNDECIDABLE_MARKERS 黑名单 / scope_conflict / threshold_conflict

那个方向是错的:

1. **正则不懂语义** —— 换个说法("不少于一千组")就漏;
2. **黑名单只会越长**, 加到没人敢动为止;
3. **它们在错误的层** —— 这些检查都在问"这条命题长得对不对",
   而真正该问的是"**这条命题该不该被验证**"。那是流程问题,
   用模式匹配去解, 天生就是打补丁。

所以改成: 建模之后、验证之前, 先整体过一遍**体检闸门**。
不通的命题根本不进验证循环 —— 是"该验才验", 而不是"出了事再拦"。

## 留下的三条都在代码层, 因为它们是契约不是猜测

    · conclusive verdict 必须挂可复核的证据
    · 量词是结构化字段, 不靠中文措辞隐含
    · 审计 0 LLM: refuted 必须给见证值, proved 必须有输出

判据: **"结论对不对"可以问 LLM, "结论有没有证据"必须问机器。**
"""

from __future__ import annotations

import pytest

from agent.tools.research._claim import (
    QUANTIFIERS,
    audit_claims,
    detect_quantifier,
    normalize_claim,
    quantifier_conflict,
)


# ============================================================
# 1. 量词 —— 保留在代码层, 因为它是结构化字段不是文本猜测
# ============================================================
class TestDetectQuantifier:
    @pytest.mark.parametrize("text,expected", [
        ("对所有正整数 n, n^2 小于 2n", "forall"),
        ("对任意 n 都成立", "forall"),
        ("每个顶点最多三条边", "forall"),
        ("存在正整数 n 使 n^2+n+41 是合数", "exists"),
        ("有没有 n 使 n^2 < 2n", "exists"),
        ("至少存在一个反例", "exists"),
        ("不存在这样的 n", "none"),
    ])
    def test_detects(self, text, expected):
        assert detect_quantifier(text) == expected

    def test_none_beats_exists(self):
        """★ 顺序陷阱: "不存在" 里含 "存在"。

        如果按"先匹配到的赢", "不存在这样的 n" 会被判成 exists ——
        而 exists 和 none 的结论正好相反, 判错等于把结论整个翻过来。
        """
        assert detect_quantifier("不存在这样的 n") == "none"

    def test_no_quantifier_returns_none(self):
        """抽不到就返回 None, 不猜。公式命题的量词常是隐含的。"""
        assert detect_quantifier("sin(x)^2 + cos(x)^2 = 1") is None
        assert detect_quantifier("") is None

    def test_question_marks_beat_negative(self):
        """★ 疑问句必须先于否定判定: "有没有" 的后两个字就是 "没有"。"""
        assert detect_quantifier("有没有 n 使它成立") == "exists"
        assert detect_quantifier("没有 n 使它成立") == "none"


class TestQuantifierConflict:
    def test_p4_actual_failure_is_caught(self):
        """★ P4 实跑原题。

        question = "对所有正整数 n, n^2 < 2n 吗"   (n=2 起就假)
        实跑拆成 "存在正整数 n 使得 n^2 < 2n"        → 判 proved
        """
        got = quantifier_conflict(
            "对所有正整数 n, n^2 小于 2n 吗",
            "存在正整数 n 使得 n^2 小于 2n",
        )
        assert got is not None
        assert (got[0], got[1]) == ("forall", "exists")

    def test_no_conflict_when_question_has_no_quantifier(self):
        """问题里没有量词 → 不报冲突。宁可漏检也不能瞎报。"""
        assert quantifier_conflict("证明 sin^2+cos^2=1", "对所有实数 x, sin^2+cos^2=1") is None

    def test_conflicting_claim_becomes_undecidable(self):
        spec = normalize_claim(
            "存在正整数 n 使得 n^2 小于 2n", scope="n 为正整数",
            question="对所有正整数 n, n^2 小于 2n 吗",
        )
        assert spec["decidable"] is False
        assert any("量词冲突" in w for w in spec["warn"])

    def test_illegal_quantifier_value_is_not_fatal(self):
        spec = normalize_claim("存在 n 使 P", scope="n=1..10", quantifier="some_exists")
        assert spec["decidable"] is True
        assert spec["quantifier"] == "exists"

    def test_all_legal_quantifiers_accepted(self):
        for q in QUANTIFIERS:
            assert normalize_claim("P 成立", scope="n=1..10", quantifier=q)["quantifier"] == q


# ============================================================
# 2. ★ 体检闸门 —— 取代那三条正则防线的一道
# ============================================================
def _fake_llm(answers: dict[int, dict]):
    """返回体模仿 _call_llm: 给定 index 返回预设答案。"""
    import json

    rows = [{"index": i, **a} for i, a in answers.items()]
    return lambda *a, **k: json.dumps({"results": rows}, ensure_ascii=False)


class TestPrecheckGate:
    def test_blocks_question_substitution(self):
        """★ 体检拦下"拆的不是原问题"。这正是量词/范围/阈值三条正则共同的职责。"""
        from agent.tools.research._precheck import apply_precheck, run_precheck

        specs = [normalize_claim("sum(k=1..n) k^2 = n(n+1)(2n+1)/6", scope="n=1..500")]
        checks = run_precheck(
            "用归纳法证明的那段证明是优美的",
            specs,
            _fake_llm({1: {"same_question": False, "decidable": True,
                           "why": "问题问'优美', 命题问'求和公式成立'"}}),
        )
        apply_precheck(specs, checks)
        assert specs[0]["decidable"] is False
        assert specs[0]["state"] == "undecidable"
        assert any("不是在回答原问题" in w for w in specs[0]["warn"])

    def test_blocks_self_contradictory_claim(self):
        """★ 用户明确要求的一类: 命题本身矛盾就不该去验。

        「最小的不存在的素数」—— 不存在的东西没有最小值, 命题本身是空的。
        """
        from agent.tools.research._precheck import apply_precheck, run_precheck

        specs = [normalize_claim("最小的不存在的素数", scope="n=1..100")]
        checks = run_precheck(
            "最小的不存在的素数是多少",
            specs,
            _fake_llm({1: {"same_question": True, "decidable": False,
                           "why": "自相矛盾: 不存在的对象没有最小值"}}),
        )
        apply_precheck(specs, checks)
        assert specs[0]["decidable"] is False
        assert any("不可判定" in w for w in specs[0]["warn"])

    def test_blocks_vacuous_truth(self):
        """空洞真命题: 为真只因为约束弱到没有反例空间。"""
        from agent.tools.research._precheck import apply_precheck, run_precheck

        specs = [normalize_claim("不存在 m 使 2^m-1 是素数且 n<2", scope="n<2")]
        checks = run_precheck(
            "最小的不存在的素数是多少", specs,
            _fake_llm({1: {"same_question": False, "decidable": False,
                           "why": "n<2 几乎没有反例空间, 是证明了空集"}}),
        )
        apply_precheck(specs, checks)
        assert specs[0]["decidable"] is False

    def test_vacuous_truth_not_skipped_by_conditional_gate(self):
        """★ 条件触发不能把空洞真命题当干净放过去 ★

        真实退化过一次: 给 run_precheck 加了 only_suspicious(本地筛一遍,
        干净的跳过 LLM 复读)之后, 这条命题的量词是 none、措辞也正常,
        于是被 _looks_clean 判为"干净"直接放行 —— 闸门形同虚设,
        而 test_blocks_vacuous_truth 当场就红了。

        空洞真命题的量词/措辞都看不出问题, 只有对照 scope 才暴露。
        这条测试钉住的是那个修复, 防止"加优化"时再次悄悄拆掉防线。
        """
        from agent.tools.research._precheck import _looks_clean

        spec = normalize_claim("不存在 m 使 2^m-1 是素数且 n<2", scope="n<2")
        assert _looks_clean("最小的不存在的素数是多少", spec) is False, (
            "空洞真命题被当成干净放行了 —— 条件触发削弱了闸门"
        )

        # 对照: 量词一致、scope 正常、措辞中性 —— 这种不该被送去问 LLM
        ok = normalize_claim("对所有偶数 n 满足 4<=n<=1000 存在素数 p,q 使 p+q=n",
                             scope="4<=n<=1000")
        assert _looks_clean("对所有偶数 n 在 4..1000 内, 是否都能表示为两个素数之和", ok) is True, (
            "干净的命题被误判为可疑 —— 条件触发没省下任何成本"
        )

    def test_good_claim_passes(self):
        from agent.tools.research._precheck import apply_precheck, run_precheck

        specs = [normalize_claim("存在 n 使 n^2+n+41 是合数", scope="1<=n<=100")]
        checks = run_precheck(
            "存在正整数 n 使得 n^2+n+41 是合数吗", specs,
            _fake_llm({1: {"same_question": True, "decidable": True, "why": ""}}),
        )
        apply_precheck(specs, checks)
        assert specs[0]["decidable"] is True

    def test_two_reasons_are_both_reported(self):
        """既不是同一件事、又不可判定 → 两条理由都要落到 warn 里。"""
        from agent.tools.research._precheck import apply_precheck, run_precheck

        specs = [normalize_claim("求和公式成立", scope="n=1..500")]
        checks = run_precheck(
            "这段证明是优美的", specs,
            _fake_llm({1: {"same_question": False, "decidable": False, "why": "两回事"}}),
        )
        apply_precheck(specs, checks)
        warns = " ".join(specs[0]["warn"])
        assert "不是在回答原问题" in warns and "不可判定" in warns


class TestPrecheckFailsOpen:
    """★ 闸门失灵时必须**倒向放行**。

    这是个明确的取舍, 不是疏忽:
      闸门坏了还硬拦  -> 工具完全不可用, 用户什么都验不了
      闸门坏了就放行  -> 多花 token 验一条可能有问题的命题
    后者的损害小得多 —— 而且证据强制和 0 LLM 审计仍在下面兜底。
    """

    def test_llm_raises_and_we_let_everything_through(self):
        from agent.tools.research._precheck import apply_precheck, run_precheck

        def boom(*a, **k):
            raise RuntimeError("API 超时")

        specs = [normalize_claim("P 成立", scope="n=1..10")]
        checks = run_precheck("P 成立吗", specs, boom)
        assert checks == {}
        stats = apply_precheck(specs, checks)
        assert stats["llm_failed"] == 1
        assert specs[0]["decidable"] is True, "闸门坏了不该把功能一起锁死"

    def test_unparseable_output_lets_everything_through(self):
        from agent.tools.research._precheck import run_precheck

        specs = [normalize_claim("P 成立", scope="n=1..10")]
        assert run_precheck("q", specs, lambda *a, **k: "抱歉我不能这样做") == {}

    def test_partial_coverage_lets_uncovered_through(self):
        """只体检到一半 → 没覆盖到的那几条放行, 不能凭空判死。"""
        from agent.tools.research._precheck import apply_precheck, run_precheck

        specs = [
            normalize_claim("P1", scope="n=1..10"),
            normalize_claim("P2", scope="n=1..10"),
        ]
        checks = run_precheck(
            "q", specs,
            _fake_llm({1: {"same_question": False, "decidable": True, "why": "换问题"}}),
        )
        apply_precheck(specs, checks)
        assert specs[0]["decidable"] is False
        assert specs[1]["decidable"] is True, "没被体检到的命题不该被判死"

    def test_out_of_range_index_ignored(self):
        from agent.tools.research._precheck import run_precheck

        specs = [normalize_claim("P", scope="n=1..10")]
        assert run_precheck("q", specs, _fake_llm({99: {"same_question": False,
                                                         "decidable": False}})) == {}


# ============================================================
# 3. ★ 那三条正则防线必须真的删掉
# ============================================================
class TestRemovedDefensesStayRemoved:
    """★ 防止有人"顺手加回来"。

    这三条在 P4.5 前两轮确实抓到过真问题, 所以后来者很容易觉得
    "删了可惜, 加回来吧" —— 而那正是回到打补丁的老路。
    留这条测试是为了让删除变成一个**有记录的、不可悄悄回退**的决定。
    """

    def test_module_no_longer_exports_them(self):
        from agent.tools.research import _claim

        for name in ("scope_conflict", "threshold_conflict",
                     "question_is_undecidable", "UNDECIDABLE_MARKERS"):
            assert not hasattr(_claim, name), (
                f"{name} 又回来了。它是靠正则匹配自然语言的防线 —— "
                f"换个说法就漏, 且属于'出了事再拦'的打补丁思路。"
                f"该做的由 _precheck 闸门负责(语义判断)。"
            )

    def test_loop_does_not_use_blacklist(self):
        import ast
        import inspect
        import textwrap

        from agent.tools.research._loop import specs_from_planning

        tree = ast.parse(textwrap.dedent(inspect.getsource(specs_from_planning)))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert "question_is_undecidable" not in names | attrs, (
            "specs_from_planning 又调回黑名单判定了"
        )

    def test_gate_is_wired_into_the_loop(self):
        """反过来也钉一下: 闸门必须真的接在闭环里, 不是写了个没人调的模块。"""
        import ast
        import inspect
        import textwrap

        from agent.tools.research.research_investigate import research_investigate

        src = textwrap.dedent(inspect.getsource(research_investigate))
        tree = ast.parse(src)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "run_precheck" in names and "apply_precheck" in names, (
            "体检闸门没接进 research_investigate —— 它得在建模后、验证前跑"
        )
        # 顺序: apply_precheck 必须在 run_loop 之前
        assert src.index("apply_precheck") < src.index("run_loop"), (
            "闸门必须跑在验证循环**之前** —— 跑在之后就只是事后拦了"
        )


# ============================================================
# 4. 留在代码层的硬契约
# ============================================================
class TestHardContractsStay:
    """删掉正则防线的同时, 这几条**不许**跟着一起没 ——
    它们是契约, 不是猜测。
    """

    def test_refuted_needs_counterexample(self):
        spec = normalize_claim("P 不成立", scope="n=1..100")
        ev = [{"id": "ev_0001", "claim_id": spec["claim_id"], "kind": "code",
               "verdict": "refuted", "reason": "不成立", "stdout": ""}]
        res = audit_claims([spec], ev)
        hits = [i for i in res["issues"] if i["code"] == "refuted_without_counterexample"]
        assert hits and hits[0]["level"] == "error", res["issues"]

    def test_proved_needs_support(self):
        spec = normalize_claim("P 成立", scope="n=1..100")
        ev = [{"id": "ev_0001", "claim_id": spec["claim_id"], "kind": "code",
               "verdict": "proved", "reason": "成立", "stdout": ""}]
        res = audit_claims([spec], ev)
        hits = [i for i in res["issues"] if i["code"] == "proved_without_support"]
        assert hits and hits[0]["level"] == "error", res["issues"]

    def test_conclusive_with_evidence_is_clean(self):
        spec = normalize_claim("P 成立", scope="n=1..100")
        ev = [{"id": "ev_0001", "claim_id": spec["claim_id"], "kind": "code",
               "verdict": "proved", "reason": "成立",
               "stdout": "n=40 -> 1681 = 41^2"}]
        assert audit_claims([spec], ev)["n_errors"] == 0
