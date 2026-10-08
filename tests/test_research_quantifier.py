"""P5-prep: "用户问题 → 命题"这一段的护栏测试。

## 为什么这组测试独立于 test_research_p2.py

P4 实跑(2026-10-08)暴露的失败全部集中在**建模**这一步, 而不是验证:

- 「对所有正整数 n, n^2 < 2n」(n=2 起就是假命题) → 拆成「存在 n 使 n^2 < 2n」→ 判 `proved`
- 「哥德巴赫猜想成立吗」 → 拆成「每个奇素数是三个奇素数之和」(弱哥德巴赫) → 判 `refuted`
- 「这段证明是优美的」 → 判 `proved` ×2

这三条都是**在拿到命题之前**就错了。所以这里测的是"命题规范化"这一层,
而不是已经拆好之后的验证逻辑。

## 三道防线

1. `detect_quantifier` / `quantifier_conflict` —— 量词不许被偷换
2. 命题回显 —— 流式输出里量词和"不可判定原因"必须可见
3. `audit_claims` 第 6 类检查 —— refuted 必须拿得出反例
"""

from __future__ import annotations

import pytest

from agent.tools.research._claim import (
    QUANTIFIERS,
    audit_claims,
    detect_quantifier,
    normalize_claim,
    quantifier_conflict,
    question_is_undecidable,
)


# ============================================================
# 1. 量词识别
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
        ("没有 n 使它成立", "none"),
    ])
    def test_detects(self, text, expected):
        assert detect_quantifier(text) == expected

    def test_none_beats_exists(self):
        """★ 顺序陷阱: "不存在" 里含 "存在"。

        如果按"先匹配到的赢", "不存在这样的 n" 会被判成 exists ——
        而 exists 和 none 的结论正好相反, 判错等于把结论整个翻过来。
        """
        assert detect_quantifier("不存在这样的 n") == "none"
        assert detect_quantifier("不存在反例") == "none"

    def test_no_quantifier_returns_none(self):
        """抽不到就返回 None, 不猜。公式命题的量词常是隐含的。"""
        assert detect_quantifier("sin(x)^2 + cos(x)^2 = 1") is None
        assert detect_quantifier("") is None


# ============================================================
# 2. ★ 量词冲突 —— P4 实跑失败的直接拦截点
# ============================================================
class TestQuantifierConflict:
    def test_p4_actual_failure_is_caught(self):
        """★ P4 实跑原题: 这是本组测试存在的全部理由。

        question = "对所有正整数 n, n^2 小于 2n 吗"   (n=2 起就假)
        实跑拆成 "存在正整数 n 使得 n^2 小于 2n"        → 判 proved

        量词一换, 结论从"假"翻成"真"。
        """
        got = quantifier_conflict(
            "对所有正整数 n, n^2 小于 2n 吗",
            "存在正整数 n 使得 n^2 小于 2n",
        )
        assert got is not None, "P4 的实测失败没被抓住"
        q_from, q_to, _msg = got
        assert (q_from, q_to) == ("forall", "exists")

    def test_same_quantifier_no_conflict(self):
        assert quantifier_conflict(
            "存在正整数 n 使 n^2+n+41 是合数", "存在正整数 n 使 n^2+n+41 是合数"
        ) is None

    def test_explicit_field_wins_over_text(self):
        """LLM 自己填的 quantifier 字段优先于从文本抽的。"""
        got = quantifier_conflict("对所有 n, P", "P 对每个 n 都行(措辞绕)", explicit="exists")
        assert got is not None and got[1] == "exists"

    def test_no_conflict_when_question_has_no_quantifier(self):
        """问题里没有量词 → 不报冲突。宁可漏检也不能瞎报。"""
        assert quantifier_conflict("证明 sin^2+cos^2=1", "对所有实数 x, sin^2+cos^2=1") is None


# ============================================================
# 3. normalize_claim: 冲突的命题必须被挡下
# ============================================================
class TestQuantifierGuardInNormalize:
    def test_conflicting_claim_becomes_undecidable(self):
        """量词被偷换的命题**不许进验证循环**。

        判 undecidable 而不是只 warn: 命题验得再对, 验的也不是用户问的那个,
        跑一轮验证纯属浪费 token 并且产出一个看起来很像结论的东西。
        """
        spec = normalize_claim(
            "存在正整数 n 使得 n^2 小于 2n",
            scope="n 为正整数",
            question="对所有正整数 n, n^2 小于 2n 吗",
        )
        assert spec["decidable"] is False
        assert spec["state"] == "undecidable"
        assert any("量词冲突" in w for w in spec["warn"])

    def test_matching_claim_stays_decidable(self):
        spec = normalize_claim(
            "存在正整数 n 使 n^2+n+41 是合数",
            scope="1<=n<=100",
            question="存在正整数 n 使得 n^2+n+41 是合数吗",
        )
        assert spec["decidable"] is True
        assert spec["quantifier"] == "exists"

    def test_quantifier_recorded_from_text(self):
        """LLM 没填 quantifier 字段时, 从命题文本兜底抽出来。"""
        spec = normalize_claim("对所有 n>=1, 2^n+1 是素数", scope="n=1..32")
        assert spec["quantifier"] == "forall"

    def test_illegal_quantifier_value_is_not_fatal(self):
        """LLM 编了个新值 → 当没填, 走文本抽取, 只 warn 不抛。"""
        spec = normalize_claim(
            "存在 n 使 P", scope="n=1..10", quantifier="some_kind_of_exists",
        )
        assert spec["decidable"] is True
        assert spec["quantifier"] == "exists"
        assert any("不是合法值" in w for w in spec["warn"])

    def test_all_legal_quantifiers_accepted(self):
        for q in QUANTIFIERS:
            spec = normalize_claim("P 成立", scope="n=1..10", quantifier=q)
            assert spec["quantifier"] == q


# ============================================================
# 4. ★ 审计第 6 类: refuted 必须拿得出反例
# ============================================================
class TestRefutedNeedsCounterexample:
    """P4 实跑里哥德巴赫被判 refuted, 而它拆出来的弱哥德巴赫**是已被证明的定理** ——
    也就是说那个 refuted 本身就是错的。错在哪? 拿不出反例。"""

    def _spec(self):
        return normalize_claim("P 不成立", scope="n=1..100")

    def test_refuted_without_output_is_error(self):
        spec = self._spec()
        ev = [{
            "id": "ev_0001", "claim_id": spec["claim_id"], "kind": "code",
            "verdict": "refuted", "reason": "不成立", "stdout": "",
        }]
        res = audit_claims([spec], ev)
        hits = [i for i in res["issues"] if i["code"] == "refuted_without_counterexample"]
        assert hits, res["issues"]
        assert hits[0]["level"] == "error", hits[0]
        assert res["n_errors"] >= 1

    def test_refuted_with_counterexample_is_clean(self):
        spec = self._spec()
        ev = [{
            "id": "ev_0001", "claim_id": spec["claim_id"], "kind": "code",
            "verdict": "refuted", "reason": "找到反例",
            "stdout": "反例: n = 40 -> 值 = 1681 = 41^2",
        }]
        res = audit_claims([spec], ev)
        assert res["n_errors"] == 0, res["issues"]

    def test_thin_counterexample_is_warning_only(self):
        """反例太短 → warn 不是 error。太严会把正常的小反例误伤。"""
        spec = self._spec()
        ev = [{
            "id": "ev_0001", "claim_id": spec["claim_id"], "kind": "code",
            "verdict": "refuted", "reason": "r", "stdout": "no",
        }]
        res = audit_claims([spec], ev)
        codes = {i["code"] for i in res["issues"]}
        assert "thin_counterexample" in codes
        assert "refuted_without_counterexample" not in codes


# ============================================================
# 5. 报告里量词必须可见
# ============================================================
class TestQuantifierVisibleInReport:
    def test_report_echoes_quantifier(self):
        """量词藏在中文措辞里就会被读漏 —— 报告要单独列一行。

        "对所有 n²<2n" 和 "存在 n 使 n²<2n" 的结论正好相反, 而两条 claim_text
        看上去很像。报告不单独打出来, 读者根本分不出这两条结论的方向。
        """
        from agent.tools.research._claim import render_report

        spec = normalize_claim("存在正整数 n 使 n^2+n+41 是合数", scope="1<=n<=100")
        ev = [{
            "id": "ev_0001", "claim_id": spec["claim_id"], "kind": "code",
            "verdict": "proved", "reason": "n=40", "stdout": "n=40 -> 1681=41^2",
        }]
        md = render_report([spec], ev, question="存在正整数 n 使 n^2+n+41 是合数吗")
        assert "量词" in md
        assert "存在" in md

    def test_report_surfaces_warnings(self):
        from agent.tools.research._claim import render_report

        spec = normalize_claim(
            "存在正整数 n 使得 n^2 小于 2n", scope="n 为正整数",
            question="对所有正整数 n, n^2 小于 2n 吗",
        )
        assert not spec["decidable"]
        md = render_report([spec], [], question="对所有正整数 n, n^2 小于 2n 吗")
        assert "量词冲突" in md, md[:600]


# ============================================================
# 6. ★ 防线四: 原问题不可判定 → 一条都不许进验证循环
# ============================================================
class TestUndecidableQuestionBlocksEverything:
    """★ 这是量词防线挡不住的**另一种**偷换。

    P4.5 第一轮实测现场(「这段证明是优美的」):

        - undecidable: "'优美'是主观审美判断, 不存在可形式化的真假判据"   ← 模型知道
        - proved:     "sum(k=1..n) k² = n(n+1)(2n+1)/6"                    ← 却还是验了

    模型一边承认原问题不可判定, 一边去验它某个"组成部分"。对用户来说那个 proved
    依然是答非所问 —— 而且因为混在一堆 undecidable 里面, **比纯硬吹更隐蔽**。
    """

    def test_detects_undecidable_question(self):
        assert question_is_undecidable("用归纳法证明的那段证明是优美的") == "优美"
        assert question_is_undecidable("这个猜想是否有趣") == "有趣"
        assert question_is_undecidable("对所有正整数 n, n^2 小于 2n 吗") is None

    def test_p4_actual_aesthetic_case_is_blocked(self):
        """P4.5 实测原题: 连"可判定"的组成部分也不许验。"""
        from agent.tools.research._loop import specs_from_planning

        planning = {
            "specs": [{
                # 模型在同一次建模里给出的"组成部分" —— 量词完全正确, 与审美毫无关系
                "claim_text": "对所有 n>=1, sum(k=1..n) k^2 = n(n+1)(2n+1)/6",
                "scope": "n=1..500", "how": "symbolic", "quantifier": "forall",
            }],
            "undecidable": ["'优美'是主观审美判断, 不存在可形式化的真假判据"],
        }
        specs, _ = specs_from_planning(
            planning, question="用归纳法证明的那段证明是优美的"
        )
        auto = [s for s in specs if s.get("decidable", True)]
        assert not auto, f"原问题不可判定, 却仍有 {len(auto)} 条要自动验证"
        assert all(s["state"] == "undecidable" for s in specs)

    def test_warning_explains_why(self):
        from agent.tools.research._loop import specs_from_planning

        specs, _ = specs_from_planning(
            {"specs": [{"claim_text": "求和公式成立", "scope": "n=1..10", "how": "symbolic"}]},
            question="这段证明是优美的吗",
        )
        warns = " ".join(w for s in specs for w in s.get("warn", []))
        assert "不可判定" in warns, warns

    def test_decidable_question_is_unaffected(self):
        """别把正常问题也一起拦了 —— 误报会让整个工具变得没法用。"""
        from agent.tools.research._loop import specs_from_planning

        specs, _ = specs_from_planning(
            {"specs": [{
                "claim_text": "存在正整数 n 使 n^2+n+41 是合数",
                "scope": "1<=n<=100", "how": "enumerate", "quantifier": "exists",
            }]},
            question="存在正整数 n 使得 n^2+n+41 是合数吗",
        )
        assert any(s.get("decidable") for s in specs)


# ============================================================
# 7. ★ 证据不重复落库 —— 否则历史 verdict 会"复活"
# ============================================================
class TestEvidenceNotDuplicated:
    """★ 这是四道防线全部上线后基线**一个数字都没改善**的真正原因。

    失效链(P4.5 实测):

    1. `_finish_without_verify` 把 `ws.load_evidence()`(全量历史)喂给 finalize
    2. finalize 的 `add_evidence()` 把它们又 INSERT 一遍
       → 同一次验证结果在 evidence 表里存了 3 份(实测 21/22/23、56/57/58、105/106/107)
    3. `add_evidence()` 顺带 `UPDATE claim.updated_at`
       → 13:49 的旧 `proved` 在"本轮"被刷新, 混进基线统计
    4. 于是刚修好的防线四(审美题已全部拦下)看起来毫无效果

    注意第 4 步的隐蔽性: `_plan.json` 和 `_status.json("无可验命题")` 都证明
    防线**确实生效了**, 只有 db 里的数字是错的。查工作区不看 db 就会误判成"修复无效"。
    """

    def test_finish_without_verify_does_not_reload_history(self):
        """无可验命题时必须传空列表, 不能 load_evidence()。

        ★ 用 AST 检查而不是 `assert "load_evidence" not in inspect.getsource(...)` ——
          源码里那串 `load_evidence` 出现在**注释**中(解释这个坑的来历),
          文本匹配会误判成"代码还在调用它"。我第一版就是这么写的, 立刻红。
          文本断言看起来更直白, 但它既拦不住真问题也会被注释误伤。
        """
        import ast
        import inspect
        import textwrap

        from agent.tools.research.research_investigate import _finish_without_verify

        tree = ast.parse(textwrap.dedent(inspect.getsource(_finish_without_verify)))
        calls = [
            n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        assert "load_evidence" not in calls, (
            f"_finish_without_verify 又把历史证据喂回 finalize 了(调用了 {calls}) —— "
            f"同一次验证会被重复记录, 且旧 verdict 会复活"
        )
        assert "load_evidence" not in [
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
        ]

    def test_state_tracks_only_new_evidence(self):
        from agent.tools.research._loop import new_state

        st = new_state("q", "th", 3)
        assert "new_evidence" in st
        assert st["new_evidence"] == []

    def test_run_loop_finalize_uses_new_evidence(self):
        import ast
        import inspect
        import textwrap

        from agent.tools.research._loop import run_loop

        tree = ast.parse(textwrap.dedent(inspect.getsource(run_loop)))
        calls = [
            n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        assert "load_evidence" not in calls, (
            f"run_loop 收尾时不能调 ws.load_evidence()(调用了 {calls}), 只能落本次新增的证据"
        )

    def test_verify_one_appends_to_new_evidence(self):
        from agent.tools.research._loop import new_state, verify_one

        st = new_state("q", "th", 1)
        spec = normalize_claim("P 不成立", scope="n=1..10")
        spec["claim_id"] = "c_test_1"
        st["specs"] = [spec]
        # 不给 code → 走"没有可执行的验证代码"分支, 0 LLM
        ev = verify_one(spec, WorkspaceStub(), st, decide={"how": "symbolic"}, code="")
        assert ev["verdict"] == "failed"
        assert st["new_evidence"] == [ev]


class WorkspaceStub:
    """够 _execute/record 用的最小 workspace 替身。"""

    def __init__(self):
        self.path = "."
        self.evidence_count = lambda: 0

    def append_evidence(self, ev):
        pass

    def append_attempt(self, a):
        pass


# ============================================================
# 8. ★ 防线五: 范围偷换 —— 量词对, scope 照样能被换掉
# ============================================================
class TestScopeGuard:
    """★ P4.5 第二轮实测失效现场。量词防线对此完全无能为力。

        问: "存在 n>100000 使得 n!+1 是素数吗"
        跑: "存在 n, 200 < n <= 500, 使得 n!+1 是素数"  → proved

    量词是 exists, 和问题的 exists **完全一致** —— 量词防线一声不响。
    但 scope 从 ">100000" 缩到了 "200..500", 验的完全是另一段。
    """

    def test_p4_actual_scope_shrink_is_caught(self):
        from agent.tools.research._claim import scope_conflict

        msg = scope_conflict(
            "存在 n>100000 使得 n!+1 是素数吗",
            "200 < n <= 500",
        )
        assert msg is not None, "P4.5 实测的范围偷换没被抓住"
        assert "范围冲突" in msg

    def test_wider_scope_is_allowed(self):
        """验证范围**更宽**是合理的保守, 不该报冲突。"""
        from agent.tools.research._claim import scope_conflict

        assert scope_conflict("n>100000 时 P 成立吗", "n=1..200000") is None
        assert scope_conflict("n<=500 时 P 成立吗", "n=1..1000") is None

    def test_no_numeric_scope_is_not_flagged(self):
        from agent.tools.research._claim import scope_conflict

        assert scope_conflict("存在 n>100000 使 P", "n 是足够大的整数") is None

    def test_conflicting_claim_becomes_undecidable(self):
        spec = normalize_claim(
            "存在 n 使得 n!+1 是素数", scope="200 < n <= 500",
            quantifier="exists",
            question="存在 n>100000 使得 n!+1 是素数吗",
        )
        assert spec["decidable"] is False
        assert any("范围冲突" in w for w in spec["warn"])

    @pytest.mark.parametrize("q,scope", [
        ("n>100000 时成立", "1..500"),
        ("n<=500 时成立", "1000..2000"),
    ])
    def test_both_directions(self, q, scope):
        from agent.tools.research._claim import scope_conflict

        assert scope_conflict(q, scope) is not None


# ============================================================
# 9. ★ 审计第 7 类: proved 也得有实质支撑
# ============================================================
class TestProvedNeedsSupport:
    """对称于 `refuted_without_counterexample`。反着来一样不可信。

    P4.5 实跑: 「最小的不存在的素数是多少」被判 proved, 理由是
    "不存在 m >= 1 使得 n = 2^m-1 是素数且 n < 2" —— **空洞真命题**:
    为真只因为约束弱到几乎没有反例空间, 而且偷换成了 2^m-1。
    """

    def test_proved_without_output_is_error(self):
        spec = normalize_claim("P 成立", scope="n=1..100")
        ev = [{
            "id": "ev_0001", "claim_id": spec["claim_id"], "kind": "code",
            "verdict": "proved", "reason": "成立", "stdout": "",
        }]
        res = audit_claims([spec], ev)
        hits = [i for i in res["issues"] if i["code"] == "proved_without_support"]
        assert hits, res["issues"]
        # ★ 必须连 level 一起断言。只 assert code 的话, 把 add("error") 改成
        #   add("warn") 测试照样绿 —— 而那样这条检查就只剩提示作用了。
        #   这个洞是靠反向验证抓到的: 变异降级 level, 测试没红。
        assert hits[0]["level"] == "error", hits[0]
        assert res["n_errors"] >= 1

    def test_proved_with_output_is_clean(self):
        spec = normalize_claim("P 成立", scope="n=1..100")
        ev = [{
            "id": "ev_0001", "claim_id": spec["claim_id"], "kind": "code",
            "verdict": "proved", "reason": "成立",
            "stdout": "n=40 -> 1681 = 41^2 是合数",
        }]
        res = audit_claims([spec], ev)
        assert res["n_errors"] == 0, res["issues"]


# ============================================================
# 10. ★ 防线八: 阈值偷换 —— 范围外的另一种偷换
# ============================================================
class TestThresholdGuard:
    """★ P4.5 第三轮实测失效现场。范围防线对这个完全无能为力。

        问: 哥德巴赫猜想成立吗(每个大偶数 = 两素数之和, 即**至少 1 组**)
        跑: 拆成 "对所有偶数 n 满足 4<=n<=200000, **存在至少 1000 组**素数对"
            → 代码发现 n=4 只有 1 组 → VERDICT: REFUTED

    逻辑上一点没错(n=4 确实凑不出 1000 组), 但哥德巴赫问的是"至少 1 组"。
    范围 4..200000 完全合理, 所以 `scope_conflict` 一声不响 ——
    **偷换不一定发生在范围上, 也可能发生在数量阈值上。**
    """

    def test_p4_actual_threshold_inflation_is_caught(self):
        from agent.tools.research._claim import threshold_conflict

        msg = threshold_conflict(
            "哥德巴赫猜想成立吗",
            "对所有偶数 n 满足 4 <= n <= 200000, 存在至少 1000 组素数对",
        )
        assert msg is not None, "P4.5 实测的阈值偷换没被抓住"
        assert "阈值" in msg

    def test_matching_threshold_is_allowed(self):
        """问题和命题都说"至少 1 组" → 正常, 不该报。"""
        from agent.tools.research._claim import threshold_conflict

        assert threshold_conflict(
            "每个大偶数能写成至少 1 组两素数之和吗",
            "对所有偶数 n, 存在至少 1 组素数对",
        ) is None

    def test_formula_numbers_are_not_thresholds(self):
        """★ 别把 "n^2+n+41" 里的 41 当成阈值。误报多了就没人看这个告警。"""
        from agent.tools.research._claim import threshold_conflict

        assert threshold_conflict(
            "n^2+n+41 是素数吗", "对所有 n>=2, n^2+n+41 是素数"
        ) is None
        assert threshold_conflict(
            "哥德巴赫猜想成立吗", "存在 n 使 n^2+n+41 是合数"
        ) is None

    def test_no_threshold_context_is_not_flagged(self):
        from agent.tools.research._claim import threshold_conflict

        assert threshold_conflict("n^2+n+41 是素数吗", "n^2+n+41 是素数") is None

    def test_conflicting_claim_becomes_undecidable(self):
        spec = normalize_claim(
            "对所有偶数 n 满足 4 <= n <= 200000, 存在至少 1000 组素数对",
            scope="n=4..200000",
            quantifier="forall",
            question="哥德巴赫猜想成立吗",
        )
        assert spec["decidable"] is False
        assert any("阈值" in w for w in spec["warn"])


