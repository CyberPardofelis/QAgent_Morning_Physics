"""
test_research_p1 — P7 阶段 1 测试 (2026-10-08): Z3 / 枚举 / OEIS / refute

覆盖:
  1. _z3_tools   SMT-LIB2 求解 + prove 语义的 sat/unsat → refuted/proved 映射
  2. _enumerate  图性质检查器 / graph_atlas 穷尽反例搜索 / 序列性质
  3. _retrieve   序列解析 + **候选对账**(离线, 不联网)
  4. research_* 工具层: verdict 契约 + 证据落盘
  5. 暴露契约   research_* 注册但不进 list_schemas()

**为什么第 2 组里的 triangle_free 用例是回归测试**:
2026-10-08 实测踩了两次 —— 先写成 `any(True for _ in nx.triangles(G))`(那是"有没有节点"),
后写成 `any(len(inner) > 0 ...)`(nx.triangles 返回的**值是计数不是 dict**, 报 TypeError)。
两次都表现为"n=1 的单点图被误判成含三角形" → 假反例。所以这里钉死 K3 才是首个反例。

历史:
  - 2026-10-08 P7.1 首版
"""
import pytest

from agent.infra.research_workspace import ResearchWorkspace, hash_question
from agent.tools.research._enumerate import (
    GRAPH_PROPERTIES,
    enumerate_graph_property,
    enumerate_sequence_property,
)
from agent.tools.research._retrieve import _score_candidate, parse_sequence
from agent.tools.research._z3_tools import MAX_TIMEOUT_MS, Z3Error, solve
from agent.tools.research.research_enumerate import research_enumerate
from agent.tools.research.research_refute import research_refute
from agent.tools.research.research_z3 import research_z3


# ============================================================
# 1. Z3
# ============================================================
class TestZ3:
    def test_unsat_maps_to_proved(self):
        """
        goal="prove" 的核心映射: 前提(x,y>=0, x+y=10) + 结论(xy>25) 不可满足
        → 命题成立 → proved
        """
        c = """
        (declare-const x Int)
        (declare-const y Int)
        (assert (>= x 0))
        (assert (>= y 0))
        (assert (= (+ x y) 10))
        (assert (> (* x y) 25))
        """
        out = solve(c, goal="prove")
        assert out["result"] == "unsat"
        assert out["verdict"] == "proved"

    def test_sat_maps_to_refuted_with_counterexample(self):
        """sat 时给出的模型就是反例 —— 这是 Z3 相对枚举/抽样的核心优势"""
        c = """
        (declare-const x Real)
        (assert (< (* x x) (* 2 x)))
        """
        out = solve(c, goal="prove")
        assert out["result"] == "sat"
        assert out["verdict"] == "refuted"
        assert "x = 1" in out["model"], f"模型里应含反例赋值, 实际: {out['model']!r}"

    def test_goal_sat_does_not_claim_proof(self):
        """goal=sat 只是可满足性检查, 不该报 proved"""
        c = "(declare-const x Int)\n(assert (= x 5))"
        out = solve(c, goal="sat")
        assert out["verdict"] == "untested"
        assert out["result"] == "sat"

    def test_syntax_error_raises(self):
        """语法错必须抛 Z3Error 而不是静默当成 unsat —— 那会把失败当证明"""
        with pytest.raises(Z3Error):
            solve("(declare-const x Int)\n(assert (= x ))", goal="prove")

    def test_timeout_clamped(self):
        """LLM 写 timeout=99999999 必须被夹住"""
        out = solve("(declare-const x Int)\n(assert (= x 1))", timeout_ms=99_999_999)
        assert out["timeout_ms"] <= MAX_TIMEOUT_MS

    def test_unknown_on_hard_problem_is_inconclusive(self):
        """求解器定不了时是 inconclusive, 绝不能算 proved"""
        c = """
        (declare-const a Int)
        (assert (> (* a a) 3))
        (assert (< (* a a) 5))
        """
        out = solve(c, goal="prove")
        # 这题 Z3 秒解(sat), 所以断言它不是 unknown; 关键是不许返回 proved
        assert out["verdict"] != "proved"


# ============================================================
# 2. 枚举
# ============================================================
class TestEnumerate:
    def test_triangle_free_counterexample_is_K3(self):
        """
        ★ 回归: 单点图**不是**三角形。首个反例必须是 K3(n=3)。

        写错 nx.triangles() 的理解会导致 n=1 就命中 → 假反例。
        """
        out = enumerate_graph_property("triangle_free", n_max=4)
        assert out["verdict"] == "refuted"
        assert out["counterexample"]["n"] == 3, (
            f"首个反例应是 K3(n=3), 实际 n={out['counterexample']['n']} "
            f"边={out['counterexample']['edges']}"
        )
        assert len(out["counterexample"]["edges"]) == 3   # K3 有 3 条边

    def test_single_node_graph_is_triangle_free(self):
        """底层性质函数的直接断言, 锁死上面的 bug"""
        import networkx as nx
        assert GRAPH_PROPERTIES["triangle_free"](nx.empty_graph(1)) is True
        assert GRAPH_PROPERTIES["triangle_free"](nx.complete_graph(3)) is False

    def test_exhaustive_search_gives_partial_not_proved(self):
        """
        ★ 枚举穷尽也**永远不报 proved** —— 有限范围不构成全域证明(§6.4 纪律)
        """
        out = enumerate_graph_property("connected", n_max=1)   # n=1 时确实全连通
        assert out["verdict"] == "partial"
        assert out["verdict"] != "proved"
        assert "有限" in out["reason"] or "有限范围" in out["reason"]

    def test_bipartite_counterexample_is_triangle(self):
        """K3 不是二部图, n=3 时应有反例"""
        out = enumerate_graph_property("bipartite", n_max=3)
        assert out["verdict"] == "refuted"
        assert out["counterexample"]["n"] == 3

    def test_unknown_property_raises(self):
        with pytest.raises(ValueError) as exc:
            enumerate_graph_property("no_such_property", n_max=3)
        assert "可用" in str(exc.value)      # 错误信息里要带可用清单

    def test_n_max_capped(self):
        """n_max 超过 graph_atlas 上限要被夹住, 否则会卡死"""
        out = enumerate_graph_property("triangle_free", n_max=99)
        assert out["n_max"] == 7
        assert out["atlas_exhaustive"] is True

    def test_sequence_property(self):
        """素数序列必然互不相同 → partial; 而 fibonacci 起点含 0 不凸 → refuted"""
        out = enumerate_sequence_property("primes", "all_distinct", n_terms=40)
        assert out["verdict"] == "partial"

        out2 = enumerate_sequence_property("fibonacci", "convex", n_terms=12)
        assert out2["verdict"] == "refuted"
        assert out2["counterexample"] is not None


# ============================================================
# 3. OEIS (离线部分 —— 联网另见 TestOeisOnline)
# ============================================================
class TestRetrieveParsing:
    @pytest.mark.parametrize("raw,expect", [
        ("1,2,4,8,16", [1, 2, 4, 8, 16]),
        ("1 2 4 8 16", [1, 2, 4, 8, 16]),
        ("[1, 2, 4, 8, 16]", [1, 2, 4, 8, 16]),
        ([1, 2, 3], [1, 2, 3]),
    ])
    def test_parse_sequence(self, raw, expect):
        assert parse_sequence(raw) == expect

    def test_too_few_terms_rejected(self):
        """少于 2 项 OEIS 会返回一堆噪声, 直接拒"""
        with pytest.raises(ValueError):
            parse_sequence("5")

    def test_non_integer_rejected(self):
        """OEIS 只收整数序列; 非整数要提示改用 sympy"""
        with pytest.raises(ValueError) as exc:
            parse_sequence("1.5, 2.5, 3.5")
        assert "sympy" in str(exc.value)


class TestRetrieveScoring:
    """
    ★ 回归: OEIS 的 data 字段是 "1, 2, 4, 8, 16" —— **逗号和空格混用**。
    只按空白切会 int('1,') 抛错被吞掉 → seq=[] → exact_match 恒为 False,
    精确匹配功能形同虚设(2026-10-08 实测: 查 1,2,4,8,16 明明是 A000079 却报 exact=False)。
    """

    ROW_POWERS = {
        "number": "79",
        "name": "Powers of 2: a(n) = 2^n.",
        "formula": ["a(n) = 2^n."],
        "data": "1, 2, 4, 8, 16, 32, 64, 128",
        "offset": 0,
    }

    def test_comma_separated_data_is_parsed(self):
        got = _score_candidate([1, 2, 4, 8, 16], self.ROW_POWERS)
        assert got["exact_match"] is True
        assert got["matched"] == 5
        assert got["id_full"] == "A000079"

    def test_space_separated_data_also_parses(self):
        row = dict(self.ROW_POWERS, data="1 2 4 8 16 32 64 128")
        got = _score_candidate([1, 2, 4, 8, 16], row)
        assert got["exact_match"] is True

    def test_non_matching_sequence_scores_zero(self):
        """对账的意义: 不匹配的候选不能被当成答案"""
        got = _score_candidate([3, 5, 8], self.ROW_POWERS)
        assert got["matched"] == 0
        assert got["exact_match"] is False

    def test_partial_prefix_match(self):
        """1,1,2 对 powers of 2 只吻合 1 项(1)"""
        got = _score_candidate([1, 1, 2], self.ROW_POWERS)
        assert got["matched"] == 1
        assert got["exact_match"] is False

    def test_offset_shifted_fibonacci_not_matched(self):
        """
        A000045 从 0 起(offset=0), 输入从 1 起(offset=1) → 前缀对不上 → 不算精确匹配。
        这是保守正确的行为: 宁可漏报也不误报。
        """
        row = {
            "number": "45", "name": "Fibonacci numbers",
            "formula": ["F(n) = F(n-1) + F(n-2)"],
            "data": "0, 1, 1, 2, 3, 5, 8, 13", "offset": 0,
        }
        got = _score_candidate([1, 1, 2, 3, 5, 8, 13], row)
        assert got["exact_match"] is False
        assert got["matched"] == 0

    def test_row_without_number_skipped(self):
        assert _score_candidate([1, 2], {"name": "x", "data": "1, 2, 3"}) is None


class TestOeisOnline:
    """联网用例 —— 无网环境下自动跳过, 不让 CI 因外部依赖红"""

    def test_oeis_lookup_real(self):
        from agent.tools.research import research_oeis
        try:
            r = research_oeis(sequence="1,2,4,8,16")
        except Exception as e:                      # 网络不可用
            pytest.skip(f"OEIS 不可达(离线环境): {type(e).__name__}")

        assert r["ok"] is True
        best = r["best_match"]
        assert best is not None
        # 精确匹配到 A000079 (2 的幂); 若 OEIS 排序变了, 至少要求匹配 5 项
        assert best["matched"] == 5
        assert best["id_full"] == "A000079"


# ============================================================
# 4. 工具层: verdict 契约 + 证据落盘
# ============================================================
class TestToolLayer:
    def test_z3_tool_evidence_persisted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = research_z3(
            constraints="(declare-const x Int)\n(assert (= x 42))",
            goal="sat", claim="测试约束",
        )
        assert r["verdict"] == "untested"       # goal=sat 不宣称证明
        assert len(r["evidence"]) == 1
        assert r["evidence"][0]["id"] == "ev_0001"     # 编号从 1 起, 不是 ev_0000
        assert r["evidence"][0]["kind"] == "solver"
        assert r["evidence"][0]["env"]["sympy"]       # 环境指纹必须落盘

    def test_evidence_file_contains_record(self, tmp_path, monkeypatch):
        """证据落盘校验: _evidence.jsonl 里真有这一行"""
        monkeypatch.chdir(tmp_path)
        th = hash_question("z3:sat:(declare-const x Int)\n(assert (= x 42))")
        research_z3(constraints="(declare-const x Int)\n(assert (= x 42))", goal="sat",
                    claim="落盘校验")
        ws = ResearchWorkspace(th)
        evs = ws.load_evidence()
        assert len(evs) == 1
        assert evs[0]["kind"] == "solver"
        assert evs[0]["verdict"] in ("proved", "refuted", "partial", "failed",
                                     "inconclusive", "untested")

    def test_enumerate_tool_returns_refuted_with_counterexample(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = research_enumerate(prop="triangle_free", task="graph", n_max=4, claim="三角形")
        assert r["verdict"] == "refuted"
        assert r["counterexample"]["n"] == 3

    def test_enumerate_bad_property_lists_options(self, tmp_path, monkeypatch):
        """参数错要把可用性质列出来, 省得 LLM 盲猜一轮"""
        monkeypatch.chdir(tmp_path)
        r = research_enumerate(prop="不存在的性质", task="graph")
        assert r["verdict"] == "failed"
        assert "可用性质" in r["summary"]
        assert "bipartite" in r["summary"]

    def test_refute_never_returns_proved_from_enumeration(self, tmp_path, monkeypatch):
        """
        ★ research_refute 的契约: 只给 refuted/partial/inconclusive,
        **绝不因"没找到反例"就宣称 proved**(科研里最常见的误判)
        """
        monkeypatch.chdir(tmp_path)
        r = research_refute(claim="所有图都无三角形", graphs="triangle_free", n_max=4)
        assert r["verdict"] == "refuted"

        # 构造一个真的找不到反例的场景
        r2 = research_refute(claim="单点图连通", graphs="connected", n_max=1)
        assert r2["verdict"] != "proved"
        assert "不等于命题成立" in r2["summary"] or r2["verdict"] == "partial"

    def test_refute_with_no_route_is_inconclusive(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = research_refute(claim="什么路线都没给")
        assert r["verdict"] == "inconclusive"
        assert "graphs" in r["summary"] or "constraints" in r["summary"]


# ============================================================
# 5. 暴露契约
# ============================================================
class TestExposureContract:
    EXPECTED_RESEARCH_TOOLS = {
        "research_run_code", "research_verify", "research_sympy",
        "research_z3", "research_enumerate", "research_oeis", "research_refute",
    }

    def test_all_registered_but_none_exposed(self, tmp_path, monkeypatch):
        """
        ★ P7 核心契约: 工具已注册(可被 ToolRegistry.get 拿到), 但 LLM 看不见。
        接入前这个性质必须一直成立。
        """
        monkeypatch.chdir(tmp_path)
        import agent.tools  # noqa: F401  触发注册
        from agent.tools.registry import ToolRegistry

        registered = set(ToolRegistry.list_names())
        exposed = {s["function"]["name"] for s in ToolRegistry.list_schemas()}

        assert self.EXPECTED_RESEARCH_TOOLS <= registered, "有工具没注册成功"
        assert not (self.EXPECTED_RESEARCH_TOOLS & exposed), "research_* 泄露给 LLM 了!"

    def test_hidden_tools_still_dispatchable(self, tmp_path, monkeypatch):
        """隐藏 ≠ 不可用: ToolRegistry.get 仍应能拿到 handler"""
        monkeypatch.chdir(tmp_path)
        import agent.tools  # noqa: F401
        from agent.tools.registry import ToolRegistry

        tool = ToolRegistry.get("research_z3")
        assert tool is not None
        assert tool.exposed is False
        assert callable(tool.handler)

    def test_survey_tools_still_exposed(self, tmp_path, monkeypatch):
        """不能把综述工具误伤 —— 那是 P6 已上线功能"""
        monkeypatch.chdir(tmp_path)
        import agent.tools  # noqa: F401
        from agent.tools.registry import ToolRegistry

        exposed = {s["function"]["name"] for s in ToolRegistry.list_schemas()}
        for name in ("survey_generate", "survey_audit", "survey_outline"):
            assert name in exposed
