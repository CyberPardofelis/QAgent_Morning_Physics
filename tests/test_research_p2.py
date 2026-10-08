"""
test_research_p2 — P7 阶段 2 测试 (2026-10-08): 命题层 / 长期库 / 循环 / 报告 / 审计

覆盖:
  1. _claim    命题规范化 + 可判定性判定 + **证据绑定**(报告渲染)+ 审计(0 LLM)
  2. _db       research.db: claim 内容哈希合并 / evidence 只 append / 判定历史
  3. _loop     决策优先级 / 完整循环(注入 code_provider, **不烧 token**)
  4. 工具层    research_report / research_audit
  5. 暴露契约  11 个 research_* 注册但全部不暴露

**为什么第 1 组的 render_report 测试是核心**:
"结论必须挂 ev_ 编号" 这条纪律如果只写在 prompt 里, 换个模板就绕开了。
这里钉死: 拿不到证据的命题**不允许**出现在结论区, 必须进"未验证区"。

**为什么第 3 组注入 code_provider**:
P0 的教训 —— "生成代码"和"执行代码"耦合后就没法脱离 LLM 单测。
循环逻辑(重试/统计/收尾/审计)必须能在 0 token 下测, 所以 code_provider 是注入的。

历史:
  - 2026-10-08 P7.2 首版
"""

from agent.infra.research_workspace import ResearchWorkspace
from agent.tools.research._claim import (
    audit_claims,
    dedupe_claims,
    normalize_claim,
    render_report,
)


# ============================================================
# 1. 命题规范化
# ============================================================
class TestNormalizeClaim:
    def test_wellformed_claim(self):
        s = normalize_claim("sum(k=1..n) k^2 = n(n+1)(2n+1)/6", "n>=1 全体整数", "symbolic")
        assert s["decidable"] is True
        assert s["state"] == "pending"
        assert s["claim_id"] and len(s["claim_id"]) == 12
        assert s["warn"] == []

    def test_missing_scope_warns_but_not_undecidable(self):
        """缺 scope 要警告, 但不能直接判死 —— 用户可能就是要验个简单命题"""
        s = normalize_claim("1+1=2", scope="")
        assert s["decidable"] is True
        assert any("变量域" in w for w in s["warn"])

    def test_explicit_undecidable(self):
        s = normalize_claim("这个猜想优美吗", how="undecidable")
        assert s["decidable"] is False
        assert s["state"] == "undecidable"

    def test_aesthetic_question_auto_undecidable(self):
        """没有 scope 时, 审美类表述自动判不可判定 —— 这类问题不该被硬凑成断言"""
        s = normalize_claim("这个证明优美吗")
        assert s["decidable"] is False
        assert s["state"] == "undecidable"

    def test_explicit_claim_id_is_preserved(self):
        """
        ★ 回归: 显式传入 claim_id 必须被沿用, 不能被 **extra 覆盖掉重算值。
        (审计从 _specs.jsonl 重建时要靠原 id 对齐证据, 否则证据全成孤儿)
        """
        s = normalize_claim("x=1", scope="x in Z", claim_id="deadbeef1234")
        assert s["claim_id"] == "deadbeef1234"

    def test_same_text_same_id(self):
        a = normalize_claim("命题A", "范围1")
        b = normalize_claim("命题A", "范围1")
        assert a["claim_id"] == b["claim_id"]


class TestDedupeClaims:
    def test_exact_duplicates_dropped(self):
        """完全相同(同文本同范围)的重复要去掉 —— LLM 经常把一条命题写两遍"""
        a = normalize_claim("同一命题", "n<=10")
        b = normalize_claim("同一命题", "n<=10")
        specs, dropped = dedupe_claims([a, b])
        assert len(specs) == 1
        assert dropped == 1

    def test_different_scope_not_merged(self):
        """
        ★ 不同范围**不合并**: n<=10 无反例 与 n<=20 查出反例 是两条不同结论,
        合成一条就抹掉了"反例出现在 11~20"这个最有价值的信息。
        """
        a = normalize_claim("同一命题", "n<=10")
        b = normalize_claim("同一命题", "n<=20")
        specs, dropped = dedupe_claims([a, b])
        assert len(specs) == 2
        assert dropped == 0


# ============================================================
# 2. 报告渲染 —— 证据绑定的硬规则
# ============================================================
def _spec(text, scope, cid):
    return {"claim_text": text, "scope": scope, "how": "symbolic",
            "decidable": True, "state": "pending", "claim_id": cid, "warn": []}


class TestRenderReport:
    def test_claim_with_evidence_goes_to_conclusion(self):
        spec = _spec("命题A", "n<=100", "cid_a")
        ev = {"id": "ev_0001", "claim_id": "cid_a", "verdict": "proved",
              "reason": "化简为 0", "kind": "code", "stdout": "", "stderr": "", "elapsed_ms": 5}
        md = render_report([spec], [ev], question="Q", env={"sympy": "1.14.0"})
        concl, unver = md.split("## 未验证区")
        assert "命题A" in concl
        assert "ev_0001" in concl
        assert "命题A" not in unver

    def test_claim_without_evidence_cannot_reach_conclusion(self):
        """
        ★ 核心纪律: 拿不到证据的命题**绝不**出现在结论区。
        prompt 里写过这条没用, 这里才是代码强制的位置。
        """
        spec = _spec("裸命题", "n<=100", "cid_x")
        md = render_report([spec], [], question="Q")
        concl, unver = md.split("## 未验证区")
        assert "裸命题" not in concl
        assert "裸命题" in unver

    def test_partial_must_carry_scope_caveat(self):
        """partial 不能被当完整结论 —— 报告里必须带范围限定"""
        spec = _spec("只在 n<=7 成立", "n<=7", "cid_p")
        ev = {"id": "ev_0002", "claim_id": "cid_p", "verdict": "partial",
              "reason": "穷尽 n<=7", "kind": "code", "stdout": "", "stderr": "", "elapsed_ms": 5}
        md = render_report([spec], [ev])
        assert "不是完整结论" in md
        assert "n<=7" in md

    def test_repeated_verification_history_shown(self):
        """反复验证要能看出历史 —— 结论翻车时全靠这条"""
        spec = _spec("反复命题", "x", "cid_r")
        evs = [
            {"id": "ev_0001", "claim_id": "cid_r", "verdict": "proved", "reason": "",
             "kind": "code", "stdout": "", "stderr": "", "elapsed_ms": 1},
            {"id": "ev_0002", "claim_id": "cid_r", "verdict": "refuted", "reason": "",
             "kind": "code", "stdout": "", "stderr": "", "elapsed_ms": 1},
        ]
        md = render_report([spec], evs)
        assert "被验证了 2 次" in md
        assert "proved → refuted" in md

    def test_env_fingerprint_printed(self):
        """没有环境指纹, 证据半年后无法复现 = 等于没有证据"""
        spec = _spec("x", "y", "cid_e")
        ev = {"id": "ev_0001", "claim_id": "cid_e", "verdict": "proved", "reason": "",
              "kind": "code", "stdout": "", "stderr": "", "elapsed_ms": 1}
        md = render_report([spec], [ev], env={"sympy": "1.14.0", "python": "3.10.6"})
        assert "sympy=1.14.0" in md and "python=3.10.6" in md


# ============================================================
# 3. 审计 —— 0 LLM 的硬约束
# ============================================================
class TestAudit:
    def _ev(self, eid, cid, verdict, kind="code", stdout=None):
        # ★ stdout 默认带上 VERDICT 标记, 贴近真实证据形态。
        #   真实链路里 judge_verdict 靠 `print("VERDICT: PROVED")` 判定,
        #   所以一条 proved 证据的 stdout **必然非空**。
        #   P4.5 加了 `proved_without_support` 检查(空 stdout 不算证明)之后,
        #   这个 fixture 不补 stdout 就会红 —— 那是 fixture 不真实, 不是检查太严。
        if stdout is None:
            stdout = f"VERDICT: {verdict.upper()}\n检查完成"
        return {"id": eid, "claim_id": cid, "verdict": verdict, "kind": kind,
                "reason": "", "stdout": stdout, "stderr": "", "elapsed_ms": 1}

    def test_clean_case_passes(self):
        specs = [_spec("好命题", "n<=10", "cid1")]
        evs = [self._ev("ev_0001", "cid1", "proved")]
        r = audit_claims(specs, evs, report_md="### PROVED — 好命题\n- 证据: `ev_0001`\n")
        assert r["ok"] is True
        assert r["n_errors"] == 0

    def test_verdict_conflict_detected(self):
        """★ 同一命题先证成后被反驳 —— 最该抓的一类翻车"""
        specs = [_spec("矛盾命题", "x", "cid2")]
        evs = [self._ev("ev_0001", "cid2", "proved"), self._ev("ev_0002", "cid2", "refuted")]
        r = audit_claims(specs, evs)
        assert r["ok"] is False
        codes = {i["code"] for i in r["issues"]}
        assert "verdict_conflict" in codes

    def test_bare_assertion_detected(self):
        """★ 结论行宣称 PROVED 却没挂 ev_ 编号 = 裸断言"""
        specs = [_spec("裸断言命题", "x", "cid3")]
        evs = [self._ev("ev_0001", "cid3", "proved")]
        md = "### PROVED — 裸断言命题\n- 没有任何证据编号\n"
        r = audit_claims(specs, evs, report_md=md)
        assert r["ok"] is False
        assert "bare_assertion" in {i["code"] for i in r["issues"]}

    def test_proved_from_citation_only_is_error(self):
        """★ 拿文献当证明 = proved 只有 citation 类证据"""
        specs = [_spec("引用型伪证明", "x", "cid4")]
        evs = [self._ev("ev_0001", "cid4", "proved", kind="citation")]
        r = audit_claims(specs, evs)
        assert r["ok"] is False
        assert "proved_from_citation" in {i["code"] for i in r["issues"]}

    def test_orphan_evidence_detected(self):
        """证据指向不存在的命题"""
        specs = [_spec("存在的命题", "x", "cid_ok")]
        evs = [self._ev("ev_0001", "cid_不存在", "proved")]
        r = audit_claims(specs, evs)
        assert "orphan_evidence" in {i["code"] for i in r["issues"]}

    def test_proved_with_partial_is_info_not_error(self):
        """proved + partial 共存通常正常(范围不同), 只提醒不报错"""
        specs = [_spec("混合命题", "n<=10", "cid5")]
        evs = [self._ev("ev_0001", "cid5", "partial"), self._ev("ev_0002", "cid5", "proved")]
        r = audit_claims(specs, evs)
        assert r["ok"] is True      # 不算硬伤
        assert any(i["code"] == "proved_with_partial" for i in r["issues"])


# ============================================================
# 4. 长期库
# ============================================================
class TestDb:
    def test_claim_merged_by_content_hash(self, tmp_path, monkeypatch):
        """★ 同命题不同任务必须合并成一行 —— 这是 research.db 存在的理由"""
        monkeypatch.chdir(tmp_path)
        from agent.tools.research import _db

        _db.init_schema()
        cid = "testmerge12"
        _db.upsert_claim(cid, "同一命题", task_hash="t1", scope="n<=10", verdict="partial")
        _db.upsert_claim(cid, "同一命题", task_hash="t2", scope="n<=20", verdict="proved")
        _db.upsert_claim(cid, "同一命题", task_hash="t3", scope="n<=30", verdict="proved")

        rows = [c for c in _db.list_claims() if c["id"] == cid]
        assert len(rows) == 1, "同内容哈希的命题必须合并成一行"
        assert rows[0]["verdict"] == "proved"        # 取最新判定
        # scope 取**最新**的, 不累积 —— 累积会让 scope 无限变长且分不清哪次有效
        assert rows[0]["scope"] == "n<=30"

    def test_evidence_only_appends(self, tmp_path, monkeypatch):
        """★ 证据永不覆盖 —— 覆盖了就永远查不清当时跑过什么"""
        monkeypatch.chdir(tmp_path)
        from agent.tools.research import _db

        _db.init_schema()
        _db.upsert_claim("cid_ev", "命题", task_hash="t1")
        _db.add_evidence({"claim_id": "cid_ev", "kind": "code", "verdict": "proved",
                          "reason": "r1", "code": "print(1)", "stdout": "", "stderr": "",
                          "elapsed_ms": 1}, task_hash="t1")
        _db.add_evidence({"claim_id": "cid_ev", "kind": "code", "verdict": "refuted",
                          "reason": "r2", "code": "print(2)", "stdout": "", "stderr": "",
                          "elapsed_ms": 1}, task_hash="t1")
        evs = _db.list_evidence(claim_id="cid_ev")
        assert len(evs) == 2
        assert _db.claim_verdict_history("cid_ev") == ["proved", "refuted"]


# ============================================================
# 5. 循环 (注入 code_provider, 0 token)
# ============================================================
class TestLoop:
    def test_decide_prefers_refutation(self):
        """★ 先找反例 —— 找到 refuted 越早知道越好"""
        from agent.tools.research._loop import decide_next, new_state

        st = new_state("Q", "th", 3)
        a = normalize_claim("可证实命题", "n<=10", "symbolic", priority=1)
        b = normalize_claim("可反驳命题", "n<=8", "enumerate", priority=5)
        st["specs"] = [a, b]
        st["by_id"] = {a["claim_id"]: a, b["claim_id"]: b}
        st["pending"] = [a["claim_id"], b["claim_id"]]

        d = decide_next(st)
        assert d["claim_id"] == b["claim_id"], "应优先选 enumerate(可反驳)"

    def test_decide_none_when_all_done(self):
        from agent.tools.research._loop import decide_next, new_state
        st = new_state("Q", "th", 3)
        st["pending"] = []
        assert decide_next(st) is None

    def test_run_loop_end_to_end_without_llm(self, tmp_path, monkeypatch):
        """
        ★ 完整闭环 0 token 跑通: 两轮 -> 报告 + 审计, 结论与证据一一对应。
        code_provider 注入固定代码, 完全不碰 LLM。
        """
        monkeypatch.chdir(tmp_path)
        from agent.tools.research._loop import new_state, run_loop

        ws = ResearchWorkspace("loopcase01", question="测试问题")
        st = new_state("测试问题", "loopcase01", max_rounds=2)

        proved = normalize_claim("1+1 等于 2", "全体", "symbolic", priority=1)
        refuted = normalize_claim("所有素数都是奇数", "n<=20", "enumerate", priority=2)
        st["specs"] = [proved, refuted]
        st["by_id"] = {proved["claim_id"]: proved, refuted["claim_id"]: refuted}
        st["pending"] = [refuted["claim_id"], proved["claim_id"]]

        CODES = {
            "所有素数都是奇数": (
                "print('反例: n=2 -> 值 = 2 是偶数')\n"
                "print('VERDICT: REFUTED')\n"
            ),
            "1+1 等于 2": "import sympy\nprint('化简:', sympy.simplify(1+1-2))\n"
                          "print('VERDICT: PROVED')\n",
        }

        def provider(spec, decide, attempt, feedback=None):
            return CODES.get(spec["claim_text"], "")

        chunks = list(run_loop("测试问题", ws, st, code_provider=provider, max_rounds=2))
        assert any("🔬" in c for c in chunks)

        # 证据落盘
        evs = ws.load_evidence()
        assert len(evs) >= 2
        assert {e["verdict"] for e in evs} >= {"proved", "refuted"}

        # 报告: 结论与证据一一对应
        md = ws.load_report()
        assert md and "## 结论" in md
        assert "ev_0001" in md
        assert "审计" in md

        # 两轮都验完了
        assert st["stats"]["proved"] >= 1
        assert st["stats"]["refuted"] >= 1

    def test_run_loop_retries_on_failure(self, tmp_path, monkeypatch):
        """失败才重试; untested/partial 不重试(重试同题没意义)"""
        monkeypatch.chdir(tmp_path)
        from agent.tools.research._loop import new_state, run_loop

        ws = ResearchWorkspace("retrycase1", question="Q")
        st = new_state("Q", "retrycase1", max_rounds=1)
        spec = normalize_claim("会崩的命题", "x", "symbolic")
        st["specs"] = [spec]
        st["by_id"] = {spec["claim_id"]: spec}
        st["pending"] = [spec["claim_id"]]

        calls = {"n": 0}

        def provider(spec, decide, attempt, feedback=None):
            calls["n"] += 1
            # 第一次崩, 第二次成功
            return "raise ValueError('boom')\n" if attempt == 0 else (
                "print('VERDICT: PROVED')\n")

        list(run_loop("Q", ws, st, code_provider=provider, max_rounds=1))

        assert calls["n"] >= 2, "失败后应该重试"
        assert st["stats"]["proved"] == 1
        assert len(ws.load_attempts()) >= 1, "失败尝试要记进 _attempts.jsonl"


# ============================================================
# 6. 工具层
# ============================================================
class TestToolLayerP2:
    def test_report_tool_empty_workspace(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from agent.tools.research.research_report import research_report
        r = research_report(task_hash="nonexistent1")
        assert r["ok"] is False
        assert "空" in r["summary"] or "没有" in r["summary"]

    def test_audit_tool_detects_bare_assertion(self, tmp_path, monkeypatch):
        """端到端: 造一个裸断言报告, 审计必须抓到"""
        monkeypatch.chdir(tmp_path)
        from agent.tools.research.research_audit import research_audit

        ws = ResearchWorkspace("auditcase1", question="Q")
        spec = normalize_claim("可疑命题", "x", "symbolic")
        ws.save_plan({"normalized": [spec]})
        ws.append_spec({"claim_text": spec["claim_text"], "scope": "x",
                        "how": "symbolic", "claim_id": spec["claim_id"]})
        ws.append_evidence({"id": "ev_0001", "claim_id": spec["claim_id"],
                            "kind": "code", "verdict": "proved", "reason": "",
                            "code": "", "stdout": "", "stderr": "", "elapsed_ms": 1,
                            "env": {}})
        ws.save_report("### PROVED — 可疑命题\n- 这行故意不挂证据编号\n")

        r = research_audit(task_hash="auditcase1")
        assert r["ok"] is False
        assert "bare_assertion" in {i["code"] for i in r["issues"]}
        assert "不可信" in r["summary"]

    def test_plan_json_extraction_handles_fences(self):
        """LLM 经常给 markdown 围栏或前后解说, 提取要扛住"""
        from agent.tools.research.research_plan import _extract_json

        assert _extract_json('```json\n{"specs": [1]}\n```') == {"specs": [1]}
        assert _extract_json('好的:\n{"specs": [2]}\n以上') == {"specs": [2]}
        assert _extract_json("完全不是 JSON") == {}


# ============================================================
# 7. 暴露契约
# ============================================================
class TestExposureP2:
    ALL = {
        "research_run_code", "research_verify", "research_sympy", "research_z3",
        "research_enumerate", "research_oeis", "research_refute",
        "research_plan", "research_investigate", "research_report", "research_audit",
    }

    def test_all_registered_none_exposed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import agent.tools  # noqa: F401
        from agent.tools.registry import ToolRegistry

        assert self.ALL <= set(ToolRegistry.list_names()), "有工具没注册成功"
        exposed = {s["function"]["name"] for s in ToolRegistry.list_schemas()}
        assert not (self.ALL & exposed), "research_* 泄露给 LLM 了!"

    def test_investigate_is_streaming_l2(self, tmp_path, monkeypatch):
        """编排主工具必须是流式 + L2, 否则长任务看不到进度还会被成本闸门挡"""
        monkeypatch.chdir(tmp_path)
        import agent.tools  # noqa: F401
        from agent.tools.registry import ToolRegistry

        t = ToolRegistry.get("research_investigate")
        assert t.is_streaming is True
        assert t.cost_tier == "L2"
        assert t.exposed is False
