"""P4: 科研闭环评估集。

回答那个唯一重要的问题: 这玩意儿到底行不行?

## 为什么有这层测试

P0-P2 的测试证明的是"每个零件能转"。评估集回答的是另一件事 ——
**拼起来之后, 它给出的 verdict 靠不靠谱**。两者不是一回事:

- 一个永远返回 `refuted` 的系统能通过所有单元测试, 但在证实类题上全错;
- 一个永远返回 `proved` 的系统(这是模型最常犯的病)能通过所有单元测试,
  但在陷阱题上全错。

所以这个文件测的不是"工具能不能跑", 而是"**结论对不对**"。

## 分层

`tier=deterministic` 的题直接调指定工具, 0 LLM —— 可进 CI, 每次都跑。
`tier=llm` 的题走完整链路(建模或代码生成), 要真花 token —— 默认跳过,
需要 `QAGENT_RESEARCH_EVAL_LLM=1` 才跑。

## 陷阱题比准确率重要

`category=trap` 的题要求系统**认输**, `expect_verdict` 是一个允许集合
(`inconclusive` / `untested` / `partial` / `failed`), 不含 `proved` / `refuted`。

一个会硬吹的系统在普通题上也能拿到高准确率 —— 只有陷阱题能区分它。
所以这两类指标**分开报**, 不合并成一个数字。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from agent.tools.research._provenance import VERDICTS

CASES_PATH = Path(__file__).parent / "research_cases.json"

#: 陷阱题的期望是「认输」。判成 proved / refuted 都算硬吹, 一律算错。
OVERCONFIDENT = frozenset({"proved", "refuted"})

#: 跑了要花 LLM token, 默认不放进 CI。
ENV_LLM_EVAL = "QAGENT_RESEARCH_EVAL_LLM"

#: 证据的合法 kind —— 对应文档 §6.4, 与 _provenance.build_evidence 的注释一致。
#: 取值必须照抄, 不要猜: 我第一版写成 ("code","z3","oeis","enumerate"),
#: 而代码里实际是 code / solver / citation —— 测试自己先红了才暴露。
REPRODUCIBLE_KINDS = frozenset({"code", "solver", "lean", "citation"})

#: ★ claim 的**内部状态** → _provenance.VERDICTS 的 6 档判定。
#:
#: 实跑发现（2026-10-08）：闭环落库的 claim.state 空间**比** 6 档判定大，
#: 多出 `undecidable` 和 `pending` 两个:
#:
#:   undecidable —— 建模阶段就判定"这个问题不可判定"，因而根本不进验证循环。
#:                  **这是系统表现最好的一种认输**，trap_04 实跑就走的这条。
#:   pending     —— 拆出来了但还没轮到验证。
#:
#: 两者都不是"第 7、8 档判定"，而是**判定之前的状态**：P0 的 6 档是证据层契约
#: （build_evidence 会把表外值降级成 untested），而这里是命题层状态。
#:
#: 所以评估集必须先归一化再比对 —— 第一版没做这件事，导致 trap_04 明明答对了
#: 却因为返回 'undecidable'（不在期望集合里）被记成失败。
STATE_TO_VERDICT = {
    "undecidable": "inconclusive",   # 认输：判定为不可判定
    "pending": "untested",           # 认输：压根没验证
}


def _normalize(verdict: str | None) -> str | None:
    """把 claim 的内部状态归一化到 6 档判定。"""
    if not verdict:
        return None
    return STATE_TO_VERDICT.get(verdict, verdict)


#: LLM 层直接调 P2 的闭环入口, 绕过 UI。
_LLM_TIER_SKIPPED = f"设 {ENV_LLM_EVAL}=1 才跑 LLM 评估层(会真花 token)"


# ============================================================
# 1. 评估集自身的 schema —— 数据错了后面全白跑
# ============================================================
def _load() -> dict[str, Any]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


CASES: list[dict[str, Any]] = _load()["cases"]


def _expected(case: dict[str, Any]) -> set[str]:
    """expect_verdict 允许是 str 或 list[str](陷阱题的允许集合)。"""
    v = case["expect_verdict"]
    return {v} if isinstance(v, str) else set(v)


def _dispatch(case: dict[str, Any]) -> dict[str, Any]:
    """按题面指定的工具与参数调用, 走 ToolRegistry —— 顺带验接线。

    走 registry 而不是直接 import 函数, 是为了让评估集同时覆盖
    「工具注册对不对」; 直接 import 的话注册断了评估集照样绿。
    """
    import agent.tools  # noqa: F401  触发注册
    from agent.tools.registry import ToolRegistry

    spec = case["deterministic_call"]
    tool = ToolRegistry.get(spec["tool"])
    assert tool is not None, f"{case['id']}: 工具 {spec['tool']} 没注册"
    return tool.handler(**spec["kwargs"])


class TestEvalSetSchema:
    def test_loads(self):
        assert CASES, "评估集是空的"
        assert len(CASES) >= 15, f"题量太少({len(CASES)}), 统计没有意义"

    def test_ids_unique(self):
        ids = [c["id"] for c in CASES]
        assert len(ids) == len(set(ids)), "有重复 id, pytest 参数化会互相覆盖"

    def test_verdicts_in_project_enum(self):
        """期望值必须落在 _provenance.VERDICTS 里。

        写错一个 verdict(比如凭空造个 undecidable)会让这题永远红,
        而报错信息看起来像"系统没做好", 极具误导性。
        """
        valid = set(VERDICTS)
        bad = [(c["id"], v) for c in CASES for v in _expected(c) if v not in valid]
        assert not bad, f"期望 verdict 不在项目枚举内: {bad} (合法值: {VERDICTS})"

    def test_tier_and_call_shape(self):
        for c in CASES:
            cid = c["id"]
            assert c["tier"] in ("deterministic", "llm"), f"{cid}: 非法 tier"
            call = c.get("deterministic_call")
            if c["tier"] == "deterministic":
                assert call and "tool" in call and "kwargs" in call, f"{cid}: 缺调用定义"
            else:
                assert not call, f"{cid}: llm 层不该带确定性调用"
                assert c.get("question"), f"{cid}: llm 层缺 question(闭环入口参数)"

    def test_deterministic_tier_is_actually_llm_free(self):
        """★ 确定性层不得调用任何要花 LLM 的工具。

        这条是被真实 bug 逼出来的: trap_03 / trap_05 原本标成
        `tier=deterministic`, 调用却是 `research_plan` —— 那是建模,
        要真花 token。于是「确定性层 0 LLM」这个前提是假的, 评估集
        跑起来变成半个 LLM 层, 而且默认 skip 逻辑完全失效。

        分类写在数据里没人审, 这条断言就是那个审。
        """
        llm_backed = {"research_plan", "research_investigate", "research_report",
                      "research_audit"}
        for c in CASES:
            if c["tier"] != "deterministic":
                continue
            tool = c["deterministic_call"]["tool"]
            assert tool not in llm_backed, (
                f"{c['id']}: 标成 deterministic 却调了 {tool}(要花 LLM)。"
                f"要么改成 tier=llm, 要么换真 0 LLM 的工具。"
            )

    def test_traps_never_expect_overconfidence(self):
        """陷阱题的允许集合里不该出现 proved / refuted —— 否则它就不是陷阱了。"""
        for c in CASES:
            if c["category"] == "trap":
                assert not (_expected(c) & OVERCONFIDENT), f"{c['id']}: 陷阱题却允许硬吹"

    def test_covers_four_categories(self):
        got = {c["category"] for c in CASES}
        assert got == {"refute", "prove", "trap", "quantifier"}, f"分类覆盖不全: {got}"


# ============================================================
# 2. 确定性层: 14 道题, 0 LLM, 每次 CI 都跑
# ============================================================
_DET = [c for c in CASES if c["tier"] == "deterministic"]


@pytest.mark.parametrize("case", _DET, ids=[c["id"] for c in _DET])
def test_deterministic_case(case, tmp_path, monkeypatch):
    """逐题: 实际 verdict 必须落在期望集合内。"""
    monkeypatch.chdir(tmp_path)  # 产物落 tmp, 不污染真实 data/research/
    got = _dispatch(case)
    actual = got.get("verdict")
    assert actual in _expected(case), (
        f"{case['id']}: 期望 {sorted(_expected(case))}, 实得 {actual!r}\n"
        f"  题目: {case['claim']}\n"
        f"  范围: {case.get('scope', '-')}\n"
        f"  摘要: {str(got.get('summary', ''))[:300]}"
    )


def test_trap_cases_never_exposed_in_ci_tier():
    """CI 层不该出现陷阱题 —— 出现了说明有人把题错误地分到 deterministic 层。

    原先这里是一条 parametrize 逐题测『陷阱题不许硬吹』, 但 CI 层一道陷阱题都
    没有(全在 llm 层), 于是它变成 empty parameter set 静默 skip —— 看起来有
    防线, 其实一行都没跑。这类"参数为空就跳过"的写法是假绿, 删掉换成本条
    显式断言。
    """
    ci_traps = [c["id"] for c in _DET if c["category"] == "trap"]
    assert not ci_traps, (
        f"{ci_traps} 被分到了 deterministic 层。陷阱题问的是『模型会不会硬吹』, "
        f"工具层只会机械执行 —— 分到 CI 层等于把这道题废掉。"
    )


def test_ci_tier_has_no_trap_case():
    """把一个绕不过去的事实写在测试里, 而不是让空参数集静默 skip。

    所有陷阱题都是「要问 LLM 才会犯错」的问题 —— 工具层只会机械执行,
    不会硬吹。所以 CI 层一道陷阱题都没有。

    代价是: **CI 抓不到「模型面对黎曼猜想会不会硬吹」**。那道防线只在
    QAGENT_RESEARCH_EVAL_LLM=1 时才成立。这条测试存在的意义是让任何人
    看到报告时都立刻看见这个缺口, 而不是以为陷阱题已经被守住了。
    """
    if any(c["category"] == "trap" for c in _DET):
        pytest.skip("CI 层已有陷阱题, 缺口已补上")
    pytest.skip(
        f"CI 层 0 道陷阱题 —— 全部 {len([c for c in CASES if c['category'] == 'trap'])} 道"
        f"在 llm 层。防硬吹靠 test_conclusive_verdict_must_carry_evidence(结构性)"
        f"+ llm 层, CI 本身测不到『模型会不会硬吹』。"
    )


# ============================================================
# 2b. 证据强制 —— 0 LLM 就能守住的「防硬吹」结构性防线
# ============================================================
class TestEvidenceMandatory:
    """★ 这是 CI 层唯一能验的防硬吹机制, 也是最要紧的一条。

    工具层不会硬吹(它只执行), 但**结论和证据一旦脱钩, 硬吹就回来了** ——
    只要某个 conclusive verdict 能不挂证据就返回。所以:

    **任何 proved / refuted, 必须挂上可复核的 evidence。**

    这条不需要 LLM、不需要花钱, 且直接堵死「无证据下结论」这条路。
    """

    def test_code_without_verdict_marker_is_untested(self, tmp_path, monkeypatch):
        """★★ P0 最核心的一条纪律, 却是整个测试套件里唯一没人守的。

        **代码跑通 ≠ 命题验证了。** 只要没有 `print("VERDICT: ...")` 标记,
        一律判 `untested`, 哪怕 stdout 里全是 True。

        这条为什么关键: 如果缺标记时默认判 proved, 那么"我写了一段代码,
        它跑出了我期望的结果"就会直接变成一条 proved 结论 —— 而 LLM 只要
        省略最后一行 print 就能凭空造出证明。

        ★ 这条测试是 P4 反向验证逼出来的: 把 `_provenance` 里那个
        `return "untested"` 改成 `return "proved"`, 原本 100+ 个测试全绿。
        纪律写在注释里、写进文档了, 但没人钉它。
        """
        monkeypatch.chdir(tmp_path)
        import agent.tools  # noqa: F401
        from agent.tools.registry import ToolRegistry

        handler = ToolRegistry.get("research_run_code").handler

        # 一段跑得通、结果为真的代码, 但**故意不写 VERDICT 标记**
        got = handler(code="print('n^2+n+41 在 n=2..39 全是素数:', all([True]))")
        assert got.get("verdict") == "untested", (
            f"没写 VERDICT 标记却判成 {got.get('verdict')!r} —— "
            f"代码跑通不等于命题验证了, 省略一行 print 就能凭空造出结论"
        )

        # 对照: 同一个代码加上标记, verdict 才有意义
        marked = handler(code="print('VERDICT: PARTIAL')")
        assert marked.get("verdict") == "partial"

    def test_marker_is_what_decides_verdict_not_stdout(self, tmp_path, monkeypatch):
        """★ 判定只认标记, 不认 stdout。打印一堆 "PROVED" 字样不算数。

        这条挡的是另一头: LLM 学会「多打印点像结论的东西」来糊弄判定。
        """
        monkeypatch.chdir(tmp_path)
        import agent.tools  # noqa: F401
        from agent.tools.registry import ToolRegistry

        handler = ToolRegistry.get("research_run_code").handler
        got = handler(code="print('VERDICT: PROVED')\nprint('RESULT: PROVED, all true!')")
        # 最后一个标记说了算
        assert got.get("verdict") in ("proved", "untested"), got.get("verdict")

        noise = handler(code="print('PROVED!')\nprint('the claim is PROVED')")
        assert noise.get("verdict") == "untested", (
            f"stdout 里没有标记, 却判成 {noise.get('verdict')!r}"
        )

    def test_conclusive_verdict_must_carry_evidence(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        offenders = []
        for case in _DET:
            got = _dispatch(case)
            v = got.get("verdict")
            if v not in OVERCONFIDENT:
                continue
            ev = got.get("evidence") or []
            if not ev:
                offenders.append(f"{case['id']}: verdict={v} 但 evidence 为空")
                continue
            first = ev[0]
            if not first.get("reason"):
                offenders.append(f"{case['id']}: 证据缺 reason(等于没法复核)")
            if first.get("kind") not in REPRODUCIBLE_KINDS:
                offenders.append(
                    f"{case['id']}: 证据 kind={first.get('kind')!r} 不在 {sorted(REPRODUCIBLE_KINDS)}"
                )

        assert not offenders, "无证据就下结论:\n  " + "\n  ".join(offenders)

    def test_conclusive_verdict_records_env_fingerprint(self, tmp_path, monkeypatch):
        """证据必须带环境指纹。半年后没有版本号, 反过来复现不了。

        查的是**证据记录本身**的 env, 不是工具返回的顶层 env ——
        顶层 env 只是便利字段(且历史上只有 research_verify 有),
        真正被持久化进 _evidence.jsonl 的是 evidence[i]["env"]。
        """
        monkeypatch.chdir(tmp_path)
        offenders = []
        for case in _DET:
            got = _dispatch(case)
            if got.get("verdict") not in OVERCONFIDENT:
                continue
            for ev in got.get("evidence") or []:
                env = ev.get("env") or {}
                if not env:
                    offenders.append(f"{case['id']}/{ev.get('id')}: 证据缺 env 指纹")
        assert not offenders, "结论缺环境指纹:\n  " + "\n  ".join(offenders)

    def test_fingerprint_covers_decisive_dependencies(self):
        """★ 指纹必须覆盖**下过结论的**那个库。

        这条是被真实缺陷逼出来的: 指纹原本只记 sympy/numpy/scipy/networkx,
        不记 z3。而 z3 的 unsat->proved 是唯一能给全域级结论的路径 ——
        版本不同的 z3 对同一约束可能给不同结果, 不记版本等于结论不可复现。
        """
        from agent.tools.research._provenance import env_fingerprint

        fp = env_fingerprint()
        for mod in ("sympy", "numpy", "networkx", "z3"):
            assert fp.get(mod), f"指纹缺 {mod}(值不能为空)"
            assert fp[mod] != "absent", f"{mod} 没装? 结论却下出来了"

    def test_z3_evidence_carries_z3_version(self, tmp_path, monkeypatch):
        """Z3 结论的证据里必须能查到 z3 版本号。"""
        monkeypatch.chdir(tmp_path)
        import agent.tools  # noqa: F401
        from agent.tools.registry import ToolRegistry

        got = ToolRegistry.get("research_z3").handler(
            constraints="(declare-const x Int)\n(assert (> x 41))\n(assert (< x 43))",
            goal="sat",
        )
        ev = (got.get("evidence") or [{}])[0]
        env = ev.get("env") or {}
        assert env.get("z3"), f"Z3 证据缺 z3 版本: env={env}"




# ============================================================
# 3. 准确率报告 —— 陷阱与常规分开统计, 不合并成一个数字
# ============================================================
class TestAccuracyReport:
    def test_deterministic_accuracy_split_by_category(self, tmp_path, monkeypatch):
        """跑全部确定性题, 按类别出准确率。失败要能指出是哪题。"""
        monkeypatch.chdir(tmp_path)
        stats: dict[str, list[bool]] = {}
        failures: list[str] = []

        for case in _DET:
            try:
                actual = _dispatch(case).get("verdict")
            except Exception as exc:  # noqa: BLE001  工具抛异常也算这题没做对
                failures.append(f"{case['id']}: 抛异常 {exc!r}")
                stats.setdefault(case["category"], []).append(False)
                continue
            ok = actual in _expected(case)
            stats.setdefault(case["category"], []).append(ok)
            if not ok:
                failures.append(
                    f"{case['id']}: 期望 {sorted(_expected(case))}, 实得 {actual!r}"
                )

        report = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(stats.items())}
        print(f"\n确定性层准确率 {report}")
        assert not failures, "以下题目判定错误:\n  " + "\n  ".join(failures)

    def test_quantifier_contrast_disagree(self):
        """★ 量词对照: 同一命题的『存在』版与『所有』版期望必须相反。

        这条不跑任何东西, 纯静态断言, 但它是 §18.3 量词误读缺陷的哨兵 ——
        如果哪天有人把两道的期望改成一样, 这条会立刻红。
        """
        quant = {c["id"]: _expected(c) for c in CASES if c["category"] == "quantifier"}
        assert len(quant) >= 3, "量词对照题不足"

        pair = ("quant_01_exists_tight", "quant_02_all_tight")
        if all(p in quant for p in pair):
            a, b = quant[pair[0]], quant[pair[1]]
            assert a != b, f"{pair} 期望相同, 量词对照失效"
            assert a & OVERCONFIDENT, f"{pair[0]} 应有明确结论"
            assert b & OVERCONFIDENT, f"{pair[1]} 应有明确结论"
            assert a != b, "『存在』与『所有』的期望必须相反"


# ============================================================
# 4. LLM 层: 默认跳过, 显式开才跑
# ============================================================
class TestLLMTier:
    def test_llm_cases_skipped_by_default(self):
        if os.environ.get(ENV_LLM_EVAL) == "1":
            pytest.skip("已显式开启 LLM 评估层")
        llm_cases = [c for c in CASES if c["tier"] == "llm"]
        assert llm_cases, "评估集里没有 llm 层题目, 量词缺陷就没人盯了"
        pytest.skip(_LLM_TIER_SKIPPED)

    def test_llm_eval_baseline(self):
        """★ LLM 层完整基线 —— 一次闭环跑完, 产出三个指标。

        **故意不 chdir 到 tmp_path。** `_call_llm` 要从项目根读 provider 配置,
        chdir 到临时目录后配置读不到, 每次闭环 0.2 秒就"建模失败"返回空 ——
        表现是全部题 verdict=None, 看着像模型全挂, 其实是测试自己把路断了。
        产物落在仓库的 data/research/ 下, 那本来就在 .gitignore 里。

        三个指标必须**分开看**, 因为它们量的不是一回事:
          1. 判定准确率    —— verdict 落在期望集合内
          2. 陷阱题硬吹率  —— 陷阱题里出现**任何** conclusive 结论就算翻车
          3. 结论自洽率    —— 同一任务内不同命题的 verdict 互相矛盾

        ★ 断言口径是**"不差于 LLM_BASELINE"**，不是"全部正确"。理由见 LLM_BASELINE
        的注释：现状离达标还很远，写成裸 assert 就是永久红灯，写成"全部正确"就是
        自欺欺人。退步红、达标绿、变好提示更新表。
        """
        if os.environ.get(ENV_LLM_EVAL) != "1":
            pytest.skip(_LLM_TIER_SKIPPED)

        llm_cases = [c for c in CASES if c["tier"] == "llm"]
        regressed, blown_now, contradictory_now, improvable = [], [], [], []
        since = datetime.now().isoformat(timespec="seconds")

        for case in llm_cases:
            cid = case["id"]
            verdicts = _run_closed_loop(case["question"], since=since)
            actual = _fold_verdicts(verdicts)
            baseline = LLM_BASELINE.get(cid)
            expected = _expected(case)

            # ── 指标 1: 判定 ──
            if actual in expected:
                if baseline is not None and actual != baseline:
                    improvable.append(f"{cid}: 已达标({actual!r})，基线记的是 {baseline!r}，请更新表")
            elif baseline is not None and actual == baseline:
                pass                      # 和基线一样差 —— 不算退步
            else:
                regressed.append(
                    f"{cid}: 期望 {sorted(expected)}，基线 {baseline!r}，实得 {actual!r}"
                    f"（全部: {verdicts}）"
                )

            # ── 指标 2: 陷阱题硬吹 ──
            if case["category"] == "trap" and any(v in OVERCONFIDENT for v in verdicts):
                blown_now.append(cid)

            # ── 指标 3: 结论自洽 ──
            if _contradiction(verdicts):
                contradictory_now.append(cid)

        blown_base = [c for c in LLM_BASELINE
                      if c.startswith("trap_") and LLM_BASELINE[c] in OVERCONFIDENT]
        contra_base = sorted(LLM_BASELINE_CONTRADICTORY)

        print("\n=== LLM 层评估基线 ===")
        print(f"  判定退步 {len(regressed)}/{len(llm_cases)}"
              f" | 陷阱硬吹 {len(blown_now)}/{len(blown_base)}"
              f" | 结论矛盾 {len(contradictory_now)}/{len(contra_base)}")
        print("  (基线: 判定错误 1/8 | 陷阱硬吹 1/5 | 结论矛盾 0/8 —— 详见文档 §21.6)")

        assert not regressed, "判定**退步**了（基线见 LLM_BASELINE）:\n  " + "\n  ".join(regressed)
        assert set(blown_now) <= set(blown_base), (
            "新的陷阱题被硬吹了:\n  "
            + "\n  ".join(sorted(set(blown_now) - set(blown_base)))
        )
        assert set(contradictory_now) <= set(contra_base), (
            "新的题目出现结论矛盾:\n  "
            + "\n  ".join(sorted(set(contradictory_now) - set(contra_base)))
        )
        if improvable:
            print("\n已达标，基线表可以更新了:\n  " + "\n  ".join(improvable))

        print(
            "\n⚠ 当前离 P5 接入门槛还很远"
            f"（陷阱硬吹 {len(blown_now)}/{len(blown_base)}）。"
            f"exposed=False 必须保持，见文档 §19.3。"
        )


#: ★ LLM 层当前实测基线
#:
#: **2026-10-08 P4.5 八道防线后**（判定错误 1/8、陷阱硬吹 1/5、结论矛盾 0/8）
#:
#: 三轮演进：
#:   P4        判定错误 5/8 · 陷阱硬吹 4/5 · 结论矛盾 3/8
#:   四道防线   判定错误 2/8 · 陷阱硬吹 2/5 · 结论矛盾 0/8
#:   八道防线   判定错误 1/8 · 陷阱硬吹 1/5 · 结论矛盾 0/8
#:
#: 关键单点修复：`quant_02`「对所有正整数 n, n²<2n」(假命题) proved → refuted。
#:
#: 这张表**故意记录着还不好看的数字**: goldbach 仍有一轮判 refuted。
#: 直接把这些题写成 assert 通过是自欺欺人（把现状粉饰成达标）；
#: 写成裸 assert 又是永久红灯，没法进 CI。所以取中间那条路：
#:
#:   **测试断言"不差于这张表"** —— 退步立即红，达标或变好则绿并提示更新本表。
#:
#: 为什么值得这么麻烦：这张表是"P5 接入前必须跨过的门槛"的量化定义。
#: 门槛从 1/8 降到 0/8 应该是**一次有意识的决定**，而不是某次跑歪了 unnoticed。
LLM_BASELINE = {
    "trap_01_goldbach": "inconclusive",          # ⚠ 仍摇摆(出现过 refuted), 见文档 §21.5
    "trap_02_riemann": "inconclusive",
    "trap_03_aesthetic": "inconclusive",
    "trap_04_factorial_prime_beyond": "inconclusive",   # 修前是 proved(超算力范围被缩小)
    "trap_05_unspecified_minimum": "inconclusive",      # 修前是 proved(空洞真命题)
    "quant_01_exists_tight": "proved",
    "quant_02_all_tight": "refuted",              # 修前是 proved
    "quant_03_exists_polynomial_composite": "proved",
}

#: 实测出现"同一任务内 proved 与 refuted 并存"的题。
#: 修复后是**空集** —— 之前 4 道里有 3 道中招, 那是旧证据复活造成的假象(见文档 §20.9)。
LLM_BASELINE_CONTRADICTORY: frozenset[str] = frozenset()


def _run_closed_loop(question: str, since: str = "") -> list[str]:
    """跑一遍完整闭环, 返回该任务下**每条**命题的判定。

    ★ research_investigate 是**流式**的(Iterator[str]), 不返回结构化结果 ——
    结论落在 research.db 的 claim 表里。所以只能: 收干流 → 按 task_hash
    读回 claim 的 verdict。第一版我写成 `got.get("verdict")`, 那是在对
    字符串调 .get, 永远炸。

    Args:
        since: ISO 时间戳, 只取 updated_at 不早于它的 claim。

    ★ `since` 是必须的, 不是优化。task_hash 按 question 内容生成, 所以**重跑同一个
    问题会落进同一个工作区和同一个 task_hash**。而 claim 表的主键是内容哈希 ——
    上次跑拆出的命题如果和这次不同, 旧记录会**残留**在表里。

    实测踩过: 审美题第一次跑留了 1 条 undecidable(13:49), 修复后重跑新增 4 条(14:14),
    读出来 5 条混在一起 —— 于是"同一个任务里既有 proved 又有 undecidable"被误判成
    **结论矛盾**, 基线数据直接不可信。所以按 updated_at 切一刀。
    """
    from agent.infra.research_workspace import hash_question
    from agent.tools.research.research_investigate import research_investigate

    for _chunk in research_investigate(question=question):
        pass

    return _all_claim_verdicts(hash_question(question), since=since)


def _all_claim_verdicts(task_hash: str, since: str = "") -> list[str]:
    """取该任务下每条命题的判定, 已归一化到 6 档。"""
    from agent.tools.research._db import list_claims

    rows = list_claims(task_hash=task_hash)
    if since:
        rows = [r for r in rows if (r.get("updated_at") or "") >= since]
    return [_normalize(r.get("verdict")) for r in rows]


def _fold_verdicts(verdicts: list[str]) -> str | None:
    """把一个任务的多条判定折成单个代表 verdict。

    规则: 有 conclusive 就取第一个 conclusive(那才是这个任务的"结论");
    全都不 conclusive 则取最后一个非空判定。

    为什么要折 —— 评估集每题只有一个期望 verdict。"没给出任何结论"和
    "给出了错结论"是两回事, 前者应该算认输而不是算错。
    """
    vs = [v for v in verdicts if v]
    if not vs:
        return None
    for v in vs:
        if v in OVERCONFIDENT:
            return v
    return vs[-1]


def _contradiction(verdicts: list[str]) -> str | None:
    """检测同一任务内 proved 与 refuted 并存。

    同一个问题拆出的多条命题**可以**有不同判定(真反例: 一条"存在 n 使 P"
    和一条"所有 n 满足 Q" 本来就该一真一假)。但只要模型拆出来的是
    **同一条命题的不同范围版本**, 一条 proved 一条 refuted 就是错的。

    这里做不了精确的语义等价判断, 所以只在"同一任务内两种 conclusive
    并存"时报警 —— 宁可多报也不漏报: 这个信号出现得很少, 值得人工看一眼。
    """
    conclusive = [v for v in verdicts if v in OVERCONFIDENT]
    if "proved" in conclusive and "refuted" in conclusive:
        return f"同一任务里既有 proved 又有 refuted: {conclusive}"
    return None


