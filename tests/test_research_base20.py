"""small-math-agent-benchmark 20 题 —— 确定性层 (0 LLM)。

这一层测的是「沙箱 + sympy/networkx 能不能算对」, 与 LLM 水平无关:
验证代码是我方写的标准答案, 不是 LLM 生成的。所以 0 token, 进 CI。

20 题全部 gold=proved (都是「验证某恒等式/性质」或「求出某确定值」),
任何非 PROVED 都算没答对。

对照: 同一批题让 LLM 自己写验证器时是 14/20, 见 run_base20_llm.py。
两个数字的差值 = 「工具能力」与「LLM 翻译能力」之间的差。
"""
from __future__ import annotations

import importlib.util
import io
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# tests/tools 不是包(没有 __init__.py), 且不能在 import 期 chdir —— LLM 层
# 要从项目根读配置。所以按文件路径加载, 不靠 sys.path。
_VERIFIER_PATH = os.path.join(ROOT, "tests", "tools", "base20_verifiers.py")
_spec = importlib.util.spec_from_file_location("base20_verifiers", _VERIFIER_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GOLD = _mod.GOLD
VERIFIER_CODE = _mod.VERIFIER_CODE

CASES_JSON = os.path.join(ROOT, "tests", "base20_cases.json")
with io.open(CASES_JSON, encoding="utf-8") as _f:
    _CASES = {c["id"]: c for c in json.load(_f)["cases"]}


def _run_via_sandbox(code: str, tmp_path, monkeypatch):
    """走真实的 research_run_code 沙箱, 不直接 exec —— 这一层要测的就是沙箱。"""
    monkeypatch.chdir(tmp_path)
    import agent.tools  # noqa: F401
    from agent.tools.registry import ToolRegistry

    return ToolRegistry.get("research_run_code").handler(code=code)


class TestBase20Deterministic:
    """20 题逐题跑, 失败要能指出是哪题。"""

    @pytest.mark.parametrize("cid", sorted(VERIFIER_CODE))
    def test_case_proved(self, cid, tmp_path, monkeypatch):
        code = VERIFIER_CODE[cid]
        r = _run_via_sandbox(code, tmp_path, monkeypatch)
        err = r.get("error")
        assert not err, f"{cid}: 沙箱报错 {err}"

        stdout = r.get("stdout", "")
        assert "VERDICT:" in stdout, (
            f"{cid}: 代码没打 VERDICT 标记 —— 没标记就是 untested\n{stdout[-500:]}"
        )
        got = stdout.split("VERDICT:")[-1].strip().split()[0].lower()
        assert got == GOLD, (
            f"{cid}: gold={GOLD} 实得={got}\n"
            f"  domain: {_CASES[cid]['domain']}\n"
            f"  stdout: {stdout[-400:]}"
        )

    def test_all_twenty_cases_present(self):
        """评估集必须真的 20 题 —— 少一题会让准确率虚高。"""
        assert len(_CASES) == 20, f"题数不对: {len(_CASES)}"
        assert len(VERIFIER_CODE) == 20, f"验证代码数不对: {len(VERIFIER_CODE)}"

    def test_cases_json_is_not_empty_when_checked_out(self):
        """★ 评估集必须能被提交进 git。

        真发生过: .gitignore 的 `data/` 没有前导斜杠, 匹配任意层级,
        评估集放在 tests/data/ 下被静默忽略 —— 别人 clone 下来题全空,
        而 git 不报错、测试也不报错(参数化后是 0 个 case)。
        """
        assert len(_CASES) == 20, (
            "评估集读出来是空的。检查它是否被 .gitignore 挡下 "
            "(注意 `data/` 无前导斜杠会匹配任意层级)"
        )
        for cid, c in _CASES.items():
            assert c.get("probes"), f"{cid} 的 probes 为空, 题目数据不完整"

    def test_verifier_code_all_emit_verdict(self):
        """每段验证代码都必须会打 VERDICT —— 否则跑完拿不到判定。

        这是「验证器自己坏掉」的第一道闸门。真实发生过: Ramsey 那版把
        顶点范围硬编码成 6, K5 分支恒判有单色三角形, 整题误报 REFUTED,
        而代码本身跑得很顺, 不看输出根本发现不了。
        """
        for cid, code in VERIFIER_CODE.items():
            assert "VERDICT:" in code, f"{cid} 没有 VERDICT 标记"

    def test_ramsey_verifier_takes_n_from_parameter(self):
        """Ramsey 验证器的顶点范围必须来自参数, 不能硬编码。

        这条针对一个已经发生过的 bug: has_mono_triangle() 里写死
        combinations(range(6), 3), K5 分支因此也去取 6 顶点三元组,
        cnt5 恒为 0, 整题被判 REFUTED —— 而实际 R(3,3)=6 是成立的。

        ★ 用 AST 查而不是文本匹配 ★ 第一次写这条时用 `assert "range(6"
          not in code`, 结果被**注释里**那句「早期版本硬编码 range(6)」
          判红 —— 断言源码文本会被注释里的同名字符串误伤。
        """
        import ast

        tree = ast.parse(VERIFIER_CODE["graph-014-ramsey-3-3"])
        fn = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "has_mono_triangle"),
            None,
        )
        assert fn is not None, "找不到 has_mono_triangle 定义"
        assert fn.args.args, "has_mono_triangle 没有参数"
        arg_names = [a.arg for a in fn.args.args]
        assert "n" in arg_names, f"没有 n 参数: {arg_names}"

        # 函数体里的 range(...) 实参必须是变量, 不能是字面量 6
        range_arg_names: set[str] = set()
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "range" and node.args):
                first = node.args[0]
                if isinstance(first, ast.Name):
                    range_arg_names.add(first.id)
                else:
                    assert not (isinstance(first, ast.Constant) and first.value == 6), (
                        "又出现硬编码的 range(6) —— 顶点范围必须取自参数 n"
                    )
        assert "n" in range_arg_names, (
            f"range() 的参数里没用上 n, 实际用了: {sorted(range_arg_names)}"
        )

    def test_llm_failure_modes_are_recorded(self):
        """第一轮 LLM 跑挂的题必须在 JSON 里记下失败模式。

        记下来的意义: 下次看到同样的失败, 能立刻判断是「新问题」还是
        「已知的老问题」——没记下来就每次都重新查一遍。
        """
        known = [cid for cid, c in _CASES.items() if c.get("llm_failed_as")]
        assert len(known) == 7, f"记录的失败题数变了: {len(known)}"
        for cid in known:
            assert _CASES[cid].get("llm_failure_mode"), f"{cid} 记了失败但没写原因"

    def test_known_unresolved_is_explicitly_marked(self):
        """已知未解的题必须显式标记, 且写清为什么加判据也修不好。

        这条防止一件具体的事: num-020-biszection-root 连续三轮都是
        REFUTED, 期间加过两轮 prompt 判据(范围判据 + REFUTED 语义),
        都没修好。如果不显式记下来, 下一轮看到它失败又会想去「再加一条
        判据」—— 而根因是 LLM 把判定分支写死成字面量, 措辞管不住。
        """
        unresolved = [cid for cid, c in _CASES.items() if c.get("known_unresolved")]
        assert unresolved == ["num-020-bisection-root"], (
            f"已知未解清单变了: {unresolved} —— "
            f"若已修好, 请把 known_unresolved 去掉并更新 llm_failure_mode"
        )
        mode = _CASES["num-020-bisection-root"]["llm_failure_mode"]
        assert "判据" in mode and "无效" in mode, (
            "已知未解必须写清「为什么加判据也修不好」, 否则下轮会重复加防线"
        )