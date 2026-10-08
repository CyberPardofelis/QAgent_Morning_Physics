"""
test_research_sandbox — P7 科研闭环 S0 沙箱测试 (2026-10-08 加)

覆盖 4 组行为:
  1. AST 静态门: 越界 import / 危险名字 / 语法错误 一律拒 (不执行)
  2. 执行: 正常跑、异常原文回传、超时杀整棵进程树
  3. 子进程环境隔离: 不把 API key 传进沙箱
  4. 输出截断: 保留头尾 (尾部是 traceback 的关键)

**为什么第 2 组里的超时用例最重要**:
P0 验收项明确要求"Windows 上必须能实际杀掉跑飞的子进程, 光有超时参数不算"。
只 kill 父进程的话, 孙进程会一直持有 stdout 管道, communicate() 永久阻塞。
所以这里断言的是**真的返回了**, 而且耗时在超时附近。

真实隔离 = S1 (Docker --network=none, 见 docs/RESEARCH_CLOSED_LOOP_PLAN.md §5.4)。
S0 不是安全边界, 它防的是 LLM 生成 `import os` 这类手滑。

跑法:
    .\\venv\\Scripts\\python.exe -m pytest tests\\test_research_sandbox.py -v
"""
import os
import time

import pytest

from agent.tools.research._sandbox import (
    MAX_OUTPUT_CHARS,
    SandboxRejected,
    _child_env,
    _truncate,
    check_code,
    run_code,
)


# ============================================================
# 1. AST 静态门
# ============================================================
class TestCheckCode:
    """静态门 —— 过了才执行, 拒绝时抛 SandboxRejected"""

    # ---------- 白名单内: 必须放行 ----------
    @pytest.mark.parametrize("code", [
        "import math\nprint(math.pi)",
        "import sympy\nfrom sympy import symbols, factor\nprint(factor(symbols('x')**2-1))",
        "import numpy as np\nprint(np.arange(3).sum())",
        "import networkx as nx\nprint(nx.complete_graph(3).number_of_edges())",
        "from itertools import combinations\nprint(len(list(combinations(range(5), 2))))",
        "print(1 + 1)",                       # 根本不 import
    ])
    def test_allowed_code_passes(self, code):
        check_code(code)  # 不抛就算过

    # ---------- 黑名单: 必须拒绝 ----------
    @pytest.mark.parametrize("code,expect", [
        ("import os",                          "os"),
        ("import subprocess",                   "subprocess"),
        ("import socket",                       "socket"),
        ("import ctypes",                       "ctypes"),
        ("import shutil",                       "shutil"),
        ("from os import path",                 "os"),
        ("import sympy, os",                    "os"),          # 白名单里夹带私货
        ("x = eval('1+1')",                     "eval"),
        ("exec('pass')",                        "exec"),
        ("f = open('/etc/passwd')",             "open"),
        ("mod = __import__('os')",              "__import__"),
        ("g = globals()",                       "globals"),
        ("import math\n__builtins__['eval']('1')", "__builtins__"),
        ("from . import sibling",               "relative"),    # 相对导入在沙箱无意义
    ])
    def test_banned_code_rejected(self, code, expect):
        """拒绝信息里要带上具体是哪个东西 —— LLM 要靠这个改代码"""
        with pytest.raises(SandboxRejected) as exc:
            check_code(code)
        assert expect in str(exc.value), f"拒绝原因里应出现 {expect}, 实际: {exc.value}"

    def test_syntax_error_rejected(self):
        """语法错 → 在解析阶段就拒, 不要跑出个莫名 NameError"""
        with pytest.raises(SandboxRejected) as exc:
            check_code("def (:")
        assert "语法错误" in str(exc.value)

    def test_error_message_has_line_number(self):
        """报错要带行号, 否则 LLM 不知道改哪"""
        code = "import math\nimport os\n"
        with pytest.raises(SandboxRejected) as exc:
            check_code(code)
        assert "line 2" in str(exc.value)


# ============================================================
# 2. 执行
# ============================================================
class TestRunCode:
    """真实起子进程跑"""

    def test_simple_run(self, tmp_path):
        r = run_code("print(1 + 1)", tmp_path / "t1")
        assert r.ok is True
        assert r.stdout.strip() == "2"
        assert r.error_type == ""
        assert r.elapsed_ms >= 0

    def test_scientific_libs_available(self, tmp_path):
        """白名单里的科学库真能 import 并算 —— 不然这个沙箱没有意义"""
        r = run_code(
            "import sympy\n"
            "x = sympy.symbols('x')\n"
            "print(sympy.factor(x**2 - 1))\n",
            tmp_path / "t2", timeout=30,
        )
        assert r.ok is True, r.stderr
        assert "(x - 1)*(x + 1)" in r.stdout

    def test_exception_message_returned(self, tmp_path):
        """
        异常原文必须回传 —— 这是"喂回 LLM 修"的燃料, 吞掉就等于闭环断了。
        """
        r = run_code('print("before")\nraise ValueError("boom-42")', tmp_path / "t3")
        assert r.ok is False
        assert r.error_type == "ValueError"
        assert "boom-42" in r.stderr
        assert "Traceback" in r.stderr
        assert "before" in r.stdout      # 崩之前的 print 也要保住

    def test_timeout_kills_process(self, tmp_path):
        """
        ★ P0 核心验收项: 死循环必须在超时附近真的被杀掉并返回。
        """
        started = time.time()
        r = run_code("while True:\n    pass", tmp_path / "t4", timeout=2)
        elapsed = time.time() - started

        assert r.timed_out is True
        assert r.ok is False
        assert "超时" in r.stderr
        # 返回了 = 没卡死。若只 kill 父进程, communicate() 会一直等管道 → 这里会挂到测试超时
        assert elapsed < 15, f"超时后 {elapsed:.1f}s 才返回, 疑似没杀干净进程树"
        # 且耗时贴着超时值, 说明是真等到超时才动手, 不是秒退
        assert elapsed >= 1.5, f"只花了 {elapsed:.1f}s 就返回, 可能没真等到超时"

    def test_rejected_code_not_executed(self, tmp_path):
        """被拒的代码连子进程都不该起 —— 用"目录里没有 main.py"证明没执行"""
        scratch = tmp_path / "t5"
        try:
            run_code("import os\nprint(os.getcwd())", scratch)
            pytest.fail("应该抛 SandboxRejected")
        except SandboxRejected:
            pass
        assert not (scratch / "main.py").exists(), "AST 门拒绝的代码不该被写盘执行"

    def test_check_false_returns_rejected_result(self, tmp_path):
        """check=False 时不抛异常, 把拒绝信息放进 result —— 给工具层用"""
        r = run_code("import os", tmp_path / "t6", check=False)
        assert r.rejected is True
        assert "os" in r.reject_reason

    def test_output_truncated_flagged(self, tmp_path):
        """超长输出要打 truncated 标记, 否则 LLM 不知道内容被掐过"""
        r = run_code("print('x' * 20000)", tmp_path / "t7")
        assert r.ok is True
        assert r.truncated is True
        assert len(r.stdout) < 20000

    def test_relative_scratch_path(self, tmp_path, monkeypatch):
        """
        ★ 回归测试: scratch 传**相对路径**必须能跑。

        2026-10-08 实测踩坑: Popen(cwd=相对路径) 会先 chdir, 再用命令行里的相对
        路径找 _runner.py, 于是路径双重拼接 → 子进程秒退 → SandboxBootstrapError。
        当时单测全绿是因为 tmp_path 给的是绝对路径, 一接 workspace
        (data/research/working/...) 的相对路径就 100% 挂。
        """
        monkeypatch.chdir(tmp_path)
        r = run_code("print('relative ok')", "rel_scratch_dir")
        assert r.ok is True, f"相对路径 scratch 跑挂: {r.stderr}"
        assert "relative ok" in r.stdout
        assert r.error_type != "SandboxBootstrapError"

    def test_nested_relative_path(self, tmp_path, monkeypatch):
        """多级相对目录(workspace 真实形态: a/b/c/)同样要能跑"""
        monkeypatch.chdir(tmp_path)
        r = run_code("print(42)", os.path.join("data", "research", "working", "abc123", "scratch"))
        assert r.ok is True, f"嵌套相对路径跑挂: {r.stderr}"
        assert "42" in r.stdout


# ============================================================
# 3. 子进程环境隔离
# ============================================================
class TestChildEnv:
    """
    沙箱子进程不能拿到父进程的 API key。
    否则 LLM 生成的代码可以在沙箱里直接调 LLM API 烧钱, 且完全绕过 CostGuard。
    """

    def test_no_api_keys_leak(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-should-not-leak")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-should-not-leak")
        env = _child_env()
        upper = {k.upper() for k in env}
        assert "DEEPSEEK_API_KEY" not in upper
        assert "OPENAI_API_KEY" not in upper

    def test_keeps_python_essentials(self):
        """跑 python 必需的变量必须在, 否则子进程起不来"""
        env = _child_env()
        upper = {k.upper() for k in env}
        for must in ("PATH", "SYSTEMROOT", "PYTHONIOENCODING"):
            assert must in upper, f"子进程环境缺 {must}"
        assert env["PYTHONIOENCODING"] == "utf-8"


# ============================================================
# 4. 截断策略
# ============================================================
class TestTruncate:
    def test_short_untouched(self):
        assert _truncate("abc") == "abc"

    def test_keeps_head_and_tail(self):
        """
        头尾都要留 —— 头部有 LLM 自己 print 的上下文, 尾部是 traceback 的关键行。
        只留一半等于把排错信息扔了。
        """
        text = "HEAD_MARKER" + ("y" * 20000) + "TAIL_MARKER"
        out = _truncate(text)
        assert out.startswith("HEAD_MARKER")
        assert out.endswith("TAIL_MARKER")
        assert "被截断" in out
        assert len(out) < MAX_OUTPUT_CHARS * 2
