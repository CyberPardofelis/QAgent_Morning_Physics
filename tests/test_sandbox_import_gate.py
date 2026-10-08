"""沙箱 AST 门的 import 校验 —— 根模块 + 符号名。

## 为什么加符号检查

根模块白名单有个盲区: `from sympy import maximize` 的根模块 `sympy`
是白名单里的, 根名校验**放行** —— 但 SymPy 根本没有 maximize 这个名字。

真实后果(2026-10-08 实测): 线性规划题 LLM 写的顶点枚举思路**完全正确**,
就因为多 import 一个不存在的名字, 整份验证器 ImportError 崩掉, verdict 变
failed。之前只能靠运行时崩溃 + 重试侥幸救回来。

这类错误有个别的错误没有的性质: **机器查一下就知道**。不需要赌 LLM
听不听话, 不需要加 prompt 措辞, 更不需要正则去猜 LLM 会怎么写。

## 与"闸门失灵倒向放行"一致

解析失败(模块装不上/懒加载/平台差异)时**放行**, 交给运行时 ImportError。
误杀一段本来能跑的代码, 代价大于让它崩一次重试 —— 这是取舍, 不是疏忽。
"""
from __future__ import annotations

import pytest

from agent.tools.research._sandbox import SandboxRejected, check_code


class TestImportModuleRoots:
    """根模块白名单 —— 这部分行为没变, 钉住它。"""

    @pytest.mark.parametrize("code", [
        "import os\n",
        "import subprocess\n",
        "from socket import socket\n",
        "import ctypes\n",
        "from . import x\n",
    ])
    def test_dangerous_root_rejected(self, code):
        with pytest.raises(SandboxRejected):
            check_code(code)

    @pytest.mark.parametrize("code", [
        "import sympy\n",
        "from sympy import symbols\n",
        "import math\n",
        "from math import gcd\n",
    ])
    def test_allowed_root_passes(self, code):
        check_code(code)


class TestImportSymbolExistence:
    """★ 新增: 被导入的符号必须真的存在 ★"""

    def test_sympy_maximize_is_caught(self):
        """实战案例: SymPy 没有 maximize, 整份正确的验证器被它弄崩。"""
        code = "from sympy import symbols, maximize\nprint('x')\n"
        with pytest.raises(SandboxRejected) as e:
            check_code(code)
        assert "maximize" in str(e.value)

    def test_minimize_also_caught(self):
        with pytest.raises(SandboxRejected):
            check_code("from sympy import minimize\n")

    def test_nonexistent_symbol_caught(self):
        with pytest.raises(SandboxRejected) as e:
            check_code("from sympy import totally_not_a_real_name_xyz\n")
        assert "totally_not_a_real_name_xyz" in str(e.value)

    def test_nonexistent_symbol_in_submodule(self):
        """多层模块也要查 —— `sympy.matrices` 同样不认这个名字。"""
        with pytest.raises(SandboxRejected):
            check_code("from sympy.matrices import not_a_matrix_thing\n")

    def test_numpy_nonexistent_caught(self):
        with pytest.raises(SandboxRejected):
            check_code("from numpy import not_a_numpy_function_abc\n")

    def test_error_message_suggests_alternative(self):
        """报错要给出替代建议 —— 只说"不存在"LLM 未必知道该用什么。"""
        with pytest.raises(SandboxRejected) as e:
            check_code("from sympy import maximize\n")
        assert "sym" in str(e.value) or "solve" in str(e.value)


class TestImportSymbolMustNotOverReject:
    """★ 配套: 合法写法一个都不能误杀 ★

    误杀的代价(好代码被拒 → LLM 困惑 → 重试 → 更贵)大于漏放的代价
    (崩一次 → 重试)。只查「该拦的」而不管「不该拦的」, 优化就过头了。
    """

    @pytest.mark.parametrize("code", [
        "from sympy import symbols, isprime\n",
        "from sympy import symbols as sym\n",           # 带 as
        "from sympy import S, oo, I, pi\n",            # 大写 / 特殊
        "from sympy import *\n",                        # 星号交给运行时
        "from sympy.matrices import Matrix\n",          # 子模块
        "from sympy.abc import symbols\n",
        "from math import gcd, isqrt, factorial\n",
        "from itertools import combinations, product\n",
        "from fractions import Fraction\n",
        "from z3 import Int, Solver, sat\n",
        "from scipy.optimize import linprog\n",         # 真的存在
        "from networkx import petersen_graph\n",
        "from numpy import linspace, array\n",
        "from collections import defaultdict, Counter\n",
        "import sympy as sp\nfrom sympy import Rational\n",
    ])
    def test_legal_imports_pass(self, code):
        """这些全是评估集里真实用到的写法, 一个都不许拒。"""
        check_code(code)


class TestSymbolCheckFailsOpen:
    """解析不了就放行 —— 别让检查本身把工具弄瘫。"""

    def test_unresolvable_module_passes(self):
        """模块名合法但解析不出来时, 交给运行时, 不在这里误杀。"""
        # sympy 存在, 但用一个不存在的深层路径
        check_code("from sympy.zzz_not_a_real_submodule import foo\n")

    def test_wildcard_import_passes(self):
        check_code("from sympy import *\n")
