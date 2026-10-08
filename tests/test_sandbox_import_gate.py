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


class TestModuleAttributeExistence:
    """★ 属性访问也要查: `sympy.isqrt(...)` 这种写法静态门以前完全看不见 ★

    ## 为什么要补这道门

    实战(2026-10-08 V2 hard-001 Pell-61): 同一份代码里
    `AttributeError: module 'sympy' has no attribute 'isqrt'` **连犯三次**,
    5 轮重试烧掉 33k token 最后一轮都没跑成。

    根因不是模型笨 —— 是**门的位置错了**: `_check_symbol_exists` 只管
    `from X import Y`, 而 `sympy.isqrt(...)` 是**属性访问**, 静态门压根没看,
    于是它一路畅通跑到运行时才崩。崩一次的代价本该是"下一轮改对",
    但它下一轮又写了一遍同样的错 —— 因为反馈环路上没有任何东西提醒它。
    """

    def test_sympy_isqrt_is_caught_with_cross_module_hint(self):
        """实战案例: SymPy 没有 isqrt, 正确写法是 math.isqrt。"""
        with pytest.raises(SandboxRejected) as ei:
            check_code("import sympy\nprint(sympy.isqrt(17))\n")
        msg = str(ei.value)
        assert "isqrt" in msg
        # ★ 必须给出跨模块的正确写法 ★ 只说"不存在"的话, 它下一轮还会写错
        assert "math.isqrt" in msg

    def test_sympy_convergents_attribute_caught(self):
        """sympy 没有 convergents(上面那对 _NAME_HINTS 已收录)。"""
        with pytest.raises(SandboxRejected) as ei:
            check_code("import sympy\nprint(sympy.convergents(0.5))\n")
        assert "continued_fraction_periodic" in str(ei.value)

    def test_nested_module_attribute_caught(self):
        """`sympy.ntheory.xxx` 也要查 —— value 是 Attribute 不是 Name。"""
        with pytest.raises(SandboxRejected):
            check_code("import sympy\nprint(sympy.ntheory.is_not_a_thing(3))\n")

    def test_aliased_module_attribute_caught(self):
        """`import sympy as sp` 之后写 `sp.isqrt` 一样要拦。"""
        with pytest.raises(SandboxRejected) as ei:
            check_code("import sympy as sp\nprint(sp.isqrt(17))\n")
        assert "math.isqrt" in str(ei.value)


class TestModuleAttributeMustNotOverReject:
    """★ 配套: 局部变量的同名方法一个都不能误杀 ★

    这是这道门最大的误杀面 —— `result.isqrt(...)`、`data.comb(...)`、
    `model.eval(...)` 全是普通方法调用。如果对任意 `名字.属性` 都去
    importlib 查一遍, 这些会被当成"模块属性"全杀掉。
    """

    @pytest.mark.parametrize("code", [
        "import sympy\nresult = 1\nprint(result.isqrt)\n",
        "import sympy\nclass R:\n    def isqrt(self):\n        return 1\n"
        "print(R().isqrt())\n",
        "import math\nprint(math.isqrt(17))\n",
        "import sympy\nprint(sympy.sqrt(2))\n",
        "import sympy\nprint(sympy.ntheory.primetest.isprime(7))\n",
        "import sympy\nx = sympy.Symbol('x')\nprint(x.is_real)\n",
        # from-import 绑定的名字**不是模块**, 不能当模块查
        "from sympy import sqrt\nprint(sqrt.isqrt)\n",
        "import itertools\nprint(itertools.combinations)\n",
        "import sympy\nprint(sympy.__version__)\n",
    ])
    def test_legal_attribute_access_passes(self, code):
        check_code(code)

    def test_math_isqrt_hint_is_not_itself_rejected(self):
        """提示里推荐的写法自己必须能过门, 否则就是死路。"""
        check_code("import math\nprint(math.isqrt(17))\n")
        check_code("import sympy\nprint(sympy.isprime(7))\n")
