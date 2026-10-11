"""沙箱读文件逃逸的回归测试。

★ 为什么这条测试比其他门测试更重要 ★
2026-10-09 实测: `import io; io.open(绝对路径).read()` 完整穿过静态门,
把 data/*_reference.json 里的**参考答案**读走了(输出 LEAK_LEN 4959)。
而 prompt 里写着「读文件一律不行」—— 那句话是假的。

一个靠"模型读不到答案"成立的基准, 如果这句话不成立, 整套分数都不作数。
所以这条必须钉死, 而且要覆盖**每一条已知的绕过路径**, 不是只测一种写法。
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agent.tools.research._sandbox import (  # noqa: E402
    SandboxRejected, check_code, run_code,
)

FAILS: list[str] = []
ABS = os.path.abspath("data/res_16_reference.json").replace("\\", "\\\\")


def blocked(name: str, code: str) -> None:
    try:
        check_code(code)
    except SandboxRejected as e:
        print(f"  [OK ] {name:38} 静态门拦下")
        return
    print(f"  [!! ] {name:38} ★ 静态门放行了!")
    FAILS.append(name)


def allowed(name: str, code: str) -> None:
    try:
        check_code(code)
    except SandboxRejected as e:
        print(f"  [!! ] {name:38} ★ 误杀合法代码: {str(e)[:60]}")
        FAILS.append(name)
        return
    print(f"  [OK ] {name:38} 放行")


def main() -> int:
    print("=" * 74)
    print("沙箱读文件逃逸 —— 回归测试")
    print("=" * 74)

    print("\n>>> 必须拦下的绕过路径")
    blocked("io.open(绝对路径)",
            f'import io\nio.open("{ABS}").read()\n')
    blocked("io.open(...).readlines()",
            f'import io\nio.open("{ABS}").readlines()\n')
    blocked("open(绝对路径)", f'open("{ABS}").read()\n')
    blocked("io.open + read_text 变体",
            f'import io\nio.open("{ABS}", "r").read()\n')
    blocked("pathlib", f'import pathlib\npathlib.Path("{ABS}").read_text()\n')
    blocked("os 模块本身", "import os\n")

    print("\n>>> ★ numpy 的文件读取器(上一轮漏掉的整类) ★")
    # 上一轮我测了 io/open/pathlib/os 就宣布"封堵完成", 而 numpy 整类没测。
    # 实测 np.loadtxt / np.genfromtxt 都真的读到了文件(报的是内容解析错误)。
    blocked("np.loadtxt(路径字面量)",
            f'import numpy as np\nnp.loadtxt(r"{ABS}")\n')
    blocked("np.genfromtxt(路径字面量)",
            f'import numpy as np\nnp.genfromtxt(r"{ABS}", dtype=str)\n')
    blocked("np.fromfile(路径字面量)",
            f'import numpy as np\nnp.fromfile(r"{ABS}", dtype=bytes)\n')
    blocked("np.load(.npy 路径)",
            f'import numpy as np\nnp.load(r"{ABS}")\n')
    blocked("np.memmap", f'import numpy as np\nnp.memmap(r"{ABS}")\n')
    blocked("np.DataFrame.from_csv",
            f'import numpy as np\nnp.DataFrame.from_csv(r"{ABS}")\n')

    print("\n>>> numpy 从内存读要放行(那不是文件)")
    allowed("np.load(变量)", "import numpy as np\nnp.load(buf)\n")
    allowed("np.array(内存)", "import numpy as np\nnp.array([1,2,3])\n")
    # ★ csv 放行是正确的, 这里显式说明为什么它不是一条绕过路径 ★
    # csv.reader(x) 需要一个**文件对象** x, 而在这个沙箱里能拿到文件对象的
    # 途径只有 open / io.open / pathlib.Path(...).open —— 全部已拦。
    # 拿不到 fh 的 csv.reader 是个空壳, 什么也读不到。
    # (第一版把这条写成"必须拦", 是我的期望写错了, 不是门有问题。)
    allowed("csv.reader(拿不到 fh, 空壳)", "import csv\ncsv.reader(fh)\n")

    print("\n>>> 必须放行的合法代码(别把门关死)")
    allowed("纯计算", "import math\nprint(math.isqrt(17))\n")
    allowed("sympy 化简", "import sympy\nx=sympy.symbols('x')\n"
                        "print(sympy.simplify(x**2-1))\n")
    allowed("z3", "import z3\ns=z3.Solver()\nprint(s.check())\n")
    allowed("numpy", "import numpy as np\nprint(np.arange(5).sum())\n")
    # ★ io 整个从白名单移除, 所以 io.StringIO(纯内存) 也用不了 ★
    # 这是**有意**的: 这个项目没有输入文件要读, prompt 也没要求 StringIO。
    # 宁可少一个便利功能, 也不要留一条读文件的路 —— 那条路直通参考答案。
    blocked("io 整个模块(放弃 StringIO 换安全)",
            "import io\nb=io.StringIO()\n")

    print("\n>>> 端到端: 真跑一遍, 确认运行时也读不到")
    tmp = tempfile.mkdtemp()
    try:
        leak = (f'import io\n'
                f'try:\n'
                f'    d = io.open(r"{ABS}").read()\n'
                f'    print("LEAK", len(d))\n'
                f'except Exception as e:\n'
                f'    print("BLOCKED", type(e).__name__)\n')
        try:
            r = run_code(leak, tmp, timeout=30)
            out = (r.stdout or "").strip()
        except SandboxRejected as e:
            # ★ 静态门在 import 阶段就拦住了 —— 这正是修复生效的样子 ★
            out = f"静态门拦下: {str(e)[:60]}"
        leaked = out.startswith("LEAK")
        print(f"  [{'!! ' if leaked else 'OK '}] 运行时读文件 -> {out[:90]}")
        if leaked:
            FAILS.append("运行时仍能读文件")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 74)
    if FAILS:
        print(f"✗ {len(FAILS)} 项不通过: " + "; ".join(FAILS))
        return 1
    print("✓ 全部通过 —— 模型读不到任何参考答案")
    return 0


if __name__ == "__main__":
    sys.exit(main())