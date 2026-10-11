"""把 20 道 verifier 全部跑一遍自测, 并汇总通过情况。

子代理报告"全部通过"是**证据不是结论** —— 它们可能改了 common.py、
可能自测其实跳过了、可能退出码骗人。这里独立重跑, 谁都不信。
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
VDIR = os.path.join(HERE, "verifiers_r")
ROOT = os.path.dirname(os.path.dirname(HERE))

sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from agent.tools.research._proc import run_capture  # noqa: E402


def main() -> int:
    files = sorted(f for f in os.listdir(VDIR)
                   if f.startswith("res_") and f.endswith(".py"))
    if not files:
        print("verifiers_r 下没有 verifier 文件")
        return 1

    ok_files, bad_files, skipped = [], [], []
    print("=" * 74)
    print(f"逐个跑自测 (共 {len(files)} 个文件)")
    print("=" * 74)
    for f in files:
        # 走 run_capture: 强制子进程 UTF-8, 否则下面的 `"正例" in l` 恒为 False
        p = run_capture([sys.executable, os.path.join(VDIR, f)],
                        cwd=ROOT, timeout=900)
        out = (p.stdout or "") + (p.stderr or "")
        if "[SKIP]" in out:
            skipped.append(f)
            print(f"  [SKIP] {f}")
        elif p.returncode == 0:
            ok_files.append(f)
            # 只显示正例那一行 —— 反例也要过才有意义, 但不必全贴
            pos = [l for l in out.splitlines() if "正例" in l]
            print(f"  [OK  ] {f:34} {pos[0].strip()[:56] if pos else ''}")
        else:
            bad_files.append(f)
            print(f"  [FAIL] {f:34} exit={p.returncode}")
            for l in out.splitlines()[-6:]:
                print(f"         {l}")

    # ---- 契约一致性: verify() 必须在, 返回四键齐全 ----
    print("\n" + "=" * 74)
    print("契约一致性检查")
    print("=" * 74)
    import importlib.util
    contract_bad = []
    for f in files:
        mod_name = f[:-3]
        try:
            spec = importlib.util.spec_from_file_location(
                mod_name, os.path.join(VDIR, f))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        except Exception as e:
            contract_bad.append(f"{f}: import 失败 {type(e).__name__}: {e}")
            continue
        if not hasattr(m, "verify"):
            contract_bad.append(f"{f}: 缺 verify()")
            continue
        try:
            r = m.verify("", "", "")
        except Exception as e:
            contract_bad.append(f"{f}: verify(空输入) 抛异常 {type(e).__name__}")
            continue
        need = {"correct", "certificate", "final_answer", "details"}
        if not isinstance(r, dict):
            contract_bad.append(f"{f}: verify() 没返回 dict")
            continue
        miss = need - set(r)
        if miss:
            contract_bad.append(f"{f}: 返回缺键 {miss}")
        if r.get("correct") is not False:
            contract_bad.append(f"{f}: 空输入的 correct 应为 False, 实为 "
                                f"{r.get('correct')!r}  ← 可能是'没给制品也放过'")
    for b in contract_bad:
        print(f"  [!!] {b}")
    if not contract_bad:
        print(f"  [OK ] {len(files)} 个 verify() 全部: 签名齐全 / 空输入判 False")

    print("\n" + "=" * 74)
    print(f"自测通过 {len(ok_files)} | 环境跳过 {len(skipped)} | "
          f"自测失败 {len(bad_files)}")
    if bad_files:
        print("  失败:", ", ".join(bad_files))
    if skipped:
        print("  跳过:", ", ".join(skipped))
    if contract_bad:
        print(f"  契约问题 {len(contract_bad)} 处")
    print("=" * 74)
    return 1 if (bad_files or contract_bad) else 0


if __name__ == "__main__":
    sys.exit(main())