# -*- coding: utf-8 -*-
"""干净重克隆 Mathlib v4.9.0 到新目录, 然后 lake exe cache get。

上一轮的教训(记录下来免得再犯): 我在同一个目录上反复"修补"依赖,
结果把 mathlib4 根仓库的 HEAD 也带偏到了 Cli 的 commit,
根目录的 lakefile.lean 变成了 `package Cli` —— 一堆连锁的 unknown executable cache。
根仓库和它管理的子目录共用 git 操作时, 一次 cwd 搞错就污染全局。
**大仓库一旦状态可疑, 重克隆比修便宜。**
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\pythonProject\Others\mathlib4-v490")
LAKE = str(
    Path(os.path.expandvars(r"%USERPROFILE%\.elan\toolchains"))
    / "leanprover--lean4---v4.9.0" / "bin" / "lake.exe"
)
PROXY = "https://gh-proxy.com/https://github.com/"
URL = PROXY + "leanprover-community/mathlib4"
GIT_ENV = {
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": f"url.{PROXY}.insteadOf",
    "GIT_CONFIG_VALUE_0": "https://github.com/",
}


def run(cmd, cwd=None, label=""):
    t0 = time.time()
    print(f"\n>>> {label}", flush=True)
    p = subprocess.run(
        cmd, cwd=cwd, env={**os.environ, **GIT_ENV},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print(((p.stdout or "") + (p.stderr or ""))[-2500:], flush=True)
    print(f"--- exit={p.returncode} {time.time()-t0:.1f}s", flush=True)
    return p.returncode


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if run(["git", "init", "-q"], cwd=ROOT, label="init") != 0:
        return 1
    run(["git", "remote", "add", "origin", URL], cwd=ROOT, label="remote add")
    if run(
        ["git", "fetch", "--depth", "1", "origin", "refs/tags/v4.9.0"],
        cwd=ROOT, label="fetch tag v4.9.0",
    ) != 0:
        return 1
    if run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=ROOT, label="checkout") != 0:
        return 1

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
        capture_output=True, text=True,
    ).stdout.strip()
    print("HEAD =", head, flush=True)
    tc = (ROOT / "lean-toolchain").read_text(encoding="utf-8").strip()
    print("lean-toolchain =", tc, flush=True)
    lf = (ROOT / "lakefile.lean").read_text(encoding="utf-8")
    print("root package =", lf.splitlines()[2] if len(lf.splitlines()) > 2 else "?",
          flush=True)
    assert "package Cli" not in lf, "根 lakefile 又是 Cli 的 —— 克隆有问题"
    assert tc == "leanprover/lean4:v4.9.0", f"toolchain 不匹配: {tc}"

    # 依赖按 manifest 钉死的 rev 拉(不走 lake update, 那会把依赖拉到默认分支)
    import json
    mf = json.loads((ROOT / "lake-manifest.json").read_text(encoding="utf-8"))
    for e in mf.get("packages", []):
        name, rev, url = e.get("name"), e.get("rev"), e.get("url")
        if not (name and rev and url):
            continue
        dest = ROOT / ".lake" / "packages" / name
        dest.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q"], cwd=dest, label=f"init {name}")
        run(["git", "remote", "add", "origin", PROXY + url[len("https://github.com/"):]],
            cwd=dest, label=f"remote {name}")
        run(["git", "fetch", "-q", "--depth", "50", "origin", rev],
            cwd=dest, label=f"fetch {name}@{rev[:8]}")
        run(["git", "checkout", "-q", rev], cwd=dest, label=f"checkout {name}")

    rc = run([LAKE, "exe", "cache", "get"], cwd=ROOT, label="lake exe cache get")
    print("\n== cache get exit =", rc, flush=True)
    return 0 if rc == 0 else 3


if __name__ == "__main__":
    sys.exit(main())