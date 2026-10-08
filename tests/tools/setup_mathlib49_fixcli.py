# -*- coding: utf-8 -*-
"""补齐 Cli 包 (上次浅取 merge commit 缺文件), 然后 lake exe cache get。"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\pythonProject\Others\mathlib4")
PKGS = ROOT / ".lake" / "packages"
LAKE = str(
    Path(os.path.expandvars(r"%USERPROFILE%\.elan\toolchains"))
    / "leanprover--lean4---v4.9.0" / "bin" / "lake.exe"
)
PROXY = "https://gh-proxy.com/https://github.com/"
GIT_ENV = {
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": f"url.{PROXY}.insteadOf",
    "GIT_CONFIG_VALUE_0": "https://github.com/",
}
CLI_URL = PROXY + "leanprover/lean4-cli"
CLI_REV = "a11566029bd9ec4f68a65394e8c3ff1af74c1a29"


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
    dest = PKGS / "Cli"
    # 上一轮 rmtree 带 ignore_errors, Windows 上 .git 里的只读文件会残留
    if dest.exists():
        shutil.rmtree(dest, onerror=lambda f, p, e: (os.chmod(p, 0o777), f(p)))
    dest.mkdir(parents=True, exist_ok=True)

    if run(["git", "init", "-q"], cwd=dest, label="git init") != 0:
        return 1
    if run(["git", "remote", "add", "origin", CLI_URL], cwd=dest, label="remote add") != 0:
        return 1
    # merge commit 要完整取, --depth 1 拿不到树
    if run(["git", "fetch", "--depth", "50", "origin", CLI_REV], cwd=dest, label="fetch") != 0:
        return 1
    if run(["git", "checkout", CLI_REV], cwd=dest, label="checkout") != 0:
        return 1
    print("Cli files:", sorted(x.name for x in dest.iterdir())[:20], flush=True)

    rc = run([LAKE, "exe", "cache", "get"], cwd=ROOT, label="lake exe cache get")
    print("\n== cache get exit =", rc, flush=True)
    return 0 if rc == 0 else 3


if __name__ == "__main__":
    sys.exit(main())