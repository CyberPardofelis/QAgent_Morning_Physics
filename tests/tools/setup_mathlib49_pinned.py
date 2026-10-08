# -*- coding: utf-8 -*-
"""按 lake-manifest.json 的锁定 revision 拉 Mathlib 依赖。

坑: `lake update` 会把依赖全部解析到**默认分支最新**, 那些版本的 lakefile.toml
    是 lake5+ 格式(要求 source 键), 而 mathlib v4.9.0 配的是 lake 4.x,
    于是 lake 报 "missing required key: source" —— 跟网络无关, 是版本漂移。

修法: 恢复 mathlib v4.9.0 自带的 lake-manifest.json, 按里面钉死的 rev 逐个 clone,
      然后只跑 `lake exe cache get` (不再 lake update)。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\pythonProject\Others\mathlib4")
PKGS = ROOT / ".lake" / "packages"
LEAN_BIN = os.path.expandvars(
    r"%USERPROFILE%\.elan\toolchains\leanprover--lean4---v4.9.0\bin"
)
LAKE = str(Path(LEAN_BIN) / "lake.exe")
PROXY = "https://gh-proxy.com/https://github.com/"

GIT_ENV = {
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": f"url.{PROXY}.insteadOf",
    "GIT_CONFIG_VALUE_0": "https://github.com/",
}


def run(cmd, cwd=None, env=None, label=""):
    t0 = time.time()
    print(f"\n>>> {label}", flush=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(
        cmd, cwd=cwd, env=e, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    out = (p.stdout or "") + (p.stderr or "")
    print(out[-3000:], flush=True)
    print(f"--- exit={p.returncode}  {time.time()-t0:.1f}s", flush=True)
    return p.returncode, out


def main():
    # 1) 恢复 mathlib v4.9.0 自带的 manifest (lake update 已覆盖过)
    run(
        ["git", "checkout", "--", "lake-manifest.json"],
        cwd=ROOT, env=GIT_ENV, label="恢复 lake-manifest.json",
    )
    mf = json.loads((ROOT / "lake-manifest.json").read_text(encoding="utf-8"))
    entries = mf.get("packages", [])
    print(f"\nmanifest 钉住 {len(entries)} 个依赖", flush=True)

    # 2) 把漂移掉的依赖目录清掉, 按锁定 rev 重拉
    for e in entries:
        name = e.get("name")
        rev = e.get("rev")
        url = e.get("url")
        if not name or not rev or not url:
            continue
        dest = PKGS / name
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        # url 里的 .git 去掉; 走代理
        proxy_url = PROXY + re.sub(r"^https://github\.com/", "", url)
        rc, _ = run(
            ["git", "clone", url, str(dest)],
            cwd=ROOT, env=GIT_ENV, label=f"clone {name}",
        )
        if rc != 0:
            # clone 可能已把 origin 设成裸 github, 用代理重设再 fetch
            run(
                ["git", "remote", "set-url", "origin", proxy_url],
                cwd=dest, env=GIT_ENV, label=f"set-url {name}",
            )
            rc, _ = run(
                ["git", "fetch", "--depth", "1", "origin", rev],
                cwd=dest, env=GIT_ENV, label=f"fetch {name}@{rev[:8]}",
            )
            if rc != 0:
                print(f"!! {name} 拉取失败", flush=True)
                continue
        run(
            ["git", "checkout", rev],
            cwd=dest, env=GIT_ENV, label=f"checkout {name}@{rev[:8]}",
        )

    # 3) 只拉 oleans 缓存, 不再 lake update
    rc, _ = run([LAKE, "exe", "cache", "get"], cwd=ROOT, label="lake exe cache get")
    print("\n== 结果 ==", flush=True)
    print("cache get exit =", rc, flush=True)
    return 0 if rc == 0 else 3


if __name__ == "__main__":
    sys.exit(main())