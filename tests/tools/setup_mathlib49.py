# -*- coding: utf-8 -*-
"""给 Lean 4.9.0 装匹配的 Mathlib (tag v4.9.0)。

背景:
  - Mathlib 最新 v4.34.1 要求 Lean 4.19+, 我装的是 4.9.0
  - 实测 tag v4.9.0 的 lean-toolchain 正是 leanprover/lean4:v4.9.0 -> 匹配
  - Mathlib 不走 GitHub Release 资产, 走 cache.mathlib.org (Azure Blob, 国内可达)
  - 大件放 D:\\pythonProject\\Others (用户约定)

步骤:
  1. 浅克隆 tag v4.9.0
  2. lake update (可能不需要, shallow clone 已有 lean-toolchain)
  3. lake exe cache get  <- 这一步拉 oleans 缓存, 省掉几小时编译
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = r"D:\pythonProject\Others\mathlib4"
LEAN_BIN = os.path.expandvars(
    r"%USERPROFILE%\.elan\toolchains\leanprover--lean4---v4.9.0\bin"
)
LAKE = os.path.join(LEAN_BIN, "lake.exe")
GIT = "git"
PROXY = "https://gh-proxy.com/https://github.com/"
URL = PROXY + "leanprover-community/mathlib4.git"

# lake 拉依赖(batteries/importGraph 等)时走的是裸 github.com, 国内连不上。
# 用环境变量形式的 git 配置注入 insteadOf 重写, **不动用户的全局 git config**。
GIT_ENV = {
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": f"url.{PROXY}.insteadOf",
    "GIT_CONFIG_VALUE_0": "https://github.com/",
}


def run(cmd, cwd=None, env=None, label=""):
    t0 = time.time()
    print(f"\n>>> {label or ' '.join(cmd)}", flush=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=e,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (p.stdout or "") + (p.stderr or "")
    print(out[-4000:], flush=True)
    print(f"--- exit={p.returncode}  {time.time()-t0:.1f}s  {label}", flush=True)
    return p.returncode, out


def main():
    os.makedirs(r"D:\pythonProject\Others", exist_ok=True)

    if os.path.isdir(os.path.join(ROOT, ".git")):
        print("[1/3] 已存在克隆, 跳过", flush=True)
    else:
        if os.path.isdir(ROOT):
            shutil.rmtree(ROOT)
        rc, _ = run(
            [
                GIT,
                "clone",
                "--depth",
                "1",
                "--branch",
                "v4.9.0",
                "--single-branch",
                URL,
                ROOT,
            ],
            label="git clone --depth 1 -b v4.9.0",
        )
        if rc != 0:
            print("!! clone 失败", flush=True)
            return 1

    tc = os.path.join(ROOT, "lean-toolchain")
    if os.path.isfile(tc):
        print("[toolchain]", open(tc).read().strip(), flush=True)

    # 先 lake update 拉依赖(batteries/importGraph/Qq 等), 再 cache get 拉 oleans
    rc, _ = run(
        [LAKE, "update"], cwd=ROOT, env=GIT_ENV, label="lake update"
    )
    if rc != 0:
        print("!! lake update 失败", flush=True)
        return 2

    # lake exe cache get  -- 拉 oleans, 否则要本地编译几小时
    rc, out = run(
        [LAKE, "exe", "cache", "get"],
        cwd=ROOT,
        env=GIT_ENV,
        label="lake exe cache get",
    )
    if rc != 0:
        print("!! cache get 失败, 可能需要先 lake update", flush=True)
        return 2

    print("\n== Mathlib v4.9.0 就绪 ==", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())