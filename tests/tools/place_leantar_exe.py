# -*- coding: utf-8 -*-
"""把已下载的 leantar zip 解出 leantar.exe, 放到 lake 期望的名字上。

lake 的 Cache/IO.lean 是「下载 zip -> 解压成 <cache>/leantar-<ver>.exe -> 再跑」。
下载那一步走 curl 直连 GitHub Releases(被重置), 但它**先检查 exe 是否已存在** ——
所以只要把 exe 直接摆到那个名字上, 就绕过下载了。
"""
import os
import shutil
import sys
import zipfile
from pathlib import Path

CACHE = Path(os.path.expandvars(r"%USERPROFILE%\.cache\mathlib"))
VER = "0.1.13"
ZIP = CACHE / f"leantar-{VER}.exe.zip"
EXE = CACHE / f"leantar-{VER}.exe"


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    if EXE.is_file():
        print(f"已就位: {EXE}")
        return 0
    if not ZIP.is_file():
        print(f"缺 zip: {ZIP} —— 先跑 fetch_leantar.py")
        return 1
    with zipfile.ZipFile(ZIP) as z:
        member = next(n for n in z.namelist() if n.endswith("leantar.exe"))
        with z.open(member) as src, open(EXE, "wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"解出 {EXE}  ({EXE.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())