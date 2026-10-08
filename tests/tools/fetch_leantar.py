# -*- coding: utf-8 -*-
"""把 leantar 放到 lake 期望的位置, 绕过 lake 用 curl 直连 GitHub Releases 被重置的问题。

背景: `lake exe cache get` 的第一步是下载 leantar (Mathlib 的 tar 解包器),
      地址 https://github.com/digama0/leangz/releases/download/v0.1.13/...
      lake 调的是 curl, 不经过 git, 所以 GIT_CONFIG 的 insteadOf 代理**不生效**,
      实测卡在 "curl: (35) Recv failure: Connection was reset"。

lake 会从 <user>/.cache/mathlib/ 找已下载的 leantar, 所以直接摆进去即可。
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

DEST_DIR = Path(os.path.expandvars(r"%USERPROFILE%\.cache\mathlib"))
DEST = DEST_DIR / "leantar-0.1.13.exe.zip"
REAL = ("https://github.com/digama0/leangz/releases/download/"
        "v0.1.13/leantar-v0.1.13-x86_64-pc-windows-msvc.zip")
PROXIES = [
    "https://gh-proxy.com/" + REAL,
    REAL,
]


def fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    if DEST.is_file() and DEST.stat().st_size > 100_000:
        print(f"已存在, 跳过: {DEST} ({DEST.stat().st_size} bytes)")
        return 0

    last = None
    for url in PROXIES:
        for attempt in range(3):
            t0 = time.time()
            try:
                print(f"[尝试] {url}  (第{attempt+1}次)", flush=True)
                data = fetch(url)
                if len(data) < 100_000:
                    raise RuntimeError(f"下载过小, 疑似错误页: {len(data)} bytes")
                tmp = DEST.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(DEST)
                print(f"[成功] {len(data)} bytes -> {DEST}  "
                      f"({time.time()-t0:.1f}s)")
                return 0
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"  失败: {type(e).__name__}: {e}", flush=True)
                time.sleep(2)
    print(f"\n全部失败, 最后一个错误: {last}")
    return 1


if __name__ == "__main__":
    sys.exit(main())