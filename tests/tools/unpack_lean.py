"""解压 Lean4 工具链(.tar.zst)并装进 elan 的 toolchains 目录。

Windows 自带的 tar 没有 zstd 过滤器(报 "Can't initialize filter"),
所以用 Python 的 zstandard 解。
"""
import io
import os
import shutil
import sys
import tarfile

SRC = r"D:\pythonProject\Others\lean4-win.tar.zst"
TC = os.path.join(os.environ["USERPROFILE"], ".elan", "toolchains",
                  "leanprover--lean4---v4.9.0")


def main() -> None:
    import zstandard

    os.makedirs(TC, exist_ok=True)
    print(f"解压到 {TC} ...")
    dctx = zstandard.ZstdDecompressor()
    with open(SRC, "rb") as f:
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(TC, filter="data")
    print("解压完成")

    # tar 里通常是 lean-4.9.0-windows/ 一层, 去掉它让 elan 认得
    entries = os.listdir(TC)
    print("顶层:", entries[:6])
    inner = [e for e in entries if e.startswith("lean-4.9.0")]
    if inner:
        src = os.path.join(TC, inner[0])
        for name in os.listdir(src):
            dst = os.path.join(TC, name)
            if os.path.exists(dst):
                continue
            shutil.move(os.path.join(src, name), dst)
        try:
            os.rmdir(src)
        except OSError:
            pass
        print("已去掉顶层目录层")

    binpath = os.path.join(TC, "bin")
    print("bin:", os.listdir(binpath)[:6] if os.path.isdir(binpath) else "缺 bin/")
    lean_exe = os.path.join(binpath, "lean.exe")
    if os.path.exists(lean_exe):
        print(f"lean.exe 就位 ({os.path.getsize(lean_exe)} bytes)")
    else:
        print("!! 找不到 lean.exe")
        sys.exit(1)


if __name__ == "__main__":
    main()