"""查 Mathlib 各版本要求的 Lean 工具链版本。

为什么要查: Mathlib v4.34.1(2026 最新)要求 Lean 4.19+, 而我装的是 4.9.0。
版本不对, 缓存下来的 .olean 根本加载不了 —— 所以必须先确定
「哪个 Mathlib tag 对应 Lean 4.9.0」, 而不是随便抓最新的。

方法: 读每个 release 的 lean-toolchain 文件内容。
"""
import base64
import json
import subprocess

TAGS = ["v4.34.1", "v4.33.0", "v4.30.0", "v4.20.0", "v4.15.0", "v4.10.0", "v4.9.0"]


def get(url: str) -> str | None:
    p = subprocess.run(["curl.exe", "-sS", "-L", "--connect-timeout", "15",
                        "--max-time", "40", url], capture_output=True)
    if p.returncode != 0:
        return None
    return p.stdout


def toolchain_for(tag: str) -> str | None:
    """读 Mathlib 该 tag 的 lean-toolchain 文件。"""
    u = (f"https://api.github.com/repos/leanprover-community/mathlib4"
         f"/contents/lean-toolchain?ref={tag}")
    txt = get(u)
    if not txt:
        return None
    try:
        d = json.loads(txt.decode("utf-8-sig", errors="replace"))
    except Exception:
        return None
    if "content" not in d:
        return f"(无内容: {d.get('message', '?')})"
    raw = base64.b64decode(d["content"]).decode("utf-8", errors="replace")
    return raw.strip()


def main() -> None:
    print("Mathlib tag -> 要求的 Lean 工具链")
    print("=" * 56)
    for t in TAGS:
        tc = toolchain_for(t)
        mark = ""
        if tc and "4.9" in tc:
            mark = "  <== 匹配我装的 Lean 4.9.0"
        print(f"  {t:10} -> {tc or '(取不到)'}{mark}")

    print()
    print("说明: 我现在装的是 leanprover/lean4:v4.9.0")
    print("若上面没有匹配的 tag, 说明 4.9.0 太老 —— 要么升 Lean, "
          "要么 Mathlib 就用不了。")


if __name__ == "__main__":
    main()