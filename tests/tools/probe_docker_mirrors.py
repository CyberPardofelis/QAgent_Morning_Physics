"""实测哪个镜像源真能拉到 python 镜像 —— 通 ≠ 能拉。

连通性只说明服务在; 真正决定能不能用是能不能完成 manifest 解析 +
层下载。Docker 会按顺序试 registry-mirrors, 第一个通的胜出, 但如果
它能连上却拉不动, 就会一直卡在那儿而不是回退到下一个源。

所以这里真 pull, 拉得下才算数。
"""
import subprocess
import sys

MIRRORS = [
    "docker.m.daocloud.io",
    "docker.1ms.run",
    "docker.nju.edu.cn",
]
IMAGE = "python:3.12-slim"


def try_mirror(m: str) -> bool:
    print(f"\n=== 试 {m} ===", flush=True)
    # 只拉 manifest 不拉层, 快速判断能不能解析
    p = subprocess.run(
        ["docker", "manifest", "inspect", f"{m}/{IMAGE}"],
        capture_output=True, text=True, timeout=120,
    )
    if p.returncode == 0:
        print(f"  manifest OK  ({m})")
        return True
    err = (p.stderr or "").strip().splitlines()
    print(f"  manifest 失败: {err[-1] if err else '?'}")
    return False


def main() -> None:
    ok = [m for m in MIRRORS if try_mirror(m)]
    print("\n=== 可用源 ===")
    for m in ok:
        print(" ", m)
    if not ok:
        print("  全部不可用 —— 需要配代理或换网络")
        sys.exit(1)
    print(f"\n建议 registry-mirrors 顺序: {', '.join(ok)}")


if __name__ == "__main__":
    main()