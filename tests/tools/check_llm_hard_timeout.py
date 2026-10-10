"""验 `_call_with_hard_timeout` 真的能兜住"连接活着但结果不来"。

★ 为什么要有这个 ★ 2026-10-10 实测: part1 的 cvxpy-04 上一次 LLM 调用挂住
  **4 分钟无任何返回**(TCP `Established`、CPU 零增长、stdout 零输出),
  而 `kwargs["timeout"]` 是**确实传进去了**的。

  原因: httpx/OpenAI 的 timeout 是**分阶段**的, 而 **read timeout 量的是
  "两个数据块之间的间隔"** —— 服务端只要还在发心跳, 就永远不超时。
  于是"超时"这个保证恰好在"连接活着但结果不来"这个场合失效。

⇒ 所以判据不是"超时参数有没有传", 而是"**传了之后能不能真的回来**"。
   这个假 client 会**永远阻塞**, 正是那个场景的最小复现。
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.tools.llm import _call_with_hard_timeout  # noqa: E402

FAILS: list[str] = []


def ck(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if cond else '!! '}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


class _Completions:
    def create(self, **kw):          # noqa: D401 - 就是不返回
        time.sleep(3600)
        raise AssertionError("不该走到这里")


class _Chat:
    completions = _Completions()


class HangingClient:
    chat = _Chat()


class Boom:
    class _C:
        @staticmethod
        def create(**kw):
            raise RuntimeError("上游 500")
    chat = type("chat", (), {"completions": _C})()


class Fine:
    class _C:
        @staticmethod
        def create(**kw):
            return "OK-RESP"
    chat = type("chat", (), {"completions": _C})()


def main() -> int:
    print("=" * 74)
    print("① 真的挂住 -> 必须在 wall 秒内放弃")
    print("=" * 74)
    t = time.time()
    try:
        _call_with_hard_timeout(HangingClient(), {}, 3)
        ck("挂住时抛 TimeoutError", False, "居然正常返回了")
    except TimeoutError as e:
        el = time.time() - t
        ck("挂住时抛 TimeoutError", True, str(e)[:70])
        ck(f"在预算内放弃(实测 {el:.1f}s, 期望 ~8s)", 5 <= el <= 14, f"{el:.1f}s")
    except Exception as e:  # noqa: BLE001
        ck("挂住时抛 TimeoutError", False, f"{type(e).__name__}: {e}")

    print()
    print("=" * 74)
    print("② 正常返回 -> 必须原样通过, 不能被硬墙误伤")
    print("=" * 74)
    t = time.time()
    got = _call_with_hard_timeout(Fine(), {}, 30)
    ck("正常调用原样返回", got == "OK-RESP", f"{got!r} 耗时 {time.time()-t:.1f}s")

    print()
    print("=" * 74)
    print("③ 上游报错 -> 错误要**原样抛出去**, 不能被吞成超时")
    print("=" * 74)
    try:
        _call_with_hard_timeout(Boom(), {}, 30)
        ck("上游错误被抛出", False, "居然没抛")
    except RuntimeError as e:
        ck("上游错误原样抛出(不被吞成 TimeoutError)", True, str(e)[:60])
    except Exception as e:  # noqa: BLE001
        ck("上游错误原样抛出", False, f"变成了 {type(e).__name__}: {e}")

    print(f"\n失败 {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
