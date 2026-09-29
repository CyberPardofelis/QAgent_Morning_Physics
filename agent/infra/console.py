"""
agent/infra/console — 控制台编码兜底 (9-29)

要解决的问题 (实测):
    项目里大量 `print("... ❌ ...")` 这类带 emoji / 符号的进度输出。
    Windows 默认控制台是 cp936(GBK), 编码不了 ✅❌⚠️⟵⊕ 等字符,
    一旦 print 到这种字符就是 `UnicodeEncodeError`,
    **直接把整个业务流程打断** —— 实测 `survey_generate` 在 APS 召回失败时
    就崩在 `_build_pool.py` 的一行 print 上。

修法:
    在应用入口把 stdout/stderr 换成"遇到编不了的字符就替换"而不是抛异常。
    一处生效, 覆盖全项目所有 print, 且不改变任何输出内容(只是不崩了)。
    需要 UTF-8 控制台的用户依然能看到 emoji(Windows Terminal / chcp 65001)。
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

# 这些字符在 cp936 里编不了, 出现时用它们替换
_FALLBACK = {"✅": "[OK]", "❌": "[X]", "⚠️": "[!]", "⚠": "[!]",
             "⟵": "->", "⊕": "+", "⏳": "...", "🔍": "[search]",
             "📚": "[kb]", "📝": "[note]", "✍": "[write]",
             "🧭": "[map]", "🔎": "[check]", "💡": "[tip]",
             "📊": "[stat]", "🚫": "[no]", "👁": "[see]"}


def harden_stream(stream, name: str) -> bool:
    """
    把一个文本流改成"编不出来的字符替换掉"而不是抛异常。

    返回是否成功改造。
    """
    if stream is None:
        return False
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        # 被重定向成文件对象等, 多数情况已支持 utf-8, 只能靠写文件时注意
        return False
    try:
        # errors="replace" 是关键: 原来默认 "strict" 遇错就抛
        reconfigure(errors="replace")
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("[console] 无法重配置 %s: %s", name, e)
        return False


def safe_print(*args, **kwargs) -> None:
    """
    永远不抛异常的 print。

    优先用原生 print(靠 harden_stream 兜底), 失败时退回到
    "先剔除编不了的符号再打印", 再失败就打印占位提示。
    供库代码里那些"纯进度输出"使用 —— 进度输出崩掉业务是不可接受的。
    """
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
        return
    except UnicodeEncodeError:
        pass
    except Exception:  # noqa: BLE001
        return
    # 第二遍: 把编不了的字符剔掉
    try:
        text = kwargs.pop("sep", " ").join(str(a) for a in args)
        cleaned = "".join(ch if _printable(ch) else "?" for ch in text)
        print(cleaned, **kwargs)
    except Exception:  # noqa: BLE001
        try:
            print(f"[输出失败: 非 ASCII 字符过多] {len(args)} 段", **kwargs)
        except Exception:  # noqa: BLE001
            pass


def _printable(ch: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "cp936"
    try:
        ch.encode(enc, errors="strict")
        return True
    except (UnicodeEncodeError, LookupError):
        return False
    except Exception:  # noqa: BLE001
        return True


def install() -> bool:
    """
    应用启动时调用一次。返回是否成功防护了 stdout。
    """
    ok_out = harden_stream(sys.stdout, "stdout")
    ok_err = harden_stream(sys.stderr, "stderr")
    if not (ok_out and ok_err):
        logger.info(
            "[console] 未能重配置控制台编码, 库内请改用 console.safe_print "
            "(stdout 可重配置=%s, stderr 可重配置=%s)", ok_out, ok_err,
        )
    return ok_out
