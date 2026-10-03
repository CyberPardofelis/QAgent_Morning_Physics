"""
agent.tools.surveys._embed — 综述用的句向量编码（子进程隔离）

为什么要单独一个子进程:
    `.env` 记着一句 `0xC0000005 根因 = QWebEngineView + BGE 共享 native`。
    综述流水线跑在 QAgent 主进程里（同进程有 QWebEngineView），就地加载
    sentence-transformers 就是在赌那个段错误不发生。
    子进程里崩了只是这次编码失败，**主进程照常继续**，综述还能出结果
    （只是退回关键词归属）。

为什么不走 agent/local_embedder.Embedder:
    那个类被 QSettings 的 `embedding_enabled` 门控着，而综述需要 embedding
    但用户可能为了躲段错误把它关掉 —— 这里要的是"综述自己该用就用"。

模型默认 BAAI/bge-m3（多语言，中文强，实测中文 fact claim 分离度 0.073，
是 bge-base-zh 的两倍）。可用 QAGENT_SURVEY_EMBED_MODEL 覆盖。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "BAAI/bge-m3"
# 直连 huggingface.co 在本机超时(WinError 10060), 国内走镜像
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def _model() -> str:
    return os.getenv("QAGENT_SURVEY_EMBED_MODEL", "").strip() or DEFAULT_MODEL


def _endpoint() -> str:
    return os.getenv("HF_ENDPOINT", "").strip() or DEFAULT_ENDPOINT


def _child_script() -> Path:
    return Path(__file__).with_name("_embed_worker.py")


def available() -> bool:
    """能否编码。子进程失败一律返回 False, 绝不把异常抛给流水线。"""
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


def encode(texts: Sequence[str], timeout: int = 900) -> np.ndarray | None:
    """编码成 L2 归一化向量。失败返回 None —— 调用方负责降级。"""
    items = [t if t and t.strip() else "(空)" for t in texts]
    if not items:
        return None
    if not available():
        return None
    tmpdir = tempfile.mkdtemp(prefix="survey_embed_")
    src = os.path.join(tmpdir, "in.json")
    dst = os.path.join(tmpdir, "out.npy")
    try:
        with open(src, "w", encoding="utf-8") as f:
            json.dump({"model": _model(), "texts": items}, f, ensure_ascii=False)
        env = dict(os.environ)
        env["HF_ENDPOINT"] = _endpoint()
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run(   # noqa: S603 - 固定脚本路径, 输入输出走临时文件
            [sys.executable, str(_child_script()), src, dst],
            capture_output=True, text=True, timeout=timeout, env=env,
            # 10-03: 不指定 encoding 时 subprocess 用**系统 locale**(本机 GBK)解码,
            # 而子进程输出的是 UTF-8(sentence-transformers 的下载进度/告警),
            # 结果 UnicodeDecodeError 整个调用炸掉。必须显式指定。
            encoding="utf-8", errors="replace",
        )
        if r.returncode != 0 or not os.path.exists(dst):
            print(f"[_embed] 子进程失败 rc={r.returncode}: {(r.stderr or '')[:300]}",
                  flush=True)
            return None
        return np.load(dst)
    except Exception as e:
        print(f"[_embed] 编码异常: {type(e).__name__}: {e}", flush=True)
        return None
    finally:
        for p in (src, dst):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


__all__ = ["DEFAULT_MODEL", "available", "encode"]
