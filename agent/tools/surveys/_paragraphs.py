"""
agent.tools.surveys._paragraphs — 章节 markdown 段落切分 / 拼接 (9-27)

为什么需要:
    单段重写的前提是"能把一个已写好的章节稳定地切成段, 重写其中几段, 再拼回去"。
    综述章节的实际结构比想象的复杂:
      - 可能有二级标题 (## 某小节)
      - 可能有列表项 (- / * / 1.)
      - 空行是段落分隔, 但列表项之间的空行**不应**切段
      - 可能有连续多个空行
      - 可能有引用块 (>) 和代码块 (```)

    切分规则 (尽量少切, 不破坏语义):
      1. 先按"空行 + 下一个非空行不是列表/引用/标题"切 —— 这样列表整体留在一段
      2. 标题单独成段
      3. 其余按空行切
    拼接时用 "\n\n".join, 保证往返一致(见 roundtrip 测试)。
"""
from __future__ import annotations

import re
from typing import Any

# 列表 / 引用 / 标题 —— 这些行不能被当作新段落的开头
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|>\s+)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_FENCE_RE = re.compile(r"^\s*```")


def split_paragraphs(markdown: str) -> list[dict[str, Any]]:
    """
    把章节 markdown 切成段。

    Returns:
        [{"idx": int, "kind": "heading"|"body"|"list", "text": str, "preview": str}, ...]
        kind 用于给 LLM 的改写指令提供结构信息。
    """
    if not markdown or not markdown.strip():
        return []

    lines = markdown.replace("\r\n", "\n").split("\n")
    n = len(lines)

    def next_content_idx(i: int) -> int:
        """从 i 起找第一个非空行的下标, 没有返回 -1。"""
        j = i
        while j < n and not lines[j].strip():
            j += 1
        return j if j < n else -1

    paras: list[dict[str, Any]] = []
    buf: list[str] = []
    in_fence = False

    def flush():
        if not buf:
            return
        text = "\n".join(buf).rstrip()
        if text.strip():
            stripped = text.strip()
            if _HEADING_RE.match(stripped):
                kind = "heading"
            elif _LIST_RE.match(stripped):
                kind = "list"
            else:
                kind = "body"
            preview = " ".join(stripped.split())[:60]
            paras.append({
                "idx": len(paras), "kind": kind, "text": text, "preview": preview,
            })
        buf.clear()

    i = 0
    while i < n:
        line = lines[i]

        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buf.append(line)
            i += 1
            continue

        if in_fence:
            buf.append(line)
            i += 1
            continue

        if not line.strip():
            j = next_content_idx(i)
            if j < 0:
                i += 1
                continue
            nxt = lines[j]
            if not buf:
                i = j          # 跳过连续空行
                continue
            last = buf[-1]
            if _LIST_RE.match(last):
                # 列表项之间的空行只是分隔, 继续留在同一段
                buf.append(line)
                i += 1
                continue
            if _HEADING_RE.match(last):
                # 标题必须独立成段(否则 "## 标题" 会粘住后面的正文)
                flush()
                i = j
                continue
            if _LIST_RE.match(nxt) or _HEADING_RE.match(nxt):
                # 下一块是结构块, 这个空行是块间分隔, 归给下一块
                flush()
                i = j
                continue
            flush()
            i = j
            continue

        buf.append(line)
        i += 1

    flush()
    return paras


def join_paragraphs(paras: list[dict[str, Any]]) -> str:
    """段列表 → markdown。与 split_paragraphs 严格互逆。"""
    parts = [p["text"].rstrip() for p in paras if p.get("text", "").strip()]
    return "\n\n".join(parts)


def render_outline_for_rewrite(paras: list[dict[str, Any]], targets: list[int]) -> str:
    """
    给 LLM 看的"当前章节结构 + 哪几段要重写"。

    保留未重写段的**内容**(作为上下文), 只把目标段标成 [要重写]。
    """
    tset = set(targets)
    lines: list[str] = []
    for p in paras:
        if p["idx"] in tset:
            lines.append(f"[第 {p['idx']} 段 — 要重写]\n{p['text']}\n")
        else:
            lines.append(f"[第 {p['idx']} 段 — 保持不变]\n{p['text']}\n")
    return "\n".join(lines)


def extract_context(
    paras: list[dict[str, Any]],
    targets: list[int],
    before: int = 1,
    after: int = 1,
) -> str:
    """取目标段前后各 N 段作为写作上下文(不含目标段本身)。"""
    if not targets:
        return ""
    lo = max(0, min(targets) - before)
    hi = min(len(paras) - 1, max(targets) + after)
    picked = [p for p in paras[lo:hi + 1] if p["idx"] not in set(targets)]
    if not picked:
        return ""
    return "\n\n".join(f"[第 {p['idx']} 段] {p['text']}" for p in picked)
