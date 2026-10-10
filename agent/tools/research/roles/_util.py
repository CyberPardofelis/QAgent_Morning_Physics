"""四个角色共用的底座: 一次无状态 LLM 调用 + 稳健的 JSON 抽取。

## 为什么每个角色都是"一次独立调用"

`agent.tools.llm._call_llm(prompt, ...)` 是**无状态单发**: 不传历史,
不共享 conversation。这正是我们要的 —— Explorer / Worker / Verifier
必须在**互不可见**的上下文里各自工作。

不这么做的话, Verifier 会看见 Worker 的代码和 Explorer 的推理,
于是它验证的不是"这个制品对不对", 而是"这段推理我同不同意"。

## JSON 抽取为什么不用正则
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

#: 带语言标记的围栏。group(1)=语言, group(2)=正文
_FENCE = re.compile(r"```([A-Za-z0-9_+.\-]*)[ \t]*\r?\n(.*?)(?:```|\Z)",
                    re.DOTALL)
_BARE_FENCE = re.compile(r"```(.*?)(?:```|\Z)", re.DOTALL)

from agent.tools.llm import _call_llm

__all__ = ["ask", "extract_json", "extract_code", "Budget"]


class Budget:
    """整道题的预算: 轮次 / 某个方法重复次数 / token。"""

    def __init__(self, max_turns: int = 8, max_same_method: int = 2,
                 max_tokens: int | None = None):
        self.max_turns = max_turns
        self.max_same_method = max_same_method
        self.max_tokens = max_tokens
        self.turns = 0

    def spend(self) -> None:
        self.turns += 1

    def left(self) -> int:
        return max(0, self.max_turns - self.turns)

    def exhausted(self) -> bool:
        return self.turns >= self.max_turns


def ask(prompt: str, *, temperature: float = 0.3, max_tokens: int = 8000,
        timeout: int = 180, **kw) -> str:
    """一次 LLM 调用。上游失败时 `_call_llm` 返回的是**错误字符串**而不是抛异常,
    所以这里原样透传 —— 由调用方判断"拿到的是不是答案"。

    ★ 别在这里把错误吞掉转成 "" ★ 旧回路 `extract_code` 返回空 ->
    走 NoCode 分支 -> 白烧几轮, 而且日志上看不出是"模型没写代码"还是"上游挂了"。
    """
    return _call_llm(prompt, temperature=temperature, max_tokens=max_tokens,
                     timeout=timeout, **kw)


#: 上游失败的特征串。`_call_llm` 失败时**返回错误字符串**而不是抛异常,
#: 所以调用方必须自己分清"拿到了答案"和"拿到了报错"。
_ERROR_HEADS = ("❌", "[llm] 调用失败", "调用失败（", "API Error", "Traceback")


def upstream_error(raw: str | None) -> str | None:
    """这段文本是**上游报错**而不是模型输出吗? 是的话返回报错全文, 否则 None。

    ## 为什么必须单独分出来

    实测 2026-10-10: provider 没配对时, Explorer 连续 6 轮都返回
    `"❌ ... 调用失败 ... The supported API model names are ... but you passed ..."`。
    `extract_json` 对这段当然解析不出东西, 于是 orchestrator 记成
    **「Explorer 的输出里没解析出计划 JSON」**, 还给它一句 lesson:
    「下一轮把要求说得更直白一点」。

    ★ 两个问题 ★
    1. 那句 lesson 是**错的** —— 问题不在 Explorer 的输出格式, 在 API 400。
       照着它走只会把下一轮也带歪。
    2. 它把 6 轮预算全烧在同一个基础设施错误上, 最后报 `INCONCLUSIVE`
       并声称"用完预算没能判定" —— **那是假的**, 一次成功的调用都没有过。

    ⇒ 上游失败必须**立刻中止**, 并且报一个跟 INCONCLUSIVE 不同的状态:
      混进去会污染所有统计("它是不是不会做这题"和"它压根没调到模型"是两回事)。
    """
    if not raw:
        return None
    head = raw.lstrip()[:200]
    for h in _ERROR_HEADS:
        if h in head:
            return raw.strip()[:600]
    return None


def _scan_object(text: str, start: int) -> str | None:
    """从 `text[start]` 处的 `{` 开始做括号配平扫描, 取出完整对象。

    比正则 `\\{.*\\}` 强的地方: 数组里带 `{}`、字符串里带 `}` 都不会提前截断。
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth, in_str, esc = 0, False, False
    for j in range(start, len(text)):
        ch = text[j]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
    return None


def extract_json(text: str) -> dict[str, Any] | None:
    """从 LLM 输出里抽出第一个能解析成 dict 的 JSON 对象。

    认围栏 ```json ... ```, 也认裸的 {...}。宁可返回 None,
    也不要返回一个"猜出来的"结构 —— 猜出来的字段会让下游以为模型说了什么。
    """
    if not text:
        return None
    # 优先找围栏里的
    for fence in ("```json", "```JSON", "```"):
        i = text.find(fence)
        while i >= 0:
            body_start = i + len(fence)
            obj = _scan_object(text, body_start) if text[body_start:body_start + 1] == "{" \
                else None
            if obj is None:
                # 围栏里可能不是紧跟 {，找第一个 {
                j = text.find("{", body_start)
                obj = _scan_object(text, j) if j >= 0 else None
            if obj:
                try:
                    v = json.loads(obj)
                    if isinstance(v, dict):
                        return v
                except Exception:  # noqa: BLE001
                    pass
            i = text.find(fence, body_start)
    # 再退回全文扫
    for i, ch in enumerate(text):
        if ch == "{":
            obj = _scan_object(text, i)
            if obj:
                try:
                    v = json.loads(obj)
                    if isinstance(v, dict):
                        return v
                except Exception:  # noqa: BLE001
                    continue
    return None


#: 围栏代码块。★ 语言标记必须一起捕获 ★
#:   第一版把 "```python\n" 和裸 "```" 都当开栏标记, 裸标记会匹配到 "```python"
#:   的开头, 把 "python" 这个词留在代码第一行 -> `unterminated string literal`。
#:
#: ★ 第二版还踩了一层 ★ 就算改用正则把语言名剥掉, 裸围栏那一路仍会把
#:   "python\nimport sympy\n..." 整个收进来 —— 而 **`python` 单独一行在语法上
#:   是合法的**(它是个名字表达式), 所以"抽出来能 parse"这条判据**抓不住它**,
#:   只在运行时炸 NameError。
#:   ⇒ 必须按**跨度**去重: 带语言标记的正则吃过的围栏, 裸围栏那路不许再收。
_LANG_WORDS = frozenset({
    "python", "py", "python3", "ipython", "json", "javascript", "js",
    "typescript", "ts", "bash", "sh", "shell", "zsh", "powershell", "ps1",
    "c", "cpp", "java", "go", "rust", "ruby", "text", "txt", "output",
})


def _blocks(text: str) -> list[str]:
    """收集所有围栏代码块, 按长度从大到小。

    先用带语言标记的正则吃掉围栏; 裸围栏只处理**它没吃过的**那些。
    最后再剥一次残留的语言名行 —— 双重保险, 因为"能 parse"证明不了
    首行不是语言名。
    """
    found: list[str] = []
    consumed: list[tuple[int, int]] = []
    for m in _FENCE.finditer(text):
        body = m.group(2)
        if body and body.strip():
            found.append(_strip_lang_line(body.strip("\n")))
        consumed.append(m.span())

    def _covered(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in consumed)

    for m in _BARE_FENCE.finditer(text):
        if _covered(m.start(), m.end()):
            continue
        body = m.group(1)
        if body and body.strip():
            found.append(_strip_lang_line(body.strip("\n")))

    found.sort(key=len, reverse=True)
    seen, out = set(), []
    for f in found:
        k = f.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(f)
    return out


def _strip_lang_line(src: str) -> str:
    """剥掉首行孤零零的语言标记(``python`` / ``json`` 之类)。"""
    lines = src.split("\n")
    if lines and lines[0].strip().lower() in _LANG_WORDS:
        lines = lines[1:]
    return "\n".join(lines)


def _parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def _salvage(src: str) -> str:
    """把散文行从前面剥掉, 直到剩下的能解析。

    实测 2026-10-10 的 smoke: 三轮全挂在**抽取**上, 不是模型不会写 ——
      · 代码块第一行是中文说明, 里面带 `→` -> SyntaxError
      · 代码块第一行是"用 python 从 n=0 开始", Python 当成表达式 -> NameError
      · 说明里带 `²` -> SyntaxError
    三次都白烧一个 Explorer 轮次。所以: **抽出来的必须真能 parse。**
    """
    lines = src.split("\n")
    # 从后往前找起点: 越靠后的起点越可能正好落在真正的代码上
    for start in range(min(len(lines), 12)):
        cand = "\n".join(lines[start:])
        if cand.strip() and _parses(cand):
            return cand
    return src


def extract_code(text: str, lang_hint: str = "") -> str:
    """抽代码块, 并保证抽出来的东西**真的是 Python**。

    顺序: 最长的块 → 其它块 → 剥掉前导散文。都不行就返回最长块,
    让沙箱如实报语法错误(那也是有用的信息)。
    """
    if not text:
        return ""
    cands = _blocks(text)
    if not cands:
        # 没有围栏: 有些模型直接给代码。整体能 parse 就用, 否则剥散文。
        whole = text.strip()
        return _salvage(whole) if whole and not _parses(whole) else whole
    for c in cands:
        if _parses(c):
            return c
    best = cands[0]
    salv = _salvage(best)
    for c in cands[1:]:
        s = _salvage(c)
        if _parses(s) and len(s) > len(salv):
            salv = s
    return salv