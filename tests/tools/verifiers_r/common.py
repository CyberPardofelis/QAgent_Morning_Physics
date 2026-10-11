"""R 套题 verifier 的公共件：制品解析 + 检查收集。

## 为什么不解析自由文本

verifier 如果去"理解" agent 的 stdout 措辞, 就会**跟着 agent 的写法走** ——
agent 换个说法 verifier 就失效, 而且失效方式是静默的(correct 变 False,
看起来像"agent 做错了")。

所以只认**规范化制品行**: `CONSTRUCT: <json>` / `ANSWER: <json>` /
`COUNTEREXAMPLE: <json>`。JSON 解析失败 = 没提交制品 = 明确失败。
"""
from __future__ import annotations

import json
import re
from typing import Any

#: 制品行的**标记**。只匹配标记本身, JSON 由下面的大括号配平扫描去取。
_ARTIFACT_TAG = re.compile(
    r"^\s*(CONSTRUCT|ANSWER|COUNTEREXAMPLE)\s*:", re.IGNORECASE | re.MULTILINE)

#: VERDICT 标记。
_VERDICT = re.compile(r"^\s*VERDICT\s*:\s*([A-Za-z]+)", re.IGNORECASE | re.MULTILINE)


def _extract_json(text: str, start: int) -> str | None:
    """从 start 开始取出一个**完整的** JSON 对象(允许跨行、允许嵌套)。

    ★ 为什么不能用 `\\{.*\\}` 单行正则 ★
    实测翻车(2026-10-09 res-002): agent 用 `json.dumps(obj, indent=2)` 打印,
    制品行变成
        CONSTRUCT: {
          "marks": [...],
          "length": 170
        }
    单行正则只吃到 `CONSTRUCT: {`, 于是**判它"没交制品"** —— 而它交得好好的。
    这是"verifier 太严于表示形式"的第 4 次, 这次不在单个 verifier 上打补丁,
    而是**把解析器本身改对**, 一笔修一整类。

    大括号配平时必须跳过字符串字面量, 否则字符串里一个 `{` 就会让计数错位。
    """
    i = start
    n = len(text)
    while i < n and text[i].isspace():   # 含换行 —— 只跳空格/tab 会漏掉
        i += 1
    if i >= n or text[i] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, n):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    return None


def parse_artifact(stdout: str, tag: str | None = None) -> dict | None:
    """抓制品行并解析成 dict。找不到或 JSON 坏了都返回 None。

    ★ 多行 JSON 也认 ★ 见 `_extract_json` 的说明。
    """
    if not stdout:
        return None
    for m in _ARTIFACT_TAG.finditer(stdout):
        if tag and m.group(1).upper() != tag.upper():
            continue
        raw = _extract_json(stdout, m.end())
        if raw is None:
            continue
        try:
            v = json.loads(raw)
        except Exception:
            continue
        if isinstance(v, dict):
            return v
    return None


def parse_verdict(stdout: str) -> str | None:
    if not stdout:
        return None
    hits = _VERDICT.findall(stdout)
    return hits[-1].lower() if hits else None


class Check:
    """收集多项检查的结果。任何一项失败 -> correct 为 False。"""

    def __init__(self) -> None:
        self.details: dict[str, Any] = {}
        self.failures: list[str] = []

    def expect(self, name: str, cond: bool, msg: str = "") -> bool:
        self.details[name] = bool(cond)
        if not cond:
            self.failures.append(f"{name}: {msg or '不满足'}")
        return bool(cond)

    def note(self, name: str, value: Any) -> None:
        """记录一个不影响对错的事实（用于诊断）。"""
        self.details[name] = value

    @property
    def correct(self) -> bool:
        return not self.failures

    def reason(self, limit: int = 3) -> str:
        if not self.failures:
            return "全部检查通过"
        return "; ".join(self.failures[:limit])


def strip_accents(s: str) -> str:
    """数学记号归一化: 去掉空白、全角字符、常见的等价写法。"""
    s = s.replace("−", "-").replace("–", "-").replace(" ", "")
    return s