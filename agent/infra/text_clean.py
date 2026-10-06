"""
agent/infra/text_clean — LLM 输出文本的字节级清洗 (10-06 加)

**要解决的问题(实测, 不是假想)**
10-06 跑覆盖空白回补, MiniMax-M3 的同一次调用, 连着 6 次输出里 6 次混进 NUL
字节(\\x00)。长相是这样的:

    "Toff\\x00oli\\x00\\x00 GHZ global addressing"
    "Rydberg dressing — this is a\\x00 specific subtopic, papers exist"

`Toffoli` 变成 `Toff\\x00oli`, 破折号变成 `\\x00\\x00 `。这是**输出流在字节
层面被破坏**, 不是模型"打了个看不见的字符"。

**为什么 `json.loads(strict=False)` 治不了**
`strict=False` 只放行 `\\n` `\\t` `\\r` 那三个标准控制符, NUL 照样报
`Expecting value`。踩了两轮才定位到 —— 之前一直怀疑是截断、是格式不稳定。

**为什么在这里修, 而不是给每个解析点打补丁**
代码库里解析 LLM 输出的地方有二十多处(`_organize._parse_theme_json`、
`_pool._utils._safe_json_loads_array`、`_fact_audit`、`_relevance_filter`、
`_actions/_decide`、`_search_more`、`_write_outline`、`memory/extractor` …),
每一处单独加一行 `replace("\\x00", "")` 意味着:
  - 漏一处就漏一处, 而漏掉的症状是"那个工具偶尔静默失败", 极难归因
  - 二十处重复同一个修复, 以后换策略要改二十遍

NUL 是**传输层**损坏, 那就在它进入程序的那个边界一次性清掉:
`_call_llm` 返回之前。所有工具的 JSON 解析都在那之下, 一处生效全覆盖。

**`json_control_char_ok` 是给"已经拿到一段脏文本"的地方留的后门**
比如解析历史日志里存的旧输出。优先用 `sanitize_llm_text`。

**为什么直接删而不是替换成空格**
实测脏字节总是插在词中间(`Toff\\x00oli`), 换成空格会留下 `Toff oli`
这种"看着像真的"的碎片, 比缺字更难发现。删掉至少还原成可读的 `Toffoli`。
"""
from __future__ import annotations

import json
from typing import Any


def strip_nul(text: str) -> str:
    """删掉 NUL 字节。**永不抛异常**, 传入非 str 原样返回。"""
    if not isinstance(text, str):
        return text
    if "\x00" not in text:
        return text
    return text.replace("\x00", "")


def strip_control_chars(text: str) -> str:
    """删掉 JSON 字符串里**未转义的控制字符**(\\t \\n \\r 保留)。

    比 strip_nul 更彻底: 模型除了 NUL, 偶尔也会吐裸 \\x0b \\x0c \\x07 之类。
    这些在 `json.loads(strict=False)` 下同样炸。

    ⚠️ 会破坏"字符串里本来就该有裸换行"的容错 —— 但那种文本本来就不是
    合法 JSON(strict=False 才放行), 而我们最终是要拿数据去用的。
    """
    if not isinstance(text, str):
        return text
    return "".join(
        ch for ch in text
        if ch in "\t\n\r" or (ord(ch) >= 0x20 and ord(ch) != 0x7F)
    )


def sanitize_llm_text(text: str) -> str:
    """LLM 输出的标准清洗入口: 先删 NUL, 再删其余控制字符。

    调用点应该在 **LLM 响应进入程序的地方**(目前是 `_call_llm` 的 return),
    而不是各个解析点 —— 见模块 docstring 的"为什么在这里修"。
    """
    if not isinstance(text, str):
        return text
    if "\x00" in text:
        text = text.replace("\x00", "")
    # 其余控制字符很罕见, 先探一下再动手, 免得每次都遍历长字符串
    if any(ord(c) < 0x20 and c not in "\t\n\r" for c in text) or "\x7f" in text:
        text = strip_control_chars(text)
    return text


def loads_maybe_dirty(text: str) -> Any:
    """解析**已知可能带 NUL / 裸控制字符**的 JSON 文本。

    给那些拿不到 `_call_llm` 出口的路径用(读历史产物、旧日志、复现现场)。
    先洗再 `strict=False`, 两道都留着: 洗一遍解决 NUL, strict=False 兜住
    裸 \\n \\t。
    """
    if not isinstance(text, str):
        return None
    try:
        return json.loads(strip_control_chars(strip_nul(text)), strict=False)
    except Exception:                                       # noqa: BLE001
        return None
