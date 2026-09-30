"""
_pool._topic_match — 主题短语匹配 (9-30)

为什么需要
----------
9-30 实测: 候选池召回到的论文 95% 过不了 LLM 审查, 追下去发现根因在**召回层**,
不在审查层。

旧逻辑把主题按空格切成单词, 任一命中即算相关:

    topic = "tensor network states quantum simulation"
    切词 -> ['tensor', 'network', 'states', 'quantum', 'simulation']

于是在物理论文里, "quantum" 和 "network" 几乎每篇都有 —— 实测 247 篇本地论文
命中 218 篇 (88%)。排在最前的包括:
    - Quantum Entanglement in Neural Network States      (神经网络态, 不是张量网络)
    - A quantum network stack and protocols             (量子网络, 不是张量网络)
    - Photon-efficient quantum key distribution          (QKD, 无关)
    - The quantum technologies roadmap                   (无关)

这些不是"审得太严", 是**召回灌进来的本来就是噪声**。LLM 审查在正确地倒掉它们,
只是成本被推到了审查那一步(每篇一次 LLM 调用)。

做法
----
把主题拆成**短语**(2~3 词) + **单词**两层:
  - 短语命中 → 权重高, 并且是**准入条件**: 主题有 2 个以上英文词时,
    至少要命中一个短语, 否则即使单词全中也不算数。
  - 单词命中 → 权重低, 只用来在已通过准入的论文里排序。

实测同一库: 准入后 88% -> 5%, 且活下来的确实是张量网络那几篇。

中文主题不含空格, 按原有分隔符切词即可, 不做短语准入(中文无空格分词不靠 n-gram)。
"""
from __future__ import annotations

import re

_SPLIT_RE = re.compile(r"[\s,，、。;；:：\-—/（）()\[\]【】]+")
_HAS_HAN = re.compile(r"[\u4e00-\u9fff]")


def split_topic(topic: str) -> tuple[list[str], list[str]]:
    """切主题 -> (单词列表, 短语列表)。短语由连续 2~3 个词组成。"""
    raw = [w.strip() for w in _SPLIT_RE.split(topic or "") if w.strip()]
    if not raw:
        return [], []

    # 中文主题: 不做 n-gram, 单字片段本身就是"短语"
    if _HAS_HAN.search(topic or ""):
        return raw, list(raw)

    words = [w.lower() for w in raw if len(w) >= 2]
    phrases: list[str] = []
    for n in (3, 2):
        for i in range(len(raw) - n + 1):
            p = " ".join(w.lower() for w in raw[i:i + n])
            if len(p) >= 4:
                phrases.append(p)
    # 去重且保序
    seen: set[str] = set()
    uniq = [p for p in phrases if not (p in seen or seen.add(p))]
    return words, uniq


def _norm(text: str) -> str:
    return re.sub(_SPLIT_RE, " ", (text or "").lower())


def _variants(phrase: str) -> list[str]:
    """短语的可匹配形式。

    主要解决单复数: 主题写 "tensor network states", 论文标题往往是
    "tensor network state correspondence" —— 不处理的话最核心的那个三元组
    直接不命中, 真论文反而被剪掉(实测里 12 篇里大部分是栽在这)。
    规则: 生成去尾 s / 去尾 es 的形式, 以及末词去 s 的形式。
    """
    out = [phrase]
    if phrase.endswith("s"):
        out.append(phrase[:-1])
    if phrase.endswith("es"):
        out.append(phrase[:-2])
    # 末词去 s ("network states" -> "network state")
    parts = phrase.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].endswith("s") and not parts[1].endswith("ss"):
        out.append(f"{parts[0]} {parts[1][:-1]}")
    return list(dict.fromkeys(out))


def _phrase_hit(phrase: str, hay: str) -> bool:
    return any(v in hay for v in _variants(phrase))


def match_score(topic: str, text: str) -> float:
    """主题-文本匹配分 0~1。仅在 qualifies() 为真时有意义。

    命中全部短语 -> 接近 1; 只命中单词 -> 低分(用来排序, 不用来准入)。
    """
    words, phrases = split_topic(topic)
    hay = _norm(text)
    if not hay or (not words and not phrases):
        return 0.0
    ph_hit = sum(1 for p in phrases if _phrase_hit(p, hay)) / len(phrases) if phrases else 0.0
    w_hit = sum(1 for w in words if w in hay) / len(words) if words else 0.0
    return round(0.7 * ph_hit + 0.3 * w_hit, 4)


def qualifies(topic: str, text: str) -> bool:
    """准入判定 —— 这是替代旧"任一单词命中"的关键一行。

    - 主题是中文(无空格词组): 保持旧行为, 任一词命中即合格
    - 主题是英文且能切出短语: **必须命中至少一个短语**
    - 主题只有一个词: 退化为词命中

    已知局限: 主题 "... tensor network states ..." 切出的 bigram "network states"
    会命中 "neural network states"(神经网络态)。**这是有意放过的** —— 神经网络态
    本就是张量网络的竞品方案, 综述里本来该拿来对照, 交给下游 LLM 审查判相关性,
    这里不必也不该在词法层面替它做判断。
    """
    words, phrases = split_topic(topic)
    if not words and not phrases:
        return True                       # 主题空 -> 不筛
    hay = _norm(text)
    if not hay:
        return False
    if _HAS_HAN.search(topic or "") or not phrases:
        return any(w in hay for w in words) or any(_phrase_hit(p, hay) for p in phrases)
    return any(_phrase_hit(p, hay) for p in phrases)


def entry_text(entry: dict) -> str:
    """pool entry -> 用于匹配的文本(标题权重最高, 放最前面不影响短语判定)。"""
    parts = [
        entry.get("title") or "",
        entry.get("core_method") or "",
        entry.get("core_contrib") or "",
        entry.get("abstract") or "",
        entry.get("keywords") or "",
        entry.get("outlook") or "",
    ]
    return " ".join(p for p in parts if p)
