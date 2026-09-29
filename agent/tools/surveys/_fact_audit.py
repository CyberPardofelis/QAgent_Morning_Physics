"""
agent.tools.surveys._fact_audit — fact 落地审查 (9-29 新增)

要解决的问题 (实测):
    上一版综述里"离子阱 / 利用每个离子内的附加能级 / 无需辅助离子"这组细节,
    论文摘要里**根本没有** —— 原文只提了 ion trap 和 nonorthogonal,
    抽取器看到两个词就自行补全成一条完整机制。
    写作阶段只是照抄 fact, 所以无论写作 prompt 怎么约束都拦不住。
    **编造的源头在 fact 抽取阶段。**

为什么必须用 LLM 而不是确定性检查:
    实测 181 条真实 fact 的 claim 是**中文**、论文原文是**英文**,
    词面覆盖率中位数 0.021 —— 确定性 token 重叠检查对跨语言场景
    根本不成立(改用"只查英文术语"又会被残留标注污染, 且中文编造细节看不见)。
    唯一能真正判断"claim 有没有超出原文"的是语义审查。

成本 (可接受):
    每 8 条 fact 一次调用, 181 条 ≈ 23 次。相对抽取本身(每篇 1 次)是小头。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

AUDIT_PROMPT = """你是事实核查员。下面每条是某篇论文抽取出的"原子事实", 以及它声称的原文出处。

【任务】判断每条事实的 claim 是否**完全**有原文依据。

重点检查 claim 是否**补充了原文没有的细节**, 典型形态:
- 原文只提到某个平台(如 ion trap), claim 就自行补出该平台上的**具体实现机制**
  (如"利用离子内的附加能级"、"无需辅助离子") —— 原文没写就是编造
- 原文说"方法 A 与方法 B 相关", claim 就写成"通过 A 实现 B"这种因果
- 原文没有的数字、年份、人名、装置型号
- 原文是"我们提出 X", claim 把 X 的适用范围/效果夸大了

判定标准:
- claim 的**技术内容**必须有原文支持 → grounded: true
- claim 加入了原文没有的技术细节/因果/数字 → grounded: false, 并在 unsupported 里写清楚多了什么
- claim 只是把原文换了说法/语言(中文译述) → grounded: true

不要因为 claim 是中文、原文是英文就判 false。只看**内容**是否有依据。

【论文原文】{source}

【待核查事实】{facts}

【输出格式】严格 JSON 数组, 不要 markdown:
[
  {{"id": "<fact_id>", "grounded": true, "unsupported": ""}},
  {{"id": "<fact_id>", "grounded": false, "unsupported": "原文只提到 X, claim 自行补出 Y"}}
]

只输出 JSON, 覆盖每一条给定的 id。"""


def _audit_batch(facts: list[dict[str, Any]], source: str, provider: str, model: str) -> dict[str, dict]:
    """审查一批 fact, 返回 {fact_id: {"grounded": bool, "unsupported": str}}"""
    from agent.tools.llm import _call_llm

    lines = []
    for f in facts:
        quote = (f.get("support_quote") or "").strip()
        lines.append(
            f'- id={f.get("fact_id")}\n'
            f'  claim: {(f.get("claim") or "").strip()}\n'
            f'  出处原文: {quote[:400] or "(抽取器未给出)"}'
        )
    prompt = AUDIT_PROMPT.format(source=source[:4000] or "(无)", facts="\n".join(lines))

    try:
        out = _call_llm(prompt, provider=provider, model=model,
                        enable_thinking=False, temperature=0.1,
                        max_tokens=8000, timeout=300)
    except Exception as e:
        logger.warning("[fact_audit] batch failed, 保守放行: %s", e)
        return {}
    if not out:
        return {}
    if "<think>" in out:
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()

    i = out.find("[")
    if i < 0:
        return {}
    body = out[i:]
    for cand in (body, body + "]", body + '"]', body + '"]}', body + "]}"):
        try:
            v = json.loads(cand)
            if isinstance(v, list):
                break
            v = None
        except Exception:
            v = None
    if not isinstance(v, list) or not v:
        return {}

    res: dict[str, dict] = {}
    for item in v:
        if isinstance(item, dict) and item.get("id"):
            res[str(item["id"])] = {
                "grounded": bool(item.get("grounded", True)),
                "unsupported": str(item.get("unsupported") or ""),
            }
    return res


def audit_fact_grounding(
    facts: list[dict[str, Any]],
    paper_source: dict[str, str] | None = None,
    provider: str = "minimax",
    model: str = "MiniMax-M3",
    batch_size: int = 8,
) -> dict[str, Any]:
    """
    审查 fact 是否有原文支撑, 就地写回 _grounded / _unsupported_terms。

    审查不覆盖到的 fact 默认**保守放行**(_grounded=True) —— 审查器本身也会出错,
    不该因为审查器故障就丢掉本来可靠的事实; 真正要命的是"通过了审查但其实是编造",
    而那由 prompt 纪律 + 抽查兜底。

    Returns: 统计 dict {audited, grounded, ungrounded, failed}
    """
    if not facts:
        return {"audited": 0, "grounded": 0, "ungrounded": 0, "failed": 0}

    paper_source = paper_source or {}
    stats = {"audited": 0, "grounded": 0, "ungrounded": 0, "failed": 0}

    # 按论文分组(同一篇的 fact 一起审, 原文只需给一次)
    by_paper: dict[str, list[dict]] = {}
    for f in facts:
        by_paper.setdefault(f.get("paper_id", "?"), []).append(f)

    for _pid, group in by_paper.items():
        src = paper_source.get(_pid, "")
        for i in range(0, len(group), batch_size):
            batch = group[i:i + batch_size]
            got = _audit_batch(batch, src, provider, model)
            if not got:
                stats["failed"] += len(batch)
                continue
            for f in batch:
                stats["audited"] += 1
                r = got.get(f.get("fact_id"))
                if r is None:
                    stats["failed"] += 1
                    continue
                if r["grounded"]:
                    f["_grounded"] = True
                    stats["grounded"] += 1
                else:
                    f["_grounded"] = False
                    f["_unsupported_terms"] = [r["unsupported"]] if r["unsupported"] else ["审查未通过"]
                    stats["ungrounded"] += 1
    return stats
