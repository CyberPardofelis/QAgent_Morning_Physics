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

【任务】为每条事实找出**支撑它的原文英文片段**, 并判断该 claim 是否超出原文。

【关键 — 必须逐字引用, 不能意译】
对每条 claim, 找出论文原文中**最具体、最能支撑它的那几个英文单词**, 原样抄进
`evidence` 字段。只能抄论文原文里真实存在的英文, **不许翻译、不许改写、不许拼接**。

这个设计是为了防止审查器自己产生幻觉: 抄不出来的, 就说明 claim 超出原文。

【判定】
- `grounded: true` 且 evidence 抄得出 -> claim 有据
- `grounded: false` -> 以下任一情况:
  a) 抄不出支撑片段(说明 claim 的技术内容原文没有)
  b) 原文只提到某平台/现象, claim 自行补出了**具体实现机制**。
     例: 原文有 "ion trap" 和 "nonorthogonal measurements", claim 写
     "利用每个离子内的附加能级实现该非正交测量" —— 原文根本没提"附加能级", 是编的。
  c) 原文说 A 与 B 相关, claim 写成"通过 A 实现 B"这种因果
  d) claim 有原文没有的数字/年份/装置型号
  e) claim 夸大了原文的适用范围或效果

- claim 只是把原文换成中文说法 -> grounded: true, evidence 抄原文里对应的英文词组

【论文原文】{source}

【待核查事实】{facts}

【输出格式】严格 JSON 数组, 不要 markdown:
[
  {{"id": "<fact_id>", "grounded": true, "evidence": "<原文里逐字抄的英文词组>"}},
  {{"id": "<fact_id>", "grounded": false, "evidence": "", "unsupported": "原文只提到 X, claim 自行补出 Y"}}
]

只输出 JSON, 覆盖每一条给定的 id。evidence 必须是**原文中逐字存在**的字符串。"""


def _norm_for_match(s: str) -> str:
    """归一化以便把引用片段和原文比对: 小写 + 只留字母数字空格。"""
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def _verify_evidence(evidence: str, source: str) -> bool:
    """
    引用片段是否真的在原文里。

    这是审查环节的**机械校验** —— 光靠 LLM 说"grounded: true" 是不够的,
    实测它会把"利用每个离子内的附加能级"这种明显编造放过去(因为源文确实有
    ion trap / nonorthogonal 两个词, 审查器看到就信了)。
    强制它逐字引用, 引不出来就等于没依据。
    """
    ev = _norm_for_match(evidence)
    if not ev:
        return False
    src = _norm_for_match(source)
    if ev in src:
        return True
    # 退一步: 逐个有内容的词都在原文里(容忍语序/标点差异)
    words = [w for w in ev.split() if len(w) > 2]
    if not words:
        return False
    return all(w in src for w in words)


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
                "evidence": str(item.get("evidence") or ""),
                "unsupported": str(item.get("unsupported") or ""),
            }
    return res


def audit_fact_grounding(
    facts: list[dict[str, Any]],
    paper_source: dict[str, str] | None = None,
    provider: str = "minimax",
    model: str = "MiniMax-M3",
    # 9-30 user 拍板: 8 -> 20。345 条 fact 按 8 一批要 43 次调用,
    # 两道 fact 审查加起来占整轮 42%。提到 20 就是 18 次, 质量不受影响 ——
    # 每条 fact 判定的输入没变, 只是塞进更长的 prompt。
    batch_size: int = 20,
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
                # 机械校验: 审查器说 grounded 还不够, 引用的原文片段必须真的在原文里。
                # 实测纯靠 LLM 判断会把明显编造放过去(源文恰有 ion trap/nonorthogonal,
                # 审查器看到就信, 忽略了 claim 自行补出的"离子附加能级")。
                if r["grounded"] and src:
                    if not _verify_evidence(r.get("evidence", ""), src):
                        r["grounded"] = False
                        r["unsupported"] = (
                            r.get("unsupported")
                            or f"引用片段在原文中找不到: {r.get('evidence','')[:60]!r}"
                        )
                if r["grounded"]:
                    f["_grounded"] = True
                    f["_evidence"] = r.get("evidence", "")
                    stats["grounded"] += 1
                else:
                    f["_grounded"] = False
                    f["_unsupported_terms"] = [r["unsupported"]] if r["unsupported"] else ["审查未通过"]
                    stats["ungrounded"] += 1
    return stats
