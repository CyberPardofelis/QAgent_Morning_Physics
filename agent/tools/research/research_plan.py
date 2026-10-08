"""
agent.tools.research.research_plan — 问题 → 可判定命题 (P7 阶段 2, L1)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §8 条款 3

这一层做的是**"能不能验"的前置筛选**, 不是"开始验"。
价值在于: 把用户随口一句"这个猜想靠谱吗"变成机器能动的原子命题,
或者干脆判定它不可判定并说清为什么 —— 后者同样是正当产物。

★ 不可判定的命题不许硬凑成可判定形式(§8 条款 3)。
   把开放问题包装成一个"看起来可判定"的断言, 是科研自动化里最贵的失败模式:
   它会一路跑到出报告才崩, 而且崩在用户已经相信结论的时候。

⚠️ exposed 默认 False (QAGENT_RESEARCH_EXPOSE=0)。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace, hash_question
from agent.prompts_research import build_modeling_prompt
from agent.tools.llm import _call_llm
from agent.tools.registry import register_tool

from ._claim import normalize_claim
from ._flags import RESEARCH_EXPOSED

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(raw: str) -> dict[str, Any]:
    """
    从 LLM 输出里抠 JSON。失败返回 {} —— 调用方据此如实报"建模失败",
    不要拿半截文本当结果。
    """
    if not raw:
        return {}
    m = _FENCE_RE.search(raw)
    text = (m.group(1) if m else raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退化路径: 抓最外层大括号(模型有时在 JSON 前后加解说)
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            pass
    return {}


@register_tool(name="research_plan", cost_tier="L1", exposed=RESEARCH_EXPOSED)
def research_plan(
    question: str,
    task_hash: str = "",
) -> dict[str, Any]:
    """
    把研究问题拆成机器可判定的原子命题。

    Args:
        question: 研究问题(自然语言)
        task_hash: 复用已有工作区; 不传按 question 自动生成

    Returns:
        {ok, specs: [{claim_text, scope, how, decidable, warn}], undecidable: [...],
         summary, workspace}
    """
    th = task_hash or hash_question(question)
    ws = ResearchWorkspace(th, question=question)
    ws.set_status(phase="plan", step=1, total=1, message="建模中")

    raw = _call_llm(
        build_modeling_prompt(question),
        temperature=0.3,      # 建模要稳
        max_tokens=3000,
        timeout=180,
    )
    data = _extract_json(raw)

    if not data:
        ws.set_status(phase="plan", step=1, total=1, message="建模失败")
        return {
            "ok": False, "specs": [], "undecidable": [],
            "summary": "建模失败: LLM 没吐出可解析的 JSON。请重试或把问题写得更具体"
                       "(含明确的变量与范围更容易拆出可判定命题)。",
            "raw_head": (raw or "")[:300],
            "workspace": str(ws.path),
            "cost": {"tokens": 1, "llm_calls": 1, "elapsed_ms": 0},
        }

    from ._loop import specs_from_planning

    specs, dropped = specs_from_planning(data, question=question)
    dec = [s for s in specs if s["decidable"]]
    und = [s for s in specs if not s["decidable"]]
    ws.save_plan({
        "specs": data.get("specs", []),
        "undecidable": data.get("undecidable", []),
        "normalized": specs,
    })

    summary_parts = [f"拆出 {len(dec)} 条可判定命题"]
    if dropped:
        summary_parts.append(f"去重 {dropped} 条")
    if und:
        summary_parts.append(f"{len(und)} 条判为不可自动验证(进'需人判断'区)")
    for s in dec:
        scope = s.get("scope") or "⚠ 未给范围"
        summary_parts.append(f"- [{s.get('how')}] {s['claim_text'][:48]} — 范围: {scope}")

    logger.info("[research_plan] %s -> %d 可判定 / %d 不可判定", question[:40], len(dec), len(und))
    return {
        "ok": True,
        "specs": specs,
        "n_decidable": len(dec),
        "undecidable": [s["claim_text"] for s in und],
        "summary": "; ".join(summary_parts[:1] + [""] + summary_parts[1:]),
        "workspace": str(ws.path),
        "cost": {"tokens": 1, "llm_calls": 1, "elapsed_ms": 0},
    }


def normalize_or_report(text: str, scope: str = "", how: str = "") -> dict[str, Any]:
    """给外部调用方用的小封装: 规范化一条命题并把 warning 提到明面上"""
    spec = normalize_claim(text, scope, how)
    return {
        "claim_id": spec["claim_id"],
        "decidable": spec["decidable"],
        "state": spec["state"],
        "warn": spec["warn"],
    }
