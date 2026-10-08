"""
agent.tools.research.research_report — 报告渲染 (P7 阶段 2, L0)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §8 条款 1

把 workspace 里已有的 specs + 证据渲染成 markdown。
**0 LLM** —— 报告是证据的投影, 不该有"模型觉得怎么说比较顺"这回事。

★ 证据绑定在这一层是**硬规则**: 拿不到证据的命题不会进结论区,
   会被归到"未验证区"。改渲染模板很容易绕开 prompt 的要求, 但绕不开这里。

⚠️ exposed 默认 False。
"""
from __future__ import annotations

import logging
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace
from agent.tools.registry import register_tool

from ._claim import render_report
from ._flags import RESEARCH_EXPOSED
from ._provenance import env_fingerprint

logger = logging.getLogger(__name__)


@register_tool(name="research_report", cost_tier="L0", exposed=RESEARCH_EXPOSED)
def research_report(
    task_hash: str,
    fmt: str = "md",
    save: bool = True,
) -> dict[str, Any]:
    """
    渲染某个任务的报告。

    Args:
        task_hash: workspace 的 task_hash (research_verify/investigate 的返回里有)
        fmt: md / json
        save: True = 同时把报告写回 workspace 的 _report.md

    Returns:
        md 格式: {ok, report, saved_path}
        json 格式: {ok, specs, evidence, env}
    """
    ws = ResearchWorkspace(task_hash)

    plan = ws.load_plan() or {}
    specs = plan.get("normalized") or []
    if not specs:
        # 兜底: 从 _specs.jsonl 重建(plan 丢了但 specs 还在的情况)
        for s in ws.load_specs():
            from ._claim import normalize_claim
            specs.append(normalize_claim(s.get("claim_text", ""), s.get("scope", ""),
                                         s.get("how", "")))
    evidence = ws.load_evidence()
    status = ws.load_status() or {}

    if not specs and not evidence:
        return {
            "ok": False,
            "summary": f"task_hash={task_hash} 的工作区是空的, 没有可渲染的内容。",
            "workspace": str(ws.path),
            "cost": {"tokens": 0, "elapsed_ms": 0},
        }

    md = render_report(
        specs, evidence,
        question=status.get("question") or plan.get("question", ""),
        env=env_fingerprint(),
    )

    saved = ""
    if save:
        ws.save_report(md)
        saved = str(ws.path / "_report.md")

    if fmt == "json":
        return {
            "ok": True,
            "specs": specs,
            "evidence": evidence,
            "env": env_fingerprint(),
            "summary": f"{len(specs)} 条命题 / {len(evidence)} 条证据",
            "saved_path": saved,
            "workspace": str(ws.path),
            "cost": {"tokens": 0, "elapsed_ms": 0},
        }

    logger.info("[research_report] %s -> %d 命题 / %d 证据", task_hash, len(specs), len(evidence))
    return {
        "ok": True,
        "report": md,
        "saved_path": saved,
        "summary": f"{len(specs)} 条命题 / {len(evidence)} 条证据",
        "workspace": str(ws.path),
        "cost": {"tokens": 0, "elapsed_ms": 0},
    }
