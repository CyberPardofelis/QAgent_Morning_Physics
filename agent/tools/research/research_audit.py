"""
agent.tools.research.research_audit — 证据链审计 (P7 阶段 2, L0)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §0 / §8

============================ 为什么必须有它 ================================
§0 说得很直白: "LLM 说'我验证过了'不算数"。
P0/P1 的判定纪律(VERDICT 标记、枚举不许报 proved)是**执行期**的约束。
但报告出来之后, 还有一类翻箱是没被挡住的:

  - 同一命题先证成后被反驳 —— 环境变了?命题写错了?
  - 报告里写 PROVED, 但磁盘证据里根本没有对应记录
  - proved 判定只有 citation 类证据(拿文献当证明)
  - 结论行没挂 ev_ 编号(裸断言)
  - 证据指向了不存在的命题(孤儿)

这一层**不调 LLM**: 它只读磁盘上的 specs + evidence + report 三样东西,
不看 LLM 说了什么。这正是它能当"硬约束"的原因 —— prompt 是软约束, 审计是硬约束。

⚠️ exposed 默认 False。
"""
from __future__ import annotations

import logging
from typing import Any

from agent.infra.research_workspace import ResearchWorkspace
from agent.tools.registry import register_tool

from ._claim import audit_claims, normalize_claim
from ._flags import RESEARCH_EXPOSED

logger = logging.getLogger(__name__)


@register_tool(name="research_audit", cost_tier="L0", exposed=RESEARCH_EXPOSED)
def research_audit(
    task_hash: str = "",
    task_hash_list: str = "",
) -> dict[str, Any]:
    """
    对已完成任务跑证据链审计(0 LLM)。

    Args:
        task_hash: 单个任务(留空则审计 _specs 里的全部)
        task_hash_list: 逗号分隔的多个 task_hash(审计"结论翻车"的推荐用法)

    Returns:
        {ok, issues[], n_errors, n_warnings, summary, audited}
    """
    hashes = [h.strip() for h in (task_hash_list or task_hash or "").split(",") if h.strip()]

    if not hashes:
        from agent.infra.research_workspace import list_workspaces
        hashes = [w["task_hash"] for w in list_workspaces()[:5]]
        if not hashes:
            return {
                "ok": False, "issues": [], "n_errors": 0, "n_warnings": 0,
                "summary": "没有任何可审计的任务 —— 先跑 research_verify 或 research_investigate。",
                "cost": {"tokens": 0, "elapsed_ms": 0},
            }

    all_issues: list[dict[str, Any]] = []
    n_claims = n_ev = 0
    per_task: list[dict[str, Any]] = []

    for h in hashes:
        ws = ResearchWorkspace(h)
        specs = _load_specs(ws)
        evidence = ws.load_evidence()
        report = ws.load_report() or ""
        if not specs and not evidence:
            continue

        r = audit_claims(specs, evidence, report_md=report)
        n_claims += r["n_claims"]
        n_ev += r["n_evidence"]
        per_task.append({
            "task_hash": h,
            "ok": r["ok"],
            "n_claims": r["n_claims"],
            "n_evidence": r["n_evidence"],
        })
        for it in r["issues"]:
            all_issues.append({**it, "task_hash": h})

    n_err = sum(1 for i in all_issues if i["level"] == "error")
    n_warn = sum(1 for i in all_issues if i["level"] == "warn")

    if not per_task:
        summary = f"审计了 {len(hashes)} 个 task_hash, 都没有内容(空工作区)"
    elif n_err:
        summary = (
            f"🔴 发现 {n_err} 处硬伤 / {n_warn} 处警告 "
            f"(跨 {len(per_task)} 个任务, {n_claims} 条命题, {n_ev} 条证据)。"
            f"**报告结论在修复前不可信。**"
        )
    else:
        summary = (
            f"✅ 审计通过: {len(per_task)} 个任务 / {n_claims} 条命题 / {n_ev} 条证据, "
            f"无硬伤{n_warn and f'(有 {n_warn} 处提醒)' or ''}"
        )

    logger.info("[research_audit] %s", summary)
    return {
        "ok": n_err == 0,
        "issues": all_issues,
        "n_errors": n_err,
        "n_warnings": n_warn,
        "per_task": per_task,
        "audited": [p["task_hash"] for p in per_task],
        "summary": summary,
        "cost": {"tokens": 0, "elapsed_ms": 0},
    }


def _load_specs(ws: ResearchWorkspace) -> list[dict[str, Any]]:
    """specs 优先取 _plan.json 的规范化版本, 没有就从 _specs.jsonl 重建"""
    plan = ws.load_plan() or {}
    normalized = plan.get("normalized") or []
    if normalized:
        return normalized
    out = []
    for s in ws.load_specs():
        out.append(normalize_claim(
            s.get("claim_text", ""), s.get("scope", ""), s.get("how", ""),
            claim_id=s.get("claim_id"),      # 沿用原 id, 别重算(否则证据成孤儿)
        ))
    return out
