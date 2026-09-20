"""
agent.tools.surveys.survey_audit — 独立审计工具 (L0, 0 LLM)

从原 surveys.py 抽出 (P1.7 重构):
- @register_tool survey_audit
- 对已有 survey 跑覆盖度审计 (不重生成, 0 LLM)
- 每章节候选池内/外引用 + 0 引用论文列表 + 候选池覆盖度
"""
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from agent.tools.registry import register_tool

from ._crud import get_theme, list_sections
from ._pool import _audit_section_paper_ids

logger = logging.getLogger(__name__)


@register_tool(name="survey_audit", is_streaming=False, cost_tier="L0")
def survey_audit(theme_id: int, topic: str = "") -> str:
    """
    对已有 survey 跑覆盖度审计 (P6.4, 0 LLM, L0 档)。

    不重生成章节, 只对 surveys.db 已落库的 theme + sections 跑:
      - 每章节候选池内/外引用统计
      - 总引用次数 / 唯一 paper_id 数
      - 候选池 0 引用列表 (从 workspace _pool.json 读, 找不到则跳过)
      - 候选池外引用 (幻觉风险) 列表

    Args:
        theme_id: survey_theme.id
        topic: 可选, 用来从 workspace data/surveys/working/{hash}/_pool.json 找候选池
               不传则只跑章节级审计 (无候选池覆盖度)

    Returns:
        markdown 审计报告
    """
    # 1. 读 theme
    theme = get_theme(theme_id)
    if not theme:
        return f"❌ 找不到 theme_id={theme_id} 的 survey"

    # 2. 读 sections
    secs = list_sections(theme_id)
    if not secs:
        return f"❌ theme_id={theme_id} 没任何 section"

    # 3. 找候选池 (从 workspace _pool.json)
    pool: list[dict[str, Any]] = []
    pool_topic = topic or theme.get("name", "")
    if pool_topic:
        from agent.infra.survey_workspace import hash_topic
        ws_path = Path("data/surveys/working") / hash_topic(pool_topic)
        pool_file = ws_path / "_pool.json"
        if pool_file.exists():
            with contextlib.suppress(Exception):
                pool = json.loads(pool_file.read_text(encoding="utf-8")).get("papers", [])

    pool_ids = {p["paper_id"] for p in pool}

    # 4. 每章节 audit
    section_audit = []
    all_in_pool: set = set()
    all_out_pool: set = set()
    for sec in secs:
        content = sec.get("content_md", "")
        bucket = _audit_section_paper_ids(content, pool_ids)
        section_audit.append({
            "section_id": sec.get("id"),
            "title": sec.get("title", ""),
            "type": sec.get("section_type", ""),
            "in_pool": bucket["in_pool"],
            "out_of_pool": bucket["out_of_pool"],
        })
        all_in_pool.update(bucket["in_pool"])
        all_out_pool.update(bucket["out_of_pool"])

    # 5. 报告
    report = [
        f"## survey_audit — theme_id={theme_id}",
        "",
        f"- topic: {theme.get('name', '?')}",
        f"- description: {theme.get('description', '')[:100]}",
        f"- 章节数: {len(secs)}",
        f"- 候选池: {len(pool)} 篇" + ("" if pool else " (没找到 workspace, 候选池覆盖度审计跳过)"),
        "",
        "## 每章节引用统计",
        "",
        "| section_id | 章节 | 类型 | 候选池内 | 候选池外(幻觉) |",
        "|---|---|---|---|---|",
    ]
    for r in section_audit:
        report.append(
            f"| {r['section_id']} | {r['title'][:40]} | {r['type']} | "
            f"{len(r['in_pool'])} | {len(r['out_of_pool'])} |"
        )

    report.extend([
        "",
        "## 总计",
        f"- 唯一 paper_id 引用: {len(all_in_pool)} 个 (候选池内), {len(all_out_pool)} 个 (候选池外, 幻觉风险)",
    ])

    if pool:
        uncited = pool_ids - all_in_pool
        coverage = (len(all_in_pool) * 100 // max(1, len(pool_ids)))
        report.extend([
            f"- 候选池覆盖度: {coverage}% ({len(all_in_pool)} / {len(pool_ids)})",
            "",
            f"## ⚠️ 候选池 0 引用论文 ({len(uncited)} / {len(pool_ids)}):",
            "",
        ])
        for pid in sorted(uncited):
            p = next((x for x in pool if x["paper_id"] == pid), None)
            title = (p or {}).get("title", "?")[:80]
            report.append(f"- `[[{pid}]]`: {title}")

    if all_out_pool:
        report.extend([
            "",
            f"## ❌ 候选池外引用 (幻觉风险) ({len(all_out_pool)} 个):",
            "",
        ])
        for pid in sorted(all_out_pool):
            report.append(f"- `[[{pid}]]`")

    return "\n".join(report)
