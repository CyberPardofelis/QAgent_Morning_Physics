"""
knowledge._scan — 拆出自原 knowledge.py (9-20 体检重构)

包含函数: scan_local_papers, migrate_json_to_db
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from agent.tools import config as _config
from agent.tools.config import KNOWLEDGE_DIR, PAPER_DIR  # noqa: F401
from agent.tools.database import get_db_connection, save_paper_record
from agent.tools.registry import register_tool

from ._analyze import analyze_paper

logger = logging.getLogger(__name__)
@register_tool(name="scan_local_papers", is_streaming=False)
def scan_local_papers() -> str:
    """扫描 PAPER_DIR，分析新出现的 PDF（增量更新）"""
    conn = get_db_connection()
    existing_ids = {row['paper_id'] for row in conn.execute("SELECT paper_id FROM papers").fetchall()}
    conn.close()

    new_papers = []
    skipped = 0
    for pdf_file in _config.PAPER_DIR.glob("*.pdf"):
        paper_id = pdf_file.stem
        if paper_id in existing_ids:
            skipped += 1
            continue
        result = analyze_paper(paper_id, pdf_path=str(pdf_file))
        if result.startswith("✅"):
            new_papers.append(paper_id)
        else:
            return f"扫描中止：分析 {paper_id} 失败\n{result}"

    if not new_papers and not skipped:
        return "扫描完成：本地目录中暂无 PDF 文件。"
    if not new_papers:
        return f"扫描完成：跳过已分析的 {skipped} 篇，没有新论文。"

    lines = [f"扫描完成：新增分析 {len(new_papers)} 篇（跳过 {skipped} 篇）：\n"]
    for pid in new_papers:
        lines.append(f"- `{pid}`")
    return "\n".join(lines)


def migrate_json_to_db():
    """将 KNOWLEDGE_DIR 下所有 .json 文件（除了 index.json）导入数据库"""
    for json_file in _config.KNOWLEDGE_DIR.glob("*.json"):
        if json_file.name == "index.json":
            continue
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        if "paper_id" not in data:
            data["paper_id"] = json_file.stem
        if "analyzed_at" not in data:
            data["analyzed_at"] = datetime.now().isoformat()
        save_paper_record(data)
