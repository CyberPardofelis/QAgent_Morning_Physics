"""
agent.tools.memory — 通用记忆库的 LLM 工具

两个工具：
- save_memory(content, memory_type?, importance?, related_paper_id?, scope?)
  → 显式保存一条记忆。LLM 看到用户说"记住"/"我的偏好"/"我决定"等时调用
- recall_memories(query, limit?, memory_type?, related_paper_id?, min_importance?)
  → 检索相关记忆。LLM 觉得需要"翻一下老黄历"时调用
"""
import json
import logging
from typing import Optional

from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)


# ============================================================
# save_memory: 保存一条记忆
# ============================================================
@register_tool(name="save_memory", is_streaming=False)
def save_memory(
    content: str,
    memory_type: str = "general",
    importance: float = 0.5,
    related_paper_id: str = "",
    metadata_json: str = "",
) -> str:
    """
    保存一条长期记忆到记忆库（通用 scope，未来可按功能分组）。

    何时调用：
    - 用户明确说"记住…"、"记下…"、"别忘…"、"我的偏好是…"、"我决定…"、"约定…"
    - 用户表达了对自己有长期价值的信息（研究偏好、决策、对论文的笔记、洞见）

    重要：content 必须是**完整的自包含陈述**，不能是"我的偏好"这种简略词
    —— 未来 recall 检索时是按 content 做语义匹配，完整句才能搜到。
    错误示例: content="是"
    正确示例: content="用户偏好 LaTeX 公式用 \\begin{aligned} 块排版"

    参数:
        content           记忆内容（必须完整陈述，含主语）
        memory_type       记忆类型: user_preference / research_fact / paper_note / reflection / task_state / general
        importance        重要性 0-1，默认 0.5（普通事实）；用户的明确偏好/决策建议 0.7-0.9
        related_paper_id  关联论文 ID（可空；如果是 paper_note 必须填）
        metadata_json     额外元数据 JSON 字符串（可空），例：'{"topic": "QAOA"}'

    返回:
        Markdown 格式的保存结果
    """
    try:
        from agent.memory import get_db_manager  # lazy import
    except ImportError as e:
        return f"❌ 记忆库模块未加载: {e}"

    if not content or not content.strip():
        return "❌ content 不能为空"

    # 校验 importance
    try:
        imp = float(importance)
    except (TypeError, ValueError):
        return f"❌ importance 必须是 0-1 之间的数字，实际: {importance}"
    if not (0.0 <= imp <= 1.0):
        return f"❌ importance 必须在 [0, 1]，实际: {imp}"

    # 校验 memory_type
    valid_types = ("user_preference", "research_fact", "paper_note", "reflection", "task_state", "general")
    if memory_type not in valid_types:
        return (
            f"❌ memory_type '{memory_type}' 不支持。\n"
            f"支持: {', '.join(valid_types)}"
        )

    # 校验 paper_note 必须填 related_paper_id
    if memory_type == "paper_note" and not related_paper_id:
        return "❌ paper_note 类型必须填 related_paper_id（论文 ID）"

    # 解析 metadata
    meta_dict = None
    if metadata_json and metadata_json.strip():
        try:
            meta_dict = json.loads(metadata_json)
            if not isinstance(meta_dict, dict):
                return "❌ metadata_json 必须是 JSON object（dict），不是 list 或基本类型"
        except json.JSONDecodeError as e:
            return f"❌ metadata_json 不是合法 JSON: {e}"

    # 调 processor
    try:
        from agent.memory.processor import MemoryProcessor
        p = MemoryProcessor()
        mid = p.save(
            content=content.strip(),
            memory_type=memory_type,
            importance=imp,
            related_paper_id=related_paper_id.strip() if related_paper_id else None,
            metadata=meta_dict,
        )
    except Exception as e:
        logger.exception("[save_memory] failed")
        return f"❌ 保存失败: {e}"

    return (
        f"✅ 记忆已保存\n\n"
        f"- ID: `{mid[:8]}`\n"
        f"- 类型: `{memory_type}`\n"
        f"- 重要性: {imp:.2f}\n"
        f"- 内容: {content[:80]}{'...' if len(content) > 80 else ''}\n"
        f"{f'- 关联论文: `{related_paper_id}`' if related_paper_id else ''}\n"
        f"\n未来对话中可被 `recall_memories` 检索到。"
    )


# ============================================================
# recall_memories: 检索记忆
# ============================================================
@register_tool(name="recall_memories", is_streaming=False)
def recall_memories(
    query: str,
    limit: int = 5,
    memory_type: str = "",
    related_paper_id: str = "",
    min_importance: float = 0.0,
    method: str = "vector",
) -> str:
    """
    从记忆库检索相关记忆。

    何时调用：
    - 用户说"我之前说过…"、"上次我们讨论过…"、"我对 XX 的偏好是…" → 想确认/复用
    - 上下文看起来需要长期记忆时（用户引用过去决策、对某话题有延续讨论）
    - 工具 `query_knowledge` 查的是论文，**这个工具查的是用户偏好/决策/笔记**

    参数:
        query              检索文本（自然语言，问题/关键词均可）
        limit              最多返回条数（默认 5）
        memory_type        限定类型: user_preference / research_fact / paper_note / reflection / task_state（可空）
        related_paper_id   限定关联论文（可空；如果填了就走精确匹配不走向量）
        min_importance     最低重要性 0-1（默认 0）
        method             检索方法: vector / keyword / hybrid（默认 vector）

    返回:
        Markdown 格式的记忆列表
    """
    try:
        from agent.memory.processor import MemoryProcessor
    except ImportError as e:
        return f"❌ 记忆库模块未加载: {e}"

    if not query or not query.strip():
        return "❌ query 不能为空"

    try:
        lim = int(limit)
    except (TypeError, ValueError):
        return f"❌ limit 必须是整数，实际: {limit}"
    lim = max(1, min(lim, 50))  # 夹到 [1, 50]

    try:
        mi = float(min_importance) if min_importance else 0.0
    except (TypeError, ValueError):
        return f"❌ min_importance 必须是数字，实际: {min_importance}"

    if method not in ("vector", "keyword", "hybrid"):
        method = "vector"

    try:
        p = MemoryProcessor()
        memories = p.recall(
            query=query.strip(),
            limit=lim,
            memory_type=memory_type.strip() if memory_type else None,
            related_paper_id=related_paper_id.strip() if related_paper_id else None,
            min_importance=mi if mi > 0 else None,
            method=method,
        )
    except Exception as e:
        logger.exception("[recall_memories] failed")
        return f"❌ 检索失败: {e}"

    if not memories:
        return f"🔍 未找到与「{query}」相关的记忆。"

    lines = [f"🔍 找到 {len(memories)} 条相关记忆：\n"]
    for i, m in enumerate(memories, 1):
        ctype = m.get("memory_type") or "general"
        imp = m.get("importance", 0.5)
        sim = m.get("_similarity", 0.0)
        method_used = m.get("_method", "")
        content = m.get("content", "")
        rid = m.get("id", "")[:8]
        paper = m.get("related_paper_id")

        score_str = ""
        if sim:
            score_str = f" (相关度 {sim:.2f})"
        elif method_used == "paper_id_exact":
            score_str = " (精确匹配)"

        lines.append(f"**{i}. [{ctype}]** {content}{score_str}")
        meta_bits = [f"重要性 {imp:.2f}"]
        if paper:
            meta_bits.append(f"论文 `{paper}`")
        meta_bits.append(f"ID `{rid}…`")
        lines.append(f"   <sub>{' · '.join(meta_bits)}</sub>\n")

    return "\n".join(lines)
