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
# ============================================================
# 重复检测阈值（可调）
# ============================================================
SIM_MERGE_THRESHOLD = 0.99    # 临时紧缩到0.99防对立观点被误合并 (2026-08-24 用户报 BGE 不能区分对立/同一事实)
SIM_SIMILAR_THRESHOLD = 0.75  # 0.75-0.95 判定为"类似"，提示用户判断


@register_tool(name="save_memory", is_streaming=False)
def save_memory(
    content: str,
    memory_type: str = "general",
    importance: float = 0.5,
    related_paper_id: str = "",
    metadata_json: str = "",
    force_save: bool = False,
) -> str:
    """保存一条长期记忆到记忆库（通用 scope，未来可按功能分组）。

    何时调用：
    - 用户明确说"记住…"、"记下…"、"别忘…"、"我的偏好是…"、"我决定…"、"约定…"
    - 用户表达了对自己有长期价值的信息（研究偏好、决策、对论文的笔记、洞见）

    重要：content 必须是**完整的自包含陈述**，不能是"是"这种简略词
    —— 未来 recall 检索时是按 content 做语义匹配，完整句才能搜到。

    重要：保存时会做**去重检测**：
    - 相似度 >= 0.95 → 视为重复，**不直接保存**，返回已有条目信息，请用 update_memory 工具合并
    - 相似度 0.75-0.95 → 视为类似，提示已存在类似条目，让 LLM 判断是否强制保存
    - 相似度 < 0.75 → 直接保存
    如确认要新建（不是合并），设 force_save=True。

    参数:
        content           记忆内容（必须完整陈述，含主语）
        memory_type       记忆类型
        importance        重要性 0-1
        related_paper_id  关联论文 ID（可空）
        metadata_json     额外元数据 JSON 字符串（可空）
        force_save        强制跳去去重检查直接保存（默认 False）
    """
    try:
        from agent.memory import get_db_manager
    except ImportError as e:
        return f"❌ 记忆库模块未加载: {e}"

    if not content or not content.strip():
        return "❌ content 不能为空"

    try:
        imp = float(importance)
    except (TypeError, ValueError):
        return f"❌ importance 必须是 0-1 之间的数字，实际: {importance}"
    if not (0.0 <= imp <= 1.0):
        return f"❌ importance 必须在 [0, 1]，实际: {imp}"

    valid_types = ("user_preference", "research_fact", "paper_note", "reflection", "task_state", "general")
    if memory_type not in valid_types:
        return f"❌ memory_type '{memory_type}' 不支持。" + chr(10) + f"支持: {', '.join(valid_types)}"

    if memory_type == "paper_note" and not related_paper_id:
        return "❌ paper_note 类型必须填 related_paper_id（论文 ID）"

    meta_dict = None
    if metadata_json and metadata_json.strip():
        try:
            meta_dict = json.loads(metadata_json)
            if not isinstance(meta_dict, dict):
                return "❌ metadata_json 必须是 JSON object（dict）"
        except json.JSONDecodeError as e:
            return f"❌ metadata_json 不是合法 JSON: {e}"

    try:
        from agent.memory.processor import MemoryProcessor
        p = MemoryProcessor()

        if not force_save:
            dup = p.check_duplicate(
                content.strip(),
                sim_merge=SIM_MERGE_THRESHOLD,
                sim_similar=SIM_SIMILAR_THRESHOLD,
                top_k=3,
            )
            verdict = dup.get("verdict", "unique")
            top_match = dup.get("top_match")
            all_matches = dup.get("all_matches", [])

            if verdict == "duplicate" and top_match:
                existing = top_match["memory"]
                sim = top_match["similarity"]
                paper_line = (chr(10) + f"- 关联论文: `{existing.get('related_paper_id')}`" if existing.get("related_paper_id") else "")
                return (
                    f"⚠️ **已存在高度相似记忆（sim={sim:.3f} >= {SIM_MERGE_THRESHOLD}）**" + chr(10) + chr(10) +
                    f"**已存在条目**:" + chr(10) +
                    f"- ID: `{existing['id'][:8]}`" + chr(10) +
                    f"- 类型: `{existing.get('memory_type', 'general')}`" + chr(10) +
                    f"- 重要性: {existing.get('importance', 0.5):.2f}" + chr(10) +
                    f"- 内容: {existing.get('content', '')[:200]}{'...' if len(existing.get('content', '')) > 200 else ''}" + chr(10) +
                    f"{paper_line}" + chr(10) +
                    f"- 创建于: {existing.get('created_at', '')}" + chr(10) + chr(10) +
                    f"**建议操作**:" + chr(10) +
                    f"1. 如果新内容覆盖了旧条目 → 调用 update_memory(existing_id, new_content) 合并" + chr(10) +
                    f"2. 如果确实是新主题（虽然相似）→ 再次调用 save_memory(..., force_save=True) 强制新建" + chr(10) +
                    f"3. 如果旧条目不再重要 → 用 viewer 删除它后重试"
                )
            if verdict == "opposite" and top_match:
                # 三元组识别出对立观点, 不要合并
                existing = top_match["memory"]
                judge_reason = top_match.get("judge_reason", "")
                return (
                    f"⚠️ **检测到对立观点（三元组判断）**" + chr(10) + chr(10) +
                    f"**已存在条目（与新内容矛盾）**:" + chr(10) +
                    f"- ID: `{existing['id'][:8]}`" + chr(10) +
                    f"- 类型: `{existing.get('memory_type', 'general')}`" + chr(10) +
                    f"- 内容: {existing.get('content', '')[:200]}{'...' if len(existing.get('content', '')) > 200 else ''}" + chr(10) + chr(10) +
                    f"**判断依据**: {judge_reason}" + chr(10) + chr(10) +
                    f"**不要合并** — 两条都是有效观点, 需要同时保留以跟踪演化." + chr(10) +
                    f"如果确认要覆盖旧观点，调用 update_memory 合并即可."
                )
            elif verdict == "similar" and all_matches:
                lines_list = [f"⚠️ **发现类似记忆（sim 0.75-0.95）**:" + chr(10)]
                for m in all_matches[:3]:
                    mm = m["memory"]
                    lines_list.append(
                        f"- `{mm['id'][:8]}` (sim={m['similarity']:.3f}): {mm.get('content', '')[:100]}"
                    )
                lines_list.append(chr(10) + "请判断：" + chr(10))
                lines_list.append("1. 合并到最像的条目 → update_memory(...)")
                lines_list.append("2. 真的是新内容 → 再次 save_memory(..., force_save=True)")
                return chr(10).join(lines_list)

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

    paper_line2 = (chr(10) + f"- 关联论文: `{related_paper_id}`" if related_paper_id else "")
    return (
        f"✅ 记忆已保存" + chr(10) + chr(10) +
        f"- ID: `{mid[:8]}`" + chr(10) +
        f"- 类型: `{memory_type}`" + chr(10) +
        f"- 重要性: {imp:.2f}" + chr(10) +
        f"- 内容: {content[:80]}{'...' if len(content) > 80 else ''}" + chr(10) +
        f"{paper_line2}" + chr(10) +
        chr(10) + f"未来对话中可被 recall_memories 检索到。"
    )


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


# ============================================================
# update_memory: 更新已有记忆
# ============================================================
@register_tool(name="update_memory", is_streaming=False)
def update_memory(
    memory_id: str,
    new_content: str = "",
    new_importance: float = -1.0,
) -> str:
    """更新一条已有记忆。用于去重时合并（覆盖旧内容）或调整重要性。

    参数:
        memory_id       已有记忆的 ID（8 位 short id 即可，函数会自动补全）
        new_content     新内容（覆盖原 content；空字符串表示不改）
        new_importance  新重要性 0-1；-1 表示不改
    """
    try:
        from agent.memory.processor import MemoryProcessor
        p = MemoryProcessor()
    except ImportError as e:
        return f"❌ 记忆库模块未加载: {e}"

    if not memory_id or not memory_id.strip():
        return "❌ memory_id 不能为空"

    if not new_content and new_importance < 0:
        return "❌ 至少传 new_content 或 new_importance 之一"

    full_id = memory_id.strip()
    if len(full_id) == 8:
        all_mems = p.list_all(limit=2000)
        match = next((m for m in all_mems if m["id"].startswith(full_id)), None)
        if not match:
            return f"❌ 找不到 id 前 8 位为 {full_id} 的记忆"
        full_id = match["id"]

    try:
        if new_content and new_content.strip():
            p.update_content(full_id, new_content.strip())
        if 0.0 <= new_importance <= 1.0:
            p.update_importance(full_id, new_importance)
        updated = p.db.get(full_id)
    except Exception as e:
        logger.exception("[update_memory] failed")
        return f"❌ 更新失败: {e}"

    if not updated:
        return f"❌ 找不到 id={full_id[:8]} 的记忆（可能已被删除）"

    paper_line3 = (chr(10) + f"- 关联论文: `{str(updated.get('related_paper_id', ''))}`" if updated.get("related_paper_id") else "")
    return (
        f"✅ 记忆已更新" + chr(10) + chr(10) +
        f"- ID: `{updated['id'][:8]}`" + chr(10) +
        f"- 类型: `{updated.get('memory_type', 'general')}`" + chr(10) +
        f"- 重要性: {updated.get('importance', 0.5):.2f}" + chr(10) +
        f"- 内容: {updated.get('content', '')[:200]}{'...' if len(updated.get('content', '')) > 200 else ''}" + chr(10) +
        f"{paper_line3}"
    )
