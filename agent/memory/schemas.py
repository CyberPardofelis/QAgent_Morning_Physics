"""
agent.memory.schemas — 路径常量、记忆类型枚举、元数据 schema

所有路径相对 data/ 根目录解析（与 agent/infra/db.py 一致）。
"""
from pathlib import Path
from typing import Dict, Any


# ============================================================
# 路径常量（与 paper 向量库分开，独立存储）
# ============================================================
DATA_DIR = Path("data")
VECTOR_DIR = DATA_DIR / "vector_store"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# 记忆库 SQLite（通过 agent/infra/db.py 注册后用 connect("memory") 访问）
# 这里保留直接路径常量用于 vector_store 自管理 metadata pickle
MEMORY_FAISS_PATH = VECTOR_DIR / "memories.faiss"
MEMORY_META_PATH = VECTOR_DIR / "memories_meta.pkl"

# ============================================================
# 记忆类型（QAgent 任务场景定制，对照 WAgent 的 emotion/imagery/plot）
# ============================================================
MEMORY_TYPES: Dict[str, str] = {
    "user_preference": "用户的研究偏好/审美/格式习惯（'我喜欢看详细推导'、'输出别用 emoji'）",
    "research_fact":   "研究中的客观决策/约定/事实（'我决定用 VQE 路线'、'和导师约定周报格式'）",
    "paper_note":      "对某篇论文的笔记/摘录（关联 paper_id，例如 'Elben2022 的方法核心是...'）",
    "reflection":      "对话中产生的洞见/总结（'我发现 TEE 在 NISQ 上其实不实用'）",
    "task_state":      "进行中的任务状态（'上次卡在 QAOA 电路优化那一步'、'等 7/15 投稿结果'）",
    "general":         "其他通用记忆（兜底类型）",
}

# 当前默认 scope（v1 全部 = default；未来 per-function 时改这里）
DEFAULT_SCOPE = "default"

# BGE-large-en-v1.5 输出维度
# 注意：这是 QAgent 现有 Embedder 的维度。如果未来换 model 改这里
DEFAULT_EMBEDDING_DIM = 1024

# ============================================================
# 元数据 schema 校验
# ============================================================
def validate_metadata(metadata: dict) -> bool:
    """
    校验 SQLite 存 metadata JSON 字段的合法性。

    不强制要求字段，但要求是 dict 且可 JSON 序列化。
    """
    if not isinstance(metadata, dict):
        return False
    # 限制大小（10KB 内，避免 LLM 错误塞超大 JSON）
    import json
    try:
        s = json.dumps(metadata, ensure_ascii=False)
        if len(s) > 10240:
            return False
    except (TypeError, ValueError):
        return False
    return True


# ============================================================
# Trigger keywords（用于 prompts.py 的 MEMORY_GUIDE 提示 LLM）
# LLM 看到这些词时应该考虑调 save_memory 工具
# ============================================================
TRIGGER_KEYWORDS: tuple = (
    "记住", "记下", "别忘", "不要忘记",
    "我的偏好", "我喜欢", "我讨厌", "我不喜欢",
    "我的决定", "我决定", "约定", "说好了",
    "关于这篇论文", "这篇 paper",
    "我发现了", "我意识到", "我学到了", "明白了",
)


def content_has_trigger_keyword(text: str) -> bool:
    """
    检查文本是否包含触发关键词（用于 processor 写侧判断，
    也方便外部手动判断 LLM 是否应该调 save_memory）。

    简单子串匹配 — 不做分词、不区分大小写。
    """
    if not text:
        return False
    return any(kw in text for kw in TRIGGER_KEYWORDS)
