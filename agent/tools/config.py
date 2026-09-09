import os
from pathlib import Path
from dotenv import load_dotenv

# 关键：模块加载时触发 .env 读取，否则模块级常量拿不到 .env 里的 key
# （注意：这只在第一次 import 时执行；后续 .env 修改需要重启进程才能生效）
load_dotenv()

# 路径配置
DATA_DIR = Path("data")
PAPER_DIR = DATA_DIR / "papers"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
INDEX_FILE = KNOWLEDGE_DIR / "index.json"
VECTOR_DIR = "vector_data"  # 向量索引存放路径
DB_PATH = KNOWLEDGE_DIR / "papers.db"

# 确保目录存在
os.makedirs(PAPER_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# 注意：
# - DEEPSEEK_API_KEY 在这里用 .env 兜底
# - 工具内部实际调 LLM 时（_call_llm），优先用 agent.infra.api_keys.get_api_key("deepseek")
#   （走 3 层兜底：.api_keys.json > QSettings > .env），保证改 key 后立即生效
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
EMBEDDING_ENABLED = os.getenv("EMBEDDING_ENABLED", "false").lower() in ("true", "1", "yes")


# ============================================================
# Survey 工具: 候选池最小论文数 (2026-09-08 抽 env)
# ============================================================
# 之前硬编码 30 (9-08 17:00 user 拍板), 现统一读 env, 改阈值不用动代码。
# 默认 30 保持向后兼容; user 想临时调大就改 .env / export。
#
#   - 未设 / 空字符串 / 非整数 / < 1 → 默认 30 (跟历史行为一致)
#   - 设了合法正整数                  → 用环境变量值
#
# 注意: load_dotenv() 在文件顶部, 这里再 getenv 能拿到 .env 里的值。
#      但 .env 修改需重启进程 (load_dotenv 注释说"只在第一次 import 时执行")。
def _resolve_survey_min_pool() -> int:
    raw = os.getenv("QAGENT_SURVEY_MIN_POOL", "").strip()
    if not raw:
        return 30
    try:
        v = int(raw)
        return v if v >= 1 else 30
    except ValueError:
        import logging
        logging.getLogger(__name__).warning(
            "[config] QAGENT_SURVEY_MIN_POOL=%r 不是合法整数, 走默认 30", raw,
        )
        return 30


SURVEY_MIN_POOL_PAPERS = _resolve_survey_min_pool()


def get_min_pool_papers() -> int:
    """候选池最小论文数 (survey 工具统一阈值). 见 SURVEY_MIN_POOL_PAPERS 注释."""
    return SURVEY_MIN_POOL_PAPERS


# ============================================================
# Survey 工具: 章节/大纲 LLM 输出 max_tokens (2026-09-08 加)
# ============================================================
# 9-08 23:50 user 拍板: 从 10000 放宽到 20000, 配合 outline/section 视角改造
# (claim-centric 视角下章节更长、sub_points 5-8 个, 10000 容易截断)
#
# 抽 env, 改阈值不用动代码:
#   - 未设 / 空字符串 / 非整数 / < 1000 → 默认 20000
#   - 设了合法正整数 (>= 1000)         → 用环境变量值
def _resolve_survey_max_section_tokens() -> int:
    raw = os.getenv("QAGENT_SURVEY_SECTION_MAX_TOKENS", "").strip()
    if not raw:
        return 20000
    try:
        v = int(raw)
        return v if v >= 1000 else 20000
    except ValueError:
        import logging
        logging.getLogger(__name__).warning(
            "[config] QAGENT_SURVEY_SECTION_MAX_TOKENS=%r 不是合法整数, 走默认 20000", raw,
        )
        return 20000


SURVEY_MAX_SECTION_TOKENS = _resolve_survey_max_section_tokens()


def get_max_section_tokens() -> int:
    """章节/大纲 LLM 输出 max_tokens (survey 工具统一阈值).
    见 SURVEY_MAX_SECTION_TOKENS 注释."""
    return SURVEY_MAX_SECTION_TOKENS


# ============================================================
# Survey 工具: 候选池相关性质量控制 (2026-09-09 加)
# ============================================================
# 用户实测反馈: 量子态验证综述里召回了 30+ 篇层析/密钥分发论文, 主题边界不清。
# 解决: 三道防线 (按"召回→粗筛→精筛" 顺序, 任一道独立可用, 一起效果最好)
#   1. chroma 召回时用相似度阈值 (0.97) 代替 top_k, 太远的不进 pool
#   2. recall 完后对 pool 跑 LLM per-paper 相关性评分 (0-10), 阈值踢出
#   3. arxiv 兜底可走 batch 模式 (多 query 召回 union/dedup), 减少单一 query 误召
#
# 3 个 env 全部抽参, user 觉得阈值太严/太松, 改 env 不用动代码。

def _resolve_survey_relevance_threshold() -> int:
    raw = os.getenv("QAGENT_SURVEY_RELEVANCE_THRESHOLD", "").strip()
    if not raw:
        return 7
    try:
        v = int(raw)
        # 0 = 不启用 relevance filter; 1-10 = 阈值, <7 太严, >8 太松
        return v if 0 <= v <= 10 else 7
    except ValueError:
        import logging
        logging.getLogger(__name__).warning(
            "[config] QAGENT_SURVEY_RELEVANCE_THRESHOLD=%r 不是合法整数, 走默认 7", raw,
        )
        return 7


SURVEY_RELEVANCE_THRESHOLD = _resolve_survey_relevance_threshold()


def get_relevance_threshold() -> int:
    """per-paper 相关性评分阈值 (0=禁用, 1-10). 见 SURVEY_RELEVANCE_THRESHOLD 注释."""
    return SURVEY_RELEVANCE_THRESHOLD


def _resolve_survey_chroma_sim_threshold() -> float:
    raw = os.getenv("QAGENT_SURVEY_CHROMA_SIM_THRESHOLD", "").strip()
    if not raw:
        return 0.97
    try:
        v = float(raw)
        # 0.0-1.0 之外的值视为非法; 0.95-0.99 是常用区间
        return v if 0.0 <= v <= 1.0 else 0.97
    except ValueError:
        import logging
        logging.getLogger(__name__).warning(
            "[config] QAGENT_SURVEY_CHROMA_SIM_THRESHOLD=%r 不是合法浮点, 走默认 0.97", raw,
        )
        return 0.97


SURVEY_CHROMA_SIM_THRESHOLD = _resolve_survey_chroma_sim_threshold()


def get_chroma_sim_threshold() -> float:
    """chroma 召回相似度阈值 (cosine similarity, 0.0-1.0). 见 SURVEY_CHROMA_SIM_THRESHOLD 注释."""
    return SURVEY_CHROMA_SIM_THRESHOLD


def _resolve_survey_relevance_batch_size() -> int:
    raw = os.getenv("QAGENT_SURVEY_RELEVANCE_BATCH_SIZE", "").strip()
    if not raw:
        return 5
    try:
        v = int(raw)
        return v if 1 <= v <= 20 else 5
    except ValueError:
        import logging
        logging.getLogger(__name__).warning(
            "[config] QAGENT_SURVEY_RELEVANCE_BATCH_SIZE=%r 不是合法整数, 走默认 5", raw,
        )
        return 5


SURVEY_RELEVANCE_BATCH_SIZE = _resolve_survey_relevance_batch_size()


def get_relevance_batch_size() -> int:
    """relevance 评分每批论文数 (默认 5). 见 SURVEY_RELEVANCE_BATCH_SIZE 注释."""
    return SURVEY_RELEVANCE_BATCH_SIZE