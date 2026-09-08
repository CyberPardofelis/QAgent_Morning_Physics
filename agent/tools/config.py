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