"""
API Key 存储管理（参考 WAgent 实现）。

设计目标：
- 不再用 QSettings（注册表）明文存储 API key
- 存到独立文件 data/.api_keys.json（点文件，git 默认忽略，不显眼）
- 自动迁移：检测到 QSettings 里有旧 key 时，第一次启动自动搬到 .api_keys.json
- 简单可读，不引入加密依赖（避免动 venv）

注意：这是"不显眼的安全"，不是真加密。
真要防本地读取：上 keyring（系统凭据管理器）或 cryptography.fernet 加密文件。
但代价是引入新依赖、用户体验更复杂。当前方案对"不在共享机器"的个人用户够用。
"""
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 文件路径
# ============================================================
DATA_DIR = Path("data")
KEYS_FILE = DATA_DIR / ".api_keys.json"

# 迁移来源（QSettings 里的旧 key）
LEGACY_SETTINGS_KEYS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}


# ============================================================
# 读写（带线程锁）
# ============================================================
_lock = threading.RLock()  # 可重入锁：set_api_key 内部 _save_to_disk + _invalidate_cache 会嵌套
_cache: Optional[Dict[str, str]] = None


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_from_disk() -> Dict[str, str]:
    """从 .api_keys.json 读所有 key（文件不存在返回空 dict）"""
    _ensure_data_dir()
    if not KEYS_FILE.exists():
        return {}
    try:
        with open(KEYS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("%s 内容不是 dict，重置为空", KEYS_FILE)
            return {}
        # 过滤非字符串值
        return {k: v for k, v in data.items() if isinstance(v, str)}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("读取 %s 失败: %s（视为空）", KEYS_FILE, e)
        return {}


def _save_to_disk(keys: Dict[str, str]) -> None:
    _ensure_data_dir()
    # 原子写：先写临时文件再 rename
    tmp_file = KEYS_FILE.with_suffix(".json.tmp")
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(keys, f, ensure_ascii=False, indent=2)
        tmp_file.replace(KEYS_FILE)
    except OSError as e:
        logger.error("保存 %s 失败: %s", KEYS_FILE, e)
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        raise


def _migrate_from_legacy_qsettings() -> bool:
    """
    从 QSettings 迁移旧的明文 API key 到 .api_keys.json。
    返回 True 表示做了迁移。

    只在 .api_keys.json 不存在 / 为空时跑一次。
    """
    # 避免引入循环 import：延迟 import
    from config.settings import get_settings

    settings = get_settings()
    migrated: Dict[str, str] = {}
    for provider, qkey in LEGACY_SETTINGS_KEYS.items():
        try:
            legacy_value = settings.value(qkey, "")
            if legacy_value and isinstance(legacy_value, str):
                migrated[provider] = legacy_value
        except Exception as e:
            logger.debug("读取 QSettings %s 失败: %s", qkey, e)

    if not migrated:
        return False

    # 合并到现有缓存（不清空已有）
    existing = _load_from_disk()
    for provider, key in migrated.items():
        if provider not in existing:
            existing[provider] = key
    _save_to_disk(existing)
    logger.info("从 QSettings 迁移了 %d 个 API key 到 %s", len(migrated), KEYS_FILE)
    return True


def _get_keys_with_migration() -> Dict[str, str]:
    """获取 keys，自动做一次性迁移"""
    global _cache
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is not None:
            return _cache
        keys = _load_from_disk()
        # 如果磁盘文件为空，尝试从 QSettings 迁移
        if not keys:
            _migrate_from_legacy_qsettings()
            keys = _load_from_disk()
        _cache = keys
        return keys


def _invalidate_cache() -> None:
    global _cache
    with _lock:
        _cache = None


# ============================================================
# 公共接口
# ============================================================
def get_api_key(provider: str) -> str:
    """
    读取指定 provider 的 API key。
    找不到返回空字符串。

    查找顺序（按优先级）：
    1. data/.api_keys.json（用户在前端配的，最权威）
    2. QSettings 里同名 key（兼容老用户从 SettingsDialog 改的，暂未迁移到 .api_keys.json）
    3. .env 里的 DEEPSEEK_API_KEY / OPENAI_API_KEY 兜底
    """
    keys = _get_keys_with_migration()
    key = keys.get(provider, "")
    if key:
        return key

    # 兜底 1：QSettings（老用户改过 key 但没迁）
    try:
        from config.settings import get_settings
        settings = get_settings()
        qkey = f"{provider.upper()}_API_KEY"
        qs_value = settings.value(qkey, "") or ""
        if qs_value:
            logger.debug("[api_keys] %s 走 QSettings 兜底 (key=%s)", provider, qkey)
            return qs_value
    except Exception:
        pass

    # 兜底 2：.env 里的 key
    try:
        import config as _cfg
        fallback = getattr(_cfg, f"{provider.upper()}_API_KEY", "") or ""
        if fallback:
            logger.debug("[api_keys] %s 走 .env 兜底", provider)
            return fallback
    except Exception:
        pass

    return ""


def set_api_key(provider: str, key: str) -> None:
    """
    设置 API key。空字符串等同于删除。
    """
    keys = _get_keys_with_migration()
    if key:
        keys[provider] = key
    else:
        keys.pop(provider, None)
    with _lock:
        _save_to_disk(keys)
        _invalidate_cache()


def get_all_api_keys() -> Dict[str, str]:
    """返回所有 key 的副本（避免外部修改）"""
    return dict(_get_keys_with_migration())


def delete_api_key(provider: str) -> None:
    """删除指定 provider 的 key"""
    set_api_key(provider, "")


def has_api_key(provider: str) -> bool:
    """是否存在非空 key"""
    return bool(get_api_key(provider))


def keys_file_path() -> Path:
    """返回 key 文件路径（用于 SettingsDialog 显示）"""
    return KEYS_FILE
