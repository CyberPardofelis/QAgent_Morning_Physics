"""
_pool._expand — 引文图扩展层 (9-30)

背景
----
实测 (OpenAlex 响应头 X-RateLimit-Credits-Used):
    search= 关键词检索            10 credits / 次
    cites: / cited_by: / doi:      1 credit  / 次, per_page 给满 200 条
关键词检索每天只有 100 次上限, 引文图有 1000 次。所以:
**关键词检索只用来拿种子, 产能必须走引文图。**

为什么是"扩展"而不是"再多搜几轮"
--------------------------------
实测 arXiv 与 OpenAlex 在不同学科的重合度差异极大:
    量子物理 / 张量网络 (RMP)   重合 6/8
    微纳加工 / 纳米剪纸         重合 1/78  ← arXiv 基本触及不到材料类期刊
引用一篇综述的参考文献, 等于直接拿到它认为该引的全部基座工作, 比关键词检索准得多。

三道约束 (9-30 user 拍板)
------------------------
1. 深度限制 max_rounds   —— 限制迭代轮数
2. 广度限制 breadth      —— 不是每篇文章、每个引用都值得扩
3. 剪枝     prune        —— 先看内容决定值不值得扩/留, 关系不大的先丢

不收敛的问题就靠这三条压住: 200 → 200 → 200 本身不会停, 但
frontier 每轮被剪枝 + 截断, 总请求数被 max_requests 兜底, 就收敛了。
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ._openalex import _openalex_work_to_pool_entry

logger = logging.getLogger(__name__)

# OpenAlex 单次引文图查询最多塞多少个 seed
MAX_SEEDS_PER_REQUEST = 50


# ============================================================
# 配置
# ============================================================
@dataclass
class ExpandConfig:
    """扩展层预算。默认值按"一次 40 篇综述的跑"给的。"""

    # 1) 深度限制: 迭代轮数。1 = 只扩一轮(种子直接取参考文献/被引), 2 = 扩到引用者的引用者
    max_rounds: int = 2

    # 2) 广度限制: 每轮从多少篇论文出发扩展。
    #    注意这是"选哪几篇去扩", 不是"拿多少结果" —— 结果数由 per_page 决定且不计费。
    forward_breadth: int = 40
    backward_breadth: int = 40

    # 单次请求最多取多少条 (OpenAlex 上限 200, 不计费所以默认给满)
    per_page: int = 200

    # credit 兜底: 引文图 1 credit/次, 1000/天, 这里默认最多花 120 次
    max_requests: int = 120

    # 全局产出上限, 防止前沿爆掉撑爆内存/后续 LLM 审查
    max_candidates: int = 2000

    # 只沿综述通道走 (type:review) —— 用户"不一定需要完整文章, 可以只找综述"的需求
    reviews_only: bool = False

    # 剪枝: 标题+摘要与主题词的重合度低于此值直接丢。0 = 不按主题剪
    min_topic_overlap: float = 0.0

    # 剪枝: 被引量低于此值丢 (0 = 不按影响力剪)。综述基座常是高被引, 调这个要小心
    min_cited_by: int = 0

    def requests_for(self) -> int:
        """**上界**: 按每轮 breadth 都塞满算出的最大请求数。

        forward / backward 是**两个独立方向, 各自打包各自发** ——
        所以批次数要分别算, 不能把两边 seed 加起来再除以 50 (那会把批次算重一倍)。
        """
        n = 0
        for _ in range(self.max_rounds):
            n += _ceil_div(self.forward_breadth, MAX_SEEDS_PER_REQUEST)
            n += _ceil_div(self.backward_breadth, MAX_SEEDS_PER_REQUEST)
        return n


def _ceil_div(a: int, b: int) -> int:
    return max(1, (a + b - 1) // b) if a else 0


# ============================================================
# 结果
# ============================================================
@dataclass
class ExpansionResult:
    entries: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


# ============================================================
# 剪枝
# ============================================================
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "for", "and", "in", "on", "to", "with",
    "by", "from", "is", "are", "was", "were", "be", "been",
    "we", "our", "their", "its", "it", "as", "at", "that", "this",
    "these", "those", "using", "use", "based", "via",
})


def _terms(text: str) -> set[str]:
    """分词出实词。英文按空格切; 中文按 2-gram (无分词依赖, 够用)。"""
    s = (text or "").lower()
    out: set[str] = set()
    for w in re.findall(r"[a-z0-9]+", s):
        if len(w) > 2 and w not in _STOPWORDS:
            out.add(w)
    han = re.findall(r"[\u4e00-\u9fff]", s)
    for i in range(len(han) - 1):
        out.add(han[i] + han[i + 1])
    return out


def topic_overlap(topic: str, *texts: str) -> float:
    """主题词在文本中的覆盖比例 0~1。文本里没有任何主题词时返回 0。"""
    tp = _terms(topic)
    if not tp:
        return 1.0
    hay = _terms(" ".join(t for t in texts if t))
    if not hay:
        return 0.0
    return len(tp & hay) / len(tp)


def select_frontier(
    entries: list[dict[str, Any]],
    breadth: int,
) -> list[dict[str, Any]]:
    """广度限制: 从候选里挑 breadth 篇值得继续扩的。

    排序依据 (这个顺序是刻意的):
      1. 有摘要的优先 —— 没摘要连剪枝都做不了, 扩了也是盲扩
      2. 被引多的优先 —— 综述的参考文献里高被引的更可能是基座工作
    """
    def key(e: dict[str, Any]):
        has_abs = 1 if (e.get("abstract") or "").strip() else 0
        return (has_abs, int(e.get("cited_by_count") or 0))

    return sorted(entries, key=key, reverse=True)[: max(0, breadth)]


def prune(
    candidates: list[dict[str, Any]],
    seen_ids: set[str],
    topic: str = "",
    min_topic_overlap: float = 0.0,
    min_cited_by: int = 0,
) -> tuple[list[dict], dict[str, int]]:
    """剪枝: 丢掉 已经见过 / 没标题 / 跑题 / 被引过低 的。

    Returns:
        (kept, dropped_stats) —— dropped_stats 按原因计数, 用于控制台/日志交代
    """
    kept: list[dict[str, Any]] = []
    drops = {"seen": 0, "no_title": 0, "off_topic": 0, "low_cited": 0, "no_id": 0}
    for c in candidates:
        wid = _entry_oa_id(c)
        if not wid:
            drops["no_id"] += 1
            continue
        if wid in seen_ids:
            drops["seen"] += 1
            continue
        if not (c.get("title") or "").strip():
            drops["no_title"] += 1
            continue
        if min_cited_by and int(c.get("cited_by_count") or 0) < min_cited_by:
            drops["low_cited"] += 1
            continue
        if min_topic_overlap > 0:
            ov = topic_overlap(topic, c.get("title", ""), c.get("abstract", ""))
            if ov < min_topic_overlap:
                drops["off_topic"] += 1
                continue
        seen_ids.add(wid)
        kept.append(c)
    return kept, drops


def _entry_oa_id(entry: dict[str, Any]) -> str:
    """取 OpenAlex work id, 支持 W123 / 完整 URL / 包在 _openalex_id 里。"""
    for k in ("_openalex_id", "openalex_id", "id"):
        v = entry.get(k)
        if v:
            s = str(v).rstrip("/").rsplit("/", 1)[-1]
            if s.startswith("W"):
                return s
    return ""


# ============================================================
# 主循环
# ============================================================
def _batches(items: list[str], size: int = MAX_SEEDS_PER_REQUEST):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def expand_from_seeds(
    seeds: list[dict[str, Any]],
    config: ExpandConfig | None = None,
    topic: str = "",
    fetch: Callable[..., list[dict]] | None = None,
) -> ExpansionResult:
    """从种子出发做引文图扩展。

    Args:
        seeds: pool entry 列表 (需带 OpenAlex id, 字段见 _openalex_work_to_pool_entry)
        config: 预算/约束; 默认 ExpandConfig()
        topic: 主题串, 供 min_topic_overlap 剪枝用
        fetch: 注入的取数函数, 签名 (direction, seed_ids, per_page, reviews_only)
               -> list[pool entry]。默认走真实 OpenAlex; 测试注入假的。

    Returns:
        ExpansionResult(entries=所有通过的候选, stats=轮次/请求数/剪枝计数)
    """
    cfg = config or ExpandConfig()
    get = fetch or _default_fetch

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "rounds": 0, "requests": 0, "max_requests": cfg.max_requests,
        "dropped": {}, "stopped_reason": "",
    }

    # 种子先进池
    frontier: list[dict[str, Any]] = []
    kept, drops = prune(seeds, seen, topic, cfg.min_topic_overlap, cfg.min_cited_by)
    frontier.extend(kept)
    out.extend(kept)
    _merge_drops(stats["dropped"], drops)

    for rnd in range(1, cfg.max_rounds + 1):
        if not frontier:
            stats["stopped_reason"] = stats["stopped_reason"] or "frontier 空(剪枝后无可扩展)"
            break
        if stats["requests"] >= cfg.max_requests:
            stats["stopped_reason"] = "达到 max_requests 上限"
            break
        if len(out) >= cfg.max_candidates:
            stats["stopped_reason"] = "达到 max_candidates 上限"
            break

        # 广度限制: 前向/后向各取一批 seed 去扩
        fwd_seeds = select_frontier(frontier, cfg.forward_breadth)
        bwd_seeds = select_frontier(frontier, cfg.backward_breadth)

        nxt: list[dict[str, Any]] = []
        for direction, batch in (("forward", fwd_seeds), ("backward", bwd_seeds)):
            ids = [i for i in (_entry_oa_id(e) for e in batch) if i]
            for chunk in _batches(ids):
                if stats["requests"] >= cfg.max_requests:
                    stats["stopped_reason"] = "达到 max_requests 上限"
                    break
                stats["requests"] += 1
                try:
                    raw = get(direction, chunk, cfg.per_page, cfg.reviews_only)
                except Exception as e:
                    # 不静默: 扩展层是并集的一路, 塌了就只剩另一路了 (9-30 实测教训)
                    print(
                        f"[expand] {direction} 扩展失败 (seeds={len(chunk)}): "
                        f"{type(e).__name__}: {str(e)[:120]}",
                        flush=True,
                    )
                    continue
                entries = [_openalex_work_to_pool_entry(w, f"openalex_{direction}_api")
                           for w in (raw or [])]
                kept, drops = prune(
                    entries, seen, topic, cfg.min_topic_overlap, cfg.min_cited_by)
                _merge_drops(stats["dropped"], drops)
                nxt.extend(kept)
                out.extend(kept)
                if len(out) >= cfg.max_candidates:
                    break
        stats["rounds"] = rnd
        frontier = nxt
        if len(out) >= cfg.max_candidates:
            stats["stopped_reason"] = "达到 max_candidates 上限"
            break

    if not stats["stopped_reason"]:
        stats["stopped_reason"] = "达到 max_rounds 上限(收敛)"

    stats["total"] = len(out)
    print(
        f"[expand] 种子 {len(seeds)} → 候选 {len(out)}  "
        f"轮次 {stats['rounds']}/{cfg.max_rounds}  请求 {stats['requests']}  "
        f"剪枝 {stats['dropped']}  停止原因: {stats['stopped_reason']}",
        flush=True,
    )
    return ExpansionResult(entries=out[: cfg.max_candidates], stats=stats)


def _default_fetch(
    direction: str,
    seed_ids: list[str],
    per_page: int,
    reviews_only: bool,
) -> list[dict]:
    """真实 OpenAlex 取数。

    ⚠️ 方向映射别搞反 (9-30 实测确认 OpenAlex 命名反直觉):
        "forward"  -> cites:         谁引用了它
        "backward" -> cited_by:      它引用了谁 (= 它的参考文献)
    """
    from agent.tools.openalex import openalex_cites, openalex_references

    fn = openalex_cites if direction == "forward" else openalex_references
    return fn(seed_ids, per_page=per_page, reviews_only=reviews_only,
              caller=f"expand_{direction}")


def _merge_drops(acc: dict[str, int], new: dict[str, int]) -> None:
    for k, v in (new or {}).items():
        acc[k] = acc.get(k, 0) + int(v)
