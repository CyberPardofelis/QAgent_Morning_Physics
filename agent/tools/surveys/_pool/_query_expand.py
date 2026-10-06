"""
_pool._query_expand — Phase A 的 LLM 查询扩展 (10-06 加)

**为什么必须有这一步**
---------------------
实测(10-06, topic="neutral atom quantum computing"):
  - OpenAlex 全库命中 82,214 篇
  - 本地语料 247 篇里, 提到 "neutral atom" 的有 **0 篇**
  - 最终候选池 **39 篇**

39 不是这个领域就这么大, 是三处硬编码卡出来的:
  - APS / IOP / arxiv 三路**全部拿 `topic` 原词去搜**, 没有任何查询扩展
  - 每一路 `k = min(need, 30)` 封顶
  - 相关性阈值 8/10 又把召回的再筛掉一层

而且**这个失败不会自愈**: 语料不变的话, 换一天跑还是 39 篇。

`_pool_audit.py` 的 retry loop 注释写着"换 query / 换 source", 实际只把 k
翻倍, query 一个字没换 —— 这是比没写更坏的一种注释。

**这个函数解决两件事**
1. **召回量**: 6 条英文检索式各打一个方向, 召回面比 1 条宽得多
2. **细分方向**: 拆出来的 sub_directions 直接就是综述的主题骨架。宏观主题
   跑出来"章节之间区分度低"就是这么来的 —— 没有细分, 归纳只能按最表层切。
   用户实测: 细分方向上这个工具的效果明显好于宏观方向。

**失败必须响亮**: 扩展挂了只回退成"用原词搜", 不能让 Phase A 挂掉, 但
**一定要说出来** —— 静默回退的话, 用户看到的还是 39 篇, 根本不知道
"本该有 6 条检索式却没跑起来"。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.infra.console import safe_print

logger = logging.getLogger(__name__)

_PROVIDER = "minimax"
_MODEL = "MiniMax-M3"

# 检索式里必须有的拉丁字母数。没有的(纯中文)拿去喂 arxiv/OpenAlex 等于没搜 ——
# 这正是中文 topic 候选池 0 篇的原因, 不能再让它悄悄发生。
_LATIN = re.compile(r"[A-Za-z]{2,}")


def _pick_queries(parsed: dict[str, Any], n: int) -> tuple[list[str], list[dict[str, Any]]]:
    """从模型输出里挑出**能用的**英文检索式, 顺带把细分方向带回来。

    为什么要逐条筛而不是直接用: 模型很爱在 queries 里混一两条中文或者
    把原主题原样抄一遍当"细分方向"。那些喂给英文 API 等于没喂, 还会让
    "6 条检索式"这个数字变得好看但没意义 —— 所以宁可少给几条。
    """
    dirs: list[dict[str, Any]] = []
    for d in (parsed.get("sub_directions") or []):
        if not isinstance(d, dict):
            continue
        nm = str(d.get("name") or "").strip()
        if not nm:
            continue
        dirs.append({"name": nm, "why": str(d.get("why") or "").strip()})

    out: list[str] = []
    for q in (parsed.get("queries") or []):
        if isinstance(q, str):            # 容错: 模型偶尔直接给字符串数组
            q = {"q": q}
        if not isinstance(q, dict):
            continue
        text = str(q.get("q") or "").strip()
        if not text or len(_LATIN.findall(text)) < 2:
            continue
        if text not in out:
            out.append(text)
        if len(out) >= n:
            break
    return out, dirs


def expand_queries(topic: str, n: int = 6) -> dict[str, Any]:
    """主题 → {"queries": [...英文检索式...], "sub_directions": [...], "ok": bool}。

    **永不抛异常**: 扩展是优化, 不是前置条件。挂了就 ok=False + queries=[],
    调用方回退用原 topic 搜 —— 至少跟今天的行为一样。
    """
    empty = {"queries": [], "sub_directions": [], "ok": False, "error": "未调用"}
    if not (topic or "").strip():
        return empty

    from agent.prompts_survey import build_query_expand_prompt
    from agent.tools.llm import _call_llm

    prompt = build_query_expand_prompt(topic, n=n)
    try:
        raw = _call_llm(prompt, provider=_PROVIDER, model=_MODEL,
                        enable_thinking=False, temperature=0.3,
                        max_tokens=4000, timeout=180)
    except Exception as e:                              # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:100]}"
        safe_print(f"[query_expand] ❌ LLM 调用异常: {err}", flush=True)
        return {**empty, "error": f"LLM 调用异常: {err}"}

    if not raw:
        safe_print("[query_expand] ❌ LLM 返回空, 回退用原 topic 检索", flush=True)
        return {**empty, "error": "LLM 返回空"}

    # 剥 <think> / ```json 包裹
    text = raw
    if "<think>" in text and "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        safe_print(f"[query_expand] ❌ 输出里找不到 JSON (返回 {len(raw)} 字符)",
                   flush=True)
        return {**empty, "error": "输出里找不到 JSON"}
    try:
        parsed = json.loads(m.group(0))
    except Exception as e:                              # noqa: BLE001
        safe_print(f"[query_expand] ❌ JSON 解析失败: {str(e)[:80]}", flush=True)
        return {**empty, "error": f"JSON 解析失败: {str(e)[:80]}"}

    queries, dirs = _pick_queries(parsed, n)
    if not queries:
        safe_print("[query_expand] ❌ 模型没给出可用的英文检索式, 回退用原 topic",
                   flush=True)
        return {**empty, "error": "没有可用的英文检索式"}

    safe_print(
        f"[query_expand] ✅ {len(queries)} 条英文检索式 / {len(dirs)} 个细分方向:\n"
        + "".join(f"    · {q}\n" for q in queries),
        flush=True,
    )
    return {"queries": queries, "sub_directions": dirs, "ok": True, "error": ""}
