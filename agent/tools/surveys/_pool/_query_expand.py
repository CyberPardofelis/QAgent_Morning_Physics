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
from agent.infra.text_clean import sanitize_llm_text

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


def _parse_gap_verdict(raw: str) -> dict[str, Any] | None:
    """从模型输出里抠出 gap 判定 JSON。抠不出来返回 None。

    **三个实测踩过的坑**
    1. `<think>` 块**没闭合**(被 max_tokens 截断在思考里): 后面根本没有 JSON,
       而贪婪的 `\\{.*\\}` 会从 think 散文里某个 `{` 一路吃到结尾, 拼出一段
       语法上像 JSON 的垃圾。所以"有 <think> 却没有 </think>"要单独识别。
    2. **NUL 字节**。10-06 实测 6 次调用有 6 次被这个卡死, 长相是:
           "Toff\\x00oli\\x00\\x00 GHZ global addressing"
       —— `Toffoli` 变成 `Toff\\x00oli`, 破折号变成 `\\x00\\x00 `。
       这是**输出流在字节层面被破坏**, 不是模型"打了隐藏字符"。
       `json.loads(strict=False)` 治不了: strict=False 只放行 `\\n \\t \\r` 那些
       标准控制符, NUL 照样报 "Expecting value"。
    3. 字符串里有裸换行/制表(那 31 条空白本来就带换行和全角括号, 模型原样抄),
       这个 `strict=False` 能治。
    """
    if not raw:
        return None
    # 先洗掉 NUL 再谈解析 —— 见上面第 2 点。
    # ⚠️ 这层现在是**第二道**防线: 主防线已经放在 `_call_llm` 的出口
    # (infra/text_clean.sanitize_llm_text), 下面所有工具的 JSON 解析都在
    # 那之下。这里保留是因为本函数的 docstring 承诺"从模型输出里抠出 JSON",
    # 而它也可能被直接喂进历史产物/复现现场这类**没走过 _call_llm** 的文本。
    raw = sanitize_llm_text(raw)
    text = raw
    if "<think>" in text and "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        # 输出全花在思考里 —— 明确说出来, 别让上层报成"JSON 找不到"
        return {"__truncated_think__": True}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0), strict=False)
    except Exception:                                   # noqa: BLE001
        return None


def seen_gaps(ranked) -> set:
    """从 (score, gap, query) 三元组里取出出现过的 gap 集合。"""
    return {g for _sc, g, _q in ranked}


def gaps_to_queries(topic: str, gaps: list[str],
                    max_queries: int = 4) -> dict[str, Any]:
    """覆盖空白 → 按可补性排序, 给出 top-N 检索式(10-06 加)。

    **为什么不能把 gap 原话当检索词**
    实测(10-06) 模型吐的空白长这样:
        "缺少具体的纠错协议(表面码、颜色码、订阅码)在原子平台上端到端执行的实测结果"
    这是一句**描述**, 不是检索式 —— 整句喂给 arxiv 等于零召回。中间必须过一次
    LLM 把"缺什么"翻译成"搜什么"。

    **为什么用打分而不是二分类 —— 这是踩了两轮才换过来的**
    第一版让模型输出 `fillable: true/false` + `why`。结果是**标签和理由系统性
    自相矛盾**:
        gap:   "多处理器互联与模块化扩展未覆盖(演示规模停留在 280 比特/单数组)"
        label: unfillable
        why:   "Modular-interconnect papers (photonic links, conveyor belts) exist"
    理由在说"这方向的文献存在", 那恰恰是可补的铁证。改了 prompt 要求理由只能
    落在"缺的是别的领域 / 缺的是领域现状", **还是照样矛盾** —— 根因不是措辞,
    是"这方向有没有文献"这种二值判断对模型来说本身就不稳。

    换成 0-10 打分之后, 我们不再问它"是不是", 只问"有多可能", 再按分数取
    top-N(max_queries)。排序比二分类稳得多, 而且成本本来就被 max_queries 闸住了。

    **`why` 字段直接不要**
    既然它已被证明不可靠, 就别回显给用户 —— 上面那种自相矛盾的句子出现在
    最终综述的"文献覆盖的空白"里, 比没有理由更糟。给不出理由的判定, 就只标
    "判定补不上"。

    返回 {"fillable": [{gap, query, score}], "unfillable": [gap],
           "ok": bool, "error": str}
    """
    empty = {"fillable": [], "unfillable": [], "ok": False, "error": "未调用"}
    gaps = [g.strip() for g in (gaps or []) if (g or "").strip()]
    if not gaps:
        return {**empty, "error": "没有空白"}
    if not (topic or "").strip():
        return {**empty, "error": "没有主题"}

    from agent.prompts_survey import build_gap_fill_prompt
    from agent.tools.llm import _call_llm

    prompt = build_gap_fill_prompt(topic, gaps, max_queries)
    parsed: dict[str, Any] | None = None
    last_err = ""
    # 重试一次: 实测同一份 prompt 的输出格式**不稳定** —— 同样的输入,
    # 一次返回 6380 字符合法 JSON, 下一次 think 就吃光预算只吐半截。
    # 一次重试的成本(1 次调用)远低于"整轮回补静默消失"。
    for attempt in range(2):
        try:
            raw = _call_llm(
                prompt, provider=_PROVIDER, model=_MODEL,
                enable_thinking=False, temperature=0.2,
                # enable_thinking=False **挡不住** MiniMax 输出 think, 那部分照样
                # 吃 max_tokens。31 条空白逐条打分, think 轻松上万字符 ——
                # 6000 的预算全被 think 吃掉, 返回 23635 字符里一个 JSON 都没有。
                max_tokens=20000, timeout=240)
        except Exception as e:                          # noqa: BLE001
            last_err = f"LLM 调用异常: {type(e).__name__}: {str(e)[:80]}"
            safe_print(f"[gap_fill] ❌ {last_err}", flush=True)
            continue
        if not raw:
            last_err = "LLM 返回空"
        else:
            got = _parse_gap_verdict(raw)
            if got is None:
                last_err = f"输出里抠不出 JSON (返回 {len(raw)} 字符)"
            elif got.get("__truncated_think__"):
                last_err = (f"输出全在思考里就被截断 (返回 {len(raw)} 字符, "
                            f"think 未闭合)")
            else:
                parsed = got
                break
        safe_print(f"[gap_fill] ⚠ 第 {attempt + 1} 次: {last_err}", flush=True)
    if parsed is None:
        safe_print(f"[gap_fill] ❌ 判定失败: {last_err}, 跳过回补", flush=True)
        return {**empty, "error": last_err}

    # 打分 + 检索式, 按分降序
    # **靠 idx 对回原始空白, 不靠模型回传的 gap 文本**: 实测模型经常把空白
    # 原文改写(截断、换标点、漏字), 10-06 连跑 3 次有 2 次对不上, 于是同一条
    # 既在 ranked 里(改写版)又被当成"模型没回的"补进 unfillable(原文版),
    # 31 条空白一度数出 59 条"不回补"。idx 是 prompt 里写死的对齐键, 稳。
    scored: list[tuple[int, str, str]] = []
    for row in (parsed.get("gaps") or []):
        if not isinstance(row, dict):
            continue
        q = str(row.get("query") or "").strip()
        if not q or len(_LATIN.findall(q)) < 2:
            continue
        try:
            sc = int(row.get("score", 0))
        except (TypeError, ValueError):
            sc = 0
        idx = row.get("idx")
        gap = None
        if isinstance(idx, int) and 1 <= idx <= len(gaps):
            gap = gaps[idx - 1]
        if not gap:
            # 没有 idx 或者越界 -> 退回按文本匹配, 再不行就丢掉这一条
            txt = str(row.get("gap") or "").strip()
            gap = txt if txt in gaps else None
            if not gap:
                continue
        scored.append((max(0, min(10, sc)), gap, q))

    # 同一条 gap 可能被模型重复输出, 保留分最高的那个
    best: dict[str, tuple[int, str, str]] = {}
    for sc, gap, q in scored:
        if gap not in best or sc > best[gap][0]:
            best[gap] = (sc, gap, q)
    ranked = sorted(best.values(), key=lambda x: -x[0])

    fillable = [{"gap": g, "query": q, "score": sc} for sc, g, q in ranked[:max_queries]]
    chosen = {f["gap"] for f in fillable}
    unfillable = [g for _sc, g, _q in ranked if g not in chosen]
    # 模型漏回的空白也要落到 unfillable, 不能凭空消失(但**不能重复计数**)
    unfillable += [g for g in gaps if g not in seen_gaps(ranked)]
    # 去重保序: 同一 gap 只留第一次出现的位置
    _seen: set[str] = set()
    unfillable = [g for g in unfillable if not (g in _seen or _seen.add(g))]

    safe_print(
        f"[gap_fill] 打分 {len(ranked)}/{len(gaps)} 条, "
        f"取前 {len(fillable)} 条回补"
        + ("\n    " + "\n    ".join(
            f"{f['score']}分 · {f['query'][:64]}" for f in fillable)
           if fillable else "\n    (没有可用检索式 —— 本轮不回补)"),
        flush=True,
    )
    return {"fillable": fillable, "unfillable": unfillable,
            "ok": True, "error": ""}
