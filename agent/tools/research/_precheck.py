"""命题体检闸门 —— P4.5 第三轮重构。

## 为什么要有这一道, 以及为什么它替代了另外三条

P4.5 前两轮的做法是**加防线**: 偷换量词就查量词, 缩小范围就查范围,
加阈值就查阈值。于是攒到八条, 其中三条是拿正则去匹配自然语言:

    UNDECIDABLE_MARKERS 黑名单 / scope_conflict / threshold_conflict

那个方向是错的, 三个理由:

1. **正则不懂语义**。"这段证明是优美的"能被关键词抓到, 但
   "对所有偶数 n 满足 4<=n<=200000 存在至少 1000 组素数对" 里哪个数字是
   "原问题没有要求的限定" —— 正则只能靠模式猜, 换个说法就漏。
2. **黑名单只会越长**。"优美/有趣/最优算法/本质上" 之后必然要加
   "最小的""不存在的""超算力"… 加到某个长度就没人敢动它了。
3. **它们在错误的层**。这些检查都在问"这条命题长得对不对",
   而真正该问的是"**这条命题该不该被验证**"。这是个**流程问题**,
   不是格式问题 —— 用模式匹配去解流程问题, 天生打补丁。

所以改成: **建模之后、验证之前, 先整体体检一遍。**
体检不通的命题根本不进验证循环, 而不是进去了再被拦下。

体检交给 LLM 做语义对照 —— 这正是它擅长而正则不擅长的:
"命题 P 和问题 Q 是不是同一件事"、"这条命题本身自洽吗、能不能写成代码断言"。

## 为什么仍然保留 0 LLM 的硬检查

闸门是 LLM 判的, 它自己也会失手。所以三条**结构性契约**留在代码层, 不交给 LLM:

    · conclusive verdict 必须挂可复核的证据
    · 量词是结构化字段, 不靠中文措辞隐含
    · 审计 0 LLM: 证据链自洽、refuted 必须给见证值

判据很简单: **"结论对不对"可以问 LLM, "结论有没有证据"必须问机器。**
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: 体检结果里, 命题被判"不通"的取值
BLOCKED = "blocked"

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def _extract_json(text: str) -> dict[str, Any]:
    """从 LLM 输出里抠 JSON。第一版 P2 的同款容错(代码块围栏 + 前后解说)。"""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = _FENCE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def build_precheck_prompt(question: str, specs: list[dict[str, Any]]) -> str:
    """
    构造命题体检 prompt。

    ★ 这里的判据是**成对的**: 每条命题都要同时回答
      (a) 它和原问题是同一件事吗?
      (b) 它本身可判定吗(能写成代码断言)?
    只问 (a) 会漏掉"命题本身是空的/矛盾的"; 只问 (b) 会漏掉偷换。
    """
    items = []
    for i, s in enumerate(specs, 1):
        items.append(
            f"{i}. claim_text: {s.get('claim_text','')}\n"
            f"   scope: {s.get('scope','') or '(未给)'}\n"
            f"   quantifier: {s.get('quantifier','') or '(未给)'}"
        )
    listing = "\n".join(items) if items else "(没有命题)"

    return f"""你是一个严格的研究审稿人。给你一个**原始问题**和一份**命题拆解**,
请逐条判断: 这些命题到底该不该被拿去验证。

【原始问题】
{question}

【待体检的命题】
{listing}

【对每条命题回答两个问题】

Q1「same_question」: 这条命题和原始问题断言的是**同一件事**吗?
   以下都是**不是同一件事**(它们是实测抓到的真实翻车):
   · 换量词: 问题是"存在 n 使 P", 命题变成"对所有 n 有 P"
   · 换范围: 问题是"n>100000", 命题只覆盖 200..500
   · 加限定: 问题问哥德巴赫(偶数=两素数之和, 即至少 1 组),
             命题变成"存在至少 1000 组素数对" —— 拿加强版去反驳原问题
   · 换对象: 问题问"这段证明是优美的", 命题变成"求和公式成立"
     (优美没有判定标准; 证明用了归纳法 ≠ 证明是优美的)
   · 加无关限定: 问题没要求"最多三条边", 命题里加了

Q2「decidable」: 这条命题**本身**能写成代码断言吗?它自洽吗?
   判 no 的典型:
   · 没有真假判据("优美""有趣""是否值得")
   · **自相矛盾**: "最小的不存在的素数" —— 不存在的东西没有最小值
   · 约束弱到没有反例空间: "不存在 m 使 2^m-1 是素数且 n<2"
     —— n<2 压根没几个数, 这不是证明, 是证明了空集
   · 连"多大范围算够"都说不清

【判据】
宁可判 no。**在 no 和 yes 之间, 犹豫时选 no** ——
漏放一条不该验的命题, 代价是给用户一个看似可信的错误结论;
误拦一条好命题, 代价只是让人手动看一眼。

【输出】只输出 JSON, 不要任何解释:
{{"results": [
  {{"index": 1, "same_question": true/false, "decidable": true/false,
   "why": "一句话, 判 no 时必须说明差在哪 / 为什么不可判定"}}
]}}"""


def run_precheck(
    question: str,
    specs: list[dict[str, Any]],
    call_llm,
) -> dict[int, dict[str, Any]]:
    """
    跑一遍体检。返回 {index(从1起): {same_question, decidable, why}}。

    ★ **体检失败(调不通 / JSON 坏 / 少返回了)时一律判通** ——
    闸门不能因为自己坏了就把所有命题拦下, 那等于功能瘫掉。
    这是"闸门失灵时倒向放行"的取舍: 放行的后果是多花 token 验一条
    可能有问题的命题; 拦截的后果是工具完全不可用。
    拦不住的那部分, 由代码层的证据强制和审计兜底。
    """
    if not specs:
        return {}
    try:
        raw = call_llm(build_precheck_prompt(question, specs), temperature=0.0,
                       max_tokens=2000, timeout=120)
    except Exception as e:
        logger.warning("[research.precheck] 体检调用失败, 本轮放行: %s", e)
        return {}

    data = _extract_json(raw)
    rows = data.get("results")
    if not isinstance(rows, list):
        logger.warning("[research.precheck] 体检返回不可解析, 本轮放行: %r", str(raw)[:200])
        return {}

    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("index", 0))
        except (TypeError, ValueError):
            continue
        if idx < 1 or idx > len(specs):
            continue
        out[idx] = {
            "same_question": bool(row.get("same_question", True)),
            "decidable": bool(row.get("decidable", True)),
            "why": str(row.get("why", "")).strip(),
        }

    if len(out) < len(specs):
        logger.warning("[research.precheck] 只体检到 %d/%d 条, 缺的按放行处理",
                       len(out), len(specs))
    return out


def apply_precheck(specs: list[dict[str, Any]], checks: dict[int, dict[str, Any]]) -> dict[str, int]:
    """
    把体检结果写回 specs: 不通的标成 undecidable(不进验证循环)。

    Returns:
        统计 {"checked": n, "blocked": m, "llm_failed": k}
    """
    stats = {"checked": 0, "blocked": 0, "llm_failed": 0}
    if not checks:
        stats["llm_failed"] = 1
        return stats

    for i, spec in enumerate(specs, 1):
        res = checks.get(i)
        if res is None:
            continue                       # 体检没覆盖到这条 → 放行(见上面取舍)
        stats["checked"] += 1
        bad_reasons = []
        if not res["same_question"]:
            bad_reasons.append(
                f"体检判定: 不是在回答原问题 —— {res['why'] or '命题与问题断言的不是同一件事'}"
            )
        if not res["decidable"]:
            bad_reasons.append(
                f"体检判定: 命题本身不可判定 —— {res['why'] or '无法写成代码断言'}"
            )
        if not bad_reasons:
            continue

        spec["decidable"] = False
        spec["state"] = "undecidable"
        spec["precheck"] = res
        spec["warn"] = list(spec.get("warn") or []) + bad_reasons
        stats["blocked"] += 1

    return stats
