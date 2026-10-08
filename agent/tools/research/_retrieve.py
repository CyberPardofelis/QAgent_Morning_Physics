"""
agent.tools.research._retrieve — OEIS 整数序列检索 (P7 阶段 1)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md §1.5.2

================================ 为什么 OEIS 值得单独接 =================================
它和 arXiv 完全不是一类东西(§1.5.2):
  arXiv  → "这个结论谁证过"        → 文献证据 (走已有 search_arxiv)
  OEIS  → "这个序列是什么"        → 结构证据 (本模块)

闭环里的用法: 算出一个新序列 → 查 OEIS 认出已知结构 → **拿到 OEIS 给的公式后回头用
SymPy 验证** —— 后两步合起来才是完整证据。OEIS 自己给的编号不算证据,
"你的序列与 A000079 的前 5 项完全一致" 才算。

=============================== 本模块比裸 API 多做的一件事 ===========================
**逐项对账**。OEIS 的搜索接口对任何序列都会返回 10 条结果(模糊匹配), 直接把
第一条当答案就是典型的检索幻觉。所以这里对每个候选算吻合比例,
只有 `exact_match=True`(输入序列是它的前缀)的才标记为强匹配, 并按吻合度排序。

实测返回形态 (2026-10-08): GET https://oeis.org/search?q=1,2,4,8,16&fmt=json
    → 裸 JSON **数组**(不是 {"results": [...]}), 每项含 number/name/formula/data/offset/...

历史:
  - 2026-10-08 P7.1 首版
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_OEIS_URL = "https://oeis.org/search"
_UA = "QAgent/0.1 (research verification tool; contact: local)"

#: OEIS 搜索一次固定返 10 条; 没必要都要, 但要足够挑出真匹配
_DEFAULT_KEEP = 5


class RetrieveError(Exception):
    """网络/解析层失败 —— 与"没查到"是两回事"""


def parse_sequence(seq: str) -> list[int]:
    """
    解析 '1,2,4,8,16' 或 '1 2 4 8 16' 或 '[1, 2, 4]' → [1,2,4,8,16]

    Raises:
        ValueError: 少于 2 项, 或含非整数项
    """
    if isinstance(seq, (list, tuple)):
        items = list(seq)
    else:
        s = str(seq).strip().strip("[](){}")
        items = [t for t in s.replace(",", " ").split() if t]

    out: list[int] = []
    for it in items:
        try:
            out.append(int(str(it).strip()))
        except ValueError as e:
            raise ValueError(
                f"序列里 {it!r} 不是整数。OEIS 只收整数序列; "
                f"非整数序列请用符号计算(sympy)而不是查 OEIS"
            ) from e

    if len(out) < 2:
        raise ValueError(
            f"至少要 2 项才能查 OEIS, 当前只有 {len(out)} 项 —— "
            f"项太少 OEIS 会返回一堆无关结果"
        )
    return out


def lookup(
    sequence: str | list[int],
    max_terms: int = 10,
    timeout: float = 30.0,
    keep: int = _DEFAULT_KEEP,
) -> dict[str, Any]:
    """
    查 OEIS 并逐项对账。

    Args:
        sequence: 整数序列字符串或 list
        max_terms: 最多送多少项去查(太长 OEIS 会拒)
        timeout: HTTP 超时秒数
        keep: 返回前几条候选

    Returns:
        {ok, query, n_terms, candidates: [{number, name, formula, exact_match,
          matched, ratio, data_head, offset}], best, verdict, reason}
    """
    try:
        import httpx
    except ImportError as e:                                  # pragma: no cover
        raise RetrieveError("httpx 未安装, 无法访问 OEIS") from e

    terms = parse_sequence(sequence)[:max_terms]
    query = ",".join(str(t) for t in terms)

    try:
        resp = httpx.get(
            _OEIS_URL,
            params={"q": query, "fmt": "json"},
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _UA},
        )
    except Exception as e:
        raise RetrieveError(f"访问 OEIS 失败: {type(e).__name__}: {e}") from e

    if resp.status_code != 200:
        raise RetrieveError(f"OEIS 返回 HTTP {resp.status_code}")

    try:
        data = resp.json()
    except Exception as e:
        raise RetrieveError(f"OEIS 返回的不是合法 JSON: {e}") from e

    # 实测: fmt=json 返回的是**裸数组**。防御性地兼容 {"results": [...]} 形态。
    rows = (
        (data.get("results", []) or data.get("matches", []))
        if isinstance(data, dict) else (data or [])
    )

    candidates = []
    for row in rows:
        entry = _score_candidate(terms, row)
        if entry:
            candidates.append(entry)

    # 强匹配优先, 其次按吻合比例
    candidates.sort(key=lambda c: (c["exact_match"], c["ratio"]), reverse=True)
    candidates = candidates[:keep]

    exact = [c for c in candidates if c["exact_match"]]

    if not candidates:
        verdict = "untested"
        reason = (f"OEIS 没有返回 {query} 的匹配项 —— "
                  f"这通常意味着**这是个新序列**(好信号), 而不是查询失败。")
    elif exact:
        best = exact[0]
        verdict = "partial"
        reason = (
            f"命中 OEIS {best['number']} ({best['name'][:60]}), "
            f"输入序列与它的前 {best['matched']} 项**完全一致**。"
            f"⚠️ 这是**已知结构**, 不等于你证明了什么 —— "
            f"下一步应该拿它的公式 {best['formula'][:50]} 回去用 sympy 验证。"
        )
    else:
        best = candidates[0]
        verdict = "untested"
        reason = (
            f"OEIS 返回 {len(candidates)} 条候选但**无完全匹配**"
            f"(最接近的是 {best['number']}, 吻合 {best['matched']}/{len(terms)} 项)。"
            f"不要把它当成答案 —— 这说明这个序列可能不是标准序列, 或者你该换检索式。"
        )

    return {
        "ok": True,
        "query": query,
        "n_terms": len(terms),
        "candidates": candidates,
        "best": exact[0] if exact else (candidates[0] if candidates else None),
        "n_exact_matches": len(exact),
        "verdict": verdict,
        "reason": reason,
    }


def _score_candidate(terms: list[int], row: dict[str, Any]) -> dict[str, Any] | None:
    """
    给一个 OEIS 条目打分: 逐项对账, 算吻合数与比例。

    这步是本模块的核心 —— 没有它, 返回的第一条就是噪声。
    """
    number = row.get("number")
    if not number:
        return None

    data_str = row.get("data") or ""
    # ⚠️ OEIS 的 data 字段形如 "1, 2, 4, 8, 16, 32, ..." —— **逗号和空格混用**。
    #    只按空白切的话 int('1,') 抛 ValueError, 被下面 except 掉 → seq=[] →
    #    matched 恒为 0 → 所有候选 exact_match=False, 精确匹配形同虚设。
    #    (2026-10-08 实测: 查 1,2,4,8,16 明明是 A000079 却报 exact=False)
    try:
        seq = [int(t) for t in re.split(r"[,\s]+", data_str.strip("[]").strip()) if t]
    except ValueError:
        seq = []

    matched = 0
    for i, t in enumerate(terms):
        if i < len(seq) and seq[i] == t:
            matched += 1
        else:
            break          # 不连续就不算对上了, 序列对账要前缀一致

    ratio = matched / len(terms) if terms else 0.0
    formula_list = row.get("formula") or []
    formula = formula_list[0] if isinstance(formula_list, list) and formula_list else str(formula_list)

    return {
        "number": number,
        # OEIS JSON 的 number 是裸数字 "79", 给人看的规范写法是 "A000079"
        "id_full": f"A{int(number):06d}" if str(number).isdigit() else str(number),
        "name": (row.get("name") or "").strip(),
        "formula": formula,
        "matched": matched,
        "ratio": round(ratio, 3),
        "exact_match": matched == len(terms) and len(seq) >= len(terms),
        "data_head": seq[:max(len(terms), 8)],
        "offset": row.get("offset"),
        "url": f"https://oeis.org/A{int(number):06d}" if str(number).isdigit() else "",
    }
