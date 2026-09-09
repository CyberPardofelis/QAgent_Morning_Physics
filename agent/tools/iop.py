"""
agent.tools.iop — IOP 期刊 (IOP Publishing) 工具集 (2026-09-09 新增)

IOP 是 quant-ph 第二大主阵地 (仅次于 APS):
- NJP (New Journal of Physics): 全 OA 期刊, 覆盖 quant-ph 极广
- J. Phys. A: 理论物理 / 量子信息基础
- J. Phys. B: 原子分子光物理
- J. Opt.: 光学
- J. Phys. D: 应用物理
- J. Phys. Condensed Matter: 凝聚态
- J. Phys. Conference Series: 会议系列

为什么走 OpenAlex (不直接爬 iopscience.iop.org):
- 实测 iopscience.iop.org/search 200 但返回 hCaptcha challenge
- articleSearch API 路径已 404 (已废弃)
- OpenAlex 完整覆盖 IOP 全部期刊, 公开 API 不限流, 拿 DOI/作者/摘要/PDF URL

提供的工具:
- search_iop: 关键词搜索, 默认 NJP + J. Phys. A + J. Phys. B + J. Opt. (quant-ph 主阵地)
"""
import logging
import requests
from agent.tools.registry import register_tool
# 2026-09-09 P0.6: 跟 search_aps 共用 OpenAlex 客户端
from agent.tools.openalex import (
    openalex_search as _openalex_search,
    format_paper_markdown as _openalex_format_paper,
    resolve_source_ids as _resolve_openalex_source_ids,
    OPENALEX_IOP_SOURCES as _OPENALEX_IOP_SOURCES,
    OPENALEX_IOP_DEFAULT_QUANT_PH as _OPENALEX_IOP_DEFAULT_QUANT_PH,
)

logger = logging.getLogger(__name__)


# ============================================================
# search_iop: IOP 期刊关键词搜索
# 走 OpenAlex API (避免 iopscience.iop.org 反爬)
# ============================================================
@register_tool(name="search_iop", is_streaming=False, cost_tier="L0")
def search_iop(
    query: str,
    max_results: int = 10,
    journals: str = "",
) -> str:
    """
    在 IOP 期刊中按关键词搜索论文（走 OpenAlex, 不调 LLM）。

    Args:
        query: 自由文本查询, e.g. "quantum error correction" / "topological phase"
        max_results: 最多返回多少篇 (1-200, 默认 10)
        journals: 逗号分隔的期刊 short_name, 限定搜索范围
                  可选: njp, jphysa, jphysb, jphysd, jphyscm, jopt, jphysconf
                  空 = 默认 quant-ph 集 (njp, jphysa, jphysb, jopt)

    Returns:
        Markdown 格式的论文列表, 含 DOI 链接 / 作者 / 日期 / 期刊 / 摘要 / PDF (OA)

    触发场景:
      - 用户说 "搜一下 NJP 上 X 主题的论文" / "IOP 有没有 X 的文章" / "在 New Journal of Physics 找 X"
      - 用户说 "search IOP for X" / "find NJP paper on X"
    """
    if not query or not query.strip():
        return "❌ query 不能为空"

    query = query.strip()
    max_results = max(1, min(int(max_results), 200))

    # 解析 journals 参数
    if journals and journals.strip():
        wanted = [j.strip() for j in journals.split(",") if j.strip()]
    else:
        wanted = _OPENALEX_IOP_DEFAULT_QUANT_PH
    source_ids = _resolve_openalex_source_ids(wanted, _OPENALEX_IOP_SOURCES)
    if not source_ids:
        return f"❌ 没有任何合法期刊 short_name (传入={wanted!r}, 可用={list(_OPENALEX_IOP_SOURCES.keys())})"

    # 哪些 short_name 落空了
    resolved_short = set(_OPENALEX_IOP_SOURCES.keys())
    unmatched = [w for w in wanted if w.lower() not in resolved_short]

    try:
        results = _openalex_search(
            query=query,
            source_ids=source_ids,
            max_results=max_results,
        )
    except Exception as e:
        return f"❌ IOP 搜索失败 (OpenAlex): {e}"

    if not results:
        return (
            f"未找到匹配 '{query}' 的 IOP 论文\n"
            f"- 搜索范围: {', '.join(wanted)}\n"
            f"- 建议: 换个关键词, 或放宽 journals (例: 空 = 默认 NJP+J.Phys.A+J.Phys.B+J.Opt.)"
        )

    journal_label = ", ".join(wanted)
    header_lines = [
        f"找到 {len(results)} 篇 IOP 论文 (query='{query}', 期刊: {journal_label}):\n",
    ]
    body = "\n\n".join(_openalex_format_paper(w, i) for i, w in enumerate(results, 1))
    footer = ""
    if unmatched:
        footer = f"\n\n⚠ 未识别的 short_name: {unmatched} (可用的: {list(_OPENALEX_IOP_SOURCES.keys())})"
    return "\n".join(header_lines) + body + footer
