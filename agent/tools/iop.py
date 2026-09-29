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

from agent.tools.openalex import (
    _DEFAULT_TIMEOUT as _OPENALEX_TIMEOUT,
)
from agent.tools.openalex import (
    _RETRY_ON_5XX as _OPENALEX_RETRY,
)
from agent.tools.openalex import (
    OPENALEX_IOP_DEFAULT_QUANT_PH as _OPENALEX_IOP_DEFAULT_QUANT_PH,
)
from agent.tools.openalex import (
    OPENALEX_IOP_SOURCES as _OPENALEX_IOP_SOURCES,
)
from agent.tools.openalex import (
    format_paper_markdown as _openalex_format_paper,
)

# 2026-09-09 P0.6: 跟 search_aps 共用 OpenAlex 客户端
# 9-18 加: 顺手导入 timeout/retry 常量, 错误信息能告诉用户具体配置
from agent.tools.openalex import (
    openalex_search as _openalex_search,
)
from agent.tools.openalex import (
    resolve_source_ids as _resolve_openalex_source_ids,
)
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

from agent.infra.console import safe_print  # 控制台 cp936 编不了 emoji 时不崩


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
        safe_print("[search_iop] ❌ query 为空", flush=True)
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
        safe_print(f"[search_iop] ❌ 没有任何合法期刊 short_name (传入={wanted!r})", flush=True)
        return (
            f"❌ IOP 搜索失败: 没有任何合法期刊 short_name\n"
            f"- 传入: {wanted}\n"
            f"- 可用 short_name: {list(_OPENALEX_IOP_SOURCES.keys())}\n"
            f"- 提示: 用逗号分隔, 例: 'njp,jphysa,jopt'"
        )

    # 哪些 short_name 落空了
    resolved_short = set(_OPENALEX_IOP_SOURCES.keys())
    unmatched = [w for w in wanted if w.lower() not in resolved_short]

    short_query = (query[:60] + "...") if len(query) > 60 else query
    safe_print(
        f"[search_iop] start  query={short_query!r}  journals={wanted}  "
        f"max_results={max_results}  source_ids={len(source_ids)}个",
        flush=True,
    )

    try:
        results = _openalex_search(
            query=query,
            source_ids=source_ids,
            max_results=max_results,
            caller="search_iop",     # 9-18 加: 控制台标签用 search_iop
        )
    except ValueError as e:
        safe_print(f"[search_iop] ❌ 入参错: {e}", flush=True)
        return f"❌ IOP 搜索失败 (入参): {e}"
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("[timeout]"):
            return (
                f"❌ IOP 搜索超时 (OpenAlex)\n"
                f"- 网络慢或被墙, >{_OPENALEX_TIMEOUT}s 没响应\n"
                f"- 建议: 1) 检查网络/VPN  2) 稍后重试  3) max_results 调小"
            )
        if msg.startswith("[conn]"):
            return (
                "❌ IOP 搜索失败: 网络连不上 OpenAlex\n"
                "- 可能是: DNS 失败 / 断网 / 防火墙\n"
                "- 建议: 1) 检查网络  2) 试 web_search 兜底"
            )
        if msg.startswith("[http 4xx]"):
            return (
                f"❌ IOP 搜索被 OpenAlex 拒绝 (HTTP 4xx)\n"
                f"- 通常是 query 写法有问题或触发了 OpenAlex 限流\n"
                f"- 详情: {msg}\n"
                f"- 建议: 1) 简化 query  2) 去掉特殊字符  3) 等几秒重试"
            )
        if msg.startswith("[http 5xx]"):
            return (
                f"❌ IOP 搜索失败: OpenAlex 服务端连续错 (HTTP 5xx)\n"
                f"- OpenAlex 自己挂了, 已重试 {_OPENALEX_RETRY} 次\n"
                f"- 建议: 1) 稍后重试  2) 试 web_search 兜底"
            )
        if msg.startswith("[json]"):
            return (
                f"❌ IOP 搜索失败: OpenAlex 返回非 JSON\n"
                f"- 服务端可能挂了 / 限流\n"
                f"- 详情: {msg}"
            )
        return f"❌ IOP 搜索失败 (OpenAlex): {msg}"

    if not results:
        safe_print(
            f"[search_iop] ⚠ 0 results  query={short_query!r}  journals={wanted}",
            flush=True,
        )
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
    safe_print(
        f"[search_iop] done  返回 {len(results)} 篇  journals={wanted}",
        flush=True,
    )
    return "\n".join(header_lines) + body + footer
