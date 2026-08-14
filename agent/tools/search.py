"""
联网搜索工具 (RAG 场景优化版，移植自 WAgent)。

设计:
- 多 provider 架构: bocha(中文主推) / tavily(海外主推) / ddg(保底, 免 key)
- 按 query 中英文自动选 provider, 也可手动指定
- 默认开启正文摘要 (RAG 场景需要, 直接喂 embedding)
- 缺 key 时自动降级: bocha -> tavily -> ddg

API key 通过 agent.infra.api_keys 管理, key 名: bocha / tavily
"""
import json
import logging
import re
from typing import List, Dict, Optional
import requests
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================
def _get_api_key(provider: str) -> str:
    """延迟 import 避免循环依赖"""
    from agent.infra.api_keys import get_api_key
    return get_api_key(provider)


def _truncate(text: str, max_length: int) -> str:
    """截断文本并加省略号"""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


# ============================================================
# Provider 实现
# ============================================================
def search_bocha(query: str, num_results: int, max_content_length: int) -> List[Dict]:
    """
    博查搜索 API (国内主推, 中文源强)
    文档: https://open.bochaai.com/
    Endpoint: https://api.bochaai.com/v1/web-search
    """
    api_key = _get_api_key("bocha")
    if not api_key:
        raise RuntimeError("BOCHA_API_KEY 未配置")

    url = "https://api.bochaai.com/v1/web-search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "summary": True,           # 关键: 让博查直接返回网页摘要
        "count": min(num_results, 50),  # 博查 count 上限 50
        "freshness": "noLimit",    # 不过滤时效
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != 200:
        raise RuntimeError(f"博查 API 错误: code={data.get('code')}, msg={data.get('msg')}")

    pages = data.get("data", {}).get("webPages", {}).get("value", []) or []
    results: List[Dict] = []
    for p in pages[:num_results]:
        content = p.get("summary") or p.get("snippet") or ""
        results.append({
            "title": p.get("name", ""),
            "link": p.get("url", ""),
            "snippet": p.get("snippet", ""),
            "content": _truncate(content, max_content_length),
            "source": p.get("siteName", ""),
            "date": p.get("dateLastCrawled", ""),
            "language": p.get("language", ""),
            "provider": "bocha",
        })
    return results


def search_tavily(query: str, num_results: int, fetch_content: bool, max_content_length: int) -> List[Dict]:
    """
    Tavily 搜索 API (海外主推, 英文源强)
    文档: https://docs.tavily.com/
    Endpoint: https://api.tavily.com/search
    """
    api_key = _get_api_key("tavily")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY 未配置")

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": min(num_results, 20),  # Tavily max 20
        "search_depth": "advanced",          # advanced 召回更准
        "include_answer": False,             # 不要 LLM 总结, 我们要原始数据
        "include_raw_content": fetch_content,
    }

    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results: List[Dict] = []
    for r in data.get("results", [])[:num_results]:
        content = r.get("raw_content") or r.get("content") or ""
        results.append({
            "title": r.get("title", ""),
            "link": r.get("url", ""),
            "snippet": _truncate(r.get("content", ""), 200),
            "content": _truncate(content, max_content_length),
            "source": "",
            "score": r.get("score"),
            "provider": "tavily",
        })
    return results


def search_ddg(query: str, num_results: int, fetch_content: bool, max_content_length: int) -> List[Dict]:
    """
    DuckDuckGo HTML 搜索 (保底, 免 key, 但国内经常超时)
    保留作为最后兜底
    """
    from bs4 import BeautifulSoup

    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    result_divs = soup.find_all("div", class_="result")
    results: List[Dict] = []

    for div in result_divs[:num_results]:
        link_elem = div.find("a", class_="result__a")
        if not link_elem:
            continue
        title = link_elem.get_text(strip=True)
        link = link_elem.get("href")
        if link and link.startswith("/"):
            link = "https://duckduckgo.com" + link
        elif link and link.startswith("//"):
            link = "https:" + link
        snippet_elem = div.find("a", class_="result__snippet")
        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

        results.append({
            "title": title,
            "link": link,
            "snippet": snippet,
            "content": "",
            "source": "",
            "provider": "ddg",
        })

    if fetch_content:
        for r in results:
            if r["link"]:
                r["content"] = _fetch_webpage_content(r["link"], max_content_length)

    return results


def _fetch_webpage_content(url: str, max_length: int) -> str:
    """网页正文抓取 (DDG fallback 路径)"""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text:
                cleaned = " ".join(text.split())
                return _truncate(cleaned, max_length)
    except ImportError:
        try:
            from bs4 import BeautifulSoup
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                cleaned = " ".join(text.split())
                return _truncate(cleaned, max_length)
        except Exception:
            pass
    except Exception:
        pass
    return ""


# ============================================================
# Provider 选择 (auto 模式)
# ============================================================
_CN_RE = re.compile(r'[\u4e00-\u9fff]')


def _is_chinese(text: str) -> bool:
    return bool(_CN_RE.search(text))


def _provider_status() -> Dict[str, bool]:
    """检查各 provider 是否可用 (key 已配)"""
    return {
        "bocha": bool(_get_api_key("bocha")),
        "tavily": bool(_get_api_key("tavily")),
    }


def _select_provider(query: str, requested: str) -> str:
    """
    选择最终使用的 provider
    - requested == "auto": 按 query 语言 + key 可用性自动选, 带降级
    - requested 显式指定: 检查 key 存在
    """
    status = _provider_status()
    requested = (requested or "auto").lower()

    if requested != "auto":
        if requested in ("bocha", "tavily") and not status.get(requested):
            raise RuntimeError(f"要求使用 {requested} 但未配置 API key, 请先在设置里填 key")
        if requested not in ("bocha", "tavily", "ddg", "auto"):
            raise RuntimeError(f"未知 provider: {requested}, 可选 bocha/tavily/ddg/auto")
        return requested

    # auto 模式
    is_cn = _is_chinese(query)
    if is_cn:
        if status["bocha"]:
            return "bocha"
        if status["tavily"]:
            return "tavily"
        return "ddg"
    else:
        if status["tavily"]:
            return "tavily"
        if status["bocha"]:
            return "bocha"
        return "ddg"


# ============================================================
# 工具入口
# ============================================================
@register_tool(name="web_search", is_streaming=False)
def web_search(
    query: str,
    num_results: int = 5,
    fetch_content: bool = True,
    max_content_length: int = 2000,
    provider: str = "auto",
) -> str:
    """
    联网搜索 (RAG 场景优化版)。

    自动根据 query 中英文选择搜索源 (中文走博查, 英文走 Tavily, 都不可用时降级 DuckDuckGo)。
    默认返回网页正文摘要, 便于直接喂给 RAG 流程做 chunking 和 embedding。
    """
    try:
        chosen = _select_provider(query, provider)
        logger.info("web_search: query=%r provider=%s", query, chosen)

        if chosen == "bocha":
            results = search_bocha(query, num_results, max_content_length)
        elif chosen == "tavily":
            results = search_tavily(query, num_results, fetch_content, max_content_length)
        else:
            results = search_ddg(query, num_results, fetch_content, max_content_length)

        status = _provider_status()
        return json.dumps({
            "query": query,
            "provider": chosen,
            "count": len(results),
            "results": results,
            "providers_available": status,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("web_search 失败")
        return json.dumps({"error": f"搜索失败: {e}"}, ensure_ascii=False)
