"""
_pool._utils — 拆出自原 _pool.py (9-20 体检重构)

包含函数: _arxiv_author_name, _safe_json_loads_array, _extract_paper_ids, _audit_section_paper_ids, _safe_paper_id
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# paper_id 抽取用 regex
#
# 9-27 实测修复: 原来 3 条 regex **全都只认 arXiv 格式**, 匹配不到 APS/IOP 的
# DOI 格式 `10.1103/2401.12345` / `10.1103/RevModPhys.90.035006`。
# APS 是主要数据源之一, 于是 APS 引用对审计完全不可见:
#   - in_pool 恒为 0        → 覆盖度审计失真
#   - 幻觉的 APS 引用抓不到 → 审计形同虚设
#   - 实测: 一段 10892 字符的章节因误判触发 2 次重写, 被砍到 3075 字符
_PAPER_ID_SIMPLE_RE = re.compile(r"\[(\d{4,5}\.\d{4,5}|\d{7})\]")
_PAPER_ID_ARXIV_RE = re.compile(r"arXiv[:\s]+(\d{4,5}\.\d{4,5}|\d{7})")
_PAPER_ID_PAPERS_MD_RE = re.compile(r"papers/(\d{4,5}\.\d{4,5}|\d{7})\.md")
# DOI 形式 (APS / IOP 等出版商): 10.NNNN/suffix
# 后缀允许字母数字和 . - _ / (部分 DOI 后缀含斜杠, 如 10.1088/1367-2630/aad1ea)
_PAPER_ID_DOI_RE = re.compile(r"\[(10\.\d{4,9}/[^\]\s\)]+)\]")
# 链接形式: papers/<doi>.md —— 后缀非贪婪, 否则会把 ".md" 一起吞进去
_PAPER_ID_DOI_LINK_RE = re.compile(r"papers/(10\.\d{4,9}/[^)\s]*?)\.md")
# 裸 DOI: 结尾必须是词字符, 避免把后面的标点/斜杠也吃进来;
# 前面不能是 papers/, 否则会把 papers/<doi>.md 里的 ".md" 也当成 id
_PAPER_ID_DOI_BARE_RE = re.compile(
    r"(?<!papers/)\b(10\.\d{4,9}/[A-Za-z0-9._/-]*[A-Za-z0-9._-])\b"
)
def _arxiv_author_name(a: Any) -> str:
    """
    兼容不同 arxiv 库的 Author 表示 (9-18 修复: 'Author' object has no attribute 'text').

    历史:
      - 旧 arxiv (< 1.x): Author 可能有 .name / .text / str 多种
      - arxiv 2.4.0 (current): Author 只有 .name (str), .text 抛 AttributeError
      - 偶发: 某些 fork (arxiv-py) Author 可能是 str 直接

    这里按"先试 .name, 再试 .text, 否则 str() 兜底"的顺序拿名字,
    任何一步抛 AttributeError 都跳过, 绝不传播给上层。
    """
    if a is None:
        return ""
    if isinstance(a, str):
        return a.strip()
    for attr in ("name", "text"):
        try:
            v = getattr(a, attr, None)
        except AttributeError:
            continue
        if v:
            return str(v).strip()
    # 兜底: 直接 str() (Author.__str__ 通常就是名字)
    try:
        s = str(a).strip()
    except Exception:
        return ""
    # str() 给的是 Python 默认 repr (例 '<...EmptyAuthor object at 0x...>'),
    # 这种没用, 当空处理 (避免污染 keywords)
    if s.startswith("<") and "object at 0x" in s:
        return ""
    return s


def _safe_json_loads_array(text: str) -> list | None:
    """
    容错 parse JSON 数组 (LLM 经常在 closing `"}]` 前就停, P6.2 实测多次踩)。

    策略 (逐级):
      1. 直接 json.loads
      2. 补 '"}]' / ']' 等简单收尾
      3. truncate 到最后一个完整 `{}` 之前, 补 `]`
      4. regex 抽所有单层完整 {...} 对象重建数组

    返 None 表示解析失败 (不是空数组)
    """
    if not text:
        return None
    s = text.strip()
    # 1. 直接
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):  # 偶尔 LLM 返单 object 不返 array
            return [v]
    except (json.JSONDecodeError, SyntaxError):
        pass
    # 2. 补 "}]
    for tail in ['"}]', '"]', '"', '}', ']']:
        try:
            v = json.loads(s.rstrip(",\n ") + tail)
            if isinstance(v, list):
                return v
        except (json.JSONDecodeError, SyntaxError):
            pass
    # 3. truncate 到最后一个 } (P6.2 4b 案例: 末尾是 "y" 缺 }, 截掉)
    last_brace = s.rfind('}')
    if last_brace > 0:
        try:
            v = json.loads(s[: last_brace + 1] + ']')
            if isinstance(v, list):
                return v
        except (json.JSONDecodeError, SyntaxError):
            pass
    # 4. regex 抽所有单层完整 {...} 对象 (不允许嵌套 {})
    obj_re = re.compile(r'\{[^{}]*\}', re.DOTALL)
    objs = obj_re.findall(s)
    if objs:
        try:
            v = json.loads("[" + ",".join(objs) + "]")
            if isinstance(v, list):
                return v
        except (json.JSONDecodeError, SyntaxError):
            pass
    return None


def _extract_paper_ids(text: str) -> list[str]:
    """从 markdown 文本抽所有 paper_id 引用, 各种格式都识别, 去重保序\n\n    9-27: 补上 DOI 形式(APS/IOP)。原来只认 arXiv, 导致 APS 引用对审计\n    完全不可见 —— 实测会把 10892 字符的章节误判触发重写并砍到 3075。"""
    if not text:
        return []
    # 9-27: 按**文档出现顺序**返回。原来是逐条 pattern 遍历, 结果按格式分组
    # (所有 arXiv 排前面, 所有 DOI 排后面), 与 docstring 说的"保序"不符,
    # 覆盖度报告里读起来也乱。改成收集 (位置, id) 后统一按位置排序。
    hits: list[tuple[int, str]] = []
    for pattern in (
        _PAPER_ID_SIMPLE_RE,
        _PAPER_ID_DOI_RE,
        _PAPER_ID_ARXIV_RE,
        _PAPER_ID_PAPERS_MD_RE,
        _PAPER_ID_DOI_LINK_RE,
        _PAPER_ID_DOI_BARE_RE,
    ):
        for m in pattern.finditer(text):
            hits.append((m.start(), m.group(1)))
    seen: set[str] = set()
    out: list[str] = []
    for _pos, pid in sorted(hits, key=lambda t: t[0]):
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def _audit_section_paper_ids(content_md: str, pool_ids: set) -> dict[str, list[str]]:
    """候选池内/外分桶 (plan §3.2 覆盖度审计核心)"""
    cites = _extract_paper_ids(content_md or "")
    in_pool = []
    out_pool = []
    for c in cites:
        if c in pool_ids:
            in_pool.append(c)
        else:
            out_pool.append(c)
    return {
        "in_pool": sorted(set(in_pool)),
        "out_of_pool": sorted(set(out_pool)),
        "total_cites": len(cites),
    }


def _safe_paper_id(work: dict) -> str:
    """
    把 OpenAlex work 转成 pool paper_id.

    优先级:
      1. DOI (例 "10.1103/PhysRevLett.123.100501" → "10.1103/PhysRevLett.123.100501")
         (DOI 是稳定唯一 ID, 不会跟 arxiv ID 冲突)
      2. arxiv ID (从 primary_location.source 或 authorships 抓, OpenAlex 部分 work 有)
      3. OpenAlex ID (例 "W123456" → "W123456")
      4. title hash (兜底, 不会重复但不可读)

    返回 paper_id (字符串).
    """
    doi = (work.get("doi") or "").replace("https://doi.org/", "").strip()
    if doi:
        return doi

    # arxiv ID: 从 primary_location.source 或 external_ids 找
    # OpenAlex 实际不会把 arxiv ID 放在常规字段, 这里只是兜底
    for loc_key in ("primary_location", "best_oa_location"):
        loc = work.get(loc_key) or {}
        src = loc.get("source") or {}
        # 有些 source 字段是 OpenAlex source object
        if isinstance(src, dict) and src.get("id") and "arxiv" in (src.get("display_name") or "").lower():
            arxiv_id = (loc.get("pdf_url") or "").split("/")[-1].replace(".pdf", "")
            if arxiv_id:
                return arxiv_id

    # OpenAlex ID (例 "https://openalex.org/W123456" → "W123456")
    oa_id = (work.get("id") or "").replace("https://openalex.org/", "").strip()
    if oa_id:
        return oa_id

    # 兜底: title hash
    import hashlib
    title = work.get("title") or work.get("display_name") or ""
    return "OA-" + hashlib.md5(title.encode("utf-8")).hexdigest()[:12]
