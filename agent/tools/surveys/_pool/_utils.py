"""
_pool._utils — 拆出自原 _pool.py (9-20 体检重构)

包含函数: _arxiv_author_name, _safe_json_loads_array, _extract_paper_ids, _audit_section_paper_ids, _safe_paper_id
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .._paper_id import (
    ARXIV_ID_ALT,
    dedup_key,
    normalize_arxiv_id,
    normalize_doi,
    normalize_paper_id,
)

logger = logging.getLogger(__name__)

# paper_id 抽取用 regex
#
# 9-27 实测修复: 原来 3 条 regex **全都只认 arXiv 格式**, 匹配不到 APS/IOP 的
# DOI 格式 `10.1103/2401.12345` / `10.1103/RevModPhys.90.035006`。
# APS 是主要数据源之一, 于是 APS 引用对审计完全不可见:
#   - in_pool 恒为 0        → 覆盖度审计失真
#   - 幻觉的 APS 引用抓不到 → 审计形同虚设
#   - 实测: 一段 10892 字符的章节因误判触发 2 次重写, 被砍到 3075 字符
#
# 10-04: 三条 arXiv regex 里的 id 形态改成引用 _paper_id.ARXIV_ID_ALT, 让
# "推导侧产出什么形态" 和 "抽取侧认什么形态" 共用同一份定义。以前各写各的,
# 于是旧格式 arXiv(带 archive 前缀)推导得出、却抽不出来 -> in_pool 恒为 0。
_PAPER_ID_SIMPLE_RE = re.compile(rf"\[({ARXIV_ID_ALT})\]")
_PAPER_ID_ARXIV_RE = re.compile(rf"arXiv[:\s]+({ARXIV_ID_ALT})")
_PAPER_ID_PAPERS_MD_RE = re.compile(rf"papers/({ARXIV_ID_ALT})\.md")
# DOI 形式 (APS / IOP / JHEP 等出版商): 10.NNNN/suffix
# 后缀允许字母数字和 . - _ / (部分 DOI 后缀含斜杠, 如 10.1088/1367-2630/aad1ea)
#
# 10-04 改: 原来用 `[^\]\s\)]` 把右括号排掉了, 于是 **JHEP 的 DOI 全被截断** ——
# `10.1007/JHEP11(2013)158` 只抽到 `10.1007/JHEP11`, 跟池子里的 id 对不上 ->
# 判成"引用了池外的幻觉 id" -> 触发重写。JHEP 是高能领域主要刊物之一, 这条对
# 物理用户是实打实的坑。括号改成"方括号内可以随便, 只要不是 ] 或空白"。
_PAPER_ID_DOI_RE = re.compile(r"\[(10\.\d{4,9}/[^\]\s]+)\]")
# 链接形式: papers/<doi>.md —— 后缀非贪婪, 否则会把 ".md" 一起吞进去
_PAPER_ID_DOI_LINK_RE = re.compile(r"papers/(10\.\d{4,9}/[^)\s]*?)\.md")
# 裸 DOI: 中间允许圆括号(JHEP), 但**结尾**必须是词字符, 避免把后面的标点/斜杠/
# 右括号也吃进来; 前面不能是 papers/, 否则会把 papers/<doi>.md 的 ".md" 当成 id
_PAPER_ID_DOI_BARE_RE = re.compile(
    r"(?<!papers/)\b(10\.\d{4,9}/[A-Za-z0-9._/()-]*[A-Za-z0-9._-])\b"
)
# 10-04 加: 无 DOI 的 OpenAlex 文献。_safe_paper_id 的第 3/4 优先级会产出
# `W3104022488` / `OA-<md5>` 两种形态, 之前只在 _organize 的编号侧认, 抽取侧
# 完全不认 -> 正文里挂着 [W3104022488] 而覆盖度审计看不见它。实测存量里有 2 条。
_PAPER_ID_OPENALEX_RE = re.compile(r"\[(W\d+|OA-[0-9a-f]{8,})\]")


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
        _PAPER_ID_OPENALEX_RE,   # 10-04 加
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
    """候选池内/外分桶 (plan §3.2 覆盖度审计核心)

    10-04: 两侧都过一遍 dedup_key 再比。池子里的 id 是 API 推导出来的规范形态,
    正文里的 id 是模型照抄的 —— 大小写 / URL 前缀 / arXiv 版本号都可能不同。
    不归一就比, 同一篇论文会被判成"引用了池外的幻觉 id", 然后触发无意义的
    重写(9-27 那次砍掉 72% 内容的机制)。
    """
    pool_keys = {dedup_key(p) for p in (pool_ids or set()) if p}
    cites = _extract_paper_ids(content_md or "")
    in_pool = []
    out_pool = []
    for c in cites:
        if dedup_key(c) in pool_keys:
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

    10-04: 三个分支全部改走 _paper_id 的归一函数。原来 2/3 两个分支各写各的,
    而且 2 号分支 `.replace(".pdf","")` **保留了版本号** —— 同一篇论文走 arxiv API
    得 `2209.02093`, 走 OpenAlex 这条路得 `2209.02093v1`, 两个 id 双双通过去重,
    于是重复烧一次相关性审查 + 一次 fact 抽取, 参考文献表里出现重复条目。
    """
    doi = normalize_doi(work.get("doi") or "")
    if doi:
        return doi

    # arxiv ID: 从 primary_location.source 或 external_ids 找
    # OpenAlex 实际不会把 arxiv ID 放在常规字段, 这里只是兜底
    for loc_key in ("primary_location", "best_oa_location"):
        loc = work.get(loc_key) or {}
        src = loc.get("source") or {}
        # 有些 source 字段是 OpenAlex source object
        if isinstance(src, dict) and src.get("id") and "arxiv" in (src.get("display_name") or "").lower():
            arxiv_id = normalize_arxiv_id(loc.get("pdf_url") or "")
            if arxiv_id:
                return arxiv_id

    # OpenAlex ID (例 "https://openalex.org/W123456" → "W123456")
    oa_id = normalize_paper_id(work.get("id") or "")
    if oa_id:
        return oa_id

    # 兜底: title hash
    import hashlib
    title = work.get("title") or work.get("display_name") or ""
    return "OA-" + hashlib.md5(title.encode("utf-8")).hexdigest()[:12]
