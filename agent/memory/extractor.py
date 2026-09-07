"""
agent.memory.extractor \u2014 LLM-driven concept & relation extraction

\u4ece\u7528\u6237\u8bed\u53e5\u4e2d\u63d0\u53d6 4 \u4e2a\u5b57\u6bb5:
  - core_concepts: 1-3 \u4e2a\u6838\u5fc3\u6982\u5ff5 + canonical + category
  - relations: 0-3 \u4e2a\u4e09\u5143\u7ec4 (subj, pred, obj, polarity)
  - negation: bool
  - (\u4e0d\u9700\u8981\u5ba1\u91cd\u63d0\u53d6 polarity: relations \u91cc\u9762\u5df2\u542b)

\u8c03\u7528 LLM (deepseek) \u63d0\u53d6, \u5931\u8d25\u4e0d\u62a5\u9519, \u8fd4\u56de\u7a7a dict (\u8c03\u7528\u8005\u8d2f\u5e8f\u5e95\u7ea7 keyword-only).

\u62bd\u53d6\u8d28\u91cf\u4f9d\u8d56 prompt + LLM \u9009\u62e9. \u9ed8\u8ba4 deepseek-chat (\u5feb\u4e14\u4ef7\u5e08).
\u4e0d\u8981\u5728\u5b66\u4e60\u8bb0\u5fc6\u91cc\u5199\u8fd9\u4e2a prompt \u7684\u590d\u5236, \u53cd\u6b63 LLM \u4f1a\u91cd\u8bfb.
"""
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
# \u8bed\u4e49\u8bcd\u8868\uff08symmetric vs asymmetric\uff09
# ============================================================
# symmetric: \u4e3b\u5ba2\u5bf9\u8c03\u540e\u8bed\u4e49\u7b49\u4ef7 (\u4f8b: "A \u53ef\u7528\u4e8e B" \u2248 "B \u53ef\u7528\u4e8e A")
# asymmetric: \u4e3b\u5ba2\u5bf9\u8c03\u540e\u662f\u53cd\u5411\u8bc4\u4ef7 (\u4f8b: "A \u4f18\u4e8e B" \u2260 "B \u4f18\u4e8e A", \u5bf9\u7acb)
SYMMETRIC_PREDS = frozenset({"applies_to", "supports", "about", "has_part", "related_to"})
ASYMMETRIC_PREDS = frozenset({"better_than", "uses", "requires", "converges_to", "implies"})

# \u8bb0\u5fc6\u5e93\u53ef\u7528\u7684\u5168\u90e8\u6807\u51c6\u8c13\u8bcd
ALL_PREDS = SYMMETRIC_PREDS | ASYMMETRIC_PREDS


# ============================================================
# LLM \u62bd\u53d6 prompt
# ============================================================
EXTRACTION_PROMPT = """\u4f60\u662f\u4e00\u4e2a\u79d1\u7814\u8bb0\u5fc6\u5e93\u7684\u6982\u5ff5\u62bd\u53d6\u52a9\u624b\u3002\u4ece\u7528\u6237\u8f93\u5165\u7684\u53e5\u5b50\u4e2d\u63d0\u53d6\u7ed3\u6784\u5316\u5b57\u6bb5, \u4ee5 JSON \u8f93\u51fa\u3002

## \u8f93\u51fa schema
```json
{
  "core_concepts": [
    {"term": "\u6982\u5ff5\u539f\u6587", "canonical": "\u5f52\u4e00\u5316\u5f62\u5f0f(\u82f1\u6587\u5c0f\u5199\u4e0b\u5212\u7ebf)", "category": "\u4e0a\u4f4d\u6982\u5ff5\u7c7b\u578b"}
  ],
  "relations": [
    {
      "subj_canonical": "\u4e3b\u4f53\u5f52\u4e00\u5316",
      "subj_category": "\u4e3b\u4f53\u7c7b\u578b",
      "pred": "\u6807\u51c6\u8c13\u8bcd(\u4ece\u4e0b\u65b9\u8bcd\u8868\u9009)",
      "obj_canonical": "\u5ba2\u4f53\u5f52\u4e00\u5316",
      "obj_category": "\u5ba2\u4f53\u7c7b\u578b",
      "polarity": "support|oppose|neutral"
    }
  ],
  "negation": true|false
}
```

## \u6807\u51c6\u8c13\u8bcd\u8bcd\u8868 (\u5fc5\u987b\u4ece\u8fd9\u4e9b\u91cc\u9009, \u4e0d\u8981\u81ea\u521b)
- applies_to: X \u9002\u7528\u4e8e Y / X \u53ef\u7528\u4e8e Y / X \u5904\u7406 Y / X \u7528\u4e8e Y (symmetric \u5bf9\u79f0)
- converges_to: X \u6536\u655b\u4e8e Y (asymmetric \u975e\u5bf9\u79f0)
- better_than: X \u4f18\u4e8e Y (asymmetric \u975e\u5bf9\u79f0)
- supports: X \u652f\u6301 Y / X \u8bc1\u660e Y (symmetric)
- about: X \u5173\u4e8e Y (symmetric)
- has_part: X \u5305\u542b Y (asymmetric)
- uses: X \u7528 Y / X \u4f7f\u7528 Y (asymmetric)
- requires: X \u9700\u8981 Y / X \u4f9d\u8d56 Y (asymmetric)
- related_to: X \u4e0e Y \u76f8\u5173 (symmetric, \u5151\u5e95)

## \u6838\u5fc3\u6982\u5ff5\u5f52\u4e00\u5316\u89c4\u5219
- \u4e2d\u6587\u672f\u8bed -> \u62fc\u97f3\u4e0b\u5212\u7ebf (\u4f8b: "\u51f8\u4f18\u5316" -> "convex_optimization")
- \u540c\u4e49\u6982\u5ff5\u7528\u540c\u4e00 canonical (\u4f8b: "\u51f8\u4f18\u5316\u95ee\u9898" \u4e5f -> "convex_optimization")
- \u8bba\u6587\u540d\u5e26\u5e74\u4efd (\u4f8b: "Elben2022" -> "elben_2022")
- \u7b97\u6cd5/\u65b9\u6cd5\u540d\u5c0f\u5199\u4e0b\u5212\u7ebf (\u4f8b: "Gilbert\u7b97\u6cd5" -> "gilbert_algorithm")
- category \u4e0e\u4e0a\u4f4d\u6982\u5ff5\u7c7b\u578b: quantum_algorithm / optimization / hardware / paper / method / problem / protocol \u7b49

## \u91cd\u8981
- \u5982\u679c\u662f\u4e8b\u5b9e\u9648\u8ff0 (X \u5173\u7cfb Y), \u5fc5\u586b relations
- \u5982\u679c\u662f\u95f2\u804a/\u65e0\u660e\u786e\u4e8b\u5b9e, relations=[], core_concepts \u4e5f\u8981\u5c3d\u91cf\u62bd
- \u5426\u5b9a\u8bcd("\u4e0d"/"\u6ca1"/"\u9519")\u5728 negation \u5b57\u6bb5
- \u4e3b\u5ba2\u5bf9\u8c03("QAOA \u4f18\u4e8e VQE" vs "VQE \u4f18\u4e8e QAOA")\u7b97\u4e24\u4e2a\u4e0d\u540c\u5173\u7cfb, polarity \u90fd\u662f support
- symmetric \u8c13\u8bcd\u4e3b\u5ba2\u5bf9\u8c03\u540e\u8bed\u4e49\u7b49\u4ef7(\u5982 A \u53ef\u7528\u4e8e B \u2248 B \u53ef\u7528\u4e8e A),asymmetric \u53cd\u4e4b

## \u8f93\u51fa\u8981\u6c42
**\u53ea\u8f93\u51fa JSON**, \u4e0d\u8981\u4efb\u4f55 markdown / \u89e3\u91ca / \u524d\u7f00\u3002"""


# ============================================================
# LLM \u8c03\u7528
# ============================================================
def _call_llm_extract(content: str) -> Optional[Dict[str, Any]]:
    """
    \u8c03 deepseek \u62bd\u53d6\u5b57\u6bb5. \u5931\u8d25\u8fd4\u56de None.
    """
    try:
        from agent.llm_clients import get_client, LLMClientError
        client = get_client("deepseek")
    except Exception as e:
        logger.warning("[MemoryExtractor] \u62ff\u4e0d\u5230 deepseek client: %s", e)
        return None
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content
    except Exception as e:
        logger.warning("[MemoryExtractor] LLM 调用失败: %s", e)
        return None
    # 2026-09-07 P0 fix: 容错 JSON parse (LLM 经常缺 '"}]' 收尾, 同 survey_generate 修法)
    # 之前 json.loads(text) 失败直接返 None, 实际 LLM 输出有 90% 缺 '"}]' 收尾
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # 容错: 补 '"}]' / ']' 收尾 + 抽完整 object 重建
        for tail in ['"}]', '"]', ']']:
            try:
                fixed = text.rstrip(",\n ") + tail
                v = json.loads(fixed)
                if isinstance(v, (dict, list)):
                    logger.info("[MemoryExtractor] 容错 parse 成功 (补 %r)", tail)
                    return v
            except Exception:
                pass
        # 抽单层完整 {...} 对象
        import re as _re
        m = _re.search(r"\{[^{}]*\"canonical\"[^{}]*\}", text)
        if m:
            try:
                return json.loads("[" + m.group(0) + "]")
            except Exception:
                pass
        logger.warning("[MemoryExtractor] JSON parse 失败(含容错): %s; text 末 200 chars: %r", e, text[-200:])
        return None


def _normalize_extract(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    \u5f52\u4e00\u5316 LLM \u8fd4\u56de\u7684\u7ed3\u6784. \u4e22\u6389\u8106\u5f31\u5b57\u6bb5, \u9ed8\u8ba4\u503c\u586b\u4e2a\u5217\u8868.
    """
    if not isinstance(raw, dict):
        return {"core_concepts": [], "relations": [], "negation": False}
    concepts = raw.get("core_concepts", []) or []
    rels = raw.get("relations", []) or []
    # \u8fc7\u6ee4\u4e0d\u5408\u6cd5\u7684 concepts
    clean_concepts = []
    for c in concepts:
        if not isinstance(c, dict):
            continue
        canonical = c.get("canonical", "")
        if not canonical or not isinstance(canonical, str):
            continue
        clean_concepts.append({
            "term": c.get("term", canonical),
            "canonical": canonical.lower().strip(),
            "category": c.get("category", "general") or "general",
        })
    # \u8fc7\u6ee4\u4e0d\u5408\u6cd5\u7684 relations
    clean_rels = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        subj = r.get("subj_canonical", "")
        obj = r.get("obj_canonical", "")
        pred = r.get("pred", "")
        if not (subj and obj and pred):
            continue
        if pred not in ALL_PREDS:
            # LLM \u4ea4\u4e86\u4e0d\u8ba4\u8bc6\u7684\u8c13\u8bcd, \u4e22
            continue
        clean_rels.append({
            "subj_canonical": subj.lower().strip(),
            "subj_category": r.get("subj_category", "general") or "general",
            "pred": pred,
            "obj_canonical": obj.lower().strip(),
            "obj_category": r.get("obj_category", "general") or "general",
            "polarity": r.get("polarity", "support") or "support",
        })
    negation = bool(raw.get("negation", False))
    return {
        "core_concepts": clean_concepts,
        "relations": clean_rels,
        "negation": negation,
    }


def extract(content: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    \u4ece content \u62bd\u53d6 4 \u5b57\u6bb5. \u9ed8\u8ba4\u8c03 LLM, \u53ef\u9009 fallback \u4e0d\u8c03.

    \u8fd4\u56de:
        {
          "core_concepts": [{"term", "canonical", "category"}, ...],
          "relations": [{"subj_canonical", "subj_category", "pred", "obj_canonical", "obj_category", "polarity"}, ...],
          "negation": bool
        }
    \u5931\u8d25\u8fd4\u56de\u7a7a\u7ed3\u6784 (\u4e0d\u62a5\u9519).
    """
    if not content or not content.strip():
        return {"core_concepts": [], "relations": [], "negation": False}
    if not use_llm:
        return {"core_concepts": [], "relations": [], "negation": False}
    raw = _call_llm_extract(content.strip())
    if raw is None:
        return {"core_concepts": [], "relations": [], "negation": False}
    return _normalize_extract(raw)


def is_pred_symmetric(pred: str) -> bool:
    """\u8c13\u8bcd\u662f\u5426\u5bf9\u79f0 (\u4e3b\u5ba2\u5bf9\u8c03\u540e\u8bed\u4e49\u7b49\u4ef7)"""
    return pred in SYMMETRIC_PREDS
