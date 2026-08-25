"""
agent.memory.judger \u2014 \u4e09\u5143\u7ec4\u5224\u65ad\u7b97\u6cd5

\u5224\u65ad\u4e24\u4e2a\u8bb0\u5fc6\u662f\u201c\u540c\u4e00\u4e8b\u5b9e\u201d / \u201c\u5bf9\u7acb\u201d / \u201c\u540c\u4e3b\u9898\u4e0d\u540c\u65b9\u6cd5\u201d / \u201c\u540c\u4e3b\u9898\u540c\u65b9\u6cd5\u201d / \u201c\u4e0d\u76f8\u5173\u201d.

\u8f93\u5165: \u4e24\u4e2a\u62bd\u53d6\u7ed3\u679c dict (extractor.extract \u8fd4\u56de)
\u8f93\u51fa: (verdict, reason) \u5143\u7ec4

verdict \u53d6\u503c:
  - same_fact: \u540c\u4e00\u4e8b\u5b9e (\u53ef\u5408\u5e76)
  - opposite: \u5bf9\u7acb / \u5426\u5b9a (\u4e0d\u80fd\u5408\u5e76, \u8981\u4fdd\u7559)
  - related_diff_method: \u540c\u4e3b\u9898\u4e0d\u540c\u65b9\u6cd5
  - related_shared_category: \u540c\u4e3b\u9898\u7c7b\u522b\u4e0d\u540c\u5177\u4f53\u6982\u5ff5
  - related_partial: \u90e8\u5206\u6982\u5ff5\u91cd\u53e0
  - unrelated: \u4e0d\u76f8\u5173
"""
import logging
from typing import List, Dict, Any, Tuple, Optional, Set

from .extractor import is_pred_symmetric

logger = logging.getLogger(__name__)


def _concept_categories(extract: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """\u8fd4\u56de {(canonical, category), ...}"""
    return {
        (c.get("canonical", ""), c.get("category", ""))
        for c in extract.get("core_concepts", [])
        if c.get("canonical")
    }


def _category_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> Set[str]:
    """\u8fd4\u56de\u5171\u4eab\u7684 category \u96c6\u5408 (\u7528\u4e8e\u5feb\u901f\u8fc7\u6ee4)"""
    a_cats = {c.get("category", "") for c in a.get("core_concepts", [])}
    b_cats = {c.get("category", "") for c in b.get("core_concepts", [])}
    return a_cats & b_cats


def judge(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[str, str]:
    """
    \u4e3b\u5165\u53e3: \u5224\u65ad\u4e24\u4e2a\u62bd\u53d6\u7ed3\u679c\u7684\u5173\u7cfb.

    \u5224\u65ad\u6d41\u7a0b (\u4f18\u5148\u7ea7\u4ece\u9ad8\u5230\u4f4e):
      1. negation \u51b2\u7a81 -> opposite
      2. \u65e0\u5171\u4eab category \u4e14\u65e0\u5171\u4eab\u6982\u5ff5 -> unrelated
      3. \u9010\u4e2a\u5173\u7cfb\u5bf9\u6bd4:
         3a. \u540c subj+pred, \u4e0d\u540c obj -> related_diff_method
         3b. subj/obj/polarity \u5168\u540c -> same_fact
         3c. \u4e3b\u5ba2\u5bf9\u8c03 + \u5bf9\u79f0\u8c13\u8bcd -> same_fact
         3d. \u4e3b\u5ba2\u5bf9\u8c03 + \u975e\u5bf9\u79f0\u8c13\u8bcd -> opposite
         3e. subj/obj \u540c, pred \u4e0d\u540c -> related_diff_predicate
         3f. \u540c (subj, pred), polarity \u4e0d\u540c -> opposite
      4. \u5171\u4eab\u5177\u4f53\u6982\u5ff5 -> related_partial
      5. \u5171\u4eab category -> related_shared_category
      6. \u9ed8\u8ba4 -> unrelated
    """
    a_rels = a.get("relations", [])
    b_rels = b.get("relations", [])

    # 1. negation \u51b2\u7a81
    if a.get("negation") != b.get("negation") and (a.get("negation") or b.get("negation")):
        return "opposite", "negation \u51b2\u7a81"

    # 2. \u65e0\u5171\u4eab category \u4e0e\u4e0a\u4e0b\u6587\u4e0d\u540c
    a_concepts = _concept_categories(a)
    b_concepts = _concept_categories(b)
    shared_concepts = a_concepts & b_concepts
    shared_cats = _category_overlap(a, b)

    if not shared_concepts and not shared_cats:
        return "unrelated", "\u65e0\u5171\u4eab category \u4e5f\u65e0\u5171\u4eab\u5177\u4f53\u6982\u5ff5"

    # 3. \u9010\u5bf9\u5173\u7cfb\u68c0\u67e5
    for r_a in a_rels:
        for r_b in b_rels:
            if r_a["pred"] != r_b["pred"]:
                continue
            s_a, o_a = r_a["subj_canonical"], r_a["obj_canonical"]
            s_b, o_b = r_b["subj_canonical"], r_b["obj_canonical"]
            pol_a = r_a.get("polarity", "support")
            pol_b = r_b.get("polarity", "support")

            # 3a. \u540c subj+pred, \u4e0d\u540c obj -> related_diff_method
            if s_a == s_b and o_a != o_b and pol_a == pol_b:
                return "related_diff_method", (
                    f"\u540c ({s_a}, {r_a['pred']}), \u4e0d\u540c obj ({o_a} vs {o_b})"
                )

            # 3b. subj/obj/polarity \u5168\u540c
            if s_a == s_b and o_a == o_b:
                if pol_a == pol_b:
                    return "same_fact", (
                        f"\u540c ({s_a}, {r_a['pred']}, {o_a}, {pol_a})"
                    )
                # 3f. \u540c (subj, pred, obj), polarity \u4e0d\u540c -> opposite
                return "opposite", (
                    f"\u540c ({s_a}, {r_a['pred']}, {o_a}) \u4f46 polarity \u4e0d\u540c ({pol_a} vs {pol_b})"
                )

            # 3c/3d. \u4e3b\u5ba2\u5bf9\u8c03
            if s_a == o_b and o_a == s_b:
                if is_pred_symmetric(r_a["pred"]):
                    return "same_fact", (
                        f"\u4e3b\u5ba2\u5bf9\u8c03 + \u5bf9\u79f0\u8c13\u8bcd ({r_a['pred']}) = \u540c\u4e00\u4e8b\u5b9e"
                    )
                return "opposite", (
                    f"\u4e3b\u5ba2\u5bf9\u8c03 + \u975e\u5bf9\u79f0\u8c13\u8bcd ({r_a['pred']}) = \u53cd\u5411\u8bc4\u4ef7"
                )

            # 3e. \u540c subj+obj, \u4e0d\u540c pred
            if s_a == s_b and o_a == o_b and r_a["pred"] != r_b["pred"]:
                return "related_diff_predicate", (
                    f"\u540c\u6982\u5ff5\u5bf9, \u4e0d\u540c pred ({r_a['pred']} vs {r_b['pred']})"
                )

    # 4. \u5171\u4eab\u5177\u4f53\u6982\u5ff5
    if shared_concepts:
        return "related_partial", (
            f"\u5171\u4eab\u5177\u4f53\u6982\u5ff5 {shared_concepts}, \u4f46\u5173\u7cfb\u672a\u5339\u914d"
        )

    # 5. \u5171\u4eab category
    if shared_cats:
        return "related_shared_category", (
            f"\u5171\u4eab category {shared_cats}, \u4f46\u5177\u4f53\u6982\u5ff5\u4e0d\u540c"
        )

    return "unrelated", "\u65e0\u6709\u6548\u5173\u7cfb"


def can_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, str]:
    """
    \u9ad8\u5c42\u63a5\u53e3: \u662f\u5426\u53ef\u4ee5\u5408\u5e76 (\u540c\u4e00\u4e8b\u5b9e).

    \u8fd4\u56de: (can_merge: bool, reason: str)
    """
    verdict, reason = judge(a, b)
    if verdict == "same_fact":
        return True, reason
    return False, f"{verdict}: {reason}"
