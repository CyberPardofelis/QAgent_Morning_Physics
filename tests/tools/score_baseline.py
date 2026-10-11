"""把 RewardScorer.txt 的打分器接进项目, 并给出**现状基线**。

## 为什么先做这个

用户的目标是「60 题全部正确 + 过程质量分最大化」。现在这套工具是
单发式的(生成代码→跑→判→重试), **没有类型化轨迹**, 所以：
  · 打分器里 method_diversity / cross_validation / exploration / certification
    这四项**全都拿不到分**(轨迹里根本没有对应 step)
  · 也就是说现在的分数反映的是「架构差距」, 不是「题做得好不好」

所以第一件事不是写新循环, 是**先量出这个差距有多大**, 否则改完也不知道
有没有变好。量完再决定投多少。

## 用法

    python tests/tools/score_baseline.py --suite R          # 只跑 R 20 题
    python tests/tools/score_baseline.py --suite all        # 60 题
    python tests/tools/score_baseline.py --offline          # 不调 LLM, 只统计已有证据
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 打分器本体 —— 原样引入, 不改它的权重和判定逻辑
import trace_adapter as _adapter
import reward_scorer as RS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUITES = {
    "base": "baseproblems.txt",
    "V2": "baseproblems_V2.txt",
    "R": "baseproblems_R.txt",
}
WORK = os.path.join("data", "research", "working")


def load_suite(name: str) -> list[dict]:
    raw = io.open(os.path.join(ROOT, SUITES[name]), encoding="utf-8").read()
    return json.loads(raw[raw.find("{"):raw.rfind("}") + 1])["items"]


def to_task_spec(item: dict) -> RS.TaskSpec:
    """题库条目 -> 打分器的 TaskSpec。

    ★ recommended_tools 要从题库搬过来, 但只是给打分器做"工具匹配度"参照,
      **不进 agent 的 prompt** —— 否则就成了给答案的提示。★
    """
    return RS.TaskSpec(
        id=item["id"],
        prompt=item["prompt"],
        recommended_tools=item.get("recommended_tools", []),
        expected_methods=item.get("expected_methods", []),
        requires_certificate="certificate" in (item.get("challenge_type") or [])
                              or "certificate" in (item.get("tags") or []),
        tags=item.get("tags", []),
    )


#: ★ 如实翻译 ★ 实现见 trace_adapter.py(单独一个文件, 因为它有独立测试)。
#:
#: 旧实现在这里把 tool 恒填 "python"、method 恒填 "brute_force", 导致
#: exploration / cross_validation 这两个维度测的是**适配器的词汇表**而不是
#: agent 的研究行为 —— 实测"贴两个假 experiment 标签"就能白赚 17.5 分。
trace_from_evidence = _adapter.trace_from_evidence
