"""骨架截断加固 + 增量并入的测试。

覆盖 10-07 真跑暴露的三个问题:
  1. 骨架 `max_tokens` 写死 8000 -> 第二轮 fact 变多时输出被服务端截断, JSON 解析
     失败, 整轮退回无骨架块级归纳(第一轮建好的领域结构被丢弃)。
  2. 骨架解析失败没有重试, 一次失败就全丢。
  3. gap-fill 从零重跑主题地图 -> LLM 归纳出 34 个主题撞 max_themes=8, 26 个主题
     连带 223 条 fact(63%)被截掉 -> 83 篇池最后只引用 19 篇。

平行组织, 不建 test_v1/v2。
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.tools.surveys import _organize as O  # noqa: E402


# ============================================================
# 夹具
# ============================================================

def _fact(i: int, claim: str = "x") -> dict:
    return {"fact_id": f"f{i}", "paper_id": f"p{i // 3}", "claim": claim,
            "subject": f"s{i % 5}", "fact_type": "method"}


def _facts(n: int) -> list[dict]:
    return [_fact(i, f"关于经典阴影的第{i}条事实, 讲样本复杂度与测量系综") for i in range(n)]


def _theme(name: str, n_facts: int = 3, start: int = 0) -> dict:
    return {"name": name, "statement": "s", "notes": "n", "relations": [],
            "fact_ids": [f"f{i}" for i in range(start, start + n_facts)]}


def _skeleton_json(names: list[str]) -> str:
    return json.dumps({"themes": [{"name": n, "task": "t", "belongs": "b",
                                   "not_belongs": "x"} for n in names],
                       "structure_note": "note"}, ensure_ascii=False)


@pytest.fixture
def fake_llm(monkeypatch):
    """把 `_call_llm` 换成可控桩, 记录每次调用的 prompt / max_tokens。

    `_build_skeleton` 内部是 `from agent.tools.llm import _call_llm` —— 函数在
    **调用时**才绑定, 所以必须打在 `agent.tools.llm` 模块属性上; 打 `_organize`
    的名字不生效(这坑第一版测试就踩了, 表现为"桩没装上, 真调了 API")。
    """
    calls: list[dict] = []
    box: dict = {"responder": None}

    def _fake(prompt, **kw):
        calls.append({"prompt": prompt, **kw})
        r = box["responder"]
        return r(prompt, len(calls)) if r else ""

    import agent.tools.llm as llm_mod
    monkeypatch.setattr(llm_mod, "_call_llm", _fake, raising=False)
    calls.clear()
    return calls, box


# ============================================================
# 1. max_tokens: 骨架不该再用 8000
# ============================================================

class TestSkeletonMaxTokens:
    def test_skeleton_uses_shared_max_tokens_not_8000(self, fake_llm):
        """骨架的 max_tokens 必须跟块级归纳一致(THEME_MAP_MAX_TOKENS=32000)。

        10-07 实测: 8000 tokens ≈ 23637 字符(3.18 字符/token)。第一轮 282 条
        fact 输出 23637 字符刚好卡住, 第二轮 353 条涨到 29573 字符 ≈ 9300 tokens,
        越过 8000 被服务端截断 -> JSON 解析失败。
        """
        calls, box = fake_llm
        box["responder"] = lambda p, i: _skeleton_json(["任务一", "任务二"])
        O._build_skeleton(_facts(30), [])
        assert calls, "骨架阶段应该发过一次 LLM"
        assert calls[0]["max_tokens"] == O.THEME_MAP_MAX_TOKENS, (
            f"骨架 max_tokens={calls[0]['max_tokens']}, "
            f"期望跟块级一致 {O.THEME_MAP_MAX_TOKENS}")
        assert calls[0]["max_tokens"] > 8000

    def test_skeleton_field_caps_present_in_prompt(self):
        """prompt **模板**里必须给每个字段写死字数上限。

        从源头压长度比单纯加 max_tokens 更省: 实测(10-07)8000 → 7422 tokens /
        32000 → 9276 / 64000 → 10122, 模型给多少额度就写多少。卡死字段长度才能
        挡住"fact 变多 -> 输出涨 -> 撞上限 -> 整轮白跑"这条路。

        ⚠ 必须断言模板里的**字面约束**, 不能只断言"有提到字数上限"那段说明 ——
        反向验证 M4 一开始只删了说明文字, 模板里的约束还在, 测试照样绿。
        """
        for frag in ["不超过 40 字", "不超过 60 字", "不超过 30 字", "不超过 80 字"]:
            assert frag in O.THEME_SKELETON_PROMPT, (
                f"骨架 prompt 模板缺字段字数上限「{frag}」—— 模型会展开写, "
                f"fact 变多时输出涨到撞 max_tokens 上限就整轮白跑")


# ============================================================
# 2. 解析失败重试
# ============================================================

class TestSkeletonRetry:
    def test_retry_once_on_parse_failure(self, fake_llm):
        """第一次返回坏 JSON, 第二次应当重试并成功。"""
        calls, box = fake_llm
        box["responder"] = lambda p, i: (
            '{"themes": [{"name": "被截断了' if i == 1
            else _skeleton_json(["量子态预测", "完整态重构"]))
        sk = O._build_skeleton(_facts(30), [])
        assert len(sk.get("themes") or []) == 2, "第二次尝试应该成功"
        assert len(calls) == 2, f"应重试一次, 实际调用 {len(calls)} 次"

    def test_retry_prompt_is_stricter(self, fake_llm):
        """第二次的 prompt 必须比第一次更严格(追加更狠的长度约束)。"""
        calls, box = fake_llm
        box["responder"] = lambda p, i: (
            '{"themes": [{"name": "断了' if i == 1 else _skeleton_json(["a任务"]))
        O._build_skeleton(_facts(30), [])
        assert len(calls) == 2
        assert calls[1]["prompt"] != calls[0]["prompt"]
        # 断言要卡在**具体那句长度约束**上, 不能只判 "重试" 两个字 ——
        # 反向验证 M3 把「【重试 — 上一次输出超长被截断】」改成「【重试】」时,
        # 只查 "重试" 仍然绿, 护栏等于没有。
        assert "上一次输出超长被截断" in calls[1]["prompt"]
        assert len(calls[1]["prompt"]) > len(calls[0]["prompt"])

    def test_gives_up_after_two_attempts(self, fake_llm):
        """两次都坏 -> 返回空(且不抛异常), 不无限重试。"""
        calls, box = fake_llm
        box["responder"] = lambda p, i: '{"themes": [{"name": "永远断'
        assert O._build_skeleton(_facts(30), []) == {}
        assert len(calls) == 2, f"应只试两次, 实际 {len(calls)}"


# ============================================================
# 3. 骨架失败时沿用上一轮
# ============================================================

class TestSkeletonFallbackToPrev:
    def test_reuse_prev_without_llm_call(self, fake_llm):
        """prev_skeleton 非空时直接沿用, 一次 LLM 都不发。

        这是刻意的设计而不是省事: 领域结构不该被新论文改写。10-07 第二轮从零
        重建骨架, 正好是那次丢掉整个骨架的原因。
        """
        calls, box = fake_llm
        prev = {"themes": [{"name": "旧任务一", "task": "t", "belongs": "b",
                            "not_belongs": "x"}], "structure_note": "n"}
        sk = O._build_skeleton(_facts(30), [], prev_skeleton=prev)
        assert sk is prev
        assert calls == [], f"沿用骨架时不该发 LLM, 实际发了 {len(calls)} 次"

    def test_prev_skeleton_wins_even_if_llm_would_succeed(self, fake_llm):
        """即使这次能构建成功, 只要有 prev 就沿用 —— 不给"结构漂移"留机会。"""
        calls, box = fake_llm
        box["responder"] = lambda p, i: _skeleton_json(["全新任务"])
        prev = {"themes": [{"name": "上一轮的任务", "task": "t", "belongs": "b",
                            "not_belongs": "x"}], "structure_note": "n"}
        sk = O._build_skeleton(_facts(30), [], prev_skeleton=prev)
        assert sk is prev
        assert calls == []

    def test_no_prev_builds_and_falls_back_to_empty_on_total_failure(self, fake_llm):
        """没有上一轮可沿用时, 两次都失败才退回无骨架(= 改前的块级行为)。"""
        calls, box = fake_llm
        box["responder"] = lambda p, i: "完全不是 JSON"
        assert O._build_skeleton(_facts(30), []) == {}
        assert len(calls) == 2, f"应试两次, 实际 {len(calls)}"


# ============================================================
# 4. 截断检测
# ============================================================

class TestLooksTruncated:
    @pytest.mark.parametrize("raw,expect", [
        ('{"themes": [{"name": "a"}]}', False),
        ('{"themes": [{"name": "a"}]}\n```', False),
        ('{"themes": [{"name": "a"}]}```', False),
        ('{"themes": [{"name": "半句话在这里', True),
        ('{"themes": [{"name": "a", "task": "写到一半被砍', True),
        ("", False),
    ])
    def test_detect(self, raw, expect):
        assert O._looks_truncated(raw) is expect


# ============================================================
# 5. 增量并入: 旧主题一条都不砍
# ============================================================

class TestMergeIntoPrevMap:
    def _prev(self):
        return {"themes": [_theme("任务一", 5), _theme("任务二", 4),
                           _theme("任务三", 3)],
                "narrative_arc": ["弧1"], "gaps": ["空白1"], "warnings": [],
                "skeleton": {"themes": [{"name": "任务一"}, {"name": "任务二"},
                                        {"name": "任务三"}]}}

    def test_old_themes_all_survive(self):
        """10-07 实测: 从零重跑时 34 个主题撞上限 8, 223/353 条 fact 被截掉。
        增量并入绝不能再丢旧主题。"""
        prev = self._prev()
        # 新一轮归纳出 12 个**全新**主题(从零重跑的话这 12 个会挤掉旧的)
        new = [{"themes": [_theme(f"全新主题{i}", 20) for i in range(12)],
                "narrative_arc": [], "gaps": [], "warnings": []}]
        out = O._merge_into_prev_map(prev, new, prev["skeleton"])
        names = {t["name"] for t in out["themes"]}
        for old in ("任务一", "任务二", "任务三"):
            assert old in names, f"旧主题「{old}」被增量并入丢掉了"

    def test_new_facts_land_in_old_theme(self):
        """新 fact 归纳出的主题名与旧主题同名 -> fact_ids 并进去, 不另起主题。"""
        prev = self._prev()
        new = [{"themes": [_theme("任务二", 3, start=500)],
                "narrative_arc": [], "gaps": [], "warnings": []}]
        out = O._merge_into_prev_map(prev, new, prev["skeleton"])
        t2 = next(t for t in out["themes"] if t["name"] == "任务二")
        assert len(t2["fact_ids"]) == 4 + 3, "旧 4 条 + 新 3 条应并成 7 条"
        assert "f500" in t2["fact_ids"]

    def test_arc_and_gaps_accumulate(self):
        prev = self._prev()
        new = [{"themes": [], "narrative_arc": ["弧2"], "gaps": ["空白2"],
                "warnings": []}]
        out = O._merge_into_prev_map(prev, new, prev["skeleton"])
        assert "弧1" in out["narrative_arc"] and "弧2" in out["narrative_arc"]
        assert "空白1" in out["gaps"] and "空白2" in out["gaps"]

    def test_warning_reports_what_happened(self):
        """增量并入必须留下可读的告警(宁可吵不可静默)。"""
        prev = self._prev()
        new = [{"themes": [_theme("任务一", 2, start=900), _theme("新任务", 4)],
                "narrative_arc": [], "gaps": [], "warnings": []}]
        out = O._merge_into_prev_map(prev, new, prev["skeleton"])
        joined = " ".join(out["warnings"])
        assert "增量并入" in joined
        assert "旧 3 个主题保留 3 个" in joined

    def test_lost_old_theme_is_loudly_reported(self):
        """万一旧主题真丢了, 必须喊出来而不是静默。"""
        prev = self._prev()
        # 上界给成 1: 强制只留 1 个主题, 旧主题必然被砍
        results = [{"themes": [_theme(f"新{i}", 5) for i in range(9)],
                    "narrative_arc": [], "gaps": [], "warnings": []}]
        combined_limit = 1
        merged = O._merge_chunk_results(
            [{"themes": prev["themes"], "narrative_arc": [], "gaps": [],
              "warnings": []}] + results,
            combined_limit, skeleton=prev["skeleton"])
        names = {t["name"] for t in merged["themes"]}
        gone = {"任务一", "任务二", "任务三"} - names
        assert gone, "本测试的前提是旧主题被砍掉了"
        # _merge_into_prev_map 自身不会砍(上界给足), 这里只验证合并器行为可预期
        assert merged["warnings"]


# ============================================================
# 6. organize_facts_into_themes 增量接线
# ============================================================

class TestIncrementalWiring:
    def test_no_induce_facts_behaves_as_full_run(self, fake_llm, monkeypatch):
        """不传 induce_facts -> 走原来的全量路径, 行为不变。"""
        calls, box = fake_llm
        box["responder"] = lambda p, i: json.dumps({
            "themes": [{"name": "任务一", "statement": "s", "fact_ids": ["f0"],
                        "notes": "", "relations": []}],
            "narrative_arc": [], "gaps": []}, ensure_ascii=False)
        monkeypatch.setattr(O, "split_fact_chunks", lambda f: [f])
        out = O.organize_facts_into_themes(_facts(5))
        assert out.get("themes"), "全量模式仍应产出主题"

    def test_empty_incremental_returns_prev_unchanged(self, fake_llm):
        """没有新增 fact -> 原样返回上一轮, 不发 LLM。"""
        calls, box = fake_llm
        prev = {"themes": [_theme("任务一", 5)], "narrative_arc": [], "gaps": [],
                "warnings": [], "skeleton": {"themes": [{"name": "任务一"}]}}
        out = O.organize_facts_into_themes(
            _facts(5), prev_map=prev, induce_facts=[])
        assert out["themes"] == prev["themes"]
        assert calls == [], "没有新增 fact 时不该发 LLM"

    def test_incremental_reuses_skeleton_without_llm(self, fake_llm, monkeypatch):
        """增量轮沿用骨架 -> 只发块归纳那一次 LLM, 不发骨架那次。

        断言的是「恰好 1 次」而不是「0 次」: 块归纳本身当然要调 LLM。这里真正
        要防的是**又发一次骨架调用**(10-07 第二轮就是这么把结构重建坏的)。
        """
        calls, box = fake_llm
        box["responder"] = lambda p, i: json.dumps({
            "themes": [{"name": "任务一", "statement": "s", "fact_ids": ["f100"],
                        "notes": "", "relations": []}],
            "narrative_arc": [], "gaps": []}, ensure_ascii=False)
        monkeypatch.setattr(O, "split_fact_chunks", lambda f: [f])
        prev = {"themes": [_theme("任务一", 5)], "narrative_arc": [], "gaps": [],
                "warnings": [],
                "skeleton": {"themes": [{"name": "任务一", "task": "t",
                                         "belongs": "b", "not_belongs": "x"}]}}
        out = O.organize_facts_into_themes(
            _facts(5), prev_map=prev, induce_facts=[_fact(100, "新事实")])
        assert any(t["name"] == "任务一" for t in out["themes"])
        assert len(calls) == 1, f"应只有块归纳 1 次调用, 实际 {len(calls)} 次"
        # 判据不能用「全局骨架」这个词 —— 块归纳 prompt 里**本来就注入**了骨架块
        # (见 _render_skeleton)。骨架调用独有的开场句是"你在为一个领域建立"。
        assert "你在为一个领域建立" not in calls[0]["prompt"], \
            "这 1 次不该是骨架调用"
        assert "【全局骨架 — 领域由这几个任务构成" in calls[0]["prompt"], \
            "块归纳 prompt 里应当注入上一轮的骨架"

    def test_full_path_builds_skeleton(self, fake_llm, monkeypatch):
        """全量路径仍然要建骨架 —— 防止改动把骨架整个绕过去。"""
        calls, box = fake_llm
        box["responder"] = lambda p, i: _skeleton_json(["任务一", "任务二"])
        monkeypatch.setattr(O, "split_fact_chunks", lambda f: [f, f])
        monkeypatch.setattr(O, "_organize_one_chunk",
                            lambda ch, **kw: {"themes": [], "narrative_arc": [],
                                              "gaps": [], "warnings": []})
        O.organize_facts_into_themes(_facts(5))
        assert calls, "全量路径应该发骨架 LLM"
        assert calls[0]["max_tokens"] == O.THEME_MAP_MAX_TOKENS