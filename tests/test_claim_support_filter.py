# -*- coding: utf-8 -*-
"""命题层出处过度归属 —— 支撑性过滤的 unit test (10-08)

背景
----
范式层(condense_facts_by_paradigm)把同一主题下**属于一条技术路线**的 fact
全吸进一条命题, 于是 `claim["paper_ids"]` 装的是"这条路线涉及的所有论文",
而不是"真正支撑这句陈述的论文"。

实测(workspace 89309897a97c, 42 条命题)后果:
    q1 陈述 = "由 Huang/Kueng/Preskill 提出的**原始**全局 Clifford 阴影协议"
    实际挂了 20 篇论文 —— 其中 16 篇只是**拿这个协议去做别的事**
    (贝叶斯分析 / 自适应泡利阴影 / bootstrap / 浅层阴影 / 费米子 / 张量网络 ...)
    写成正文就是"HKP 提出了原始协议"这句话后面跟 20 个引用,
    字面上等于说 20 个团队各自独立发现了这个协议。

分布(证明是定点 bug 而非普遍现象): 42 条命题 paper_ids 中位数 = 1,
只有 q1(20 篇) 和 q5(6 篇) 离群。

本测试锁定要实现的四件事:
  1. 只挂真正支撑的论文
  2. **不动 fact_ids**(技术路线归属仍然成立, fact 覆盖率不能因此掉)
  3. 判定失败/解析失败 -> 原样返回, 绝不静默降级成空
  4. 论文数不超过阈值的命题根本不触发 LLM(成本闸门)
"""
from __future__ import annotations

import inspect
import json

import pytest

from agent.tools.surveys import _paradigm as P


# ---------------------------------------------------------------- 纯函数

def _claims_fixture():
    """真实 q1 缩影: 一条 4 篇论文的命题, 只有 1 篇真正支撑。"""
    facts = [
        {"fact_id": "f1", "paper_id": "p_hkp", "claim": "HKP 提出的经典阴影协议用随机全局 Clifford"},
        {"fact_id": "f2", "paper_id": "p_hkp", "claim": "以 O(log M) 次测量预测 M 个态函数"},
        {"fact_id": "f3", "paper_id": "p_bayes", "claim": "经典阴影方法为有限测量条件下的量子估计提供机会"},
        {"fact_id": "f4", "paper_id": "p_bootstrap", "claim": "经典阴影是一种用极少测量构建近似经典描述的方法"},
    ]
    return [{
        "claim_id": "q1",
        "statement": "随机Clifford测量的经典阴影协议: HKP 提出的原始经典阴影协议",
        "paper_ids": ["p_hkp", "p_bayes", "p_bootstrap"],
        "fact_ids": ["f1", "f2", "f3", "f4"],
    }], facts


def test_support_filter_keeps_only_supports():
    """只保留判为 S 的论文。"""
    claims, facts = _claims_fixture()
    titles = {"p_hkp": "Original Protocol", "p_bayes": "Bayesian", "p_bootstrap": "Bootstrap"}

    out = P.filter_claim_support(
        claims, facts, paper_titles=titles,
        call=lambda prompt: '{"judgments":[{"i":1,"v":"S"},{"i":2,"v":"U"},{"i":3,"v":"U"}]}',
    )
    assert out[0]["paper_ids"] == ["p_hkp"]


def test_support_filter_does_not_touch_fact_ids():
    """关键: fact_ids 一个都不能动 —— 技术路线归属仍然成立。"""
    claims, facts = _claims_fixture()
    out = P.filter_claim_support(
        claims, facts,
        call=lambda p: '{"judgments":[{"i":1,"v":"S"},{"i":2,"v":"U"},{"i":3,"v":"U"}]}',
    )
    assert out[0]["fact_ids"] == ["f1", "f2", "f3", "f4"], \
        "过滤出处不能连带删掉 fact, 否则 fact 覆盖率会凭空掉一截"


def test_support_filter_skips_small_claims():
    """成本闸门: 论文数 <= 阈值的命题压根不调 LLM。

    反向验证 M6 把 `if len(pids) < min_papers` 改成 `if len(pids) < 0` 时, 断言
    必须能接住 —— 关键是**断言 call 一次都没被调用**, 而不只是"结果看起来对"。
    闸门失效时结果照样对(小命题本来也只有 1-2 篇), 但每次运行会多烧几十次
    LLM 调用。所以这里要盯的是调用次数。
    """
    claims = [{"claim_id": "q1", "statement": "s", "paper_ids": ["a", "b"], "fact_ids": []}]
    calls = []

    def _boom(prompt):
        calls.append(prompt)
        raise AssertionError("小命题不该触发 LLM")

    out = P.filter_claim_support(claims, [], call=_boom)
    assert calls == [], f"小命题不该调 LLM, 实际调了 {len(calls)} 次"
    assert out[0]["paper_ids"] == ["a", "b"], "跳过的命题必须原样返回"


def test_support_filter_gate_scales_with_min_papers():
    """门槛是参数: 给 5 就该跳过 4 篇的, 给 2 就该处理 3 篇的。"""
    facts = [{"fact_id": f"f{i}", "paper_id": f"p{i}", "claim": "x"} for i in range(4)]
    claims = [{"claim_id": "q1", "statement": "s",
               "paper_ids": ["p0", "p1", "p2", "p3"],
               "fact_ids": ["f0", "f1", "f2", "f3"]}]

    calls = []
    P.filter_claim_support(
        claims, facts, min_papers=5,
        call=lambda p: (calls.append(p), '{"judgments":[]}')[1])
    assert calls == [], "min_papers=5 时 4 篇的命题必须跳过"

    calls2 = []
    P.filter_claim_support(
        claims, facts, min_papers=2,
        call=lambda p: (calls2.append(p), '{"judgments":[{"i":1,"v":"S"}]}')[1])
    assert len(calls2) == 1, "min_papers=2 时 4 篇的命题必须真的调 LLM"


def test_support_filter_keeps_original_on_llm_failure():
    """LLM 抛异常 -> 原样返回, 不能降级成空引用(宁可吵不可静默)。"""
    claims, facts = _claims_fixture()

    def _raise(prompt):
        raise RuntimeError("network down")

    out = P.filter_claim_support(claims, facts, call=_raise)
    assert out[0]["paper_ids"] == ["p_hkp", "p_bayes", "p_bootstrap"], \
        "LLM 挂了必须原样保留, 而不是悄悄清空出处"


def test_support_filter_keeps_original_on_unparseable_output():
    """输出无法解析 -> 原样返回。"""
    claims, facts = _claims_fixture()
    out = P.filter_claim_support(claims, facts, call=lambda p: "我觉得这些都挺相关的")
    assert out[0]["paper_ids"] == ["p_hkp", "p_bayes", "p_bootstrap"]


def test_support_filter_ignores_bad_indices():
    """模型混入越界/乱填编号时, 它们不能变成出处。

    必须给全 3 篇都表态, 否则会先撞上"必须全覆盖"的护栏提前返回,
    测不到编号校验本身。
    """
    claims, facts = _claims_fixture()
    out = P.filter_claim_support(
        claims, facts,
        call=lambda p: '{"judgments":[{"i":1,"v":"S"},{"i":2,"v":"U"},{"i":3,"v":"U"},'
                       '{"i":99,"v":"S"},{"i":0,"v":"S"},{"i":"x","v":"S"},'
                       '{"i":-1,"v":"S"},{"i":true,"v":"S"}]}',
    )
    assert out[0]["paper_ids"] == ["p_hkp"], "越界/乱填编号不能变成出处"


def test_support_filter_never_returns_empty_for_nonempty_input():
    """若模型把所有论文都判 U, 不能产出空 paper_ids —— 那会让该句直接被丢掉。"""
    claims, facts = _claims_fixture()
    out = P.filter_claim_support(
        claims, facts,
        call=lambda p: '{"judgments":[{"i":1,"v":"U"},{"i":2,"v":"U"},{"i":3,"v":"U"}]}',
    )
    assert out[0]["paper_ids"], "全判 U 时应退回原集合, 不能留空"
    assert out[0]["paper_ids"] == ["p_hkp", "p_bayes", "p_bootstrap"]


def test_support_filter_records_removed_papers():
    """被移除的论文要留痕, 否则事后无法审计为什么某篇没被引。"""
    claims, facts = _claims_fixture()
    out = P.filter_claim_support(
        claims, facts,
        call=lambda p: '{"judgments":[{"i":1,"v":"S"},{"i":2,"v":"U"},{"i":3,"v":"U"}]}',
    )
    assert set(out[0]["support_removed_pids"]) == {"p_bayes", "p_bootstrap"}
    assert out[0]["support_judged"] == 3


def test_support_filter_uses_only_own_facts_not_whole_library():
    """回归: 只把本条命题自己的 fact 喂给模型。

    踩过的坑 —— 用全库 fact 建 paper->内容 的索引, 于是每篇论文"看起来"在讲
    命题说的那件事(其实那些 fact 是它在**别的命题**里贡献的)。实测 q1 的 prompt
    从 3653 字符涨到 8543, 20 篇里 12 篇被判 S, 收紧彻底失效。
    真实运行(workspace 89309897a97c)里 q1 探针判 4 篇 S, 带 bug 判 12 篇。
    """
    facts = [
        {"fact_id": "f1", "paper_id": "p_hkp", "claim": "HKP 提出的原始经典阴影协议"},
        # p_noise 在 q1 里只贡献这一条, 明显是"使用"原始协议去做别的
        {"fact_id": "f9", "paper_id": "p_noise", "claim": "纠错经典阴影估计 k 阶约化密度矩阵"},
        # 同属 p_noise, 但只属于 q2 —— 绝不能混进 q1 的判定输入
        {"fact_id": "z1", "paper_id": "p_noise", "claim": "经典阴影协议由 HKP 于 2020 年提出并阐释其性质"},
    ]
    claims = [{
        "claim_id": "q1",
        "statement": "HKP 提出的原始经典阴影协议",
        "paper_ids": ["p_hkp", "p_noise"],
        "fact_ids": ["f1", "f9"],
    }]
    seen = {}

    def _call(prompt):
        seen["p"] = prompt
        return '{"judgments":[{"i":1,"v":"S"},{"i":2,"v":"U"}]}'

    # min_papers=2: 默认门槛是 3, 这条只有 2 篇会被**跳过** —— 那样断言会
    # 假绿(根本没走到判定逻辑)。显式降到 2 才能真正测到输入范围。
    out = P.filter_claim_support(claims, facts, min_papers=2, call=_call)
    assert out[0]["paper_ids"] == ["p_hkp"]
    assert "2020 年提出并阐释" not in seen["p"], \
        "别的命题的 fact 混进了判定输入, 会让论文显得比实际更相关"


def test_support_filter_handles_empty_input():
    assert P.filter_claim_support([], []) == []
    claims = [{"claim_id": "q1", "statement": "s", "paper_ids": [], "fact_ids": []}]
    out = P.filter_claim_support(claims, [])
    assert out[0]["paper_ids"] == []


# ------------------------------------------------- prompt 必须三分类且无弃权

def test_support_prompt_demands_all_papers_and_no_abstain():
    """prompt 铁律: 必须覆盖全部编号。

    这是实测踩出来的: 第一版写了"判不准的不要选", 20 篇只挑出 1 篇(召回 1/4)。
    改成强制三选一后精确命中 4/4、零误判。
    """
    src = P.SUPPORT_JUDGE_PROMPT
    assert "必须对全部" in src
    assert "一篇都不能漏" in src
    assert "S" in src and "U" in src and "X" in src


# ------------------------------------------------------ 接进范式层主流程

def test_paradigm_claims_call_support_filter():
    """接线护栏: condense_facts_by_paradigm 必须过一遍支撑过滤。"""
    src = inspect.getsource(P) if hasattr(P, "condense_facts_by_paradigm") else ""
    from agent.tools.surveys import _organize as OG
    src = inspect.getsource(OG.condense_facts_by_paradigm)
    assert "filter_claim_support" in src


def test_filter_exported():
    assert hasattr(P, "filter_claim_support")


def test_support_filter_rejects_out_of_range_index():
    """越界编号必须被丢掉, 不能挤掉真出处。

    反向验证 M7 把 `if not (1 <= idx <= n)` 改成 `if False` 时, 原本这里没有任何
    断言能接住 —— 直到模型吐出 {"i": 99, "v": "U"} 这种坏编号, 越界项如果被接受
    就会进 keep 之外或错误覆盖真实判定。
    """
    claims = [{
        "claim_id": "q1", "statement": "s",
        "paper_ids": ["p1", "p2", "p3"], "fact_ids": ["f1", "f2", "f3"],
    }]
    facts = [
        {"fact_id": "f1", "paper_id": "p1", "claim": "a"},
        {"fact_id": "f2", "paper_id": "p2", "claim": "b"},
        {"fact_id": "f3", "paper_id": "p3", "claim": "c"},
    ]
    # i=99 越界判 S, i=0 越界判 S —— 若不校验, keep 会变成 p1,p2,p3
    out = P.filter_claim_support(
        claims, facts,
        call=lambda p: '{"judgments":[{"i":1,"v":"S"},{"i":2,"v":"U"},'
                       '{"i":3,"v":"U"},{"i":99,"v":"S"},{"i":0,"v":"S"}]}',
    )
    assert out[0]["paper_ids"] == ["p1"], "越界编号必须被忽略, 不能变成出处"


def test_support_filter_rejects_invalid_verdict_value():
    """非法判定值(如 "SUP")必须被丢掉, 而不是当 S 用。

    反向验证 M8 把别名表换成首字母截断时就是被这条接住的 ——
    "SUP" 被砍成 "S", 第二篇就成了假支撑的出处。
    三篇都表态, 避免先撞上"必须全覆盖"提前返回。
    """
    claims = [{
        "claim_id": "q1", "statement": "s",
        "paper_ids": ["p1", "p2", "p3"], "fact_ids": ["f1", "f2", "f3"],
    }]
    facts = [
        {"fact_id": "f1", "paper_id": "p1", "claim": "a"},
        {"fact_id": "f2", "paper_id": "p2", "claim": "b"},
        {"fact_id": "f3", "paper_id": "p3", "claim": "c"},
    ]
    # p2 的 "SUP" 非法 -> 被丢弃 -> p2 不是支撑; p1/p3 合法判定
    out = P.filter_claim_support(
        claims, facts,
        call=lambda p: '{"judgments":[{"i":1,"v":"S"},{"i":2,"v":"SUP"},{"i":3,"v":"U"}]}',
    )
    assert out[0]["paper_ids"] == ["p1"], "非法判定值不能被首字母截断成 S"


def test_support_filter_rejects_bool_index():
    """bool 是 int 的子类, True 会被当成编号 1 —— 必须显式排除。"""
    from agent.tools.surveys._paradigm import _parse_support_judgments
    got = _parse_support_judgments(
        '{"judgments":[{"i":true,"v":"U"},{"i":1,"v":"S"}]}', n=2)[0]
    assert got == {1: "S"}, "i=true 不该被当成编号 1(那会把 U 覆盖成 S)"


def test_support_filter_requires_full_coverage():
    """模型只判了一部分 -> 原样保留, 不能把"没提到"当成"仅使用"。

    这是真 bug(写测试时才发现): 未被判的编号 `verdicts.get(i)` 返回 None != "S",
    于是被 keep 直接排除 —— "模型没提到"被当成了"判为仅使用", 静默删出处。
    漏判和判 U 是两回事, 必须区分。
    """
    claims = [{
        "claim_id": "q1", "statement": "s",
        "paper_ids": ["p1", "p2", "p3"], "fact_ids": ["f1", "f2", "f3"],
    }]
    facts = [
        {"fact_id": "f1", "paper_id": "p1", "claim": "a"},
        {"fact_id": "f2", "paper_id": "p2", "claim": "b"},
        {"fact_id": "f3", "paper_id": "p3", "claim": "c"},
    ]
    # 只判了 1 篇, p2/p3 没表态
    out = P.filter_claim_support(
        claims, facts,
        call=lambda p: '{"judgments":[{"i":1,"v":"S"}]}',
    )
    assert out[0]["paper_ids"] == ["p1", "p2", "p3"], \
        "漏判不等于判 U, 必须原样保留"


def test_support_filter_applies_when_fully_covered():
    """对照组: 覆盖完整时正常收紧(证明上一条不是被"永不收紧"蒙混过关)。"""
    claims = [{
        "claim_id": "q1", "statement": "s",
        "paper_ids": ["p1", "p2", "p3"], "fact_ids": ["f1", "f2", "f3"],
    }]
    facts = [
        {"fact_id": "f1", "paper_id": "p1", "claim": "a"},
        {"fact_id": "f2", "paper_id": "p2", "claim": "b"},
        {"fact_id": "f3", "paper_id": "p3", "claim": "c"},
    ]
    out = P.filter_claim_support(
        claims, facts,
        call=lambda p: '{"judgments":[{"i":1,"v":"S"},{"i":2,"v":"U"},{"i":3,"v":"U"}]}',
    )
    assert out[0]["paper_ids"] == ["p1"]


def test_support_filter_gate_skips_declared_but_factless_papers():
    """第二道闸门: 声明挂在命题上、但没贡献 fact 的论文不算候选。

    反向验证 M6 把 pids 闸门改成 `if False` 时, 真正兜住的是这道 cand 闸门 ——
    所以 M6 的变异要同时打穿两道才咬得住。这里锁住 cand 闸门本身。
    """
    calls = []

    def _call(prompt):
        calls.append(prompt)
        return '{"judgments":[{"i":1,"v":"S"}]}'

    # 声明 3 篇(够过 pids 闸门=3), 但只有 p1 有 fact -> cand 只有 1 篇
    claims = [{
        "claim_id": "q1", "statement": "s",
        "paper_ids": ["p1", "p2", "p3"], "fact_ids": ["f1"],
    }]
    facts = [{"fact_id": "f1", "paper_id": "p1", "claim": "a"}]
    out = P.filter_claim_support(claims, facts, call=_call)
    assert calls == [], "候选不足门槛时不该调 LLM"
    assert out[0]["paper_ids"] == ["p1", "p2", "p3"], "跳过的命题原样返回"


def test_threshold_constant_is_sane():
    """阈值太小会把正常命题也拖进 LLM 判定(成本), 太大则放过头。"""
    assert P.SUPPORT_FILTER_MIN_PAPERS >= 3, \
        "1-2 篇的命题本来就该是精确归属, 不需要再花一次 LLM 去确认"