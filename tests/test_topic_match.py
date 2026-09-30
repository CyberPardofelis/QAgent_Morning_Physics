"""
test_topic_match.py — 主题短语匹配 (9-30)

核心: 防止"quantum"+"network" 这种拆词匹配把量子网络/QKD 误召成张量网络。
"""
from __future__ import annotations

import importlib
import sys


def _mod(name: str):
    if name not in sys.modules:
        importlib.import_module(name)
    return sys.modules[name]


TM = _mod("agent.tools.surveys._pool._topic_match")

TOPIC = "tensor network states quantum simulation"


# ============================================================
# 切词
# ============================================================

def test_split_makes_bigrams_and_trigrams():
    words, phrases = TM.split_topic(TOPIC)
    assert "tensor" in words and "network" in words
    assert "tensor network" in phrases
    assert "tensor network states" in phrases
    assert "quantum simulation" in phrases


def test_split_chinese_uses_terms_directly():
    words, phrases = TM.split_topic("张量网络态 量子模拟")
    assert words == ["张量网络态", "量子模拟"]
    assert phrases == words, "中文不做 n-gram, 片段本身就是短语"


def test_split_empty():
    assert TM.split_topic("") == ([], [])
    assert TM.split_topic("   ") == ([], [])


def test_single_word_has_no_phrases():
    words, phrases = TM.split_topic("holography")
    assert words == ["holography"]
    assert phrases == [], "单词主题切不出短语"


# ============================================================
# 准入: 核心行为
# ============================================================

def test_true_positive_qualifies():
    t = "Matrix product states and projected entangled pair states: Conformal field theory"
    assert TM.qualifies(TOPIC, t + " tensor network representations in two dimensions")


def test_quantum_network_is_rejected():
    """
    回归核心: 旧逻辑会因为同时命中 'quantum' + 'network' 放行,
    但这篇讲的是量子网络, 不是张量网络。
    """
    t = "A quantum network stack and protocols for reliable entanglement distribution"
    assert "quantum" in t and "network" in t, "前提: 单词都命中"
    assert not TM.qualifies(TOPIC, t), "'quantum network' 不该被当成 'tensor network'"


def test_qkd_is_rejected():
    t = "Photon-efficient quantum key distribution using time-energy entanglement"
    assert not TM.qualifies(TOPIC, t)


def test_neural_network_states_accepted_by_design():
    """神经网络态会放行, 这是**有意为之**, 不是漏网。

    主题切出的 bigram "network states" 确实出现在 "neural network states" 里,
    词法层面区分不了。但神经网络态本就是张量网络的竞品方案, 综述里该拿来对照 ——
    交给下游 LLM 审查判相关性, 而不是这里用词法替它做判断。
    只要求分数明显低于真·张量网络, 让排序仍把它排后面。
    """
    t = "Quantum Entanglement in Neural Network States"
    real = "Tensor network state correspondence and holography"
    assert TM.qualifies(TOPIC, t), "神经网络态有意放过"
    assert TM.match_score(TOPIC, t) < TM.match_score(TOPIC, real), \
        "放过但分数要明显低于真·张量网络, 让排序仍然把它排后面"


def test_holography_only_phrase_hits():
    t = "Tensor network state correspondence and holography"
    assert TM.qualifies(TOPIC, t)


def test_single_word_topic_falls_back_to_word():
    # 无词干还原: 主题词必须原样出现。"holography" 匹配不上 "holographic",
    # 这是刻意的 —— 词形还原风险大于收益。
    assert TM.qualifies("holography", "Holography and the entropy of black holes")
    assert not TM.qualifies("holography", "quantum key distribution")


def test_plural_forms_match():
    """单复数: 主题写 states, 标题写 state, 最核心的三元组不能因此不命中。"""
    assert TM.qualifies(TOPIC, "Tensor network state correspondence")
    assert TM.qualifies("quantum simulations of spin chains", "quantum simulation protocol")


def test_chinese_topic_keeps_old_behavior():
    assert TM.qualifies("张量网络态", "本文研究张量网络态在量子模拟中的应用")
    assert not TM.qualifies("张量网络态", "本文研究量子密钥分发协议的安全性")


def test_empty_text_never_qualifies():
    assert not TM.qualifies(TOPIC, "")
    assert not TM.qualifies(TOPIC, "   ")


def test_empty_topic_is_permissive():
    assert TM.qualifies("", "anything at all")


# ============================================================
# 分数: 只在合格之后用来排序
# ============================================================

def test_score_range_and_ordering():
    hi = TM.match_score(TOPIC, "tensor network states for quantum simulation of lattices")
    mid = TM.match_score(TOPIC, "quantum simulation with tensor network methods")
    lo = TM.match_score(TOPIC, "a review of tensor network algorithms")
    assert 0.0 <= lo <= mid <= hi <= 1.0


def test_score_zero_when_nothing_matches():
    assert TM.match_score(TOPIC, "photon-efficient key distribution protocol") == 0.0


def test_score_empty_inputs():
    assert TM.match_score("", "x") == 0.0
    assert TM.match_score(TOPIC, "") == 0.0


# ============================================================
# entry_text
# ============================================================

def test_entry_text_concatenates_fields():
    e = {"title": "T", "core_method": "M", "core_contrib": "C",
         "abstract": "A", "keywords": "K", "outlook": "O"}
    t = TM.entry_text(e)
    for v in "TMCAKO":
        assert v in t


def test_entry_text_skips_missing_fields():
    t = TM.entry_text({"title": "OnlyTitle"})
    assert t.strip() == "OnlyTitle"


def test_entry_text_roundtrip_qualifies():
    e = {"title": "Tensor network state correspondence and holography",
         "abstract": "we discuss tensor network states"}
    assert TM.qualifies(TOPIC, TM.entry_text(e))


def test_entry_text_rejects_quantum_network():
    e = {"title": "A quantum network stack",
         "abstract": "quantum network protocols for entanglement"}
    assert not TM.qualifies(TOPIC, TM.entry_text(e))


# ============================================================
# 实测回归: 旧逻辑 vs 新逻辑
# ============================================================

def test_prunes_88pct_noise_on_real_library():
    """
    真实数据回归 (本地 summary_items, 9-30 实测):
    旧"任一单词命中" 218/247 = 88%; 新短语准入 12/247 = 5%。
    这里不依赖真实库, 用实测抓到的代表样本锁定行为。
    """
    noise = [
        "A quantum network stack and protocols for reliable entanglement distribution",
        "Photon-efficient quantum key distribution using time-energy entanglement",
        "Multi-partite entanglement can speed up quantum key distribution",
        "The quantum technologies roadmap: a European community view",
        "Control of quantum phenomena: past, present and future",
    ]
    for t in noise:
        assert not TM.qualifies(TOPIC, t), f"该放行的噪声: {t}"

    # 注意: 这里给的是**完整 entry 文本**(标题+摘要), 跟管线里 entry_text() 一致。
    # 只喂标题会误判 —— 很多真论文标题里没有 "tensor network" 字样,
    # 但摘要里有(ER ansatz / PEPS 那类都是在摘要里才点明用张量网络)。
    signal = [
        ("Tensor network state correspondence and holography",
         "We establish the correspondence between tensor network states and holography."),
        ("Causal structure of the entanglement renormalization ansatz",
         "We analyze the tensor network parameters of the ER ansatz and their fixed points."),
    ]
    for title, abstract in signal:
        t = f"{title} {abstract}"
        assert TM.qualifies(TOPIC, t), f"该毙掉的正例: {title}"


def test_known_limitation_synonym_vocabulary_dropped():
    """
    **如实锁定的已知漏召**: 短语匹配认不出同义术语。

    "Matrix product states and PEPS: Conformal field theory" 是张量网络领域的核心
    综述, 但标题里没有 "tensor network" 这个词, 用的�� MPS / PEPS 这套术语,
    短语门控会把它剪掉。

    这是词法匹配的固有上限, 不打算用硬编码同义词表去补(那等于把领域知识焊死在
    通用工具里)。留给引文图扩展去补 —— 种子过审之后沿引用边走, MPS/PEPS 那类论文
    自然会被带出来, 不依赖词法。

    这个测试的作用是: 哪天有人想改门控逻辑时, 会看到这里的代价是什么。
    """
    t = "Matrix product states and projected entangled pair states: Conformal field theory"
    assert not TM.qualifies(TOPIC, t), "当前行为: 标题认不出 MPS 术语"
    # 但只要**任意字段**(通常摘要)提到 tensor network, 就能进 —— 这才是管线实际行为
    assert TM.qualifies(
        TOPIC,
        t + " We review the tensor network representation of these states.",
    ), "摘要里提了就该放行, 不该误杀"
