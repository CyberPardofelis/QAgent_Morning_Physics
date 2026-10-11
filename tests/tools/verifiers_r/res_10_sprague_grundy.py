"""res-010-sprague-grundy 的独立 verifier。

题目: 对 data/res_10_rules.txt 给定的取石子游戏, 计算 n=0..999 的 SG 值,
给出周期规律, 并用两套独立实现交叉验证。

★ 规则从文件读, SG 值由 verifier **自己算** ★
data/res_10_rules.txt 说: 单堆, 每次取 1、3 或 4 颗, 正常规则。
SG 递推是教科书定义:
    g(0) = 0
    g(n) = mex { g(n-m) : m ∈ moves, m <= n }
verifier 从规则文件里**正则抽 moves**(不是写死 [1,3,4] —— 规则文件才是
题目的输入, 写死等于无视输入; 也顺带防了"规则文件改了判据还按老规则算")。

★ 判据 ★
  1. 制品里的 sg_values 必须是 1000 个非负整数, 且与 verifier 自算的
     g[0..999] **逐项相等**。为什么逐项: SG 值没有"差不多", g(n) 差 1
     就是错(它直接决定胜负判定), 而"周期对但前缀错"是常见错误来源
     (很多人只报周期, 前几十项靠猜)。
  2. agent 报的 period 必须被 verifier 自算的序列验证:
     - period p 是合法周期: g[n] == g[n+p] 对**所有** onset <= n <= 999-p 成立
     - p 必须是**最小**周期: 若存在更小的合法 p', 报的 p 就不叫周期规律
       (报 7*1000 = 7000 也满足 g[n]==g[n+7000], 但那是废话不是规律)
     - 允许 agent 自报 onset(周期性起点); 不报就当 0, 但 verifier 会
       记录自己算出的最小 onset 供诊断。
  3. data/res_10_reference.json 只做**交叉核对**: 它的 sg_1000 / period
     与 verifier 自算的结果一起写进 details。若两者不一致, 以 verifier
     自算的为准(判据基于自算), 并在 details 里标出来 —— 参考文件是
     数据方的产物, 不是判据。
"""
from __future__ import annotations

import json
import os
import re
import sys

try:
    from .common import Check, parse_artifact, parse_verdict
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import Check, parse_artifact, parse_verdict

#: 题库 verifier.params.states
N_STATES = 1000
_SEQ_LEN = N_STATES            # g[0..999]
#: 周期上限: 拿一个"随便的"大数当周期, 检查它是不是合法的最小周期
_MAX_PERIOD = 1000

_RULES_REL = os.path.join("data", "res_10_rules.txt")
_REFERENCE_REL = os.path.join("data", "res_10_reference.json")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOTS = [
    os.path.abspath(os.path.join(_HERE, "..", "..", "..")),
    os.getcwd(),
]

_SG_CACHE: dict[tuple, list[int]] = {}


def _find(rel: str) -> str | None:
    for root in _ROOTS:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    return None


def read_moves() -> tuple[list[int], str]:
    """
    从规则文件里抽出可取的石子数。返回 (moves, 来源说明)。

    认得的写法: "每次只能取 1、3 或 4 颗" / "可取 1, 3, 4" / "moves: [1,3,4]"。
    抽不到就退回 [1,3,4](规则文件的默认表述), 并在 details 里说明。
    """
    p = _find(_RULES_REL)
    if p is None:
        return [1, 3, 4], f"规则文件缺失, 退回默认 moves=[1,3,4]({_RULES_REL})"
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        return [1, 3, 4], f"规则文件读不了({e}), 退回默认 moves=[1,3,4]"

    # 优先找"每次只能取 1、3 或 4 颗"这一句里的数字。
    # ★ 必须从"取"一直吃到"颗"(或行尾)★ 只抓数字类字符会在"或"处截断,
    #   把 moves=[1,3,4] 读成 [1,3] —— 那是把规则读错了, 判据会跟着错。
    m = re.search(r"取\s*([^。\n]*?)(?:颗|个|粒|颗石子)", text)
    cand = m.group(1) if m else None
    if cand is None:
        m2 = re.search(r"moves?\s*[:=]\s*\[([^\]]*)\]", text, re.IGNORECASE)
        cand = m2.group(1) if m2 else None
    if cand is None:
        return [1, 3, 4], f"{_RULES_REL} 里没抽到 moves, 退回默认 [1,3,4]"
    moves = sorted({int(x) for x in re.findall(r"\d+", cand) if int(x) > 0})
    if not moves:
        return [1, 3, 4], f"{_RULES_REL} 抽出的 moves 为空, 退回默认 [1,3,4]"
    return moves, f"{_RULES_REL}: moves={moves}"


def compute_sg(moves, n_max: int = _SEQ_LEN - 1) -> list[int]:
    """
    verifier 自己实现 SG 递推: g(n) = mex{g(n-m)}。

    为什么这一段就是判据本身: SG 值有唯一定义, 按定义算一遍的序列
    是唯一的真值。任何"周期""规律"都必须能被这条序列证伪。
    """
    key = (tuple(moves), n_max)
    if key in _SG_CACHE:
        return _SG_CACHE[key]
    g = [0] * (n_max + 1)
    for n in range(1, n_max + 1):
        seen = {g[n - m] for m in moves if m <= n}
        v = 0
        while v in seen:
            v += 1
        g[n] = v
    _SG_CACHE[key] = g
    return g


def period_of(g, p: int, onset: int = 0) -> bool:
    """g[n] == g[n+p] 对所有 onset <= n <= len(g)-1-p 是否成立。"""
    if p <= 0 or p >= len(g) - onset:
        return False
    return all(g[n] == g[n + p] for n in range(onset, len(g) - p))


def min_period(g, max_p: int = _MAX_PERIOD, onset: int = 0) -> tuple[int, int]:
    """
    返回 (最小合法周期, 该周期下能成立的**最小** onset)。

    onset 的取法: 条件 "对所有 n>=c 有 g[n]==g[n+p]" 随 c 变弱, 所以合法的
    c 构成一个后缀 —— 从 0 往上扫, 第一个成立的就是最小 onset。
    (曾经写成"往上推到最后一个成立", 得到的会是最弱起点, 反例: 周期 2 的
    序列上会报 onset=997, 看着像"周期性很晚才出现", 完全是误导。)
    """
    for p in range(1, max_p + 1):
        if period_of(g, p, onset):
            # 扫描上界必须留在 len(g)-p-1: 再往上就没有 n 可比了,
            # period_of 会因为"比较区间为空/太短"而误报不成立。
            for c in range(onset, len(g) - p):
                if not period_of(g, p, c):
                    return p, max(onset, c - 1)
            return p, onset
    return -1, -1


def _min_period_over_all_onsets(g, max_p: int = _MAX_PERIOD) -> tuple[int, int]:
    """
    不预设 onset 的最小周期 —— 与 agent 可能自报的 onset 无关, 更严。
    注意周期性若只在后半段成立, p 更小的"伪周期"也可能在那里合法, 所以
    这里同时要求 onset=0 上成立(题面要的是整条 SG 序列的周期)。
    """
    for p in range(1, max_p + 1):
        if period_of(g, p, 0):
            for c in range(0, len(g) - p):
                if not period_of(g, p, c):
                    return p, max(0, c - 1)
            return p, 0
    return -1, -1


def _load_reference() -> dict | None:
    p = _find(_REFERENCE_REL)
    if p is None:
        return None
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    except Exception:
        return None


def _as_int_list(raw) -> list[int] | None:
    if not isinstance(raw, (list, tuple)):
        return None
    out: list[int] = []
    for v in raw:
        if isinstance(v, bool):
            return None
        if isinstance(v, float):
            if not v.is_integer():
                return None
            v = int(v)
        if isinstance(v, str):
            try:
                f = float(v.strip())
            except ValueError:
                return None
            if not f.is_integer():
                return None
            v = int(f)
        if not isinstance(v, int):
            return None
        out.append(v)
    return out


def verify(stdout: str, code: str, claim: str) -> dict:
    chk = Check()
    chk.note("verdict", parse_verdict(stdout))

    moves, rules_src = read_moves()
    g = compute_sg(moves)
    chk.note("rules_source", rules_src)
    chk.note("moves", moves)
    chk.note("sg_first_10", g[:10])

    p_min, o_min = _min_period_over_all_onsets(g)
    chk.note("verifier_min_period", p_min)
    chk.note("verifier_onset", o_min)

    # 交叉核对: 参考文件只作记录, 不作判据
    ref = _load_reference()
    if ref is not None:
        ref_sg = ref.get("sg_1000")
        same = isinstance(ref_sg, list) and len(ref_sg) == len(g) and \
            [int(x) for x in ref_sg] == g
        chk.note("reference_sg_matches_verifier", bool(same))
        chk.note("reference_period", ref.get("period"))
        chk.note("reference_onset", ref.get("onset"))
        if not same:
            chk.note("reference_mismatch", "data/res_10_reference.json 与 verifier "
                                           "自算不一致 —— 以自算为准")

    art = parse_artifact(stdout, "ANSWER") or parse_artifact(stdout, "CONSTRUCT")
    if not chk.expect("artifact_present", art is not None,
                      "未提交 ANSWWER 制品(需要 {\"sg_values\": [...1000 项], "
                      "\"period\": p})"):
        return _finish(chk, None, {})
    assert art is not None

    # ---- 1. SG 值逐项比对 ----
    raw = None
    for key in ("sg_values", "sg", "grundy", "values", "sg_first_1000"):
        if art.get(key) is not None:
            raw = art[key]
            break
    got = _as_int_list(raw)
    if not chk.expect("sg_values_present", got is not None,
                      f"制品里没有整数 sg_values: {str(raw)[:60]!r}"):
        return _finish(chk, art, {"period": p_min})
    assert got is not None

    chk.expect("sg_values_length", len(got) >= _SEQ_LEN,
               f"只给了 {len(got)} 项, 题目要求前 {_SEQ_LEN} 项")

    first_bad = None
    for i in range(min(len(got), _SEQ_LEN)):
        if got[i] != g[i]:
            first_bad = (i, got[i], g[i])
            break
    chk.expect("sg_values_match", first_bad is None,
               "" if first_bad is None else
               f"第 {first_bad[0]} 项不符: 报 {first_bad[1]}, 自算 {first_bad[2]}")
    chk.note("sg_nonnegative", all(v >= 0 for v in got[:_SEQ_LEN]))

    # ---- 2. 周期规律 ----
    p_raw = art.get("period") or art.get("period_p")
    try:
        p = int(float(p_raw))
    except (TypeError, ValueError):
        p = -1
    o_raw = art.get("onset") or art.get("preperiod") or art.get("start")
    try:
        onset = int(float(o_raw))
    except (TypeError, ValueError):
        onset = 0
    onset = max(0, onset)
    chk.note("claimed_period", p)
    chk.note("claimed_onset", onset)

    chk.expect("period_positive", p > 0, f"period 非法: {p_raw!r}")
    if p > 0:
        chk.expect("period_is_real", period_of(g, p, onset),
                   f"p={p} 在 onset={onset} 上不是合法周期"
                   f"(g[n] != g[n+{p}] 至少有一处)")
        # ★ 最小性 ★ 否则报 p=7000 也"合法", 但那是废话不是规律
        chk.expect("period_is_minimal", p == p_min,
                   f"报的周期 {p} 不是最小周期(自算最小为 {p_min}); "
                   "周期规律必须是最小的那一个")

    extra = {"moves": moves, "period": p_min, "onset": o_min,
             "sg_first_10": g[:10], "sg_checked": min(len(got), _SEQ_LEN)}
    if first_bad is not None:
        extra["first_mismatch"] = {"index": first_bad[0],
                                   "reported": first_bad[1], "computed": first_bad[2]}
    return _finish(chk, art, extra)


def _finish(chk: Check, art: dict | None, extra: dict) -> dict:
    correct = chk.correct
    if correct:
        cert = (f"moves={extra['moves']} 下自算 g[0..{extra['sg_checked'] - 1}] "
                f"与制品逐项相同; 最小周期 {extra['period']}(起点 {extra['onset']}) "
                f"经自算序列验证")
    else:
        cert = chk.reason()
    final = json.dumps(extra, ensure_ascii=False, sort_keys=True) if extra else None
    return {"correct": correct, "certificate": cert, "final_answer": final,
            "details": dict(chk.details)}


if __name__ == "__main__":
    moves, src = read_moves()
    g = compute_sg(moves)
    p_min, o_min = _min_period_over_all_onsets(g)
    print(f"[INFO] 规则来源: {src}")
    print(f"[INFO] verifier 自算 SG: g[0..19] = {g[:20]}")
    print(f"[INFO] 自算最小周期 = {p_min}, onset = {o_min} "
          f"(g[n]==g[n+{p_min}] 在 n>={o_min} 上恒成立)")
    ref = _load_reference()
    if ref is not None:
        same = [int(x) for x in (ref.get("sg_1000") or [])] == g
        print(f"[INFO] 与 data/res_10_reference.json 交叉核对: sg 一致={same}, "
              f"参考 period={ref.get('period')}, 参考 onset={ref.get('onset')}")

    good = {"sg_values": g, "period": p_min, "onset": o_min}
    r = verify("VERDICT: PROVED\nANSWER: " + json.dumps(good) + "\n", "", "")
    assert r["correct"], f"正例应通过: {r}"
    print("[OK] 正例(自算 SG + 最小周期) ->", r["certificate"])

    # 反例 A: SG 值从第 37 项起整体错一位(周期仍"对得上", 但前缀错)
    bad_sg = list(g)
    bad_sg[37] += 1
    r = verify("ANSWER: " + json.dumps({"sg_values": bad_sg, "period": p_min}) + "\n",
               "", "")
    assert not r["correct"] and not r["details"]["sg_values_match"], r
    print("[OK] 反例A(SG 值第 37 项错) ->", r["certificate"])

    # 反例 B: 周期是 7 的整数倍 —— g[n]==g[n+p] 仍成立, 但不是最小周期
    r = verify("ANSWER: " +
               json.dumps({"sg_values": g, "period": p_min * 100}) + "\n", "", "")
    assert not r["correct"] and not r["details"]["period_is_minimal"], r
    print("[OK] 反例B(周期报成 100 倍, 不是最小周期) ->", r["certificate"])

    # 反例 C: 只给 30 项 —— 题目要求前 1000 项
    r = verify("ANSWER: " +
               json.dumps({"sg_values": g[:30], "period": p_min}) + "\n", "", "")
    assert not r["correct"] and not r["details"]["sg_values_length"], r
    print("[OK] 反例C(只给 30 项) ->", r["certificate"])

    # 反例 D: 周期报成 6(不是这个序列的周期)
    r = verify("ANSWER: " + json.dumps({"sg_values": g, "period": 6}) + "\n", "", "")
    assert not r["correct"] and not r["details"]["period_is_real"], r
    print("[OK] 反例D(周期报 6, 不成立) ->", r["certificate"])

    print("[OK] 全部自测通过")