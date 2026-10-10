"""Verifier —— 拿到题面和制品, **自己重算**, 给裁决。

## 它看不到什么
  · Worker 的代码
  · Explorer 的推理和方法选择理由
  · Worker 的任何自评字段

这不是形式主义。实测(旧回路): agent 会打出 `verified_200_terms: true`,
而打分器见到非空证书就给满分 —— **裁决者看见了结论, 于是结论自证**。
把推理和代码挡在外面, 是为了让"这个制品对不对"成为一个**独立**的问题。

## ★ Verifier 的诚实边界 ★
必须说清楚, 否则这个组件会被高估:

  1. **Verifier 和 Worker 可能是同一个模型。** 这里保证的是**上下文隔离**,
     不是"不同大脑"。同模型在同上下文里会自欺; 隔开上下文会减轻, 但不能根除。
  2. **机器裁决是真的。** Verifier 写的是一个检查程序, **由沙箱执行**。
     所以它的判断来自实际计算, 不是来自它的自信程度。写错的检查程序会算错,
     但不会"因为看起来对就通过"。
  3. **检查程序可能退化成一个恒真的空壳。** 这是真实风险, 所以:
     · 我们要求检查程序**同时打印它算出来的证据**, 证据为空 = 不采信;
     · 我们写**两份互相独立**的检查程序跑同一个制品。
       ★ 两个是**提高置信度**用的, 不是准入门槛 ★
       2026-10-10 改判据的理由: 只要有**一个**由沙箱实际执行、带证据的结论在,
       正确性就**已经**被验证了。让第二个检查器的缺席去推翻它,
       等于让 LLM 抖动决定一道题的生死 ——
       实测 res-03 Sylvester 就是这么把一道**答对的题**丢掉的。
       两份都验而结论相反, 那才是真降级。

  换句话说: 这套东西能挡住"随口宣布", 挡不住"两个检查程序一起想歪了"。
  要再进一步就得每题配人写的基准验证器, 那是另一笔账。

## hint 只诊断, 不指方法
"矩阵构造有误" 是诊断; "用 Paley II 构造" 是方法提示 —— 后者不给 Explorer,
否则整个"无提示"设计就漏了。
"""
from __future__ import annotations

import json
from typing import Any

from .._sandbox import SandboxRejected, run_code
from ._util import ask, extract_code, extract_json, upstream_error

__all__ = ["check", "VERDICTS"]

VERDICTS = ("PROVED", "REFUTED", "PARTIAL", "INCONCLUSIVE")

_ROLE = """\
你是一位**独立复核方**。你拿到的是一道题, 以及别人交上来的一个答案。

你的任务不是判断"他做得对不对", 而是**自己动手重算一遍**, 然后报告你的计算结果。

★ 你看不到他是怎么做的 ★ 你只看到题面和那个成品对象。
这是故意的: 如果你看见了他的推理, 你就会倾向于认可他的思路, 而不是独立核对。

【怎么复核】
写一个 Python 程序, 由机器在沙箱里执行。程序必须:
  1. 从**题面**把定义/条件/约束**自己解析或重新写一遍**(不要相信制品里的
     任何自述字段 —— 那是他写的, 可能编);
  2. 用制品里的对象做一次**真实的检验**: 代回原式、逐条验性质、重新枚举一遍;
  3. 把你**实际算出来的数字/中间结果**打印出来 —— 每一步算对了都要有痕迹;
  4. 最后打印一行 `VERDICT: <结论>`。

【结论怎么选】
  PROVED        你的检验全部通过, 而且这足以说明命题成立
  REFUTED       你找到了不成立的反例(交出来)
  PARTIAL       只验证了一部分范围, 或只有数值证据而无证明
  INCONCLUSIVE  你无法判定 —— 这是合法结论, 不要为了给个答案而硬判

★ 只做"看着像"的重构式验证(拿他的字段跟自己的字段比字符串)是**无效复核** ★
你要验的是**数学事实**: 他的对象满不满足题面要求的性质。

【不要做的事】
  · 不要在结论里写"他应该用什么方法"。只说你算出了什么、哪里不过。
  · 不要因为"看起来很合理"就判 PROVED —— 你没算就不算。

【执行环境 —— 不写清楚, 你的检查程序会直接被拒, 白跑一轮】
代码跑在一个**白名单沙箱**里, **零预置**:
  · ★ 没有任何预置名字 ★ 每个符号都要自己 import。
    踩过的坑(2026-10-10): 直接写 `Real('x')` -> NameError。
    要写 `from z3 import Real`。
  · 可 import 的只有:
      math cmath itertools functools fractions decimal statistics random json re
      string operator copy heapq bisect collections typing enum dataclasses abc
      numbers array time datetime csv textwrap
      sympy numpy scipy networkx z3
  · `open` / `eval` / `exec` / `__import__` 禁用
    (`.eval` 这个**属性**可以用 —— z3 的 `model.eval(expr)` 是标准写法)
  · 读不到任何文件; 有硬超时。
  · ★ 最后一件事最容易漏 ★ 你的程序跑完必须**打印** `VERDICT: <结论>` 这一行。
    算完就 assert、或者只 print 中间量而不打 VERDICT, 机器读不到结论 ——
    你会看到「跑了但没打出 VERDICT 行」, 这一轮就白跑了。
"""

_OUT = """\
【输出格式】
先给一个 ```python 代码块, 然后(可选)用一句话说明你为什么这样设计这个检查。
代码块之外不要写 VERDICT 行 —— VERICT 由**程序运行时**打印, 不是你手写。
"""


def _is_lean_artifact(art) -> bool:
    """这个制品是不是 Lean 源码。

    ★ 判据是「制品里有没有 Lean 源码」, 不是「题面像不像 Lean 题」 ★
    Worker 交出来什么, 就用什么去复核它 —— 两边判据必须一致。
    否则会出现「Worker 交了 Lean, Verifier 拿 Python 去查」这种荒唐的错配:
    实测两个检查器全被 Python 沙箱拒掉, 0/2 验过, 而真相是**接线问题**,
    报出来却像「检查程序报错」—— 把接线问题伪装成了数学问题。

    ★ 两种形状都要认 ★ 2026-10-10: Worker 换成原生工具调用后, 制品形状变了 ——
      新形态是 `{"answer":..., "reason":..., "tool_result": {"proved":true,
      "sorry_used":false, "source":"...", "reason":"ok", ...}}`,
      源码与判决都在**下一层** `tool_result` 里; 旧形态是顶层直接放
      `source` / `lean_proved`。只认顶层的话, Verifier 会把一份 Lean 制品
      当成普通结果拿 Python 去查 —— 于是**又一次**"接线问题伪装成数学问题"。
    """
    if not isinstance(art, dict):
        return False
    if "source" in art and "lean_proved" in art:
        return True                       # 旧形态
    tr = art.get("tool_result")           # 新形态
    return isinstance(tr, dict) and "source" in tr and "proved" in tr


_LEAN_ROLE = "\n".join([
    "你是一位**独立复核方**。待复核的对象是一段 Lean 4 源码。",
    "",
    "【输出格式】",
    "先给一个 lean 代码块 —— 那是你**自己独立写的证明**, 不是把别人的抄一遍。",
    "代码块之外不要写 VERDICT 行。",
    "",
    "【判据(由编译器给出, 不是你手写)】",
    "  · 编译通过 + **没有 sorry** = PROVED",
    "  · 任何 error / warning / sorry = 没证好",
    "",
    "★ 硬性 ★",
    "  · 头两行 `import Mathlib` + `open scoped BigOperators`(工具层会补, 但写对能少一轮往返)。",
    "  · 必须是**完整的 theorem/example 声明**; 只给一个 tactic 片段(如 `rw [add_pow]`)",
    "    会被判 `unexpected identifier; expected command`。",
    "  · **绝对不许用 `sorry`**。编译器对 sorry 放行、退出码仍是 0,",
    "    只看退出码会把「没证」当成「证好了」。",
    "  · 不确定 Mathlib 定理名就别 `exact <名字>`, 改用自动 tactic:",
    "    `by simp` / `by norm_num` / `by ring` / `by linarith` / `by nlinarith` / `by aesop`。",
    "",
    "★ Lean 没有运行时, 所以「检查程序」就是**编译器**本身 ★",
])


def check(problem: str, artifact: dict[str, Any] | None,
         scratch_dir: str, *, timeout: int = 180,
         n_checkers: int = 2, max_retries: int = 2, **kw) -> dict:
    """独立复核。返回

        {verdict, agree, outcome_kind, n_confirmed, n_total,
         confident, checkers: [{verdict, evidence, code}], evidence, note}

    ## ★ 判据只有一条: **正确性有没有被验证过** ★

    只要有一个检查程序**由沙箱实际执行**、并打印了证据, 那个结论就成立。
    第二个检查器是**提高置信度**, 不是抬高门槛 ——
    它多的时候结论更可信, 它缺席的时候不该把已经验成的东西推翻。

    实测 2026-10-10 (res-03 Sylvester) 就是在这一点上丢了一道**答对的题**:
    Worker 交出 `X = [[181/840, -1/24], [-1/56, 1/8]]`(与人工解逐项相同,
    残差 0, det(M)=1680), 一个检查器验完给了 PROVED + 证据,
    另一个没交出检查程序 —— 旧逻辑把这当成"两个结论矛盾", 判 INCONCLUSIVE,
    note 还写「两个独立检查程序结论不一致: ['PROVED']」。**说的是没发生过的事。**

    ## 真正降级的只有两种

    · **矛盾**: 两个都验了, 结论相反 —— 这时任何单一结论都不可信
    · **没验成**: 一个带证据的检查都没有 —— 这时候才是"不知道"

    少一个检查器既不是矛盾也不是"不知道"。
    """
    if artifact is None:
        return {"verdict": "INCONCLUSIVE", "agree": False, "checkers": [],
                "outcome_kind": "NO_ARTIFACT", "note": "没有制品可复核"}

    runs = []
    want = max(1, n_checkers)
    # ★ Lean 制品走编译, Python 制品走沙箱 ★ 判据必须和 Worker 交出来的东西对上
    lean_mode = _is_lean_artifact(artifact)
    role = _LEAN_ROLE if lean_mode else _ROLE
    art_view = json.dumps(artifact, ensure_ascii=False, indent=2)[:20000]

    def _one() -> dict:
        raw = ask(f"{role}\n\n【题目】\n{problem}\n\n"
                  f"【待复核的成品对象】\n"
                  f"{art_view}\n\n"
                  f"{_OUT}", temperature=0.2, **kw)
        err = upstream_error(raw)
        if err is not None:
            return {"verdict": None, "evidence": "", "code": "",
                    "fail_kind": "INFRA", "note": f"上游调用失败: {err[:200]}"}
        code = extract_code(raw)
        if not code.strip():
            return {"verdict": None, "evidence": "", "code": "",
                    "fail_kind": "INFRA", "note": "没给出检查程序"}

        # ★ Lean 分支: 检查 = 编译 ★
        #   判据 = exit 0 且无 error 且**无 sorry**;
        #   证据 = 编译器的判定, **不是**模型的自述。
        if lean_mode:
            from .._lean import check_lean, lean_available
            from .worker import _lean_prelude       # 补 import/scoped, 见那里的说明
            if not lean_available():
                return {"verdict": None, "evidence": "", "code": code,
                        "fail_kind": "BAD_CHECKER",
                        "note": "Lean 4 / Mathlib 不可用 —— 后端没接上, 不是这道题不会"}
            code = _lean_prelude(code)
            try:
                lr = check_lean(code, timeout=min(timeout, 300))
            except Exception as e:  # noqa: BLE001
                return {"verdict": None, "evidence": "", "code": code,
                        "fail_kind": "BAD_CHECKER",
                        "note": f"编译抛异常 {type(e).__name__}: {e}"}
            ok = bool(lr.get("proved")) and not lr.get("sorry_used")
            # ★★★ 这里我第一版写错了, 跑出来才发现 ★★★
            #   原来"编译不过" -> REFUTED。**编译不过不是命题为假。**
            #   后果实测: lean-04 两个检查器都没编过 -> 一起报 REFUTED -> 2/2 一致
            #   -> roles 判 REFUTED。而 `Nat.sum_range_choose n` 是**真能证出来的**
            #   (我自己编过), 所以这是一个被**当成结论采纳**的假否定。
            #
            # ★ Lean 有一条固有的不对称 ★
            #   检查器拿出一个能编译、无 sorry 的证明 -> 可以确认 PROVED;
            #   拿不出来 -> **什么都不能说明**。证伪要的是反例, 是另一个程序,
            #   不是"我这份编译不过"。所以 Lean 检查器**只能确认, 不能反驳**。
            #   这和 Python 那条路一致: run_code 失败返回 verdict=None,
            #   表示"这个检查器没出结果", 而不是"它认为答案错了"。
            if not ok:
                return {"verdict": None, "evidence": "", "code": code,
                        "fail_kind": "BAD_CHECKER",
                        "note": ("这份检查程序没编译通过 —— 这是**它自己没干成**, "
                                 "不是命题为假。Lean 证伪要反例, 不能靠编不过。"
                                 f"编译器: {str(lr.get('detail') or '')[:200]}"),
                        "stdout_tail": str(lr.get("detail") or "")[-800:]}
            return {
                "verdict": "PROVED",
                "evidence": (f"Lean 编译 exit={lr.get('exit_code')} "
                             f"error={lr.get('error_count')} "
                             f"sorry={lr.get('sorry_used')}"),
                "code": code,
                "stdout_tail": str(lr.get("detail") or "")[-800:],
                "error_type": None,
            }
        try:
            sb = run_code(code, scratch_dir, timeout=timeout, check=False)
        except SandboxRejected as e:
            return {"verdict": None, "evidence": "", "code": code,
                    "fail_kind": "BAD_CHECKER", "note": f"检查程序被沙箱拒了: {e}"}
        out = sb.stdout or ""      # ★ 是 stdout 不是 print_out ★ 同 worker.py
        return {
            "verdict": _verdict_from(out),
            "evidence": _evidence_from(out),
            "code": code,
            "stdout_tail": out[-1500:],
            # ★ stderr 也要存 ★ 之前只存了 stdout, 于是"检查程序报错了"这件事
            #   在诊断里完全看不见 —— 只能看到"没结论", 看不到为什么。
            "stderr_tail": (sb.stderr or "")[-800:],
            "error_type": sb.error_type or None,
        }

    for _ in range(want):
        runs.append(_one())

        # ★★ 补位式, 不是陪审式 ★★ 2026-10-10
        #   原来 `for _ in range(want): runs.append(_one())` —— 第二个检查器
        #   **无论第一个结果如何都照跑**。
        #
        #   但判据本身已经写死了:「第二个检查器是**提高置信度**, 不是准入门槛」。
        #   那它就不该阻塞主流程。实测代价: 每个 verifier 轮次最坏
        #   2 + 2*max_retries = 6 次 LLM 调用, 而 1 轮就答对的题只花 113 秒 ——
        #   对照下午我直连 MCP 的一道题只要 1~3 次工具调用。
        #
        #   正确用法是**补位**: 第一个交出了带证据的结论就够了, 第二个留到
        #   真正需要"第二票"(第一个没出结果)时才上场。真出现矛盾时两个都在场,
        #   判据一条没松。
        if _usable(runs[-1]):
            break

    # ★ 重试**基础设施性**的缺失 ★ 这是补能力, 不是加门。
    #   一个检查器"没交出程序"多半是 LLM 抖动 / 沙箱抖动, 不是"它认为答案不对"。
    #   补一次再判, 比直接判负诚实得多 —— 也比反复重问 Worker 便宜。
    for _ in range(max(0, max_retries)):
        need = [i for i, r in enumerate(runs) if r.get("fail_kind") == "INFRA"]
        if not need:
            break
        for i in need:
            runs[i] = _one()

    return _decide(runs, problem, n_planned=want)


def _why_failed(r: dict) -> str:
    """这个检查器为什么没算出一个『带证据的结论』。

    ★ 必须说清楚是哪一步断的 ★
    以前这三种情况全被压成一句话"没给出带证据的结论",
    于是看到它的人无从下手 —— 而事实上它们的修法完全不同:
    没给代码 → 提示词问题; 被沙箱拒 → 代码用了禁用 API; 跑了但没打 VERDICT 行 → 格式问题。
    """
    if not (r.get("code") or "").strip():
        return "没给出检查程序"
    if r.get("note") and "沙箱拒" in str(r["note"]):
        return f"沙箱拒了({r['note']})"
    et = r.get("error_type")
    if et:
        return f"检查程序报错 {et}: {str(r.get('stderr_tail') or r.get('stdout_tail') or '')[-200:]}"
    if not r.get("verdict"):
        tail = str(r.get("stdout_tail") or "")[-200:]
        return f"跑了但没打出 VERDICT 行(输出尾部: {tail!r})" if tail \
            else "跑了但没打出 VERDICT 行, 也没有任何输出"
    return "有结论但没打印证据"


def _usable(r: dict) -> bool:
    """这个检查器交出了**带证据的结论**吗?

    ★ 判据是「有结论且有证据」, 不是「有结论」 ★
    一个没打印证据的 PROVED 不能采信 —— 那正是恒真空壳的特征。
    """
    return bool(r.get("verdict") in VERDICTS and r.get("evidence"))


def _decide(runs: list[dict], problem: str = "",
            n_planned: int | None = None) -> dict[str, Any]:
    """把若干检查器的原始结果汇成裁决。

    ★ 判据: 有没有**被验证过**, 不是有没有**全体一致** ★

    旧逻辑 `agree =  unanimity and completeness` 里有两件事被搅在一起:
      · 结论真矛盾   -> 该降级
      · 有人没出结果 -> 这是**没验过**, 不是"验过而且不通过"
    少一个检查器就整体判负, 等于让 LLM 抖动决定一道题的生死 ——
    res-03 就是这么丢的。

    现在:
      · 有带证据的结论, 且无矛盾 -> **采纳**; 缺席的记进 n_confirmed/n_total
      · 矛盾                     -> 降级
      · 一个带证据的都没有       -> 才是"不知道"

    降级比原来少, 但**每一次降级都是真降级** ——
    这比"多降一级看起来更安全"有用得多, 因为假阴性的代价是白丢一道答对的题。
    """
    good = [r for r in runs if r.get("verdict") in VERDICTS and r.get("evidence")]
    votes = {r["verdict"] for r in good}
    n_confirmed, n_total = len(good), len(runs)

    # ---- 1. 真矛盾: 两个都验了, 结论相反 ----
    if len(votes) > 1:
        return {"verdict": "INCONCLUSIVE", "agree": False,
                "outcome_kind": "CONTRADICTION", "checkers": runs,
                "n_confirmed": n_confirmed, "n_total": n_total,
                "note": f"两个独立检查程序给出了**矛盾**的结论: {sorted(votes)} —— "
                        f"这时候任何单一结论都不可信",
                "hint": _hint_from(runs, problem)}

    # ---- 2. 一个带证据的都没有: 这才是真的"不知道" ----
    if not good:
        why = "; ".join(f"#{i + 1} {_why_failed(r)}" for i, r in enumerate(runs))
        all_infra = all(r.get("fail_kind") == "INFRA" for r in runs)
        return {"verdict": "INCONCLUSIVE", "agree": False,
                "outcome_kind": "NO_USABLE_RESULT", "checkers": runs,
                "n_confirmed": 0, "n_total": n_total,
                "failed_kind": "INFRA" if all_infra else "CHECKER",
                "note": f"检查程序没给出带证据的结论 —— {why}"
                        + ("(全部是基础设施失效, 不是答案有问题)" if all_infra else "")}

    # ---- 3. 验过了, 无矛盾 -> 采纳 ----
    verdict = next(iter(votes))
    missing = [(i + 1, _why_failed(r)) for i, r in enumerate(runs) if not _usable(r)]
    # ★ n_total 报**本来该有几个**, 不是实际跑了几个 ★
    #   补位式之后第一个成功就 break, 于是 n_confirmed==n_total==1,
    #   看着像"全票", 而第二个根本没上场 —— 那会把"1 个验过"混进"2 个一致"。
    planned = max(n_total, n_planned or 0)
    out: dict[str, Any] = {
        "verdict": verdict, "agree": True,
        "outcome_kind": "UNANIMOUS" if not missing else "VERIFIED",
        "checkers": runs,
        "n_confirmed": n_confirmed, "n_total": planned,
        "second_checker_ran": len(runs) > 1,
        "confident": not missing and len(runs) >= 2,
        "evidence": good[0]["evidence"],
    }
    if missing:
        if len(runs) <= 1:
            out["note"] = (f"正确性**已被验证**: 1/{planned} 个检查程序由沙箱实际执行"
                           f"并给出带证据的 {verdict}。第二个检查器按**补位式**未上场"
                           f"(第一个已给出带证据的结论) —— 它是提高置信度的, "
                           f"不是准入门槛, 所以不该阻塞主流程。")
        else:
            out["note"] = (f"正确性**已被验证**: {n_confirmed}/{planned} 个检查程序"
                           f"由沙箱实际执行并给出带证据的 {verdict}; "
                           f"其余没出结果(缺席, 不是反对): "
                           + "; ".join(f"#{i} {w}" for i, w in missing))
    if verdict not in ("PROVED", "REFUTED"):
        out["hint"] = _hint_from(good, problem)
    return out


def _verdict_from(stdout: str) -> str | None:
    import re
    m = re.search(r"^\s*VERDICT\s*:\s*([A-Za-z]+)", stdout or "",
                  re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    v = m.group(1).upper()
    return v if v in VERDICTS else None


def _evidence_from(stdout: str) -> str:
    """检查程序自己算出来的痕迹。VERDICT 行之外的东西就是证据。

    ★ 空证据 = 不采信 ★ 一个只会打印 "VERDICT: PROVED" 的检查程序
    什么也没算, 它的结论是白给的。
    """
    keep = []
    for line in (stdout or "").splitlines():
        s = line.strip()
        if not s or s.upper().startswith("VERDICT"):
            continue
        keep.append(s)
    return "\n".join(keep)[-1200:]


def _hint_from(runs: list[dict], problem: str) -> str:
    """从检查程序的诊断里提炼一句"哪里错了"。★ 不给方法 ★"""
    lines = []
    for r in runs:
        for ln in (r.get("stdout_tail") or "").splitlines()[-12:]:
            s = ln.strip()
            if s and not s.upper().startswith("VERDICT"):
                lines.append(s)
    return " / ".join(lines[-6:])[:600]