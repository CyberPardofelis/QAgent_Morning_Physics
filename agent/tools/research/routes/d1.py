"""D1 路线 —— D 的继任者, 用于验证 D 上的 bug 修复与架构改动。

## 与 D 的关系

**D 一行都不改。** D 已经有 40 题 x 4 轮的实测数据(基线 + 三个改动轮),
那些数据是本项目的对照组; 改动 D 就等于毁掉对照。

所以 D1 站在两个原则之上:

1. **复用而不是复制。** `Lane` / `make_lanes` / `board_intents` / `lane_prompt` /
   `explorer_prompt` / `run_lanes` / `lane_trace_fields` 全部**从 d.py 导入**,
   不重写。只在 `solve_d1` 里实现**不同的地方**。这样两边不会漂移 ——
   交接文档 §5 里那些"同一件事在两层各写一遍 = 两边迟早不一致"的教训。

2. **D 不动。** 没有任何 monkey-patch, 没有全局开关。

## D1 相对 D 的四处改动

### ① probe 轮丢掉制品 (bug, 实测 4 轮里让 lean-04 三次"做对了交不出去")

D 的主循环:

    results = run_lanes(runners)
    ...
    if action == "probe":
        continue          # ← 这一轮的制品**全丢**

`probe` 的本意是"这一轮不强制 final"(见 `run_probe` 的 docstring), 但
D 实现成了"这一轮的结果不算数"。后果实测:

    v2        turn2 = execute, lane2 proved=True  -> 收进答案 -> CORRECT
    v1type    turn2 = probe,   lane2 proved=True  -> 丢弃     -> INCONCLUSIVE
    v2desc    turn3 = probe,   三条 lane 共 6278 字 -> 丢弃     -> INCONCLUSIVE
    v3digest  turn3 = probe,   lane2 proved=True  -> 丢弃     -> INCONCLUSIVE

基线之所以对, **纯粹因为它成功的 lane 恰好落在 execute 轮上**。

**D1 的做法**: probe 轮照旧 `continue`(不提前收工, 保留 probe 的语义),
但把它见过的最好制品**存下来当兜底**。预算耗尽且 execute 轮颗粒无收时,
交出这个兜底, 而不是交出空答案。
★ 只兜底**非占位符**的制品 —— 否则等于把 oeis-03 那个 bug 请回来。

### ② 重复方法检测漏掉 probe 轮 (bug)

D 里:

    if action == "execute":
        fp = pl.fingerprint
        if used_fps.get(fp, 0) >= 2: ... continue
        used_fps[fp] = used_fps.get(fp, 0) + 1

只在 execute 时计数。实测 v3digest 的 lean-04 是 **turn2、turn3 连着两个
probe**, 同一个错误假设被 probe 了两次而没有触发"换一条路"。
**D1: 两种 action 都计数。**

### ③ 只渲染本轮要用的工具 (架构: 压 prompt)

实测 worker 的 prompt 中位 **40.6k 字符**, 构成:

    头(catalog 13.8k + 题面 + 方法 + 环境) 14.5k
    + 工具返回累积(每步最多 MAX_FEEDBACK=6k, 保留最近 2*KEEP_TURNS 轮) ~26k

**catalog 一个字都没随题面变, 却占第一轮的 95%。** 而 Explorer 已经把该用的
工具名放进计划里了(`plan["tools"]`), lane 任务书里也写着。

**D1 的做法**: 只给那几个工具渲染完整说明, 其余只留名字一行。
1 个工具时 catalog 从 13788 → 1078 字符, 整个 prompt 省 **31%**。

★ 安全兜底 ★ Explorer 有时会填**库名**而不是工具名(实测 `cvxpy` /
  `ortools` 都不在注册表里)。所以: 过滤结果为空时**退回全量** ——
  宁可多花 token, 也不能把某个工具的说明藏起来让 worker 看不见。

### ④ LLM verifier (新增, 一次调用)

交接文档 §6 说 Verifier 是坏的, 因为"检查程序被沙箱拒, 0/2 验过"。
那是**程序化**验证的失败: 检查器要落成代码、要在沙箱里跑, 于是被拒。

D1 换一条路: **不写检查器, 用一次 LLM 调用做复核**。

它能验的是基准值匹配**根本管不了**的三件事:

  · 这个答案**答的是不是这道题**(基准值匹配只比数字, 不管跑题)
  · 答案**是不是从 worker 实际拿到的工具输出里来的**
    (判据只比答案文本, 碰不到"证据链")
  · 推理链**自不自洽**(符号/单位/下标/取值范围)

它**不**重新解一遍题(那是 worker 的活, 翻倍的代价)。

★ 关键约束 ★ verifier **必须看到证据**, 只给它最终答案, 它就只是第二个
  猜的, 不比现在的判据强。所以它吃的是: 题面 + 最终答案 + **工具调用序列**
  + 最后一次工具返回。

★ 定位: **只做旁证, 不改判分** ★
  基准值命中时那是硬事实, LLM 复核是启发式, 启发式不该推翻硬事实。
  verifier 的价值在**判据瞎掉的地方** —— UNJUDGED 的题、以及"答案蒙对了
  但过程可疑"的题。

## 返回

与 `solve_d` 同构, 外加 `verify` 字段。`verdict` 取值多一个 `VERIFIED_*`
前缀的旁证标记, 但 WORKER_ONLY/DISPUTED/INCONCLUSIVE 的语义不变。
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent.tools.research.routes.d import (  # noqa: E402
    DEFAULT_LANES, Lane, board_intents, explorer_prompt, lane_prompt,
    make_lanes, lane_trace_fields, run_lanes,
    # ★ 这三个是 d.py 的下划线私有名, 但它们是**判定逻辑本身**, 不是
    #   "内部实现细节": _is_placeholder 决定"交出来的是不是内容",
    #   _agreement 决定"几条 lane 数值上一致吗", _lanes_from_plan 决定分派。
    #   重新实现这三份 = 把 oeis-03 那个 bug 的修复和它可能的回归一起复制一遍。
    #   所以 D1 复用它们, 而不是各写一份 —— 宁可破一次下划线约定。
    _agreement, _is_placeholder, _lanes_from_plan)
from agent.tools.research.roles import worker as _WK  # noqa: E402
from agent.tools.research.roles.catalog import render_catalog  # noqa: E402

__all__ = ["solve_d1", "focused_catalog", "verify_by_llm",
           "verify_by_llm_tools", "MAX_FOCUS_TOOLS", "VERIFY_BUDGET"]

#: 最多给几个工具渲染完整说明。再多就不是"聚焦"而是全量了。
MAX_FOCUS_TOOLS = 3

#: verifier 的工具预算。刻意很小 —— 它放开手就会重解整题, 成本翻倍。
VERIFY_BUDGET = 2


# ====================================================================== ③ 压缩
def _one_line(desc: str, limit: int = 70) -> str:
    """把一段工具描述压成**一行能干什么**。

    ★ 为什么按句号切而不是按字数硬截 ★
      硬截会切出半句话 —— 实测 `gap_invariants` 的描述第一句有 60+ 字,
      按 56 截出来是「给单个群算一组可辨识不变量(阶/是否交换/可解/单/幂零/完备/导出链长/中心/…」
      带着一个截断的括号, 读起来比不截还糟。
    """
    d = (desc or "").strip()
    if not d:
        return ""
    for sep in ("。", "\n", "; "):
        if sep in d:
            d = d.split(sep)[0]
            break
    d = d.strip()
    return d[:limit] + ("…" if len(d) > limit else "")


def explorer_catalog(tools: list[dict] | None) -> str:
    """给 **Explorer** 用的目录: 只有「名字 + 一行能干什么」。

    ## 为什么 Explorer 可以不看完整 catalog

      实测 2026-10-11 跑满 40 题: Explorer 的 prompt 中位 **21518 字符**,
      其中全量 catalog 占 **93.6%**。而 Explorer 的职责是**选路** ——
      它决定 `plan["tools"]`, 真正**调工具的是 Worker**。
      参数 schema、属性 description("沙箱读不到文件系统"那段契约)
      全是给 Worker 的, 对 Explorer 是纯噪声。

      实测简述版: 10058 字符 -> **362 字符**, 省 96%。

    ## 为什么不能改成默认

      B/C/D 三条对照路线也走 `explorer_prompt`, 改了它们的选路质量
      就跟着变, 而它们是这轮实验的**对照组**。
      ⇒ 只在 D1 里显式调用, 且 `plan_prompt` 的 `catalog` 形参默认 None。

    ## 砍掉的东西会不会有用

      会 —— 所以这里只给**名字**, 并明确告诉它"细节由 worker 自己读"。
      万一它选错了工具, worker 那一侧拿到的仍是**完整聚焦版** catalog,
      而且 `focused_catalog` 的兜底逻辑保证"命中不到就退回全量"。
    """
    if not tools:
        return ""
    rows: list[str] = []
    seen: set[str] = set()
    for t in tools:
        n = str(t.get("name") or "")
        if not n or n in seen:
            continue
        seen.add(n)
        lib = t.get("library") or ""
        tail = f"  [{lib}]" if lib else ""
        rows.append(f"  {n}{tail}: {_one_line(t.get('description'))}")
    body = "\n".join(rows)
    # ★ 这里**不要再加 "【工具目录】" 标题** ★
    #   `plan_prompt` 里已经硬编码了 `【工具目录】(...)` + `{cat}`,
    #   我第一版又自己加了一个 -> 渲染出来是
    #       【工具目录】(这是这台机器上真实可用的东西, 就这些)
    #       【工具目录】(就这些, 名字必须逐字用, 不要发明)
    #   两个同名区块叠在一起, 读起来像出过一次故障。
    #   ⇒ 只返回 body + 尾注, 标题交给调用方那一层统一加。
    return (body + "\n"
            f"★ 这里只列了每个工具**能干什么**。参数怎么传、沙箱有哪些规矩,\n"
            f"  **worker 自己会读到完整说明** —— 你只需要选对工具名。★")
def focused_catalog(tools: list[dict] | None,
                    focus: list[str] | None) -> tuple[str, list[str], bool]:
    """只渲染 `focus` 里的工具, 其余只留名字。

    返回 `(catalog文本, 真正聚焦到的工具名, 是否做了过滤)`。

    ★ 为什么宁可退回全量 ★
      Explorer 偶尔填**库名**而不是工具名(实测 `cvxpy` / `ortools` 都不在
      注册表里)。若按"没命中就只渲染空集", worker 就**看不见任何工具的说明**
      —— 那比多花 13k 字符糟糕得多。命中不到时一律用全量。
    """
    full = render_catalog(tools)
    if not tools:
        return full, [], False
    want = {str(x).strip() for x in (focus or []) if str(x).strip()}
    if not want:
        return full, [], False
    by_name = {str(t.get("name")): t for t in tools}
    hit = [n for n in (list(want)) if n in by_name]
    if not hit:
        # ★ 兜底: 一个都没命中 -> 全量
        return full, [], False
    if len(hit) >= len(tools):
        return full, [], False
    keep = [t for n in sorted(hit) if (t := by_name[n]) is not None]
    rest = [n for n in by_name if n not in set(hit)]
    text = render_catalog(keep)
    if rest:
        text += ("\n\n【本次没展开的工具(只是名字留个印象, 详细说明已省掉;\n"
                 "  确实需要其中某个时, 直接照着名字调它)】\n"
                 + ", ".join(sorted(rest)))
    return text, sorted(hit), True


# ====================================================================== ④ verifier
def _tool_succeeded(obs) -> bool:
    """这次工具调用**成功**了吗?

    ★ 为什么不能写 `bool(obs.get("ok"))` ★
      `registry.call_tool` 的契约是「**失败时**一定有 `ok: False` + `error`;
      成功时**原样返回工具自己的输出**」。而实测(2026-10-11) 17 个工具里
      **只有 `z3_smt2` 成功时不带 `ok` 键** —— 它用
      `status` / `is_sat` / `is_unsat` 表达结果:

          {'status': 'unsat', 'is_sat': False, 'is_unsat': True,
           'model': None, 'stdout': 'status = unsat\\n', 'elapsed_ms': 225}

      于是 `obs.get("ok")` 给出 **None -> bool(None) == False**,
      一次**成功且否证了论断**的 Z3 调用被记成 "失败"。
      第一版 verifier 就是这么写的, 实跑 z3-01 时它把
      `unsat`(正是它要的证据)标成 ok=False, 又拿这个错标记去给人看。
      同一个家族的错误 2026-10-11 已经在 `_digest_one` 上栽过一次
      (Lean 用 `proved` 不填 `ok`), 这里是第二处。

    ⇒ 判据: **`ok` 不存在 = 成功**(而不是失败)。
    """
    if not isinstance(obs, dict):
        return False
    if "ok" in obs:
        return bool(obs.get("ok"))
    return "error" not in obs


_VERIFY_HEAD = """你要复核**另一个人**刚交上来的一份科研计算答案。
你没有工具, 不要重算 —— 你要做的是**判断这份答案有没有被它自己拿到的证据支撑**。

【题目】
{problem}

【他交出的答案】
{answer}

【他实际调过的工具(按顺序)】
{calls}

【最后一次工具返回】
{last_result}

请只判断下面这四件事, 并输出 JSON:

1. answers_question: 这个答案**答的**是不是上面那道题? (跑题 / 答非所问 = false)
2. grounded: 结论是不是**从上面的工具返回里**得出来的? 出现工具返回里
   根本没有的数字/结论, 而 he 又说不出怎么来的 = false
3. coherent: 量纲/符号/下标约定/取值范围有没有自相矛盾的地方?
4. verdict:
     "SUPPORTED"   上面三条都站得住
     "UNSUPPORTED" 有一条明确不成立(把证据原文引出来)
     "UNCERTAIN"    看不出来 —— **这是合法且常见的答案, 不要为了给结论而编**

weakest_link 用一句话指出最薄弱的那个环节; reason 不超过 80 字。

只输出 JSON, 不要任何解释或标题。"""


def _compact_calls(result: dict, limit: int = 12) -> str:
    """把工具调用序列压成几行给人看: 工具名 + 成败 + 关键片段。"""
    out: list[str] = []
    for t in (result.get("tool_calls") or [])[:limit]:
        if not isinstance(t, dict):
            continue
        tool = str(t.get("tool"))
        # ★ 这里的 ok 是 worker 记在 tool_calls 上的, 同样可能是 None ★
        #   worker 对 Lean 有 `proved` 兜底, 但 z3_smt2 这一族没有 ——
        #   判成 "?" 会让 verifier 以为"不知道成没成", 白丢一条证据。
        raw = t.get("ok")
        ok = _tool_succeeded({"ok": raw}) if raw is not None else True
        mark = "ok" if ok else "fail"
        err = str((t.get("error") or ""))[:80].replace("\n", " ")
        val = ""
        if ok and t.get("args"):
            val = "  参数=" + json.dumps(t.get("args"), ensure_ascii=False,
                                        default=str)[:110]
        out.append(f"  [{mark}] {tool}{('  ' + err) if err else ''}{val}")
    if not out:
        return "  (他一次工具都没调)"
    more = len(result.get("tool_calls") or []) - limit
    if more > 0:
        out.append(f"  …… 还有 {more} 次没列")
    return "\n".join(out)


def verify_by_llm(problem: str, result: dict, *, ask, **kw) -> dict | None:
    """一次 LLM 调用做复核。**旁证** —— 返回一个 dict, 不参与判分。

    ⚠ 这是**不带工具**的降级版。真正有意义的复核是 `verify_by_llm_tools` ——
      只看答案文本的复核比现有的值比对强不了多少, 它答的是"顺不顺眼",
      不是"对不对"。没有工具时这个函数只应该作为**兜底**。

    返回 `None` 表示没验成(调用失败/解析不出), 这种情况必须如实标成
    "没验", **不能**默认成通过。
    """
    art = result.get("artifact") or {}
    answer = str(art.get("answer") or result.get("answer") or "")
    if not answer.strip():
        return {"verdict": "UNCERTAIN", "answers_question": None,
                "grounded": None, "coherent": None, "checked": [],
                "weakest_link": "没有交出任何答案, 无从复核",
                "reason": "artifact 为空", "tool_budget_used": 0}
    prompt = _VERIFY_HEAD.format(
        problem=problem[:2000],
        answer=answer[:3000],
        calls=_compact_calls(result),
        last_result=json.dumps(art.get("tool_result") or result.get("tool_result"),
                               ensure_ascii=False, default=str)[:2000],
    )
    try:
        raw = ask(prompt, temperature=0.1, **kw)
    except Exception as e:  # noqa: BLE001
        return {"verdict": "UNCERTAIN", "answers_question": None,
                "grounded": None, "coherent": None, "checked": [],
                "weakest_link": "verifier 调用失败",
                "reason": f"{type(e).__name__}: {e}"[:120], "tool_budget_used": 0}

    from agent.tools.research.roles._util import extract_json
    obj = extract_json(raw)
    if not isinstance(obj, dict) or "verdict" not in obj:
        return {"verdict": "UNCERTAIN", "answers_question": None,
                "grounded": None, "coherent": None, "checked": [],
                "weakest_link": "verifier 输出解析失败",
                "reason": str(raw)[:120], "tool_budget_used": 0}
    return _pack_verify(obj, [])


def _pack_verify(obj: dict, checked: list[dict]) -> dict:
    v = str(obj.get("verdict") or "").upper()
    if v not in ("SUPPORTED", "UNSUPPORTED", "UNCERTAIN"):
        v = "UNCERTAIN"
    return {
        "verdict": v,
        "answers_question": obj.get("answers_question"),
        "grounded": obj.get("grounded"),
        "coherent": obj.get("coherent"),
        "checked": checked,
        "weakest_link": str(obj.get("weakest_link") or "")[:200],
        "reason": str(obj.get("reason") or "")[:200],
        "tool_budget_used": len(checked),
    }


# ------------------------------------------------------------------ 带工具的复核
_VERIFY_T0 = """你要复核**另一个人**刚交上来的一份科研计算答案。

你没有资格重解这道题 —— **你的活是复核, 不是重做**。
所以每一步你只准做一件事: **挑出答案里最关键的那一个论断, 调工具把它验掉**。

【题目】
{problem}

【他的答案】
{answer}

【他调过的工具(按顺序)】
{calls}

【可用工具】
{catalog}

现在输出 JSON, 二选一:

A) 要验一个论断 —— 填 tool / args:
{{"action":"check", "claim":"你要验的那句话(一句话)", "tool":"工具名", "args":{{...}}}}
   ★ args 必须严格符合上面工具的参数; 拿不准就换一个更小的论断来验。

B) 已经够了 / 验不动 —— 填 verdict:
{{"action":"final", "verdict":"SUPPORTED"|"UNSUPPORTED"|"UNCERTAIN",
  "answers_question":true/false, "grounded":true/false, "coherent":true/false,
  "weakest_link":"一句话", "reason":"不超过80字"}}

★★ UNCERTAIN 是合法且常见的答案。**验不动就直说, 编一个结论比不验更糟。 ★

只输出 JSON。"""

_VERIFY_TN = """你刚才要验的是: {claim}
用 {tool} 跑出来的返回是:
{result}

这个论断因此:
  · 被证实 -> 在最终 JSON 里 grounded=true
  · 被推翻 -> verdict="UNSUPPORTED", weakest_link 写清哪一条被推翻了, 并引用返回里的原文
  · 验不了 -> verdict="UNCERTAIN", reason 说清卡在哪

只输出最终 JSON(字段同上一轮: verdict / answers_question / grounded / coherent /
weakest_link / reason)。"""


def verify_by_llm_tools(problem: str, result: dict, *, ask,
                        catalog: str = "", budget: int = 2,
                        call=None, **kw) -> dict:
    """带工具的复核: 最多 `budget` 次工具调用, 每次调用一次 LLM。

    ## 为什么 verifier 需要工具

    交接文档 §6 记的 verifier 失败是**程序化**验证的失败 —— 检查器要落成
    代码、要在沙箱里跑, 于是被拒, 0/2 验过。换成本文件这条路就绕开了它。

    有工具之后, 复核才有 teeth:

      · Lean 题: 把提交的证明丢给 `lean_check` -> **编译器裁决**, 一次工具
        调用、零额外 LLM, 这正是交接文档一直想要而没得到的东西
      · 数值题: 独立复算关键那一步, 和他的答案对比
      · OEIS 题: `oeis_lookup` 查序列, 比对公式对不对

    ## 为什么给它预算而不是放开手

    verifier 一放开就会**重解整题**, 那等于把成本翻倍, 而复核的价值恰恰
    在于"用最小代价否掉一个可疑论断"。所以:
      · `budget` 很小(默认 2)
      · 提示里写死"你的活是复核, 不是重做"
      · 它可以随时 `final` 提前收工

    ## 为什么仍然只是**旁证**

    基准值命中时那是硬事实。启发式复核不该推翻硬事实, 所以这个函数的
    返回值**不参与判分**, 只作为诊断信号附在结果里。
    """
    art = result.get("artifact") or {}
    answer = str(art.get("answer") or result.get("answer") or "")
    checked: list[dict] = []
    # ★ 每一步的**原始输出**必须落盘 ★
    #   2026-10-11 实测踩到了: 我发现 20 题的 UNCERTAIN 是自己的分支造的,
    #   想用新逻辑重放这批数据, 却发现原始 JSON 一个都没留 —— 那 20 题
    #   的真实结论**永久取不回来了**, 只能重跑一遍(又是一轮 LLM 成本)。
    #   ⇒ 只要将来可能想改判定逻辑, 就必须把原料留下。
    raw_steps: list[dict] = []
    if not answer.strip():
        return {"verdict": "UNCERTAIN", "answers_question": None,
                "grounded": None, "coherent": None, "checked": [],
                "weakest_link": "没有交出任何答案, 无从复核",
                "reason": "artifact 为空", "tool_budget_used": 0}

    convo = [_VERIFY_T0.format(
        problem=problem[:2000], answer=answer[:3000],
        calls=_compact_calls(result), catalog=catalog[:4000])]

    from agent.tools.research.roles._util import extract_json

    def done(obj: dict) -> dict:
        """收口: 统一把逐步原始输出挂进结果。"""
        out = _pack_verify(obj, checked)
        out["raw_steps"] = list(raw_steps)
        return out

    for step in range(max(1, int(budget)) + 1):
        try:
            raw = ask("\n\n".join(convo), temperature=0.1, **kw)
        except Exception as e:  # noqa: BLE001
            return done({"verdict": "UNCERTAIN",
                         "weakest_link": "verifier 调用失败",
                         "reason": f"{type(e).__name__}: {e}"[:120]})
        obj = extract_json(raw)
        raw_steps.append({"step": step, "raw": str(raw)[:1500],
                          "obj": obj if isinstance(obj, dict) else None})
        if not isinstance(obj, dict):
            return done({"verdict": "UNCERTAIN",
                         "weakest_link": "verifier 输出解析失败",
                         "reason": str(raw)[:120]})
        act = str(obj.get("action") or "").lower()
        if act == "final" or step >= budget:
            return done(obj)

        name = str(obj.get("tool") or "").strip()
        args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
        # ★ 这里曾经无条件把"没给工具名"压成 UNCERTAIN ★
        #   实测 2026-10-11 跑满 40 题: **20 题**栽在这条分支上。
        #   模型那一轮其实**已经给了结论** —— 例如
        #     "工具返回 proved=true、reason=ok, 明确确认代码编译成功",
        #     "在(4/5,1)处确实得到 p=-119/625<0, 足以构成反例",
        #   只是忘了把 action 从 "check" 改成 "final"(它以为上一轮
        #   已经调过工具了)。我却把它的 verdict 连同 reason 一起丢掉,
        #   换成"要调工具但没给工具名"。
        #
        #   ⇒ 后果不是"统计不好看", 是**掩盖了 verifier 的真实判断**:
        #     真 UNCERTAIN 只有 5 题, 却被 20 条假 UNCERTAIN 淹没。
        #
        # ⇒ 修法: 没给工具名时, **先看它有没有给结论**。
        #   给了就采信(它已经拿到工具返回了), 没给才落 UNCERTAIN。
        if not name or call is None:
            if str(obj.get("verdict") or "").strip():
                return done(
                    {**obj,
                     "weakest_link": obj.get("weakest_link")
                     or f"它说要调 {obj.get('tool') or '工具'} 却没给工具名, "
                        f"但已给出结论, 按结论采信"})
            return done({**obj, "action": "",
                         "verdict": "UNCERTAIN",
                         "weakest_link": "verifier 要调工具但没给工具名",
                         })
        claim = str(obj.get("claim") or "")[:200]
        try:
            obs = call(name, args)
        except Exception as e:  # noqa: BLE001
            obs = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
        checked.append({"claim": claim, "tool": name, "args": args,
                        "ok": _tool_succeeded(obs),
                        "brief": json.dumps(obs, ensure_ascii=False,
                                            default=str)[:300]})
        convo.append(_VERIFY_TN.format(
            claim=claim or "(未说明)", tool=name,
            result=json.dumps(obs, ensure_ascii=False, default=str)[:2500]))

    return done({"verdict": "UNCERTAIN",
                 "weakest_link": "工具预算用尽, 没验完",
                 "reason": f"用了 {len(checked)} 次工具"})

# ====================================================================== 主循环
def solve_d1(problem: str, *, tools=None, hints: str | None = None,
             budget=None, scratch_dir: str | None = None,
             verbose: bool = False,
             max_lanes: int = DEFAULT_LANES, verify: bool = True,
             verify_budget: int = VERIFY_BUDGET,
             lean_explorer: bool = False, **llm_kw):
    """D1 = D + 两处 bug 修复 + prompt 聚焦 + LLM verifier。

    与 `solve_d` 的差别**只有四处**, 其余辅助全部从 d.py 导入复用:

      ① probe 轮的制品不再被丢掉(留作兜底)
      ② 重复方法检测同时覆盖 probe 和 execute
      ③ 每轮按计划里的工具**聚焦** catalog(prompt 省 ~31%)
      ④ 可选的 LLM verifier(带工具, 只做旁证)

    ## `lean_explorer`: Explorer 也用简述目录(可开关, 供 A/B 对照)

      默认 **False** = 与已跑完的 d1run 那轮完全一致, 这样开关本身
      不会污染与历史数据的可比性。置 True 后 Explorer 只看到
      「名字 + 一行能干什么」(10058 -> 362 字符), 它的 prompt
      从 21518 降到约 2000。

      ★ 风险 ★ Explorer 靠 catalog 决定 `plan["tools"]`, 而 worker
      是**按这个字段去聚焦 catalog** 的 —— 如果 Explorer 因为信息
      太少而选错/漏选工具名, 会连带影响聚焦。`focused_catalog` 的
      兜底(命中不到就退回全量)能挡住"选了库里没有的名字",
      但**挡不住"漏选了本该用的第二个工具"**。
      所以这个开关必须靠 A/B 跑分判断, 不能只看字符数。

    ★ `verbose` 必须**显式声明**, 不能落进 `**llm_kw` ★
      调用方(`run_part2_d.py`)是按 `solve_d` 的签名调的, 一定会传
      `verbose=False`。而 `llm_kw` 最后会一路透传到
      `_call_llm(prompt, ..., **kw)` —— **它没有 `**kw`**。
      于是每一次 D1 实跑都会在第一个 Explorer 调用上
      `TypeError: _call_llm() got an unexpected keyword argument 'verbose'`,
      整题 CRASH。签名里少写一个装饰性参数 = 100% 挂, 且单元测试
      (假 ask 什么参数都吃) 完全看不出来。
    """
    import shutil
    import tempfile
    import time

    from agent.tools.research.roles import _registry as _reg
    # ★ 必须从 `_util` 导, 不能写 `from ...roles import ask` ★
    #   `roles/__init__.py` 只 re-export 了 Budget, **没有** ask ——
    #   `from package import name` 在 name 不是子模块时会直接 ImportError。
    #   d.py 用的就是这个路径, 这里跟着它走。
    from agent.tools.research.roles._util import Budget, ask as _ask
    from agent.tools.research.roles import explorer as _EX

    budget = budget or Budget(max_turns=3)
    own_dir = scratch_dir is None
    scratch = scratch_dir or tempfile.mkdtemp(prefix="d1_route_")
    started = time.time()
    trace: list[dict] = []
    used_fps: dict[str, int] = {}
    lanes_seen = 0
    peak_conc = 0
    # ★ 修复① probe 轮见过的好制品, 预算耗尽时兜底
    fallback: dict | None = None
    fallback_note = ""
    n_focused = 0

    def _finish(res: dict) -> dict:
        if verify and res.get("artifact"):
            try:
                res["verify"] = verify_by_llm_tools(
                    problem, res, ask=lambda p, **k: _ask(p, **k, **llm_kw),
                    catalog=render_catalog(tools),
                    budget=verify_budget, call=_reg.call)
            except Exception as e:  # noqa: BLE001
                res["verify"] = {"verdict": "UNCERTAIN", "checked": [],
                                 "weakest_link": "verifier 炸了",
                                 "reason": f"{type(e).__name__}: {e}"[:120],
                                 "tool_budget_used": 0}
        else:
            res["verify"] = None
        return res

    try:
        while not budget.exhausted():
            budget.spend()
            turn = budget.turns

            prompt = explorer_prompt(problem, hints=hints,
                                     history=list(trace), tools=tools,
                                     turn=turn,
                                     catalog=(explorer_catalog(tools)
                                              if lean_explorer else None))
            raw = _ask(prompt, temperature=0.6, **llm_kw)
            obj = _EX.extract_json(raw)
            pl = _EX.Plan(obj) if isinstance(obj, dict) else None
            if not isinstance(pl, dict):
                trace.append({"turn": turn, "role": "explorer", "action": None,
                              "outcome": "Explorer 的输出里没解析出计划 JSON",
                              "lesson": None, "lesson_given": False})
                continue

            lanes = _lanes_from_plan(pl)[:max(1, int(max_lanes or 1))]
            action = pl.action
            trace.append({
                "turn": turn, "role": "explorer", "action": action,
                "method": pl.get("method"), "tools": pl.tools,
                "rationale": pl.get("rationale"),
                "lanes": [{"lane": x.idx, "intent": x.intent} for x in lanes],
                "n_lanes": len(lanes), "done": pl.get("done"),
                "outcome": f"计划: {action}, 工具={pl.tools}, "
                           f"分派 {len(lanes)} 条 lane"})
            lanes_seen += len(lanes)

            # ★ 修复② 重复方法检测**两种 action 都算** ★
            #   D 只在 execute 时计数, 于是"连着 probe 两次同一个错误假设"
            #   不会触发换路 —— 实测 v3digest 的 lean-04 正是 turn2/turn3 连 probe。
            fp = pl.fingerprint
            if used_fps.get(fp, 0) >= 2:
                trace[-1].update({
                    "outcome": f"这个方法({action})已经原样试过 2 次, 不给第三次机会",
                    "lesson": "换一条根本不同的路", "lesson_given": True})
                continue
            used_fps[fp] = used_fps.get(fp, 0) + 1

            # ★ 修复③ 聚焦 catalog: 只渲染计划里点名的工具 ★
            want = list(pl.tools)[:MAX_FOCUS_TOOLS]
            for ln in lanes:
                want = list(dict.fromkeys(want + list(ln.tools)[:2]))[:MAX_FOCUS_TOOLS]
            cat, hit, did = focused_catalog(tools, want)
            if did:
                n_focused += 1
                trace[-1]["catalog_focus"] = hit

            runners = []
            for ln in lanes:
                def _mk(ln=ln):
                    def _run(**_kw):
                        board = board_intents(lanes, for_lane=ln.idx)
                        plan = {"method": ln.intent or pl.get("method"),
                                "rationale": pl.get("rationale"),
                                "tools": ln.tools or pl.tools}
                        if action == "probe":
                            return _WK.run_probe(
                                lane_prompt(problem, lane=ln, board=board,
                                            catalog=cat, probe=True,
                                            probe_spec=pl.get("probe_spec") or "",
                                            hints=hints),
                                plan, scratch, timeout=180, catalog=cat,
                                prior=list(trace), **llm_kw)
                        return _WK.run_execute(
                            lane_prompt(problem, lane=ln, board=board,
                                        catalog=cat, probe=False, hints=hints),
                            plan, scratch, timeout=420, catalog=cat,
                            prior=list(trace), **llm_kw)
                    return _run
                runners.append(_mk())

            results = run_lanes(runners)
            peak_conc = max(peak_conc,
                            max((int(r.get("_concurrent") or 1) for r in results),
                                default=1))

            for r in results:
                trace.append({
                    "turn": turn, "role": "worker", "action": action,
                    "lane": int(r.get("lane") or 0),
                    "method": pl.get("method"), "tools": pl.tools,
                    "rationale": pl.get("rationale"),
                    "outcome": _WK.brief(r) if hasattr(r, "get") else str(r),
                    "error_type": r.get("error_type"),
                    "has_artifact": r.get("has_artifact"),
                    "artifact": r.get("artifact"),
                    "tool_calls": r.get("tool_calls"),
                    "findings": r.get("findings"),
                    "switches": r.get("switches"),
                    "lane_error": r.get("error"),
                    # ★ 把"形状重试"的原因落盘 ★
                    #   实测 2026-10-11 跑满 40 题: 324 次 LLM 调用里
                    #   **134 次是形状重试**(每轮 1.17 次, 占全部调用的 41%) ——
                    #   是单项最大的浪费来源, 比工具调用、比 verifier 都大。
                    #   而 d.py 的 trace 条目里**没有**这个字段, 于是
                    #   "模型到底为什么重试"在数据里**一个字都看不到**。
                    #   没有它就没法优化 —— 这是可观测性缺口, 不是算法问题。
                    "shape_problems": r.get("shape_problems") or [],
                    "n_llm_calls": r.get("n_llm_calls"),
                })

            # ★ 修复① probe 轮: 仍然不收工(那是 probe 的本意), 但**把制品留下** ★
            #   D 在这里直接 continue, 于是这一轮 lane 的成果全丢。
            #   实测 4 轮里 lean-04 三次"做对了交不出去"就是这么来的。
            if action == "probe":
                for r in results:
                    if r.get("has_artifact") and r.get("artifact") \
                            and not _is_placeholder(r):
                        if fallback is None:
                            fallback = r
                            fallback_note = (f"turn{turn} 的 probe 轮里 lane"
                                             f"{r.get('lane')} 交出了制品")
                trace.append({
                    "turn": turn, "role": "note", "action": None,
                    "outcome": "probe 轮不收工, 但制品已留作兜底"
                               + (f"({fallback_note})" if fallback else "(本轮无制品)"),
                    "lesson": None, "lesson_given": False})
                continue

            good = [r for r in results
                    if r.get("has_artifact") and r.get("artifact")]

            if bool(pl.get("done")) and good:
                trace.append({
                    "turn": turn, "role": "note", "action": None,
                    "outcome": "Explorer 判定 done 且本轮已交出制品 -> 收工, "
                               "不再消耗剩余轮次",
                    "lesson": None, "lesson_given": False})
            if not good:
                trace.append({
                    "turn": turn, "role": "note", "action": None,
                    "outcome": f"{len(lanes)} 条 lane 都没交出答案",
                    "lesson": "收卷时必须输出 "
                              '{"final": "...", "reason": "..."}',
                    "lesson_given": True})

            real = [r for r in good if not _is_placeholder(r)]
            if real:
                good = real
                dropped = len([r for r in results
                               if r.get("has_artifact") and r.get("artifact")
                               and _is_placeholder(r)])
                if dropped:
                    trace.append({
                        "turn": turn, "role": "note", "action": None,
                        "outcome": f"{dropped} 条 lane 交出的是占位符"
                                   "('...' / '你的答案' / 空), 已剔除",
                        "lesson": None, "lesson_given": False})
            if fallback is not None and not good:
                good = [fallback]
                trace.append({
                    "turn": turn, "role": "note", "action": None,
                    "outcome": "本轮 execute 没成果, 用先前 probe 轮留下的制品兜底",
                    "lesson": None, "lesson_given": False})

            if good:
                agree_n, n_good, agree_why = _agreement(results)
                disputed = bool(n_good >= 2 and agree_n < 2)
                best = good[0]
                return _finish({
                    "verdict": "DISPUTED" if disputed else "WORKER_ONLY",
                    "artifact": best.get("artifact"),
                    "evidence": None,
                    "trace": list(trace), "turns": budget.turns,
                    "seconds": round(time.time() - started, 1),
                    "budget_exhausted": False,
                    "n_lanes": len(lanes), "lanes_seen": lanes_seen,
                    "concurrent": peak_conc,
                    "catalog_focused_rounds": n_focused,
                    "agree": agree_n, "n_good": n_good,
                    "agreement_note": agree_why,
                    "note": f"lane{best.get('lane')} 交出制品"
                            + (f"(共 {len(lanes)} 条 lane 并发)"
                               if len(lanes) > 1 else "")
                            + f"。{agree_why}",
                })

        # ---- 预算耗尽: 用 probe 轮留下的兜底制品, 别交空答案 ----
        if fallback is not None:
            trace.append({
                "turn": budget.turns, "role": "note", "action": None,
                "outcome": f"预算耗尽, 但 {fallback_note} —— 交出它, 不交空答案",
                "lesson": None, "lesson_given": False})
            return _finish({
                "verdict": "INCONCLUSIVE", "artifact": fallback.get("artifact"),
                "evidence": None, "trace": list(trace), "turns": budget.turns,
                "seconds": round(time.time() - started, 1),
                "budget_exhausted": True, "n_lanes": 0,
                "lanes_seen": lanes_seen, "concurrent": peak_conc,
                "catalog_focused_rounds": n_focused,
                "agree": 0, "n_good": 1,
                "agreement_note": "来自 probe 轮的兜底制品, 没有交叉验证",
                "note": f"预算耗尽, 用 {fallback_note}",
            })

        return {
            "verdict": "INCONCLUSIVE", "artifact": None, "evidence": None,
            "trace": list(trace), "turns": budget.turns,
            "seconds": round(time.time() - started, 1),
            "budget_exhausted": True, "n_lanes": 0,
            "lanes_seen": lanes_seen, "concurrent": peak_conc,
            "catalog_focused_rounds": n_focused,
            "note": "预算耗尽, 没能判定 —— 那是诚实结论, 不是失败",
            "verify": None,
        }
    finally:
        if own_dir:
            shutil.rmtree(scratch, ignore_errors=True)
