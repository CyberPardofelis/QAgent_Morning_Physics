"""Worker —— 按 Explorer 给的方法**调工具**, 交回证据。

## ★ 这个角色在 2026-10-10 被整个换掉了 ★

### 旧形态(为什么废掉)

Worker 收到一句话"按计划把问题做出来", 然后被要求交一段**完整的 Python 程序**,
由它自己去 import 库、自己 print 出 `ANSWER: {...}` 一行, 再丢进白名单沙箱跑。
那一整套约定连带出四类与数学毫无关系的失败(实测 part1 13 题的聚类):

    NameError          3 次   忘了 import / 写错名字
    SandboxRejected    3 次   用了沙箱禁掉的东西(os/globals/open/...)
    NotCode            2 次   直接把答案 JSON 或散文当代码交上来
    NoArtifact         1 次   跑通了但没打 ANSWER: 行

最刺眼的一条: `z3-03` 第 2 轮那个 JSON **就是正确答案**(max=14/5),
却被"必须交程序不交答案"这条规则拒了, 又烧掉两轮预算, 最后 INCONCLUSIVE。

★ 那些规则是**沙箱的伴生物**, 不是能力要求 ★
原生工具调用不需要它们: 工具实现由我写一次、测一次、schema 固定, 模型只出参数。

### 新形态

模型每一步只输出一个 JSON:

    {"tool": "工具名", "args": {...}}      调工具
    {"final": "答案", "reason": "..."}     答案出来了
    {"give_up": "卡在哪一步"}               做不下去了

`args` 走 `registry.call_tool` **结构化校验**后执行, 结果原样回灌给下一轮。
于是: 不用 import、不过沙箱、没有 ANSWER: 协议、没有"交程序还是交答案"的两难。

实测(C 路线 36 题直跑): 每题 2~3 次 LLM 调用, 判对 33/35。

## 形状纠正为什么**不占**步数预算

模型常把内容想清楚了却没包成 JSON(实测有一次吐了 29130 字符散文)。
那是**协议**失败不是进展失败, 计入预算等于拿整道题的思考额度去罚一个括号。
所以每一步之内允许 `SHAPE_RETRIES` 次纠正重试, 且要**说清差在哪个键** ——
只说"格式错了"的话它下一句还会吐散文。

## 这个角色的 schema 里仍然没有 verdict

不变。**宣布结论不要钱, 交出工具算出来的东西才费力气。**
Worker 交的是工具的原始返回 + 模型自己的 `final` 文本, 不交"我认为这是最优的"。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from . import _registry as _reg
from ._util import ask, extract_json

__all__ = ["WorkerResult", "run_probe", "run_execute", "parse_artifact",
           "SHAPE_RETRIES", "shape_problem"]

#: 每步之内, 输出形状不对时的纠正重试次数(不占步数预算)
SHAPE_RETRIES = 2

#: 回灌给模型的文本上限。★ 不用 `json.dumps(...)[:N]` 盲切 ★
#:   那样会把 traceback 的**尾部**(也就是异常类型和消息)切掉,
#:   而那正是模型改代码要的唯一信息。
MAX_FEEDBACK = 6000

#: 对话里保留最近几轮的**完整**工具返回。
#: ★ 实测 cvxpy-04 prompt 涨到 24k ★ —— 见 `_run_loop` 里的压缩逻辑。
#:   2 轮够模型看清"刚试过什么、结果如何", 又不会让 prompt 失控。
KEEP_TURNS = 2

# ============================================================== 收敛机制
#: 同一个工具连调这么多次, 就认定它进了循环
STALL_N = 3
#: 剩下的步数少于这个数, 强制要求换路/交卷(最后一次机会)
FORCE_SWITCH_AT = 3


def _err_class(err: str) -> str:
    """把一条工具错误归成**类别** token(不是具体名字)。

    ★ 为什么要按"类别"而不是按"名字"判循环 ★
      2026-10-11 实测 lean-04-multistep: 17 次工具调用, **0 次成功**,
      其中 11 次是"猜一个不存在的 Mathlib 定理名" ——
      `Nat.sum_choose_eq` → `binomial_theorem` → `sum_range_choose` → …
      每次名字都不同, 但**根因完全一样**: 记不住 Mathlib 的 API。

      而 stall 检测原来只数"连续调了同一个工具几次"
      (`for x in reversed(tried): if x != name: break`)。
      这 17 次全是 `lean_check`, 于是 stall **触发了 6 次**, 每次都告诉模型
      "你已经连续 N 次调同一个工具了, **换一个工具**"。

      ★ 而 `lean_check` 是这道题**唯一**能用的工具 ★
      换工具的建议在这道题上是有害的 —— 它把唯一的路也堵上了。
      模型确实换了(去 `#check` 枚举), 但换的是**用法**, 不是工具, 于是照旧。

    ⇒ 于是把"卡住"的定义从「同一个工具连着调 N 次」放宽成
      「**同一种错误**连着出现 N 次」。后者才对应真正的循环。
    """
    s = str(err or "")
    # 顺序有意义: 先匹配更具体的
    #
    # ★ `unknown constant` 与 `unknown identifier` 必须归成**同一类** ★
    #   实测 lean-04 那 7 次里, 前 4 次是前者、后 2 次是后者, 根因完全
    #   一样("这个名字不存在")。我第一版把它们分成两类, 于是从后往前数
    #   连续同类只够得着 2 次, 永远触发不了 STALL_N=3 —— 改动等于没做。
    #   归类必须按**根因**, 不按错误信息的措辞。
    for pat, tag in (
        (r"unknown constant|unknown identifier|no declaration|"
         r"not a valid|unresolved name|failed to synthesize|"
         r"omega could not prove|unsolved goals|application type mismatch|"
         r"type mismatch", "name-or-shape"),
        (r"field notation|unexpected token|expected token|parse|syntax",
         "syntax"),
        (r"参数类型不对|缺参数|SandboxRejected", "contract"),
        (r"Timeout|timed out|超时", "timeout"),
        (r"AssertionError", "assertion"),
        (r"AttributeError", "attribute"),
        (r"TypeError", "typeerror"),
        (r"NameError", "nameerror"),
        (r"ValueError", "valueerror"),
        (r"KeyError", "keyerror"),
        (r"ZeroDivision", "divzero"),
        (r"IndexError", "index"),
    ):
        if re.search(pat, s, re.I):
            return tag
    return "other" if s else "none"


def _failed_ids(findings: list[dict]) -> list[str]:
    """从已失败的调用里抠出"它试过的标识符/名字"。

    ★ 为什么要这个 ★
      模型记不住 Mathlib 的名字, 于是换一个名字再试 —— 而 harness
      **已经知道**它试过哪些了(`unknown constant 'X'` 里就有 X),
      却从来没把这些名字告诉它。加进提醒里, 就是零成本的"记忆外挂"。
    """
    out: list[str] = []
    for d in (findings or []):
        if not isinstance(d, dict):
            continue
        for m in re.finditer(
                r"(?:unknown constant|unknown identifier|invalid "
                r"field notation with '@'|no declaration|not found)\s*"
                r"['\"]?([A-Za-z_][A-Za-z0-9_'.]*)", str(d.get("error") or "")):
            # ★ 必须 rstrip 掉尾引号 ★
            #   Lean 的名字带点(`Nat.sum_choose_eq`), 所以字符类里必须留 `'`,
            #   但那样**结尾的引号也会被一起吞掉** ->
            #   `Nat.sum_choose_eq'`。这个字符串是要**回灌给模型**的,
            #   它照抄一个带尾巴的名字, 下一次照样 unknown constant。
            nm = m.group(1).rstrip("'")
            if nm and nm not in out:
                out.append(nm)
    return out[:12]


def _loop_note(tool: str, times: int, tried: list,
               findings: list | None = None,
               only_tool: bool = False) -> str:
    """告诉模型它在一个循环里, 并点名已经试过的工具。"""
    others = list(dict.fromkeys(t for t in tried if t != tool))
    if others:
        alt = "\n你已经试过(也都卡住了): " + ", ".join(others)
    elif only_tool:
        # ★★★ 这里原来输出的是"你还没试过别的工具" ★★★
        #   对"只有一个工具可用"的题, 那是一句**有害**的话:
        #   模型于是以为"该换个工具", 而能换的都换不动。
        #   实测 lean-04: 17 次全是 lean_check, 提醒里点名"你还没试过
        #   别的工具", 模型就去 `#check` 枚举常量 —— 换了用法, 没换根因。
        alt = ("\n★ 这道题**只有这一个工具**能用, 换工具不是出路 ★\n"
               "要换的是**思路**: 换建模方式 / 换一个更小的规模先试 / "
               "把你已经确认的中间结论直接组织成答案。")
    else:
        alt = "\n你还没试过别的工具。"
    names = _failed_ids(findings or [])
    black = ""
    if names:
        black = ("\n★ 你已经试过下面这些名字, **它们都不存在**, "
                 "不要再原样试了 ★\n  " + ", ".join(names))
    return (
        f"\n★ 你已经**连续 {times} 次**调同一个工具 `{tool}` 了 ★\n"
        f"{alt}{black}\n"
        f"**不要再原样调它**。要么换一个工具, 要么换一个思路"
        f"(换建模方式 / 换数值方法 / 降规模先看小例子 / "
        f"把已经算出来的中间结果直接组织成答案)。\n"
        f"★ 如果你其实已经有足够的信息了, **现在就交 final** —— "
        f"交一个基于已有证据的答案, 远好过烧光预算什么都不交。 ★")


def _force_note(left: int) -> str:
    return (
        f"\n★ 只剩 {left} 步了 ★\n"
        f"**这两步内必须做决定**:\n"
        f"  · 你已经有答案/证据了 -> 立刻 `{{\"final\": ...}}`\n"
        f"  · 还没有 -> 换一个**和之前完全不同**的工具或思路\n"
        f"  · 真的做不出来 -> `{{\"give_up\": ...}}`, 说清卡在哪\n"
        f"★ 交一个不完整但**基于真实计算**的答案, 好过什么都不交 ★")


def _clip_tool_result(obs: dict) -> str:
    """把工具返回整理成给模型看的文本 —— **异常消息必须排在最前面**。"""
    if not isinstance(obs, dict):
        return str(obs)[:MAX_FEEDBACK]
    keep = {k: v for k, v in obs.items()
            if k in ("ok", "value", "status", "error_type", "error",
                     "proved", "reason", "detail", "assignments", "model",
                     "solution", "solutions", "classes", "candidates",
                     "note", "var_domains", "domain_sensitive",
                     "relaxation_value", "bound", "bound_agrees_with_value",
                     "prelude_added", "sorry_used", "aborted", "gap_errors",
                     "_coerced_types", "_data_loaded", "_data_errors",
                     "elapsed_ms", "found", "path", "length", "n_classes",
                     "non_abelian_ids", "simple_ids", "missing_fields",
                     "computed_fields", "element_order_multiset",
                     "seq_note", "terms_note", "stdout")}
    body = json.dumps(keep, ensure_ascii=False, default=str)
    if len(body) <= MAX_FEEDBACK:
        return body
    # ★ 太长时保头**也**保尾 ★ 尾部常有 exit_code / elapsed / 关键结论
    head = body[: MAX_FEEDBACK // 2]
    tail = body[-MAX_FEEDBACK // 2:]
    return head + "\n...[中间截断]...\n" + tail


def shape_problem(obj: dict | None, raw: str) -> str | None:
    """这个输出能不能直接执行? 能返回 None, 不能返回**具体**的原因。

    ★ 为什么必须具体 ★ 只说"格式错了", 模型下一句还是吐散文(实测如此)。
    """
    if obj is None:
        head = (raw or "").strip()
        if not head:
            return "你这条回复是空的"
        if head[0] != "{":
            return (f"你的回复开头是 {head[0]!r} 而不是 '{{' —— "
                    f"我要的**唯一**是一个 JSON 对象, 不要标题/解释/散文")
        return "没能配平出一个完整 JSON(可能是引号没转义或括号不配对)"
    if "final" in obj or "give_up" in obj:
        return None
    tool = obj.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        keys = sorted(map(str, obj))[:8]
        if any(k in obj for k in ("code", "source", "expr", "objective")):
            return (f"你给的键是 {keys} —— 那是**工具的参数**, 不是一次调用。"
                    f'要写成 {{"tool": "工具名", "args": {{这些参数}}}}')
        return f"JSON 里没有 tool 键(你给的是 {keys})"
    if "args" in obj and not isinstance(obj["args"], dict):
        return (f"args 必须是对象, 你给的是 {type(obj['args']).__name__}"
                f' —— 写成 {{"tool": ..., "args": {{...}}}}')
    return None


_SHAPE_HINT = """
★ 你上一条回复我没能取出可执行的动作 ★
{detail}
请**只**输出一个 JSON 对象 —— 不要标题、不要推理过程、不要散文。必须是:
  {{"tool": "工具名", "args": {{...}}}}              调一个工具
  {{"final": "答案", "reason": "怎么得到的"}}         已经有答案了
  {{"give_up": "卡在哪一步"}}                        做不下去了
工具名**逐字**来自上面的目录。哪怕你上一条已经把内容想清楚了, 也请**重新**
写成上面那个 JSON —— 我只读 JSON。
"""


class WorkerResult(dict):
    """Worker 的产出。注意: **没有 correct/verdict 字段, 也不该加**。"""

    @property
    def ok(self) -> bool:
        return bool(self.get("ran")) and not self.get("error_type")

    @property
    def artifact(self) -> dict | None:
        return self.get("artifact")


#: 键名像"对答案的断言"。配合"值是布尔/判定词"一起用, 单看任一个都会误伤。
_CLAIM_KEY = re.compile(
    r"^(is_|has_|proposition_|claim|conclusion|verdict|verified|unique|"
    r"optimal|minimal|irreducible|correct|holds|solved|proven|proved|"
    r"valid|exists|answer_is)", re.IGNORECASE)
_CLAIM_VALUES = frozenset({True, False, "true", "false", "yes", "no",
                           "proved", "refuted", "partial", "inconclusive",
                           "pass", "fail", "ok", "correct", "wrong"})


def strip_self_claims(art: dict | None) -> tuple[dict | None, list[str]]:
    """★ 剥掉制品里的"对答案的断言" ★

    实测 2026-10-10 smoke: 即使 prompt 里明写"不要交你对答案的评价",
    Worker 照样交上来 `{"proposition_is_false": true, "witness": {...}}`。
    **prompt 层的禁令拦不住** —— 宣布结论不要钱, 所以必须在数据层剥。

    不靠"已知字段名"白名单(结构上抓不到新名字), 靠**形状**:
    键名像断言 **且** 值是布尔/判定词, 才算自评。真的把布尔当答案的题不会误伤,
    因为光名字像不够, 光值是布尔也不够。
    """
    if not isinstance(art, dict):
        return art, []
    out, stripped = {}, []
    for k, v in art.items():
        key_like = bool(_CLAIM_KEY.match(str(k)))
        val_like = (v is True or v is False
                    or (isinstance(v, str) and v.strip().lower() in _CLAIM_VALUES))
        if key_like and val_like:
            stripped.append(f"{k}={v!r}")
            continue
        out[k] = v
    return out, stripped


def _tool_task_prompt(problem: str, plan: dict, catalog: str, *,
                      probe: bool, probe_spec: str = "",
                      prior: list | None = None) -> str:
    head = (f"你要做一次**{'小规模试验' if probe else '完整求解'}**。\n\n"
            f"【问题】\n{problem}\n\n"
            f"【同事给你的方法】\n{(plan.get('method') or '')[:1500]}")
    if plan.get("rationale"):
        head += f"\n他选这些工具的理由: {(plan.get('rationale') or '')[:600]}"
    if probe and probe_spec:
        head += f"\n\n【这次试验要看的】\n{probe_spec}"
    # ★★ 前几轮的调查结果 —— ReAct 闭环的另一半 ★★
    #   旧版 Worker **完全看不到** history: 它每轮只拿到 problem + method,
    #   于是每次都像第一次调这个工具, 前面算出来的中间结果全烂在
    #   上一轮的上下文里。
    #   实测 2026-10-10 (part2 零提示 20 题): 13 道没交出答案的题里
    #   **13 道在重复调同一个工具**(`numeric_code` 最多 x16) —— 那不是探索,
    #   是**信息断在 Worker 和 Explorer 之间**。
    if prior:
        blocks = []
        for i, h in enumerate(prior[-3:], start=1):
            f = h.get("findings") or {}
            lines = [f"  (前 {i} 轮) 第{h.get('turn')}轮: {str(h.get('method') or '')[:120]}"]
            for c in (f.get("calls") or [])[:5]:
                v = str(c.get("value") or c.get("error") or "")[:90]
                lines.append(f"      · {c.get('tool')}: {v}")
            if f.get("artifact_preview"):
                lines.append(f"      制品: {str(f['artifact_preview'])[:120]}")
            if f.get("stopped_because"):
                lines.append(f"      停在哪: {f['stopped_because']}")
            blocks.append("\n".join(lines))
        head += ("\n\n【前几轮已经查到的】(★ 别重复已经做过的事, 也别丢弃已得的结果 ★\n"
                 + "\n".join(blocks))
    return (
        head
        + "\n\n【怎么做事】\n"
        + "你**不写代码**去跑 —— 你只能**调工具**。\n"
        + "每一步只输出一个 JSON 对象, 不要任何别的文字:\n"
        + '  {"tool": "工具名", "args": {...}}        调工具\n'
        + '  {"final": "你的答案", "reason": "怎么得到的"}   答案出来了\n'
        + '  {"give_up": "卡在哪一步"}               做不下去了\n'
        + "★ 我会把工具的返回结果发回给你, 你再决定下一步 ★\n"
        + "★ `args` 必须符合工具的 inputSchema; 报类型错就按 hint 改 ★\n"
        + ("★ 这一轮**不需要**最终答案, 你的任务是把现象查清楚 ★\n"
           if probe else
           "★ 答案必须来自**工具的返回**, 不许心算填进去 ★\n")
        + "\n【这台机器上实际有什么】\n" + _reg.render_env()
        + "\n\n【可用工具】\n" + catalog)


def _call_ok(obs) -> bool | None:
    """这次工具调用**成功**了吗? —— 全文件**唯一**的判据。

    ★ 为什么不能写 `bool(obs.get("ok"))` ★
      `magi_tools.registry.call_tool` 的契约是「**失败时**一定返回
      `{"ok": False, "error": ...}`; **成功时原样返回工具自己的输出**」
      (`registry.py` 里就是一句 `return t["_fn"](**kw)`)。所以**成功路径
      根本没有 `ok` 键** —— `obs.get("ok")` 给 None, `bool(None)` 是 False。

      实测 2026-10-11 的 17 个工具, 成败信号分三族:

        · 有 `ok`            sympy_* / graph_* / numeric_code / oeis_lookup
        · `proved` 而无 `ok`  lean_check / gap_*      (证成 / 证伪)
        · `status` 无 `ok`   z3_smt2                  (sat / unsat / unknown)

      ★ 这个坑已经栽了两次 ★
        1. `_digest_one` 原来只认 `ok`, 于是 Lean 题的 findings 里
           `ok=None`, `brief()` 数成「0 次成功」—— 明明 proved=True。
        2. `tool_calls` 那条更彻底: 它只写了 `obs.get("ok")`, **连
           `proved` 都没带上**, 于是 `brief()` 的 `_succeeded()` 永远
           读不到任何 Lean 证据。实测 2026-10-11 D1 跑 lean-04:
           「调了 12 次工具(**0 次成功**): lean_check ...」,
           同一行 trace 里就写着 `lean_check: proved=True` ——
           **诊断信息自相矛盾, 看着像全失败**。

    ⇒ 判据: `ok` 在就用它; 不在就看 `proved`; 都不在就**没有 error 即成功**。

    ★★ 为什么 Lean 族和 z3 族**故意不对称** ★★
      这看着像不一致, 其实是两个不同的问题, 强行统一才会真出错:

      · Lean 家族有**明确的裁决字段** `proved`(True 证成 / False 证伪)。
        `findings` 会回到模型的下一轮, 若把"证伪"报成 ok=True, 模型会
        以为这条路走通了、继续在错的假设上打转。所以尊重 `proved`。
      · z3 家族的 `status: unsat` 只是"无解"。对"证明这就是全部解集"
        这种用法, unsat 恰恰是**想要的**结果 —— 它没有"我期望什么"这层
        语义, 所以那里只问"跑没跑通", 没有 error 就算成功。

      ⚠ 别把它们"顺手统一"成同一个语义。那会把 Lean 的证伪报成成功,
        直接影响模型下一步的行为。
    """
    if not isinstance(obs, dict):
        return None
    if "ok" in obs:
        return bool(obs.get("ok"))
    if "proved" in obs:
        return bool(obs.get("proved"))
    return "error" not in obs


def _digest_one(tool: str, args: dict, obs) -> dict:
    """一次工具调用 -> 一行**可决策**的摘要。

    ★ 要保留什么 ★ 不是 stdout 全文(那会撑爆上下文), 而是**能让人判断
      "这条路走到哪了"的东西**:
        · ok / status / value —— 算出来的结果
        · error_type + 首行 —— 为什么不行(Traceback 全文没人读)
        · 规模信息(耗时) —— 判断是不是"太大跑不动"
    """
    if not isinstance(obs, dict):
        return {"tool": tool, "ok": None, "value": str(obs)[:120]}
    d = {"tool": tool, "ok": _call_ok(obs)}
    if obs.get("error_type"):
        d["error"] = f"{obs.get('error_type')}: {str(obs.get('error'))[:90]}"
    # ★★ 编译/求解类工具**失败时必须带回"为什么" ★★
    #   交接文档记这条已修("11次→5次, 408s→233s"), 但实测 **一次都没触发过**:
    #   它找的是 `obs["errors"]` / `obs["output"]`, 而 `lean_check` 实际返回的键是
    #       proved / reason / **detail** / **output_tail** / exit_code / error_count
    #   —— `errors` 和 `output` **根本不存在**, 于是四个 if 全落空,
    #   `d["error"]` 恒为空。2026-10-11 重跑 40 题, lean-04 于是又盲改 21 次
    #   烧了 451 秒, 和"修好之前"一模一样。
    #
    #   ⇒ 这里**按优先级多试几个键**, 而不是押注某一个字段名 ——
    #     工具层改字段名时, 这个分支不该再静默退化成"只留 proved=False"。
    if obs.get("ok") is False or obs.get("proved") is False:
        # ① 结构化的报错列表(有的工具给 list, 给什么算什么)
        for key in ("errors", "error_list", "messages"):
            errs = obs.get(key)
            if isinstance(errs, list) and errs:
                d["error"] = "编译报错: " + " | ".join(
                    str(e).strip().splitlines()[0][:90] for e in errs[:3])
                break
        # ② 编译器给的**第一手诊断**(lean_check 走的是这条)
        if "error" not in d:
            det = str(obs.get("detail") or "").strip()
            if det:
                d["error"] = "诊断: " + " | ".join(
                    x[:90] for x in det.splitlines() if x.strip())[:180]
        # ③ 完整输出的尾巴(短版本)
        if "error" not in d:
            for key in ("output_tail", "output", "stdout"):
                txt = str(obs.get(key) or "").strip()
                lines = [x for x in txt.splitlines() if x.strip()][:3]
                if lines:
                    d["error"] = f"{key}: " + " | ".join(x[:90] for x in lines)
                    break
    for k in ("status", "proved", "found", "n_candidates", "n_classes",
              "elapsed_ms"):
        if k in obs and obs[k] is not None:
            d["value"] = f"{k}={str(obs[k])[:60]}"
            break
    if "value" in obs and obs["value"] is not None:
        d["value"] = json.dumps(obs["value"], ensure_ascii=False,
                                default=str)[:200]
    if "error" not in d and not obs.get("ok") and obs.get("error"):
        first = str(obs["error"]).strip().splitlines()
        d["error"] = (first[-1] if first else str(obs["error"]))[:110]
    return d


def _build_findings(res: dict) -> dict:
    """Worker 这一轮的**调查摘要** —— 交回给 Explorer 的正式产物。"""
    f: dict = {"calls": res.get("findings_calls") or []}
    art = res.get("artifact")
    if isinstance(art, dict) and art.get("answer"):
        f["artifact_preview"] = str(art.get("answer"))[:300]
    elif res.get("final_answer"):
        f["artifact_preview"] = str(res.get("final_answer"))[:300]
    et = res.get("error_type")
    if et:
        f["stopped_because"] = {
            "UnparsedOutput": "模型输出形状不对, 没交出动作",
            "GaveUp": "模型主动放弃",
            "RanOutOfSteps": "步数用完, 没交出答案",
            "BackendUnwired": "后端不可用",
        }.get(et, et)
    sw = res.get("switches") or []
    if sw:
        f["switched"] = [f"第{x.get('step')}步:{x.get('trigger')}" for x in sw[:6]]
    n = len(f["calls"])
    if n:
        f["summary"] = f"共调 {n} 次工具: " + ", ".join(
            f"{c['tool']}({'ok' if c.get('ok') else 'fail'})" for c in f["calls"][:8])
    return f


def _run_loop(problem: str, plan: dict, catalog: str, *, probe: bool,
              scratch_dir: str, timeout: int, prior: list | None = None,
              **kw) -> WorkerResult:
    """一步一个 JSON -> 调工具 -> 回灌结果。最多 `max_steps` 步。"""
    max_steps = 4 if probe else 8
    spec = (plan.get("probe_spec")
            or "规模尽量小, 只要能判断这条路可行不可行就够了")
    convo = [_tool_task_prompt(problem, plan, catalog, probe=probe,
                               probe_spec=spec, prior=prior)]
    #: ★ ReAct 闭环: 逐步攒下"每次调工具拿到了什么", 收尾时交回 Explorer ★
    #: (旧版只交 brief(), 而 brief 是"最后一步的尾部" —— 信息量最低的一段)
    findings: list[dict] = []
    res: WorkerResult = WorkerResult({"ran": True, "tool_calls": [],
                                      "lang": "tool", "scratch": scratch_dir,
                                      # ★ 所有会被 `res[...] = res[...] or ...`
                                      #   读到的键, **初始化时就给 None** ★
                                      #   我第一版只写了 ran/tool_calls/lang/scratch,
                                      #   于是 `res["error_type"] = res["error_type"] or "..."`
                                      #   在真出错的分支上直接 `KeyError: 'error_type'` ——
                                      #   而正常路径永远不读它, 所以不跑到就不知道。
                                      #   "字典子类 + 读后写" 这个模式, 初始化必须自备全部键。
                                      "error_type": None, "stderr": "",
                                      "has_artifact": False, "artifact": None,
                                      "stripped_self_claims": [],
                                      "final_answer": None, "final_reason": "",
                                      "lean": None, "last_tool_result": None,
                                      "shape_problems": [], "switches": []})
    last_obs: dict | None = None
    ctx = {"lean_proved": None, "lean_sorry": None, "lean_reason": None}

    for k in range(1, max_steps + 1):
        act = None
        for attempt in range(SHAPE_RETRIES + 1):
            raw = ask("\n\n".join(convo), temperature=0.2, **kw)
            obj = extract_json(raw)
            why = shape_problem(obj, raw)
            if why is None:
                act = obj
                break
            res.setdefault("shape_problems", []).append(
                {"step": k, "attempt": attempt, "why": why})
            if attempt < SHAPE_RETRIES:
                convo.append(_SHAPE_HINT.format(detail=why))
                convo.append(f"你上一条回复的开头是: {(raw or '')[:160]!r}")
                continue
            break
        if act is None:
            res["error_type"] = res["error_type"] or "UnparsedOutput"
            res["stderr"] = (str(res.get("shape_problems") or "")[-300:])
            res["findings_calls"] = findings
            res["findings"] = _build_findings(res)
            return res

        if "give_up" in act:
            res["error_type"] = res["error_type"] or "GaveUp"
            res["stderr"] = str(act.get("give_up"))[:400]
            res["findings_calls"] = findings
            res["findings"] = _build_findings(res)
            return res
        if "final" in act:
            res["final_answer"] = str(act.get("final"))
            res["final_reason"] = str(act.get("reason") or "")[:600]
            art, stripped = strip_self_claims(
                {"answer": res["final_answer"], "reason": res["final_reason"],
                 "tool_result": last_obs})
            res["artifact"] = art
            res["stripped_self_claims"] = stripped
            res["has_artifact"] = True
            res["findings_calls"] = findings
            res["findings"] = _build_findings(res)
            return res

        name = str(act["tool"]).strip()
        args = act.get("args") if isinstance(act.get("args"), dict) else {}
        t0 = time.time()
        obs = _reg.call(name, args)
        # ★★★ 这一步是 ReAct 闭环的关键: 把"拿到了什么"**结构化攒起来** ★★★
        #   旧版只把返回塞进本轮的 convo, 然后 Worker 结束时只交一个
        #   `brief()`(最后一步的尾部截断) 给 Explorer。
        #   ⇒ Explorer 看到的"发生了什么"是信息量最低的那一段,
        #      它既没法判断这条路走到哪了, 也无法据此决定换不换路。
        #   实测 2026-10-10 (part2 零提示 20 题): 13 道没交出答案的题
        #   里有 13 道在重复调同一个工具(`numeric_code` 最多 x16) ——
        #   因为**每一轮 Worker 都从零开始**, 前几轮的中间结果全烂在
        #   上一轮上下文里, 谁也没把结论交出去。
        findings.append(_digest_one(name, args, obs))
        res["tool_calls"].append({"tool": name, "args": args,
                                  "ms": int((time.time() - t0) * 1000),
                                  # ★ 必须用 `_call_ok`, 不能 `obs.get("ok")` ★
                                  #   这条记录是 `brief()` 里「N 次成功」的唯一
                                  #   数据源; 原来只写 obs.get("ok"), 于是
                                  #   lean_check(proved) / z3_smt2(status) 调了
                                  #   多少次都被数成 0 次成功 —— 见 `_call_ok`
                                  #   的 docstring。
                                  "ok": _call_ok(obs)})
        if name == "lean_check" and isinstance(obs, dict):
            ctx["lean_proved"] = obs.get("proved")
            ctx["lean_sorry"] = bool(obs.get("sorry_used"))
            ctx["lean_reason"] = obs.get("reason")
            # ★ 从 ctx 逐项取, 别写 `ctx["lean"]` ★ 我第一版那么写了,
            #   而 ctx 里从来没有 "lean" 这个键 —— 于是**第一道 Lean 题直接崩**
            #   (KeyError: 'lean'), 整道题 CRASH。
            #   一个只在某一条分支上执行的赋值, 不跑到它就永远不知道错了;
            #   这和"只有某条路径才走的死锁"是同一类。
            res["lean"] = {k: ctx[k] for k in
                           ("lean_proved", "lean_sorry", "lean_reason")}
        last_obs = obs
        convo.append(f"【第 {k} 步: 你调了 {name}】\n工具返回:\n"
                     + _clip_tool_result(obs)
                     + "\n\n请给出你的下一个 JSON:")
        # ★ 对话必须有上限 ★ 实测 2026-10-10 (cvxpy-04-sos):
        #   SDP 求解结果动辄几千字符, 我每步都**原样**追加, 8 步下来 prompt
        #   涨到 **24k 字符**。后果有两个:
        #     · 每轮都变慢(那一题烧了 18.6 分钟 / 14 次工具调用);
        #     · 更危险的是**挤掉 max_tokens** —— 答案 JSON 可能被截断,
        #       而截断的 JSON 表现为"模型没交出东西", 看着像它不会。
        #   ⇒ 保留**最近**若干轮, 更早的压成一行摘要(只留工具名与成败)。
        #     这不是丢信息: 被压掉的那些轮模型已经读过了, 它要的是"我试过什么"。
        if len(convo) > 2 * KEEP_TURNS + 1:
            head = convo[0]                       # 题面永远保留
            old = convo[1:-2 * KEEP_TURNS]
            hist = "\n".join(
                f"  (更早) 第{i}步: {re.search(r'你调了 (\\w+)', t).group(1) if re.search(r'你调了 (\\w+)', t) else '?'}"
                for i, t in enumerate(old, 2))
            convo[:] = [head, f"【你已经试过的】\n{hist}"] + convo[-2 * KEEP_TURNS:]

        # ---- 收敛: 同一个工具连调就明确推它换路 ----
        # ★ 为什么必须有 ★ 实测 2026-10-10 (part2 零提示 20 题):
        #   13 道"没交出答案"的题里, **13 道都在重复调同一个工具** ——
        #   `numeric_code` 最多 x16、`gap_eval` x10、`lean_check` x8,
        #   而"唯一工具数"几乎全是 1。那不是探索, 是**卡在循环里**。
        #
        #   ★ 光在 SYSTEM 提示里写"要换路"没用 ★
        #   模型**看不到自己调了几次** —— 它不知道"我已经调了 8 次了"。
        #   ⇒ 必须由 harness 计数, 然后把事实告诉它。
        tried = [x.get("tool") for x in res["tool_calls"]]
        # ★ 必须数**连续**次数, 不是累计 ★
        #   第一版数的是 `sum(1 for x in tried if x == name)` —— 那是**累计**,
        #   于是交替调 A/B/C 各 3 次也会被判"卡在 A 上"(实测交替调用确实误触发了)。
        #   "卡住"的定义是**连着几次都在同一条路上**, 而不是"这条路过很多次"。
        n_same = 0
        for x in reversed(tried):
            if x != name:
                break
            n_same += 1
        # ★ 还要看**错误类别**有没有在原地打转 ★
        #   只数工具名会漏掉最贵的那种卡住: 同一个工具、每次试**不同的**
        #   错东西。实测 lean-04 的 17 次全是 lean_check 但根因是同一个
        #   ("记不住 Mathlib 的定理名"), 工具计数完全看不出它在打转。
        n_cls = 0
        cur_cls = ""
        # ★★★ 循环变量**必须叫别的名字** ★★★
        #   我第一版写成 `for k in reversed(findings)` ——
        #   而外层 `for k in range(1, max_steps+1)` 的 `k` 就是**第几步**。
        #   于是内层循环结束后 `k` 变成了 findings 的一个元素(字符串),
        #   紧接着 `left = max_steps - k` 直接
        #     TypeError: unsupported operand type(s) for -: 'int' and 'str'
        #   而 `check_convergence.py` / `check_react_loop.py` 当场崩掉。
        #
        #   教训: 我给 `_err_class` / `_failed_ids` / `_loop_note` 都写了单测,
        #   全绿 —— **但那些是纯函数, 不经过这个循环**。
        #   "测过了"和"这条路径被执行过"是两件事。
        for _d in reversed(findings):
            if not isinstance(_d, dict):
                break
            _c = _err_class(_d.get("error") or "")
            if _c in ("none", "other"):
                break
            if not cur_cls:
                cur_cls = _c
            elif _c != cur_cls:
                break
            n_cls += 1
        left = max_steps - k
        if n_same >= STALL_N or n_cls >= STALL_N:
            res.setdefault("switches", []).append(
                {"step": k, "tool": name, "times": n_same,
                 "err_class": cur_cls, "n_same_err": n_cls,
                 "trigger": "stall"})
            # ★ 这道题能用几个工具? 只有 1 个时不能说"换工具" ★
            convo.append(_loop_note(
                name, max(n_same, n_cls), tried, findings,
                only_tool=len(set(t for t in tried if t)) <= 1))
        # ★ 和上面的 elif 改成独立 if ★
        #   用 elif 时, 只要 stall 触发了 budget_low 永远不会触发 ——
        #   而"要交卷了"和"你卡住了"是**两件都要说**的事。
        if left <= FORCE_SWITCH_AT:
            res.setdefault("switches", []).append(
                {"step": k, "trigger": "budget_low", "left": left})
            convo.append(_force_note(left))

    res["error_type"] = res["error_type"] or "RanOutOfSteps"
    res["last_tool_result"] = last_obs
    res["findings_calls"] = findings
    res["findings"] = _build_findings(res)
    return res


def run_probe(problem: str, plan: dict, scratch_dir: str, *,
              timeout: int = 90, catalog: str | None = None,
              prior: list | None = None, **kw) -> WorkerResult:
    """小规模实验。目的是拿证据, 不是出答案 —— 所以不强制 final。"""
    return _run_loop(problem, plan, catalog or _reg.render_catalog(),
                     probe=True, scratch_dir=scratch_dir,
                     timeout=timeout, prior=prior, **kw)


def run_execute(problem: str, plan: dict, scratch_dir: str, *,
                timeout: int = 420, catalog: str | None = None,
                prior: list | None = None, **kw) -> WorkerResult:
    """全量求解, 产出可复核的制品。"""
    return _run_loop(problem, plan, catalog or _reg.render_catalog(),
                     probe=False, scratch_dir=scratch_dir,
                     timeout=timeout, prior=prior, **kw)


def _lean_prelude(code: str) -> str:
    """补上 Lean 编译必需的抬头。**实现搬到了工具层**, 这里只做转发。

    ## 为什么不再自己补

    抬头的真正实现是 `magi_tools.tools.lean._with_prelude`, 因为 `lean_check`
    自己也必须补 —— 不然直接调 `lean_check("theorem t : 1+1=2 := by norm_num")`
    会因为缺 `import Mathlib` 报 unknown tactic(实测 0.2 秒就挂)。

    同一件事在两层各写一遍 = 两边迟早不一致。2026-10-10 就是这么出事的:
    tools 层补对了, roles 层这份旧实现还在按"只补 scoped"的逻辑拼,
    拼出 `open scoped BigOperators` 排在 `import Mathlib` **前面**的源码 ——
    而 Lean 要求 import 必须最先, 于是"自带 import 的源码反而编译不过"。

    留这个名字是因为 `verifier.py` 还在 import 它; 删一个被引用的名字
    比留一个转发函数更容易在某天变成 ImportError。
    """
    try:
        from magi_tools.tools.lean import _with_prelude
    except Exception:  # noqa: BLE001 - 工具层不在时退回"原样返回"
        return code
    return _with_prelude(code, True)[0]


def looks_like_code(src: str) -> tuple[bool, str]:
    """这段东西**是程序**吗? 返回 (是不是, 原因)。

    ## 现在它还用得上吗

    Worker 不再自己写程序了, 所以主路径不再用它。但 `check_extract_code.py`
    仍在测它, 而"一段 JSON 数据 / 一段散文不是程序"这条判据本身没错 ——
    它属于**代码判别**而不是 Worker 的职责。留着, 不再从 Worker 调用。

    判据: 能被 `json.loads` 解析 -> 是数据; 只有一个字面量表达式 -> 是数据。
    """
    import ast

    s = (src or "").strip()
    if not s:
        return False, "空的"
    try:
        import json as _json
        _json.loads(s)
        return False, f"这是**一段 JSON 数据**({s[:40]}...), 不是程序"
    except Exception:  # noqa: BLE001
        pass
    try:
        tree = ast.parse(s)
    except SyntaxError as e:
        return False, f"不是可解析的 Python: {e.msg}"
    nodes = [n for n in ast.walk(tree)
             if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign,
                               ast.Expr, ast.For, ast.While, ast.If,
                               ast.With, ast.FunctionDef, ast.ClassDef,
                               ast.Import, ast.ImportFrom, ast.Return))]
    if not nodes:
        return False, "里面没有任何可执行语句"
    if len(nodes) == 1 and isinstance(nodes[0], ast.Expr) \
            and isinstance(nodes[0].value, (ast.Constant, ast.Dict,
                                           ast.List, ast.Tuple, ast.Set)):
        return False, "只有一个数据字面量, 没有计算"
    return True, ""


def parse_artifact(stdout: str) -> dict[str, Any] | None:
    """兼容旧协议: 从 stdout 抓 `ANSWER: {...}`。

    ★ 新形态下 Worker 不再 print 这一行 ★ 这条留着是因为编排器之外
    还有地方 import 它(verifier 的旧路径), 而删掉一个被引用的名字
    比留一个空函数更容易在某天变成 ImportError。
    """
    if not stdout:
        return None
    for m in re.finditer(r"^\s*(CONSTRUCT|ANSWER|COUNTEREXAMPLE)\s*:",
                         stdout, re.IGNORECASE | re.MULTILINE):
        j = stdout.find("{", m.end())
        if j < 0:
            continue
        try:
            v = json.loads(stdout[j:stdout.find("\n", j) if "\n" in stdout[j:] else len(stdout)])
            if isinstance(v, dict):
                return v
        except Exception:  # noqa: BLE001
            continue
    return None


def brief(r: WorkerResult, limit: int = 900) -> str:
    """给 Explorer 看的一句话摘要。"""
    bits = []
    calls = r.get("tool_calls") or []
    if calls:
        # ★ 成功要认 `ok` **和** `proved` ★
        #   Lean 工具成功时不填 `ok`(它用 `proved` 表达), 于是这里永远数出
        #   "0 次成功" —— 实测 lean-02 重跑时 findings 明明 proved=True,
        #   摘要却写「调了 1 次工具(0 次成功)」。看着像失败, 排查时会被带偏。
        def _succeeded(c) -> bool:
            return bool(c.get("ok")) or c.get("proved") is True
        ok = sum(1 for c in calls if _succeeded(c))
        bits.append(f"调了 {len(calls)} 次工具({ok} 次成功): "
                    + ", ".join(c.get("tool", "?") for c in calls[-6:]))
    else:
        bits.append("一次工具都没调")
    if r.get("error_type"):
        bits.append(f"状态: 失败({r['error_type']})")
    else:
        bits.append("状态: 给出结论")
    if r.get("has_artifact"):
        bits.append("交出了制品")
    if r.get("lean"):
        # ★ 键名是 lean_proved / lean_sorry, 不是 proved / sorry ★
        #   我第一版在这里写 `.get("proved")`, 于是 **Lean 的判决在摘要里
        #   永远是 None** —— 而 Explorer 就是靠这份摘要决定要不要换方法的。
        #   它看不见"编译器其实判过了", 于是会带着一个假的无结论去换路。
        #   和 ① 同一个形状: 键名不对, 症状是"某个值莫名其妙是 None"。
        lg = r["lean"]
        bits.append(f"lean_check: proved={lg.get('lean_proved')} "
                    f"sorry={lg.get('lean_sorry')} reason={lg.get('lean_reason')}")
    s = " | ".join(bits)
    last = r.get("last_tool_result")
    if last:
        s += "\n--- 最后一次工具返回(尾部) ---\n" + _clip_tool_result(last)[-limit:]
    err = (r.get("stderr") or "")[-400:]
    if err:
        s += "\n--- 报错 ---\n" + err
    return s
