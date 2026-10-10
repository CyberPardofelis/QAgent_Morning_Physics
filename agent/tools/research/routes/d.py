"""D 路线 —— Explorer 分派成 N 条 lane, N 个 Worker **并发**跑。

## D 与 B 的关系

D 是**独立的第四条路线**, 不是 B 的改造:

    B:  Explorer -> 单个 Worker(串行)          -> Verifier(当前关着)
    D:  Explorer -> 分派 N 条 lane -> N×Worker(并发) -> Explorer 汇总
                                  ^
        且 solve 接受 hints —— 调用方可以指定工具/方法/思路

**B 的代码一行都不改**。D 复用 explorer/worker/orchestrator 的现有能力,
新增的全部逻辑都住在这个文件里。这不是为了"干净", 而是因为 B 已经有
40 题(part1) 的实测数据 —— 复用它做对照才有意义。

## ★ 为什么 worker 之间**不做直接沟通** ★

Worker 之间直接对话会引入**顺序依赖**: A 要等 B 回答, 而并行的全部
价值就是不等待。更糟的是它毁掉可复现性 —— 同一道题跑两次, lane 之间
对话顺序不同, 结果就不同。

但"完全无沟通"有个真实问题: **重复分派**。Explorer 可能同时派两个
worker 去干同一件事。

⇒ 折中: **共享只读板**。
  · Explorer 分派时把各 lane 的**意图**写上板 —— 这是去重机制;
  · Worker 开工前能读到"别人在干什么", 但**读不到别人的结果**;
  · 结果汇总仍然只由 Explorer(有全局视野的那个角色)在下一轮做。

这样 lane 之间的"沟通"是**单向的、经协调员间接的**, 既不等待也不污染。

## 并发必须可审计

写了线程池但实际串行, **数据上完全看不出来** —— 只是慢一点。
所以 `run_lanes` 返回的同时并发计数, 留给 `check_d_route.py` 验。
"""
from __future__ import annotations

import os
import re
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent.tools.research.roles import explorer as EX  # noqa: E402
from agent.tools.research.roles import worker as WK     # noqa: E402

__all__ = ["Lane", "make_lanes", "board_intents", "lane_prompt",
           "explorer_prompt", "run_lanes", "lane_trace_fields",
           "solve_d", "DEFAULT_LANES"]

#: 一轮最多开几条 lane。开太多会让每条的预算薄到交不出制品。
DEFAULT_LANES = 3


class Lane(dict):
    """一条 lane = 一个 worker 的任务书。"""

    @property
    def idx(self) -> int:
        return int(self.get("idx", 0))

    @property
    def intent(self) -> str:
        return str(self.get("intent") or "")

    @property
    def tools(self) -> list[str]:
        t = self.get("tools")
        return [str(x) for x in t] if isinstance(t, list) else []

    @property
    def label(self) -> str:
        return f"lane{self.idx}"


def make_lanes(n: int = DEFAULT_LANES,
               intents: list[str] | None = None) -> list[Lane]:
    """把 Explorer 的分派意图切成 n 条 lane。

    ★ 为什么允许 intents 为空 ★
      Explorer 有时给不出好的分派(返回 None 或一条意图)。那时我们
      **不猜**它想干什么 —— 退化成"一条 lane, 什么都不指定", 也就是
      退回 B 的行为。这比编造意图诚实。
    """
    n = max(1, min(int(n or 1), 8))
    out: list[Lane] = []
    for i in range(n):
        it = ""
        if intents and i < len(intents):
            it = str(intents[i] or "").strip()
        out.append(Lane({"idx": i, "intent": it}))
    return out


def board_intents(lanes: list[Lane], *, for_lane: int | None = None) -> str:
    """共享只读板: **别的** lane 的意图。

    ★ 只有意图, 没有结果 ★ 这是去重机制而不是协作机制 ——
      让 lane 看得见"别人在干什么", 但看不见"别人做到了什么"。
      看得见结果会引入等待和污染(见模块开头)。

    ★ 为什么排除自己那一条 ★
      第一版把三条全列上, 于是每条 lane 的 prompt 里都出现了**自己**
      的意图文本 —— 它本来就在【你这 lane 的任务】里, 重复一遍不算错,
      但会让"这条 lane 在干什么"在文本上无法区分来源, 而我们正要靠
      这个做审计。⇒ `for_lane` 存在时只列别人的。
    """
    if len(lanes) <= 1:
        return ""
    rows = [f"  {ln.label}: {ln.intent or '(未指定意图)'}"
            for ln in lanes if for_lane is None or ln.idx != for_lane]
    if not rows:
        return ""
    return ("\n【并行分工】(这些是**别的 lane 各自要做什么**, "
            "不是已经拿到的结果 —— 你看不到也不需要看别人的结果)\n"
            + "\n".join(rows))


def explorer_prompt(problem: str, *, hints: str | None = None,
                    history: list[dict] | None = None,
                    tools: list[dict] | None = None,
                    turn: int = 0) -> str:
    """Explorer 的 prompt = B 的 prompt + 两个 D 特有的区块。

    ★ 复用 `EX.plan_prompt` 而不是重写 ★
      B 的 prompt 里那些踩坑攒下来的规矩(工具 id 必须逐字来自目录、
      没有硬路由表、probe/execute 怎么选)全部沿用。
      重写一遍就等于把这些经验扔了, 而且两份会漂移。

    ★ hints 是**独立区块**, 不拼进题面 ★
      拼进题面的话 Explorer 分不清哪句是题、哪句是建议;
      独立区块让"这是别人给的方法建议"成为一个**有身份的信息**。
    """
    base = EX.plan_prompt(problem, history, tools)
    blocks: list[str] = []

    if hints:
        blocks.append(
            "【调用方提示】(★ 这是**别人给的方法建议**, 不是题面 ★\n"
            f"{hints}\n"
            "你可以采纳、可以补充、也可以判断它不合适并说明理由 —— "
            "但**先看它**, 它比你的默认倾向更贴近出题人的意图。")

    blocks.append(
        f"【并行分工】(D 路线的特有能力)\n"
        f"这一轮你可以决定**开几条 lane 并行查**(建议 {DEFAULT_LANES} 条)。\n"
        "每条 lane 是一个独立 worker, 它们**同时**开始, **互相看不见对方的结果**。\n"
        '在计划里加一个 `lanes` 字段:\n'
        '  "lanes": {"n": 3, "intents": ["我给每条 lane 的一句话分工", ...]}\n'
        "★ 分工必须**互不重叠** —— 它们没有机会互相通气, 重了就白跑 ★\n"
        "★ `ints` 里的每一条都要说清这条路**在查什么**, 而不是工具名 ★")

    out = base.replace("【你现在的任务】", "\n\n".join(blocks) + "\n\n【你现在的任务】")
    if turn:
        out = out.replace("【你现在的任务】",
                          f"【你现在的任务】(第 {turn + 1} 轮)")
    return out


def lane_prompt(problem: str, *, lane: Lane, board: str,
                catalog: str, probe: bool, probe_spec: str | None = None,
                hints: str | None = None) -> str:
    """一条 lane 的 worker prompt。

    ★ 唯一区别于 B 的地方: 多了 (a) 自己的意图 (b) 只读板。
      Worker 本身怎么调工具, 完全走 `WK._tool_task_prompt` —— 不重写。

    ★ 为什么把 lane 信息拼进 `problem` 而不改 Worker ★
      `_tool_task_prompt(problem, plan, ...)` 会把 problem 原样嵌进它自己的
      模板(header 里那句"【问题】\\n{problem}")。所以只要把 lane 的上下文
      拼在 problem 前面传进去, Worker 那侧**一个字节都不用改** ——
      D 于是不可能污染 B, 这正是"独立新路线"要的效果。
    """
    ctx: list[str] = [problem]
    if hints:
        ctx.append("【调用方提示】(别人给的方法建议, 可以照做也可以不采)\n" + hints)
    if lane.intent:
        ctx.append(f"【你这 lane 的任务】(第 {lane.idx + 1} 条)\n{lane.intent}")
    if board:
        ctx.append(board)
    if probe and probe_spec:
        ctx.append(f"【这次试验要看的】\n{probe_spec}")
    return WK._tool_task_prompt("\n\n".join(ctx), {"method": lane.intent},
                                catalog, probe=probe,
                                probe_spec=probe_spec or "")


def run_lanes(runners: list[Callable[..., dict]],
              *, max_workers: int | None = None) -> list[dict]:
    """并发跑 N 条 lane。

    ★ 返回值里带 `_concurrent` —— 这是**可审计的并发证据** ★
      线程池写错了(比如误用了串行 map)数据上完全看不出来, 只是慢。
      把"同时最多有几条在跑"记下来, 测试才能真的验到它。
    """
    if not runners:
        return []

    state = {"cur": 0, "max": 0}
    import threading
    lock = threading.Lock()

    def wrap(fn):
        def _inner(**kw):
            with lock:
                state["cur"] += 1
                state["max"] = max(state["max"], state["cur"])
            try:
                return fn(**kw)
            finally:
                with lock:
                    state["cur"] -= 1
        return _inner

    wrapped = [wrap(f) for f in runners]
    # ★ max_workers 默认 = lane 数, 不能更小 ★
    #   设成 1 就退化成串行, 而那正是我们想防的失败模式。
    mw = max_workers or len(wrapped)
    pool = ThreadPoolExecutor(max_workers=mw)
    futs = {pool.submit(f): i for i, f in enumerate(wrapped)}
    out: list[dict] = []
    try:
        # ★★ 早到的直接赢, 不等全部 ★★
        #   第一版是 `for fu in futs: fu.result()` —— 那等价于"全部到齐再看",
        #   于是 `with ThreadPoolExecutor` 退出时 shutdown(wait=True)
        #   **把三条 lane 全等完**。后果实测很明显: D 平均 233s/题,
        #   比 B 的串行还慢 —— 因为 lane0 早就交出答案, 却在陪 lane1/lane2
        #   烧完各自的 8 步预算。
        #
        #   ⇒ 并行的全部价值就是"先到先赢"。这里改成: 任何一条 lane
        #     **交出制品**就立即返回, 剩下的设成守护线程、不 join。
        while futs:
            done, _ = wait(futs, return_when=FIRST_COMPLETED)
            for fu in done:
                i = futs.pop(fu)
                try:
                    r = fu.result()
                except Exception as e:  # noqa: BLE001
                    # ★ 一条 lane 崩了**不能带崩整轮** ★
                    #   并行的意义就是"三条路总有一条通" ——
                    #   若第一条异常就把整轮扔掉, 并行反而比串行更脆。
                    r = {"has_artifact": False,
                         "error": f"{type(e).__name__}: {e}"}
                # ★ lane 编号**必须显式写** ★
                #   WorkerResult 是 dict 子类, 它自己**没有** lane 键,
                #   而 `i` 就是 lane 号。
                #   (第一版漏了这行, 于是 note 里出现 `laneNone` ——
                #    不影响判决, 但 trace 与日志都丢了"是哪一路"的信息。)
                r["lane"] = i
                r["_concurrent"] = state["max"]
                out.append(r)
            # ★★ 不早退 ★★
            #   我一度改成"任何 lane 交出制品就立刻返回"想省墙钟时间 —— 那是错的:
            #   `has_artifact` 只说明 worker **自己认为**做完了, 不代表答案对。
            #   实测 res-08 就是 C 路线交出了答案但判 WRONG。
            #   早退等于"lane0 交出一个错答案, lane1 正在算对的, 我不等它",
            #   拿正确率去赌速度。
            #   ⇒ 等全部跑完, 再用 `_agreement` 交叉检查。
            #     并行的真正红利不是"同时跑", 是**天然产生冗余证据**。
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    for r in out:
        r.setdefault("_concurrent", state["max"])
    return out


def _nums_in(text: str) -> set[float]:
    out: set[float] = set()
    for m in re.finditer(r"-?\d+(?:\.\d+)?", text or ""):
        try:
            out.add(float(m.group(0)))
        except Exception:  # noqa: BLE001
            pass
    return out


def _agreement(results: list[dict], tol: float = 1e-6) -> tuple[int, int, str]:
    """几条 lane 的结论**数值上是否一致** —— 并行白送的交叉检查。

    ★ 为什么需要它 ★
      `has_artifact=True` 只说明 worker **自己认为**做完了, **不代表对**。
      实测 res-08 就是交出了答案但判 WRONG。
      没有 Verifier 时(它的检查程序 0/2 验过, 不可用), 我们没有任何
      信号说"这条答案是不是真的"。

    ★ 并行的红利就在这里 ★
      三条 lane **独立**解同一道题, 却**互相看不见** ——
      所以它们一致时的一致, 是真的交叉验证, 不是抄的。
      这是零成本的冗余证据, 不加白不加。

    ⇒ 一致且 >=2 条交出 -> 高置信; 不一致 -> 标 disputed,
      让分数**可解释**(不是"某条 lane 说的", 而是"几条 lane 都这么说")。

    返回 (一致的 lane 数, 交出制品的 lane 数, 文字描述)。
    """
    good = [r for r in results
            if r.get("has_artifact") and isinstance(r.get("artifact"), dict)]
    n_good = len(good)
    if n_good < 2:
        return (0, n_good, f"只有 {n_good} 条 lane 交出制品, 无从交叉验证")
    sets = [_nums_in(str(r["artifact"].get("answer") or ""))
            for r in good]
    # 以**数字最多**的那条为基准(信息量最大), 看别的 lane 有多少比例命中
    base_i = max(range(len(sets)), key=lambda i: len(sets[i]))
    base = sets[base_i]
    if not base:
        return (0, n_good, "交出的答案里没有可比对的数值")
    agree = 1
    for i, s in enumerate(sets):
        if i == base_i:
            continue
        if not s:
            continue
        hit = sum(1 for v in s if any(abs(v - b) <= tol * max(1.0, abs(b))
                                      for b in base))
        if hit / len(s) >= 0.8:
            agree += 1
    if agree >= 2:
        return (agree, n_good,
                f"{agree}/{n_good} 条 lane 的数值结论一致 —— 交叉验证通过")
    return (1, n_good,
            f"{n_good} 条 lane 交出的数值结论**互不相符** —— 答案存疑")


#: 答案短于这个长度且全是占位内容 -> 不算"真交出了东西"
PLACEHOLDER_MAX = 40
_PLACEHOLDER_PAT = re.compile(
    r"^\s*(?:\.{2,}|…+|-+)?\s*(?:你的?答案|答案|your\s+answer|answer|"
    r"结果|result|待定|todo|n/?a|null|none)\s*(?:\.{2,}|…+)?\s*$",
    re.IGNORECASE)


def _is_placeholder(result: dict) -> bool:
    """这条 lane 交出来的是不是"占位符"而不是内容?

    ★ 要害 ★ `has_artifact=True` 只说明 worker 自己认为做完了。
      实测 2026-10-11 oeis-03: 三条 lane 都 has_artifact, 但其中一条的
      制品是字面的 `"..."` —— 我按序号取第一条, **恰好选中了它**。

    ⇒ 这里只判"有没有内容", 不判内容对不对(那是 `_agreement` 的事)。
    """
    art = result.get("artifact") or {}
    if not isinstance(art, dict):
        return True
    txt = str(art.get("answer") or "").strip()
    if not txt:
        return True
    if len(txt) <= PLACEHOLDER_MAX and _PLACEHOLDER_PAT.match(txt):
        return True
    return bool(len(txt) <= PLACEHOLDER_MAX
                and re.fullmatch(r"[.。…\s\-]+", txt))


def lane_trace_fields() -> tuple[str, ...]:
    """lane 的 worker trace 条目必须带的字段。

    ★ 与 B 的 worker trace 字段保持一致, 只多一个 `lane` ★
      多出来的那个是**必须的**: 并发回来顺序不定, 没有 lane 标记
      就分不清"这条是第几路的", 而 explorer 的历史渲染靠顺序理解
      "先试了什么后试了什么"。
    """
    return ("turn", "role", "action", "method", "tools", "rationale",
            "outcome", "error_type", "has_artifact", "tool_calls",
            "findings", "switches", "lane")


# ====================================================================== 编排主循环
def _lanes_from_plan(pl: dict) -> list[Lane]:
    """从 Explorer 的计划里读出分派。

    ★ 读不到就退回一条 lane, 而不是猜一个 ★
      Explorer 给不出好分派时, 编造 N 条意图比老老实实开一条更糟 ——
      那会让 trace 里出现**根本没打算做**的事, 事后无法审计。
    """
    spec = pl.get("lanes")
    if not isinstance(spec, dict):
        return make_lanes(1)
    try:
        n = int(spec.get("n") or 1)
    except (TypeError, ValueError):
        n = 1
    intents = spec.get("intents")
    if not isinstance(intents, list):
        intents = []
    # ★ explorer 决定开几条, 但**上限 3**, 且**有 hints 时强制 1** ★
    #   有 hints = 题面点名了工具 = 没有选择空间, 多开 lane 纯浪费,
    #   还让交叉验证失效(都做同一件事,"一致"什么也证明不了)。
    n = max(1, min(n, DEFAULT_LANES))
    return make_lanes(n, intents)


def solve_d(problem: str, *, hints: str | None = None,
            budget=None, tools: list[dict] | None = None,
            scratch_dir: str | None = None, verbose: bool = False,
            verify: bool = False, max_lanes: int = DEFAULT_LANES,
            **llm_kw) -> dict:
    """D 路线主循环: Explorer 分派 -> N 条 lane 并发 -> Explorer 汇总。

    ## `hints` 是第一版方案里调用方唯一能插手的地方

    它对应现实里"这件事要求用某个方法做"的场景:
    **提示是一个有身份的信息**("这是别人给的方法建议"), 不是题面的一部分。

    ## ★ 为什么 D 默认 `verify=False` ★
      B 和 C 现在也都关着 Verifier —— 它的判据是"检查程序有没有被沙箱
      **实际执行**过", 而实测两道题的检查器全被拒、0/2 验过。
      在那之前加上它, D 的结果会全是 INCONCLUSIVE, 且**分不清**是
      "扇出没用"还是"verifier 又拒了"。三条路线保持同一口径才可比。

    ## 返回

        {verdict, artifact, trace, turns, seconds, lanes_seen, concurrent}
    """
    import shutil
    import tempfile
    import time

    from agent.tools.research.roles._util import Budget, ask as _ask
    from agent.tools.research.roles import explorer as _EX
    from agent.tools.research.roles import worker as _WK
    from agent.tools.research.roles.catalog import render_catalog

    budget = budget or Budget(max_turns=3)
    own_dir = scratch_dir is None
    scratch = scratch_dir or tempfile.mkdtemp(prefix="d_route_")
    started = time.time()
    trace: list[dict] = []
    cat = render_catalog(tools)
    used_fps: dict[str, int] = {}
    lanes_seen = 0
    peak_conc = 0

    try:
        while not budget.exhausted():
            budget.spend()
            turn = budget.turns

            # ---- 1. Explorer 出计划(含分派) ----
            prompt = explorer_prompt(problem, hints=hints,
                                     history=list(trace), tools=tools,
                                     turn=turn)
            raw = _ask(prompt, temperature=0.6, **llm_kw)
            obj = _EX.extract_json(raw)
            pl = _EX.Plan(obj) if isinstance(obj, dict) else None
            if not isinstance(pl, dict):
                trace.append({
                    "turn": turn, "role": "explorer", "action": None,
                    "outcome": "Explorer 的输出里没解析出计划 JSON",
                    "lesson": None, "lesson_given": False})
                continue

            # ★★ lane 数**只由 explorer 决定**, 我不做任何强制 ★★
            #   上限 max_lanes(默认 3), 读不到就 1 条。
            #
            #   ★ 曾经的错: 我把 hints 变成"强制 1 lane" ★
            #     那等于**剥夺 explorer 的第二次机会** —— lane 0 出问题时,
            #     explorer 连"再加一条路"都做不到。而 explorer 本来就是
            #     "看上一轮结果决定换不换路"这个角色, 强制恰好删掉了它。
            #     简单题一条 lane 够用, **复杂题不够** —— 而我按简单题的
            #     结论去限制复杂题, 是拿测试集的形状去卡设计。
            #
            #   ⇒ hints 里只**提一句**"这题大概一条 lane 就够", 让 explorer
            #     自己判断; 它若看到 lane 0 走不通, 下一轮自己会开到 2/3 条。
            lanes = _lanes_from_plan(pl)[:max(1, int(max_lanes or 1))]
            action = pl.action
            trace.append({
                "turn": turn, "role": "explorer", "action": action,
                "method": pl.get("method"), "tools": pl.tools,
                "rationale": pl.get("rationale"),
                "lanes": [{"lane": x.idx, "intent": x.intent} for x in lanes],
                "n_lanes": len(lanes),
                "done": pl.get("done"),
                "outcome": f"计划: {action}, 工具={pl.tools}, "
                           f"分派 {len(lanes)} 条 lane"})
            lanes_seen += len(lanes)

            if action == "execute":
                fp = pl.fingerprint
                if used_fps.get(fp, 0) >= 2:
                    trace[-1].update({
                        "outcome": "这个方法已经原样试过 2 次, 不给第三次机会",
                        "lesson": "换一条根本不同的路", "lesson_given": True})
                    continue
                used_fps[fp] = used_fps.get(fp, 0) + 1

            # ---- 2. 并发跑 N 条 lane ----
            # ★ 每条 lane 算**自己那份**只读板(不含自己那一行) ★
            #   共享同一份板会让每条 lane 都读到自己 —— 文本上无法区分
            #   "这是我自己的任务" 与 "这是我看到分工表", 而审计依赖这个区分。
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

            # ---- 3. N 份结果全进 trace(带 lane 标记) ----
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
                })

            if action == "probe":
                continue

            # ---- 4. 收制品: 任一 lane 交出就算这轮成功 ----
            good = [r for r in results
                    if r.get("has_artifact") and r.get("artifact")]

            # ★ explorer 说 done 且这轮真交出了 -> 立刻收工, 别再开下一轮 ★
            #   (2026-10-11 修正: 我把 `done` 字段**只记进 trace 却从不读它**,
            #    于是 max_turns=3 总是被跑满 —— 第一轮就够的题也在空转后面两轮,
            #    每轮又要开 lane 等它们跑完。这就是 D 慢的另一半原因。)
            #   语义: explorer 在**上一轮之后**才说 done, 那时收工才有意义;
            #   若它第一轮就 done, 说明它认为"这轮就能出答案", 出不出由结果决定。
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
                continue

            # ★★ 先**剔除占位符答案**, 再选 ★★
            #   实测 2026-10-11 oeis-03-candidates: 三条 lane 都"交出了制品",
            #   但 trace 里 lane0 的 制品预览 就是字面的 "..."(它调了 4 次
            #   oeis_lookup 撞了两次 stall, 收尾只写了半句话), 而 lane1/lane2
            #   各 1~2 次就交出了完整答案。
            #   ★ 我原来写 `best = good[0]` —— 按 lane 序号取第一条,
            #     **恰好选中了最差的那条**, 于是整题被判错。
            #
            #   `has_artifact` 只说明 worker 自己认为做完了, 不代表交出了
            #   **内容**。三选的失败形态实测有两种: "..." / "你的答案" / 空。
            #   ⇒ 选之前先按内容分类, 占位符的一律不选。
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
                                   f"('...' / '你的答案' / 空), 已剔除",
                        "lesson": None, "lesson_given": False})

            # ★ 多条 lane 都交出: 用**交叉验证**决定这条答案有多可信 ★
            #   lane 之间互不可见这一点白送了交叉验证的机会 ——
            #   几条 lane 独立算出**同一个数**, 这证据比"某条 lane 说的"强。
            agree_n, n_good, agree_why = _agreement(results)
            disputed = bool(n_good >= 2 and agree_n < 2)

            #   剔除占位符后取 lane 序号最小的那条 —— 可复现。
            best = good[0]
            art = best.get("artifact")
            if not verify:
                return {
                    "verdict": "DISPUTED" if disputed else "WORKER_ONLY",
                    "artifact": art,
                    "evidence": None,
                    "trace": list(trace), "turns": budget.turns,
                    "seconds": round(time.time() - started, 1),
                    "budget_exhausted": False,
                    "n_lanes": len(lanes), "lanes_seen": lanes_seen,
                    "concurrent": peak_conc,
                    # ★ 交叉验证结果要随答案一起交出去 ★
                    #   调用方据此就知道"这条答案有几条独立 lane 背书"。
                    "agree": agree_n, "n_good": n_good,
                    "agreement_note": agree_why,
                    "note": f"lane{best.get('lane')} 交出制品"
                            + (f"(共 {len(lanes)} 条 lane 并发)"
                               if len(lanes) > 1 else "")
                            + f"。{agree_why}",
                }

        return {
            "verdict": "INCONCLUSIVE", "artifact": None, "evidence": None,
            "trace": list(trace), "turns": budget.turns,
            "seconds": round(time.time() - started, 1),
            "budget_exhausted": True,
            "n_lanes": 0, "lanes_seen": lanes_seen, "concurrent": peak_conc,
            "note": "预算耗尽, 没能判定 —— 那是诚实结论, 不是失败",
        }
    finally:
        if own_dir:
            shutil.rmtree(scratch, ignore_errors=True)