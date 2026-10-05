"""
scripts/dedup_facts_llm.py — 同论文内 fact 的语义去重 (10-05)

用户诉求 (原话): "重复抽取, 你就找是否有相同语义的合并一下。如果没有, 只是
不同的方面的话, 那就都保留。"

为什么必须靠 LLM 而不是词面相似度
--------------------------------
先做了词面勘察 (64612 对, 258 篇论文), 结论是**词面只能预筛, 不能判同**:

  - 词元集合 Jaccard 丢词序, 3443 组候选里绝大多数是误报
    ("二维无限系统" == "无限二维系统" 该合, 但
     "适用于非均匀的一维量子系统" vs "适用于有限系统和非均匀系统" 不该合)
  - 换成字符 3-gram(保序) 降到 1693 组, >=0.65 那层抽样 14 组全判真重复
  - 但 g3<0.45 那层(占 97%)抽样 18 对, 仍有 3-4 对真重复:
    "监控型随机电路…测量诱导相变" vs "受监控随机量子线路…测量诱导相变现象"
    (g3=0.15) —— 改写幅度大到词面几乎重合不上

所以: lexical 决定"送不送 LLM", LLM 决定"合不合"。

**全量跑而不是只跑候选层**: 258 篇(51% 有词面候选覆盖 80% fact)。
剩下 124 篇共 871 条 fact 只有全量才抓得到, 而那正是词面抓不到的那类改写。
用户 10-05 拍板"全量跑"。

判同的判据 (写进 prompt, 与用户措辞对齐)
----------------------------------------
- **合并**: 同一篇论文里, 两条 fact 说的是同一个命题(同义改写/换了措辞/
  一条是另一条的子集且没有新增信息)
- **保留**: 只是同一篇论文的**不同侧面** —— 不同的方法/不同的结果/不同的条件/
  不同的 regime。宁可留重复, 不可误合并: facts.py 模块头写明"合错不可逆"。

安全措施
--------
1. 合并前**备份** facts.db
2. 每条合并写 `fact_merge` 审计行(kept/dropped 原文 + 判同理由), 可逐条还原
3. 跨模型的 JSON 解析容错 + 抢救, 复用 _organize 的错误识别
4. 支持 --dry / --limit / --papers, 支持断点续跑(结果落盘 json)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.infra.db import connect  # noqa: E402
from agent.prompts_survey import strip_paper_id_from_claim  # noqa: E402
from agent.tools.surveys._parallel import TaskResult, run_ordered  # noqa: E402

RESULT_JSON = ROOT / "output" / "dedup_llm_result.json"
# 单次调用的 fact 块上限。**两个维度都要卡, 而且条数比字符更关键。**
#
# 10-05 实测(MiniMax-M3, enable_thinking=False 但照样思考):
#     16 条 (1259 字符 prompt) ->  25s  正常
#     16 条 (1500 字符 prompt) ->  62s  正常
#     75 条 (4893 字符 prompt) -> >720s 撞穿 timeout×3
# 注意第三行: 4893 字符**没有**超 3000 的字符预算(75 条 claim 都很短, 扣掉
# 模板后正好一块), 但它照样卡死。所以决定耗时的是**条数** —— 判同要逐条两两
# 对比, 思考量随条数暴涨, 而不是随字符涨。字符上限管的是 prompt 体积, 条数
# 上限管的是思考深度, 两个都得卡。
CHUNK_CHARS = 3000
CHUNK_FACTS = 20
RESUME_HINT = "加 --resume 重跑可续"
# 判同**没做完**的标记。截断/空返回/异常都算 —— 它们的结果是"偏保守"而不是
# "确认没有重复", 两者必须能在结果文件里区分开。
INCOMPLETE_MARK = "incomplete"
# 记这两个词就代表该篇判同未完成(用于 --redo-incomplete 与人眼复核)
_INCOMPLETE_WORDS = ("截断", "API 错误", "返回空")
# ⚠ 10-05: 这个数从 8000 提到 32000, 因为 8000 会被 **<think> 吃光**。
# 实测(MiniMax-M3, 14 条 fact): 模型**已经正确判出**了同义组 [0][5][10][12],
# 但 8000 token 全花在思考里, JSON 答案一个字没输出就被截断。
# 跟 _organize 的宏观层是同一个坑(那边是 8000 -> 32000): MiniMax 即使
# enable_thinking=False 也会输出大段 <think>, 那部分照样吃 max_tokens。
# enable_thinking=False 挡不住, 只能给足预算。
DEDUP_MAX_TOKENS = 32000

DEDUP_PROMPT = """下面是从**同一篇论文**里抽出的若干条事实陈述(claim)。
请判断: 其中哪些 claim 说的是**同一个命题**(只是措辞不同 / 同义改写 /
互相重复), 把它们归为一组。

判据(务必严格):
- 合并: 两条 claim 的**核心断言完全一致**。例如
  "该算法能计算平移不变的无限二维系统的基态" 与
  "该算法能计算平移不变的二维无限系统的基态" -> 同一命题。
- 合并: 一条是另一条的**子集且没有引入新信息**。例如
  "折叠算法被评估用于寻找基态和热平衡态" 与 "折叠算法被评估其求基态的能力"
  -> 后者信息更少, 归为一组(但保留信息量更大的那条)。
- **不合并**: 同一篇论文的**不同侧面**。不同的方法 / 不同的结果 / 不同的适用
  条件 / 不同的物理 regime / 不同的实验平台, 即使用词相似也是不同命题。
  这是最重要的一条: 宁可留下重复, 也不要把不同侧面合并掉。

只输出 JSON, 不要任何解释:
{{"groups": [[0, 2], [5]], "singletons": [1, 3, 4]}}

说明: groups 里每个列表是一组同义 claim 的**下标**(只列下标 >=2 的组);
singletons 列其余下标。没有重复时 groups 为空数组。

论文: {title}
事实陈述:
{facts}
"""


def load_paper_facts(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """按论文分组取 fact。多出处的事实只取一次(它已经是合并后的结果)。"""
    rows = conn.execute("""
        SELECT f.id, f.content, f.fact_type, s.source_id AS paper
          FROM fact f JOIN fact_source s ON s.fact_id = f.id
         WHERE f.content IS NOT NULL AND TRIM(f.content) <> ''
         ORDER BY s.source_id, f.id
    """).fetchall()
    by_paper: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_paper[r["paper"]].setdefault(
            r["id"],
            {"id": r["id"], "content": strip_paper_id_from_claim(r["content"]),
             "fact_type": r["fact_type"] or ""},
        )
    out: dict[str, list[dict]] = {}
    for paper, facts in by_paper.items():
        if len(facts) > 1:
            out[paper] = list(facts.values())
    return out


def chunk_facts(facts: list[dict]) -> list[list[int]]:
    """按字符 + 条数双上限切块, 返回每块的下标列表。

    为什么条数也要卡(10-05): 实测 75 条 claim(总字符才 4893, 没超字符预算)
    照样卡死超过 720s, 而 16 条只要 25-62s。判同是"逐条两两比", 模型的
    思考量随**条数**暴涨, 不随字符涨 —— 短 claim 堆到 75 条一样会烧穿预算。
    """
    blocks: list[list[int]] = []
    cur, size = [], 0
    for i, f in enumerate(facts):
        n = len(f["content"])
        if cur and (size + n > CHUNK_CHARS or len(cur) >= CHUNK_FACTS):
            blocks.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    if cur:
        blocks.append(cur)
    return blocks


def parse_groups(raw: str, valid: set[int]) -> list[list[int]]:
    """解析 LLM 返回的分组。**只接受全部下标都在范围内**的组。

    宁可少合并也不误合并: 模型偶尔会吐 0-based / 1-based 混用, 或者把
    下标写错, 这类组直接丢(会被算作"未判定", 用户可见)。
    """
    if not raw:
        return []
    s = raw.strip()
    if "```" in s:
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("{"):
                s = p
                break
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(s[start:end + 1])
    except Exception:
        return []
    groups = []
    for g in data.get("groups") or []:
        if not isinstance(g, list) or len(g) < 2:
            continue
        idx = set()
        for x in g:
            try:
                i = int(x)
            except (TypeError, ValueError):
                idx = set()
                break
            idx.add(i)
        if len(idx) >= 2 and idx <= valid:
            groups.append(sorted(idx))
    return groups


def judge_paper(paper: str, facts: list[dict], title: str,
                provider: str, model: str, temperature: float) -> tuple[list[list[int]], str]:
    """对一篇论文的所有 fact 判同。返回 (groups, note)。"""
    from agent.tools.llm import _call_llm
    from agent.tools.surveys._organize import _is_llm_error_response

    groups: list[list[int]] = []
    notes: list[str] = []
    blocks = chunk_facts(facts) or [list(range(len(facts)))]
    for k, block in enumerate(blocks, 1):
        if len(block) < 2:
            continue
        lines = "\n".join(
            f"[{i}] ({facts[i]['fact_type']}) {facts[i]['content']}" for i in block)
        prompt = DEDUP_PROMPT.format(title=title or paper, facts=lines)
        out = _call_llm(prompt, provider=provider, model=model,
                        enable_thinking=False, temperature=temperature,
                        max_tokens=DEDUP_MAX_TOKENS, timeout=240)
        api_err = _is_llm_error_response(out)
        if api_err:
            notes.append(f"块 {k} API 错误: {api_err}")
            continue
        if not out:
            notes.append(f"块 {k} LLM 返回空")
            continue
        g = parse_groups(out, set(block))
        if not g and looks_truncated(out):
            # **关键**: 截断不是"没有重复"。10-05 实测模型已经正确判出了同义组,
            # 但 8000 token 全被 <think> 吃光, JSON 一个字没输出。
            # 早先的写法把这种情况记成"未判出重复", 等于把一次失败汇报成
            # 一个结论 —— 而且是"这里没有重复"这种看着挺安心的结论。
            notes.append(f"块 {k} 被 max_tokens={DEDUP_MAX_TOKENS} 截断, "
                         f"判同未完成(不是'没有重复')")
            continue
        groups.extend(g)
        if not g:
            notes.append(f"块 {k} 判定无重复")
    return groups, "; ".join(notes)


def looks_truncated(out: str) -> bool:
    """判断返回是否被截断 —— 用来把"没判出来"和"真的没有"分开。

    判据: 有 <think> 却没有闭合, 或者 JSON 括号没配平。
    """
    if not isinstance(out, str) or not out:
        return False
    s = out.strip()
    if "<think>" in s and "</think>" not in s:
        return True
    if "{" in s and s.count("{") > s.count("}"):
        return True
    return bool("[" in s and s.count("[") > s.count("]"))


def pick_keeper(facts: list[dict], idxs: list[int], conn: sqlite3.Connection) -> int:
    """同义组里保留哪一条: **信息量最大**的。

    排序键: 出处数多的优先(多出处 = 多篇论文独立说过, 证据更强), 其次正文长的。
    绝不取第一条 —— 那会随机丢掉更完整的那份表述。
    """
    def score(i: int) -> tuple[int, int]:
        fid = facts[i]["id"]
        n_src = conn.execute("SELECT COUNT(*) FROM fact_source WHERE fact_id=?",
                             (fid,)).fetchone()[0]
        return (n_src, len(facts[i]["content"]))
    return max(idxs, key=score)


def preflight(provider: str, model: str) -> None:
    """发第一条之前先验两件事, 否则一次跑砸掉的是**整天**的预算。

    1. 模型名有效: MiniMax 的白名单在 llm_clients.MINIMAX_VALID_MODELS。
       10-05 踩过: 写了不存在的 "MiniMax-M3.1", 258 次调用全部 400,
       而 cost_guard 是"先计费再发请求", 于是额度全被空转吃掉, 一组没判出来。
    2. 额度够不够: 本次要判的论文数 x 每篇块数(估 1.2) vs 今日剩余。
    """
    if provider == "minimax":
        from agent.llm_clients import MINIMAX_VALID_MODELS
        if model not in MINIMAX_VALID_MODELS:
            raise SystemExit(
                f"[中止] 模型名 {model!r} 不在 MiniMax 白名单里。\n"
                f"  有效: {', '.join(MINIMAX_VALID_MODELS)}\n"
                f"  (写错不会立刻暴露 —— 每次调用都 400, 而 cost_guard 先计费,\n"
                f"   整轮跑完额度就没了还一组结果都没有。)")
    from agent.cost_guard import get_cost_guard
    from config import Config
    used, ceil = get_cost_guard().current_count(), Config.get_cost_ceiling_daily()
    print(f"[preflight] 今日额度 {used}/{ceil}, 剩余 {max(0, ceil - used)}")
    if used >= ceil:
        raise SystemExit("[中止] 今日 LLM 额度已用满, 等跨天重置或调高 "
                         f"cost_guard 上限(当前 {ceil})。")
    return None


def _write_results(results: dict) -> None:
    """原子落盘: 先写临时文件再 os.replace。

    10-05 踩过: 直接 `RESULT_JSON.write_text(...)` 每篇调一次, 进程正好在
    写一半时被杀掉, 留下一个**截断的 json** —— 之后 `--resume` 读到它就
    json.load 失败, 前面所有已完成的工作一起报废。写临时文件 + 原子替换
    才能保证磁盘上永远是一个完整文件。
    """
    tmp = RESULT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, RESULT_JSON)


def _entry(groups: list, note: str) -> dict:
    """一条判同结果的落盘格式。

    `note` 一定要存进去: 10-05 第一版只存了分组, 于是"这一篇因为被截断没判完"
    这个事实只活在日志里。进程一重启, `--resume` 看到 key 存在就当"判过了",
    那 4 篇截断 + 5 篇 API 错误的会**被静默跳过**并当成"确认无重复"。
    这就是"没判出来"被当成"没有"的老问题, 换了个地方复发。
    """
    return {
        "groups": groups,
        "note": note,
        "complete": not any(w in note for w in _INCOMPLETE_WORDS),
    }


def _read_results() -> dict:
    """读旧格式(纯 {paper: [[ids]]})并补成新格式。"""
    raw = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[k] = v
        else:                       # 旧格式: 只有分组, 无法知道是否判完
            out[k] = {"groups": v, "note": "", "complete": False}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只判不合并")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 篇")
    ap.add_argument("--papers", default="", help="逗号分隔的 paper_id")
    ap.add_argument("--provider", default="minimax")
    # ⚠ 10-05: 这里曾经写 "MiniMax-M3.1", 那是**编的** —— MiniMax 的有效模型
    # 见 agent/llm_clients.MINIMAX_VALID_MODELS, 只有 ...M2 / M2.1 / M2.5 /
    # M2.7 / M3。写错的后果不轻: 每次调用都 400 unknown model, 而 cost_guard
    # 是**先计费再发请求**(llm.py 的 cost_guard_charge 在 create 之前), 于是
    # 258 篇全跑完 = 258 次全部失败却照样扣掉当日预算, 0 组结果。
    # 这个坑是我自己踩的: 凭印象写了个看起来更"新"的名字。
    ap.add_argument("--model", default="MiniMax-M3")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--resume", action="store_true", help="跳过已完整判完的论文")
    ap.add_argument("--redo-incomplete", action="store_true",
                    help="只重判没判完的(截断/API 错误/空返回), 完整判过的不动")
    ap.add_argument("--workers", type=int, default=6,
                    help="并发路数(默认 6; 1 = 串行, 排障用)")
    args = ap.parse_args()

    from agent.infra.db import DB_FILES
    db = ROOT / "data" / DB_FILES["facts"]

    done: dict = {}
    if (args.resume or args.redo_incomplete) and RESULT_JSON.exists():
        done = _read_results()
        n_done = sum(1 for v in done.values() if v.get("complete"))
        n_bad = len(done) - n_done
        print(f"断点续跑: 已判 {len(done)} 篇(完整 {n_done} / 未判完 {n_bad})")

    with connect("facts") as conn:
        titles = {r[0]: (r[1] or "") for r in conn.execute(
            "SELECT source_id, quote FROM fact_source WHERE quote <> ''")}
        by_paper = load_paper_facts(conn)
        todo = dict(by_paper)
        if args.papers:
            want = {p.strip() for p in args.papers.split(",") if p.strip()}
            todo = {k: v for k, v in todo.items() if k in want}
        if args.limit:
            todo = dict(sorted(todo.items(), key=lambda kv: len(kv[1]))[:args.limit])
        if args.resume:
            # 只跳过**完整**判完的。未判完的必须重判 —— 否则那 4 篇截断 + 5 篇
            # API 错误会被当成"判过了, 没有重复", 而它们其实是"没判出来"。
            todo = {k: v for k, v in todo.items()
                    if not (done.get(k, {}).get("complete"))}
        if args.redo_incomplete:
            todo = {k: v for k, v in todo.items()
                    if k in done and not done[k].get("complete")}
        n_facts = sum(len(v) for v in todo.values())
        print("=" * 68)
        print("LLM 语义去重" + ("(dry-run, 不合并)" if args.dry else ""))
        print("=" * 68)
        print(f"2+ fact 的论文 {len(by_paper)} 篇, 本次待判 {len(todo)} 篇 / {n_facts} 条 fact")
        print(f"模型 {args.provider}/{args.model}, 单块上限 {CHUNK_CHARS} 字符")
        preflight(args.provider, args.model)
        if args.dry:
            for p, v in sorted(todo.items(), key=lambda kv: -len(kv[1]))[:15]:
                print(f"  {len(v):>3} 条  {p}")

        results: dict = dict(done)
        t0 = time.time()
        # **从小的开始**, 不是从大的:
        #   1) 立刻能看到真实速率, 而不是卡在第一篇最大的上等半小时
        #   2) 单篇耗时随条数超线性增长, 先跑小的能尽早把额度换成结果
        #   3) 出问题时(超时/额度耗尽)已经有一大批结果落盘, 而不是零
        order = sorted(todo.items(), key=lambda kv: len(kv[1]))
        lock = threading.Lock()
        n_done = 0

        def _judge(item):
            """纯函数: 只发请求、只返回数据, **不改任何共享状态**。

            这是并发唯一安全的形态 —— 改了共享状态就要考虑加锁, 而加锁之后
            顺序又乱了。详见 _parallel 模块头的说明。
            """
            paper, facts = item
            groups, note = judge_paper(paper, facts, titles.get(paper, ""),
                                       args.provider, args.model, args.temperature)
            payload = [[facts[i]["id"] for i in g] for g in groups]
            return paper, payload, note

        def _worker(item):
            """判同 + **立刻**落盘。

            ⚠ 不能写成 `run_ordered(...)` 跑完再遍历结果: 它是全部跑完才返回,
            也就是说到那一刻为止一篇都没落盘。中途崩/被杀/断电, 前面几百次
            调用全部白花(而且每次都计了费)。10-05 就这么白跑过一轮。
            锁只保护 RESULTS_JSON 的写与计数器, 顺序无关紧要(落盘是 dict)。
            """
            nonlocal n_done
            try:
                paper, payload, note = _judge(item)
            except BaseException as e:                      # noqa: BLE001
                return TaskResult(ok=False, exc=e)
            with lock:
                n_done += 1
                n, el = n_done, time.time() - t0
                results[paper] = _entry(payload, note)
                _write_results(results)
            mark = f"合 {len(payload)} 组" if payload else (
                "无重复" if results[paper]["complete"] else "未判完")
            print(f"[{n}/{len(order)}] {len(item[1]):>3} 条 {paper} -> {mark}"
                  f" (累计 {el:.0f}s, 预计剩余 {el / n * (len(order) - n) / 60:.1f} 分)"
                  + (f"  ⚠ {note}" if note else ""), flush=True)
            return TaskResult(ok=True, value=(paper, payload, note))

        print(f"并发 {args.workers} 路(复用 surveys._parallel, 含限流退避重试)")
        n_failed = 0
        for res in run_ordered(order, _worker, max_workers=args.workers, desc="dedup"):
            # TaskResult 不是裸值, 不能直接解包(10-05 踩过)
            if not res.ok:
                n_failed += 1
                print(f"失败: {type(res.exc).__name__}: {res.exc} ({RESUME_HINT})",
                      flush=True)
        _write_results(results)
        if n_failed:
            print(f"⚠ {n_failed} 篇失败(异常), 未判完 —— {RESUME_HINT}")

        n_groups = sum(len(v.get("groups") or []) for v in results.values())
        n_merges = sum(max(0, len(g) - 1)
                       for v in results.values() for g in (v.get("groups") or []))
        n_trunc = sum(1 for v in results.values() if "截断" in (v.get("note") or ""))
        n_err = sum(1 for v in results.values() if "错误" in (v.get("note") or ""))
        n_incomplete = sum(1 for v in results.values() if not v.get("complete"))
        print(f"\n判同完成: {n_groups} 组 / 可合并 {n_merges} 条")
        print(f"覆盖 {len(results)} 篇, 其中未判完 {n_incomplete} 篇"
              f"(截断 {n_trunc} / API 错误 {n_err})")
        if n_incomplete:
            print("⚠ 未判完的块结果偏保守(宁可漏合也不错合), "
                  "补跑: --redo-incomplete")
        if args.dry:
            print("dry-run: 未合并任何数据。")
            return 0

        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location("spf", ROOT / "scripts" / "strip_placeholder_facts.py")
        spf = module_from_spec(spec)
        sys.modules["spf"] = spf
        spec.loader.exec_module(spf)

        if n_merges:
            dst = shutil.copy2(db, db.with_suffix(db.suffix + f".bak-dedup-{datetime.now():%m%d-%H%M}"))
            print(f"已备份 -> {Path(dst).name}")
        conn.executescript(spf.MERGE_AUDIT_SQL)
        conn.execute("BEGIN")
        conn.execute("PRAGMA defer_foreign_keys = ON")
        fk_before = spf.fk_snapshot(conn)
        for paper, entry in results.items():
            groups = entry.get("groups") or []
            if not groups:
                continue
            facts = by_paper.get(paper) or []
            ids = [f["id"] for f in facts]        # group 里存的就是这些 id
            for g in groups:
                idxs = [ids.index(fid) for fid in g if fid in set(ids)]
                if len(idxs) < 2:
                    # id 已经不在库里 = 上一轮已经合过了。**必须跳过**,
                    # 否则 --redo-incomplete 第二遍会把同一组合并重复应用。
                    continue
                keeper = pick_keeper(facts, idxs, conn)
                for drop in idxs:
                    if drop == keeper:
                        continue
                    spf.apply_merge(conn, facts[keeper]["id"], facts[drop]["id"],
                                    reason="semantic_merge")
        problems, notes = spf.check_integrity(conn, fk_before)
        for x in notes:
            print(f"[提示] {x}")
        if problems:
            conn.rollback()
            print("\n[自检不通过, 已回滚]")
            for x in problems:
                print(f"   - {x}")
            return 1
        conn.commit()
        print(f"完成: fact {conn.execute('SELECT COUNT(*) FROM fact').fetchone()[0]} 条, "
              f"合并审计 {conn.execute('SELECT COUNT(*) FROM fact_merge').fetchone()[0]} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
