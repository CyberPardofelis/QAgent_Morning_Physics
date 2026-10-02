"""
surveys._actions._extract_facts — 拆出自原 _actions.py (9-20 体检重构)

包含函数: action_extract_facts
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any

from .._parallel import run_ordered
from .._pool import _safe_json_loads_array
from ._helpers import _friendly_error

logger = logging.getLogger(__name__)


def action_extract_facts(args: dict[str, Any], state: dict[str, Any], ws) -> Iterator[str]:
    """
    extract_facts action (step 3 真实实现):
      - 对每篇 paper 调 minimax LLM 抽 3-5 条 fact (JSON 数组)
      - 复用 P6 build_fact_prompt + _safe_json_loads_array 容错
      - fact 写 workspace/_facts.jsonl, 加 state['facts']
      - 单篇失败不阻塞 batch
    """
    from agent.prompts_survey import build_fact_prompt
    from agent.tools.llm import _call_llm

    _PROVIDER = "minimax"
    _MODEL = "MiniMax-M3"

    # 默认抽 state['pool'] 全部; 也可指定 paper_ids 子集
    target_ids = args.get("paper_ids")
    targets = [p for p in state["pool"] if p["paper_id"] in target_ids] if target_ids else list(state["pool"])

    if not targets:
        yield "⚠️ extract_facts: 0 篇目标 (paper_ids 不在 pool 中)\n"
        yield f"👁️ 观察: extract_facts 0 篇, 累计 facts={len(state['facts'])} 条\n"
        return

    yield f"📝 extract_facts: 抽 {len(targets)} 篇, 调 {_PROVIDER}/{_MODEL}\n"

    # 9-30 实测结论: **fact 抽取不能合批**, 已从 4 篇/批回退到 1 篇/批。
    #
    # 我原本按字符预算做多篇合批(省 4 倍调用), 实测 12 篇分 3 批跑下来:
    #     26 条 fact, 但只来自 **4 篇** —— 另外 8 篇产出为零
    # 模型把多篇 prompt 当成**一个整体**处理: 只挑 1 篇抽, 其余直接无视。
    # (哪怕在 prompt 里补一句"每篇都要各自抽 4-12 条"也没用。)
    # 代价太大: 丢掉 2/3 的论文, 综述覆盖面直接塌。所以这里退回逐篇。
    #
    # 省调用的正确位置在**审查**: 语义审查/主题对齐是逐条判定,
    # 批大小只影响调用次数不影响判定质量, 那两处 8 -> 20 的提批是安全的。
    batches = [[p] for p in targets]

    # 10-02 并发: 每次调用**只抽一篇**, 每篇的 prompt / 请求 / 解析完全独立,
    # 所以并发不改动任何单次调用的输入 —— 这是并发和"合批"的本质区别。
    # (合批会改输入, 实测丢 2/3 论文; 并发不会, 见本文件顶部注释。)
    # 下面 42-49 次串行调用占了全流程等待时间的最大一块, 并发后约 1/4。
    def _extract_one(batch: list[dict]) -> tuple[list[str], list | None, Exception | None]:
        """纯函数: 发请求 + 解析。**不许碰 state / ws** —— 那是主线程的活。"""
        blocks = []
        for p in batch:
            blocks.append(build_fact_prompt(
                paper_id=p["paper_id"],
                title=p.get("title", ""),
                year=str(p.get("year", "")),
                journal=p.get("journal", ""),
                core_method=p.get("core_method", ""),
                core_contrib=p.get("core_contrib", ""),
                abstract=p.get("abstract", ""),
                outlook=p.get("outlook", ""),
            ))
        fact_prompt = "\n\n".join(blocks) if len(blocks) > 1 else blocks[0]
        batch_pids = [p["paper_id"] for p in batch]
        if len(batch) > 1:
            # 9-30 实测事故: 多篇合抽时, 模型**不会原样回抄 paper_id**(长的 DOI 串
            # 抄不准/截断/干脆不写) -> 校验全部落空 -> 51 篇抽出 0 条 fact。
            # 教训: 让模型抄长字符串是不可靠的, 要抄**批内序号**。
            idx_hint = ", ".join(f"{i+1}={pid}" for i, pid in enumerate(batch_pids))
            fact_prompt = (
                f"【本次共 {len(batch)} 篇论文, 编号如下】\n{idx_hint}\n"
                f"**每条 fact 的 paper_id 字段必须填上面表格里的编号(1 到 {len(batch)}), "
                f"不要填完整的 paper_id 字符串。**\n"
                f"**每篇论文都要各自抽 4-12 条 fact, 不是这一批总共 4-12 条。**\n"
                f"每篇论文各自独立判断信息量, 不要把一篇的细节拆到另一篇上。\n\n"
                + fact_prompt
            )
        try:
            out = _call_llm(
                fact_prompt,
                provider=_PROVIDER,
                model=_MODEL,
                enable_thinking=False,
                temperature=0.2,
                max_tokens=10000 * max(1, len(batch)),
                timeout=180 + 60 * len(batch),
            )
            # 剥 <think>
            if out and "<think>" in out:
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
            # 容错 parse JSON 数组
            m = re.search(r"\[[\s\S]*", out or "")
            if not m:
                raise ValueError(f"LLM 输出无 JSON 数组: {(out or '')[:200]}")
            facts = _safe_json_loads_array(m.group(0))
            if facts is None:
                raise ValueError(f"JSON parse 失败 (含容错): {(out or '')[:200]}")
            return batch_pids, facts, None
        except Exception as e:      # noqa: BLE001 — 单批失败不阻塞后面的批
            return batch_pids, None, e

    from agent.prompts_survey import clean_fact
    # 按**原始顺序**归并(不是完成顺序) —— fact_id 序号和 _facts.jsonl 行序都靠它
    for res in run_ordered(batches, _extract_one, desc="extract_facts"):
        if res.ok:
            batch_pids, facts, err = res.value
        else:
            # _extract_one 内部已经把异常收进返回值了, 走到这里说明是它之外的意外
            # (比如 build_fact_prompt 抛了) —— 记一条, 不静默
            batch_pids, facts, err = [], None, res.exc
        if facts is None:
            # 单批失败不阻塞后面的批, 不 yield 详情
            for paper_id in (batch_pids or ["?"]):
                state["errors"].append({
                    "step": state["step"], "action": "extract_facts",
                    "paper": paper_id,
                    "msg": _friendly_error(err or RuntimeError("抽取失败"), "extract_facts"),
                })
            continue
        # 批内序号 -> 真实 paper_id
        idx2pid = {str(i + 1): pid for i, pid in enumerate(batch_pids)}
        idx2pid.update({pid: pid for pid in batch_pids})
        for _i, raw in enumerate(facts):
            if not isinstance(raw, dict):
                continue
            # 9-29: 规范化 —— 去模型自写的 [paper_id] 标注、重编全局唯一 id、
            # 加确定性质检标记(术语是否超出 support_quote / 是否带自指措辞)。
            # 实测: 上一版"离子阱/附加能级"那组编造**出在抽取阶段**, 写作阶段
            # 只是照抄, 所以 prompt 纪律 + 确定性检查要同时上。
            # 9-30: 多篇合抽时, 出处靠 fact 自带的 paper_id 字段回来(可能是批内序号)。
            # **单篇时根本不用查表** —— 批次里只有一篇, paper_id 是已知的。
            # 踩过的坑: 单篇时模型经常**不填** paper_id 字段(老代码靠 clean_fact 的
            # setdefault 兜住), 改成"先查表再覆盖"后查不到就全丢, 49 篇抽出 0 条。
            fpid = (
                batch_pids[0] if len(batch_pids) == 1
                else idx2pid.get(str(raw.get("paper_id") or "").strip(), "")
            )
            if not fpid:
                state["errors"].append({
                    "step": state["step"], "action": "extract_facts",
                    "paper": str(raw.get("paper_id"))[:60] or "?",
                    "msg": f"多篇合抽时 paper_id 缺失或不在本批 {batch_pids[:2]}..., 已丢弃该 fact",
                })
                continue
            f = clean_fact(raw, fpid, _i)
            ws.append_fact(f)
            state["facts"].append(f)

    # 9-08 17:20 user 拍板: 只显示"抽了 N 条 fact" 高层
    yield f"👁️ 抽完 {len(targets)} 篇, 累计 {len(state['facts'])} 条 fact\n"

    # ========== 9-29: fact 落地审查 ==========
    # 编造的源头在抽取阶段(实测"离子内附加能级"原文没有, 抽取器自行补全),
    # 写作阶段只是照抄。这里对抽出的 fact 做一次语义审查, 超纲的打 _grounded=False,
    # 写作阶段会据此剔除。
    if state["facts"]:
        from agent.tools.surveys._fact_audit import audit_fact_grounding
        pool = state.get("pool") or []
        src = {
            p.get("paper_id", ""): " ".join([
                p.get("title", "") or "", p.get("abstract", "") or "",
                p.get("core_method", "") or "", p.get("core_contrib", "") or "",
            ])
            for p in pool
        }
        yield "🔎 fact 落地审查中(核对 claim 有没有超出原文)...\n"
        try:
            st = audit_fact_grounding(state["facts"], src)
            failed_note = f" / 未能判定 {st['failed']}" if st["failed"] else ""
            yield (
                f"👁️ 审查: 通过 {st['grounded']} / 超纲 {st['ungrounded']}{failed_note}"
                f" (共 {len(state['facts'])} 条)\n"
            )
            if st["ungrounded"]:
                state["errors"].append({
                    "step": state["step"], "action": "fact_grounding_audit",
                    "msg": f"{st['ungrounded']} 条 fact 的 claim 超出原文, 写作时会剔除",
                })
        except Exception as e:
            friendly = _friendly_error(e, "fact_audit")
            state["errors"].append({
                "step": state["step"], "action": "fact_grounding_audit",
                "msg": friendly,
            })
            yield f"👁️ 审查失败 ({friendly}), 沿用原 fact\n"

    # ========== 9-29: 写入跨来源 fact 库 ==========
    # 一次抽取不再是"用完即弃": 通过审查的 fact 进 facts.db, 带出处。
    # 同一命题从不同论文抽出时, 内容哈希会让它们收敛成**一条 fact + 多个出处**,
    # 引用时就能看出"几个独立工作都这么说"。
    if state.get("facts"):
        try:
            from agent.tools.facts import add_facts_from_extraction
            fst = add_facts_from_extraction(state["facts"], source_type="paper_abstract")
            yield (
                f"📚 已入 fact 库: 新增 {fst['new_facts']} 条, "
                f"并入已有 fact 的新出处 {fst['multi_source']} 个"
                + (f", 未过审跳过 {fst['ungrounded']} 条" if fst["ungrounded"] else "")
                + "\n"
            )
        except Exception as e:
            friendly = _friendly_error(e, "fact_store")
            state["errors"].append({
                "step": state["step"], "action": "fact_store",
                "msg": friendly,
            })
            yield f"⚠ fact 库写入失败 ({friendly}), 不影响本次综述\n"

    # ========== 9-09 user 拍板: pre-outline fact 主题对齐审查 (mechanism B) ==========
    # 跟 _relevance_filter_pool (mechanism A) 配套 — paper 严判只过滤 pool,
    # 但 pool 内的 fact 仍可能是从邻近子领域 paper 抽出来的"挂羊头卖狗肉" 内容
    # 这里对每条 fact 做主题对齐审查, 不删非对齐 fact, 只打标 (audit 用)
    from agent.tools.config import get_fact_review_threshold
    fact_threshold = get_fact_review_threshold()
    if fact_threshold > 0 and state["facts"]:
        from agent.tools.surveys._pool import _review_facts_topic_alignment
        yield "🔎 fact 主题对齐审查中...\n"
        try:
            scored_facts, fact_stats = _review_facts_topic_alignment(
                topic=state["topic"],
                facts=state["facts"],
                threshold=fact_threshold,
            )
            # 把审查结果写回每条 fact
            for _i, f in enumerate(state["facts"]):
                fid = f.get("fact_id")
                if not fid:
                    # 用同样的回退 key 逻辑
                    pid = f.get("paper_id", "?")
                    claim_hash = hash(str(f.get("claim", ""))[:50]) & 0xFFFF
                    fid = f"{pid}:{_i}:{claim_hash}"
                s = scored_facts.get(fid, {})
                f["_topic_alignment_score"] = s.get("score", 5)
                f["_is_topic_aligned"] = s.get("is_topic_aligned", True)
                f["_alignment_reason"] = s.get("reason", "")
            # 存 stats 给 finalize 用
            state["_fact_review_stats"] = fact_stats
            yield (
                f"👁️ fact 审查: 对齐 {fact_stats['aligned']} / 邻近 {fact_stats['misaligned']}"
                f" / fallback {fact_stats['fallback']} (共 {fact_stats['total']} 条)\n"
            )
        except Exception as e:
            friendly = _friendly_error(e, "fact_review")
            state["errors"].append({
                "step": state["step"], "action": "fact_review",
                "msg": friendly,
            })
            yield f"👁️ fact 审查失败 ({friendly}), 跳过 — 用原始 fact 列表\n"
    else:
        # fact 审查关闭 或 无 fact, 不 yield 任何东西
        pass
