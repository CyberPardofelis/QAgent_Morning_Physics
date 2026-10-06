"""反向验证: 空白回补(gaps → 再检索)的每条护栏都配一个"改坏它, 测试必须变红"的变异。

用户要求(9-28): 测试不能只是"跑绿了就算"。

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_gap_fill.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QE = ROOT / "agent" / "tools" / "surveys" / "_pool" / "_query_expand.py"
SG = ROOT / "agent" / "tools" / "surveys" / "survey_generate.py"
CFG = ROOT / "agent" / "tools" / "config.py"
PROMPTS = ROOT / "agent" / "prompts_survey.py"
TEST = "tests/test_survey_gap_fill.py"

# (编号, 说明, 文件, 变异前, 变异后, 期望失败的测试名片段)
MUTATIONS = [
    # ---------- 解析层: NUL / think / 重试 ----------
    ("G1", "NUL 字节不再洗掉 —— 10-06 实测 6/6 次调用全中招, 整轮回补静默消失",
     QE,
     '    if "\\x00" in raw:\n        raw = raw.replace("\\x00", "")',
     "    pass",
     "test_real_raw_parses_to_all_31_gaps"),

    ("G2", "裸换行/制表不再用 strict=False 放行 —— 31 条空白本来就带换行",
     QE,
     "        return json.loads(m.group(0), strict=False)",
     "        return json.loads(m.group(0))",
     "test_parse_tolerates_raw_control_chars"),

    ("G3", "think 未闭合不再单独识别 —— 贪婪 \\{.*\\} 会从 think 散文里拼出垃圾",
     QE,
     '    elif "<think>" in text:\n'
     '        # 输出全花在思考里 —— 明确说出来, 别让上层报成"JSON 找不到"\n'
     '        return {"__truncated_think__": True}',
     "    elif False:\n        pass",
     "test_parse_flags_unclosed_think"),

    ("G4", "重试次数从 2 降到 1 —— 实测同一 prompt 输出格式不稳定, 一次就放弃等于整轮消失",
     QE,
     "    for attempt in range(2):",
     "    for attempt in range(1):",
     "test_retries_once_on_unparseable"),

    ("G5", "max_tokens 从 20000 掉回 6000 —— think 吃光预算, 实测返回 23635 字符零 JSON",
     QE,
     "                max_tokens=20000, timeout=240)",
     "                max_tokens=6000, timeout=240)",
     "test_gap_fill_budget_big_enough_for_think"),

    # ---------- 打分选择: idx 对齐 / 去重 / 计数守恒 ----------
    ("G6", "改成信模型回传的 gap 文本 —— 改写版进 ranked, 原文又被当漏回补进 unfillable",
     QE,
     "        idx = row.get(\"idx\")\n        gap = None\n"
     "        if isinstance(idx, int) and 1 <= idx <= len(gaps):\n"
     "            gap = gaps[idx - 1]\n",
     "        idx = row.get(\"idx\")\n        gap = None\n"
     "        if False:\n            gap = None\n",
     "test_idx_aligns_to_original_gap_text"),

    ("G7", "同一条 gap 的重复输出保留最低分 —— 会刷掉真正该回补的那条",
     QE,
     "        if gap not in best or sc > best[gap][0]:\n            best[gap] = (sc, gap, q)",
     "        if gap not in best:\n            best[gap] = (sc, gap, q)",
     "test_dup_gap_keeps_highest_score"),

    ("G8", "模型漏回的空白不再补进 unfillable —— 空白会凭空消失(静默说谎)",
     QE,
     "    unfillable += [g for g in gaps if g not in seen_gaps(ranked)]",
     "    unfillable += []",
     "test_gaps_model_omitted_still_land_in_unfillable"),

    ("G9", "unfillable 不再去重 —— 同一条空白会跟着 theme_map 里的重复条目一起膨胀",
     QE,
     "    _seen: set[str] = set()\n"
     "    unfillable = [g for g in unfillable if not (g in _seen or _seen.add(g))]",
     "    unfillable = list(unfillable)",
     "test_duplicate_input_gaps_do_not_multiply_output"),

    ("G10", "不按分数排序 —— 打分最高的空白被丢掉, 排序是这个方案的命根子",
     QE,
     "    ranked = sorted(best.values(), key=lambda x: -x[0])",
     "    ranked = list(best.values())",
     "test_idx_aligns_to_original_gap_text"),
    ("G11", "英文检索式不筛 —— 拿中文去喂 arxiv/OpenAlex, 一篇都回不来",
     QE,
     "        if not q or len(_LATIN.findall(q)) < 2:\n            continue",
     "        if not q:\n            continue",
     "test_chinese_query_is_dropped"),

    ("G12", "max_queries 闸门失效 —— 31 条全去检索, 一轮就是几十次调用",
     QE,
     '    fillable = [{"gap": g, "query": q, "score": sc} for sc, g, q in ranked[:max_queries]]',
     '    fillable = [{"gap": g, "query": q, "score": sc} for sc, g, q in ranked]',
     "test_idx_aligns_to_original_gap_text"),

    ("G13", "判定失败静默返回空 —— 用户看不到\"本该回补却没回补\"",
     QE,
     '        return {**empty, "error": last_err}',
     "        return {**empty}",
     "test_gives_up_after_two_attempts"),

    # ---------- 管线接线: 顺序 / 复用 / 上限 / 标注 ----------
    ("G14", "回补被挪到宏观层之后 —— 宏观板块会基于一张马上就过期的地图归类",
     SG,
     "        theme_map, pool, usable, fill_log = yield from _phase_gap_fill(",
     "        _unused_gf = (_phase_gap_fill,",
     "test_gap_fill_wired_after_theme_map_before_macros"),

    ("G15", "回补整段不再受开关控制 —— QAGENT_SURVEY_GAP_FILL_ROUNDS=0 关不掉",
     SG,
     "    if get_gap_fill_rounds() > 0:",
     "    if True:",
     "test_gap_fill_switch_off"),

    ("G16", "新论文上限失效 —— 每篇都要 summarize + 抽 fact, 成本会失控",
     SG,
     "        if len(new_papers) > cap:\n"
     '            yield f"   (候选 {len(new_papers)} 篇 > 上限 {cap}, 只取前 {cap} 篇)\\n"\n'
     "            new_papers = new_papers[:cap]",
     "        if False:\n            new_papers = new_papers",
     "test_gap_fill_caps_papers"),

    ("G17", "空白标注被拿掉 —— 跑完之后清单跟没跑过一样, 用户分不清试过没试过",
     SG,
     '        tmap["gaps"] = _annotate_gaps(tmap.get("gaps") or [], filled_gaps)',
     '        tmap["gaps"] = list(tmap.get("gaps") or [])',
     "test_gap_fill_wired_after_theme_map_before_macros"),

    ("G18", "标注里改回回显模型理由 —— 实测那个理由会跟\"补不上\"自相矛盾",
     SG,
     '    return [f"（已检索回补）{g}" if g in filled else f"（判定补不上）{g}"\n'
     "            for g in gaps]",
     '    return [f"（已检索回补）{g}: {filled_reason}" if g in filled '
     'else f"（判定补不上）{g}: {filled_reason}"\n            for g in gaps]',
     "test_annotate_never_shows_a_reason"),

    ("G19", "三个闸门放开区间 —— 12 条检索式 × 200 篇是四位数次调用",
     CFG,
     '    "QAGENT_SURVEY_GAP_FILL_QUERIES", 3, 1, 12)',
     '    "QAGENT_SURVEY_GAP_FILL_QUERIES", 3, 1, 99)',
     "test_gap_fill_bounds"),

    ("G20", "回补轮数上限放开 —— 多轮换不来东西, 第二轮的空白往往还是补不上",
     CFG,
     '    "QAGENT_SURVEY_GAP_FILL_ROUNDS", 1, 0, 5)',
     '    "QAGENT_SURVEY_GAP_FILL_ROUNDS", 1, 0, 20)',
     "test_gap_fill_bounds"),

    ("G22", "回补里又调了 ws 上不存在的方法 —— 10-06 真跑前一直绿的 P0",
     SG,
     "        # **不要再调 ws.save_facts(facts)** —— 10-06 实测这行必崩:",
     "        ws.save_facts(facts)\n"
     "        # **不要再调 ws.save_facts(facts)** —— 10-06 实测这行必崩:",
     "test_gap_fill_only_calls_workspace_methods_that_exist"),

    ("G21", "prompt 不再要求英文检索式 —— 模型会直接给中文, 那一轮成本全白付",
     PROMPTS,
     "**每条都要给一条英文检索式**",
     "**每条都要给一条检索式**",
     "test_gap_fill_prompt_demands_english_query"),
]


def _run_test(frag: str) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q", "-k", frag,
         "-p", "no:cacheprovider", "--no-header", "-x"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "")[-500:]


def main() -> int:
    os.chdir(ROOT)
    results: list[tuple[str, bool, str]] = []
    for mid, desc, path, before, after, frag in MUTATIONS:
        src = path.read_text(encoding="utf-8")
        if before not in src:
            results.append((mid, False, f"变异锚点没找到, 测试无效: {desc}"))
            continue
        with tempfile.TemporaryDirectory() as td:
            bak = Path(td) / path.name
            shutil.copy2(path, bak)
            try:
                path.write_text(src.replace(before, after, 1), encoding="utf-8")
                rc, out = _run_test(frag)
                if rc == 0:
                    results.append(
                        (mid, False, f"改坏后测试仍然绿 —— 这条护栏没咬到: {desc}"))
                else:
                    results.append((mid, True, f"变红 ✓  {desc}"))
            finally:
                shutil.copy2(bak, path)

    print("=" * 74)
    ok = sum(1 for _, good, _ in results if good)
    for mid, good, msg in results:
        print(f"  {'✓' if good else '✗'} {mid:<4} {msg}")
    print("=" * 74)
    print(f"反向验证 {ok}/{len(results)} 生效")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
