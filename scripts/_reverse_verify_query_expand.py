"""反向验证: 查询扩展的每条护栏都配一个"改坏它, 对应测试必须变红"的变异。

用户要求(9-28): 测试不能只是"跑绿了就算"。

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_query_expand.py
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
BUILD = ROOT / "agent" / "tools" / "surveys" / "_pool" / "_build_pool.py"
CFG = ROOT / "agent" / "tools" / "config.py"
TEST = "tests/test_pool_query_expand.py"

# (编号, 说明, 文件, 变异前, 变异后, 期望失败的测试名片段)
MUTATIONS = [
    ("Q1", "英文检索式筛选整个去掉 —— 中文检索式会直接喂给 arxiv/OpenAlex",
     QE,
     "        if not text or len(_LATIN.findall(text)) < 2:\n            continue",
     "        if not text:\n            continue",
     "test_pick_queries_drops_chinese_only"),

    ("Q2", "拉丁词门槛从 2 放宽到 1 —— 单个词也算检索式",
     QE,
     "_LATIN = re.compile(r\"[A-Za-z]{2,}\")",
     "_LATIN = re.compile(r\"[A-Za-z]\")",
     "test_pick_queries_drops_single_latin_token"),

    ("Q3", "检索式去重失效 —— 同一条 query 反复出现, 同一批结果反复进池",
     QE,
     "        if text not in out:\n            out.append(text)",
     "        out.append(text)",
     "test_pick_queries_dedups"),

    ("Q4", "字符串数组容错删掉 —— 模型偶尔直接给字符串, 整批会被丢掉",
     QE,
     "        if isinstance(q, str):            # 容错: 模型偶尔直接给字符串数组\n"
     "            q = {\"q\": q}",
     "        if False:\n            q = {\"q\": q}",
     "test_pick_queries_accepts_bare_string_array"),

    ("Q5", "LLM 异常不再被吞掉 —— 扩展一挂整个 Phase A 跟着崩",
     QE,
     "    except Exception as e:                              # noqa: BLE001",
     "    except ZeroDivisionError as e:",
     "test_expand_fails_loudly_on_exception"),

    ("Q6", "失败时返回空 error —— 用户看不到扩展为什么没跑起来",
     QE,
     "        return {**empty, \"error\": f\"LLM 调用异常: {err}\"}",
     "        return {**empty}",
     "test_expand_fails_loudly_on_exception"),

    ("Q7", "全中文检索式被静默接受 —— 等于拿中文去搜英文 API, 一篇都回不来",
     QE,
     "    if not queries:\n        safe_print(\"[query_expand] ❌ 模型没给出可用的英文检索式, 回退用原 topic\",\n"
     "                   flush=True)\n        return {**empty, \"error\": \"没有可用的英文检索式\"}",
     "    if not queries:\n        return {**empty, \"ok\": True, \"error\": \"\"}",
     "test_expand_fails_when_only_chinese_queries"),

    ("Q8", "<think> 包裹不再剥 —— 真实输出形态里 JSON 在 think 后面",
     QE,
     "    if \"<think>\" in text and \"</think>\" in text:\n"
     "        text = text.rsplit(\"</think>\", 1)[1]",
     "    if False:\n        text = text",
     "test_expand_strips_think_and_fence"),

    ("Q9", "batch 段改成一路挂了就整段放弃 —— 扩展会拖垮另外两个源",
     BUILD,
     "            except Exception as e:                      # noqa: BLE001\n"
     "                # 单路挂了就少一路, 绝不能因为一个源失败就丢掉另外两个的结果\n"
     "                external_status[label] = f\"exception:{type(e).__name__}:{str(e)[:80]}\"",
     "            except Exception as e:                      # noqa: BLE001\n"
     "                break",
     "test_batch_pass_never_drops_other_sources"),

    ("Q10", "新增的三个 source 从 source_priority 拿掉 —— 排序退化, 扩展召回被顶出 max_pool",
     BUILD,
     '        "aps_batch_expanded_query": 7,\n'
     '        "iop_batch_expanded_query": 8,\n'
     '        "arxiv_batch_expanded_query": 9,',
     '        "aps_batch_expanded_query": 9,\n'
     '        "iop_batch_expanded_query": 9,\n'
     '        "arxiv_batch_expanded_query": 9,',
     "test_expanded_sources_have_sort_priority"),

    ("Q11", "build_status 不再带细分方向 —— 下游(主题归纳 / gaps 回环)拿不到骨架",
     BUILD,
     '            "sub_directions": qexp.get("sub_directions") or [],',
     '            "sub_directions": [],',
     "test_build_status_carries_query_expansion"),

    ("Q12", "单 query 召回也换成扩展词 —— 断掉\"扩展挂了也能照常跑\"的退路",
     BUILD,
     "            aps_papers = _openalex_aps_search(topic, aps_k)",
     "            aps_papers = _openalex_aps_search(_qexp_queries[0] if _qexp_queries else topic, aps_k)",
     "test_single_query_recall_uses_raw_topic_only"),

    ("Q13", "查询扩展开关失效 —— QAGENT_SURVEY_QUERY_EXPAND=0 关不掉, 白花调用",
     CFG,
     "    return _SURVEY_QUERY_EXPAND_N if _SURVEY_QUERY_EXPAND_ENABLED else 0",
     "    return _SURVEY_QUERY_EXPAND_N",
     "test_query_expand_switch_off_returns_zero"),

    ("Q14", "n 的合法区间放开 —— 拆 50 个方向会灌进一堆噪点, 每篇还要过 LLM 审查",
     CFG,
     '    "QAGENT_SURVEY_QUERY_EXPAND_N", 6, 1, 12)',
     '    "QAGENT_SURVEY_QUERY_EXPAND_N", 6, 1, 50)',
     "test_query_expand_n_respects_bound"),

    ("Q15", "prompt 不再要求检索式是英文 —— 模型会直接给中文检索式",
     ROOT / "agent" / "prompts_survey.py",
     "检索式是喂给 arxiv / OpenAlex 的, 它们都是英文 API, 所以 queries **必须是英文**。",
     "检索式随便写什么都行, 中文英文都行。",
     "test_expand_ok_returns_queries_and_dirs"),
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
