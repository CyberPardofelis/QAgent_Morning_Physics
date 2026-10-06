"""反向验证: 分块范围块的每条护栏都配一个「改坏它, 测试必须变红」的变异。

重点在 M1 —— 范围块被摘掉/挪位之后, Phase B 的**返回结构完全不变**,
合并层、渲染层、UI 全都察觉不到, 所以只能靠 prompt 内容断言。

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_chunk_scope.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORG = ROOT / "agent" / "tools" / "surveys" / "_organize.py"
TEST = "tests/test_theme_chunk_scope.py"

# (编号, 说明, 文件, 变异前, 变异后, 期望变红的测试名片段)
MUTATIONS = [
    ("M1", "把 {scope_line} 占位符从 prompt 里删掉 —— 模型完全看不到自己读了几块",
     ORG,
     "子问题构成, 它们之间是什么关系\"。\n\n{scope_line}\n\n【语料分布",
     "子问题构成, 它们之间是什么关系\"。\n\n【语料分布",
     "prompt_template_has_scope_line_placeholder"),

    ("M2", "范围块挪到语料分布**之后** —— 模型先把一块的分布当成了全领域分布",
     ORG,
     "{scope_line}\n\n【语料分布 — 客观数据, 先看这个】\n"
     "下面的簇是按 fact 的 `subject`(核心对象)自动聚的, 反映了这批文献**实际**在讨论什么。\n"
     "{n_subject_histogram}\n",
     "【语料分布 — 客观数据, 先看这个】\n"
     "下面的簇是按 fact 的 `subject`(核心对象)自动聚的, 反映了这批文献**实际**在讨论什么。\n"
     "{n_subject_histogram}\n\n{scope_line}\n",
     "scope_line_precedes_subject_histogram"),

    ("M3", "块序号写死成 1 —— 8 块的 prompt 全都自称第 1 块, 等于没做",
     ORG,
     "        f\"【本次范围 — 只读第 {chunk_idx}/{n_chunks} 块】\\n\"\n"
     "        f\"你看到的 {facts_n} 条 fact 是**全库 {total_facts} 条里的第 \"\n"
     "        f\"{chunk_idx}/{n_chunks} 块**, 其余的块由各自独立的调用分别归纳。\\n\"",
     "        f\"【本次范围 — 只读第 1/{n_chunks} 块】\\n\"\n"
     "        f\"你看到的 {facts_n} 条 fact 是**全库 {total_facts} 条里的第 \"\n"
     "        f\"1/{n_chunks} 块**, 其余的块由各自独立的调用分别归纳。\\n\"",
     "every_chunk_gets_its_own_index"),

    ("M4", "只给本块条数, 不给全库总数 —— 模型不知道这是 1/N",
     ORG,
     'f"你看到的 {facts_n} 条 fact 是**全库 {total_facts} 条里的第 "',
     'f"你看到的 {facts_n} 条 fact 是**一批 fact, 属于第 "',
     "every_chunk_gets_its_own_index"),

    ("M5", "折半后仍写全库总数当本块条数 —— 复现刚被测出来的撒谎",
     ORG,
     "    if n_chunks <= 1 and facts_n >= total_facts:",
     "    if n_chunks <= 1:",
     "halved_single_chunk_never_claims_it_is_everything"),

    ("M6", "折半后谎称看到了全部 —— 模型以为漏掉的那些不存在",
     ORG,
     "            f\"全部语料共 {total_facts} 条, **你这一轮只看到了前 {facts_n} 条**\"",
     "            f\"全部语料共 {total_facts} 条, **你已经全部看过了**\"",
     "halved_single_chunk_never_claims_it_is_everything"),

    ("M7", "不告诉模型分布表只统计了这一块 —— 它会拿 1/N 的锚当全领域分布",
     ORG,
     '        f"- 下面的语料分布也**只统计了这一块** —— 它反映的是这块在谈什么,\\n"\n'
     '        f"  不是全领域的分布。\\n"',
     "",
     "scope_line_says_histogram_is_chunk_local"),

    ("M8", "分块入口不再传 chunk_idx —— 每块都退回默认第 1 块",
     ORG,
     "                chunk_idx=i, n_chunks=len(chunks), total_facts=len(facts)))",
     "                ))",
     "every_chunk_gets_its_own_index"),

    ("M9", "折半重试那次 format 漏传 scope_line —— 折半时模型看不到范围了",
     ORG,
     "            scope_line=_chunk_scope_line(len(facts), total_facts or len(facts), chunk_idx, n_chunks),\n"
     "        )\n"
     "    return facts, prompt, parsed, salvaged, last_err, n_api_err",
     "        )\n"
     "    return facts, prompt, parsed, salvaged, last_err, n_api_err",
     "every_format_call_site_passes_scope_line"),
]


def _run_test(frag: str) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q", "-k", frag,
         "-p", "no:cacheprovider", "--no-header", "-x"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "")[-400:]


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
                rc, _ = _run_test(frag)
                if rc == 0:
                    results.append(
                        (mid, False, f"改坏后测试仍然绿 —— 这条护栏没咬到: {desc}"))
                else:
                    results.append((mid, True, f"变红 ✓  {desc}"))
            finally:
                shutil.copy2(bak, path)

    print("=" * 78)
    ok = sum(1 for _, good, _ in results if good)
    for mid, good, msg in results:
        print(f"  {'✓' if good else '✗'} {mid:<4} {msg}")
    print("=" * 78)
    print(f"反向验证 {ok}/{len(results)} 生效")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
