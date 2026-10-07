"""候选池跨来源去重的反向验证。

**方法: 基线对照。** 每个变异先跑一遍目标测试确认**基线是绿的**, 再改坏源码重跑,
必须变红。基线不绿就说明这条护栏不存在 —— 直接判 FAIL, 而不是"变异后仍红所以
护栏生效了"。

⚠ 另两个必查项(踩过的坑):
  - `str.replace(..., 1)` 只改第一处, 模式在源码里重复出现时那就是半个变异,
    测出来的红是假的。每个变异都先自检 `src.count(anchor) == 1`。
  - 环境变量要**继承** os.environ 再改, 不能整个替换(Windows 上会丢掉
    SystemRoot, 子进程连 pytest 都起不来, 表现为"基线全红")。

用法: venv\\Scripts\\python.exe -X utf8 scripts\\_reverse_verify_pool_dedup.py
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILS = ROOT / "agent" / "tools" / "surveys" / "_pool" / "_utils.py"
ARXIV = ROOT / "agent" / "tools" / "surveys" / "_pool" / "_arxiv.py"
OA = ROOT / "agent" / "tools" / "surveys" / "_pool" / "_openalex.py"
SEARCH = ROOT / "agent" / "tools" / "surveys" / "_actions" / "_search_more.py"

T = "tests/test_pool_dedup.py::"

MUTATIONS = [
    ("D1", "身份键只看 paper_id, 忽略 doi 字段(退回 10-07 的 bug 状态)", UTILS,
     '    doi = normalize_doi(entry.get("doi") or "")',
     '    doi = ""  # [变异] 不看 doi 字段',
     T + "TestIdentityKey::test_doi_beats_arxiv_id"),

    ("D2", "优先级反过来: arXiv 版赢(违反 user「以 OpenAlex 为准」)", UTILS,
     "    is_doi_form = 1 if looks_like_doi(entry.get(\"paper_id\") or \"\") else 0",
     "    is_doi_form = 0 if looks_like_doi(entry.get(\"paper_id\") or \"\") else 1",
     T + "TestPriority::test_openalex_doi_form_wins_over_arxiv"),

    ("D3", "不做折叠, 原样返回(去重整个失效)", UTILS,
     "        cur = best.get(k)",
     "        cur = object() if False else best.get(k + '\\x00__never__')",
     T + "TestDedupAcrossSources::test_identical_paper_id_many_times_collapses_to_one"),

    ("D4", "合并只在 else 分支做(cur 赢时落败方信息蒸发)", UTILS,
     """        else:
            winner, loser = e, cur
        # ⚠ 合并和 alias **两个分支都要做**。第一版写在 else 里, 于是"旧的那条
        # 恰好更好"时(预印本摘要更全、OpenAlex 侧摘要空)落败方信息直接蒸发。
        merged = _fill_gaps(winner, loser)""",
     """        else:
            winner, loser = e, cur
        # ⚠ 合并和 alias **两个分支都要做**。第一版写在 else 里, 于是"旧的那条
        # 恰好更好"时(预印本摘要更全、OpenAlex 侧摘要空)落败方信息直接蒸发。
        merged = winner""",
     T + "TestDedupAcrossSources::test_loser_metadata_fills_gaps"),

    ("D5", "arXiv 召回不读 <arxiv:doi>(跨来源折叠失去依据)", ARXIV,
     '"doi": (getattr(r, "doi", "") or "").strip(),',
     '"doi": "",  # [变异] 不读 arxiv:doi',
     T + "TestRecallFields::test_arxiv_module_reads_doi",
     True),   # 两个入口都有这行, 必须全换 —— 只换第一处剩下一处, 测试照样绿

    ("D6", "OpenAlex 侧不写 doi 字段", OA,
     '"doi": normalize_doi(work.get("doi") or ""),',
     '"doi": "",  # [变异] 不写 doi',
     T + "TestRecallFields::test_openalex_module_sets_doi"),

    ("D7", "去重挪到 summarize 之后(重复论文照样各烧一次 LLM)", SEARCH,
     """    raw_papers, dd = dedup_across_sources(raw_papers)""",
     """    raw_papers, dd = raw_papers, {"n_in": 0, "n_out": 0, "n_arxiv_to_doi": 0,
                                  "n_dedup_same": 0}""",
     T + "TestRecallFields::test_search_more_dedupes_before_summarize"),

    ("D8", "去重不跟已有池子比(gap fill 连打 5 次各塞一遍)", SEARCH,
     '    for p in (state.get("pool") or []):\n        if p.get("paper_id"):',
     '    for p in []:  # [变异] 不看已有池子\n        if p.get("paper_id"):',
     T + "TestRecallFields::test_search_more_checks_existing_pool"),

    ("D9", "新条目不带 doi(下次追加又认不出跨来源重复)", SEARCH,
     '"doi": p.get("doi") or (paper_id if looks_like_doi(paper_id) else ""),',
     '"doi": "",  # [变异]',
     T + "TestRecallFields::test_search_more_carries_doi_into_pool"),
]


def run(nodeid: str) -> bool:
    import os
    import tempfile
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    # ⚠ 必须隔离字节码缓存。脚本是「写变异 -> 立刻起 pytest 子进程」, 两次
    # 写盘的 mtime 常常落在**同一秒**里, 而 CPython 判断 .pyc 是否失效用的
    # 就是秒级 mtime + 源文件大小 —— 于是子进程加载的是**变异前的**字节码,
    # 护栏当然"没咬到"。实测踩过: D4 变异明明把 abstract 从 ARXIV-ABSTRACT
    # 改成空串(进程内 exec 验证过), 脚本却报"仍绿"。
    # PYTHONPYCACHEPREFIX 指到一个用完就删的目录, 每次都重新编译。
    # 注意: 这个坑只会造成**假 FAIL**(变异没生效 -> 测试仍绿), 不会造成假 PASS。
    cache = tempfile.mkdtemp(prefix="rvc_pycache_")
    env["PYTHONPYCACHEPREFIX"] = cache
    try:
        p = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "pytest", nodeid, "-q", "--no-header",
             "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=env, timeout=600)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    return p.returncode == 0


def main() -> None:
    results: list[tuple[str, str, str]] = []
    backup: dict[pathlib.Path, str] = {}
    try:
        for mut in MUTATIONS:
            mid, desc, path, anchor, repl, target = mut[:6]
            # 可选第 7 项: True = 全部匹配都替换。模式在源码里出现多次时
            # (D5 的 arxiv:doi 读取两个入口各有一行) 只改第一处就是半个变异,
            # 测出来的红是假的。
            replace_all = bool(mut[6]) if len(mut) > 6 else False
            src = backup.get(path) or path.read_text(encoding="utf-8")
            backup[path] = src

            n = src.count(anchor)
            want = None if replace_all else 1
            if want is not None and n != want:
                results.append((mid, desc, f"SKIP 锚点出现 {n} 次(应为 1) —— 变异没生效, 不算数"))
                continue
            if replace_all and n < 1:
                results.append((mid, desc, "SKIP 锚点 0 次 —— 变异没生效, 不算数"))
                continue
            if not run(target):
                results.append((mid, desc, "FAIL 基线不绿 —— 这条护栏本来就不存在, 不能证明"))
                continue
            path.write_text(src.replace(anchor, repl), encoding="utf-8")
            try:
                # 变异必须**逐字核对**, 而且判据是「旧锚点还在不在」, 不是
                # 「新代码在不在」—— 后者第一版写反了, 结果 D4 判成绿:
                # 锚点只覆盖到 `else:` 那两行, 后面的
                # `merged = _fill_gaps(...)` 原地不动, 把新写的
                # `merged = winner` 又覆盖回去, 行为压根没变。
                # 正确判据: 替换之后 `anchor` 这个片段应当**整个消失**。
                wrote = path.read_text(encoding="utf-8")
                if anchor.strip() and anchor.strip() in wrote:
                    print(f"    [!] {mid}: 旧锚点仍然存在于变异后的文件里 —— "
                          f"变异被原代码盖回去了, 这条护栏不算数")
                still_green = run(target)
            finally:
                path.write_text(src, encoding="utf-8")
            results.append((mid, desc,
                            "FAIL 改坏后测试仍绿 —— 护栏没咬到" if still_green
                            else "PASS 改坏后变红"))
    finally:
        for p, s in backup.items():
            p.write_text(s, encoding="utf-8")

    print("=" * 78)
    print("反向验证: 候选池跨来源去重")
    print("=" * 78)
    n_pass = 0
    for mid, desc, verdict in results:
        print(f"  [{verdict.split()[0]}] {mid}: {desc}")
        if verdict.startswith("PASS"):
            n_pass += 1
    print("-" * 78)
    print(f"{n_pass}/{len(results)} 护栏被证实有效")
    if n_pass != len(results):
        raise SystemExit(1)
    print("全部源码已还原")


if __name__ == "__main__":
    main()