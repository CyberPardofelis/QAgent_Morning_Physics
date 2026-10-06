"""反向验证: 相关性审查保护网每条护栏都配一个「改坏它, 测试必须变红」的变异。

重点在 M1 —— 它复现的就是原始 bug 本身: 保护网挪到 strict 分支之后
(等价于让它在 strict 路径永不触发)。行为断言里 M1 只会让 4 个测试变红,
剩下全绿, 所以额外还有一条 AST 结构断言专门钉顺序。

用法: venv\\Scripts\\python.exe scripts\\_reverse_verify_relevance_fallback.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RF = ROOT / "agent" / "tools" / "surveys" / "_pool" / "_relevance_filter.py"
AUDIT = ROOT / "agent" / "tools" / "surveys" / "_pool" / "_pool_audit.py"
TEST = "tests/test_relevance_filter_fallback.py"

# (编号, 说明, 文件, 变异前, 变异后, 期望变红的测试名片段)
MUTATIONS = [
    ("M1", "复现原 bug: 保护网在 strict 下不触发 —— 整段重新变回死代码",
     RF,
     "    if pool_dict and all(\n"
     "        p.get(\"_relevance_reason\", \"\").startswith((\"[fallback]\", \"[batch_failed]\"))\n"
     "        for p in pool_dict.values()\n"
     "    ):",
     "    if not strict and pool_dict and all(\n"
     "        p.get(\"_relevance_reason\", \"\").startswith((\"[fallback]\", \"[batch_failed]\"))\n"
     "        for p in pool_dict.values()\n"
     "    ):",
     "fallback_net_sits_before_strict_branch"),

    ("M2", "保护网条件 all 改成 any —— 只要有一篇没评就全量放行",
     RF,
     "    if pool_dict and all(",
     "    if pool_dict and any(",
     "mixed_real_and_fallback"),

    ("M3", "保护网只告警不放行 —— 消息说了 API 挂了, 池子照样交空",
     RF,
     "        print(msg, flush=True)\n"
     "        logger.warning(msg)\n"
     "        return dict(pool_dict)",
     "        print(msg, flush=True)\n"
     "        logger.warning(msg)",
     "strict_all_batch_failed_returns_full_pool"),

    ("M4", "marker 只认 [fallback] 不认 [batch_failed] —— 整批失败时保护网失效",
     RF,
     "startswith((\"[fallback]\", \"[batch_failed]\"))",
     "startswith(\"[fallback]\")",
     "strict_all_batch_failed_returns_full_pool"),

    ("M5", "调用方把 strict=True 改成 strict=False —— 生产路径整个变了",
     AUDIT,
     "relevance_threshold, strict=True)",
     "relevance_threshold, strict=False)",
     "caller_actually_uses_strict_true"),

    ("M6", "告警去掉归因关键词 —— 用户只看到「不可用」, 不知道该查 API 还是换选题",
     RF,
     "f\"不是主题不匹配。本轮跳过相关性过滤, 原样保留全部候选\"\n"
     "            f\"(topic={topic[:50]!r})。\"\n"
     "            f\"如果最终候选池为空, 排查方向应该是 API 连通性/额度, 而不是换主题词\"",
     "f\"已跳过相关性过滤, 原样保留全部候选(topic={topic[:50]!r})\"",
     "warning_points_at_api_not_at_topic"),

    ("M7", "strict 分支改成无条件返空 —— 不管 LLM 判了什么都交空池(严判被'简化'掉)",
     RF,
     "    if strict:\n"
     "        if not kept:",
     "    if strict:\n"
     "        if True:",
     "strict_all_relevant_still_filters_by_threshold"),
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

    print("=" * 78)
    ok = sum(1 for _, good, _ in results if good)
    for mid, good, msg in results:
        print(f"  {'✓' if good else '✗'} {mid:<4} {msg}")
    print("=" * 78)
    print(f"反向验证 {ok}/{len(results)} 生效")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
