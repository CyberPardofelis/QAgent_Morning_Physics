"""骨架加固 + 增量并入 的反向验证。

**方法: 基线对照。** 每个变异都先跑一遍目标测试确认**基线是绿的**, 再改坏源码
重跑, 必须变红。基线不绿就说明这条护栏根本不存在 —— 直接判 FAIL 而不是
"变异后仍然是红的, 所以护栏生效了"。

⚠ 为什么必须做基线对照: pytest 只要返回非 0 就算红, 而「测试自己 TypeError 崩了」
和「断言失败」都会返回非 0。10-07 在 skeleton 那轮验证时踩过这个坑 ——
先判成 15/15 通过, 实际有两条护栏从来没生效过。

另外每个变异都先 `src.count(anchor)` 自检唯一性: `str.replace(..., 1)` 只改第一处,
模式在源码里重复出现时那就是个"半个变异", 测出来的红是假的。
(同族教训: 惰性正则跨块匹配、手抄整句易错。)

用法: venv\\Scripts\\python.exe -X utf8 scripts\\_reverse_verify_skeleton_hardening.py
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORG = ROOT / "agent" / "tools" / "surveys" / "_organize.py"
GEN = ROOT / "agent" / "tools" / "surveys" / "survey_generate.py"
LLM = ROOT / "agent" / "tools" / "llm.py"

# (编号, 描述, 文件, 锚点, 替换, 目标测试 nodeid)
MUTATIONS = [
    ("M1", "骨架 max_tokens 退回 8000(回到真跑时的截断配置)", ORG,
     """out = _call_llm(prompt, provider=provider, model=model,
                            enable_thinking=False, temperature=temperature,
                            max_tokens=THEME_MAP_MAX_TOKENS, timeout=240)""",
     """out = _call_llm(prompt, provider=provider, model=model,
                            enable_thinking=False, temperature=temperature,
                            max_tokens=8000, timeout=240)""",
     "tests/test_theme_skeleton_hardening.py::TestSkeletonMaxTokens::"
     "test_skeleton_uses_shared_max_tokens_not_8000"),

    ("M2", "骨架只试一次(删掉解析失败重试)", ORG,
     "    for attempt in (1, 2):",
     "    for attempt in (1,):",
     "tests/test_theme_skeleton_hardening.py::TestSkeletonRetry::"
     "test_retry_once_on_parse_failure"),

    ("M3", "重试时不再追加更严格的长度约束", ORG,
     "【重试 — 上一次输出超长被截断】",
     "【重试】",
     "tests/test_theme_skeleton_hardening.py::TestSkeletonRetry::"
     "test_retry_prompt_is_stricter"),

    ("M4", "骨架 task 字段的字数上限从 prompt 模板里删掉", ORG,
     '"task": "这个任务要解决什么问题(**不超过 40 字**)",',
     '"task": "这个任务要解决什么问题(1-2 句)",',
     "tests/test_theme_skeleton_hardening.py::TestSkeletonMaxTokens::"
     "test_skeleton_field_caps_present_in_prompt"),

    ("M5", "有 prev 骨架时改成重建(不再沿用)", ORG,
     """    if prev_skeleton and (prev_skeleton.get("themes") or []):
        safe_print(
            f"[skeleton] ♻️ 沿用上一轮骨架 {len(prev_skeleton['themes'])} 个任务 \"""",
     """    if False and prev_skeleton and (prev_skeleton.get("themes") or []):
        safe_print(
            f"[skeleton] ♻️ 沿用上一轮骨架 {len(prev_skeleton['themes'])} 个任务 \"""",
     "tests/test_theme_skeleton_hardening.py::TestSkeletonFallbackToPrev::"
     "test_prev_skeleton_wins_even_if_llm_would_succeed"),

    ("M6", "增量判定退回 truthy(空列表会走全量重跑)", ORG,
     """incremental = (prev_map is not None and bool(prev_map.get("themes"))
                   and induce_facts is not None)""",
     "incremental = bool(induce_facts)",
     "tests/test_theme_skeleton_hardening.py::TestIncrementalWiring::"
     "test_empty_incremental_returns_prev_unchanged"),

    ("M7", "增量合并时砍掉旧主题(回到 max_themes 上限)", ORG,
     "combined, len(prev_names) + n_new, skeleton=skeleton)",
     "combined, 1, skeleton=skeleton)",
     "tests/test_theme_skeleton_hardening.py::TestMergeIntoPrevMap::"
     "test_old_themes_all_survive"),

    ("M8", "gap fill 不再传 induce_facts(退回从零重跑)", GEN,
     "            prev_map=theme_map, induce_facts=new_usable)",
     "            prev_map=theme_map)",
     "tests/test_survey_gap_fill.py"),

    # ---- _call_llm 的 MiniMax thinking 开关(骨架截断的真正根因) ----
    ("M9", "minimax 不再显式发 thinking=disabled(退回 10-07 的 bug 状态)", LLM,
     '            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}',
     '            pass  # [变异] 不发 disabled',
     "tests/test_llm_thinking_switch.py::TestMinimaxThinkingDisabled::"
     "test_disabled_flag_is_sent"),

    ("M10", "minimax + enable_thinking=True 时也发 enabled(实测会 400)", LLM,
     """    if provider == "minimax":
        # MiniMax 接口对 frequency_penalty / presence_penalty 支持有限, 统一剥离避免 400。
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)
        if not enable_thinking:""",
     """    if provider == "minimax":
        # MiniMax 接口对 frequency_penalty / presence_penalty 支持有限, 统一剥离避免 400。
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)
        if True:""",
     "tests/test_llm_thinking_switch.py::TestMinimaxThinkingDisabled::"
     "test_never_sends_enabled_to_minimax"),

    ("M11", "不再为 minimax 剥掉 frequency/presence penalty(会 400)", LLM,
     """        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)
        if not enable_thinking:""",
     """        if not enable_thinking:""",
     "tests/test_llm_thinking_switch.py::TestMinimaxThinkingDisabled::"
     "test_no_penalties_for_minimax"),

    ("M12", "provider 判断放宽成 `if provider:`(minimax 分支吞掉所有 provider)", LLM,
     """    if provider == "minimax":
        # MiniMax 接口对 frequency_penalty / presence_penalty 支持有限, 统一剥离避免 400。""",
     """    if provider:
        # MiniMax 接口对 frequency_penalty / presence_penalty 支持有限, 统一剥离避免 400。""",
     "tests/test_llm_thinking_switch.py::TestOtherProvidersUnchanged::"
     "test_deepseek_disabled_keeps_penalties"),

    ("M13", "timeout 不再透传给 SDK(回退 9-04 修过的 bug)", LLM,
     "    kwargs[\"timeout\"] = timeout",
     "    pass  # [变异] 不透传 timeout",
     "tests/test_llm_thinking_switch.py::TestRegressionGuard::"
     "test_timeout_still_passthrough"),
]


def run(nodeid: str) -> bool:
    """True = 绿。

    ⚠ 环境要**继承** os.environ 再改 PYTHONPATH, 不能整个替换 ——
    Windows 上把 env 换成只有 PYTHONPATH/PATH 会丢掉 SystemRoot / TEMP /
    PATHEXT 等, 子进程连 pytest 都起不来, 表现为"基线全红"。踩过一次。
    """
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", nodeid, "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=600)
    if p.returncode != 0 and "ERROR" in p.stdout and "passed" not in p.stdout:
        print(f"    [debug] pytest 异常: {p.stdout[-400:]}")
    return p.returncode == 0


def main() -> None:
    results: list[tuple[str, str, str]] = []
    backup: dict[pathlib.Path, str] = {}

    try:
        for mid, desc, path, anchor, repl, target in MUTATIONS:
            src = backup.get(path) or path.read_text(encoding="utf-8")
            backup[path] = src

            n = src.count(anchor)
            if n != 1:
                results.append((mid, desc, f"SKIP 锚点出现 {n} 次(应为 1) —— 变异没生效, 不算数"))
                continue

            # ---- 基线对照: 变异前必须绿, 否则这条护栏不存在 ----
            if not run(target):
                results.append((mid, desc, "FAIL 基线不绿 —— 这条护栏本来就不存在, 不能证明"))
                continue

            path.write_text(src.replace(anchor, repl, 1), encoding="utf-8")
            try:
                still_green = run(target)
            finally:
                path.write_text(src, encoding="utf-8")

            if still_green:
                results.append((mid, desc, "FAIL 改坏后测试仍绿 —— 护栏没咬到"))
            else:
                results.append((mid, desc, "PASS 改坏后变红"))
    finally:
        for p, s in backup.items():
            p.write_text(s, encoding="utf-8")

    print("=" * 78)
    print("反向验证: 骨架加固 + 增量并入")
    print("=" * 78)
    n_pass = 0
    for mid, desc, verdict in results:
        print(f"  [{verdict.split()[0]}] {mid}: {desc}")
        print(f"          {verdict}")
        if verdict.startswith("PASS"):
            n_pass += 1
    print("-" * 78)
    print(f"{n_pass}/{len(results)} 护栏被证实有效")
    if n_pass != len(results):
        raise SystemExit(1)
    print("全部源码已还原")


if __name__ == "__main__":
    main()