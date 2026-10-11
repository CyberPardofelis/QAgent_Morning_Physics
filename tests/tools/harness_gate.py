"""评测前置门：harness 自证"我能评到每一道题"。

## 为什么要有这道门

DS 的判断(2026-10-09): 静默漏网必须从"跑完之后检查"变成"跑之前就保证"。

真实事故: `run_r_scored.py` 的 `best_evidence()` 只取 `kind=="code"`,
而 Lean 走 `_verify_via_lean` 落的是 `kind="lean"` —— 于是 res-008/009/018
三道**从头到尾没进过 verifier**, 直接落进 unverified。它们的证据就在磁盘上,
好好的。**"没跑到"和"判错"在统计上长得一样**, 所以这种洞必须用断言堵。

## 四层

  L0 benchmark/ 新题库的参考答案独立复算通过 —— GT 错则 verifier 稳定判错,
     而分数看起来完全正常("模型不会做"), 这种错从分数里看不出来
  L1 每道题都有 verifier, 且签名齐全
  L2 每道题用「已知正确的制品」跑 -> 必须 True; 用「已知错误的制品」跑 -> 必须 False
     (这就是 verifier 自带的咬合+不误杀自测, 这里把它**前置**成评测的门)
  L3 任何一道不通过 -> **拒绝开始评测**

## 怎么用

    python tests/tools/harness_gate.py              # 全跑(含 L0)
    python tests/tools/harness_gate.py --skip-l0    # 只跑旧的 L1-L3
    python tests/tools/harness_gate.py --quiet      # 只看结论

返回码非 0 = 评测数据不可信, 不要跑。
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from agent.tools.research._proc import run_capture  # noqa: E402

VDIR = os.path.join(HERE, "verifiers_r")


def l1_every_problem_has_verifier() -> list[str]:
    import score_baseline as SB
    import verifier_map as VM
    bad = []
    for it in SB.load_suite("R"):
        cid = it["id"]
        if cid not in VM.PROBLEM_TO_VERIFIER:
            bad.append(f"{cid}: 没有登记 verifier —— correctness 无法计算")
            continue
        try:
            mod = VM.load_verifier(cid)
        except Exception as e:
            bad.append(f"{cid}: verifier 加载失败 {type(e).__name__}: {e}")
            continue
        if not hasattr(mod, "verify"):
            bad.append(f"{cid}: verifier 缺 verify()")
            continue
        try:
            r = mod.verify("", "", "")
        except Exception as e:
            bad.append(f"{cid}: verify(空输入) 抛异常 {type(e).__name__}")
            continue
        miss = {"correct", "certificate", "final_answer", "details"} - set(r)
        if miss:
            bad.append(f"{cid}: 返回缺键 {miss}")
        if r.get("correct") is not False:
            # 空输入都放过 = "没给答案也算对", 那这道题等于没验
            bad.append(f"{cid}: 空输入 correct={r.get('correct')!r} 应为 False")
    return bad


def l2_bite_tests_pass() -> list[str]:
    """逐个跑 verifier 自带的 `__main__` —— 正例必须过、反例必须被拦。"""
    if not os.path.isdir(VDIR):
        return [f"verifiers_r 目录不存在: {VDIR}"]
    files = sorted(f for f in os.listdir(VDIR)
                   if f.startswith("res_") and f.endswith(".py"))
    if not files:
        return ["verifiers_r 下没有任何 verifier"]
    bad = []
    for f in files:
        try:
            # ★ 必须走 run_capture ★ 它强制 env 里的 PYTHONIOENCODING=utf-8。
            # 手写 subprocess.run(..., encoding="utf-8") 看着一样, 实则子进程
            # 按 cp936 写字节 -> 下面那句 `if "正例" not in out` 恒真 ->
            # 20 个全部通过的验证器被报成 20 条失败。见 _proc.py 文档。
            p = run_capture([sys.executable, os.path.join(VDIR, f)],
                            cwd=ROOT, timeout=1200)
        except subprocess.TimeoutExpired:
            bad.append(f"{f}: 自测超时")
            continue
        out = (p.stdout or "") + (p.stderr or "")
        if "[SKIP]" in out:
            continue                      # 环境不具备, 由报告单独标
        if p.returncode != 0:
            bad.append(f"{f}: 自测 exit={p.returncode}")
            for l in out.splitlines()[-4:]:
                bad.append("      " + l)
            continue
        # 自测必须**两个方向都有**, 只测正例的等于没测
        if "正例" not in out or "反例" not in out:
            bad.append(f"{f}: 自测只跑了一个方向 —— 必须同时有正例与反例")
    return bad



def l3_solvability() -> list[str]:
    """★ 可解性门: 这道题模型**拿到它需要的一切**了吗? ★

    2026-10-09 实测事故: 8 道题的输入数据只在 data/res_*.txt 里, 而沙箱禁读,
    题面又只说了"数据在某个文件" —— 模型**拿不到数据却被要求把数据抄进代码**,
    于是自己发明了题面(res-013 的"发明第三个变量")。那 8 道题从头到尾不可解,
    却算进了基线。跑完才发现, 代价是整个基线不可信。

    所以这道门要在**跑之前**断言:
      1. 题目引用的输入数据真的进了 claim(不是"某个文件里有")
      2. prompt 里要求的制品字段, 和验证器实际读的字段对得上
      3. claim 里没有任何"去读文件"这种拿到手就废的指示
    """
    import problem_data
    import problem_spec as PS
    import score_baseline as SB

    bad: list[str] = []
    for it in SB.load_suite("R"):
        cid = it["id"]
        claim = PS.build_claim(it["prompt"], it.get("expected"),
                               PS.data_hint_for(cid), problem_id=cid)

        # (1) 输入可达
        if not problem_data.input_reachable(cid):
            bad.append(f"{cid}: 输入数据不可达 —— 数据文件缺失或没内联")
        elif cid in problem_data.DATA_FILES:
            data = problem_data.read_data(cid) or ""
            probe = next((t for t in data.split()[:200]
                          if len(t) >= 3 and t in claim), None)
            if probe is None:
                bad.append(f"{cid}: 数据文件有内容, 但 claim 里找不到 —— "
                           f"模型拿不到题目输入")

        # (3) 拿到手就废的指示
        if "沙箱禁读文件" in claim and "本题的输入数据" not in claim:
            bad.append(f"{cid}: claim 里有『沙箱禁读文件』但没内联数据")

        # (2) 字段名一致
        # ★ 只查"答案字段", 不查自评字段 ★
        # 自评字段(verified/optimal/unique/irreducible...)是**故意**从 spec 里
        # 剥掉的 —— 让 agent 填它等于让它自己给自己打分。第一版这里没排除,
        # 结果 8 道题全部误报。
        # ★ 又一次"新检查误伤哪一类" ★ 这个坑我这轮踩了三次:
        #   制品门杀 sin^2+cos^2 / 映射表键对不上 / 这道门把自评字段当缺失。
        #   每加一条检查都要问: 它会把哪一类本来正常的东西判成异常?
        exp = it.get("expected") or {}
        val = exp.get("value") if isinstance(exp, dict) else None
        all_fields = list(val.keys()) if isinstance(val, dict) else []
        try:
            import problem_spec as _PS
            from agent.tools.research.research_verify import SELF_EVAL_FIELDS
            answer_only = [f for f in all_fields
                           if f not in SELF_EVAL_FIELDS]
        except Exception:
            answer_only = all_fields
        fields = answer_only
        spec = PS.expected_to_spec(exp, it["prompt"]) if fields else ""
        if fields and not spec:
            bad.append(f"{cid}: 有契约字段但生成不出 spec —— "
                       f"agent 看不到该交什么")
        for f in fields:
            if f not in spec:
                bad.append(f"{cid}: 契约字段 `{f}` 没进 spec")
    return bad


def l0_reference_truth() -> list[str]:
    """L0: 题库参考答案必须先独立复算过。

    ★ 为什么排在 L1 前面 ★
    verifier 是拿 ground_truth 判分的。GT 错了, verifier 就稳定地判错,
    而分数**看起来完全正常** —— 只是"模型在这些题上不会做"。
    这种错误从分数里看不出来, 只能靠独立复算抓, 所以它是入库门不是评测门。

    ★ 为什么不在这里 import check_reference_truth ★
    L1/L2/L3 跑的是旧 R 套, 本门是给 benchmark/ 新题库用的。
    子进程跑是为了不让题库一坏就把这道门本身带崩 —— 门的失败必须是结论。
    """
    script = os.path.join(HERE, "check_reference_truth.py")
    if not os.path.exists(os.path.join(ROOT, "benchmark")):
        return []                      # 还没导入新题库, L0 暂不适用
    if not os.path.exists(script):
        return [f"找不到 {script} —— L0 无法执行"]
    try:
        p = subprocess.run([sys.executable, script, "--json",
                            os.path.join(ROOT, "data", "research",
                                         "_reference_truth_report.json")],
                           cwd=ROOT, capture_output=True, text=True,
                           timeout=3600, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return ["L0 复算超过 1 小时未结束 —— 门自己的状态未知, 视为不通过"]
    except Exception as e:  # noqa: BLE001
        return [f"L0 复算脚本自身崩溃 {type(e).__name__}: {e} —— 门坏掉时必须报不通过"]

    if p.returncode == 0:
        return []
    try:
        import json
        rep = json.loads(io.open(
            os.path.join(ROOT, "data", "research", "_reference_truth_report.json"),
            encoding="utf-8").read())
        tally = rep.get("tally", {})
        bad = [r for r in rep.get("results", [])
               if r["status"] in ("FAIL", "ERROR", "NEGCTL_FAIL", "NO_RECOMPUTE")]
    except Exception:
        return [f"L0 复算未通过(退出码 {p.returncode}), 明细也读不出来"]

    if p.returncode == 1:
        detail = ", ".join(f"{r['id']}({r['status']})" for r in bad[:8])
        more = f" 等 {len(bad)} 题" if len(bad) > 8 else ""
        return [f"参考答案复算不通过 {len(bad)} 题: {detail}{more} "
                f"—— GT 错则分数全错, 详见 data/research/_reference_truth_report.json"]
    return [f"{tally.get('UNVERIFIABLE', 0)} 题参考答案复算不出来(缺文件/缺环境/"
            f"不良定义), 需要人工确认后才能开跑"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--skip-l0", action="store_true",
                    help="跳过 L0(题库参考答案复算) —— 只在你明确知道自己在干什么时用")
    args = ap.parse_args()

    # ★ 门不该在自己的结论上崩 ★
    # Windows 控制台 stdout 是 cp936, 印不出 '✗' -> UnicodeEncodeError ->
    # **恰好在最该报"不要开始跑"的那一刻抛栈退出**, 退出码还一样是 1,
    # 于是读日志的人分不清是"门没过"还是"门自己坏了"。
    # 这不是排版问题: 一个诊断工具的失败必须是结论, 不能是崩溃。
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 74)
    print("评测前置门 —— harness 自证能评到每一题")
    print("=" * 74)
    failures: list[str] = []
    layers = []
    if not args.skip_l0:
        layers.append(("★ L0 参考答案复算(GT 对不对)", l0_reference_truth))
    layers += [
        ("L1 每题都有可用 verifier", l1_every_problem_has_verifier),
        ("L2 咬合+不误杀自测全过", l2_bite_tests_pass),
        ("★ L3 可解性(输入可达/字段一致)", l3_solvability),
    ]
    for label, fn in layers:
        problems = fn()
        if problems:
            failures.extend(problems)
            print(f"  [!!] {label}")
            if not args.quiet:
                for p in problems[:12]:
                    print(f"        - {p}")
                if len(problems) > 12:
                    print(f"        ... 另有 {len(problems)-12} 处")
        else:
            print(f"  [OK ] {label}")

    print("=" * 74)
    if failures:
        print(f"✗ 前置门未过({len(failures)} 处) —— **评测数据不可信, 不要开始跑**")
        return 1
    print("✓ 前置门通过 —— 可以开始评测")
    return 0


if __name__ == "__main__":
    sys.exit(main())