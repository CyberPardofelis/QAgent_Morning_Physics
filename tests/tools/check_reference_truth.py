"""复算闸门: 题库参考答案正确性不过, 就不许评测。

★ 这个门为什么必须存在 ★
ground_truth 是别人写的。它错的时候, verifier 会拿错的参考答案判分,
于是每道题都判失败 —— 而分数看起来完全正常("模型不会做这些题")。
这种错误**从分数里看不出来**, 只能靠独立复算抓。

★ 闸门自己的可信度怎么保证 ★
一个只会输出 PASS 的门比没有门更危险。所以这里带一道**负控实验**:
把每题的 ground_truth 数值字段扰动一遍再跑同一个 fn,
要求至少一道 check 从 ok 翻成 not-ok。翻不动 = 那道题的判定逻辑是恒真的,
门在这里没干活, 直接报 NEGCTL_FAIL。

★ 报不出来就说报不出来 ★
UNVERIFIABLE(缺文件/缺环境/GT 本身不良定义) 和 FAIL(GT 错) 分开报。
把"我没验过"混进"验过了没问题"里, 闸门就变成了橡皮章。

用法:
    python tests/tools/check_reference_truth.py            # 全跑
    python tests/tools/check_reference_truth.py --negctl-only
退出码: 0 = 可评测; 1 = 有 FAIL(题库必须改); 2 = 只有 UNVERIFIABLE(需人工补)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_recompute import RECOMPUTE  # noqa: E402

# Windows 控制台默认 GBK, 一个外来字符就能让 print 崩在半路, 报告全丢。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BANKS = [
    os.path.join(ROOT, "benchmark", "part1_tool_capability.json"),
    os.path.join(ROOT, "benchmark", "part2_research_capability.json"),
]
REPORT = os.path.join(ROOT, "data", "research", "_reference_truth_report.json")


# --------------------------------------------------------------------------
def probe_env():
    """题库需要的通道到底在不在。在评测之前就该知道, 不要跑一半才发现。"""
    out = {}

    # Python 包
    for mod in ("qutip", "z3", "cvxpy", "ortools", "networkx", "sympy",
                "scipy", "mpmath", "numpy"):
        try:
            __import__(mod)
            out[f"pkg:{mod}"] = True
        except Exception:
            out[f"pkg:{mod}"] = False

    # 项目自带工具
    try:
        sys.path.insert(0, os.path.join(ROOT, "agent", "tools", "research"))
        import research_tools  # noqa: F401
        out["tool:research_tools"] = True
    except Exception:
        out["tool:research_tools"] = False

    try:
        sys.path.insert(0, os.path.join(ROOT, "agent", "tools", "research"))
        import research_oeis  # noqa: F401
        out["tool:research_oeis"] = True
    except Exception:
        out["tool:research_oeis"] = False

    out["tool:loogle_search"] = os.path.exists(os.path.join(
        ROOT, "agent", "tools", "research", "research_loogle.py"))

    # 数据文件
    for rel in ("data/nx_04_graph.gml", "data/jss_04.json",
                "data/tsp_200.txt", "data/tsp_200_opt.txt"):
        out[f"file:{rel}"] = os.path.exists(os.path.join(ROOT, rel))

    # Lean 工具链
    lean = os.path.join("D:\\pythonProject\\Others", "mathlib4-v490")
    out["toolchain:lean4.9.0"] = os.path.isdir(lean)
    return out


# --------------------------------------------------------------------------
def _numeric_leaves(obj, prefix=()):
    """收集 gt 里所有数值叶子, 供负控扰动。返回 [(键元组, 值)]。

    ★ 用键元组而不是 '.a.b' 字符串 ★ 键名本身可能带点(negativity_at_0.4),
    用分隔符拼路径会被切错。
    """
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += _numeric_leaves(v, prefix + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += _numeric_leaves(v, prefix + (i,))
    elif isinstance(obj, bool):
        found.append((prefix, obj))           # bool 也扰动, 单独判定
    elif isinstance(obj, (int, float)):
        found.append((prefix, obj))
    return found


def _set_by_path(obj, keys, value):
    """按键元组写回。"""
    cur = obj
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value


def _perturb(gt, leaves):
    """把 gt 深拷贝后扰动所有数值/布尔叶子。"""
    mutated = json.loads(json.dumps(gt))
    for keys, val in leaves:
        if isinstance(val, bool):
            _set_by_path(mutated, keys, not val)
        elif isinstance(val, int):
            _set_by_path(mutated, keys, val + 7 if val >= 0 else val - 7)
        else:
            _set_by_path(mutated, keys, val * 1.37 + 0.011)
    return mutated


# --------------------------------------------------------------------------
def run_bank(path, negctl=True):
    with io.open(path, encoding="utf-8") as fh:
        bank = json.load(fh)
    rows = []
    for item in bank["items"]:
        iid = item["id"]
        gt = item.get("ground_truth") or {}
        fn = RECOMPUTE.get(iid)
        if fn is None:
            rows.append({"id": iid, "part": item.get("part"),
                         "status": "NO_RECOMPUTE",
                         "reason": "RECOMPUTE 里没有这一条 —— 题库加了新题但没写复算"})
            continue

        t0 = time.time()
        try:
            res = fn(gt)
        except Exception as e:  # noqa: BLE001
            res = {"status": "ERROR", "derived": {}, "checks": [],
                   "reason": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc(limit=3)}
        res["id"] = iid
        res["part"] = item.get("part")
        res["tool_under_test"] = item.get("tool_under_test")
        res["elapsed"] = round(time.time() - t0, 2)

        # ---- 负控: 只对"我判 PASS"的题做 ----
        # 已经 FAIL 的题不用测: 它本来就没说"对", 扰动后也还是错,
        # ok 数只会停在 0 再往下 —— 那是设计缺陷不是判定不敏感。
        if negctl and res["status"] == "PASS":
            leaves = _numeric_leaves(gt)
            if not leaves:
                res["negctl"] = "N/A (GT 里没有数值/布尔字段可扰动)"
            else:
                mutated = _perturb(gt, leaves)
                try:
                    mres = fn(mutated)
                    before_ok = sum(1 for c in res["checks"] if c[1])
                    after_ok = sum(1 for c in mres.get("checks", []) if c[1])
                    flipped = after_ok < before_ok
                    res["negctl"] = "OK" if flipped else "NEGCTL_FAIL"
                    res["negctl_detail"] = (
                        f"扰动 {len(leaves)} 个字段 -> ok {before_ok} 降到 {after_ok}"
                        + ("" if flipped else "  <-- 扰动答案后判定没变, 这道门在此处没干活"))
                    if not flipped:
                        res["status"] = "NEGCTL_FAIL"
                except Exception as e:  # noqa: BLE001
                    res["negctl"] = f"NEGCTL_ERROR: {type(e).__name__}: {e}"
        elif negctl:
            res["negctl"] = "N/A (该题已非 PASS, 无需负控)"
        rows.append(res)
    return {"bank": os.path.basename(path), "part": bank.get("part"),
            "name": bank.get("name"), "items": rows}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negctl-only", action="store_true")
    ap.add_argument("--json", default=REPORT)
    args = ap.parse_args()

    env = probe_env()
    results = [run_bank(p) for p in BANKS]

    tally = {}
    failures, unver, negctl_bad = [], [], []
    for b in results:
        for r in b["items"]:
            tally[r["status"]] = tally.get(r["status"], 0) + 1
            if r["status"] in ("FAIL", "ERROR", "NEGCTL_FAIL", "NO_RECOMPUTE"):
                failures.append(r)
            elif r["status"] == "UNVERIFIABLE":
                unver.append(r)
            if r.get("negctl") == "NEGCTL_FAIL":
                negctl_bad.append(r)

    total = sum(tally.values())
    print("=" * 78)
    print("复算闸门 —— 题库参考答案独立复算")
    print("=" * 78)
    print("状态分布: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    print(f"总计 {total} 题\n")

    print("-" * 78)
    print("[参考答案是错的 / 跑挂了]")
    for r in failures:
        print(f"  !! {r['id']:<32} {r['status']}")
        for label, ok, detail in r.get("checks", []):
            if not ok:
                print(f"       - {label}: {detail}")
        if r.get("reason"):
            print(f"       ! {r['reason']}")
        if r.get("negctl_detail"):
            print(f"       ~ 负控: {r['negctl_detail']}")

    print("-" * 78)
    print("[复算不出来 —— 不是'对', 是'没验']")
    for r in unver:
        print(f"  -- {r['id']:<32} {r.get('reason', '')[:100]}")

    print("-" * 78)
    print("[环境通道]")
    for k, v in sorted(env.items()):
        print(f"  {'OK ' if v else 'MISS'}  {k}")

    negctl_n = sum(1 for b in results for r in b["items"]
                   if str(r.get("negctl", "")).startswith("OK"))
    print("-" * 78)
    print(f"[负控] 扰动 GT 后判定确实翻转的题: {negctl_n} 题"
          f"  (共 {total - len(unver)} 题跑了比对)")
    if negctl_bad:
        print(f"  负控失败 {len(negctl_bad)} 题 —— 这些题的判定逻辑对该字段不敏感")
        for r in negctl_bad:
            print(f"    !! {r['id']}: {r.get('negctl_detail', '')}")

    report = {"env": env, "tally": tally,
              "results": [r for b in results for r in b["items"]]}
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    io.open(args.json, "w", encoding="utf-8", newline="\n").write(
        json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\n明细: {args.json}")

    hard = tally.get("FAIL", 0) + tally.get("ERROR", 0) \
        + tally.get("NEGCTL_FAIL", 0) + tally.get("NO_RECOMPUTE", 0)
    if hard:
        print(f"\n>>> 闸门不放行: {hard} 题的参考答案需要修")
        return 1
    if unver:
        print(f"\n>>> 需要人工处理: {len(unver)} 题复算不出来")
        return 2
    print("\n>>> 闸门放行")
    return 0


if __name__ == "__main__":
    sys.exit(main())