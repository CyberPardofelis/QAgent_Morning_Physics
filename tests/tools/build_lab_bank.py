"""把 DS 给的新题库落成仓库里的正式题库文件。

## 为什么要写这个脚本而不是手工复制

part1 那份 JSON **本身就解析不了**: line 122 `"argmax": [8/5, 6/5]`
—— JSON 语法里没有除法。手工"修一下"很容易顺手改错别的,
而且我改了也不会有人知道。所以: 解析 -> 报告 -> 结构化修正 -> 落盘。

## 这一版做了什么

  1. 修 `[8/5, 6/5]` -> `[1.6, 1.2]`(同一道 LP 题在 cvxpy-01 里本来就是小数,
     说明是笔误; 而且 8/5, 6/5 是有理数, 直接写字符串 "8/5" 会让下游类型混乱)
  2. **砍掉 qutip 整组**(4 题)。本机 Python 3.14 装不了 qutip,
     留着就是 4 道永远"环境不具备"的题 —— 混进统计里只会污染分母。
     ★ 工具认证测试里放一道跑不了的工具, 等于没测这个工具, 还假报了一整组的分 ★
  3. 给每道题补 `data_ref`(题面要读的文件)和 `verifier_spec` 的空位,
     方便后面挂独立验证器。
  4. part2 原样保留 —— 它已经核对过: tool_hints 全 null、题面无工具名。

## 为什么不改 part2 的题面措辞

part2 的题面里其实还有"软"的方法暗示(res-12 的"枚举所有拓扑"、
res-09 的"不超过已知最优解 1.10 倍")。**这一版不动它们** ——
因为那是题面设计层面的判断, 该你拍板, 不该我在脚本里偷偷改。
它们会被 `audit_new_problem_bank.py` 的软检查列出来供你决定。
"""
from __future__ import annotations

import json
import os
import re
import sys

BASE_SRC = (r"C:\Users\Nightingale\.minimax\v2\assets\2026\10\09"
            r"\15-04-31-590-asset_20261009-150431-590_"
            r"75a846261199_9740f800-BASEPROBLEMS_BASE.txt")
RES_SRC = (r"C:\Users\Nightingale\.minimax\v2\assets\2026\10\09"
           r"\15-04-31-593-asset_20261009-150431-593_"
           r"e7a9ff794cb4_a5a03d6c-BASEPROBLEMS_RES.txt")
ROOT = r"D:\pythonProject\QAgent"
OUT_DIR = os.path.join(ROOT, "data", "lab")

#: 本机装不上的工具 -> 整组砍掉。这些是"环境不具备", 不是"模型不会",
#: 混进正确率分母就是污染测量。
DROP_TOOLS = {"qutip_quantum"}

#: ★ DS 用的是**函数名**(`z3_solve`), 工具目录用的是**库名**(`z3`) ★
#: 9 个工具**一个都对不上**, 后果有两层:
#:   1. part1 最核心的判据「有没有真用上指定工具」36 道全部判 False
#:   2. 更糟: 题面写"使用 z3_solve", 目录里只有 `z3`, 而 Explorer 的规矩是
#:      "tools 里的 id 必须逐字来自目录" —— **题面和目录互相矛盾**
#: 所以题库这一侧要向目录对齐, 而不是反过来改目录(目录是给模型看的, 短名更清楚)。
TOOL_ALIAS = {
    "lean_check": "lean4",
    "z3_solve": "z3",
    "sympy_simplify": "sympy",
    "cvxpy_solve": "cvxpy",
    "networkx_analyze": "networkx",
    "scipy_numerical": "scipy",
    "mpmath_high_precision": "mpmath",
    "ortools_cp_sat": "ortools",
    "oeis_lookup": "oeis",
}

#: JSON 里出现的 `a/b` 形式的有理数字面量。
_FRAC = re.compile(r"(?<![\w.])(\d+)\s*/\s*(\d+)(?![\w.])")


def fix_fraction_literals(text: str) -> tuple[str, list[str]]:
    r"""把裸的 `8/5` 换成 `1.6`。

    只处理**最内层**方括号(`[ ... ]` 里不再有方括号)——
    这样嵌套数组 `[[3,4],[4,3]]` 不会被误伤, 而 `[8/5, 6/5]` 一定会被覆盖。

    ★ 第一版写错了 ★ 我用了 `\[([^\[\]]*?[\d\s])[^\[\]]*?\]`,
    捕获组因为惰性量词只咬住一个字符(实测 group(1) == "8"),
    于是 `_FRAC.search` 永远不命中, 函数**什么都不改也不报错** ——
    看起来跑通了, 实际上是个恒等函数。写完这类"看着对"的加工函数,
    一定要拿那行坏数据单测它。
    """
    notes: list[str] = []

    def _one(f: re.Match) -> str:
        a, b = int(f.group(1)), int(f.group(2))
        if b == 0:
            return f.group(0)
        notes.append(f"{f.group(0)} -> {a / b!r}")
        return repr(a / b)

    def _sub(m: re.Match) -> str:
        body = m.group(1)
        return "[" + _FRAC.sub(_one, body) + "]" if _FRAC.search(body) else m.group(0)

    return re.sub(r"\[([^\[\]]*)\]", _sub, text), notes


def salvage_json(text: str) -> tuple[dict | None, list[str]]:
    """尽量把 part1 救回来: 先原样解析, 失败再按分数字面量修一版。"""
    try:
        return json.loads(text), []
    except json.JSONDecodeError as e:
        first_err = (f"原样解析失败 @ line {e.lineno}: {e.msg}")
    fixed, notes = fix_fraction_literals(text)
    try:
        d = json.loads(fixed)
        return d, [first_err] + [f"  修正 {n}" for n in notes]
    except json.JSONDecodeError as e2:
        return None, [first_err, f"修正后仍失败 @ line {e2.lineno}: {e2.msg}"]


def strip_unsupported(d: dict) -> tuple[dict, list[str]]:
    """砍掉装不上的工具组。"""
    kept, dropped = [], []
    for it in d["items"]:
        if it.get("tool_under_test") in DROP_TOOLS:
            dropped.append(it["id"])
            continue
        kept.append(it)
    d["items"] = kept
    d.setdefault("dropped", {})
    d["dropped"] = {
        "tools": sorted(DROP_TOOLS),
        "ids": dropped,
        "why": "本机 Python 3.14 无 wheel; 工具认证题放一道跑不了的等于没测还假报分",
    }
    return d, dropped


def align_tool_names(d: dict) -> list[str]:
    """把 DS 的函数名对齐到工具目录的 id。

    要改三处, 少改一处就还是矛盾的:
      `tool_under_test`(统计口径) / `tool_hints`(题面给模型看的) /
      题面正文里出现的工具名
    """
    notes: list[str] = []
    for it in d["items"]:
        old = it.get("tool_under_test")
        new = TOOL_ALIAS.get(old)
        if not new:
            continue
        it["tool_under_test"] = new
        it["tool_under_test_original"] = old
        notes.append(f"{it['id']}: {old} -> {new}")
        h = it.get("tool_hints")
        if isinstance(h, str):
            for o, n in TOOL_ALIAS.items():
                h = h.replace(o, n)
            it["tool_hints"] = h
        p = it.get("prompt")
        if isinstance(p, str):
            for o, n in TOOL_ALIAS.items():
                p = re.sub(rf"\b{re.escape(o)}\b", n, p)
            it["prompt"] = p
    return notes


def data_refs(d: dict) -> None:
    """把题面里引用的 data/xxx 抽成显式字段, 方便后面挂输入数据。"""
    for it in d["items"]:
        blob = (str(it.get("prompt", "")) + " "
                + json.dumps(it.get("ground_truth", {}), ensure_ascii=False))
        it["data_ref"] = sorted(set(re.findall(
            r"data/[\w./-]+\.(?:txt|json|gml|csv)", blob)))


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    report: list[str] = []

    # ---------- part1 ----------
    raw1 = open(BASE_SRC, encoding="utf-8").read()
    d1, notes1 = salvage_json(raw1)
    report.append("== part1 ==")
    report += [f"  {n}" for n in notes1] or ["  原样即可解析"]
    if d1 is None:
        report.append("  !! part1 救不回来, 中止")
        print("\n".join(report))
        return 1
    before = len(d1["items"])
    d1, dropped = strip_unsupported(d1)
    aligned = align_tool_names(d1)
    data_refs(d1)
    report.append(f"  砍掉 {len(dropped)} 题({', '.join(dropped)}), "
                  f"{before} -> {len(d1['items'])}")
    report.append(f"  工具名对齐 {len(aligned)} 题(DS 用函数名, 目录用库名)")
    # 对齐后必须与目录**逐字**一致 —— 否则又是题面和目录互相矛盾
    try:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from agent.tools.research.roles.catalog import build_catalog
        ids = {t["id"] for t in build_catalog()}
        leftover = sorted({i["tool_under_test"] for i in d1["items"]
                           if i["tool_under_test"] not in ids})
        report.append("  ★ 对齐后仍不在目录里的工具: "
                      + (", ".join(leftover) if leftover else "无"))
    except Exception as e:  # noqa: BLE001
        report.append(f"  ★ 校验目录失败: {type(e).__name__}: {e}")

    # ---------- part2 ----------
    raw2 = open(RES_SRC, encoding="utf-8").read()
    d2 = json.loads(raw2)
    data_refs(d2)
    report.append("== part2 ==")
    report.append(f"  原样解析 OK, {len(d2['items'])} 题")

    for name, d in (("part1_tool_capability.json", d1),
                    ("part2_research_capability.json", d2)):
        p = os.path.join(OUT_DIR, name)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
        # 落盘后**立刻重新解析一次** —— 写出去的必须是能读的
        json.load(open(p, encoding="utf-8"))
        report.append(f"  写出并复读通过: data/lab/{name} "
                      f"({os.path.getsize(p)} 字节)")

    print("\n".join(report))
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())