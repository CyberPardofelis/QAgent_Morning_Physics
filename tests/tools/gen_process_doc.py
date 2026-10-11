import io
import json

D = json.load(io.open("data/research/all60_process.json", encoding="utf-8"))
SUITE_TITLE = {
    "base": ("base 套 20 题（`baseproblems.txt`）",
             "➖ = 只有 agent 自述，**无独立验证器，不能当正确率读**"),
    "V2": ("V2 套 20 题（`baseproblems_V2.txt`）",
           "➖ 同上；这一套按「裸枚举必死」设计，规模题居多"),
    "R": ("R 套 20 题（`baseproblems_R.txt`）",
          "✅/❌ = **独立验证器**判定（自己重算，不读 agent 代码与自由文本）"),
}

out = []
out.append("## 一眼看懂的三个模式\n")
out.append("""从下面 60 题的过程里能读出三条反复出现的模式。它们比单题的对错更有指导意义。

**模式 A｜宣称结论的成本是零，交制品的成本不低。**
凡是失败题里 `自判=proved/refuted` 而验证器判"没交制品"的，都是这个模式
（res-015 / 017 / 019 / 020）。今晚的四个案例里，
- res-006 交了 `a=b=c=d=0` 的平凡关系并宣称 PROVED
- res-013 交了 89 组全错的解并宣称 PROVED
- res-014 交的动作序列连魔方都解不开，也宣称 PROVED
→ 声明 vs 推导的分离已经做了（`_verdict.py`），但**机器门还没在这轮跑测里生效**
（`run_all60.py` 漏传 `answer_fields`，已修）。

**模式 B｜策略标签换了，但方法没真的换。**
res-019 连续 4 轮都声明 `construct`，res-011 连续 6 轮都声明 `solver`，
res-015 连续 3 轮 `symbolic`。换标签不等于换计算形态 ——
`# STRATEGY:` 现在只是**标签**，机器没有检查它与实际 import 的库是否一致
（res-019 声明 construct，实际用的是 `collections/itertools/numpy/random`，
没有任何群论工具）。

**模式 C｜"强制换方法"这道门在多数失败题上没开火。**
它只在**超时**和**用错模块**时触发。而下面这些失败形态
（格式不对、精度不够、Lean 编译不过、公式对不上）都不属于这两类，
所以门正确地没开火 —— 但也意味着这些题**没有走"换一条路"这条路**。

""")

for suite in ("base", "V2", "R"):
    title, note = SUITE_TITLE[suite]
    rows = [r for r in D if r["suite"] == suite]
    out.append(f"\n---\n\n## {title}\n")
    out.append(f"> {note}\n")
    for r in rows:
        libs = sorted({l for rd in r["rounds"] for l in rd["libs"]})
        strats = [rd["strat"] for rd in r["rounds"]]
        # 策略是否在轮次间变化过
        uniq = sorted(set(strats))
        out.append(f"\n### {r['mark']} `{r['id']}`\n")
        out.append(f"> {r['prompt']}\n")
        out.append(f"- **要交什么**（题库契约字段）: "
                   f"{'、'.join(f'`{x}`' for x in r['ask_fields']) or '（只要判定，不交对象）'}")
        out.append(f"- **过程**: {r['n_ev']} 轮尝试 / {r['n_attempt']} 次重试反馈，"
                   f"{r['seconds']}s，{r['tokens']} token")
        out.append(f"- **声明的策略**: {' → '.join(strats)}"
                   + ("　*(多轮同标签 = 没换路)*" if len(uniq) == 1 and r["n_ev"] > 1 else ""))
        out.append(f"- **实际用的库**: {'、'.join(libs) or '（无 import）'}")
        out.append(f"- **agent 判定**: `{r['agent_verdict']}`"
                   + (f"　｜**过程分 {r['score']}**　短板 {r['weak']}"
                      if r["score"] is not None else ""))
        out.append(f"- **判据**: {r['judged_by']}")
        if r["cert"]:
            out.append(f"- **验证器判词**: {r['cert']}")
        if r["attempts"]:
            out.append("- **逐轮它被告知了什么**:")
            for a in r["attempts"][-4:]:
                out.append(f"  - 第{a['i']}轮 · `{str(a['type'])[:70]}` {a['detail']}")
    out.append("")

io.open("data/research/_all60_process.md", "w",
        encoding="utf-8").write("\n".join(out))
print("已生成 data/research/_all60_process.md",
      len("\n".join(out)), "字符")