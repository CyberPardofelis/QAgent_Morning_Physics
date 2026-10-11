# 进展、问题与失败题目详情（R 套，独立验证基线）

> 生成时间：2026-10-09 14:00
> 数据：`data/research/r_scored.json`、`data/research/all60_process.json`、
> `data/research/r_baseline2.log`、`data/research/_fails.txt`
> 抓取脚本：`tests/tools/_fail_detail.py`（可重跑刷新）
> 配套：`readme_minimax.md`（环境）、`all60_process.md`（60 题逐题过程）

---

## 一、当前进展

### 一句话

**R 套 20 题，独立验证 9 题通过**（3 → 5 → 9，逐轮修复的效果可测）。剩余 11 题已能干净地分成三类，
没有假阴性混在里面。

### 三轮基线的演进

| 轮次 | 结果 | 关键改动 |
|---|---|---|
| 第 1 轮 | **3/20** | 起点 |
| 第 2 轮 | **5/20** | 补传 `answer_fields`（机器门第一次真的运行） |
| 第 3 轮 | **9/20** | DeepSeek P0 六条全部落地 + res-12 前导加号修复 |

### 本轮修复清单（全部有回归测试）

| 修复 | 问题 | 测试 |
|---|---|---|
| P0-1 输入数据内联进题面 | 8 道题的输入只在文件里，模型读不到却被告诫"把数据抄进代码"→ 那 8 道不可解 | `check_input_reachable.py` |
| P0-2 封 numpy 整类读取 | `np.loadtxt/fromfile/genfromtxt/load/memmap/DataFrame.from_csv` 全能绕过 AST 门读走参考答案 | `check_sandbox_no_read.py` |
| P0-3 恢复 `verdict.derive` 接线 | 该段被**我自己的重写脚本**整段删掉，`declared_vs_derived` 恒为 0 | `check_verdict_derive.py` + 端到端 |
| P0-4 `answer_spec` 进 prompt | 参数加了但 `research_verify` 没有这个形参，等于没接 | 端到端 |
| P0-5 `methods_for` 真过滤 | 认题是装饰品：不同形状返回同一份全量目录 | `check_methods_catalog.py` |
| P0-6 `harness_gate` 加 L3 可解性门 | 让"题不可解"在**跑之前**被拦住 | `harness_gate.py` |
| res-12 前导加号 | `Fraction("+")` 抛异常 → **正确的答案记成 verifier 异常，0 分** | `run_all_verifiers.py` |

全量回归 **1417 passed / 5 skipped**，前置门 L1/L2/L3 全过。

### 维度分（现在是有意义的数字）

```
exploration       0.00  ▏  ← 唯一还是 0，因为回路里真的没有探索阶段
tool_quality      0.31  ▎
cross_validation  0.36  ▎
self_correction   0.46  ▍
method_diversity  0.49  ▍
correctness       0.53  ▌
calibration       0.53  ▌
certification     0.61  ▋
平均过程分 22.7
```

**对比修复前**：`exploration` / `cross_validation` 是**结构性常数**——适配器把
`tool` 恒填 `"python"`、`method` 恒填 `"brute_force"`，且永远产不出
`experiment`/`cross_check` 类型的步骤。实测「凭空贴两个 experiment 标签」能让
总分 42.2 → 59.7，**agent 行为零变化**。现在每个数都从真实证据算出来，有方差。

---

## 二、当前问题（未解决）

### P1｜三道「格式合规」失败：机器门在喊，模型不听话

res-002 / 011 / 019。**特征完全一致**：代码跑通了、机器门也判了 PROVED，
但没交出规定格式的制品，于是独立验证器无从核对，给 0 分。

res-019 最有代表性——**连续 5 轮都被同一句话告知**：

```
「你判了 PROVED, 但**没有交出本题要求的对象** —— 机器无法独立复核
  一个没交出来的东西。必须打一行制品(字段名照抄):
  print("ANSWER: " + json.dumps({"group": ...}))」
```

它改了策略（`construct` ↔ `construct_and_check` 交替）、换了库
（`collections/itertools/numpy/random/sympy`），**但制品一直没交**。

> **这正是 DeepSeek 说的**：「宣布结论」成本为零，交制品要费力气，
> 模型总选前者。我把它做成了判定层的规则（降级），**但它仍然能宣布，
> 只是宣布了没用** —— 没有在**行动层**封死这条路。

### P2｜res-016 TSP 压根没跑：`Blocked`

`0 轮 / 1 次反馈`，反馈内容是 `Blocked`。跑测命令里已经
`set QAGENT_COST_GUARD=off`，**但它还是被拦住了**。原因未查明
（可能是上游限流返回错误串，也可能是环境变量没生效）。

→ 定位清楚之前，**任何"它做错了"的判断对这道题都不成立**。

### P3｜res-011 的制品门没开火，但验证器说缺制品

机器侧判词是「代码显式标记 VERDICT: PROVED；机器复核通过」——说明
`_missing_artifact_reason` **没有触发**；而独立验证器判
`artifact_present` 失败。两者对不上。

→ 要么 `answer_fields` 没传到（跑测脚本问题），要么它交了制品但字段名对。
**未查清。**

### P4｜res-014：6 轮里有 5 轮 `NoCode`

模型五次没吐出可执行代码。这不是数学问题，是**基本输出纪律**问题。
Lean 路径上 `elapsed_ms=0`、`库=['Mathlib']` 说明我的过程抽取器对
`kind=lean` 的证据读得不对（Lean 源码不是 Python import）。

### P5｜验证器内部一致性：判对了但子项失败

`res-013` 判 OK 但 `duplicate_entries` 失败；`res-017` 判 OK 但
`range_declared` 失败。说明**分项判据和总判定之间没有严格的蕴含关系** ——
有些失败项不影响最终 correct。这是新暴露的一层，之前没注意过。

### P6｜`exploration = 0.00` 是真实的 0

适配器现在**能**产出 `experiment` 步了（从真实证据翻译），但**回路里没有任何
阶段会去做小规模实验**。这是 DeepSeek 排序里的第二项「Explorer 显式化」，未做。

---

## 三、失败题目逐题详情

分三类：**真失败（能力）5 道** / **格式合规 3 道** / **没跑到证据 3 道**。

### A. 真失败（能力问题）

---

#### res-006 PSLQ —— 真残差不够

- 题面：找 `a·ln2 + b·ln3 + c·ln5 + d·ln7 ≈ 0`，`(a,b,c,d)` 不全为零，系数 ≤ 1e6
- 要交：`relation`、`residual`
- 过程：3 轮证据 / 4 次反馈

| 轮 | 判定 | 策略 | 用了什么 | 结果 |
|---|---|---|---|---|
| 1 | failed | symbolic | decimal,itertools,math,sympy | 静态门拦：`sympy` 里没有 `mpmath` 这个属性 |
| 4 | failed | symbolic | json,sympy | 静态门拦：`sympy` 里没有 `pslq` 这个属性 |
| 5 | **proved** | numeric_highprecision | decimal,json,math | 机器复核通过 |

- **失败子项**：`residual_small` —— 残差没到题目要求的 < 1e-50

> **值得注意**：它连撞两次"sympy 里有 mpmath/pslq"的幻觉，但静态门两次都抓住了，
> 并且**最后一次换了正确的高精度路线**。机制在起作用，是精度没做够。

---

#### res-007 OEIS —— 公式复算不上

- 题面：给序列前 30 项，识别 OEIS 条目、提公式、验证前 100 项
- 要交：`oeis_id`、`formula`、`verified_100_terms`
- 过程：**1 轮 / 0 次反馈**，97ms
- 失败子项：`formula_recomputable` —— 验证器无法用它给的公式复算出序列

> **单轮就宣布成功**。反例模式：只交了个 `polynomial` 字符串，
> 但它和序列的真正生成式对不上。

---

#### res-008 Lean 二项式和 —— 凭记忆写 API，6 轮全败

- 要交：无（真假判定题）
- 过程：6 轮 / 6 次反馈，**全部 inconclusive**

逐轮编译器原话（这是最有信息量的部分）：

```
第1轮  5 处编译错误   object file '.\.\.lake\build\lib\Mathl…'
第2轮  3 处编译错误   tactic 'rewrite' failed, did not find instance
第3轮  3 处编译错误   unexpected token '='; expected ',' or
第4轮  2 处编译错误   unknown constant 'Nat.add_pow'      ← ★
第5轮  3 处编译错误、用了 sorry 或 axiom
第6轮  2 处编译错误、用了 sorry 或 axiom
```

- 失败子项：`lean_proved`

> `unknown constant 'Nat.add_pow'` 是典型症状：**模型在凭记忆写 Mathlib API，
> 而 4.9.0 里没有这个名字**。第5、6轮开始用 `sorry` 兜底 —— 那等于没证。
> DeepSeek 的诊断（「凭记忆写 API + 错误信息不可读」）在这里**完全成立**。

---

#### res-009 Lean 二次公式 —— 同病，错误数剧烈波动

| 轮 | 编译错误数 | 备注 |
|---|---|---|
| 1 | 8 | 用了 sorry |
| 2 | 6 | |
| 3 | **20** | 越改越糟 |
| 4 | 4 | 错误少了但仍用 sorry |
| 5 | 6 | |
| 6 | 5 | |

- 失败子项：`lean_proved` + `no_sorry`

> **第3轮 20 个错误**是最刺眼的：它在错误的方向上"努力"。
> 而且 6 轮里 5 轮用了 `sorry` —— 它在用兜底掩盖没证成。

---

#### res-020 假设修正循环 —— 反例对，但没交够

- 题面：检验「对所有 n≥1，2^n+1 是素数」（实际 n=3 就假，9=3²）
- 要交：`verdict`、`evidence`、`refined_conjecture`
- 过程：3 轮
- 失败子项：`counterexample_minimal_n_is_3`、`search_range_declared`

> **方向是对的**（判 refuted），但没给出"最小反例恰为 n=3"这个关键结论，
> 也没声明搜索范围。**内容正确但没答完题**。

---

### B. 格式合规（答案可能对，但交不出来）

---

#### res-002 Golomb 尺 —— 拿编造的"已知最优值"判了假

- 要交：`marks`、`length`、`distinct_distances`
- 过程：**1 轮 / 0 次反馈**，381ms

```
COUNTEREXAMPLE: optimal_length=283 > 200, so no 15-mark Golomb ruler has length <= 200
VERDICT: REFUTED
```

- 失败子项：`artifact`

> ★ **这是个假反例**。15 标 Golomb 尺的**最优长度是 170**（我独立构造并验证过：
> 105 个距离全互异、长度 170 ≤ 200）。它引用的 "283" 是幻觉。
>
> R4 那道门（「和源码里硬编码的已知值比不上」）本该拦住，但那个常量是它
> **运行时算/编的**而不是写在源码里的字面量，所以门看不见。
> **反例本身是假的，而凭证机制无法识别这一点** —— 这是凭证层的边界。

---

#### res-011 对角数独 —— 判对了但没交制品（见 P3）

- 要交：`solution`、`unique`
- 过程：**1 轮 / 0 次反馈**，1315ms，z3
- 判定：机器门说 PROVED，验证器说 `artifact_present` 失败

> **两者不一致，未查清。**

---

#### res-019 60 阶群 —— 5 轮被同一句话告知，5 轮不改

- 要交：`group`、`is_nonabelian`、`is_A5`、`generators`
- 过程：11 条证据 / 11 次反馈 —— 每一轮都是「先判 PROVED → 再被制品门否决」

| 轮 | 策略 | 用的库 | 结果 |
|---|---|---|---|
| 1 | construct_and_check | itertools,json | 判 PROVED → 制品门否决 |
| 2 | construct | collections,itertools,json,numpy | 同样 |
| 3 | construct | itertools,json | 同样 |
| 4 | construct | collections,itertools,json,sympy | 同样 |
| 5 | construct_and_check | itertools,json | 同样 |
| 6 | construct_and_check | itertools,json,random | KeyError 崩了 |

> **它换了策略、换了库，就是不交制品。** 这是 P1 那个问题最干净的一次复现：
> 判定层的否决**不足以**让模型改变行为。

---

### C. 没跑到证据（不是判错）

#### res-003 Hadamard-28 —— 6 轮里 3 次被强制换方法挡回

```
第0轮 NoCode
第1轮 NoCode
第2轮 静态门拦：sympy.polys.galoistools 里没有 gf_irred_p_givens
第3轮 StrategyUnchanged   ← 强制换方法生效
第4轮 StrategyUnchanged
第5轮 StrategyUnchanged   ← 连续两次后放行，它仍然没换
```

> 它想用**有限域上的构造**（`galoistools`）做 Hadamard，
> 而题目是 **H Hᵀ = 28I 的整数矩阵** —— 方法选错了但方向不算离谱。
> 强制换方法这道门**正确地识别了**（第3轮就拦），但拦不住它坚持。

#### res-014 三阶魔方 —— 6 轮里 5 轮没吐出代码

```
第0~3轮 NoCode
第4轮 AssertionError
第5轮 NoCode
```
> 基本输出纪律问题，不是数学问题。

#### res-016 TSP-500 —— `Blocked`，一次没跑（见 P2）

---

## 四、给下一个人的三句话

1. **9/20 是可信的**——每道都有独立验证器，每个验证器都有咬合+不误杀双向自测，
   评测前置门三层全过。但**base / V2 那 40 题仍只有 agent 自述，不能当正确率读**。
2. **最该攻的是 Lean 两道**（res-008/009，V2 里还有两道）：不是格式问题也不是合规问题，
   是明确的"没做到"，且错误信息完全可读（`unknown constant 'Nat.add_pow'`），
   指向「凭记忆写 API + 不查证」。60 题里 Lean 占 4 道。
3. **两个机制已被证明"能工作但不足以改变行为"**：制品门（res-019 连喊 5 轮）、
   强制换方法（res-003 连挡 3 轮）。它们拦得住"假通过"，拦不住"不改"。
   下一步应该在**行动层**（工具调用/终止条件）封死，而不是在判定层继续加规则。