# 60 题过程与结果清单

> 数据来源：`data/research/all60_scored.json` + `data/research/all60_process.json`
> 原始日志 `data/research/all60.log`（3409 秒，一次跑完三套）
> 生成脚本 `tests/tools/gen_process_doc.py`（改代码后重跑即可刷新）
>
> ⚠️ **这一轮的结果不能当基线**：`run_all60.py` 在 11:35 启动后，我又往主回路接了
> `_verdict.derive` 和方法目录，**跑测跨越了代码变更**。base/V2 段是旧代码的结果，
> R 段半新半旧。要拿基线必须用最终代码重跑。

## 图例

| 标记 | 含义 |
|---|---|
| ✅ | **独立验证器**判定通过（`tests/tools/verifiers_r/`，自己重算，不读 agent 代码与自由文本） |
| ❌ | 独立验证器判定不通过 |
| ❓ | 没跑到有判定的证据 / 没交出可复核的制品 |
| ➖ | 只有 agent 自述，**这一套没有独立验证器，不能当正确率读** |

## 总览

| 套 | ✅ | ❌ | ❓ | ➖ | 合计耗时 |
|---|---|---|---|---|---|
| base | — | — | — | 20 | 约 4 分钟 |
| V2 | — | — | — | 20 | 约 12 分钟 |
| **R** | **3** | **15** | 2 | 0 | 约 40 分钟 |

**R 套 15 道"错"的构成**（这是本轮最重要的信息）：

| 类别 | 题数 | 题号 |
|---|---|---|
| **没交制品**（不是算错） | 7 | res-004 005 011 015 016 017 019 020 中的 artifact 类 |
| 真错（数学/能力） | 6 | 006 残差不够 / 007 公式不对 / 008 009 Lean 用 sorry / 010 SG 第2项错 / 013 格式错 |
| 没跑到证据 | 2 | res-003、res-014 |

---## 一眼看懂的三个模式

从下面 60 题的过程里能读出三条反复出现的模式。它们比单题的对错更有指导意义。

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



---

## base 套 20 题（`baseproblems.txt`）

> ➖ = 只有 agent 自述，**无独立验证器，不能当正确率读**


### ➖ `num-001-euler-prime-polynomial`

> 验证 n^2+n+41 在 n=0 到 39 时全为素数，并检查 n=40 时的值。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，20.3s，5346 token
- **声明的策略**: symbolic
- **实际用的库**: json、sympy
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `num-002-goldbach-small`

> 对 4 到 1000 的所有偶数，找出一对素数和。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 2 轮尝试 / 1 次重试反馈，13.7s，10215 token
- **声明的策略**: enumerate → enumerate　*(多轮同标签 = 没换路)*
- **实际用的库**: json、sympy
- **agent 判定**: `proved`　｜**过程分 3.67**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `MachineGate: 代码标了 REFUTED 并给出反例 COUNTEREXAMPLE: {"n": 98}, 但机器用另一种独立方式` 检查偶数范围: 4 .. 1000 (共 499 个) 成功找到素数和分解的偶数个数: 453 未找到素数和分解的偶数: [98, 122, 124, 126, 128, 148, 150, 190, 192, 208, 210, 212, 220, 222,…

### ➖ `num-003-carmichael-561`

> 验证 561 是合数，且对所有与 561 互素的 a，a^560 ≡ 1 mod 561。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 2 轮尝试 / 1 次重试反馈，22.9s，10871 token
- **声明的策略**: symbolic → symbolic　*(多轮同标签 = 没换路)*
- **实际用的库**: json、sympy
- **agent 判定**: `refuted`　｜**过程分 3.67**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `代码执行出错(TypeError), 没有产生可复核的结果` === Part A: 561 是合数 === 561 的因式分解: {3: 1, 11: 1, 17: 1} sympy.isprime(561) = False 非平凡因子: [3, 11, 17] 561 = {3: 1, 11: 1, 17: 1} =…

### ➖ `num-004-100-factorial-trailing-zeros`

> 求 100! 末尾有多少个 0。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，9.1s，5221 token
- **声明的策略**: symbolic
- **实际用的库**: json、sympy
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'tool_quality', 'exploration']
- **判据**: 自判

### ➖ `num-005-inclusion-exclusion-100`

> 求 1 到 100 中不能被 2、3、5 整除的数的个数。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 2 轮尝试 / 1 次重试反馈，13.3s，10440 token
- **声明的策略**: enumerate → (未声明)
- **实际用的库**: json、sympy
- **agent 判定**: `proved`　｜**过程分 10.67**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `代码执行出错(AttributeError), 没有产生可复核的结果` 

### ➖ `alg-006-fibonacci-matrix`

> 验证矩阵 [[1,1],[1,0]]^n = [[F_{n+1},F_n],[F_n,F_{n-1}]] 对 n<=20 成立。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 2 轮尝试 / 1 次重试反馈，21.8s，10944 token
- **声明的策略**: symbolic → symbolic　*(多轮同标签 = 没换路)*
- **实际用的库**: json、sympy
- **agent 判定**: `proved`　｜**过程分 2.33**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `代码执行出错(TypeError), 没有产生可复核的结果` All n in 1..20 hold numerically: True Symbolic induction base case OK: True Symbolic induction step OK: False

### ➖ `alg-007-am-gm`

> 验证对任意 a,b>0，有 (a+b)/2 >= sqrt(ab)。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，8.9s，5082 token
- **声明的策略**: symbolic
- **实际用的库**: sympy、z3
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `alg-008-circle-max-linear-sum`

> 在 x^2+y^2=1 下求 x+y 的最大值。

- **要交什么**（题库契约字段）: `max`、`argmax`
- **过程**: 2 轮尝试 / 1 次重试反馈，26.6s，11082 token
- **声明的策略**: symbolic → symbolic　*(多轮同标签 = 没换路)*
- **实际用的库**: json、sympy
- **agent 判定**: `proved`　｜**过程分 2.33**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `代码执行出错(TypeError), 没有产生可复核的结果` upper_bound_identity = 3*x**2 + 2*x*y + 3*y**2 - 4 candidate_constraint = 0 candidate_value = sqrt(2) gap_is_square_identity = Fal…

### ➖ `alg-009-determinant-multiplicative`

> 随机生成 100 组 3x3 矩阵，验证 det(AB)=det(A)det(B)。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，5.3s，4983 token
- **声明的策略**: symbolic
- **实际用的库**: random、sympy
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `comb-010-eight-queens`

> 求 8x8 棋盘上 8 皇后问题的解数。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 3 轮尝试 / 2 次重试反馈，23.2s，16157 token
- **声明的策略**: enumerate → solver → enumerate
- **实际用的库**: json、z3
- **agent 判定**: `proved`　｜**过程分 9.83**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `代码执行出错(AssertionError), 没有产生可复核的结果` 
  - 第1轮 · `代码执行出错(AssertionError), 没有产生可复核的结果` 

### ➖ `logic-011-de-morgan`

> 验证德摩根律：not(A and B) = (not A) or (not B)，not(A or B) = (not A) and (not B)。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，6.3s，5178 token
- **声明的策略**: symbolic
- **实际用的库**: sympy
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'tool_quality', 'exploration']
- **判据**: 自判

### ➖ `opt-012-small-lp`

> 最大化 x+y，约束 x+2y<=4，3x+y<=6，x>=0，y>=0。

- **要交什么**（题库契约字段）: `max`、`argmax`
- **过程**: 1 轮尝试 / 0 次重试反馈，8.6s，5212 token
- **声明的策略**: solver
- **实际用的库**: json、sympy
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'tool_quality', 'exploration']
- **判据**: 自判

### ➖ `comb-013-hanoi`

> 验证 n 层汉诺塔最少需要 2^n - 1 步，n<=10。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，9.6s，5732 token
- **声明的策略**: symbolic
- **实际用的库**: json、sympy、z3
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'tool_quality', 'exploration']
- **判据**: 自判

### ➖ `graph-014-ramsey-3-3`

> 验证 R(3,3)=6：6 个顶点的完全图任意红蓝染色必有红色或蓝色三角形；5 个顶点存在反例。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，15.2s，6356 token
- **声明的策略**: solver
- **实际用的库**: itertools、json、z3
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'tool_quality', 'exploration']
- **判据**: 自判

### ➖ `graph-015-labeled-trees-5`

> 求 5 个标号顶点的树有多少棵。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 2 轮尝试 / 1 次重试反馈，21.9s，10879 token
- **声明的策略**: enumerate → symbolic
- **实际用的库**: heapq、itertools、json、networkx、sympy
- **agent 判定**: `proved`　｜**过程分 2.67**　短板 ['correctness', 'tool_quality', 'exploration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `代码被沙箱静态门拒绝: line 60: `networkx` 里没有 `enumerate_trees` 这个属性(A` 

### ➖ `graph-016-petersen-properties`

> 验证 Petersen 图有 10 个顶点、3-正则、直径 2、围长 5。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，7.2s，5019 token
- **声明的策略**: construct
- **实际用的库**: json、networkx
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `prob-017-birthday-paradox`

> 计算 23 人中至少两人生日相同的概率，验证是否大于 1/2。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，5.5s，4939 token
- **声明的策略**: symbolic
- **实际用的库**: fractions、json、sympy
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'tool_quality', 'exploration']
- **判据**: 自判

### ➖ `prob-018-two-dice-expectation`

> 求掷两个公平六面骰子点数和的期望。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，6.6s，4808 token
- **声明的策略**: enumerate
- **实际用的库**: fractions、json
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `alg-019-pythagorean-formula`

> 验证 a=m^2-n^2，b=2mn，c=m^2+n^2 对 m>n>0 满足 a^2+b^2=c^2。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，5.0s，4979 token
- **声明的策略**: symbolic
- **实际用的库**: sympy
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `num-020-bisection-root`

> 用二分法求 x^3-2x-5=0 在 [2,3] 的实根，精度 1e-6。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，6.9s，5442 token
- **声明的策略**: numeric
- **实际用的库**: json、math、sympy
- **agent 判定**: `partial`　｜**过程分 0.0**　短板 ['correctness', 'cross_validation', 'tool_quality']
- **判据**: 自判


---

## V2 套 20 题（`baseproblems_V2.txt`）

> ➖ 同上；这一套按「裸枚举必死」设计，规模题居多


### ➖ `hard-001-pell-61`

> 求 Pell 方程 x^2 - 61 y^2 = 1 的最小正整数解 (x,y)。

- **要交什么**（题库契约字段）: `x`、`y`
- **过程**: 6 轮尝试 / 6 次重试反馈，73.9s，35184 token
- **声明的策略**: symbolic → symbolic → symbolic → symbolic → symbolic → symbolic　*(多轮同标签 = 没换路)*
- **实际用的库**: json、math、sympy
- **agent 判定**: `failed`　｜**过程分 0.0**　短板 ['correctness', 'cross_validation', 'exploration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第2轮 · `代码执行出错(AttributeError), 没有产生可复核的结果` 
  - 第3轮 · `代码执行出错(NameError), 没有产生可复核的结果` Convergents of sqrt(61): First few convergents (h/k form): Smallest solution found: None
  - 第4轮 · `代码被沙箱静态门拒绝: line 77: `sympy` 里没有 `isqrt` 这个属性(AttributeError` 
  - 第5轮 · `代码执行出错(AssertionError), 没有产生可复核的结果` 

### ➖ `hard-002-partition-p100`

> 计算整数分拆数 p(100) 的精确值。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，5.4s，5082 token
- **声明的策略**: symbolic
- **实际用的库**: json
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `hard-003-sum-of-three-cubes-33`

> 求整数 x,y,z 使 x^3 + y^3 + z^3 = 33。

- **要交什么**（题库契约字段）: `x`、`y`、`z`
- **过程**: 1 轮尝试 / 0 次重试反馈，7.1s，4968 token
- **声明的策略**: solver
- **实际用的库**: json、z3
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'tool_quality', 'exploration']
- **判据**: 自判

### ➖ `hard-004-catalan-50`

> 计算第 50 个 Catalan 数 C_50 的精确值。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 2 轮尝试 / 1 次重试反馈，8.4s，9744 token
- **声明的策略**: symbolic → symbolic　*(多轮同标签 = 没换路)*
- **实际用的库**: json、math、sympy
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'tool_quality', 'exploration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `代码被沙箱静态门拒绝: line 11: `sympy.functions.combinatorial.numbers`` 

### ➖ `hard-005-collatz-up-to-1e6`

> 验证 Collatz 猜想对所有 1 <= n <= 10^6 都最终到达 1，并给出最长链的起点和步数。

- **要交什么**（题库契约字段）: `max_start`、`max_steps`
- **过程**: 1 轮尝试 / 0 次重试反馈，6.9s，5220 token
- **声明的策略**: enumerate
- **实际用的库**: json
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `hard-006-mersenne-127`

> 用 Lucas-Lehmer 检验证明 2^127 - 1 是素数。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，6.1s，4976 token
- **声明的策略**: symbolic
- **实际用的库**: json、sympy
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `hard-007-ramsey-3-4`

> 验证 R(3,4)=9：证明任意 9 个顶点的红蓝完全图必有红色三角形或蓝色 K4，并给出 8 个顶点无反例的着色。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 2 轮尝试 / 1 次重试反馈，63.1s，14446 token
- **声明的策略**: solver → solver　*(多轮同标签 = 没换路)*
- **实际用的库**: itertools、json、z3
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `MachineGate: 输出**自相矛盾**: 有的行说某一部分成立, 有的行说另一部分不成立, 而你判了 REFUTED。
  这道题是` 8-vertex coloring found: True 8-vertex coloring INVALID: ('red_triangle', (0, 1, 2)) Checking K_9 universal property... K_9 check …

### ➖ `hard-008-twin-primes-1e7`

> 统计 1 到 10^7 之间的孪生素数对 (p, p+2) 的数量。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，4.8s，4957 token
- **声明的策略**: numeric
- **实际用的库**: （无 import）
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `hard-009-smallest-factorial-over-1e1000`

> 求最小的正整数 n 使 n! > 10^1000。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，10.2s，5464 token
- **声明的策略**: numeric
- **实际用的库**: json、math
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `hard-010-squarefree-1e6`

> 统计 1 到 10^6 中无平方因子数的个数。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，10.6s，5220 token
- **声明的策略**: symbolic
- **实际用的库**: array、json、sympy
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `hard-011-petersen-chromatic-3`

> 证明 Petersen 图的色数是 3：给出 3 着色，并证明不能 2 着色。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，11.9s，5948 token
- **声明的策略**: solver
- **实际用的库**: itertools、json、z3
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `hard-012-erdos-straus-1000`

> 验证 Erdos-Straus 猜想对 2 <= n <= 1000 成立：4/n = 1/x + 1/y + 1/z 有正整数解。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，14.1s，6180 token
- **声明的策略**: enumerate
- **实际用的库**: fractions、json、math
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `hard-013-bbp-pi-hex-digit`

> 用 BBP 公式计算 pi 的十六进制展开中第 10^6 位数字。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，41.3s，8645 token
- **声明的策略**: numeric
- **实际用的库**: decimal、json
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `hard-014-sum-cubes-1000`

> 验证 1^3 + 2^3 + ... + n^3 = (n(n+1)/2)^2 对 n=1000 成立，并给出两边精确大整数。

- **要交什么**（题库契约字段）: `lhs`、`rhs`
- **过程**: 3 轮尝试 / 2 次重试反馈，15.9s，15546 token
- **声明的策略**: symbolic → symbolic → symbolic　*(多轮同标签 = 没换路)*
- **实际用的库**: json、sympy
- **agent 判定**: `proved`　｜**过程分 2.83**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `代码执行出错(NameError), 没有产生可复核的结果` 
  - 第1轮 · `代码执行出错(NameError), 没有产生可复核的结果` 

### ➖ `hard-015-fermat-little-lean`

> 在 Lean 4 + Mathlib 中形式化并证明费马小定理：若 p 为素数且 p 不整除 a，则 a^(p-1) ≡ 1 mod p。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，42.7s，1099 token
- **声明的策略**: (未声明)
- **实际用的库**: Mathlib
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'tool_quality', 'exploration']
- **判据**: 自判

### ➖ `hard-016-sqrt2-irrational-lean`

> 在 Lean 4 + Mathlib 中形式化并证明 sqrt(2) 是无理数。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，8.7s，984 token
- **声明的策略**: (未声明)
- **实际用的库**: Mathlib
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'tool_quality', 'exploration']
- **判据**: 自判

### ➖ `hard-017-100-queens-sat`

> 用 Z3 或 SAT 求解器求 100 皇后问题的一个解，并验证无冲突。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 4 轮尝试 / 3 次重试反馈，146.9s，23307 token
- **声明的策略**: solver → solver → solver → solver　*(多轮同标签 = 没换路)*
- **实际用的库**: json、time、z3
- **agent 判定**: `proved`　｜**过程分 2.56**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `代码执行出错(NameError), 没有产生可复核的结果` Solver returned: unknown in 15.05s
  - 第1轮 · `代码显式标记 VERDICT: INCONCLUSIVE` SOLVER_STATUS: unknown INCONCLUSIVE_REASON: solver returned unknown within the time budget VERDICT: INCONCLUSIVE
  - 第2轮 · `代码显式标记 VERDICT: INCONCLUSIVE` STATUS: unknown VERDICT: INCONCLUSIVE

### ➖ `hard-018-zeta2-pi2-6`

> 数值验证 Basel 问题：sum_{n=1}^{N} 1/n^2 收敛到 pi^2/6，取 N=10^7，误差小于 1e-6。

- **要交什么**（题库契约字段）: `sum`、`pi2_over_6`、`error`
- **过程**: 1 轮尝试 / 0 次重试反馈，8.6s，5593 token
- **声明的策略**: numeric
- **实际用的库**: json、math
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判

### ➖ `hard-019-no-flt-n3-small`

> 验证不存在正整数 a,b,c < 1000 满足 a^3 + b^3 = c^3。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 2 轮尝试 / 1 次重试反馈，442.0s，11116 token
- **声明的策略**: (未声明) → enumerate
- **实际用的库**: json、sympy、z3
- **agent 判定**: `proved`　｜**过程分 5.33**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `执行超时被终止, 没拿到判定结果` 

### ➖ `hard-020-min-n-with-n-divisors`

> 求最小的正整数 n，使 n 恰好有 1000 个正因子。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 2 轮尝试 / 1 次重试反馈，27.4s，11120 token
- **声明的策略**: construct → enumerate
- **实际用的库**: functools、itertools、json、math、sympy
- **agent 判定**: `proved`　｜**过程分 10.67**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 自判
- **逐轮它被告知了什么**:
  - 第0轮 · `代码执行出错(ValueError), 没有产生可复核的结果` Number of exponent partitions found: 0 First few partitions: []


---

## R 套 20 题（`baseproblems_R.txt`）

> ✅/❌ = **独立验证器**判定（自己重算，不读 agent 代码与自由文本）


### ✅ `res-001-ramsey-34-lower-bound`

> 证明 R(3,4) > 8：构造一个 8 个顶点的红蓝完全图，既无红色三角形，也无蓝色 K4，并给出机器可验证的证书。

- **要交什么**（题库契约字段）: `coloring`、`certificate`
- **过程**: 3 轮尝试 / 2 次重试反馈，35.8s，18471 token
- **声明的策略**: construct_and_check → construct_and_check → construct_and_check　*(多轮同标签 = 没换路)*
- **实际用的库**: itertools、json、networkx
- **agent 判定**: `proved`　｜**过程分 60.17**　短板 ['exploration', 'cross_validation', 'self_correction']
- **判据**: 独立
- **验证器判词**: {"claim": "R(3,4) > 8", "n": 8, "red_edges": [[0, 3], [0, 4], [0, 7], [1, 5], [1, 7], [2, 3], [2, 6], [2, 7], [3, 5], [4, 6]], "enumerated_triples": 56, "enumer
- **逐轮它被告知了什么**:
  - 第0轮 · `MachineGate: 输出**自相矛盾**: 有的行说某一部分成立, 有的行说另一部分不成立, 而你判了 REFUTED。
  这道题是` K4): False CONSTRUCT: {"coloring": [["B", "R", "R", "B", "R", "R", "B", "B"], ["B", "B", "R", "R", "B", "R", "R", "B"], ["B", "B",…
  - 第1轮 · `MachineGate: 输出**自相矛盾**: 有的行说某一部分成立, 有的行说另一部分不成立, 而你判了 REFUTED。
  这道题是` , 'red', 'blue', 'blue', 'red', None] CONSTRUCT: {"coloring": [[null, "red", "red", "red", "red", "red", "red", "blue"], ["red", n…

### ❌ `res-002-golomb-ruler-15`

> 构造一个 15 阶 Golomb 尺，长度不超过 200，并给出所有两两距离互不相同的证书。

- **要交什么**（题库契约字段）: `marks`、`length`、`distinct_distances`
- **过程**: 1 轮尝试 / 0 次重试反馈，931.2s，35488 token
- **声明的策略**: construct_and_check
- **实际用的库**: itertools、json
- **agent 判定**: `partial`　｜**过程分 0.0**　短板 ['correctness', 'cross_validation', 'exploration']
- **判据**: 独立

### ❓ `res-003-hadamard-28-paley`

> 构造一个 28 阶 Hadamard 矩阵，并验证 H H^T = 28 I。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 4 轮尝试 / 6 次重试反馈，112.0s，25280 token
- **声明的策略**: construct_and_check → construct_and_check → construct → construct
- **实际用的库**: json、numpy
- **agent 判定**: `None`
- **判据**: none
- **逐轮它被告知了什么**:
  - 第2轮 · `NoCode` 
  - 第3轮 · `NoCode` 
  - 第4轮 · `MachineGate: 输出**自相矛盾**: 有的行说某一部分成立, 有的行说另一部分不成立, 而你判了 REFUTED。
  这道题是` 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1], [1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1,…
  - 第5轮 · `MachineGate: 输出**自相矛盾**: 有的行说某一部分成立, 有的行说另一部分不成立, 而你判了 REFUTED。
  这道题是` 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1], [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1,…

### ❌ `res-004-sos-inequality`

> 证明或证伪：对任意实数 x,y,z，有 x^4 + y^4 + z^4 + xyz >= x^2 y^2 + y^2 z^2 + z^2 x^2 + x + y + z - 3。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 1 次重试反馈，77.2s，8588 token
- **声明的策略**: (未声明)
- **实际用的库**: sympy、z3
- **agent 判定**: `untested`　｜**过程分 0.0**　短板 ['correctness', 'cross_validation', 'certification']
- **判据**: 独立
- **逐轮它被告知了什么**:
  - 第0轮 · `代码执行出错(SympifyError), 没有产生可复核的结果` Searching for counterexample (real x, y, z) ... z3 result: sat Counterexample candidate: x=-2, y=2, z=2

### ❌ `res-005-berlekamp-massey`

> 给定整数序列前 100 项（由某个线性递推生成），求最小线性递推式，并用前 200 项验证。

- **要交什么**（题库契约字段）: `recurrence`、`order`、`verified_200_terms`
- **过程**: 1 轮尝试 / 0 次重试反馈，13.4s，5897 token
- **声明的策略**: symbolic
- **实际用的库**: json、sympy
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 独立

### ❌ `res-006-pslq-integer-relation`

> 在高精度下寻找整数 a,b,c,d，使 a*log(2)+b*log(3)+c*log(5)+d*log(7) ≈ 0，且 (a,b,c,d) 不全为零且 |a|,|b|,|c|,|d| <= 10^6。

- **要交什么**（题库契约字段）: `relation`、`residual`
- **过程**: 2 轮尝试 / 1 次重试反馈，61.0s，14529 token
- **声明的策略**: solver → numeric
- **实际用的库**: decimal、fractions、json、math、numpy、sympy、z3
- **agent 判定**: `partial`　｜**过程分 0.0**　短板 ['correctness', 'cross_validation', 'exploration']
- **判据**: 独立
- **验证器判词**: residual_small: 残差 1.385e-05 > 容差 9.437e-17 (PSLQ 在 |系数|<=1e6 下的最好残差约 9.437e-19)
- **逐轮它被告知了什么**:
  - 第0轮 · `代码被沙箱静态门拒绝: line 114: `sympy` 里没有 `integer_relation` 这个属性(At` 

### ❌ `res-007-oeis-identify-prove`

> 给定一个整数序列前 30 项，识别其 OEIS 条目，提取一个闭式或递推公式，并验证前 100 项一致。

- **要交什么**（题库契约字段）: `oeis_id`、`formula`、`verified_100_terms`
- **过程**: 4 轮尝试 / 5 次重试反馈，160.8s，45028 token
- **声明的策略**: (未声明) → solver → symbolic → symbolic
- **实际用的库**: json、math、sympy
- **agent 判定**: `inconclusive`　｜**过程分 0.0**　短板 ['correctness', 'cross_validation', 'exploration']
- **判据**: 独立
- **验证器判词**: formula_recomputable: 没有可独立复算的公式制品(formula 文本判不了: 'a(n) = n!  (also: a(0)=1, a(n) = n*a(n-1))'); 题库契约字段是 formula(字符串), 例如 "F[n] = F[n-1] + F[n-2]" 或 "F[n] = 3*F
- **逐轮它被告知了什么**:
  - 第1轮 · `NoCode` 
  - 第2轮 · `代码显式标记 VERDICT: INCONCLUSIVE` 26220), (21, 54, 39750), (22, 57, 60060), (23, 60, 90385), (24, 62, 135600), (25, 65, 202750), (26, 68, 302280), (27, 70, 449640),…
  - 第3轮 · `MachineGate: 你判了 REFUTED, 但没有出示**凭证行**, 机器无法复核这个判定。
  必须打这一行(字段名和值可以改,` 4, expected=4717 n=16: alt=2042975, expected=7300 n=17: alt=3124550, expected=11240 n=18: alt=13123110, expected=17205 n=19: alt=2…
  - 第4轮 · `NoCode` 

### ❌ `res-008-lean-binomial`

> 在 Lean 4 + Mathlib 中形式化并证明：sum_{k=0}^{n} C(n,k) = 2^n，不允许 sorry。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 6 轮尝试 / 6 次重试反馈，60.0s，7809 token
- **声明的策略**: (未声明) → (未声明) → (未声明) → (未声明) → (未声明) → (未声明)　*(多轮同标签 = 没换路)*
- **实际用的库**: Mathlib
- **agent 判定**: `inconclusive`　｜**过程分 0.0**　短板 ['correctness', 'cross_validation', 'tool_quality']
- **判据**: 独立
- **验证器判词**: lean_proved: Lean 没判为 proved(exit=1, errors=1, sorry/axiom=0); no_sorry: 证明里用了 sorry / 自带 axiom —— 等于没证
- **逐轮它被告知了什么**:
  - 第2轮 · `Lean 未证成: 1 处编译错误` C:\Users\NIGHTI~1\AppData\Local\Temp\tmptwupf7lc\Check.lean:3:8: error: 'sum_range_pow' has already been declared 'sum_range_pow' …
  - 第3轮 · `Lean 未证成: 2 处编译错误` C:\Users\NIGHTI~1\AppData\Local\Temp\tmpiso228eh\Check.lean:4:8: error: unknown identifier 'sum_range_choose' C:\Users\NIGHTI~1\Ap…
  - 第4轮 · `Lean 未证成: 3 处编译错误、用了 sorry 或 axiom(等于没证)` C:\Users\NIGHTI~1\AppData\Local\Temp\tmpqpja7nsp\Check.lean:4:11: error: function expected at range term has type ?m.13 C:\Users\N…
  - 第5轮 · `Lean 未证成: 1 处编译错误、用了 sorry 或 axiom(等于没证)` C:\Users\NIGHTI~1\AppData\Local\Temp\tmp210u1icy\Check.lean:7:5: error: unknown constant 'Finset.sum_pow_mul_choose_eq' 'sum_range…

### ❌ `res-009-lean-quadratic`

> 在 Lean 4 中形式化并证明二次公式：若 a≠0，则 a x^2 + b x + c = 0 当且仅当 x = (-b ± sqrt(b^2-4ac)) / (2a)（在适当域假设下）。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 6 轮尝试 / 6 次重试反馈，93.3s，15699 token
- **声明的策略**: (未声明) → (未声明) → (未声明) → (未声明) → (未声明) → (未声明)　*(多轮同标签 = 没换路)*
- **实际用的库**: Mathlib
- **agent 判定**: `inconclusive`　｜**过程分 0.0**　短板 ['correctness', 'cross_validation', 'tool_quality']
- **判据**: 独立
- **验证器判词**: lean_proved: Lean 没判为 proved(exit=1, errors=7, sorry/axiom=0); no_sorry: 证明里用了 sorry / 自带 axiom —— 等于没证
- **逐轮它被告知了什么**:
  - 第2轮 · `Lean 未证成: 9 处编译错误、用了 sorry 或 axiom(等于没证)` Users\NIGHTI~1\AppData\Local\Temp\tmpu50qsplv\Check.lean:47:22: error: unknown identifier 'le_of_ge' C:\Users\NIGHTI~1\AppData\Loc…
  - 第3轮 · `Lean 未证成: 6 处编译错误、用了 sorry 或 axiom(等于没证)` f the pattern in the target expression ?a ^ 2 = ?b ^ 2 case mp a b c x : ℝ ha : a ≠ 0 h : b ^ 2 - 4 * a * c ≥ 0 hx : a * x ^ 2 + b…
  - 第4轮 · `Lean 未证成: 5 处编译错误、用了 sorry 或 axiom(等于没证)` heck.lean:40:28: error: tactic 'rewrite' failed, equality or iff proof expected ?m.12594 case mpr.inl a b c x : ℝ ha : a ≠ 0 hdisc…
  - 第5轮 · `Lean 未证成: 7 处编译错误、用了 sorry 或 axiom(等于没证)` b c : ℝ ha : a ≠ 0 x : ℝ h : x = (-b + √(b ^ 2 - 4 * a * c)) / (2 * a) ⊢ a * (-b + √(b ^ 2 - 4 * a * c)) ^ 2 * (2 * a) + b * (-b +…

### ❌ `res-010-sprague-grundy`

> 对给定的一种取石子游戏（规则见 data/res_10_rules.txt），计算前 1000 个状态的 Sprague-Grundy 值，给出周期规律并用两套独立实现交叉验证。

- **要交什么**（题库契约字段）: `sg_values`、`period`、`verified_by_second_impl`
- **过程**: 1 轮尝试 / 0 次重试反馈，17.0s，5769 token
- **声明的策略**: (未声明)
- **实际用的库**: functools、json
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 独立
- **验证器判词**: sg_values_match: 第 2 项不符: 报 2, 自算 0; period_is_real: p=4 在 onset=0 上不是合法周期(g[n] != g[n+4] 至少有一处); period_is_minimal: 报的周期 4 不是最小周期(自算最小为 7); 周期规律必须是最小的那一个

### ❌ `res-011-sudoku-diagonal-z3`

> 用 Z3 求解一个对角数独（两条主对角线也含 1-9），并证明解唯一。

- **要交什么**（题库契约字段）: `solution`、`unique`
- **过程**: 6 轮尝试 / 6 次重试反馈，76.4s，36291 token
- **声明的策略**: solver → solver → solver → solver → solver → solver　*(多轮同标签 = 没换路)*
- **实际用的库**: itertools、json、z3
- **agent 判定**: `inconclusive`　｜**过程分 0.0**　短板 ['correctness', 'cross_validation', 'certification']
- **判据**: 独立
- **验证器判词**: FAILED: artifact_present: 未提交 CONSTRUCT 制品
- **逐轮它被告知了什么**:
  - 第2轮 · `代码执行出错(AssertionError), 没有产生可复核的结果` 
  - 第3轮 · `代码执行出错(AssertionError), 没有产生可复核的结果` 
  - 第4轮 · `代码执行出错(TypeError), 没有产生可复核的结果` 
  - 第5轮 · `代码执行出错(AssertionError), 没有产生可复核的结果` 

### ✅ `res-012-minimal-polynomial`

> 求 alpha = sqrt(2) + sqrt(3) + sqrt(5) 在 Q 上的极小多项式，并验证其次数和不可约性。

- **要交什么**（题库契约字段）: `polynomial`、`degree`、`irreducible`
- **过程**: 1 轮尝试 / 0 次重试反馈，13.7s，5408 token
- **声明的策略**: symbolic
- **实际用的库**: json、sympy
- **agent 判定**: `proved`　｜**过程分 36.0**　短板 ['exploration', 'method_diversity', 'cross_validation']
- **判据**: 独立
- **验证器判词**: p(alpha)==0(精确), deg=8, irreducible=True, coeffs=[Fraction(576, 1), Fraction(0, 1), Fraction(-960, 1), Fraction(0, 1), Fraction(352, 1), Fraction(0, 1), Fractio

### ❌ `res-013-groebner-system`

> 在有限域 F_101 上求解多项式方程组（见 data/res_13_system.txt），给出所有解并验证。

- **要交什么**（题库契约字段）: `solutions`、`verified`
- **过程**: 1 轮尝试 / 0 次重试反馈，4.1s，5058 token
- **声明的策略**: solver
- **实际用的库**: itertools、json
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 独立
- **验证器判词**: FAILED: solutions_wellformed: 解的格式不对: [8, 24, 97]

### ❓ `res-014-rubiks-6-moves`

> 给定一个 6 步内可解的三阶魔方打乱状态，用 IDA* 或双向 BFS 找到最短解，并验证步数最少。

- **要交什么**（题库契约字段）: `moves`、`length`、`optimal`
- **过程**: 3 轮尝试 / 4 次重试反馈，126.0s，41599 token
- **声明的策略**: solver → solver → solver　*(多轮同标签 = 没换路)*
- **实际用的库**: collections、functools、json
- **agent 判定**: `None`
- **判据**: none
- **逐轮它被告知了什么**:
  - 第0轮 · `NoCode` 
  - 第1轮 · `代码执行出错(ValueError), 没有产生可复核的结果` 
  - 第2轮 · `代码执行出错(AssertionError), 没有产生可复核的结果` 
  - 第3轮 · `NoCode` 

### ❌ `res-015-cyclic-inequality-counterexample`

> 检验猜想：对任意正实数 a,b,c，有 a/(b+c) + b/(c+a) + c/(a+b) >= 3/2。若成立给证明，若不成立给反例。

- **要交什么**（题库契约字段）: `verdict`、`proof_method`
- **过程**: 3 轮尝试 / 2 次重试反馈，58.2s，20491 token
- **声明的策略**: symbolic → symbolic → symbolic　*(多轮同标签 = 没换路)*
- **实际用的库**: random、sympy、z3
- **agent 判定**: `proved`　｜**过程分 0.0**　短板 ['correctness', 'certification', 'exploration']
- **判据**: 独立
- **验证器判词**: FAILED: artifact_present: 既没给证明制品也没给反例制品
- **逐轮它被告知了什么**:
  - 第0轮 · `代码执行出错(Z3Exception), 没有产生可复核的结果` c)**2*(a - b - c))/(2*(a + b)*(a + c)*(b + c)) numerator expanded = 2*a**3 - 2*a**2*b - 2*a**2*c - 2*a*b**2 + 6*a*b*c - 2*a*c**2 +…
  - 第1轮 · `代码执行出错(NameError), 没有产生可复核的结果` Combined expression: (2*a*(a + b)*(a + c) + 2*b*(a + b)*(b + c) + 2*c*(a + c)*(b + c) - 3*(a + b)*(a + c)*(b + c))/(2*(a + b)*(a +…

### ❌ `res-016-tsp-heuristic-500`

> 对给定 500 城市 TSP 实例，找到一个长度不超过已知最优解 1.05 倍的可行巡回。

- **要交什么**（题库契约字段）: `tour`、`length`
- **过程**: 1 轮尝试 / 0 次重试反馈，75.9s，5824 token
- **声明的策略**: construct
- **实际用的库**: json、math、random
- **agent 判定**: `partial`　｜**过程分 0.0**　短板 ['correctness', 'cross_validation', 'exploration']
- **判据**: 独立
- **验证器判词**: tour_present: 制品里没有 tour 列表

### ❌ `res-017-counterexample-hunt`

> 对猜想：'对任意 n>=2，n^2+n+1 无平方因子'，搜索 1e6 以内反例。若找到给出反例，若没找到给出启发式理由和独立验证。

- **要交什么**（题库契约字段）: `verdict`、`counterexample`
- **过程**: 2 轮尝试 / 1 次重试反馈，143.2s，11126 token
- **声明的策略**: enumerate → (未声明)
- **实际用的库**: json、random、sympy
- **agent 判定**: `refuted`　｜**过程分 5.33**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 独立
- **验证器判词**: artifact_present: 未提交 COUNTEREXAMPLE/ANSWER 制品行
- **逐轮它被告知了什么**:
  - 第0轮 · `MachineGate: 你的输出**自相矛盾**: 一边说「范围内未找到 / 没找到反例」, 一边又判了 REFUTED。
  REFUT` Search range: n = 2 .. 1000000 Counterexample found in [2, LIMIT]? True Counterexample in [2, 1000] (sympy.factorint cross-check)?…

### ✅ `res-018-formal-fermat-small`

> 在 Lean 4 中形式化并证明：对素数 p=5，2^5 ≡ 2 (mod 5)，并对 p=7 同样验证。用 ZMod 表达。

- **要交什么**（题库契约字段）: （只要判定，不交对象）
- **过程**: 1 轮尝试 / 0 次重试反馈，10.9s，1169 token
- **声明的策略**: (未声明)
- **实际用的库**: Mathlib
- **agent 判定**: `proved`　｜**过程分 32.0**　短板 ['tool_quality', 'exploration', 'method_diversity']
- **判据**: 独立
- **验证器判词**: Lean 4 + Mathlib 编译通过且无 sorry/无自带 axiom; 独立复核 2^5 mod 5 == 2、2^7 mod 7 == 2 均成立

### ❌ `res-019-group-construction-60`

> 构造一个 60 阶非交换群，给出其乘法表和生成元，并验证群公理、非交换性、以及是否为 A5。

- **要交什么**（题库契约字段）: `group`、`is_nonabelian`、`is_A5`、`generators`
- **过程**: 4 轮尝试 / 3 次重试反馈，49.1s，26969 token
- **声明的策略**: construct → construct → construct → construct　*(多轮同标签 = 没换路)*
- **实际用的库**: collections、itertools、json、numpy、random
- **agent 判定**: `proved`　｜**过程分 1.29**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 独立
- **验证器判词**: artifact_present: 未提交 CONSTRUCT 制品(需要 group / generators / is_A5)
- **逐轮它被告知了什么**:
  - 第0轮 · `代码执行出错(NameError), 没有产生可复核的结果` 
  - 第1轮 · `MachineGate: 输出**自相矛盾**: 有的行说某一部分成立, 有的行说另一部分不成立, 而你判了 REFUTED。
  这道题是` 1], [3, 2, 1, 0, 4], [3, 2, 4, 1, 0], [3, 4, 0, 1, 2], [3, 4, 1, 2, 0], [3, 4, 2, 0, 1], [4, 0, 1, 2, 3], [4, 0, 2, 3, 1], [4, 0, …
  - 第2轮 · `代码执行出错(KeyError), 没有产生可复核的结果` A5 阶 = 60 闭合: True 结合律(抽样): True 单位元存在(idx=0): True 逆元存在: True 非交换性: True, 例子: ((0, 1, 3, 4, 2), (0, 2, 1, 4, 3))

### ❌ `res-020-hypothesis-refine-loop`

> 给定一个可能为假的猜想（见 data/res_20_conjecture.txt）。任务：先搜反例；若有反例，修改猜想为最小修正版并给出反例边界；若在小范围无反例，给出部分证明或独立数值证据。

- **要交什么**（题库契约字段）: `verdict`、`evidence`、`refined_conjecture`
- **过程**: 3 轮尝试 / 2 次重试反馈，51.8s，19939 token
- **声明的策略**: (未声明) → symbolic → (未声明)
- **实际用的库**: itertools、json、math、sympy
- **agent 判定**: `refuted`　｜**过程分 1.67**　短板 ['correctness', 'exploration', 'calibration']
- **判据**: 独立
- **验证器判词**: artifact_present: 未提交 COUNTEREXAMPLE/ANSWER 制品
- **逐轮它被告知了什么**:
  - 第0轮 · `MachineGate: 输出**自相矛盾**: 有的行说某一部分成立, 有的行说另一部分不成立, 而你判了 REFUTED。
  这道题是` subset T satisfies max(S1)+max(S2) > max(T) for ALL partitions of A\\T into two nonempty parts. Equivalently: for every T, the par…
  - 第1轮 · `代码执行出错(NameError), 没有产生可复核的结果` 
