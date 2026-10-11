# 参考答案复算闸门 —— 题库可信度报告

> 面向: 正在平行重构 `agent/tools/research` 流程的那位
> 产出: mavis, 2026-10-09
> 配套代码: `tests/tools/reference_recompute.py` / `check_reference_truth.py`
> 明细: `data/research/_reference_truth_report.json`

---

## 0. 一句话

**新题库 `benchmark/` 里 DS 给的 60 道题, 有 7 道 `ground_truth` 是错的。**
GT 错 -> verifier 稳定判错 -> 分数看起来完全正常 -> 从分数里读不出来。
所以我加了一道**题库入库闸门**, 把能自己修的都修了。

### 现在的状态(修完之后)

```
状态分布: PASS=49, UNVERIFIABLE=11   (FAIL=0  ERROR=0  NEGCTL_FAIL=0)
负控: 32 题翻脸成功 / 0 题漏网
闸门退出码 2 —— 只剩"没验", 没有"验出错的"
```

**11 道"没验"里, 7 道是设计上就不该有标量参考答案**(见 §5), 4 道需要 DS 定夺。

---

## 1. 闸门怎么用

```powershell
$env:QAGENT_COST_GUARD='off'    # 否则配额耗尽返回错误串, 静默空转
$env:PYTHONIOENCODING='utf-8'   # Windows 控制台是 GBK, 印不出外来字符
python tests/tools/check_reference_truth.py    # 闸门本体
python tests/tools/harness_gate.py             # L0 = 本闸门, 排在 L1-L3 前
python tests/tools/harness_gate.py --skip-l0   # 只跑旧的 L1-L3
```

退出码: `0` = 可评测; `1` = 有 FAIL(题库必须改); `2` = 只有 UNVERIFIABLE(需人工补)。

**当前退出码 = 2**(49 PASS / 11 UNVERIFIABLE / 0 FAIL)。

**接进 harness_gate 的方式**: `l0_reference_truth()` 用**子进程**调闸门。
不用 import —— L0 一坏不该把整道门带崩, **门的失败必须是结论而不是崩溃**。

### 复算流程(可重跑, 种子固定)

```powershell
python tests/tools/import_new_banks.py           # DS 的两个附件 -> benchmark/
python tests/tools/make_missing_data.py          # 补齐题库引用但缺失的数据文件
python tests/tools/apply_reference_corrections.py  # 按复算结果订正 GT(可重入)
python tests/tools/check_reference_truth.py      # 闸门
```

---

## 2. 闸门自己的两条纪律

### 2.1 复算必须绕开被测工具

用 OR-Tools 复算 ortools 题 = 用错的答案验错的答案, 恒等于真。所以每条复算
路径都是另一样东西:

| 被测工具 | 复算走的是 |
|---|---|
| ortools_cp_sat | 纯 Python 网格穷举 / CP-SAT 只在"找见证"时用 |
| z3_solve | 纯算术消元 / 可行域顶点枚举 / 回溯 |
| cvxpy_solve | 配方 / 手算 2x2 PSD 条件 / 解析顶点 |
| scipy_numerical | mpmath / 解析闭式(与 Simpson 双路互验) |
| mpmath_* | Machin 公式(自算 π)、Euler–Maclaurin(自算 ζ(3))、符号翻转二分 |
| qutip_quantum | 纯 numpy |
| networkx_analyze | 手写 BFS / 暴力删点删边求最小割 |

### 2.2 闸门自带负控

把每道 PASS 题的 GT 数值字段扰动一遍再跑同一个 fn, 要求至少一道 check 从
ok 翻成 not-ok。**翻不动 = 这道题的判定逻辑是恒真的, 门在这里没干活。**

**负控抓到了门自己的两个洞**:`res-02` 和 `res-10` 只校验构造、不绑定
`gt.exists`, 扰动答案时纹丝不动。补上绑定检查后:
**22 题翻脸成功 / 0 题漏网**。

没有负控的门比没有门更危险 —— 它只会输出 PASS。

---

## 3. 参考答案确认是错的(7 道, 已按复算值订正)

**改动全部落在 `benchmark/*.json` 里, 并且每条都写进题库顶层的 `_corrections`
—— 记了原值、新值、复算路径、出错原因。可以直接对着
`tests/tools/reference_recompute.py` 核, 不需要信我。**
订正脚本 `tests/tools/apply_reference_corrections.py` 可重跑(按 id 覆盖, 不累积)。

| 题 | GT 原值 | 复算真值 | 复算方法 |
|---|---|---|---|
| `z3-01-lin-int` | 8 组解 | **7 组** | 纯 Python 双层 for。x=-5 -> y=11, 违反 \|y\|≤10 |
| `ortools-01-int-prog` | max=16 @(4,2) | **17 @(3,4)** | 纯网格穷举。(3,4) 可行: 10≤10, 15≤15 |
| `cvxpy-02-qp` | min=1.0 | **0.0** | 配方 f=(x-1)²+(y-2)², (1,2) 可行 |
| `cvxpy-04-sos` | verdict=true | **false**, 全局最小 **-0.43036** | 801² 网格 + 坐标下降。f(0.5,0.5)=-0.125 |
| `scipy-04` | 2.4053 | **0.2300428481** | 闭式 (1/2)√(π/a)e^{-b²/4a} + 60 万段 Simpson |
| `res-03-sylvester` | X 四分量全错 | [[0.21547619, -0.04166667],[-0.01785714, 0.125]] | Fraction 精确消元, 残差 0 |
| `res-20-cross-method` | 4.4605, 2.5198, 2.0197 | **1.2679492, 3.0, 4.7320508** | Faddeev–LeVerrier + numpy |

### ★ 两条最值得记的经验 ★

**① `res-20` 是"半对"的 GT**: 它给的三个数**迹 9.0 恰好对, 但积 22.70 ≠ det(M)=18**。
只核迹的 verifier 会放它过去。**任何"特征值/谱"类参考答案, 至少要再核一个
不变量(积、和的平方、或者代回原式)。**

**② `cvxpy-04` 的题面预设了假命题**: 题面写"验证 x^4+y^4+1-x\*y-x-y ≥ 0 成立",
而它是**假的**。我改了题面(去掉预设, 改成"判定是否成立, 成立给证书否则给反例"),
GT 改成 false + 反例点。
**注意: 如果不改题面只改 GT, 这题就变成"模型必须推翻题面前提"——
按 Part 1 的定义(题面明示工具、测"用工具的能力")这其实是道好陷阱题;
但题面预设假命题本身是缺陷, 会让人误以为模型答错了就是模型不会。**

---

## 4. 已修好的东西

### 4.1 数据文件(3 道题从"不可解"变成"可跑")

| 文件 | 状态 |
|---|---|
| `data/jss_04.json` | **新建**。3 作业×3 工序 JSS, 随机搜出并用 CP-SAT 确认**最优 makespan 恰为 11**, 与 GT 自洽 |
| `data/nx_04_graph.gml` | **新建**。20 顶点双社区, 模块度 **0.4 上下**(不是卡在 GT 的 0.3 阈值上), louvain 稳定得 2 块 |
| `data/tsp_200.txt` | **新建**。200 城市欧氏实例(seed 424242) |
| `data/tsp_200_opt.txt` | **新建**。最近邻 + 2-opt + Or-opt 得到的最好巡回 |

生成脚本: `tests/tools/make_missing_data.py`(可重跑, 种子固定)。

⚠️ **`tsp_200_opt.txt` 是"我们搜到的最好巡回", 不是证明过的最优解。**
所以"≤ 1.10 × reference"这个判据比"≤ 1.10 × 真最优"**略严**, 对被测模型不利。
如果 DS 那边有真最优值, 换掉这个文件即可。

### 4.2 依赖

- `qutip 5.3.1` **已装**(Part1 四道 qutip 题之前跑不了)。
  注意: 那四道题的**参考答案**本来就能用纯 numpy 验, 缺的只是跑题环境。

### 4.3 判据改写(3 道, 从"不是良定义标量"改成行为性判据)

`res-13` / `res-18` / `res-19` 的原 GT 都是**单个标量**(临界 r、p_c、周期窗口位置),
但这些量依赖扫描分辨率、瞬态丢弃长度、随机种子 —— 题面没定死, 所以**不是良定义标量**。

我**没有**替 DS 定一套物理协议(那是重设计题), 而是改成**行为性判据**:

```json
{
  "criterion_type": "behavioral",
  "must_report": ["lambda_max(r) 曲线", "Feigenbaum 累积点 r_inf, 判据 |r_inf-3.5699|<5e-3",
                  "显式声明瞬态丢弃长度与后继迭代长度", "显式声明判混沌用的判据"],
  "feasible_checks": ["curve_present", "r_inf_in_[3.565,3.575]", "protocol_declared"],
  "not_checkable_as_scalar": "chaotic r 的精确集合"
}
```

**这跟 Part 2 本来就该有的判据形态是一致的** —— 那六道(混沌/量子/渗流)
如果继续拿文献标量当 GT, 背答案就过。

### 4.4 自洽性修正(1 道)

`res-11` 原 GT 列 3 个同构类还带"等/其他"占位, 而 60 阶群实际有 13 个,
**GT 自身不自洽**。改成混合判据: 可验证事实(唯一的 60 阶非交换单群是 A5 +
不变量必须能把各类分开)+ 显式标注"完整清单需 GAP/Sage, 本环境验不了"。

### 4.5 补齐可核对字段(1 道)

`res-09-tsp-200` 的 GT 里只有 `ratio_tolerance`, **没有任何参考长度**,
于是判据阈值无处落脚(负控也抓不到)。补上 `reference_length` /
`length_threshold` / `reference_is_proven_optimal: false` /
`reference_method`。**"≤1.10 × reference" 因此比"≤1.10 × 真最优"略严**,
对被测模型不利, 这一点写进了 GT 本身。

---

## 5. 我修不了、需要 DS 确认的(11 道 UNVERIFIABLE)

**这 11 道没有一道是"我还没查"** —— 全都是"按现有信息查不了":

### 5a. 设计上就不该有标量参考答案(7 道)—— 已改成行为性判据

`res-11` / `res-13` / `res-18` / `res-19` 我已经改写成判据形态(见 §4.3/4.4),
复算层按设计返回 UNVERIFIABLE。剩下这三道是我改不动、也不该我改的:

| 题 | 为什么查不了 |
|---|---|
| `lean-01..04` | GT 只有 statement/mathlib_hint, **不是可复算的数值答案**。Lean 4.9.0 在 `D:\pythonProject\Others\mathlib4-v490`, 但要真跑 Lean 编译才算验过, 本闸门不做 |
| `oeis-03` | 要求"列出至少 3 个匹配序列" —— **答案是开放集合**, 没有对错可判 |
| `res-14` | Lorenz Lyapunov 谱依赖积分精度与瞬态长度, 复算成本高且与实现强相关 |
| `res-16` | GHZ vs W 的 SLOCC 不等价性, 见下 |

### 5b. `res-16` 值得单独说

要分开 GHZ 和 W 需要 3-tangle。我试了四个不变量**全部不成立**:

| 不变量 | GHZ | W | 结论 |
|---|---|---|---|
| rank(ρ_A) 的 Schmidt 秩 | 2 | 2 | 分不开 |
| rank(ρ_AB) | 2 | 2 | 分不开 |
| Pauli 反对称化算符 A^Γ | 直接得 0 | 0 | A^Γ 同时湮灭 \|000⟩ 和 \|111⟩, 拿它当 3-tangle 会得出"GHZ 的 3-tangle=0" |
| 跨 A\|BC 的 negativity | log2=1 | log2=1 | 分不开 |

3-tangle 在文献里有多种算符写法与归一化(CKW 给 GHZ 3/4, 另一些归一到 1),
差一个常数就翻盘。**我拒绝再猜一个约定**, 已如实标成 UNVERIFIABLE。

> **如果你要接这道题: 请先固定一个 3-tangle 定义并给出参考实现, 或者换一道
> 不依赖 3-tangle 的判据题。**

### 5c. 已验证但不做的事(2 道)

- `ortools-03-golomb`: GT 的 `optimal_length=55` 我只验了"存在长度 55 的见证"
  (两两差 105 个互异), **最优性没复算** —— 复算层不做搜索证明
- `res-10-golomb-15`: 找到了长度 ≤200 的见证(CP-SAT), GT 的
  `optimal_length=151` 同样**没复算**

---

## 6. 给流程重构那边的 5 条提醒

1. **判据必须行为性**。Part 2 拿文献标量当 GT, 模型背答案就过。
   上面 4.3 的 `must_report` / `feasible_checks` 形态可以直接用。

2. **GT 与题面必须自洽**。`cvxpy-04` 题面预设假命题、`res-11` GT 自相矛盾,
   都是"题库自己跟自己打架"。**题面和 GT 要一起复核, 不能只看 GT。**

3. **`unknown` ≠ `unsat`**。Part 1 缺 L3「陷阱」层 —— z3 把 `unknown` 当 `unsat`、
   set 迭代顺序导致 `break`、版本敏感 API(`Nat.add_pow`) —— 现在完全空缺,
   而它测的正是"模型用工具时会犯什么错", 价值最高。

4. **环境通道要在跑之前确认**。`loogle_search` **在仓库里根本不存在**
   (`agent/tools/research/` 下只有 `research_oeis`), 而 `lean-03` 的 note
   明写"考察能否用 loogle 检索 Mathlib"。`research_oeis` 我也没在
   `agent/tools/research/` 找到 —— DS 假设的工具通道和仓库现状对不上。
   闸门的环境探针会把这类缺失直接列出来。

5. **题库和 verifier 冻结**。新架构定型前不要动 `benchmark/` 和
   `reference_recompute.py`, 否则新旧分数不可比。架构版本标签已加:
   `QAGENT_RESEARCH_ARCH` 环境变量 → 每条证据的 `arch_version` 字段。

---

## 7. 我自己写错的地方(15 个, 全部已修)

这一节留着是因为**"验证器自己有 bug"比 GT 错更隐蔽**, 而且这一轮我踩了 15 个:

### 7a. 复算路径算错(会给出错误结论的)

1. **Ramsey 回溯把一个顶点的所有边设成同一颜色** —— 每步 2 种选择而不是 2^v 种,
   于是只搜到全同染色的子空间, 真实的 C5 构造一个都搜不到,
   **会误报 "R(3,4)>8 不成立"**
2. **Euler–Maclaurin 有限和到 N 而不是 N-1** —— 多算 N⁻³ = 1.5625e-8,
   结果对到小数点后 7 位, 肉眼扫一眼像对的
3. **Erdős–Straus 用了多余的整除条件** —— n=5 有解 1/2+1/4+1/20 = 4/5,
   但整除条件会把它漏掉, **误报"有反例"**
4. **拓扑计数只查并封闭** —— 得 2271, 加交集闭合才是 355。
   2271 和 355 "差不多同一个量级", 很容易被当成四舍五入放过
5. **特征值用"扫符号翻转"找根** —— 本题 λ=3 是重根, 变号法整个漏掉,
   看上去像矩阵只有 2 个特征值
6. **CPTP 的 Kraus 手猜成 K=diag(√.7,√.3)** —— 不保迹。改用 X 基测量+制备
7. **OEIS 三道全部差一位下标**(题面序列从第 1 项起, 闭式从第 0 项起)
8. **Chudnovsky 求 π 的递推式写错** —— 换成 Machin 公式 arctan 级数
9. **JSS 机器冲突约束用"每次 new BoolVar() 的 OnlyEnforceIf"** —— 恒真的
   守卫变量让约束看着写了其实没约束, **makespan 被算小, 于是"随机搜找不到
   makespan=11"看起来像不可能, 其实是尺子坏了**
10. **JSS 复算把 9 个工序写成 6 个** —— 枚举 720 种先后(而不是 9!=362880),
    算出的 makespan=8 **小于下界 11**, 荒谬值暴露了枚举量不够
11. **`nx.read_gml` 返回字符串节点 id** —— 用 int 索引直接 KeyError: 0

### 7b. 负控抓到的洞(判定逻辑对 GT 不敏感)

12. **`res-02` / `res-10` 只校验构造、不绑定 `gt.exists`** —— 扰动 GT 时纹丝不动
13. **`res-20` 不读新加的 `invariant_checks`** —— 同上
14. **`cvxpy-04` 不读 `value_at_counterexample` / `global_min` / `argmin`** —— 同上
15. **`res-09` 的 GT 里根本没有参考长度字段** —— 判据阈值无处落脚,
    这是**题库字段缺失**, 补了 `reference_length` + `length_threshold` 才修好

**共同点: 都是"看起来算出来了"但其实算错了, 或者"输出了 PASS 但其实什么都没查"。**
跟 `_provenance.py` / `_BLOCKED_MARKERS` 那两次是同一族 ——
**工具报的数字不能直接信, 得有第二条独立路径或者负控盯着。**

### 7c. 还有一个更阴的:审计轨迹是垃圾但改动生效了

`apply_reference_corrections.py` 里 `log.append((cid, "OK", entry["kind"]))`
误写成 append `kind` 而不是 `entry`。结果: **GT/prompt/verifier 的改动照常生效,
但 `_corrections` 里存的是 `'数值订正'` 这种字符串, 而且不报错。**
我是在验证审计轨迹时才发现的 —— 如果不验证, 这份"可追溯记录"就是假的,
下一个人会照着它去核, 核不出来。

> **教训: 生成审计记录的代码, 它的输出本身要被检查。**
> 一条 assert `isinstance(entry, dict)` 就够, 但更正确的是**定期把审计记录
> 拿回来读一遍**。

---

## 8. 已知还没解决的

- `z3-03-optimize` 和 `cvxpy-01-lp` **是同一道 LP**(两库复算值一致:
  14/5 @ (8/5, 6/5))。Part 1 号称 10 工具×4 题, 实际只有 **9 个独立问题**。
- `expected_behaviors` 字段(Part 2 二十道全有)**没有任何代码消费**。
- `loogle_search` / `research_oeis` / `research_tools` 三个模块在
  `agent/tools/research/` 下**都找不到**。DS 的题面假设了这些通道存在。
- Part 2 题库与 verifier 继续冻结, 等架构定型后再定。

---

## 9. 交付物清单

| 路径 | 说明 |
|---|---|
| `tests/tools/reference_recompute.py` | 60 题的独立复算路径 |
| `tests/tools/check_reference_truth.py` | 闸门本体(含负控 + 环境探针) |
| `tests/tools/import_new_banks.py` | DS 附件 -> `benchmark/`(含 JSON 语法错修复) |
| `tests/tools/make_missing_data.py` | 补齐 4 个缺失数据文件(种子固定, 可重跑) |
| `tests/tools/apply_reference_corrections.py` | 按复算结果订正 GT(可重入) |
| `tests/tools/harness_gate.py` | 加了 L0 层 |
| `benchmark/part1_tool_capability.json` | 40 题 + `_corrections` 审计轨迹 |
| `benchmark/part2_research_capability.json` | 20 题 + `_corrections` 审计轨迹 |
| `data/jss_04.json` / `nx_04_graph.gml` / `tsp_200.txt` / `tsp_200_opt.txt` | 新建的数据文件 |
| `data/research/_reference_truth_report.json` | 逐题复算明细 |
| `data/research/_reference_truth_findings.md` | 复算报告(同一批结论的另一份) |