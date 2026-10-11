# 科研验证闭环 —— 工作环境与当前问题

> 写给下一个接手的人（含 DeepSeek）看。**先读第 3 节「当前问题」**，那里是这个项目
> 真正卡住的地方。第 1 节是环境事实，第 2 节是架构现状。
>
> 最后更新：2026-10-09 12:30 | 分支 `research-closed-loop` | HEAD `8480fa0` + 大量未提交改动

---

## 1. 工作环境

### 机器 / 运行时

| 项 | 值 |
|---|---|
| OS | Windows 10/11 (`Windows-AMD64`) |
| Python | **3.14.7**（很新，部分科学计算包没有预编译 wheel） |
| CPU / 内存 | 20 核 / 63.8 GB |
| Python 解释器 | `C:\Python314\python.exe` |
| 项目根 | `D:\pythonProject\QAgent` |

> ★ Python 3.14 是这个项目的一个持续性麻烦：不少科学计算库还没出 wheel，
> 只能源码编译，而本机**没有 MSVC**（见下）。新加包前先确认有 wheel。

### 已装的关键依赖

```
numpy 2.5.2   sympy 1.14.0   z3 5.1.0   networkx 3.6.1
scipy 1.18.0  mpmath 1.3.0    ortools 9.15.6755   python-sat 1.9   cvxpy 1.9.3
```

**装不上的**：

| 包 | 原因 | 处置 |
|---|---|---|
| `kociemba` | 编译要 MSVC，本机没有 | 自写纯 Python 三阶魔方（更符合"独立第二实现"的要求） |
| `sage` / `gap` | 体量/依赖不现实 | 用 sympy + 自写置换群构造替代 |
| `galois` / `fpylll` | 3.14 无 wheel | 自写 Berlekamp-Massey（~20 行）/ 用 mpmath 的 PSLQ |

> `pip install` 要分批装：**一个包装不上会中断整批**（踩过：`kociemba` 失败导致
> `ortools`/`python-sat`/`cvxpy` 全没装上）。

### 外部依赖（不在本仓库）

| 资源 | 路径 | 说明 |
|---|---|---|
| **Mathlib** | `D:\pythonProject\Others\mathlib4-v490` | Lean 4.9.0 的形式化库，形式化题必需 |
| leantar | `%USERPROFILE%\.cache\mathlib\` | 装 Mathlib 时手动放的，lake 的下载器走 curl 直连 |

**Mathlib 的三个坑**（装它花了一整晚）：

1. 必须用 tag **`v4.9.0`**（最新 v4.34.1 要 Lean 4.19+）
2. `lake update` 会把依赖解析到默认分支、破坏 manifest → 必须按钉死的 rev checkout
3. leantar 走 curl 直连 GitHub，git 代理无效，需手动下 `leantar-0.1.13.exe.zip` 解出同名 exe
4. **旧的 `D:\pythonProject\Others\mathlib4` 已经坏了**（根仓库 HEAD 变成了 Cli 的 commit），不要再用

### 仓库里的三套题库

| 文件 | 题数 | 有独立验证器 | 判据来源 |
|---|---|---|---|
| `baseproblems.txt` | 20 | ❌ | 仅 agent 自述 |
| `baseproblems_V2.txt` | 20 | ❌ | 仅 agent 自述 |
| `baseproblems_R.txt` | 20 | ✅ `tests/tools/verifiers_r/` | 独立验证器 |

打分器：`RewardScorer.txt`（原始文件，代码+散文混在一起）→
`tests/tools/reward_scorer.py`（代码）+ `data/research/reward_scorer_notes.md`（散文）。

### 题库引用的数据文件

题库有 5 道题引用了 `data/` 下本该存在、实际缺失的输入文件，我方补齐了
（`tests/tools/build_res_data.py` / `build_res_data2.py` 生成）：

```
res_05_sequence.txt  res_07_sequence.txt  res_10_rules.txt
res_11_sudoku.txt    res_13_system.txt    res_14_rubiks.txt
res_16_tsp.txt       res_20_conjecture.txt
```

每个都有配套的 `*_reference.json`（**参考答案，绝不能给模型看到**）。

> ★ **更正：这三个短名文件不能删**（2026-10-10 核实过，之前的"是垃圾"是错的）★
>
> `res_05_seq.txt` / `res_07_seq.txt` / `res_14_cube.txt` 与对应的 `_sequence` / `_rubiks`
> 版本字节相同，但**短名才是题库 `baseproblems_R.txt` 的 `verifier.params` 指向的名字**，
> 而且 `res_07_oeis.py` 在运行时**真的去读** `data/res_07_seq.txt`。
> 删掉它不会报错，只会让验证器静默退回内置实例 —— 又是一例"长得像别的故障的静默降级"。
>
> 顺带查清了一件更要紧的事：**题面内联的数据与验证器判定基准是同一个**
> （`python tests/tools/check_input_vs_reference.py`，res-005/007/014 三项全过）。
> 这不是废话 —— res-005/014 的验证器是自建实例，若数据后来被重生成而没人核对，
> 验证器就会判一道跟题面无关的题，agent 交对了也判错，**表现和"模型不会"一模一样**。

---

## 2. 架构现状

### 模块地图（`agent/tools/research/`，28 个模块 7534 行）

| 模块 | 行数 | 职责 |
|---|---|---|
| `_provenance.py` | 1217 | **判定核心**：`judge_verdict()` + 一批正则机器门 |
| `research_verify.py` | 783 | 主入口：生成代码 → 跑 → 判定 → 重试 |
| `_sandbox.py` | 673 | 白名单沙箱（AST 静态门 + 子进程隔离） |
| `_claim.py` | 596 | 命题规范化 / 量词 |
| `_loop.py` | 376 | 多轮 ReAct 骨架（claim 级） |
| `_methods.py` | ~180 | 题型识别 + 方法目录 |
| `_verdict.py` | ~150 | **判定从制品推导** |
| `_strategy.py` | ~150 | 强制换方法（`# STRATEGY:` 标签） |
| `_lean.py` | 306 | Lean 后端 + Mathlib 接入 + axiom 后门锁 |
| `research_{plan,investigate,refute,sympy,z3,enumerate,oeis,audit,report,lean}.py` | — | 其余工具入口 |

### 一次验证的完整链路

```
claim（含脱敏后的输出格式约定）
  ↓
_generate_code()
  ├─ plan_method()          规模/结构线索
  ├─ method_catalog_block() 可查的方法目录（不是按题型塞的提示）
  └─ # STRATEGY: <name>     模型自己声明用的方法
  ↓
_sandbox: AST 静态门 → 子进程执行（20s/题 可调）
  ↓
judge_verdict()            A 线机器门（正则看 stdout 措辞矛盾）
  ↓
_verdict.derive()          ★ 从**制品**推导判定，不采信模型声明
  ↓
失败/被推翻/自认定不了 → 重试（最多 5 轮，RESEARCH_MAX_RETRY）
```

### 两个正交的判定层（**不要合并**）

| 层 | 管什么 | 实测覆盖 |
|---|---|---|
| **凭证层** `_refuted_without_credential` | REFUTED 有没有交出"具体输入 + 违反它的值" | 5 条真实假反例里拦 4 条 |
| **一致性层** `_refuted_contradicts_its_own_output` | 交出来的东西和自己 stdout 的其余部分矛盾吗 | 5 条里拦 5 条 |

判决实验结论：**凭证层漏掉的正好是"凭证在场但自相矛盾"**（Pell-61 那条把正解标成反例，
同一份输出里又写「方程成立? True」）。两层管的是**存在性** vs **一致性**，正交。

---

## 3. 当前问题

### 3.1 ⚠️ 最紧急：R 套这轮明显退步，原因未查明

同一套 20 题，**11-09 那轮 5~6/20；今天 11:35 起跑的这轮，13 题只对 1 道**。

今天早上（跑测启动之前）改过的东西：

| 改动 | 预期收益 | 可能反噬 |
|---|---|---|
| 删掉 Pell 技法提示（含 D=61 答案） | 不泄题 | ★ **Pell 类题失去选型引导** |
| 删掉 `COMPOUND_CLAIM` 里的 R(3,4)=9 | 不泄题 | 双向题失去具体教例 |
| `_redact` 改白名单式 | 不泄解法方向 | 格式提示变少，合规率可能下降 |
| 自评字段从契约里剥掉 | 不诱导编造 | agent 交出的字段变少，验证器判"缺字段" |
| `io` 移出沙箱白名单 | 堵答案泄露 | **`io.StringIO` 也用不了** |

**这些改动都还没单独验证过是不是元凶。** 也可能是模型随机性
（同一道 BBP / 100 皇后每次都不同，早就反复观察到）。**两种解释都还没排除。**

> ★ 这就是"拿一个丢一个"。**下一步不该加功能，该先定位这个回退。**
> 做法：对改动逐个回滚二分，找出真正伤性能的那一个。

### 3.2 base / V2 共 40 题没有独立验证器

用户已明确否掉"再手写 40 个验证器"这条路（理由：验证器本身也要验证，无限 regress）。
改为多 Agent 协同：**把独立性从"我手写的正确性"换成"构造上的独立性"**。

**现状**：R 套 20 题有独立验证器（`tests/tools/verifiers_r/`，21 个文件，每个都有
「咬合 + 不误杀」双向自测）。base/V2 那 40 题现在报的是**agent 自述**，不能当正确率读。

**边界（必须承认）**：多 Agent 不能完全取代独立验证器 —— 两个 agent 可能
**同一种方式错**。已实测：res-013 的 agent 自己发明了第三个变量 x,y,z 然后认真解了
一个不存在的题。如果第二个 agent 也这么读题，两边会一致地错。
→ 手写验证器保留为**每套抽 3 道的抽查**，专抓这类共模失败。

### 3.3 60 题跑测的中途结果不可用作基线

`data/research/all60_scored.json` / `all60.log` 是一次跨代码变更的长跑：
11:35 启动，之后又接入了 `_verdict.derive`、方法目录、artifact 要求。
**base/V2 段是旧代码的结果，R 段是半新半旧。** 要拿基线必须用最终代码重跑。

### 3.4 Explorer 阶段还没做（`exploration = 0.00`）

打分器的 `exploration` 维度（权重 0.07）需要轨迹里有 `experiment`/`search` 类型的步骤。
现在适配器**能产出**这类步骤了（今天重写了 `tests/tools/trace_adapter.py`，
从 `# STRATEGY:` 标签和真实 import 读 method/tool，不再硬编码），但**回路里没有
一个真的去做小规模实验的阶段**，所以 0 分是真实的 0。

DS 给的顺序：**Verifier 独立化 → Explorer 显式化 → Critic → Debate**。
第一项今天做了一半（判定改为从制品推导），第二项还没开始。

### 3.5 已知的历史遗留风险

- `_provenance.py` 1217 行，绝大部分是正则门。它**判的是"话说得像不像"**，
  不是数学。B 线的 20 个独立验证器才是判数学的。
  **这两套裁判并存** —— `_provenance` 会不会误杀、误放，目前只有 R 套的
  20 个 verifier 能发现一部分。
- `RESEARCH_MAX_RETRY=5`，单题最坏 5×150s 沙箱超时 ≈ 12.5 分钟
- 沙箱默认超时 20 秒（`RESEARCH_SANDBOX_TIMEOUT`），评测时用 `--timeout` 覆盖
- `QAGENT_COST_GUARD` 不关的话，配额耗尽会返回**错误字符串**而不是异常 →
  `extract_code` 返回空 → 走 NoCode 分支 → 白烧几轮。评测必须 `set QAGENT_COST_GUARD=off`
- ★ **但反过来：跑单元测试时绝对不能设 `QAGENT_COST_GUARD=off`** ★
  实测 2026-10-10：`tests/test_cost_guard.py` 的 4 个用例 +
  `tests/test_user_settings_isolation.py` 的 1 个**依赖这个开关本身存在**。
  带着 `off` 跑全量 = 1417 passed 变成 1412 passed + 5 failed，
  而那 5 条失败**与任何改动无关**，纯粹是调用姿势 —— 去掉就 17 passed。
  干净基线：`python -X utf8 -m pytest tests -q --no-header -p no:cacheprovider`
  → **1417 passed / 5 skipped**。看到 cost guard 相关失败，先怀疑自己多设了这个变量。

---

## 4. 已修的（都有回归测试，别退回去）

| 修的 | 问题 | 测试 |
|---|---|---|
| 沙箱读文件 | `io.open` 绕过 AST 门，模型能直接读走 `data/*_reference.json` 的**答案** | `check_sandbox_no_read.py` |
| certificate 方向 | **15/20 verifier 判错时返回非空诊断串** → 打分器给满分证书 | `check_score_contract.py` |
| prompt 硬编码答案 | `R(3,4)=9 是真的`（res-001 的答案）、Pell 的 `(1766319049,226153980)`、Catalan C50 真值，**无条件注入每条 prompt** | `scan_answer_leak.py` |
| 「不要报告」名单 | 那行字等于宣布"你的答案就是不可约/唯一/最优" | 同上 |
| `_redact` 启发式 | 「含中文就放行」让 `proof_method: "Cauchy-Schwarz 或 SOS"` 原样进 prompt = 给了 res-015 解法方向 | 同上 |
| 重复评分入口 | `score_baseline.main` 有 hash 漂移 + `final_correct` 硬写 None，单独跑 0/20 | 已删除 |
| trace 适配器 | tool/method 硬编码（`"python"`/`"brute_force"`）→ 维度测的是适配器词汇表 | `tests/tools/trace_adapter.py` |
| 判定从声明 | 「宣布结论」成本为零，agent 总选它 → 判定改为从制品推导 | `check_verdict_derive.py` |

全量回归：**1417 passed / 5 skipped**；7 个自建套件全绿。

### ★ 一条必须遵守的工程纪律

**加任何机器门/判据/提示之前，先问"它会误伤哪一类"。**

实例（今天当场发生的）：加了"宣称 PROVED 却无制品就降级"这条规则后，
全量回归挂了 9 道 —— 因为 `sin²(x)+cos²(x)=1` 这种**真假判定题不需要交制品**，
那个计算本身就是证明。现在这条规则带 `require_artifact` 开关，
只有"题目要求交一个对象"时才生效。

> 反例一律适用，不受该开关影响。

---

## 5. 怎么跑

```powershell
cd D:\pythonProject\QAgent
$env:PYTHONIOENCODING="utf-8"
$env:QAGENT_COST_GUARD="off"          # ★ 必须关，否则配额耗尽会静默空转

# 60 题全量(base+V2+R)，约 2 小时
python tests/tools/run_all60.py

# 只跑 R 套(唯一有独立验证器的一套)
python tests/tools/run_r_scored.py

# 单题复跑
python tests/tools/rerun_v2_case.py hard-001-pell-61

# 全量回归
python -m pytest tests/ -q

# 自建套件(每个都是独立的 x 方向回归)
python tests/tools/check_proved_machine_gate.py     # 36/36 机器门咬合+不误杀
python tests/tools/check_refuted_machine_gate.py
python tests/tools/check_verdict_derive.py          # 判定从制品推导
python tests/tools/check_score_contract.py          # 失败诊断不能当证书
python tests/tools/check_sandbox_no_read.py         # 模型读不到答案
python tests/tools/check_crash_extraction.py        # 崩溃不丢已打印的制品
python tests/tools/check_strategy_switch.py         # 强制换方法
python tests/tools/check_berlekamp_convention.py    # 两种 BM 记法都接受

# 诊断工具
python tests/tools/scan_answer_leak.py              # 机械扫描答案泄露
python tests/tools/audit_verifier_contract.py       # verifier×题库字段名一致性
python tests/tools/count_gate_firings.py            # 各门开火次数
python tests/tools/credential_vs_fallback.py        # 两层正交性判决实验
python tests/tools/harness_gate.py                  # 评测前置门(不过就别跑)
python tests/tools/run_all_verifiers.py             # 20 个 verifier 自测
```

### PowerShell 陷阱（踩过很多次）

- **不支持 `&&`**
- **内联 `python -c @"..."@` 三引号 heredoc 在嵌套引号下会炸** —— 复杂脚本一律写成
  `.py` 文件再跑，别硬塞进 `-c`
- `Get-Content` 读 UTF-8 文件要显式 `-Encoding UTF8`，否则中文全乱码
- **编辑工具的精确匹配对 CRLF 文件会失败** —— 这些源码文件是 CRLF，
  批量替换要用 Python 脚本（`io.open(..., newline='')` 读写）

---

## 6. 交接给外部时要强调的三点

1. **"分数"要问它是哪一种**。这个项目里有三个不同的数：
   - agent 自述 verdict（base/V2 只能得到这个，**不是正确率**）
   - 独立验证器判定的 correct（只有 R 套有）
   - 打分器的过程分（8 维加权，其中 7 维的可靠性有限，见 3.1 和 trace 适配器部分）

2. **判据必须行为性、不能句法性**。"这组系数能不能复现那条序列"✓；
   "字段名和我的 C 数组一样吗"✗（实测三次假阴性全出在后者）。

3. **子代理报告"全部通过"是证据不是结论**。今天独立核实发现：
   - 子代理审计说 res-019 的字段名 `is_A5` 泄题 —— 实际 `A5` 就在题面里，不泄
   - 子代理报告的 20 个 verifier，我独立重跑发现 5 处契约不一致（它自己修的）