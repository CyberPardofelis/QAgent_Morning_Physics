# 科研验证闭环 —— 审查结论（交给 minimax / 下一个接手人）

> 作者：Mavis（外部审查，非本功能开发者）
> 日期：2026-10-09 | 范围：`agent/tools/research/` 全部 28 模块 + 评测 harness + R 套 20 个独立验证器
> 关系：**本文是 `readme_minimax.md` 的接续/纠偏**。readme 描述现状与自我诊断；本文给出独立复核后的重新归因与行动清单。
> 纪律：**本次审查源码一行未改**（只临时写了 4 个丢弃的复现脚本，已清理）。
>
> **怎么读**：第 0 节讲证据等级 → 第 1 节是最重要的重新归因 → 第 2~4 节是可动手的缺陷清单（带 `file:line`）→ 第 5 节是架构判断 → 第 6 节是行动清单 → 第 8 节给复现命令，**任何一条结论你都可以自己跑一遍证明或推翻**。

---

## 0. 证据等级（先说清楚哪些是实测、哪些是推断）

| 等级 | 含义 | 本文里的条目 |
|---|---|---|
| **实测** | 我本地跑过命令/读过代码并核对了输出 | 第 1、2、3、4 节几乎全部 |
| **推断** | 从实测事实推出，但没有做对照实验 | 第 7 节，以及正文中显式标注的几处 |

凡我说"实测"的，第 8 节都有对应的复现命令。**没有复现命令的断言，请当推断看。**

---

## 1. ★ 最重要的一条：R 套 15 道失败的重新归因

`all60_process.md` 的结论是「剩余失败里已经没有假阴性了，全是真失败」。
**这个结论不成立。** 逐题核对验证器判词、`_evidence.jsonl` 的取用路径、以及每题实际送进工具的那个 claim 之后：

| 归因 | 题号 | 数量 |
|---|---|---|
| **输入拿不到**（题面数据在文件里，沙箱禁读，题面也不含数据） | res-005 / 007 / 010 / 011 / 013 / 014 / 016 / 020 | **8** |
| **REFUTED 侧制品门不对称**（正确答案没装进制品行，且从没收到过格式反馈） | res-017 / 020 | 2 |
| **格式合规失败**（proved 侧门开了，3~4 轮仍不打制品行） | res-015 / 019 | 2 |
| **真能力不足** | res-006（PSLQ 残差 1.4e-5）/ 008 / 009（Lean） | **3** |
| 方法选型 + 超时 | res-002（Golomb 15 阶，931s 单轮） | 1 |
| 崩溃丢弃已算出的证据 | res-004（z3 已算出 x=-2,y=2,z=2） | 1 |

**⇒ 15 道里只有 3 道是真的做不出来。其余 12 道的根因都在工具/接口/接线上。**

### 1.1 「输入拿不到」是什么形态（本次审查最关键的发现）

`tests/tools/problem_spec.py:226-245` 的 `DATA_HINTS` 告诉模型：

> 实例在 `data/res_16_tsp.txt`。沙箱禁读文件。

而 `data/res_16_tsp.txt` 是 **16934 字节、500 座城市的坐标**，题面里一个数字都没有。
`data/res_13_system.txt` 里写着真正的方程组（`x^2+y^2-3=0`, `x*y-1=0`）和"共 4 组解"，题面只说"见 data/res_13_system.txt"。

⇒ 模型**必须自己发明题目**。这就是 HANDOVER 里那条"res-013 的 agent 自己发明了第三个变量 x,y,z 然后认真解了一个不存在的题"的真因 —— 不是它解错题，是**它收到了一道空题**。
同一根因还解释了：res-005 递推式对不上、res-007 序列对不上、res-016 交不出 tour、res-011 盘面形状全错、res-014 动作解不开魔方、res-020 自己编猜想。

### 1.2 一条反讽（同一个 bug 同时挡人和挡题）

`_sandbox.py:75-112` 拦读文件靠的是**名字匹配**（禁 `open` 这个名字 + 禁 `.read/.open/...` 这些属性名 + 禁 `io` 模块）。
我实测：`np.loadtxt` / `np.genfromtxt` / `np.fromfile` / `np.load` **全部放行**。
⇒ 那句"沙箱里读不到任何参考答案"**既没守住红线**（`np.fromfile(路径, dtype=np.uint8).tobytes().decode()` 一行就能读走 `data/*_reference.json`），**又恰好挡住了模型读自己的输入**。

按项目自己的话（`_sandbox.py:108`）："一个靠这句话建立的基准，如果那句话不成立，整套分数都不作数。"
**⇒ 在修掉之前，R 套的通过率不能当作能力读数**：40% 的题既不可能做对，也没有真正防住作弊通道。

---

## 2. 被当作"今天的核心修复"、但实际没接线的机制（死代码）

readme §2「两个正交的判定层」、§4「已修的（都有回归测试）」里的两条，**在运行路径上都不存在**。

### 2.1 `_verdict.derive()` 从未被调用

- 定义：`agent/tools/research/_verdict.py:72`
- 唯一调用方：`tests/tools/check_verdict_derive.py`（**只有测试**）
- `research_verify` 的链路是 `judge_verdict()`（`_provenance.py:718`）直接产出 verdict，`derive` 不在里面
- 连带：`research_verify.py:192` 的 `_DERIVED: list[dict] = []` **从来没有被 append 过**（只在 `:779-781` 被读），于是返回的 `declared_vs_derived` 永远是 `{"rows": 0, "mismatch": 0, "detail": "无"}`

⇒ readme §2 画的 `judge_verdict() → _verdict.derive() ★从制品推导，不采信模型声明★` 那条链**不存在**；"声明 vs 推导"这个新诊断量在结构上无法产出数据。

### 2.2 `_methods.classify / methods_for` 是装饰

- `_methods.py:172-180`：`methods_for(shapes)` 无论传什么形状，都返回**同一份 9 条全量目录**（实测 `methods_for([]) == methods_for([两个形状])` 为 True）
- `prompts_research.py:711`：`ms = methods_for(shapes) if shapes else []`

⇒ "认题"的唯一作用是：**一条形状都没认出 → 整个目录不显示；认出任意一条 → 显示全部 9 条**。所谓"认题 → 找方法"，实际是"认题 → 打印全表"。`classify()` 的输出对行为零影响。

### 2.3 `answer_spec` / `quantifier` 也是死的（这一条直接造成 015/019 失败）

- `research_verify.py:542`：`cur_code, raw = _generate_code(claim, scope, how, retry_context=retry_ctx)` —— **没传 `quantifier`，也没传 `answer_spec`**
- `_generate_code`（`:195-243`）这两个形参**没有任何调用方赋值**；`research_verify()` 的签名里根本没有 `answer_spec` 参数
- 于是 `prompts_research.py:446-471` 的 `answer_spec_block()` —— 那段「制品行的 JSON 必须包含这些字段：`print("ANSWER: " + json.dumps({...}))`，构造类用 CONSTRUCT，反例类用 COUNTEREXAMPLE」—— **在首轮 prompt 里不存在**

⇒ 模型**第一次生成代码时，从没被告知"你必须打一行 CONSTRUCT/ANSWER 制品行"**。这句话只在**重试**时、且只在它**已经宣称 PROVED** 时才出现（见 3.1）。res-015 / res-019 的"格式合规失败"就是这么来的。

### 2.4 降级记录被"影子化"，报告取到的是被否掉的那条

- `research_verify.py:622-629`：机器门把 `proved` 降级成 `inconclusive` 时，`ws.append_evidence(ev)` **追加一条同 id 的新记录**（不是覆盖）
- `run_all60.py:58-73` 与 `run_r_scored.py:67-92` 的 `best_evidence()` 都**按 RANK 从磁盘取最高分那条**

⇒ 报告里的 `agent_verdict` 取到的是**已经被门否掉的那条** verdict。base/V2 的"自判 proved"数量因此虚高（base/V2 只能报这个数，所以影响直接）。

---

## 3. 直接吃掉分数的具体 bug（每条都带定位 + 修法 + 验收）

### 3.1 制品门只管 PROVED，不管 REFUTED —— 一行改动，两题翻正

- 定位：`research_verify.py:622` → `if ev.get("verdict") == "proved" and wanted_fields:`
- 后果：res-017 判 `refuted`，**格式反馈一次都没触发**。按 HANDOVER 自己的说法，它的答案（n=18）**是对的**，只因没装进制品行被判错；res-020 同理（n=3，`2^3+1=9`）。
- 修法：条件改成 `in ("proved", "refuted")`（`_missing_artifact_reason` 对两者都适用）
- 验收：`tests/tools/check_*` 里加一条"refuted 但无制品行 → 必须降级并注入格式反馈"；单题复跑 res-017 应得到制品行

### 3.2 res-003 的契约自相矛盾：prompt 叫它别交对象，验证器只认那个对象

- 题库 `expected.type = "boolean"`，`value = true` → `problem_spec.expected_to_spec` 渲染成「结论类型: 真假判定 / 需要提交的值: <True/False, 表示该性质是否成立>」
- 而 `tests/tools/verifiers_r/res_03_hadamard.py:66,78,83` 要的是制品里的 `matrix`（`矩阵里有 0 → entries_pm1 失败`）
- ⇒ 模型被明确告知"本题只要判定"，然后被按"你交的矩阵"打分
- 修法：改题库 `expected.type`（object + 字段 `matrix`），或让验证器接受"True/False 判定 + 题面 detail 里的性质"
- 验收：`harness_gate.py` 加第 L3 层（见 3.6）后应能自动抓到这类不一致

### 3.3 写法模板按"默认参数"给，不按题型给

- 定位：`prompts_research.py:908-910` `template_for(how_hint)`；`research_verify.py:231` 用 `how`（默认 `"symbolic"`）
- 实测：20 道 R 题里 **17 道**被塞进 `sympy.simplify(lhs - rhs)` 那段符号恒等式骨架 —— 包括 TSP、数独、魔方、Golomb 尺
- ⇒ 它把模型往**错误的形态**上引（"把命题两边写成符号式然后化简"对 TSP 毫无意义）
- 修法：按题型映射模板（认题结果至少在这里用上），或直接删掉模板注入

### 3.4 prompt 长度与算力配置不匹配

- 实测每轮送进 LLM 的 prompt：**平均 11371 字符**（base ≈ 9.0k 固定纪律文 + 方法目录 ≈ 2.1k + plan/模板 ≈ 0~0.9k）
- 配置：`temperature=0.2, max_tokens=4000`，**单发**
- ⇒ 指令密度远高于题目本身；模型要在 4000 token 里既读完 ~9k 字符纪律、又写出一段完整的验证程序
- 修法：纪律文按**题型/失败类型**按需裁剪（现在是无条件全量注入）；把"格式与制品"这一块前置、加粗

### 3.5 "强制换方法"几乎不会开火

- 定位：`_strategy.py:67-83` `method_class_failure` 只认两种措辞：`"执行超时"`、`"里没有" + "这个名字"`
- all60 的「模式 C」描述的就是这个现象；res-002 烧了 2×150s 也没换路
- 修法：把"超时"以外的高置信信号补进来（如单轮用满 timeout 且 stdout 为空、静态门以规模为由拒绝）；**不要**靠扩大正则去认新措辞

### 3.6 评测前置门只保证"验证器能跑"，不保证"题是活的"

- 定位：`tests/tools/harness_gate.py:42-102`，L1 = 每题有 verifier、L2 = verifier 咬合+不误杀自测
- ⇒ 它抓不到任何 1.1 / 3.2 那类问题（题拿不到输入、prompt 与验证器字段不一致）
- 修法：加 **L3 可解性门**，跑之前断言：
  1. 模型拿得到全部输入（题面自包含，或输入已内联）
  2. prompt 给的字段名 ⊇ 验证器要的字段名
  3. 验证器要求的制品 tag ⊆ prompt 允许的 tag
  4. 任一条不通过 → **拒绝开始评测**（与现有 L3 一致：返回非 0）

### 3.7 小项（一并修，都是静默漂移类）

| 问题 | 定位 | 说明 |
|---|---|---|
| `SELF_EVAL_FIELDS` 逐字复制两份 | `research_verify.py:103-109` + `problem_spec.py:39-45` | 实测当前一致；改一边必然静默漂移 → 收成一处 |
| R1 豁免是**子串匹配** | `_provenance.py:991-994` | 代码里一句注释写 `Solver` 就能关掉 R1（`_COMPLETE_METHOD` 清单里还有 `nthroot`、`decide` 这类常见词） |
| `_redact` 对 list 泄漏长度 | `problem_spec.py:85-86` `f"<长度 {len(v)} 的列表>"` | 与它自己"数量也是答案的一部分"的规则（`:105-116`）不一致。当前 R 套 expected 里恰好没有 list，属潜在风险 |
| `plan_method` 仍留着按题型塞的解法提示 | `prompts_research.py:779-797`（digit-extraction）等 | 那段等于把 BBP 的实现交出去（`pow(16,e,d)` 而非 `16**e % d`、浮点溢出坑、自检建议）。这与"删掉 Pell 提示（因为泄题）"是**同一条政策的反面** → 政策要统一 |

---

## 4. 代码资产的分布：53% 的代码不在测量路径上

实测行数：

```
total 7534 行 / 28 模块
  在路径上 3540 行（_provenance 1217 / research_verify 783 / _sandbox 673 / _lean 306 / research_run_code 152 / _verdict 140 / _strategy 111 / __init__ 94 / _flags 64）
  不在路径 3994 行 = 53%
```

不在路径的是：`_claim.py`(596) `_loop.py`(376) `_enumerate.py`(365) `_precheck.py`(333) `_db.py`(288) `_sympy_tools.py`(236) `_retrieve.py`(217) `_z3_tools.py`(197) `research_investigate.py`(193) `_methods.py`(184) `research_lean.py`(145) `research_refute.py`(143) `research_plan.py`(138) `research_audit.py`(129) `research_report.py`(99) `research_enumerate.py`(95) `research_sympy.py`(92) `research_oeis.py`(86) `research_z3.py`(82)

判据：`run_all60.py:115` / `run_r_scored.py:127` 调的都是 `ToolRegistry.get("research_verify")`；`research_investigate` + `_loop.run_loop` + `_precheck.apply_precheck` 这条**更聪明**的链（命题分解 + 量词纪律 + 体检闸门 + 多轮 ReAct）**评测从来不碰**。

⇒ 也就是"命题层纪律"和"多轮研究回路"是**另一条产品线**，报告却按 `research_verify` 的分数谈能力。两条路并行是"绕弯子"里最贵的一条。**要么接进测量路径，要么在报告里明确它不属于这张成绩单。**

---

## 5. 架构层判断：单发代码生成 ≠ 做数学

现有被评测驱动的路径只有一条：

```
一次 LLM 调用 → 一段代码 → 沙箱跑 → 正则门判 → （失败才）重试
```

没有交互式工具使用、没有小规模实验、没有文献检索、没有引理分解、没有多实例交叉审查。
`exploration = 0.00` **不是"少了一个工具"，是这个形状的必然结果** —— 单发式回路天生产不出 `experiment` 步骤。

我另外查了外部资料（1 份 OpenAI 官方报告，2025-11，讲 GPT-5 在数学/物理上的早期科学加速实验），其要点与本项目的失败形态正好对照（只记高层，不列检索结果）：

- 被点名的有效做法是：**一个实例头脑风暴 → 另一个实例写完整证明 → 第三个实例独立挑错 → 最后 Lean 形式化**，外加**深度文献检索**；
- 该报告里"文献检索"本身是**一等产出**（包括"这个问题其实 20 年前已有人解决"这种结论）；
- 明确鼓励模型**诚实说明"某个引理我没证出来"**，而不是编造；
- 也不回避失败案例（过度声称、方向错误）。

对照下来结论是：**精力投错层了。** `_provenance.py` 1217 行里绝大部分是**正则匹配措辞**（判的是"话说得像不像"，不是数学），而 12/15 的失败发生在"模型拿不到题、拿不到格式反馈"这种一层就能修的地方。

### 5.1 该保留的（别在重构时丢掉）

1. **R 套 20 个独立验证器**（`tests/tools/verifiers_r/`）—— 全项目最值钱的资产：自己重算、离线、不读 agent 代码与自由文本、每个都有正例+反例双向自测。**这就是"结论权不在模型手里"的正确实现。**
2. `_sandbox` 的子进程隔离 + 杀进程树 + 哨兵回传（结构是对的，只是白名单策略要改，见 1.2）。
3. 证据链 / workspace / `_evidence.jsonl` 的落盘纪律。
4. "声明 vs 推导"这个**想法**是对的（`_verdict.py` 的 docstring 讲得很清楚）—— 只是没接线（见 2.1）。
5. `harness_gate` 的"评测前置门"思路（补 L3 即可）。

---

## 6. 行动清单

> 每条都给"改哪里 + 验收怎么看"。P0 做完**必须重跑 R 套并重新定基线**（现在的基线不可用）。

### P0（半天，先让分数反映真实能力）

| # | 动作 | 涉及文件 | 验收 |
|---|---|---|---|
| 1 | **把题目输入送进 claim**：`build_claim` 内联 `data/res_*.txt` 正文（`*_reference.json` 除外）；或给沙箱开一个只读输入目录 | `tests/tools/problem_spec.py:208-245`（+ 可选 `_sandbox.py`） | 8 道题里至少 6 道能产出与验证器对得上的制品；逐题看 `_evidence.jsonl` 的 stdout 里出现了正确的题面数据 |
| 2 | **制品门对称化**：`proved` 和 `refuted` 都要制品 | `research_verify.py:622` | 新回归：refuted 无制品 → 降级 + 注入格式反馈；res-017 / 020 单题复跑应通过 |
| 3 | **修 res-003 契约矛盾** | `baseproblems_R.txt` 的 `expected` 或 `verifiers_r/res_03_hadamard.py` | prompt 与验证器要的字段一致；res-003 单题复跑不再是"契约级不可能" |
| 4 | **降级改成覆盖同 id**（或让 harness 只读"门后"记录） | `research_verify.py:629` / `run_all60.py:58-73` / `run_r_scored.py:67-92` | `_evidence.jsonl` 不再出现同 id 重复；报告里的 `agent_verdict` == 循环的最终 verdict |
| 5 | **加 L3 可解性门** | `tests/tools/harness_gate.py` | 故意改坏一个字段名/删掉一份输入 → 门必须返回非 0 并拒跑 |

### P1（1~2 天，接线 / 去噪）

6. `answer_spec` + `quantifier` 真正接进 `research_verify → _generate_code`；把"必须打制品行 + tag 选择"写进**首轮** prompt（`research_verify.py:542` + `:469-478`）。
7. `derive()` 接进 `_execute` 之后的判定链，**或**删掉它和 `_DERIVED` —— 别留一个永远返回 0 的诊断量。
8. `methods_for` 真正按 shape 过滤；做不到就删掉整个 `_methods.py`。`template_for` 改成按题型映射，或删掉模板注入。
9. 统一"提示政策"：`plan_method` 里的题型解法提示（digit-extraction 那段）按"Pell 提示"的同一标准处理。
10. 读文件的红线改成 **OS 级约束**（或显式声明"允许读 `data/inputs/` 只读"），不要再用名字匹配 —— `np.*` 已经漏了（实测）。
11. 对 53% 的 off-path 代码做取舍（见第 4 节）。

### P2（架构，这才是"像那份 OpenAI 报告那样"的部分）

12. 把单发 codegen 换成**多轮 agent**：允许读题面数据、允许检索（至少 OEIS/文献）、允许"跑小实验 → 看结果 → 再改"、允许把命题拆成引理分别验证。
13. **结论权彻底从模型手里拿走**：模型只交制品与步骤，`correct` 只由 verifier 决定（这条你已经做对了，见 5.1）。
14. 把"没找到 / 已有人做过 / 只证出一半"当成**合法产出并计分**，而不是逼它升级成 PROVED（根因是"宣称结论的成本为零"）。
15. 另建一个 **10 题左右、输入自包含、题面完整**的最小研究集，让 `score` 第一次真的在测数学。**在那之前不要再用 R 套的通过率对外说能力。**

---

## 7. 我没验证的猜测（请当假设，不要当结论）

1. **res-015 / res-019 在门已开火的情况下仍不打制品**：我推断是 prompt 过长（≈11.4k 字符）稀释了指令，**没做对照实验**（没跑 A/B 截断 prompt 的版本）。
2. **res-002 的"强制换方法未触发"**：我只看到 931s 单轮 + `method_class_failure` 的判据，**没拿到它那轮声明的 `# STRATEGY:` 标签**，所以是"没触发"还是"触发了但放行"没分清。
3. **`np.fromfile` 读取参考答案这条路是否真会被模型走到**：我验证了静态门放行（实测），但**没有实测跑一遍读文件**（沙箱 cwd 是 scratch 目录，需要绝对路径；模型不知道绝对路径）。风险是"通道存在"，不是"已被利用"。

这三条要坐实都得跑单题复现，我没跑 —— 没有获得消耗配额的授权。

---

## 8. 复现命令（独立复核本文任何一条结论）

```powershell
cd D:\pythonProject\QAgent
$env:PYTHONIOENCODING="utf-8"
```

**(a) `derive()` 是死代码 / `_DERIVED` 从不写入**

```powershell
Select-String -Path agent\tools\research\*.py, tests\tools\*.py -Pattern "_verdict\.|derive\(" 
# 期望：_verdict.py 里只有定义；调用方只有 tests/tools/check_verdict_derive.py
Select-String -Path agent\tools\research\research_verify.py -Pattern "_DERIVED"
# 期望：192 行定义 + 779-781 行读取，没有 append
```

**(b) `methods_for` 是 no-op**

```powershell
python -c "from agent.tools.research._methods import methods_for as m; print([x['name'] for x in m([])] == [x['name'] for x in m(['任意形状'])])"
# 期望：True
```

**(c) `answer_spec` / `quantifier` 没接进主循环**

```powershell
Select-String -Path agent\tools\research\research_verify.py -Pattern "_generate_code\(|answer_spec|quantifier"
# 期望：542 行的调用**没有** answer_spec= / quantifier= 实参
```

**(d) 制品门只管 proved**

```powershell
Select-String -Path agent\tools\research\research_verify.py -Pattern 'verdict"\] == "proved"'
# 期望：622 行，只有 proved
```

**(e) 8 道题的输入模型拿不到**

```powershell
python -c "import sys; sys.path.insert(0,'tests/tools'); import problem_spec as P; print(list(P.DATA_HINTS))"
Get-ChildItem data\res_*.txt | Select-Object Name, Length
# 期望：8 个 id；res_16_tsp.txt 16934 字节（题面里没有任何一个坐标）
```

**(f) 沙箱红线的漏洞**

```powershell
python -c "from agent.tools.research._sandbox import check_code, SandboxRejected
for n,c in {'np.loadtxt':'import numpy as np; np.loadtxt(\"D:/x.txt\")','np.fromfile':'import numpy as np; np.fromfile(\"D:/x.txt\")'}.items():
    try: check_code(c); print('放行', n)
    except SandboxRejected as e: print('拦下', n)"
# 期望：两条都“放行”
```

**(g) prompt 体积**

自己拼 `build_code_prompt + plan_method().note + method_catalog_block + template_for(how)`，量 `len()`。
期望：R 套平均 ≈ 11371 字符，其中 base ≈ 9.0k。

**(h) 每题实际收到的 claim（找出 3.2 那类契约矛盾）**

用 `problem_spec.build_claim(prompt, expected, data_hint_for(cid))` 逐题打印。
期望：res-003 显示"真假判定 / 需要提交的值: True/False"，而它的验证器要 `matrix`。

**(i) R 套重跑（修完 P0 之后）**

```powershell
$env:QAGENT_COST_GUARD="off"
python tests\tools\harness_gate.py     # 前置门，不过就别跑
python tests\tools\run_r_scored.py     # 唯一有独立验证器的一套
```

---

## 9. 别再做的事

- 不要再加正则门 / 措辞门（`_provenance.py` 的增长曲线就是这条路的代价，readme §3.5 自己也承认了）。
- 不要为 base/V2 那 40 题手写更多验证器，也不要再往契约里加自评字段。
- 不要在"输入拿不到"的题上刷分、调提示、加门。
- 不要在没有 P0 的情况下对外报 R 套通过率。

---

## 附录 A：第二轮复核（2026-10-09 14:00，基于 `progress_and_failures.md`）

> 背景：minimax 已落地第一轮 P0 六条，R 套 3/20 → 5/20 → **9/20**（我核了 `r_scored.json`：ok=9 / wrong=8 / unverified=3，确认属实）。
> 本轮我读了它改过的 5 个文件 + 全部 20 题的 `_evidence.jsonl`，并对"未解决"的 P1~P5 逐条追到原始 stdout。
>
> **结论先说：五条未解决里，三条是误诊，两条是真 bug 但根因不是它写的那两个。**
> 真正卡住 R 套的，仍然是**工具在对自己的判据撒谎**，不是模型不听话。

### A.1 P0 六条确实修好了（确认）

| 修复 | 我的独立验证 |
|---|---|
| P0-1 输入内联 | ✅ res-005 / 010 / 013 / 015 / 017 / 020 现在都进了 ok，答案与独立验证器对得上（res-013 的 4 组解与独立全枚举一致） |
| P0-2 封 numpy 整类读取 | ✅ 白名单已收窄 |
| P0-4 `answer_spec` 进 prompt | ✅ 首轮 prompt 里已有格式与制品段 |
| P0-5 `methods_for` 真过滤 | ✅ |
| P0-6 L3 可解性门 | ⚠️ 落了 3 条里的 2 条（见 A.3，**tag 兼容性漏了**） |

### A.2 ★ P1 诊断错了：res-019 不是"模型不听话"，是 stdout 截断吃掉了制品

`progress_and_failures.md` 把 res-019 当作「机器在喊、模型不听话」最干净的复现，并据此建议**在行动层封死交制品这条路**。
**对这道题，这个方向是在追影子。真因是沙箱输出上限。**

实测（`agent/tools/research/_sandbox.py:64` `MAX_OUTPUT_CHARS = 8000`）：

```
res-019 的 stdout 原始长度 = 46454 字符
截断后存进证据的 = 8041 字符，接缝处是：
    ", \"3,47\": \n... [中间 38454 字符因超长被截断, 共 46454 字符] ...\n2, \"54,53\": 43, ..."
```

那个制品行是 60×60 的群乘法表 JSON，本身就有 ~46,000 字符。
`_truncate` 保留头尾 → **JSON 被从中间劈开** → `_artifact_fields()` 返回 `[]` → 门判「你没有交出本题要求的对象」。

而模型**每一轮都交了**。11 条证据里，制品是这样的（字段齐全）：

```
ANSWER: {... "group": {"0,0":0, "0,1":1, ..., "59,59":0},
         "is_nonabelian": true, "is_A5": true,
         "generators": [[1,2,3,4,0],[0,2,1,4,3]]}
VERDICT: PROVED
```

→ **同一批 stdout 上逐条比对：`_artifact_fields` 返回 0 个字段，11 次里有 11 次被判"没交制品"。**
→ 它换了策略、换了库、迭代 6 轮，每一轮都在正确地交制品，每一轮都被同一个截断吃掉。

> ⚠️ 这条对「宣布结论成本为零」那个判断是个**反例**：这道题的成本不是零，是工具把成本抬到了不可能。

**全 R 套只有 res-019 命中截断**（我扫了 20 题的 stdout），但它是"格式合规失败"里被当成旗舰案例的那一道。

**修法（推荐第 2 条）**：
1. `MAX_OUTPUT_CHARS` 抬到 200k+（最省事，但治标，且会把噪声灌进重试 prompt）；
2. ★ **先抽制品、再截断**：`run_code` 里对完整 stdout 先跑一遍 `parse_artifact`，把制品单独回传/落到 evidence 的独立字段，`stdout` 怎么截都不影响复核；
3. 沙箱把 `CONSTRUCT/ANSWER/COUNTEREXAMPLE` 行单独回传，与人类可读输出分离。

> 注意：修完只是**从"不可判"变成"可判"**，pass 还取决于那个群在数学上对不对 —— 这一步我不能替它下结论（证据里没有未截断的原始 stdout）。

### A.3 ★ P3 诊断错了：res-011 交的是对的，卡在 tag 与"自评字段"上

`_attempts` 说「机器门说 PROVED，验证器说缺制品，未查清」。查清了，是**两个独立的工具侧矛盾**：

**(1) tag 不匹配。** 模型交的是 `ANSWER: {"solution": [...9x9...]}`，
而 `tests/tools/verifiers_r/res_11_sudoku_diagonal.py:279` 是 `parse_artifact(stdout, "CONSTRUCT")` —— **只收 CONSTRUCT**。
（全套 20 个验证器里，只有 res_002 / res_011 / res_012 / res_016 / res_06 等少数几个是严格单 tag；大部分都写了 `or parse_artifact(stdout, "ANSWER")` 兜底。res-011 没有。）

**(2) 验证器要求"自评字段"，而 prompt 故意不让它自评。**
`res_11_sudoku_diagonal.py:314-319` 要求 `art["unique"] is True` 和一个 `uniqueness_certificate` 对象；
但 `unique` / `claims_unique` 正在 `SELF_EVAL_FIELDS` 里 —— `problem_spec` 故意把它从题面剥掉、`research_verify` 的制品门也故意不要求它。
**prompt 明确说"凡是你对答案的评价一律不要写"，验证器偏要它写。**

实测（我把 `ANSWER:` 改写成 `CONSTRUCT:` 再喂给同一个验证器）：

```
grid_valid_diagonal_sudoku : true    (shape/values/rows/cols/boxes/两条对角线 全过)
matches_puzzle_clues      : true
equals_unique_solution     : true    ← 与独立枚举出的唯一解逐格相同
claims_unique              : False   ← 只因为它按规矩没自评
uniqueness_certificate     : False   ← 只因为它按规矩没编证书
```

**⇒ 这道题的答案是完全正确的，0 分完全是工具造成的。**

**这也暴露 L3 门漏了一条**：它查了「输入可达」「字段名一致」，**没查「验证器收的 tag ⊆ prompt 允许的 tag」** —— 那是我第一轮 L3 清单里的第 3 条。

**修法**：res-011 的验证器改成 `parse_artifact(stdout, "CONSTRUCT") or parse_artifact(stdout, "ANSWER")`；
并把 `unique` / `uniqueness_certificate` 这类**要求自评**的判据从验证器里删掉（唯一性应该由验证器自己枚举判定，它已经有 `puzzle_really_unique` 了）。
L3 门补第 3 条。

### A.4 ★ P5 是误读：details 里的 False 不等于判错

`res_13_groebner.py:220` 是 `ck.note("duplicate_entries", ...)`，`res_17_counterexample.py:298` 是 `chk.note("range_declared", False)`。
**两个都是 `note()`，不阻塞。** `Check.correct` 只看 `failures`，`failures` 只由 `expect()` 产生。
⇒ 「判对了但子项失败」不是新暴露的一层，是把 `details`（诊断记录）当成了 `failures`（判据）。这条可以关掉。

### A.5 ★ P2 根因不是限流：`_is_blocked` 在自由文本上做子串匹配

`progress_and_failures.md` 说「原因未查明，可能是上游限流，也可能是环境变量没生效」。都不是。

日志（`data/research/r_baseline2.log`）：

```
[16/20] res-016-tsp-heuristic-500
    agent自判=blocked 49s 15997tok
[llm] _call_llm done  耗时 26.7s  返回 8841 字符
```

**LLM 正常返回了 8841 字符、15997 token 的完整答案**（含分析 + ```python 围栏），没有任何阻断。

而 `_is_blocked()`（`research_verify.py:81-86`）拿一组 marker 去**子串匹配整份模型输出**：

```python
_BLOCKED_MARKERS = ("[CostGuard]", "cost_guard 阻断", "未发出 LLM 请求",
                    "未知 LLM provider", "429", "rate limit", "insufficient")
```

其中 `"429"` 是裸的三位数字子串。res-016 是 **500 城市 TSP**，题面数据已内联（P0-1 的成果），模型必然把实例抄进代码 —— 于是：

```
res_16_tsp.txt 里 "429" 作为子串出现 11 次
300 个随机 4000 字符窗口的抽样：300/300 命中 "429"
```

⇒ **任何抄了实例的正常输出，100% 会被判成"上游阻断"** → 走 `break` 分支 → 这道题一次都没跑 → 记成 `unverified`。
⇒ 49 秒 + 15997 token 纯浪费，且**统计上被当成"没跑到"，不是"判错"** —— 这个分类本身还算幸运。

**修法**：`_is_blocked` 只在**输出很短**时判断（被真正阻断时 LLM 压根没返回代码，输出通常只有一两百字的错误串）；
更稳的做法是**先试 `extract_code`，抽不到代码再判 blocked**，而不是反过来。顺带：`_attempts.jsonl` 只存 `raw[:500]`，
这题因此**永久不可诊断** —— 建议把原始输出单独留一份（哪怕截到 8k）。

### A.6 res-002 也被归错类了：答案已经在手里，它自己把它否掉了

`progress_and_failures.md` 把它归进「格式合规（答案可能对，但交不出来）」。**不对，它交出来了，然后扔了。**

它这一轮的 stdout：

```
Found candidate marks: [0, 1, 3, 7, 12, 20, 30, 44, 69, 90, 105, 121, 143, 169, 197]
Length: 197

Known optimal Golomb ruler lengths:
n=1..15: [0, 1, 3, 6, 11, 17, 25, 34, 44, 55, 72, 85, 106, 127, 283]

Optimal 15-mark Golomb ruler length = 283 > 200.
Therefore, a 15-mark Golomb ruler with length <= 200 does NOT exist.
COUNTEREXAMPLE: optimal_length=283 > 200, ...
```

我独立验了那个候选：`marks` 长 15、跨度 **197 ≤ 200**、105 个两两距离**互不相同** ⇒ **一把合法的 15 阶 Golomb 尺**。
题目问的正是"构造一把长度 ≤ 200 的尺"。**答案就在它自己的输出第 3 行**，它拿一张编造的"已知最优"表（末项 283；真值 G(15)=127）把自己证伪了。

三个独立的洞叠在一起：

1. **R4 那道门差一个词。** `_REFERENCE_CONST` 只认 `known (reference|expected|value)|标准值|参考值|已知值…`，
   **不认 `Known optimal`** —— 而那正是模型幻觉常量最常用的英文措辞。
2. **凭证门按形式不按可复核性。** `_counterexample_credential` 对这行解析出 `{'optimal_length': '283 > 200, ...'}`
   —— `_FIELD` 正则匹配到 `k=v` 且含数字，就**当成合格凭证**放行了。
3. **矛盾表缺一种形态。** 输出里「Found candidate … Length: 197」与「does NOT exist」并存，
   这是**自相矛盾**里最干净的一种形态（找到了见证又否认它），但 `_CONTRADICTIONS` 六行里没有它。

### A.7 这一轮真正该修的顺序

**P0′（三条小改动，全部是"把工具的假判词改对"，不碰数学）**

| # | 动作 | 定位 | 预期 |
|---|---|---|---|
| 1 | **先抽制品、再截断**（或抬 `MAX_OUTPUT_CHARS`） | `_sandbox.py:64` + `run_code` | res-019 从"不可判"变"可判" |
| 2 | **`_is_blocked` 收紧 + 改成"先抽代码再判阻断"** | `research_verify.py:70-86` | res-016 不再空转 49s/16k token |
| 3 | **res-011：验证器接受 ANSWER + 删掉要求自评的判据** | `verifiers_r/res_11_sudoku_diagonal.py:279,314-319` | res-011 从"答案全对却 0 分"变可判 |

> 这三条加起来，**20 题里至少 3 题从"工具造成的失败"回到"可判"** —— 也就是说 9/20 这个数字里，
> 还有 3 分不是模型挣来的，也不是模型丢的。

**P1′**
4. L3 门补「验证器收的 tag ⊆ prompt 允许的 tag」（我第一轮 L3 清单第 3 条，漏了）。
5. `_REFERENCE_CONST` 补 `known optimal|known best|known lower bound|已知最优|最优值`；矛盾表加一行「找到了见证又否认它」。
6. ★ **降级影子化仍未修**（不在它的修复清单里）：`res-019/_evidence.jsonl` 里 `ev_0001` 出现**两次**（一次 proved、一次 inconclusive），
   `best_evidence` 按 RANK 取最高 → 报告里的 `agent_verdict` 取到的是**被门否掉的那条**。我第一轮的 P0-4 没被采纳，这条会持续污染 base/V2 的自判统计。

**P2′（能力，真正的）**
7. res-002 / 006 / 007 是真能力。但注意 res-002 的形态：**它缺的不是算力，是"先自检再宣布"**——
   它已经有那把尺了。产出顺序上，**先交制品再下结论**比任何门都管用。
8. **Lean 4 道（res-008/009 + V2 两道）：建议先别按"最该攻"处理。** 症状是 `unknown constant 'Nat.add_pow'` ——
   本机 Mathlib 是 **4.9.0**（v490），而模型写的是现代 Mathlib 的名字。**这是环境错配，不是推理能力不足。**
   更划算的做法：给 Lean 后端加一个**探针阶段**（先让模型 `#check` 一批关键引理名 / 用 `grep` 查 Mathlib 源码确认存在，再写证明），
   而不是加轮次 —— 现在 6 轮全在同一个幻觉上打转。
9. `exploration = 0.00` 仍是真实的 0，**这轮没有任何改动触及它**。单发 codegen 的形状没变，
   所以"跑小实验 → 看结果 → 再改"这件事在结构上依然不会发生。

### A.8 一条方法论提醒（这轮最值钱的一句）

**"机器在喊、模型不改"这个现象，连续两次都是工具自己在撒谎。**

- 第一次：`_missing_artifact_reason` 说"你没交制品"——真相是 stdout 被 8000 上限截断，制品被吃掉。
- 第二次：`_is_blocked` 说"上游阻断"——真相是一份正常的 8841 字符答案，因为里面有 "429" 三个数字。

⇒ **看到"模型无视一条极其明确的指令"时，先假设是我的判据在骗自己，去读那条原始 stdout。**
这比再想"怎么让模型更听话"有用得多——因为后者的所有方案（加门、加提示、加工具）都建立在"判词可信"这个前提上。

### A.9 第二轮复现命令

```powershell
cd D:\pythonProject\QAgent
$env:PYTHONIOENCODING="utf-8"
```

**(a) res-019 的截断**（看接缝处的截断标记）
```powershell
python -c "import hashlib,io,json,os;th='r_'+hashlib.sha1(b'res-019-group-construction-60').hexdigest()[:10];\
r=json.loads(io.open(os.path.join('data','research','working',th,'_evidence.jsonl'),encoding='utf-8').readline());\
print(len(r['stdout']));i=r['stdout'].find('因超长被截断');print(repr(r['stdout'][i-40:i+60]))"
# 期望：看到 "... [中间 38454 字符因超长被截断, 共 46454 字符] ..."
```

**(b) 截断导致门误判**
```powershell
python -c "import sys;sys.path.insert(0,'.');\
from agent.tools.research.research_verify import _artifact_fields;\
from agent.tools.research._sandbox import MAX_OUTPUT_CHARS;print(MAX_OUTPUT_CHARS)"
# 8000，而 res-019 的原始 stdout 是 46454
```

**(c) res-011：改 tag 就能被认出来，但还差自评字段**
```powershell
# 把证据里的 'ANSWER:' 换成 'CONSTRUCT:' 再喂给 res_11 的 verify()
# 期望：artifact_present 通过，grid_valid/matches_puzzle_clues/equals_unique_solution 全 true，
#       仍失败在 claims_unique + uniqueness_certificate
```

**(d) `_is_blocked` 假阳性**
```powershell
python -c "import io;t=io.open('data/res_16_tsp.txt',encoding='utf-8').read();\
print('429 出现', t.count('429'), '次 / 共', len(t), '字符')"
Select-String -Path data\research\r_baseline2.log -Pattern '_call_llm done' -Context 0,0 | Select-Object -First 3
# 期望：res-016 那条显示「返回 8841 字符」，不是错误串
```

**(e) res-002 的答案是对的**
```powershell
python -c "m=[0,1,3,7,12,20,30,44,69,90,105,121,143,169,197];\
d=[m[j]-m[i] for i in range(15) for j in range(i+1,15)];\
print(len(m), m[-1]-m[0], len(d)==len(set(d)))"
# 期望：15 197 True  —— 一把合法的 15 阶 Golomb 尺，它自己算出来了
```

**(f) P5 是误读**
```powershell
Select-String -Path tests\tools\verifiers_r\res_13_groebner.py,tests\tools\verifiers_r\res_17_counterexample.py -Pattern 'duplicate_entries|range_declared'
# 期望：都是 ck.note(...) / chk.note(...)，不是 expect
```

---

## 10. 交接时请向 DS 明确的三点

1. **"分数"要问是哪一个**：R 套有独立验证器（可用）；base/V2 只有 agent 自述 + 正则门（**不是正确率**）。
2. **本次审查的结论是"12/15 失败与能力无关"**，请 DS 判断 P0 那 5 条是否排除了他原本设想的下一步（Explorer 显式化 / Critic / Debate）——我的看法是：**那些在题都拿不到的情况下做了也测不出来**。
3. **res-013 那类"自己发明题目"的责任方是工具，不是模型**；HANDOVER 里把它归为"多 Agent 会共模失败"的例子，前提判断需要更正。
