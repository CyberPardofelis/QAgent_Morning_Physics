# 独立 verifier 契约（R 套 20 题）

## 为什么需要它

DS 的打分器里 `correctness` 权重 0.40，而且明确写了：

> D 只负责诚实打分……**这就是为什么 correctness 一定得来自外部 verifier，不能由打分器自己写答案。**

现在项目里 **0 道题有独立 verifier**，导致：
- `final_correct` 永远是 None → correctness 恒为 0 → 总分封顶 60
- 更糟：**我无法判断「60 题全部正确」到底达没达成**，只能信 agent 自己说的 VERDICT

所以 verifier 不是加分项，是**度量前提**。

## 铁律

1. **verifier 不得 import 或复用 agent 的代码**。它必须自己算。
2. **verifier 不得读 agent 的 stdout 来"理解"它的意思**。只按下面的规范化制品解析。
   理由：解析自由文本 = 让 verifier 迁就 agent 的写法，一旦 agent 换写法
   verifier 就失效。规范化制品是硬约定。
3. **verifier 自身必须能自测**：文件底部带 `if __name__ == "__main__":`，
   用一个**自己构造的**正确答案 + 一个**故意错的**答案各跑一遍，
   断言"对的过、错的不过"。没有自测的 verifier 不收。

4. **verifier 不得要求「题面没让 agent 交」的字段**。这是工具侧自相矛盾，
   表现为"答案其实是对的却被判错"，而且**和「模型不会」长得一模一样**。

   实测反面教材 `res_11_sudoku_diagonal.py`（2026-10-10）：
   - 题面输出契约明写「只交 `solution`」，并显式禁止交「是否唯一」这类**自评**
     （`research_verify.SELF_EVAL_FIELDS` 就是干这个的）；
   - 而 verifier 一边 `parse_artifact(stdout, "CONSTRUCT")` 只认 `CONSTRUCT:` 标签
     （harness 发给 agent 的是 `ANSWER:`），一边硬性要求 `unique is True` 和
     `uniqueness_certificate`。
   - 三处叠加 → 一个**完全遵守题面**的 agent 必然判错。

   规矩：**先看 `problem_spec.build_claim()` 实际发给 agent 的字段清单**，
   verifier 的必填项必须是它的子集；要额外的证据，就去题面里加，而不是在
   verifier 里偷偷加。

5. **判据必须行为性，不能句法性**。

   > 能不能**生成 / 满足**那个东西？
   > 不是"它长得像不像我算出来的那个"。

   同一次实测里，这条判据的两头都出了问题：
   - `_certificate_proves_unsat()` 只是在制品字符串里找 `"unsat"` 字样。
     它**分不清**真的 Z3 查询输出和编的一句话 —— 于是：该交的被拒（假阴性），
     随便写个 `{"note": "unsat"}` 就能过（假阳性）。**两头都错**。
   - 「制品行的标签是 `CONSTRUCT` 还是 `ANSWER`」同样是句法性的。
     标签由题面决定，verifier 不该挑。

   有行为性判据就别退化到句法性：res-011 的唯一性其实**已经被独立枚举证实**
   （`_solve_clues(..., limit=2)` 只找出一个解），证书那一项没增加任何证明力，
   只增加了误杀面。正确做法是把它降级成 `ck.note(...)` 记录在 `details` 里。

## 制品契约

agent 的验证代码必须打印下列之一（沿用已有的 VERDICT 契约）：

```
VERDICT: PROVED | REFUTED | PARTIAL | INCONCLUSIVE
```

外加**按题型**打印的制品行：

| 题型 | 制品行 | 例 |
|---|---|---|
| 反例类 | `COUNTEREXAMPLE: <json>` | `{"n": 3, "value": 9}` |
| 构造类 | `CONSTRUCT: <json>` | `{"matrix": [[...]], "kind": "coloring"}` |
| 数值/闭式类 | `ANSWER: <json>` | `{"poly": "x^8-...", "degree": 8}` |
| 布尔类(Lean/通用) | 无需制品，看 `VERDICT` + Lean 编译输出 |

解析时从 stdout 里正则抓 `<TAG>: <json到行尾>`，JSON 解析失败即判定为"未提交制品"。

## verifier 的返回结构

```python
def verify(stdout: str, code: str, claim: str) -> dict:
    """
    返回:
      correct      : bool        —— 答案是否**真的**满足题目要求(独立判定)
      certificate  : str|None    —— 机器可验证的证书文本; correct=False 时给失败原因
      final_answer : str|None    —— 结构化答案, 供打分器 calibration 维度使用
      details      : dict        —— 分项校验结果, 便于诊断
    """
```

**不要**返回一个"大概对"的布尔值。任何一项硬性要求不满足，`correct` 就是 False，
并在 `details` 里写清是哪一项。

## 每道题的具体校验点

见各自文件的 docstring。通用要求：

- 构造类：把 agent 给的对象**逐项重新验证**（例如 Hadamard 矩阵要真的算 HH^T）
- 反例类：**代回原命题**验证，且要确认它是"最小/范围内"的（若题目要求）
- 闭式类：代回原方程/原式验证
- Lean 类：跑 `lake env lean` 或复用 `_lean.check_lean`，确认 exit==0 且无 sorry

## 目录

```
tests/tools/verifiers_r/
    res_01_ramsey.py
    res_02_golomb.py
    ...
    common.py        # 制品解析 + 通用断言辅助
```

`common.py` 提供：

```python
parse_artifact(stdout, tag) -> dict | None   # 抓 "<TAG>: <json>"
require(cond, msg, details) -> None          # 不满足就记进 details 再抛
class Check:                                 # 收集多项检查结果
    def __init__(self): self.details = {}; self.failures = []
    def expect(self, name, cond, msg=""): ...
    def correct(self) -> bool: ...
```

## 自测要求

每个 verifier 文件底部：

```python
if __name__ == "__main__":
    # 1) 自己造一个正确答案 -> correct 必须是 True
    # 2) 自己造一个错误答案 -> correct 必须是 False
    # 两行输出: [OK] 正例 / [OK] 反例; 任一不成立 exit 1
```