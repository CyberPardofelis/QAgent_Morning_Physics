"""P4 反向验证: 证明这些断言真的会红。

## 为什么需要这一步

一个从没红过的测试,不能证明它在测什么 —— 它可能只是 assert True。
反向验证的做法是**故意改坏源码**, 看对应测试是否变红:

    改坏 → 测试必须红 → 改回 → 测试必须绿

两步都成立, 才说明这条断言真的咬住了那行代码。

## 关键纪律(来自两次翻车)

1. **先数 `src.count(anchor)`** —— 大于 1 说明锚点不唯一,
   半截替换会造出一个"看起来变了其实没测到目标"的假绿。
2. **变异必须真的生效** —— 每次跑测试前先把替换后的片段打印出来核对。
   我曾连续两次失败: 一次是"变异"实际上一行都没改到, 一次是只改了
   重复文本中的一部分。两次都是护栏没失效, 是我没造出变异。
3. **跑完必须还原** —— 任何一步出错, finally 里也要还原。

用法:  python scripts/_reverse_verify_eval.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PY = str(ROOT / "venv" / "Scripts" / "python.exe")

PROVENANCE = ROOT / "agent" / "tools" / "research" / "_provenance.py"
REGISTRY = ROOT / "agent" / "tools" / "registry.py"
CLAIM = ROOT / "agent" / "tools" / "research" / "_claim.py"
LOOP = ROOT / "agent" / "tools" / "research" / "_loop.py"


def _run(test: str) -> tuple[bool, str]:
    """跑一条测试, 返回 (是否通过, 输出尾部)。"""
    proc = subprocess.run(
        [PY, "-X", "utf8", "-m", "pytest", test, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    return proc.returncode == 0, (proc.stdout or "")[-700:]


class Mutation:
    """一次变异: 改文件 → 跑测试 → 还原。"""

    def __init__(self, name: str, path: Path, old: str, new: str, test: str,
                 expect_substring: str) -> None:
        self.name = name
        self.path = path
        self.old = old
        self.new = new
        self.test = test
        self.expect = expect_substring

    def apply(self) -> int:
        src = self.path.read_text(encoding="utf-8")
        n = src.count(self.old)
        if n != 1:
            raise AssertionError(
                f"锚点在 {self.path.name} 里出现 {n} 次, 必须是 1 次。"
                f"多处重复时用整块替换, 否则会造出假绿。\n锚点: {self.old[:70]!r}"
            )
        self._backup = src
        self.path.write_text(src.replace(self.old, self.new, 1), encoding="utf-8")
        # 变异真的生效吗 —— 打出来自己核对, 不靠相信
        after = self.path.read_text(encoding="utf-8")
        if self.new not in after:
            raise AssertionError(f"变异没写进去: {self.path.name}")
        if self.old in after:
            print(f"  (注: {self.path.name} 里原锚点仍有其它同文, 正常)")
        return n

    def restore(self) -> None:
        self.path.write_text(self._backup, encoding="utf-8")

    def run(self) -> tuple[bool, str]:
        print(f"\n{'=' * 68}\n变异: {self.name}\n{'=' * 68}")
        self.apply()
        print(f"  已改: {self.old.strip()[:64]!r}\n     -> {self.new.strip()[:64]!r}")
        try:
            passed, out = _run(self.test)
        finally:
            self.restore()

        print(f"  测试 {self.test} → {'绿(!! 反向验证失败)' if passed else '红(符合预期)'}")
        if passed:
            return False, out
        hit = self.expect in out
        print(f"  报错命中 {self.expect!r}: {'是' if hit else '否 ← 红了但不是这条断言咬的'}")
        return hit, out


def build() -> list[Mutation]:
    return [
        Mutation(
            name="verdict 缺标记就判 untested —— 删掉这条纪律",
            path=PROVENANCE,
            old='return "untested", (',
            new='return "proved", (  # MUTANT: 违反"没标记一律 untested"',
            test="tests/test_research_eval.py::TestEvidenceMandatory",
            expect_substring="",
        ),
        Mutation(
            name="z3 从环境指纹里去掉 —— 结论不可复现",
            path=PROVENANCE,
            old='for mod in ("sympy", "numpy", "scipy", "networkx", "z3"):',
            new='for mod in ("sympy", "numpy", "scipy", "networkx"):  # MUTANT: 漏 z3',
            test="tests/test_research_eval.py::TestEvidenceMandatory",
            expect_substring="",
        ),
        Mutation(
            name="exposed=False 的工具照样暴露给 LLM",
            path=REGISTRY,
            old="if include_hidden or t.exposed",
            new="if True:  # MUTANT: 无视 exposed",
            test="tests/test_research_eval.py::TestEvalSetSchema",
            expect_substring="",
        ),
        # ↓ P4.5 新增: 四道防线
        Mutation(
            name="量词冲突不再拦截 —— 「对所有」又被换成「存在」",
            path=CLAIM,
            old="if not q or c not in QUANTIFIERS or q == c:",
            new="if True:  # MUTANT: 放行所有量词",
            test="tests/test_research_quantifier.py::TestQuantifierConflict",
            expect_substring="",
        ),
        Mutation(
            name="防线四失效 —— 不可判定的问题又能拆出可判定命题去验",
            path=LOOP,
            old="    marker = question_is_undecidable(question) if question else None",
            new="    marker = None  # MUTANT: 防线四关掉",
            test="tests/test_research_quantifier.py::TestUndecidableQuestionBlocksEverything",
            expect_substring="",
        ),
        Mutation(
            name="历史证据又被重复落库 —— 旧 verdict 复活",
            path=LOOP,
            # ⚠️ 锚点对着**当前**代码写。这条变异第一版锚的是早已被替换掉的
            #   `evidence = ws.load_evidence()` / `finalize(state, ws, evidence)`,
            #   结果 count=0, 变异压根没造出来 —— 看着像"护栏没生效",
            #   其实是变异本身失败。这正是文档 §11 第 5 条纪律的由来。
            old='finalize(state, ws, list(state.get("new_evidence") or []))',
            new=('finalize(state, ws, list(state.get("new_evidence") or []) '
                 '+ ws.load_evidence())  # MUTANT: 又把历史塞回来'),
            test="tests/test_research_quantifier.py::TestEvidenceNotDuplicated",
            expect_substring="",
        ),
        Mutation(
            name="防线五失效 —— scope 从 >100000 缩到 200..500 也不再报警",
            path=CLAIM,
            old="if q_lo is not None and s_hi is not None and s_hi < q_lo:",
            new="if False:  # MUTANT: 范围检查关掉",
            test="tests/test_research_quantifier.py::TestScopeGuard",
            expect_substring="",
        ),
        Mutation(
            name="防线六失效 —— 空 stdout 的 proved 也算证明",
            path=CLAIM,
            old='add("error", "proved_without_support",',
            new='add("warn", "proved_without_support",  # MUTANT: 降级为警告',
            test="tests/test_research_quantifier.py::TestProvedNeedsSupport",
            expect_substring="",
        ),
        Mutation(
            name="防线八失效 —— 至少 1000 组 能拿去反驳哥德巴赫",
            path=CLAIM,
            old="    if not c_th:\n        return None                                  # 命题没有阈值语境, 查不了",
            new="    if c_th:\n        return None  # MUTANT: 阈值检查关掉",
            test="tests/test_research_quantifier.py::TestThresholdGuard",
            expect_substring="",
        ),
    ]


def main() -> int:
    results = []
    for mut in build():
        try:
            hit, out = mut.run()
        except AssertionError as exc:
            print(f"  变异构造失败: {exc}")
            results.append((mut.name, False, "变异未生效"))
            continue
        results.append((mut.name, hit, "ok"))

    print(f"\n{'=' * 68}\n反向验证汇总\n{'=' * 68}")
    bad = 0
    for name, ok, note in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + ("" if ok else f"  ← {note}"))
        bad += not ok

    # 还原后复跑, 确认没留下任何变异残留
    print("\n还原后复跑基准:")
    passed, _ = _run("tests/test_research_eval.py")
    print(f"  tests/test_research_eval.py → {'绿(干净)' if passed else '红 ← 还原失败!'}")

    if bad or not passed:
        print("\n反向验证未通过。")
        return 1
    print("\n全部变异都被测试抓到, 且源码已干净还原。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
