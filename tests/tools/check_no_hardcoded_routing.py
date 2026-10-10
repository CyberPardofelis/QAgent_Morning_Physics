"""★ 守线检查: 新结构里不得出现"题型 → 方法"的硬路由 ★

## 为什么这道检查是整个重构的保险

这次重构的全部意义只有一句话: **方法选择必须是 Explorer 的决定**。

而"硬编码"有两种藏法, 第二种更致命:
  1. Python 里写 `if "组合结构" in claim: use_sat()`
  2. 把同一张表**搬进 prompt 字符串**里

第 2 种看起来"更灵活", 但效果完全一样 —— 方法选择还是被程序做完了,
只是换了个地方。肉眼 review prompt 很难发现, 而模型**一定会**照着那张表走,
于是重构悄悄退化回原样, 而所有指标看上去都正常。

所以这道检查扫的是**所有面向模型的字符串**(prompt 模板), 找:
  · 题型词(组合结构/丢番图/不等式/群论/数值优化/存在性…)
  · "如果…就用…"这种条件句式
  · 工具名与题型同时出现的映射句

## 它会把哪一类正常东西判成异常(先问这个)
`catalog.py` 里每条工具都带 `not_for`(不擅长什么), 那是**工具文档**,
必须留 —— 它对所有问题一视同仁, 不针对任何题型, 所以在豁免名单里。
"probe"/"execute" 是 Worker 的**动作类型**, 不是题型, 也不管。
真正要禁的是"如果题目是 X, 就该用 Y"这种**问题类型 → 方法**的对应。
"""
from __future__ import annotations

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ROLES = os.path.join(ROOT, "agent", "tools", "research", "roles")

#: 题型词。出现即可疑 —— 因为题型不该决定方法。
SHAPE_WORDS = ("组合结构", "丢番图", "不等式", "群论", "数值优化",
               "存在性", "构造型", "反例类", "闭式类", "图论题")

#: 条件句式: "如果…就用…"
COND_PATTERNS = ("如果是", "若题目", "对于.*类问题", "这类问题应该",
                 "遇到.*就用", "应该用.{0,6}solve", "首选")

#: 明确豁免的文件/理由 —— 写清楚为什么豁免, 不许无理由豁免。
EXEMPT = {
    # ★ 理由已更新 (2026-10-10) ★
    #   旧理由: "工具文档(not_for/does), 对所有问题一视同仁, 不针对题型" ——
    #   那说的是**手写的 11 条库名表**。现在 `catalog.py` 只是转发
    #   `_registry.render_catalog()`, 而那份文本来自 MAGI-Tools 的工具描述
    #   (`name` / `description` / `inputSchema`), 对所有问题同样一视同仁。
    "catalog.py": "只转发工具注册表的描述(name/description/inputSchema), "
                  "不含任何题型词, 也不做题型->工具的映射",
    "_registry.py": "桥接层: 只做定位/转发/类型标注, 不含任何提示词文本",
    "_util.py": "只有 LLM 调用与 JSON 抽取, 无 prompt 文本",
    "orchestrator.py": "只做循环与预算; 它连题型词都不该出现",
}

#: 允许出现的词: 动作类型与角色名, 不是题型。
ALLOWED_SUBSTRINGS = ("probe", "execute", "explorer", "worker", "verifier")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """找出所有 docstring 的 lineno。

    ★ 为什么必须排除 docstring ★ 第一版没排除, 结果 `explorer.py` 立刻红了 ——
    因为我在模块 docstring 里举了「组合结构」当**反面例子**。
    文档里描述"不要做什么"和 prompt 里真的**去做**它, 是两回事。
    这道检查要盯的是会被送给模型的那些字符串。
    """
    out: set[int] = set()

    def _mark(body: list) -> None:
        if not body:
            return
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            out.add(first.value.lineno)

    if isinstance(tree, ast.Module):
        _mark(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _mark(node.body)
    return out


def _string_literals(tree: ast.AST) -> list[tuple[int, str]]:
    skip = _docstring_nodes(tree)
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and len(node.value) >= 12
                and getattr(node, "lineno", -1) not in skip):
            out.append((node.lineno, node.value))
    return out


def check_file(path: str) -> list[str]:
    fname = os.path.basename(path)
    if fname in EXEMPT:
        return []
    src = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"{fname}: 语法错误 line {e.lineno}: {e.msg}"]

    problems: list[str] = []
    for lineno, text in _string_literals(tree):
        hits = [w for w in SHAPE_WORDS if w in text]
        if hits:
            problems.append(f"{fname}:{lineno} 出现题型词 {hits} —— "
                            f"题型不该决定方法")
            continue
        cond = [p for p in COND_PATTERNS
                if p.replace(".*", "") in text or (p.endswith("用") and p in text)]
        if cond and any(a in text.lower() for a in ALLOWED_SUBSTRINGS) is False:
            problems.append(f"{fname}:{lineno} 疑似条件路由句 {cond} —— "
                            f"『如果…就用…』是把选型写死")
    return problems


def main() -> int:
    files = sorted(f for f in os.listdir(ROLES) if f.endswith(".py"))
    problems: list[str] = []
    print("=" * 70)
    print("守线检查: roles/ 里不得有『题型 → 方法』的硬路由")
    print("=" * 70)
    for f in files:
        p = check_file(os.path.join(ROLES, f))
        if p:
            problems += p
            print(f"  [!!] {f}")
        else:
            why = EXEMPT.get(f, "")
            print(f"  [OK ] {f}" + (f"   (豁免: {why})" if why else ""))
    print("=" * 70)
    if problems:
        print(f"x 发现 {len(problems)} 处硬路由 —— 重构等于没做")
        for p in problems:
            print("  - " + p)
        return 1
    print("o 未发现硬路由: 方法选择确实留给了 Explorer")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())