"""机械扫描:真正进入 prompt 的文本里, 有没有硬编码的答案。

★ 为什么用脚本而不是靠人读 ★
今晚的事故正是"人读漏了"—— 我把 expected 的字段名写进 prompt, 又忘了
和另一处同步。人眼读长文本必漏, 脚本不会。

扫描对象: build_claim 拼出来的**完整 claim 字符串**(那才是真正进 prompt 的),
外加 prompt 里所有静态常量块。
判据: 题库/参考文件里出现过的**具体数值**, 是不是原样出现在 claim 里。
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import problem_spec as PS  # noqa: E402
import score_baseline as SB  # noqa: E402

#: 题面里天然出现的数字(规模标记)不算泄露, 只查"答案型"的数字:
#: 出现在 *_reference.json 或 expected 里的具体答案数值。
def reference_numbers() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    d = os.path.join("data")
    for f in os.listdir(d):
        if not f.endswith("_reference.json"):
            continue
        raw = io.open(os.path.join(d, f), encoding="utf-8").read()
        nums = set(re.findall(r"(?<![\w.])\d{2,}(?![\w.])", raw))
        if nums:
            out[f] = nums
    return out


def main() -> int:
    items = SB.load_suite("R")
    refs = reference_numbers()
    print("=" * 78)
    print("机械扫描: 真正进入 prompt 的 claim 里有没有硬编码答案")
    print("=" * 78)

    leaks: list[str] = []
    for it in items:
        cid = it["id"]
        raw_prompt = it["prompt"]
        claim = PS.build_claim(it["prompt"], it.get("expected"),
                               PS.data_hint_for(cid))
        # ★ 只看"题面之外新增的部分" ★
        # 否则会自匹配: 我自己的占位符里就有 "True" 这三个字母,
        # 拿 "true" 去 claim 里 find 必然命中 —— 第一版就是这么误报的。
        added = claim.replace(raw_prompt, "", 1)
        exp = it.get("expected") or {}
        val = exp.get("value") if isinstance(exp, dict) else None
        if not isinstance(val, dict):
            continue
        for k, v in val.items():
            if isinstance(v, bool):
                continue          # 布尔值已统一脱敏成占位符, 单独验证那个函数
            if isinstance(v, (int, float)):
                c = str(v)
                if len(c) < 2:
                    continue
                if re.search(r"(?<![\w.])" + re.escape(c) + r"(?![\w.])", added):
                    leaks.append(f"{cid}.{k} = {c} 出现在题面之外新增的部分里")

    if leaks:
        print(f"\n!! 发现 {len(leaks)} 处:")
        for l in leaks:
            print("   -", l)
    else:
        print("\n✓ 没有 expected 里的答案数值被写进 claim(题面之外新增的部分)")

    # 布尔脱敏单独验: 直接调 _redact
    print("\n" + "-" * 78)
    print("布尔值脱敏(直接测 _redact)")
    bad_bool = []
    for it in items:
        exp = it.get("expected") or {}
        val = exp.get("value") if isinstance(exp, dict) else None
        if not isinstance(val, dict):
            continue
        for k, v in val.items():
            if isinstance(v, bool):
                r = PS._redact(v)
                if r.lower() == str(v).lower():
                    bad_bool.append(f"{it['id']}.{k}")
    if bad_bool:
        print("   !! 布尔没脱敏:", bad_bool)
    else:
        print("   ✓ 所有布尔值都脱敏成占位符")

    # 第二个检查: *_reference.json 的**完整文件名**有没有出现在 prompt 里
    print("\n" + "-" * 78)
    print("参考文件完整路径是否出现在 prompt 里")
    bad_ref = []
    for it in items:
        claim = PS.build_claim(it["prompt"], it.get("expected"),
                               PS.data_hint_for(it["id"]))
        for f in refs:
            # ★ 必须匹配完整文件名 ★ 只匹配 stem 会自命中:
            # stem 是 "res_05", 而输入文件叫 "res_05_sequence.txt", 前缀相同。
            if f in claim:
                bad_ref.append(f"{it['id']}: claim 里出现了 {f}")
    if bad_ref:
        for b in bad_ref:
            print("   !!", b)
    else:
        print("   ✓ 没有任何 _reference.json 文件名进入 prompt")

    # 第三个检查: data_hint 里说的文件名是否真的存在且是输入文件
    print("\n" + "-" * 78)
    print("data_hint 引用的文件是否存在")
    import re as _re
    for it in items:
        hint = PS.data_hint_for(it["id"])
        if not hint:
            continue
        for m in _re.finditer(r"data/([\w./]+)", hint):
            p = os.path.join("data", m.group(1))
            ok = os.path.isfile(p)
            ref = "reference" in m.group(1)
            print(f"   {it['id']:38} {m.group(1):26} "
                  f"{'存在' if ok else '缺失'}"
                  f"{'  <-- 这是参考答案, 不该给模型!' if ref else ''}")
            if not ok or ref:
                leaks.append(f"{it['id']}: data_hint 指向 {m.group(1)}")

    print("\n" + "=" * 78)
    print(f"结论: {len(leaks)} 处问题" if leaks else "结论: 干净")
    print("=" * 78)
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())