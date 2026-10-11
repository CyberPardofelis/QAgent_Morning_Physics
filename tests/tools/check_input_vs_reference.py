"""交叉核对: 题面内联的输入数据 vs 验证器自建的判定基准。

## 为什么必须查

res-005 / res-014 的验证器是子代理写的, 当时 `data/res_05_seq.txt` 等文件
**还不存在**, 所以验证器自带了一个固定实例。后来这轮把输入数据补齐了
(`build_res_data2.py` 写了 res_05_seq.txt / res_07_seq.txt / res_14_cube.txt),
于是现在的局面是:

    题面发给 agent 的  ->  data/res_05_sequence.txt 里那 200 项
    验证器拿来判对错的 ->  res_05_berlekamp.SEQUENCE (代码里 make_sequence 生成的)

**如果这两条不是同一个序列, 验证器就在判一道跟题面无关的题** ——
agent 交对了也会被判错, 而且表现和"模型不会"一模一样。

res-007 的验证器已经写成"文件存在就用文件", 所以它天然一致;
res-005 / res-014 是自建实例, 必须显式核对。
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "verifiers_r"))
sys.path.insert(0, ROOT)


def _ints_in(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", text)]


def check_res005() -> str:
    import res_05_berlekamp as V
    txt = open(os.path.join(ROOT, "data", "res_05_sequence.txt"),
               encoding="utf-8").read()
    nums = _ints_in(txt)
    # 文件里混着题面里的数字(模数 2147483647 / 系数说明), 先看能不能直接对上
    seq = V.SEQUENCE
    # 找文件里那一段连续 200 项的整数
    hit = None
    for L in range(len(nums) - 199):
        if nums[L:L + 200] == seq:
            hit = L
            break
    if hit is not None:
        return (f"res-005 [OK] 题面数据与验证器 SEQUENCE 完全一致 "
                f"(在文件第 {hit} 个数字处起连续 200 项)")
    # 不一致时给出第一处差异, 便于定位是"数据写错了"还是"用了不同实例"
    detail = ""
    for L in range(len(nums) - 199):
        seg = nums[L:L + 200]
        d = next((i for i in range(200) if seg[i] != seq[i]), None)
        if d is None:
            continue
        detail = (f"最像的一段起于第 {L} 个数字, 第 {d} 项起不一致: "
                  f"题面={seg[d]} 验证器={seq[d]}")
        break
    return f"res-005 [!!] 题面数据与验证器 SEQUENCE 不一致 —— {detail or '(找不到 200 项连续段)'}"


def check_res014() -> str:
    """res-014: 题面发的是**颜色**, 验证器内部是**贴纸编号**, 要证明是同一个魔方。

    转换: `_build_stickers` 的面序是 ("U","D","F","B","R","L"), 与题面声明的
    "面序 U D F B R L" 一致, 所以 sticker_index // 9 就是这个贴纸的**家面**。
    于是一个状态里每个位置的颜色 = 该位置贴纸的家面。

    我**不**去猜"哪个颜色对应哪个面"(那是题库自己的约定), 而是反过来:
    若存在一个"字母 -> 家面"的双射能把 54 个位置全对上, 两者就是同一个状态。
    双射不存在 = 题面发的魔方和验证器判的不是同一个 -> agent 交对了也判错。
    """
    import res_14_rubiks as V
    txt = open(os.path.join(ROOT, "data", "res_14_cube.txt"),
               encoding="utf-8").read()
    faces = ("U", "D", "F", "B", "R", "L")
    state = list(V._scramble_state())          # 54 个贴纸编号

    # --- 解析题面: "U:  R W W   R W W   Y W W" -> 9 个字母 ---
    # ★ 只收"恰好 9 个单字符"的行 ★ 题面后面还有 `U: 法向 (0,1,0)...` 这种行,
    # 它也匹配 `X:` 但不是状态块。第一版没做这个过滤, 把法向行当成了状态行。
    letters: list[str] = []
    for line in txt.splitlines():
        m = re.match(r"\s*([UDLRFB]):\s+(.*)$", line.strip())
        if not m:
            continue
        cells = [c for c in m.group(2).split() if len(c) == 1 and c.isalpha()]
        if len(cells) != 9:
            continue                      # 法向/方向说明行, 跳过
        letters.extend(cells)
        if len(letters) == 54:
            break
    if len(letters) != 54:
        return f"res-014 [??] 题面只解析出 {len(letters)} 个色块(应为 54), 需人工看格式"

    # --- 题面每个位置的家面(由验证器给出) ---
    home = [faces[s // 9] for s in state]

    # --- 求"字母 -> 家面"的一致性 ---
    m2f: dict[str, set[str]] = {}
    for lt, hf in zip(letters, home):
        m2f.setdefault(lt, set()).add(hf)
    bad = {lt: sorted(v) for lt, v in m2f.items() if len(v) != 1}
    if bad:
        first = sorted(bad.items())[0]
        return (f"res-014 [!!] 同一个颜色出现在不同家面上({first[0]} -> {first[1]})"
                f" —— 题面与验证器不是同一个魔方")
    img = {lt: next(iter(v)) for lt, v in m2f.items()}
    if sorted(img.values()) != sorted(faces):
        return (f"res-014 [!!] 颜色与家面不是双射({img}) —— 题面与验证器不是同一个魔方")

    pretty = " ".join(f"{k}->{v}" for k, v in sorted(img.items()))
    return (f"res-014 [OK] 题面 54 个色块与验证器 _scramble_state() 是同一个魔方 "
            f"(颜色↔家面 双射: {pretty})")


def check_res007_uses_file() -> str:
    """res-007 的验证器到底读的是文件还是内置实例?"""
    import res_07_oeis as V
    seq, src = V.target_sequence()
    if src.endswith(".txt"):
        return f"res-007 [OK] 验证器读的是文件({src}), 与题面同源"
    return (f"res-007 [!!] 验证器**没读文件**, 退回内置实例({src}) —— "
            f"题面给的 30 项和它判的不是一个序列")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    print("=" * 70)
    print("题面输入数据 vs 验证器判定基准 —— 交叉核对")
    print("=" * 70)
    bad = 0
    for fn in (check_res005, check_res007_uses_file, check_res014):
        line = fn()
        bad += line.count("[!!]")
        print("  " + line)
    print("=" * 70)
    sys.exit(1 if bad else 0)