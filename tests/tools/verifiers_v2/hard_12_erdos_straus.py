"""Erdos-Straus 猜想的有界验证器（基准 hard-012）。

题目：在有穷范围内验证：对每个整数 2 <= n <= 1000，
方程 4/n = 1/x + 1/y + 1/z 存在正整数解 (x, y, z)。

文件顶部的公共签名
------------------
    def find_solution(n) -> tuple
        对单个 n 做完备枚举，返回 (x, y, z)；无解或搜索预算耗尽时抛异常
        （NoSolution 表示枚举完备却未命中；BudgetExhausted 表示没能搜完）。
        ——「没找到」与「不存在」在此处是两种不同状态。

    def verify() -> dict
        返回 {"verdict": str, "detail": str, "witness": dict}
        verdict ∈ {"proved", "refuted", "inconclusive"}
        proved       = 范围内每个 n 都找到解，且每个解都代回原式校验通过
        refuted      = 完备枚举跑完且确实找不到解（真实反例）
        inconclusive = 搜索预算耗尽 / 异常，未能完成全覆盖

用法
----
    python tests/tools/verifiers_v2/hard_12_erdos_straus.py

相对 LLM 常见翻车点的显式防护
----------------------------
* 「没找到」不等于「不存在」：只有 _search_one 返回 None（枚举被证明完备）
  才允许输出 refuted；预算耗尽一律 inconclusive。
* 「已验证数量」只在找到解并代回原式校验通过后才 +1，
  杜绝“已验证 0 个却给出确定结论”的自相矛盾输出。
* 全部算术用 int / Fraction，不用浮点，避免精度误差造出假反例。
"""

from fractions import Fraction

N_MIN = 2
N_MAX = 1000

# 单个 n 的 (x, y) 枚举步数预算。超出只说明这一遍没搜完，
# 不构成 n 无解的证据。
SEARCH_BUDGET = 2_000_000


class NoSolution(Exception):
    """完备枚举已跑完，确实不存在解。"""


class BudgetExhausted(Exception):
    """枚举未跑完，结论待定。"""


def _search_one(n, budget=SEARCH_BUDGET):
    """对单个 n 做完备枚举，返回 (x, y, z)；无解返回 None。

    完备性论证
    ----------
    任意解都可重排为 x <= y <= z（方程对三个变量完全对称）。

    1) x 的范围恰为 [n//4 + 1, 3n//4]：
       - 1/x <= 4/n（1/x 是三项中最大项，故不超过总和）=> x >= n/4；
         x = n/4 时余项为 0、不可能，故 x >= n//4 + 1。
       - 1/x + 1/y + 1/z <= 3/x（y, z >= x => 1/y, 1/z <= 1/x）
         => 4/n <= 3/x => x <= 3n//4。
       ==> 任何解的 x 都落在枚举区间内，不漏解。

    2) 固定 x 后余项 r = 4/n - 1/x = p/q（既约，p, q > 0），需 1/y + 1/z = r：
       - r = 1/y + 1/z > 1/z 无关，此处用 r > 1/y（z 有限、1/z > 0）
         => y > q/p => y >= q//p + 1。
       - r = 1/y + 1/z <= 2/y（z >= y => 1/z <= 1/y）=> y <= 2q//p。
       再与 y >= x 取交集。==> 任何解的 y 都落在枚举区间内，不漏解。

    3) 给定 (x, y)，z 由方程唯一确定：
       p/q = 1/y + 1/z => p*y*z = q*z + q*y => z = q*y / (p*y - q)。
       分母 p*y - q > 0（由 y > q/p 保证），若整除则 z 唯一。
       ==> 不存在“漏掉的 z”：每个 (x, y) 至多一个合法 z。
    """
    total = Fraction(4, n)
    steps = 0

    for x in range(n // 4 + 1, (3 * n) // 4 + 1):
        r = total - Fraction(1, x)
        if r <= 0:
            continue
        p, q = r.numerator, r.denominator  # Fraction 保证既约且为正

        y_lo = max(x, q // p + 1)
        y_hi = (2 * q) // p
        if y_lo > y_hi:
            continue

        for y in range(y_lo, y_hi + 1):
            steps += 1
            if steps > budget:
                raise BudgetExhausted("n=%d 枚举步数超过预算 %d" % (n, budget))
            denom = p * y - q
            num = q * y
            if num % denom:
                continue
            z = num // denom
            if z < y:  # 违反 y <= 2q/p 假设的数值，不可取
                continue
            return (x, y, z)

    return None  # 上面所有 (x, y) 都试完了仍无解 —— 真正的“无解”


def find_solution(n):
    """返回 (x, y, z)；完备枚举后确认无解则抛 NoSolution，预算耗尽抛 BudgetExhausted。"""
    sol = _search_one(n)
    if sol is None:
        raise NoSolution("n=%d 在完备枚举范围内无正整数解" % n)
    x, y, z = sol
    if Fraction(1, x) + Fraction(1, y) + Fraction(1, z) != Fraction(4, n):
        raise AssertionError("自校验失败：n=%d, (x,y,z)=%s" % (n, sol))
    return sol


def verify() -> dict:
    """返回 {"verdict", "detail", "witness"}。见文件顶部签名说明。"""
    verified = 0
    witness = {}
    failures = []
    budget_hit = []

    for n in range(N_MIN, N_MAX + 1):
        try:
            x, y, z = find_solution(n)   # 内部已代回原式自校验
        except BudgetExhausted:
            budget_hit.append(n)
            continue
        except NoSolution:
            failures.append(n)
            continue
        verified += 1
        if n in (2, 3, 4, 5, 7, 997, 999, 1000):
            witness["n=%d" % n] = [x, y, z]

    expected = N_MAX - N_MIN + 1
    detail = "已验证的 n 数量: %d / %d (范围 %d..%d)" % (verified, expected, N_MIN, N_MAX)

    if verified == expected:
        return {
            "verdict": "proved",
            "detail": detail + "；全部解均代回原式 1/x+1/y+1/z==4/n 校验通过。",
            "witness": witness,
        }

    if budget_hit:
        # 关键：没搜完就不能下结论，哪怕一个反例都没找到。
        return {
            "verdict": "inconclusive",
            "detail": detail + "；%d 个 n 因搜索预算耗尽未定论（首个: n=%d），"
                              "预算耗尽不构成任何存在/不存在结论。" % (len(budget_hit), budget_hit[0]),
            "witness": witness,
        }

    if failures:
        return {
            "verdict": "refuted",
            "detail": detail + "；完备枚举后确认无解的 n: %s" % (failures,),
            "witness": witness,
        }

    return {
        "verdict": "inconclusive",
        "detail": detail + "；存在未归类的中间状态，不下结论。",
        "witness": witness,
    }


if __name__ == "__main__":
    result = verify()
    print("检查范围: %d 到 %d" % (N_MIN, N_MAX))
    print(result["detail"])
    if result["witness"]:
        print("抽样见证 (n -> [x, y, z]):")
        for k in sorted(result["witness"], key=lambda s: int(s.split("=")[1])):
            print("  %s -> %s" % (k, result["witness"][k]))
    print("VERDICT: %s" % result["verdict"].upper())