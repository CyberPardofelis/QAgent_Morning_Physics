import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from agent.tools.research._sandbox import run_code  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
REF = os.path.abspath(os.path.join(ROOT, "data", "res_16_reference.json"))
INP = os.path.abspath(os.path.join(ROOT, "data", "res_16_tsp.txt"))


def run(code, label):
    tmp = tempfile.mkdtemp()
    try:
        r = run_code(code, tmp, timeout=40)
        out = (r.stdout or "").strip().replace("\n", " | ")[:120]
        err = (r.stderr or "").strip().replace("\n", " | ")[:90]
        print(f"  {label:26} -> {out or '(空)'}")
        if err:
            print(f"  {'':26}    stderr: {err}")
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


print("=" * 74)
print("验 1: numpy 能不能绕过 AST 门读走文件")
print("=" * 74)
run("import numpy as np\n"
    f"a = np.genfromtxt(r'{REF}', dtype=str, encoding='utf8')\n"
    "print('READ', len(a), 'lines; head=', str(a[:3])[:80])\n",
    "np.genfromtxt 读答案文件")
run("import numpy as np\n"
    f"a = np.genfromtxt(r'{INP}', dtype=str, encoding='utf8')\n"
    "print('READ', len(a), 'lines; head=', str(a[:3])[:80])\n",
    "np.genfromtxt 读自己的输入")

print()
print("=" * 74)
print("验 2: 题面/claim 里到底有没有题目的输入数据")
print("=" * 74)
import json  # noqa: E402
import score_baseline as SB  # noqa: E402
import problem_spec as PS  # noqa: E402

for suite in ("R",):
    for it in SB.load_suite(suite):
        cid = it["id"]
        hint = PS.data_hint_for(cid)
        if not hint:
            continue
        claim = PS.build_claim(it["prompt"], it.get("expected"), hint)
        # 题面里有没有真的出现数据本身(而不只是"去读那个文件")
        has_data_inline = any(
            tok in claim for tok in ("U:  R W W", "[0, 1, 3, 7", "x =", "1 2 3"))
        print(f"  {cid:36} hint 提到文件={'data/' in hint} "
              f"claim 里含数据={has_data_inline}")
        print(f"       hint 内容: {hint[:100]}")

print()
print("=" * 74)
print("验 3: 制品门是不是只管 PROVED 不管 REFUTED")
print("=" * 74)
src = open(os.path.join(ROOT, "agent", "tools", "research",
                        "research_verify.py"), encoding="utf-8").read()
print('  `ev.get("verdict") == "proved" and wanted_fields` 出现:',
      'ev.get("verdict") == "proved" and wanted_fields' in src)
print('  → REFUTED 的制品要求只有 _provenance 的凭证门, 没有 answer_fields 那一路')