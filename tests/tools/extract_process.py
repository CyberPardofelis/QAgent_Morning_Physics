"""从落盘证据生成 60 题的「过程 + 结果」清单。

★ 为什么要从证据生成而不是手写 ★
手写 60 条必然记错、漏项, 而且改一轮就得重写。这里只抽取**可核对的字段**,
叙述性判断留给人在文档里加。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import score_baseline as SB  # noqa: E402

TAG = {"base": "b", "V2": "v2", "R": "r"}
RANK = {"failed": 0, "untested": 1, "inconclusive": 2,
        "partial": 3, "refuted": 4, "proved": 5}
_STRAT = re.compile(r"^\s*#\s*STRATEGY\s*:\s*([a-z_]+)\s*$",
                    re.IGNORECASE | re.MULTILINE)
_IMPORT = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)",
                     re.MULTILINE)


def th_of(suite: str, cid: str) -> str:
    return TAG[suite] + "_" + hashlib.sha1(cid.encode()).hexdigest()[:10]


def load(suite: str, cid: str) -> tuple[list[dict], list[dict]]:
    d = os.path.join("data", "research", "working", th_of(suite, cid))
    out = []
    for name, k in (("_evidence.jsonl", "ev"), ("_attempts.jsonl", "at")):
        p = os.path.join(d, name)
        recs = ([json.loads(x) for x in io.open(p, encoding="utf-8") if x.strip()]
                if os.path.isfile(p) else [])
        out.append(recs)
    return out[0], out[1]


def brief(rec: dict, limit: int = 150) -> str:
    s = (rec.get("reason") or rec.get("stdout") or "").strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s[:limit] + ("…" if len(s) > limit else "")


def libs_of(code: str) -> list[str]:
    return sorted({m.split(".")[0] for m in _IMPORT.findall(code or "")})


def strat_of(code: str) -> str:
    m = _STRAT.search(code or "")
    return m.group(1).lower() if m else "(未声明)"


def main() -> None:
    scored = {}
    p = "data/research/all60_scored.json"
    if os.path.isfile(p):
        for r in json.load(io.open(p, encoding="utf-8"))["rows"]:
            scored[(r["suite"], r["id"])] = r

    out = []
    for suite in ("base", "V2", "R"):
        items = SB.load_suite(suite)
        for it in items:
            cid = it["id"]
            ev, at = load(suite, cid)
            row = scored.get((suite, cid), {})
            cat = row.get("category", "?")
            mark = {"ok": "✅", "wrong": "❌", "self_reported": "➖",
                    "unverified": "❓"}.get(cat, "?")

            prompt = (it.get("prompt") or "").replace("\n", " ").strip()
            exp = it.get("expected") or {}
            val = exp.get("value") if isinstance(exp, dict) else None
            fields = list(val.keys()) if isinstance(val, dict) else []

            out.append({
                "suite": suite, "id": cid, "mark": mark, "cat": cat,
                "prompt": prompt[:110],
                "ask_fields": fields,
                "n_ev": len(ev), "n_attempt": len(at),
                "seconds": row.get("seconds"),
                "tokens": row.get("tokens"),
                "agent_verdict": row.get("agent_verdict"),
                "judged_by": row.get("judged_by"),
                "score": row.get("score"),
                "weak": row.get("weak_dims"),
                "cert": (row.get("cert") or "")[:160],
                "rounds": [
                    {
                        "i": r.get("attempt"),
                        "v": r.get("verdict"),
                        "strat": strat_of(r.get("code")),
                        "libs": libs_of(r.get("code")),
                        "ms": r.get("elapsed_ms"),
                        "crash": bool(r.get("crashed_with_output")),
                        "why": brief(r),
                    } for r in ev
                ],
                "attempts": [
                    {
                        "i": a.get("attempt"),
                        "type": a.get("error_type"),
                        "detail": brief(a, 130),
                    } for a in at
                ],
            })

    io.open("data/research/all60_process.json", "w",
            encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"已生成 {len(out)} 条 -> data/research/all60_process.json")
    for suite in ("base", "V2", "R"):
        rs = [r for r in out if r["suite"] == suite]
        print(f"  {suite}: {len(rs)} 题")


if __name__ == "__main__":
    main()