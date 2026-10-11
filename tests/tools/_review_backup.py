"""一次性工具: 把当前架构(v1-single-shot)的原始证据快照一份。

为什么必须做: 跑测脚本每道题开头 shutil.rmtree(ws) 清工作区,
证据只存在磁盘上, 一旦被清掉就永久没了 —— res-019 那份 46454 字符的
原始 stdout 就是这么丢的(只剩 8000 上限截断后的版本)。
架构重构之前, 这是旧架构唯一的原始记录。
"""
import io
import json
import os
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "research")
DST = os.path.join(SRC, "_snapshots", "2026-10-09-baseline-v1")

if os.path.isdir(DST):
    print(f"快照已存在, 跳过: {os.path.relpath(DST, ROOT)}")
    raise SystemExit(0)

os.makedirs(DST, exist_ok=True)
n_files = 0

src_work = os.path.join(SRC, "working")
if os.path.isdir(src_work):
    shutil.copytree(src_work, os.path.join(DST, "working"))
    n_files = sum(len(f) for _, _, f in os.walk(os.path.join(DST, "working")))

for name in ("r_scored.json", "all60_scored.json", "r_baseline2.log",
             "all60.log", "all60_process.json"):
    p = os.path.join(SRC, name)
    if os.path.isfile(p):
        shutil.copy2(p, os.path.join(DST, name))
        n_files += 1

manifest = {
    "taken_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "why": "旧架构(单发 codegen)的唯一原始证据记录; 架构重构前必须冻结",
    "arch_version": "v1-single-shot",
    "note": "所有 *_evidence.jsonl 都是 MAX_OUTPUT_CHARS=8000 截断后的 stdout, "
            "不是原始输出。截断问题已记录在 Conclude_DeepSeek.md 附录 A.2。",
    "scored": {
        "r_scored": "ok=9 wrong=8 unverified=3",
    },
}
io.open(os.path.join(DST, "_SNAPSHOT.json"), "w", encoding="utf-8").write(
    json.dumps(manifest, ensure_ascii=False, indent=2))

size = sum(os.path.getsize(os.path.join(r, f))
           for r, _, fs in os.walk(DST) for f in fs)
print(f"快照完成: {n_files} 个文件, {size/1024:.0f} KB")
print(f"位置: {os.path.relpath(DST, ROOT)}")