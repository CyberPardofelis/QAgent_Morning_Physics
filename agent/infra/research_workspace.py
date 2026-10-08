"""
agent.infra.research_workspace — 单个科研验证任务的工作区 (P7 阶段 0)

对标 agent/infra/survey_workspace.py: 同构的"一个任务一个目录 + 中间产物全落盘",
好处是**任何时候进程崩了/被 Ctrl-C 了, 已有证据还在磁盘上**, 可以断点续跑,
也可以让我(调试方)直接读文件核对 LLM 的说法, 不靠它的自述。

目录: data/research/working/{question 的 SHA1 前 12 位}/
├── _plan.json       建模结果: 问题 → 可判定命题        (P2 才有内容)
├── _specs.jsonl     每行一条待验命题
├── _evidence.jsonl  每行一条证据 (代码/输出/判定/环境指纹) ← 唯一事实源
├── _attempts.jsonl  每行一次失败尝试的**错误原文** ← 喂回 LLM 的燃料
├── _status.json     进度/当前步/token 花费
├── _report.md       最终结论 + 证据链
└── scratch/         沙箱唯一可写目录 (S0 执行器在这跑)

公开 API:
  hash_question(q) -> str
  ResearchWorkspace(task_hash)
      .save_plan(d) / .load_plan()
      .save_specs(list) / .append_spec(d) / .load_specs() / .clear_specs()
      .append_evidence(d) -> int        (返回当前总条数, 下一条 index 用)
      .load_evidence()
      .append_attempt(d) / .load_attempts()
      .set_status(...) / .load_status()
      .save_report(md) / .load_report()
      .scratch_dir  -> Path
      .summary()
  list_workspaces()

历史:
  - 2026-10-08 P7.0 首版
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 数据根目录 —— 与 surveys 对称
_DEFAULT_ROOT = Path("data/research/working")


def hash_question(question: str) -> str:
    """question 的 SHA1 前 12 位, 作为工作区目录名 (对标 hash_topic)"""
    if not question:
        return f"empty_{int(time.time())}"
    return hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:12]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ResearchWorkspace:
    """
    单个科研验证任务的工作区。

    用法:
        ws = ResearchWorkspace(hash_question("验证黄金分割比"), question="...")
        ws.append_spec({"claim_text": "...", "scope": "n=1..1000"})
        n = ws.append_evidence(ev)          # n = 1, 下一条 index 是 1
        ws.set_status(phase="verify", step=1, total=3)
    """

    def __init__(
        self,
        task_hash: str,
        root: Path | None = None,
        question: str = "",       # 写 _status.json 时留个可读的原文
    ) -> None:
        self.task_hash = task_hash
        self.question = question
        self.root = Path(root) if root else _DEFAULT_ROOT
        self.path = self.root / task_hash
        self.path.mkdir(parents=True, exist_ok=True)

    # ============== _plan.json ==============
    def save_plan(self, plan: dict[str, Any]) -> None:
        """建模结果: 问题 → 可判定命题 + 每条的验证方法建议 (P2 填)"""
        payload = {
            "question": self.question,
            "task_hash": self.task_hash,
            "updated_at": _now(),
            **(plan or {}),
        }
        (self.path / "_plan.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_plan(self) -> dict[str, Any] | None:
        p = self.path / "_plan.json"
        if not p.exists():
            return None
        return _safe_load_json(p)

    # ============== _specs.jsonl ==============
    def append_spec(self, spec: dict[str, Any]) -> int:
        """追加一条待验命题, 返回当前条数"""
        return self._append_jsonl("_specs.jsonl", spec)

    def save_specs(self, specs: list[dict[str, Any]]) -> None:
        """覆盖式保存 (重新建模时用)"""
        self.clear_specs()
        for s in specs:
            self.append_spec(s)

    def load_specs(self) -> list[dict[str, Any]]:
        return self._read_jsonl("_specs.jsonl")

    def clear_specs(self) -> int:
        """清空并返回清掉的条数"""
        p = self.path / "_specs.jsonl"
        n = len(self._read_jsonl("_specs.jsonl"))
        if p.exists():
            p.unlink()
        return n

    # ============== _evidence.jsonl ==============
    def append_evidence(self, evidence: dict[str, Any]) -> int:
        """
        追加一条证据, 返回追加后的总条数。

        **永不覆盖**: 同一条命题验 10 次就是 10 条证据摆在那儿,
        结论前后不一致时才查得出来 (与 facts 的 fact/fact_source 同思路)。
        """
        return self._append_jsonl("_evidence.jsonl", evidence)

    def load_evidence(self) -> list[dict[str, Any]]:
        return self._read_jsonl("_evidence.jsonl")

    def evidence_count(self) -> int:
        return len(self._read_jsonl("_evidence.jsonl"))

    # ============== _attempts.jsonl ==============
    def append_attempt(self, attempt: dict[str, Any]) -> int:
        """
        记录一次失败尝试的**错误原文**。

        这份文件是科研闭环的燃料: 下一轮把错误喂回 LLM 让它改。
        所以只存错误原文就够了, 存了也别改 —— LLM 反复犯同一个错是重要信号。
        """
        return self._append_jsonl("_attempts.jsonl", attempt)

    def load_attempts(self) -> list[dict[str, Any]]:
        return self._read_jsonl("_attempts.jsonl")

    # ============== _status.json (持续更新) ==============
    def set_status(
        self,
        phase: str = "",
        step: int = 0,
        total: int = 0,
        message: str = "",
        **extra: Any,
    ) -> None:
        payload = {
            "question": self.question,
            "task_hash": self.task_hash,
            "updated_at": _now(),
            "phase": phase,
            "step": step,
            "total": total,
            "message": message,
            **extra,
        }
        (self.path / "_status.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_status(self) -> dict[str, Any] | None:
        p = self.path / "_status.json"
        if not p.exists():
            return None
        return _safe_load_json(p)

    # ============== _report.md ==============
    def save_report(self, report_md: str) -> None:
        (self.path / "_report.md").write_text(report_md or "", encoding="utf-8")

    def load_report(self) -> str | None:
        p = self.path / "_report.md"
        return p.read_text(encoding="utf-8") if p.exists() else None

    # ============== scratch/ (沙箱唯一可写处) ==============
    @property
    def scratch_dir(self) -> Path:
        d = self.path / "scratch"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def clear_scratch(self) -> None:
        """清空 scratch —— 每次验证跑完清一次, 别让 main.py 堆积"""
        d = self.path / "scratch"
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()

    # ============== 元信息 ==============
    def summary(self) -> dict[str, Any]:
        """工作区摘要 (UI 列表 / 调试时先看这个)"""
        st = self.load_status() or {}
        evs = self.load_evidence()
        verdicts: dict[str, int] = {}
        for e in evs:
            v = e.get("verdict", "?")
            verdicts[v] = verdicts.get(v, 0) + 1
        return {
            "task_hash": self.task_hash,
            "question": self.question or st.get("question", ""),
            "phase": st.get("phase", ""),
            "step": st.get("step", 0),
            "total": st.get("total", 0),
            "updated_at": st.get("updated_at", ""),
            "specs": len(self.load_specs()),
            "evidence": len(evs),
            "attempts": len(self.load_attempts()),
            "verdicts": verdicts,
            "path": str(self.path),
        }

    # ============== 内部 ==============
    def _append_jsonl(self, name: str, row: dict[str, Any]) -> int:
        p = self.path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return self._count_lines(name)

    def _read_jsonl(self, name: str) -> list[dict[str, Any]]:
        p = self.path / name
        if not p.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue      # 半行(进程被杀在写入中途)跳过, 不让整个文件读不出来
        return out

    def _count_lines(self, name: str) -> int:
        return len(self._read_jsonl(name))


def list_workspaces(root: Path | None = None) -> list[dict[str, Any]]:
    """所有工作区摘要, 按更新时间倒序 (UI 列表 / 调试巡检用)"""
    base = Path(root) if root else _DEFAULT_ROOT
    if not base.exists():
        return []
    items: list[dict[str, Any]] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        try:
            ws = ResearchWorkspace(d.name, root=base)
            items.append(ws.summary())
        except Exception:
            continue
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return items


def _safe_load_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
