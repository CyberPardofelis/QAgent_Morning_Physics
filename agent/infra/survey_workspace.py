"""
agent.infra.survey_workspace — 综述生成(P6)工作区管理

设计:
  - 每次 survey_generate(topic, ...) 跑下来,所有中间产物落到
    data/surveys/working/{topic_hash}/ 下一个独立目录
  - 目录名用 topic 的 SHA1 前 12 位(防特殊字符路径问题)
  - 5 个文件:
      _pool.json      — 候选池 (Phase A)
      _facts.jsonl    — 事实提取结果 (Phase C, 每行 1 个 fact)
      _outline.json   — 大纲 (Phase D)
      _status.json    — 运行状态 (进度条/恢复用, 持续更新)
      _audit.md       — 覆盖度审计 (Phase F)

公开 API:
  hash_topic(topic) -> str
  SurveyWorkspace(topic_hash) 类
      .save_pool(papers), .load_pool() -> List[dict]
      .append_fact(fact), .load_facts() -> List[dict]
      .save_outline(outline), .load_outline() -> dict
      .set_status(phase, progress, total, message)
      .save_audit(audit_md)
      .path  (Path 对象, 仅读)

  list_workspaces() -> List[dict]
      (供 UI surveys_dialog 列表用, 返回按时间倒序的所有 workspace 摘要)

历史:
  - 2026-09-05: P6.2 首版
"""
import contextlib
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 数据根
_DEFAULT_ROOT = Path("data/surveys/working")


def hash_topic(topic: str) -> str:
    """topic 字符串 → 12 位 hash,作为工作区目录名"""
    if not topic:
        return f"empty_{int(time.time())}"
    return hashlib.sha1(topic.strip().encode("utf-8")).hexdigest()[:12]


class SurveyWorkspace:
    """
    单个 survey 生成任务的工作区。

    使用:
        ws = SurveyWorkspace("a1b2c3d4e5f6")
        ws.save_pool([{...}, {...}])
        ws.append_fact({...})
        ws.save_outline({...})
    """

    def __init__(
        self,
        topic_hash: str,
        root: Path | None = None,
        topic: str = "",  # 写 _status.json 时留个可读 topic, 不强制
    ):
        self.topic_hash = topic_hash
        self.topic = topic
        self.root = Path(root) if root else _DEFAULT_ROOT
        self.path = self.root / topic_hash
        self.path.mkdir(parents=True, exist_ok=True)

    # ============== _pool.json ==============
    def save_pool(self, papers: list[dict[str, Any]]) -> None:
        p = self.path / "_pool.json"
        payload = {
            "topic": self.topic,
            "topic_hash": self.topic_hash,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "size": len(papers),
            "papers": papers,
        }
        p.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_pool(self) -> list[dict[str, Any]]:
        p = self.path / "_pool.json"
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data.get("papers", [])
        except Exception:
            return []

    # ============== _facts.jsonl ==============
    def append_fact(self, fact: dict[str, Any]) -> None:
        p = self.path / "_facts.jsonl"
        # 每行一条 fact, json 序列化
        line = json.dumps(fact, ensure_ascii=False)
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def clear_facts(self) -> int:
        """清空 _facts.jsonl, 返回清掉的条数。

        9-30 加: 之前只 append 不清, 于是**同一个主题重跑会累积 fact**。
        实测 workspace 里堆到 1089 行, 同一篇论文积累了 42 条(配额本来 4-12),
        跨运行的陈旧 fact 和新一轮的混在一起。

        这个坑还间接造成过一次事故: 某次 fact 抽取全军覆没(抽了 0 条),
        `extract_state["facts"] or ws.load_facts()` 静默端上了历史累积的 685 条,
        综述照常产出、**看起来完全正常, 实际建在旧数据上**。

        一轮 survey 应该产出它自己那一轮的 fact, 所以抽取开始前先清。
        (phase=write 这类续跑不调抽取, 不受影响。)
        """
        p = self.path / "_facts.jsonl"
        if not p.exists():
            return 0
        try:
            n = sum(1 for _ in p.open("r", encoding="utf-8"))
        except OSError:
            return 0
        p.write_text("", encoding="utf-8")
        return n

    def load_facts(self) -> list[dict[str, Any]]:
        p = self.path / "_facts.jsonl"
        if not p.exists():
            return []
        facts = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                facts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return facts

    # ============== _claims.json ==============
    def save_claims(self, claims: list[dict[str, Any]]) -> None:
        """落命题层产物 (fact 合并后的命题)。

        10-02 崩溃续跑: 命题合并要 1000+ 秒, 丢了重跑很贵。
        """
        p = self.path / "_claims.json"
        p.write_text(
            json.dumps({"topic": self.topic, "created_at":
                        datetime.now().isoformat(timespec="seconds"),
                        "size": len(claims), "claims": claims},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_claims(self) -> list[dict[str, Any]] | None:
        """读命题层产物。**没有文件返回 None**(区别于"有文件但空列表")。"""
        p = self.path / "_claims.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            claims = data.get("claims")
            return claims if isinstance(claims, list) and claims else None
        except Exception:
            return None

    # ============== _narratives.json ==============
    def save_narratives(self, narratives: dict[str, dict[str, Any]]) -> None:
        p = self.path / "_narratives.json"
        p.write_text(
            json.dumps({"topic": self.topic, "created_at":
                        datetime.now().isoformat(timespec="seconds"),
                        "narratives": narratives}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_narratives(self) -> dict[str, dict[str, Any]] | None:
        p = self.path / "_narratives.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            nar = data.get("narratives")
            return nar if isinstance(nar, dict) and nar else None
        except Exception:
            return None

    # ============== _checkpoint.json (10-02 崩溃续跑) ==============
    # 为什么需要: v60e 那轮跑了 76 分钟 / 151 次调用, 卡死在主题归纳那一步,
    # 全轮作废 —— 而 fact 其实早就落盘在 _facts.jsonl 里了。流水线是严格串行的,
    # 前面每个阶段的产物都已经在手上, 只是没有"跑到哪了"的记录。
    #
    # 关键是 signature: 改了 max_pool / target_chars 就不能续, 否则会把
    # 旧参数下的中间产物接到新参数上 —— 那种错最难查(输出看着完全正常)。
    def save_checkpoint(self, stage: str, **extra: Any) -> None:
        p = self.path / "_checkpoint.json"
        payload = {
            "topic": self.topic,
            "topic_hash": self.topic_hash,
            "stage": stage,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        payload.update(extra)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

    def load_checkpoint(self) -> dict[str, Any] | None:
        p = self.path / "_checkpoint.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) and data.get("stage") else None
        except Exception:
            return None

    def clear_checkpoint(self) -> None:
        p = self.path / "_checkpoint.json"
        if p.exists():
            p.unlink()

    # ============== _outline.json ==============
    def save_outline(self, outline: dict[str, Any]) -> None:
        p = self.path / "_outline.json"
        p.write_text(
            json.dumps(outline, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_outline(self) -> dict[str, Any] | None:
        p = self.path / "_outline.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ============== _status.json (持续更新) ==============
    def set_status(
        self,
        phase: str,
        progress: int = 0,
        total: int = 0,
        message: str = "",
    ) -> None:
        p = self.path / "_status.json"
        payload = {
            "topic": self.topic,
            "topic_hash": self.topic_hash,
            "phase": phase,
            "progress": progress,
            "total": total,
            "message": message,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        # 写 tmp 再 rename, 防止中途崩溃留半截文件
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(p)

    def load_status(self) -> dict[str, Any] | None:
        p = self.path / "_status.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ============== _audit.md ==============
    def save_audit(self, audit_md: str) -> None:
        p = self.path / "_audit.md"
        p.write_text(audit_md, encoding="utf-8")

    def load_audit(self) -> str | None:
        p = self.path / "_audit.md"
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8")

    # ============== 元信息 ==============
    def summary(self) -> dict[str, Any]:
        """给 UI list_workspaces 用, 1 行摘要"""
        pool = self.load_pool()
        facts = self.load_facts()
        outline = self.load_outline()
        status = self.load_status()
        return {
            "topic_hash": self.topic_hash,
            "topic": self.topic,
            "path": str(self.path),
            "pool_size": len(pool),
            "facts_count": len(facts),
            "sections_count": len((outline or {}).get("sections", [])),
            "phase": (status or {}).get("phase", "?"),
            "updated_at": (status or {}).get("updated_at", ""),
        }


# ============== list_workspaces ==============
def list_workspaces(root: Path | None = None) -> list[dict[str, Any]]:
    """
    列出 data/surveys/working/ 下所有 workspace, 按 _status.json 的 updated_at 倒序。
    无 _status.json 的(可能是中断的)也列出, 标 phase="??"。
    """
    r = Path(root) if root else _DEFAULT_ROOT
    if not r.exists():
        return []
    items = []
    for sub in r.iterdir():
        if not sub.is_dir():
            continue
        # 从目录名恢复 topic_hash, topic 从 _status.json 拿
        status_file = sub / "_status.json"
        topic = ""
        phase = "??"
        updated_at = ""
        if status_file.exists():
            try:
                s = json.loads(status_file.read_text(encoding="utf-8"))
                topic = s.get("topic", "")
                phase = s.get("phase", "??")
                updated_at = s.get("updated_at", "")
            except Exception:
                pass
        # 数一下产物
        pool_file = sub / "_pool.json"
        facts_file = sub / "_facts.jsonl"
        outline_file = sub / "_outline.json"
        audit_file = sub / "_audit.md"
        pool_size = 0
        if pool_file.exists():
            with contextlib.suppress(Exception):
                pool_size = json.loads(pool_file.read_text(encoding="utf-8")).get("size", 0)
        facts_count = 0
        if facts_file.exists():
            facts_count = sum(1 for _ in facts_file.read_text(encoding="utf-8").splitlines() if _.strip())
        sections_count = 0
        if outline_file.exists():
            with contextlib.suppress(Exception):
                sections_count = len(json.loads(outline_file.read_text(encoding="utf-8")).get("sections", []))
        # 目录修改时间兜底(没 _status.json 时)
        if not updated_at:
            try:
                updated_at = datetime.fromtimestamp(sub.stat().st_mtime).isoformat(timespec="seconds")
            except Exception:
                updated_at = ""
        items.append({
            "topic_hash": sub.name,
            "topic": topic,
            "path": str(sub),
            "pool_size": pool_size,
            "facts_count": facts_count,
            "sections_count": sections_count,
            "phase": phase,
            "updated_at": updated_at,
            "has_audit": audit_file.exists(),
        })
    # 倒序
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return items


# ============== 模块自检 ==============
if __name__ == "__main__":
    # sanity
    h = hash_topic("量子纠错码近期进展")
    print(f"hash: {h}")
    ws = SurveyWorkspace(h, topic="量子纠错码近期进展")
    print(f"path: {ws.path}")
    ws.save_pool([{"paper_id": "p1", "title": "t1", "year": 2024}])
    ws.append_fact({"fact_id": "f1", "paper_id": "p1", "claim": "x"})
    ws.append_fact({"fact_id": "f2", "paper_id": "p1", "claim": "y"})
    ws.save_outline({"sections": [{"title": "intro", "section_type": "intro"}]})
    ws.set_status("fact_extraction", progress=1, total=1, message="done")
    ws.save_audit("# audit\nok")
    print(f"pool: {ws.load_pool()}")
    print(f"facts: {len(ws.load_facts())}")
    print(f"outline: {ws.load_outline()}")
    print(f"status: {ws.load_status()}")
    print(f"summary: {ws.summary()}")
    print(f"list: {list_workspaces()}")
    print("→ ok")
