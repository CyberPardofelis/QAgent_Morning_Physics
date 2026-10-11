# math_research_scorer.py
"""
研究过程质量打分器（Research Process Reward Scorer）

用途：
  - 输入：一个 agent 在数学研究任务上的完整轨迹（AgentTrace）
  - 输出：0-100 的过程质量分 + 分维度明细 + 反馈 + 改进建议
  - 定位：可作为 RL/自我博弈的 reward function 或验证器"第二裁判"
  - 提示：这个打分器就是"判别器 D"，题目生成器 + agent 是"生成器 G"

设计原则：
  1. correctness 是入场券，不是全部
  2. 鼓励多次尝试、方法切换、独立交叉验证、证书化
  3. 惩罚"一把梭就交卷"、"只有暴力搜索"、"从不验证"
  4. 输出的 feedback / suggestions 是给 agent 自我改进用的
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import json, random


# ─────────────────────────── 数据结构 ───────────────────────────

@dataclass
class TraceStep:
    """一步轨迹。step_type 建议取值见 STEP_TYPES。"""
    type: str
    content: str = ""
    tool: Optional[str] = None          # 例如 "z3", "sympy", "lean4"
    method: Optional[str] = None        # 例如 "sat_encoding", "brute_force", "sos"
    success: Optional[bool] = None
    meta: Dict[str, Any] = field(default_factory=dict)


STEP_TYPES = {
    "plan",         # 制定计划 / 分解问题
    "search",       # 检索文献 / OEIS / 定理
    "experiment",   # 数值实验 / 小范围枚举
    "code",         # 写代码
    "run",          # 跑代码 / 跑工具
    "observe",      # 观察结果
    "verify",       # 验证（同一路径）
    "cross_check",  # 独立路径交叉验证
    "reflect",      # 反思 / 修正思路
    "conclude",     # 给最终结论
}


@dataclass
class AgentTrace:
    task_id: str
    steps: List[TraceStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    final_correct: Optional[bool] = None       # 由外部 verifier 提供
    certificate: Optional[str] = None          # 机器可验证证书
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    id: str
    prompt: str
    recommended_tools: List[str] = field(default_factory=list)
    expected_methods: List[str] = field(default_factory=list)
    requires_certificate: bool = False
    tags: List[str] = field(default_factory=list)


@dataclass
class ScoreReport:
    total: float
    breakdown: Dict[str, float]
    penalties: Dict[str, float]
    feedback: List[str]
    suggestions: List[str]
    weak_dims: List[str]          # 分数最低的若干维度，供课程生成器参考

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "breakdown": self.breakdown,
            "penalties": self.penalties,
            "feedback": self.feedback,
            "suggestions": self.suggestions,
            "weak_dims": self.weak_dims,
        }


# ─────────────────────────── 权重 ───────────────────────────

DEFAULT_WEIGHTS = {
    "correctness":      0.40,   # 最终答案对不对
    "method_diversity": 0.10,   # 用了多少种方法
    "self_correction":  0.10,   # 失败后能不能有效修正
    "cross_validation": 0.10,   # 独立路径交叉验证
    "certification":    0.10,   # 机器可验证证书
    "tool_quality":     0.08,   # 工具匹配度
    "exploration":      0.07,   # 小实验 / 反例搜索
    "calibration":      0.05,   # 结论的不确定性标注
}


# ─────────────────────────── 分维度打分 ───────────────────────────

def score_correctness(trace: AgentTrace, task: TaskSpec) -> float:
    if trace.final_correct is None:
        return 0.0
    return 1.0 if trace.final_correct else 0.0


def score_method_diversity(trace: AgentTrace, task: TaskSpec) -> float:
    methods = {s.method for s in trace.steps if s.method}
    n = len(methods)
    if n >= 3: return 1.0
    if n == 2: return 0.7
    if n == 1: return 0.3
    return 0.0


def score_self_correction(trace: AgentTrace, task: TaskSpec) -> float:
    fails = recovers = 0
    last_failed = False
    for s in trace.steps:
        if s.success is False:
            fails += 1
            last_failed = True
        elif s.success is True and last_failed:
            recovers += 1
            last_failed = False
    if fails == 0:
        # 没失败过：可能题太简单，也可能第一次就选对，也可能作弊
        return 0.5
    if recovers == 0:
        return 0.1
    if recovers >= fails:
        return 1.0
    return 0.5 + 0.5 * recovers / fails


def score_cross_validation(trace: AgentTrace, task: TaskSpec) -> float:
    verify_like = [s for s in trace.steps
                   if s.type in ("verify", "cross_check", "run") and s.success]
    tools   = {s.tool   for s in verify_like if s.tool}
    methods = {s.method for s in verify_like if s.method}
    # 要么 ≥2 个不同工具，要么 ≥2 条不同方法路径
    if len(tools) >= 2 or len(methods) >= 2:
        return 1.0
    if len(tools) == 1 or len(methods) == 1:
        return 0.4
    return 0.0


def score_certification(trace: AgentTrace, task: TaskSpec) -> float:
    if trace.certificate:
        return 1.0
    if task.requires_certificate:
        return 0.0
    has_verified = any(s.type == "verify" and s.success for s in trace.steps)
    if has_verified:
        return 0.6
    return 0.2


def score_tool_quality(trace: AgentTrace, task: TaskSpec) -> float:
    used = {s.tool for s in trace.steps if s.tool}
    if not used:
        return 0.0
    if not task.recommended_tools:
        return min(1.0, len(used) / 3)
    recommended = set(task.recommended_tools)
    hit = used & recommended
    target = min(3, len(recommended))
    return min(1.0, len(hit) / max(1, target))


def score_exploration(trace: AgentTrace, task: TaskSpec) -> float:
    score = 0.0
    for s in trace.steps:
        if s.type == "experiment":
            score += 0.3
        if s.type == "search":
            score += 0.2
        if s.meta.get("counterexample_search"):
            score += 0.4
    return min(1.0, score)


def score_calibration(trace: AgentTrace, task: TaskSpec) -> float:
    text = (trace.final_answer or "").lower()
    hedged = any(w in text for w in [
        "uncertain", "不确定", "可能", "猜测", "conjecture",
        "假设", "hypothesis", "heuristic", "启发", "未证明"
    ])
    if trace.final_correct and not hedged: return 1.0
    if trace.final_correct and hedged:     return 0.7
    if (not trace.final_correct) and hedged: return 0.4
    return 0.0


# ─────────────────────────── 惩罚项 ───────────────────────────

def compute_penalties(trace: AgentTrace, task: TaskSpec) -> Dict[str, float]:
    p: Dict[str, float] = {}
    has_verify = any(s.type in ("verify", "cross_check") for s in trace.steps)
    if not has_verify:
        p["no_verification"] = 20.0

    methods = {s.method for s in trace.steps if s.method}
    if methods and methods <= {"brute_force"}:
        p["brute_force_only"] = 10.0

    if trace.steps and trace.steps[-1].type == "conclude" and not has_verify:
        p["premature_conclusion"] = 5.0

    if trace.final_correct and len(trace.steps) < 3:
        p["too_few_steps"] = 15.0

    return p


# ─────────────────────────── 反馈生成 ───────────────────────────

def generate_feedback(
    breakdown: Dict[str, float],
    penalties: Dict[str, float],
    trace: AgentTrace,
    task: TaskSpec,
) -> (List[str], List[str]):
    fb: List[str] = []
    sug: List[str] = []

    if breakdown["correctness"] == 0:
        fb.append("最终答案未通过验证器。")
        sug.append("回到问题表述，确认是否误解题意；优先写最小复现。")

    if breakdown["method_diversity"] < 0.7:
        fb.append("方法单一，缺少切换。")
        sug.append("至少尝试 3 种不同方法（枚举 / 符号 / SAT / 数值 / SDP）。")

    if breakdown["self_correction"] < 0.5:
        fb.append("遇到失败后没有有效恢复。")
        sug.append("失败时先分类错误：编码错 / 思路错 / 工具选错，再针对性修正。")

    if breakdown["cross_validation"] < 0.7:
        fb.append("缺少独立交叉验证。")
        sug.append("用两套不同实现或不同工具算出同一结论再下结论。")

    if breakdown["certification"] < 0.5:
        fb.append("缺少机器可验证证书。")
        sug.append("尽量给 SAT/UNSAT 证书、Lean 证明、SOS 证书或反例分解。")

    if breakdown["tool_quality"] < 0.5:
        fb.append("工具使用不足或不匹配。")
        if task.recommended_tools:
            sug.append(f"本题推荐工具：{', '.join(task.recommended_tools)}")

    if breakdown["exploration"] < 0.5:
        fb.append("缺少小实验或反例搜索。")
        sug.append("先在小范围枚举/随机测试，验证猜想方向再上重武器。")

    if breakdown["calibration"] < 0.5:
        fb.append("结论缺乏不确定性标注。")
        sug.append("给出结论时标注证据强度，区分'已验证'与'未验证'。")

    for k, v in penalties.items():
        fb.append(f"惩罚项：{k} (-{v})")

    return fb, sug


# ─────────────────────────── 主入口 ───────────────────────────

def score_trace(
    trace: AgentTrace,
    task: TaskSpec,
    weights: Optional[Dict[str, float]] = None,
) -> ScoreReport:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    breakdown = {
        "correctness":      score_correctness(trace, task),
        "method_diversity": score_method_diversity(trace, task),
        "self_correction":  score_self_correction(trace, task),
        "cross_validation": score_cross_validation(trace, task),
        "certification":    score_certification(trace, task),
        "tool_quality":     score_tool_quality(trace, task),
        "exploration":      score_exploration(trace, task),
        "calibration":      score_calibration(trace, task),
    }

    base = sum(breakdown[k] * w[k] for k in w) * 100.0
    penalties = compute_penalties(trace, task)
    total = max(0.0, base - sum(penalties.values()))

    feedback, suggestions = generate_feedback(breakdown, penalties, trace, task)

    # 找出最弱的 3 个维度，供课程生成器用
    weak = sorted(breakdown.items(), key=lambda kv: kv[1])[:3]
    weak_dims = [k for k, v in weak if v < 0.7]

    return ScoreReport(
        total=round(total, 2),
        breakdown={k: round(v, 3) for k, v in breakdown.items()},
        penalties={k: round(v, 2) for k, v in penalties.items()},
        feedback=feedback,
        suggestions=suggestions,
        weak_dims=weak_dims,
    )


# ─────────────────────────── Gym 风格环境 ───────────────────────────

class MathResearchEnv:
    """
    简易 gym 风格环境：
        obs = env.reset(task_id)
        obs, reward, done, info = env.step(TraceStep(...))
    - reward 只在 episode 结束时给出（由 score_trace 计算）
    - info["report"] 含明细 + 反馈 + 建议
    - info["weak_dims"] 可直接喂给出题器，形成 GAN 式自我博弈
    """

    def __init__(
        self,
        tasks: List[TaskSpec],
        verifier: Callable[[TaskSpec, AgentTrace], Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.tasks = tasks
        self.verifier = verifier
        self.config = config or {}
        self.current_task: Optional[TaskSpec] = None
        self.current_trace: Optional[AgentTrace] = None
        self.step_count = 0

    def reset(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        self.current_task = self._pick_task(task_id)
        self.current_trace = AgentTrace(task_id=self.current_task.id, steps=[])
        self.step_count = 0
        return {
            "task_id": self.current_task.id,
            "prompt": self.current_task.prompt,
            "tool_hint": self.current_task.recommended_tools,
            "requires_certificate": self.current_task.requires_certificate,
        }

    def step(self, step: TraceStep):
        assert self.current_trace is not None, "call reset() first"
        self.current_trace.steps.append(step)
        self.step_count += 1

        max_steps = self.config.get("max_steps", 50)
        done = step.type == "conclude" or self.step_count >= max_steps
        reward = 0.0
        info: Dict[str, Any] = {}

        if done:
            verdict = self.verifier(self.current_task, self.current_trace) or {}
            self.current_trace.final_correct = verdict.get("correct")
            self.current_trace.certificate  = verdict.get("certificate")
            if "final_answer" in verdict:
                self.current_trace.final_answer = verdict["final_answer"]

            report = score_trace(self.current_trace, self.current_task)
            reward = report.total
            info["report"] = report.to_dict()
            info["weak_dims"] = report.weak_dims

        obs = {
            "step_count": self.step_count,
            "last_tool": self.current_trace.steps[-1].tool if self.current_trace.steps else None,
        }
        return obs, reward, done, info

    def _pick_task(self, task_id: Optional[str]) -> TaskSpec:
        if task_id:
            for t in self.tasks:
                if t.id == task_id:
                    return t
            raise KeyError(f"task not found: {task_id}")
        return random.choice(self.tasks)


# ─────────────────────────── Demo ───────────────────────────

def _demo_verifier(task: TaskSpec, trace: AgentTrace) -> Dict[str, Any]:
    """Demo 用：假设 task id 以 res-001 开头时答对，其余随机。"""
    return {"correct": trace.final_correct, "certificate": trace.certificate}


def _pretty(report: ScoreReport) -> None:
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    task = TaskSpec(
        id="res-001-ramsey-34-lower-bound",
        prompt="构造 8 顶点红蓝图，无红 K3 无蓝 K4，并给机器可验证证书。",
        recommended_tools=["z3", "networkx", "python"],
        expected_methods=["sat_encoding", "brute_check"],
        requires_certificate=True,
        tags=["ramsey", "sat", "certificate"],
    )

    # 轨迹 A：一把梭交卷，无验证，无证书
    trace_A = AgentTrace(
        task_id=task.id,
        steps=[
            TraceStep(type="plan", content="直接猜一个矩阵", method="guess"),
            TraceStep(type="conclude", content="这就是答案", method="guess"),
        ],
        final_answer="一个随手写的邻接矩阵（未验证）",
        final_correct=False,
        certificate=None,
    )

    # 轨迹 B：多步、多方法、失败后修正、有证书
    trace_B = AgentTrace(
        task_id=task.id,
        steps=[
            TraceStep(type="plan",   content="先试 SAT 编码",       method="sat_encoding"),
            TraceStep(type="code",   content="写 z3 脚本",          tool="z3",       method="sat_encoding"),
            TraceStep(type="run",    content="K8 直接编码无解",     tool="z3",       method="sat_encoding", success=False),
            TraceStep(type="reflect",content="改用随机搜索 + 对称性剪枝", method="random_search"),
            TraceStep(type="code",   content="写 networkx 搜索脚本", tool="networkx", method="random_search"),
            TraceStep(type="run",    content="找到一个候选着色",    tool="networkx", method="random_search", success=True),
            TraceStep(type="verify", content="暴力验证无红 K3 无蓝 K4", tool="python", method="brute_check", success=True),
            TraceStep(type="cross_check", content="独立 SAT 验证反例合法性", tool="z3", method="sat_encoding", success=True),
            TraceStep(type="conclude", content="给出邻接矩阵 + 证书", method="conclude"),
        ],
        final_answer="邻接矩阵 + 两套独立验证输出的证书（已验证）",
        final_correct=True,
        certificate="python: no_red_K3 and no_blue_K4; z3: UNSAT for mono K4",
    )

    print("=== 轨迹 A：一把梭 ===")
    _pretty(score_trace(trace_A, task))

    print("\n=== 轨迹 B：研究流程 ===")
    _pretty(score_trace(trace_B, task))
