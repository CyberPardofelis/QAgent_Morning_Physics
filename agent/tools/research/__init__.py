"""
agent.tools.research — 科研验证闭环 (P7)

设计稿: docs/RESEARCH_CLOSED_LOOP_PLAN.md (阶段划分见 §10)

这个包回答的是和综述完全不同的问题:
    综述(P6): 领域里有什么说法?
    科研(P7): 这个命题到底成不成立, **凭什么**?

模块结构 (LLM-heavy 的与纯本地的分开, 纯本地的可脱离 LLM 单测):
- _sandbox        S0 受限执行器 (子进程 + AST 门 + 超时杀进程树) ★ 闭环的地基
- _provenance     证据编号 / 环境指纹 / verdict 判定 (run 了 ≠ 成立, 纪律在这)
- _sympy_tools    SymPy 符号计算 (编译成代码后交给沙箱, 不直接调)
- _flags          暴露开关 (QAGENT_RESEARCH_EXPOSE) + 超时/重试默认值
- research_run_code  L0 工具: 跑一段代码, 记一条证据
- research_verify    L1 工具: ★ 主入口, 命题 → 生成代码 → 执行 → 判定 → 证据链

状态 (2026-10-08):
- [x] P7.0 registry exposed 开关 / S0 沙箱 / 证据链 / workspace / prompts
- [x] P7.0 research_run_code + research_verify (research_sympy 已接在 _sympy_tools)
- [ ] P7.1 反例枚举 + Z3 + OEIS
- [ ] P7.2 ReAct 编排 + 命题库 + 报告 + 审计
- [ ] P7.3 Docker 真隔离 (等雪风装 Docker Desktop)

⚠️ **本包 import 时不会自动注册到 agent.tools.__init__** —— 刻意如此。
   research_* 工具 exposed=False (见 _flags), LLM 看不见, 调好后再用
   QAGENT_RESEARCH_EXPOSE=1 接入。调试请直接调:
       from agent.tools.research.research_verify import research_verify
   或:
       ToolRegistry.get("research_verify").handler(...)
"""
import logging

logger = logging.getLogger(__name__)

# ============================================================
# 重导出: LLM 工具 (@register_tool)
# 注意 import 本模块即触发装饰器 → 工���进入 ToolRegistry,
# 但因为 exposed=False, 它们**不会**出现在 list_schemas() 里。
# ============================================================
# ============================================================
# 重导出: 内部 API (workspace 在 agent.infra, 这里转发方便调试)
# ============================================================
from agent.infra.research_workspace import (  # noqa: E402
    ResearchWorkspace,
    hash_question,
    list_workspaces,
)

from ._claim import audit_claims, normalize_claim, render_report
from ._enumerate import GRAPH_PROPERTIES, SEQUENCE_PROPERTIES
from ._flags import RESEARCH_EXPOSED
from ._loop import decide_next, new_state, run_loop
from ._provenance import VERDICTS, build_evidence, claim_id_for, judge_verdict, record
from ._retrieve import lookup as oeis_lookup
from ._sandbox import SandboxRejected, SandboxResult, run_code
from ._sympy_tools import sympy_op
from ._z3_tools import Z3Error
from ._z3_tools import solve as z3_solve
from .research_audit import research_audit
from .research_enumerate import research_enumerate
from .research_investigate import research_investigate
from .research_oeis import research_oeis
from .research_plan import research_plan
from .research_refute import research_refute
from .research_report import research_report
from .research_run_code import research_run_code
from .research_sympy import research_sympy
from .research_verify import extract_code, research_verify
from .research_z3 import research_z3

__all__ = [
    # LLM 工具
    "research_run_code", "research_verify", "research_sympy",
    "research_z3", "research_enumerate", "research_oeis", "research_refute",
    "research_plan", "research_investigate", "research_report", "research_audit",
    # 内部 API
    "run_code", "SandboxResult", "SandboxRejected",
    "sympy_op", "judge_verdict", "build_evidence", "record", "claim_id_for", "VERDICTS",
    "extract_code",
    "z3_solve", "Z3Error", "oeis_lookup",
    "GRAPH_PROPERTIES", "SEQUENCE_PROPERTIES",
    "normalize_claim", "render_report", "audit_claims",
    "decide_next", "new_state", "run_loop",
    "ResearchWorkspace", "hash_question", "list_workspaces",
    "RESEARCH_EXPOSED",
]
