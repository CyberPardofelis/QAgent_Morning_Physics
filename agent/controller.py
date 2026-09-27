"""
agent.controller — 流式对话与工具调用编排（v3 — 实时流式 + 工具结果切气泡）

设计目标：
- 实时 yield delta.content（保留 QAgent 流式感）
- 工具调用循环：每个工具的结果在子回合里被 LLM 看到
- DIRECT_OUTPUT 工具（generate_daily_summary_*）返回的完整内容按 ### 标题切气泡
- 单一循环，无 if/elif 工具分发
- 2026-09-02 改(抄 SAgent):加 <think>...</think> 标签流式剥离状态机,
  处理 MiniMax M3 把 thinking 直接拼在 content 字段的情况
"""
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections.abc import Iterator  # noqa: E402

from agent.cost_guard import get_cost_guard  # noqa: E402
from agent.llm_clients import (  # noqa: E402
    PROVIDERS,
    LLMClientError,
    classify_llm_error,
    resolve_legacy_model,
)
from agent.llm_clients import (  # noqa: E402
    get_client as _get_client,
)
from agent.prompts import SYSTEM_PROMPT  # noqa: E402
from agent.tools.registry import ToolRegistry  # noqa: E402
from config import Config  # noqa: E402

logger = logging.getLogger(__name__)

# 触发所有工具模块的 @register_tool 装饰器执行
import agent.tools  # noqa: E402, F401

# 直接输出工具：这些工具返回的内容就是最终答案，需要按章节切气泡
DIRECT_OUTPUT_TOOLS = {"generate_daily_summary_arxiv", "generate_aps_daily_summary"}


# ============================================================
# 工具调用日志(2026-09-02 抄 SAgent)
# UI 不显示工具原始输出,但控制台/日志要能看到:工具名 + 关键参数 +
# 结果长度 + 结果头尾片段,方便排查 silent failure。
# ============================================================
_LOG_PREVIEW_BYTES = 200


def _brief_args_for_log(fn_name: str, args: dict) -> str:
    """工具参数格式化(控制台用)。长字符串截断,大字典缩到 200 字内。"""
    try:
        s = json.dumps(args, ensure_ascii=False, default=str)
    except Exception:
        s = repr(args)
    if len(s) > _LOG_PREVIEW_BYTES:
        s = s[:_LOG_PREVIEW_BYTES] + f"... <truncated total={len(s)}>"
    return s


def _harmonize_assistant_reasoning(messages: list, provider_name: str, enable_thinking: bool) -> None:
    """
    2026-09-04 P0 fix:每轮 LLM 调用前,**就地**统一所有 assistant 消息的 `reasoning_content` 字段。

    背景:
    - DeepSeek API 在 thinking 模式下,**强制要求**所有 assistant 消息都有 `reasoning_content` 字段
      (空字符串也算有,不能缺),否则报 400:
      "The reasoning_content in the thinking mode must be passed back to the API."
    - 之前的 v3 修法只在 `run_agent_stream` 开头跑一次 pre-loop(给所有 assistant 补 `""`),
      但 while loop 里**新加的** assistant 消息(没 reasoning_content)会在下一轮触发 400
    - 修法:把"补 reasoning_content"逻辑抽成 helper,每轮调 LLM 前跑一次(in-place 修改)

    行为矩阵:
    - deepseek + enable_thinking=True  → 给所有 assistant 补 `reasoning_content=""`
    - deepseek + enable_thinking=False → 剥掉所有 assistant 的 `reasoning_content`
    - minimax (无论 thinking 怎么设)   → 剥掉所有 assistant 的 `reasoning_content`(minimax 不用)
    - openai / compare / custom       → 剥掉所有 assistant 的 `reasoning_content`

    注意:in-place 修改 messages(跟原 pre-loop 一致,不返新 list)。
    """
    if provider_name == "deepseek" and enable_thinking:
        # 补 reasoning_content=""(缺的 assistant)
        for _m in messages:
            if _m.get("role") == "assistant" and "reasoning_content" not in _m:
                _m["reasoning_content"] = ""
    else:
        # 其它场景:剥掉所有 reasoning_content
        for _m in messages:
            if _m.get("role") == "assistant" and "reasoning_content" in _m:
                _m.pop("reasoning_content", None)


def _brief_result_for_log(result: str) -> str:
    """工具结果格式化(控制台用)。头尾各 100 字,中间省略。
    空结果 → '<empty>' 标记,方便排查 silent failure。"""
    if not result:
        return "<empty>"
    if len(result) <= _LOG_PREVIEW_BYTES:
        return repr(result)
    head = result[:_LOG_PREVIEW_BYTES // 2]
    tail = result[-_LOG_PREVIEW_BYTES // 2:]
    return f"{head!r} ... <truncated total={len(result)}> ... {tail!r}"


# ============================================================
# 思考过程过滤(2026-09-02 抄 SAgent)
# MiniMax M3 把 thinking 用 <think>...</think> 标签直接拼在 delta.content 字段里
# (不是用 reasoning_content 字段),用一段轻量级状态机在流式输出时实时剥离。
# 这是 2026-09-02 用户报的"minimax 模式下两个空气泡" 根因。
# ============================================================
_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"


def _is_prefix_of_tag(prefix: str, tag: str) -> bool:
    """检查 prefix 是不是 tag 的前缀(用于跨 chunk 标签边界检测)"""
    return len(prefix) < len(tag) and tag.startswith(prefix)


def _strip_think_tags(chunk_content: str, state: list):
    """
    流式剥离 <think>...</think> 标签。

    参数:
        chunk_content: 当前 chunk 的 content(可能只含部分标签)
        state: 2 元素 list,作为 mutable 状态盒子: [in_thinking, pending_buf]
               调用方在循环外初始化 [False, ""],跨 chunk 保持状态。

    返回:
        list[str]: 应当 yield 给 UI 的内容片段(可能 0/1/多个)

    设计:
      - 把 chunk_content 累积到 state[1] (pending_buf),状态机对 pending_buf 操作
      - NORMAL 状态:yield 非 thinking 标签内容;如果 pending 是 open 标签的前缀,等待
      - THINKING 状态:丢弃 thinking 内容;如果 pending 是 close 标签的前缀,等待
      - 状态机跨 chunk 保持(state 参数),close 后继续找下一个 open(支持连续 thinking 块)
    """
    out = []
    # 累积新 chunk 到 pending
    state[1] += chunk_content
    pending = state[1]
    in_thinking = state[0]

    while pending:
        if in_thinking:
            if _THINK_CLOSE_TAG in pending:
                idx = pending.find(_THINK_CLOSE_TAG)
                pending = pending[idx + len(_THINK_CLOSE_TAG):]
                in_thinking = False
                # 循环,处理 close 后的 pending
            else:
                # 看 pending 中任何 < 位置是否可能是 close 标签前缀
                cut = len(pending)
                for i, ch in enumerate(pending):
                    if ch == '<':
                        candidate = pending[i:]
                        if _THINK_CLOSE_TAG.startswith(candidate) and len(candidate) < len(_THINK_CLOSE_TAG):
                            cut = i
                            break
                if cut < len(pending):
                    # 丢弃 cut 之前的 thinking,保留 cut 之后(可能是 close 前缀)
                    pending = pending[cut:]
                    break
                else:
                    # 整段都不是 close 前缀的开始,丢弃
                    pending = ""
        else:
            # NORMAL
            if _THINK_OPEN_TAG in pending:
                idx = pending.find(_THINK_OPEN_TAG)
                before = pending[:idx]
                if before:
                    out.append(before)
                pending = pending[idx + len(_THINK_OPEN_TAG):]
                in_thinking = True
            else:
                # 看 pending 中任何 < 位置是否可能是 open 标签前缀
                cut = len(pending)
                for i, ch in enumerate(pending):
                    if ch == '<':
                        candidate = pending[i:]
                        if _THINK_OPEN_TAG.startswith(candidate) and len(candidate) < len(_THINK_OPEN_TAG):
                            cut = i
                            break
                if cut < len(pending):
                    # yield cut 之前(确定非标签),保留 cut 之后(等更多)
                    if cut > 0:
                        out.append(pending[:cut])
                    pending = pending[cut:]
                    break
                else:
                    # 整段都不是 open 前缀的开始,整段 yield
                    out.append(pending)
                    pending = ""

    state[0] = in_thinking
    state[1] = pending
    return out


# 2026-09-04 加:_strip_think_tags_from_text — 一次性剥离整段文本里的
# <think>...</think> 块, 用于:
# - 历史消息加载时(history_manager / conversation_controller)
# - 发送给 LLM 前的清洗(_clean_messages_for_api)
# - UI 旧对话回放时(避免老 thinking 污染气泡)
# 因为整段文本已知完整, 不会跨 chunk, 用 regex 一行搞定
# (state machine 留给流式路径 _strip_think_tags 用)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_tags_from_text(text: str) -> str:
    """
    从一段完整文本里剥掉所有 <think>...</think> 块。

    用法:
    - controller._strip_think_tags_from_text(conv_msg["content"])
    - 若 text 不含 <think> 标签, 直接返回原对象(零拷贝优化)
    """
    if not text or "<think>" not in text:
        return text
    return _THINK_RE.sub("", text).strip()

# mode -> (provider_name, default_model)
MODE_TO_PROVIDER_MODEL = {
    "openai":            ("openai",   None),
    "deepseek-v4-flash": ("deepseek", "deepseek-v4-flash"),
    "deepseek-v4-pro":   ("deepseek", "deepseek-v4-pro"),
    "minimax-m3":        ("minimax",  "MiniMax-M3"),
    "minimax-vl-01":     ("minimax",  "MiniMax-VL-01"),
    # 旧 mode 兼容
    "deepseek-flash":    ("deepseek", "deepseek-v4-flash"),
    "deepseek-pro":      ("deepseek", "deepseek-v4-pro"),
    "deepseek":          ("deepseek", "deepseek-v4-pro"),
    # 自定义 mode（worker 用 extra_params 显式传 model）
    "custom":            (None,       None),
}


# bubble 分隔符（兼容旧 UI 代码）
NEW_BUBBLE = "\x1cNEW_BUBBLE"


# 2026-09-02 P0 fix: 工具循环死循环兜底（模块级常量, 方便 smoke test 验证 + 调优）
# LLM 持续调同工具 / 工具持续返空 / 工具持续报错 都会让 LLM 反复重试,
# 无 MAX_ROUNDS 会撞 cost_guard 50 ceiling + 内存累积 + 费用爆
# 2026-09-07 用户拍板: 综述生成任务多工具组合 (survey_generate 内部 9 次 LLM 调
# + chat 多轮 follow-up) 20 轮不够, 调到 50
MAX_ROUNDS = 50


def _resolve_client_and_model(mode: str) -> tuple:
    if mode not in MODE_TO_PROVIDER_MODEL:
        raise LLMClientError(
            "unknown",
            f"不支持的 mode: {mode!r}。支持: {list(MODE_TO_PROVIDER_MODEL.keys())}",
        )
    provider_name, default_model = MODE_TO_PROVIDER_MODEL[mode]
    if provider_name is None:
        return None, None, "custom"
    client = _get_client(provider_name)
    if default_model:
        model = default_model
    elif provider_name == "openai":
        model = Config.get_openai_model() or PROVIDERS["openai"].default_model
    elif provider_name == "deepseek":
        model = Config.get_deepseek_model() or PROVIDERS["deepseek"].default_model
    elif provider_name == "minimax":
        # MiniMax 支持运行时切换 model（如 MiniMax-M3 / MiniMax-VL-01），
        # 也允许用户在 QSettings 里配 MINIMAX_MODEL 覆盖
        try:
            from config.settings import get_settings
            cfg_model = (get_settings().value("MINIMAX_MODEL", "") or "").strip()
        except Exception:
            cfg_model = ""
        model = cfg_model or PROVIDERS["minimax"].default_model
    else:
        model = PROVIDERS[provider_name].default_model
    return client, model, provider_name


# resolve_legacy_model 已在 agent.llm_clients 中定义（统一来源）


# ============================================================
# 流式输出（核心）
# ============================================================
def run_agent_stream(
    messages: list,
    mode: str = "deepseek",
    model: str | None = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 40000,
    # 2026-09-04 改:180s → 600s
    # 之前默认 180s,慢模型(MiniMax-M3 等)跑长 prompt 容易撞 timeout,
    # 用户报"控制台只看到 LLM round 1 start,然后什么也没了"就是撞了
    # 600s 跟 _call_llm 里的 daily summary 默认一致(防 daily summary 工具也撞)
    timeout: int = 600,
    enable_thinking: bool = True,
    reasoning_effort: str = "high",
    cost_tier: str | None = None,  # 2026-09-02 P0.7 加:None = 从 Config 读
    **_unused,
) -> Iterator[str]:
    """
    流式调用 LLM。

    yield 规则：
    - delta.content → 实时 yield（流式感）
    - NEW_BUBBLE   → UI 创建新气泡
    - DIRECT_OUTPUT 工具（generate_daily_summary_*）返回后，按 ### 标题切气泡
    - 工具执行完毕 → yield 一条 [工具 name 执行完毕] 提示
    - LLM 错误 → yield 友好错误消息
    - CostGuard 阻断（100%）→ yield 阻断消息 + 直接 return

    2026-09-02 P0.7 加 cost_tier 参数:
    - "L0": 只本地检索工具, 0 次 LLM
    - "L1": + 单次 LLM 工具
    - "L2": + 重型多轮（P4 / P6 实现后才有）
    留空 = 从 Config.get_cost_tier() 读, 默认 L2
    """
    client, default_model, provider_name = _resolve_client_and_model(mode)
    actual_model = model or default_model
    if actual_model is None:
        raise LLMClientError("unknown", f"mode={mode!r} 需要显式传 model 参数")

    # 旧 model 名兼容
    actual_model, enable_thinking, reasoning_effort = resolve_legacy_model(
        actual_model, enable_thinking
    )

    # 2026-09-02 P0.7:cost_tier 过滤工具 + CostGuard 计数
    if cost_tier is None:
        cost_tier = Config.get_cost_tier()
    if cost_tier not in ("L0", "L1", "L2"):
        cost_tier = "L2"
    # 工具 schema 按档位过滤
    tools = ToolRegistry.list_schemas_filtered(cost_tier)
    logger.info(
        "[controller] 成本控制档位=%s, 工具数 %d (全部 %d)",
        cost_tier, len(tools), len(ToolRegistry.list_schemas()),
    )

    # 9-27 fix: 删掉这里的预扣。
    # 之前在 run_agent_stream 开头 charge 一次, while 循环里每轮又 charge 一次,
    # 第一轮被重复计费。而且 openai 早 return 路径会走 _stream_openai_simple
    # (那里自己 charge), 于是这条预扣是纯浪费。计费现在只发生在真正发请求的地方。

    # 2026-09-04 P0 fix:DeepSeek thinking 模式下 messages 里所有 assistant 必须有
    # reasoning_content 字段。之前的 v3 修法在 run_agent_stream 开头跑一次,
    # 但 while loop 里新加的 assistant 消息没被覆盖 → 第 2 轮 400。
    # 修法:_harmonize_assistant_reasoning() 在 while loop 每轮开头调一次
    # (见下面 while True 块),这里只在 run_agent_stream 开头也跑一次做兜底。
    _harmonize_assistant_reasoning(messages, provider_name, enable_thinking)

    # 2026-08-31 v2 加:openai 模式早 return, 跳过下面的 while 循环。
    # 但 openai 模式也不该带 reasoning_content(openai 不会用,且会污染历史)。
    # 在早 return 前统一剥掉历史里的 reasoning_content,跟 while 循环里的逻辑保持一致。
    if provider_name == "openai":
        for _m in messages:
            if _m.get("role") == "assistant" and "reasoning_content" in _m:
                _m.pop("reasoning_content", None)

    # OpenAI 模式：不支持工具调用
    if provider_name == "openai":
        try:
            yield from _stream_openai_simple(
                client=client, model=actual_model, messages=messages,
                temperature=temperature, max_tokens=min(max_tokens, 8192),
            )
        except Exception as e:
            yield _format_openai_error(e)
            yield NEW_BUBBLE
        return

    # DeepSeek 模式：思考 + 工具调用循环
    # MiniMax 模式：走同一路径（MiniMax OpenAI 兼容接口支持工具调用 + thinking extra_body）
    _round = 0
    # 9-27: 计费守卫在这里取一次, 循环里每轮 charge。
    # 预扣(循环外)已删除 —— 每次 charge 都对应一次真实 LLM 请求, 不多扣。
    cost_guard = get_cost_guard()
    # 9-27: turn 级 deadline。之前只有单次调用的 timeout, 工具执行无超时,
    # 整个 turn 的墙上时间没有上界。详见 Config.get_turn_deadline_seconds。
    _deadline = time.monotonic() + Config.get_turn_deadline_seconds()
    # 2026-09-02 加(抄 SAgent):跨 chunk 持续的状态盒,用于剥离 <think> 标签
    _think_state = [False, ""]
    while True:
        _round += 1

        # 9-27: turn 级预算检查。放在 MAX_ROUNDS 之后、计费之前 ——
        # 超预算时不该再扣费, 也不该再发请求。
        _remaining = _deadline - time.monotonic()
        if _remaining <= 0:
            elapsed = Config.get_turn_deadline_seconds()
            print(f"[controller] ⚠ 超过 turn 预算 {elapsed}s, 优雅收尾(round={_round})", flush=True)
            logger.warning("[controller] turn 预算耗尽, 中止 (round=%d)", _round)
            yield (
                f"\n\n[已停止: 超过单次对话的时间预算 {elapsed}s]\n"
                f"已完成 {_round - 1} 轮。\n"
                f"可以在设置里调大 'turn_deadline_seconds'。\n"
            )
            yield NEW_BUBBLE
            return

        # 2026-09-02 P0 fix: 超过 MAX_ROUNDS 强制退出, yield 友好提示给用户
        # 9-27 fix: 这个检查必须排在 cost_guard 计费**之前**。
        # 之前顺序反了 —— 第 51 轮先扣费再被 MAX_ROUNDS 拦下, 白扣一次。
        if _round > MAX_ROUNDS:
            print(f"[controller] ⚠ 工具循环超过 {MAX_ROUNDS} 轮, 强制退出(round={_round})", flush=True)
            logger.warning(
                "[controller] 工具循环超过 %d 轮, 强制退出(可能 LLM 死循环或工具持续返空)",
                MAX_ROUNDS,
            )
            yield (
                f"\n\n[已停止: 工具循环超过 {MAX_ROUNDS} 轮]\n"
                f"可能原因:\n"
                f"  - LLM 持续调同一工具(死循环)\n"
                f"  - 工具持续返回空结果\n"
                f"  - 工具持续返回错误\n\n"
                f"已自动终止,避免内存/费用爆。如需继续,请重新提问。"
            )
            yield NEW_BUBBLE
            return

        # 9-21 加: 每轮 LLM 调用前 charge (修复 cost_guard 设计 bug)
        # 9-27: 工具内部的 LLM 调用 (agent/tools/llm.py 的 _call_llm) 也各自计费,
        #       所以一次用户对话可能是 50 次 controller 调用 + 上百次工具调用
        is_ok, guard_msg = cost_guard.check_and_increment()
        if not is_ok:
            yield guard_msg
            yield NEW_BUBBLE
            return
        if guard_msg:
            # 80% 警告: 插到 LLM 输出前面,让用户看到
            yield guard_msg

            return

        # 2026-09-04 P0 fix:每轮都补 reasoning_content
        # 之前只在 run_agent_stream 开头跑一次 pre-loop,但 while loop 里新加的
        # assistant 消息(可能没 reasoning_content)会触发 DeepSeek 400:
        #   "The reasoning_content in the thinking mode must be passed back to the API."
        # 修法:每轮调 LLM 前再跑一次,确保 messages 里所有 assistant 都有 reasoning_content
        # (deepseek + thinking) 或 都没有(其它 provider)
        _harmonize_assistant_reasoning(messages, provider_name, enable_thinking)

        print(f"[controller] LLM round {_round} start  mode={provider_name} model={actual_model} messages={len(messages)}", flush=True)
        # 9-27: 把单次调用的 timeout 压到"剩余预算"以内。
        # 否则 timeout=600 的一次调用可能直接吃光整个 turn 预算,
        # 导致上面的预算检查在下一轮才发现超期, 用户白等 10 分钟。
        # 留 5s 余量给流式收尾/工具派发。
        _effective_timeout = timeout
        _budget_left = int(_deadline - time.monotonic()) - 5
        if _budget_left > 0:
            _effective_timeout = min(timeout, max(_budget_left, 10))
        request_params = {
            "model": actual_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "timeout": _effective_timeout,
        }
        # 2026-08-31 修(从 SAgent 照搬):用 provider 显式分支代替通用 enable_thinking
        # 之前 `if enable_thinking:` 一刀切,对 minimax/local_qwen 都不准确。
        # SAgent 2026-08-24 修复的 3 路分支:
        #   - deepseek  + enable=True  → extra_body.thinking.enabled + reasoning_effort
        #   - minimax   + enable=False → extra_body.thinking.disabled (防 adaptive thinking 默认开)
        #   - 其他 provider           → 不动 extra_body
        # 2026-09-04 注:历史消息的 reasoning_content 清洗已在 _harmonize_assistant_reasoning 里统一处理,
        # 这里只负责 per-request 的 extra_body 设置。
        if provider_name == "deepseek":
            if enable_thinking:
                request_params["extra_body"] = {"thinking": {"type": "enabled"}}
                request_params["reasoning_effort"] = reasoning_effort
        elif provider_name == "minimax" and not enable_thinking:
            # MiniMax M3 默认 adaptive thinking,enable=False 时必须显式关
            # 否则用户取消"思考模式"对 MiniMax 完全无效,LLM 仍输出 reasoning_content,
            # 第二轮报 "must be passed back" 错
            request_params["extra_body"] = {"thinking": {"type": "disabled"}}

        # MiniMax 接口对 frequency_penalty / presence_penalty 支持有限（部分模型会回 400），
        # 这里只在非 MiniMax 时透传，避免不必要的失败。
        if provider_name == "minimax":
            request_params.pop("frequency_penalty", None)
            request_params.pop("presence_penalty", None)

        try:
            stream = client.chat.completions.create(**request_params)
        except Exception as e:
            if provider_name == "minimax":
                yield _format_minimax_error(e)
            else:
                yield _format_deepseek_error(e)
            yield NEW_BUBBLE
            return

        current_content = ""
        current_reasoning = ""
        tool_calls_dict: dict = {}
        # 2026-09-04 加:检测到 LLM 在调用 DIRECT_OUTPUT 工具前输出 prose,
        # 后续 content 都不 yield(避免幻觉污染 UI)。系统 prompt + tool description
        # 已经明确告诉 LLM 不要 pre-call prose,这里是兜底。
        _discard_prose_for_direct_output = False

        # 2026-09-04 加:心跳日志(每 5s 一次),让用户知道 LLM 还在跑(没卡死)
        # 之前 MiniMax-M3 跑 9.7k prompt + 30k output 可能要 60-200s,
        # 用户看到控制台"LLM round 1 start"然后什么都没了 → 误以为卡死
        import threading as _th
        import time as _t
        _heartbeat_state = {"last_print": _t.time(), "stop": False}

        def _heartbeat():
            while not _heartbeat_state["stop"]:
                _t.sleep(5.0)
                if _heartbeat_state["stop"]:
                    break
                _elapsed = int(_t.time() - _heartbeat_state["last_print"])
                print(f"[controller] ⏳ LLM 还在流式响应... 累计 {_elapsed}s  (round {_round})", flush=True)

        _hb_thread = _th.Thread(target=_heartbeat, daemon=True)
        _hb_thread.start()

        try:
            for chunk in stream:
                delta = chunk.choices[0].delta
                # 2026-09-02 改(抄 SAgent):把 reasoning_content 字段的累加 + content 字段的
                # <think> 标签剥离严格分开
                # - deepseek: extra_body.thinking 开启时 reasoning 走独立字段 → 这里累加
                # - MiniMax M3: 把 thinking 直接拼在 content 字段(用 <think> 标签) → 走 _strip_think_tags 状态机
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    # 静默收集 reasoning — 不进 current_content、不 yield 给 UI
                    current_reasoning += rc or ""
                    continue
                if delta.content:
                    # 2026-09-02 改(抄 SAgent):过滤 MiniMax 把 thinking 直接拼在 content 字段里用
                    # <think>...</think> 标签包着的部分(不是用 reasoning_content 字段)。
                    # 用轻量级状态机: tracking 是否在 thinking 块内 + buffer 累积跨 chunk 的标签边界。
                    # 这就是用户报的"minimax 模式两个空气泡" 的真正根因 ——
                    # 之前 QAgent controller 直接 yield delta.content,让 <think>...</think> 标签
                    # 污染 UI 之外,有时因为标签跨 chunk 还会让 content 看起来"几乎为空"。
                    if _discard_prose_for_direct_output:
                        # 2026-09-04:已检测到 DIRECT_OUTPUT 工具调用,后续 prose 都是幻觉,
                        # 静默吞掉,不进 current_content、不 yield
                        continue
                    for fc in _strip_think_tags(delta.content, _think_state):
                        if fc:
                            current_content += fc
                            # ✅ 实时 yield — 保留 QAgent 流式感
                            yield fc
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_dict:
                            tool_calls_dict[idx] = {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": ""},
                            }
                        if tc.function.arguments:
                            tool_calls_dict[idx]["function"]["arguments"] += tc.function.arguments
                        # 2026-09-04:一旦看到 DIRECT_OUTPUT 工具调用,标记"丢弃后续 prose"
                        if tc.function.name in DIRECT_OUTPUT_TOOLS:
                            _discard_prose_for_direct_output = True
                            # 同时把已 yield 的 current_content 也从 history 抹掉
                            if current_content:
                                logger.warning(
                                    "[controller] LLM 在 DIRECT_OUTPUT 工具 %s 前已 yield %d 字符 prose,"
                                    "疑似幻觉,history 中已抹掉(但 UI 气泡里已经看到——这是 QTextBrowser 没法 rewind 的代价)。",
                                    tc.function.name, len(current_content),
                                )
                                current_content = ""
        finally:
            # 心跳线程停
            _heartbeat_state["stop"] = True

        # 把当前 assistant 消息加入历史
        assistant_msg: dict = {"role": "assistant", "content": current_content}
        # 2026-08-31 修(从 SAgent 照搬):加 provider 判断,只在 deepseek 模式
        # 才把 reasoning_content 加到 assistant_msg。
        # 其他 provider(minimax/openai)即使输出了 reasoning_content,也不存到历史,
        # 避免下次切到非 deepseek 模式时残留在 messages 里触发 400。
        if current_reasoning and provider_name == "deepseek":
            # DeepSeek thinking 模式要求多轮/工具循环时把 reasoning_content 原样传回
            assistant_msg["reasoning_content"] = current_reasoning
        if tool_calls_dict:
            assistant_msg["tool_calls"] = list(tool_calls_dict.values())
        messages.append(assistant_msg)

        # 没有工具调用：本轮结束
        if not tool_calls_dict:
            break

        # ========== 工具调用（统一通过 ToolRegistry 分发） ==========
        for tool_call in assistant_msg["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as e:
                yield NEW_BUBBLE
                yield f"[工具 {fn_name} 参数解析失败: {e}]"
                yield NEW_BUBBLE
                continue

            # 2026-09-02 改(抄 SAgent):控制台日志(不影响 UI,方便排查 silent failure)
            args_brief = _brief_args_for_log(fn_name, args)
            logger.info(
                "[controller] [tool-call] ▶ %s id=%s args=%s  (round %d)",
                fn_name, tool_call.get("id", "?"), args_brief, _round,
            )
            t0 = time.monotonic()

            # 工具结果收集（同步或流式）
            # 9-08 17:15 修: commit 6d91bc3 把 "yield chunk 给 UI" 删了, 改成只 collect
            #   导致 survey_generate 内部 ReAct 进度 ("--- Step X ---" / "💭 思考" / "➡️ 行动" / "👁️ 观察")
            #   只进 messages 历史, 用户看不到任何"工具在做什么" 的提示
            # 修: 边 collect 边 yield, 这样 UI 能实时看到工具内部进度, 收集到的 result_text 仍进 LLM 历史
            collected_chunks: list[str] = []
            dispatch_failed = False
            try:
                for chunk in ToolRegistry.dispatch(fn_name, args):
                    if chunk:
                        collected_chunks.append(chunk)
                        yield chunk   # 9-08 17:15 修: 同步 yield 给 UI (跟之前 commit 6d91bc3 之前一致)
            except Exception as e:
                dispatch_failed = True
                logger.exception("[controller] [tool-call] ✗ %s dispatch 异常: %s", fn_name, e)
            result_text = "".join(collected_chunks)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "[controller] [tool-call] %s %s id=%s len=%d elapsed=%dms preview=%s  (round %d)",
                "✗" if dispatch_failed or not result_text else "✓",
                fn_name, tool_call.get("id", "?"),
                len(result_text), elapsed_ms,
                _brief_result_for_log(result_text), _round,
            )

            # 工具结果加入 LLM 历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result_text,
            })

            # DIRECT_OUTPUT 工具:2026-09-04 用户拍板,取消 ### 切气泡。
            # 之前把 LLM 的总结按 `### 标题` 切到多个 bubble,理由是早期 chat 系统
            # 单个 bubble 高度有限(超出会被截断/不滚)。现在 QTextBrowser 接管气泡渲染,
            # 自带滚动条,单 bubble 也能放几万字 markdown → 不再需要切。
            # 行为:整个 LLM 输出 yield 进同一个 bubble(用户滚动阅读)。
            # 数据库层不受影响:parse_summary_markdown 解析完整 markdown 入库,
            # 跟气泡怎么切没关系。
            if fn_name in DIRECT_OUTPUT_TOOLS:
                yield NEW_BUBBLE
                yield result_text
                return

            # 普通工具：2026-09-05 P0 fix 简化: 不切气泡 (之前 NEW_BUBBLE × 2 产生 2 个空气泡)
            # 把完成消息附在当前气泡末尾, 让后续 LLM 输出继续在同一气泡
            # 用户实测: "工具调用时显示了多个空气泡" 根因
            yield f"\n\n[工具 {fn_name} 执行完毕]\n\n"
            yield NEW_BUBBLE  # 1 个 NEW_BUBBLE 够, 让 LLM 后续内容在新气泡


def _dispatch_tool(fn_name: str, args: dict) -> str:
    """调工具，收集完整结果（同步或流式），返回字符串"""
    collected: list[str] = []
    for chunk in ToolRegistry.dispatch(fn_name, args):
        if chunk:
            collected.append(chunk)
    return "".join(collected)


# ============================================================
# 错误格式化（薄封装，逻辑统一在 agent.llm_clients.classify_llm_error）
# ============================================================
def _format_openai_error(e: Exception) -> str:
    return classify_llm_error("OpenAI", e)


def _format_deepseek_error(e: Exception) -> str:
    return classify_llm_error("DeepSeek", e)


def _format_minimax_error(e: Exception) -> str:
    return classify_llm_error("MiniMax", e)


# ============================================================
# OpenAI 简单流式
# ============================================================
def _stream_openai_simple(client, model, messages, temperature, max_tokens) -> Iterator[str]:
    # 9-21 加: cost_guard 计数 (之前 OpenAI 模式完全不走 cost_guard)
    # 这是修复 cost_guard 设计 bug 的一部分
    cost_guard = get_cost_guard()
    is_ok, guard_msg = cost_guard.check_and_increment()
    if not is_ok:
        yield guard_msg
        yield NEW_BUBBLE
        return
    if guard_msg:
        yield guard_msg

    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True,
        temperature=temperature, max_tokens=max_tokens,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ============================================================
# 非流式（兼容旧调用）
# ============================================================
def call_model(client, model_name, user_input) -> str:
    # 9-21 加: cost_guard 计数 (之前完全不走 cost_guard)
    cost_guard = get_cost_guard()
    is_ok, guard_msg = cost_guard.check_and_increment()
    if not is_ok:
        return f"{guard_msg}\n(cost_guard 阻断, 拒绝非流式 LLM 调用)"
    if guard_msg:
        # 80% 警告 (非流式只返 string, 警告直接拼到结果里)
        print(guard_msg, flush=True)

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    response = client.chat.completions.create(
        model=model_name, messages=msgs, temperature=0.3, max_tokens=8192,
    )
    return response.choices[0].message.content


def run_agent(user_input: str, mode: str = "deepseek") -> str:
    client, model, _ = _resolve_client_and_model(mode)
    if client is None or model is None:
        raise LLMClientError("unknown", f"mode={mode!r} 在非流式入口不支持")
    return call_model(client, model, user_input)
