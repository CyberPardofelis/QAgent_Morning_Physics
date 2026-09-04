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
import json
import re
import logging
import time
from typing import Iterator, Optional

from agent.llm_clients import (
    get_client as _get_client,
    get_deepseek_client,
    get_openai_client,
    get_minimax_client,
    LLMClientError,
    PROVIDERS,
    resolve_legacy_model,
    classify_llm_error,
)
from agent.prompts import SYSTEM_PROMPT
from agent.tools.registry import ToolRegistry
from agent.cost_guard import get_cost_guard
from agent.stream_utils import split_markdown_bubbles
from config import Config

logger = logging.getLogger(__name__)

# 触发所有工具模块的 @register_tool 装饰器执行
import agent.tools  # noqa: F401


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
MAX_ROUNDS = 20


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
    model: Optional[str] = None,
    temperature: float = 0.3,
    top_p: float = 0.9,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 40000,
    timeout: int = 180,
    enable_thinking: bool = True,
    reasoning_effort: str = "high",
    cost_tier: Optional[str] = None,  # 2026-09-02 P0.7 加:None = 从 Config 读
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

    # CostGuard: 进入 LLM 前 check + increment,100% 阻断
    cost_guard = get_cost_guard()
    is_ok, guard_msg = cost_guard.check_and_increment()
    if not is_ok:
        yield guard_msg
        yield NEW_BUBBLE
        return
    if guard_msg:
        # 80% 警告: 插到 LLM 输出前面,让用户看到
        yield guard_msg

    # 2026-08-31 v3 加(根因修复):当 deepseek 模式 + thinking 模式 + 带 tools 时,
    # DeepSeek 强制要求 messages 里**所有** assistant 消息都有 `reasoning_content` 字段
    # (空字符串 "" 也算有,不能缺)。
    # 否则报 400:
    #   "The reasoning_content in the thinking mode must be passed back to the API."
    # 这是 QAgent 一直 400 的真正根因(测试 2026-08-31 验证):
    # - Test 1: thinking on + tools + 空 assistant (没 reasoning_content) → 400
    # - Test 2: thinking on + tools + assistant 加 reasoning_content="" → OK
    # - Test 3: 不带 extra_body + tools + 空 assistant → 400
    # 之前 v1/v2 修复只在"切到非思考"或"切到非 deepseek 模式"时剥历史,治标不治本。
    # 空气泡(预分配的 content="" assistant)是从 conversation_controller 来的,
    # 它没 reasoning_content 字段 → DeepSeek 校验失败。
    # 修法:deepseek + thinking 模式下,给所有 assistant 消息补 reasoning_content=""(如果缺)
    if provider_name == "deepseek" and enable_thinking:
        for _m in messages:
            if _m.get("role") == "assistant" and "reasoning_content" not in _m:
                _m["reasoning_content"] = ""

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
    # 2026-09-02 加(抄 SAgent):跨 chunk 持续的状态盒,用于剥离 <think> 标签
    _think_state = [False, ""]
    while True:
        _round += 1
        # 2026-09-02 P0 fix: 超过 MAX_ROUNDS 强制退出, yield 友好提示给用户
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
                f"已自动终止,避免内存/费用爆。如需继续,请重新提问。\n"
            )
            yield NEW_BUBBLE
            return
        print(f"[controller] LLM round {_round} start  mode={provider_name} model={actual_model} messages={len(messages)}", flush=True)
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
            "timeout": timeout,
        }
        # 2026-08-31 修(从 SAgent 照搬):用 provider 显式分支代替通用 enable_thinking
        # 之前 `if enable_thinking:` 一刀切,对 minimax/local_qwen 都不准确。
        # SAgent 2026-08-24 修复的 3 路分支:
        #   - deepseek  + enable=True  → extra_body.thinking.enabled + reasoning_effort
        #   - minimax   + enable=False → extra_body.thinking.disabled (防 adaptive thinking 默认开)
        #   - 其他 provider           → 不动 extra_body
        # 2026-08-31 新增:deepseek 切到非思考时,清洗历史里所有 assistant 消息的 reasoning_content。
        # 否则 DeepSeek 报 400:
        #   "The reasoning_content in the thinking mode must be passed back to the API."
        # (历史里带 reasoning_content 字段 → 当前请求必须开 thinking,否则 API 端校验失败)
        # 2026-08-31 v2 修:跨模式切换场景(用户从 deepseek-thinking 切到 minimax/openai/compare),
        # 也需要剥 reasoning_content,因为非 deepseek 模式用不上、留着会触发校验失败。
        if provider_name == "deepseek":
            if enable_thinking:
                request_params["extra_body"] = {"thinking": {"type": "enabled"}}
                request_params["reasoning_effort"] = reasoning_effort
            else:
                # 剥掉历史里所有 assistant 消息的 reasoning_content,避免 400
                for _m in messages:
                    if _m.get("role") == "assistant" and "reasoning_content" in _m:
                        _m.pop("reasoning_content", None)
        elif provider_name == "minimax":
            if not enable_thinking:
                # MiniMax M3 默认 adaptive thinking,enable=False 时必须显式关
                # 否则用户取消"思考模式"对 MiniMax 完全无效,LLM 仍输出 reasoning_content,
                # 第二轮报 "must be passed back" 错
                request_params["extra_body"] = {"thinking": {"type": "disabled"}}
            # 2026-08-31 v2 加:minimax 模式一律剥 reasoning_content(其他 provider 用不上,留着会污染历史)
            for _m in messages:
                if _m.get("role") == "assistant" and "reasoning_content" in _m:
                    _m.pop("reasoning_content", None)
        else:
            # 2026-08-31 v2 加:openai / compare / custom 等其他 provider 一律剥 reasoning_content
            # 避免历史里的 deepseek thinking 残留触发 400
            for _m in messages:
                if _m.get("role") == "assistant" and "reasoning_content" in _m:
                    _m.pop("reasoning_content", None)

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
            collected_chunks: list[str] = []
            dispatch_failed = False
            try:
                for chunk in ToolRegistry.dispatch(fn_name, args):
                    if chunk:
                        collected_chunks.append(chunk)
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

            # DIRECT_OUTPUT 工具：按 ### 标题切气泡，yield 完 break
            if fn_name in DIRECT_OUTPUT_TOOLS:
                yield NEW_BUBBLE
                yield from _split_into_bubbles(result_text)
                return

            # 普通工具：yield 一条简短完成提示 + 切新气泡（让后续 LLM 输出在另一个气泡里）
            yield NEW_BUBBLE
            yield f"[工具 {fn_name} 执行完毕]"
            yield NEW_BUBBLE


def _dispatch_tool(fn_name: str, args: dict) -> str:
    """调工具，收集完整结果（同步或流式），返回字符串"""
    collected: list[str] = []
    for chunk in ToolRegistry.dispatch(fn_name, args):
        if chunk:
            collected.append(chunk)
    return "".join(collected)


def _split_into_bubbles(content: str) -> Iterator[str]:
    """
    把 DIRECT_OUTPUT 工具返回的长 Markdown 内容按 ### 标题切气泡。
    每段 yield 一次（用 NEW_BUBBLE 分隔）。
    如果没有 ### 标题，整个内容一个气泡。
    """
    if not content or not content.strip():
        return

    bubbles = split_markdown_bubbles(content)
    for bubble in bubbles:
        if bubble.strip():
            yield bubble
            yield NEW_BUBBLE


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
