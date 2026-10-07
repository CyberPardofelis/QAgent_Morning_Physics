"""
工具内部调 LLM 的统一入口（agent.tools.llm._call_llm）。

之前版本直接用 agent.tools.config 模块级常量（DEEPSEEK_API_KEY），
但 config 只在 import 时读一次 .env，且没用 agent.infra.api_keys 的 3 层兜底，
所以改 key 后不生效，QSettings 里的 key 也读不到。

新版修复：
- 改用 agent.infra.api_keys.get_api_key(provider) 动态读（保证永远最新）
- 旧 model 名兼容 + 错误分类统一在 agent.llm_clients
- 支持 provider 参数（"deepseek" / "minimax" / ...），默认 deepseek 保持向后兼容

2026-09-04 用户拍板:工具内部调 LLM 默认跟主界面 chat mode 走
- 之前硬编码 default provider="deepseek" model="deepseek-v4-flash"
- 跟 chat_window.mode_combo 选啥没关系
- 现在:_call_llm 不传 provider/model 时,自动从 QSettings `chat_mode` 读,
  复用 controller 的 MODE_TO_PROVIDER_MODEL 映射
- 思考模式也跟(`chat_think_mode`)
- 工具如果想强制用某个 model(比如 _smoke 测试),还是显式传 provider/model
"""
import logging
import threading

from agent.cost_guard import charge as cost_guard_charge
from agent.infra.text_clean import sanitize_llm_text
from agent.llm_clients import (
    PROVIDERS,
    classify_llm_error,
    get_client,
    resolve_legacy_model,
)

# 10-02: 综述流水线开始并发调用 _call_llm 之后, 这些 print 会在多线程里交错,
# 严重时把两条日志黏成一行("...done 耗时 12.3s[llm] _call_llm start ..."),
# 阶段耗时统计就废了。print(x, flush=True) 不是原子的(字符串和换行是两次写)。
# 只锁 print 本身 —— 不锁请求, 并发度不受影响。
_LOG_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _LOG_LOCK:
        print(msg, flush=True)

logger = logging.getLogger(__name__)


def _get_provider_key(provider: str) -> str:
    """动态从 agent.infra.api_keys 读 key(3 层兜底:.api_keys.json > QSettings > .env)"""
    try:
        from agent.infra.api_keys import get_api_key
        return get_api_key(provider)
    except Exception as e:
        logger.warning("[tools.llm] 读 %s api_key 失败: %s", provider, e)
        # 最后兜底:读模块级常量(仅 deepseek / openai 有)
        try:
            from agent.tools import config as _cfg
            return getattr(_cfg, f"{provider.upper()}_API_KEY", "") or ""
        except Exception:
            return ""


def _resolve_chat_mode_params() -> tuple:
    """
    2026-09-04 加:从 QSettings 读主界面 chat mode,返回 (provider, model, enable_thinking)。

    复用 controller.MODE_TO_PROVIDER_MODEL 映射,跟主界面 controller 用同一份。
    优先级:
      1. QSettings `chat_mode` 存在且在 MODE_TO_PROVIDER_MODEL 里 → 走它
      2. 兜底:provider="deepseek", model="deepseek-v4-flash", enable_thinking=False
         (跟旧版默认行为一致,保证向后兼容)

    注意:不是所有 mode 都有 model 字段(e.g. "openai" 走 Config.get_openai_model())
    这种情况下:
      - openai:用 Config.get_openai_model() 或 PROVIDERS["openai"].default_model

    Returns:
        (provider: str, model: str, enable_thinking: bool)

    2026-09-04 修:不能顶层 import agent.controller(循环 import:
    controller -> tools.registry -> tools.llm -> controller)
    改为函数内 lazy import。
    """
    default = ("deepseek", "deepseek-v4-flash", False)
    try:
        from agent.controller import MODE_TO_PROVIDER_MODEL
    except Exception:
        return default

    try:
        from config.settings import get_settings
        settings = get_settings()
    except Exception:
        return default

    try:
        mode = (settings.value("chat_mode", "", type=str) or "").strip()
    except Exception:
        return default
    if not mode or mode not in MODE_TO_PROVIDER_MODEL:
        return default

    provider_name, default_model = MODE_TO_PROVIDER_MODEL[mode]
    if provider_name is None:
        # "custom" mode(worker 模式,无默认 provider/model)
        return default

    # 解析 model(provider-specific 兜底)
    if default_model:
        model = default_model
    else:
        # provider-specific(参考 controller._resolve_client_and_model)
        try:
            from config import _config as Config
            if provider_name == "openai":
                model = Config.Config.get_openai_model() or PROVIDERS["openai"].default_model
            elif provider_name == "deepseek":
                model = Config.Config.get_deepseek_model() or PROVIDERS["deepseek"].default_model
            elif provider_name == "minimax":
                cfg_model = (settings.value("MINIMAX_MODEL", "") or "").strip()
                model = cfg_model or PROVIDERS["minimax"].default_model
            else:
                model = PROVIDERS[provider_name].default_model
        except Exception:
            model = PROVIDERS[provider_name].default_model

    # 思考模式跟主界面:仅在 provider=deepseek 时同步(其它 provider 思考模式行为未验证)
    think = False
    if provider_name == "deepseek":
        try:
            think_str = settings.value("chat_think_mode", "false", type=str)
            think = (think_str == "true")
        except Exception:
            think = False

    return (provider_name, model, think)


def _call_llm(
        prompt: str,
        model: str = None,   # 2026-09-04 改:None 表示跟主界面 chat mode
        enable_thinking: bool = None,  # None = 跟主界面
        reasoning_effort: str = "high",  # "high" 或 "max"
        temperature: float = 0.3,
        top_p: float = 0.8,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 40000,
        timeout: int = 180,
        provider: str = None,  # 2026-09-04 改:None = 跟主界面
) -> str:
    """
    工具内部用的 LLM 调用(同步、非流式)。
    用于 generate_daily_summary_arxiv / generate_aps_daily_summary / analyze_paper 等工具内的总结生成。

    2026-09-04 用户拍板:默认 provider/model/enable_thinking 跟主界面 chat mode 走。
    - 之前硬编码 provider="deepseek" model="deepseek-v4-flash" enable_thinking=False
    - 现在:不传时,从 QSettings `chat_mode` 读,复用 controller.MODE_TO_PROVIDER_MODEL
    - 工具如果想强制用某个 model(smoke 测试等),显式传 provider/model 即可

    参数:
        provider: LLM provider id("deepseek" / "minimax" / "openai"),None = 跟主界面
        model:    具体 model 名称,None = 跟主界面
        enable_thinking: 是否走思考模式,None = 跟主界面(仅 deepseek 时同步)
        reasoning_effort / temperature / ...: 透传给 chat.completions.create

    返回:成功时为 LLM 输出文本;失败时为带错误提示的字符串(不再 print 错误就吞掉)。
    """
    # 2026-09-04:默认参数从 chat mode 读
    if provider is None or model is None or enable_thinking is None:
        chat_provider, chat_model, chat_think = _resolve_chat_mode_params()
        if provider is None:
            provider = chat_provider
        if model is None:
            model = chat_model
        if enable_thinking is None:
            enable_thinking = chat_think

    # 校验 provider
    if provider not in PROVIDERS:
        return f"❌ 未知 LLM provider: {provider!r}。支持: {list(PROVIDERS.keys())}"

    cfg = PROVIDERS[provider]
    display_name = cfg.display_name

    # 1. 拿 client(自动处理 3 层兜底 + 缓存)
    try:
        client = get_client(provider)
    except Exception as e:
        # get_client 内部已经封装错误,e 是 LLMClientError
        return classify_llm_error(display_name, e)

    # 2. 旧 model 名兼容(仅 deepseek 有 legacy 映射,其他 provider 透传)
    model, enable_thinking, reasoning_effort = resolve_legacy_model(model, enable_thinking)

    # 3. 构造请求参数
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    # ---- 2026-07-07 修: MiniMax 的 thinking 开关 ----
    #
    # **原来这里有个 bug, 让所有工具调用都在为无用的思考烧 token。**
    #
    # 之前 `enable_thinking=False` 时**什么都不传**, 于是 MiniMax-M3 用它自己的
    # 默认值 —— 而 M3 默认是 **adaptive thinking**。也就是说: 工具里写
    # `enable_thinking=False` 期望"别思考, 直接给答案", 实际拿到的是
    # **96% 都是 <think>...</think> 的返回串**。
    #
    # 实测(10-07, 经典阴影综述的骨架 prompt, 31722 字符):
    #   不传 flag:      content=33715 字符, completion_tokens=10761, 105s
    #   type=disabled:  content= 1090 字符, completion_tokens=  499,  14s
    # JSON 解析结果两边一模一样(都是 7 个骨架任务) —— 也就是说那 97% 的输出
    # 和 87% 的等待时间**全是白花的**。
    #
    # 真正咬人的地方是**它吃 max_tokens**: 骨架原来给 `max_tokens=8000`,
    # 96% 被 think 占掉, 答案 JSON 挤在最后被截断, `_parse_theme_json` 对
    # "截在字符串中间"无解 -> 第二轮退回无骨架的块级归纳, 第一轮建好的领域
    # 结构被整块丢弃(实测 10-07 经典阴影第二轮)。
    #
    # `agent/controller.py` 早就写对了这条分支(主界面对话路径), 但**工具内部
    # 调用路径一直漏着**。这里对齐它。
    #
    # ⚠ 顺带修掉第二个 bug: `thinking.type` 对 M3 只接受 `"disabled"`,
    #   传 `"enabled"` 直接 400(`invalid thinking.type: "enabled"`)——
    #   实测确认过。所以 minimax + enable_thinking=True 时**不能**走上面那
    #   个 `enabled` 分支, 要用它的默认 adaptive 行为。
    # ⚠ 顺序有讲究: 先按 enable_thinking 统一填, **再**按 provider 裁剪。
#   第一版把 minimax 整条拆成独立分支, 结果 `else` 那个填 penalty 的分支只有
#   非 minimax 才走得到, 后面那两个 pop 就成了**死代码** —— 反向验证 M11
#   「不再为 minimax 剥 penalty」改坏了却测不出来, 因为压根没往里塞过。
#   回归测试 test_no_penalties_for_minimax 也是这么"绿得毫无意义"的。
#
    # `enable_thinking and provider != "minimax"`: M3 显式 "enabled" 会 400
    # (实测 `invalid thinking.type: "enabled"`), 所以 minimax 走它自己的默认
    # adaptive 行为, 不塞 extra_body。
    if enable_thinking and provider != "minimax":
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        })

    if provider == "minimax":
        # MiniMax 接口对 frequency_penalty / presence_penalty 支持有限, 统一剥离避免 400。
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)
        if not enable_thinking:
            # ↓ 这一行才是本 bug 的正事: 不显式关, M3 就在思考(见上面的实测数据)
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    # 2026-09-04 P0 fix:把 timeout 传给 client.chat.completions.create()
    # 之前 timeout 是 _call_llm 的入参但没被透传 — daily summary 类工具传 timeout=600
    # 完全被忽略,默认 180s 撞 timeout(用户 9 月 4 日实测 MiniMax-M3 跑 9.7k prompt 卡 180s)
    # OpenAI Python SDK 的 chat.completions.create(**kwargs) 接受 timeout kwarg 透传
    # 给底层 HTTPX 请求,直接放 kwargs 里即可
    kwargs["timeout"] = timeout

    try:
        import time as _t
        _t0 = _t.time()
        _log(f"[llm] _call_llm start  provider={provider} model={model} thinking={enable_thinking} prompt 长度={len(prompt)} 字符 timeout={timeout}s")
        # 9-27 P0 fix: 工具内部的 LLM 调用之前完全不走 cost_guard。
        # survey_generate 一次运行会在 _call_llm 里发 100+ 次请求 (每篇论文抽 fact /
        # 相关性评分 / 写章节 ...), 守卫却只看见 controller 那 ~50 次 —— ceiling
        # 完全约束不住真实开销。计费收口到这里。
        _is_ok, _guard_msg = cost_guard_charge()
        if not _is_ok:
            _log(f"[llm] _call_llm 被 cost_guard 阻断: {_guard_msg}")
            return f"❌ {_guard_msg.strip()}\n(cost_guard 阻断, 未发出 LLM 请求)"
        if _guard_msg:
            _log(f"[llm] cost_guard: {_guard_msg.strip()}")

        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        out = msg.content
        # 2026-09-05 P6.0 fix: deepseek 推理模型在 enable_thinking=False 时把"实际答案"放 reasoning_content,
        # message.content 字段为空字符串(实测 deepseek-v4-flash / pro 都是 0 chars,但 reasoning_content
        # 几 K 字符)。P6 prompt demo 实测发现,严重影响 _call_llm 工具调用。兜底:content 空但
        # reasoning_content 非空时,优先 reasoning_content。
        if not (out and out.strip()):
            rc = getattr(msg, "reasoning_content", None)
            if rc and rc.strip():
                out = rc
                _log(f"[llm] _call_llm fallback: content 空 → reasoning_content ({len(rc)} 字符)")
        # 10-06: **在出口洗掉 NUL / 控制字符**, 而不是给二十多个解析点各打一个补丁。
        # 实测 MiniMax-M3 连着 6 次输出 6 次混进 \x00(`Toffoli` → `Toff\x00oli`),
        # 症状是所有 JSON 解析点随机失败, 而根因只有一个 —— 输出流在字节层面
        # 被破坏。`json.loads(strict=False)` 治不了(strict=False 只放行 \n \t \r)。
        # 这里洗一次, 下面所有工具的 JSON 解析全部受益。详见 infra/text_clean.py。
        out = sanitize_llm_text(out)
        _log(f"[llm] _call_llm done  耗时 {_t.time()-_t0:.1f}s  返回 {len(out or '')} 字符")
        return out
    except Exception as e:
        _log(f"[llm] _call_llm 失败: {e}")
        return classify_llm_error(display_name, e)
