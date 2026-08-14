# agent/controller.py
import json
import re
from openai import OpenAI
from config import OPENAI_MODEL, DEEPSEEK_MODEL, DEEPSEEK_API_KEY, EMBEDDING_ENABLED
from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    scan_local_papers, run_python, update_paper_comment,
    generate_daily_summary_arxiv, analyze_today_topic_arxiv,
    generate_aps_daily_summary, download_aps_paper,
    search_arxiv, download_arxiv_paper, analyze_paper
)
from agent.llm_clients import get_openai_client, get_deepseek_client
from agent.tool_definitions import BASE_TOOLS, EMBEDDING_TOOLS, EMBEDDING_TOOL_NAMES
from agent.stream_utils import split_markdown_bubbles

if EMBEDDING_ENABLED:
    from agent.tools import (search_papers, query_knowledge)

# 直接输出工具：这些工具返回的内容就是最终答案，不需要再交给模型处理
DIRECT_OUTPUT_TOOLS = {"generate_daily_summary_arxiv", "generate_aps_daily_summary"}

# =============================
# 创建 DeepSeek client
# =============================
def get_deepseek_client():   # 保留原函数，避免外部依赖改动
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

# =============================
# 内部辅助：将一段文本按 Markdown 标题分割为气泡并 yield
# =============================
NEW_BUBBLE = "\x1cNEW_BUBBLE"
NEW_BUBBLE_BATCH = "\x1cNEW_BUBBLE_BATCH"

def _yield_bubbles(content: str):
    """对任意文本内容，按 Markdown 三级+标题分割气泡并 yield"""
    if not content:
        return
    bubbles = split_markdown_bubbles(content)
    for bubble in bubbles:
        yield bubble
        yield NEW_BUBBLE

# =============================
# 流式输出函数（核心）
# =============================
def run_agent_stream(
    messages: list,
    mode: str = "deepseek",
    # 以下全部可配置
    model: str = None,                     # 如果为 None，则使用 config.DEEPSEEK_MODEL
    temperature: float = 0.3,
    top_p: float = 0.9,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 40000,                # 根据 model 自动调整也行，但就给个默认值
    timeout: int = 180,
    enable_thinking: bool = True,          # 是否开启思考模式
    reasoning_effort: str = "high"):

    if mode == "openai":
        client = get_openai_client()
        model = OPENAI_MODEL
        return _stream_openai(client, model, messages)
    else:  # deepseek
        client = get_deepseek_client()
        model = DEEPSEEK_MODEL

    if EMBEDDING_ENABLED:
        TOOLS = BASE_TOOLS + EMBEDDING_TOOLS
    else:
        TOOLS = BASE_TOOLS

    while True:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body={"thinking": {"type": "enabled"}} if enable_thinking else {},
            reasoning_effort=reasoning_effort if enable_thinking else None
        )

        # 用于累积当前子回合（sub-turn）的完整响应
        current_sub_turn_message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": []
        }
        tool_call_dict = {}

        for chunk in stream:
            delta = chunk.choices[0].delta

            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                current_sub_turn_message["reasoning_content"] += delta.reasoning_content

            if delta.content:
                current_sub_turn_message["content"] += delta.content
                # ✅ 关键修改：不再在此处实时 yield，改为完整累积后再分割气泡

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_dict:
                        tool_call_dict[idx] = {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments or ""}
                        }
                    else:
                        if tc.function.arguments:
                            tool_call_dict[idx]["function"]["arguments"] += tc.function.arguments

        if tool_call_dict:
            current_sub_turn_message["tool_calls"] = list(tool_call_dict.values())
            for tc in current_sub_turn_message["tool_calls"]:
                print(f"模型请求调用工具: {tc['function']['name']}")
        else:
            current_sub_turn_message.pop("tool_calls", None)

        # 将当前子回合的完整消息追加到历史
        messages.append(current_sub_turn_message)

        # ========== ✅ 对累积的文本内容进行气泡分割并输出 ==========
        accumulated_text = current_sub_turn_message.get("content", "")
        # has_tool_calls = bool(current_sub_turn_message.get("tool_calls"))

        if accumulated_text:
            for chunk_text in _yield_bubbles(accumulated_text):
                yield chunk_text

        # 如果没有工具调用，说明本轮已结束
        if not current_sub_turn_message.get("tool_calls"):
            break

        # ========== 工具调用执行 ==========
        tool_responses = []
        for tool_call in current_sub_turn_message["tool_calls"]:
            fn_name = tool_call["function"]["name"]

            if not EMBEDDING_ENABLED and fn_name in EMBEDDING_TOOL_NAMES:
                content = f"工具 {fn_name} 需要 embedding 支持，但当前已在配置中禁用。"
                tool_responses.append((tool_call["id"], content))
                yield NEW_BUBBLE
                yield content
                yield NEW_BUBBLE
                continue

            try:
                args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as e:
                content = f"工具 {fn_name} 参数解析失败: {e}"
                tool_responses.append((tool_call["id"], content))
                yield NEW_BUBBLE
                yield content
                yield NEW_BUBBLE
                continue

            result = None
            if fn_name == "search_papers":
                result = search_papers(args["query"])
            elif fn_name == "run_python":
                result = run_python(args["code"])
            elif fn_name == "search_arxiv":
                result = search_arxiv(args["query"], args.get("max_results", 10))
            elif fn_name == "download_arxiv_paper":
                result = download_arxiv_paper(args["arxiv_id"])
            elif fn_name == "analyze_paper":
                paper_id = args.get("paper_id")
                pdf_path = args.get("pdf_path")
                user_comment = args.get("user_comment", "")
                if not paper_id:
                    yield NEW_BUBBLE
                    yield "❌ 缺少论文标识符（paper_id）"
                    yield NEW_BUBBLE
                    continue
                try:
                    result = analyze_paper(paper_id, pdf_path, user_comment)
                    yield NEW_BUBBLE
                    yield f"✅ 论文分析完成，标题：{result.get('title', 'N/A')}"
                    yield NEW_BUBBLE
                except Exception as e:
                    yield NEW_BUBBLE
                    yield f"❌ 分析失败：{e}"
                    yield NEW_BUBBLE
            elif fn_name == "update_paper_comment":
                paper_id = args.get("paper_id")
                user_comment = args.get("user_comment", "")
                if not paper_id:
                    yield NEW_BUBBLE
                    yield "❌ 缺少论文标识符（paper_id）"
                    yield NEW_BUBBLE
                    continue
                if not user_comment:
                    yield NEW_BUBBLE
                    yield "❌ 缺少评论内容（user_comment）"
                    yield NEW_BUBBLE
                    continue
                try:
                    result = update_paper_comment(paper_id, user_comment)
                    yield NEW_BUBBLE
                    yield f"✅ 评论已更新至论文 {paper_id}，现有评论：{result['user_comments']}"
                    yield NEW_BUBBLE
                except Exception as e:
                    yield NEW_BUBBLE
                    yield f"❌ 更新评论失败：{e}"
                    yield NEW_BUBBLE
            elif fn_name == "scan_local_papers":
                result = scan_local_papers()
            elif fn_name == "query_knowledge":
                result = query_knowledge(
                    args.get("keyword"),
                    args.get("topic"),
                    args.get("author")
                )
            elif fn_name == "generate_daily_summary_arxiv":
                result = generate_daily_summary_arxiv()
            elif fn_name == "analyze_today_topic_arxiv":
                result = analyze_today_topic_arxiv(args["topic"])
            elif fn_name == "generate_aps_daily_summary":
                journal = args.get("journal")
                try:
                    result = generate_aps_daily_summary(journal)
                except Exception as e:
                    error_msg = f"❌ aps总结生成失败：{e}"
                    yield NEW_BUBBLE
                    yield error_msg
                    yield NEW_BUBBLE
                    result = error_msg
            elif fn_name == "download_aps_paper":
                result = download_aps_paper(args["doi"])
            else:
                result = f"未知工具: {fn_name}"

            # 将结果转换为字符串
            if result is None:
                content = f"工具 {fn_name} 执行无返回"
            elif isinstance(result, str):
                content = result
            else:
                content = json.dumps(result, ensure_ascii=False)
            tool_responses.append((tool_call["id"], content))

        # 将所有 tool 消息追加到 messages
        for tool_call_id, content in tool_responses:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content
            })

        # 检查本轮调用的工具是否属于直接输出类型
        called_tool_names = {tc["function"]["name"] for tc in current_sub_turn_message["tool_calls"]}
        if called_tool_names & DIRECT_OUTPUT_TOOLS:
            for _, content in tool_responses:
                bubbles = split_markdown_bubbles(content)
                for bubble in bubbles:
                    yield ('\n' + bubble).lstrip('\n')
                    yield NEW_BUBBLE
            break

def _stream_openai(client, model, user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.3,
        max_tokens=8192
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def call_model(client, model_name, user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    if EMBEDDING_ENABLED:
        TOOLS = BASE_TOOLS + EMBEDDING_TOOLS
    else:
        TOOLS = BASE_TOOLS

    while True:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            fn = tool_call.function.name
            if not EMBEDDING_ENABLED and fn in EMBEDDING_TOOL_NAMES:
                result = f"工具 {fn} 需要 embedding 支持，但当前已在配置中禁用。"
                messages.append(msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                continue
            args = json.loads(tool_call.function.arguments)
            result = None

            if fn == "search_papers":
                result = search_papers(args["query"])
            elif fn == "run_python":
                result = run_python(args["code"])
            elif fn == "search_arxiv":
                result = search_arxiv(args["query"], args.get("max_results", 10))
            elif fn == "download_arxiv_paper":
                result = download_arxiv_paper(args["arxiv_id"])
            elif fn == "analyze_paper":
                result = analyze_paper(
                    args["paper_id"],
                    args.get("pdf_path"),
                    args.get("user_comment", "")
                )
            elif fn == "scan_local_papers":
                result = scan_local_papers()
            elif fn == "query_knowledge":
                result = query_knowledge(
                    args.get("keyword"),
                    args.get("topic"),
                    args.get("author")
                )
            elif fn == "generate_daily_summary_arxiv":
                result = generate_daily_summary_arxiv()
            elif fn == "analyze_today_topic_arxiv":
                result = analyze_today_topic_arxiv(args["topic"])
            elif fn == "generate_aps_daily_summary":
                result = generate_aps_daily_summary(args["journal"])
            elif fn == "download_aps_paper":
                result = download_aps_paper(args["doi"])
            else:
                result = "Unknown tool"

            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
            })

def run_agent(user_input, mode="deepseek"):
    if mode == "openai":
        client = get_openai_client()
        return call_model(client, OPENAI_MODEL, user_input)
    if mode == "deepseek":
        client = get_deepseek_client()
        return call_model(client, DEEPSEEK_MODEL, user_input)
    if mode == "compare":
        openai_client = get_openai_client()
        deepseek_client = get_deepseek_client()
        r1 = call_model(openai_client, OPENAI_MODEL, user_input)
        r2 = call_model(deepseek_client, DEEPSEEK_MODEL, user_input)
        return {"OpenAI": r1, "DeepSeek": r2}
