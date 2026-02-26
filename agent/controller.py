import json
from openai import OpenAI
from config import OPENAI_MODEL, DEEPSEEK_MODEL, DEEPSEEK_API_KEY
from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    search_papers, run_python, generate_daily_summary_arxiv, analyze_today_topic_arxiv, update_paper_comment,
    search_arxiv, download_paper, analyze_paper, scan_local_papers, query_knowledge
)
from agent.llm_clients import get_openai_client, get_deepseek_client

# =============================
# 工具定义
# =============================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search local research papers using vector index",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute python code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "Search papers about certain topic and not care about time on arXiv by query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "download_paper",
            "description": "Download an arXiv paper by ID to local folder",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"}
                },
                "required": ["arxiv_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_paper",
            "description": "分析一篇论文（通过 arXiv ID 或本地 PDF），提取结构化信息并保存到知识库。如果论文已存在，会覆盖原有记录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "arXiv ID 或 PDF 文件名（不含扩展名）"},
                    "pdf_path": {"type": "string", "description": "本地 PDF 路径（可选，如果不提供则尝试在默认目录查找）"},
                    "user_comment": {"type": "string", "description": "用户对论文的评论（可选）"}
                },
                "required": ["paper_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_paper_comment",
            "description": "为已分析过的论文追加一条用户评论（仅更新评论，不重新分析论文内容）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "论文标识符（与 analyze_paper 使用的 ID 一致）"},
                    "user_comment": {"type": "string", "description": "要添加的评论内容"}
                },
                "required": ["paper_id", "user_comment"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scan_local_papers",
            "description": "Scan local PDF folder and analyze new papers (incremental)",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge",
            "description": "Search the knowledge base by keyword/topic/author",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "topic": {"type": "string"},
                    "author": {"type": "string"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_daily_summary_arxiv",
            "description": "Generate a daily summary of new quantum physics papers from arXiv (triggered by 'good morning, arxiv')",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_today_topic_arxiv",
            "description": "Analyze today's or new arXiv papers on a specific topic (e.g., quantum algorithms, quantum error correction)",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The research topic to focus on"}
                },
                "required": ["topic"]
            }
        }
    }
]



# =============================
# 创建 DeepSeek client
# =============================
def get_deepseek_client():
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )


# =============================
# 流式输出函数（核心）
# =============================
def run_agent_stream(messages: list, mode="deepseek"):
    """
    流式调用 DeepSeek (支持思考模式、工具调用)，返回 token 生成器
    """
    if mode == "openai":
        client = get_openai_client()
        model = OPENAI_MODEL
        return _stream_openai(client, model, messages)
    else:  # deepseek
        client = get_deepseek_client()
        model = DEEPSEEK_MODEL

    while True:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
            temperature=0.3,
            max_tokens=2000,
            extra_body={"thinking": {"type": "enabled"}} if mode == "deepseek" else {}
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
                yield delta.content  # 实时输出文本块

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
        else:
            # 关键：如果没有工具调用，删除 tool_calls 字段（避免空数组）
            current_sub_turn_message.pop("tool_calls", None)

        # 将当前子回合的完整消息（包含 reasoning_content）追加到历史
        # 这是 DeepSeek 多子回合能够继续推理的必要条件
        messages.append(current_sub_turn_message)

        # 如果没有工具调用，说明本轮（可能包含多子回合）已结束，可以退出循环
        if not current_sub_turn_message.get("tool_calls"):
            break

        # ========== 关键修改点 2 ==========
        # 执行工具调用，并将结果以 "tool" 角色追加
        for tool_call in current_sub_turn_message["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            # 注意：这里需要异常处理，因为参数可能不完整
            try:
                args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as e:
                error_result = f"工具 {fn_name} 参数解析失败: {e}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": error_result
                })
                continue
            result = None

            if fn_name == "search_papers":
                result = search_papers(args["query"])
            elif fn_name == "run_python":
                result = run_python(args["code"])
            elif fn_name == "search_arxiv":
                result = search_arxiv(args["query"], args.get("max_results", 10))
            elif fn_name == "download_paper":
                result = download_paper(args["arxiv_id"])
            elif fn_name == "analyze_paper":
                paper_id = args.get("paper_id")
                pdf_path = args.get("pdf_path")
                user_comment = args.get("user_comment", "")
                if not paper_id:
                    yield "❌ 缺少论文标识符（paper_id）"
                    continue
                try:
                    result = analyze_paper(paper_id, pdf_path, user_comment)
                    yield f"✅ 论文分析完成，标题：{result.get('title', 'N/A')}"
                except Exception as e:
                    yield f"❌ 分析失败：{e}"
            elif fn_name == "update_paper_comment":
                paper_id = args.get("paper_id")
                user_comment = args.get("user_comment", "")
                if not paper_id:
                    yield "❌ 缺少论文标识符（paper_id）"
                    continue
                if not user_comment:
                    yield "❌ 缺少评论内容（user_comment）"
                    continue
                try:
                    result = update_paper_comment(paper_id, user_comment)
                    yield f"✅ 评论已更新至论文 {paper_id}，现有评论：{result['user_comments']}"
                except Exception as e:
                    yield f"❌ 更新评论失败：{e}"
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
            else:
                result = f"未知工具: {fn_name}"

            # 在添加 tool 消息之前
            if not tool_call.get("id"):
                print(f"错误：tool_call 缺少 id，跳过该工具调用")
                continue
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],  # 必须有此字段，且值来自对应的工具调用ID
                "content": json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
            })

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
        max_tokens=2000
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def call_model(client, model_name, user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

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
            args = json.loads(tool_call.function.arguments)
            result = None

            if fn == "search_papers":
                result = search_papers(args["query"])
            elif fn == "run_python":
                result = run_python(args["code"])
            elif fn == "search_arxiv":
                result = search_arxiv(args["query"], args.get("max_results", 10))
            elif fn == "download_paper":
                result = download_paper(args["arxiv_id"])
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