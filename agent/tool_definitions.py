"""
工具 schema 集中管理（参考 WAgent 实现）。

所有 LLM 可调用的工具的 OpenAI function calling schema 在这里按 name 索引声明。
实现分散在各工具文件，通过 @register_tool 装饰器按 name 自动绑定。

新增工具流程：
1. 在本文件 TOOL_SCHEMAS 添加新 key
2. 在对应工具文件中用 @register_tool(name="your_tool_name") 装饰函数
3. 完成。controller 无需改动。

注意：TOOL_SCHEMAS 只是数据声明，不依赖任何工具实现，避免循环 import。
"""
from typing import Dict, Any


# 所有 LLM 可调用的工具 schema 在这里声明
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # ============================================================
    # run_python: 执行任意 Python 代码（子进程运行）
    # ============================================================
    "run_python": {
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

    # ============================================================
    # search_arxiv: 在 arXiv 上搜索论文
    # ============================================================
    "search_arxiv": {
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

    # ============================================================
    # download_arxiv_paper: 下载 arXiv 论文 PDF
    # ============================================================
    "download_arxiv_paper": {
        "type": "function",
        "function": {
            "name": "download_arxiv_paper",
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

    # ============================================================
    # analyze_paper: 分析论文并保存到知识库
    # ============================================================
    "analyze_paper": {
        "type": "function",
        "function": {
            "name": "analyze_paper",
            "description": "分析一篇论文（通过 arXiv ID 或本地 PDF），提取结构化信息并保存到知识库。如果论文已存在，会覆盖原有记录。当用户说到本地时，将提供的论文id作为本地pdf文件名，优先在本地查找。",
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

    # ============================================================
    # update_paper_comment: 更新论文评论
    # ============================================================
    "update_paper_comment": {
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

    # ============================================================
    # scan_local_papers: 增量扫描本地 PDF
    # ============================================================
    "scan_local_papers": {
        "type": "function",
        "function": {
            "name": "scan_local_papers",
            "description": "Scan local PDF folder and analyze new papers (incremental)",
            "parameters": {"type": "object", "properties": {}}
        }
    },

    # ============================================================
    # generate_daily_summary_arxiv: arXiv 每日总结
    # ============================================================
    "generate_daily_summary_arxiv": {
        "type": "function",
        "function": {
            "name": "generate_daily_summary_arxiv",
            "description": "Generate a detailed daily summary of new quantum physics papers from arXiv (triggered by 'good morning, arxiv')",
            "parameters": {"type": "object", "properties": {}}
        }
    },

    # ============================================================
    # analyze_today_topic_arxiv: arXiv 某主题详细分析
    # ============================================================
    "analyze_today_topic_arxiv": {
        "type": "function",
        "function": {
            "name": "analyze_today_topic_arxiv",
            "description": "Generate a detailed daily summary of new arXiv papers on a specific topic (e.g., quantum algorithms, quantum error correction)",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The research topic to focus on"}
                },
                "required": ["topic"]
            }
        }
    },

    # ============================================================
    # generate_aps_daily_summary: APS 期刊每日总结
    # ============================================================
    "generate_aps_daily_summary": {
        "type": "function",
        "function": {
            "name": "generate_aps_daily_summary",
            "description": "Generate a detailed daily summary of new quantum information papers from an APS journal. Use this when the user says 'good morning' followed by a journal name like 'PRL' or 'PRA'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "journal": {
                        "type": "string",
                        "enum": ["prl", "pra", "prb", "prc", "prd", "pre", "prx", "prxquantum", "rmp", "prapplied"],
                        "description": "The APS journal code (e.g., 'prl' for Physical Review Letters, 'pra' for Physical Review A)."
                    }
                },
                "required": ["journal"]
            }
        }
    },

    # ============================================================
    # download_aps_paper: 下载 APS 论文 PDF
    # ============================================================
    "download_aps_paper": {
        "type": "function",
        "function": {
            "name": "download_aps_paper",
            "description": "通过DOI下载APS期刊论文（PRL, PRA, PRB等）的PDF到本地，如果DOI开头是10.1103，即为APS.当需要下载 APS 论文时，请直接调用 `download_aps_paper` 工具，并传入 DOI 参数。该工具会自动处理所有细节（包括获取重定向、下载PDF），无需额外运行 Python 代码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doi": {"type": "string", "description": "论文DOI，例如 10.1103/PhysRevLett.136.080203"}
                },
                "required": ["doi"]
            }
        }
    },

    # ============================================================
    # query_knowledge: 知识库查询
    # ============================================================
    "query_knowledge": {
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

    # ============================================================
    # get_recent_papers: 获取最近分析的论文
    # ============================================================
    "get_recent_papers": {
        "type": "function",
        "function": {
            "name": "get_recent_papers",
            "description": "获取最近分析的论文列表（按时间倒序），用于回顾最近看过的文章",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 15, "description": "最多返回条数"}
                }
            }
        }
    },

    # ============================================================
    # web_search: 联网搜索（多 provider，移植自 WAgent）
    # provider 链: bocha(中文主) -> tavily(海外) -> ddg(保底免 key)
    # key 通过 .api_keys.json 管理 (key 名: bocha / tavily)
    # ============================================================
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "联网搜索实时信息（新闻、百科、技术文档等），RAG 场景优化版本。"
                "自动根据 query 中英文选择搜索源：中文优先博查，英文优先 Tavily，"
                "两者都没配 key 时降级到 DuckDuckGo。返回网页标题、链接、摘要和正文，"
                "可直接用于 chunking 和 embedding。"
                "支持 auto/bocha/tavily/ddg 四种 provider 模式。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，例如 '北京下周天气' 或 'Python asyncio best practices'"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回结果条数，默认 5，最多 10",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "fetch_content": {
                        "type": "boolean",
                        "description": "是否抓取网页正文。RAG 场景推荐开启（默认 true）；只想看标题链接时设 false 可加快速度",
                        "default": True
                    },
                    "max_content_length": {
                        "type": "integer",
                        "description": "单条结果正文的最多字符数，默认 2000（RAG 友好）",
                        "default": 2000,
                        "minimum": 200,
                        "maximum": 8000
                    },
                    "provider": {
                        "type": "string",
                        "enum": ["auto", "bocha", "tavily", "ddg"],
                        "description": (
                            "搜索源选择。auto=按 query 语言自动选（推荐）；"
                            "bocha=博查（中文强，需 BOCHA_API_KEY）；"
                            "tavily=Tavily（海外强，需 TAVILY_API_KEY）；"
                            "ddg=DuckDuckGo（保底，免 key 但国内常超时）"
                        ),
                        "default": "auto"
                    }
                },
                "required": ["query"]
            }
        }
    },

    # ============================================================
    # save_memory: 保存一条长期记忆
    # ============================================================
    "save_memory": {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "保存一条长期记忆到记忆库，用于跨对话记住用户的偏好、决策、"
                "对论文的笔记、洞见等。当用户说【记住】【记下】【我的偏好是】【我决定】【约定】等时必须调用。"
                "content 必须是完整的自包含陈述（含主语），不能是【是】这种简略词。"
                "用户对长期不变信息的偏好/决策，importance 建议 0.7-0.9；普通事实 0.5。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "记忆内容（完整自包含陈述，含主语，例如【用户偏好 LaTeX 公式用 aligned 块排版】）"
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["user_preference", "research_fact", "paper_note", "reflection", "task_state", "general"],
                        "description": (
                            "记忆类型: user_preference=用户偏好, research_fact=研究事实/决策, "
                            "paper_note=论文笔记（必须填 related_paper_id）, "
                            "reflection=洞见, task_state=任务状态, general=其他"
                        ),
                        "default": "general"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5,
                        "description": "重要性 0-1。用户的明确偏好/决策建议 0.7-0.9；普通事实 0.5"
                    },
                    "related_paper_id": {
                        "type": "string",
                        "description": "关联论文 ID（paper_note 类型必填）"
                    },
                    "metadata_json": {
                        "type": "string",
                        "description": "额外元数据 JSON 字符串（可空），例如 '{\"topic\": \"QAOA\"}'"
                    }
                },
                "required": ["content"]
            }
        }
    },

    # ============================================================
    # recall_memories: 检索记忆
    # ============================================================
    "recall_memories": {
        "type": "function",
        "function": {
            "name": "recall_memories",
            "description": (
                "从记忆库检索相关记忆。当用户引用过去决策（如【我之前说过…】）、"
                "延续某话题讨论、需要确认用户偏好时调用。"
                "和 query_knowledge 的区别：query_knowledge 查的是论文，recall_memories 查的是用户偏好/决策/笔记。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索文本（自然语言，问题/关键词均可）"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 5,
                        "description": "最多返回条数（默认 5）"
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["user_preference", "research_fact", "paper_note", "reflection", "task_state"],
                        "description": "限定记忆类型（可空）"
                    },
                    "related_paper_id": {
                        "type": "string",
                        "description": "限定关联论文（可空；填了就走精确匹配不走向量检索）"
                    },
                    "min_importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.0,
                        "description": "最低重要性 0-1（默认 0）"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["vector", "keyword", "hybrid"],
                        "default": "vector",
                        "description": "检索方法: vector(默认，语义) / keyword(关键词) / hybrid(混合)"
                    }
                },
                "required": ["query"]
            }
        }
    },
}


def get_tool_schema(name: str):
    """按名字取单个 schema（方便 controller / 测试用）"""
    return TOOL_SCHEMAS.get(name)


def get_all_tool_names():
    """所有已注册工具的 LLM 端名字"""
    return list(TOOL_SCHEMAS.keys())
