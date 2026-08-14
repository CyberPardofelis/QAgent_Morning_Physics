# agent/tool_definitions.py
"""
工具定义（与具体实现解耦）
"""

# 不依赖 embedding 的基础工具（始终可用）
BASE_TOOLS = [
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
    {
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
            "name": "generate_daily_summary_arxiv",
            "description": "Generate a detailed daily summary of new quantum physics papers from arXiv (triggered by 'good morning, arxiv')",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
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
    {
        "type": "function",
        "function": {
            "name": "generate_aps_daily_summary",
            "description": "Generate a detailed daily summary of new quantum information papers from an APS journal. Use this when the user says 'good morning' followed by a journal name like 'PRL' or 'PRA'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "journal": {
                        "type": "string",
                        "enum": ["prl", "pra", "prb", "prc", "prd", "pre", "prx","prxquantum","rmp","prapplied"],
                        "description": "The APS journal code (e.g., 'prl' for Physical Review Letters, 'pra' for Physical Review A)."
                    }
                },
                "required": ["journal"]
            }
        }
    },
    {
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
    }
]

# 依赖 embedding 的工具（仅在 EMBEDDING_ENABLED = True 时启用）
EMBEDDING_TOOLS = [
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
]

EMBEDDING_TOOL_NAMES = {tool["function"]["name"] for tool in EMBEDDING_TOOLS}