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
    # 2026-09-04 用户拍板:run_python 工具停用(安全考虑)
    # 之前 LLM 会用 run_python 跑 PyMuPDF 绕过 pdftotext 截断问题
    # → 现在 run_python 完全从 LLM 工具列表移除(连 schema 都不暴露)
    # 后续考虑:加 sandbox/资源限制/白名单 import 后再开放

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
            "description": (
                "Generate a detailed daily summary of new arXiv quant-ph papers. "
                "抓 https://arxiv.org/list/quant-ph/new,只取 'New submissions' 块(跳过 cross-list 和 replacement),"
                "LLM 生成结构化总结(每篇 paper 的核心贡献/方法/意义/关键词),返回完整 markdown。\n"
                "**触发场景**:\n"
                "  - 用户说'早上好'/'今日 arxiv'/'arxiv 每日总结'/'今天有什么新论文'\n"
                "  - 用户说'fetch arxiv daily summary' / '今天 quant-ph'\n"
                "**重要约束**:\n"
                "  - 这是 DIRECT_OUTPUT 工具 — 调用时**不要**写'我帮你抓了 N 篇' / '日期范围' / 'arXiv ID 列表' 之类的 prose\n"
                "  - 你**没有任何 fetch 能力**,在调工具前编出来的 paper 数/日期/arXiv ID 一定是幻觉(用户点开链接找不到)\n"
                "  - 调完工具后,直接输出工具返回的完整 markdown 即可"
            ),
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
            "description": (
                "Generate a detailed daily summary of new quantum information papers from an APS journal. "
                "调用后会通过 Playwright 抓取对应期刊的 recent 页面,提取真实 paper 列表(标题/作者/abstract/真 DOI),"
                "然后 LLM 生成结构化总结(核心贡献/方法/意义/关键词),最后以 markdown 形式完整返回。\n"
                "**触发场景**:\n"
                "  - 用户说'早上好'/'每日总结'/'今天有什么新论文' + 期刊名(PRL/PRA/PRB/...)\n"
                "  - 用户说'抓一下 PRA 今天的论文' / '帮我看下 PRL 最新'\n"
                "  - 用户说'fetch PRA daily summary' / 'summarize PRL recent'\n"
                "**重要约束**:\n"
                "  - 这是 DIRECT_OUTPUT 工具 — 调用时**不要**在你的回复里写'我帮你抓了 N 篇' / '日期范围' / 'DOI 列表' 之类的 prose\n"
                "  - 你**没有任何 fetch 能力**,在调工具前编出来的 paper 数/日期/DOI 一定是幻觉(用户点开全是 404)\n"
                "  - 调完工具后,直接输出工具返回的完整 markdown 即可,不要压缩/总结/转表格"
            ),
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
            "description": "Search the knowledge base by keyword/topic/author (FTS5 全文索引, 精确匹配)",
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
    # semantic_search: 语义检索（P1 新增 2026-09-02）
    # 走 P0.5 section-aware chunk + P0 chroma 向量库
    # 跟 query_knowledge 区别:query_knowledge 是关键词精确匹配,这个是语义相似
    # ============================================================
    "semantic_search": {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "基于语义相似度检索知识库(走 chroma + bge-large-en-v1.5 embedding,本地免费)。"
                "支持 metadata 过滤(year / journal / section / topic_keyword)。"
                "返回按论文分组的 chunk 列表,每条带 section / 相似度 / 摘要片段。"
                "适用场景: '找和 X 论文的 method 部分类似的内容' / '搜量子纠错码阈值'。"
                "跟 query_knowledge 的区别:query_knowledge 是关键词精确匹配,这个是语义近似。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "自然语言查询, e.g. 'quantum error correction threshold'"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                        "description": "返回多少篇论文(每篇可能命中多个 chunk)"
                    },
                    "year": {
                        "type": "integer",
                        "default": 0,
                        "description": "按发表年份过滤(0=不过滤)"
                    },
                    "journal": {
                        "type": "string",
                        "default": "",
                        "description": "按期刊过滤(arXiv / PRL / PRA / PRB 等,空=不过滤)"
                    },
                    "section": {
                        "type": "string",
                        "default": "",
                        "description": "按章节类别过滤(introduction / method / result / discussion 等,空=不过滤)"
                    },
                    "topic_keyword": {
                        "type": "string",
                        "default": "",
                        "description": "按关键词模糊过滤(在 keywords 字段里 contains 匹配,空=不过滤)"
                    }
                },
                "required": ["query"]
            }
        }
    },

    # ============================================================
    # P2.2: index_paper_to_chroma（纯本地, L0）
    # 把单篇论文从 papers.db + PDF 切分后索引到 chroma
    # ============================================================
    "index_paper_to_chroma": {
        "type": "function",
        "function": {
            "name": "index_paper_to_chroma",
            "description": (
                "把单篇论文切分后索引到 chroma papers collection(纯本地,无 LLM 调用)。"
                "数据源:papers.db 元数据 + data/papers/{paper_id}.pdf。"
                "幂等:同 paper_id 先删后插。"
                "适用:论文已 analyze 但没入 chroma, 或者快速补索引。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "论文 ID(arXiv id 或 DOI 等)"
                    }
                },
                "required": ["paper_id"]
            }
        }
    },

    # ============================================================
    # P2.3: reindex_all（纯本地, L0）
    # 全量重建 chroma papers collection
    # ============================================================
    "reindex_all": {
        "type": "function",
        "function": {
            "name": "reindex_all",
            "description": (
                "全量重建 chroma papers collection(走 P0.5 chunker 切 + chroma 写入)。"
                "适用:切分策略升级 / embedding model 切换 / 需要全量刷新。"
                "不调 LLM,纯本地;每 10 篇 sleep 0.5s 防 CPU 100%。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 0,
                        "description": "最多重建多少篇(0=全部,给测试用)"
                    }
                }
            }
        }
    },

    # ============================================================
    # P2.4: sync_recent_to_chroma（纯本地, L0）
    # 兜底:扫最近 summary_items,对未入 chroma 的 paper 切 + 入
    # ============================================================
    "sync_recent_to_chroma": {
        "type": "function",
        "function": {
            "name": "sync_recent_to_chroma",
            "description": (
                "兜底:扫最近 N 天的 summary_items(P0.6 抓文时落库),对尚未入 chroma 的 paper "
                "走 P0.5 chunker 切 + 入库(纯本地,无 LLM)。"
                "适用:抓文后 analyze 失败 / summary_items 落库但 chroma 没数据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "default": 7,
                        "description": "查最近 N 天的 summary_item"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "最多处理多少条(防单次调用太慢)"
                    }
                }
            }
        }
    },

    # ============================================================
    # P2.5: semantic_search_keywords — 关键词/摘要级语义检索(纯本地, L0)
    # 走 chroma paper_keywords collection(独立于 papers chunks)
    # 跟 semantic_search 区别: 这个粒度 = 1 篇 1 条 record(summary_item 粒度)
    # ============================================================
    "semantic_search_keywords": {
        "type": "function",
        "function": {
            "name": "semantic_search_keywords",
            "description": (
                "基于语义相似度检索 summary_items 库(走 chroma paper_keywords collection,本地免费)。"
                "支持 metadata 过滤(source / since_date / in_papers_db_only)。"
                "返回按相似度排序的 summary_item 列表, 每条带 paper_id / title / source / date / distance。"
                "跟 `semantic_search` 区别: `semantic_search` 走 papers chunks(已精细分析的论文,粒度=section级);"
                "这个走 summary_items(抓文级,粒度=论文级,适用于召回没分析过的最新抓文)。"
                "适用: '找和拓扑码贡献最像的论文' / '召回最近抓的 X 主题论文'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "自然语言查询, e.g. 'topological code fault tolerance'"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 30,
                        "description": "返回多少条 record(每条 = 1 篇 summary_item)"
                    },
                    "source": {
                        "type": "string",
                        "default": "",
                        "description": "按 source 过滤(arxiv_daily / aps_daily / reindex 等, 空=不过滤)"
                    },
                    "since_date": {
                        "type": "string",
                        "default": "",
                        "description": "按 summary_date >= since_date 过滤(ISO date 如 '2026-09-01', 空=不过滤)"
                    },
                    "in_papers_db_only": {
                        "type": "boolean",
                        "default": False,
                        "description": "只返回已精细分析过的 summary_item"
                    }
                },
                "required": ["query"]
            }
        }
    },

    # ============================================================
    # P3: similar_papers — 找相似论文(字段匹配优先 + chroma 召回补充, 纯本地 L0)
    # 设计: 字段匹配(同 keywords / authors / research_topic 打分) + chroma 关键词近邻
    # ============================================================
    "similar_papers": {
        "type": "function",
        "function": {
            "name": "similar_papers",
            "description": (
                "找相似论文: 字段匹配(同 keywords / authors / research_topic 打分)优先, "
                "再加 3-5 个 chroma 关键词语义近邻作多样性补充。走 papers.db 字段计算 + chroma paper_keywords 检索,纯本地无 LLM。"
                "返回 Markdown 列表,每条带 paper_id / 命中字段 / 字段匹配分或语义相似度。"
                "适用: '找和 X 论文方向类似的论文' / '推荐相关工作'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "源论文 ID(arXiv id 或 DOI 等,必须在 papers.db 里)"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20,
                        "description": "返回多少条"
                    },
                    "n_field": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 0,
                        "maximum": 15,
                        "description": "字段匹配拿多少条(0 = 全用语义)"
                    },
                    "n_semantic": {
                        "type": "integer",
                        "default": 4,
                        "minimum": 0,
                        "maximum": 10,
                        "description": "chroma 召回多少条语义近邻(0 = 全用字段)"
                    }
                },
                "required": ["paper_id"]
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
    "update_memory": {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": (
                "更新一条已有记忆。用于去重时合并（覆盖旧内容）或调整重要性。"
                "何时调用：save_memory 返回【已存在相似记忆】时；用户说【刚才记错了，应该是 XX】更正时；"
                "用户说【那条更重要了/没那么重要了】调整重要性时。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "已有记忆的 ID（save_memory 返回的 8 位 short id 即可，函数会自动补全完整 id）"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "新内容（覆盖原 content；空字符串表示不改）"
                    },
                    "new_importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": -1.0,
                        "description": "新重要性 0-1；-1 表示不改"
                    }
                },
                "required": ["memory_id"]
            }
        }
    },

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

    # ============================================================
    # survey_generate: 综述生成（P6 主工具, L2 档）
    # ============================================================
    # 2026-09-05 P6.2 上线
    # 6 段 pipeline: 候选池 → fact → 大纲 → 分章节 → 落库 → audit
    # 走 agent.prompts_survey 的 3 个 prompt 模板 + agent.infra.survey_workspace 工作区
    "survey_generate": {
        "type": "function",
        "function": {
            "name": "survey_generate",
            "description": (
                "生成综述(P6 主工具, L2 档默认关)。"
                "给定主题(topic), 走 6 段 pipeline 自动产出 markdown 综述章节, 落 surveys.db。"
                "候选池从 summary_items.db 走关键词匹配(0 LLM), 然后每篇 paper 1 次 LLM 提 fact,"
                "再 1 次 LLM 生成大纲, 接着每章 1 次 LLM 写 markdown, 最后 0 LLM 跑覆盖度审计。"
                "返回 markdown 报告含: theme_id, section_id 列表, 候选池 0 引用列表, 审计路径。"
                "适用: 用户说'生成 X 主题综述' / 'survey X' / '综述 X' 时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "综述主题, 例 '量子纠错码近期进展'"
                    },
                    "max_pool": {
                        "type": "integer",
                        "default": 40,
                        "minimum": 1,
                        "maximum": 200,
                        "description": "候选池上限, 默认 40 (单轮 6 段 pipeline token 估算 5-15K)"
                    },
                    "max_steps": {
                        "type": "integer",
                        "default": 10,
                        "description": "预留 P4 ReAct 共享接口, 本轮未使用"
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "True = 不调 LLM, 只构建候选池(快, 用于 UI 测)"
                    }
                },
                "required": ["topic"]
            }
        }
    },

    # ============================================================
    # survey_audit: 对已有 survey 跑覆盖度审计(P6.4, L0 档, 0 LLM)
    # ============================================================
    "survey_audit": {
        "type": "function",
        "function": {
            "name": "survey_audit",
            "description": (
                "对已落库的 survey 主题跑覆盖度审计(P6.4, L0 档, 0 LLM 调用)。"
                "不重生成章节, 只对 surveys.db 已有 theme + sections 跑:"
                "每章节候选池内/外引用统计, 总唯一引用, 候选池 0 引用列表(从 workspace _pool.json 读), 候选池外引用列表。"
                "传 topic 字段可启用候选池覆盖度审计(否则只跑章节级)。"
                "适用: 用户说'审计 X 综述' / 'survey audit X' / '看看 X 综述覆盖率' 时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "theme_id": {
                        "type": "integer",
                        "description": "survey_theme.id, surveys.db 里的主键"
                    },
                    "topic": {
                        "type": "string",
                        "description": "可选, 综述主题(用来从 workspace 找候选池); 不传则跳过候选池覆盖度"
                    }
                },
                "required": ["theme_id"]
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
