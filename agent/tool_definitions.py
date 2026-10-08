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
from typing import Any

# 所有 LLM 可调用的工具 schema 在这里声明
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
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
            "description": (
                "在 arXiv 上搜索论文, 走 arxiv API. "
                "返回 Markdown 格式论文列表 (ID/标题/作者/日期, summary_only=True 时不含摘要全文). "
                "**不要把工具返回的 markdown 完整复制到对话窗口** — 自己组织回复, 只列关键信息."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词, 英文"},
                    "max_results": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                    "summary_only": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "9-18 加: True (默认) 只返 ID/标题/作者/日期, "
                            "不含 200 字符摘要全文. False 才返完整摘要."
                        )
                    }
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
            "description": (
                "下载 arXiv 论文 PDF 到本地. "
                "**注意**: arxiv 反爬严格, 主路径失败会自动 fallback 到 PDF URL, "
                "两条路径都拒时建议改走 analyze_paper (不下载 PDF, 走 API 拉元数据)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv ID, 支持 '2501.12345' / 'arXiv:2501.12345' / URL"}
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
    # search_aps: APS 期刊关键词搜索 (2026-09-09 P0.6 新增)
    # 走 OpenAlex, 覆盖 PRL/PRA/PRB/PRC/PRD/PRE/PRX/PRXQuantum/RMP
    # 默认 quant-ph 主阵地: PRL + PRA + PRX Quantum + PRX + RMP
    # ============================================================
    "search_aps": {
        "type": "function",
        "function": {
            "name": "search_aps",
            "description": (
                "在 APS 期刊 (PRL/PRA/PRB/PRC/PRD/PRE/PRX/PRXQuantum/RMP 等) 中按关键词搜索论文, "
                "走 OpenAlex 公开 API (不爬 journals.aps.org, 避免反爬)。返回 Markdown 格式论文列表, "
                "含 DOI 链接 / 作者 / 发布日期 / 期刊 / 摘要 / 引用数 / PDF (OA)。\n"
                "**触发场景**:\n"
                "  - 用户说 '搜一下 APS 上 X 主题的论文' / 'PRL 有没有 X 的文章' / '在 PRA 找 X'\n"
                "  - 用户说 'search APS for X' / 'find PRL paper on X' / 'APS X 方向'\n"
                "**重要约束**:\n"
                "  - journals 参数是逗号分隔的 short_name 列表, 例: 'prl,pra,prxquantum'\n"
                "  - 空 = 默认 quant-ph 集 (prl, pra, prxquantum, prx, rmp)\n"
                "  - 是 DIRECT_OUTPUT 工具, 调完工具后直接输出工具返回的 markdown"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "自由文本查询, e.g. 'quantum error correction' / 'topological insulator'"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 200,
                        "description": "最多返回多少篇"
                    },
                    "journals": {
                        "type": "string",
                        "default": "",
                        "description": (
                            "逗号分隔的 APS 期刊 short_name, 限定搜索范围. "
                            "可选: prl, pra, prb, prc, prd, pre, prx, prxquantum, prapplied, "
                            "prresearch, prmaterials, rmp. "
                            "空 = 默认 quant-ph 集 (prl, pra, prxquantum, prx, rmp)"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },

    # ============================================================
    # search_iop: IOP 期刊关键词搜索 (2026-09-09 P0.6 新增)
    # 走 OpenAlex, 覆盖 NJP / J. Phys. A/B/D/CM / J. Opt. / J. Phys. Conf.
    # 默认 quant-ph 主阵地: NJP + J. Phys. A + J. Phys. B + J. Opt.
    # ============================================================
    "search_iop": {
        "type": "function",
        "function": {
            "name": "search_iop",
            "description": (
                "在 IOP 期刊 (NJP / J. Phys. A/B/D/Condensed Matter / J. Opt. / Conf. Series) 中按关键词搜索论文, "
                "走 OpenAlex 公开 API (不爬 iopscience.iop.org, 避免 hCaptcha)。返回 Markdown 格式论文列表, "
                "含 DOI 链接 / 作者 / 发布日期 / 期刊 / 摘要 / 引用数 / PDF (OA)。\n"
                "**触发场景**:\n"
                "  - 用户说 '搜一下 NJP 上 X 主题的论文' / 'IOP 有没有 X 的文章' / '在 New Journal of Physics 找 X'\n"
                "  - 用户说 'search IOP for X' / 'find NJP paper on X'\n"
                "**重要约束**:\n"
                "  - journals 参数是逗号分隔的 short_name 列表, 例: 'njp,jphysa,jopt'\n"
                "  - 空 = 默认 quant-ph 集 (njp, jphysa, jphysb, jopt)\n"
                "  - 是 DIRECT_OUTPUT 工具, 调完工具后直接输出工具返回的 markdown"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "自由文本查询, e.g. 'quantum error correction' / 'topological phase'"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 200,
                        "description": "最多返回多少篇"
                    },
                    "journals": {
                        "type": "string",
                        "default": "",
                        "description": (
                            "逗号分隔的 IOP 期刊 short_name, 限定搜索范围. "
                            "可选: njp, jphysa, jphysb, jphysd, jphyscm, jopt, jphysconf. "
                            "空 = 默认 quant-ph 集 (njp, jphysa, jphysb, jopt)"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },

    # ============================================================
    # query_knowledge: 知识库查询 (本地摘要库, 不是搜外网)
    # ============================================================
    "query_knowledge": {
        "type": "function",
        "function": {
            "name": "query_knowledge",
            "description": (
                "在**本地知识库** (papers.db 已分析的论文) 中按 keyword/topic/author 搜索. "
                "走 SQLite FTS5 全文索引. 返回 Markdown 论文列表 (ID/作者/方向/关键词).\n"
                "**重要约束**:\n"
                "  - 三个参数 (keyword/topic/author) 至少传一个, 全空 = 全表扫描会拒\n"
                "  - **不是搜外网**. 想搜外网用: search_arxiv / search_aps / search_iop / web_search\n"
                "  - keyword 走 FTS5, 对 : / \" / ^ / ( / + 等字符敏感, 太复杂会报 FTS5 syntax error\n"
                "  - topic (research_topic LIKE) / author (authors LIKE) 用普通 LIKE, 任意字符都行\n"
                "  - keyword / topic 都可空, author 也可空\n"
                "**触发场景**:\n"
                "  - 用户问 '我之前分析过的 X 论文' / '知识库里有没有 X' / '按作者 Y 找论文'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "default": "",
                        "description": "FTS5 全文匹配, 例 'quantum error correction'. 避免特殊字符 (: \" ( + 等)."
                    },
                    "topic": {
                        "type": "string",
                        "default": "",
                        "description": "research_topic LIKE 模糊匹配, 例 'quantum computing'. 任意字符都行."
                    },
                    "author": {
                        "type": "string",
                        "default": "",
                        "description": "authors LIKE 模糊匹配, 例 'Bennett'. 任意字符都行."
                    }
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
                "**9-09 改: 只返总数 (例: '找到 5 条相关 summary_item'), 不返 paper_id / title / 摘要 / 相似度 等具体内容。**"
                "LLM 需要单篇详情时, 改调 `query_knowledge(paper_id)` 按 id 精确查, 不要在本工具结果里展开。"
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
                    },
                    "summary_only": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "9-18 加: True (默认) 只返 title/link/snippet (snippet 截 200 字符), "
                            "**不返 content 全文**. 防止 LLM 把 2KB+ 全文 JSON 复制到对话窗口污染 UI. "
                            "False 才返完整 RAG 友好 JSON (含 content 全文, 适合 embedding/chunking)."
                        )
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
                    },
                    "concept_aware": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "概念召回开关，默认关闭。开启会额外发 1 次 LLM 请求抽 query 的概念，"
                            "默认关闭以保证本工具 0 次 LLM 调用（L0 档）。"
                            "仅在用户明确提到某个概念/术语且向量召回不准时才开启。"
                        )
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
                    "organize": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "走**主题驱动**路线: 先从 fact 内容归纳出领域主题地图, "
                            "再围绕主题写叙述。章节结构由文献决定而非预设模板, "
                            "写出来更像综述(关系/对比/推论占比高得多), 引文也不会编造。"
                            "默认 False = 旧的 5 段模板路线(保留, 可对照回退)。"
                        )
                    },
                    "phase": {
                        "type": "string",
                        "enum": ["all", "plan", "write", "organize"],
                        "default": "all",
                        "description": (
                            "执行阶段。all=一次跑完(默认)。"
                            "plan=只生成大纲就停(候选池→fact→大纲), 之后可用 survey_outline 改大纲。"
                            "write=从 workspace 读回 pool/fact/大纲(可含用户改动)只写章节正文, "
                            "不重复跑 Phase A 和 fact 抽取, 省掉几十次 LLM 调用。"
                        )
                    },
                    "section_idx": {
                        "type": "integer",
                        "default": -1,
                        "description": (
                            "配合 phase=\"write\": >=0 时只写第 section_idx 章(单章重写), "
                            "其余章节不动。-1(默认)=写全部。"
                        )
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
    # ============================================================
    # survey_outline / survey_outline_export: 大纲查看与编辑 (L0, 0 LLM)
    # ============================================================
    "survey_rewrite_paragraphs": {
        "type": "function",
        "function": {
            "name": "survey_rewrite_paragraphs",
            "description": (
                "重写已生成综述里某一章的指定段落, 其余段落保持原样(L2)。"
                "整章重写要 20000 tokens 全部重来, 只改一段时用这个更省。"
                "用户说\"第3章第2段太空了补一下\"/\"gap那章最后一段没给反例\"时用。"
                "建议先用 dry_run=True 看清段落编号再动手。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "综述主题, 必须与生成时完全一致"
                    },
                    "section_idx": {
                        "type": "integer",
                        "description": "章节序号(0-based, 对应大纲里的 [i] 标号)"
                    },
                    "para_indices": {
                        "type": "string",
                        "description": (
                            "要重写的段号, 逗号分隔, 如 \"1\" 或 \"0,2,3\"。"
                            "不传则重写整章的所有正文段(不含标题)。"
                            "先用 dry_run=True 看本章有哪几段。"
                        )
                    },
                    "instruction": {
                        "type": "string",
                        "description": "额外要求, 如 \"补一个2023年后的反例\"/\"语气收紧\"/\"更具体一些\""
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "True=只切分并展示段落结构, 不调 LLM 不落库(0 token, 用来试参数)"
                    }
                },
                "required": ["topic", "section_idx"]
            }
        }
    },

    "survey_outline": {
        "type": "function",
        "function": {
            "name": "survey_outline",
            "description": (
                "查看/修改某主题的综述大纲(0 LLM 调用, 纯本地)。"
                "典型流程: survey_generate(phase=\"plan\") 生成大纲 → survey_outline(action=\"view\") "
                "给用户看 → survey_outline(action=\"update\", sections=...) 改 → "
                "survey_generate(phase=\"write\") 按改后的大纲写正文。"
                "用户说\"看看大纲\"/\"改一下大纲\"/\"第3章标题改成X\"/\"加一章XX\"时用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "综述主题, 必须与生成时完全一致(大纲按 topic hash 定位)"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["view", "update"],
                        "default": "view",
                        "description": "view=查看当前大纲; update=用 sections 覆盖更新"
                    },
                    "sections": {
                        "type": "string",
                        "description": (
                            "action=\"update\" 时必填。JSON 字符串, 形如 "
                            "{\"sections\":[{\"title\":\"...\",\"section_type\":\"intro\","
                            "\"sub_points\":[\"...\"],\"is_transition\":\"no\"}]}; "
                            "也接受顶层直接是数组。section_type 可用 5 段保底类型"
                            "(intro/progress/method_compare/gap/outlook) 或 custom。"
                            "先 view 拿到现版, 改完整体传回来(未提及的章节会丢, 所以要传全量)。"
                        )
                    }
                },
                "required": ["topic"]
            }
        }
    },

    "survey_outline_export": {
        "type": "function",
        "function": {
            "name": "survey_outline_export",
            "description": "导出某主题大纲的可编辑 JSON 模板(0 LLM)。用户想把大纲拿到外部编辑器改时用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "综述主题"}
                },
                "required": ["topic"]
            }
        }
    },

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

    # ================================================================
    # 2026-10-08 P0 加: P7 科研验证闭环 (research_*)
    #
    # 这批工具全部 exposed=False(见 registry.ToolDef.exposed), 只注册不暴露:
    # ToolRegistry.get() 拿得到 handler, 但 list_schemas() 默认跳过它们,
    # 所以 LLM 看不见。要打开需设环境变量 QAGENT_RESEARCH_EXPOSE=1,
    # 由 agent/tools/research/_flags.py 读取。
    #
    # 为什么这么绕: 闭环的价值全在"结论靠不靠谱"上。在评估集测出来之前
    # 放给 LLM, 等于让它拿着一个可能硬吹的工具去回答用户。
    # 详见 docs/RESEARCH_CLOSED_LOOP_PLAN.md §16。
    # ================================================================

    "research_run_code": {
        "type": "function",
        "function": {
            "name": "research_run_code",
            "description": (
                "在受限沙箱里执行一段 Python 并返回输出 + 证据编号 (L0 档, 0 LLM 调用)。"
                "沙箱有 AST 白名单门 + 子进程隔离 + 超时杀进程树, "
                "只防手滑(误删文件/死循环), 不是安全边界。"
                "★ 纪律: 判定结论不靠你读 stdout, 而靠代码里 print('VERDICT: PROVED'/'REFUTED'/"
                "'PARTIAL'/'INCONCLUSIVE') 标记。没有这个标记一律判 untested —— "
                "代码跑通不等于命题验证了。"
                "允许 import: math/sympy/numpy/scipy/networkx/z3/itertools/functools/re 等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 源码, 不要 markdown 代码块包裹"},
                    "purpose": {"type": "string", "description": "这段代码要验什么(一句话, 进证据记录)"},
                    "timeout": {"type": "integer", "description": "硬超时秒数, 默认 20"}
                },
                "required": ["code"]
            }
        }
    },

    "research_verify": {
        "type": "function",
        "function": {
            "name": "research_verify",
            "description": (
                "验证一条命题并返回带证据的结论 (L1 档, 会调一次 LLM 生成验证代码)。"
                "流程: 把命题和范围喂给 LLM → 拿到验证代码 → 沙箱执行 → "
                "按 VERDICT 标记落证据。verdict 只有 6 档: "
                "proved / refuted / partial / failed / inconclusive / untested。"
                "★ partial = 有限范围抽查通过, 不等于证明了 —— 有限枚举永远只能给 partial。"
                "不给 code 时会请 LLM 生成; 自己已经写好验证代码就用 code 直接传(省一次 LLM)。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "要验证的命题, 写成可判定的形式(带明确范围)"},
                    "scope": {"type": "string", "description": "变量范围, 如 'n=2..100' / 'n 为正整数'"},
                    "how": {"type": "string", "description": "symbolic(默认) / enumerate, 提示 LLM 该用什么手法"},
                    "timeout": {"type": "integer", "description": "代码执行硬超时秒数, 默认 20"},
                    "task_hash": {"type": "string", "description": "复用已有工作区; 不传按 claim 自动生成"},
                    "code": {"type": "string", "description": "自己写好的验证代码(可选)。传了就跳过 LLM 生成"},
                    "dry_run": {"type": "boolean", "description": "True = 只跑沙箱门不真执行, 默认 False"}
                },
                "required": ["claim"]
            }
        }
    },

    "research_sympy": {
        "type": "function",
        "function": {
            "name": "research_sympy",
            "description": (
                "做一次符号计算并落一条证据 (L0 档, 0 LLM 调用)。"
                "适合恒等式化简、微积分、化简到 0 判等、极限这类一步到位的符号题。"
                "返回 {result, verdict, evidence_id}, result 是化简后的字符串。"
                "想验恒等式, op 传 'simplify' 再看 result 是否等于 0。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "sympy 表达式源码, 如 '(x**2+2*x+1)'"},
                    "op": {"type": "string", "description": "simplify(默认) / diff / integrate / limit / factor / expand"},
                    "params": {"type": "string", "description": "op 的参数, 如 integrate 传 'x', limit 传 'x,0,1'"},
                    "timeout": {"type": "integer", "description": "硬超时秒数, 默认 20"}
                },
                "required": ["expr"]
            }
        }
    },

    "research_z3": {
        "type": "function",
        "function": {
            "name": "research_z3",
            "description": (
                "求解 SMT-LIB2 约束并落证据 (L0 档, 0 LLM 调用)。"
                "goal='sat' 找一组满足约束的解; goal='prove' 走『前提(命题成立) ∧ 结论(命题不成立)』:"
                "unsat → proved(无反例即成立), sat → refuted(模型本身就是反例)。"
                "★ goal='prove' 是唯一能给全域级结论的工具 —— 枚举再大也只是 partial。"
                "constraints 是完整的 SMT-LIB2 文本, 如 "
                "'(declare-const x Int)(assert (> x 41))(assert (< x 43))'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "constraints": {"type": "string", "description": "SMT-LIB2 约束文本(含 declare-const)"},
                    "goal": {"type": "string", "description": "sat(默认, 找解) / prove(证命题成立)"},
                    "timeout_ms": {"type": "integer", "description": "求解超时毫秒, 默认 5000"},
                    "claim": {"type": "string", "description": "这条约束对应哪个命题(进证据记录)"}
                },
                "required": ["constraints"]
            }
        }
    },

    "research_lean": {
        "type": "function",
        "function": {
            "name": "research_lean",
            "description": (
                "用 Lean 4 kernel 验证形式化证明 (L0 档, 0 LLM 调用)。"
                "★ 这是**最强的一档机器判定**: kernel 验证的是证明项本身"
                "(proof term), 检查每个推理步骤的合法性, 而不是代码逻辑。"
                "sympy 验的是『我对这个式子的化简』, Lean 验的是『整个推理链合法』。"
                "\n\n★ 什么时候该用它 ★\n"
                "· 需要全域/无穷层面的结论时(有限枚举证明不了『对所有 n』)\n"
                "· 命题能写成形式化陈述时(定理、性质、算法正确性)\n"
                "· 想要比符号化简更强的保证时\n\n"
                "★ 什么时候不该用 ★\n"
                "· 数值计算 / 具体反例搜索 → 用 research_run_code 或 research_enumerate\n"
                "· 开放式问题(如哥德巴赫猜想本身)→ Lean 也证不了\n"
                "· 需要 Mathlib 的大型定理(费马小定理、sqrt2 无理)→ 当前环境 Mathlib 未就绪\n"
                "\n★ 极其重要 ★\n"
                "Lean 遇到 `sorry` 会**报 warning 但编译通过, 退出码仍是 0**。"
                "所以本工具单独检查 sorry: 返回里 sorry_used=true 表示"
                "证明**没有写完**, verdict 会降级成 inconclusive。"
                "不要看到编译成功就当它证明了。"
                "\n不依赖 Mathlib 的核心定理可以直接用, 例如:\n"
                "    theorem t (a : Nat) : a + 0 = a := Nat.add_zero a"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Lean 4 源码, 若干 theorem 声明。"
                            "不依赖 Mathlib 的核心定理可直接用(Nat.add_zero / decide 等); "
                            "需要 import Mathlib 的定理当前环境跑不了。"
                        )
                    },
                    "claim": {"type": "string", "description": "这段代码想证明的命题(进证据记录, 供报告引用)"},
                    "timeout": {"type": "integer", "description": "编译硬超时秒数, 默认 300"}
                },
                "required": ["code"]
            }
        }
    },

    "research_enumerate": {
        "type": "function",
        "function": {
            "name": "research_enumerate",
            "description": (
                "在可穷举对象上检查性质并找反例 (L0 档, 0 LLM 调用)。"
                "task='graph' 遍历非同构图(用 networkx graph atlas, n_max<=7 时是穷尽的), "
                "task='sequence' 检查素数/斐波那契等序列的前缀。"
                "★ 枚举找到反例是 refuted(真反例); **枚举找不到反例只能给 partial**, "
                "永远不能给 proved —— 有限枚举证明不了全域。"
                "返回里 atlas_exhaustive=True 表示这次是穷尽性搜索, 不是抽样。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prop": {"type": "string", "description": "要检查的性质名, 如 bipartite / triangle_free / connected / is_prime"},
                    "task": {"type": "string", "description": "graph(默认) / sequence"},
                    "n_max": {"type": "integer", "description": "图枚举的顶点数上限, 默认 7; >7 只抽样不穷尽"},
                    "kind": {"type": "string", "description": "task=sequence 时的序列名: primes / fibonacci / partition_count / squares"},
                    "n_terms": {"type": "integer", "description": "task=sequence 时的项数, 默认 20"},
                    "claim": {"type": "string", "description": "这条检查对应哪个命题(进证据记录)"}
                },
                "required": ["prop"]
            }
        }
    },

    "research_oeis": {
        "type": "function",
        "function": {
            "name": "research_oeis",
            "description": (
                "查 OEIS 并逐项对账, 返回最强匹配 (L0 档, 0 LLM 调用)。"
                "★ 搜索接口对任何序列都返一堆模糊匹配, 所以本工具**逐项比对**"
                "前 keep 项是否真的吻合, 返回精确命中的那条。"
                "适用: 用户给你一串数问'这是什么序列'。"
                "注意: OEIS 是整数序列词典, 不是文献库 —— 找论文用别的工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {"type": "string", "description": "逗号分隔的整数, 如 '1,2,4,8,16'"},
                    "max_terms": {"type": "integer", "description": "拿前几项去搜, 默认 10"},
                    "keep": {"type": "integer", "description": "返回前几条候选, 默认 5"},
                    "claim": {"type": "string", "description": "对应哪个命题(进证据记录)"}
                },
                "required": ["sequence"]
            }
        }
    },

    "research_refute": {
        "type": "function",
        "function": {
            "name": "research_refute",
            "description": (
                "依次尝试图枚举 → Z3 找反例, 找到立刻返回 (L0 档, 0 LLM 调用)。"
                "两条路都给: graphs 走图性质枚举, constraints 走 SMT。"
                "适合'我想推翻这个说法'的场景。找到反例返回 refuted 并附反例本身;"
                "两条路都没找到, 返回 partial —— **不是 proved**, 换更强的工具(Z3 goal=prove)。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "要反驳的命题"},
                    "graphs": {"type": "string", "description": "图性质名(空则跳过图枚举这一路)"},
                    "constraints": {"type": "string", "description": "SMT-LIB2 约束(空则跳过 Z3 这一路)"},
                    "n_max": {"type": "integer", "description": "图枚举上限, 默认 7"},
                    "timeout_ms": {"type": "integer", "description": "Z3 超时毫秒"}
                },
                "required": ["claim"]
            }
        }
    },

    "research_plan": {
        "type": "function",
        "function": {
            "name": "research_plan",
            "description": (
                "把研究问题拆成机器可判定的原子命题 (L1 档, 调一次 LLM)。"
                "返回 {specs: [{claim_text, scope, how, decidable}], undecidable: [...]}。"
                "★ 关键设计: 拆不出来的东西**不许硬拆**。像'这段证明是优美的'、"
                "'最小的不存在的素数是多少'这种, 应该进 undecidable 而不是编一个可判定命题。"
                "认输比硬答有价值。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "研究问题(自然语言)"},
                    "task_hash": {"type": "string", "description": "复用已有工作区; 不传按 question 自动生成"}
                },
                "required": ["question"]
            }
        }
    },

    "research_investigate": {
        "type": "function",
        "function": {
            "name": "research_investigate",
            "description": (
                "跑一次完整研究任务(流式, L2 档)。建模 → 逐条命题生成验证代码 → 沙箱执行 → "
                "失败自动带上下文重试 → 落报告。决策用规则不用 LLM, 所以可预测。"
                "★ 这是唯一会自己花不少 token 的入口, 用户问的是真问题时才用。"
                "plan_only=True 可只建模不验证。"
                "★ 已知局限: '用户问题 → 命题'这段没有验证器, 量词可能被误读"
                "(把『存在 n 使 X』读成『所有 n』)。结论出来后建议自己核一眼量词。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "研究问题"},
                    "max_rounds": {"type": "integer", "description": "最多验证几轮(每轮一条命题), 默认 3"},
                    "task_hash": {"type": "string", "description": "复用工作区; 不传按 question 生成"},
                    "plan_only": {"type": "boolean", "description": "True = 只建模不验证, 默认 False"}
                },
                "required": ["question"]
            }
        }
    },

    "research_report": {
        "type": "function",
        "function": {
            "name": "research_report",
            "description": (
                "渲染某个任务的报告 (L0 档, 0 LLM 调用)。"
                "报告会显式分出『已验证区 / 未验证区』—— 没有证据支撑的断言不许进前者。"
                "task_hash 从 research_verify / research_investigate 的返回里拿。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_hash": {"type": "string", "description": "工作区的 task_hash"},
                    "fmt": {"type": "string", "description": "md(默认) / json"},
                    "save": {"type": "boolean", "description": "True = 同时写回 workspace 的 _report.md, 默认 True"}
                },
                "required": ["task_hash"]
            }
        }
    },

    "research_audit": {
        "type": "function",
        "function": {
            "name": "research_audit",
            "description": (
                "对已完成任务跑证据链审计 (L0 档, 0 LLM 调用)。"
                "查的是: 有没有无证据的结论、证据链断没断、code 和 verdict 对不对得上。"
                "★ 0 LLM —— prompt 是软约束, 审计才是硬约束。"
                "翻车复盘 / 出报告前自查时用这个。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_hash": {"type": "string", "description": "单个任务; 留空则审计该工作区 _specs 里的全部"},
                    "task_hash_list": {"type": "string", "description": "逗号分隔的多个 task_hash(审计'结论翻车'的推荐用法)"}
                },
                "required": []
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
