"""
ui.summary_kb_dialog — 论文摘要知识库 dialog(2026-09-03 翻页+搜索升级)

职责:
- 列出 summary_items.db 全部条目(默认按 summary_date 倒序,**分页**)
- 关键词/标题/作者/全文 检索(走 FTS5,**不要** Python 端 LIKE 遍历)
- 双击行 → 弹详情(只读)

数据源:
- 抓文时(arxiv/aps)同步落库 → summary_items(title/authors/abstract + LLM core_method/core_contrib/outlook/keywords)
- analyze_paper 完成后协同更新 keywords/core_method/core_contrib
- LLM 总结(generate_daily_summary_*)结束后同步补 LLM 字段(2026-09-03 新增)

跟论文分析知识库(KnowledgeBaseDialog)的区别:
- summary_kb: 粒度细(只存"扫过"的论文摘要), 来源多(arxiv_daily/aps_daily/reindex/manual)
- paper_kb: 粒度粗(精细分析过的论文), 只有 analyze_paper 走过才有

2026-09-03 升级:
- 分页:默认 page_size=50, 加上一页/下一页 + 跳转输入 + 当前页/总页显示
- 搜索:走 search_summary_items_fts(FTS5 MATCH,NOT Python LIKE 遍历)
- 详情 dialog:增加 outlook 字段

2026-09-04 升级:
- 关键词列显示修复:`_format_keywords_cell` 按逗号切,每行 1-2 个,
  不再 `[:60]` 一行截掉换行/逗号
"""
import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from agent.tools.knowledge import semantic_search_keywords
from agent.tools.summary import (
    count_summary_items,
    list_summary_items,
    search_summary_items_fts,
)
from ui.styles import get_stylesheet

# 2026-09-03:每页条数(用户可调)
DEFAULT_PAGE_SIZE = 50
PAGE_SIZE_OPTIONS = [20, 50, 100, 200]


class SummaryKBDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("论文摘要知识库")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(get_stylesheet())

        # 分页状态(2026-09-03 新)
        self._page_size = DEFAULT_PAGE_SIZE
        self._page_offset = 0
        self._total = 0
        self._current_results: list = []   # 当前页的结果缓存
        self._search_keyword: str = ""    # 当前搜索关键词(决定走 list 还是 FTS5)

        layout = QVBoxLayout(self)

        # ---- 状态条 ----
        status_layout = QHBoxLayout()
        self.status_label = QLabel("")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # ---- 搜索区域 ----
        search_layout = QHBoxLayout()
        self.keyword_edit = QLineEdit()
        self.keyword_edit.setPlaceholderText("输入关键词 / 标题 / 模糊查询...")
        self.search_combo = QComboBox()
        self.search_combo.addItems([
            "全部",
            "全文检索(FTS5)",
            "按标题",
            "按作者",
            "按关键词",
            "语义检索",
            "来源=arxiv_daily",
            "来源=aps_daily",
            "来源=reindex",
        ])
        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self.perform_search)
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.load_recent)
        self.keyword_edit.returnPressed.connect(self.perform_search)

        search_layout.addWidget(QLabel("搜索:"))
        search_layout.addWidget(self.keyword_edit)
        search_layout.addWidget(self.search_combo)
        search_layout.addWidget(search_btn)
        search_layout.addWidget(reset_btn)
        search_layout.addStretch()
        layout.addLayout(search_layout)

        # ---- 结果表格 ----
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels([
            "论文ID", "标题", "来源", "日期", "已分析", "关键词"
        ])
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.doubleClicked.connect(self.open_detail_dialog)
        layout.addWidget(self.result_table)

        # ---- 分页控制(2026-09-03 新)----
        pager_layout = QHBoxLayout()
        self.first_btn = QPushButton("⏮ 首页")
        self.first_btn.clicked.connect(self._goto_first)
        self.prev_btn = QPushButton("◀ 上一页")
        self.prev_btn.clicked.connect(self._goto_prev)
        self.next_btn = QPushButton("下一页 ▶")
        self.next_btn.clicked.connect(self._goto_next)
        self.last_btn = QPushButton("末页 ⏭")
        self.last_btn.clicked.connect(self._goto_last)

        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems([str(x) for x in PAGE_SIZE_OPTIONS])
        self.page_size_combo.setCurrentText(str(self._page_size))
        self.page_size_combo.currentTextChanged.connect(self._on_page_size_changed)

        self.jump_edit = QLineEdit()
        self.jump_edit.setPlaceholderText("页码")
        self.jump_edit.setFixedWidth(60)
        self.jump_edit.returnPressed.connect(self._on_jump)
        jump_btn = QPushButton("跳转")
        jump_btn.clicked.connect(self._on_jump)

        self.page_info_label = QLabel("第 1 / 1 页")

        pager_layout.addWidget(self.first_btn)
        pager_layout.addWidget(self.prev_btn)
        pager_layout.addWidget(self.next_btn)
        pager_layout.addWidget(self.last_btn)
        pager_layout.addStretch()
        pager_layout.addWidget(QLabel("每页:"))
        pager_layout.addWidget(self.page_size_combo)
        pager_layout.addSpacing(20)
        pager_layout.addWidget(QLabel("跳到:"))
        pager_layout.addWidget(self.jump_edit)
        pager_layout.addWidget(jump_btn)
        pager_layout.addSpacing(20)
        pager_layout.addWidget(self.page_info_label)
        layout.addLayout(pager_layout)

        # 初始:列最近(第 1 页)
        self.load_recent()

    # ============================================================
    # 翻页逻辑(2026-09-03 新)
    # ============================================================
    def _on_page_size_changed(self, txt: str):
        try:
            self._page_size = int(txt)
        except ValueError:
            return
        self._page_offset = 0
        self._refresh()

    def _goto_first(self):
        self._page_offset = 0
        self._refresh()

    def _goto_prev(self):
        self._page_offset = max(0, self._page_offset - self._page_size)
        self._refresh()

    def _goto_next(self):
        if self._page_offset + self._page_size < self._total:
            self._page_offset += self._page_size
            self._refresh()
        else:
            QMessageBox.information(self, "提示", "已经是最后一页了")

    def _goto_last(self):
        if self._total == 0:
            return
        last_offset = ((self._total - 1) // self._page_size) * self._page_size
        self._page_offset = last_offset
        self._refresh()

    def _on_jump(self):
        try:
            page_num = int(self.jump_edit.text().strip())
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效页码(整数)")
            return
        if page_num < 1:
            page_num = 1
        total_pages = max(1, (self._total + self._page_size - 1) // self._page_size)
        if page_num > total_pages:
            page_num = total_pages
        self._page_offset = (page_num - 1) * self._page_size
        self._refresh()

    def _update_pager_ui(self):
        """根据 _page_offset / _total 更新按钮可用性 + 页码显示"""
        total_pages = max(1, (self._total + self._page_size - 1) // self._page_size)
        current_page = (self._page_offset // self._page_size) + 1 if self._total > 0 else 0
        self.page_info_label.setText(f"第 {current_page} / {total_pages} 页(共 {self._total} 条)")
        # 边界处理
        self.first_btn.setEnabled(self._page_offset > 0)
        self.prev_btn.setEnabled(self._page_offset > 0)
        self.next_btn.setEnabled(self._page_offset + self._page_size < self._total)
        self.last_btn.setEnabled(self._page_offset + self._page_size < self._total)

    # ============================================================
    # 加载逻辑
    # ============================================================
    def _refresh(self):
        """根据当前状态(list vs 搜索)重新加载当前页"""
        if self._search_keyword:
            # 搜索模式:走 FTS5 + offset
            items = search_summary_items_fts(
                keyword=self._search_keyword,
                limit=self._page_size,
                offset=self._page_offset,
            )
            # FTS5 没单独 count, 用 list 全部再过滤(限搜索,limit 1000)
            # 注意: FTS5 模式下 _total 是"匹配总数"不是数据库总数
            all_matches = search_summary_items_fts(
                keyword=self._search_keyword,
                limit=10000,
                offset=0,
            )
            self._total = len(all_matches)
            self._current_results = items
            self.display_items(items, info=f"(搜索 '{self._search_keyword}' · 第 {self._page_offset // self._page_size + 1} 页)")
        else:
            # 列表模式
            items = list_summary_items(limit=self._page_size, offset=self._page_offset)
            self._total = count_summary_items()
            self._current_results = items
            self.display_items(items, info="(按日期倒序)")

        self._update_pager_ui()
        self._update_status()

    def _update_status(self):
        """更新顶部状态条"""
        try:
            n = count_summary_items()
            self.status_label.setText(f"📊 summary_items.db 共 {n} 条")
        except Exception as e:
            self.status_label.setText(f"⚠ 状态查询失败: {e}")

    def load_recent(self):
        """重置回 list 模式 + 第 1 页"""
        self._search_keyword = ""
        self.keyword_edit.clear()
        self.search_combo.setCurrentIndex(0)
        self._page_offset = 0
        self._refresh()

    def display_items(self, items, info=""):
        """把 summary_item 列表填到表格"""
        self.result_table.setRowCount(len(items))
        for row, it in enumerate(items):
            self.result_table.setItem(row, 0, QTableWidgetItem(it.get("paper_id", "")))
            self.result_table.setItem(row, 1, QTableWidgetItem((it.get("title") or "")[:80]))
            self.result_table.setItem(row, 2, QTableWidgetItem(it.get("source", "")))
            self.result_table.setItem(row, 3, QTableWidgetItem(it.get("summary_date", "")))
            in_db = "✅" if it.get("in_papers_db") else "—"
            self.result_table.setItem(row, 4, QTableWidgetItem(in_db))
            # 2026-09-04 改:关键词显示不再 [60] 一行截掉
            # 之前 `keywords[:60]` 把换行/逗号分隔的多关键词全压一行 → 用户报
            # "randomized measurement\n,\nRényi entropy\n..." 抽象排版
            # 新逻辑:按逗号切,每行 1-2 个,最多显示 8 个,允许换行
            kw = self._format_keywords_cell(it.get("keywords") or "")
            self.result_table.setItem(row, 5, QTableWidgetItem(kw))
        self.result_table.resizeColumnsToContents()

    @staticmethod
    def _format_keywords_cell(raw: str, max_items: int = 8) -> str:
        """
        格式化关键词单元格(2026-09-04):
        - 按逗号/顿号/分号/换行/制表符切,strip, 去空
        - 最多显示 max_items 个,末尾加 "+N more"
        - 每行一个(简单清晰, 不要 2-gram 那种挤版)
        - 全空返 "—"
        """
        if not raw or not raw.strip():
            return "—"
        # 兼容中英文标点:逗号(,/,), 顿号(、), 分号(;),
        # 换行(\n), 制表(\t)
        parts = re.split(r"[,\n\t;、,]+", raw)
        parts = [p.strip() for p in parts if p and p.strip()]
        if not parts:
            return "—"
        shown = parts[:max_items]
        text = "\n".join(shown)
        if len(parts) > max_items:
            text += f"\n+{len(parts) - max_items} more"
        return text

    def perform_search(self):
        """执行搜索 — 2026-09-03 重写:走 FTS5 / SQL 过滤,不再 Python 端遍历"""
        keyword = self.keyword_edit.text().strip()
        mode = self.search_combo.currentText()

        try:
            # "全部" 或没关键词 → 退到 list 模式
            if mode == "全部" or not keyword:
                self._search_keyword = ""
                self._page_offset = 0
                self._refresh()
                return

            # "全文检索(FTS5)" — 主路径,走 FTS5
            if mode == "全文检索(FTS5)":
                self._search_keyword = keyword
                self._page_offset = 0
                self._refresh()
                return

            # "按标题" / "按作者" / "按关键词" — 走 FTS5(它索引了 title/authors/keywords)
            if mode in ("按标题", "按作者", "按关键词"):
                # FTS5 已经把 title/abstract/core_method/core_contrib/keywords 都索引了
                # 区分用:先 FTS5 拿候选,再在 Python 端按 mode 二次过滤
                candidates = search_summary_items_fts(
                    keyword=keyword, limit=10000, offset=0,
                )
                if mode == "按标题":
                    filtered = [it for it in candidates
                                if keyword.lower() in (it.get("title") or "").lower()]
                elif mode == "按作者":
                    filtered = [it for it in candidates
                                if keyword.lower() in (it.get("authors") or "").lower()]
                else:  # 按关键词
                    filtered = [it for it in candidates
                                if keyword.lower() in (it.get("keywords") or "").lower()]
                self._search_keyword = keyword
                self._total = len(filtered)
                self._page_offset = 0
                page_items = filtered[self._page_offset:self._page_offset + self._page_size]
                self._current_results = page_items
                self.display_items(page_items, info=f"({mode} '{keyword}' · {self._total} 条)")
                self._update_pager_ui()
                self._update_status()
                return

            if mode == "语义检索":
                # 走 paper_keywords collection 语义检索
                md_result = semantic_search_keywords(query=keyword, top_k=30)
                pids = []
                for line in md_result.split("\n"):
                    if "- ID: `" in line:
                        pid = line.split("- ID: `")[1].split("`")[0]
                        if pid not in pids:
                            pids.append(pid)
                if not pids:
                    QMessageBox.information(self, "提示", f"未找到匹配 '{keyword}' 的论文")
                    return
                # 用 paper_id 列表从 summary_items 拿详情
                # 走 SQL IN 比 list 500 条过滤快
                from agent.infra.db import connect
                placeholders = ",".join("?" * len(pids))
                with connect("summary_items") as conn:
                    rows = conn.execute(
                        f"SELECT * FROM summary_item WHERE paper_id IN ({placeholders})",
                        pids,
                    ).fetchall()
                items = [dict(r) for r in rows]
                # 按 pids 顺序排序
                items.sort(key=lambda it: pids.index(it.get("paper_id", "")) if it.get("paper_id") in pids else 999)
                self._search_keyword = keyword
                self._total = len(items)
                self._page_offset = 0
                page_items = items[self._page_offset:self._page_offset + self._page_size]
                self._current_results = page_items
                self.display_items(page_items, info=f"(语义检索 '{keyword}' · {self._total} 条)")
                self._update_pager_ui()
                self._update_status()
                return

            if mode.startswith("来源="):
                src = mode.split("=")[1]
                self._search_keyword = ""
                self._page_offset = 0
                items = list_summary_items(source=src, limit=self._page_size, offset=0)
                # 全部数(用于分页)
                from agent.infra.db import connect
                with connect("summary_items") as conn:
                    self._total = conn.execute(
                        "SELECT COUNT(*) FROM summary_item WHERE source = ?", (src,)
                    ).fetchone()[0]
                self._current_results = items
                self.display_items(items, info=f"(来源={src})")
                self._update_pager_ui()
                self._update_status()
                return

        except Exception as e:
            QMessageBox.critical(self, "搜索错误", f"搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return

    def open_detail_dialog(self, index):
        """双击行:弹详情 dialog(只读)"""
        row = index.row()
        pid_item = self.result_table.item(row, 0)
        if not pid_item:
            return
        paper_id = pid_item.text()
        # 优先从当前页缓存里找
        target = next((it for it in self._current_results if it.get("paper_id") == paper_id), None)
        if not target:
            # 当前页没,直接按 paper_id + source 查最新一条
            # (翻页后双击不在当前页的,需要补查)
            try:
                # 简单点:用 list_summary_items 不带 source,拿最近一条
                items = list_summary_items(limit=10000, offset=0)
                target = next((it for it in items if it.get("paper_id") == paper_id), None)
            except Exception:
                pass
        if not target:
            QMessageBox.warning(self, "错误", f"找不到 {paper_id} 的详情")
            return
        dlg = SummaryItemDetailDialog(target, self)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.show()


class SummaryItemDetailDialog(QDialog):
    """summary_item 详情 dialog(只读) — 2026-09-03 加 outlook"""
    def __init__(self, item: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"摘要详情: {item.get('paper_id', '?')}")
        self.setMinimumSize(700, 600)
        self.setStyleSheet(get_stylesheet())
        layout = QVBoxLayout(self)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setMarkdown(self._render(item))
        layout.addWidget(text)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    @staticmethod
    def _render(it: dict) -> str:
        parts = [
            f"# {it.get('title', '(无标题)')}",
            f"**论文 ID**: `{it.get('paper_id', '?')}`",
            f"**来源**: {it.get('source', '?')}",
            f"**日期**: {it.get('summary_date', '?')}",
        ]
        if it.get("authors"):
            parts.append(f"**作者**: {it['authors'][:200]}")
        if it.get("keywords"):
            parts.append(f"**关键词**: {it['keywords']}")
        if it.get("in_papers_db"):
            parts.append("**状态**: ✅ 已精细分析")
        if it.get("analysis_id"):
            parts.append(f"**关联 papers.id**: {it['analysis_id']}")
        if it.get("abstract"):
            parts.append("\n## 摘要")
            parts.append(it["abstract"][:1500])
        if it.get("core_contrib"):
            parts.append("\n## 核心贡献")
            parts.append(it["core_contrib"][:1500])
        if it.get("core_method"):
            parts.append("\n## 方法")
            parts.append(it["core_method"][:1500])
        # 2026-09-03 加:outlook 字段
        if it.get("outlook"):
            parts.append("\n## 意义与展望")
            parts.append(it["outlook"][:1500])
        return "\n\n".join(parts)
