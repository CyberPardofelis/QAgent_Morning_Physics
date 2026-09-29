"""
ui.fact_kb_dialog — 信息知识库 dialog(2026-09-29 加)

职责:
- 列出 facts.db 的原子事实(默认按 出处数降序 + 更新时间倒序, **分页**)
- 按 content/subject/fact_type/paper_id/来源类型 检索
- 详情面板显示: fact 本体 + **全部出处**(这正是 [fact, paper] 结构的核心价值 ——
  一个命题被几篇独立论文支持, 一眼可见)
- CRUD: 新建 / 编辑 / 删除 fact 与其出处

窗口模型(2026-09-29 用户要求):
    **不强制前置, 不阻塞其他窗口**。所以:
    - 主对话框用 show() 而非 exec(), 不 setModal
    - 新建/编辑/删除确认都用**子对话框**且默认不模态
      (Qt 的 QMessageBox 默认就是模态的, 这里的确认框用自建的非模态提示,
       或者用 question 的父窗口但不强 raise)
    - 不调 raise_() / activateWindow()

跟其他知识库的区别:
    - 论文摘要知识库: 粒度是"扫过的论文", 一篇一行
    - 论文分析知识库: 粒度是"分析过的论文", 一篇一行
    - 论文综述库: 粒度是"主题 + 章节"
    - **信息知识库: 粒度是"事实", 多篇论文可能共用一条事实** ← 独有
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from agent.tools import facts as F
from ui.styles import get_stylesheet

DEFAULT_PAGE_SIZE = 50
PAGE_SIZE_OPTIONS = [20, 50, 100, 200]

FACT_TYPE_LABELS = {
    "": "(全部类型)",
    "method": "方法",
    "result": "结果",
    "conclusion": "结论",
    "limitation": "局限",
    "hypothesis": "假设",
    "claim": "论断",
    "fact": "其他",
}

SOURCE_TYPE_LABELS = {
    "": "(全部来源)",
    "paper_abstract": "论文摘要",
    "paper_fulltext": "论文全文",
    "conversation": "对话",
    "manual": "手工添加",
    "derived": "推导",
}


class FactEditDialog(QDialog):
    """
    fact 编辑对话框(新建/编辑共用)。

    非模态: 不设 setModal, 不 exec —— 用户可以留着它切去主窗口。
    """

    def __init__(self, fact=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑事实" if fact else "新建事实")
        self.setMinimumWidth(560)
        self.setStyleSheet(get_stylesheet())
        self._fact = fact

        v = QVBoxLayout(self)
        form = QFormLayout()

        self.content_edit = QPlainTextEdit()
        self.content_edit.setPlaceholderText(
            "自足、去语境化的陈述。\n"
            "主语应该是方法/概念本身, 不要写'本文提出'/'该方法'。\n\n"
            "示例: 稳定化解析延拓(SAC)框架从多项式估计中重构密度矩阵的非多项式解析函数"
        )
        self.content_edit.setMinimumHeight(120)
        if fact:
            self.content_edit.setPlainText(fact.get("content", ""))

        self.type_combo = QComboBox()
        for k, label in FACT_TYPE_LABELS.items():
            self.type_combo.addItem(label, k)
        if fact:
            idx = self.type_combo.findData(fact.get("fact_type", ""))
            if idx >= 0:
                self.type_combo.setCurrentIndex(idx)

        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText(
            "讨论的核心对象(跨论文对齐用)。2-6 词, 用领域通用叫法。\n"
            "示例: 经典阴影 / SIC-POVM 态层析"
        )
        if fact:
            self.subject_edit.setText(fact.get("subject", ""))

        form.addRow("事实内容 *", self.content_edit)
        form.addRow("类型", self.type_combo)
        form.addRow("核心对象", self.subject_edit)
        v.addLayout(form)

        # 出处(新建时至少要有一个来源, 否则 fact 不可溯源)
        self.src_type = QComboBox()
        for k, label in SOURCE_TYPE_LABELS.items():
            if k:
                self.src_type.addItem(label, k)

        src_box = QFormLayout()
        src_box.addRow("来源类型", self.src_type)
        self.source_id_edit = QLineEdit()
        self.source_id_edit.setPlaceholderText("arXiv id / DOI / 对话 id")
        src_box.addRow("来源标识 *", self.source_id_edit)
        self.quote_edit = QPlainTextEdit()
        self.quote_edit.setPlaceholderText("原文摘录(可选, 但强烈建议填 —— 审查靠它判断是否超纲)")
        self.quote_edit.setMaximumHeight(80)
        src_box.addRow("原文摘录", self.quote_edit)
        if fact and fact.get("sources"):
            s0 = fact["sources"][0]
            i = self.src_type.findData(s0.get("source_type", ""))
            if i >= 0:
                self.src_type.setCurrentIndex(i)
            self.source_id_edit.setText(s0.get("source_id", ""))
            self.quote_edit.setPlainText(s0.get("quote", ""))
        v.addWidget(QLabel("出处(至少一个)"))
        v.addLayout(src_box)

        btns = QHBoxLayout()
        btns.addStretch()
        ok = QPushButton("保存")
        cancel = QPushButton("取消")
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

        ok.clicked.connect(self._on_save)
        cancel.clicked.connect(self.reject)

    def _on_save(self):
        content = self.content_edit.toPlainText().strip()
        if not content:
            QMessageBox.warning(self, "内容为空", "事实内容不能为空。")
            return
        src = self.source_id_edit.text().strip()
        if not src:
            QMessageBox.warning(self, "缺来源", "必须填来源标识 —— 事实必须可溯源。")
            return
        self._result = {
            "content": content,
            "fact_type": self.type_combo.currentData() or "fact",
            "subject": self.subject_edit.text().strip(),
            "source_type": self.src_type.currentData() or "manual",
            "source_id": src,
            "quote": self.quote_edit.toPlainText().strip(),
        }
        self.accept()

    def result(self):
        return getattr(self, "_result", None)


class FactKBDialog(QDialog):
    """信息知识库: 原子事实 + 多出处。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("信息知识库")
        self.resize(1200, 760)
        self.setStyleSheet(get_stylesheet())
        # 非模态: 允许在主窗口和其他窗口之间自由切换
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)

        self._page = 0
        self._page_size = DEFAULT_PAGE_SIZE
        self._all = []
        self._editing = None      # 当前正在编辑 fact 的子对话框

        v = QVBoxLayout(self)

        # ---------- 搜索栏 ----------
        top = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(
            "搜索事实内容 / 核心对象 / 来源论文 id...（回车搜索）"
        )
        self.search_edit.returnPressed.connect(self._do_search)
        top.addWidget(self.search_edit, 3)

        self.type_combo = QComboBox()
        for k, label in FACT_TYPE_LABELS.items():
            self.type_combo.addItem(label, k)
        self.type_combo.currentIndexChanged.connect(self._do_search)
        top.addWidget(self.type_combo, 1)

        self.source_combo = QComboBox()
        for k, label in SOURCE_TYPE_LABELS.items():
            self.source_combo.addItem(label, k)
        self.source_combo.currentIndexChanged.connect(self._do_search)
        top.addWidget(self.source_combo, 1)

        self.multi_only = QComboBox()
        self.multi_only.addItem("全部", 0)
        self.multi_only.addItem("仅多出处(≥2 来源)", 1)
        self.multi_only.currentIndexChanged.connect(self._do_search)
        top.addWidget(self.multi_only, 1)

        btn_new = QPushButton("新建")
        btn_new.clicked.connect(self._on_new)
        top.addWidget(btn_new)

        v.addLayout(top)

        # ---------- 统计条 ----------
        self.stats_label = QLabel()
        v.addWidget(self.stats_label)

        # ---------- 左表 / 右详情 ----------
        split = QSplitter(Qt.Horizontal)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["出处数", "类型", "核心对象", "内容", "更新时间"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.itemDoubleClicked.connect(self._on_edit)
        split.addWidget(self.table)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        rv.addWidget(self.detail, 1)

        btn_edit = QPushButton("编辑所选事实")
        btn_edit.clicked.connect(self._on_edit)
        btn_add_src = QPushButton("为所选事实添加出处")
        btn_add_src.clicked.connect(self._on_add_source)
        btn_del = QPushButton("删除所选事实")
        btn_del.clicked.connect(self._on_delete)
        rb = QHBoxLayout()
        rb.addWidget(btn_edit)
        rb.addWidget(btn_add_src)
        rb.addWidget(btn_del)
        rb.addStretch()
        rv.addLayout(rb)
        split.addWidget(right)
        split.setSizes([820, 380])
        v.addWidget(split, 1)

        # ---------- 分页 ----------
        pager = QHBoxLayout()
        self.page_label = QLabel()
        prev_btn = QPushButton("上一页")
        next_btn = QPushButton("下一页")
        prev_btn.clicked.connect(lambda: self._goto_page(self._page - 1))
        next_btn.clicked.connect(lambda: self._goto_page(self._page + 1))
        jump = QLineEdit()
        jump.setFixedWidth(60)
        jump.setPlaceholderText("页码")
        jump.returnPressed.connect(lambda: self._goto_page(int(jump.text() or 1) - 1))
        size_combo = QComboBox()
        for n in PAGE_SIZE_OPTIONS:
            size_combo.addItem(f"{n} 条/页", n)
        size_combo.setCurrentIndex(PAGE_SIZE_OPTIONS.index(DEFAULT_PAGE_SIZE))
        size_combo.currentIndexChanged.connect(self._set_page_size)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh)

        pager.addWidget(prev_btn)
        pager.addWidget(next_btn)
        pager.addWidget(jump)
        pager.addWidget(self.page_label, 1)
        pager.addWidget(size_combo)
        pager.addWidget(refresh)
        v.addLayout(pager)

        self._load()

    # ---------------- 数据 ----------------

    def _query(self) -> list[dict]:
        rows = F.list_facts(
            fact_type=self.type_combo.currentData() or "",
            source_type=self.source_combo.currentData() or "",
            min_sources=2 if self.multi_only.currentData() == 1 else 0,
            limit=5000, order_by="n_sources",
        )
        q = self.search_edit.text().strip().lower()
        if q:
            rows = [
                r for r in rows
                if q in (r.get("content") or "").lower()
                or q in (r.get("subject") or "").lower()
            ]
        return rows

    def _load(self):
        try:
            self._all = self._query()
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
            self._all = []
        self._goto_page(0)

    def refresh(self):
        self._load()

    def _do_search(self):
        self._page = 0
        self._load()

    def _set_page_size(self, idx):
        self._page_size = PAGE_SIZE_OPTIONS[idx]
        self._page = 0
        self._goto_page(0)

    def _goto_page(self, page):
        n = max(1, (len(self._all) + self._page_size - 1) // self._page_size)
        self._page = max(0, min(page, n - 1))
        start = self._page * self._page_size
        rows = self._all[start:start + self._page_size]

        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(str(row.get("n_sources", 0))))
            self.table.setItem(r, 1, QTableWidgetItem(row.get("fact_type", "")))
            self.table.setItem(r, 2, QTableWidgetItem(row.get("subject", "")))
            item = QTableWidgetItem((row.get("content") or "").replace("\n", " ")[:300])
            item.setToolTip(row.get("content", ""))
            self.table.setItem(r, 3, item)
            self.table.setItem(r, 4, QTableWidgetItem(str(row.get("created_at", ""))[:19]))
            self.table.item(r, 0).setData(Qt.UserRole, row.get("id", ""))

        total_pages = n
        self.page_label.setText(
            f"第 {self._page + 1} / {total_pages} 页    共 {len(self._all)} 条事实"
        )
        st = F.stats()
        self.stats_label.setText(
            f"事实 {st.get('n_facts', 0)} 条 · 出处 {st.get('n_sources', 0)} 个 · "
            f"多出处 {st.get('multi_source', 0)} 条 · 未过审 {st.get('ungrounded', 0)} 条"
        )

    # ---------------- 详情 ----------------

    def _current_fact_id(self) -> str | None:
        r = self.table.currentRow()
        if r < 0:
            return None
        it = self.table.item(r, 0)
        return it.data(Qt.UserRole) if it else None

    def _on_select(self):
        fid = self._current_fact_id()
        if not fid:
            return
        f = F.get_fact(fid)
        if not f:
            return
        lines = [
            f"【{f.get('fact_type', '')}】{f.get('content', '')}",
            f"核心对象: {f.get('subject', '') or '(未填)'}",
            f"fact_id : {f['id']}",
            f"更新时间: {f.get('updated_at', '')}",
            f"出处 ({len(f.get('sources', []))} 个):",
        ]
        for i, s in enumerate(f.get("sources", []), 1):
            mark = "" if s.get("grounded") else "   ⚠ 未通过落地审查"
            lines.append(
                f"  [{i}] {SOURCE_TYPE_LABELS.get(s.get('source_type'), s.get('source_type'))}"
                f" / {s.get('source_id', '')}{mark}"
            )
            if s.get("quote"):
                lines.append(f"      原文: {s['quote'][:300]}")
        if f.get("_subject_issues"):
            lines.append("")
            lines.append("⚠ 核心对象问题: " + "; ".join(f["_subject_issues"]))
        self.detail.setPlainText("\n".join(lines))

    # ---------------- CRUD ----------------

    def _on_new(self):
        # 非模态子窗口: 留着也能切去主窗口
        dlg = FactEditDialog(None, self)
        self._editing = dlg      # 保活, 避免被 GC
        dlg.finished.connect(lambda _r: self._after_edit(dlg))

    def _on_edit(self):
        fid = self._current_fact_id()
        if not fid:
            QMessageBox.information(self, "未选中", "请先在左侧选中一条事实。")
            return
        f = F.get_fact(fid)
        dlg = FactEditDialog(f, self)
        self._editing = dlg
        dlg.finished.connect(lambda _r: self._after_edit(dlg))

    def _after_edit(self, dlg):
        res = dlg.result()
        if not res:
            return
        try:
            F.add_fact(
                content=res["content"], fact_type=res["fact_type"],
                subject=res["subject"], source_type=res["source_type"],
                source_id=res["source_id"], quote=res["quote"],
            )
            QMessageBox.information(
                self, "已保存",
                "已保存。\n注意: 内容哈希决定 fact_id —— 如果改动后与已有事实"
                "内容相同, 会自动合并到那一条上并追加本次出处。",
            )
            self._load()
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
        finally:
            self._editing = None

    def _on_add_source(self):
        fid = self._current_fact_id()
        if not fid:
            QMessageBox.information(self, "未选中", "请先选中一条事实。")
            return
        stype, ok = QInputDialog.getItem(
            self, "添加出处", "来源类型:",
            [SOURCE_TYPE_LABELS[k] for k in SOURCE_TYPE_LABELS if k], 0, False,
        )
        if not ok:
            return
        key = next(k for k, v in SOURCE_TYPE_LABELS.items() if v == stype)
        sid, ok = QInputDialog.getText(self, "添加出处", "来源标识(arXiv id / DOI / 对话 id):")
        if not ok or not sid.strip():
            return
        try:
            r = F.add_fact(
                content=F.get_fact(fid)["content"],
                fact_type=F.get_fact(fid)["fact_type"],
                subject=F.get_fact(fid)["subject"],
                source_type=key, source_id=sid.strip(),
            )
            if not r["source_added"]:
                QMessageBox.information(self, "已存在", "该出处已存在, 未重复添加。")
            else:
                QMessageBox.information(self, "已添加", "出处已追加。")
            self._load()
        except Exception as e:
            QMessageBox.critical(self, "添加失败", str(e))

    def _on_delete(self):
        fid = self._current_fact_id()
        if not fid:
            QMessageBox.information(self, "未选中", "请先选中一条事实。")
            return
        f = F.get_fact(fid)
        reply = QMessageBox.question(
            self, "删除确认",
            f"确定删除这条事实及其全部出处?\n\n{f.get('content', '')[:200]}\n\n"
            f"(共 {len(f.get('sources', []))} 个出处会一并删除, 不可恢复)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        from agent.infra.db import connect
        try:
            with connect("facts") as conn:
                conn.execute("DELETE FROM fact_source WHERE fact_id = ?", (fid,))
                conn.execute("DELETE FROM fact WHERE id = ?", (fid,))
            self.detail.clear()
            self._load()
        except Exception as e:
            QMessageBox.critical(self, "删除失败", str(e))
