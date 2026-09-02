"""
ui.settings_dialog — 设置对话框

8 个标签页（按 WAgent 风格组织 + 保留 QAgent 专属）：
- 首选项        字体大小
- LLM API       DeepSeek + OpenAI（API key 走 agent.infra.api_keys，model 走 QSettings）
- 搜索服务      博查 (Bocha) + Tavily（API key 走 agent.infra.api_keys）
- 系统提示词    默认 prompt
- 数据状态      走 health_check
- 期刊设置      QAgent 专属：journals.yaml section 字段
- Embedding 设置 QAgent 专属：embedding 配置
- 每日总结范围  QAgent 专属：daily_summary 勾选

API key 存到 data/.api_keys.json（agent.infra.api_keys 统一管理，含 QSettings 兜底），
避免明文在 QSettings/注册表里。
"""
import os
import json
import yaml
from pathlib import Path
from PySide6.QtCore import Qt, QSettings, Signal
from config.settings import get_settings
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QCheckBox,
    QComboBox, QDialogButtonBox, QLabel, QTabWidget, QWidget,
    QScrollArea, QTextEdit, QPushButton, QMessageBox, QGroupBox,
    QHBoxLayout, QSpinBox,
)
from agent.infra.api_keys import (
    get_all_api_keys, set_api_key, has_api_key,
)
from agent.infra.db import health_check
from agent.llm_clients import reset_client_cache
from ui.styles import (
    get_stylesheet, list_themes, set_theme,
    get_current_theme_name, load_user_themes, get_theme,
)

# 数据根目录（与 agent.infra.db._data_root() 一致）
_DATA_ROOT = Path("data").resolve()


class SettingsDialog(QDialog):
    default_prompt_changed = Signal(str)

    def __init__(self, parent=None, tab_filter=None):
        """
        2026-09-02 加 tab_filter 参数:让同一个 dialog 类支持 2 个子 dialog
        (主菜单"设置"下拉里的"首选项" / "期刊与论文")。

        用法:
            SettingsDialog()                              # 显示全部 8 个 tab(旧行为,保留兼容)
            SettingsDialog(tab_filter=["首选项", "LLM API", "搜索服务", "系统提示词", "数据状态"])
            SettingsDialog(tab_filter=["期刊设置", "Embedding 设置", "每日总结范围"])

        tab_filter 里 tab 名跟 self._TAB_ORDER 里的 key 一一对应。
        """
        super().__init__(parent)
        self._tab_filter = tab_filter  # None = 显示全部
        # 2026-09-02 加:tab 名 → (widget 构造 callable) 映射,顺序就是显示顺序
        # 集中管理,加新 tab 只改这一处 + 写 builder
        self._TAB_BUILDERS: list = []  # 后面组装
        self.setWindowTitle("设置")
        self.setMinimumSize(720, 620)
        # 2026-09-02 改:用 get_stylesheet() 跟主题
        self.setStyleSheet(get_stylesheet())
        # 2026-09-02 改(抄 WAgent):解除与主窗口的父子关系 → Z 序独立,避免父对象 close 时自动 delete
        self.setParent(None)

        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        # 2026-09-02 修:用户报"首选项左上角横滚条还在"
        # 三重处理:禁用 scroll buttons + 禁 drawBase + 设固定高度
        tab_bar = self.tab_widget.tabBar()
        tab_bar.setUsesScrollButtons(False)   # 禁撕角(QToolButton)
        tab_bar.setDrawBase(False)             # 禁 tab bar 底部基线
        tab_bar.setFixedHeight(34)             # 固定高度(防 1-2px 横条 bug)
        tab_bar.setExpanding(False)            # tabs 不强制填满宽度(放不下时不要滚动)
        main_layout.addWidget(self.tab_widget)

        # 2026-09-02 v5 加:保存所有"可能被隐藏"的 tab widget 引用,防 GC 释放 C++
        # 之前用 QWidget(self) 让这些 tab 变成 dialog 直接子 widget (浮在 (0,0)),
        # addTab 不 re-parent 那些被 filter 排除的 tab → 它们带着内嵌的 QScrollArea 浮在
        # 标题栏下方 → 用户报"首选项左上角横滚条"反复 5 轮的真正根因
        # 现在改 QWidget() (无 parent) + 这个 dict 持有 Python 引用 → addTab 被调时
        # Qt 自动 re-parent 进 QStackedWidget,不被调时也不会浮在外面
        self._hidden_tabs = {}

        # ----- 1. 首选项（仅字体） -----
        pref_tab = QWidget()
        pref_layout = QVBoxLayout(pref_tab)
        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        # tab_filter 为 None 时显示全部,否则只显示 filter 里的 tab
        if tab_filter is not None:
            # 调整窗口标题 + 尺寸提示(子 dialog 比全集小)
            # 2026-09-02 v2 改:标题用菜单项同名("首选项" / "期刊与论文"),比"设置 - 首选项"直观
            # v3 修:用 == 严格匹配(不能用 issubset,否则部分 filter 会被错判为全集子集)
            # v4 加:"期刊与论文" 支持 4 tab(含 2026-09-02 加的"综述总结")
            if tab_filter == ["首选项", "LLM API", "搜索服务", "系统提示词", "数据状态"]:
                self.setWindowTitle("首选项")
            elif tab_filter == ["期刊设置", "Embedding 设置", "每日总结范围", "综述总结"]:
                self.setWindowTitle("期刊与论文")
            elif tab_filter == ["期刊设置", "Embedding 设置", "每日总结范围"]:
                self.setWindowTitle("期刊与论文")  # 兼容 3 tab 旧 filter
            else:
                self.setWindowTitle(f"设置({','.join(tab_filter)})")
            self.setMinimumSize(640, 520)

        # 主题下拉(实时切换) — 2026-09-02 加(抄 WAgent 主题系统)
        theme_row = QHBoxLayout()
        self.theme_combo = QComboBox()
        self._populate_theme_combo()
        self.theme_combo.setCurrentIndex(
            max(0, self.theme_combo.findData(get_current_theme_name()))
        )
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        theme_row.addWidget(self.theme_combo, stretch=1)

        # ➕ 自定义主题按钮
        self.new_theme_btn = QPushButton("➕ 自定义…")
        self.new_theme_btn.setToolTip("新建/管理自定义主题 (从 8 个核心色派生整套配色)")
        self.new_theme_btn.clicked.connect(self._on_new_theme)
        theme_row.addWidget(self.new_theme_btn)

        # ⚙ 编辑当前主题 (只对 user: 主题显示)
        if get_current_theme_name().startswith("user:"):
            edit_btn = QPushButton("⚙ 编辑")
            edit_btn.clicked.connect(self._on_edit_current)
            theme_row.addWidget(edit_btn)

        form_layout.addRow("主题:", theme_row)

        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems(["10px", "11px", "12px", "13px", "14px", "15px", "16px"])
        self.font_size_combo.setCurrentText("14px")
        form_layout.addRow("全局字体大小:", self.font_size_combo)
        pref_layout.addLayout(form_layout)
        pref_layout.addStretch()
        # 2026-09-02 改:widget 始终构造(因 _load_config 等会读 self.xxx 属性),
        # 只在 addTab 时按 tab_filter 过滤
        if self._should_show("首选项"):
            self.tab_widget.addTab(pref_tab, "首选项")
        else:
            self._hidden_tabs["首选项"] = pref_tab  # 2026-09-02 v5:防 GC 释放

        # ----- 2. LLM API -----
        llm_tab = self._build_llm_tab()  # 始终构造,因 _load_config 读 self.deepseek_api_key 等
        if self._should_show("LLM API"):
            self.tab_widget.addTab(llm_tab, "LLM API")
        else:
            self._hidden_tabs["LLM API"] = llm_tab

        # ----- 3. 搜索服务 -----
        search_tab = self._build_search_tab()  # 始终构造,_load_config 读 self.bocha_api_key 等
        if self._should_show("搜索服务"):
            self.tab_widget.addTab(search_tab, "搜索服务")
        else:
            self._hidden_tabs["搜索服务"] = search_tab

        # ----- 4. 系统提示词 -----
        # 2026-09-02 改:给所有可能被隐藏的 tab widget 加 self 作 parent,
        # 2026-09-02 v5 改:不再用 QWidget(self) — 那会变成 dialog 直接子 widget
        #   (addTab 不会 re-parent 那些被 filter 排除的 tab),浮在 (0,0) 位置显示
        #   内含的 QScrollArea → 标题栏下方出现 1-2px 横滚条(用户报 N 次的 bug)
        # 改用: QWidget() 创建 + self._hidden_tabs[tab_name] = tab 保存引用防 GC
        # addTab() 在被 filter 包含时会 re-parent 进去 QStackedWidget,不会被浮在外面
        prompt_tab = QWidget()
        prompt_layout = QVBoxLayout(prompt_tab)
        info_label = QLabel("设置默认系统提示词（将作为所有新对话的初始系统提示，分组可单独覆盖）")
        info_label.setWordWrap(True)
        prompt_layout.addWidget(info_label)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("输入默认系统提示词...")
        prompt_layout.addWidget(self.prompt_edit)
        reset_btn = QPushButton("恢复默认提示词")
        reset_btn.clicked.connect(self.reset_default_prompt)
        prompt_layout.addWidget(reset_btn)
        prompt_layout.addStretch()
        if self._should_show("系统提示词"):
            self.tab_widget.addTab(prompt_tab, "系统提示词")
        else:
            self._hidden_tabs["系统提示词"] = prompt_tab

        # ----- 5. 数据状态 -----
        db_tab = self._build_db_tab()  # 始终构造(用 self.db_status_text)
        if self._should_show("数据状态"):
            self.tab_widget.addTab(db_tab, "数据状态")
        else:
            self._hidden_tabs["数据状态"] = db_tab

        # ----- 6. 期刊设置（QAgent 专属） -----
        journal_tab = QWidget()
        journal_layout = QVBoxLayout(journal_tab)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        journal_layout.addWidget(scroll_area)
        journal_content = QWidget()
        scroll_area.setWidget(journal_content)
        self.journal_form_layout = QFormLayout(journal_content)
        self.journal_form_layout.setSpacing(10)
        self.journal_section_inputs = {}
        self._create_journal_inputs()
        journal_layout.addStretch()
        if self._should_show("期刊设置"):
            self.tab_widget.addTab(journal_tab, "期刊设置")
        else:
            self._hidden_tabs["期刊设置"] = journal_tab  # 2026-09-02 v5:防 GC 释放

        # ----- 7. Embedding 设置（QAgent 专属） -----
        embed_tab = QWidget()
        embed_layout = QVBoxLayout(embed_tab)
        embed_form = QFormLayout()
        embed_form.setSpacing(10)
        self.embedding_enabled_embed = QCheckBox()
        embed_form.addRow("启用 Embedding:", self.embedding_enabled_embed)
        self.embedder_type = QComboBox()
        self.embedder_type.addItems(["local", "openai"])
        embed_form.addRow("Embedder 类型:", self.embedder_type)
        self.openai_embedding_model = QLineEdit()
        embed_form.addRow("OpenAI Embedding 模型:", self.openai_embedding_model)
        self.local_embedder_name = QLineEdit()
        embed_form.addRow("本地 Embedder 名称:", self.local_embedder_name)
        self.model_path = QLineEdit()
        embed_form.addRow("模型路径:", self.model_path)
        self.data_dir = QLineEdit()
        embed_form.addRow("数据目录:", self.data_dir)
        self.vector_dir = QLineEdit()
        embed_form.addRow("向量存储目录:", self.vector_dir)
        embed_layout.addLayout(embed_form)
        embed_layout.addStretch()
        if self._should_show("Embedding 设置"):
            self.tab_widget.addTab(embed_tab, "Embedding 设置")
        else:
            self._hidden_tabs["Embedding 设置"] = embed_tab  # 2026-09-02 v5:防 GC 释放

        # ----- 8. 每日总结范围（QAgent 专属） -----
        daily_scope_tab = QWidget()
        daily_scope_layout = QVBoxLayout(daily_scope_tab)
        daily_scroll = QScrollArea()
        daily_scroll.setWidgetResizable(True)
        daily_scope_layout.addWidget(daily_scroll)
        daily_content = QWidget()
        daily_content_layout = QVBoxLayout(daily_content)
        daily_scroll.setWidget(daily_content)
        self.daily_checkboxes = {}
        self._populate_daily_checkboxes(daily_content_layout)
        daily_scope_layout.addStretch()
        if self._should_show("每日总结范围"):
            self.tab_widget.addTab(daily_scope_tab, "每日总结范围")
        else:
            self._hidden_tabs["每日总结范围"] = daily_scope_tab  # 2026-09-02 v5:防 GC 释放

        # ----- 9. 综述总结（2026-09-02 P0.7 加:成本控制 3 档 + ceiling）
        # 综述生成功能(P6)待 ReAct 内核(P4)完成后再实现。
        # 现在这里只放成本控制 3 档 UI:防止给课题组用时 token 爆钱。
        survey_tab = QWidget()
        survey_layout = QVBoxLayout(survey_tab)
        survey_layout.setSpacing(12)

        # ---- 顶部说明 ----
        info = QLabel(
            "<b>综述总结</b><br>"
            "本 tab 当前承载 <b>成本控制 3 档</b>(P0.7, 2026-09-02):<br>"
            "• <b>L0 静默</b>: 只本地检索/SQL/下载, 0 次 LLM 调用 — 适合查本地知识库<br>"
            "• <b>L1 轻量</b>: + 单次 LLM 工具 (analyze_paper / save_memory 等) — 默认档<br>"
            "• <b>L2 重型</b>: + ReAct 多轮 + 综述生成 (P4 / P6 阶段实现后才有) — 给个人用时<br><br>"
            "切档位立即生效, 不需重启。下次 LLM 调用时按档位过滤可用工具。"
        )
        info.setWordWrap(True)
        info.setProperty("role", "muted")
        survey_layout.addWidget(info)

        # ---- 成本档位 segmented ----
        tier_group = QGroupBox("当前成本档位")
        tier_layout = QVBoxLayout(tier_group)
        self.cost_tier_combo = QComboBox()
        self.cost_tier_combo.addItems([
            "L0 (纯本地, 0 次 LLM)",
            "L1 (单次 LLM, 默认)",
            "L2 (重型多轮, 给个人用)",
        ])
        # QComboBox 数据用 "L0" / "L1" / "L2",UI 显示长描述
        self.cost_tier_combo.setItemData(0, "L0")
        self.cost_tier_combo.setItemData(1, "L1")
        self.cost_tier_combo.setItemData(2, "L2")
        tier_layout.addWidget(self.cost_tier_combo)
        tier_hint = QLabel(
            "注意: L2 默认开是给个人用户测方便。给课题组用时建议调到 L1。"
        )
        tier_hint.setWordWrap(True)
        tier_hint.setProperty("role", "muted")
        tier_layout.addWidget(tier_hint)
        survey_layout.addWidget(tier_group)

        # ---- 每日 ceiling ----
        ceiling_group = QGroupBox("每日 LLM 调用上限")
        ceiling_layout = QFormLayout(ceiling_group)
        self.cost_ceiling_spin = QSpinBox()
        self.cost_ceiling_spin.setRange(1, 9999)
        self.cost_ceiling_spin.setSuffix(" 次/天")
        ceiling_layout.addRow("上限:", self.cost_ceiling_spin)
        ceiling_hint = QLabel(
            "达到上限的 80% 时弹窗提醒, 100% 时硬阻断。明天 0 点自动重置。"
        )
        ceiling_hint.setWordWrap(True)
        ceiling_hint.setProperty("role", "muted")
        ceiling_layout.addRow("", ceiling_hint)
        survey_layout.addWidget(ceiling_group)

        # ---- 今日用量 + 重置 ----
        usage_group = QGroupBox("今日用量")
        usage_layout = QHBoxLayout(usage_group)
        self.cost_usage_label = QLabel("今日已用: 0 / 0 次")
        self.cost_usage_label.setProperty("role", "title")
        usage_layout.addWidget(self.cost_usage_label)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._refresh_cost_usage)
        usage_layout.addWidget(refresh_btn)
        reset_btn = QPushButton("重置今日计数")
        reset_btn.clicked.connect(self._reset_cost_usage)
        usage_layout.addWidget(reset_btn)
        usage_layout.addStretch()
        survey_layout.addWidget(usage_group)

        # ---- 综述生成 placeholder ----
        # 等 P6 阶段填候选池 / claim-source pairing / surveys.db 等
        # 现在只放一行说明,告诉用户这个 tab 的另一半用途
        placeholder_group = QGroupBox("综述生成 (P6, 待实现)")
        placeholder_layout = QVBoxLayout(placeholder_group)
        placeholder_label = QLabel(
            "综述生成功能等 P4 (ReAct 调研内核) 完成后, 2026-09-02 之后排期。<br>"
            "届时会新增: 候选池构建 / claim-source pairing / surveys.db 章节落库 / 行内 markdown 链接等。<br>"
            "ROADMAP_NEXT §7 综述生成 (L2, 3-4 天) 已规划。"
        )
        placeholder_label.setWordWrap(True)
        placeholder_label.setProperty("role", "muted")
        placeholder_layout.addWidget(placeholder_label)
        survey_layout.addWidget(placeholder_group)

        survey_layout.addStretch()
        if self._should_show("综述总结"):
            self.tab_widget.addTab(survey_tab, "综述总结")
        else:
            self._hidden_tabs["综述总结"] = survey_tab  # 防 GC 释放

        # 底部按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        # 加载所有配置
        # 2026-09-02 加:WAgent 主题系统 — 启动时从 data/themes.json 加载用户自定义主题
        load_user_themes()
        self._load_config()
        self._load_journals_config()
        self._load_font_size()
        self._refresh_db_status()

    # ============================================================
    # 2026-09-02 加:tab_filter helper
    # ============================================================
    def _should_show(self, tab_name: str) -> bool:
        """tab_filter 为 None 时全部显示;否则只在 list 里才显示。"""
        if self._tab_filter is None:
            return True
        return tab_name in self._tab_filter

    # ============================================================
    # LLM API Tab
    # ============================================================
    def _build_llm_tab(self) -> QWidget:
        # MiniMax 组的输入框被显式 setMinimumHeight(48) 撑高后,tab 整体高度
        # 可能超过 dialog 的 620px 最小高度 → 用 QScrollArea 兜底,内容超长可滚。
        tab = QWidget()
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer_layout.addWidget(scroll)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setSpacing(12)

        info = QLabel(
            "API Key 存储在 <code>data/.api_keys.json</code>（含 QSettings/.env 兜底）。\n"
            "Base URL 不填则使用 provider 默认值。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # ---------- DeepSeek ----------
        ds_group = QGroupBox("DeepSeek")
        ds_layout = QFormLayout(ds_group)
        self.deepseek_api_key = QLineEdit()
        self.deepseek_api_key.setEchoMode(QLineEdit.Password)
        self.deepseek_api_key.setPlaceholderText("sk-...")
        ds_layout.addRow("API Key:", self.deepseek_api_key)
        self.deepseek_model = QLineEdit()
        self.deepseek_model.setPlaceholderText("deepseek-v4-flash")
        ds_layout.addRow("默认 Model:", self.deepseek_model)
        layout.addWidget(ds_group)

        # ---------- OpenAI ----------
        oa_group = QGroupBox("OpenAI")
        oa_layout = QFormLayout(oa_group)
        self.openai_api_key = QLineEdit()
        self.openai_api_key.setEchoMode(QLineEdit.Password)
        self.openai_api_key.setPlaceholderText("sk-proj-...")
        oa_layout.addRow("API Key:", self.openai_api_key)
        self.openai_model = QLineEdit()
        self.openai_model.setPlaceholderText("gpt-4o-mini")
        oa_layout.addRow("默认 Model:", self.openai_model)
        layout.addWidget(oa_group)

        # ---------- MiniMax（MiniMax） ----------
        mm_group = QGroupBox("MiniMax（国内）")
        mm_layout = QFormLayout(mm_group)
        self.minimax_api_key = QLineEdit()
        self.minimax_api_key.setEchoMode(QLineEdit.Password)
        self.minimax_api_key.setPlaceholderText("eyJhbGciOi...")
        mm_layout.addRow("API Key:", self.minimax_api_key)
        self.minimax_model = QLineEdit()
        self.minimax_model.setPlaceholderText("MiniMax-M3")
        mm_layout.addRow("默认 Model:", self.minimax_model)
        self.minimax_base_url = QLineEdit()
        self.minimax_base_url.setPlaceholderText("https://api.MiniMax.com/v1  (留空用默认)")
        mm_layout.addRow("Base URL:", self.minimax_base_url)
        mm_hint = QLabel(
            "国内 MiniMax 官方 API，OpenAI 兼容接口，支持工具调用 + 思考模式。"
            "可换模型：MiniMax-M3、MiniMax-VL-01 等。"
        )
        mm_hint.setWordWrap(True)
        # 2026-09-02 改:用 role="muted" 走全局 QSS 里的 text_muted token,跟主题走
        mm_hint.setProperty("role", "muted")
        mm_hint.setMinimumHeight(43)  # 跟 QLineEdit 等高,避免 hint 行被压扁
        mm_layout.addRow("", mm_hint)
        # 给 MiniMax 三个输入框强制更高的 min-height + 缩小圆角,
        # 避免 global QSS 的 border-radius:20px + padding:12px 让 widget 看着"圆胖矮",
        # 在某些 DPI/字号下视觉上比 DeepSeek/OpenAI 明显矮一截
        for le in (self.minimax_api_key, self.minimax_model, self.minimax_base_url):
            le.setMinimumHeight(48)
            le.setStyleSheet(
                "QLineEdit { padding: 10px 16px; border-radius: 10px; }"
            )
        layout.addWidget(mm_group)

        # ---------- 兼容旧字段 ----------
        # 旧 controller1.py 还在读 DEEPSEEK_MODEL_CHAT，保留设置项但加说明
        ds_chat_group = QGroupBox("兼容设置（老 controller 用的字段，新代码忽略）")
        ds_chat_layout = QFormLayout(ds_chat_group)
        self.deepseek_chat_model = QLineEdit()
        self.deepseek_chat_model.setPlaceholderText("deepseek-v4-flash")
        ds_chat_layout.addRow("DeepSeek Chat Model:", self.deepseek_chat_model)
        layout.addWidget(ds_chat_group)

        layout.addStretch()
        return tab

    # ============================================================
    # 搜索服务 Tab
    # ============================================================
    def _build_search_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        info = QLabel(
            "联网搜索服务用于 web_search 工具的 RAG 检索。<br>"
            "• <b>博查 (Bocha)</b>：中文源强，国内稳定，需付费（按调用量）<br>"
            "• <b>Tavily</b>：海外源强，免费档每月 1000 次<br>"
            "• <b>DuckDuckGo</b>：保底免 key，但国内访问常超时<br><br>"
            "填了 key 后 web_search 会按 query 中英文自动选 provider。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # ---------- Bocha ----------
        bocha_group = QGroupBox("博查 (Bocha) — 推荐中文场景")
        bocha_layout = QFormLayout(bocha_group)
        self.bocha_api_key = QLineEdit()
        self.bocha_api_key.setEchoMode(QLineEdit.Password)
        self.bocha_api_key.setPlaceholderText("sk-...  （在 https://open.bochaai.com 申请）")
        bocha_layout.addRow("API Key:", self.bocha_api_key)
        bocha_test_btn = QPushButton("测试连接")
        bocha_test_btn.clicked.connect(self._test_bocha)
        bocha_layout.addRow("", bocha_test_btn)
        layout.addWidget(bocha_group)

        # ---------- Tavily ----------
        tavily_group = QGroupBox("Tavily — 推荐海外/英文场景")
        tavily_layout = QFormLayout(tavily_group)
        self.tavily_api_key = QLineEdit()
        self.tavily_api_key.setEchoMode(QLineEdit.Password)
        self.tavily_api_key.setPlaceholderText("tvly-...  （在 https://tavily.com 申请）")
        tavily_layout.addRow("API Key:", self.tavily_api_key)
        tavily_test_btn = QPushButton("测试连接")
        tavily_test_btn.clicked.connect(self._test_tavily)
        tavily_layout.addRow("", tavily_test_btn)
        layout.addWidget(tavily_group)

        ddg_note = QLabel(
            "<b>DuckDuckGo</b>：无需配置 key，国内访问可能超时，仅作保底。"
        )
        # 2026-09-02 改:走主题 token
        ddg_note.setProperty("role", "muted")
        ddg_note.setWordWrap(True)
        layout.addWidget(ddg_note)

        layout.addStretch()
        return tab

    def _test_bocha(self):
        from agent.infra.api_keys import get_api_key
        from agent.tools.search import search_bocha
        key = self.bocha_api_key.text().strip()
        if not key:
            QMessageBox.warning(self, "提示", "请先填写博查 API Key")
            return
        old = get_api_key("bocha")
        try:
            set_api_key("bocha", key)
            results = search_bocha("test", 1, 100)
            if results:
                QMessageBox.information(self, "成功",
                    f"博查连接成功！\n\n示例：\n• 标题: {results[0]['title'][:60]}")
            else:
                QMessageBox.warning(self, "失败", "博查返回 0 条结果，key 可能无效")
        except Exception as e:
            QMessageBox.critical(self, "失败", f"博查连接失败：\n{e}")
        finally:
            set_api_key("bocha", old)

    def _test_tavily(self):
        from agent.infra.api_keys import get_api_key
        from agent.tools.search import search_tavily
        key = self.tavily_api_key.text().strip()
        if not key:
            QMessageBox.warning(self, "提示", "请先填写 Tavily API Key")
            return
        old = get_api_key("tavily")
        try:
            set_api_key("tavily", key)
            results = search_tavily("test", 1, False, 100)
            if results:
                QMessageBox.information(self, "成功",
                    f"Tavily 连接成功！\n\n示例：\n• 标题: {results[0]['title'][:60]}")
            else:
                QMessageBox.warning(self, "失败", "Tavily 返回 0 条结果，key 可能无效")
        except Exception as e:
            QMessageBox.critical(self, "失败", f"Tavily 连接失败：\n{e}")
        finally:
            set_api_key("tavily", old)

    # ============================================================
    # 数据状态 Tab
    # ============================================================
    def _build_db_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        title = QLabel("数据库状态")
        # 2026-09-02 改:用 role="title" 走主题 — font-weight 走 QSS,font-size 跟随全局
        title.setProperty("role", "title")
        layout.addWidget(title)

        path_label = QLabel(f"数据目录: {_DATA_ROOT}")
        # 2026-09-02 改:走主题 muted token
        path_label.setProperty("role", "muted")
        layout.addWidget(path_label)

        check_btn = QPushButton("立即检查")
        check_btn.clicked.connect(self._refresh_db_status)
        layout.addWidget(check_btn)

        self.db_status_text = QTextEdit()
        self.db_status_text.setReadOnly(True)
        layout.addWidget(self.db_status_text)

        return tab

    def _refresh_db_status(self):
        try:
            status = health_check()
            lines = []
            for name, info in status.items():
                if "error" in info:
                    lines.append(f"❌ {name}: {info['error']}")
                elif not info["exists"]:
                    lines.append(f"⚪ {name}: 未创建（首次启动时自动建表）")
                else:
                    lines.append(
                        f"✅ {name}: {info['size_bytes'] / 1024:.1f} KB, "
                        f"{info['table_count']} 张表"
                    )
            # chroma 向量库状态（2026-09-02 P0 新增）
            try:
                from agent.vector_db import health_check as chroma_health_check
                cinfo = chroma_health_check()
                lines.append("")
                lines.append("—— 向量库（chroma，P0）——")
                if not cinfo.get("exists"):
                    lines.append("⚪ chroma: 目录未创建（首次检索时自动建）")
                else:
                    lines.append(f"✅ chroma path: {cinfo['path']}")
                    lines.append(f"   collection: {cinfo['collection']}")
                    lines.append(f"   sqlite: {cinfo.get('sqlite_size_kb', 0):.1f} KB")
                    lines.append(f"   embedding: {Path(cinfo['embedding_model']).name}")
            except Exception as ce:
                lines.append(f"❌ chroma 检查失败: {ce}")
            self.db_status_text.setPlainText("\n".join(lines))
        except Exception as e:
            self.db_status_text.setPlainText(f"检查失败: {e}")

    # ============================================================
    # 期刊设置（QAgent 专属）
    # ============================================================
    def _create_journal_inputs(self):
        yaml_path = self._get_yaml_path()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                journals_data = yaml.safe_load(f)
            journals = journals_data.get('journals', {})
        except Exception as e:
            print(f"读取期刊配置失败: {e}")
            journals = {}
        while self.journal_form_layout.rowCount() > 0:
            self.journal_form_layout.removeRow(0)
        self.journal_section_inputs.clear()
        for journal_key, journal_info in journals.items():
            section_input = QLineEdit(journal_info.get('section', ''))
            self.journal_section_inputs[journal_key] = section_input
            self.journal_form_layout.addRow(f"{journal_key} section:", section_input)

    def _get_yaml_path(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, 'agent', 'journals.yaml')

    def _load_journals_config(self):
        yaml_path = self._get_yaml_path()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                journals_data = yaml.safe_load(f)
            journals = journals_data.get('journals', {})
            for journal_key, section_input in self.journal_section_inputs.items():
                if journal_key in journals:
                    section_input.setText(journals[journal_key].get('section', ''))
        except Exception as e:
            print(f"加载期刊配置失败: {e}")

    def _save_journals_config(self):
        yaml_path = self._get_yaml_path()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                journals_data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            journals_data = {'journals': {}}
        if 'journals' not in journals_data:
            journals_data['journals'] = {}
        for journal_key, section_input in self.journal_section_inputs.items():
            if journal_key not in journals_data['journals']:
                journals_data['journals'][journal_key] = {}
            journals_data['journals'][journal_key]['section'] = section_input.text()
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(journals_data, f, allow_unicode=True, default_flow_style=False)

    # ============================================================
    # 每日总结范围（QAgent 专属）
    # ============================================================
    def _get_journal_keys_from_yaml(self):
        yaml_path = self._get_yaml_path()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return list(data.get('journals', {}).keys())
        except Exception:
            return []

    def _populate_daily_checkboxes(self, layout):
        for cb in self.daily_checkboxes.values():
            cb.deleteLater()
        self.daily_checkboxes.clear()
        cb_arxiv = QCheckBox("arXiv")
        self.daily_checkboxes["arxiv"] = cb_arxiv
        layout.addWidget(cb_arxiv)
        for key in self._get_journal_keys_from_yaml():
            cb = QCheckBox(key.upper())
            self.daily_checkboxes[key] = cb
            layout.addWidget(cb)
        layout.addStretch()

    # ============================================================
    # 加载 / 保存
    # ============================================================
    def _load_font_size(self):
        settings = get_settings()
        font_size = settings.value("font_size", "14px")
        if isinstance(font_size, str) and font_size:
            self.font_size_combo.setCurrentText(font_size)

    # ============================================================
    # 主题(2026-09-02 加,抄 WAgent 主题系统)
    # ============================================================
    def _populate_theme_combo(self):
        """填主题下拉: builtin 优先, 然后 user 自定义"""
        self.theme_combo.blockSignals(True)
        self.theme_combo.clear()
        for theme in list_themes():
            marker = " ✦" if not theme.builtin else ""
            self.theme_combo.addItem(theme.display_name + marker, theme.name)
        self.theme_combo.blockSignals(False)

    def _on_theme_changed(self, _index: int):
        """主题下拉变化:实时切换 + 持久化。"""
        name = self.theme_combo.currentData()
        if not name:
            return
        set_theme(name, persist=True)

    def _on_new_theme(self):
        """打开主题编辑器(新建模式)"""
        from ui.theme_editor import ThemeEditorDialog
        dlg = ThemeEditorDialog(self)
        dlg.theme_saved.connect(self._on_theme_editor_saved)
        dlg.exec()

    def _on_edit_current(self):
        """编辑当前正在用的主题(只对 user: 主题有效)"""
        current = get_current_theme_name()
        if not current.startswith("user:"):
            return
        from ui.theme_editor import ThemeEditorDialog
        dlg = ThemeEditorDialog(self, edit_theme_name=current)
        dlg.theme_saved.connect(self._on_theme_editor_saved)
        dlg.exec()

    def _on_theme_editor_saved(self):
        """主题编辑器保存/删除后: 刷新下拉 + 当前选择"""
        self._populate_theme_combo()
        idx = self.theme_combo.findData(get_current_theme_name())
        if idx >= 0:
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentIndex(idx)
            self.theme_combo.blockSignals(False)

    def _load_config(self):
        settings = get_settings()

        # API key：先看 .api_keys.json（agent.infra.api_keys 自动 fallback 到 QSettings）
        all_keys = get_all_api_keys()
        self.deepseek_api_key.setText(all_keys.get("deepseek", ""))
        self.openai_api_key.setText(all_keys.get("openai", ""))
        self.minimax_api_key.setText(all_keys.get("minimax", ""))
        self.bocha_api_key.setText(all_keys.get("bocha", ""))
        self.tavily_api_key.setText(all_keys.get("tavily", ""))

        # LLM 模型配置（QSettings）
        self.deepseek_model.setText(settings.value("DEEPSEEK_MODEL", "deepseek-v4-flash"))
        self.deepseek_chat_model.setText(settings.value("DEEPSEEK_MODEL_CHAT", "deepseek-v4-flash"))
        self.openai_model.setText(settings.value("OPENAI_MODEL", ""))
        self.minimax_model.setText(settings.value("MINIMAX_MODEL", ""))
        self.minimax_base_url.setText(settings.value("MINIMAX_BASE_URL", ""))

        # Embedding
        enabled = settings.value("EMBEDDING_ENABLED", False)
        if isinstance(enabled, str):
            enabled = enabled.lower() in ["true", "1", "yes"]
        self.embedding_enabled_embed.setChecked(bool(enabled))
        self.embedder_type.setCurrentText(settings.value("EMBEDDER_TYPE", "local"))
        self.openai_embedding_model.setText(settings.value("OPENAI_EMBEDDING_MODEL", ""))
        self.local_embedder_name.setText(settings.value("LOCAL_EMBEDDER_NAME", ""))
        self.model_path.setText(settings.value("MODEL_PATH", ""))
        self.data_dir.setText(settings.value("DATA_DIR", ""))
        self.vector_dir.setText(settings.value("VECTOR_DIR", ""))

        # 系统提示词
        self.prompt_edit.setPlainText(settings.value("default_system_prompt", ""))

        # 每日总结范围
        try:
            daily_journals = json.loads(settings.value("daily_summary_journals", '["arxiv"]'))
        except Exception:
            daily_journals = ["arxiv"]
        for journal_key, cb in self.daily_checkboxes.items():
            cb.setChecked(journal_key in daily_journals)

        # 2026-09-02 P0.7 加:成本控制 3 档 + ceiling
        from config import Config
        tier = Config.get_cost_tier()
        tier_idx = {"L0": 0, "L1": 1, "L2": 2}.get(tier, 2)
        if hasattr(self, "cost_tier_combo"):
            self.cost_tier_combo.setCurrentIndex(tier_idx)
        if hasattr(self, "cost_ceiling_spin"):
            self.cost_ceiling_spin.setValue(Config.get_cost_ceiling_daily())
        # 刷新今日用量显示
        self._refresh_cost_usage()

    def _refresh_cost_usage(self):
        """2026-09-02 P0.7 加:刷新"今日已用 X/Y" 标签"""
        if not hasattr(self, "cost_usage_label"):
            return
        try:
            from agent.cost_guard import get_cost_guard
            from config import Config
            used = get_cost_guard().current_count()
            ceiling = Config.get_cost_ceiling_daily()
            pct = int(used / ceiling * 100) if ceiling else 0
            self.cost_usage_label.setText(f"今日已用: {used} / {ceiling} 次 ({pct}%)")
        except Exception as e:
            self.cost_usage_label.setText(f"今日用量读取失败: {e}")

    def _reset_cost_usage(self):
        """2026-09-02 P0.7 加:重置今日计数(给用户手动清零)"""
        from agent.cost_guard import get_cost_guard
        get_cost_guard().reset()
        self._refresh_cost_usage()
        QMessageBox.information(self, "已重置", "今日 LLM 计数已清零, 明天 0 点也会自动重置。")

    def reset_default_prompt(self):
        from agent.prompts import SYSTEM_PROMPT
        self.prompt_edit.setPlainText(SYSTEM_PROMPT)

    def accept(self):
        settings = get_settings()

        # 字体
        settings.setValue("font_size", self.font_size_combo.currentText())

        # API Key：写到 .api_keys.json（agent.infra.api_keys 统一管理）
        set_api_key("deepseek", self.deepseek_api_key.text().strip())
        set_api_key("openai", self.openai_api_key.text().strip())
        set_api_key("minimax", self.minimax_api_key.text().strip())
        set_api_key("bocha", self.bocha_api_key.text().strip())
        set_api_key("tavily", self.tavily_api_key.text().strip())

        # LLM 模型（QSettings）
        settings.setValue("DEEPSEEK_MODEL", self.deepseek_model.text().strip())
        settings.setValue("DEEPSEEK_MODEL_CHAT", self.deepseek_chat_model.text().strip())
        settings.setValue("OPENAI_MODEL", self.openai_model.text().strip())
        settings.setValue("MINIMAX_MODEL", self.minimax_model.text().strip())
        settings.setValue("MINIMAX_BASE_URL", self.minimax_base_url.text().strip())

        # Embedding
        settings.setValue("EMBEDDING_ENABLED", self.embedding_enabled_embed.isChecked())
        settings.setValue("EMBEDDER_TYPE", self.embedder_type.currentText())
        settings.setValue("OPENAI_EMBEDDING_MODEL", self.openai_embedding_model.text())
        settings.setValue("LOCAL_EMBEDDER_NAME", self.local_embedder_name.text())
        settings.setValue("MODEL_PATH", self.model_path.text())
        settings.setValue("DATA_DIR", self.data_dir.text())
        settings.setValue("VECTOR_DIR", self.vector_dir.text())

        # 期刊
        self._save_journals_config()

        # 系统提示词
        new_prompt = self.prompt_edit.toPlainText()
        settings.setValue("default_system_prompt", new_prompt)
        self.default_prompt_changed.emit(new_prompt)

        # 每日总结范围
        selected = [key for key, cb in self.daily_checkboxes.items() if cb.isChecked()]
        settings.setValue("daily_summary_journals", json.dumps(selected))

        # 2026-09-02 P0.7 加:成本控制 3 档 + ceiling
        if hasattr(self, "cost_tier_combo"):
            tier_idx = self.cost_tier_combo.currentIndex()
            tier = ["L0", "L1", "L2"][tier_idx] if 0 <= tier_idx <= 2 else "L2"
            from config import Config
            Config.set_cost_tier(tier)
        if hasattr(self, "cost_ceiling_spin"):
            from config import Config
            Config.set_cost_ceiling_daily(self.cost_ceiling_spin.value())

        # API key 改了 → 清掉 client 缓存
        reset_client_cache()

        QMessageBox.information(self, "提示",
                                "配置已保存。部分更改（如字体）需要重启应用才能生效。")
        super().accept()
