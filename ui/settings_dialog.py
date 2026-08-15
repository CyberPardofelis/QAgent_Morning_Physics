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
    QHBoxLayout,
)
from agent.infra.api_keys import (
    get_all_api_keys, set_api_key, has_api_key,
)
from agent.infra.db import health_check
from agent.llm_clients import reset_client_cache
from ui.styles import GLOBAL_STYLESHEET

# 数据根目录（与 agent.infra.db._data_root() 一致）
_DATA_ROOT = Path("data").resolve()


class SettingsDialog(QDialog):
    default_prompt_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumSize(720, 620)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        main_layout = QVBoxLayout(self)
        tab_widget = QTabWidget()
        tab_widget.setDocumentMode(True)
        main_layout.addWidget(tab_widget)

        # ----- 1. 首选项（仅字体） -----
        pref_tab = QWidget()
        pref_layout = QVBoxLayout(pref_tab)
        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems(["10px", "11px", "12px", "13px", "14px", "15px", "16px"])
        self.font_size_combo.setCurrentText("14px")
        form_layout.addRow("全局字体大小:", self.font_size_combo)
        pref_layout.addLayout(form_layout)
        pref_layout.addStretch()
        tab_widget.addTab(pref_tab, "首选项")

        # ----- 2. LLM API -----
        tab_widget.addTab(self._build_llm_tab(), "LLM API")

        # ----- 3. 搜索服务 -----
        tab_widget.addTab(self._build_search_tab(), "搜索服务")

        # ----- 4. 系统提示词 -----
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
        tab_widget.addTab(prompt_tab, "系统提示词")

        # ----- 5. 数据状态 -----
        tab_widget.addTab(self._build_db_tab(), "数据状态")

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
        tab_widget.addTab(journal_tab, "期刊设置")

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
        tab_widget.addTab(embed_tab, "Embedding 设置")

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
        tab_widget.addTab(daily_scope_tab, "每日总结范围")

        # 底部按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        # 加载所有配置
        self._load_config()
        self._load_journals_config()
        self._load_font_size()
        self._refresh_db_status()

    # ============================================================
    # LLM API Tab
    # ============================================================
    def _build_llm_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
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
        mm_hint.setStyleSheet("color: #888; font-size: 11px;")
        mm_layout.addRow("", mm_hint)
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
        ddg_note.setStyleSheet("color: gray; font-size: 11px;")
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
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        path_label = QLabel(f"数据目录: {_DATA_ROOT}")
        path_label.setStyleSheet("color: gray; font-size: 11px;")
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

        # API key 改了 → 清掉 client 缓存
        reset_client_cache()

        QMessageBox.information(self, "提示",
                                "配置已保存。部分更改（如字体）需要重启应用才能生效。")
        super().accept()
