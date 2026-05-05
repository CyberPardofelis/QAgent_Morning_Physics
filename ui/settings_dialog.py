import os
import yaml
import json
from PySide6.QtCore import Qt, QSettings, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QCheckBox,
    QComboBox, QDialogButtonBox, QLabel, QTabWidget, QWidget,
    QScrollArea, QTextEdit, QPushButton, QMessageBox
)
from ui.styles import GLOBAL_STYLESHEET

class SettingsDialog(QDialog):
    default_prompt_changed = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumSize(700, 500)  # 稍微增大以容纳标签页
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # 主布局
        main_layout = QVBoxLayout(self)

        # 创建标签页
        tab_widget = QTabWidget()
        tab_widget.setDocumentMode(True)  # 扁平风格
        main_layout.addWidget(tab_widget)

        # ----- 标签页1：首选项 -----
        pref_tab = QWidget()
        pref_layout = QVBoxLayout(pref_tab)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        # 字体大小
        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems(["10px", "11px", "12px", "13px", "14px", "15px", "16px"])
        self.font_size_combo.setCurrentText("14px")
        form_layout.addRow("全局字体大小:", self.font_size_combo)

        # 通用 API 配置
        self.openai_api_key = QLineEdit()
        self.openai_api_key.setEchoMode(QLineEdit.Password)
        form_layout.addRow("OpenAI API Key:", self.openai_api_key)

        self.openai_model = QLineEdit()
        form_layout.addRow("OpenAI Model:", self.openai_model)

        self.deepseek_api_key = QLineEdit()
        self.deepseek_api_key.setEchoMode(QLineEdit.Password)
        form_layout.addRow("DeepSeek API Key:", self.deepseek_api_key)

        self.deepseek_model = QLineEdit()
        form_layout.addRow("DeepSeek Model:", self.deepseek_model)

        self.deepseek_chat_model = QLineEdit()
        form_layout.addRow("DeepSeek Chat Model:", self.deepseek_chat_model)

        # Embedding 开关（首选项版本）
        self.embedding_enabled_pref = QCheckBox()
        form_layout.addRow("启用 Embedding:", self.embedding_enabled_pref)

        # 添加一个说明标签（可选）
        form_layout.addRow("", QLabel("注：详细 Embedding 配置请前往「Embedding 设置」标签页"))

        pref_layout.addLayout(form_layout)
        pref_layout.addStretch()

        # ----- 标签页2：期刊设置 -----
        journal_tab = QWidget()
        journal_layout = QVBoxLayout(journal_tab)

        # 使用 QScrollArea 容纳期刊配置
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        journal_layout.addWidget(scroll_area)

        # 期刊配置内容部件
        journal_content = QWidget()
        scroll_area.setWidget(journal_content)
        self.journal_form_layout = QFormLayout(journal_content)
        self.journal_form_layout.setSpacing(10)

        # 存储每个期刊的输入控件
        self.journal_section_inputs = {}
        self._create_journal_inputs()  # 从 YAML 加载并创建输入框

        journal_layout.addStretch()

        # ----- 标签页3：Embedding 设置 -----
        embed_tab = QWidget()
        embed_layout = QVBoxLayout(embed_tab)

        embed_form = QFormLayout()
        embed_form.setSpacing(10)

        # Embedding 开关（Embedding 页面版本）
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

        # ----- 标签页4：系统提示词 -----
        prompt_tab = QWidget()
        prompt_layout = QVBoxLayout(prompt_tab)

        # 说明标签
        info_label = QLabel("设置默认系统提示词（将作为所有新对话的初始系统提示，分组可单独覆盖）")
        info_label.setWordWrap(True)
        prompt_layout.addWidget(info_label)

        # 多行文本编辑框
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("输入默认系统提示词...")
        prompt_layout.addWidget(self.prompt_edit)

        # 恢复默认按钮
        reset_btn = QPushButton("恢复默认提示词")
        reset_btn.clicked.connect(self.reset_default_prompt)
        prompt_layout.addWidget(reset_btn)

        # 标签页添加到 tab_widget
        tab_widget.addTab(pref_tab, "首选项")
        tab_widget.addTab(journal_tab, "期刊设置")
        tab_widget.addTab(embed_tab, "Embedding 设置")
        tab_widget.addTab(prompt_tab, "系统提示词")

        # ----- 底部按钮 -----
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        #----- 标签页5：每日总结范围 -----

        daily_scope_tab = QWidget()
        daily_scope_layout = QVBoxLayout(daily_scope_tab)

        # 使用 QScrollArea 容纳复选框列表
        daily_scroll = QScrollArea()
        daily_scroll.setWidgetResizable(True)
        daily_scope_layout.addWidget(daily_scroll)

        daily_content = QWidget()
        daily_content_layout = QVBoxLayout(daily_content)
        daily_scroll.setWidget(daily_content)

        # 存储复选框的字典 self.daily_checkboxes
        self.daily_checkboxes = {}

        # 加载期刊列表（包括 arxiv 和 YAML 中的 journal keys）
        self._populate_daily_checkboxes(daily_content_layout)

        daily_scope_layout.addStretch()
        tab_widget.addTab(daily_scope_tab, "每日总结范围")

        # 加载现有配置
        self._load_config()
        self._load_journals_config()
        self._load_font_size()

        # 同步两个 Embedding 开关的状态（双向连接）
        self.embedding_enabled_pref.stateChanged.connect(
            lambda state: self.embedding_enabled_embed.setCheckState(state)
        )
        self.embedding_enabled_embed.stateChanged.connect(
            lambda state: self.embedding_enabled_pref.setCheckState(state)
        )

    def reset_default_prompt(self):
        from agent.prompts import SYSTEM_PROMPT
        self.prompt_edit.setPlainText(SYSTEM_PROMPT)

    def _load_font_size(self):
        """从 QSettings 加载字体大小设置"""
        settings = QSettings("CyberPardofelis", "LLM科研助手")
        font_size = settings.value("font_size", "10px")
        self.font_size_combo.setCurrentText(font_size)

    def _create_journal_inputs(self):
        """读取 YAML 文件，为每个期刊创建 section 输入框"""
        yaml_path = self._get_yaml_path()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                journals_data = yaml.safe_load(f)
            journals = journals_data.get('journals', {})
        except Exception as e:
            print(f"读取期刊配置失败: {e}")
            journals = {}

        # 清空原有布局（防止重复）
        while self.journal_form_layout.rowCount() > 0:
            self.journal_form_layout.removeRow(0)

        self.journal_section_inputs.clear()
        for journal_key, journal_info in journals.items():
            section_input = QLineEdit(journal_info.get('section', ''))
            self.journal_section_inputs[journal_key] = section_input
            self.journal_form_layout.addRow(f"{journal_key} section:", section_input)

    def _get_yaml_path(self):
        """获取 journals.yaml 的绝对路径"""
        # SettingsDialog 所在文件位于项目根目录，agent 是子目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, 'agent', 'journals.yaml')

    def _load_config(self):
        """从 QSettings 加载首选项和 Embedding 配置"""
        settings = QSettings("CyberPardofelis", "LLM科研助手")

        # 首选项
        self.openai_api_key.setText(settings.value("OPENAI_API_KEY", ""))
        self.openai_model.setText(settings.value("OPENAI_MODEL", ""))
        self.deepseek_api_key.setText(settings.value("DEEPSEEK_API_KEY", ""))
        self.deepseek_model.setText(settings.value("DEEPSEEK_MODEL", ""))
        self.deepseek_chat_model.setText(settings.value("DEEPSEEK_MODEL_CHAT", ""))

        # Embedding 开关（同时设置两个）
        enabled = settings.value("EMBEDDING_ENABLED", False)
        if isinstance(enabled, str):
            enabled = enabled.lower() in ["true", "1", "yes"]
        self.embedding_enabled_pref.setChecked(enabled)
        self.embedding_enabled_embed.setChecked(enabled)

        # Embedding 其他配置
        self.embedder_type.setCurrentText(settings.value("EMBEDDER_TYPE", "local"))
        self.openai_embedding_model.setText(settings.value("OPENAI_EMBEDDING_MODEL", ""))
        self.local_embedder_name.setText(settings.value("LOCAL_EMBEDDER_NAME", ""))
        self.model_path.setText(settings.value("MODEL_PATH", ""))
        self.data_dir.setText(settings.value("DATA_DIR", ""))
        self.vector_dir.setText(settings.value("VECTOR_DIR", ""))

        # 加载每日总结范围
        daily_journals = json.loads(settings.value("daily_summary_journals", '["arxiv"]'))
        for journal_key, cb in self.daily_checkboxes.items():
            cb.setChecked(journal_key in daily_journals)

    def _load_journals_config(self):
        """从 YAML 文件加载期刊 section 值到输入框"""
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
        """将用户修改的期刊 section 值写回 YAML 文件"""
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

    def accept(self):
        """保存所有配置"""
        settings = QSettings("CyberPardofelis", "LLM科研助手")

        # 保存字体大小
        settings.setValue("font_size", self.font_size_combo.currentText())

        # 保存首选项 API 配置
        settings.setValue("OPENAI_API_KEY", self.openai_api_key.text())
        settings.setValue("OPENAI_MODEL", self.openai_model.text())
        settings.setValue("DEEPSEEK_API_KEY", self.deepseek_api_key.text())
        settings.setValue("DEEPSEEK_MODEL", self.deepseek_model.text())
        settings.setValue("DEEPSEEK_MODEL_CHAT", self.deepseek_chat_model.text())

        # 保存 Embedding 配置（开关以任一为准，两者已同步）
        settings.setValue("EMBEDDING_ENABLED", self.embedding_enabled_pref.isChecked())
        settings.setValue("EMBEDDER_TYPE", self.embedder_type.currentText())
        settings.setValue("OPENAI_EMBEDDING_MODEL", self.openai_embedding_model.text())
        settings.setValue("LOCAL_EMBEDDER_NAME", self.local_embedder_name.text())
        settings.setValue("MODEL_PATH", self.model_path.text())
        settings.setValue("DATA_DIR", self.data_dir.text())
        settings.setValue("VECTOR_DIR", self.vector_dir.text())

        # 保存期刊配置到 YAML
        self._save_journals_config()

        # 应用字体大小（需要重启，但可以提示）
        QMessageBox.information(self, "提示",
                                "配置已保存。部分更改（如字体）需要重启应用才能生效。")
        new_prompt = self.prompt_edit.toPlainText()
        settings = QSettings("CyberPardofelis", "LLM科研助手")
        settings.setValue("default_system_prompt", new_prompt)
        self.default_prompt_changed.emit(new_prompt)

        # 保存每日总结范围
        selected = [key for key, cb in self.daily_checkboxes.items() if cb.isChecked()]
        settings.setValue("daily_summary_journals", json.dumps(selected))

        super().accept()

    def _get_journal_keys_from_yaml(self):
        """从 journals.yaml 读取所有期刊键名（如 prl, pra...）"""
        yaml_path = self._get_yaml_path()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return list(data.get('journals', {}).keys())
        except Exception as e:
            print(f"读取期刊列表失败: {e}")
            return []

    def _populate_daily_checkboxes(self, layout):
        """创建每日总结范围的复选框"""
        # 清除已有
        for cb in self.daily_checkboxes.values():
            cb.deleteLater()
        self.daily_checkboxes.clear()

        # 添加 arxiv
        cb_arxiv = QCheckBox("arXiv")
        self.daily_checkboxes["arxiv"] = cb_arxiv
        layout.addWidget(cb_arxiv)

        # 添加 YAML 中的期刊
        journal_keys = self._get_journal_keys_from_yaml()
        for key in journal_keys:
            cb = QCheckBox(key.upper())  # 显示大写，如 PRL
            self.daily_checkboxes[key] = cb
            layout.addWidget(cb)

        layout.addStretch()
