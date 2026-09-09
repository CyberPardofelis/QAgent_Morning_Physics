from PySide6.QtCore import QSettings
from config.settings import get_settings
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTextEdit, QHBoxLayout,
    QPushButton, QMessageBox
)
from ui.styles import get_stylesheet

class GroupSystemPromptDialog(QDialog):
    def __init__(self, group_id, group_name, parent=None):
        super().__init__(parent)
        self.group_id = group_id
        self.group_name = group_name
        self.setWindowTitle(f"设置系统提示词 - {group_name}")
        self.setMinimumSize(600, 400)
        self.setStyleSheet(get_stylesheet())  # 2026-09-02 改:用 get_stylesheet() 跟主题

        layout = QVBoxLayout(self)

        # 提示标签
        label = QLabel(f"为分组「{group_name}」设置自定义系统提示词（留空则使用全局默认）")
        layout.addWidget(label)

        # 文本编辑框
        self.prompt_edit = QTextEdit()
        layout.addWidget(self.prompt_edit)

        # 按钮
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_prompt)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # 加载已有设置
        self.load_prompt()

    def load_prompt(self):
        settings = get_settings()
        prompt = settings.value(f"group_system_prompt/{self.group_id}")
        if prompt:
            self.prompt_edit.setPlainText(prompt)

    def save_prompt(self):
        # 检查分组是否仍然存在（防止删除后保存）
        parent = self.parent()
        if parent and hasattr(parent, 'history_manager'):
            groups = parent.history_manager.get_all_groups()
            if not any(g['group_id'] == self.group_id for g in groups):
                QMessageBox.warning(self, "错误", "该分组已被删除，无法保存")
                self.reject()
                return

        new_prompt = self.prompt_edit.toPlainText().strip()
        settings = get_settings()
        if new_prompt:
            settings.setValue(f"group_system_prompt/{self.group_id}", new_prompt)
        else:
            settings.remove(f"group_system_prompt/{self.group_id}")  # 空字符串视为使用默认
        self.accept()
