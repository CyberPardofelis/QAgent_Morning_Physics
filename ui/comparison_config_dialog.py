# ui/comparison_config_dialog.py
import json
from PySide6.QtCore import QSettings
from config.settings import get_settings
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QDoubleSpinBox, QSpinBox, QGroupBox, QPushButton, QMessageBox
)

class ComparisonConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("对比模式设置")
        self.setMinimumWidth(600)
        self.settings = get_settings()

        layout = QVBoxLayout(self)

        # 模型 A 配置组
        self.group_a = self._create_model_group("模型 A")
        layout.addWidget(self.group_a)

        # 模型 B 配置组
        self.group_b = self._create_model_group("模型 B")
        layout.addWidget(self.group_b)

        # 按钮
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_config)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.load_config()

    def _create_model_group(self, title):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)

        # 模型类型
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型:"))
        model_combo = QComboBox()
        model_combo.addItems(["openai", "deepseek-flash", "deepseek-pro"])
        model_layout.addWidget(model_combo)
        layout.addLayout(model_layout)

        # 思维链
        think_cb = QCheckBox("启用思维链 (思考模式)")
        layout.addWidget(think_cb)

        # 温度
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        temp_spin = QDoubleSpinBox()
        temp_spin.setRange(0.0, 2.0)
        temp_spin.setSingleStep(0.05)
        temp_layout.addWidget(temp_spin)
        layout.addLayout(temp_layout)

        # top_p
        topp_layout = QHBoxLayout()
        topp_layout.addWidget(QLabel("Top P:"))
        topp_spin = QDoubleSpinBox()
        topp_spin.setRange(0.0, 1.0)
        topp_spin.setSingleStep(0.05)
        topp_layout.addWidget(topp_spin)
        layout.addLayout(topp_layout)

        # 重复惩罚 (frequency_penalty)
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequency Penalty:"))
        freq_spin = QDoubleSpinBox()
        freq_spin.setRange(-2.0, 2.0)
        freq_spin.setSingleStep(0.1)
        freq_layout.addWidget(freq_spin)
        layout.addLayout(freq_layout)

        # 存在惩罚 (presence_penalty)
        pres_layout = QHBoxLayout()
        pres_layout.addWidget(QLabel("Presence Penalty:"))
        pres_spin = QDoubleSpinBox()
        pres_spin.setRange(-2.0, 2.0)
        pres_spin.setSingleStep(0.1)
        pres_layout.addWidget(pres_spin)
        layout.addLayout(pres_layout)

        # 保存控件引用
        group.model_combo = model_combo
        group.think_cb = think_cb
        group.temp_spin = temp_spin
        group.topp_spin = topp_spin
        group.freq_spin = freq_spin
        group.pres_spin = pres_spin
        return group

    def load_config(self):
        config_a = json.loads(self.settings.value("comparison/config_a", '{"model":"deepseek-flash","thinking":true,"temperature":0.3,"top_p":0.9,"frequency_penalty":0.0,"presence_penalty":0.0}'))
        config_b = json.loads(self.settings.value("comparison/config_b", '{"model":"deepseek-pro","thinking":true,"temperature":0.3,"top_p":0.9,"frequency_penalty":0.0,"presence_penalty":0.0}'))

        self._apply_config(self.group_a, config_a)
        self._apply_config(self.group_b, config_b)

    def _apply_config(self, group, config):
        group.model_combo.setCurrentText(config.get("model", "deepseek-flash"))
        group.think_cb.setChecked(config.get("thinking", True))
        group.temp_spin.setValue(config.get("temperature", 0.3))
        group.topp_spin.setValue(config.get("top_p", 0.9))
        group.freq_spin.setValue(config.get("frequency_penalty", 0.0))
        group.pres_spin.setValue(config.get("presence_penalty", 0.0))

    def _get_config(self, group):
        return {
            "model": group.model_combo.currentText(),
            "thinking": group.think_cb.isChecked(),
            "temperature": group.temp_spin.value(),
            "top_p": group.topp_spin.value(),
            "frequency_penalty": group.freq_spin.value(),
            "presence_penalty": group.pres_spin.value(),
        }

    def save_config(self):
        config_a = self._get_config(self.group_a)
        config_b = self._get_config(self.group_b)
        self.settings.setValue("comparison/config_a", json.dumps(config_a))
        self.settings.setValue("comparison/config_b", json.dumps(config_b))
        QMessageBox.information(self, "保存成功", "对比模式配置已保存")
        self.accept()