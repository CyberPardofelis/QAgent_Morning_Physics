# tree_panel.py
import json
from datetime import datetime
from PySide6.QtCore import Qt, QSettings
from config.settings import get_settings
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTreeWidget,
    QTreeWidgetItem, QMenu, QInputDialog, QMessageBox
)
from agent.history_manager import Conversation


class ConversationTreePanel(QWidget):
    """左侧对话树面板 — 原样提取 ChatWindow 中树相关代码"""

    def __init__(self, history_manager, parent=None):
        super().__init__(parent)
        self.history_manager = history_manager
        self._setup_ui()
        self._connect_signals()

        # 外部回调
        self.on_conversation_selected = None
        self.on_create_conversation = None
        self.on_edit_group_prompt = None
        self.on_delete_conversation = None
        self.on_move_conversation = None   # 新增：移动对话后的回调

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.new_conv_btn = QPushButton("＋ 新建对话")
        self.new_conv_btn.setObjectName("newConvBtn")
        layout.addWidget(self.new_conv_btn)

        self.new_group_btn = QPushButton("＋ 新建分组")
        self.new_group_btn.setObjectName("newGroupBtn")
        layout.addWidget(self.new_group_btn)

        self.conv_tree = QTreeWidget()
        self.conv_tree.setHeaderHidden(True)
        self.conv_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.conv_tree.customContextMenuRequested.connect(self.show_tree_context_menu)
        self.conv_tree.itemClicked.connect(self.on_tree_item_clicked)
        layout.addWidget(self.conv_tree)

    def _connect_signals(self):
        self.new_conv_btn.clicked.connect(
            lambda: self.on_create_conversation and self.on_create_conversation(None)
        )
        self.new_group_btn.clicked.connect(self.create_new_group)

    # -------------------- 树刷新 --------------------
    def refresh_tree(self, current_conv_id=None):
        grouped = self.history_manager.get_conversations_by_group()
        groups_info = {g["group_id"]: g for g in self.history_manager.get_all_groups()}
        self.conv_tree.clear()
        group_items = {}
        for group_id, group_meta in groups_info.items():
            group_item = QTreeWidgetItem(self.conv_tree)
            group_item.setText(0, group_meta['name'])
            group_item.setData(0, Qt.UserRole, f"group:{group_id}")
            group_items[group_id] = group_item
        for group_id, convs in grouped.items():
            if group_id is None:
                continue
            if group_id in group_items:
                for conv_meta in convs:
                    self._add_conversation_item(group_items[group_id], conv_meta)
        ungrouped = grouped.get(None, [])
        if ungrouped:
            ungrouped_item = QTreeWidgetItem(self.conv_tree)
            ungrouped_item.setText(0, "未分组")
            ungrouped_item.setData(0, Qt.UserRole, "group:ungrouped")
            for conv_meta in ungrouped:
                self._add_conversation_item(ungrouped_item, conv_meta)

        if current_conv_id:
            self._highlight_conversation_in_tree(current_conv_id)

    def _add_conversation_item(self, parent, conv_meta):
        item = QTreeWidgetItem(parent)
        updated = datetime.fromisoformat(conv_meta["updated_at"]).strftime("%m-%d %H:%M")
        item.setText(0, f"{conv_meta['title']}\n{updated}")
        item.setData(0, Qt.UserRole, f"conv:{conv_meta['conv_id']}")

    def on_tree_item_clicked(self, item, _):
        data = item.data(0, Qt.UserRole)
        if data and data.startswith("conv:"):
            conv_id = data.split(":", 1)[1]
            if self.on_conversation_selected:
                self.on_conversation_selected(conv_id)

    def _highlight_conversation_in_tree(self, conv_id):
        def find_item(root):
            for i in range(root.childCount()):
                child = root.child(i)
                if child.data(0, Qt.UserRole) == f"conv:{conv_id}":
                    return child
                found = find_item(child)
                if found:
                    return found
            return None
        item = find_item(self.conv_tree.invisibleRootItem())
        if item:
            self.conv_tree.setCurrentItem(item)

    # -------------------- 右键菜单 --------------------
    def show_tree_context_menu(self, pos):
        item = self.conv_tree.itemAt(pos)
        if not item:
            menu = QMenu()
            new_group_action = QAction("新建分组", self)
            new_group_action.triggered.connect(self.create_new_group)
            menu.addAction(new_group_action)
            menu.exec(self.conv_tree.mapToGlobal(pos))
            return
        data = item.data(0, Qt.UserRole)
        if data and data.startswith("group:"):
            group_id = data.split(":", 1)[1]
            menu = QMenu()
            rename_action = QAction("重命名分组", self)
            rename_action.triggered.connect(lambda: self.rename_group(group_id))
            delete_action = QAction("删除分组", self)
            delete_action.triggered.connect(lambda: self.delete_group(group_id))
            new_conv_action = QAction("在此分组新建对话", self)
            new_conv_action.triggered.connect(
                lambda: self.on_create_conversation and self.on_create_conversation(group_id)
            )
            set_prompt_action = QAction("设置系统提示词", self)
            set_prompt_action.triggered.connect(
                lambda: self.on_edit_group_prompt and self.on_edit_group_prompt(group_id)
            )
            menu.addActions([rename_action, delete_action, new_conv_action, set_prompt_action])
            menu.exec(self.conv_tree.mapToGlobal(pos))
        elif data and data.startswith("conv:"):
            conv_id = data.split(":", 1)[1]
            conv = self.history_manager.get_conversation(conv_id)
            if not conv:
                return
            menu = QMenu()
            rename_action = QAction("重命名对话", self)
            rename_action.triggered.connect(lambda: self.rename_conversation(conv))
            delete_action = QAction("删除对话", self)
            delete_action.triggered.connect(
                lambda: self.on_delete_conversation and self.on_delete_conversation(conv_id)
            )
            move_menu = QMenu("移动到分组", self)
            move_to_ungrouped = QAction("未分组", self)
            move_to_ungrouped.triggered.connect(lambda: self.move_conversation(conv_id, None))
            move_menu.addAction(move_to_ungrouped)
            for group in self.history_manager.get_all_groups():
                group_action = QAction(group["name"], self)
                group_action.triggered.connect(
                    lambda checked, gid=group["group_id"]: self.move_conversation(conv_id, gid)
                )
                move_menu.addAction(group_action)
            menu.addMenu(move_menu)
            menu.addAction(rename_action)
            menu.addAction(delete_action)
            menu.exec(self.conv_tree.mapToGlobal(pos))

    # -------------------- 分组管理 --------------------
    def create_new_group(self):
        name, ok = QInputDialog.getText(self, "新建分组", "输入分组名称：", text="新分组")
        if ok and name.strip():
            self.history_manager.create_group(name.strip())
            self.refresh_tree()

    def rename_group(self, group_id):
        current_name = self.history_manager.groups[group_id]["name"]
        new_name, ok = QInputDialog.getText(self, "重命名分组", "新名称：", text=current_name)
        if ok and new_name.strip():
            self.history_manager.rename_group(group_id, new_name.strip())
            self.refresh_tree()

    def delete_group(self, group_id):
        reply = QMessageBox.question(
            self, "删除分组", "确定删除分组？该分组下的对话将移至「未分组」。",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.history_manager.delete_group(group_id, move_to_group_id=None)
            settings = get_settings()
            settings.remove(f"group_system_prompt/{group_id}")
            self.refresh_tree()

    def move_conversation(self, conv_id, target_group_id):
        """移动对话到指定分组，并通知外部更新当前对话的 group_id"""
        self.history_manager.move_conversation(conv_id, target_group_id)
        # 关键修复：调用回调，让外部同步当前对话的 group_id
        if self.on_move_conversation:
            self.on_move_conversation(conv_id, target_group_id)
        self.refresh_tree()

    def rename_conversation(self, conv: Conversation):
        new_title, ok = QInputDialog.getText(self, "重命名对话", "输入新标题：", text=conv.title)
        if ok and new_title.strip():
            conv.title = new_title.strip()
            conv.updated_at = datetime.now().isoformat()
            self.history_manager.save_conversation(conv)
            self.refresh_tree()