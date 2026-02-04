"""
用户管理GUI界面
User Management GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable, List, Dict, Any
from .user_manager import UserManager, User


class UserManagementDialog:
    """用户管理对话框"""
    
    def __init__(self, parent, log_callback: Optional[Callable] = None):
        """初始化用户管理对话框
        
        Args:
            parent: 父窗口
            log_callback: 日志回调函数
        """
        self.parent = parent
        self.log = log_callback if log_callback else print
        
        # 创建窗口（宽度改为1400，左右各700）
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("用户管理 & 批量添加账号")
        self.dialog.geometry("1400x700")
        self.dialog.resizable(True, True)
        
        # 先隐藏窗口，避免白屏
        self.dialog.withdraw()
        
        # 居中显示
        self._center_window()
        
        # 创建用户管理器
        self.user_manager = UserManager()
        
        # 创建界面
        self._create_widgets()
        
        # 加载数据
        self._refresh_user_list()
        
        # 所有内容准备完成后再显示窗口
        self.dialog.deiconify()
        self.dialog.lift()
        self.dialog.focus_force()
    
    def _center_window(self):
        """将窗口居中显示"""
        self.dialog.update_idletasks()
        width = 1400
        height = 700
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === 创建左右分栏 ===
        # 左侧：用户管理
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 右侧：批量添加账号
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # === 左侧：用户管理区域 ===
        left_title = ttk.Label(left_frame, text="用户管理", font=("Microsoft YaHei UI", 12, "bold"), foreground="blue")
        left_title.pack(pady=(0, 10))
        
        # === 上半部分：用户列表区域 ===
        user_frame = ttk.LabelFrame(left_frame, text="管理员列表", padding="10")
        user_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建用户Treeview
        user_columns = ("user_id", "user_name", "transfer_recipient", "account_count", "enabled", "description")
        self.user_tree = ttk.Treeview(user_frame, columns=user_columns, show="headings", height=8)
        
        # 定义列标题和宽度
        user_column_config = {
            "user_id": ("用户ID", 100),
            "user_name": ("用户名称", 120),
            "transfer_recipient": ("转账收款人", 120),
            "account_count": ("账号数量", 80),
            "enabled": ("状态", 60),
            "description": ("备注", 200)
        }
        
        for col, (heading, width) in user_column_config.items():
            self.user_tree.heading(col, text=heading)
            self.user_tree.column(col, width=width, anchor=tk.CENTER)
        
        # 添加滚动条
        user_scrollbar_y = ttk.Scrollbar(user_frame, orient=tk.VERTICAL, command=self.user_tree.yview)
        user_scrollbar_x = ttk.Scrollbar(user_frame, orient=tk.HORIZONTAL, command=self.user_tree.xview)
        self.user_tree.configure(yscrollcommand=user_scrollbar_y.set, xscrollcommand=user_scrollbar_x.set)
        
        # 布局
        self.user_tree.grid(row=0, column=0, sticky="nsew")
        user_scrollbar_y.grid(row=0, column=1, sticky="ns")
        user_scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        user_frame.grid_rowconfigure(0, weight=1)
        user_frame.grid_columnconfigure(0, weight=1)
        
        # 配置标签颜色
        self.user_tree.tag_configure("enabled", foreground="green")
        self.user_tree.tag_configure("disabled", foreground="gray")
        
        # 绑定选择事件（选中管理员时刷新账号列表）
        self.user_tree.bind("<<TreeviewSelect>>", self._on_user_selected)
        # 绑定双击事件（编辑管理员）
        self.user_tree.bind("<Double-Button-1>", lambda e: self._edit_user())
        
        # === 用户操作按钮 ===
        user_button_frame = ttk.Frame(left_frame)
        user_button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(user_button_frame, text="➕ 添加管理员", command=self._add_user, width=12).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(user_button_frame, text="✏️ 编辑管理员", command=self._edit_user, width=12).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(user_button_frame, text="🗑️ 删除管理员", command=self._delete_user, width=12).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(user_button_frame, text="🔄 刷新", command=self._refresh_user_list, width=10).pack(side=tk.LEFT, padx=(0, 5))
        
        # === 下半部分：该管理员的账号列表区域 ===
        account_frame = ttk.LabelFrame(left_frame, text="该管理员的账号列表（可勾选后移除）", padding="10")
        account_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建账号Treeview（带勾选框）
        account_columns = ("phone", "nickname", "user_id", "status", "owner")
        self.account_tree = ttk.Treeview(account_frame, columns=account_columns, show="tree headings", height=8)
        
        # 配置勾选框列
        self.account_tree.heading("#0", text="", anchor=tk.CENTER)
        self.account_tree.column("#0", width=40, anchor=tk.CENTER, stretch=False, minwidth=40)
        
        # 定义列标题和宽度
        account_column_config = {
            "phone": ("手机号", 120),
            "nickname": ("昵称", 150),
            "user_id": ("用户ID", 100),
            "status": ("最新状态", 100),
            "owner": ("当前管理员", 100)
        }
        
        for col, (heading, width) in account_column_config.items():
            self.account_tree.heading(col, text=heading)
            self.account_tree.column(col, width=width, anchor=tk.CENTER)
        
        # 绑定点击事件（用于切换勾选状态）
        self.account_tree.bind("<Button-1>", self._on_account_tree_click)
        
        # 初始化勾选状态
        self.account_checked_items = {}  # {item_id: True/False}
        
        # 添加滚动条
        account_scrollbar_y = ttk.Scrollbar(account_frame, orient=tk.VERTICAL, command=self.account_tree.yview)
        account_scrollbar_x = ttk.Scrollbar(account_frame, orient=tk.HORIZONTAL, command=self.account_tree.xview)
        self.account_tree.configure(yscrollcommand=account_scrollbar_y.set, xscrollcommand=account_scrollbar_x.set)
        
        # 布局
        self.account_tree.grid(row=0, column=0, sticky="nsew")
        account_scrollbar_y.grid(row=0, column=1, sticky="ns")
        account_scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        account_frame.grid_rowconfigure(0, weight=1)
        account_frame.grid_columnconfigure(0, weight=1)
        
        # === 账号操作按钮 ===
        account_button_frame = ttk.Frame(left_frame)
        account_button_frame.pack(fill=tk.X)
        
        ttk.Button(account_button_frame, text="全选", command=self._select_all_accounts, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(account_button_frame, text="反选", command=self._invert_account_selection, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(account_button_frame, text="📋 添加账号", command=self._add_accounts_to_user, width=12).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(account_button_frame, text="🗑️ 移除选中账号", command=self._remove_owner_from_selected, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(account_button_frame, text="🔄 刷新", command=self._refresh_current_user_accounts, width=10).pack(side=tk.LEFT, padx=(0, 5))
        
        # === 右侧：批量添加账号区域 ===
        self._create_batch_add_widgets(right_frame)
    
    def _refresh_user_list(self):
        """刷新用户列表"""
        # 清空现有项
        for item in self.user_tree.get_children():
            self.user_tree.delete(item)
        
        # 添加用户
        for user in self.user_manager.get_all_users():
            account_count = len(self.user_manager.get_user_accounts(user.user_id))
            status = "启用" if user.enabled else "禁用"
            tag = "enabled" if user.enabled else "disabled"
            
            # 显示多个收款人（逗号分隔，最多显示3个）
            recipients_display = ", ".join(user.transfer_recipients[:3])
            if len(user.transfer_recipients) > 3:
                recipients_display += f" 等{len(user.transfer_recipients)}个"
            
            # 用户ID显示：去掉 "user_" 前缀，只显示数字部分
            display_user_id = user.user_id.replace("user_", "") if user.user_id.startswith("user_") else user.user_id
            
            values = (
                display_user_id,
                user.user_name,
                recipients_display,
                account_count,
                status,
                user.description
            )
            
            # 使用 tags 参数存储完整的 user_id，用于后续操作
            item_id = self.user_tree.insert("", tk.END, values=values, tags=(tag, user.user_id))
        
        # 不自动刷新账号列表，等待用户选择管理员
    
    def _on_user_selected(self, event):
        """当选中管理员时，刷新该管理员的账号列表"""
        selection = self.user_tree.selection()
        if not selection:
            # 清空账号列表
            for item in self.account_tree.get_children():
                self.account_tree.delete(item)
            self.account_checked_items = {}
            return
        
        # 获取选中的用户ID（从 tags 中获取完整的 user_id）
        item = selection[0]
        tags = self.user_tree.item(item, 'tags')
        # tags 中第二个元素是完整的 user_id（第一个是 enabled/disabled）
        user_id = tags[1] if len(tags) > 1 else tags[0]
        
        # 刷新该用户的账号列表
        self._refresh_user_accounts(user_id)
    
    def _refresh_user_accounts(self, user_id: str):
        """刷新指定管理员的账号列表
        
        Args:
            user_id: 用户ID
        """
        # 清空现有项
        for item in self.account_tree.get_children():
            self.account_tree.delete(item)
        
        self.account_checked_items = {}
        
        # 获取该用户的所有账号
        assigned_phones = set(self.user_manager.get_user_accounts(user_id))
        
        if not assigned_phones:
            # 没有分配的账号
            return
        
        # 从数据库加载账号信息
        from .local_db import LocalDatabase
        db = LocalDatabase()
        
        # 对每个分配的账号，获取详细信息
        for phone in assigned_phones:
            # 获取账号汇总信息
            summary = db.get_account_summary(phone)
            
            if summary:
                nickname = summary.get('nickname', '-') or '-'
                account_user_id = summary.get('user_id', '-') or '-'
                
                # 获取最新记录以获取状态和管理员
                records = db.get_history_records(phone, limit=1)
                if records:
                    last_status = records[0].get('状态', '未处理')
                    owner_name = records[0].get('owner', '-') or '-'
                else:
                    last_status = '未处理'
                    owner_name = '-'
            else:
                # 数据库中没有记录，显示基本信息
                nickname = '-'
                account_user_id = '-'
                last_status = '未处理'
                owner_name = '-'
            
            values = (phone, nickname, account_user_id, last_status, owner_name)
            item_id = self.account_tree.insert("", tk.END, text="□", values=values)
            self.account_checked_items[item_id] = False
    
    def _refresh_current_user_accounts(self):
        """刷新当前选中管理员的账号列表"""
        # 获取选中的管理员
        user_id, _ = self._get_selected_user_id()
        if not user_id:
            return
        
        self._refresh_user_accounts(user_id)
    
    def _on_account_tree_click(self, event):
        """处理账号树的点击事件"""
        region = self.account_tree.identify("region", event.x, event.y)
        if region == "tree":
            # 点击了勾选框列
            item = self.account_tree.identify_row(event.y)
            if item:
                # 切换勾选状态
                current_state = self.account_checked_items.get(item, False)
                new_state = not current_state
                self.account_checked_items[item] = new_state
                
                # 更新显示
                self.account_tree.item(item, text="☑" if new_state else "□")
    
    def _select_all_accounts(self):
        """全选所有账号"""
        for item_id in self.account_tree.get_children():
            self.account_checked_items[item_id] = True
            self.account_tree.item(item_id, text="☑")
    
    def _invert_account_selection(self):
        """反选账号"""
        for item_id in self.account_tree.get_children():
            current_state = self.account_checked_items.get(item_id, False)
            new_state = not current_state
            self.account_checked_items[item_id] = new_state
            self.account_tree.item(item_id, text="☑" if new_state else "□")
    
    def _get_selected_user_id(self):
        """获取选中管理员的完整 user_id
        
        Returns:
            tuple: (user_id, user_name) 或 (None, None)
        """
        selection = self.user_tree.selection()
        if not selection:
            return None, None
        
        item = selection[0]
        tags = self.user_tree.item(item, 'tags')
        values = self.user_tree.item(item, 'values')
        
        # tags 中第二个元素是完整的 user_id（第一个是 enabled/disabled）
        user_id = tags[1] if len(tags) > 1 else None
        user_name = values[1] if len(values) > 1 else None
        
        return user_id, user_name
    
    def _add_accounts_to_user(self):
        """为选中的管理员添加账号（从数据库未分配账号中选择）"""
        # 获取选中的管理员
        user_id, user_name = self._get_selected_user_id()
        if not user_id:
            messagebox.showwarning("提示", "请先选择一个管理员")
            self.dialog.lift()
            self.dialog.focus_force()
            return
        
        # 从数据库获取所有账号（优化后的查询应该很快）
        from .local_db import LocalDatabase
        db = LocalDatabase()
        all_summaries = db.get_all_accounts_summary(limit=10000)
        
        # 获取所有已分配的账号
        all_assigned_phones = set()
        for uid in self.user_manager.users.keys():
            all_assigned_phones.update(self.user_manager.get_user_accounts(uid))
        
        # 筛选出未分配的账号
        unassigned_accounts = []
        for summary in all_summaries:
            phone = summary.get('phone', '')
            if phone not in all_assigned_phones:
                unassigned_accounts.append(summary)
        
        if not unassigned_accounts:
            messagebox.showinfo("提示", "数据库中没有未分配的账号")
            self.dialog.lift()
            self.dialog.focus_force()
            return
        
        # 打开选择对话框
        SelectUnassignedAccountsDialog(
            self.dialog, 
            self.user_manager, 
            user_id,
            user_name,
            unassigned_accounts, 
            lambda: self._on_accounts_changed(user_id),
            self.log
        )
    
    def _assign_selected_accounts(self):
        """将选中的账号分配给选中的管理员"""
        # 获取选中的管理员
        user_id, user_name = self._get_selected_user_id()
        if not user_id:
            messagebox.showwarning("提示", "请先选择一个管理员")
            self.dialog.lift()
            self.dialog.focus_force()
            return
        
        # 获取选中的账号
        selected_phones = []
        for item_id in self.account_tree.get_children():
            if self.account_checked_items.get(item_id, False):
                values = self.account_tree.item(item_id, 'values')
                if values:
                    selected_phones.append(values[0])  # 手机号在第一列
        
        if not selected_phones:
            messagebox.showwarning("提示", "请先勾选要分配的账号")
            self.dialog.lift()
            self.dialog.focus_force()
            return
        
        # 确认分配
        result = messagebox.askyesno(
            "确认分配",
            f"确定要将 {len(selected_phones)} 个账号分配给 '{user_name}' 吗？"
        )
        
        if result:
            # 批量分配
            count = self.user_manager.batch_assign_accounts(selected_phones, user_id)
            
            self.log(f"✓ 已为 {count} 个账号分配管理员: {user_name}")
            messagebox.showinfo("成功", f"已成功分配 {count} 个账号给 '{user_name}'")
            
            # 确保焦点返回
            self.dialog.lift()
            self.dialog.focus_force()
            
            # 刷新显示
            self._refresh_user_list()
    
    def _remove_owner_from_selected(self):
        """移除选中账号的管理员"""
        # 获取选中的管理员
        user_id, user_name = self._get_selected_user_id()
        if not user_id:
            messagebox.showwarning("提示", "请先选择一个管理员")
            self.dialog.lift()
            self.dialog.focus_force()
            return
        
        # 获取选中的账号
        selected_phones = []
        selected_items = []
        for item_id in self.account_tree.get_children():
            if self.account_checked_items.get(item_id, False):
                values = self.account_tree.item(item_id, 'values')
                if values:
                    selected_phones.append(values[0])
                    selected_items.append(item_id)
        
        if not selected_phones:
            messagebox.showwarning("提示", "请先勾选要移除的账号")
            self.dialog.lift()
            self.dialog.focus_force()
            return
        
        # 确认移除
        result = messagebox.askyesno(
            "确认移除",
            f"确定要从 '{user_name}' 移除 {len(selected_phones)} 个账号吗？\n\n这不会删除账号，只是取消分配关系。"
        )
        
        if result:
            # 批量移除
            count = 0
            for phone in selected_phones:
                if self.user_manager.unassign_account(phone):
                    count += 1
            
            self.log(f"✓ 已从 '{user_name}' 移除 {count} 个账号")
            messagebox.showinfo("成功", f"已成功移除 {count} 个账号")
            
            # 确保焦点返回
            self.dialog.lift()
            self.dialog.focus_force()
            
            # 刷新显示
            self._on_accounts_changed(user_id)
    
    def _on_accounts_changed(self, user_id: str):
        """当账号分配关系改变时，刷新显示
        
        Args:
            user_id: 管理员的完整 user_id
        """
        # 刷新用户列表（更新账号数量）
        self._refresh_user_list()
        
        # 刷新该用户的账号列表
        self._refresh_user_accounts(user_id)
    
    def _on_user_selected(self, event):
        """当选中用户时，刷新该用户的账号列表"""
        selection = self.user_tree.selection()
        if not selection:
            # 清空账号列表
            for item in self.account_tree.get_children():
                self.account_tree.delete(item)
            return
        
        # 获取选中的用户ID（从 tags 中获取完整的 user_id）
        item = selection[0]
        tags = self.user_tree.item(item, 'tags')
        # tags 中第二个元素是完整的 user_id（第一个是 enabled/disabled）
        user_id = tags[1] if len(tags) > 1 else tags[0]
        
        # 刷新该用户的账号列表
        self._refresh_user_accounts(user_id)
    
    def _add_user(self):
        """添加管理员（从数据库未分配账号中选择）"""
        # 从数据库获取所有账号
        from .local_db import LocalDatabase
        db = LocalDatabase()
        all_summaries = db.get_all_accounts_summary(limit=10000)
        
        # 获取已分配的账号
        assigned_phones = set()
        for uid in self.user_manager.users.keys():
            assigned_phones.update(self.user_manager.get_user_accounts(uid))
        
        # 筛选未分配的账号
        unassigned_accounts = [s for s in all_summaries if s['phone'] not in assigned_phones]
        
        if not unassigned_accounts:
            messagebox.showinfo("提示", "数据库中没有未分配的账号")
            self.dialog.lift()
            self.dialog.focus_force()
            return
        
        # 打开选择账号作为管理员对话框
        SelectAccountAsOwnerDialog(
            self.dialog, 
            self.user_manager, 
            unassigned_accounts, 
            self._refresh_user_list,
            self.log
        )
    
    def _edit_user(self):
        """编辑用户"""
        # 获取选中的管理员
        user_id, _ = self._get_selected_user_id()
        if not user_id:
            messagebox.showwarning("提示", "请先选择要编辑的用户")
            self.dialog.lift()
            self.dialog.focus_force()
            return
        
        user = self.user_manager.get_user(user_id)
        if user:
            UserEditDialog(self.dialog, self.user_manager, user, self._refresh_user_list)
    
    def _delete_user(self):
        """删除用户"""
        # 获取选中的管理员
        user_id, user_name = self._get_selected_user_id()
        if not user_id:
            messagebox.showwarning("提示", "请先选择要删除的用户")
            self.dialog.lift()
            self.dialog.focus_force()
            return
        
        try:
            # 获取账号数量
            account_count = len(self.user_manager.get_user_accounts(user_id))
            
            # 确认删除
            if account_count > 0:
                result = messagebox.askyesno(
                    "确认删除",
                    f"用户 '{user_name}' 有 {account_count} 个账号\n\n删除用户将清除所有账号分配关系\n\n是否继续？"
                )
            else:
                result = messagebox.askyesno(
                    "确认删除",
                    f"确定要删除用户 '{user_name}' 吗？"
                )
            
            if result:
                if self.user_manager.delete_user(user_id):
                    self.log(f"✓ 已删除管理员: {user_name}")
                    
                    # 清空账号列表（因为删除的用户可能正在显示）
                    for item in self.account_tree.get_children():
                        self.account_tree.delete(item)
                    self.account_checked_items = {}
                    
                    # 刷新用户列表
                    self._refresh_user_list()
                    
                    # 显示成功提示
                    messagebox.showinfo("成功", f"已成功删除管理员 '{user_name}'")
                    
                    # 确保焦点返回到用户管理窗口
                    self.dialog.lift()
                    self.dialog.focus_force()
                else:
                    messagebox.showerror("错误", "删除管理员失败")
                    self.dialog.lift()
                    self.dialog.focus_force()
        except Exception as e:
            print(f"删除管理员时出错: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"删除管理员时出错: {e}")
            self.dialog.lift()
            self.dialog.focus_force()
    
    def _batch_add_accounts(self):
        """批量添加账号到账号文件（已集成到主窗口右侧，此方法保留用于兼容性）"""
        # 不再打开新对话框，功能已集成到主窗口右侧
        pass
    
    def _create_batch_add_widgets(self, parent_frame):
        """创建批量添加账号的界面组件（集成到主窗口右侧）
        
        Args:
            parent_frame: 父容器
        """
        # 标题
        right_title = ttk.Label(parent_frame, text="批量添加账号", font=("Microsoft YaHei UI", 12, "bold"), foreground="blue")
        right_title.pack(pady=(0, 10))
        
        # 说明文字
        info_label = ttk.Label(
            parent_frame,
            text="批量添加账号到账号文件\n格式：手机号----密码（每行一个）\n例如：13800138000----password123",
            font=("Microsoft YaHei UI", 9),
            foreground="gray",
            justify=tk.LEFT
        )
        info_label.pack(pady=(0, 10))
        
        # === 管理员选择区域 ===
        owner_frame = ttk.LabelFrame(parent_frame, text="选择管理员（可选）", padding="10")
        owner_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 获取所有启用的用户
        users = self.user_manager.get_all_users()
        enabled_users = [u for u in users if u.enabled]
        
        # 创建下拉选择框
        owner_select_frame = ttk.Frame(owner_frame)
        owner_select_frame.pack(fill=tk.X)
        
        ttk.Label(owner_select_frame, text="管理员:", width=10).pack(side=tk.LEFT, padx=(0, 5))
        
        self.batch_owner_var = tk.StringVar(value="不分配")
        self.batch_owner_combo = ttk.Combobox(
            owner_select_frame,
            textvariable=self.batch_owner_var,
            state='readonly',
            width=25
        )
        
        # 填充用户列表
        self.batch_user_list = []
        owner_options = ["不分配"]
        
        for user in enabled_users:
            display_text = f"{user.user_name} (ID: {user.user_id})"
            owner_options.append(display_text)
            self.batch_user_list.append(user)
        
        self.batch_owner_combo['values'] = owner_options
        self.batch_owner_combo.current(0)  # 默认选择"不分配"
        self.batch_owner_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 提示文字
        hint_label = ttk.Label(
            owner_frame,
            text="💡 提示：选择管理员后，添加的账号将自动分配给该用户",
            font=("Microsoft YaHei UI", 8),
            foreground="gray"
        )
        hint_label.pack(pady=(5, 0))
        
        # === 账号输入区域 ===
        text_frame = ttk.LabelFrame(parent_frame, text="账号列表（已有账号只显示手机号，新增账号格式：手机号----密码）", padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 添加滚动条的文本框
        text_container = ttk.Frame(text_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.batch_accounts_text = tk.Text(text_container, height=15, width=50, yscrollcommand=scrollbar.set)
        self.batch_accounts_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.batch_accounts_text.yview)
        
        # 统计信息
        self.batch_stats_var = tk.StringVar(value="待添加: 0 个账号")
        ttk.Label(parent_frame, textvariable=self.batch_stats_var, foreground="gray").pack(pady=(0, 10))
        
        # 绑定文本变化事件
        self.batch_accounts_text.bind('<KeyRelease>', self._on_batch_text_changed)
        
        # 加载已有账号到文本框
        self._load_existing_accounts_to_batch()
        
        # 按钮区域
        button_frame = ttk.Frame(parent_frame)
        button_frame.pack(fill=tk.X)
        
        # 第一行按钮
        button_row1 = ttk.Frame(button_frame)
        button_row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(button_row1, text="➕ 添加到账号文件", command=self._batch_add_accounts_action, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_row1, text="🗑️ 删除选中账号", command=self._batch_delete_accounts, width=15).pack(side=tk.LEFT, padx=(0, 5))
        
        # 第二行按钮
        button_row2 = ttk.Frame(button_frame)
        button_row2.pack(fill=tk.X)
        
        ttk.Button(button_row2, text="🧹 清空所有账号", command=self._batch_clear_all_accounts, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_row2, text="关闭", command=self.dialog.destroy, width=10).pack(side=tk.RIGHT)
    
    def _on_batch_text_changed(self, event=None):
        """批量添加文本变化时更新统计信息"""
        text = self.batch_accounts_text.get("1.0", tk.END).strip()
        if not text:
            self.batch_stats_var.set("待添加: 0 个账号")
            return
        
        # 简单统计行数
        lines = [line.strip() for line in text.split('\n') if line.strip() and not line.strip().startswith('#')]
        self.batch_stats_var.set(f"当前: {len(lines)} 行")
    
    def _load_existing_accounts_to_batch(self):
        """加载已有账号到批量添加文本框（只显示手机号）"""
        try:
            # 获取账号文件路径
            from .config import ConfigLoader
            config = ConfigLoader().load()
            accounts_file = config.accounts_file
            
            if not accounts_file:
                return
            
            # 使用加密账号文件管理器读取
            try:
                from .encrypted_accounts_file import EncryptedAccountsFile
            except ImportError:
                try:
                    from encrypted_accounts_file import EncryptedAccountsFile
                except ImportError:
                    from src.encrypted_accounts_file import EncryptedAccountsFile
            
            encrypted_file = EncryptedAccountsFile(accounts_file)
            accounts_list = encrypted_file.read_accounts()
            
            if accounts_list:
                # 只显示手机号（不显示密码）
                lines = [phone for phone, password in accounts_list]
                text = '\n'.join(lines)
                
                # 插入到文本框
                self.batch_accounts_text.delete("1.0", tk.END)
                self.batch_accounts_text.insert("1.0", text)
                
                # 更新统计
                self._on_batch_text_changed()
                
                self.log(f"✓ 已加载 {len(accounts_list)} 个已有账号")
        except Exception as e:
            # 如果加载失败，不影响使用（可能是新文件）
            print(f"[批量添加] 加载已有账号失败: {e}")
    
    def _batch_add_accounts_action(self):
        """批量添加账号到账号文件的实际操作"""
        # 获取账号文本
        text = self.batch_accounts_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请输入账号信息", parent=self.dialog)
            return
        
        # 获取账号文件路径（从配置中读取）
        from .config import ConfigLoader
        config = ConfigLoader().load()
        accounts_file = config.accounts_file
        
        if not accounts_file:
            messagebox.showerror("错误", "未配置账号文件路径", parent=self.dialog)
            return
        
        # 解析账号（使用多线程加速）
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        lines = text.split('\n')
        new_accounts = []
        invalid_lines = []
        parse_lock = threading.Lock()
        
        def parse_line(line_data):
            """解析单行账号（线程安全）"""
            line_num, line = line_data
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                return None
            
            # 如果只有手机号（已有账号），跳过
            if line.isdigit() and len(line) == 11:
                return None
            
            # 检查格式
            if '----' not in line:
                return ('error', line_num, "格式错误（缺少----分隔符）")
            
            parts = line.split('----')
            
            # 标准格式：手机号----密码（每行一个账号）
            if len(parts) != 2:
                return ('error', line_num, "格式错误（应为：手机号----密码）")
            
            phone = parts[0].strip()
            password = parts[1].strip()
            
            if not phone.isdigit() or len(phone) != 11:
                return ('error', line_num, "手机号格式错误")
            
            if not password:
                return ('error', line_num, "密码为空")
            
            return ('success', line_num, phone, password)
        
        # 使用线程池并行解析（最多8个线程）
        max_workers = min(8, len(lines))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有解析任务
            line_data = [(i+1, line) for i, line in enumerate(lines)]
            futures = {executor.submit(parse_line, data): data for data in line_data}
            
            # 收集结果
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                
                if result[0] == 'error':
                    _, line_num, error_msg = result
                    with parse_lock:
                        invalid_lines.append(f"第{line_num}行：{error_msg}")
                elif result[0] == 'success':
                    _, line_num, phone, password = result
                    with parse_lock:
                        new_accounts.append((phone, password, line_num))
        
        if not new_accounts:
            if invalid_lines:
                messagebox.showerror("错误", "没有有效的账号\n\n" + "\n".join(invalid_lines[:5]), parent=self.dialog)
            return
        
        # 去重：检查输入中的重复账号
        seen_phones = {}  # {phone: (password, line_num)}
        unique_accounts = []
        input_duplicates = []  # 输入中的重复账号
        
        for phone, password, line_num in new_accounts:
            if phone in seen_phones:
                # 发现重复
                existing_password, existing_line = seen_phones[phone]
                if existing_password == password:
                    # 密码相同，记录为重复
                    input_duplicates.append(f"第{line_num}行：手机号 {phone} 重复（与第{existing_line}行相同）")
                else:
                    # 密码不同，使用最后出现的密码
                    input_duplicates.append(f"第{line_num}行：手机号 {phone} 重复但密码不同（将使用此密码）")
                    # 更新为最新的密码
                    seen_phones[phone] = (password, line_num)
                    # 从unique_accounts中移除旧的，添加新的
                    unique_accounts = [(p, pw) for p, pw in unique_accounts if p != phone]
                    unique_accounts.append((phone, password))
            else:
                seen_phones[phone] = (password, line_num)
                unique_accounts.append((phone, password))
        
        # 显示无效行和重复警告
        warnings = []
        if invalid_lines:
            warnings.append(f"格式错误: {len(invalid_lines)} 行")
        if input_duplicates:
            warnings.append(f"输入重复: {len(input_duplicates)} 行")
        
        if warnings:
            warning_message = "发现以下问题：\n\n"
            
            if invalid_lines:
                warning_message += "【格式错误】\n" + "\n".join(invalid_lines[:3])
                if len(invalid_lines) > 3:
                    warning_message += f"\n... 还有 {len(invalid_lines) - 3} 行\n"
                warning_message += "\n"
            
            if input_duplicates:
                warning_message += "【输入重复】\n" + "\n".join(input_duplicates[:3])
                if len(input_duplicates) > 3:
                    warning_message += f"\n... 还有 {len(input_duplicates) - 3} 行\n"
                warning_message += "\n"
            
            warning_message += f"是否继续添加 {len(unique_accounts)} 个有效账号？"
            
            result = messagebox.askyesno("警告", warning_message, parent=self.dialog)
            if not result:
                return
        
        # 使用去重后的账号列表
        new_accounts = unique_accounts
        
        # 使用加密账号文件管理器读取现有账号（检查重复）
        from pathlib import Path
        try:
            from .encrypted_accounts_file import EncryptedAccountsFile
        except ImportError:
            try:
                from encrypted_accounts_file import EncryptedAccountsFile
            except ImportError:
                from src.encrypted_accounts_file import EncryptedAccountsFile
        
        encrypted_file = EncryptedAccountsFile(accounts_file)
        existing_accounts_dict = {}  # {phone: (password, owner)}
        
        try:
            existing_accounts_list = encrypted_file.read_accounts()
            for item in existing_accounts_list:
                if len(item) >= 2:
                    phone = item[0]
                    password = item[1]
                    owner = item[2] if len(item) > 2 else None
                    existing_accounts_dict[phone] = (password, owner)
        except Exception as e:
            print(f"[批量添加] 读取账号文件失败: {e}")
            # 如果读取失败，继续（可能是新文件）
        
        # 分类账号：新增、更新、重复（密码相同）
        accounts_to_add = []  # 新增的账号
        accounts_to_update = []  # 需要更新密码的账号
        duplicate_accounts = []  # 完全重复的账号（手机号和密码都相同）
        
        for phone, password in new_accounts:
            if phone in existing_accounts_dict:
                existing_password, existing_owner = existing_accounts_dict[phone]
                if existing_password == password:
                    # 密码相同，完全重复
                    duplicate_accounts.append(phone)
                else:
                    # 密码不同，需要更新
                    accounts_to_update.append((phone, password, existing_owner))
            else:
                # 新账号
                accounts_to_add.append((phone, password))
        
        # 使用加密账号文件管理器处理账号
        try:
            # 确保目录存在
            Path(accounts_file).parent.mkdir(parents=True, exist_ok=True)
            
            # 处理更新密码的账号（先删除旧的，再添加新的）
            if accounts_to_update:
                # 读取所有现有账号
                all_accounts = encrypted_file.read_accounts()
                
                # 创建更新映射
                update_map = {phone: (password, owner) for phone, password, owner in accounts_to_update}
                
                # 更新账号列表
                updated_accounts = []
                for item in all_accounts:
                    phone = item[0]
                    if phone in update_map:
                        # 更新密码，保留管理员
                        new_password, existing_owner = update_map[phone]
                        if existing_owner:
                            updated_accounts.append((phone, new_password, existing_owner))
                        else:
                            updated_accounts.append((phone, new_password))
                    else:
                        # 保持不变
                        updated_accounts.append(item)
                
                # 写回文件
                if encrypted_file.write_accounts(updated_accounts):
                    self.log(f"✓ 成功更新 {len(accounts_to_update)} 个账号的密码（已加密）")
                else:
                    messagebox.showerror("错误", "更新账号密码失败", parent=self.dialog)
                    return
            
            # 追加新账号（自动加密）
            if accounts_to_add:
                if encrypted_file.append_accounts(accounts_to_add):
                    self.log(f"✓ 成功添加 {len(accounts_to_add)} 个账号到账号文件（已加密）")
                else:
                    messagebox.showerror("错误", "写入账号文件失败", parent=self.dialog)
                    return
            
            if duplicate_accounts:
                self.log(f"⚠️ 跳过 {len(duplicate_accounts)} 个完全重复的账号")
            
            # === 自动分配管理员 ===
            owner_selection = self.batch_owner_var.get()
            assigned_count = 0
            
            if owner_selection != "不分配":
                # 获取选中的用户
                selected_user = None
                for user in self.batch_user_list:
                    display_text = f"{user.user_name} (ID: {user.user_id})"
                    if display_text == owner_selection:
                        selected_user = user
                        break
                
                if selected_user:
                    # 只为新增的账号分配管理员（更新密码的账号保留原管理员）
                    phones_to_assign = [phone for phone, _ in accounts_to_add]
                    if phones_to_assign:
                        assigned_count = self.user_manager.batch_assign_accounts(phones_to_assign, selected_user.user_id)
                        self.log(f"✓ 已为 {assigned_count} 个新账号分配管理员: {selected_user.user_name}")
            
            # 显示成功消息
            success_message = "操作完成\n\n"
            
            if accounts_to_add:
                success_message += f"新增账号: {len(accounts_to_add)} 个\n"
            
            if accounts_to_update:
                success_message += f"更新密码: {len(accounts_to_update)} 个\n"
            
            if duplicate_accounts:
                success_message += f"跳过重复: {len(duplicate_accounts)} 个\n"
            
            if assigned_count > 0:
                success_message += f"\n已为新账号分配管理员: {owner_selection}\n"
            
            # 确保对话框置顶
            self.dialog.lift()
            self.dialog.focus_force()
            messagebox.showinfo("成功", success_message, parent=self.dialog)
            
            # 刷新用户列表（更新账号数量）
            self._refresh_user_list()
            
        except Exception as e:
            messagebox.showerror("错误", f"写入账号文件失败: {e}", parent=self.dialog)
    
    def _batch_delete_accounts(self):
        """删除选中的账号"""
        # 获取文本框中的内容
        text = self.batch_accounts_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请先输入要删除的账号", parent=self.dialog)
            return
        
        # 解析要删除的账号
        phones_to_delete = []
        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 如果只有手机号（已有账号）
            if line.isdigit() and len(line) == 11:
                phones_to_delete.append(line)
                continue
            
            if '----' not in line:
                continue
            
            # 标准格式
            parts = line.split('----', 1)
            phone = parts[0].strip()
            if phone.isdigit() and len(phone) == 11:
                phones_to_delete.append(phone)
        
        if not phones_to_delete:
            messagebox.showwarning("提示", "没有有效的账号可删除", parent=self.dialog)
            return
        
        # 确认删除
        result = messagebox.askyesno(
            "确认删除",
            f"确定要删除 {len(phones_to_delete)} 个账号吗？\n\n" +
            "删除操作将：\n" +
            "1. 从账号文件中删除\n" +
            "2. 从数据库中删除所有记录\n" +
            "3. 删除登录缓存文件\n" +
            "4. 删除账号信息缓存\n\n" +
            "此操作不可恢复！",
            parent=self.dialog
        )
        
        if not result:
            return
        
        # 获取账号文件路径
        from .config import ConfigLoader
        config = ConfigLoader().load()
        accounts_file = config.accounts_file
        
        if not accounts_file:
            messagebox.showerror("错误", "未配置账号文件路径", parent=self.dialog)
            return
        
        from pathlib import Path
        
        # 使用加密账号文件管理器
        try:
            from .encrypted_accounts_file import EncryptedAccountsFile
        except ImportError:
            try:
                from encrypted_accounts_file import EncryptedAccountsFile
            except ImportError:
                from src.encrypted_accounts_file import EncryptedAccountsFile
        
        encrypted_file = EncryptedAccountsFile(accounts_file)
        
        # 删除账号（自动加密）
        try:
            if encrypted_file.delete_accounts(phones_to_delete):
                self.log(f"✓ 已从账号文件删除 {len(phones_to_delete)} 个账号（已加密）")
            else:
                messagebox.showerror("错误", "删除账号失败", parent=self.dialog)
                return
        except Exception as e:
            messagebox.showerror("错误", f"删除账号失败: {e}", parent=self.dialog)
            return
        
        # 从数据库删除
        from .local_db import LocalDatabase
        db = LocalDatabase()
        deleted_db_count = 0
        
        for phone in phones_to_delete:
            try:
                # 删除历史记录
                conn = db._get_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM history_records WHERE phone = ?", (phone,))
                conn.commit()
                conn.close()
                deleted_db_count += 1
            except Exception as e:
                self.log(f"⚠️ 删除数据库记录失败 ({phone}): {e}")
        
        self.log(f"✓ 已从数据库删除 {deleted_db_count} 个账号的记录")
        
        # 删除登录缓存
        from .login_cache_manager import LoginCacheManager
        from .adb_bridge import ADBBridge
        adb = ADBBridge()
        cache_manager = LoginCacheManager(adb)
        deleted_cache_count = 0
        
        for phone in phones_to_delete:
            try:
                if cache_manager.delete_cache(phone):
                    deleted_cache_count += 1
            except Exception as e:
                self.log(f"⚠️ 删除登录缓存失败 ({phone}): {e}")
        
        self.log(f"✓ 已删除 {deleted_cache_count} 个账号的登录缓存")
        
        # 删除账号信息缓存
        from .account_cache import get_account_cache
        account_cache = get_account_cache()
        deleted_account_cache_count = 0
        
        for phone in phones_to_delete:
            try:
                account_cache.clear(phone)
                deleted_account_cache_count += 1
            except Exception as e:
                self.log(f"⚠️ 删除账号缓存失败 ({phone}): {e}")
        
        self.log(f"✓ 已删除 {deleted_account_cache_count} 个账号的信息缓存")
        
        # 从用户管理器中移除账号分配
        removed_assignments = 0
        for phone in phones_to_delete:
            try:
                if self.user_manager.unassign_account(phone):
                    removed_assignments += 1
            except Exception as e:
                self.log(f"⚠️ 移除账号分配失败 ({phone}): {e}")
        
        if removed_assignments > 0:
            self.log(f"✓ 已移除 {removed_assignments} 个账号的管理员分配")
        
        # 显示成功消息
        # 确保对话框置顶
        self.dialog.lift()
        self.dialog.focus_force()
        messagebox.showinfo(
            "删除成功",
            f"已成功删除 {len(phones_to_delete)} 个账号\n\n" +
            f"- 账号文件: {len(phones_to_delete)} 个\n" +
            f"- 数据库记录: {deleted_db_count} 个\n" +
            f"- 登录缓存: {deleted_cache_count} 个\n" +
            f"- 账号缓存: {deleted_account_cache_count} 个\n" +
            f"- 管理员分配: {removed_assignments} 个",
            parent=self.dialog
        )
        
        # 清空文本框
        self.batch_accounts_text.delete("1.0", tk.END)
        self._on_batch_text_changed()
        
        # 刷新用户列表
        self._refresh_user_list()
    
    def _batch_clear_all_accounts(self):
        """清空所有账号"""
        # 确认清空
        result = messagebox.askyesno(
            "确认清空",
            "确定要清空所有账号吗？\n\n" +
            "此操作将删除账号文件中的所有账号！\n\n" +
            "此操作不可恢复！",
            icon='warning',
            parent=self.dialog
        )
        
        if not result:
            return
        
        # 询问是否清理缓存
        clear_cache = messagebox.askyesno(
            "清理缓存",
            "是否同时清理所有账号的缓存？\n\n" +
            "包括：\n" +
            "- 登录缓存文件\n" +
            "- 账号信息缓存\n" +
            "- 数据库记录\n\n" +
            "建议选择'是'以彻底清理",
            parent=self.dialog
        )
        
        # 获取账号文件路径
        from .config import ConfigLoader
        config = ConfigLoader().load()
        accounts_file = config.accounts_file
        
        if not accounts_file:
            messagebox.showerror("错误", "未配置账号文件路径", parent=self.dialog)
            return
        
        from pathlib import Path
        
        # 使用加密账号文件管理器
        try:
            from .encrypted_accounts_file import EncryptedAccountsFile
        except ImportError:
            try:
                from encrypted_accounts_file import EncryptedAccountsFile
            except ImportError:
                from src.encrypted_accounts_file import EncryptedAccountsFile
        
        encrypted_file = EncryptedAccountsFile(accounts_file)
        
        # 如果需要清理缓存，先读取所有账号
        all_phones = []
        if clear_cache:
            try:
                accounts_list = encrypted_file.read_accounts()
                for item in accounts_list:
                    if len(item) >= 1:
                        all_phones.append(item[0])
            except Exception as e:
                print(f"[清空账号] 读取账号失败: {e}")
        
        # 清空账号文件（自动加密）
        try:
            if encrypted_file.clear_accounts():
                self.log(f"✓ 已清空账号文件（已加密）")
            else:
                messagebox.showerror("错误", "清空账号文件失败", parent=self.dialog)
                return
        except Exception as e:
            messagebox.showerror("错误", f"清空账号文件失败: {e}", parent=self.dialog)
            return
        
        # 清理缓存
        if clear_cache and all_phones:
            self.log(f"正在清理 {len(all_phones)} 个账号的缓存...")
            
            # 删除数据库记录
            from .local_db import LocalDatabase
            db = LocalDatabase()
            deleted_db_count = 0
            
            for phone in all_phones:
                try:
                    conn = db._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM history_records WHERE phone = ?", (phone,))
                    conn.commit()
                    conn.close()
                    deleted_db_count += 1
                except Exception as e:
                    self.log(f"⚠️ 删除数据库记录失败 ({phone}): {e}")
            
            self.log(f"✓ 已从数据库删除 {deleted_db_count} 个账号的记录")
            
            # 删除登录缓存
            from .login_cache_manager import LoginCacheManager
            from .adb_bridge import ADBBridge
            adb = ADBBridge()
            cache_manager = LoginCacheManager(adb)
            deleted_cache_count = 0
            
            for phone in all_phones:
                try:
                    if cache_manager.delete_cache(phone):
                        deleted_cache_count += 1
                except Exception as e:
                    self.log(f"⚠️ 删除登录缓存失败 ({phone}): {e}")
            
            self.log(f"✓ 已删除 {deleted_cache_count} 个账号的登录缓存")
            
            # 删除账号信息缓存
            from .account_cache import get_account_cache
            account_cache = get_account_cache()
            
            try:
                account_cache.clear()  # 清空所有缓存
                self.log(f"✓ 已清空所有账号信息缓存")
            except Exception as e:
                self.log(f"⚠️ 清空账号缓存失败: {e}")
            
            # 清空所有用户的账号分配
            try:
                for user in self.user_manager.get_all_users():
                    assigned_phones = self.user_manager.get_user_accounts(user.user_id)
                    for phone in assigned_phones:
                        self.user_manager.unassign_account(phone)
                self.log(f"✓ 已清空所有管理员的账号分配")
            except Exception as e:
                self.log(f"⚠️ 清空账号分配失败: {e}")
        
        # 显示成功消息
        # 确保对话框置顶
        self.dialog.lift()
        self.dialog.focus_force()
        if clear_cache:
            messagebox.showinfo(
                "清空成功",
                f"已成功清空所有账号\n\n" +
                f"- 账号文件: 已清空\n" +
                f"- 数据库记录: 已删除\n" +
                f"- 登录缓存: 已删除\n" +
                f"- 账号缓存: 已清空\n" +
                f"- 管理员分配: 已清空",
                parent=self.dialog
            )
        else:
            messagebox.showinfo(
                "清空成功",
                "已成功清空账号文件\n\n" +
                "缓存文件未清理，如需清理请重新执行清空操作",
                parent=self.dialog
            )
        
        # 清空文本框
        self.batch_accounts_text.delete("1.0", tk.END)
        self._on_batch_text_changed()
        
        # 刷新用户列表
        self._refresh_user_list()
    
    def _assign_unassigned_accounts(self):
        """分配未分配的账号"""
        # 获取账号文件中的所有账号
        from .config import ConfigLoader
        config = ConfigLoader().load()
        accounts_file = config.accounts_file
        
        if not accounts_file:
            messagebox.showerror("错误", "未配置账号文件路径")
            return
        
        from pathlib import Path
        if not Path(accounts_file).exists():
            messagebox.showerror("错误", "账号文件不存在")
            return
        
        # 读取所有账号
        all_phones = []
        try:
            with open(accounts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '----' in line:
                        phone = line.split('----')[0].strip()
                        all_phones.append(phone)
        except Exception as e:
            messagebox.showerror("错误", f"读取账号文件失败: {e}")
            return
        
        if not all_phones:
            messagebox.showinfo("提示", "账号文件中没有账号")
            return
        
        # 获取未分配的账号
        unassigned_phones = self.user_manager.get_unassigned_accounts(all_phones)
        
        if not unassigned_phones:
            messagebox.showinfo("提示", "所有账号都已分配")
            return
        
        # 打开未分配账号对话框
        UnassignedAccountsDialog(self.dialog, self.user_manager, unassigned_phones, self._refresh_user_list, self.log)


class SelectAccountAsOwnerDialog:
    """从未分配账号中选择作为管理员对话框"""
    
    def __init__(self, parent, user_manager: UserManager, unassigned_accounts: List[Dict[str, Any]], 
                 callback: Callable, log_callback: Callable):
        """初始化选择账号作为管理员对话框
        
        Args:
            parent: 父窗口
            user_manager: 用户管理器
            unassigned_accounts: 未分配的账号列表
            callback: 完成后的回调函数
            log_callback: 日志回调函数
        """
        self.parent = parent
        self.user_manager = user_manager
        self.unassigned_accounts = unassigned_accounts
        self.callback = callback
        self.log = log_callback
        
        # 创建窗口
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"添加管理员 - 从未分配账号中选择")
        self.dialog.geometry("700x600")
        self.dialog.resizable(True, True)
        
        # 居中显示
        self._center_window()
        
        # 创建界面
        self._create_widgets()
    
    def _center_window(self):
        """将窗口居中显示"""
        self.dialog.update_idletasks()
        width = 700
        height = 600
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 说明文字
        info_label = ttk.Label(
            main_frame,
            text=f"从数据库未分配账号中选择一个账号作为管理员\n\n共 {len(self.unassigned_accounts)} 个未分配账号",
            font=("Microsoft YaHei UI", 10, "bold"),
            foreground="blue"
        )
        info_label.pack(pady=(0, 15))
        
        # 账号列表区域
        list_frame = ttk.LabelFrame(main_frame, text="未分配账号列表", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 创建Treeview
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("phone", "nickname", "user_id", "latest_date")
        self.account_tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        
        column_config = {
            "phone": ("手机号", 120),
            "nickname": ("昵称", 150),
            "user_id": ("用户ID", 120),
            "latest_date": ("最新记录日期", 120)
        }
        
        for col, (heading, width) in column_config.items():
            self.account_tree.heading(col, text=heading)
            self.account_tree.column(col, width=width, anchor=tk.CENTER)
        
        # 添加滚动条
        scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.account_tree.yview)
        scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.account_tree.xview)
        self.account_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        self.account_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # 填充账号列表
        for account in self.unassigned_accounts:
            phone = account.get('phone', '')
            nickname = account.get('nickname', '-') or '-'
            user_id = account.get('user_id', '-') or '-'
            latest_date = account.get('latest_date', '-') or '-'
            
            values = (phone, nickname, user_id, latest_date)
            self.account_tree.insert("", tk.END, values=values)
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="添加为管理员", command=self._add_as_owner, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="取消", command=self.dialog.destroy, width=10).pack(side=tk.LEFT)
    
    def _add_as_owner(self):
        """将选中账号添加为管理员"""
        # 获取选中的账号
        selection = self.account_tree.selection()
        if not selection:
            messagebox.showwarning("提示", "请选择一个账号")
            return
        
        item = selection[0]
        values = self.account_tree.item(item, 'values')
        phone = values[0]
        nickname = values[1]
        account_user_id = values[2]
        
        # 使用账号的昵称作为管理员名称
        owner_name = nickname if nickname and nickname != '-' else phone
        
        # 使用账号的用户ID作为转账收款人
        recipient = account_user_id if account_user_id and account_user_id != '-' else phone
        
        # 直接使用账号的用户ID作为管理员ID
        # 如果用户ID无效，使用手机号作为后备方案
        if account_user_id and account_user_id != '-':
            user_id = f"user_{account_user_id}"
        else:
            # 如果没有用户ID，使用手机号的哈希值
            import hashlib
            user_id = f"user_{hashlib.md5(phone.encode()).hexdigest()[:8]}"
        
        # 检查用户ID是否已存在
        if self.user_manager.get_user(user_id):
            messagebox.showerror("错误", f"该账号已被添加为管理员")
            return
        
        # 创建用户对象
        user = User(
            user_id=user_id,
            user_name=owner_name,
            transfer_recipients=[recipient],
            description=f"账号: {phone}, 用户ID: {account_user_id}",
            enabled=True
        )
        
        # 添加用户
        if self.user_manager.add_user(user):
            # 将该账号分配给新创建的管理员
            self.user_manager.batch_assign_accounts([phone], user_id)
            
            self.log(f"✓ 已添加管理员: {owner_name}")
            self.log(f"  - 手机号: {phone}")
            self.log(f"  - 用户ID: {account_user_id}")
            self.log(f"  - 转账收款人: {recipient}")
            messagebox.showinfo("成功", f"已成功添加管理员\n\n管理员: {owner_name}\n手机号: {phone}\n用户ID: {account_user_id}\n转账收款人: {recipient}")
            
            # 刷新父窗口
            self.callback()
            
            # 关闭对话框
            self.dialog.destroy()
        else:
            messagebox.showerror("错误", "添加管理员失败")


class UserEditDialog:
    """用户编辑对话框"""
    
    def __init__(self, parent, user_manager: UserManager, user: Optional[User], callback: Callable):
        """初始化用户编辑对话框
        
        Args:
            parent: 父窗口
            user_manager: 用户管理器
            user: 用户对象（None表示新建）
            callback: 完成后的回调函数
        """
        self.parent = parent
        self.user_manager = user_manager
        self.user = user
        self.callback = callback
        self.is_new = (user is None)
        
        # 创建窗口
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("添加用户" if self.is_new else "编辑用户")
        self.dialog.geometry("500x450")
        self.dialog.resizable(False, False)
        
        # 居中显示
        self._center_window()
        
        # 创建界面
        self._create_widgets()
        
        # 如果是编辑模式，填充数据
        if not self.is_new:
            self._load_user_data()
    
    def _center_window(self):
        """将窗口居中显示"""
        self.dialog.update_idletasks()
        width = 500
        height = 450
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 用户ID
        row1 = ttk.Frame(main_frame)
        row1.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(row1, text="用户ID:", width=15).pack(side=tk.LEFT)
        self.user_id_var = tk.StringVar()
        user_id_entry = ttk.Entry(row1, textvariable=self.user_id_var)
        user_id_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if not self.is_new:
            user_id_entry.config(state='readonly')  # 编辑模式下ID不可修改
        
        # 用户名称
        row2 = ttk.Frame(main_frame)
        row2.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(row2, text="用户名称:", width=15).pack(side=tk.LEFT)
        self.user_name_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.user_name_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 转账收款人（多个）
        row3 = ttk.Frame(main_frame)
        row3.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(row3, text="转账收款人:", width=15).pack(side=tk.LEFT, anchor=tk.N)
        
        recipients_frame = ttk.Frame(row3)
        recipients_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 收款人列表文本框
        self.recipients_text = tk.Text(recipients_frame, height=4, width=40)
        self.recipients_text.pack(fill=tk.BOTH, expand=True)
        
        # 提示文字
        hint_label = ttk.Label(recipients_frame, text="（每行一个手机号，支持多个收款人）", foreground="gray", font=("Microsoft YaHei UI", 8))
        hint_label.pack(anchor=tk.W)
        
        # 备注说明
        row4 = ttk.Frame(main_frame)
        row4.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(row4, text="备注说明:", width=15).pack(side=tk.LEFT, anchor=tk.N)
        self.description_text = tk.Text(row4, height=4, width=40)
        self.description_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 启用状态
        row5 = ttk.Frame(main_frame)
        row5.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(row5, text="状态:", width=15).pack(side=tk.LEFT)
        self.enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row5, text="启用", variable=self.enabled_var).pack(side=tk.LEFT)
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="保存", command=self._save, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="取消", command=self.dialog.destroy, width=10).pack(side=tk.LEFT)
    
    def _load_user_data(self):
        """加载用户数据"""
        if self.user:
            self.user_id_var.set(self.user.user_id)
            self.user_name_var.set(self.user.user_name)
            # 加载多个收款人（每行一个）
            recipients_text = "\n".join(self.user.transfer_recipients)
            self.recipients_text.insert("1.0", recipients_text)
            self.description_text.insert("1.0", self.user.description)
            self.enabled_var.set(self.user.enabled)
    
    def _save(self):
        """保存用户"""
        # 验证输入
        user_id = self.user_id_var.get().strip()
        user_name = self.user_name_var.get().strip()
        recipients_text = self.recipients_text.get("1.0", tk.END).strip()
        description = self.description_text.get("1.0", tk.END).strip()
        enabled = self.enabled_var.get()
        
        if not user_id:
            messagebox.showerror("错误", "请输入用户ID")
            return
        
        if not user_name:
            messagebox.showerror("错误", "请输入用户名称")
            return
        
        if not recipients_text:
            messagebox.showerror("错误", "请至少输入一个转账收款人手机号")
            return
        
        # 解析收款人列表
        transfer_recipients = []
        invalid_lines = []
        
        for line_num, line in enumerate(recipients_text.split('\n'), 1):
            phone = line.strip()
            if not phone:
                continue
            
            # 验证手机号格式
            if not phone.isdigit() or len(phone) != 11:
                invalid_lines.append(f"第{line_num}行：{phone}")
                continue
            
            transfer_recipients.append(phone)
        
        if not transfer_recipients:
            error_msg = "没有有效的收款人手机号"
            if invalid_lines:
                error_msg += "\n\n格式错误的手机号：\n" + "\n".join(invalid_lines[:5])
            messagebox.showerror("错误", error_msg)
            return
        
        if invalid_lines:
            result = messagebox.askyesno(
                "警告",
                f"发现 {len(invalid_lines)} 个格式错误的手机号：\n\n" + "\n".join(invalid_lines[:5]) +
                (f"\n... 还有 {len(invalid_lines) - 5} 个" if len(invalid_lines) > 5 else "") +
                f"\n\n是否继续保存 {len(transfer_recipients)} 个有效收款人？"
            )
            if not result:
                return
        
        # 创建用户对象
        user = User(
            user_id=user_id,
            user_name=user_name,
            transfer_recipients=transfer_recipients,
            description=description,
            enabled=enabled
        )
        
        # 保存
        if self.is_new:
            if self.user_manager.add_user(user):
                messagebox.showinfo("成功", f"已添加用户: {user_name}\n收款人数量: {len(transfer_recipients)}")
                self.callback()
                self.dialog.destroy()
            else:
                messagebox.showerror("错误", "添加用户失败（用户ID可能已存在）")
        else:
            if self.user_manager.update_user(user):
                messagebox.showinfo("成功", f"已更新用户: {user_name}\n收款人数量: {len(transfer_recipients)}")
                self.callback()
                self.dialog.destroy()
            else:
                messagebox.showerror("错误", "更新用户失败")


class QuickAssignOwnerDialog:
    """快速分配管理员对话框（从主界面调用）"""
    
    def __init__(self, parent, phones: List[str], callback: Callable):
        """初始化快速分配对话框
        
        Args:
            parent: 父窗口
            phones: 要分配的手机号列表
            callback: 完成后的回调函数
        """
        self.parent = parent
        self.phones = phones
        self.callback = callback
        
        # 创建用户管理器
        self.user_manager = UserManager()
        
        # 创建窗口
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"分配管理员 - 已选择 {len(phones)} 个账号")
        self.dialog.geometry("500x400")
        self.dialog.resizable(False, False)
        
        # 居中显示
        self._center_window()
        
        # 创建界面
        self._create_widgets()
    
    def _center_window(self):
        """将窗口居中显示"""
        self.dialog.update_idletasks()
        width = 500
        height = 400
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 说明文字
        info_label = ttk.Label(
            main_frame,
            text=f"为 {len(self.phones)} 个账号分配管理员",
            font=("Microsoft YaHei UI", 10, "bold"),
            foreground="blue"
        )
        info_label.pack(pady=(0, 15))
        
        # 用户选择
        select_frame = ttk.LabelFrame(main_frame, text="选择管理员", padding="10")
        select_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 获取所有用户
        users = self.user_manager.get_all_users()
        
        if not users:
            ttk.Label(
                select_frame,
                text="暂无用户，请先在用户管理中添加用户",
                foreground="red"
            ).pack(pady=20)
            
            ttk.Button(main_frame, text="关闭", command=self.dialog.destroy, width=10).pack()
            return
        
        # 用户列表（使用Listbox）
        listbox_frame = ttk.Frame(select_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        self.user_listbox = tk.Listbox(
            listbox_frame,
            yscrollcommand=scrollbar.set,
            font=("Microsoft YaHei UI", 10),
            height=10
        )
        scrollbar.config(command=self.user_listbox.yview)
        
        self.user_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 填充用户列表
        self.user_list = []
        for user in users:
            if user.enabled:
                display_text = f"{user.user_name} (ID: {user.user_id})"
                self.user_listbox.insert(tk.END, display_text)
                self.user_list.append(user)
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="分配", command=self._assign, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="取消", command=self.dialog.destroy, width=10).pack(side=tk.LEFT)
    
    def _assign(self):
        """执行分配"""
        # 获取选中的用户
        selection = self.user_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请选择一个用户")
            return
        
        user = self.user_list[selection[0]]
        
        # 批量分配
        count = self.user_manager.batch_assign_accounts(self.phones, user.user_id)
        
        messagebox.showinfo("成功", f"已为 {count} 个账号分配管理员: {user.user_name}")
        self.callback()
        self.dialog.destroy()


class AccountAssignDialog:
    """账号分配对话框"""
    
    def __init__(self, parent, user_manager: UserManager, user: User, callback: Callable):
        """初始化账号分配对话框
        
        Args:
            parent: 父窗口
            user_manager: 用户管理器
            user: 用户对象
            callback: 完成后的回调函数
        """
        self.parent = parent
        self.user_manager = user_manager
        self.user = user
        self.callback = callback
        
        # 创建窗口
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"分配账号 - {user.user_name}")
        self.dialog.geometry("600x500")
        self.dialog.resizable(True, True)
        
        # 居中显示
        self._center_window()
        
        # 创建界面
        self._create_widgets()
    
    def _center_window(self):
        """将窗口居中显示"""
        self.dialog.update_idletasks()
        width = 600
        height = 500
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 说明文字
        info_label = ttk.Label(
            main_frame,
            text=f"为用户 '{self.user.user_name}' 分配账号\n每行输入一个手机号",
            foreground="blue"
        )
        info_label.pack(pady=(0, 10))
        
        # 文本框
        text_frame = ttk.LabelFrame(main_frame, text="手机号列表（每行一个）", padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.phones_text = tk.Text(text_frame, height=20, width=50)
        self.phones_text.pack(fill=tk.BOTH, expand=True)
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="分配", command=self._assign, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="取消", command=self.dialog.destroy, width=10).pack(side=tk.LEFT)
    
    def _assign(self):
        """执行分配"""
        # 获取手机号列表
        text = self.phones_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请输入手机号")
            return
        
        # 解析手机号
        phones = []
        for line in text.split('\n'):
            phone = line.strip()
            if phone and phone.isdigit() and len(phone) == 11:
                phones.append(phone)
        
        if not phones:
            messagebox.showerror("错误", "没有有效的手机号")
            return
        
        # 批量分配
        count = self.user_manager.batch_assign_accounts(phones, self.user.user_id)
        
        messagebox.showinfo("成功", f"已为用户 '{self.user.user_name}' 分配 {count} 个账号")
        self.callback()
        self.dialog.destroy()



class BatchAddAccountsDialog:
    """批量添加账号对话框"""
    
    def __init__(self, parent, log_callback: Callable, user_manager: UserManager = None, refresh_callback: Callable = None):
        """初始化批量添加账号对话框
        
        Args:
            parent: 父窗口
            log_callback: 日志回调函数
            user_manager: 用户管理器（可选）
            refresh_callback: 刷新主界面回调函数（可选）
        """
        self.parent = parent
        self.log = log_callback
        self.user_manager = user_manager if user_manager else UserManager()
        self.refresh_callback = refresh_callback
        
        # 创建窗口
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("批量添加账号")
        self.dialog.geometry("700x700")
        self.dialog.resizable(True, True)
        
        # 先隐藏窗口，避免白屏
        self.dialog.withdraw()
        
        # 居中显示
        self._center_window()
        
        # 创建界面
        self._create_widgets()
        
        # 所有内容准备完成后再显示窗口
        self.dialog.deiconify()
        self.dialog.lift()
        self.dialog.focus_force()
    
    def _center_window(self):
        """将窗口居中显示"""
        self.dialog.update_idletasks()
        width = 700
        height = 700
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 说明文字
        info_label = ttk.Label(
            main_frame,
            text="批量添加账号到账号文件\n\n格式：手机号----密码（每行一个）\n例如：13800138000----password123",
            font=("Microsoft YaHei UI", 9),
            foreground="blue",
            justify=tk.LEFT
        )
        info_label.pack(pady=(0, 15))
        
        # === 管理员选择区域 ===
        owner_frame = ttk.LabelFrame(main_frame, text="选择管理员（可选）", padding="10")
        owner_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 获取所有启用的用户
        users = self.user_manager.get_all_users()
        enabled_users = [u for u in users if u.enabled]
        
        # 创建下拉选择框
        owner_select_frame = ttk.Frame(owner_frame)
        owner_select_frame.pack(fill=tk.X)
        
        ttk.Label(owner_select_frame, text="管理员:", width=10).pack(side=tk.LEFT, padx=(0, 5))
        
        self.owner_var = tk.StringVar(value="不分配")
        self.owner_combo = ttk.Combobox(
            owner_select_frame,
            textvariable=self.owner_var,
            state='readonly',
            width=30
        )
        
        # 填充用户列表
        self.user_list = []
        owner_options = ["不分配"]
        
        for user in enabled_users:
            display_text = f"{user.user_name} (ID: {user.user_id})"
            owner_options.append(display_text)
            self.user_list.append(user)
        
        self.owner_combo['values'] = owner_options
        self.owner_combo.current(0)  # 默认选择"不分配"
        self.owner_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 提示文字
        hint_label = ttk.Label(
            owner_frame,
            text="💡 提示：选择管理员后，添加的账号将自动分配给该用户",
            font=("Microsoft YaHei UI", 9),
            foreground="gray"
        )
        hint_label.pack(pady=(10, 0))
        
        # === 账号输入区域 ===
        # 文本框
        text_frame = ttk.LabelFrame(main_frame, text="账号列表（已有账号只显示手机号，新增账号格式：手机号----密码）", padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 添加滚动条的文本框
        text_container = ttk.Frame(text_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.accounts_text = tk.Text(text_container, height=15, width=60, yscrollcommand=scrollbar.set)
        self.accounts_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.accounts_text.yview)
        
        # 统计信息
        self.stats_var = tk.StringVar(value="待添加: 0 个账号")
        ttk.Label(main_frame, textvariable=self.stats_var, foreground="gray").pack(pady=(0, 10))
        
        # 绑定文本变化事件
        self.accounts_text.bind('<KeyRelease>', self._on_text_changed)
        
        # 加载已有账号到文本框
        self._load_existing_accounts()
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        # 第一行按钮
        button_row1 = ttk.Frame(button_frame)
        button_row1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(button_row1, text="➕ 添加到账号文件", command=self._add_accounts, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_row1, text="🗑️ 删除选中账号", command=self._delete_accounts, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_row1, text="🧹 清空所有账号", command=self._clear_all_accounts, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_row1, text="关闭", command=self.dialog.destroy, width=10).pack(side=tk.RIGHT)
    
    def _on_text_changed(self, event=None):
        """文本变化时更新统计信息"""
        text = self.accounts_text.get("1.0", tk.END).strip()
        if not text:
            self.stats_var.set("待添加: 0 个账号")
            return
        
        # 简单统计行数
        lines = [line.strip() for line in text.split('\n') if line.strip() and not line.strip().startswith('#')]
        self.stats_var.set(f"当前: {len(lines)} 行")
    
    def _load_existing_accounts(self):
        """加载已有账号到文本框（只显示手机号）"""
        try:
            # 获取账号文件路径
            from .config import ConfigLoader
            config = ConfigLoader().load()
            accounts_file = config.accounts_file
            
            if not accounts_file:
                return
            
            # 使用加密账号文件管理器读取
            try:
                from .encrypted_accounts_file import EncryptedAccountsFile
            except ImportError:
                try:
                    from encrypted_accounts_file import EncryptedAccountsFile
                except ImportError:
                    from src.encrypted_accounts_file import EncryptedAccountsFile
            
            encrypted_file = EncryptedAccountsFile(accounts_file)
            accounts_list = encrypted_file.read_accounts()
            
            if accounts_list:
                # 只显示手机号（不显示密码）
                lines = [phone for phone, password in accounts_list]
                text = '\n'.join(lines)
                
                # 插入到文本框
                self.accounts_text.delete("1.0", tk.END)
                self.accounts_text.insert("1.0", text)
                
                # 更新统计
                self._on_text_changed()
                
                self.log(f"✓ 已加载 {len(accounts_list)} 个已有账号")
        except Exception as e:
            # 如果加载失败，不影响使用（可能是新文件）
            print(f"[批量添加] 加载已有账号失败: {e}")
    
    def _add_accounts(self):
        """添加账号到账号文件"""
        # 获取账号文本
        text = self.accounts_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请输入账号信息")
            return
        
        # 获取账号文件路径（从配置中读取）
        from .config import ConfigLoader
        config = ConfigLoader().load()
        accounts_file = config.accounts_file
        
        if not accounts_file:
            messagebox.showerror("错误", "未配置账号文件路径")
            return
        
        # 解析账号（使用多线程加速）
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        lines = text.split('\n')
        new_accounts = []
        invalid_lines = []
        parse_lock = threading.Lock()
        
        def parse_line(line_data):
            """解析单行账号（线程安全）"""
            line_num, line = line_data
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                return None
            
            # 检查格式
            if '----' not in line:
                return ('error', line_num, "格式错误（缺少----分隔符）")
            
            parts = line.split('----')
            
            # 标准格式：手机号----密码（每行一个账号）
            if len(parts) != 2:
                return ('error', line_num, "格式错误（应为：手机号----密码）")
            
            phone = parts[0].strip()
            password = parts[1].strip()
            
            if not phone.isdigit() or len(phone) != 11:
                return ('error', line_num, "手机号格式错误")
            
            if not password:
                return ('error', line_num, "密码为空")
            
            return ('success', line_num, phone, password)
        
        # 使用线程池并行解析（最多8个线程）
        max_workers = min(8, len(lines))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有解析任务
            line_data = [(i+1, line) for i, line in enumerate(lines)]
            futures = {executor.submit(parse_line, data): data for data in line_data}
            
            # 收集结果
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                
                if result[0] == 'error':
                    _, line_num, error_msg = result
                    with parse_lock:
                        invalid_lines.append(f"第{line_num}行：{error_msg}")
                elif result[0] == 'success':
                    _, line_num, phone, password = result
                    with parse_lock:
                        new_accounts.append((phone, password, line_num))
        
        if not new_accounts:
            messagebox.showerror("错误", "没有有效的账号\n\n" + "\n".join(invalid_lines[:5]))
            return
        
        # 去重：检查输入中的重复账号
        seen_phones = {}  # {phone: (password, line_num)}
        unique_accounts = []
        input_duplicates = []  # 输入中的重复账号
        
        for phone, password, line_num in new_accounts:
            if phone in seen_phones:
                # 发现重复
                existing_password, existing_line = seen_phones[phone]
                if existing_password == password:
                    # 密码相同，记录为重复
                    input_duplicates.append(f"第{line_num}行：手机号 {phone} 重复（与第{existing_line}行相同）")
                else:
                    # 密码不同，使用最后出现的密码
                    input_duplicates.append(f"第{line_num}行：手机号 {phone} 重复但密码不同（将使用此密码）")
                    # 更新为最新的密码
                    seen_phones[phone] = (password, line_num)
                    # 从unique_accounts中移除旧的，添加新的
                    unique_accounts = [(p, pw) for p, pw in unique_accounts if p != phone]
                    unique_accounts.append((phone, password))
            else:
                seen_phones[phone] = (password, line_num)
                unique_accounts.append((phone, password))
        
        # 显示无效行和重复警告
        warnings = []
        if invalid_lines:
            warnings.append(f"格式错误: {len(invalid_lines)} 行")
        if input_duplicates:
            warnings.append(f"输入重复: {len(input_duplicates)} 行")
        
        if warnings:
            warning_message = "发现以下问题：\n\n"
            
            if invalid_lines:
                warning_message += "【格式错误】\n" + "\n".join(invalid_lines[:3])
                if len(invalid_lines) > 3:
                    warning_message += f"\n... 还有 {len(invalid_lines) - 3} 行\n"
                warning_message += "\n"
            
            if input_duplicates:
                warning_message += "【输入重复】\n" + "\n".join(input_duplicates[:3])
                if len(input_duplicates) > 3:
                    warning_message += f"\n... 还有 {len(input_duplicates) - 3} 行\n"
                warning_message += "\n"
            
            warning_message += f"是否继续添加 {len(unique_accounts)} 个有效账号？"
            
            result = messagebox.askyesno("警告", warning_message)
            if not result:
                return
        
        # 使用去重后的账号列表
        new_accounts = unique_accounts
        
        # 使用加密账号文件管理器读取现有账号（检查重复）
        from pathlib import Path
        try:
            from .encrypted_accounts_file import EncryptedAccountsFile
        except ImportError:
            try:
                from encrypted_accounts_file import EncryptedAccountsFile
            except ImportError:
                from src.encrypted_accounts_file import EncryptedAccountsFile
        
        encrypted_file = EncryptedAccountsFile(accounts_file)
        existing_accounts_dict = {}  # {phone: (password, owner)}
        
        try:
            existing_accounts_list = encrypted_file.read_accounts()
            for item in existing_accounts_list:
                if len(item) >= 2:
                    phone = item[0]
                    password = item[1]
                    owner = item[2] if len(item) > 2 else None
                    existing_accounts_dict[phone] = (password, owner)
        except Exception as e:
            print(f"[批量添加] 读取账号文件失败: {e}")
            # 如果读取失败，继续（可能是新文件）
        
        # 分类账号：新增、更新、重复（密码相同）
        accounts_to_add = []  # 新增的账号
        accounts_to_update = []  # 需要更新密码的账号
        duplicate_accounts = []  # 完全重复的账号（手机号和密码都相同）
        
        for phone, password in new_accounts:
            if phone in existing_accounts_dict:
                existing_password, existing_owner = existing_accounts_dict[phone]
                if existing_password == password:
                    # 密码相同，完全重复
                    duplicate_accounts.append(phone)
                else:
                    # 密码不同，需要更新
                    accounts_to_update.append((phone, password, existing_owner))
            else:
                # 新账号
                accounts_to_add.append((phone, password))
        
        if not result:
            return
        
        # 使用加密账号文件管理器处理账号
        try:
            # 确保目录存在
            Path(accounts_file).parent.mkdir(parents=True, exist_ok=True)
            
            # 处理更新密码的账号（先删除旧的，再添加新的）
            if accounts_to_update:
                # 读取所有现有账号
                all_accounts = encrypted_file.read_accounts()
                
                # 创建更新映射
                update_map = {phone: (password, owner) for phone, password, owner in accounts_to_update}
                
                # 更新账号列表
                updated_accounts = []
                for item in all_accounts:
                    phone = item[0]
                    if phone in update_map:
                        # 更新密码，保留管理员
                        new_password, existing_owner = update_map[phone]
                        if existing_owner:
                            updated_accounts.append((phone, new_password, existing_owner))
                        else:
                            updated_accounts.append((phone, new_password))
                    else:
                        # 保持不变
                        updated_accounts.append(item)
                
                # 写回文件
                if encrypted_file.write_accounts(updated_accounts):
                    self.log(f"✓ 成功更新 {len(accounts_to_update)} 个账号的密码（已加密）")
                else:
                    messagebox.showerror("错误", "更新账号密码失败")
                    return
            
            # 追加新账号（自动加密）
            if accounts_to_add:
                if encrypted_file.append_accounts(accounts_to_add):
                    self.log(f"✓ 成功添加 {len(accounts_to_add)} 个账号到账号文件（已加密）")
                else:
                    messagebox.showerror("错误", "写入账号文件失败")
                    return
            
            if duplicate_accounts:
                self.log(f"⚠️ 跳过 {len(duplicate_accounts)} 个完全重复的账号")
            
            # === 自动分配管理员 ===
            owner_selection = self.owner_var.get()
            assigned_count = 0
            
            if owner_selection != "不分配":
                # 获取选中的用户
                selected_user = None
                for user in self.user_list:
                    display_text = f"{user.user_name} (ID: {user.user_id})"
                    if display_text == owner_selection:
                        selected_user = user
                        break
                
                if selected_user:
                    # 只为新增的账号分配管理员（更新密码的账号保留原管理员）
                    phones_to_assign = [phone for phone, _ in accounts_to_add]
                    if phones_to_assign:
                        assigned_count = self.user_manager.batch_assign_accounts(phones_to_assign, selected_user.user_id)
                        self.log(f"✓ 已为 {assigned_count} 个新账号分配管理员: {selected_user.user_name}")
            
            # 刷新主界面账号列表
            if self.refresh_callback:
                try:
                    self.refresh_callback()
                    self.log(f"✓ 已刷新主界面账号列表")
                except Exception as e:
                    self.log(f"⚠️ 刷新主界面失败: {e}")
            
            # 显示成功消息
            success_message = "操作完成\n\n"
            
            if accounts_to_add:
                success_message += f"新增账号: {len(accounts_to_add)} 个\n"
            
            if accounts_to_update:
                success_message += f"更新密码: {len(accounts_to_update)} 个\n"
            
            if duplicate_accounts:
                success_message += f"跳过重复: {len(duplicate_accounts)} 个\n"
            
            if assigned_count > 0:
                success_message += f"\n已为新账号分配管理员: {owner_selection}\n"
            
            success_message += "\n主界面账号列表已自动刷新"
            
            # 确保对话框置顶
            self.dialog.lift()
            self.dialog.focus_force()
            messagebox.showinfo("成功", success_message, parent=self.dialog)
            
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("错误", f"写入账号文件失败: {e}")
    
    def _delete_accounts(self):
        """删除选中的账号"""
        # 获取文本框中的内容
        text = self.accounts_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请先输入要删除的账号")
            return
        
        # 解析要删除的账号
        phones_to_delete = []
        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '----' not in line:
                continue
            
            # 检查是否是批量格式
            if '----fc5678' in line:
                # 批量格式
                parts = line.split('----')
                for part in parts[1:]:
                    part = part.strip()
                    if part.startswith('fc5678'):
                        phone = part[6:]
                    else:
                        phone = part
                    
                    if phone.isdigit() and len(phone) == 11:
                        phones_to_delete.append(phone)
            else:
                # 标准格式
                parts = line.split('----', 1)
                phone = parts[0].strip()
                if phone.isdigit() and len(phone) == 11:
                    phones_to_delete.append(phone)
        
        if not phones_to_delete:
            messagebox.showwarning("提示", "没有有效的账号可删除")
            return
        
        # 确认删除
        result = messagebox.askyesno(
            "确认删除",
            f"确定要删除 {len(phones_to_delete)} 个账号吗？\n\n" +
            "删除操作将：\n" +
            "1. 从账号文件中删除\n" +
            "2. 从数据库中删除所有记录\n" +
            "3. 删除登录缓存文件\n" +
            "4. 删除账号信息缓存\n\n" +
            "此操作不可恢复！"
        )
        
        if not result:
            return
        
        # 获取账号文件路径
        from .config import ConfigLoader
        config = ConfigLoader().load()
        accounts_file = config.accounts_file
        
        if not accounts_file:
            messagebox.showerror("错误", "未配置账号文件路径")
            return
        
        from pathlib import Path
        
        # 使用加密账号文件管理器
        try:
            from .encrypted_accounts_file import EncryptedAccountsFile
        except ImportError:
            try:
                from encrypted_accounts_file import EncryptedAccountsFile
            except ImportError:
                from src.encrypted_accounts_file import EncryptedAccountsFile
        
        encrypted_file = EncryptedAccountsFile(accounts_file)
        
        # 删除账号（自动加密）
        try:
            if encrypted_file.delete_accounts(phones_to_delete):
                self.log(f"✓ 已从账号文件删除 {len(phones_to_delete)} 个账号（已加密）")
            else:
                messagebox.showerror("错误", "删除账号失败")
                return
        except Exception as e:
            messagebox.showerror("错误", f"删除账号失败: {e}")
            return
        
        # 从数据库删除
        from .local_db import LocalDatabase
        db = LocalDatabase()
        deleted_db_count = 0
        
        for phone in phones_to_delete:
            try:
                # 删除历史记录
                conn = db._get_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM history_records WHERE phone = ?", (phone,))
                conn.commit()
                conn.close()
                deleted_db_count += 1
            except Exception as e:
                self.log(f"⚠️ 删除数据库记录失败 ({phone}): {e}")
        
        self.log(f"✓ 已从数据库删除 {deleted_db_count} 个账号的记录")
        
        # 删除登录缓存
        from .login_cache_manager import LoginCacheManager
        from .adb_bridge import ADBBridge
        adb = ADBBridge()
        cache_manager = LoginCacheManager(adb)
        deleted_cache_count = 0
        
        for phone in phones_to_delete:
            try:
                if cache_manager.delete_cache(phone):
                    deleted_cache_count += 1
            except Exception as e:
                self.log(f"⚠️ 删除登录缓存失败 ({phone}): {e}")
        
        self.log(f"✓ 已删除 {deleted_cache_count} 个账号的登录缓存")
        
        # 删除账号信息缓存
        from .account_cache import get_account_cache
        account_cache = get_account_cache()
        deleted_account_cache_count = 0
        
        for phone in phones_to_delete:
            try:
                account_cache.clear(phone)
                deleted_account_cache_count += 1
            except Exception as e:
                self.log(f"⚠️ 删除账号缓存失败 ({phone}): {e}")
        
        self.log(f"✓ 已删除 {deleted_account_cache_count} 个账号的信息缓存")
        
        # 从用户管理器中移除账号分配
        removed_assignments = 0
        for phone in phones_to_delete:
            try:
                if self.user_manager.remove_account_from_all_users(phone):
                    removed_assignments += 1
            except Exception as e:
                self.log(f"⚠️ 移除账号分配失败 ({phone}): {e}")
        
        if removed_assignments > 0:
            self.log(f"✓ 已移除 {removed_assignments} 个账号的管理员分配")
        
        # 刷新主界面账号列表
        if self.refresh_callback:
            try:
                self.refresh_callback()
                self.log(f"✓ 已刷新主界面账号列表")
            except Exception as e:
                self.log(f"⚠️ 刷新主界面失败: {e}")
        
        # 显示成功消息
        # 确保对话框置顶
        self.dialog.lift()
        self.dialog.focus_force()
        messagebox.showinfo(
            "删除成功",
            f"已成功删除 {len(phones_to_delete)} 个账号\n\n" +
            f"- 账号文件: {len(phones_to_delete)} 个\n" +
            f"- 数据库记录: {deleted_db_count} 个\n" +
            f"- 登录缓存: {deleted_cache_count} 个\n" +
            f"- 账号缓存: {deleted_account_cache_count} 个\n" +
            f"- 管理员分配: {removed_assignments} 个\n\n" +
            "主界面账号列表已自动刷新",
            parent=self.dialog
        )
        
        # 清空文本框
        self.accounts_text.delete("1.0", tk.END)
        self._on_text_changed()
    
    def _clear_all_accounts(self):
        """清空所有账号"""
        # 确认清空
        result = messagebox.askyesno(
            "确认清空",
            "确定要清空所有账号吗？\n\n" +
            "此操作将删除账号文件中的所有账号！\n\n" +
            "此操作不可恢复！",
            icon='warning'
        )
        
        if not result:
            return
        
        # 询问是否清理缓存
        clear_cache = messagebox.askyesno(
            "清理缓存",
            "是否同时清理所有账号的缓存？\n\n" +
            "包括：\n" +
            "- 登录缓存文件\n" +
            "- 账号信息缓存\n" +
            "- 数据库记录\n\n" +
            "建议选择'是'以彻底清理"
        )
        
        # 获取账号文件路径
        from .config import ConfigLoader
        config = ConfigLoader().load()
        accounts_file = config.accounts_file
        
        if not accounts_file:
            messagebox.showerror("错误", "未配置账号文件路径")
            return
        
        from pathlib import Path
        
        # 使用加密账号文件管理器
        try:
            from .encrypted_accounts_file import EncryptedAccountsFile
        except ImportError:
            try:
                from encrypted_accounts_file import EncryptedAccountsFile
            except ImportError:
                from src.encrypted_accounts_file import EncryptedAccountsFile
        
        encrypted_file = EncryptedAccountsFile(accounts_file)
        
        # 如果需要清理缓存，先读取所有账号
        all_phones = []
        if clear_cache:
            try:
                accounts_list = encrypted_file.read_accounts()
                for phone, _ in accounts_list:
                    all_phones.append(phone)
            except Exception as e:
                print(f"[清空账号] 读取账号失败: {e}")
        
        # 清空账号文件（自动加密）
        try:
            if encrypted_file.clear_accounts():
                self.log(f"✓ 已清空账号文件（已加密）")
            else:
                messagebox.showerror("错误", "清空账号文件失败")
                return
        except Exception as e:
            messagebox.showerror("错误", f"清空账号文件失败: {e}")
            return
        
        # 清理缓存
        if clear_cache and all_phones:
            self.log(f"正在清理 {len(all_phones)} 个账号的缓存...")
            
            # 删除数据库记录
            from .local_db import LocalDatabase
            db = LocalDatabase()
            deleted_db_count = 0
            
            for phone in all_phones:
                try:
                    conn = db._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM history_records WHERE phone = ?", (phone,))
                    conn.commit()
                    conn.close()
                    deleted_db_count += 1
                except Exception as e:
                    self.log(f"⚠️ 删除数据库记录失败 ({phone}): {e}")
            
            self.log(f"✓ 已从数据库删除 {deleted_db_count} 个账号的记录")
            
            # 删除登录缓存
            from .login_cache_manager import LoginCacheManager
            from .adb_bridge import ADBBridge
            adb = ADBBridge()
            cache_manager = LoginCacheManager(adb)
            deleted_cache_count = 0
            
            for phone in all_phones:
                try:
                    if cache_manager.delete_cache(phone):
                        deleted_cache_count += 1
                except Exception as e:
                    self.log(f"⚠️ 删除登录缓存失败 ({phone}): {e}")
            
            self.log(f"✓ 已删除 {deleted_cache_count} 个账号的登录缓存")
            
            # 删除账号信息缓存
            from .account_cache import get_account_cache
            account_cache = get_account_cache()
            
            try:
                account_cache.clear()  # 清空所有缓存
                self.log(f"✓ 已清空所有账号信息缓存")
            except Exception as e:
                self.log(f"⚠️ 清空账号缓存失败: {e}")
            
            # 清空所有用户的账号分配
            try:
                self.user_manager.clear_all_account_assignments()
                self.log(f"✓ 已清空所有管理员的账号分配")
            except Exception as e:
                self.log(f"⚠️ 清空账号分配失败: {e}")
        
        # 显示成功消息
        # 确保对话框置顶
        self.dialog.lift()
        self.dialog.focus_force()
        if clear_cache:
            messagebox.showinfo(
                "清空成功",
                f"已成功清空所有账号\n\n" +
                f"- 账号文件: 已清空\n" +
                f"- 数据库记录: 已删除\n" +
                f"- 登录缓存: 已删除\n" +
                f"- 账号缓存: 已清空\n" +
                f"- 管理员分配: 已清空\n\n" +
                "主界面账号列表已自动刷新",
                parent=self.dialog
            )
        else:
            messagebox.showinfo(
                "清空成功",
                "已成功清空账号文件\n\n" +
                "缓存文件未清理，如需清理请重新执行清空操作\n\n" +
                "主界面账号列表已自动刷新",
                parent=self.dialog
            )
        
        # 清空文本框
        self.accounts_text.delete("1.0", tk.END)
        self._on_text_changed()
        
        # 刷新主界面账号列表
        if self.refresh_callback:
            try:
                self.refresh_callback()
                self.log(f"✓ 已刷新主界面账号列表")
            except Exception as e:
                self.log(f"⚠️ 刷新主界面失败: {e}")


class SelectUnassignedAccountsDialog:
    """从未分配账号中选择对话框"""
    
    def __init__(self, parent, user_manager: UserManager, user_id: str, user_name: str, 
                 unassigned_accounts: List[Dict[str, Any]], callback: Callable, log_callback: Callable):
        """初始化选择未分配账号对话框
        
        Args:
            parent: 父窗口
            user_manager: 用户管理器
            user_id: 目标用户ID
            user_name: 目标用户名称
            unassigned_accounts: 未分配的账号列表（汇总信息）
            callback: 完成后的回调函数
            log_callback: 日志回调函数
        """
        self.parent = parent
        self.user_manager = user_manager
        self.user_id = user_id
        self.user_name = user_name
        self.unassigned_accounts = unassigned_accounts
        self.callback = callback
        self.log = log_callback
        
        # 创建窗口
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"为 '{user_name}' 添加账号 - {len(unassigned_accounts)} 个未分配账号")
        self.dialog.geometry("800x600")
        self.dialog.resizable(True, True)
        
        # 居中显示
        self._center_window()
        
        # 创建界面
        self._create_widgets()
    
    def _center_window(self):
        """将窗口居中显示"""
        self.dialog.update_idletasks()
        width = 800
        height = 600
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 说明文字
        info_label = ttk.Label(
            main_frame,
            text=f"从数据库中选择账号分配给 '{self.user_name}'\n\n共 {len(self.unassigned_accounts)} 个未分配账号",
            font=("Microsoft YaHei UI", 10, "bold"),
            foreground="blue"
        )
        info_label.pack(pady=(0, 15))
        
        # 账号列表区域
        list_frame = ttk.LabelFrame(main_frame, text="未分配账号列表（可多选）", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 添加全选/反选按钮
        select_buttons = ttk.Frame(list_frame)
        select_buttons.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(select_buttons, text="全选", command=self._select_all, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(select_buttons, text="反选", command=self._invert_selection, width=8).pack(side=tk.LEFT)
        
        # 创建Treeview显示账号信息
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # 定义列
        columns = ("phone", "nickname", "user_id", "latest_date", "success_rate")
        self.account_tree = ttk.Treeview(tree_frame, columns=columns, show="tree headings", selectmode=tk.EXTENDED)
        
        # 配置勾选框列
        self.account_tree.heading("#0", text="", anchor=tk.CENTER)
        self.account_tree.column("#0", width=40, anchor=tk.CENTER, stretch=False, minwidth=40)
        
        # 配置其他列
        column_config = {
            "phone": ("手机号", 120),
            "nickname": ("昵称", 150),
            "user_id": ("用户ID", 100),
            "latest_date": ("最新记录日期", 120),
            "success_rate": ("成功率", 80)
        }
        
        for col, (heading, width) in column_config.items():
            self.account_tree.heading(col, text=heading)
            self.account_tree.column(col, width=width, anchor=tk.CENTER)
        
        # 添加滚动条
        scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.account_tree.yview)
        scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.account_tree.xview)
        self.account_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        self.account_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # 绑定点击事件（用于切换勾选状态）
        self.account_tree.bind("<Button-1>", self._on_tree_click)
        
        # 初始化勾选状态
        self.checked_items = {}  # {item_id: True/False}
        
        # 填充账号列表
        for account in self.unassigned_accounts:
            phone = account.get('phone', '')
            nickname = account.get('nickname', '-') or '-'
            user_id = account.get('user_id', '-') or '-'
            latest_date = account.get('latest_date', '-') or '-'
            success_rate = f"{account.get('success_rate', 0):.1f}%"
            
            values = (phone, nickname, user_id, latest_date, success_rate)
            item_id = self.account_tree.insert("", tk.END, text="□", values=values)
            self.checked_items[item_id] = False
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="分配选中账号", command=self._assign_selected, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="取消", command=self.dialog.destroy, width=10).pack(side=tk.LEFT)
    
    def _on_tree_click(self, event):
        """处理树的点击事件"""
        region = self.account_tree.identify("region", event.x, event.y)
        if region == "tree":
            # 点击了勾选框列
            item = self.account_tree.identify_row(event.y)
            if item:
                # 切换勾选状态
                current_state = self.checked_items.get(item, False)
                new_state = not current_state
                self.checked_items[item] = new_state
                
                # 更新显示
                self.account_tree.item(item, text="☑" if new_state else "□")
    
    def _select_all(self):
        """全选账号"""
        for item_id in self.account_tree.get_children():
            self.checked_items[item_id] = True
            self.account_tree.item(item_id, text="☑")
    
    def _invert_selection(self):
        """反选账号"""
        for item_id in self.account_tree.get_children():
            current_state = self.checked_items.get(item_id, False)
            new_state = not current_state
            self.checked_items[item_id] = new_state
            self.account_tree.item(item_id, text="☑" if new_state else "□")
    
    def _assign_selected(self):
        """分配选中的账号"""
        # 获取选中的账号
        selected_phones = []
        for item_id in self.account_tree.get_children():
            if self.checked_items.get(item_id, False):
                values = self.account_tree.item(item_id, 'values')
                if values:
                    selected_phones.append(values[0])  # 手机号在第一列
        
        if not selected_phones:
            messagebox.showwarning("提示", "请先勾选要分配的账号")
            return
        
        # 确认分配
        result = messagebox.askyesno(
            "确认分配",
            f"确定要将 {len(selected_phones)} 个账号分配给 '{self.user_name}' 吗？"
        )
        
        if result:
            # 批量分配
            count = self.user_manager.batch_assign_accounts(selected_phones, self.user_id)
            
            self.log(f"✓ 已为 '{self.user_name}' 分配 {count} 个账号")
            messagebox.showinfo("成功", f"已成功分配 {count} 个账号给 '{self.user_name}'")
            
            # 刷新父窗口
            self.callback()
            
            # 关闭对话框
            self.dialog.destroy()


class UnassignedAccountsDialog:
    """未分配账号对话框"""
    
    def __init__(self, parent, user_manager: UserManager, unassigned_phones: List[str], callback: Callable, log_callback: Callable):
        """初始化未分配账号对话框
        
        Args:
            parent: 父窗口
            user_manager: 用户管理器
            unassigned_phones: 未分配的手机号列表
            callback: 完成后的回调函数
            log_callback: 日志回调函数
        """
        self.parent = parent
        self.user_manager = user_manager
        self.unassigned_phones = unassigned_phones
        self.callback = callback
        self.log = log_callback
        
        # 创建窗口
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"分配未分配账号 ({len(unassigned_phones)} 个)")
        self.dialog.geometry("800x600")
        self.dialog.resizable(True, True)
        
        # 居中显示
        self._center_window()
        
        # 创建界面
        self._create_widgets()
    
    def _center_window(self):
        """将窗口居中显示"""
        self.dialog.update_idletasks()
        width = 800
        height = 600
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 说明文字
        info_label = ttk.Label(
            main_frame,
            text=f"发现 {len(self.unassigned_phones)} 个未分配的账号\n\n请选择要分配的账号和目标用户",
            font=("Microsoft YaHei UI", 10, "bold"),
            foreground="blue"
        )
        info_label.pack(pady=(0, 15))
        
        # === 左右分栏 ===
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 左侧：未分配账号列表
        left_frame = ttk.LabelFrame(content_frame, text="未分配账号（可多选）", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 添加全选/反选按钮
        select_buttons = ttk.Frame(left_frame)
        select_buttons.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(select_buttons, text="全选", command=self._select_all, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(select_buttons, text="反选", command=self._invert_selection, width=8).pack(side=tk.LEFT)
        
        # 账号列表
        listbox_frame = ttk.Frame(left_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        self.account_listbox = tk.Listbox(
            listbox_frame,
            yscrollcommand=scrollbar.set,
            font=("Microsoft YaHei UI", 10),
            selectmode=tk.MULTIPLE,  # 允许多选
            height=20
        )
        scrollbar.config(command=self.account_listbox.yview)
        
        self.account_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 填充账号列表
        for phone in self.unassigned_phones:
            self.account_listbox.insert(tk.END, phone)
        
        # 右侧：用户列表
        right_frame = ttk.LabelFrame(content_frame, text="选择管理员", padding="10")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 用户列表
        user_listbox_frame = ttk.Frame(right_frame)
        user_listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        user_scrollbar = ttk.Scrollbar(user_listbox_frame, orient=tk.VERTICAL)
        self.user_listbox = tk.Listbox(
            user_listbox_frame,
            yscrollcommand=user_scrollbar.set,
            font=("Microsoft YaHei UI", 10),
            height=20
        )
        user_scrollbar.config(command=self.user_listbox.yview)
        
        self.user_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        user_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 填充用户列表
        self.user_list = []
        users = self.user_manager.get_all_users()
        for user in users:
            if user.enabled:
                display_text = f"{user.user_name} (ID: {user.user_id})"
                self.user_listbox.insert(tk.END, display_text)
                self.user_list.append(user)
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="分配选中账号", command=self._assign_selected, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="取消", command=self.dialog.destroy, width=10).pack(side=tk.LEFT)
    
    def _select_all(self):
        """全选账号"""
        self.account_listbox.selection_set(0, tk.END)
    
    def _invert_selection(self):
        """反选账号"""
        for i in range(self.account_listbox.size()):
            if self.account_listbox.selection_includes(i):
                self.account_listbox.selection_clear(i)
            else:
                self.account_listbox.selection_set(i)
    
    def _assign_selected(self):
        """分配选中的账号"""
        # 获取选中的账号
        account_selection = self.account_listbox.curselection()
        if not account_selection:
            messagebox.showwarning("提示", "请选择要分配的账号")
            return
        
        # 获取选中的用户
        user_selection = self.user_listbox.curselection()
        if not user_selection:
            messagebox.showwarning("提示", "请选择管理员")
            return
        
        user = self.user_list[user_selection[0]]
        
        # 获取选中的手机号
        phones_to_assign = [self.unassigned_phones[i] for i in account_selection]
        
        # 确认分配
        result = messagebox.askyesno(
            "确认分配",
            f"确定要将 {len(phones_to_assign)} 个账号分配给 '{user.user_name}' 吗？"
        )
        
        if result:
            # 批量分配
            count = self.user_manager.batch_assign_accounts(phones_to_assign, user.user_id)
            
            self.log(f"✓ 已为 {count} 个账号分配管理员: {user.user_name}")
            messagebox.showinfo("成功", f"已成功分配 {count} 个账号给 '{user.user_name}'")
            
            # 从列表中移除已分配的账号
            for i in reversed(list(account_selection)):
                self.account_listbox.delete(i)
                self.unassigned_phones.pop(i)
            
            # 刷新父窗口
            self.callback()
            
            # 如果所有账号都已分配，关闭对话框
            if not self.unassigned_phones:
                messagebox.showinfo("完成", "所有账号都已分配")
                self.dialog.destroy()
            else:
                # 更新标题
                self.dialog.title(f"分配未分配账号 ({len(self.unassigned_phones)} 个)")



class SelectUserDialog:
    """选择用户对话框"""
    
    def __init__(self, parent, user_manager: UserManager, users: List[User], 
                 callback: Callable, log_callback: Callable):
        """初始化选择用户对话框
        
        Args:
            parent: 父窗口
            user_manager: 用户管理器
            users: 用户列表
            callback: 完成后的回调函数
            log_callback: 日志回调函数
        """
        self.parent = parent
        self.user_manager = user_manager
        self.users = users
        self.callback = callback
        self.log = log_callback
        
        # 创建窗口
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("选择管理员")
        self.dialog.geometry("500x400")
        self.dialog.resizable(True, True)
        
        # 居中显示
        self._center_window()
        
        # 创建界面
        self._create_widgets()
    
    def _center_window(self):
        """将窗口居中显示"""
        self.dialog.update_idletasks()
        width = 500
        height = 400
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_widgets(self):
        """创建界面组件"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 说明文字
        info_label = ttk.Label(
            main_frame,
            text=f"选择一个用户作为管理员\n\n共 {len(self.users)} 个用户",
            font=("Microsoft YaHei UI", 10, "bold"),
            foreground="blue"
        )
        info_label.pack(pady=(0, 15))
        
        # 用户列表区域
        list_frame = ttk.LabelFrame(main_frame, text="用户列表", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 创建Treeview
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("user_id", "user_name", "account_count", "enabled")
        self.user_tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        
        column_config = {
            "user_id": ("用户ID", 100),
            "user_name": ("用户名称", 150),
            "account_count": ("账号数量", 100),
            "enabled": ("状态", 80)
        }
        
        for col, (heading, width) in column_config.items():
            self.user_tree.heading(col, text=heading)
            self.user_tree.column(col, width=width, anchor=tk.CENTER)
        
        # 添加滚动条
        scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.user_tree.yview)
        scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.user_tree.xview)
        self.user_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        self.user_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # 填充用户列表
        for user in self.users:
            account_count = len(self.user_manager.get_user_accounts(user.user_id))
            status = "启用" if user.enabled else "禁用"
            
            # 用户ID显示：去掉 "user_" 前缀，只显示数字部分
            display_user_id = user.user_id.replace("user_", "") if user.user_id.startswith("user_") else user.user_id
            
            values = (display_user_id, user.user_name, account_count, status)
            # 使用 tags 存储完整的 user_id
            self.user_tree.insert("", tk.END, values=values, tags=(user.user_id,))
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="确定", command=self._confirm, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="取消", command=self.dialog.destroy, width=10).pack(side=tk.LEFT)
    
    def _confirm(self):
        """确认选择"""
        # 获取选中的用户
        selection = self.user_tree.selection()
        if not selection:
            messagebox.showwarning("提示", "请选择一个用户")
            return
        
        item = selection[0]
        tags = self.user_tree.item(item, 'tags')
        values = self.user_tree.item(item, 'values')
        
        # 从 tags 中获取完整的 user_id
        user_id = tags[0] if tags else values[0]
        user_name = values[1]
        
        self.log(f"✓ 已选择管理员: {user_name}")
        messagebox.showinfo("成功", f"已选择管理员: {user_name}")
        
        # 刷新父窗口
        self.callback()
        
        # 关闭对话框
        self.dialog.destroy()
