"""
卡密管理工具 - GUI 版本
用于生成、查看、管理卡密

安全特性：
- 访问密码保护
- 默认密码：hye19911206
- 可在服务端重置密码
"""

import os
import json
import random
import string
import requests
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
from datetime import datetime, timedelta
from typing import Optional
import threading
import webbrowser
from pathlib import Path


class LicenseManagerGUI:
    """卡密管理器 GUI"""
    
    def __init__(self, root):
        """初始化"""
        self.root = root
        self.root.title("卡密管理工具 v1.0.0")
        self.root.geometry("1000x700")
        
        # 配置
        # 获取可执行文件所在目录（支持打包后的exe）
        if getattr(sys, 'frozen', False):
            # 打包后的exe，使用exe所在目录
            base_dir = os.path.dirname(sys.executable)
        else:
            # 开发环境，使用当前工作目录
            base_dir = os.getcwd()
        
        # 优先使用exe所在目录的 .env
        env_path = os.path.join(base_dir, ".env")
        if os.path.exists(env_path):
            self.config_file = env_path
        elif os.path.exists(".env"):
            self.config_file = ".env"
        elif os.path.exists("server/.env"):
            self.config_file = "server/.env"
        else:
            self.config_file = ".env"  # 默认值
        
        # 先加载配置（必须在使用 supabase_key 之前）
        self.load_config()
        
        # 标记密码已验证（在 main 函数中验证）
        self.password_verified = True
        
        # 本地缓存
        self.licenses_cache = []  # 卡密列表缓存
        self.device_counts_cache = {}  # 设备数量缓存 {license_key: count}
        self.cache_timestamp = None  # 缓存时间戳
        self.cache_valid_duration = 60  # 缓存有效期（秒）
        
        # 自动刷新
        self.auto_refresh = True
        self.refresh_interval = 15000  # 15秒
        self.last_refresh_time = None
        
        # 创建界面
        self.create_widgets()
        
        # 窗口居中显示
        self._center_main_window()
        
        # 加载数据
        self.refresh_list()
        
        # 启动自动刷新
        self.start_auto_refresh()
    
    def _center_main_window(self):
        """将主窗口居中显示在屏幕中间"""
        self.root.update()
        
        # 获取窗口实际大小
        width = 1000
        height = 700
        
        # 相对于屏幕居中
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        
        # 确保坐标不为负数
        x = max(0, x)
        y = max(0, y)
        
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def load_config(self):
        """加载配置"""
        if not os.path.exists(self.config_file):
            # 配置文件不存在，显示错误并退出
            try:
                messagebox.showerror("错误", f"未找到配置文件：{self.config_file}\n\n请确保 .env 文件存在并包含：\nSUPABASE_URL=你的数据库地址\nSUPABASE_KEY=你的数据库密钥")
            except:
                print(f"错误: 未找到配置文件 {self.config_file}")
            return False
        
        self.supabase_url = None
        self.supabase_key = None
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == 'SUPABASE_URL':
                            self.supabase_url = value
                        elif key == 'SUPABASE_KEY':
                            self.supabase_key = value
        except Exception as e:
            try:
                messagebox.showerror("错误", f"读取配置文件失败: {e}")
            except:
                print(f"错误: 读取配置文件失败 {e}")
            return False
        
        if not self.supabase_url or not self.supabase_key:
            try:
                messagebox.showerror("错误", "配置文件中缺少 SUPABASE_URL 或 SUPABASE_KEY")
            except:
                print("错误: 配置文件中缺少必要信息")
            return False
        
        return True
    
    def _center_window(self, window, parent=None):
        """将窗口居中显示在父窗口中间
        
        Args:
            window: 要居中的窗口
            parent: 父窗口，如果为None则使用self.root
        """
        if parent is None:
            parent = self.root
        
        window.update()
        width = window.winfo_width()
        height = window.winfo_height()
        
        # 相对于父窗口居中
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        x = parent_x + (parent_width // 2) - (width // 2)
        y = parent_y + (parent_height // 2) - (height // 2)
        
        # 确保窗口不会超出屏幕
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + width > screen_width:
            x = screen_width - width
        if y + height > screen_height:
            y = screen_height - height
        
        window.geometry(f'{width}x{height}+{x}+{y}')
    
    def _get_password_hash(self) -> str:
        """从服务器获取密码哈希值"""
        try:
            headers = {
                'apikey': self.supabase_key,
                'Authorization': f'Bearer {self.supabase_key}',
            }
            
            url = f"{self.supabase_url}/rest/v1/admin_password?id=eq.1&select=password_hash"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    print("[密码] 从服务器获取密码哈希")
                    return data[0]['password_hash']
        except Exception as e:
            print(f"[密码] 从服务器获取密码失败: {e}")
        
        # 如果服务器获取失败，使用默认密码
        import hashlib
        default_password = "hye19911206"
        print("[密码] 使用默认密码")
        return hashlib.sha256(default_password.encode()).hexdigest()
    
    def _verify_password(self) -> bool:
        """验证密码"""
        import hashlib
        
        # 创建自定义密码对话框
        password_dialog = tk.Toplevel(self.root)
        password_dialog.title("访问验证")
        password_dialog.geometry("400x250")
        password_dialog.resizable(False, False)
        password_dialog.configure(bg='#1a2332')
        
        # 禁止关闭窗口（必须验证或取消）
        def on_close():
            """尝试关闭窗口时的处理"""
            if messagebox.askyesno("确认", "确定要退出吗？", parent=password_dialog):
                result['verified'] = False
                password_dialog.destroy()
        
        password_dialog.protocol("WM_DELETE_WINDOW", on_close)
        
        # 居中显示
        password_dialog.update()
        width = password_dialog.winfo_width()
        height = password_dialog.winfo_height()
        x = (password_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (password_dialog.winfo_screenheight() // 2) - (height // 2)
        
        # 确保坐标不为负数
        x = max(0, x)
        y = max(0, y)
        
        password_dialog.geometry(f'{width}x{height}+{x}+{y}')
        
        # 确保窗口显示在最前面
        password_dialog.lift()
        password_dialog.attributes('-topmost', True)
        password_dialog.after(100, lambda: password_dialog.attributes('-topmost', False))
        
        # 获得焦点
        password_dialog.focus_force()
        
        # 模态对话框
        password_dialog.transient(self.root)
        password_dialog.grab_set()
        
        # 主容器
        main_frame = tk.Frame(password_dialog, bg='#1a2332', padx=40, pady=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = tk.Label(
            main_frame,
            text="🔒 访问验证",
            font=("Microsoft YaHei UI", 20, "bold"),
            fg='#4a9eff',
            bg='#1a2332'
        )
        title_label.pack(pady=(0, 10))
        
        # 副标题
        subtitle_label = tk.Label(
            main_frame,
            text="请输入管理员密码",
            font=("Microsoft YaHei UI", 10),
            fg='#8899aa',
            bg='#1a2332'
        )
        subtitle_label.pack(pady=(0, 20))
        
        # 密码输入框
        password_entry = tk.Entry(
            main_frame,
            font=("Consolas", 12),
            bg='#0d1520',
            fg='#ffffff',
            insertbackground='#ffffff',
            relief=tk.FLAT,
            bd=0,
            highlightthickness=2,
            highlightbackground='#2a3f5f',
            highlightcolor='#4a9eff',
            show='●'
        )
        password_entry.pack(fill=tk.X, ipady=10, ipadx=10, pady=(0, 20))
        password_entry.focus_set()
        
        # 结果变量
        result = {'verified': False}
        
        def verify():
            """验证密码"""
            password = password_entry.get().strip()
            
            if not password:
                messagebox.showwarning("提示", "请输入密码", parent=password_dialog)
                return
            
            # 验证密码
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            if password_hash == self.password_hash:
                result['verified'] = True
                password_dialog.destroy()
            else:
                messagebox.showerror("错误", "密码错误！", parent=password_dialog)
                password_entry.delete(0, tk.END)
                password_entry.focus_set()
        
        def cancel():
            """取消"""
            if messagebox.askyesno("确认", "确定要退出吗？", parent=password_dialog):
                result['verified'] = False
                password_dialog.destroy()
        
        # 绑定回车键
        password_entry.bind('<Return>', lambda e: verify())
        
        # 按钮容器
        button_frame = tk.Frame(main_frame, bg='#1a2332')
        button_frame.pack(fill=tk.X)
        
        # 确定按钮
        ok_btn = tk.Button(
            button_frame,
            text="确定",
            font=("Microsoft YaHei UI", 11, "bold"),
            fg='#ffffff',
            bg='#4a9eff',
            activebackground='#3a8eef',
            activeforeground='#ffffff',
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            command=verify,
            width=10,
            height=1
        )
        ok_btn.pack(side=tk.LEFT, padx=(0, 10), expand=True, fill=tk.X)
        
        # 取消按钮
        cancel_btn = tk.Button(
            button_frame,
            text="取消",
            font=("Microsoft YaHei UI", 11),
            fg='#8899aa',
            bg='#2a3f5f',
            activebackground='#1a2332',
            activeforeground='#4a9eff',
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            command=cancel,
            width=10,
            height=1
        )
        cancel_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # 等待对话框关闭
        password_dialog.wait_window()
        
        return result['verified']
        ok_btn = tk.Button(
            button_frame,
            text="确定",
            font=("Microsoft YaHei UI", 11, "bold"),
            fg='#ffffff',
            bg='#4a9eff',
            activebackground='#3a8eef',
            activeforeground='#ffffff',
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            command=verify,
            width=10,
            height=1
        )
        ok_btn.pack(side=tk.LEFT, padx=(0, 10), expand=True, fill=tk.X)
        
        # 取消按钮
        cancel_btn = tk.Button(
            button_frame,
            text="取消",
            font=("Microsoft YaHei UI", 11),
            fg='#8899aa',
            bg='#2a3f5f',
            activebackground='#1a2332',
            activeforeground='#4a9eff',
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            command=cancel,
            width=10,
            height=1
        )
        cancel_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # 等待对话框关闭
        password_dialog.wait_window()
        
        return result['verified']
    
    def reset_password(self):
        """重置密码（保存到服务器）"""
        import hashlib
        from tkinter import simpledialog
        
        # 验证当前密码
        current_password = simpledialog.askstring(
            "验证身份",
            "请输入当前密码：",
            show='*'
        )
        
        if not current_password:
            return
        
        current_hash = hashlib.sha256(current_password.encode()).hexdigest()
        
        if current_hash != self.password_hash:
            messagebox.showerror("错误", "当前密码错误！")
            return
        
        # 输入新密码
        new_password = simpledialog.askstring(
            "设置新密码",
            "请输入新密码：",
            show='*'
        )
        
        if not new_password:
            return
        
        if len(new_password) < 6:
            messagebox.showerror("错误", "密码长度至少6位！")
            return
        
        # 确认新密码
        confirm_password = simpledialog.askstring(
            "确认密码",
            "请再次输入新密码：",
            show='*'
        )
        
        if new_password != confirm_password:
            messagebox.showerror("错误", "两次输入的密码不一致！")
            return
        
        # 保存新密码到服务器
        new_hash = hashlib.sha256(new_password.encode()).hexdigest()
        
        try:
            headers = {
                'apikey': self.supabase_key,
                'Authorization': f'Bearer {self.supabase_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'password_hash': new_hash,
                'updated_at': datetime.now().isoformat()
            }
            
            url = f"{self.supabase_url}/rest/v1/admin_password?id=eq.1"
            response = requests.patch(url, headers=headers, json=data, timeout=10)
            
            if response.status_code in [200, 204]:
                self.password_hash = new_hash
                messagebox.showinfo("成功", "密码已重置并同步到服务器！\n\n所有客户端下次启动时将使用新密码。")
            else:
                messagebox.showerror("错误", f"保存到服务器失败: {response.status_code}")
        except Exception as e:
            messagebox.showerror("错误", f"保存密码失败: {e}")
    
    def create_widgets(self):
        """创建界面组件"""
        # 顶部工具栏
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="生成卡密", command=self.show_generate_dialog, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="刷新列表", command=lambda: self.refresh_list(force=True), width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="设备状态", command=self.show_device_status, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="修改设备数", command=self.modify_device_limit, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="禁用卡密", command=self.disable_selected, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="删除卡密", command=self.delete_selected, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="解绑设备", command=self.unbind_selected, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="延长有效期", command=self.extend_selected, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="重置密码", command=self.reset_password, width=12).pack(side=tk.RIGHT, padx=2)
        
        # 自动刷新开关
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="自动同步", variable=self.auto_refresh_var, 
                       command=self.toggle_auto_refresh).pack(side=tk.RIGHT, padx=5)
        
        # 筛选器
        filter_frame = ttk.Frame(self.root)
        filter_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Label(filter_frame, text="状态筛选:").pack(side=tk.LEFT, padx=5)
        self.filter_var = tk.StringVar(value="all")
        ttk.Radiobutton(filter_frame, text="全部", variable=self.filter_var, value="all", command=self.apply_filter).pack(side=tk.LEFT)
        ttk.Radiobutton(filter_frame, text="未使用", variable=self.filter_var, value="unused", command=self.apply_filter).pack(side=tk.LEFT)
        ttk.Radiobutton(filter_frame, text="已激活", variable=self.filter_var, value="active", command=self.apply_filter).pack(side=tk.LEFT)
        ttk.Radiobutton(filter_frame, text="已禁用", variable=self.filter_var, value="disabled", command=self.apply_filter).pack(side=tk.LEFT)
        
        # 统计信息
        self.stats_label = ttk.Label(filter_frame, text="", foreground="blue")
        self.stats_label.pack(side=tk.RIGHT, padx=10)
        
        # 同步时间显示
        self.sync_time_label = ttk.Label(filter_frame, text="", foreground="gray")
        self.sync_time_label.pack(side=tk.RIGHT, padx=5)
        
        # 卡密列表
        list_frame = ttk.Frame(self.root)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建表格
        columns = ("卡密", "状态", "设备限制", "设备ID", "激活时间", "过期时间", "备注")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="extended")
        
        # 设置列
        self.tree.heading("卡密", text="卡密")
        self.tree.heading("状态", text="状态")
        self.tree.heading("设备限制", text="设备限制")
        self.tree.heading("设备ID", text="设备ID")
        self.tree.heading("激活时间", text="激活时间")
        self.tree.heading("过期时间", text="过期时间")
        self.tree.heading("备注", text="备注")
        
        self.tree.column("卡密", width=200, anchor=tk.CENTER)
        self.tree.column("状态", width=80, anchor=tk.CENTER)
        self.tree.column("设备限制", width=80, anchor=tk.CENTER)
        self.tree.column("设备ID", width=120, anchor=tk.CENTER)
        self.tree.column("激活时间", width=120, anchor=tk.CENTER)
        self.tree.column("过期时间", width=120, anchor=tk.CENTER)
        self.tree.column("备注", width=150, anchor=tk.CENTER)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 右键菜单
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="复制卡密", command=self.copy_license)
        self.context_menu.add_command(label="查看详情", command=self.show_details)
        self.context_menu.add_command(label="设备状态", command=self.show_device_status)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="修改设备数", command=self.modify_device_limit)
        self.context_menu.add_command(label="禁用", command=self.disable_selected)
        self.context_menu.add_command(label="删除（永久）", command=self.delete_selected)
        self.context_menu.add_command(label="解绑设备", command=self.unbind_selected)
        
        self.tree.bind("<Button-3>", self.show_context_menu)
        
        # 状态栏
        self.status_label = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def generate_license_key(self) -> str:
        """生成卡密（完全随机，无固定前缀）"""
        chars = string.ascii_uppercase + string.digits
        parts = []
        for _ in range(5):  # 5组，每组4位
            part = ''.join(random.choices(chars, k=4))
            parts.append(part)
        return '-'.join(parts)
    
    def show_generate_dialog(self):
        """显示生成对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("生成卡密")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 居中显示
        self._center_window(dialog)
        
        # 有效期
        ttk.Label(dialog, text="有效期（天）:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        days_var = tk.StringVar(value="365")
        ttk.Entry(dialog, textvariable=days_var, width=20).grid(row=0, column=1, padx=10, pady=10)
        
        # 设备数量限制
        ttk.Label(dialog, text="设备数量:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        max_devices_var = tk.StringVar(value="1")
        devices_entry = ttk.Entry(dialog, textvariable=max_devices_var, width=20)
        devices_entry.grid(row=1, column=1, padx=10, pady=10)
        ttk.Label(dialog, text="(一个卡密可在几台设备上使用)", font=("", 8), foreground="gray").grid(row=1, column=2, padx=5, sticky=tk.W)
        
        # 生成数量
        ttk.Label(dialog, text="生成数量:").grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        count_var = tk.StringVar(value="1")
        ttk.Entry(dialog, textvariable=count_var, width=20).grid(row=2, column=1, padx=10, pady=10)
        
        # 备注
        ttk.Label(dialog, text="备注:").grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        notes_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=notes_var, width=20).grid(row=3, column=1, padx=10, pady=10)
        
        # 按钮
        def generate():
            try:
                days = int(days_var.get())
                max_devices = int(max_devices_var.get())
                count = int(count_var.get())
                notes = notes_var.get()
                
                if max_devices < 1:
                    messagebox.showerror("错误", "设备数量至少为1")
                    return
                
                dialog.destroy()
                self.generate_licenses(days, count, notes, max_devices)
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数字")
        
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="生成", command=generate, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)
    
    def generate_licenses(self, days: int, count: int, notes: str, max_devices: int = 1):
        """生成卡密
        
        Args:
            days: 有效期天数
            count: 生成数量
            notes: 备注
            max_devices: 最大设备数量
        """
        self.status_label.config(text=f"正在生成 {count} 个卡密...")
        
        def task():
            licenses = []
            for i in range(count):
                license_key = self.generate_license_key()
                expires_at = (datetime.now() + timedelta(days=days)).isoformat()
                
                headers = {
                    'apikey': self.supabase_key,
                    'Authorization': f'Bearer {self.supabase_key}',
                    'Content-Type': 'application/json',
                    'Prefer': 'return=representation'
                }
                
                data = {
                    'license_key': license_key,
                    'status': 'unused',
                    'expires_at': expires_at,
                    'max_devices': max_devices,
                    'notes': notes or f'{days}天有效期 {max_devices}设备'
                }
                
                url = f"{self.supabase_url}/rest/v1/licenses"
                response = requests.post(url, headers=headers, json=data, timeout=10)
                
                if response.status_code in [200, 201]:
                    licenses.append(license_key)
            
            # 保存到文件
            if licenses:
                filename = f"卡密_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"有效期: {days} 天\n")
                    f.write(f"设备数量: {max_devices} 台\n")
                    f.write(f"数量: {len(licenses)}\n")
                    f.write("-" * 50 + "\n\n")
                    for key in licenses:
                        f.write(f"{key}\n")
                
                self.root.after(0, lambda: messagebox.showinfo("成功", f"已生成 {len(licenses)} 个卡密\n有效期: {days}天\n设备数量: {max_devices}台\n保存到: {filename}"))
            
            # 使缓存失效，强制刷新
            self.cache_timestamp = None
            self.root.after(0, lambda: self.refresh_list(force=True))
            self.root.after(0, lambda: self.status_label.config(text="就绪"))
        
        threading.Thread(target=task, daemon=True).start()
    
    def is_cache_valid(self):
        """检查缓存是否有效"""
        if not self.cache_timestamp:
            return False
        
        elapsed = (datetime.now() - self.cache_timestamp).total_seconds()
        return elapsed < self.cache_valid_duration
    
    def refresh_list(self, force=False):
        """刷新列表
        
        Args:
            force: 是否强制刷新（忽略缓存）
        """
        # 如果缓存有效且不是强制刷新，使用缓存
        if not force and self.is_cache_valid():
            print("[缓存] 使用本地缓存数据")
            self.update_tree(self.licenses_cache)
            self.update_statistics(self.licenses_cache)
            self.status_label.config(text="就绪（使用缓存）")
            return
        
        self.status_label.config(text="正在加载...")
        
        def task():
            headers = {
                'apikey': self.supabase_key,
                'Authorization': f'Bearer {self.supabase_key}',
            }
            
            url = f"{self.supabase_url}/rest/v1/licenses?select=*&order=created_at.desc"
            filter_status = self.filter_var.get()
            if filter_status != "all":
                url += f"&status=eq.{filter_status}"
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                licenses = response.json()
                
                # 更新缓存
                self.licenses_cache = licenses
                self.cache_timestamp = datetime.now()
                print(f"[缓存] 已更新缓存，共 {len(licenses)} 条记录")
                
                # 更新表格
                self.root.after(0, lambda: self.update_tree(licenses))
                
                # 更新统计
                self.root.after(0, lambda: self.update_statistics(licenses))
                
                # 更新同步时间
                self.last_refresh_time = datetime.now()
                self.root.after(0, self.update_sync_time)
            
            self.root.after(0, lambda: self.status_label.config(text="就绪"))
        
        threading.Thread(target=task, daemon=True).start()
    
    def update_tree(self, licenses):
        """更新表格（使用缓存的设备数量）"""
        # 清空
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 状态映射（英文 -> 中文）
        status_map = {
            'unused': '未使用',
            'active': '已激活',
            'disabled': '已禁用'
        }
        
        # 如果设备数量缓存为空，批量查询一次
        if not self.device_counts_cache:
            self.device_counts_cache = self.get_all_device_counts()
        
        # 批量查询所有设备ID（用于显示）
        device_ids_cache = {}
        try:
            headers = {
                'apikey': self.supabase_key,
                'Authorization': f'Bearer {self.supabase_key}'
            }
            url = f"{self.supabase_url}/rest/v1/device_bindings?select=license_key,machine_id"
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                bindings = response.json()
                for binding in bindings:
                    key = binding.get('license_key')
                    machine_id = binding.get('machine_id', '')
                    if key:
                        if key not in device_ids_cache:
                            device_ids_cache[key] = []
                        device_ids_cache[key].append(machine_id)
        except Exception as e:
            print(f"查询设备绑定失败: {e}")
        
        # 添加数据
        for lic in licenses:
            key = lic['license_key']
            status_en = lic['status']
            status_cn = status_map.get(status_en, status_en)  # 转换为中文
            max_devices = lic.get('max_devices', 1)
            
            # 从缓存中获取设备数量
            device_count = self.device_counts_cache.get(key, 0)
            device_limit = f"{device_count}/{max_devices}"
            
            # 从 device_bindings 获取设备ID列表
            device_ids = device_ids_cache.get(key, [])
            if device_ids:
                # 显示第一个设备ID（截断显示）
                machine_id = device_ids[0]
                if len(machine_id) > 16:
                    machine_id = machine_id[:16] + '...'
                # 如果有多个设备，添加提示
                if len(device_ids) > 1:
                    machine_id += f' (+{len(device_ids)-1})'
            else:
                machine_id = '-'
            
            activated_at = lic.get('activated_at', '')[:10] if lic.get('activated_at') else '-'
            expires_at = lic.get('expires_at', '')[:10] if lic.get('expires_at') else '-'
            notes = lic.get('notes') or '-'
            
            # 根据状态设置标签（使用英文作为标签）
            tag = status_en
            self.tree.insert("", tk.END, values=(key, status_cn, device_limit, machine_id, activated_at, expires_at, notes), tags=(tag,))
        
        # 设置标签颜色
        self.tree.tag_configure("unused", foreground="green")
        self.tree.tag_configure("active", foreground="blue")
        self.tree.tag_configure("disabled", foreground="red")
    
    def get_device_count(self, license_key: str) -> int:
        """获取卡密已绑定的设备数量（同步方法，仅供内部使用）"""
        try:
            headers = {
                'apikey': self.supabase_key,
                'Authorization': f'Bearer {self.supabase_key}',
            }
            
            url = f"{self.supabase_url}/rest/v1/device_bindings?license_key=eq.{license_key}&select=machine_id"
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                devices = response.json()
                return len(devices)
        except:
            pass
        
        return 0
    
    def get_all_device_counts(self, use_cache=True) -> dict:
        """批量获取所有卡密的设备数量（一次性查询，提高性能）
        
        Args:
            use_cache: 是否使用缓存
            
        Returns:
            dict: {license_key: device_count}
        """
        # 如果使用缓存且缓存有效，直接返回
        if use_cache and self.device_counts_cache and self.is_cache_valid():
            print("[缓存] 使用设备数量缓存")
            return self.device_counts_cache
        
        try:
            headers = {
                'apikey': self.supabase_key,
                'Authorization': f'Bearer {self.supabase_key}',
            }
            
            # 一次性查询所有设备绑定
            url = f"{self.supabase_url}/rest/v1/device_bindings?select=license_key"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                devices = response.json()
                
                # 统计每个卡密的设备数量
                device_counts = {}
                for device in devices:
                    key = device.get('license_key')
                    if key:
                        device_counts[key] = device_counts.get(key, 0) + 1
                
                # 更新缓存
                self.device_counts_cache = device_counts
                print(f"[缓存] 已更新设备数量缓存")
                
                return device_counts
        except Exception as e:
            print(f"批量查询设备数量失败: {e}")
        
        return self.device_counts_cache if use_cache else {}
    
    def modify_device_limit(self):
        """修改选中卡密的设备数量限制"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要修改的卡密")
            return
        
        if len(selected) > 1:
            messagebox.showwarning("警告", "一次只能修改一个卡密的设备数量")
            return
        
        # 获取当前卡密信息
        item = selected[0]
        values = self.tree.item(item)['values']
        license_key = values[0]
        current_limit = values[2].split('/')[1]  # 从 "1/3" 中提取 "3"
        
        # 弹出输入对话框
        new_limit = tk.simpledialog.askinteger(
            "修改设备数量限制",
            f"当前卡密: {license_key}\n当前设备限制: {current_limit} 台\n\n请输入新的设备数量限制:",
            minvalue=1,
            maxvalue=100,
            initialvalue=int(current_limit)
        )
        
        if not new_limit:
            return
        
        # 在后台线程更新
        self.status_label.config(text="正在修改设备数量...")
        
        def task():
            try:
                headers = {
                    'apikey': self.supabase_key,
                    'Authorization': f'Bearer {self.supabase_key}',
                    'Content-Type': 'application/json'
                }
                
                data = {'max_devices': new_limit}
                url = f"{self.supabase_url}/rest/v1/licenses?license_key=eq.{license_key}"
                
                response = requests.patch(url, headers=headers, json=data, timeout=10)
                
                if response.status_code in [200, 204]:
                    self.root.after(0, lambda: messagebox.showinfo("成功", f"已将设备数量限制修改为 {new_limit} 台"))
                    # 使缓存失效，强制刷新
                    self.cache_timestamp = None
                    self.root.after(0, lambda: self.refresh_list(force=True))
                else:
                    self.root.after(0, lambda: messagebox.showerror("失败", f"修改失败: {response.status_code}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"修改失败: {e}"))
            finally:
                self.root.after(0, lambda: self.status_label.config(text="就绪"))
        
        threading.Thread(target=task, daemon=True).start()
    
    def update_statistics(self, licenses):
        """更新统计信息"""
        total = len(licenses)
        
        # 获取所有卡密统计
        headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}',
        }
        url = f"{self.supabase_url}/rest/v1/licenses?select=status"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            all_licenses = response.json()
            unused = sum(1 for l in all_licenses if l['status'] == 'unused')
            active = sum(1 for l in all_licenses if l['status'] == 'active')
            disabled = sum(1 for l in all_licenses if l['status'] == 'disabled')
            
            self.stats_label.config(text=f"总数: {len(all_licenses)} | 未使用: {unused} | 已激活: {active} | 已禁用: {disabled}")
    
    def disable_selected(self):
        """禁用选中的卡密"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要禁用的卡密")
            return
        
        if not messagebox.askyesno("确认", f"确定要禁用 {len(selected)} 个卡密吗？"):
            return
        
        self.status_label.config(text=f"正在禁用 {len(selected)} 个卡密...")
        
        def task():
            try:
                for item in selected:
                    license_key = self.tree.item(item)['values'][0]
                    self.disable_license(license_key)
                
                self.root.after(0, lambda: messagebox.showinfo("成功", f"已禁用 {len(selected)} 个卡密"))
                # 使缓存失效，强制刷新
                self.cache_timestamp = None
                self.root.after(0, lambda: self.refresh_list(force=True))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"禁用失败: {e}"))
            finally:
                self.root.after(0, lambda: self.status_label.config(text="就绪"))
        
        threading.Thread(target=task, daemon=True).start()
    
    def delete_selected(self):
        """删除选中的卡密（永久删除，无法恢复）"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要删除的卡密")
            return
        
        # 显示卡密列表供确认
        license_keys = [self.tree.item(item)['values'][0] for item in selected]
        confirm_msg = f"⚠️ 警告：即将永久删除以下 {len(selected)} 个卡密\n\n"
        confirm_msg += "\n".join(f"  • {key}" for key in license_keys[:5])
        if len(license_keys) > 5:
            confirm_msg += f"\n  ... 还有 {len(license_keys) - 5} 个"
        confirm_msg += "\n\n删除后：\n"
        confirm_msg += "✓ 卡密将从数据库中永久删除\n"
        confirm_msg += "✓ 所有设备绑定记录将被清除\n"
        confirm_msg += "✓ 使用该卡密的设备将立即失效\n"
        confirm_msg += "✓ 此操作无法撤销！\n\n"
        confirm_msg += "确定要继续吗？"
        
        if not messagebox.askyesno("确认删除", confirm_msg, icon='warning'):
            return
        
        self.status_label.config(text=f"正在删除 {len(selected)} 个卡密...")
        
        def task():
            success_count = 0
            failed_count = 0
            
            try:
                for item in selected:
                    license_key = self.tree.item(item)['values'][0]
                    try:
                        self.delete_license_permanently(license_key)
                        success_count += 1
                    except Exception as e:
                        print(f"删除卡密 {license_key} 失败: {e}")
                        failed_count += 1
                
                # 显示结果
                if failed_count == 0:
                    self.root.after(0, lambda: messagebox.showinfo("删除成功", 
                        f"✓ 已永久删除 {success_count} 个卡密\n\n使用这些卡密的设备将立即失效"))
                else:
                    self.root.after(0, lambda: messagebox.showwarning("部分成功", 
                        f"成功删除: {success_count} 个\n失败: {failed_count} 个"))
                
                # 使缓存失效，强制刷新
                self.cache_timestamp = None
                self.root.after(0, lambda: self.refresh_list(force=True))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"删除失败: {e}"))
            finally:
                self.root.after(0, lambda: self.status_label.config(text="就绪"))
        
        threading.Thread(target=task, daemon=True).start()
    
    def disable_license(self, license_key: str):
        """禁用卡密（同步方法，仅供内部使用）"""
        headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}',
            'Content-Type': 'application/json'
        }
        
        data = {'status': 'disabled'}
        url = f"{self.supabase_url}/rest/v1/licenses?license_key=eq.{license_key}"
        
        requests.patch(url, headers=headers, json=data, timeout=10)
    
    def delete_license_permanently(self, license_key: str):
        """永久删除卡密（同步方法，仅供内部使用）
        
        删除步骤：
        1. 删除所有设备绑定记录
        2. 删除卡密记录
        """
        headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}',
        }
        
        # 1. 删除所有设备绑定记录
        device_url = f"{self.supabase_url}/rest/v1/device_bindings?license_key=eq.{license_key}"
        requests.delete(device_url, headers=headers, timeout=10)
        
        # 2. 删除卡密记录
        license_url = f"{self.supabase_url}/rest/v1/licenses?license_key=eq.{license_key}"
        response = requests.delete(license_url, headers=headers, timeout=10)
        
        if response.status_code not in [200, 204]:
            raise Exception(f"删除失败: HTTP {response.status_code}")
    
    def unbind_selected(self):
        """解绑选中的卡密（清除所有设备绑定）"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要解绑的卡密")
            return
        
        if not messagebox.askyesno("确认", f"确定要解绑 {len(selected)} 个卡密的所有设备吗？\n这将清除所有设备绑定记录。"):
            return
        
        self.status_label.config(text=f"正在解绑 {len(selected)} 个卡密...")
        
        def task():
            try:
                for item in selected:
                    license_key = self.tree.item(item)['values'][0]
                    self.unbind_license(license_key)
                
                self.root.after(0, lambda: messagebox.showinfo("成功", f"已解绑 {len(selected)} 个卡密"))
                # 使缓存失效，强制刷新
                self.cache_timestamp = None
                self.root.after(0, lambda: self.refresh_list(force=True))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"解绑失败: {e}"))
            finally:
                self.root.after(0, lambda: self.status_label.config(text="就绪"))
        
        threading.Thread(target=task, daemon=True).start()
    
    def unbind_license(self, license_key: str):
        """解绑卡密（清除所有设备绑定，同步方法，仅供内部使用）"""
        headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}',
            'Content-Type': 'application/json'
        }
        
        # 删除所有设备绑定记录
        device_url = f"{self.supabase_url}/rest/v1/device_bindings?license_key=eq.{license_key}"
        requests.delete(device_url, headers=headers, timeout=10)
        
        # 更新卡密状态为未使用
        data = {
            'machine_id': None,
            'status': 'unused'
        }
        url = f"{self.supabase_url}/rest/v1/licenses?license_key=eq.{license_key}"
        requests.patch(url, headers=headers, json=data, timeout=10)
    
    def extend_selected(self):
        """延长选中卡密的有效期"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要延长的卡密")
            return
        
        days = tk.simpledialog.askinteger("延长有效期", "延长天数:", minvalue=1, maxvalue=3650)
        if not days:
            return
        
        self.status_label.config(text=f"正在延长 {len(selected)} 个卡密...")
        
        def task():
            try:
                for item in selected:
                    license_key = self.tree.item(item)['values'][0]
                    self.extend_license(license_key, days)
                
                self.root.after(0, lambda: messagebox.showinfo("成功", f"已延长 {len(selected)} 个卡密 {days} 天"))
                # 使缓存失效，强制刷新
                self.cache_timestamp = None
                self.root.after(0, lambda: self.refresh_list(force=True))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"延长失败: {e}"))
            finally:
                self.root.after(0, lambda: self.status_label.config(text="就绪"))
        
        threading.Thread(target=task, daemon=True).start()
    
    def extend_license(self, license_key: str, days: int):
        """延长卡密有效期（同步方法，仅供内部使用）"""
        # 查询当前过期时间
        headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}',
        }
        
        url = f"{self.supabase_url}/rest/v1/licenses?license_key=eq.{license_key}&select=expires_at"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                current_expires = datetime.fromisoformat(data[0]['expires_at'].replace('Z', '+00:00'))
                new_expires = current_expires + timedelta(days=days)
                
                headers['Content-Type'] = 'application/json'
                update_data = {'expires_at': new_expires.isoformat()}
                url = f"{self.supabase_url}/rest/v1/licenses?license_key=eq.{license_key}"
                
                requests.patch(url, headers=headers, json=update_data, timeout=10)
    
    def export_licenses(self):
        """导出卡密"""
        filter_status = self.filter_var.get()
        
        self.status_label.config(text="正在导出卡密...")
        
        def task():
            try:
                headers = {
                    'apikey': self.supabase_key,
                    'Authorization': f'Bearer {self.supabase_key}',
                }
                
                url = f"{self.supabase_url}/rest/v1/licenses?select=*&order=created_at.desc"
                if filter_status != "all":
                    url += f"&status=eq.{filter_status}"
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    licenses = response.json()
                    
                    # 状态映射（英文 -> 中文）
                    status_map = {
                        'unused': '未使用',
                        'active': '已激活',
                        'disabled': '已禁用'
                    }
                    
                    filename = f"卡密导出_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"筛选条件: {filter_status}\n")
                        f.write(f"数量: {len(licenses)}\n")
                        f.write("=" * 80 + "\n\n")
                        
                        for lic in licenses:
                            status_cn = status_map.get(lic['status'], lic['status'])
                            f.write(f"卡密: {lic['license_key']}\n")
                            f.write(f"状态: {status_cn}\n")
                            f.write(f"设备ID: {lic.get('machine_id') or '-'}\n")
                            f.write(f"过期时间: {lic.get('expires_at', '')[:10]}\n")
                            f.write(f"备注: {lic.get('notes') or '-'}\n")
                            f.write("-" * 80 + "\n\n")
                    
                    self.root.after(0, lambda: messagebox.showinfo("成功", f"已导出 {len(licenses)} 个卡密到:\n{filename}"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("失败", f"导出失败: {response.status_code}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"导出失败: {e}"))
            finally:
                self.root.after(0, lambda: self.status_label.config(text="就绪"))
        
        threading.Thread(target=task, daemon=True).start()
    
    def copy_license(self):
        """复制卡密"""
        selected = self.tree.selection()
        if not selected:
            return
        
        license_key = self.tree.item(selected[0])['values'][0]
        self.root.clipboard_clear()
        self.root.clipboard_append(license_key)
        self.status_label.config(text=f"已复制: {license_key}")
    
    def show_details(self):
        """显示详情"""
        selected = self.tree.selection()
        if not selected:
            return
        
        values = self.tree.item(selected[0])['values']
        license_key = values[0]
        
        self.status_label.config(text="正在加载详情...")
        
        def task():
            try:
                # 查询完整信息
                headers = {
                    'apikey': self.supabase_key,
                    'Authorization': f'Bearer {self.supabase_key}',
                }
                
                url = f"{self.supabase_url}/rest/v1/licenses?license_key=eq.{license_key}&select=*"
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        lic = data[0]
                        
                        # 查询绑定的设备列表
                        device_url = f"{self.supabase_url}/rest/v1/device_bindings?license_key=eq.{license_key}&select=*"
                        device_response = requests.get(device_url, headers=headers, timeout=10)
                        
                        devices_info = ""
                        devices = []
                        if device_response.status_code == 200:
                            devices = device_response.json()
                            if devices:
                                devices_info = "\n\n已绑定设备:\n"
                                now = datetime.now()
                                for i, dev in enumerate(devices, 1):
                                    machine_id = dev.get('machine_id', '')
                                    activated_at = dev.get('activated_at', '')[:19] if dev.get('activated_at') else '-'
                                    last_check_str = dev.get('last_check_at', '')
                                    
                                    # 计算在线状态
                                    status_text = '未知'
                                    last_seen = '从未连接'
                                    
                                    if last_check_str:
                                        try:
                                            last_check = datetime.fromisoformat(last_check_str.replace('Z', '+00:00'))
                                            elapsed_seconds = (now - last_check.replace(tzinfo=None)).total_seconds()
                                            
                                            if elapsed_seconds < 300:  # 5分钟内
                                                status_text = '🟢 在线'
                                                if elapsed_seconds < 60:
                                                    last_seen = f'{int(elapsed_seconds)}秒前'
                                                else:
                                                    last_seen = f'{int(elapsed_seconds/60)}分钟前'
                                            elif elapsed_seconds < 1800:  # 30分钟内
                                                status_text = '⚪ 离线'
                                                last_seen = f'{int(elapsed_seconds/60)}分钟前'
                                            else:
                                                status_text = '🔴 长时间未连接'
                                                if elapsed_seconds < 86400:
                                                    last_seen = f'{int(elapsed_seconds/3600)}小时前'
                                                else:
                                                    last_seen = f'{int(elapsed_seconds/86400)}天前'
                                        except:
                                            pass
                                    
                                    devices_info += f"\n设备 {i}:\n"
                                    devices_info += f"  设备ID: {machine_id}\n"
                                    devices_info += f"  状态: {status_text}\n"
                                    devices_info += f"  最后连接: {last_seen}\n"
                                    devices_info += f"  激活时间: {activated_at}\n"
                        
                        # 状态映射（英文 -> 中文）
                        status_map = {
                            'unused': '未使用',
                            'active': '已激活',
                            'disabled': '已禁用'
                        }
                        status_cn = status_map.get(lic['status'], lic['status'])
                        
                        details = f"""
卡密: {lic['license_key']}
状态: {status_cn}
设备限制: {lic.get('max_devices', 1)} 台
已绑定: {len(devices)} 台
主设备ID: {lic.get('machine_id') or '-'}
创建时间: {lic.get('created_at', '')[:19]}
激活时间: {lic.get('activated_at', '')[:19] if lic.get('activated_at') else '-'}
过期时间: {lic.get('expires_at', '')[:19] if lic.get('expires_at') else '-'}
备注: {lic.get('notes') or '-'}{devices_info}
                        """
                        
                        self.root.after(0, lambda: messagebox.showinfo("卡密详情", details))
                        self.root.after(0, lambda: self.status_label.config(text="就绪"))
                        return
                
                # 如果查询失败，显示基本信息
                details = f"""
卡密: {values[0]}
状态: {values[1]}
设备限制: {values[2]}
设备ID: {values[3]}
激活时间: {values[4]}
过期时间: {values[5]}
备注: {values[6]}
                """
                
                self.root.after(0, lambda: messagebox.showinfo("卡密详情", details))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"加载详情失败: {e}"))
            finally:
                self.root.after(0, lambda: self.status_label.config(text="就绪"))
        
        threading.Thread(target=task, daemon=True).start()
    
    def show_context_menu(self, event):
        """显示右键菜单"""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)
    
    def show_device_status(self):
        """显示设备在线状态"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要查看的卡密")
            return
        
        license_key = self.tree.item(selected[0])['values'][0]
        
        # 创建状态窗口
        status_window = tk.Toplevel(self.root)
        status_window.title(f"设备状态 - {license_key}")
        status_window.geometry("700x500")
        status_window.transient(self.root)
        
        # 居中显示
        self._center_window(status_window)
        
        # 标题
        title_frame = ttk.Frame(status_window)
        title_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Label(title_frame, text=f"卡密: {license_key}", font=("", 12, "bold")).pack(side=tk.LEFT)
        
        # 刷新按钮
        refresh_btn = ttk.Button(title_frame, text="刷新", command=lambda: self._refresh_device_status(license_key, tree, status_label))
        refresh_btn.pack(side=tk.RIGHT)
        
        # 设备列表
        list_frame = ttk.Frame(status_window)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        columns = ("设备ID", "状态", "最后连接", "激活时间")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        
        tree.heading("设备ID", text="设备ID")
        tree.heading("状态", text="状态")
        tree.heading("最后连接", text="最后连接")
        tree.heading("激活时间", text="激活时间")
        
        tree.column("设备ID", width=250)
        tree.column("状态", width=100)
        tree.column("最后连接", width=150)
        tree.column("激活时间", width=150)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 状态栏
        status_label = ttk.Label(status_window, text="正在加载...", relief=tk.SUNKEN)
        status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 加载数据
        self._refresh_device_status(license_key, tree, status_label)
    
    def _refresh_device_status(self, license_key: str, tree: ttk.Treeview, status_label: ttk.Label):
        """刷新设备状态"""
        status_label.config(text="正在加载...")
        
        def task():
            try:
                headers = {
                    'apikey': self.supabase_key,
                    'Authorization': f'Bearer {self.supabase_key}',
                }
                
                # 查询设备绑定
                device_url = f"{self.supabase_url}/rest/v1/device_bindings?license_key=eq.{license_key}&select=*"
                device_response = requests.get(device_url, headers=headers, timeout=10)
                
                if device_response.status_code != 200:
                    self.root.after(0, lambda: status_label.config(text=f"查询失败: {device_response.status_code}"))
                    return
                
                devices = device_response.json()
                
                # 计算在线状态
                now = datetime.now()
                device_list = []
                
                for device in devices:
                    last_check_str = device.get('last_check_at', '')
                    machine_id = device.get('machine_id', '')
                    activated_at = device.get('activated_at', '')[:19] if device.get('activated_at') else '-'
                    
                    # 计算在线状态
                    status = 'unknown'
                    status_text = '未知'
                    last_seen = '从未连接'
                    
                    if last_check_str:
                        try:
                            last_check = datetime.fromisoformat(last_check_str.replace('Z', '+00:00'))
                            elapsed_seconds = (now - last_check.replace(tzinfo=None)).total_seconds()
                            
                            # 判断在线状态
                            if elapsed_seconds < 300:  # 5分钟内
                                status = 'online'
                                status_text = '🟢 在线'
                                if elapsed_seconds < 60:
                                    last_seen = f'{int(elapsed_seconds)}秒前'
                                else:
                                    last_seen = f'{int(elapsed_seconds/60)}分钟前'
                            elif elapsed_seconds < 1800:  # 30分钟内
                                status = 'offline'
                                status_text = '⚪ 离线'
                                last_seen = f'{int(elapsed_seconds/60)}分钟前'
                            else:
                                status = 'inactive'
                                status_text = '🔴 长时间未连接'
                                if elapsed_seconds < 86400:  # 24小时内
                                    last_seen = f'{int(elapsed_seconds/3600)}小时前'
                                else:
                                    last_seen = f'{int(elapsed_seconds/86400)}天前'
                        except:
                            pass
                    
                    device_list.append({
                        'machine_id': machine_id,
                        'status': status,
                        'status_text': status_text,
                        'last_seen': last_seen,
                        'activated_at': activated_at
                    })
                
                # 更新表格
                def update_ui():
                    # 清空
                    for item in tree.get_children():
                        tree.delete(item)
                    
                    # 添加数据
                    for dev in device_list:
                        tree.insert("", tk.END, 
                                  values=(dev['machine_id'], dev['status_text'], dev['last_seen'], dev['activated_at']),
                                  tags=(dev['status'],))
                    
                    # 设置标签颜色
                    tree.tag_configure("online", foreground="green")
                    tree.tag_configure("offline", foreground="gray")
                    tree.tag_configure("inactive", foreground="red")
                    tree.tag_configure("unknown", foreground="orange")
                    
                    # 更新状态栏
                    online_count = sum(1 for d in device_list if d['status'] == 'online')
                    offline_count = sum(1 for d in device_list if d['status'] == 'offline')
                    inactive_count = sum(1 for d in device_list if d['status'] == 'inactive')
                    
                    status_label.config(text=f"总设备: {len(device_list)} | 在线: {online_count} | 离线: {offline_count} | 长时间未连接: {inactive_count}")
                
                self.root.after(0, update_ui)
                
            except Exception as e:
                self.root.after(0, lambda: status_label.config(text=f"加载失败: {e}"))
        
        threading.Thread(target=task, daemon=True).start()
    
    def apply_filter(self):
        """应用筛选（使用本地缓存，不请求网络）"""
        if not self.licenses_cache:
            # 如果缓存为空，强制刷新
            self.refresh_list(force=True)
            return
        
        filter_status = self.filter_var.get()
        
        # 从缓存中筛选
        if filter_status == "all":
            filtered_licenses = self.licenses_cache
        else:
            filtered_licenses = [lic for lic in self.licenses_cache if lic['status'] == filter_status]
        
        print(f"[缓存] 筛选 {filter_status}，共 {len(filtered_licenses)} 条")
        
        # 更新表格
        self.update_tree(filtered_licenses)
        self.status_label.config(text=f"就绪（筛选: {filter_status}，使用缓存）")
    
    def toggle_auto_refresh(self):
        """切换自动刷新"""
        self.auto_refresh = self.auto_refresh_var.get()
        if self.auto_refresh:
            self.status_label.config(text="已启用自动同步（每15秒）")
            self.start_auto_refresh()
        else:
            self.status_label.config(text="已禁用自动同步")
    
    def start_auto_refresh(self):
        """启动自动刷新（强制刷新，更新缓存）"""
        if self.auto_refresh:
            self.refresh_list(force=True)
            # 15秒后再次刷新
            self.root.after(self.refresh_interval, self.start_auto_refresh)
    
    def update_sync_time(self):
        """更新同步时间显示"""
        if self.last_refresh_time:
            elapsed = (datetime.now() - self.last_refresh_time).total_seconds()
            if elapsed < 60:
                time_str = f"{int(elapsed)}秒前"
            elif elapsed < 3600:
                time_str = f"{int(elapsed/60)}分钟前"
            else:
                time_str = f"{int(elapsed/3600)}小时前"
            
            self.sync_time_label.config(text=f"上次同步: {time_str}")
            
            # 每5秒更新一次显示
            if self.auto_refresh:
                self.root.after(5000, self.update_sync_time)


def verify_admin_password() -> bool:
    """验证管理员密码（独立函数）"""
    import hashlib
    from tkinter import messagebox
    
    # 获取密码哈希
    password_hash = None
    
    # 尝试从配置文件和服务器获取密码
    try:
        config_file = None
        if os.path.exists(".env"):
            config_file = ".env"
        elif os.path.exists("server/.env"):
            config_file = "server/.env"
        
        if config_file:
            supabase_url = None
            supabase_key = None
            
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == 'SUPABASE_URL':
                            supabase_url = value
                        elif key == 'SUPABASE_KEY':
                            supabase_key = value
            
            if supabase_url and supabase_key:
                import requests
                headers = {
                    'apikey': supabase_key,
                    'Authorization': f'Bearer {supabase_key}',
                }
                
                url = f"{supabase_url}/rest/v1/admin_password?id=eq.1&select=password_hash"
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        password_hash = data[0]['password_hash']
                        print("[密码] 从服务器获取密码哈希")
    except Exception as e:
        print(f"[密码] 从服务器获取密码失败: {e}")
    
    # 如果没有获取到，使用默认密码
    if not password_hash:
        default_password = "hye19911206"
        password_hash = hashlib.sha256(default_password.encode()).hexdigest()
        print("[密码] 使用默认密码")
    
    # 创建密码对话框（使用系统默认样式）
    password_dialog = tk.Tk()
    password_dialog.title("访问验证")
    password_dialog.resizable(False, False)
    
    # 先隐藏窗口
    password_dialog.withdraw()
    
    # 居中显示
    password_dialog.update()
    width = 350
    height = 150
    screen_width = password_dialog.winfo_screenwidth()
    screen_height = password_dialog.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    
    # 确保坐标不为负数
    x = max(0, x)
    y = max(0, y)
    
    password_dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    # 显示窗口
    password_dialog.deiconify()
    
    # 确保窗口显示在最前面
    password_dialog.lift()
    password_dialog.attributes('-topmost', True)
    password_dialog.after(100, lambda: password_dialog.attributes('-topmost', False))
    
    # 获得焦点
    password_dialog.focus_force()
    
    # 主容器
    main_frame = tk.Frame(password_dialog, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # 标题
    title_label = tk.Label(
        main_frame,
        text="请输入管理员密码",
        font=("Microsoft YaHei UI", 11)
    )
    title_label.pack(pady=(0, 15))
    
    # 密码输入框
    password_entry = tk.Entry(
        main_frame,
        font=("Consolas", 11),
        show='●',
        width=30
    )
    password_entry.pack(pady=(0, 15))
    password_entry.focus_set()
    
    # 结果变量
    result = {'verified': False}
    
    def verify():
        """验证密码"""
        password = password_entry.get().strip()
        
        if not password:
            messagebox.showwarning("提示", "请输入密码", parent=password_dialog)
            return
        
        # 验证密码
        input_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if input_hash == password_hash:
            result['verified'] = True
            password_dialog.quit()
            password_dialog.destroy()
        else:
            messagebox.showerror("错误", "密码错误！", parent=password_dialog)
            password_entry.delete(0, tk.END)
            password_entry.focus_set()
    
    def cancel():
        """取消"""
        result['verified'] = False
        password_dialog.quit()
        password_dialog.destroy()
    
    # 绑定回车键
    password_entry.bind('<Return>', lambda e: verify())
    
    # 禁止关闭窗口
    password_dialog.protocol("WM_DELETE_WINDOW", cancel)
    
    # 按钮容器
    button_frame = tk.Frame(main_frame)
    button_frame.pack()
    
    # 确定按钮
    ok_btn = tk.Button(
        button_frame,
        text="确定",
        command=verify,
        width=10
    )
    ok_btn.pack(side=tk.LEFT, padx=5)
    
    # 取消按钮
    cancel_btn = tk.Button(
        button_frame,
        text="取消",
        command=cancel,
        width=10
    )
    cancel_btn.pack(side=tk.LEFT, padx=5)
    
    # 运行对话框
    password_dialog.mainloop()
    
    return result['verified']


def main():
    """主函数"""
    try:
        # 先验证密码
        if not verify_admin_password():
            # 密码验证失败，直接退出
            return
        
        # 密码验证成功，创建主窗口
        root = tk.Tk()
        app = LicenseManagerGUI(root)
        root.mainloop()
        
    except Exception as e:
        # 捕获所有异常，显示错误信息
        import traceback
        error_msg = f"程序启动失败:\n\n{str(e)}\n\n详细信息:\n{traceback.format_exc()}"
        
        try:
            # 尝试显示错误对话框
            error_root = tk.Tk()
            error_root.withdraw()
            messagebox.showerror("启动错误", error_msg)
            error_root.destroy()
        except:
            # 如果无法显示对话框，写入日志文件
            with open("error_log.txt", "w", encoding="utf-8") as f:
                f.write(error_msg)


if __name__ == "__main__":
    main()
