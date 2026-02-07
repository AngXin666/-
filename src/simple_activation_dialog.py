"""
简化激活对话框
Simple Activation Dialog
"""

import tkinter as tk
from tkinter import messagebox
from .license_manager_simple import SimpleLicenseManager


class SimpleActivationDialog:
    """简化的激活卡密对话框"""
    
    def __init__(self, parent=None):
        try:
            print("[激活对话框] 初始化开始...")
            self.result = False
            self.license_manager = SimpleLicenseManager()
            
            print("[激活对话框] 创建窗口...")
            # 直接创建 Tk 主窗口（不使用 Toplevel）
            self.dialog = tk.Tk()
            self.dialog.title("激活卡密")
            self.dialog.geometry("500x350")
            self.dialog.resizable(False, False)
            
            print("[激活对话框] 主窗口已创建")
            
            # 关闭窗口时的处理
            self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
            
            # 创建界面
            print("[激活对话框] 创建界面组件...")
            self._create_widgets()
            
            # 居中显示
            print("[激活对话框] 居中窗口...")
            self._center_window()
            
            # 确保窗口显示在最前面
            self.dialog.lift()
            self.dialog.attributes('-topmost', True)  # 置顶
            self.dialog.after(100, lambda: self.dialog.attributes('-topmost', False))  # 100ms后取消置顶
            self.dialog.focus_force()
            
            print("[激活对话框] 窗口已显示，等待用户操作...")
            
            # 使用 mainloop 而不是 wait_window
            self.dialog.mainloop()
            
            print("[激活对话框] 对话框已关闭")
            
        except Exception as e:
            print(f"[激活对话框] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            # 确保清理资源
            if hasattr(self, 'dialog'):
                try:
                    self.dialog.destroy()
                except:
                    pass
            raise
    
    def _center_window(self):
        """将窗口居中显示"""
        self.dialog.update_idletasks()
        width = 500
        height = 350
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_widgets(self):
        """创建界面组件"""
        # 主容器
        main_frame = tk.Frame(self.dialog, padx=30, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = tk.Label(
            main_frame,
            text="激活卡密",
            font=("Microsoft YaHei UI", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # 卡密输入
        license_label = tk.Label(
            main_frame,
            text="请输入卡密:",
            font=("Microsoft YaHei UI", 10)
        )
        license_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.license_entry = tk.Entry(
            main_frame,
            font=("Consolas", 11),
            width=40
        )
        self.license_entry.pack(fill=tk.X, ipady=5, pady=(0, 15))
        self.license_entry.bind('<Return>', lambda e: self._activate())
        
        # 设备ID显示
        machine_label = tk.Label(
            main_frame,
            text="设备ID (机器码):",
            font=("Microsoft YaHei UI", 10)
        )
        machine_label.pack(anchor=tk.W, pady=(0, 5))
        
        # 设备ID容器
        machine_frame = tk.Frame(main_frame)
        machine_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.machine_id = self.license_manager.get_machine_id()
        machine_id_label = tk.Label(
            machine_frame,
            text=self.machine_id,
            font=("Consolas", 9),
            relief=tk.SUNKEN,
            bd=1,
            padx=10,
            pady=8
        )
        machine_id_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        copy_btn = tk.Button(
            machine_frame,
            text="复制",
            command=self._copy_machine_id,
            font=("Microsoft YaHei UI", 9),
            width=8
        )
        copy_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # 提示
        hint_label = tk.Label(
            main_frame,
            text="此设备ID将与卡密绑定",
            font=("Microsoft YaHei UI", 8),
            fg="gray"
        )
        hint_label.pack(pady=(0, 20))
        
        # 按钮
        button_frame = tk.Frame(main_frame)
        button_frame.pack()
        
        activate_btn = tk.Button(
            button_frame,
            text="激活",
            font=("Microsoft YaHei UI", 10),
            command=self._activate,
            width=12
        )
        activate_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(
            button_frame,
            text="取消",
            font=("Microsoft YaHei UI", 10),
            command=self._on_cancel,
            width=12
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def _copy_machine_id(self):
        """复制机器ID到剪贴板"""
        self.dialog.clipboard_clear()
        self.dialog.clipboard_append(self.machine_id)
        messagebox.showinfo("提示", "设备ID已复制到剪贴板", parent=self.dialog)
    
    def _activate(self):
        """激活卡密"""
        license_key = self.license_entry.get().strip()
        
        if not license_key:
            messagebox.showwarning("提示", "请输入卡密", parent=self.dialog)
            return
        
        # 激活
        success, message = self.license_manager.activate_license(license_key)
        
        if success:
            messagebox.showinfo("激活成功", message, parent=self.dialog)
            self.result = True
            self.dialog.quit()  # 退出主循环
            self.dialog.destroy()  # 销毁窗口
        else:
            messagebox.showerror("激活失败", message, parent=self.dialog)
    
    def _on_cancel(self):
        """取消激活"""
        self.result = False
        self.dialog.quit()  # 退出主循环
        self.dialog.destroy()  # 销毁窗口
