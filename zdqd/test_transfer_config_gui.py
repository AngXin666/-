"""
测试转账配置GUI - 验证转账目标模式选择功能
Test Transfer Config GUI - Verify Transfer Target Mode Selection
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import tkinter as tk
from tkinter import ttk
from src.gui import TransferConfigWindow


def test_transfer_config_gui():
    """测试转账配置GUI"""
    print("=" * 70)
    print("测试转账配置GUI - 转账目标模式选择")
    print("=" * 70)
    
    # 创建主窗口
    root = tk.Tk()
    root.title("转账配置GUI测试")
    root.geometry("300x200")
    
    # 日志显示
    log_text = tk.Text(root, height=8, width=40)
    log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    def log_callback(message):
        """日志回调"""
        log_text.insert(tk.END, f"{message}\n")
        log_text.see(tk.END)
        print(message)
    
    # 打开转账配置窗口按钮
    def open_transfer_config():
        try:
            # 创建转账配置窗口
            config_window = TransferConfigWindow(
                parent=root,
                log_callback=log_callback,
                accounts_file="data/accounts.txt.enc",
                gui_instance=None
            )
            log_callback("转账配置窗口已打开")
        except Exception as e:
            log_callback(f"打开转账配置窗口失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 按钮
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=5)
    
    ttk.Button(btn_frame, text="打开转账配置", command=open_transfer_config).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="退出", command=root.quit).pack(side=tk.LEFT, padx=5)
    
    log_callback("点击'打开转账配置'按钮测试GUI")
    log_callback("在转账配置窗口中可以看到三个单选按钮：")
    log_callback("  - 转给管理员自己")
    log_callback("  - 转给管理员的收款人")
    log_callback("  - 转给系统配置收款人")
    
    # 运行主循环
    root.mainloop()


if __name__ == "__main__":
    test_transfer_config_gui()
