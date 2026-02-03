"""
测试用户管理GUI的新布局
Test the new layout of User Management GUI
"""

import tkinter as tk
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.user_management_gui import UserManagementDialog


def test_layout():
    """测试新的左右分栏布局"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    def log_callback(message):
        print(f"[LOG] {message}")
    
    # 创建用户管理对话框
    dialog = UserManagementDialog(root, log_callback)
    
    print("✓ 用户管理GUI已打开")
    print("  - 窗口尺寸: 1400x700")
    print("  - 左侧: 用户管理区域")
    print("  - 右侧: 批量添加账号区域")
    print("\n请检查布局是否正确...")
    
    root.mainloop()


if __name__ == "__main__":
    test_layout()
