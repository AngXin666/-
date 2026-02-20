#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试批量添加账号修复 - 验证不会卡死
"""

import sys
import os

# 设置工作目录
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

os.chdir(application_path)
sys.path.insert(0, application_path)

def test_batch_add():
    """测试批量添加账号功能"""
    print("=" * 60)
    print("测试批量添加账号修复")
    print("=" * 60)
    
    try:
        # 导入必要的模块
        print("\n[1/4] 导入模块...")
        import tkinter as tk
        from src.user_management_gui import UserManagementDialog
        print("✓ 模块导入成功")
        
        # 创建测试窗口
        print("\n[2/4] 创建测试窗口...")
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        def log_callback(msg):
            print(f"[LOG] {msg}")
        
        # 创建用户管理对话框
        dialog = UserManagementDialog(root, log_callback)
        print("✓ 窗口创建成功")
        
        # 生成测试账号数据（100个账号）
        print("\n[3/4] 生成测试数据（100个账号）...")
        test_accounts = []
        for i in range(100):
            phone = f"1380013{i:04d}"  # 13800130000 - 13800130099
            password = f"test_pass_{i}"
            test_accounts.append(f"{phone}----{password}")
        
        test_text = "\n".join(test_accounts)
        print(f"✓ 生成了 {len(test_accounts)} 个测试账号")
        
        # 填充到文本框
        print("\n[4/4] 测试批量添加...")
        dialog.batch_accounts_text.delete("1.0", tk.END)
        dialog.batch_accounts_text.insert("1.0", test_text)
        
        # 模拟点击"添加到账号文件"按钮
        print("  执行批量添加操作...")
        
        # 设置管理员为"不分配"
        dialog.batch_owner_var.set("不分配")
        
        # 调用批量添加方法
        try:
            dialog._batch_add_accounts_action()
            print("✓ 批量添加完成，没有卡死！")
        except Exception as e:
            print(f"✗ 批量添加失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 关闭窗口
        try:
            dialog.dialog.destroy()
        except:
            pass
        
        try:
            root.quit()
            root.destroy()
        except:
            pass
        
        print("\n" + "=" * 60)
        print("✓ 测试通过！批量添加不会卡死")
        print("=" * 60)
        
        # 强制退出
        import time
        time.sleep(0.5)
        os._exit(0)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batch_add()
    
    if success:
        print("\n✓ 修复验证成功，可以打包了")
        sys.exit(0)
    else:
        print("\n✗ 修复验证失败，需要继续调试")
        sys.exit(1)
