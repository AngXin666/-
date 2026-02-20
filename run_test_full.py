#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整启动测试 - 逐步测试每个导入
"""

# ============================================================
# 【关键修复】必须在最开始就设置递归深度限制
# ============================================================
import sys
sys.setrecursionlimit(15000)

import os
import traceback

def main():
    """主函数"""
    
    # 设置工作目录
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    os.chdir(application_path)
    sys.path.insert(0, application_path)
    
    # 创建日志文件
    log_file = os.path.join(application_path, "full_test.log")
    
    def log(msg):
        """写入日志"""
        print(msg)
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except:
            pass
    
    try:
        log("=" * 60)
        log("完整启动测试")
        log("=" * 60)
        log(f"打包环境: {getattr(sys, 'frozen', False)}")
        log(f"工作目录: {os.getcwd()}")
        
        log("\n[1] 测试license_manager_simple...")
        from src.license_manager_simple import SimpleLicenseManager
        log("  OK license_manager_simple")
        
        log("\n[2] 创建license_manager实例...")
        license_manager = SimpleLicenseManager()
        log("  OK 实例创建成功")
        
        log("\n[3] 检查许可证...")
        valid, message = license_manager.check_license()
        log(f"  许可证状态: {valid}, {message}")
        
        if not valid:
            log("\n[4] 测试simple_activation_dialog...")
            from src.simple_activation_dialog import SimpleActivationDialog
            log("  OK simple_activation_dialog导入成功")
            log("  （跳过实际显示对话框）")
        
        log("\n[5] 测试adb_bridge...")
        from src.adb_bridge import ADBBridge
        log("  OK adb_bridge")
        
        log("\n[6] 测试emulator_controller...")
        from src.emulator_controller import EmulatorController
        log("  OK emulator_controller")
        
        log("\n[7] 测试gui...")
        from src.gui import main as gui_main
        log("  OK gui")
        
        log("\n[8] 测试model_manager...")
        from src.model_manager import ModelManager
        log("  OK model_manager")
        
        log("\n" + "=" * 60)
        log("所有测试通过！")
        log("=" * 60)
        
        # 显示消息框
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("成功", "完整启动测试通过！")
            root.destroy()
        except:
            pass
        
    except Exception as e:
        error_msg = f"错误: {e}\n{traceback.format_exc()}"
        log(f"\nERROR {error_msg}")
        
        # 显示错误对话框
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("错误", f"测试失败:\n{str(e)}\n\n详细信息已保存到 {log_file}")
            root.destroy()
        except:
            pass
    
    finally:
        log(f"\n日志文件: {log_file}")

if __name__ == "__main__":
    main()
