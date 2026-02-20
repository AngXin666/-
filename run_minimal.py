#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小化启动脚本 - 用于诊断打包问题
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
    log_file = os.path.join(application_path, "minimal_startup.log")
    
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
        log("最小化启动测试")
        log("=" * 60)
        log(f"Python版本: {sys.version}")
        log(f"递归深度限制: {sys.getrecursionlimit()}")
        log(f"打包环境: {getattr(sys, 'frozen', False)}")
        log(f"工作目录: {os.getcwd()}")
        log(f"可执行文件: {sys.executable}")
        
        log("\n[1/5] 导入torch...")
        import torch
        log(f"  ✓ torch {torch.__version__}")
        
        log("\n[2/5] 导入PIL...")
        from PIL import Image
        log("  ✓ PIL")
        
        log("\n[3/5] 导入src.adb_bridge...")
        from src.adb_bridge import ADBBridge
        log("  ✓ src.adb_bridge")
        
        log("\n[4/5] 导入src.page_detector_integrated...")
        from src.page_detector_integrated import PageDetectorIntegrated
        log("  ✓ src.page_detector_integrated")
        
        log("\n[5/5] 导入src.model_manager...")
        from src.model_manager import ModelManager
        log("  ✓ src.model_manager")
        
        log("\n创建ModelManager实例...")
        manager = ModelManager.get_instance()
        log("  ✓ ModelManager实例创建成功")
        
        log("\n" + "=" * 60)
        log("所有测试通过！程序可以正常启动")
        log("=" * 60)
        
        # 显示消息框
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("成功", "最小化启动测试通过！\n所有模块导入成功。")
            root.destroy()
        except:
            pass
        
    except Exception as e:
        error_msg = f"错误: {e}\n{traceback.format_exc()}"
        log(f"\n✗ {error_msg}")
        
        # 显示错误对话框
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("错误", f"启动失败:\n{str(e)}\n\n详细信息已保存到 {log_file}")
            root.destroy()
        except:
            pass
    
    finally:
        log(f"\n日志文件: {log_file}")

if __name__ == "__main__":
    main()
