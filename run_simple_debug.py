#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化调试版本 - 最小化启动
"""

# 在最开始设置递归深度
import sys
sys.setrecursionlimit(15000)

import os

# 在最开始patch subprocess
if sys.platform == 'win32':
    import subprocess
    
    _STARTUPINFO = subprocess.STARTUPINFO()
    _STARTUPINFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    _STARTUPINFO.wShowWindow = subprocess.SW_HIDE
    _CREATE_NO_WINDOW = 0x08000000
    
    _original_popen = subprocess.Popen
    _original_run = subprocess.run
    
    def _patched_popen(*args, **kwargs):
        if 'startupinfo' not in kwargs:
            kwargs['startupinfo'] = _STARTUPINFO
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = _CREATE_NO_WINDOW
        else:
            kwargs['creationflags'] |= _CREATE_NO_WINDOW
        return _original_popen(*args, **kwargs)
    
    def _patched_run(*args, **kwargs):
        if 'startupinfo' not in kwargs:
            kwargs['startupinfo'] = _STARTUPINFO
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = _CREATE_NO_WINDOW
        else:
            kwargs['creationflags'] |= _CREATE_NO_WINDOW
        return _original_run(*args, **kwargs)
    
    subprocess.Popen = _patched_popen
    subprocess.run = _patched_run

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
    log_file = os.path.join(application_path, "simple_debug.log")
    
    def log(msg):
        """写入日志"""
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except:
            pass
    
    try:
        log("=" * 60)
        log("简化调试启动")
        log("=" * 60)
        log(f"打包环境: {getattr(sys, 'frozen', False)}")
        log(f"工作目录: {os.getcwd()}")
        log(f"Python版本: {sys.version}")
        log(f"递归深度: {sys.getrecursionlimit()}")
        
        log("\n显示消息框...")
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("成功", f"程序启动成功！\n\n工作目录: {os.getcwd()}\n打包环境: {getattr(sys, 'frozen', False)}")
        root.destroy()
        
        log("\n程序正常退出")
        
    except Exception as e:
        error_msg = f"错误: {e}\n{traceback.format_exc()}"
        log(f"\nERROR: {error_msg}")
        
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("错误", f"启动失败:\n{str(e)}\n\n详细信息已保存到 {log_file}")
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    main()
