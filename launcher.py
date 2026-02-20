#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动器 - 用于启动主程序
不使用asyncio，避免打包问题
"""

import sys
import os
import subprocess

def main():
    """启动主程序"""
    # 获取当前目录
    if getattr(sys, 'frozen', False):
        # 打包后
        app_dir = os.path.dirname(sys.executable)
    else:
        # 开发环境
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 主程序路径
    main_exe = os.path.join(app_dir, '_internal', 'run.exe')
    
    if not os.path.exists(main_exe):
        # 如果在_internal中找不到，尝试当前目录
        main_exe = os.path.join(app_dir, 'run.exe')
    
    if not os.path.exists(main_exe):
        print("错误：找不到主程序")
        input("按回车键退出...")
        return
    
    # 启动主程序
    try:
        subprocess.run([main_exe], cwd=app_dir)
    except Exception as e:
        print(f"启动失败: {e}")
        input("按回车键退出...")

if __name__ == '__main__':
    main()
