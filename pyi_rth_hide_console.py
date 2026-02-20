"""
PyInstaller Runtime Hook - 隐藏所有控制台窗口
这个文件会在打包后的程序启动时最早执行
"""

import sys

# 只在Windows打包环境中执行
if sys.platform == 'win32' and getattr(sys, 'frozen', False):
    try:
        import ctypes
        import ctypes.wintypes
        
        # 获取当前进程的所有控制台窗口并隐藏
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        
        # 获取控制台窗口句柄
        console_window = kernel32.GetConsoleWindow()
        if console_window:
            # 隐藏控制台窗口
            SW_HIDE = 0
            user32.ShowWindow(console_window, SW_HIDE)
        
        # 设置进程创建标志，所有子进程都不显示窗口
        CREATE_NO_WINDOW = 0x08000000
        DETACHED_PROCESS = 0x00000008
        
    except Exception:
        pass
