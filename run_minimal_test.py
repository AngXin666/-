#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小化测试 - 只测试Tkinter GUI
"""

import sys
sys.setrecursionlimit(15000)

import os
import tkinter as tk
from tkinter import messagebox

def main():
    """最小化测试"""
    try:
        print("创建Tk窗口...")
        root = tk.Tk()
        root.title("最小化测试")
        root.geometry("400x300")
        
        print("添加标签...")
        label = tk.Label(root, text="如果你看到这个窗口，说明Tkinter工作正常！", font=("Arial", 14))
        label.pack(pady=50)
        
        print("添加按钮...")
        button = tk.Button(root, text="关闭", command=root.quit)
        button.pack(pady=20)
        
        print("显示窗口...")
        root.deiconify()
        root.lift()
        
        print("启动mainloop...")
        root.mainloop()
        
        print("程序正常退出")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            messagebox.showerror("错误", f"程序出错:\n{str(e)}")
        except:
            pass

if __name__ == "__main__":
    main()
