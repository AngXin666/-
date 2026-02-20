#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断模型加载问题
"""

import os
import sys

def main():
    """诊断模型加载"""
    print("=" * 60)
    print("模型加载诊断工具")
    print("=" * 60)
    
    # 检查打包目录的日志文件
    packed_dir = r"D:\溪盟商城自动化助手_打包"
    startup_log = os.path.join(packed_dir, "startup_error.log")
    
    print(f"\n[1] 检查启动日志: {startup_log}")
    if os.path.exists(startup_log):
        print("  ✓ 日志文件存在")
        print("\n日志内容:")
        print("-" * 60)
        try:
            with open(startup_log, "r", encoding="utf-8") as f:
                content = f.read()
                print(content)
        except Exception as e:
            print(f"  ✗ 读取失败: {e}")
    else:
        print("  ✗ 日志文件不存在")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()
