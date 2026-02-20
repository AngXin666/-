#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试打包后程序启动
"""

import subprocess
import sys
import time

# 打包程序路径
PACKED_EXE = r"D:\溪盟商城自动化助手_打包_新版\溪盟商城自动化助手.exe"

print("="*60)
print("测试打包后程序启动")
print("="*60)

print(f"\n[1] 启动程序: {PACKED_EXE}")
print("等待程序输出...")

try:
    # 使用subprocess运行，捕获输出
    process = subprocess.Popen(
        [PACKED_EXE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    # 等待5秒
    time.sleep(5)
    
    # 检查进程状态
    poll = process.poll()
    if poll is not None:
        print(f"\n[ERROR] 程序已退出，退出码: {poll}")
        
        # 读取输出
        stdout, stderr = process.communicate(timeout=1)
        
        if stdout:
            print("\n[STDOUT]:")
            print(stdout)
        
        if stderr:
            print("\n[STDERR]:")
            print(stderr)
    else:
        print("\n[OK] 程序正在运行")
        print("手动关闭程序以继续...")
        
        # 等待用户关闭
        process.wait()
        
        # 读取输出
        stdout, stderr = process.communicate(timeout=1)
        
        if stdout:
            print("\n[STDOUT]:")
            print(stdout)
        
        if stderr:
            print("\n[STDERR]:")
            print(stderr)

except Exception as e:
    print(f"\n[ERROR] 启动失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("测试完成")
print("="*60)
