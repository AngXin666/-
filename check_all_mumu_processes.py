#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查所有包含MuMu的进程"""

import psutil

print("=" * 60)
print("检查所有包含 'MuMu' 或 'mumu' 的进程")
print("=" * 60)

found_processes = []
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        proc_name = proc.info['name']
        if proc_name and ('mumu' in proc_name.lower() or 'nemu' in proc_name.lower()):
            found_processes.append(proc)
            print(f"\n进程名: {proc_name}")
            print(f"PID: {proc.info['pid']}")
            print(f"命令行: {proc.info['cmdline']}")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

print(f"\n总共找到 {len(found_processes)} 个相关进程")
