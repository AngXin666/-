#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查MuMu模拟器进程"""

import psutil

print("=" * 60)
print("检查 MuMuPlayer.exe 进程")
print("=" * 60)

found_processes = []
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        if proc.info['name'] == 'MuMuPlayer.exe':
            found_processes.append(proc)
            print(f"\n进程 PID: {proc.info['pid']}")
            print(f"命令行: {proc.info['cmdline']}")
            
            # 提取实例ID
            cmdline = proc.info['cmdline']
            if cmdline:
                for i, arg in enumerate(cmdline):
                    if arg == '-v' and i + 1 < len(cmdline):
                        print(f"  → 实例ID: {cmdline[i + 1]}")
                        break
                else:
                    print(f"  → 没有找到 -v 参数，可能是实例0")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

print(f"\n总共找到 {len(found_processes)} 个 MuMuPlayer.exe 进程")
