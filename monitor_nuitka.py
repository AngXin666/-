#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""监控Nuitka编译进度"""

import time
import os
from pathlib import Path

print("=" * 60)
print("  Nuitka编译进度监控")
print("=" * 60)
print("\n正在监控编译过程，按Ctrl+C停止监控...\n")

build_dir = Path('nuitka_build')
last_size = 0
last_file_count = 0

while True:
    try:
        if build_dir.exists():
            # 统计文件数量和总大小
            files = list(build_dir.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            if file_count != last_file_count or total_size != last_size:
                print(f"[{time.strftime('%H:%M:%S')}] 已生成 {file_count} 个文件，总大小: {total_size/1024/1024:.1f} MB")
                last_file_count = file_count
                last_size = total_size
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 等待编译开始...")
        
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        break
    except Exception as e:
        print(f"错误: {e}")
        time.sleep(5)
