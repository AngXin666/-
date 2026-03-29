#!/usr/bin/env python3
"""检查保存缓存时的日志输出"""

import os
from pathlib import Path
from datetime import datetime

def check_logs():
    """检查最近的日志文件"""
    
    # 查找日志目录
    log_dirs = [
        Path("logs"),
        Path("log"),
        Path("."),
    ]
    
    log_files = []
    for log_dir in log_dirs:
        if log_dir.exists():
            # 查找所有日志文件
            for pattern in ["*.log", "*.txt"]:
                log_files.extend(log_dir.glob(pattern))
    
    if not log_files:
        print("❌ 未找到日志文件")
        return
    
    # 按修改时间排序,找最新的
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"找到 {len(log_files)} 个日志文件\n")
    
    # 检查最近的5个日志文件
    for log_file in log_files[:5]:
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
        print(f"\n{'='*60}")
        print(f"文件: {log_file}")
        print(f"修改时间: {mtime}")
        print(f"{'='*60}")
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 查找与缓存相关的日志
            lines = content.split('\n')
            cache_lines = []
            
            for i, line in enumerate(lines):
                if any(keyword in line for keyword in [
                    '缓存', 'cache', 'Cache', 'CACHE',
                    '保存', 'save', 'Save', 'SAVE',
                    'login_cache', 'LoginCache'
                ]):
                    # 包含前后各2行的上下文
                    start = max(0, i-2)
                    end = min(len(lines), i+3)
                    cache_lines.extend(lines[start:end])
                    cache_lines.append("---")
            
            if cache_lines:
                print("\n与缓存相关的日志:")
                print('\n'.join(cache_lines[-50:]))  # 只显示最后50行
            else:
                print("未找到与缓存相关的日志")
                
        except Exception as e:
            print(f"读取失败: {e}")

if __name__ == "__main__":
    check_logs()
